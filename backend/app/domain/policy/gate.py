"""Sensitivity gate — classifies content before any external provider call.

TWO-PASS ARCHITECTURE:
  Pass 1: Keyword heuristics (sync, <1ms) — fast path for obvious personal/relationship content
  Pass 2: Embedding-prototype classification (CPU-bound, ~20-100ms) — handles novel patterns

SECURITY CONTRACT (PIPE-03, PRIV-04, PRIV-05):
  - Gate MUST run before any external API call
  - personal_profile → blocked=True
  - relationship → blocked=True
  - unclassified → blocked=True (unknown content is not allowed by default)
  - general → blocked=False (explicitly safe to send to external provider)
  - Gate decision MUST be recorded in audit trail (caller's responsibility)

USAGE:
  gate = SensitivityGate()
  decision = gate.classify(text)  # sync, uses heuristics; safe for quick check
  # or
  decision = await gate.classify_async(text)  # wraps embedding pass in asyncio.to_thread()
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from app.infrastructure.local_model import MODEL_NAME, LocalModelNotReady, load_local_model

logger = logging.getLogger(__name__)

# ── Keyword lists ─────────────────────────────────────────────────────────────

_PERSONAL_PROFILE_KEYWORDS = frozenset([
    "my name is", "i was born", "birthday", "born on", "date of birth",
    "home address", "i live at", "my address", "phone number", "my phone",
    "social security", "passport number", "driver's license",
    "my age is", "years old", "my email is",
])

_RELATIONSHIP_KEYWORDS = frozenset([
    "my wife", "my husband", "my partner", "my girlfriend", "my boyfriend",
    "my kids", "my children", "my son", "my daughter", "my mom", "my dad",
    "my parents", "my sibling", "my brother", "my sister",
    "my best friend", "my friend", "my family",
])

# ── Embedding-prototype classifier config (Pass 2) ───────────────────────────
#
# WHY NOT NLI: the previous CrossEncoder zero-shot setup
# (cross-encoder/nli-MiniLM2-L6-H768 with raw entailment logits) classified
# plainly technical conversations as "personal profile information" at 0.93+
# even after per-pair softmax and hypothesis rewording — it blocked ALL real
# content (finding F22, docs/recommendations.md). Cosine similarity against
# per-category prototype sentences with the same MiniLM model used for
# embeddings separates cleanly (measured 2026-07-08 against the eval dataset).

_GATE_EMBED_MODEL_NAME = MODEL_NAME  # shared pinned local asset contract

_CATEGORY_PROTOTYPES: dict[str, list[str]] = {
    "personal_profile": [
        "My name is John Smith and I was born on March 3rd, 1985.",
        "I have been struggling with anxiety and I am seeing a therapist.",
        "Here is my home address and my phone number.",
        "I feel depressed lately and my health has been getting worse.",
        "I am sharing details about my identity, age, and where I live.",
    ],
    "relationship": [
        "My wife and I had an argument about money yesterday.",
        "My daughter is starting school next week and I am nervous.",
        "I am having trouble trusting my best friend after what happened.",
        "My relationship with my parents has always been complicated.",
    ],
    "general": [
        "How do I implement async functions in Python?",
        "Database indexing strategies for large PostgreSQL tables.",
        "Explain how the borrow checker works in this programming language.",
        "What are best practices for designing a REST API?",
        "Explain the theory of relativity in simple terms.",
        "How should I structure a marketing plan for a startup?",
    ],
}

# Decision-rule thresholds (privacy-first asymmetry):
# a sensitive category blocks at a LOW similarity bar; 'general' allows only at
# a HIGH bar with a clear margin over the best sensitive category.
_SENSITIVE_BLOCK_SIM = 0.25
_GENERAL_ALLOW_SIM = 0.35
_GENERAL_ALLOW_MARGIN = 0.15

# How much of the document to classify (MiniLM truncates ~256 tokens anyway;
# personal disclosures overwhelmingly appear early in conversations)
_CLASSIFY_WINDOW_CHARS = 2000

_gate_model = None  # SentenceTransformer, loaded lazily
_prototype_embeddings: dict[str, "object"] | None = None  # category → ndarray


def _get_gate_model():
    """Load the embedding model + prototype embeddings on first use.

    NOT thread-safe for concurrent first-loads (acceptable: personal-scale worker
    processes one job at a time via Semaphore(1)).
    """
    global _gate_model, _prototype_embeddings
    if _gate_model is None:
        try:
            model = load_local_model()
            prototypes = {
                category: model.encode(sentences, normalize_embeddings=True)
                for category, sentences in _CATEGORY_PROTOTYPES.items()
            }
        except Exception:
            logger.error("Failed to load sensitivity gate model: local_model_not_ready")
            raise LocalModelNotReady() from None
        # Publish only a complete initialization; failure must leave no cached
        # half-model that prevents a later call from retrying the prototypes.
        _gate_model, _prototype_embeddings = model, prototypes
        logger.info("Loaded sensitivity gate model: %s", _GATE_EMBED_MODEL_NAME)
    return _gate_model


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class SensitivityDecision:
    """Result of sensitivity gate classification.

    category: "personal_profile" | "relationship" | "unclassified" | "general"
    confidence: float in [0, 1] — how confident the classifier is
    blocked: True if content must NOT be sent to external provider
    method: "heuristic" | "embedding" — which pass made the decision
    """
    category: str
    confidence: float
    blocked: bool
    method: str = "heuristic"

    @property
    def is_allowed(self) -> bool:
        """True if content is safe to send to external provider."""
        return not self.blocked


def _decide_from_similarities(similarities: dict[str, float]) -> SensitivityDecision:
    """Turn per-category max-prototype similarities into a gate decision.

    Pure function (no model) so the privacy-critical decision rule is unit-
    testable. Asymmetric by design: blocking needs little evidence, allowing
    needs strong evidence AND a clear margin — unknown stays blocked (PRIV-05).
    """
    ranked = sorted(similarities.items(), key=lambda kv: -kv[1])
    top_category, top_sim = ranked[0]
    margin = top_sim - ranked[1][1]

    if top_category in ("personal_profile", "relationship") and top_sim >= _SENSITIVE_BLOCK_SIM:
        return SensitivityDecision(
            category=top_category,
            confidence=max(0.0, min(1.0, top_sim)),
            blocked=True,
            method="embedding",
        )

    if (
        top_category == "general"
        and top_sim >= _GENERAL_ALLOW_SIM
        and margin >= _GENERAL_ALLOW_MARGIN
    ):
        return SensitivityDecision(
            category="general",
            confidence=max(0.0, min(1.0, top_sim)),
            blocked=False,
            method="embedding",
        )

    return SensitivityDecision(
        category="unclassified",
        confidence=max(0.0, min(1.0, top_sim)),
        blocked=True,
        method="embedding",
    )


# ── Gate implementation ───────────────────────────────────────────────────────

class SensitivityGate:
    """Content sensitivity classifier.

    classify() is synchronous and uses keyword heuristics only (fast path).
    For full two-pass classification including the embedding pass, use classify_async().
    """

    def _keyword_check(self, text: str) -> SensitivityDecision | None:
        """Pass 1: fast keyword heuristics. Returns decision or None if inconclusive."""
        text_lower = text.lower()

        for keyword in _PERSONAL_PROFILE_KEYWORDS:
            if keyword in text_lower:
                return SensitivityDecision(
                    category="personal_profile",
                    confidence=1.0,
                    blocked=True,
                    method="heuristic",
                )

        for keyword in _RELATIONSHIP_KEYWORDS:
            if keyword in text_lower:
                return SensitivityDecision(
                    category="relationship",
                    confidence=1.0,
                    blocked=True,
                    method="heuristic",
                )

        return None  # inconclusive — proceed to embedding pass

    def _embedding_classify(self, text: str) -> SensitivityDecision:
        """Pass 2: embedding-prototype classification. Runs synchronously (call from to_thread)."""
        model = _get_gate_model()
        doc_embedding = model.encode(
            [text[:_CLASSIFY_WINDOW_CHARS]], normalize_embeddings=True
        )[0]
        similarities = {
            category: float((proto_emb @ doc_embedding).max())
            for category, proto_emb in _prototype_embeddings.items()
        }
        return _decide_from_similarities(similarities)

    def classify(self, text: str) -> SensitivityDecision:
        """Classify text sensitivity synchronously.

        Uses keyword heuristics only (fast path). For the embedding pass, use classify_async().
        This method is safe to call from sync context; does NOT load the embedding model.

        Returns SensitivityDecision with category, confidence, blocked, method.
        If heuristics are inconclusive, returns unclassified (blocked=True) by default.
        """
        # Pass 1: keyword heuristics
        heuristic_result = self._keyword_check(text)
        if heuristic_result is not None:
            return heuristic_result

        # If heuristics are inconclusive, return unclassified (blocked by default).
        # Full embedding classification requires classify_async().
        return SensitivityDecision(
            category="unclassified",
            confidence=0.0,
            blocked=True,
            method="heuristic",
        )

    async def classify_async(self, text: str) -> SensitivityDecision:
        """Classify text sensitivity with the full embedding pass via asyncio.to_thread().

        Use from worker loop — avoids blocking the event loop during model inference.
        Keyword heuristics still run first; the embedding pass only runs if heuristics are inconclusive.
        """
        # Pass 1: keyword heuristics (sync, instant)
        heuristic_result = self._keyword_check(text)
        if heuristic_result is not None:
            return heuristic_result

        # Pass 2: embedding inference in thread pool (CPU-bound — must not block event loop)
        try:
            return await asyncio.to_thread(self._embedding_classify, text)
        except LocalModelNotReady:
            # Unavailable local model is an execution failure, not a sensitivity
            # classification. Existing dispatchers fail the claim; never publish
            # policy-skipped output merely because the gate could not load.
            raise
        except Exception as e:
            logger.error(
                "Embedding classification failed: %s — defaulting to unclassified (blocked)", e
            )
            return SensitivityDecision(
                category="unclassified",
                confidence=0.0,
                blocked=True,
                method="heuristic",
            )
