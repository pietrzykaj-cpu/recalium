"""Pure ContextPacket v1 assembly over an existing retrieval response."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping

from app.domain.context_packets.contracts import (
    ContextPacket, ExcludedEvidence, ModelContext, PacketBudget, PacketIntegrity,
    PacketQuery, PacketRetrieval, PacketSource, SelectedEvidence,
)
from app.domain.retrieval.service import RetrievalItem, RetrievalResponse
from app.domain.retrieval.diagnostics import RetrievalDiagnostics

SCHEMA_VERSION = "recalium.context-packet.v1"
ATTRIBUTION_NOTICE = (
    "Inherited records are attributed evidence from prior interactions, not the "
    "current model's personal memories."
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if not isinstance(value, (list, tuple)):
        return []
    return [entry.strip() for entry in value if isinstance(entry, str) and entry.strip()]


def _stable_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _metadata_lists(provenance: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    sources: list[Mapping[str, Any]] = [provenance]
    source_metadata = provenance.get("source_metadata")
    if isinstance(source_metadata, Mapping):
        sources.append(source_metadata)
    unresolved: list[str] = []
    flags: list[str] = []
    for source in sources:
        unresolved.extend(_strings(source.get("unresolved_questions")))
        unresolved.extend(_strings(source.get("unresolved")))
        flags.extend(_strings(source.get("flags")))
    return _stable_unique(unresolved), _stable_unique(flags)


def _retrieval_metadata(
    item: RetrievalItem, provenance: Mapping[str, Any], *, mode: str, rank: int,
    diagnostics: RetrievalDiagnostics | None = None,
) -> PacketRetrieval:
    nested = provenance.get("retrieval")
    retrieval = nested if isinstance(nested, Mapping) else {}
    lexical_score = _number(retrieval.get("lexical_score"))
    semantic_score = _number(retrieval.get("semantic_score"))
    rrf_score = _number(retrieval.get("rrf_score"))
    lexical_rank = _positive_int(retrieval.get("lexical_rank"))
    semantic_rank = _positive_int(retrieval.get("semantic_rank"))
    if diagnostics is not None:
        diagnostic = next((entry for entry in diagnostics.candidates if entry.representative_id == item.id), None)
        if diagnostic is not None:
            if diagnostic.lexical is not None:
                lexical_score, lexical_rank = diagnostic.lexical.score, diagnostic.lexical.rank
            if diagnostic.semantic is not None:
                semantic_score, semantic_rank = diagnostic.semantic.score, diagnostic.semantic.rank
            if diagnostic.fused_score is not None:
                rrf_score = diagnostic.fused_score
    if mode == "keyword" and lexical_score is None:
        lexical_score, lexical_rank = item.score, rank
    elif mode == "semantic" and semantic_score is None:
        semantic_score, semantic_rank = item.score, rank
    return PacketRetrieval(
        final_score=item.score,
        relevance_rank=rank,
        lexical_score=lexical_score,
        lexical_rank=lexical_rank,
        semantic_score=semantic_score,
        semantic_rank=semantic_rank,
        hybrid_score=item.score if mode == "hybrid" else None,
        rrf_score=rrf_score,
        link_type=item.link_type,
    )


def _estimated_tokens(content: str, chars_per_token: int) -> int:
    return math.ceil(len(content) / chars_per_token) if content else 0


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_context_packet(
    response: RetrievalResponse,
    *,
    token_budget: int | None = None,
    chars_per_token: int = 4,
    provider: str | None = None,
    model: str | None = None,
    generated_at: datetime | None = None,
    diagnostics: RetrievalDiagnostics | None = None,
) -> ContextPacket:
    """Build a non-persistent packet without mutating the retrieval response."""
    if token_budget is not None and token_budget < 0:
        raise ValueError("token_budget must be non-negative")
    if chars_per_token < 1:
        raise ValueError("chars_per_token must be at least 1")

    created = generated_at or datetime.now(timezone.utc)
    selected: list[SelectedEvidence] = []
    excluded: list[ExcludedEvidence] = []
    warnings: list[str] = []
    unresolved_questions: list[str] = []
    flags: list[str] = []
    token_used = 0

    for rank, item in enumerate(response.items, start=1):
        if isinstance(item.provenance, Mapping):
            provenance: dict[str, Any] = deepcopy(dict(item.provenance))
        else:
            provenance = {}
            warnings.append(f"malformed_provenance:{item.id}")
        unresolved, item_flags = _metadata_lists(provenance)
        unresolved_questions.extend(unresolved)
        flags.extend(item_flags)

        item_tokens = _estimated_tokens(item.content, chars_per_token)
        if token_budget is not None and token_used + item_tokens > token_budget:
            excluded.append(ExcludedEvidence(
                memory_id=item.id,
                memory_type=item.type,
                relevance_rank=rank,
                reason="context_budget_exceeded",
                estimated_tokens=item_tokens,
            ))
            continue

        selected.append(SelectedEvidence(
            memory_id=item.id,
            memory_type=item.type,
            content=item.content,
            source=PacketSource(
                archive_id=item.source_id,
                system=item.source_system,
                captured_at=item.captured_at,
            ),
            provenance=provenance,
            retrieval=_retrieval_metadata(
                item, provenance, mode=response.retrieval_mode, rank=rank,
                diagnostics=diagnostics,
            ),
            conflict_label=item.conflict_label,
            source_fact_id=item.source_fact_id,
        ))
        token_used += item_tokens

    if response.degraded_mode:
        warnings.append("semantic_retrieval_degraded")
    if response.trimming_reason == "budget_met":
        warnings.append("upstream_retrieval_budget_excluded_unreported_candidates")

    budget = PacketBudget(
        unit="estimated_tokens" if token_budget is not None else "characters",
        limit=token_budget if token_budget is not None else response.budget_limit,
        used=token_used if token_budget is not None else response.budget_used,
        estimator=(
            f"character_ratio:{chars_per_token}"
            if token_budget is not None
            else "exact_characters"
        ),
        upstream_limit=response.budget_limit,
        upstream_used=response.budget_used,
        upstream_trimming_reason=response.trimming_reason,
    )
    model_context = (
        ModelContext(provider=provider, model=model)
        if provider is not None or model is not None else None
    )
    query = PacketQuery(text=response.query, mode_used=response.retrieval_mode)
    digest_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": created.isoformat(),
        "evidence_label": "evidence_from_prior_interactions",
        "attribution_notice": ATTRIBUTION_NOTICE,
        "query": query.model_dump(),
        "selected": [record.model_dump(mode="json") for record in selected],
        "excluded": [record.model_dump(mode="json") for record in excluded],
        "unresolved_questions": _stable_unique(unresolved_questions),
        "flags": _stable_unique(flags),
        "warnings": _stable_unique(warnings),
        "budget": budget.model_dump(),
        "model_context": model_context.model_dump() if model_context else None,
    }
    digest = _canonical_digest(digest_payload)
    return ContextPacket(
        packet_id=f"cpv1-{digest[:24]}",
        generated_at=created,
        attribution_notice=ATTRIBUTION_NOTICE,
        query=query,
        selected=selected,
        excluded=excluded,
        unresolved_questions=_stable_unique(unresolved_questions),
        flags=_stable_unique(flags),
        warnings=_stable_unique(warnings),
        budget=budget,
        model_context=model_context,
        integrity=PacketIntegrity(packet_digest=digest),
    )
