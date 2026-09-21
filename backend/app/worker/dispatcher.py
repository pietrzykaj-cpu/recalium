"""Job dispatcher — routes jobs through the processing pipeline.

PIPELINE ORDER (mandatory — gate fires first, ALWAYS):
  1. Sensitivity gate (classify_async) — BLOCKS if personal/relationship/unclassified
  2. Summarize + extract facts (LLM, if provider configured and not already done)
  3. Embed (local sentence-transformers — plan 05 wires this)
  4. FTS index
  5. Conflict detection (plan 06 wires this)

SECURITY:
  - Gate runs BEFORE any external provider call — no exceptions
  - Keys read from get_settings() at dispatch time — never from job record or DB
  - Provider keys never appear in job error_message or logs

BYOK-07: Invalid/rate-limited key → retryable_failed with error captured (not silent drop)
BYOK-08: Already-completed sub-jobs (summary exists) → skip without reprocessing
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from ipaddress import ip_address
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.jobs.models import Job
from app.domain.jobs.service import (
    complete_job,
    fail_job,
    set_pending_provider,
)
from app.domain.policy.gate import SensitivityGate
from app.infrastructure.settings import get_settings

logger = logging.getLogger(__name__)

# Sensitivity gate instance (shared; stateless after model load)
_gate = SensitivityGate()

# ── LLM prompts ───────────────────────────────────────────────────────────────

SUMMARIZATION_SYSTEM_PROMPT = """You are a conversation summarizer. Write a concise summary (2-4 sentences) of the key topics and outcomes in this conversation. Focus on information that would be useful to recall in a future session."""

FACT_EXTRACTION_SYSTEM_PROMPT = """You are a fact extraction engine. Extract factual statements from the provided text.

Scan all parts of the text, not just the beginning. Identify all factual claims
(not questions, hypotheticals, or meta-commentary). For each claim, find a verbatim
quote in the text as its source span; only include facts where source_span is an
exact substring of the text.

For each fact:
1. fact_text: single declarative sentence
2. source_span: EXACT quote from text (must be verbatim substring)
3. confidence_tier: "high" (explicit), "medium" (implied), "low" (uncertain)
4. entities: named entities (people, places, organizations, products)
5. tags: 1-3 topic tags (lower-case)

Return JSON:
{"facts": [
  {
    "fact_text": "User's name is Alice.",
    "source_span": "My name is Alice",
    "confidence_tier": "high",
    "entities": ["Alice"],
    "tags": ["identity"]
  }
]}

Return {"facts": []} if no facts can be extracted."""


# ── Provider routing ──────────────────────────────────────────────────────────

# F3: extraction quality drops sharply past the first turn — models (measured
# with qwen3.5:4b, eval run 3) extract mainly from the beginning of multi-turn
# conversations, and very long conversations exceed attention/token budgets.
# Split on turn boundaries and extract per chunk.
_EXTRACT_CHUNK_CHARS = 1200

def _split_conversation(text: str, max_chunk_chars: int = _EXTRACT_CHUNK_CHARS) -> list[str]:
    """Split a conversation into contiguous verbatim slices ≤ max_chunk_chars.

    Splits preferentially at turn boundaries (lines starting with User:/
    Assistant:/Human:/AI:), packing consecutive turns greedily. A single turn
    longer than max is hard-split. Every chunk is a contiguous slice of the
    original text, so extracted source_spans stay verbatim in the full source.
    """
    import re  # noqa: PLC0415

    if len(text) <= max_chunk_chars:
        return [text]

    boundaries = [0] + [
        m.start() for m in re.finditer(r"^(?:User|Assistant|Human|AI):", text, re.MULTILINE)
        if m.start() != 0
    ] + [len(text)]

    chunks: list[str] = []
    chunk_start = 0
    chunk_end = 0
    for i in range(len(boundaries) - 1):
        seg_start, seg_end = boundaries[i], boundaries[i + 1]
        seg_len = seg_end - seg_start

        if seg_len > max_chunk_chars:
            # Oversized single turn: flush accumulator, then hard-split it
            if chunk_end > chunk_start:
                chunks.append(text[chunk_start:chunk_end])
            pos = seg_start
            while seg_end - pos > max_chunk_chars:
                chunks.append(text[pos:pos + max_chunk_chars])
                pos += max_chunk_chars
            if pos < seg_end:
                chunks.append(text[pos:seg_end])
            chunk_start = chunk_end = seg_end
        elif (chunk_end - chunk_start) + seg_len > max_chunk_chars:
            # Adding this turn would overflow: flush, start new chunk with it
            if chunk_end > chunk_start:
                chunks.append(text[chunk_start:chunk_end])
            chunk_start, chunk_end = seg_start, seg_end
        else:
            chunk_end = seg_end

    if chunk_end > chunk_start:
        chunks.append(text[chunk_start:chunk_end])

    return [c for c in chunks if c.strip()]


def _dedupe_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop facts whose normalized fact_text was already produced (chunk overlap)."""
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for fact in facts:
        key = " ".join((fact.get("fact_text") or "").lower().split()).rstrip(".")
        if key and key not in seen:
            seen.add(key)
            result.append(fact)
    return result


def _validate_spans(facts: list[dict[str, Any]], raw_source: str) -> list[dict[str, Any]]:
    """F4: verify each source_span is a verbatim substring of the raw source.

    Hallucinated spans poison provenance (the product differentiator). For a span
    that is non-empty but not a verbatim substring of the source, clear it and
    downgrade confidence to 'low' — the fact is kept but flagged as unverified via
    the empty-span rule in write_facts(). Local substring check; no LLM cost.
    """
    for fact in facts:
        span = (fact.get("source_span") or "").strip()
        if span and span not in raw_source:
            logger.info(
                "F4: cleared unverifiable source_span (%d chars) — not verbatim in source",
                len(span),
            )
            fact["source_span"] = ""
            fact["confidence_tier"] = "low"
    return facts


def _parse_json_object(raw: str) -> dict:
    """Parse the first JSON object out of model output.

    Local models decorate JSON with markdown fences or trailing junk (observed:
    valid JSON followed by a bare closing fence). Decode from the first '{'
    and ignore anything after the object.
    """
    start = raw.find("{")
    if start == -1:
        raise json.JSONDecodeError("no JSON object in output", raw, 0)
    obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    if not isinstance(obj, dict):
        raise json.JSONDecodeError("top-level JSON value is not an object", raw, 0)
    return obj


class _OllamaFact(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    fact_text: str
    source_span: str
    confidence_tier: Literal["high", "medium", "low"]
    entities: list[str]
    tags: list[str]


class _OllamaExtraction(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    facts: list[_OllamaFact]


def _ollama_final_content(raw: str) -> str:
    """Separate explicit thinking blocks, including a template-prefilled opener.

    Do not guess from prose such as "Okay" or remove tag literals inside JSON.
    An orphan closing tag must occupy its own line, as in Qwen's native output.
    """
    content = raw.strip()
    while content:
        if content.startswith("<think>"):
            end = content.find("</think>", len("<think>"))
            if end < 0:
                raise ValueError("Ollama returned an unfinished thinking block")
            content = content[end + len("</think>"):].strip()
            continue
        # Valid structured output can contain literal protocol tokens as data.
        try:
            json.loads(content)
        except json.JSONDecodeError:
            closing = re.search(r"(?m)^[ \t]*</think>[ \t]*(?:\r?\n|$)", content)
            if closing:
                content = content[closing.end():].strip()
                continue
        break
    if not content:
        raise ValueError("Ollama returned no final answer")
    return content


def _parse_ollama_facts(raw: str) -> list[dict[str, Any]]:
    """A missing/invalid facts envelope is a provider failure, never zero facts."""
    try:
        content = raw.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL)
        obj = json.loads(fenced.group(1) if fenced else content)
    except json.JSONDecodeError:
        raise ValueError("Ollama extraction returned invalid JSON") from None
    try:
        result = _OllamaExtraction.model_validate(obj)
    except ValidationError:
        # Do not include Pydantic's input values (memory text) in job errors.
        raise ValueError("Ollama extraction response does not match the facts schema") from None
    logger.info("Ollama extraction returned %d facts", len(result.facts))
    return [fact.model_dump() for fact in result.facts]


async def _ollama_chat(
    system: str, user: str, *, format_json: bool = False, allow_external: bool = True,
) -> str:
    """Call Ollama's native API and return only its final answer.

    A thinking-only Qwen checkpoint can emit reasoning despite think=False.
    Preserve the working request mode (native thinking with structured output
    can be very slow), and separate its explicit protocol boundary on return.
    """
    import httpx  # noqa: PLC0415

    settings = get_settings()
    if not allow_external and not _is_local_ollama_url(settings.ollama_base_url):
        raise PermissionError("Policy forbids remote Ollama processing")
    payload: dict[str, Any] = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
    }
    if format_json:
        payload["format"] = _OllamaExtraction.model_json_schema()
    headers = (
        {"Authorization": f"Bearer {settings.ollama_api_key}"}
        if settings.ollama_api_key
        else None
    )
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=10.0),
        follow_redirects=False, trust_env=allow_external,
    ) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return _ollama_final_content(resp.json().get("message", {}).get("content") or "")

async def _run_summarize_job(text: str, *, allow_external: bool = True) -> str | None:
    """Run LLM summarization via the configured provider/model (F1/F2).

    Reads API keys from settings at call time (never from DB or job record).
    Returns None if no provider is available (caller marks pending_provider).
    Raises exception on API error (caller converts to retryable_failed).
    """
    settings = get_settings()
    provider = _resolve_provider(settings.summarize_provider)
    if provider is None or not _provider_allowed(provider, allow_external=allow_external):
        return None
    model = _resolve_model(provider, settings.summarize_model)

    if provider == "openai":
        from openai import AsyncOpenAI  # noqa: PLC0415
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=512,
        )
        return response.choices[0].message.content

    if provider == "anthropic":
        from anthropic import AsyncAnthropic  # noqa: PLC0415
        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        response = await client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0,
            messages=[{"role": "user", "content": text}],
            system=SUMMARIZATION_SYSTEM_PROMPT,
        )
        return response.content[0].text

    # provider == "ollama"
    return await _ollama_chat(
        SUMMARIZATION_SYSTEM_PROMPT, text, allow_external=allow_external,
    )


async def _run_extract_job(text: str, *, allow_external: bool = True) -> list[dict[str, Any]]:
    """Run LLM fact extraction over turn-boundary chunks (F3).

    Models extract predominantly from the beginning of multi-turn
    conversations; chunking recovers facts from later turns and bounds input
    size for very long conversations. Facts are merged and deduped.

    Returns [] (not None) when no provider is configured — FTS still runs.
    Raises exception on API error (caller converts to retryable_failed).
    """
    settings = get_settings()
    if not _provider_allowed(
        _resolve_provider(settings.extract_provider), allow_external=allow_external,
    ):
        return []  # No provider available for extraction

    facts: list[dict[str, Any]] = []
    for chunk in _split_conversation(text):
        facts.extend(await _extract_chunk(chunk, allow_external=allow_external))
    returned_count = len(facts)
    facts = _validate_spans(facts, text)  # F4: preserve existing provenance checks
    facts = _dedupe_facts(facts)
    logger.info("Extraction facts: returned=%d retained=%d", returned_count, len(facts))
    return facts


async def _extract_chunk(text: str, *, allow_external: bool = True) -> list[dict[str, Any]]:
    """Single-call fact extraction for one chunk via the configured provider/model (F1/F2)."""
    settings = get_settings()
    provider = _resolve_provider(settings.extract_provider)
    if provider is None or not _provider_allowed(provider, allow_external=allow_external):
        return []
    model = _resolve_model(provider, settings.extract_model)

    if provider == "openai":
        from openai import AsyncOpenAI  # noqa: PLC0415
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FACT_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        data = json.loads(response.choices[0].message.content or "{}")
        return data.get("facts", [])

    if provider == "anthropic":
        from anthropic import AsyncAnthropic  # noqa: PLC0415
        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        response = await client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0,
            messages=[{"role": "user", "content": text}],
            system=FACT_EXTRACTION_SYSTEM_PROMPT,
        )
        raw = response.content[0].text
        try:
            return _parse_json_object(raw).get("facts", [])
        except json.JSONDecodeError:
            logger.warning("Anthropic returned non-JSON facts response: %s", raw[:200])
            return []

    # provider == "ollama"
    raw = await _ollama_chat(
        FACT_EXTRACTION_SYSTEM_PROMPT, text, format_json=True, allow_external=allow_external,
    )
    return _parse_ollama_facts(raw)


async def _run_link_detection_job(
    session: AsyncSession,
    raw_archive_id: "uuid.UUID",
    allow_external: bool = True,
) -> None:
    """Build memory links for all facts derived from raw_archive_id.

    Three passes — each non-fatal:
      Pass A (semantic) — for each new fact, find up to 5 semantically similar
                          facts from other archive items (cosine via pgvector),
                          create link_type='related'.
      Pass B (LLM typed) — ask LLM to classify the top-5 pairs as
                          supports|elaborates|contradicts|unrelated.
                          Only runs if an LLM provider is configured AND the
                          resolved policy allows external calls (GPT5.6 #6).
      Pass C (entity)   — for each entity in the extracted tags, find other
                          active facts that contain the same entity string,
                          create link_type='entity'.

    All DB writes go via write_links() which deduplicates triplets.
    """
    import uuid  # noqa: PLC0415

    from sqlalchemy import select, text as sa_text  # noqa: PLC0415
    from app.domain.derived_memory.models import Fact, Embedding  # noqa: PLC0415
    from app.domain.derived_memory.service import write_links  # noqa: PLC0415
    from app.domain.audit.models import AuditEvent  # noqa: PLC0415

    # Load facts that belong to this archive item
    fact_rows = (await session.execute(
        select(Fact)
        .where(Fact.raw_archive_id == raw_archive_id)
        .where(Fact.source_status == "active")
    )).scalars().all()

    if not fact_rows:
        return

    # ── Pass A: semantic links via embedding similarity ──────────────────────
    embedding_row = (await session.execute(
        select(Embedding)
        .where(Embedding.raw_archive_id == raw_archive_id)
        .where(Embedding.source_status == "active")
        .limit(1)
    )).scalar_one_or_none()

    semantic_fact_ids: list[uuid.UUID] = []
    if embedding_row is not None:
        # Find up to 5 active facts from other archives with similar embeddings
        rows = (await session.execute(
            sa_text(
                """
                SELECT f.id
                FROM facts f
                JOIN embeddings e ON e.raw_archive_id = f.raw_archive_id
                WHERE f.source_status = 'active'
                  AND e.source_status = 'active'
                  AND f.raw_archive_id != :this_id
                ORDER BY e.embedding <=> CAST(:vec AS vector)
                LIMIT 5
                """
            ),
            {
                "this_id": str(raw_archive_id),
                # numpy str() is space-separated — not valid pgvector input
                "vec": str(
                    embedding_row.embedding.tolist()
                    if hasattr(embedding_row.embedding, "tolist")
                    else list(embedding_row.embedding)
                ),
            },
        )).fetchall()
        semantic_fact_ids = [row[0] for row in rows]

    links_to_write: list[dict] = []
    for src_fact in fact_rows:
        for tgt_fact_id in semantic_fact_ids:
            links_to_write.append({
                "source_fact_id": src_fact.id,
                "target_fact_id": tgt_fact_id,
                "link_type": "related",
                "confidence": 0.7,
                "created_by": "pipeline_semantic",
                "entity_name": None,
            })

    if links_to_write:
        await write_links(session, links_to_write)
        links_to_write = []

    # ── Pass B: LLM typed classification of top-5 semantic pairs ─────────────
    # GPT5.6 #6: Pass B egresses fact text to an external LLM, so it is gated on the
    # resolved policy — a local_only mode or a sensitive-content decision blocks it.
    if allow_external and semantic_fact_ids and _has_llm_provider():
        # Load target fact texts for the LLM prompt
        tgt_facts = (await session.execute(
            select(Fact)
            .where(Fact.id.in_(semantic_fact_ids))
            .where(Fact.source_status == "active")
        )).scalars().all()
        tgt_by_id = {f.id: f for f in tgt_facts}

        for src_fact in fact_rows[:3]:  # limit to 3 source facts per archive
            for tgt_id in semantic_fact_ids[:5]:
                tgt_fact = tgt_by_id.get(tgt_id)
                if tgt_fact is None:
                    continue
                try:
                    link_type = await _classify_link_pair(src_fact.fact_text, tgt_fact.fact_text)
                    if link_type and link_type != "unrelated":
                        links_to_write.append({
                            "source_fact_id": src_fact.id,
                            "target_fact_id": tgt_fact.id,
                            "link_type": link_type,
                            "confidence": 0.8,
                            "created_by": "pipeline_llm",
                            "entity_name": None,
                        })
                except Exception as exc:
                    logger.debug("LLM link classification failed (non-fatal): %s", exc)
                    # F5: make the (non-fatal) failure observable in the audit log
                    session.add(AuditEvent(
                        event_type="link_detection_error",
                        raw_archive_id=raw_archive_id,
                        actor="pipeline_worker",
                        operation_metadata={
                            "pass": "llm_typed",
                            "source_fact_id": str(src_fact.id),
                            "target_fact_id": str(tgt_fact.id),
                            "error_reason": str(exc)[:500],
                        },
                    ))

        if links_to_write:
            await write_links(session, links_to_write)
            links_to_write = []

    # ── Pass C: entity co-mention links ──────────────────────────────────────
    # Load entity names from fact_tags for this archive's facts
    from app.domain.derived_memory.models import MemoryLink as _ML  # noqa: PLC0415, F401

    entity_rows = (await session.execute(
        sa_text(
            """
            SELECT DISTINCT ml.entity_name
            FROM memory_links ml
            WHERE ml.link_type = 'entity'
              AND ml.entity_name IS NOT NULL
              AND ml.source_fact_id = ANY(:fact_ids)
            LIMIT 20
            """
        ),
        {"fact_ids": [str(f.id) for f in fact_rows]},
    )).fetchall()
    # entity_rows from pass C won't exist yet on first run — skip if empty
    # Instead, rebuild entity links from the tags stored during extraction
    # (entities are stored as tag rows with prefix 'entity:')
    entity_tags_rows = (await session.execute(
        sa_text(
            """
            SELECT DISTINCT t.name
            FROM tags t
            JOIN fact_tags ft ON ft.tag_id = t.id
            JOIN facts f ON f.id = ft.fact_id
            WHERE f.raw_archive_id = :this_id
              AND f.source_status = 'active'
              AND t.name LIKE 'entity:%'
            """
        ),
        {"this_id": str(raw_archive_id)},
    )).fetchall()

    for (entity_tag_name,) in entity_tags_rows:
        entity_name = entity_tag_name[len("entity:"):]
        if not entity_name:
            continue
        # Find other facts that also have this entity tag
        other_fact_rows = (await session.execute(
            sa_text(
                """
                SELECT DISTINCT f.id
                FROM facts f
                JOIN fact_tags ft ON ft.fact_id = f.id
                JOIN tags t ON t.id = ft.tag_id
                WHERE t.name = :entity_tag
                  AND f.source_status = 'active'
                  AND f.raw_archive_id != :this_id
                LIMIT 10
                """
            ),
            {"entity_tag": entity_tag_name, "this_id": str(raw_archive_id)},
        )).fetchall()

        for src_fact in fact_rows:
            for (tgt_fact_id,) in other_fact_rows:
                links_to_write.append({
                    "source_fact_id": src_fact.id,
                    "target_fact_id": tgt_fact_id,
                    "link_type": "entity",
                    "confidence": 1.0,
                    "created_by": "pipeline_entity",
                    "entity_name": entity_name,
                })

    if links_to_write:
        await write_links(session, links_to_write)


async def _classify_link_pair(source_text: str, target_text: str) -> str | None:
    """Ask LLM to classify the typed relationship between two fact texts.

    Returns one of: 'supports', 'elaborates', 'contradicts', 'unrelated', or None on error.
    Prompt is minimal to reduce token usage.
    """
    prompt = (
        f"Fact A: {source_text}\n"
        f"Fact B: {target_text}\n\n"
        "How does Fact B relate to Fact A? "
        "Reply with exactly one word: supports, elaborates, contradicts, or unrelated."
    )
    settings = get_settings()
    provider = _resolve_provider(settings.extract_provider)
    if provider is None:
        return None
    model = _resolve_model(provider, settings.extract_model)

    try:
        if provider == "openai":
            from openai import AsyncOpenAI  # noqa: PLC0415
            client = AsyncOpenAI(api_key=settings.openai_api_key)
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5,
            )
            return resp.choices[0].message.content.strip().lower() or None

        if provider == "anthropic":
            from anthropic import AsyncAnthropic  # noqa: PLC0415
            client = AsyncAnthropic(api_key=settings.anthropic_api_key)
            resp = await client.messages.create(
                model=model,
                max_tokens=5,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip().lower() or None

        # provider == "ollama"
        raw = await _ollama_chat(
            "You classify relationships between facts. Reply with exactly one word.",
            prompt,
        )
        word = raw.strip().lower().split()[0] if raw.strip() else ""
        return word if word in ("supports", "elaborates", "contradicts", "unrelated") else None
    except Exception as exc:
        logger.debug("_classify_link_pair failed: %s", exc)
        raise


_PROVIDER_DEFAULT_MODEL = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
}


def _is_local_ollama_url(url: str | None) -> bool:
    """Trust only loopback and Docker Desktop's host gateway, never arbitrary LAN URLs."""
    try:
        parsed = urlsplit(url or "")
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
            return False
        # Accessing port also rejects malformed/out-of-range ports.
        parsed.port
        if parsed.hostname in {"localhost", "host.docker.internal"}:
            return True
        return ip_address(parsed.hostname).is_loopback
    except ValueError:
        return False


def _provider_allowed(provider: str | None, *, allow_external: bool) -> bool:
    return provider is not None and (
        allow_external
        or (provider == "ollama" and _is_local_ollama_url(get_settings().ollama_base_url))
    )


def _resolve_provider(configured: str) -> str | None:
    """Resolve a configured provider ("auto" | name) to an available provider (F2).

    "auto" falls back to the first configured key (openai → anthropic → ollama).
    An explicit provider whose key is missing returns None so the caller degrades
    transparently (pending_provider) instead of silently using another provider.
    """
    settings = get_settings()
    available = {
        "openai": bool(settings.openai_api_key),
        "anthropic": bool(settings.anthropic_api_key),
        "ollama": bool(settings.ollama_base_url),
    }
    choice = (configured or "auto").strip().lower()
    if choice != "auto":
        return choice if available.get(choice) else None
    for name in ("openai", "anthropic", "ollama"):
        if available[name]:
            return name
    return None


def _resolve_model(provider: str, configured_model: str) -> str:
    """Resolve the model name for a provider (F1). "auto"/empty → provider default."""
    if provider == "ollama":
        return get_settings().ollama_model
    model = (configured_model or "auto").strip()
    if model and model.lower() != "auto":
        return model
    return _PROVIDER_DEFAULT_MODEL.get(provider, "gpt-4o-mini")


def _provider_name() -> str:
    """Model used for extraction (recorded in the fact derivation_model field)."""
    settings = get_settings()
    provider = _resolve_provider(settings.extract_provider)
    if provider is None:
        return "none"
    return _resolve_model(provider, settings.extract_model)


def _active_provider_label() -> str | None:
    """The selected provider (openai|anthropic|ollama) that would service LLM calls.

    Distinct from ``_provider_name()``, which returns the MODEL label stored in
    ``fact.derivation_model``. The policy-decision audit records the *provider*, not
    the model (GPT5.6 #6). Reports the extraction provider (which also drives link
    Pass B), falling back to the summarization provider.
    """
    settings = get_settings()
    return (
        _resolve_provider(settings.extract_provider)
        or _resolve_provider(settings.summarize_provider)
    )


def _has_llm_provider() -> bool:
    """True if any LLM provider key is configured."""
    settings = get_settings()
    return bool(settings.openai_api_key or settings.anthropic_api_key or settings.ollama_base_url)


# ── Main dispatcher ───────────────────────────────────────────────────────────

async def dispatch_job(session: AsyncSession, job: Job) -> None:
    """Route a claimed job through the full pipeline.

    Pipeline order (enforced — gate MUST fire before any external call):
      1. Load raw archive content
      2. Sensitivity gate (blocks personal/relationship/unclassified)
      3. LLM summarize + extract (each provider authorized independently)
      4. FTS indexing (always runs — local, no external call)
      5. Embeddings and conflict detection wired in plan 05–06

    On any provider error: job → retryable_failed with error captured (BYOK-07).
    On missing provider: job → pending_provider (amber badge, not failure).
    On blocked content: job → completed (no LLM output, FTS still runs).
    """
    from sqlalchemy import select  # noqa: PLC0415
    from app.domain.archive.models import RawArchiveItem  # noqa: PLC0415
    from app.domain.derived_memory.service import (  # noqa: PLC0415
        write_summary,
        write_facts,
        write_fts_entry,
        write_tags,
        get_existing_summary,
    )

    # ── Step 1: Load raw archive content ────────────────────────────────────
    result = await session.execute(
        select(RawArchiveItem)
        .where(RawArchiveItem.id == job.raw_archive_id)
        .where(RawArchiveItem.deleted_at.is_(None))
    )
    archive_item = result.scalar_one_or_none()

    if archive_item is None:
        await fail_job(
            session, job,
            error=f"Raw archive item {job.raw_archive_id} not found or deleted",
            retryable=False,
        )
        return

    raw_text = archive_item.raw_content

    # ── Step 2: Sensitivity gate — MUST run before any external call ────────
    try:
        sensitivity_decision = await _gate.classify_async(raw_text)
    except Exception as e:
        # Gate failure → treat as unclassified (blocked) — fail-safe
        logger.error("Sensitivity gate failed for job %s: %s — treating as blocked", job.id, e)
        await fail_job(
            session, job,
            error=f"Sensitivity gate error: {e}",
            retryable=True,
        )
        return

    logger.info(
        "Sensitivity gate: job=%s category=%s confidence=%.2f blocked=%s",
        job.id, sensitivity_decision.category, sensitivity_decision.confidence,
        sensitivity_decision.blocked,
    )

    # F15: the gate decision must be externally observable (audit trail), not
    # just logged — evals and users verify the privacy promise through this.
    try:
        from app.domain.audit.models import AuditEvent  # noqa: PLC0415
        session.add(AuditEvent(
            event_type="sensitivity_gate",
            raw_archive_id=job.raw_archive_id,
            actor="pipeline_worker",
            operation_metadata={
                "category": sensitivity_decision.category,
                "confidence": round(sensitivity_decision.confidence, 4),
                "blocked": sensitivity_decision.blocked,
                "method": sensitivity_decision.method,
            },
        ))
        # GPT5.6 #4: persist the gate data_class onto the item so retrieval can
        # filter by category at the SQL layer (the category filter was ignored).
        archive_item.metadata_json = {
            **(archive_item.metadata_json or {}),
            "data_class": sensitivity_decision.category,
        }
        await session.commit()
    except Exception as e:
        logger.warning("Failed to write sensitivity_gate audit event (non-fatal): %s", e)
        await session.rollback()
        await session.refresh(job)

    # ── Step 2b: Resolve effective policy (gate + caller processing intent) ──
    # GPT5.6 #6: honor the caller-declared processing_mode/sensitivity_hint
    # (stored on the item's metadata, e.g. by MCP ingest), defaulting to
    # stricter — a local_only mode or a sensitive hint forbids external calls
    # even when the content gate would have allowed them.
    from app.domain.policy.resolver import resolve_effective_policy  # noqa: PLC0415

    item_metadata = archive_item.metadata_json or {}
    effective_policy = resolve_effective_policy(
        gate_allows=sensitivity_decision.is_allowed,
        data_class=sensitivity_decision.category,
        processing_mode=item_metadata.get("processing_mode"),
        sensitivity_hint=item_metadata.get("sensitivity_hint"),
    )
    logger.info(
        "Policy: job=%s allow_external=%s mode=%s hint=%s data_class=%s",
        job.id, effective_policy.allow_external, effective_policy.processing_mode,
        effective_policy.sensitivity_hint, effective_policy.data_class,
    )
    settings = get_settings()
    operation_providers = {
        "summarize": _resolve_provider(settings.summarize_provider),
        "extract": _resolve_provider(settings.extract_provider),
    }
    operation_permissions = {
        operation: {
            "provider": provider,
            "local": provider == "ollama" and _is_local_ollama_url(settings.ollama_base_url),
            "allowed": _provider_allowed(provider, allow_external=effective_policy.allow_external),
        }
        for operation, provider in operation_providers.items()
    }
    allowed_providers = {
        decision["provider"] for decision in operation_permissions.values() if decision["allowed"]
    }
    try:
        from app.domain.audit.models import AuditEvent  # noqa: PLC0415
        session.add(AuditEvent(
            event_type="policy_decision",
            raw_archive_id=job.raw_archive_id,
            actor="pipeline_worker",
            operation_metadata={
                "allow_external": effective_policy.allow_external,
                "processing_mode": effective_policy.processing_mode,
                "sensitivity_hint": effective_policy.sensitivity_hint,
                "data_class": effective_policy.data_class,
                "provider": next(iter(allowed_providers)) if len(allowed_providers) == 1 else None,
                "operations": operation_permissions,
                "reason": effective_policy.reason,
            },
        ))
        await session.commit()
        policy_audit_recorded = True
    except Exception as e:
        logger.error("Failed to write policy_decision audit event: %s", e)
        await session.rollback()
        await session.refresh(job)
        policy_audit_recorded = False

    # Fail closed for both local and external LLMs: authorization must be durable
    # before processing. The audit records permissions, not completed calls.
    if (effective_policy.allow_external or allowed_providers) and not policy_audit_recorded:
        await fail_job(
            session, job,
            error="Policy decision could not be recorded; LLM processing blocked (fail-closed).",
            retryable=True,
        )
        return

    # ── Step 3: LLM summarize + extract (only if policy allows AND provider configured) ──
    if effective_policy.allow_external or allowed_providers:
        if not _has_llm_provider():
            # No provider — run FTS first (local, no LLM needed), then mark pending_provider
            try:
                existing_summary_fts = await get_existing_summary(session, job.raw_archive_id)
                fts_text = existing_summary_fts.summary_text if existing_summary_fts else raw_text[:10000]
                await write_fts_entry(
                    session,
                    raw_archive_id=job.raw_archive_id,
                    text_content=fts_text,
                )
                logger.debug("Wrote FTS entry for job %s (no LLM provider)", job.id)
            except Exception as e:
                logger.warning("FTS indexing failed for job %s (non-fatal): %s", job.id, e)
            await set_pending_provider(
                session, job,
                reason="No LLM provider configured. Add an OpenAI, Anthropic, or Ollama key in Settings.",
            )
            return

        # BYOK-08: Check if summary already exists — skip if present
        existing_summary = await get_existing_summary(session, job.raw_archive_id)

        if existing_summary is None and operation_permissions["summarize"]["allowed"]:
            try:
                summary_text = await _run_summarize_job(
                    raw_text, allow_external=effective_policy.allow_external,
                )
                if summary_text:
                    await write_summary(
                        session,
                        raw_archive_id=job.raw_archive_id,
                        summary_text=summary_text,
                        model_used=_resolve_model(
                            operation_providers["summarize"], settings.summarize_model,
                        ),
                        derivation_method="llm_summarization",
                    )
                    logger.debug("Wrote summary for job %s", job.id)
            except Exception as e:
                error_type = type(e).__name__
                await session.rollback()  # clear any aborted tx so the status write succeeds
                await session.refresh(job)  # rollback expired the instance
                await fail_job(
                    session, job,
                    error=f"{error_type}: {str(e)[:500]}",
                    retryable=True,
                )
                return

        # Extract facts
        try:
            facts_data = (
                await _run_extract_job(raw_text, allow_external=effective_policy.allow_external)
                if operation_permissions["extract"]["allowed"] else []
            )
            enriched_facts = [
                {**f, "derivation_method": "llm_extraction", "derivation_model": _provider_name()}
                for f in facts_data
            ]
            if enriched_facts:
                created_facts = await write_facts(
                    session,
                    raw_archive_id=job.raw_archive_id,
                    facts_data=enriched_facts,
                )
                logger.debug("Wrote %d facts for job %s", len(enriched_facts), job.id)

                # Persist tags and entity tags extracted alongside each fact
                for fact_obj, fact_raw in zip(created_facts, enriched_facts):
                    tags: list[str] = fact_raw.get("tags") or []
                    entities: list[str] = fact_raw.get("entities") or []
                    # Store entities as special 'entity:<name>' tags for pass-C link detection
                    entity_tags = [f"entity:{e.strip().lower()}" for e in entities if e.strip()]
                    all_tags = tags + entity_tags
                    if all_tags:
                        try:
                            await write_tags(session, fact_obj.id, all_tags)
                        except Exception as tag_exc:
                            logger.warning(
                                "Tag persistence failed for fact %s (non-fatal): %s",
                                fact_obj.id, tag_exc,
                            )
        except Exception as e:
            error_type = type(e).__name__
            await session.rollback()  # clear any aborted tx so the status write succeeds
            await session.refresh(job)  # rollback expired the instance
            await fail_job(
                session, job,
                error=f"{error_type}: {str(e)[:500]}",
                retryable=True,
            )
            return

    # ── Step 4: FTS indexing (always runs — local, no external call) ─────────
    # Use summary text if available, otherwise raw content
    try:
        existing_summary = await get_existing_summary(session, job.raw_archive_id)
        fts_text = existing_summary.summary_text if existing_summary else raw_text[:10000]
        await write_fts_entry(
            session,
            raw_archive_id=job.raw_archive_id,
            text_content=fts_text,
        )
        logger.debug("Wrote FTS entry for job %s", job.id)
    except Exception as e:
        logger.warning("FTS indexing failed for job %s (non-fatal): %s", job.id, e)
        await session.rollback()  # aborted tx would wedge every later step
        await session.refresh(job)  # rollback expired the instance
        # FTS failure is non-fatal — job still completes

    # ── Step 5: Embeddings (local sentence-transformers — no API key needed) ──
    try:
        from app.domain.derived_memory.service import (  # noqa: PLC0415
            embed_text,
            write_embedding,
            get_existing_embedding,
        )

        existing_embedding = await get_existing_embedding(session, job.raw_archive_id)
        if existing_embedding is None:
            # Use summary text if available (more condensed signal), else raw content
            existing_summary_for_embed = await get_existing_summary(session, job.raw_archive_id)
            embed_source = (
                existing_summary_for_embed.summary_text
                if existing_summary_for_embed
                else raw_text[:10000]
            )
            vector = await embed_text(embed_source)
            await write_embedding(
                session,
                raw_archive_id=job.raw_archive_id,
                vector=vector,
            )
            logger.debug("Wrote embedding for job %s (%d dims)", job.id, len(vector))
        else:
            logger.debug("Embedding already exists for job %s — skipping (BYOK-08)", job.id)
    except Exception as e:
        # Embedding failure is non-fatal (local model) — log and continue to completion
        logger.warning("Embedding step failed for job %s (non-fatal): %s", job.id, e)
        await session.rollback()  # aborted tx would wedge every later step
        await session.refresh(job)  # rollback expired the instance

    # ── Step 6: Conflict detection (CANM-06 / GPT5.6 #10) ────────────────────
    try:
        from app.domain.conflict_detection import detect_and_group_duplicates  # noqa: PLC0415
        from app.domain.derived_memory.service import get_existing_embedding  # noqa: PLC0415
        current_embedding = await get_existing_embedding(session, job.raw_archive_id)
        if current_embedding is not None:
            detection = await detect_and_group_duplicates(
                session,
                raw_archive_id=job.raw_archive_id,
                embedding_id=current_embedding.id,
                embedding=current_embedding.embedding,
            )
            if detection is not None:
                logger.info(
                    "Conflict group %s: job=%s duplicates=%d linked_facts=%d",
                    detection.group.id, job.id,
                    len(detection.duplicate_archive_ids), detection.linked_fact_count,
                )
                # Surface the duplicate in the audit trail (the group itself
                # surfaces in the review queue via linked facts).
                try:
                    from app.domain.audit.models import AuditEvent  # noqa: PLC0415
                    session.add(AuditEvent(
                        event_type="conflict_detected",
                        raw_archive_id=job.raw_archive_id,
                        actor="pipeline_worker",
                        operation_metadata={
                            "conflict_group_id": str(detection.group.id),
                            "group_type": "duplicate",
                            "duplicate_archive_ids": [
                                str(a) for a in detection.duplicate_archive_ids
                            ],
                            "linked_fact_count": detection.linked_fact_count,
                        },
                    ))
                    await session.commit()
                except Exception as audit_exc:
                    logger.warning(
                        "conflict_detected audit write failed (non-fatal): %s", audit_exc
                    )
                    await session.rollback()
                    await session.refresh(job)
    except Exception as e:
        logger.warning("Conflict detection failed for job %s (non-fatal): %s", job.id, e)
        await session.rollback()  # aborted tx would wedge every later step
        await session.refresh(job)  # rollback expired the instance

    # ── Step 7: Link detection (passes A/B/C) ────────────────────────────────
    try:
        await _run_link_detection_job(
            session,
            job.raw_archive_id,
            allow_external=effective_policy.allow_external,
        )
        logger.debug("Link detection complete for job %s", job.id)
    except Exception as e:
        logger.warning("Link detection failed for job %s (non-fatal): %s", job.id, e)
        await session.rollback()  # aborted tx would wedge every later step
        await session.refresh(job)  # rollback expired the instance

    # ── Step 8: Deletion-race reconcile (GPT5.6 #9) ──────────────────────────
    # The source may have been deleted while this job ran its (slow, external)
    # pipeline. Any derivations written after that deletion would still be active
    # and resurrect the removed content. Under a FOR UPDATE lock that serializes
    # with cascade_delete_archive_item, re-suppress + crypto-erase all derivations
    # if the source is now deleted. Non-fatal: the job still completes.
    try:
        from app.domain.archive.service import (  # noqa: PLC0415
            suppress_new_derivations_if_deleted,
        )
        await suppress_new_derivations_if_deleted(session, job.raw_archive_id)
    except Exception as e:
        logger.warning(
            "Deletion-race reconcile failed for job %s (non-fatal): %s", job.id, e
        )
        await session.rollback()
        await session.refresh(job)

    await complete_job(session, job)
    logger.info("Completed job %s", job.id)
