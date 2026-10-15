"""Derived memory domain service — writes summaries, facts, FTS entries, and embeddings.

SECURITY: No API keys here. This service only writes derived data to DB.
PIPE-02: All write_facts calls enforce source_span/confidence_tier/derivation fields.
PIPE-01: embed_text uses local sentence-transformers when available (EMBED_BACKEND=cpu|gpu).
         When sentence-transformers is not installed, embed_text raises RuntimeError and the
         worker skips the embedding step — external providers (OpenAI/Ollama) handle it instead.
CASCADE CONTRACT: All rows default to source_status='active'.
"""
from __future__ import annotations

import asyncio
import logging
import uuid

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.derived_memory.models import Summary, Fact, FtsEntry, Embedding, Tag, FactTag, MemoryLink
from app.domain.derivations.visibility import current_sql

logger = logging.getLogger(__name__)

# ── Embedding model singleton ─────────────────────────────────────────────────
# Loaded lazily on first call. NOT at module import (avoids blocking event loop).
# _get_embed_model() is called inside asyncio.to_thread() — safe to block there.
# Will be None if sentence-transformers is not installed (EMBED_BACKEND=none).
try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SentenceTransformer = None  # type: ignore[assignment,misc]
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.info(
        "sentence-transformers not installed (EMBED_BACKEND=none). "
        "Local embeddings unavailable — use OpenAI / Anthropic / Ollama instead."
    )

_embed_model: "_SentenceTransformer | None" = None
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
# Public, canonical name of the embedding model that produced the vectors in the
# `embeddings` table. Write, retrieval, and provenance MUST reference this single
# constant so they can never drift (GPT5.6 #21). Changing it requires a re-embed
# migration — mixing models/dimensions silently corrupts semantic search.
ACTIVE_EMBEDDING_MODEL = _EMBED_MODEL_NAME


def _get_embed_model() -> "_SentenceTransformer":
    """Load embedding model lazily. Called inside asyncio.to_thread() — safe to block."""
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Rebuild with EMBED_BACKEND=cpu or EMBED_BACKEND=gpu, "
            "or configure an external provider (OPENAI_API_KEY / OLLAMA_BASE_URL)."
        )
    global _embed_model
    if _embed_model is None:
        _embed_model = _SentenceTransformer(_EMBED_MODEL_NAME)
        logger.info("Loaded embedding model: %s", _EMBED_MODEL_NAME)
    return _embed_model


async def embed_text(text: str) -> list[float]:
    """Embed text using all-MiniLM-L6-v2 in a thread pool.

    Runs synchronous sentence-transformers encode() in asyncio.to_thread() to
    avoid blocking the event loop. Returns L2-normalized vector (384 dims).

    PIPE-01: local embeddings always available — no API key needed.
    """
    def _encode() -> list[float]:
        model = _get_embed_model()
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    return await asyncio.to_thread(_encode)


async def write_summary(
    session: AsyncSession,
    raw_archive_id: uuid.UUID,
    summary_text: str,
    model_used: str,
    derivation_method: str = "llm_summarization",
) -> Summary:
    """Write a new summary for a raw archive item. source_status defaults to 'active'."""
    summary = Summary(
        raw_archive_id=raw_archive_id,
        summary_text=summary_text,
        model_used=model_used,
        derivation_method=derivation_method,
    )
    session.add(summary)
    await session.commit()
    await session.refresh(summary)
    logger.debug("Wrote summary for archive_id=%s model=%s", raw_archive_id, model_used)
    return summary


async def get_existing_summary(
    session: AsyncSession,
    raw_archive_id: uuid.UUID,
) -> Summary | None:
    """Return the first active summary for this archive item (for BYOK-08 skip check)."""
    result = await session.execute(
        select(Summary)
        .where(Summary.raw_archive_id == raw_archive_id)
        .where(Summary.source_status == "active")
        .limit(1)
    )
    return result.scalar_one_or_none()


async def write_facts(
    session: AsyncSession,
    raw_archive_id: uuid.UUID,
    facts_data: list[dict],
    *, commit: bool = True, generation_id: uuid.UUID | None = None,
) -> list[Fact]:
    """Write extracted facts for a raw archive item.

    PIPE-02 enforcement:
    - source_span is required. If empty/whitespace, confidence_tier is forced to 'low'.
    - confidence_tier must be 'high', 'medium', or 'low'.
    - derivation_method and derivation_model are required.
    """
    created: list[Fact] = []

    for item in facts_data:
        source_span = item.get("source_span", "").strip() if item.get("source_span") else ""
        confidence_tier = item.get("confidence_tier", "low")

        # PIPE-02: empty source_span → force confidence_tier to 'low'
        if not source_span:
            confidence_tier = "low"
            logger.debug(
                "Fact has empty source_span — downgrading confidence to 'low' for archive_id=%s",
                raw_archive_id,
            )

        # Validate confidence_tier
        if confidence_tier not in ("high", "medium", "low"):
            confidence_tier = "low"

        fact = Fact(
            raw_archive_id=raw_archive_id,
            generation_id=generation_id,
            fact_text=item["fact_text"],
            source_span=item.get("source_span", ""),
            confidence_tier=confidence_tier,
            derivation_method=item.get("derivation_method", "llm_extraction"),
            derivation_model=item.get("derivation_model", "unknown"),
            conflict_group_id=item.get("conflict_group_id"),
        )
        session.add(fact)
        created.append(fact)

    if created:
        if commit:
            await session.commit()
            for fact in created:
                await session.refresh(fact)
        else:
            await session.flush()
        logger.debug("Wrote %d facts for archive_id=%s", len(created), raw_archive_id)

    return created


async def write_fts_entry(
    session: AsyncSession,
    raw_archive_id: uuid.UUID,
    text_content: str,
) -> FtsEntry:
    """Write a full-text search entry and populate the tsvector column.

    Uses to_tsvector('english', ...) for English-language FTS indexing.
    search_vector is updated via raw SQL because SQLAlchemy doesn't have a TSVECTOR type.
    """
    entry = FtsEntry(
        raw_archive_id=raw_archive_id,
        text_content=text_content,
    )
    session.add(entry)
    await session.flush()  # Get ID without full commit

    # Update search_vector via SQL (to_tsvector is a PostgreSQL function)
    await session.execute(
        text(
            "UPDATE fts_entries SET search_vector = to_tsvector('english', :content) "
            "WHERE id = :id"
        ),
        {"content": text_content, "id": str(entry.id)},
    )
    await session.commit()
    await session.refresh(entry)
    logger.debug("Wrote FTS entry for archive_id=%s", raw_archive_id)
    return entry


async def write_embedding(
    session: AsyncSession,
    raw_archive_id: uuid.UUID,
    vector: list[float],
    model_name: str = _EMBED_MODEL_NAME,
) -> Embedding:
    """Write embedding vector for a raw archive item.

    PIPE-01: source_status='active'. 384 dims from all-MiniLM-L6-v2.
    Stores embedding_model so future model upgrades can detect stale rows.
    """
    embedding = Embedding(
        raw_archive_id=raw_archive_id,
        embedding=vector,
        embedding_model=model_name,
    )
    session.add(embedding)
    await session.commit()
    await session.refresh(embedding)
    logger.debug(
        "Wrote embedding for archive_id=%s model=%s dims=%d",
        raw_archive_id, model_name, len(vector),
    )
    return embedding


async def get_existing_embedding(
    session: AsyncSession,
    raw_archive_id: uuid.UUID,
) -> Embedding | None:
    """Return the first active embedding for this archive item (BYOK-08 skip check)."""
    result = await session.execute(
        select(Embedding)
        .where(Embedding.raw_archive_id == raw_archive_id)
        .where(Embedding.source_status == "active")
        .limit(1)
    )
    return result.scalar_one_or_none()


async def embedding_model_health(session: AsyncSession) -> dict:
    """Report embedding-model distribution and stale-row count (GPT5.6 #21).

    "Stale" = an active embedding whose model differs from
    ``ACTIVE_EMBEDDING_MODEL`` — a vector that must not be compared against
    current query embeddings and needs a re-embed. Retrieval already filters to
    the active model, so stale rows are silently unsearchable without this report.
    """
    result = await session.execute(
        text(
            "SELECT embedding_model, count(*) AS n FROM embeddings e "
            "WHERE source_status = 'active' " + current_sql('e') + " GROUP BY embedding_model"
        )
    )
    by_model = {row.embedding_model: int(row.n) for row in result}
    total = sum(by_model.values())
    active = by_model.get(ACTIVE_EMBEDDING_MODEL, 0)
    return {
        "active_model": ACTIVE_EMBEDDING_MODEL,
        "total": total,
        "active_model_count": active,
        "stale_count": total - active,
        "models": by_model,
    }


# ── Linkage & tagging writers ─────────────────────────────────────────────────

async def upsert_tag_names(session: AsyncSession, names: list[str]) -> dict[str, uuid.UUID]:
    """Lock all names in batch order, not a different order per fact."""
    from sqlalchemy.dialects.postgresql import insert
    result = {}
    for name in sorted({n.strip().lower() for n in names if n.strip()}):
        await session.execute(insert(Tag).values(id=uuid.uuid4(), name=name)
            .on_conflict_do_nothing(index_elements=[Tag.name]))
        result[name] = await session.scalar(select(Tag.id).where(Tag.name == name))
    return result


async def write_tags(
    session: AsyncSession,
    fact_id: uuid.UUID,
    tags: list[str],
    assigned_by: str = "pipeline",
    *, commit: bool = True, prepared_tag_ids: dict | None = None,
) -> list[FactTag]:
    """Upsert tags and create FactTag associations for a fact.

    Tag names are normalised to lower-case before upsert.
    Duplicate (fact_id, tag_id) pairs are skipped (INSERT OR IGNORE semantics).
    Returns the newly created FactTag rows (may be empty if all already existed).
    """
    if not tags:
        return []

    if not commit:
        # Managed extraction owns the commit. Sorted, database-enforced upserts
        # handle tag-name races between different archives without partial batches.
        from sqlalchemy.dialects.postgresql import insert
        tag_ids = prepared_tag_ids if prepared_tag_ids is not None else await upsert_tag_names(session, tags)
        created = []
        for name in sorted({t.strip().lower() for t in tags if t.strip()}):
            tag_id = tag_ids[name]
            await session.execute(insert(FactTag).values(fact_id=fact_id, tag_id=tag_id,
                assigned_by=assigned_by).on_conflict_do_nothing())
        await session.flush()
        return created

    created: list[FactTag] = []
    for raw_name in tags:
        name = raw_name.strip().lower()
        if not name:
            continue

        # Upsert tag by name
        result = await session.execute(
            select(Tag).where(Tag.name == name).limit(1)
        )
        tag = result.scalar_one_or_none()
        if tag is None:
            tag = Tag(name=name)
            session.add(tag)
            await session.flush()  # populate tag.id

        # Check if association already exists
        assoc_result = await session.execute(
            select(FactTag)
            .where(FactTag.fact_id == fact_id)
            .where(FactTag.tag_id == tag.id)
            .limit(1)
        )
        if assoc_result.scalar_one_or_none() is None:
            ft = FactTag(fact_id=fact_id, tag_id=tag.id, assigned_by=assigned_by)
            session.add(ft)
            created.append(ft)

    if created:
        await session.commit()
        logger.debug("Wrote %d tag associations for fact_id=%s", len(created), fact_id)

    return created


async def write_links(
    session: AsyncSession,
    links_data: list[dict],
    *, commit: bool = True,
) -> list[MemoryLink]:
    """Write memory links between facts.

    Each dict in links_data must contain:
      - source_fact_id: uuid.UUID
      - target_fact_id: uuid.UUID
      - link_type: str  ('related'|'supports'|'elaborates'|'contradicts'|'entity')
      - confidence: float  (0.0–1.0)
      - created_by: str
      - entity_name: str | None  (required when link_type='entity')

    Duplicate (source, target, type) triplets are skipped silently.
    Returns list of created MemoryLink rows.
    """
    if not links_data:
        return []

    created: list[MemoryLink] = []
    for item in links_data:
        src = item["source_fact_id"]
        tgt = item["target_fact_id"]
        ltype = item["link_type"]

        # Skip self-links (DB constraint would reject them anyway)
        if src == tgt:
            continue

        # Check for existing triplet
        existing = await session.execute(
            select(MemoryLink)
            .where(MemoryLink.source_fact_id == src)
            .where(MemoryLink.target_fact_id == tgt)
            .where(MemoryLink.link_type == ltype)
            .limit(1)
        )
        if existing.scalar_one_or_none() is not None:
            continue

        link = MemoryLink(
            source_fact_id=src,
            target_fact_id=tgt,
            link_type=ltype,
            entity_name=item.get("entity_name"),
            confidence=float(item.get("confidence", 1.0)),
            created_by=item.get("created_by", "pipeline_semantic"),
        )
        session.add(link)
        created.append(link)

    if created:
        if commit:
            await session.commit()
            for link in created:
                await session.refresh(link)
        else:
            await session.flush()
        logger.debug("Wrote %d memory links", len(created))

    return created
