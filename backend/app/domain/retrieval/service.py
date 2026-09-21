"""Retrieval service — keyword, semantic, and hybrid search.

Architecture spec: docs/architecture/retrieval-and-ranking.md
- Keyword: PostgreSQL FTS via websearch_to_tsquery('english', query)
- Semantic: pgvector cosine similarity on embeddings table
- Hybrid: RRF merge (k=60, top-50 per mode, top-20 merged)
- Budget trimming: canonical → facts → summaries → raw (strict priority; no mid-item truncation)
- LRU cache: 256 entries, 60s TTL
- Audit event emitted for every retrieve() call

SRCH-01, SRCH-02, SRCH-03, SRCH-04, SRCH-05, SRCH-06, MCP-01, MCP-03
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal

from cachetools import TTLCache
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.audit.models import AuditEvent
from app.domain.derived_memory.service import ACTIVE_EMBEDDING_MODEL
from app.domain.retrieval.diagnostics import MetadataMergeDiagnostic, RetrievalDiagnosticsCollector

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RRF_K: int = 60
RRF_CANDIDATES_PER_MODE: int = 50
RRF_FINAL_TOP_N: int = 20
RRF_MIN_THRESHOLD: float = 1 / (RRF_K + 25)  # ≈ 0.01176
# Context budget is measured in CHARACTERS, not tokens (F7). As a rule of thumb
# ~1 token ≈ 4 characters for English text. The MCP/API `budget` parameter uses
# this same character unit.
CHAR_BUDGET: int = 2000
DEFAULT_BUDGET: int = CHAR_BUDGET  # backward-compatible alias (deprecated; use CHAR_BUDGET)

# ── In-process LRU cache ──────────────────────────────────────────────────────
_cache: TTLCache = TTLCache(maxsize=256, ttl=60)


# ── Request/Response models ───────────────────────────────────────────────────
@dataclass
class RetrievalFilters:
    category: str | None = None
    source_system: str | None = None
    time_range_start: str | None = None
    time_range_end: str | None = None
    canonical_only: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass
class RetrievalRequest:
    query: str
    mode: Literal["keyword", "semantic", "hybrid"] = "hybrid"
    budget: int = DEFAULT_BUDGET
    filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    actor: str = "user_ui"
    limit: int = RRF_FINAL_TOP_N


@dataclass
class RetrievalItem:
    id: str
    type: Literal["canonical", "fact", "summary", "excerpt"]
    content: str
    score: float
    source_id: str
    source_system: str
    captured_at: str
    conflict_label: str | None
    provenance: dict
    source_fact_id: str | None = None
    link_type: str | None = None


@dataclass
class RetrievalResponse:
    query: str
    retrieval_mode: str
    budget_used: int
    budget_limit: int
    trimming_reason: Literal["budget_met", "result_exhausted"]
    items: list[RetrievalItem]
    degraded_mode: bool = False


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_key(req: RetrievalRequest) -> str:
    payload = json.dumps({
        "q": req.query,
        "mode": req.mode,
        "budget": req.budget,
        "filters": {
            "category": req.filters.category,
            "source_system": req.filters.source_system,
            "time_range_start": req.filters.time_range_start,
            "time_range_end": req.filters.time_range_end,
            "canonical_only": req.filters.canonical_only,
            "tags": sorted(req.filters.tags),
        },
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def invalidate_cache() -> None:
    """Invalidate the entire retrieval cache."""
    _cache.clear()
    logger.debug("Retrieval cache invalidated")


CACHE_INVALIDATE_CHANNEL = "recalium_cache_invalidate"


async def notify_cache_invalidation(session: AsyncSession) -> None:
    """Clear the retrieval cache and broadcast the change to all app processes (F8).

    Replaces easily-forgotten manual invalidation after memory writes. The
    in-process cache is cleared immediately; the Postgres NOTIFY makes the
    invalidation event-driven and multi-process safe (any process running
    cache_invalidation_listener() clears its own cache).
    """
    invalidate_cache()
    try:
        await session.execute(text(f"NOTIFY {CACHE_INVALIDATE_CHANNEL}"))
    except Exception as exc:
        logger.debug("cache NOTIFY failed (non-fatal): %s", exc)


async def cache_invalidation_listener() -> None:
    """Background task: LISTEN for cache-invalidation events and clear the cache (F8).

    Reconnects on error; the 60s TTL remains as a safety net. Started from the
    FastAPI lifespan alongside the worker.
    """
    import asyncio  # noqa: PLC0415
    import asyncpg  # noqa: PLC0415
    from app.infrastructure.settings import get_settings  # noqa: PLC0415

    dsn = get_settings().database_url.replace("+asyncpg", "")
    while True:
        conn = None
        try:
            conn = await asyncpg.connect(dsn)
            await conn.add_listener(CACHE_INVALIDATE_CHANNEL, lambda *_: invalidate_cache())
            logger.info("Cache invalidation listener connected (channel=%s)", CACHE_INVALIDATE_CHANNEL)
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("Cache invalidation listener error (reconnecting in 5s): %s", exc)
            await asyncio.sleep(5)
        finally:
            if conn is not None:
                try:
                    await conn.close()
                except Exception:
                    pass


# ── RRF helpers ───────────────────────────────────────────────────────────────

def rrf_score(rank: int, k: int = RRF_K) -> float:
    """Reciprocal Rank Fusion score. rank is 1-based."""
    return 1.0 / (k + rank)


def _fusion_key(candidate: dict) -> str:
    """Stable cross-modal identity for RRF (GPT5.6 #4).

    Keyword and semantic modes key their rows differently (``fts_entries.id`` vs
    ``embeddings.id``), so fusing on the raw row id never combines the two modes'
    votes for the same conversation — the same item appears twice and the cross-modal
    agreement that RRF exists to reward is lost. Fuse instead on the underlying
    retrievable unit: canonical items by their own id, everything else by its source
    archive item, so a keyword excerpt and a semantic summary of the same
    conversation collapse into one ranked result.
    """
    if candidate.get("type") == "canonical":
        return f"canonical:{candidate['id']}"
    if candidate.get("type") == "fact":
        # Each fact is its own atomic retrievable unit — do not collapse multiple
        # distinct facts of one conversation into a single archive-level result.
        return f"fact:{candidate['id']}"
    return f"archive:{candidate.get('source_id') or candidate['id']}"


def _merge_rrf(
    keyword_candidates: list[dict],
    semantic_candidates: list[dict],
    diagnostics: RetrievalDiagnosticsCollector | None = None,
) -> list[dict]:
    """Fuse two ranked candidate lists with RRF on a stable identity (GPT5.6 #4).

    Each mode's list is assumed already ranked best-first. Votes are accumulated per
    stable fusion key; the representative shown for a fused unit is its highest-priority
    type (canonical → fact → summary → excerpt). Returns representative candidate dicts
    with their combined RRF score, sorted desc and capped at ``RRF_FINAL_TOP_N``.
    """
    priority = {t: i for i, t in enumerate(_PRIORITY_ORDER)}
    scores: dict[str, float] = {}
    representative: dict[str, dict] = {}
    observations: dict[str, list[tuple[str, int, dict, float]]] = {}
    reasons: dict[str, str] = {}

    for channel, ranked in (("lexical", keyword_candidates), ("semantic", semantic_candidates)):
        for rank, candidate in enumerate(ranked, start=1):
            key = _fusion_key(candidate)
            contribution = rrf_score(rank)
            scores[key] = scores.get(key, 0.0) + contribution
            observations.setdefault(key, []).append((channel, rank, candidate, contribution))
            current = representative.get(key)
            if current is None or priority.get(candidate.get("type"), 99) < priority.get(
                current.get("type"), 99
            ):
                representative[key] = candidate
                reasons[key] = "first_visible_candidate" if current is None else "higher_memory_type_priority"

    if diagnostics:
        diagnostics.thresholds.update({"rrf_k": RRF_K, "rrf_minimum": RRF_MIN_THRESHOLD, "rrf_final_top_n": RRF_FINAL_TOP_N})

    # Merge provenance into the chosen representative without changing its identity,
    # score, or ordering. Conflicting scalars retain the representative's value.
    merged_metadata: dict[str, MetadataMergeDiagnostic] = {}
    for key, visible in observations.items():
        chosen = representative[key]
        ordered = [entry[2] for entry in visible if entry[2] is chosen] + [entry[2] for entry in visible if entry[2] is not chosen]
        merged = deepcopy(ordered[0].get("provenance") or {})
        conflicts: list[str] = []
        merged_paths: list[str] = []

        def merge_dict(target: dict, incoming: dict, prefix: str = "") -> None:
            for name in sorted(incoming):
                path = f"{prefix}.{name}" if prefix else name
                value = deepcopy(incoming[name])
                if name not in target or target[name] is None:
                    target[name] = value; merged_paths.append(path)
                elif isinstance(target[name], dict) and isinstance(value, dict):
                    merge_dict(target[name], value, path)
                elif isinstance(target[name], list) and isinstance(value, list):
                    for item in value:
                        if item not in target[name]: target[name].append(item)
                    merged_paths.append(path)
                elif target[name] != value:
                    conflicts.append(path)

        for other in ordered[1:]: merge_dict(merged, other.get("provenance") or {})
        chosen_copy = dict(chosen); chosen_copy["provenance"] = merged
        representative[key] = chosen_copy
        unresolved_before = {}
        for channel, _, cand, _ in visible:
            values = ((cand.get("provenance") or {}).get("source_metadata") or {}).get("unresolved_questions") or []
            if values: unresolved_before[channel] = list(values)
        unresolved_after = (merged.get("source_metadata") or {}).get("unresolved_questions") or []
        merged_metadata[key] = MetadataMergeDiagnostic(
            merged_keys=sorted(set(merged_paths)), conflict_paths=sorted(set(conflicts)),
            unresolved_before=unresolved_before, unresolved_after=list(unresolved_after),
            conflicts_before={channel: cand.get("conflict_label") for channel, _, cand, _ in visible if cand.get("conflict_label") is not None},
            conflict_after=chosen_copy.get("conflict_label"),
        )
        if diagnostics:
            diagnostics.record_fusion(key, visible, chosen_copy, scores[key], reasons[key], merged_metadata[key])

    fused = [(key, score) for key, score in scores.items() if score >= RRF_MIN_THRESHOLD]
    fused.sort(key=lambda kv: kv[1], reverse=True)

    result: list[dict] = []
    included = {key for key, _ in fused[:RRF_FINAL_TOP_N]}
    if diagnostics:
        for key, score in scores.items():
            if score < RRF_MIN_THRESHOLD: diagnostics.exclude(representative[key]["id"], "rrf_threshold", "rrf_below_threshold")
            elif key not in included: diagnostics.exclude(representative[key]["id"], "fusion_limit", "fused_top_n_limit")
    for final_rank, (key, score) in enumerate(fused[:RRF_FINAL_TOP_N], start=1):
        candidate = dict(representative[key])
        candidate["score"] = score
        result.append(candidate)
        if diagnostics: diagnostics.mark_rank(candidate["id"], final_rank)
    return result


# ── Budget trimming ───────────────────────────────────────────────────────────

_PRIORITY_ORDER = ["canonical", "fact", "summary", "excerpt"]


def apply_budget_trimming(
    items: list[RetrievalItem],
    budget: int,
    diagnostics: RetrievalDiagnosticsCollector | None = None,
) -> tuple[list[RetrievalItem], int, Literal["budget_met", "result_exhausted"]]:
    """Apply strict priority budget trimming.

    Order: canonical → fact → summary → excerpt.
    Never truncate mid-item. Skip item if it doesn't fit.
    Returns (trimmed_items, budget_used, trimming_reason).
    """
    priority_map = {t: i for i, t in enumerate(_PRIORITY_ORDER)}
    sorted_items = sorted(items, key=lambda x: (priority_map.get(x.type, 99), -x.score))

    result: list[RetrievalItem] = []
    used = 0

    for item in sorted_items:
        item_len = len(item.content)
        # We continue scanning even if this item doesn't fit — a smaller later item might.
        if used + item_len <= budget:
            result.append(item)
            used += item_len
        elif diagnostics:
            diagnostics.exclude(item.id, "priority_budget_trimming", "character_budget_exceeded")
        # If item doesn't fit: skip entirely (never truncate)
        if used >= budget:
            return result, used, "budget_met"

    return result, used, "result_exhausted"


# ── Candidate retrieval ───────────────────────────────────────────────────────

# Imports store 'chatgpt_import'/'claude_import' but the documented/MCP filter value
# is 'chatgpt'/'claude'. Accept both so the advertised filter works (GPT5.6 #4).
_SOURCE_ALIASES: dict[str, list[str]] = {
    "chatgpt": ["chatgpt", "chatgpt_import"],
    "claude": ["claude", "claude_import"],
}


def _normalize_source_filter(value: str | None) -> list[str]:
    """Map an advertised source name to the stored ``source_type`` values (GPT5.6 #4)."""
    v = (value or "").strip().lower()
    if not v:
        return []
    if v in _SOURCE_ALIASES:
        return list(_SOURCE_ALIASES[v])
    # Generic: also accept the "<v>_import" variant so future importers work.
    return list(dict.fromkeys([v, f"{v}_import"]))


def _parse_iso_dt(value: str | None) -> datetime | None:
    """Parse an ISO-8601 filter bound to an aware datetime (GPT5.6 #4).

    asyncpg binds a timestamptz parameter from a ``datetime``, not a string, so time
    filters are parsed here. A naive value is assumed UTC. Returns None on garbage so
    a malformed bound is ignored rather than crashing retrieval.
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _archive_filter_sql(f: "RetrievalFilters", alias: str = "ra") -> tuple[str, dict]:
    """SQL WHERE fragments + params for archive-backed candidates (GPT5.6 #4).

    Pushes source_system, time range, and category into the candidate query so they
    apply BEFORE the per-mode LIMIT. Post-filtering (the previous behaviour) silently
    dropped matches ranked past the top-50 per mode.
    """
    clauses: list[str] = []
    params: dict = {}
    sources = _normalize_source_filter(f.source_system)
    if sources:
        params["f_sources"] = sources
        clauses.append(f"AND {alias}.source_type = ANY(:f_sources)")
    start = _parse_iso_dt(f.time_range_start)
    if start is not None:
        params["f_time_start"] = start
        clauses.append(f"AND {alias}.ingested_at >= :f_time_start")
    end = _parse_iso_dt(f.time_range_end)
    if end is not None:
        params["f_time_end"] = end
        clauses.append(f"AND {alias}.ingested_at <= :f_time_end")
    if f.category:
        params["f_category"] = f.category.strip().lower()
        clauses.append(f"AND lower({alias}.metadata_json->>'data_class') = :f_category")
    return "\n".join(clauses), params


def _canonical_included(f: "RetrievalFilters") -> bool:
    """Whether canonical items pass the declared source/category filters (GPT5.6 #4).

    Canonical items surface as source_system='canonical' and are user-curated (not
    gate-classified), so a category filter or a non-canonical source filter excludes
    them.
    """
    if f.category:
        return False
    sources = _normalize_source_filter(f.source_system)
    if sources and "canonical" not in sources:
        return False
    return True


def _canonical_time_sql(f: "RetrievalFilters") -> tuple[str, dict]:
    """Time-range WHERE fragments + params for the canonical query (GPT5.6 #4)."""
    clauses: list[str] = []
    params: dict = {}
    start = _parse_iso_dt(f.time_range_start)
    if start is not None:
        params["f_time_start"] = start
        clauses.append("AND cm.created_at >= :f_time_start")
    end = _parse_iso_dt(f.time_range_end)
    if end is not None:
        params["f_time_end"] = end
        clauses.append("AND cm.created_at <= :f_time_end")
    return "\n".join(clauses), params


async def _keyword_candidates(
    session: AsyncSession,
    query: str,
    limit: int = RRF_CANDIDATES_PER_MODE,
    tags: list[str] | None = None,
    filters: "RetrievalFilters | None" = None,
) -> list[dict]:
    """Retrieve keyword candidates using PostgreSQL FTS.

    Declared filters (source_system, time range, category, canonical_only) are pushed
    into SQL so they apply before the LIMIT (GPT5.6 #4).
    """
    f = filters or RetrievalFilters()
    # Build optional tag filter clause — restricts to archive items whose facts carry ALL requested tags
    tag_filter_sql = ""
    tag_params: dict = {}
    if tags:
        normalised = [t.strip().lower() for t in tags if t.strip()]
        if normalised:
            tag_filter_sql = """
              AND fe.raw_archive_id IN (
                  SELECT f.raw_archive_id
                  FROM facts f
                  JOIN fact_tags ft ON ft.fact_id = f.id
                  JOIN tags tg ON tg.id = ft.tag_id
                  WHERE f.source_status = 'active'
                                        AND f.review_status = 'active'
                    AND tg.name = ANY(:tag_names)
                  GROUP BY f.raw_archive_id
                  HAVING COUNT(DISTINCT tg.name) >= :tag_count
              )
            """
            tag_params = {"tag_names": normalised, "tag_count": len(normalised)}

    archive_filter_sql, archive_filter_params = _archive_filter_sql(f, alias="ra")

    fts_rows = []
    if not f.canonical_only:
        fts_result = await session.execute(
            text(f"""
                SELECT
                    fe.id::text AS id,
                    'excerpt' AS type,
                    LEFT(fe.text_content, 500) AS content,
                    ts_rank_cd(fe.search_vector, websearch_to_tsquery('english', :q)) AS score,
                    fe.raw_archive_id::text AS source_id,
                    COALESCE(ra.source_type, 'unknown') AS source_system,
                    ra.ingested_at::text AS captured_at,
                    NULL::text AS conflict_label,
                    fe.raw_archive_id::text AS archive_id
                FROM fts_entries fe
                JOIN raw_archive ra ON ra.id = fe.raw_archive_id
                WHERE fe.source_status = 'active'
                  AND ra.deleted_at IS NULL
                  AND fe.search_vector @@ websearch_to_tsquery('english', :q)
                  {tag_filter_sql}
                  {archive_filter_sql}
                ORDER BY score DESC
                LIMIT :limit
            """),
            {"q": query, "limit": limit, **tag_params, **archive_filter_params},
        )
        fts_rows = fts_result.mappings().all()

    canon_rows = []
    if _canonical_included(f):
        canon_time_sql, canon_time_params = _canonical_time_sql(f)
        canon_result = await session.execute(
            text(f"""
                SELECT
                    cm.id::text AS id,
                    'canonical' AS type,
                    cm.content AS content,
                    ts_rank_cd(cm.search_vector, websearch_to_tsquery('english', :q)) AS score,
                    COALESCE(cm.raw_archive_id::text, cm.id::text) AS source_id,
                    'canonical' AS source_system,
                    cm.created_at::text AS captured_at,
                    NULL::text AS conflict_label,
                    cm.raw_archive_id::text AS archive_id
                FROM canonical_memory cm
                WHERE cm.source_status = 'active'
                  AND cm.status = 'active'
                  AND cm.search_vector @@ websearch_to_tsquery('english', :q)
                  {canon_time_sql}
                ORDER BY score DESC
                LIMIT :limit
            """),
            {"q": query, "limit": limit, **canon_time_params},
        )
        canon_rows = canon_result.mappings().all()

    # GPT5.6 #4: direct fact retrieval — match facts by their own FTS vector so a
    # relevant fact surfaces on its own, not only via link traversal.
    fact_rows = []
    if not f.canonical_only:
        fact_result = await session.execute(
            text(f"""
                SELECT
                    f.id::text AS id,
                    'fact' AS type,
                    f.fact_text AS content,
                    ts_rank_cd(f.search_vector, websearch_to_tsquery('english', :q)) AS score,
                    f.raw_archive_id::text AS source_id,
                    COALESCE(ra.source_type, 'unknown') AS source_system,
                    ra.ingested_at::text AS captured_at,
                    f.source_span AS source_span
                FROM facts f
                JOIN raw_archive ra ON ra.id = f.raw_archive_id
                WHERE f.source_status = 'active'
                  AND f.review_status = 'active'
                  AND ra.deleted_at IS NULL
                  AND f.search_vector @@ websearch_to_tsquery('english', :q)
                  {archive_filter_sql}
                ORDER BY score DESC
                LIMIT :limit
            """),
            {"q": query, "limit": limit, **archive_filter_params},
        )
        fact_rows = fact_result.mappings().all()

    candidates = []
    for row in list(canon_rows) + list(fts_rows):
        candidates.append({
            "id": row["id"],
            "type": row["type"],
            "content": row["content"] or "",
            "score": float(row["score"] or 0),
            "source_id": row["source_id"] or "",
            "source_system": row["source_system"] or "unknown",
            "captured_at": str(row["captured_at"] or ""),
            "conflict_label": None,
            "provenance": {
                "derivation_method": "fts_retrieval",
                "derivation_model": "postgresql_fts",
                "source_excerpt": (row["content"] or "")[:200],
            },
        })

    for row in fact_rows:
        candidates.append({
            "id": row["id"],
            "type": "fact",
            "content": row["content"] or "",
            "score": float(row["score"] or 0),
            "source_id": row["source_id"] or "",
            "source_system": row["source_system"] or "unknown",
            "captured_at": str(row["captured_at"] or ""),
            "conflict_label": None,
            "provenance": {
                "derivation_method": "fact_fts_retrieval",
                "derivation_model": "postgresql_fts",
                "source_excerpt": (row["source_span"] or row["content"] or "")[:200],
            },
        })

    # Rank the combined candidates (canonical + excerpts + facts) by FTS relevance so
    # the per-mode LIMIT keeps the most relevant across all three, not insertion order.
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:limit]


async def _semantic_candidates(
    session: AsyncSession,
    query: str,
    limit: int = RRF_CANDIDATES_PER_MODE,
    tags: list[str] | None = None,
    filters: "RetrievalFilters | None" = None,
) -> tuple[list[dict], bool]:
    """Retrieve semantic candidates using pgvector cosine similarity.

    Returns (candidates, degraded) where degraded=True if no embeddings exist.
    Declared source/time/category filters are pushed into SQL (GPT5.6 #4).
    """
    f = filters or RetrievalFilters()
    # Semantic candidates are all archive-backed (summary/excerpt); canonical_only
    # therefore yields nothing from this mode.
    if f.canonical_only:
        return [], False
    try:
        from app.domain.derived_memory.service import embed_text
        query_vector = await embed_text(query)
    except RuntimeError:
        logger.info("Semantic search unavailable: sentence-transformers not installed (degraded mode)")
        return [], True

    # Build optional tag filter clause
    tag_filter_sql = ""
    tag_params: dict = {}
    if tags:
        normalised = [t.strip().lower() for t in tags if t.strip()]
        if normalised:
            tag_filter_sql = """
              AND e.raw_archive_id IN (
                  SELECT f.raw_archive_id
                  FROM facts f
                  JOIN fact_tags ft ON ft.fact_id = f.id
                  JOIN tags tg ON tg.id = ft.tag_id
                  WHERE f.source_status = 'active'
                                        AND f.review_status = 'active'
                    AND tg.name = ANY(:tag_names)
                  GROUP BY f.raw_archive_id
                  HAVING COUNT(DISTINCT tg.name) >= :tag_count
              )
            """
            tag_params = {"tag_names": normalised, "tag_count": len(normalised)}

    archive_filter_sql, archive_filter_params = _archive_filter_sql(f, alias="ra")

    result = await session.execute(
        text(f"""
            SELECT
                e.id::text AS id,
                e.raw_archive_id::text AS raw_archive_id,
                1 - (e.embedding <=> :vec) AS score,
                ra.source_type AS source_system,
                ra.ingested_at::text AS captured_at
            FROM embeddings e
            JOIN raw_archive ra ON ra.id = e.raw_archive_id
            WHERE e.source_status = 'active'
              AND ra.deleted_at IS NULL
              AND e.embedding_model = :model
              {tag_filter_sql}
              {archive_filter_sql}
            ORDER BY e.embedding <=> :vec
            LIMIT :limit
        """),
        {
            "vec": str(query_vector),
            "model": ACTIVE_EMBEDDING_MODEL,
            "limit": limit,
            **tag_params,
            **archive_filter_params,
        },
    )
    rows = result.mappings().all()

    if not rows:
        return [], True

    candidates = []
    for row in rows:
        summary_result = await session.execute(
            text("""
                SELECT summary_text FROM summaries
                WHERE raw_archive_id = :aid AND source_status = 'active'
                LIMIT 1
            """),
            {"aid": row["raw_archive_id"]},
        )
        summary_row = summary_result.fetchone()

        if summary_row:
            content = summary_row[0]
            item_type = "summary"
        else:
            raw_result = await session.execute(
                text("SELECT LEFT(raw_content, 500) FROM raw_archive WHERE id = :aid"),
                {"aid": row["raw_archive_id"]},
            )
            raw_row = raw_result.fetchone()
            content = raw_row[0] if raw_row else ""
            item_type = "excerpt"

        candidates.append({
            "id": row["id"],
            "type": item_type,
            "content": content or "",
            "score": float(row["score"] or 0),
            "source_id": row["raw_archive_id"],
            "source_system": row["source_system"] or "unknown",
            "captured_at": str(row["captured_at"] or ""),
            "conflict_label": None,
            "provenance": {
                "derivation_method": "semantic_retrieval",
                "derivation_model": ACTIVE_EMBEDDING_MODEL,
                "source_excerpt": (content or "")[:200],
            },
        })

    return candidates, False


# ── Link traversal ────────────────────────────────────────────────────────────

async def _traverse_links(
    session: AsyncSession,
    archive_ids: list[str],
    max_links: int = 10,
) -> list[dict]:
    """Find linked facts for the given archive items and return them as candidates.

    For each archive item in the current result set, find outgoing 'related',
    'supports', 'elaborates', and 'entity' links and return the linked facts
    as additional RetrievalItem dicts (type='fact').

    Limits to max_links total to avoid flooding the context budget.
    """
    if not archive_ids:
        return []

    rows = (await session.execute(
        text("""
            SELECT
                ml.id::text AS link_id,
                ml.link_type,
                ml.confidence,
                ml.source_fact_id::text AS source_fact_id,
                ml.target_fact_id::text AS target_fact_id,
                tf.fact_text AS target_fact_text,
                tf.raw_archive_id::text AS target_archive_id,
                ra.source_type AS source_system,
                ra.ingested_at::text AS captured_at,
                lower(ra.metadata_json->>'data_class') AS category
            FROM memory_links ml
            JOIN facts sf ON sf.id = ml.source_fact_id
            JOIN facts tf ON tf.id = ml.target_fact_id
            JOIN raw_archive ra ON ra.id = tf.raw_archive_id
            WHERE sf.raw_archive_id = ANY(:archive_ids)
              AND sf.source_status = 'active'
              AND tf.source_status = 'active'
                            AND sf.review_status = 'active'
                            AND tf.review_status = 'active'
              AND ra.deleted_at IS NULL
              AND ml.link_type != 'contradicts'
            ORDER BY ml.confidence DESC
            LIMIT :max_links
        """),
        {"archive_ids": archive_ids, "max_links": max_links},
    )).mappings().all()

    seen_fact_ids: set[str] = set()
    candidates: list[dict] = []
    for row in rows:
        tgt_id = row["target_fact_id"]
        if tgt_id in seen_fact_ids:
            continue
        seen_fact_ids.add(tgt_id)
        candidates.append({
            "id": tgt_id,
            "type": "fact",
            "content": row["target_fact_text"] or "",
            "score": float(row["confidence"] or 0.5) * 0.5,  # discount vs direct match
            "source_id": row["target_archive_id"] or "",
            "source_system": row["source_system"] or "unknown",
            "captured_at": str(row["captured_at"] or ""),
            "category": row["category"] or "",
            "conflict_label": None,
            "source_fact_id": row["source_fact_id"],
            "link_type": row["link_type"],
            "provenance": {
                "derivation_method": "link_traversal",
                "derivation_model": "memory_links",
                "source_excerpt": (row["target_fact_text"] or "")[:200],
            },
        })

    return candidates


# ── Main retrieve function ────────────────────────────────────────────────────

_MAX_QUERY_CHARS = 2000


def _sanitize_fts_query(q: str) -> str:
    """F9: harden free-text search input before it reaches Postgres FTS.

    websearch_to_tsquery tolerates arbitrary punctuation, but NUL bytes are
    invalid in Postgres text and pathologically long inputs waste work. Strip
    NUL/control characters (keep tab/newline) and cap length. Normal queries are
    unchanged.
    """
    if not q:
        return q
    cleaned = "".join(ch for ch in q if ch in "\t\n" or ord(ch) >= 32)
    return cleaned[:_MAX_QUERY_CHARS].strip()


def _apply_candidate_filters(candidates: list[dict], f: RetrievalFilters) -> list[dict]:
    """Safety-net filter over the final candidate set (GPT5.6 #4).

    source_system, time range, category, and canonical_only are enforced at the SQL
    layer (before the per-mode LIMIT). This post-filter still runs because link
    traversal appends linked facts *after* the SQL queries; it re-applies the same
    source/time/category/canonical filters to those appended candidates so they
    can't bypass a declared filter. Source matching uses the same normalization as
    SQL so the advertised ``chatgpt`` matches stored ``chatgpt_import``. Category
    matching only applies to candidates that carry a ``category`` key (currently:
    linked facts from ``_traverse_links``); other candidate types were already
    filtered by category at the SQL layer before reaching here.
    """
    out = candidates
    if f.canonical_only:
        out = [c for c in out if c.get("type") == "canonical"]
    if f.source_system:
        accepted = set(_normalize_source_filter(f.source_system))
        out = [c for c in out if (c.get("source_system") or "").strip().lower() in accepted]
    if f.time_range_start:
        out = [c for c in out if (c.get("captured_at") or "") >= f.time_range_start]
    if f.time_range_end:
        out = [c for c in out if (c.get("captured_at") or "") <= f.time_range_end]
    if f.category:
        wanted = f.category.strip().lower()
        out = [c for c in out if "category" not in c or c.get("category") == wanted]
    return out


async def retrieve(
    session: AsyncSession,
    req: RetrievalRequest,
    diagnostics: RetrievalDiagnosticsCollector | None = None,
) -> RetrievalResponse:
    """Execute retrieval and return a context-budgeted response.

    Modes: keyword, semantic, hybrid.
    Always emits an AuditEvent.
    Uses in-process LRU cache for repeated identical queries.
    """
    req.query = _sanitize_fts_query(req.query)
    if diagnostics is not None:
        diagnostics.filters = {
            key: value for key, value in asdict(req.filters).items()
            if value not in (None, "", [], ())
        }
    if req.mode not in ("keyword", "semantic", "hybrid"):
        raise ValueError(
            f"Invalid retrieval mode {req.mode!r}: must be 'keyword', 'semantic', or 'hybrid'"
        )
    cache_key = _cache_key(req)
    cache_allowed = diagnostics is None
    if cache_allowed and cache_key in _cache:
        logger.debug("Retrieval cache hit for query=%r", req.query[:50])
        # NOTE: Audit events are intentionally NOT emitted on cache hits.
        # The original retrieve() call already recorded the audit event.
        # Emitting on every cache hit would inflate audit log counts.
        return _cache[cache_key]

    degraded = False
    effective_mode = req.mode

    if req.mode == "keyword":
        candidates = await _keyword_candidates(
            session, req.query, RRF_CANDIDATES_PER_MODE,
            tags=req.filters.tags or None, filters=req.filters,
        )
        if diagnostics is not None and len(candidates) >= RRF_CANDIDATES_PER_MODE:
            diagnostics.record_unknown_exclusion("lexical_sql_limit", "not_observable_at_this_stage")

    elif req.mode == "semantic":
        candidates, degraded = await _semantic_candidates(
            session, req.query, RRF_CANDIDATES_PER_MODE,
            tags=req.filters.tags or None, filters=req.filters,
        )
        if diagnostics is not None and len(candidates) >= RRF_CANDIDATES_PER_MODE:
            diagnostics.record_unknown_exclusion("semantic_sql_limit", "not_observable_at_this_stage")

    else:  # hybrid
        kw_candidates = await _keyword_candidates(
            session, req.query, RRF_CANDIDATES_PER_MODE,
            tags=req.filters.tags or None, filters=req.filters,
        )
        sem_candidates, sem_degraded = await _semantic_candidates(
            session, req.query, RRF_CANDIDATES_PER_MODE,
            tags=req.filters.tags or None, filters=req.filters,
        )

        if diagnostics is not None:
            if len(kw_candidates) >= RRF_CANDIDATES_PER_MODE:
                diagnostics.record_unknown_exclusion("lexical_sql_limit", "not_observable_at_this_stage")
            if len(sem_candidates) >= RRF_CANDIDATES_PER_MODE:
                diagnostics.record_unknown_exclusion("semantic_sql_limit", "not_observable_at_this_stage")

        if sem_degraded or not sem_candidates:
            degraded = True
            effective_mode = "keyword"
            candidates = kw_candidates
        else:
            # GPT5.6 #4: fuse on a stable cross-modal identity so the same
            # conversation ranked by both modes combines its votes (was fused on
            # per-mode row ids that can never match across modes).
            candidates = _merge_rrf(kw_candidates, sem_candidates, diagnostics=diagnostics)

    # ── Link traversal — fetch linked facts for direct candidates ────────────
    direct_archive_ids = list({c["source_id"] for c in candidates if c.get("source_id")})
    try:
        async with session.begin_nested():
            linked_candidates = await _traverse_links(session, direct_archive_ids)
        # Avoid duplicating items already in candidates
        existing_ids = {c["id"] for c in candidates}
        candidates += [lc for lc in linked_candidates if lc["id"] not in existing_ids]
    except Exception as exc:
        logger.debug("Link traversal failed (non-fatal): %s", exc)

    # GPT5.6 #4: enforce declared filters on the final candidate set
    before_filters = candidates
    candidates = _apply_candidate_filters(candidates, req.filters)
    if diagnostics is not None:
        retained_ids = {candidate["id"] for candidate in candidates}
        for candidate in before_filters:
            if candidate["id"] not in retained_ids:
                diagnostics.exclude(candidate["id"], "final_candidate_filter", "declared_filter_mismatch")

    items = [
        RetrievalItem(
            id=c["id"],
            type=c["type"],
            content=c["content"],
            score=c["score"],
            source_id=c["source_id"],
            source_system=c["source_system"],
            captured_at=c["captured_at"],
            conflict_label=c.get("conflict_label"),
            provenance=c.get("provenance", {}),
            source_fact_id=c.get("source_fact_id"),
            link_type=c.get("link_type"),
        )
        for c in candidates
    ]

    trimmed_items, budget_used, trimming_reason = apply_budget_trimming(items, req.budget, diagnostics=diagnostics)

    response = RetrievalResponse(
        query=req.query,
        retrieval_mode=effective_mode,
        budget_used=budget_used,
        budget_limit=req.budget,
        trimming_reason=trimming_reason,
        items=trimmed_items,
        degraded_mode=degraded,
    )

    audit_event = AuditEvent(
        event_type="search" if req.actor == "user_ui" else "mcp_retrieve",
        actor=req.actor,
        operation_metadata={
            "query_summary": req.query[:100],
            "result_count": len(trimmed_items),
            "retrieval_mode": effective_mode,
            "degraded_mode": degraded,
            "policy_decision": "allowed",
        },
    )
    session.add(audit_event)
    await session.flush()
    # NOTE: Commit is left to the caller (API dependency or MCP session manager).

    if cache_allowed:
        _cache[cache_key] = response
    return response
