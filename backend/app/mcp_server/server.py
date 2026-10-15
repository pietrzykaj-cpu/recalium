"""MCP server for Recalium.

Exposes a single `retrieve_memory` tool for MCP clients (AI agents).
All access events are recorded in the audit_events table (MCP-03, MCP-04).

SECURITY: Server must be bound to 127.0.0.1 only (locked architectural decision —
DNS rebinding attack prevention). This is enforced at the uvicorn/Docker level.

MCP-01: retrieve_memory tool
MCP-03: access event audit (event_type='mcp_retrieve')
MCP-04: client identity recorded in actor field
"""
from __future__ import annotations
from app.domain.derivations.visibility import current_sql, enabled
from app.domain.derivations.readers import begin_reader_snapshot

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.domain.retrieval.service import (
    RetrievalFilters,
    RetrievalRequest,
    retrieve,
)
from app.infrastructure.db import get_session_factory

logger = logging.getLogger(__name__)

# MCP app instance — exported so main.py can mount it
mcp_app = FastMCP("recalium")


def _mcp_error(
    code: str,
    message: str,
    *,
    field: str | None = None,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error_details = dict(details or {})
    if field is not None:
        error_details["field"] = field
    return {
        "status": "error",
        "error": {
            "code": code,
            "message": message,
            "details": error_details,
            "retryable": retryable,
        },
    }


@mcp_app.tool()
async def retrieve_memory(
    query: str,
    mode: str = "hybrid",
    budget: int = 2000,
    category: str | None = None,
    source_system: str | None = None,
    time_range_start: str | None = None,
    time_range_end: str | None = None,
    canonical_only: bool = False,
    tags: list[str] | None = None,
    actor: str = "mcp_client",
) -> dict[str, Any]:
    """Retrieve relevant memory items from the Recalium archive.

    Args:
        query: The search query string.
        mode: Retrieval mode — "keyword", "semantic", or "hybrid" (default: "hybrid").
        budget: Maximum character budget for returned content (default: 2000).
        category: Optional category filter.
        source_system: Optional source system filter (e.g. "chatgpt", "claude").
        time_range_start: Optional ISO 8601 datetime lower bound for captured_at.
        time_range_end: Optional ISO 8601 datetime upper bound for captured_at.
        canonical_only: If true, return only canonical memory items.
        tags: Optional list of tag names to filter results (all tags must match).
        actor: MCP client identity string (for audit trail — MCP-04).

    Returns:
        Retrieval envelope with items, scores, provenance, and context budget metadata.
        Items include source_fact_id and link_type fields when retrieved via link traversal.
    """
    filters = RetrievalFilters(
        category=category,
        source_system=source_system,
        time_range_start=time_range_start,
        time_range_end=time_range_end,
        canonical_only=canonical_only,
        tags=tags or [],
    )
    req = RetrievalRequest(
        query=query,
        mode=mode,
        budget=budget,
        filters=filters,
        actor=actor,
    )

    factory = get_session_factory()
    async with factory() as session:
        try:
            response = await retrieve(session, req)
            # GPT5.6 #11: commit so the MCP access audit event is durable
            # (retrieve() adds+flushes the event but leaves the commit to the caller).
            await session.commit()
        except ValueError as exc:
            await session.rollback()
            return _mcp_error("validation_error", str(exc), field="mode")
        except Exception:
            await session.rollback()
            logger.exception("MCP retrieve_memory failed")
            return _mcp_error("internal_error", "Retrieval failed due to an internal error.", retryable=True)

    return {
        "query": response.query,
        "retrieval_mode": response.retrieval_mode,
        "budget_used": response.budget_used,
        "budget_limit": response.budget_limit,
        "trimming_reason": response.trimming_reason,
        "degraded_mode": response.degraded_mode,
        "items": [
            {
                "id": item.id,
                "type": item.type,
                "content": item.content,
                "score": item.score,
                "source_id": item.source_id,
                "source_system": item.source_system,
                "captured_at": item.captured_at,
                "conflict_label": item.conflict_label,
                "provenance": item.provenance,
                "source_fact_id": item.source_fact_id,
                "link_type": item.link_type,
            }
            for item in response.items
        ],
    }


@mcp_app.tool()
async def get_fact_links(
    fact_id: str,
    direction: str = "outgoing",
) -> dict[str, Any]:
    """Get memory links for a specific fact.

    Args:
        fact_id: UUID string of the fact to look up links for.
        direction: 'outgoing' (default), 'incoming', or 'both'.

    Returns:
        Dict with fact_id, direction, links list, and total count.
        Each link includes link_type, confidence, entity_name, other_fact_id,
        other_fact_text, and created_by.
        Returns {"error": "..."} if fact_id is invalid or fact not found.
    """
    if direction not in ("outgoing", "incoming", "both"):
        return _mcp_error(
            "validation_error",
            "direction must be 'outgoing', 'incoming', or 'both'",
            field="direction",
        )

    try:
        import uuid as _uuid  # noqa: PLC0415
        fid = _uuid.UUID(fact_id)
    except (ValueError, AttributeError):
        return _mcp_error("validation_error", f"Invalid fact_id UUID: {fact_id!r}", field="fact_id")

    from sqlalchemy import text  # noqa: PLC0415

    factory = get_session_factory()
    async with factory() as session:
        try:
            await begin_reader_snapshot(session)
            fact_check = (await session.execute(
                text("SELECT id FROM facts f WHERE id = :fid AND source_status = 'active' AND review_status = 'active' " + current_sql('f') + " LIMIT 1"),
                {"fid": str(fid)},
            )).fetchone()
            if fact_check is None:
                return _mcp_error("not_found", f"Fact {fact_id} not found or not active", field="fact_id")

            if direction == "outgoing":
                where = "ml.source_fact_id = :fid"
                other_id_col = "ml.target_fact_id::text AS other_fact_id"
                join_clause = "JOIN facts tf ON tf.id = ml.target_fact_id"
                text_col = "tf.fact_text AS other_fact_text"
                status_filter = "AND tf.source_status = 'active' AND tf.review_status = 'active'" + current_sql('tf')
            elif direction == "incoming":
                where = "ml.target_fact_id = :fid"
                other_id_col = "ml.source_fact_id::text AS other_fact_id"
                join_clause = "JOIN facts sf ON sf.id = ml.source_fact_id"
                text_col = "sf.fact_text AS other_fact_text"
                status_filter = "AND sf.source_status = 'active' AND sf.review_status = 'active'" + current_sql('sf')
            else:
                where = "(ml.source_fact_id = :fid OR ml.target_fact_id = :fid)"
                other_id_col = """CASE WHEN ml.source_fact_id = :fid THEN ml.target_fact_id::text ELSE ml.source_fact_id::text END AS other_fact_id"""
                join_clause = "LEFT JOIN facts tf ON tf.id = ml.target_fact_id LEFT JOIN facts sf ON sf.id = ml.source_fact_id"
                text_col = """CASE WHEN ml.source_fact_id = :fid THEN tf.fact_text ELSE sf.fact_text END AS other_fact_text"""
                status_filter = f"""
                                        AND (
                                            (ml.source_fact_id = :fid AND tf.source_status = 'active' AND tf.review_status = 'active' {current_sql('tf')})
                                            OR (ml.target_fact_id = :fid AND sf.source_status = 'active' AND sf.review_status = 'active' {current_sql('sf')})
                                        )"""

            rows = (await session.execute(
                text(f"""
                    SELECT ml.id::text AS link_id, ml.link_type, ml.confidence,
                           ml.entity_name, ml.created_by, ml.created_at::text AS created_at,
                           {other_id_col}, {text_col}
                    FROM memory_links ml
                    {join_clause}
                    WHERE {where} {status_filter}
                    ORDER BY ml.created_at DESC
                """),
                {"fid": str(fid)},
            )).mappings().all()
        except Exception:
            await session.rollback()
            logger.exception("MCP get_fact_links failed")
            return _mcp_error("internal_error", "Link lookup failed due to an internal error.", retryable=True)

    return {
        "fact_id": fact_id,
        "direction": direction,
        "links": [
            {
                "link_id": row["link_id"],
                "link_type": row["link_type"],
                "confidence": float(row["confidence"] or 1.0),
                "entity_name": row["entity_name"],
                "created_by": row["created_by"],
                "created_at": row["created_at"],
                "other_fact_id": row["other_fact_id"],
                "other_fact_text": row["other_fact_text"],
            }
            for row in rows
        ],
        "total": len(rows),
    }


@mcp_app.tool()
async def list_tags(
    prefix: str | None = None,
    min_count: int = 0,
) -> dict[str, Any]:
    """List all tags with their usage counts.

    Args:
        prefix: Optional prefix to filter tag names (e.g. "entity:" for entity tags).
        min_count: Minimum fact_count to include (default: 0 = include all).

    Returns:
        Dict with tags list (id, name, fact_count, created_at) and total count.
    """
    from sqlalchemy import text  # noqa: PLC0415

    factory = get_session_factory()
    async with factory() as session:
        try:
            await begin_reader_snapshot(session)
            rows = (await session.execute(
                text(f"""
                    SELECT
                        t.id::text AS id,
                        t.name,
                        t.created_at::text AS created_at,
                        COUNT(f.id) AS fact_count
                    FROM tags t
                    LEFT JOIN fact_tags ft ON ft.tag_id = t.id
                    LEFT JOIN facts f ON f.id = ft.fact_id AND f.source_status = 'active' AND f.review_status = 'active'
                    {current_sql('f')}
                    GROUP BY t.id, t.name, t.created_at
                    {'HAVING COUNT(f.id) > 0' if enabled() else ''}
                    ORDER BY fact_count DESC, t.name ASC
                """),
            )).mappings().all()
        except Exception:
            await session.rollback()
            logger.exception("MCP list_tags failed")
            return _mcp_error("internal_error", "Tag listing failed due to an internal error.", retryable=True)

    tags = [
        {
            "id": row["id"],
            "name": row["name"],
            "fact_count": int(row["fact_count"] or 0),
            "created_at": row["created_at"],
        }
        for row in rows
    ]

    if prefix:
        tags = [t for t in tags if t["name"].startswith(prefix)]
    if min_count > 0:
        tags = [t for t in tags if t["fact_count"] >= min_count]

    return {"tags": tags, "total": len(tags)}


@mcp_app.tool()
async def ingest_memory(
    content: str = "",
    source_metadata: dict[str, Any] | None = None,
    client_identity: str | None = None,
    import_method: str = "mcp_tool",
    idempotency_key: str | None = None,
    sensitivity_hint: str | None = None,
    project_hint: str | None = None,
    processing_mode: str = "deferred",
    source_name: str | None = None,
    actor: str | None = None,
) -> dict[str, Any]:
    """Ingest raw content into the Recalium memory archive via MCP.

    Args:
        content: The raw text content to ingest (required, minimum 10 characters).
        source_metadata: Required source metadata from the client. Include source_type and
            source_name when available, plus conversation/session ids or source URI.
        client_identity: MCP client identity string for audit trail.
        import_method: Import method label, defaults to "mcp_tool".
        idempotency_key: Optional idempotency key; repeated calls return the same archive id.
        sensitivity_hint: Optional client-declared sensitivity hint.
        project_hint: Optional project/workspace hint.
        processing_mode: Requested processing mode. Stored for the worker/policy layer.
        source_name: Backward-compatible source label; source_metadata.source_name wins.
        actor: Backward-compatible client identity alias.

    Returns:
        On success: {"status": "accepted", "item_count": N, "archive_ids": [...]}
        On error: stable {"status": "error", "error": {code, message, details, retryable}}
    """
    # MCP-02: validate required fields and return descriptive errors
    if not content or not content.strip():
        return _mcp_error(
            "validation_error",
            "content is required and must be non-empty",
            field="content",
        )
    if len(content.strip()) < 10:
        return _mcp_error(
            "validation_error",
            "content too short (minimum 10 characters)",
            field="content",
        )
    if not source_metadata:
        return _mcp_error(
            "validation_error",
            "source_metadata is required for MCP ingestion",
            field="source_metadata",
        )

    # GPT5.6 #6: validate declared processing intent at the boundary so an
    # invalid mode/hint is rejected rather than silently ignored downstream.
    from app.domain.policy.resolver import (  # noqa: PLC0415
        ALLOWED_PROCESSING_MODES,
        ALLOWED_SENSITIVITY_HINTS,
        is_valid_processing_mode,
        is_valid_sensitivity_hint,
    )

    if not is_valid_processing_mode(processing_mode):
        return _mcp_error(
            "validation_error",
            f"invalid processing_mode {processing_mode!r}; allowed: {sorted(ALLOWED_PROCESSING_MODES)}",
            field="processing_mode",
        )
    if not is_valid_sensitivity_hint(sensitivity_hint):
        return _mcp_error(
            "validation_error",
            f"invalid sensitivity_hint {sensitivity_hint!r}; allowed: {sorted(ALLOWED_SENSITIVITY_HINTS)}",
            field="sensitivity_hint",
        )

    effective_actor = client_identity or actor or "mcp_client"
    source_type = str(source_metadata.get("source_type") or source_metadata.get("system") or "mcp")
    effective_source_name = str(
        source_metadata.get("source_name")
        or source_metadata.get("session_id")
        or source_metadata.get("conversation_id")
        or source_name
        or "MCP Import"
    )
    metadata = {
        "source_metadata": source_metadata,
        "client_identity": effective_actor,
        "import_method": import_method,
        "processing_mode": processing_mode,
        "transport": "mcp_sse",
    }
    if idempotency_key:
        metadata["idempotency_key"] = idempotency_key
    if sensitivity_hint:
        metadata["sensitivity_hint"] = sensitivity_hint
    if project_hint:
        metadata["project_hint"] = project_hint

    from app.domain.ingest.service import (  # noqa: PLC0415
        IdempotencyConflictError,
        ingest_text_content,
    )

    factory = get_session_factory()
    async with factory() as session:
        try:
            result = await ingest_text_content(
                session=session,
                content=content,
                source_name=effective_source_name,
                actor=effective_actor,
                source_type=source_type,
                extra_metadata=metadata,
                idempotency_key=idempotency_key,
            )
            await session.commit()
        except IdempotencyConflictError as exc:
            return _mcp_error("idempotency_conflict", str(exc), retryable=False)
        except ValueError as exc:
            return _mcp_error("validation_error", str(exc), retryable=False)
        except Exception:
            await session.rollback()
            logger.exception("MCP ingest_memory failed")
            return _mcp_error("internal_error", "Ingest failed due to an internal error.", retryable=True)

    return {
        "status": "accepted",
        "item_count": result.item_count,
        "archive_ids": [str(aid) for aid in result.archive_ids],
        "idempotent_replay": result.idempotent_replay,
        "idempotency_key": idempotency_key,
        "processing_mode": processing_mode,
    }


def create_mcp_server() -> FastMCP:
    """Return the configured MCP app instance.

    Used by main.py lifespan and by tests.
    SECURITY: Caller must bind to 127.0.0.1 only when using HTTP transport.
    """
    return mcp_app
