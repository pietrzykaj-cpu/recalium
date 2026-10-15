"""Tags and fact-links API routes.

LINK-01: GET /api/tags — list all tags with usage counts
LINK-02: GET /api/facts/{fact_id}/tags — list tags for a specific fact
LINK-03: GET /api/facts/{fact_id}/links — list outgoing memory links from a fact
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.db import get_session
from app.domain.derivations.visibility import current_sql, enabled
from app.domain.derivations.readers import begin_reader_snapshot

router = APIRouter(prefix="/api", tags=["tags"])


@router.get("/tags")
async def list_tags(
    session: AsyncSession = Depends(get_session),
) -> dict:
    """List all tags with their usage count (number of active facts tagged).

    Only counts fact associations through active facts.
    """
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

    return {
        "tags": [
            {
                "id": row["id"],
                "name": row["name"],
                "created_at": row["created_at"],
                "fact_count": int(row["fact_count"] or 0),
            }
            for row in rows
        ],
        "total": len(rows),
    }


@router.get("/facts/{fact_id}/tags")
async def list_fact_tags(
    fact_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """List tags assigned to a specific fact."""
    await begin_reader_snapshot(session)
    # Verify fact exists and is active
    fact_check = (await session.execute(
        text("SELECT id FROM facts f WHERE id = :fid AND source_status = 'active' AND review_status = 'active' " + current_sql('f') + " LIMIT 1"),
        {"fid": str(fact_id)},
    )).fetchone()
    if fact_check is None:
        raise HTTPException(status_code=404, detail="Fact not found or not active")

    rows = (await session.execute(
        text("""
            SELECT
                t.id::text AS tag_id,
                t.name,
                ft.assigned_by,
                ft.created_at::text AS assigned_at
            FROM fact_tags ft
            JOIN tags t ON t.id = ft.tag_id
            WHERE ft.fact_id = :fid
            ORDER BY t.name ASC
        """),
        {"fid": str(fact_id)},
    )).mappings().all()

    return {
        "fact_id": str(fact_id),
        "tags": [
            {
                "tag_id": row["tag_id"],
                "name": row["name"],
                "assigned_by": row["assigned_by"],
                "assigned_at": row["assigned_at"],
            }
            for row in rows
        ],
    }


@router.get("/facts/{fact_id}/links")
async def list_fact_links(
    fact_id: uuid.UUID,
    direction: str = "outgoing",
    session: AsyncSession = Depends(get_session),
) -> dict:
    """List memory links for a specific fact.

    direction='outgoing' (default): links where this fact is the source.
    direction='incoming': links where this fact is the target.
    direction='both': all links involving this fact.
    """
    if direction not in ("outgoing", "incoming", "both"):
        raise HTTPException(
            status_code=422,
            detail="direction must be 'outgoing', 'incoming', or 'both'",
        )
    await begin_reader_snapshot(session)

    # Verify fact exists and is active
    fact_check = (await session.execute(
        text("SELECT id FROM facts f WHERE id = :fid AND source_status = 'active' AND review_status = 'active' " + current_sql('f') + " LIMIT 1"),
        {"fid": str(fact_id)},
    )).fetchone()
    if fact_check is None:
        raise HTTPException(status_code=404, detail="Fact not found or not active")

    if direction == "outgoing":
        where_clause = "ml.source_fact_id = :fid"
        other_id_col = "ml.target_fact_id::text AS other_fact_id"
        other_text_join = "JOIN facts tf ON tf.id = ml.target_fact_id"
        other_text_col = "tf.fact_text AS other_fact_text"
        other_status = "AND tf.source_status = 'active' AND tf.review_status = 'active'" + current_sql('tf')
    elif direction == "incoming":
        where_clause = "ml.target_fact_id = :fid"
        other_id_col = "ml.source_fact_id::text AS other_fact_id"
        other_text_join = "JOIN facts sf ON sf.id = ml.source_fact_id"
        other_text_col = "sf.fact_text AS other_fact_text"
        other_status = "AND sf.source_status = 'active' AND sf.review_status = 'active'" + current_sql('sf')
    else:  # both
        where_clause = "(ml.source_fact_id = :fid OR ml.target_fact_id = :fid)"
        other_id_col = """
            CASE
              WHEN ml.source_fact_id = :fid THEN ml.target_fact_id::text
              ELSE ml.source_fact_id::text
            END AS other_fact_id"""
        other_text_join = """
            LEFT JOIN facts tf ON tf.id = ml.target_fact_id
            LEFT JOIN facts sf ON sf.id = ml.source_fact_id"""
        other_text_col = """
            CASE
              WHEN ml.source_fact_id = :fid THEN tf.fact_text
              ELSE sf.fact_text
            END AS other_fact_text"""
        other_status = f"""
                            AND (
                                (ml.source_fact_id = :fid AND tf.source_status = 'active' AND tf.review_status = 'active' {current_sql('tf')})
                                OR (ml.target_fact_id = :fid AND sf.source_status = 'active' AND sf.review_status = 'active' {current_sql('sf')})
                            )"""

    rows = (await session.execute(
        text(f"""
            SELECT
                ml.id::text AS link_id,
                ml.link_type,
                ml.confidence,
                ml.entity_name,
                ml.created_by,
                ml.created_at::text AS created_at,
                {other_id_col},
                {other_text_col}
            FROM memory_links ml
            {other_text_join}
            WHERE {where_clause}
              {other_status}
            ORDER BY ml.created_at DESC
        """),
        {"fid": str(fact_id)},
    )).mappings().all()

    return {
        "fact_id": str(fact_id),
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


@router.get("/tags/{tag_id}/facts")
async def list_tag_facts(
    tag_id: uuid.UUID,
    limit: int = Query(200, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """List active facts tagged with a specific tag.

    Powers the Explore tag/entity browser. Returns the tag's name plus the
    active facts carrying it; 404 if the tag does not exist.
    """
    await begin_reader_snapshot(session)
    tag_row = (await session.execute(
        text("SELECT id::text AS id, name FROM tags WHERE id = :tid LIMIT 1"),
        {"tid": str(tag_id)},
    )).mappings().fetchone()
    if tag_row is None:
        raise HTTPException(status_code=404, detail="Tag not found")

    rows = (await session.execute(
        text(f"""
            SELECT
                f.id::text AS id,
                f.fact_text,
                f.confidence_tier,
                f.review_status,
                f.raw_archive_id::text AS raw_archive_id,
                f.created_at::text AS created_at,
                COUNT(*) OVER() AS total_count
            FROM fact_tags ft
            JOIN facts f ON f.id = ft.fact_id
            WHERE ft.tag_id = :tid
              AND f.source_status = 'active'
              AND f.review_status = 'active'
              {current_sql('f')}
            ORDER BY f.created_at DESC
            LIMIT :limit
        """),
        {"tid": str(tag_id), "limit": limit},
    )).mappings().all()

    # total matching active facts (before LIMIT) — so `count` is meaningful even
    # when the returned `facts` page is truncated to `limit`.
    total = int(rows[0]["total_count"]) if rows else 0
    return {
        "tag_id": str(tag_id),
        "name": tag_row["name"],
        "facts": [
            {
                "id": row["id"],
                "fact_text": row["fact_text"],
                "confidence_tier": row["confidence_tier"],
                "review_status": row["review_status"],
                "raw_archive_id": row["raw_archive_id"],
                "created_at": row["created_at"],
            }
            for row in rows
        ],
        "count": total,
    }
