"""Facts API routes.

Provides extracted fact review data for the frontend Facts page.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.audit.models import AuditEvent
from app.domain.derived_memory.models import Fact
from app.infrastructure.db import get_session

router = APIRouter(prefix="/api/facts", tags=["facts"])

VISIBLE_REVIEW_STATUSES = ("active", "disputed", "stale")
VALID_REVIEW_STATUSES = set(VISIBLE_REVIEW_STATUSES) | {"archived", "deleted"}
VALID_CONFIDENCE_TIERS = {"high", "medium", "low"}


class UpdateFactBody(BaseModel):
    fact_text: str | None = None
    source_span: str | None = None
    confidence_tier: str | None = None
    review_status: str | None = None
    actor: str = "user_ui"


class FactActionBody(BaseModel):
    actor: str = "user_ui"


def _fact_to_dict(fact: Fact) -> dict[str, Any]:
    return {
        "id": str(fact.id),
        "raw_archive_id": str(fact.raw_archive_id),
        "fact_text": fact.fact_text,
        "source_span": fact.source_span,
        "confidence_tier": fact.confidence_tier,
        "derivation_method": fact.derivation_method,
        "derivation_model": fact.derivation_model,
        "conflict_group_id": str(fact.conflict_group_id) if fact.conflict_group_id else None,
        "source_status": fact.source_status,
        "review_status": fact.review_status,
        "created_at": fact.created_at.isoformat() if fact.created_at else None,
    }


async def _get_mutable_fact(session: AsyncSession, fact_id: uuid.UUID) -> Fact:
    fact = await session.scalar(
        select(Fact).where(Fact.id == fact_id).where(Fact.source_status == "active")
    )
    if fact is None:
        raise HTTPException(status_code=404, detail=f"Fact {fact_id} not found")
    return fact


def _validate_review_status(review_status: str) -> None:
    if review_status not in VALID_REVIEW_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"review_status must be one of: {', '.join(sorted(VALID_REVIEW_STATUSES))}",
        )


def _validate_confidence_tier(confidence_tier: str) -> None:
    if confidence_tier not in VALID_CONFIDENCE_TIERS:
        raise HTTPException(
            status_code=422,
            detail=f"confidence_tier must be one of: {', '.join(sorted(VALID_CONFIDENCE_TIERS))}",
        )


def _add_fact_audit_event(
    session: AsyncSession,
    *,
    event_type: str,
    fact: Fact,
    actor: str,
    operation_metadata: dict[str, Any],
) -> None:
    session.add(
        AuditEvent(
            event_type=event_type,
            raw_archive_id=fact.raw_archive_id,
            actor=actor,
            operation_metadata={"fact_id": str(fact.id), **operation_metadata},
        )
    )


async def _list_facts_response(
    session: AsyncSession,
    *,
    limit: int,
    offset: int,
    source_status: str,
    review_status: str | None,
    confidence_tier: str | None,
) -> dict[str, Any]:
    from app.domain.derivations.readers import begin_reader_snapshot
    await begin_reader_snapshot(session)
    query = select(Fact).where(Fact.source_status == source_status)
    if review_status is None:
        query = query.where(Fact.review_status.in_(VISIBLE_REVIEW_STATUSES))
    elif review_status != "all":
        _validate_review_status(review_status)
        query = query.where(Fact.review_status == review_status)
    if confidence_tier:
        _validate_confidence_tier(confidence_tier)
        query = query.where(Fact.confidence_tier == confidence_tier)

    count_query = select(func.count()).select_from(query.subquery())
    count = int(await session.scalar(count_query) or 0)

    rows = (await session.execute(
        query.order_by(Fact.created_at.desc()).offset(offset).limit(limit)
    )).scalars().all()

    return {
        "facts": [_fact_to_dict(fact) for fact in rows],
        "count": count,
        "limit": limit,
        "offset": offset,
    }


@router.get("")
async def list_facts_no_trailing_slash(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    source_status: str = Query("active"),
    review_status: str | None = Query(None),
    confidence_tier: str | None = Query(None),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """List extracted facts for review.

    Defaults to active facts so source-removed facts are not shown in normal
    review surfaces.
    """
    return await _list_facts_response(
        session,
        limit=limit,
        offset=offset,
        source_status=source_status,
        review_status=review_status,
        confidence_tier=confidence_tier,
    )


@router.get("/")
async def list_facts(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    source_status: str = Query("active"),
    review_status: str | None = Query(None),
    confidence_tier: str | None = Query(None),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """List extracted facts for review with trailing slash compatibility."""
    return await _list_facts_response(
        session,
        limit=limit,
        offset=offset,
        source_status=source_status,
        review_status=review_status,
        confidence_tier=confidence_tier,
    )


@router.get("/{fact_id}")
async def get_fact(
    fact_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Return a single extracted fact by id.

    Powers the memory detail view. Returns the fact regardless of review
    status so a linked/promoted fact still resolves; 404 if it does not exist.
    """
    from app.domain.derivations.visibility import enabled, HISTORY
    from app.domain.derivations.readers import begin_reader_snapshot
    await begin_reader_snapshot(session)
    fact = await session.scalar(select(Fact).where(Fact.id == fact_id).execution_options(
        **HISTORY, populate_existing=enabled()))
    if fact is None:
        raise HTTPException(status_code=404, detail=f"Fact {fact_id} not found")
    result = _fact_to_dict(fact)
    if enabled():
        from app.domain.derivations.models import DerivationGeneration, ArchiveDerivationHead
        if fact.generation_id is not None:
            generation = await session.get(DerivationGeneration, fact.generation_id)
            if generation is None or generation.published_at is None:
                raise HTTPException(status_code=404, detail='Fact not found')
        head = await session.get(ArchiveDerivationHead, fact.raw_archive_id)
        is_current = (head.active_generation_id == fact.generation_id if head and head.active_generation_id
                      else fact.generation_id is None and (head is None or head.legacy_retired_at is None))
        result.update(generation_id=str(fact.generation_id) if fact.generation_id else None,
                      generation_visibility='current' if is_current else 'history')
    return result


@router.patch("/{fact_id}")
async def update_fact(
    fact_id: uuid.UUID,
    body: UpdateFactBody,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Edit reviewable fields on an extracted fact and record an audit event."""
    fact = await _get_mutable_fact(session, fact_id)
    updated_fields: list[str] = []

    if body.fact_text is not None:
        if not body.fact_text.strip():
            raise HTTPException(status_code=422, detail="fact_text must not be empty")
        fact.fact_text = body.fact_text
        updated_fields.append("fact_text")
    if body.source_span is not None:
        if not body.source_span.strip():
            raise HTTPException(status_code=422, detail="source_span must not be empty")
        fact.source_span = body.source_span
        updated_fields.append("source_span")
    if body.confidence_tier is not None:
        _validate_confidence_tier(body.confidence_tier)
        fact.confidence_tier = body.confidence_tier
        updated_fields.append("confidence_tier")
    if body.review_status is not None:
        _validate_review_status(body.review_status)
        fact.review_status = body.review_status
        updated_fields.append("review_status")

    if not updated_fields:
        raise HTTPException(status_code=422, detail="At least one editable field is required")

    _add_fact_audit_event(
        session,
        event_type="fact_updated",
        fact=fact,
        actor=body.actor,
        operation_metadata={"updated_fields": updated_fields},
    )
    await session.flush()
    return _fact_to_dict(fact)


async def _set_fact_review_status(
    session: AsyncSession,
    *,
    fact_id: uuid.UUID,
    review_status: str,
    event_type: str,
    actor: str,
) -> Fact:
    fact = await _get_mutable_fact(session, fact_id)
    previous_status = fact.review_status
    fact.review_status = review_status
    _add_fact_audit_event(
        session,
        event_type=event_type,
        fact=fact,
        actor=actor,
        operation_metadata={"previous_review_status": previous_status, "review_status": review_status},
    )
    await session.flush()
    return fact


@router.post("/{fact_id}/dispute")
async def dispute_fact(
    fact_id: uuid.UUID,
    body: FactActionBody | None = None,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    fact = await _set_fact_review_status(
        session,
        fact_id=fact_id,
        review_status="disputed",
        event_type="fact_marked_disputed",
        actor=(body.actor if body else "user_ui"),
    )
    return _fact_to_dict(fact)


@router.post("/{fact_id}/mark-stale")
async def stale_fact(
    fact_id: uuid.UUID,
    body: FactActionBody | None = None,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    fact = await _set_fact_review_status(
        session,
        fact_id=fact_id,
        review_status="stale",
        event_type="fact_marked_stale",
        actor=(body.actor if body else "user_ui"),
    )
    return _fact_to_dict(fact)


@router.post("/{fact_id}/archive")
async def archive_fact(
    fact_id: uuid.UUID,
    body: FactActionBody | None = None,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    fact = await _set_fact_review_status(
        session,
        fact_id=fact_id,
        review_status="archived",
        event_type="fact_archived",
        actor=(body.actor if body else "user_ui"),
    )
    return _fact_to_dict(fact)


@router.delete("/{fact_id}", status_code=204)
async def delete_fact(
    fact_id: uuid.UUID,
    actor: str = Query("user_ui"),
    session: AsyncSession = Depends(get_session),
) -> None:
    await _set_fact_review_status(
        session,
        fact_id=fact_id,
        review_status="deleted",
        event_type="fact_deleted",
        actor=actor,
    )
