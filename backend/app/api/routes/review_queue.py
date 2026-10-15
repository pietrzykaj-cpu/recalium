"""Review queue API routes.

CANM-05: Groups duplicate/overlapping facts for manageable cleanup.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.archive.models import RawArchiveItem
from app.domain.derived_memory.models import ConflictGroup, Fact
from app.domain.review_queue.service import (
    InvalidResolutionActionError,
    ReviewItemNotFoundError,
    ReviewItemStaleError,
    dismiss_review_item,
    list_pending_review_items,
    resolve_review_item,
)
from app.infrastructure.db import get_session

router = APIRouter(prefix="/api/review-queue", tags=["review-queue"])


class ResolveBody(BaseModel):
    resolved_by: str = "user_ui"
    resolution_note: str = ""
    # GPT5.6 #10: 'keep' (facts coexist) or 'suppress' (keep the best, drop the rest).
    action: str = "keep"


def _fact_to_dict(fact: Fact, archive: RawArchiveItem | None) -> dict:
    return {
        "id": str(fact.id),
        "raw_archive_id": str(fact.raw_archive_id),
        "fact_text": fact.fact_text,
        "source_span": fact.source_span,
        "confidence_tier": fact.confidence_tier,
        "derivation_method": fact.derivation_method,
        "derivation_model": fact.derivation_model,
        "source_status": fact.source_status,
        "review_status": fact.review_status,
        "source_name": archive.source_name if archive else None,
        "source_type": archive.source_type if archive else None,
        "created_at": fact.created_at.isoformat() if fact.created_at else None,
    }


async def _conflict_group_details(session: AsyncSession, group_id: uuid.UUID) -> dict:
    group_result = await session.execute(
        select(ConflictGroup).where(
            ConflictGroup.id == group_id,
            ConflictGroup.source_status == "active",
        )
    )
    group = group_result.scalar_one_or_none()

    facts_result = await session.execute(
        select(Fact, RawArchiveItem)
        .join(RawArchiveItem, RawArchiveItem.id == Fact.raw_archive_id)
        .where(
            Fact.conflict_group_id == group_id,
            Fact.source_status == "active",
            Fact.review_status.in_(["active", "disputed", "stale"]),
            RawArchiveItem.deleted_at.is_(None),
        )
        .order_by(Fact.confidence_tier.asc(), Fact.created_at.asc())
    )
    facts = [_fact_to_dict(fact, archive) for fact, archive in facts_result.all()]
    return {
        "group_type": group.group_type if group else None,
        "group_source_status": group.source_status if group else None,
        "fact_count": len(facts),
        "facts": facts,
    }


async def _item_to_dict(item, session: AsyncSession, include_details: bool = True) -> dict:
    details = await _conflict_group_details(session, item.conflict_group_id) if include_details else {}
    return {
        "id": str(item.id),
        "conflict_group_id": str(item.conflict_group_id),
        "item_type": item.item_type,
        "status": item.status,
        "source_status": item.source_status,
        "resolution_note": item.resolution_note,
        "resolved_by": item.resolved_by,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "resolved_at": item.resolved_at.isoformat() if item.resolved_at else None,
        **details,
    }


@router.get("")
async def list_items(session: AsyncSession = Depends(get_session)) -> dict:
    items = await list_pending_review_items(session)
    return {"items": [await _item_to_dict(i, session) for i in items], "count": len(items)}


@router.post("/{item_id}/resolve")
async def resolve(
    item_id: uuid.UUID,
    body: ResolveBody,
    session: AsyncSession = Depends(get_session),
) -> dict:
    try:
        item = await resolve_review_item(
            session, item_id,
            resolution_note=body.resolution_note,
            resolved_by=body.resolved_by,
            action=body.action,
        )
    except ReviewItemNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ReviewItemStaleError:
        raise HTTPException(status_code=409, detail='Review group is no longer current')
    except InvalidResolutionActionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return await _item_to_dict(item, session)


@router.post("/{item_id}/dismiss")
async def dismiss(
    item_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict:
    try:
        item = await dismiss_review_item(session, item_id)
    except ReviewItemNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ReviewItemStaleError:
        raise HTTPException(status_code=409, detail='Review group is no longer current')
    return await _item_to_dict(item, session)
