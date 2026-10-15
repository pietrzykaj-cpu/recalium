"""Review queue domain service.

Implements CANM-05: groups duplicate/overlapping facts for manageable review.

GPT5.6 #10: resolving/dismissing a conflict now has a real domain effect on the
involved facts (keep or suppress) and reindexes retrieval — it is no longer a bare
status flip.

All list queries filter source_status='active' per cascade contract.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from app.domain.derivations.visibility import enabled, current, HISTORY
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.derived_memory.models import ConflictGroup, Fact
from app.domain.review_queue.models import ReviewQueueItem

logger = logging.getLogger(__name__)

# Confidence ordering for choosing which fact to keep when suppressing duplicates.
_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}

# Allowed resolution actions (GPT5.6 #10).
RESOLUTION_ACTIONS = frozenset({"keep", "suppress"})


class ReviewItemNotFoundError(Exception):
    """Raised when a review queue item cannot be found."""


class InvalidResolutionActionError(Exception):
    """Raised when an unknown resolution action is requested (GPT5.6 #10)."""

class ReviewItemStaleError(Exception):
    """Historical/mixed-generation groups cannot be partially resolved as current."""

async def _current_group(session, group_id, *, lock=False):
    if not enabled():
        return True
    all_facts = (await session.scalars(select(Fact).where(Fact.conflict_group_id == group_id,
        Fact.source_status == 'active').execution_options(**HISTORY))).all()
    if lock:
        from app.domain.derivations.service import lock_archives_and_heads
        await lock_archives_and_heads(session, [f.raw_archive_id for f in all_facts])
        all_facts = (await session.scalars(select(Fact).where(Fact.conflict_group_id == group_id,
            Fact.source_status == 'active').execution_options(**HISTORY, populate_existing=True))).all()
    ids = {f.id for f in all_facts}
    visible = set((await session.scalars(select(Fact.id).where(Fact.id.in_(ids),
        Fact.source_status=='active'))).all())
    return bool(ids) and ids == visible


def _reindex_after_curation() -> None:
    """Invalidate the retrieval cache so suppressed facts leave results immediately."""
    try:
        from app.domain.retrieval.service import invalidate_cache  # noqa: PLC0415
        invalidate_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cache invalidation after review resolution failed: %s", exc)


async def _apply_resolution(
    session: AsyncSession,
    conflict_group_id: uuid.UUID,
    action: str,
) -> dict:
    """Apply the domain effect of a resolution to a conflict group's facts (GPT5.6 #10).

    * ``keep``     — the facts are legitimately distinct: clear their conflict flag so
                     they are no longer grouped, and keep them all active.
    * ``suppress`` — the facts are true duplicates: keep the highest-confidence (then
                     earliest) fact and suppress the rest (source_status='source_removed',
                     excluded from retrieval).

    Marks the ConflictGroup resolved. Returns a summary dict.
    """
    if enabled() and not await _current_group(session, conflict_group_id, lock=True):
        raise ReviewItemStaleError('Review group is no longer current')
    facts = (
        await session.execute(
            select(Fact).where(
                Fact.conflict_group_id == conflict_group_id,
                Fact.source_status == "active",
            )
        )
    ).scalars().all()

    suppressed = 0
    kept = 0
    if action == "suppress" and facts:
        ordered = sorted(
            facts,
            key=lambda f: (_CONFIDENCE_RANK.get(f.confidence_tier, 3), f.created_at),
        )
        primary = ordered[0]
        primary.conflict_group_id = None
        kept = 1
        for fact in ordered[1:]:
            fact.source_status = "source_removed"
            suppressed += 1
    else:  # keep — the facts coexist; just clear the conflict flag.
        for fact in facts:
            fact.conflict_group_id = None
            kept += 1

    group = (
        await session.execute(
            select(ConflictGroup).where(ConflictGroup.id == conflict_group_id)
        )
    ).scalar_one_or_none()
    if group is not None:
        group.resolved_at = datetime.now(timezone.utc)

    return {"kept": kept, "suppressed": suppressed}


async def materialize_review_item(
    session: AsyncSession,
    conflict_group_id: uuid.UUID,
    item_type: str,
) -> ReviewQueueItem:
    """Create a pending ReviewQueueItem for a conflict group (CANM-05).

    Will raise a DB integrity error if conflict_group_id does not reference
    an existing conflict_groups row (FK enforcement).
    """
    item = ReviewQueueItem(
        id=uuid.uuid4(),
        conflict_group_id=conflict_group_id,
        item_type=item_type,
        status="pending",
        source_status="active",
    )
    session.add(item)
    await session.flush()
    return item


async def list_pending_review_items(
    session: AsyncSession,
) -> list[ReviewQueueItem]:
    """Return all pending, active review queue items (CANM-05)."""
    from app.domain.derivations.readers import begin_reader_snapshot
    await begin_reader_snapshot(session)
    result = await session.execute(
        select(ReviewQueueItem).where(
            ReviewQueueItem.status == "pending",
            ReviewQueueItem.source_status == "active",
        )
    )
    items = list(result.scalars().all())
    if enabled():
        items = [item for item in items if await _current_group(session, item.conflict_group_id)]
    return items


async def resolve_review_item(
    session: AsyncSession,
    item_id: uuid.UUID,
    resolution_note: str,
    resolved_by: str,
    action: str = "keep",
) -> ReviewQueueItem:
    """Resolve a review item and apply its domain effect (CANM-05 / GPT5.6 #10).

    ``action`` is ``keep`` (facts coexist; conflict flag cleared) or ``suppress``
    (keep the best fact, suppress the duplicates). Retrieval is reindexed.

    Raises ReviewItemNotFoundError if the item does not exist and
    InvalidResolutionActionError for an unknown action.
    """
    if action not in RESOLUTION_ACTIONS:
        raise InvalidResolutionActionError(
            f"Unknown resolution action {action!r}; expected one of {sorted(RESOLUTION_ACTIONS)}."
        )
    result = await session.execute(
        select(ReviewQueueItem).where(ReviewQueueItem.id == item_id)
    )
    item = result.scalar_one_or_none()
    if item is None:
        raise ReviewItemNotFoundError(f"ReviewQueueItem {item_id} not found.")
    if enabled() and (item.source_status != 'active' or item.status != 'pending'):
        raise ReviewItemStaleError('Review item is no longer actionable')

    summary = await _apply_resolution(session, item.conflict_group_id, action)

    item.status = "resolved"
    item.resolution_note = resolution_note
    item.resolved_by = resolved_by
    item.resolved_at = datetime.now(timezone.utc)
    await session.flush()
    _reindex_after_curation()
    logger.info(
        "Resolved review item %s action=%s kept=%d suppressed=%d",
        item_id, action, summary["kept"], summary["suppressed"],
    )
    return item


async def dismiss_review_item(
    session: AsyncSession,
    item_id: uuid.UUID,
) -> ReviewQueueItem:
    """Dismiss a review item (CANM-05 / GPT5.6 #10).

    Dismissal means "these are not real duplicates": the facts coexist and their
    conflict flag is cleared so they are no longer grouped. Retrieval is reindexed.

    Raises ReviewItemNotFoundError if the item does not exist.
    """
    result = await session.execute(
        select(ReviewQueueItem).where(ReviewQueueItem.id == item_id)
    )
    item = result.scalar_one_or_none()
    if item is None:
        raise ReviewItemNotFoundError(f"ReviewQueueItem {item_id} not found.")
    if enabled() and (item.source_status != 'active' or item.status != 'pending'):
        raise ReviewItemStaleError('Review item is no longer actionable')

    await _apply_resolution(session, item.conflict_group_id, "keep")

    item.status = "dismissed"
    item.resolved_at = datetime.now(timezone.utc)
    await session.flush()
    _reindex_after_curation()
    return item
