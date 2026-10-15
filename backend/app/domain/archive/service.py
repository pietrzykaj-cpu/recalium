"""Archive domain service — deletion cascade.

PRIV-01: Deletion cascade immediately suppresses all derived data.
PRIV-02: Canonical memory is marked source_removed + required_review, NOT deleted.

GPT5.6 #2: deletion also crypto-erases the raw + derived plaintext in place and
records an append-only tombstone (DB table + external ledger) so the removed
content cannot survive in a later backup or be resurrected by a restore.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.archive.models import RawArchiveItem, Tombstone
from app.domain.archive.tombstones import (
    DEFAULT_BACKUP_DIR,
    REDACTION_MARKER,
    append_tombstone_ledger,
    read_tombstone_ledger,
)
from app.domain.audit.models import AuditEvent

logger = logging.getLogger(__name__)


class ArchiveItemNotFoundError(Exception):
    """Raised when the archive item does not exist or is already deleted."""


async def _suppress_derived(session: AsyncSession, archive_id_str: str) -> None:
    """Mark all derived rows for a source as source_removed (PRIV-01/PRIV-02)."""
    from app.domain.derivations.privacy import lock_existing_heads, suppress_generations
    await lock_existing_heads(session, [uuid.UUID(archive_id_str)])
    await suppress_generations(session, archive_id_str)
    for table in ("summaries", "facts", "embeddings", "fts_entries"):
        await session.execute(
            text(
                f"UPDATE {table} SET source_status = 'source_removed' "
                "WHERE raw_archive_id = :archive_id AND source_status = 'active'"
            ),
            {"archive_id": archive_id_str},
        )

    await session.execute(
        text(
            "UPDATE conflict_groups SET source_status = 'source_removed' "
            "WHERE id IN ("
            "  SELECT DISTINCT conflict_group_id FROM facts "
            "  WHERE raw_archive_id = :archive_id AND conflict_group_id IS NOT NULL"
            ") AND source_status = 'active'"
        ),
        {"archive_id": archive_id_str},
    )

    await session.execute(
        text(
            "UPDATE review_queue_items SET source_status = 'source_removed' "
            "WHERE conflict_group_id IN ("
            "  SELECT DISTINCT conflict_group_id FROM facts "
            "  WHERE raw_archive_id = :archive_id AND conflict_group_id IS NOT NULL"
            ") AND source_status = 'active'"
        ),
        {"archive_id": archive_id_str},
    )

    await session.execute(
        text(
            "UPDATE canonical_memory "
            "SET source_status = 'source_removed', status = 'required_review' "
            "WHERE raw_archive_id = :archive_id AND source_status = 'active'"
        ),
        {"archive_id": archive_id_str},
    )


async def _erase_plaintext(session: AsyncSession, archive_id_str: str) -> None:
    """Crypto-erase the raw and derived plaintext for a source (GPT5.6 #2).

    Overwrites the content-bearing columns with a redaction marker so the removed
    bytes cannot survive in a subsequent backup. The one-way ``content_hash`` is
    preserved so tombstone reapply can still identify a restored copy. Canonical
    memory text is intentionally NOT erased (PRIV-02 keeps it for review), but it
    is suppressed and excluded from retrieval.
    """
    from app.domain.derivations.privacy import erase_projections
    await erase_projections(session, archive_id_str, REDACTION_MARKER)
    await session.execute(
        text("UPDATE raw_archive SET raw_content = :m WHERE id = :id"),
        {"m": REDACTION_MARKER, "id": archive_id_str},
    )
    await session.execute(
        text("UPDATE summaries SET summary_text = :m WHERE raw_archive_id = :id"),
        {"m": REDACTION_MARKER, "id": archive_id_str},
    )
    await session.execute(
        text(
            "UPDATE facts SET fact_text = :m, source_span = :m WHERE raw_archive_id = :id"
        ),
        {"m": REDACTION_MARKER, "id": archive_id_str},
    )
    # Also drop the FTS index payload so the text is neither stored nor searchable.
    await session.execute(
        text(
            "UPDATE fts_entries SET text_content = :m, search_vector = NULL "
            "WHERE raw_archive_id = :id"
        ),
        {"m": REDACTION_MARKER, "id": archive_id_str},
    )


async def cascade_delete_archive_item(
    session: AsyncSession,
    archive_id: uuid.UUID,
    actor: str = "user_ui",
    reason: str | None = None,
    backup_dir: str = DEFAULT_BACKUP_DIR,
) -> None:
    """Soft-delete a raw archive item and cascade source_removed to all derived data.

    Atomic: all updates committed together or none committed.
    PRIV-01: Derived data marked source_removed.
    PRIV-02: Canonical memory marked source_removed + required_review (NOT deleted).
    GPT5.6 #2: raw/derived plaintext is crypto-erased and a durable tombstone
    (DB row + external append-only ledger) is written so the content cannot survive
    in a backup or be resurrected by a restore.
    """
    result = await session.execute(
        select(RawArchiveItem).where(
            RawArchiveItem.id == archive_id,
            RawArchiveItem.deleted_at.is_(None),
        ).with_for_update()
    )
    item = result.scalar_one_or_none()
    if item is None:
        raise ArchiveItemNotFoundError(
            f"Archive item {archive_id} not found or already deleted"
        )

    now = datetime.now(timezone.utc)
    archive_id_str = str(archive_id)
    content_hash = item.content_hash

    item.deleted_at = now

    await _suppress_derived(session, archive_id_str)
    await _erase_plaintext(session, archive_id_str)

    # Durable tombstone in the database (included in future backups).
    session.add(
        Tombstone(
            source_id=archive_id,
            content_hash=content_hash,
            removal_type="delete",
            removed_at=now,
            actor=actor,
            reason=reason,
        )
    )

    audit_event = AuditEvent(
        event_type="archive_delete",
        raw_archive_id=archive_id,
        actor=actor,
        operation_metadata={
            "archive_id": archive_id_str,
            "deleted_at": now.isoformat(),
            "content_hash": content_hash,
            "erased": True,
        },
    )
    session.add(audit_event)

    await session.commit()

    # External append-only ledger (outside the DB dump) — survives restore of a
    # backup taken before this deletion. Best-effort with an error log on failure.
    append_tombstone_ledger(
        {
            "source_id": archive_id_str,
            "content_hash": content_hash,
            "removal_type": "delete",
            "removed_at": now.isoformat(),
            "actor": actor,
            "reason": reason,
        },
        backup_dir,
    )

    try:
        from app.domain.retrieval.service import invalidate_cache  # noqa: PLC0415
        invalidate_cache()
    except Exception as e:
        logger.warning("Could not invalidate retrieval cache after deletion: %s", e)

    logger.info("Cascade delete complete for archive_id=%s actor=%s", archive_id_str, actor)


async def suppress_new_derivations_if_deleted(
    session: AsyncSession,
    raw_archive_id: uuid.UUID,
) -> bool:
    """Reconcile derivations against a concurrent deletion (GPT5.6 #9).

    The processing pipeline checks ``deleted_at`` when it loads a source, then makes
    slow external calls, then writes derived rows much later. A deletion in that
    window only suppresses the rows that existed *when it ran*, so derivations
    written afterwards would stay ``active`` and resurrect the deleted content.

    Called at the end of the pipeline: takes a ``FOR UPDATE`` row lock on the source
    and, if it has since been deleted, re-runs the same suppression + crypto-erase
    over *all* its derived rows (including any just written). The matching row lock
    in ``cascade_delete_archive_item`` serializes the two paths, so both interleavings
    converge on "all derivations suppressed". Returns True if a deletion was
    reconciled, False otherwise.
    """
    row = (
        await session.execute(
            select(RawArchiveItem.deleted_at)
            .where(RawArchiveItem.id == raw_archive_id)
            .with_for_update()
        )
    ).one_or_none()
    if row is None or row[0] is None:
        # Source still live (or gone entirely) — nothing to reconcile. Release lock.
        await session.commit()
        return False

    archive_id_str = str(raw_archive_id)
    await _suppress_derived(session, archive_id_str)
    await _erase_plaintext(session, archive_id_str)
    session.add(
        AuditEvent(
            event_type="deletion_race_reconciled",
            raw_archive_id=raw_archive_id,
            actor="pipeline_worker",
            operation_metadata={
                "archive_id": archive_id_str,
                "note": "derivations written during processing suppressed post-deletion",
            },
        )
    )
    await session.commit()

    try:
        from app.domain.retrieval.service import invalidate_cache  # noqa: PLC0415
        invalidate_cache()
    except Exception as e:
        logger.warning("Cache invalidation after deletion-race reconcile failed: %s", e)

    logger.warning(
        "Deletion-race reconciled for archive_id=%s: derivations written during "
        "processing were suppressed post-deletion.",
        archive_id_str,
    )
    return True


async def reapply_tombstones(
    session: AsyncSession,
    backup_dir: str = DEFAULT_BACKUP_DIR,
) -> dict:
    """Re-suppress and re-erase any content covered by a tombstone (GPT5.6 #2).

    Called after a restore. The external append-only ledger is the source of truth
    because it survives the restore of a pre-deletion backup that lacks the DB
    tombstone. For every ledger entry we crypto-erase and suppress any restored
    ``raw_archive`` row with the same ``content_hash`` that is not already removed,
    guaranteeing deleted content never becomes retrievable again.

    Returns a summary dict: {tombstones, reapplied}.
    """
    ledger = read_tombstone_ledger(backup_dir)
    # Acquire ALL source rows before any head/generation for a multi-source restore.
    hashes = [r.get('content_hash') for r in ledger if r.get('content_hash')]
    locked = (await session.scalars(select(RawArchiveItem).where(
        RawArchiveItem.content_hash.in_(hashes), RawArchiveItem.deleted_at.is_(None))
        .order_by(RawArchiveItem.id).with_for_update())).all()
    from app.domain.derivations.privacy import lock_existing_heads
    await lock_existing_heads(session, [r.id for r in locked])
    reapplied = 0

    for record in ledger:
        content_hash = record.get("content_hash")
        if not content_hash:
            continue
        rows = (
            await session.execute(
                select(RawArchiveItem).where(
                    RawArchiveItem.content_hash == content_hash,
                    RawArchiveItem.deleted_at.is_(None),
                )
            )
        ).scalars().all()
        for item in rows:
            archive_id_str = str(item.id)
            now = datetime.now(timezone.utc)
            item.deleted_at = now
            await _suppress_derived(session, archive_id_str)
            await _erase_plaintext(session, archive_id_str)
            # Ensure a DB tombstone exists for this restored copy.
            existing = (
                await session.execute(
                    select(Tombstone).where(
                        Tombstone.source_id == item.id,
                        Tombstone.content_hash == content_hash,
                    )
                )
            ).scalar_one_or_none()
            if existing is None:
                session.add(
                    Tombstone(
                        source_id=item.id,
                        content_hash=content_hash,
                        removal_type="delete",
                        removed_at=now,
                        actor=str(record.get("actor") or "restore_reapply"),
                        reason="tombstone reapplied after restore",
                    )
                )
            session.add(
                AuditEvent(
                    event_type="tombstone_reapplied",
                    raw_archive_id=item.id,
                    actor="restore_reapply",
                    operation_metadata={
                        "archive_id": archive_id_str,
                        "content_hash": content_hash,
                    },
                )
            )
            reapplied += 1

    if reapplied:
        await session.commit()
        try:
            from app.domain.retrieval.service import invalidate_cache  # noqa: PLC0415
            invalidate_cache()
        except Exception as e:
            logger.warning("Cache invalidation after tombstone reapply failed: %s", e)

    logger.info(
        "Tombstone reapply complete: %d ledger entries, %d rows re-suppressed",
        len(ledger),
        reapplied,
    )
    return {"tombstones": len(ledger), "reapplied": reapplied}
