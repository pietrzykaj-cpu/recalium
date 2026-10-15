"""Canonical memory domain service.

Implements CANM-01 (inspect/edit/delete/promote), CANM-02 (retrieval priority),
CANM-03 (explicit action only), CANM-04 (source_span confirmation).

All read queries filter source_status='active' per cascade contract.

GPT5.6 #9: promotion integrity is server-authoritative. Source linkage and the
source-span attestation that drives the CANM-04 gate are derived from the stored
fact under a row lock — never from client-supplied fields — so a caller cannot
forge ``has_source_span``, mis-link a source, or promote removed content.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.canonical_memory.models import CanonicalMemoryItem


class PromotionNotConfirmedError(Exception):
    """Raised when promoting a fact without source_span and confirmed=False (CANM-04)."""


class CanonicalItemNotFoundError(Exception):
    """Raised when a canonical item cannot be found."""


class FactNotFoundError(Exception):
    """Raised when the fact to promote does not exist (GPT5.6 #9)."""


class SourceMismatchError(Exception):
    """Raised when a client-supplied raw_archive_id contradicts the fact's source (GPT5.6 #9)."""


class SourceRemovedError(Exception):
    """Raised when promoting a fact whose source has been removed/suppressed (GPT5.6 #9)."""


async def promote_fact_to_canonical(
    session: AsyncSession,
    fact_id: uuid.UUID,
    content: str,
    promoted_by: str,
    raw_archive_id: uuid.UUID | None = None,
    has_source_span: bool | None = None,  # deprecated/ignored — server derives it
    confirmed: bool = False,
) -> CanonicalMemoryItem:
    """Promote a derived-memory fact into canonical memory (CANM-03/CANM-04).

    GPT5.6 #9 — promotion integrity is server-authoritative:

    * The fact is loaded under a ``FOR UPDATE`` row lock, which serializes with
      deletion so a fact cannot be promoted while it is being removed.
    * Source linkage (``raw_archive_id``) and the source-span attestation that
      drives the CANM-04 confirmation gate are taken from the **stored fact**, never
      from the client. A client cannot set ``has_source_span=True`` to bypass the
      gate for a fact that has no real span. The legacy ``has_source_span`` argument
      is accepted for wire compatibility but ignored.
    * A removed/suppressed fact (or one whose source is deleted) cannot be promoted,
      so promotion can never resurrect deleted content into the highest-trust tier.
    * Server-computed provenance (verbatim source span + source ``content_hash``) is
      recorded, giving canonical memory a verifiable derivation chain.

    Client ``content`` remains user-editable curation (CANM-01).
    """
    from app.domain.archive.models import RawArchiveItem  # noqa: PLC0415
    from app.domain.derived_memory.models import Fact  # noqa: PLC0415
    from app.domain.derivations.visibility import enabled, HISTORY

    if enabled():
        # Keep the managed archive/head order even for explicit historical curation.
        # Never take a fact lock before the source lock used by deletion/publication.
        source_id = await session.scalar(select(Fact.raw_archive_id).where(
            Fact.id == fact_id).execution_options(**HISTORY))
        if source_id is None:
            raise FactNotFoundError(f"Fact {fact_id} not found.")
        from app.domain.derivations.service import lock_archives_and_heads
        await lock_archives_and_heads(session, [source_id])

    fact = (
        await session.execute(
            select(Fact).where(Fact.id == fact_id).with_for_update().execution_options(
                **HISTORY, populate_existing=enabled())
        )
    ).scalar_one_or_none()
    if fact is None:
        raise FactNotFoundError(f"Fact {fact_id} not found.")
    if enabled() and fact.generation_id is not None:
        from app.domain.derivations.models import DerivationGeneration
        generation = await session.get(DerivationGeneration, fact.generation_id)
        if generation is None or generation.published_at is None:
            raise FactNotFoundError('Unpublished candidate cannot be promoted')

    # A removed fact must not be promotable — that would resurrect deleted content.
    if fact.source_status != "active":
        raise SourceRemovedError(
            f"Fact {fact_id} is not active (source_status={fact.source_status}); "
            "it cannot be promoted."
        )

    # Source linkage is server-authoritative. Reject a contradicting client value
    # rather than silently trusting it.
    if raw_archive_id is not None and raw_archive_id != fact.raw_archive_id:
        raise SourceMismatchError(
            "raw_archive_id does not match the fact's source; promotion refused."
        )
    source_id = fact.raw_archive_id

    # The underlying source item must itself be live.
    source = (
        await session.execute(
            select(RawArchiveItem)
            .where(RawArchiveItem.id == source_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if source is None or source.deleted_at is not None:
        raise SourceRemovedError(
            "Source archive item is missing or deleted; promotion refused."
        )

    # CANM-04 gate keyed off the REAL stored span, not the client's claim.
    server_has_source_span = bool((fact.source_span or "").strip())
    if not server_has_source_span and not confirmed:
        raise PromotionNotConfirmedError(
            "Facts without a source span require confirmed=True to promote to canonical memory."
        )

    provenance = {
        "fact_id": str(fact.id),
        "raw_archive_id": str(source_id),
        "source_span": fact.source_span or "",
        "source_content_hash": source.content_hash,
        "confidence_tier": fact.confidence_tier,
        "derivation_method": fact.derivation_method,
        "derivation_model": fact.derivation_model,
        "promoted_without_source_span": not server_has_source_span,
    }

    item = CanonicalMemoryItem(
        id=uuid.uuid4(),
        fact_id=fact.id,
        raw_archive_id=source_id,
        content=content,
        promoted_from="fact",
        promoted_by=promoted_by,
        status="active",
        source_status="active",
        provenance_note=json.dumps(provenance, sort_keys=True),
    )
    session.add(item)
    await session.flush()
    return item


async def create_manual_canonical(
    session: AsyncSession,
    content: str,
    promoted_by: str,
) -> CanonicalMemoryItem:
    """Create a canonical item manually, not promoted from a fact (CANM-01)."""
    item = CanonicalMemoryItem(
        id=uuid.uuid4(),
        fact_id=None,
        raw_archive_id=None,
        content=content,
        promoted_from="manual",
        promoted_by=promoted_by,
        status="active",
        source_status="active",
    )
    session.add(item)
    await session.flush()
    return item


async def get_canonical_item(
    session: AsyncSession,
    item_id: uuid.UUID,
) -> CanonicalMemoryItem | None:
    """Return a canonical item by id, or None if not found."""
    result = await session.execute(
        select(CanonicalMemoryItem).where(CanonicalMemoryItem.id == item_id)
    )
    return result.scalar_one_or_none()


async def update_canonical_item(
    session: AsyncSession,
    item_id: uuid.UUID,
    content: str | None = None,
    status: str | None = None,
) -> CanonicalMemoryItem:
    """Partially update a canonical item's content and/or status (CANM-01).

    Raises CanonicalItemNotFoundError if the item does not exist.
    """
    item = await get_canonical_item(session, item_id)
    if item is None:
        raise CanonicalItemNotFoundError(f"Canonical item {item_id} not found.")
    if content is not None:
        item.content = content
    if status is not None:
        item.status = status
    item.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return item


async def delete_canonical_item(
    session: AsyncSession,
    item_id: uuid.UUID,
) -> None:
    """Soft-delete a canonical item by setting source_status='source_removed' (CANM-01)."""
    item = await get_canonical_item(session, item_id)
    if item is None:
        raise CanonicalItemNotFoundError(f"Canonical item {item_id} not found.")
    item.source_status = "source_removed"
    item.updated_at = datetime.now(timezone.utc)
    await session.flush()


async def mark_canonical_disputed(
    session: AsyncSession,
    item_id: uuid.UUID,
) -> CanonicalMemoryItem:
    """Mark a canonical item as disputed (CANM-01)."""
    return await update_canonical_item(session, item_id, status="disputed")


async def mark_canonical_stale(
    session: AsyncSession,
    item_id: uuid.UUID,
) -> CanonicalMemoryItem:
    """Mark a canonical item as stale (CANM-01)."""
    return await update_canonical_item(session, item_id, status="stale")


async def list_canonical_items(
    session: AsyncSession,
) -> list[CanonicalMemoryItem]:
    """Return all active canonical items (CANM-02: highest retrieval priority)."""
    result = await session.execute(
        select(CanonicalMemoryItem).where(
            CanonicalMemoryItem.source_status == "active"
        )
    )
    return list(result.scalars().all())
