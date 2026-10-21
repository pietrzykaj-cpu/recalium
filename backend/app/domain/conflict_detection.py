"""Conflict and duplicate detection for derived memory facts.

CANM-06: Flags contradictory or duplicate facts across sources by comparing
embedding vectors using pgvector cosine distance.

SECURITY: All queries filter source_status = 'active' — suppressed data is excluded.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Sequence

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.derived_memory.models import ConflictGroup, Embedding, Fact
from app.domain.derivations.visibility import enabled, current_sql

logger = logging.getLogger(__name__)

DUPLICATE_DISTANCE_THRESHOLD = 0.15
_MAX_CANDIDATES = 10
# Slice 2A: identity of the managed detection contract (threshold, candidate cap,
# archive-wide grouping, link predicate). Changing any of them requires a bump, which
# deliberately permits exactly one new detection run per generation.
DETECTION_CONTRACT_VERSION = 1


async def find_duplicate_candidates(
    session: AsyncSession,
    embedding: list[float],
    exclude_id: uuid.UUID,
) -> list[uuid.UUID]:
    """Find embeddings that are near-duplicates of the given vector.

    Uses pgvector cosine distance (<=>). Only considers rows with
    source_status = 'active' (CASCADE CONTRACT).

    Args:
        session: Async DB session.
        embedding: The query vector (384-dim list of floats).
        exclude_id: UUID of the embedding row to exclude (the source itself).

    Returns:
        List of embedding UUIDs within DUPLICATE_DISTANCE_THRESHOLD.
    """
    result = await session.execute(
        text(f"""
            SELECT id
            FROM embeddings e
            WHERE id != :exclude_id
              AND source_status = 'active'
              {current_sql('e')}
              AND embedding <=> :vec < :threshold
            ORDER BY embedding <=> :vec
            LIMIT :max_candidates
        """),
        {
            # Rows read back through pgvector are numpy arrays, whose str() is
            # space-separated and NOT valid pgvector input — coerce to list first
            "vec": str(embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)),
            "exclude_id": str(exclude_id),
            "threshold": DUPLICATE_DISTANCE_THRESHOLD,
            "max_candidates": _MAX_CANDIDATES,
        },
    )
    rows = result.fetchall()
    logger.debug("Conflict scan: found %d candidates near embedding %s", len(rows), exclude_id)
    return [uuid.UUID(str(row.id)) for row in rows]


async def create_conflict_group(
    session: AsyncSession,
    group_type: str = "duplicate",
    fact_ids: Sequence[uuid.UUID] | None = None,  # accepted for API compat; stored when schema supports it
) -> ConflictGroup:
    """Create a ConflictGroup row and return it.

    CANM-06: group_type is one of "duplicate" | "contradiction" | "overlap".

    Args:
        session: Async DB session.
        group_type: Type of conflict ("duplicate" by default).
        fact_ids: Optional list of fact UUIDs involved (for future linking table).

    Returns:
        The persisted ConflictGroup instance.
    """
    group = ConflictGroup(group_type=group_type)
    session.add(group)
    await session.commit()
    await session.refresh(group)
    logger.info("Created conflict group id=%s type=%s", group.id, group_type)
    return group


@dataclass
class DuplicateDetection:
    """Result of a duplicate scan: the group plus what it linked."""

    group: ConflictGroup
    duplicate_archive_ids: list[uuid.UUID] = field(default_factory=list)
    linked_fact_count: int = 0


async def _resolve_embedding_archive_ids(
    session: AsyncSession, embedding_ids: Sequence[uuid.UUID]
) -> list[uuid.UUID]:
    """Map embedding row ids to their distinct source ``raw_archive_id`` values."""
    ids = list(embedding_ids)
    if not ids:
        return []
    result = await session.execute(
        select(Embedding.raw_archive_id).where(Embedding.id.in_(ids))
    )
    seen: list[uuid.UUID] = []
    for (archive_id,) in result.all():
        if archive_id is not None and archive_id not in seen:
            seen.append(archive_id)
    return seen


async def link_facts_to_group(
    session: AsyncSession,
    group_id: uuid.UUID,
    raw_archive_ids: Sequence[uuid.UUID],
) -> int:
    """Assign active, not-yet-grouped facts of the given items to a conflict group.

    Returns the number of facts newly linked. Idempotent: facts already in a
    group are left untouched.
    """
    ids = [rid for rid in raw_archive_ids if rid is not None]
    if not ids:
        return 0
    result = await session.execute(
        update(Fact)
        .where(Fact.raw_archive_id.in_(ids))
        .where(Fact.source_status == "active")
        .where(Fact.conflict_group_id.is_(None))
        .values(conflict_group_id=group_id)
    )
    await session.commit()
    return int(result.rowcount or 0)


async def detect_and_group_duplicates(
    session: AsyncSession,
    *,
    raw_archive_id: uuid.UUID,
    embedding_id: uuid.UUID,
    embedding: list[float],
    generation_id: uuid.UUID | None = None,
) -> DuplicateDetection | None:
    """Detect near-duplicate embeddings and, if any, create + populate a group.

    CANM-06 / GPT5.6 #10: the prior wiring created an empty ``ConflictGroup``
    that was never linked to any fact, so duplicates never surfaced in the
    review queue. This creates the group *and* links the involved items' active
    facts to it so the conflict is reviewable.

    Returns a ``DuplicateDetection`` (group + involved items + linked fact count)
    or ``None`` when no other item is a near-duplicate.

    Slice 2A: under R2 the managed caller must pass ``generation_id`` (the trigger
    generation); detection then runs at most once per (generation, contract version).
    """
    if enabled():
        if generation_id is None:
            raise ValueError("Managed conflict detection requires the trigger generation_id")
        return await _detect_current_generations(
            session, raw_archive_id, embedding_id, embedding, generation_id)
    candidates = await find_duplicate_candidates(
        session, embedding=embedding, exclude_id=embedding_id
    )
    if not candidates:
        return None

    dup_archive_ids = [
        aid
        for aid in await _resolve_embedding_archive_ids(session, candidates)
        if aid != raw_archive_id
    ]
    if not dup_archive_ids:
        return None

    group = await create_conflict_group(session, group_type="duplicate")
    involved = [raw_archive_id, *dup_archive_ids]
    linked = await link_facts_to_group(session, group.id, involved)
    # GPT5.6 #10: materialize a review item so the conflict actually surfaces in the
    # review queue (previously the group was created and linked but never queued, so
    # the user could never act on it).
    from app.domain.review_queue.service import materialize_review_item  # noqa: PLC0415
    await materialize_review_item(session, group.id, item_type="duplicate")
    await session.commit()
    return DuplicateDetection(
        group=group,
        duplicate_archive_ids=dup_archive_ids,
        linked_fact_count=linked,
    )

async def _recorded_run(session, generation_id, *, fresh=False):
    """Slice 2A: the recorded detection run for this generation and contract version, if any."""
    from app.domain.derived_memory.models import ConflictDetectionRun
    statement = select(ConflictDetectionRun).where(
        ConflictDetectionRun.generation_id == generation_id,
        ConflictDetectionRun.contract_version == DETECTION_CONTRACT_VERSION)
    if fresh:
        statement = statement.execution_options(populate_existing=True)
    return await session.scalar(statement)


async def _detect_current_generations(session, archive_id, embedding_id, vector, generation_id):
    """Visibility/fencing plus Slice 2A replay identity (one result per generation and contract).

    Not pair identity (Slice 2B): a later archive near an already-decided pair can still
    regroup their unlinked facts.
    """
    from app.domain.derivations import service as g
    from app.domain.derived_memory.models import ConflictDetectionRun, ConflictGroupMember
    from app.domain.review_queue.service import materialize_review_item
    async with g.transaction(session):
        # Replay identity first: a recorded run means this generation was already
        # detected under this contract. Plain read before any lock or write (the lock
        # helper below inserts heads), so a replay writes nothing at all.
        if await _recorded_run(session, generation_id) is not None:
            return None
        candidates = await find_duplicate_candidates(session, vector, embedding_id)
        other_ids = await _resolve_embedding_archive_ids(session, candidates)
        sources,_ = await g.lock_archives_and_heads(session, [archive_id, *other_ids])
        if not all([await g.source_live(session,s) for s in sources.values()]):
            return None
        own = await session.scalar(select(Embedding).where(Embedding.id == embedding_id,
            Embedding.raw_archive_id == archive_id, Embedding.source_status == "active",
            Embedding.generation_id == generation_id))
        if own is None:
            return None
        # Locked re-check: a concurrent run for this generation committed meanwhile.
        if await _recorded_run(session, generation_id, fresh=True) is not None:
            return None
        def record(outcome, group_id=None, audit_id=None):
            session.add(ConflictDetectionRun(generation_id=generation_id, raw_archive_id=archive_id,
                contract_version=DETECTION_CONTRACT_VERSION, outcome=outcome,
                conflict_group_id=group_id, audit_event_id=audit_id))
        fresh = (await session.scalars(select(Embedding).where(
            Embedding.id.in_(candidates), Embedding.source_status == "active"))).all()
        other_ids = sorted({e.raw_archive_id for e in fresh if e.raw_archive_id != archive_id}, key=str)
        if not other_ids:
            record("no_candidates")
            return None
        to_link = (await session.scalars(select(Fact).where(
            Fact.raw_archive_id.in_([archive_id,*other_ids]), Fact.source_status=="active",
            Fact.review_status=="active", Fact.conflict_group_id.is_(None)).order_by(Fact.id))).all()
        if not to_link:
            record("nothing_to_link")
            return None
        group = ConflictGroup(group_type="duplicate")
        session.add(group)
        await session.flush()
        session.add_all([ConflictGroupMember(conflict_group_id=group.id, fact_id=f.id,
            raw_archive_id=f.raw_archive_id, generation_id=f.generation_id) for f in to_link])
        await session.flush()
        result = await session.execute(update(Fact).where(Fact.id.in_([f.id for f in to_link]))
            .values(conflict_group_id=group.id))
        await materialize_review_item(session, group.id, "duplicate")
        from app.domain.audit.models import AuditEvent
        audit = AuditEvent(event_type="conflict_detected", raw_archive_id=archive_id,
            actor="pipeline_worker", operation_metadata={"conflict_group_id":str(group.id),
            "group_type":"duplicate","duplicate_archive_ids":[str(a) for a in other_ids],
            "linked_fact_count":result.rowcount})
        session.add(audit)
        await session.flush()
        record("grouped", group.id, audit.id)
        detection = DuplicateDetection(group, other_ids, int(result.rowcount or 0))
    return detection
