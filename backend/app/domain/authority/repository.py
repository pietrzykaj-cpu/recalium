"""Async persistence adapter for the pure authority domain."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .contracts import AuthorityEdge, AuthorityRecord, AuthorityStateResult
from .models import AuthorityEdgeRow, AuthorityRecordRow
from .service import AuthorityGraph, AuthorityValidationError, evaluate_current_state


def _to_domain(row: AuthorityRecordRow) -> AuthorityRecord:
    return AuthorityRecord(
        id=str(row.id),
        space_id=row.space_id,
        workstream_id=row.workstream_id,
        authority_key=row.authority_key,
        record_kind=row.record_kind,
        content=row.content,
        lifecycle_status=row.lifecycle_status,
        created_at=row.created_at,
        created_by=row.created_by,
        provenance=row.provenance or {},
        withdrawal_reason=row.withdrawal_reason,
        dispute_reason=row.dispute_reason,
    )


def _edge_to_domain(row: AuthorityEdgeRow) -> AuthorityEdge:
    return AuthorityEdge(
        successor_record_id=str(row.successor_record_id),
        predecessor_record_id=str(row.predecessor_record_id),
        edge_type=row.edge_type,
        created_at=row.created_at,
        created_by=row.created_by,
        reason=row.reason,
        provenance=row.provenance or {},
    )


async def create_record(session: AsyncSession, record: AuthorityRecord) -> AuthorityRecord:
    session.add(
        AuthorityRecordRow(
            id=uuid.UUID(record.id),
            space_id=record.space_id,
            workstream_id=record.workstream_id,
            authority_key=record.authority_key,
            record_kind=record.record_kind,
            content=record.content,
            lifecycle_status=record.lifecycle_status,
            created_at=record.created_at,
            created_by=record.created_by,
            provenance=record.provenance,
            withdrawal_reason=record.withdrawal_reason,
            dispute_reason=record.dispute_reason,
        )
    )
    await session.flush()
    return record


async def load_record(session: AsyncSession, record_id: str) -> AuthorityRecord | None:
    row = await session.get(AuthorityRecordRow, uuid.UUID(record_id))
    return _to_domain(row) if row else None


async def list_scope_records(
    session: AsyncSession, *, space_id: str, workstream_id: str, authority_key: str
) -> list[AuthorityRecord]:
    result = await session.execute(
        select(AuthorityRecordRow)
        .where(
            AuthorityRecordRow.space_id == space_id,
            AuthorityRecordRow.workstream_id == workstream_id,
            AuthorityRecordRow.authority_key == authority_key,
        )
        .order_by(AuthorityRecordRow.id)
    )
    return [_to_domain(row) for row in result.scalars().all()]


async def list_scope_edges(
    session: AsyncSession, *, record_ids: list[str]
) -> list[AuthorityEdge]:
    if not record_ids:
        return []
    ids = [uuid.UUID(record_id) for record_id in record_ids]
    result = await session.execute(
        select(AuthorityEdgeRow).where(
            AuthorityEdgeRow.successor_record_id.in_(ids),
            AuthorityEdgeRow.predecessor_record_id.in_(ids),
        )
    )
    return [_edge_to_domain(row) for row in result.scalars().all()]


async def evaluate_scope(
    session: AsyncSession, *, space_id: str, workstream_id: str, authority_key: str
) -> AuthorityStateResult:
    records = await list_scope_records(
        session, space_id=space_id, workstream_id=workstream_id, authority_key=authority_key
    )
    edges = await list_scope_edges(session, record_ids=[record.id for record in records])
    return evaluate_current_state(
        records,
        edges,
        space_id=space_id,
        workstream_id=workstream_id,
        authority_key=authority_key,
    )


async def activate_record(session: AsyncSession, record_id: str) -> AuthorityRecord:
    record = await load_record(session, record_id)
    if record is None:
        raise AuthorityValidationError(f"Unknown authority record: {record_id}")
    graph = AuthorityGraph()
    graph.create_record(record)
    updated = graph.activate_record(record_id)
    row = await session.get(AuthorityRecordRow, uuid.UUID(record_id))
    row.lifecycle_status = updated.lifecycle_status
    await session.flush()
    return updated


async def supersede(
    session: AsyncSession,
    successor_id: str,
    predecessor_id: str,
    *,
    created_at: datetime,
    created_by: str,
    reason: str | None = None,
    provenance: dict | None = None,
) -> AuthorityEdge:
    """Atomically validate, activate the successor and insert its edge.

    The caller commits the surrounding SQL transaction; the flush ensures both
    writes fail together before commit.
    """
    async with session.begin_nested():
        successor = await load_record(session, successor_id)
        predecessor = await load_record(session, predecessor_id)
        if successor is None or predecessor is None:
            raise AuthorityValidationError("Both supersession records must exist")
        records = await list_scope_records(
            session,
            space_id=successor.space_id,
            workstream_id=successor.workstream_id,
            authority_key=successor.authority_key,
        )
        edges = await list_scope_edges(session, record_ids=[record.id for record in records])
        graph = AuthorityGraph()
        for record in records:
            graph.create_record(record)
        for edge in edges:
            graph._edges.append(edge)
        edge = graph.supersede(
            successor_id,
            predecessor_id,
            created_at=created_at,
            created_by=created_by,
            reason=reason,
            provenance=provenance,
        )
        successor_row = await session.get(AuthorityRecordRow, uuid.UUID(successor_id))
        successor_row.lifecycle_status = "active"
        session.add(
            AuthorityEdgeRow(
                successor_record_id=uuid.UUID(successor_id),
                predecessor_record_id=uuid.UUID(predecessor_id),
                edge_type=edge.edge_type,
                created_at=edge.created_at,
                created_by=edge.created_by,
                reason=edge.reason,
                provenance=edge.provenance,
            )
        )
        await session.flush()
        return edge


async def withdraw(
    session: AsyncSession, record_id: str, *, reason: str | None = None
) -> AuthorityRecord:
    record = await load_record(session, record_id)
    if record is None:
        raise AuthorityValidationError(f"Unknown authority record: {record_id}")
    graph = AuthorityGraph()
    graph.create_record(record)
    updated = graph.withdraw(record_id, reason=reason)
    row = await session.get(AuthorityRecordRow, uuid.UUID(record_id))
    row.lifecycle_status = updated.lifecycle_status
    row.withdrawal_reason = updated.withdrawal_reason
    await session.flush()
    return updated


async def mark_disputed(
    session: AsyncSession, record_id: str, *, reason: str | None = None
) -> AuthorityRecord:
    record = await load_record(session, record_id)
    if record is None:
        raise AuthorityValidationError(f"Unknown authority record: {record_id}")
    graph = AuthorityGraph()
    graph.create_record(record)
    updated = graph.mark_disputed(record_id, reason=reason)
    row = await session.get(AuthorityRecordRow, uuid.UUID(record_id))
    row.lifecycle_status = updated.lifecycle_status
    row.dispute_reason = updated.dispute_reason
    await session.flush()
    return updated
