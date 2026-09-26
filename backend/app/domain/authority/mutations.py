"""Guarded internal authority writes; no HTTP or MCP surface is defined here."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.audit.models import AuditEvent
from app.domain.bridge.models import (
    AuthorityMutationReceipt,
    BridgeClient,
    BridgeGrant,
    BridgeProject,
)

from .contracts import AuthorityEdge, AuthorityRecord, AuthorityStateResult
from .mutation_contracts import (
    ActivateAuthorityRecord,
    AuthorityActorContext,
    AuthorityScope,
    AuthorityStateSummary,
    AuthorityWriteResult,
    CreateAuthorityProposal,
    GuardedAuthorityMutation,
    MarkAuthorityDisputed,
    SupersedeAuthorityRecord,
    WithdrawAuthorityRecord,
)
from .repository import (
    activate_record,
    create_record,
    list_scope_edges,
    list_scope_records,
    mark_disputed,
    supersede,
    withdraw,
)
from .service import AuthorityGraph, evaluate_current_state


class AuthorityMutationError(ValueError):
    """Stable service-layer rejection for future transport adapters."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_authority_state(
    records: list[AuthorityRecord] | tuple[AuthorityRecord, ...],
    edges: list[AuthorityEdge] | tuple[AuthorityEdge, ...],
    *,
    scope: AuthorityScope,
) -> dict[str, Any]:
    """Return the platform-neutral material covered by stale-write guards.

    Timestamps are deliberately excluded. Content is represented by its UTF-8
    digest. Creator, provenance and transition reasons are included because
    they materially affect a client's interpretation of authority.
    """
    scope_key = (scope.space_id, scope.workstream_id, scope.authority_key)
    scoped_records = sorted(
        (
            record
            for record in records
            if (record.space_id, record.workstream_id, record.authority_key) == scope_key
        ),
        key=lambda record: record.id,
    )
    record_ids = {record.id for record in scoped_records}
    scoped_edges = sorted(
        (
            edge
            for edge in edges
            if edge.successor_record_id in record_ids
            and edge.predecessor_record_id in record_ids
        ),
        key=lambda edge: (
            edge.successor_record_id,
            edge.predecessor_record_id,
            edge.edge_type,
        ),
    )
    return {
        "scope": scope.model_dump(mode="json"),
        "records": [
            {
                "id": record.id,
                "record_kind": record.record_kind,
                "content_sha256": _sha256(record.content),
                "lifecycle_status": record.lifecycle_status,
                "created_by": record.created_by,
                "provenance": record.provenance,
                "withdrawal_reason": record.withdrawal_reason,
                "dispute_reason": record.dispute_reason,
            }
            for record in scoped_records
        ],
        "edges": [
            {
                "successor_record_id": edge.successor_record_id,
                "predecessor_record_id": edge.predecessor_record_id,
                "edge_type": edge.edge_type,
                "created_by": edge.created_by,
                "reason": edge.reason,
                "provenance": edge.provenance,
            }
            for edge in scoped_edges
        ],
    }


def authority_state_digest(
    records: list[AuthorityRecord] | tuple[AuthorityRecord, ...],
    edges: list[AuthorityEdge] | tuple[AuthorityEdge, ...],
    *,
    scope: AuthorityScope,
) -> str:
    return _sha256(_canonical_json(canonical_authority_state(records, edges, scope=scope)))


def _summary(
    state: AuthorityStateResult,
    records: list[AuthorityRecord] | tuple[AuthorityRecord, ...],
    edges: list[AuthorityEdge] | tuple[AuthorityEdge, ...],
    scope: AuthorityScope,
) -> AuthorityStateSummary:
    return AuthorityStateSummary(
        digest=authority_state_digest(records, edges, scope=scope),
        status=state.status,
        current_record_id=state.current_record.id if state.current_record else None,
        eligible_record_ids=[record.id for record in state.eligible_records],
        competing_record_ids=[record.id for record in state.competing_records],
        historical_record_ids=[record.id for record in state.historical_records],
        superseded_record_ids=state.superseded_record_ids,
    )


async def _load_scope(
    session: AsyncSession, scope: AuthorityScope
) -> tuple[list[AuthorityRecord], list[AuthorityEdge], AuthorityStateResult]:
    records = await list_scope_records(
        session,
        space_id=scope.space_id,
        workstream_id=scope.workstream_id,
        authority_key=scope.authority_key,
    )
    edges = await list_scope_edges(session, record_ids=[record.id for record in records])
    state = evaluate_current_state(
        records,
        edges,
        space_id=scope.space_id,
        workstream_id=scope.workstream_id,
        authority_key=scope.authority_key,
    )
    return records, edges, state


def _advisory_lock_value(parts: tuple[str, ...]) -> int:
    encoded = _canonical_json(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big", signed=True)


async def _lock(session: AsyncSession, *parts: str) -> None:
    await session.execute(
        text("SELECT pg_advisory_xact_lock(:lock)"),
        {"lock": _advisory_lock_value(tuple(parts))},
    )


async def _authorize(
    session: AsyncSession,
    actor: AuthorityActorContext,
    scope: AuthorityScope,
    *,
    capability: str,
) -> None:
    principal = (
        await session.execute(
            select(BridgeClient)
            .where(
                BridgeClient.id == actor.authorizing_client_id,
                BridgeClient.active.is_(True),
            )
            .with_for_update(read=True)
        )
    ).scalar_one_or_none()
    if principal is None:
        raise AuthorityMutationError("authentication_required")
    permission = getattr(BridgeGrant, capability)
    grant = (
        await session.execute(
            select(BridgeGrant)
            .join(BridgeProject, BridgeProject.id == BridgeGrant.project_id)
            .where(
                BridgeGrant.client_id == actor.authorizing_client_id,
                BridgeGrant.project_id == scope.space_id,
                permission.is_(True),
                or_(
                    BridgeProject.kind == "shared",
                    BridgeProject.owner_client_id == actor.authorizing_client_id,
                ),
            )
            .with_for_update(read=True)
        )
    ).scalar_one_or_none()
    if grant is None:
        raise AuthorityMutationError("permission_denied")


def _payload_digest(operation: str, actor: AuthorityActorContext, request: Any) -> str:
    payload = {
        "operation": operation,
        "actor": actor.model_dump(mode="json"),
        "request": request.model_dump(mode="json"),
    }
    return _sha256(_canonical_json(payload))


async def _receipt_or_none(
    session: AsyncSession,
    *,
    actor: AuthorityActorContext,
    request_digest: str,
    payload_digest: str,
) -> AuthorityWriteResult | None:
    receipt = await session.get(
        AuthorityMutationReceipt,
        (actor.requesting_client_id, request_digest),
    )
    if receipt is None:
        return None
    if receipt.payload_digest != payload_digest:
        raise AuthorityMutationError("idempotency_conflict")
    original = AuthorityWriteResult.model_validate(receipt.result_json)
    return original.model_copy(
        update={"outcome": "idempotent_replay", "idempotent_replay": True}
    )


def _preview_operation_id(actor: AuthorityActorContext, payload_digest: str) -> str:
    material = f"recalium-authority-preview:{actor.requesting_client_id}:{payload_digest}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, material))


async def _persist_result(
    session: AsyncSession,
    *,
    actor: AuthorityActorContext,
    operation: str,
    request: Any,
    request_digest: str,
    payload_digest: str,
    result: AuthorityWriteResult,
) -> None:
    if result.audit_event_id is None:
        raise RuntimeError("Durable authority result requires an audit event id")
    audit_id = uuid.UUID(result.audit_event_id)
    session.add(
        AuditEvent(
            id=audit_id,
            event_type="authority_mutation",
            actor=actor.requesting_client_id,
            operation_metadata={
                "operation_id": result.operation_id,
                "operation": operation,
                "requesting_client_id": actor.requesting_client_id,
                "authorizing_client_id": actor.authorizing_client_id,
                "scope": result.scope.model_dump(mode="json"),
                "request_digest": request_digest,
                "payload_digest": payload_digest,
                "expected_state_digest": getattr(request, "expected_state_digest", None),
                "before": result.before.model_dump(mode="json"),
                "after": result.after.model_dump(mode="json"),
                "affected_record_ids": result.affected_record_ids,
                "affected_edge_ids": result.affected_edge_ids,
                "reason": getattr(request, "reason", None),
                "provenance": request.provenance,
                "result": result.outcome,
            },
        )
    )
    session.add(
        AuthorityMutationReceipt(
            client_id=actor.requesting_client_id,
            request_digest=request_digest,
            payload_digest=payload_digest,
            operation_id=uuid.UUID(result.operation_id),
            operation_type=operation,
            result_json=result.model_dump(mode="json"),
            audit_event_id=audit_id,
            created_at=datetime.now(UTC),
        )
    )
    await session.flush()


async def create_authority_proposal(
    session: AsyncSession,
    actor: AuthorityActorContext,
    request: CreateAuthorityProposal,
) -> AuthorityWriteResult:
    operation = "create_authority_proposal"
    await _authorize(
        session,
        actor,
        request.scope,
        capability="can_propose_authority",
    )
    request_digest = _sha256(request.idempotency_key)
    payload_digest = _payload_digest(operation, actor, request)
    await _lock(session, "authority-receipt", actor.requesting_client_id, request_digest)
    replay = await _receipt_or_none(
        session,
        actor=actor,
        request_digest=request_digest,
        payload_digest=payload_digest,
    )
    if replay is not None:
        return replay
    await _lock(
        session,
        "authority-scope",
        request.scope.space_id,
        request.scope.workstream_id,
        request.scope.authority_key,
    )
    records, edges, state = await _load_scope(session, request.scope)
    before = _summary(state, records, edges, request.scope)
    operation_id = str(uuid.uuid4())
    audit_id = str(uuid.uuid4())
    async with session.begin_nested():
        await create_record(
            session,
            AuthorityRecord(
                id=request.record_id,
                space_id=request.scope.space_id,
                workstream_id=request.scope.workstream_id,
                authority_key=request.scope.authority_key,
                record_kind=request.record_kind,
                content=request.content,
                lifecycle_status="proposed",
                created_at=datetime.now(UTC),
                created_by=actor.requesting_client_id,
                provenance=request.provenance,
            ),
        )
        after_records, after_edges, after_state = await _load_scope(session, request.scope)
        result = AuthorityWriteResult(
            operation_id=operation_id,
            operation=operation,
            outcome="applied",
            scope=request.scope,
            affected_record_ids=[request.record_id],
            before=before,
            after=_summary(after_state, after_records, after_edges, request.scope),
            audit_event_id=audit_id,
        )
        await _persist_result(
            session,
            actor=actor,
            operation=operation,
            request=request,
            request_digest=request_digest,
            payload_digest=payload_digest,
            result=result,
        )
    return result


def _apply_to_graph(
    graph: AuthorityGraph,
    operation: str,
    request: GuardedAuthorityMutation,
    *,
    actor: AuthorityActorContext,
) -> tuple[list[str], list[str]]:
    affected_records: list[str]
    affected_edges: list[str] = []
    if operation == "activate_authority_record":
        assert isinstance(request, ActivateAuthorityRecord)
        graph.activate_record(request.record_id)
        affected_records = [request.record_id]
    elif operation == "supersede_authority_record":
        assert isinstance(request, SupersedeAuthorityRecord)
        graph.supersede(
            request.successor_record_id,
            request.predecessor_record_id,
            created_at=datetime.now(UTC),
            created_by=actor.requesting_client_id,
            reason=request.reason,
            provenance=request.provenance,
        )
        affected_records = [request.successor_record_id, request.predecessor_record_id]
        affected_edges = [
            f"{request.successor_record_id}>{request.predecessor_record_id}:supersedes"
        ]
    elif operation == "withdraw_authority_record":
        assert isinstance(request, WithdrawAuthorityRecord)
        graph.withdraw(request.record_id, reason=request.reason)
        affected_records = [request.record_id]
    elif operation == "mark_authority_disputed":
        assert isinstance(request, MarkAuthorityDisputed)
        graph.mark_disputed(request.record_id, reason=request.reason)
        affected_records = [request.record_id]
    else:
        raise AuthorityMutationError("unknown_authority_operation")
    return affected_records, affected_edges


async def _apply_to_repository(
    session: AsyncSession,
    operation: str,
    request: GuardedAuthorityMutation,
    *,
    actor: AuthorityActorContext,
) -> None:
    if operation == "activate_authority_record":
        assert isinstance(request, ActivateAuthorityRecord)
        await activate_record(session, request.record_id)
    elif operation == "supersede_authority_record":
        assert isinstance(request, SupersedeAuthorityRecord)
        await supersede(
            session,
            request.successor_record_id,
            request.predecessor_record_id,
            created_at=datetime.now(UTC),
            created_by=actor.requesting_client_id,
            reason=request.reason,
            provenance=request.provenance,
        )
    elif operation == "withdraw_authority_record":
        assert isinstance(request, WithdrawAuthorityRecord)
        await withdraw(session, request.record_id, reason=request.reason)
    elif operation == "mark_authority_disputed":
        assert isinstance(request, MarkAuthorityDisputed)
        await mark_disputed(session, request.record_id, reason=request.reason)
    else:
        raise AuthorityMutationError("unknown_authority_operation")


async def _guarded_mutation(
    session: AsyncSession,
    actor: AuthorityActorContext,
    request: GuardedAuthorityMutation,
    *,
    operation: str,
) -> AuthorityWriteResult:
    await _authorize(
        session,
        actor,
        request.scope,
        capability="can_mutate_authority",
    )
    if not request.confirm and not request.dry_run:
        raise AuthorityMutationError("confirmation_required")
    request_digest = _sha256(request.idempotency_key)
    payload_digest = _payload_digest(operation, actor, request)
    if not request.dry_run:
        await _lock(session, "authority-receipt", actor.requesting_client_id, request_digest)
        replay = await _receipt_or_none(
            session,
            actor=actor,
            request_digest=request_digest,
            payload_digest=payload_digest,
        )
        if replay is not None:
            return replay
    await _lock(
        session,
        "authority-scope",
        request.scope.space_id,
        request.scope.workstream_id,
        request.scope.authority_key,
    )
    records, edges, state = await _load_scope(session, request.scope)
    before = _summary(state, records, edges, request.scope)
    if before.digest != request.expected_state_digest:
        raise AuthorityMutationError("stale_authority_state")
    graph = AuthorityGraph.from_state(records, edges)
    affected_records, affected_edges = _apply_to_graph(
        graph,
        operation,
        request,
        actor=actor,
    )
    preview_state = graph.evaluate(
        space_id=request.scope.space_id,
        workstream_id=request.scope.workstream_id,
        authority_key=request.scope.authority_key,
    )
    preview_after = _summary(
        preview_state,
        list(graph.records),
        list(graph.edges),
        request.scope,
    )
    warnings = ["authority_state_ambiguous"] if preview_after.status == "ambiguous" else []
    if request.dry_run:
        return AuthorityWriteResult(
            operation_id=_preview_operation_id(actor, payload_digest),
            operation=operation,
            outcome="preview",
            scope=request.scope,
            affected_record_ids=affected_records,
            affected_edge_ids=affected_edges,
            before=before,
            after=preview_after,
            warnings=warnings,
        )
    operation_id = str(uuid.uuid4())
    audit_id = str(uuid.uuid4())
    async with session.begin_nested():
        await _apply_to_repository(session, operation, request, actor=actor)
        after_records, after_edges, after_state = await _load_scope(session, request.scope)
        after = _summary(after_state, after_records, after_edges, request.scope)
        if after.digest != preview_after.digest:
            raise AuthorityMutationError("postcondition_failed")
        result = AuthorityWriteResult(
            operation_id=operation_id,
            operation=operation,
            outcome="applied",
            scope=request.scope,
            affected_record_ids=affected_records,
            affected_edge_ids=affected_edges,
            before=before,
            after=after,
            warnings=warnings,
            audit_event_id=audit_id,
        )
        await _persist_result(
            session,
            actor=actor,
            operation=operation,
            request=request,
            request_digest=request_digest,
            payload_digest=payload_digest,
            result=result,
        )
    return result


async def activate_authority_record(
    session: AsyncSession,
    actor: AuthorityActorContext,
    request: ActivateAuthorityRecord,
) -> AuthorityWriteResult:
    return await _guarded_mutation(
        session, actor, request, operation="activate_authority_record"
    )


async def supersede_authority_record(
    session: AsyncSession,
    actor: AuthorityActorContext,
    request: SupersedeAuthorityRecord,
) -> AuthorityWriteResult:
    return await _guarded_mutation(
        session, actor, request, operation="supersede_authority_record"
    )


async def withdraw_authority_record(
    session: AsyncSession,
    actor: AuthorityActorContext,
    request: WithdrawAuthorityRecord,
) -> AuthorityWriteResult:
    return await _guarded_mutation(
        session, actor, request, operation="withdraw_authority_record"
    )


async def mark_authority_disputed(
    session: AsyncSession,
    actor: AuthorityActorContext,
    request: MarkAuthorityDisputed,
) -> AuthorityWriteResult:
    return await _guarded_mutation(
        session, actor, request, operation="mark_authority_disputed"
    )
