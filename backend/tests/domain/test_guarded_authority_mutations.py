"""Authority Phase 1E-A guarded internal mutation tests."""

import hashlib
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select

from app.domain.audit.models import AuditEvent
from app.domain.authority.contracts import AuthorityRecord
from app.domain.authority.models import AuthorityEdgeRow, AuthorityRecordRow
from app.domain.authority.mutation_contracts import (
    ActivateAuthorityRecord,
    AuthorityActorContext,
    AuthorityScope,
    CreateAuthorityProposal,
    MarkAuthorityDisputed,
    SupersedeAuthorityRecord,
    WithdrawAuthorityRecord,
)
from app.domain.authority.mutations import (
    AuthorityMutationError,
    activate_authority_record,
    authority_state_digest,
    create_authority_proposal,
    mark_authority_disputed,
    supersede_authority_record,
    withdraw_authority_record,
)
from app.domain.authority.repository import (
    create_record,
    evaluate_scope,
    list_scope_edges,
    list_scope_records,
)
from app.domain.authority.service import AuthorityValidationError
from app.domain.bridge.models import (
    AuthorityMutationReceipt,
    BridgeClient,
    BridgeGrant,
    BridgeProject,
)

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
SCOPE = AuthorityScope(space_id="shared", workstream_id="bootstrap", authority_key="storage")


async def grant(
    session,
    *,
    client_id="writer",
    can_write=False,
    can_propose=False,
    can_mutate=False,
    space="shared",
):
    session.add(BridgeClient(id=client_id, credential_digest=(client_id + "0" * 64)[:64]))
    session.add(BridgeProject(id=space, kind="shared"))
    await session.flush()
    session.add(
        BridgeGrant(
            client_id=client_id,
            project_id=space,
            can_read=True,
            can_write=can_write,
            can_propose_authority=can_propose,
            can_mutate_authority=can_mutate,
        )
    )
    await session.commit()
    return AuthorityActorContext(
        requesting_client_id=client_id,
        authorizing_client_id=client_id,
    )


def proposal(*, key="proposal-1", content="Use SQLite", record_id=None, provenance=None):
    return CreateAuthorityProposal(
        scope=SCOPE,
        record_id=record_id or str(uuid4()),
        record_kind="decision",
        content=content,
        provenance=provenance or {"source": "synthetic"},
        idempotency_key=key,
    )


def record(*, status="proposed", scope=SCOPE, record_id=None):
    return AuthorityRecord(
        id=record_id or str(uuid4()),
        space_id=scope.space_id,
        workstream_id=scope.workstream_id,
        authority_key=scope.authority_key,
        record_kind="decision",
        content="synthetic decision",
        lifecycle_status=status,
        created_at=NOW,
        created_by="synthetic",
        provenance={"source": "synthetic"},
    )


async def seed(session, *records):
    for item in records:
        await create_record(session, item)
    await session.commit()


async def current_digest(session, scope=SCOPE):
    records = await list_scope_records(
        session,
        space_id=scope.space_id,
        workstream_id=scope.workstream_id,
        authority_key=scope.authority_key,
    )
    edges = await list_scope_edges(session, record_ids=[item.id for item in records])
    return authority_state_digest(records, edges, scope=scope)


async def counts(session):
    receipts = await session.scalar(select(func.count()).select_from(AuthorityMutationReceipt))
    audits = await session.scalar(
        select(func.count()).select_from(AuditEvent).where(
            AuditEvent.event_type == "authority_mutation"
        )
    )
    return receipts, audits


@pytest.mark.asyncio
async def test_proposal_only_grant_creates_non_authoritative_idempotent_proposal(db_session):
    actor = await grant(db_session, can_propose=True)
    request = proposal()
    result = await create_authority_proposal(db_session, actor, request)
    await db_session.commit()
    assert result.outcome == "applied"
    assert (await evaluate_scope(
        db_session,
        space_id=SCOPE.space_id,
        workstream_id=SCOPE.workstream_id,
        authority_key=SCOPE.authority_key,
    )).status == "empty"
    saved = await db_session.get(AuthorityRecordRow, UUID(request.record_id))
    assert saved.lifecycle_status == "proposed"
    replay = await create_authority_proposal(db_session, actor, request)
    assert replay.outcome == "idempotent_replay"
    assert replay.operation_id == result.operation_id
    with pytest.raises(AuthorityMutationError, match="idempotency_conflict"):
        await create_authority_proposal(
            db_session,
            actor,
            request.model_copy(update={"content": "Different decision"}),
        )
    assert await counts(db_session) == (1, 1)


@pytest.mark.asyncio
async def test_proposal_permission_cannot_mutate_and_can_write_is_not_authority(db_session):
    actor = await grant(db_session, can_write=True, can_propose=True)
    item = record()
    await seed(db_session, item)
    request = ActivateAuthorityRecord(
        scope=SCOPE,
        record_id=item.id,
        expected_state_digest=await current_digest(db_session),
        idempotency_key="activate-denied",
        confirm=True,
    )
    with pytest.raises(AuthorityMutationError, match="authority_mutation_forbidden"):
        await activate_authority_record(db_session, actor, request)
    await db_session.rollback()
    assert (await db_session.get(AuthorityRecordRow, UUID(item.id))).lifecycle_status == "proposed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "request_type"),
    [
        (activate_authority_record, ActivateAuthorityRecord),
        (withdraw_authority_record, WithdrawAuthorityRecord),
        (mark_authority_disputed, MarkAuthorityDisputed),
    ],
)
async def test_proposal_only_grant_cannot_run_any_single_record_mutation(
    db_session, operation, request_type
):
    actor = await grant(db_session, can_propose=True)
    item = record(status="active")
    await seed(db_session, item)
    request = request_type(
        scope=SCOPE,
        record_id=item.id,
        expected_state_digest=await current_digest(db_session),
        idempotency_key=f"denied-{request_type.__name__}",
        confirm=True,
    )
    with pytest.raises(AuthorityMutationError, match="authority_mutation_forbidden"):
        await operation(db_session, actor, request)


@pytest.mark.asyncio
async def test_proposal_only_grant_cannot_supersede(db_session):
    actor = await grant(db_session, can_propose=True)
    predecessor, successor = record(status="active"), record()
    await seed(db_session, predecessor, successor)
    request = SupersedeAuthorityRecord(
        scope=SCOPE,
        successor_record_id=successor.id,
        predecessor_record_id=predecessor.id,
        expected_state_digest=await current_digest(db_session),
        idempotency_key="denied-supersede",
        confirm=True,
    )
    with pytest.raises(AuthorityMutationError, match="authority_mutation_forbidden"):
        await supersede_authority_record(db_session, actor, request)


@pytest.mark.asyncio
async def test_confirmation_dry_run_and_stale_guard(db_session):
    actor = await grant(db_session, can_mutate=True)
    item = record()
    await seed(db_session, item)
    digest = await current_digest(db_session)
    unconfirmed = ActivateAuthorityRecord(
        scope=SCOPE,
        record_id=item.id,
        expected_state_digest=digest,
        idempotency_key="unconfirmed",
    )
    with pytest.raises(AuthorityMutationError, match="confirmation_required"):
        await activate_authority_record(db_session, actor, unconfirmed)
    preview = await activate_authority_record(
        db_session,
        actor,
        unconfirmed.model_copy(update={"dry_run": True}),
    )
    assert preview.outcome == "preview"
    assert preview.after.status == "current"
    repeated_preview = await activate_authority_record(
        db_session,
        actor,
        unconfirmed.model_copy(update={"dry_run": True}),
    )
    assert repeated_preview == preview
    assert await counts(db_session) == (0, 0)
    assert (await db_session.get(AuthorityRecordRow, UUID(item.id))).lifecycle_status == "proposed"
    stale = unconfirmed.model_copy(
        update={"confirm": True, "expected_state_digest": "0" * 64, "idempotency_key": "stale"}
    )
    with pytest.raises(AuthorityMutationError, match="stale_authority_state"):
        await activate_authority_record(db_session, actor, stale)
    await db_session.rollback()
    assert (await db_session.get(AuthorityRecordRow, UUID(item.id))).lifecycle_status == "proposed"


@pytest.mark.asyncio
async def test_valid_mutation_commits_with_audit_receipt_and_replays(db_session):
    actor = await grant(db_session, can_mutate=True)
    item = record()
    await seed(db_session, item)
    request = ActivateAuthorityRecord(
        scope=SCOPE,
        record_id=item.id,
        expected_state_digest=await current_digest(db_session),
        idempotency_key="activate-once",
        confirm=True,
        provenance={"model_label": "synthetic-only"},
    )
    result = await activate_authority_record(db_session, actor, request)
    await db_session.commit()
    assert result.outcome == "applied"
    assert result.after.status == "current"
    assert await counts(db_session) == (1, 1)
    receipt = await db_session.get(
        AuthorityMutationReceipt,
        (actor.requesting_client_id, hashlib.sha256(b"activate-once").hexdigest()),
    )
    assert str(receipt.audit_event_id) == result.audit_event_id
    audit = await db_session.get(AuditEvent, receipt.audit_event_id)
    assert audit.operation_metadata["requesting_client_id"] == actor.requesting_client_id
    assert audit.operation_metadata["authorizing_client_id"] == actor.authorizing_client_id
    assert audit.operation_metadata["provenance"] == {"model_label": "synthetic-only"}
    replay = await activate_authority_record(db_session, actor, request)
    assert replay.outcome == "idempotent_replay"
    assert replay.operation_id == result.operation_id
    changed = request.model_copy(update={"reason": "different"})
    with pytest.raises(AuthorityMutationError, match="idempotency_conflict"):
        await activate_authority_record(db_session, actor, changed)


@pytest.mark.asyncio
async def test_failure_after_mutation_rolls_back_authority_audit_and_receipt(db_session, monkeypatch):
    from app.domain.authority import mutations

    actor = await grant(db_session, can_mutate=True)
    item = record()
    await seed(db_session, item)

    original_persist = mutations._persist_result

    async def fail_persistence(*args, **kwargs):
        await original_persist(*args, **kwargs)
        raise RuntimeError("synthetic persistence failure")

    monkeypatch.setattr(mutations, "_persist_result", fail_persistence)
    request = ActivateAuthorityRecord(
        scope=SCOPE,
        record_id=item.id,
        expected_state_digest=await current_digest(db_session),
        idempotency_key="rollback",
        confirm=True,
    )
    with pytest.raises(RuntimeError, match="synthetic persistence failure"):
        await activate_authority_record(db_session, actor, request)
    await db_session.rollback()
    row = await db_session.get(AuthorityRecordRow, UUID(item.id))
    assert row.lifecycle_status == "proposed"
    assert await counts(db_session) == (0, 0)


@pytest.mark.asyncio
async def test_supersession_is_atomic_and_terminal_states_do_not_resurrect(db_session):
    actor = await grant(db_session, can_mutate=True)
    predecessor, successor = record(status="active"), record()
    await seed(db_session, predecessor, successor)
    supersession = SupersedeAuthorityRecord(
        scope=SCOPE,
        successor_record_id=successor.id,
        predecessor_record_id=predecessor.id,
        expected_state_digest=await current_digest(db_session),
        idempotency_key="supersede",
        confirm=True,
        reason="replacement",
    )
    result = await supersede_authority_record(db_session, actor, supersession)
    await db_session.commit()
    assert result.after.current_record_id == successor.id
    edges = (await db_session.execute(select(AuthorityEdgeRow))).scalars().all()
    assert len(edges) == 1
    withdrawal = WithdrawAuthorityRecord(
        scope=SCOPE,
        record_id=successor.id,
        expected_state_digest=await current_digest(db_session),
        idempotency_key="withdraw",
        confirm=True,
        reason="closed",
    )
    terminal = await withdraw_authority_record(db_session, actor, withdrawal)
    await db_session.commit()
    assert terminal.after.status == "empty"
    assert terminal.after.current_record_id is None


@pytest.mark.asyncio
async def test_dispute_is_guarded_and_does_not_resurrect_predecessor(db_session):
    actor = await grant(db_session, can_mutate=True)
    predecessor, successor = record(status="active"), record()
    await seed(db_session, predecessor, successor)
    await supersede_authority_record(
        db_session,
        actor,
        SupersedeAuthorityRecord(
            scope=SCOPE,
            successor_record_id=successor.id,
            predecessor_record_id=predecessor.id,
            expected_state_digest=await current_digest(db_session),
            idempotency_key="supersede-dispute",
            confirm=True,
        ),
    )
    await db_session.commit()
    result = await mark_authority_disputed(
        db_session,
        actor,
        MarkAuthorityDisputed(
            scope=SCOPE,
            record_id=successor.id,
            expected_state_digest=await current_digest(db_session),
            idempotency_key="dispute",
            confirm=True,
            reason="conflicting evidence",
        ),
    )
    await db_session.commit()
    assert result.after.status == "empty"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "other_scope",
    [
        AuthorityScope(space_id="other", workstream_id="bootstrap", authority_key="storage"),
        AuthorityScope(space_id="shared", workstream_id="other", authority_key="storage"),
        AuthorityScope(space_id="shared", workstream_id="bootstrap", authority_key="other"),
    ],
)
async def test_cross_scope_cycle_and_model_provenance_do_not_bypass_guards(
    db_session, other_scope
):
    actor = await grant(db_session, can_mutate=True)
    first, second = record(status="active"), record()
    outsider = record(scope=other_scope)
    await seed(db_session, first, second, outsider)
    cross_scope = SupersedeAuthorityRecord(
        scope=SCOPE,
        successor_record_id=outsider.id,
        predecessor_record_id=first.id,
        expected_state_digest=await current_digest(db_session),
        idempotency_key="cross-scope",
        confirm=True,
        provenance={"model_label": "cannot-authorize"},
    )
    with pytest.raises(AuthorityValidationError):
        await supersede_authority_record(db_session, actor, cross_scope)
    await db_session.rollback()
    await supersede_authority_record(
        db_session,
        actor,
        SupersedeAuthorityRecord(
            scope=SCOPE,
            successor_record_id=second.id,
            predecessor_record_id=first.id,
            expected_state_digest=await current_digest(db_session),
            idempotency_key="forward",
            confirm=True,
        ),
    )
    await db_session.commit()
    with pytest.raises(AuthorityValidationError):
        await supersede_authority_record(
            db_session,
            actor,
            SupersedeAuthorityRecord(
                scope=SCOPE,
                successor_record_id=first.id,
                predecessor_record_id=second.id,
                expected_state_digest=await current_digest(db_session),
                idempotency_key="cycle",
                confirm=True,
            ),
        )


@pytest.mark.asyncio
async def test_explicit_activation_may_preserve_ambiguity(db_session):
    actor = await grant(db_session, can_mutate=True)
    first, second = record(status="active"), record()
    await seed(db_session, first, second)
    result = await activate_authority_record(
        db_session,
        actor,
        ActivateAuthorityRecord(
            scope=SCOPE,
            record_id=second.id,
            expected_state_digest=await current_digest(db_session),
            idempotency_key="explicit-ambiguity",
            confirm=True,
        ),
    )
    await db_session.commit()
    assert result.after.status == "ambiguous"
    assert result.warnings == ["authority_state_ambiguous"]


@pytest.mark.asyncio
async def test_model_provenance_without_capability_cannot_propose(db_session):
    actor = await grant(db_session)
    with pytest.raises(AuthorityMutationError, match="authority_proposal_forbidden"):
        await create_authority_proposal(
            db_session,
            actor,
            proposal(provenance={"author_kind": "model", "model_label": "synthetic"}),
        )
