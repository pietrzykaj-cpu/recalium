"""Actual REST/MCP coverage for the guarded authority write boundary."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from mcp.server.fastmcp.exceptions import ToolError
from sqlalchemy import func, select

from app.api import bridge as bridge_module
from app.domain.audit.models import AuditEvent
from app.domain.authority.contracts import AuthorityRecord
from app.domain.authority.models import AuthorityEdgeRow, AuthorityRecordRow
from app.domain.authority.mutation_contracts import AuthorityScope
from app.domain.authority.mutations import authority_state_digest
from app.domain.authority.repository import create_record, list_scope_edges, list_scope_records
from app.domain.bridge.models import (
    AuthorityMutationReceipt,
    BridgeClient,
    BridgeGrant,
    BridgeProject,
)

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
SCOPE = AuthorityScope(space_id="shared", workstream_id="bootstrap", authority_key="storage")


@pytest.fixture(autouse=True)
def local_bridge_host(client):
    client.base_url = "http://localhost"


@pytest.fixture
async def authority_identities(db_session):
    clients = {
        "proposer": {"propose": True, "mutate": False, "write": False, "grant": True},
        "mutator": {"propose": False, "mutate": True, "write": False, "grant": True},
        "writer": {"propose": False, "mutate": False, "write": True, "grant": True},
        "ungranted": {"propose": False, "mutate": False, "write": False, "grant": False},
    }
    db_session.add(BridgeProject(id="shared", kind="shared"))
    for name, permissions in clients.items():
        token = f"synthetic-{name}-credential-" + "x" * 32
        permissions["token"] = token
        db_session.add(
            BridgeClient(
                id=name,
                credential_digest=hashlib.sha256(token.encode()).hexdigest(),
            )
        )
    await db_session.flush()
    for name, permissions in clients.items():
        if not permissions["grant"]:
            continue
        db_session.add(
            BridgeGrant(
                client_id=name,
                project_id="shared",
                can_read=True,
                can_write=permissions["write"],
                can_propose_authority=permissions["propose"],
                can_mutate_authority=permissions["mutate"],
            )
        )
    await db_session.commit()
    return {
        name: {"Authorization": f"Bearer {permissions['token']}"}
        for name, permissions in clients.items()
    }


def scope_payload(scope=SCOPE):
    return scope.model_dump(mode="json")


def proposal_payload(*, key, record_id=None, provenance=None):
    return {
        "scope": scope_payload(),
        "record_id": record_id or str(uuid4()),
        "record_kind": "decision",
        "content": "Use SQLite for the synthetic bootstrap",
        "provenance": provenance or {"source": "synthetic-interface-test"},
        "idempotency_key": key,
    }


def mcp_context(headers):
    return SimpleNamespace(
        request_context=SimpleNamespace(
            request=SimpleNamespace(
                headers={"authorization": headers["Authorization"]}
            )
        )
    )


async def current_digest(session, scope=SCOPE):
    records = await list_scope_records(
        session,
        space_id=scope.space_id,
        workstream_id=scope.workstream_id,
        authority_key=scope.authority_key,
    )
    edges = await list_scope_edges(session, record_ids=[record.id for record in records])
    return authority_state_digest(records, edges, scope=scope)


def authority_record(*, status="proposed", scope=SCOPE, content="synthetic decision"):
    return AuthorityRecord(
        id=str(uuid4()),
        space_id=scope.space_id,
        workstream_id=scope.workstream_id,
        authority_key=scope.authority_key,
        record_kind="decision",
        content=content,
        lifecycle_status=status,
        created_at=NOW,
        created_by="synthetic-seed",
        provenance={"source": "synthetic-interface-test"},
    )


async def seed(session, *records):
    for record in records:
        await create_record(session, record)
    await session.commit()


async def authority_counts(session):
    return {
        "records": await session.scalar(select(func.count()).select_from(AuthorityRecordRow)),
        "edges": await session.scalar(select(func.count()).select_from(AuthorityEdgeRow)),
        "receipts": await session.scalar(
            select(func.count()).select_from(AuthorityMutationReceipt)
        ),
        "audits": await session.scalar(
            select(func.count()).select_from(AuditEvent).where(
                AuditEvent.event_type == "authority_mutation"
            )
        ),
    }


@pytest.mark.asyncio
async def test_rest_and_mcp_proposals_are_non_authoritative_and_preserve_provenance(
    client, authority_identities, db_session
):
    rest_payload = proposal_payload(
        key="rest-proposal",
        provenance={
            "author_kind": "model",
            "model_label": "synthetic-model",
            "session_id": "synthetic-session",
        },
    )
    rest = await client.post(
        "/bridge/v1/create_authority_proposal",
        json=rest_payload,
        headers=authority_identities["proposer"],
    )
    assert rest.status_code == 200, rest.text
    assert rest.json()["after"]["status"] == "empty"
    row = await db_session.get(AuthorityRecordRow, UUID(rest_payload["record_id"]))
    assert row.lifecycle_status == "proposed"
    assert row.created_by == "proposer"
    assert row.provenance["model_label"] == "synthetic-model"

    mcp_payload = proposal_payload(key="mcp-proposal")
    mcp = await bridge_module.create_authority_proposal(
        bridge_module.CreateAuthorityProposal(**mcp_payload),
        mcp_context(authority_identities["proposer"]),
    )
    assert mcp["after"]["status"] == "empty"
    mcp_replay = await bridge_module.create_authority_proposal(
        bridge_module.CreateAuthorityProposal(**mcp_payload),
        mcp_context(authority_identities["proposer"]),
    )
    assert mcp_replay["outcome"] == "idempotent_replay"
    assert mcp_replay["operation_id"] == mcp["operation_id"]
    with pytest.raises(ToolError, match="idempotency_conflict"):
        await bridge_module.create_authority_proposal(
            bridge_module.CreateAuthorityProposal(
                **{**mcp_payload, "content": "Changed synthetic proposal"}
            ),
            mcp_context(authority_identities["proposer"]),
        )
    current = await client.post(
        "/bridge/v1/get_current_authority",
        json=scope_payload(),
        headers=authority_identities["proposer"],
    )
    assert current.status_code == 200
    assert current.json()["status"] == "empty"
    assert current.json()["state_digest"] == mcp["after"]["digest"]


@pytest.mark.asyncio
async def test_capabilities_are_separate_for_rest_and_mcp(
    client, authority_identities, db_session
):
    item = authority_record()
    await seed(db_session, item)
    payload = {
        "scope": scope_payload(),
        "record_id": item.id,
        "expected_state_digest": await current_digest(db_session),
        "idempotency_key": "activate-denied",
        "confirm": True,
        "provenance": {"model_label": "cannot-authorize"},
    }
    rest = await client.post(
        "/bridge/v1/activate_authority_record",
        json=payload,
        headers=authority_identities["proposer"],
    )
    assert rest.status_code == 403
    assert rest.json() == {"error": "authority_mutation_forbidden"}
    with pytest.raises(ToolError, match="authority_mutation_forbidden"):
        await bridge_module.activate_authority_record(
            bridge_module.ActivateAuthorityRecord(**payload),
            mcp_context(authority_identities["proposer"]),
        )
    writer = await client.post(
        "/bridge/v1/activate_authority_record",
        json={**payload, "idempotency_key": "writer-denied"},
        headers=authority_identities["writer"],
    )
    assert writer.status_code == 403
    assert writer.json() == {"error": "authority_mutation_forbidden"}
    assert (await db_session.get(AuthorityRecordRow, UUID(item.id))).lifecycle_status == "proposed"


@pytest.mark.asyncio
async def test_authentication_and_space_permission_errors_are_distinct(
    client, authority_identities
):
    payload = proposal_payload(key="permission-errors")
    unauthenticated = await client.post(
        "/bridge/v1/create_authority_proposal",
        json=payload,
    )
    assert unauthenticated.status_code == 401
    assert unauthenticated.json() == {"error": "authentication_required"}
    ungranted = await client.post(
        "/bridge/v1/create_authority_proposal",
        json=payload,
        headers=authority_identities["ungranted"],
    )
    assert ungranted.status_code == 403
    assert ungranted.json() == {"error": "permission_denied"}


@pytest.mark.asyncio
async def test_guards_dry_run_replay_and_conflict_through_rest(
    client, authority_identities, db_session
):
    item = authority_record()
    await seed(db_session, item)
    digest = await current_digest(db_session)
    base = {
        "scope": scope_payload(),
        "record_id": item.id,
        "expected_state_digest": digest,
        "idempotency_key": "activate-once",
    }
    missing_confirmation = await client.post(
        "/bridge/v1/activate_authority_record",
        json=base,
        headers=authority_identities["mutator"],
    )
    assert missing_confirmation.status_code == 409
    assert missing_confirmation.json() == {"error": "confirmation_required"}
    missing_digest = await client.post(
        "/bridge/v1/activate_authority_record",
        json={key: value for key, value in base.items() if key != "expected_state_digest"},
        headers=authority_identities["mutator"],
    )
    assert missing_digest.status_code == 422
    assert missing_digest.json() == {"error": "invalid_input"}
    stale = await client.post(
        "/bridge/v1/activate_authority_record",
        json={**base, "expected_state_digest": "0" * 64, "confirm": True},
        headers=authority_identities["mutator"],
    )
    assert stale.status_code == 409
    assert stale.json() == {"error": "stale_authority_state"}

    before = await authority_counts(db_session)
    preview = await client.post(
        "/bridge/v1/activate_authority_record",
        json={**base, "dry_run": True},
        headers=authority_identities["mutator"],
    )
    assert preview.status_code == 200
    assert preview.json()["outcome"] == "preview"
    assert preview.json()["after"]["status"] == "current"
    assert await authority_counts(db_session) == before
    assert (await db_session.get(AuthorityRecordRow, UUID(item.id))).lifecycle_status == "proposed"

    applied = await client.post(
        "/bridge/v1/activate_authority_record",
        json={**base, "confirm": True},
        headers=authority_identities["mutator"],
    )
    assert applied.status_code == 200, applied.text
    assert applied.json()["outcome"] == "applied"
    counts = await authority_counts(db_session)
    replay = await client.post(
        "/bridge/v1/activate_authority_record",
        json={**base, "confirm": True},
        headers=authority_identities["mutator"],
    )
    assert replay.status_code == 200
    assert replay.json()["outcome"] == "idempotent_replay"
    assert replay.json()["operation_id"] == applied.json()["operation_id"]
    assert await authority_counts(db_session) == counts
    conflict = await client.post(
        "/bridge/v1/activate_authority_record",
        json={**base, "confirm": True, "reason": "changed payload"},
        headers=authority_identities["mutator"],
    )
    assert conflict.status_code == 409
    assert conflict.json() == {"error": "idempotency_conflict"}


@pytest.mark.asyncio
async def test_supersession_terminal_states_and_invalid_graph_changes(
    client, authority_identities, db_session
):
    predecessor, successor = authority_record(status="active"), authority_record()
    await seed(db_session, predecessor, successor)
    supersede_payload = {
        "scope": scope_payload(),
        "successor_record_id": successor.id,
        "predecessor_record_id": predecessor.id,
        "expected_state_digest": await current_digest(db_session),
        "idempotency_key": "supersede",
        "confirm": True,
        "reason": "synthetic replacement",
    }
    supersede_result = await client.post(
        "/bridge/v1/supersede_authority_record",
        json=supersede_payload,
        headers=authority_identities["mutator"],
    )
    assert supersede_result.status_code == 200, supersede_result.text
    assert supersede_result.json()["after"]["current_record_id"] == successor.id

    cycle = await client.post(
        "/bridge/v1/supersede_authority_record",
        json={
            **supersede_payload,
            "successor_record_id": predecessor.id,
            "predecessor_record_id": successor.id,
            "expected_state_digest": await current_digest(db_session),
            "idempotency_key": "cycle",
        },
        headers=authority_identities["mutator"],
    )
    assert cycle.status_code == 409
    assert cycle.json() == {"error": "supersession_cycle"}

    withdrawal = await client.post(
        "/bridge/v1/withdraw_authority_record",
        json={
            "scope": scope_payload(),
            "record_id": successor.id,
            "expected_state_digest": await current_digest(db_session),
            "idempotency_key": "withdraw",
            "confirm": True,
            "reason": "synthetic close",
        },
        headers=authority_identities["mutator"],
    )
    assert withdrawal.status_code == 200
    assert withdrawal.json()["after"]["status"] == "empty"
    assert withdrawal.json()["after"]["current_record_id"] is None


@pytest.mark.asyncio
async def test_dispute_and_cross_scope_failure_leave_no_partial_state(
    client, authority_identities, db_session
):
    predecessor, successor = authority_record(status="active"), authority_record()
    outside_scope = AuthorityScope(
        space_id="shared", workstream_id="other", authority_key="storage"
    )
    outside = authority_record(scope=outside_scope)
    await seed(db_session, predecessor, successor, outside)
    before = await authority_counts(db_session)
    cross_scope = await client.post(
        "/bridge/v1/supersede_authority_record",
        json={
            "scope": scope_payload(),
            "successor_record_id": outside.id,
            "predecessor_record_id": predecessor.id,
            "expected_state_digest": await current_digest(db_session),
            "idempotency_key": "cross-scope",
            "confirm": True,
        },
        headers=authority_identities["mutator"],
    )
    assert cross_scope.status_code == 404
    assert cross_scope.json() == {"error": "unknown_authority_record"}
    assert await authority_counts(db_session) == before

    supersede_result = await client.post(
        "/bridge/v1/supersede_authority_record",
        json={
            "scope": scope_payload(),
            "successor_record_id": successor.id,
            "predecessor_record_id": predecessor.id,
            "expected_state_digest": await current_digest(db_session),
            "idempotency_key": "supersede-dispute",
            "confirm": True,
        },
        headers=authority_identities["mutator"],
    )
    assert supersede_result.status_code == 200
    dispute = await client.post(
        "/bridge/v1/mark_authority_disputed",
        json={
            "scope": scope_payload(),
            "record_id": successor.id,
            "expected_state_digest": await current_digest(db_session),
            "idempotency_key": "dispute",
            "confirm": True,
            "reason": "synthetic uncertainty",
        },
        headers=authority_identities["mutator"],
    )
    assert dispute.status_code == 200
    assert dispute.json()["after"]["status"] == "empty"
    assert dispute.json()["after"]["current_record_id"] is None


@pytest.mark.asyncio
async def test_rest_mcp_preview_equivalence_and_catalog_metadata(
    client, authority_identities, db_session
):
    item = authority_record()
    await seed(db_session, item)
    payload = {
        "scope": scope_payload(),
        "record_id": item.id,
        "expected_state_digest": await current_digest(db_session),
        "idempotency_key": "equivalent-preview",
        "dry_run": True,
    }
    rest = await client.post(
        "/bridge/v1/activate_authority_record",
        json=payload,
        headers=authority_identities["mutator"],
    )
    mcp = await bridge_module.activate_authority_record(
        bridge_module.ActivateAuthorityRecord(**payload),
        mcp_context(authority_identities["mutator"]),
    )
    assert rest.status_code == 200
    assert mcp == rest.json()

    tools = {tool.name: tool for tool in await bridge_module.bridge_mcp.list_tools()}
    proposal = tools["create_authority_proposal"]
    assert proposal.annotations.readOnlyHint is False
    assert proposal.annotations.destructiveHint is False
    assert proposal.annotations.idempotentHint is True
    for name in (
        "activate_authority_record",
        "supersede_authority_record",
        "withdraw_authority_record",
        "mark_authority_disputed",
    ):
        assert tools[name].annotations.readOnlyHint is False
        assert tools[name].annotations.destructiveHint is True
        assert tools[name].annotations.idempotentHint is True


@pytest.mark.asyncio
async def test_persistence_failure_rolls_back_and_maps_without_internal_details(
    client, authority_identities, db_session, monkeypatch
):
    from app.domain.authority import mutations

    item = authority_record()
    await seed(db_session, item)
    original_persist = mutations._persist_result

    async def fail_after_persist(*args, **kwargs):
        await original_persist(*args, **kwargs)
        raise RuntimeError("synthetic persistence failure detail")

    monkeypatch.setattr(mutations, "_persist_result", fail_after_persist)
    response = await client.post(
        "/bridge/v1/activate_authority_record",
        json={
            "scope": scope_payload(),
            "record_id": item.id,
            "expected_state_digest": await current_digest(db_session),
            "idempotency_key": "persistence-rollback",
            "confirm": True,
        },
        headers=authority_identities["mutator"],
    )
    assert response.status_code == 500
    assert response.json() == {"error": "authority_persistence_failure"}
    db_session.expire_all()
    assert (await db_session.get(AuthorityRecordRow, UUID(item.id))).lifecycle_status == "proposed"
    counts = await authority_counts(db_session)
    assert counts["receipts"] == 0
    assert counts["audits"] == 0
