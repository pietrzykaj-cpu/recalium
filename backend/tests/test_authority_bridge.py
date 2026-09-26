"""Read-only authority exposure tests; disposable PostgreSQL only."""

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import hashlib
import pytest

from app.api import bridge as bridge_module
from app.domain.authority.contracts import AuthorityRecord
from app.domain.authority.repository import create_record, mark_disputed, supersede, withdraw
from app.domain.bridge.models import BridgeClient, BridgeGrant, BridgeProject

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)


def record(*, space="shared", workstream="bootstrap", key="state", status="proposed", content="synthetic", provenance=None):
    return AuthorityRecord(
        id=str(uuid4()),
        space_id=space,
        workstream_id=workstream,
        authority_key=key,
        record_kind="decision",
        content=content,
        lifecycle_status=status,
        created_at=NOW,
        created_by="synthetic-test",
        provenance=provenance or {"source": "synthetic"},
    )


@pytest.fixture
async def identity(db_session):
    token = "synthetic-bridge-token-" + "x" * 32
    db_session.add(BridgeClient(id="reader", credential_digest=hashlib.sha256(token.encode()).hexdigest()))
    db_session.add_all([
        BridgeProject(id="shared", kind="shared"),
        BridgeProject(id="other", kind="shared"),
    ])
    await db_session.flush()
    db_session.add(BridgeGrant(client_id="reader", project_id="shared", can_read=True, can_write=False))
    await db_session.commit()
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def local_bridge_host(client):
    client.base_url = "http://localhost"


async def query_mcp(payload, headers):
    headers = {"authorization": headers["Authorization"]}
    ctx = SimpleNamespace(
        request_context=SimpleNamespace(request=SimpleNamespace(headers=headers))
    )
    return await bridge_module.get_current_authority(
        bridge_module.CurrentAuthorityInput(**payload), ctx
    )


@pytest.mark.asyncio
async def test_current_historical_provenance_and_rest_mcp_equivalence(client, identity, db_session):
    predecessor = record(status="active", content="old", provenance={"source": "old"})
    successor = record(content="new", provenance={"source": "new"})
    await create_record(db_session, predecessor)
    await create_record(db_session, successor)
    await db_session.commit()
    await supersede(
        db_session,
        successor.id,
        predecessor.id,
        created_at=NOW,
        created_by="synthetic-test",
        reason="replacement",
    )
    await db_session.commit()

    payload = {
        "space_id": "shared",
        "workstream_id": "bootstrap",
        "authority_key": "state",
        "include_historical": True,
        "include_provenance": True,
    }
    response = await client.post("/bridge/v1/get_current_authority", json=payload, headers=identity)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "current"
    assert body["current_record"]["id"] == successor.id
    assert body["current_record"]["provenance"] == {"source": "new"}
    assert [item["id"] for item in body["historical_records"]] == [predecessor.id]
    assert body["deterministic"]["currentness"] == "graph_derived"

    mcp_body = await query_mcp(payload, identity)
    assert mcp_body == body
    repeat = await client.post("/bridge/v1/get_current_authority", json=payload, headers=identity)
    assert repeat.json() == body


@pytest.mark.asyncio
async def test_empty_ambiguous_and_unknown_scope_semantics(client, identity, db_session):
    empty = await client.post(
        "/bridge/v1/get_current_authority",
        json={"space_id": "shared", "workstream_id": "missing", "authority_key": "missing"},
        headers=identity,
    )
    assert empty.status_code == 200
    assert empty.json()["status"] == "empty"

    first, second = record(key="conflict", status="active"), record(key="conflict", status="active")
    await create_record(db_session, first)
    await create_record(db_session, second)
    await db_session.commit()
    ambiguous = await client.post(
        "/bridge/v1/get_current_authority",
        json={"space_id": "shared", "workstream_id": "bootstrap", "authority_key": "conflict"},
        headers=identity,
    )
    assert ambiguous.status_code == 200
    assert ambiguous.json()["status"] == "ambiguous"
    assert len(ambiguous.json()["competing_records"]) == 2

    unknown = await client.post(
        "/bridge/v1/get_current_authority",
        json={"space_id": "unknown", "workstream_id": "bootstrap", "authority_key": "state"},
        headers=identity,
    )
    assert unknown.status_code == 403
    assert unknown.json() == {"error": "permission_denied"}


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_operation", [withdraw, mark_disputed])
async def test_withdrawn_or_disputed_successor_does_not_resurrect_predecessor(
    client, identity, db_session, terminal_operation
):
    predecessor = record(key="terminal", status="active")
    successor = record(key="terminal")
    await create_record(db_session, predecessor)
    await create_record(db_session, successor)
    await db_session.commit()
    await supersede(db_session, successor.id, predecessor.id, created_at=NOW, created_by="synthetic-test")
    await db_session.commit()
    await terminal_operation(db_session, successor.id, reason="synthetic")
    await db_session.commit()

    response = await client.post(
        "/bridge/v1/get_current_authority",
        json={"space_id": "shared", "workstream_id": "bootstrap", "authority_key": "terminal"},
        headers=identity,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "empty"
    assert response.json()["current_record"] is None


@pytest.mark.asyncio
async def test_cross_space_isolation_and_input_validation(client, identity, db_session):
    visible = record(key="same", status="active", content="visible")
    hidden = record(space="other", key="same", status="active", content="hidden")
    await create_record(db_session, visible)
    await create_record(db_session, hidden)
    await db_session.commit()

    visible_response = await client.post(
        "/bridge/v1/get_current_authority",
        json={"space_id": "shared", "workstream_id": "bootstrap", "authority_key": "same"},
        headers=identity,
    )
    assert visible_response.status_code == 200
    assert visible_response.json()["current_record"]["content"] == "visible"
    assert hidden.id not in str(visible_response.json())

    denied = await client.post(
        "/bridge/v1/get_current_authority",
        json={"space_id": "other", "workstream_id": "bootstrap", "authority_key": "same"},
        headers=identity,
    )
    assert denied.status_code == 403

    malformed = await client.post(
        "/bridge/v1/get_current_authority",
        json={"space_id": "shared", "workstream_id": "", "authority_key": "same"},
        headers=identity,
    )
    assert malformed.status_code == 422


@pytest.mark.asyncio
async def test_authority_mcp_catalog_is_read_only():
    tools = await bridge_module.bridge_mcp.list_tools()
    names = {tool.name for tool in tools}
    assert "get_current_authority" in names
    tool = next(tool for tool in tools if tool.name == "get_current_authority")
    assert tool.annotations.readOnlyHint is True
    assert not any(name in names for name in {"create_authority", "activate_authority", "supersede_authority", "withdraw_authority", "dispute_authority"})
