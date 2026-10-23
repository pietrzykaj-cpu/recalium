"""Retrieve-only Bridge MCP mount (/bridge/mcp-read): disposable PostgreSQL only; never run against live DB.

The read mount must list exactly one tool, force the shared space, reuse the existing
retrieve path, and leave /bridge/mcp and the REST Bridge unchanged.
"""

import hashlib
import json
import secrets
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from app.api import bridge as bridge_module
from app.domain.bridge.contracts import RetrieveInput
from app.domain.bridge.models import BridgeClient, BridgeGrant, BridgeProject

OTHER_BRIDGE_TOOLS = {
    "build_context_packet",
    "ingest_memory",
    "get_ingest_status",
    "get_current_authority",
    "build_continuity_handoff",
    "create_authority_proposal",
    "activate_authority_record",
    "supersede_authority_record",
    "withdraw_authority_record",
    "mark_authority_disputed",
}
EXISTING_CATALOG = OTHER_BRIDGE_TOOLS | {"retrieve_memory"}
DENIED_SCOPES = [
    {"space_ids": ["sol-private"]},
    {"space_ids": ["shared", "sol-private"]},
    {"space_ids": ["other"]},
    {"space_ids": []},
    {"project_id": "shared"},
]


def read_mcp():
    return getattr(bridge_module, "bridge_mcp_read")


def read_tool():
    return getattr(bridge_module, "retrieve_memory_read")


@pytest.fixture
async def identities(db_session):
    """`reader`: shared read only. `sol`: read+write on shared and owner of sol-private (as in production)."""
    tokens = {name: secrets.token_urlsafe(32) for name in ("reader", "sol")}
    for name, token in tokens.items():
        db_session.add(
            BridgeClient(id=name, credential_digest=hashlib.sha256(token.encode()).hexdigest())
        )
    await db_session.flush()
    db_session.add_all(
        [
            BridgeProject(id="shared", kind="shared"),
            BridgeProject(id="sol-private", kind="private", owner_client_id="sol"),
        ]
    )
    await db_session.flush()
    db_session.add_all(
        [
            BridgeGrant(client_id="reader", project_id="shared", can_read=True, can_write=False),
            BridgeGrant(client_id="sol", project_id="shared", can_read=True, can_write=True),
            BridgeGrant(client_id="sol", project_id="sol-private", can_read=True, can_write=True),
        ]
    )
    await db_session.commit()
    return {name: {"Authorization": f"Bearer {token}"} for name, token in tokens.items()}


# ── Catalog isolation (no HTTP) ─────────────────────────────────────────────


async def test_read_catalog_is_exactly_retrieve_memory():
    tools = await read_mcp().list_tools()
    assert [tool.name for tool in tools] == ["retrieve_memory"]
    annotations = tools[0].annotations
    assert annotations.readOnlyHint is True and annotations.destructiveHint is False


async def test_read_catalog_registers_no_write_or_other_bridge_tool():
    names = {tool.name for tool in await read_mcp().list_tools()}
    assert not names & OTHER_BRIDGE_TOOLS
    assert await read_mcp().list_resources() == []
    assert await read_mcp().list_prompts() == []


@pytest.mark.parametrize(
    "name", ["ingest_memory", "create_authority_proposal", "activate_authority_record"]
)
async def test_unregistered_tools_are_unknown_on_read_mount(name):
    with pytest.raises(Exception, match="Unknown tool"):
        await read_mcp().call_tool(name, {"data": {}})


async def test_existing_bridge_mcp_catalog_unchanged():
    assert {tool.name for tool in await bridge_module.bridge_mcp.list_tools()} == EXISTING_CATALOG


# ── Scope rule of the read wrapper (fake MCP context, no HTTP) ──────────────


def fake_ctx(actor="reader"):
    return SimpleNamespace(
        request_context=SimpleNamespace(
            request=SimpleNamespace(
                headers={"authorization": "Bearer unused"}, state=SimpleNamespace(bridge_actor=actor)
            )
        )
    )


async def test_read_wrapper_forces_shared_and_calls_only_retrieve(monkeypatch):
    calls = []

    async def capture(ctx, operation, data):
        calls.append((operation, data.space_ids, data.project_id))
        return {"items": []}

    monkeypatch.setattr(bridge_module, "mcp_call", capture)
    await read_tool()(RetrieveInput(query="violet"), fake_ctx())
    await read_tool()(RetrieveInput(query="violet", space_ids=["shared"]), fake_ctx())
    assert calls == [("retrieve_memory", ["shared"], None), ("retrieve_memory", ["shared"], None)]


@pytest.mark.parametrize("scope", DENIED_SCOPES)
async def test_read_wrapper_rejects_other_scopes_fail_closed(monkeypatch, db_session, scope):
    from mcp.server.fastmcp.exceptions import ToolError

    from app.domain.audit.models import AuditEvent

    async def must_not_run(ctx, operation, data):  # pragma: no cover - failing path
        raise AssertionError("execute reached for a denied scope")

    monkeypatch.setattr(bridge_module, "mcp_call", must_not_run)
    with pytest.raises(ToolError, match="permission_denied"):
        await read_tool()(RetrieveInput(query="violet", **scope), fake_ctx("sol"))
    await db_session.rollback()
    rows = (await db_session.execute(select(AuditEvent))).scalars().all()
    assert [(r.event_type, r.actor, r.operation_metadata.get("operation"), r.operation_metadata.get("outcome"),
             r.operation_metadata.get("code"), r.operation_metadata.get("surface")) for r in rows] == [
        ("bridge_access", "sol", "retrieve_memory", "denied", "permission_denied", "bridge-mcp-read")
    ]


# ── HTTP end to end: both MCP mounts and REST through the full app ──────────


def rpc(method, params=None, request_id=1):
    body = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


MCP_HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}


async def test_read_mount_http_end_to_end(client, identities, db_session):
    """One test owns both session managers (each may run only once per process)."""
    from app.domain.archive.models import RawArchiveItem
    from app.domain.audit.models import AuditEvent
    from app.domain.derived_memory.models import Fact

    client.base_url = "http://localhost:8000"  # SDK host check expects host:port

    archives = {}
    for space, headers in (("shared", identities["sol"]), ("sol-private", identities["sol"])):
        response = await client.post(
            "/bridge/v1/ingest_memory",
            json={
                "project_id": space,
                "content": f"Synthetic violet note kept in {space}.",
                "source_metadata": {"author_kind": "human", "conversation_id": "synthetic"},
                "idempotency_key": f"seed-{space}",
            },
            headers=headers,
        )
        assert response.status_code == 202, response.text
        archives[space] = response.json()["archive_id"]
        db_session.add(
            Fact(
                raw_archive_id=uuid.UUID(archives[space]),
                fact_text=f"violet {space} fact",
                source_span="violet",
                confidence_tier="high",
                derivation_method="test",
                derivation_model="synthetic",
            )
        )
    await db_session.commit()

    async def audit_count():
        await db_session.rollback()
        return await db_session.scalar(select(func.count()).select_from(AuditEvent))

    async def archive_count():
        await db_session.rollback()
        return await db_session.scalar(select(func.count()).select_from(RawArchiveItem))

    async def call(path, method, params, headers, request_id=1):
        before = await audit_count()
        response = await client.post(
            path, json=rpc(method, params, request_id), headers={**MCP_HEADERS, **headers}
        )
        return response, (await audit_count()) - before

    def tool_result(response):
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        return result.get("isError", False), result["content"][0]["text"] if result["content"] else ""

    query = {"query": "violet", "mode": "keyword", "budget": 20000, "limit": 50}
    reader, sol = identities["reader"], identities["sol"]

    async with bridge_module.bridge_mcp.session_manager.run(), read_mcp().session_manager.run():
        # tools/list: read mount exactly one tool; existing mount unchanged (+1 transport row each)
        response, delta = await call("/bridge/mcp-read", "tools/list", None, reader)
        assert [t["name"] for t in response.json()["result"]["tools"]] == ["retrieve_memory"]
        assert delta == 1
        response, delta = await call("/bridge/mcp", "tools/list", None, reader)
        assert {t["name"] for t in response.json()["result"]["tools"]} == EXISTING_CATALOG
        assert delta == 1

        # shared retrieve succeeds; omitted space_ids defaults to shared (+3 each)
        params = {"name": "retrieve_memory", "arguments": {"data": {**query, "space_ids": ["shared"]}}}
        response, delta = await call("/bridge/mcp-read", "tools/call", params, reader)
        is_error, text = tool_result(response)
        assert not is_error and delta == 3
        read_items = json.loads(text)["items"]
        assert read_items and {item["source_id"] for item in read_items} == {archives["shared"]}
        assert json.loads(text)["searched_space_ids"] == ["shared"]

        params = {"name": "retrieve_memory", "arguments": {"data": query}}
        response, delta = await call("/bridge/mcp-read", "tools/call", params, reader)
        is_error, text = tool_result(response)
        assert not is_error and delta == 3
        assert [i["id"] for i in json.loads(text)["items"]] == [i["id"] for i in read_items]

        # same shared semantics as the existing MCP mount and the REST Bridge
        params = {"name": "retrieve_memory", "arguments": {"data": {**query, "space_ids": ["shared"]}}}
        response, _ = await call("/bridge/mcp", "tools/call", params, reader)
        is_error, text = tool_result(response)
        assert not is_error
        assert [i["id"] for i in json.loads(text)["items"]] == [i["id"] for i in read_items]
        rest = await client.post(
            "/bridge/v1/retrieve_memory", json={**query, "space_ids": ["shared"]}, headers=reader
        )
        assert rest.status_code == 200
        assert [i["id"] for i in rest.json()["items"]] == [i["id"] for i in read_items]

        # sol is entitled to sol-private on the existing mount ...
        params = {"name": "retrieve_memory", "arguments": {"data": {**query, "space_ids": ["sol-private"]}}}
        response, _ = await call("/bridge/mcp", "tools/call", params, sol)
        is_error, text = tool_result(response)
        assert not is_error and {i["source_id"] for i in json.loads(text)["items"]} == {archives["sol-private"]}
        # ... but the read mount rejects it, and every other scope, fail-closed (+2: transport + denied)
        for who in (sol, reader):
            for scope in DENIED_SCOPES:
                params = {"name": "retrieve_memory", "arguments": {"data": {**query, **scope}}}
                response, delta = await call("/bridge/mcp-read", "tools/call", params, who)
                is_error, text = tool_result(response)
                assert is_error and "permission_denied" in text and delta == 2, (scope, text, delta)

        # unregistered write/authority tools fail as unknown; no data write (+1 transport row)
        archives_before = await archive_count()
        ingest = {
            "name": "ingest_memory",
            "arguments": {
                "data": {
                    "destination": "shared",
                    "content": "Synthetic violet note that must never be stored.",
                    "source_metadata": {"author_kind": "human"},
                    "idempotency_key": "never",
                }
            },
        }
        response, delta = await call("/bridge/mcp-read", "tools/call", ingest, sol)
        is_error, text = tool_result(response)
        assert is_error and "Unknown tool" in text and delta == 1
        assert await archive_count() == archives_before
        proposal = {"name": "create_authority_proposal", "arguments": {"data": {}}}
        response, delta = await call("/bridge/mcp-read", "tools/call", proposal, sol)
        is_error, text = tool_result(response)
        assert is_error and "Unknown tool" in text and delta == 1

        # BridgeBoundary still applies to the read mount
        list_body = rpc("tools/list")
        before = await audit_count()
        unauthenticated = await client.post("/bridge/mcp-read", json=list_body, headers=MCP_HEADERS)
        assert unauthenticated.status_code == 401 and (await audit_count()) - before == 1
        wrong = await client.post(
            "/bridge/mcp-read", json=list_body, headers={**MCP_HEADERS, "Authorization": "Bearer " + "x" * 40}
        )
        assert wrong.status_code == 401
        for header in ({"Origin": "https://evil.example"}, {"Host": "evil.example"}):
            blocked = await client.post(
                "/bridge/mcp-read", json=list_body, headers={**MCP_HEADERS, **reader, **header}
            )
            assert blocked.status_code == 403

    # audit chain of a successful read-mount retrieve matches the existing convention
    events = (
        (await db_session.execute(select(AuditEvent).order_by(AuditEvent.occurred_at))).scalars().all()
    )
    reads = [e for e in events if e.event_type == "mcp_retrieve" and e.actor == "reader"]
    assert reads and all(e.operation_metadata["searched_space_ids"] == ["shared"] for e in reads)
    assert "violet" not in str([e.operation_metadata for e in events])
