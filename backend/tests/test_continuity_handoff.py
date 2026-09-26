"""Synthetic, provider-free continuity handoff assembly tests."""
from datetime import datetime, timezone
import hashlib
from types import SimpleNamespace

import pytest
import pytest_asyncio

from app.domain.bridge import service as bridge_service
from app.api import bridge as bridge_module
from app.domain.bridge.contracts import ContinuityHandoffInput
from app.domain.bridge.models import BridgeClient, BridgeGrant, BridgeProject
from app.domain.context_packets.service import build_context_packet
from app.domain.retrieval.service import RetrievalItem, RetrievalResponse


NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)


def authority(key, status="current", record_id=None, content=None):
    record_id = record_id or f"{key}-current"
    current = None
    competing = []
    historical = []
    if status == "current":
        current = {
            "id": record_id, "space_id": "shared", "workstream_id": "bootstrap",
            "authority_key": key, "record_kind": "decision",
            "content": content or f"Current {key}", "lifecycle_status": "active",
            "created_at": NOW.isoformat(), "created_by": "synthetic",
            "provenance": {"source": "synthetic"},
        }
    elif status == "ambiguous":
        competing = [
            {
                "id": f"{key}-a", "space_id": "shared", "workstream_id": "bootstrap",
                "authority_key": key, "record_kind": "decision",
                "content": "Competing A", "lifecycle_status": "active",
                "created_at": NOW.isoformat(), "created_by": "synthetic",
                "provenance": {"source": "synthetic-a"},
            },
            {
                "id": f"{key}-b", "space_id": "shared", "workstream_id": "bootstrap",
                "authority_key": key, "record_kind": "decision",
                "content": "Competing B", "lifecycle_status": "active",
                "created_at": NOW.isoformat(), "created_by": "synthetic",
                "provenance": {"source": "synthetic-b"},
            },
        ]
    return {
        "status": status, "space_id": "shared", "workstream_id": "bootstrap",
        "authority_key": key, "current_record": current,
        "competing_records": competing, "historical_records": historical,
        "superseded_record_ids": [],
        "deterministic": {"currentness": "graph_derived", "source": "persisted_authority_records"},
    }


def packet_payload(*, token_budget=500):
    response = RetrievalResponse(
        query="continuity",
        retrieval_mode="keyword",
        budget_used=40,
        budget_limit=2000,
        trimming_reason="result_exhausted",
        items=[
            RetrievalItem(
                id="memory-1", type="fact", content="Synthetic supporting memory.",
                score=0.8, source_id="archive-1", source_system="synthetic",
                captured_at=NOW.isoformat(), conflict_label=None,
                provenance={"source_metadata": {"unresolved_questions": ["Q1"]}},
            )
        ],
    )
    return build_context_packet(response, token_budget=token_budget, generated_at=NOW).model_dump(mode="json")


@pytest.mark.asyncio
async def test_handoff_orders_keys_and_keeps_authority_outside_memory_budget(monkeypatch):
    calls = []

    async def fake_authority(session, request, spaces):
        calls.append(("authority", request.authority_key))
        return authority(request.authority_key, status={"decision": "current", "conflict": "ambiguous", "missing": "empty"}[request.authority_key])

    async def fake_packet(session, actor, request, spaces, generated_at=None):
        calls.append(("retrieval", generated_at))
        return packet_payload(token_budget=0)

    monkeypatch.setattr(bridge_service, "_current_authority", fake_authority)
    monkeypatch.setattr(bridge_service, "_context_packet", fake_packet)
    request = ContinuityHandoffInput(
        space_id="shared", workstream_id="bootstrap",
        authority_keys=["missing", "conflict", "decision"], query="continuity",
        current_agent={"provider": "local", "model": "synthetic", "capabilities": ["read"]},
        include_provenance=True,
    )
    result = await bridge_service._continuity_handoff(None, "reader", request, {"shared": object()})
    assert [item["authority_key"] for item in result["authority_results"]] == ["conflict", "decision", "missing"]
    assert {item["status"] for item in result["authority_results"]} == {"current", "ambiguous", "empty"}
    assert result["included_memory_ids"] == []
    assert result["excluded_memory_ids"] == ["memory-1"]
    assert "authority_ambiguous:conflict" in result["warnings"]
    assert "authority_empty:missing" in result["warnings"]
    assert "AUTHORITATIVE CURRENT STATE" in result["rendered_handoff"]["text"]
    assert "SUPPORTING MEMORY (NON-AUTHORITATIVE)" in result["rendered_handoff"]["text"]
    assert "CURRENT AGENT" in result["rendered_handoff"]["text"]
    assert result["provenance"]["authority_source"] == "persisted_authority_records"
    assert [kind for kind, _ in calls[:3]] == ["authority", "authority", "authority"]
    assert calls[-1][0] == "retrieval"


@pytest.mark.asyncio
async def test_handoff_is_deterministic_and_mcp_ready(monkeypatch):
    async def fake_authority(session, request, spaces):
        return authority(request.authority_key)

    async def fake_packet(session, actor, request, spaces, generated_at=None):
        return packet_payload()

    monkeypatch.setattr(bridge_service, "_current_authority", fake_authority)
    monkeypatch.setattr(bridge_service, "_context_packet", fake_packet)
    request = ContinuityHandoffInput(
        space_id="shared", workstream_id="bootstrap",
        authority_keys=["state"], query="continuity",
        predecessors=[{"id": "previous-agent", "provider": "local", "model": "synthetic"}],
    )
    first = await bridge_service._continuity_handoff(None, "reader", request, {"shared": object()})
    second = await bridge_service._continuity_handoff(None, "reader", request, {"shared": object()})
    assert first == second
    assert first["succession_envelope"]["continuity_principle"].startswith("Continuity of history")
    assert first["context_packet"]["integrity"]["packet_digest"]


@pytest.mark.asyncio
async def test_handoff_denies_unpermitted_space(monkeypatch):
    request = ContinuityHandoffInput(space_id="hidden", workstream_id="bootstrap", authority_keys=["state"], query="x")

    async def unexpected(*args, **kwargs):
        raise AssertionError("hidden scope must not reach authority or retrieval")

    monkeypatch.setattr(bridge_service, "_current_authority", unexpected)
    with pytest.raises(bridge_service.BridgeError) as exc:
        await bridge_service._continuity_handoff(None, "reader", request, {"shared": object()})
    assert exc.value.status == 403
    assert exc.value.code == "permission_denied"
@pytest_asyncio.fixture(autouse=True)
def local_bridge_host(client):
    client.base_url = "http://localhost"


@pytest_asyncio.fixture
async def identity(db_session):
    token = "synthetic-continuity-token-" + "x" * 32
    db_session.add(BridgeClient(id="handoff-reader", credential_digest=hashlib.sha256(token.encode()).hexdigest()))
    db_session.add(BridgeProject(id="shared", kind="shared"))
    await db_session.flush()
    db_session.add(BridgeGrant(client_id="handoff-reader", project_id="shared", can_read=True, can_write=False))
    await db_session.commit()
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_rest_and_mcp_empty_authority_handoff_are_equivalent(client, identity):
    payload = {
        "space_id": "shared", "workstream_id": "bootstrap",
        "authority_keys": ["missing"], "query": "synthetic continuity",
        "current_agent": {"provider": "local", "model": "synthetic"},
    }
    response = await client.post("/bridge/v1/build_continuity_handoff", json=payload, headers=identity)
    assert response.status_code == 200, response.text
    body = response.json()
    ctx = SimpleNamespace(request_context=SimpleNamespace(request=SimpleNamespace(
        headers={"authorization": identity["Authorization"]},
    )))
    mcp_body = await bridge_module.build_continuity_handoff(
        bridge_module.ContinuityHandoffInput(**payload), ctx,
    )
    assert mcp_body == body
    assert body["authority_results"][0]["status"] == "empty"
    assert body["included_memory_ids"] == []
    assert "authority_empty:missing" in body["warnings"]
    assert body["succession_envelope"]["current_agent"]["inherited_evidence_is_not_personal_memory"] is True