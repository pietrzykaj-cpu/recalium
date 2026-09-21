"""Bridge/API integration coverage for ContextPacket; excluded from the domain suite."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_bridge_builds_packet_after_enriched_retrieval(monkeypatch) -> None:
    from app.domain.bridge.contracts import ContextPacketInput
    from app.domain.bridge import service as bridge_service

    enriched = {
        "query": "continuity", "retrieval_mode": "hybrid", "budget_used": 8,
        "budget_limit": 2000, "trimming_reason": "result_exhausted", "degraded_mode": False,
        "items": [{
            "id": "fact-1", "type": "fact", "content": "evidence", "score": 0.5,
            "source_id": "archive-1", "source_system": "bridge",
            "captured_at": "2026-09-14T00:00:00+00:00", "conflict_label": None,
            "provenance": {"authenticated_client": "sol", "project_id": "sol-private"},
        }],
    }

    async def fake_retrieve(session, actor, request, spaces):
        assert actor == "sol"
        return enriched

    monkeypatch.setattr(bridge_service, "_retrieve", fake_retrieve)
    request = ContextPacketInput(
        query="continuity", token_budget=10, current_provider="ollama", current_model="qwen3:4b",
    )
    packet = await bridge_service._context_packet(None, "sol", request, {})
    assert "retrieval_diagnostics" not in packet
    assert packet["selected"][0]["provenance"]["authenticated_client"] == "sol"
    assert packet["model_context"] == {"provider": "ollama", "model": "qwen3:4b"}


@pytest.mark.asyncio
async def test_bridge_diagnostics_are_explicitly_opt_in(monkeypatch) -> None:
    from app.domain.bridge.contracts import ContextPacketInput
    from app.domain.bridge import service as bridge_service

    async def fake_retrieve(session, actor, request, spaces, diagnostics=None):
        assert diagnostics is not None
        return {
            "query": "continuity", "retrieval_mode": "hybrid", "budget_used": 0,
            "budget_limit": 2000, "trimming_reason": "result_exhausted",
            "degraded_mode": False, "items": [],
        }

    monkeypatch.setattr(bridge_service, "_retrieve", fake_retrieve)
    packet = await bridge_service._context_packet(
        None, "sol", ContextPacketInput(query="continuity", include_diagnostics=True), {"private"},
    )
    assert packet["retrieval_diagnostics"]["version"] == "retrieval-diagnostics-v1"
    assert packet["retrieval_diagnostics"]["memory_space_ids"] == ["private"]
    assert packet["retrieval_diagnostics"]["candidates"] == []
    assert packet["evidence_label"] == "evidence_from_prior_interactions"


@pytest.mark.asyncio
async def test_opt_in_rest_and_mcp_surfaces_are_registered() -> None:
    from app.api.bridge import bridge_app, bridge_mcp

    route_paths = {route.path for route in bridge_app.routes}
    assert "/v1/build_context_packet" in route_paths
    assert "/v1/retrieve_memory" in route_paths

    tool_names = {tool.name for tool in await bridge_mcp.list_tools()}
    assert tool_names == {
        "retrieve_memory", "build_context_packet", "ingest_memory", "get_ingest_status",
    }
