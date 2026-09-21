"""Pure and mocked tests for the opt-in Ollama succession execution seam."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.domain.agent_succession.contracts import CurrentAgent, Predecessor
from app.domain.agent_succession.service import build_agent_succession_envelope, render_agent_succession_context
from app.domain.context_packets.service import build_context_packet
from app.domain.model_context.execution import execute_ollama_succession_conversation
from app.domain.retrieval.service import RetrievalItem, RetrievalResponse


NOW = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)


def settings(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "ollama_base_url": "http://localhost:11434",
        "ollama_model": "qwen3:4b",
        "ollama_api_key": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def succession_context(*, empty: bool = False, predecessors=(), decisions=None):
    items = [] if empty else [RetrievalItem(
        id="memory-fox", type="fact", content="The silver fox sleeps beneath the orange tree.",
        score=.9, source_id="archive-synthetic", source_system="synthetic",
        captured_at="2026-09-15T00:00:00Z", conflict_label=None, provenance={"source": "test"},
    )]
    packet = build_context_packet(RetrievalResponse(
        query="fox", retrieval_mode="hybrid", budget_used=48, budget_limit=500,
        trimming_reason="result_exhausted", items=items,
    ), generated_at=NOW)
    envelope = build_agent_succession_envelope(
        packet, current_agent=CurrentAgent(provider="ollama", model="qwen3:4b"),
        predecessors=list(predecessors), decisions=decisions, generated_at=NOW,
    )
    return envelope, render_agent_succession_context(envelope, max_chars=2000)


def response(content: object = "The evidence says beneath the orange tree."):
    return type("Response", (), {
        "raise_for_status": lambda self: None,
        "json": lambda self: {"message": {"content": content}},
    })()


@pytest.mark.asyncio
async def test_execution_constructs_separate_attributed_request_and_returns_current_model_output() -> None:
    envelope, context = succession_context(predecessors=[Predecessor(id="p1", provider="openai", model="gpt")])
    client = type("Client", (), {"post": AsyncMock(return_value=response())})()
    result = await execute_ollama_succession_conversation(
        settings=settings(), user_prompt="Where does it sleep?", succession_context=context, client=client,
    )
    payload = client.post.call_args.kwargs["json"]
    assert result.provider == "ollama"
    assert result.model == "qwen3:4b"
    assert result.content == "The evidence says beneath the orange tree."
    assert result.persisted is False
    assert payload["messages"][0]["role"] == "system"
    assert "INHERITED EVIDENCE" in payload["messages"][0]["content"]
    assert payload["messages"][1] == {"role": "user", "content": "Where does it sleep?"}
    assert "Where does it sleep?" not in payload["messages"][0]["content"]
    assert envelope.current_agent.model == "qwen3:4b"


@pytest.mark.asyncio
async def test_config_validation_and_local_policy_are_explicit() -> None:
    _, context = succession_context()
    client = type("Client", (), {"post": AsyncMock()})()
    with pytest.raises(ValueError, match="base URL"):
        await execute_ollama_succession_conversation(
            settings=settings(ollama_base_url=""), user_prompt="x", succession_context=context, client=client,
        )
    with pytest.raises(ValueError, match="model"):
        await execute_ollama_succession_conversation(
            settings=settings(ollama_model=""), user_prompt="x", succession_context=context, client=client,
        )
    with pytest.raises(PermissionError, match="remote"):
        await execute_ollama_succession_conversation(
            settings=settings(ollama_base_url="https://remote.example.com"), user_prompt="x", succession_context=context, client=client,
        )
    assert client.post.await_count == 0


@pytest.mark.asyncio
async def test_http_error_and_connection_failure_propagate_without_persistence() -> None:
    _, context = succession_context()
    bad_response = type("Response", (), {
        "raise_for_status": lambda self: (_ for _ in ()).throw(httpx.HTTPStatusError("bad", request=None, response=None)),
        "json": lambda self: {},
    })()
    failing = type("Client", (), {"post": AsyncMock(return_value=bad_response)})()
    with pytest.raises(httpx.HTTPStatusError):
        await execute_ollama_succession_conversation(
            settings=settings(), user_prompt="x", succession_context=context, client=failing,
        )
    unavailable = type("Client", (), {"post": AsyncMock(side_effect=httpx.ConnectError("offline"))})()
    with pytest.raises(httpx.ConnectError):
        await execute_ollama_succession_conversation(
            settings=settings(), user_prompt="x", succession_context=context, client=unavailable,
        )
    timed_out = type("Client", (), {"post": AsyncMock(side_effect=httpx.ReadTimeout("timed out"))})()
    with pytest.raises(httpx.ReadTimeout):
        await execute_ollama_succession_conversation(
            settings=settings(), user_prompt="x", succession_context=context, client=timed_out,
        )


@pytest.mark.asyncio
async def test_malformed_response_and_empty_inheritance_are_handled_without_mutation() -> None:
    envelope, empty = succession_context(empty=True)
    before = envelope.model_dump(mode="json")
    malformed = type("Client", (), {"post": AsyncMock(return_value=response({"not": "text"}))})()
    with pytest.raises(ValueError, match="usable message"):
        await execute_ollama_succession_conversation(
            settings=settings(), user_prompt="Continue.", succession_context=empty, client=malformed,
        )
    assert envelope.model_dump(mode="json") == before


@pytest.mark.asyncio
async def test_multiple_conflicting_predecessors_stay_attributed_and_headers_are_optional() -> None:
    _, context = succession_context(
        predecessors=[Predecessor(id="p1"), Predecessor(id="p2")],
        decisions=[
            {"id": "d1", "content": "Use SQLite.", "conflict": True},
            {"id": "d2", "content": "Use PostgreSQL.", "conflict": True},
        ],
    )
    client = type("Client", (), {"post": AsyncMock(return_value=response("Current model assessment."))})()
    result = await execute_ollama_succession_conversation(
        settings=settings(ollama_api_key="synthetic-key"), user_prompt="Assess.", succession_context=context, client=client,
    )
    system = result.request.messages[0].content
    assert "p1 (metadata unknown)" in system
    assert "p2 (metadata unknown)" in system
    assert "explicitly conflicting conclusions" in system
    assert "I remember" not in system
    assert client.post.call_args.kwargs["headers"] == {"Authorization": "Bearer synthetic-key"}
