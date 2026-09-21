"""Pure and mocked transport tests for the Ollama succession-context adapter."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.domain.agent_succession.contracts import CurrentAgent, Predecessor
from app.domain.agent_succession.service import build_agent_succession_envelope, render_agent_succession_context
from app.domain.context_packets.service import build_context_packet
from app.domain.model_context.ollama import (
    build_ollama_succession_request, build_ollama_succession_request_for_settings,
    submit_ollama_succession_request,
)
from app.domain.retrieval.service import RetrievalItem, RetrievalResponse


NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


def rendered(*, predecessors=(), content="A prior decision.", empty=False, decisions=None):
    items = [] if empty else [RetrievalItem(
        id="memory-1", type="fact", content=content, score=.9, source_id="archive-1",
        source_system="synthetic", captured_at="2026-09-14T00:00:00Z", conflict_label=None,
        provenance={},
    )]
    packet = build_context_packet(RetrievalResponse(
        query="decision", retrieval_mode="hybrid", budget_used=len(content), budget_limit=2000,
        trimming_reason="result_exhausted", items=items,
    ), generated_at=NOW)
    envelope = build_agent_succession_envelope(
        packet,
        current_agent=CurrentAgent(provider="ollama", model="qwen3:4b", version="local"),
        predecessors=list(predecessors),
        decisions=decisions,
        generated_at=NOW,
    )
    return render_agent_succession_context(envelope, max_chars=2000)


def test_deterministic_ollama_request_keeps_user_prompt_separate() -> None:
    context = rendered(predecessors=[Predecessor(id="p1", provider="openai", model="gpt")])
    first = build_ollama_succession_request(model="qwen3:4b", user_prompt="What should we do?", succession_context=context)
    second = build_ollama_succession_request(model="qwen3:4b", user_prompt="What should we do?", succession_context=context)
    assert first == second
    assert first.messages[0].role == "system"
    assert first.messages[1].role == "user"
    assert first.messages[1].content == "What should we do?"
    assert "INHERITED EVIDENCE" in first.messages[0].content


def test_missing_target_model_is_rejected_but_missing_predecessor_metadata_is_rendered() -> None:
    context = rendered(predecessors=[Predecessor(id="unknown")])
    request = build_ollama_succession_request(model="qwen3:4b", user_prompt="Continue.", succession_context=context)
    assert "metadata unknown" in request.messages[0].content
    with pytest.raises(ValueError, match="model"):
        build_ollama_succession_request(model=None, user_prompt="Continue.", succession_context=context)


def test_existing_ollama_settings_model_can_supply_target_without_reading_settings() -> None:
    settings = type("Settings", (), {"ollama_model": "qwen3:4b"})()
    request = build_ollama_succession_request_for_settings(
        settings, user_prompt="Continue.", succession_context=rendered(),
    )
    assert request.model == "qwen3:4b"


def test_empty_inheritance_and_multiple_predecessors_remain_attributed() -> None:
    empty = build_ollama_succession_request(model="qwen3:4b", user_prompt="Continue.", succession_context=rendered(empty=True))
    assert "No selected inherited evidence" in empty.messages[0].content
    context = rendered(predecessors=[Predecessor(id="p1"), Predecessor(id="p2")])
    request = build_ollama_succession_request(model="qwen3:4b", user_prompt="Continue.", succession_context=context)
    assert "p1 (metadata unknown)" in request.messages[0].content
    assert "p2 (metadata unknown)" in request.messages[0].content


def test_conflicting_inherited_conclusions_and_no_first_person_rewriting() -> None:
    context = rendered(
        content="I chose PostgreSQL, but this is disputed.",
        decisions=[
            {"id": "d1", "content": "Use SQLite.", "conflict": True},
            {"id": "d2", "content": "Use PostgreSQL.", "conflict": True},
        ],
    )
    request = build_ollama_succession_request(model="qwen3:4b", user_prompt="Assess it.", succession_context=context)
    system = request.messages[0].content
    assert "I remember" not in system
    assert "my prior experience" not in system
    assert "verbatim inherited evidence" in system
    assert "I chose PostgreSQL" in system
    assert "explicitly conflicting conclusions" in system


@pytest.mark.asyncio
async def test_mocked_ollama_adapter_smoke_uses_native_chat_payload() -> None:
    request = build_ollama_succession_request(model="qwen3:4b", user_prompt="Continue.", succession_context=rendered())
    response = type("Response", (), {"raise_for_status": lambda self: None, "json": lambda self: {"message": {"content": "ok"}}})()
    client = type("Client", (), {"post": AsyncMock(return_value=response)})()
    result = await submit_ollama_succession_request(client, "http://localhost:11434", request)
    assert result == {"message": {"content": "ok"}}
    assert client.post.call_args.args[0] == "http://localhost:11434/api/chat"
    assert client.post.call_args.kwargs["json"]["think"] is False
    assert client.post.call_args.kwargs["json"]["options"] == {"temperature": 0}
    assert client.post.call_args.kwargs["json"]["messages"][1]["content"] == "Continue."
