"""Mock-only certification for the Phase 2C local Ollama execution boundary."""
from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.domain.model_context.contracts import ProviderChatRequest, ProviderMessage
from app.domain.model_context.execution import (
    OllamaExecutionTimeouts,
    execute_local_ollama_provider_request,
)
from app.domain.model_context.ollama import OllamaContinuityOptions


def settings(base_url: str = "http://127.0.0.1:11434") -> SimpleNamespace:
    return SimpleNamespace(
        ollama_base_url=base_url,
        ollama_model="qwen3:4b",
        ollama_api_key="",
    )


def provider_request(model: str = "qwen3:4b") -> ProviderChatRequest:
    return ProviderChatRequest(
        model=model,
        messages=[
            ProviderMessage(role="system", content="Certified authority rules."),
            ProviderMessage(role="user", content="Certified continuity context."),
        ],
    )


def json_response(
    request: httpx.Request,
    *,
    content: object = "Final answer.",
    thinking: object | None = None,
    status: int = 200,
) -> httpx.Response:
    message: dict[str, object] = {"content": content}
    if thinking is not None:
        message["thinking"] = thinking
    return httpx.Response(status, json={"message": message}, request=request)


@pytest.mark.asyncio
async def test_certified_request_is_preserved_with_bounded_native_options() -> None:
    captured: list[tuple[httpx.Request, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append((request, json.loads(request.content)))
        return json_response(request)

    original = provider_request()
    result = await execute_local_ollama_provider_request(
        settings=settings(),
        request=original,
        approved_model="qwen3:4b",
        transport=httpx.MockTransport(handler),
    )

    assert len(captured) == 1
    sent_request, payload = captured[0]
    assert sent_request.url == "http://127.0.0.1:11434/api/chat"
    assert payload["model"] == "qwen3:4b"
    assert payload["messages"] == [message.model_dump() for message in original.messages]
    assert [message["role"] for message in payload["messages"]] == ["system", "user"]
    assert "tools" not in payload
    assert "functions" not in payload
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["keep_alive"] == "0s"
    assert payload["options"] == {
        "temperature": 0,
        "seed": 20260915,
        "num_ctx": 4096,
        "num_predict": 256,
    }
    assert result.provider == "ollama"
    assert result.model == original.model
    assert result.request == original
    assert result.content == "Final answer."
    assert result.reasoning_present is False
    assert result.http_status == 200
    assert result.elapsed_ms >= 0
    assert result.persisted is False
    assert result.inherited_context_was_attributed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    [
        "https://remote.example.com:11434",
        "ftp://127.0.0.1:11434",
        "http://user:pass@127.0.0.1:11434",
        "http://127.0.0.1:11434/redirect",
    ],
)
async def test_nonlocal_or_unexpected_endpoint_is_rejected_before_transport(
    base_url: str,
) -> None:
    transport = AsyncMock()
    with pytest.raises(PermissionError, match="local"):
        await execute_local_ollama_provider_request(
            settings=settings(base_url),
            request=provider_request(),
            approved_model="qwen3:4b",
            transport=transport,
        )
    transport.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["", "   ", "llama3.2"])
async def test_blank_or_mismatched_model_is_rejected_before_transport(model: str) -> None:
    request = provider_request(model)
    transport = AsyncMock()
    with pytest.raises(ValueError, match="model"):
        await execute_local_ollama_provider_request(
            settings=settings(),
            request=request,
            approved_model="qwen3:4b",
            transport=transport,
        )
    transport.assert_not_called()


@pytest.mark.asyncio
async def test_blank_approved_model_is_rejected_before_transport() -> None:
    transport = AsyncMock()
    with pytest.raises(ValueError, match="approved model"):
        await execute_local_ollama_provider_request(
            settings=settings(),
            request=provider_request(),
            approved_model=" ",
            transport=transport,
        )
    transport.assert_not_called()


@pytest.mark.asyncio
async def test_client_disables_proxy_inheritance_and_redirects(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, *, json, headers=None):
            request = httpx.Request("POST", url)
            return json_response(request)

    def client_factory(**kwargs):
        captured_kwargs.update(kwargs)
        return FakeClient()

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:9999")
    monkeypatch.setattr("app.domain.model_context.execution.httpx.AsyncClient", client_factory)

    await execute_local_ollama_provider_request(
        settings=settings(),
        request=provider_request(),
        approved_model="qwen3:4b",
    )

    assert captured_kwargs["trust_env"] is False
    assert captured_kwargs["follow_redirects"] is False
    timeout = captured_kwargs["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 5.0
    assert timeout.read == 60.0
    assert timeout.write == 10.0
    assert timeout.pool == 5.0


@pytest.mark.asyncio
async def test_redirect_and_non_2xx_fail_once_without_retry() -> None:
    for status in (302, 503):
        calls = 0

        def handler(request: httpx.Request, status_code: int = status) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                status_code,
                headers={"location": "https://remote.example.com/api/chat"},
                request=request,
            )

        with pytest.raises(httpx.HTTPStatusError):
            await execute_local_ollama_provider_request(
                settings=settings(),
                request=provider_request(),
                approved_model="qwen3:4b",
                transport=httpx.MockTransport(handler),
            )
        assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("refused"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.ReadTimeout("read timed out"),
    ],
)
async def test_transport_failures_are_deterministic_and_not_retried(
    failure: Exception,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise failure

    with pytest.raises(type(failure)):
        await execute_local_ollama_provider_request(
            settings=settings(),
            request=provider_request(),
            approved_model="qwen3:4b",
            transport=httpx.MockTransport(handler),
        )
    assert calls == 1


@pytest.mark.asyncio
async def test_overall_timeout_is_bounded_and_not_retried() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return json_response(request)

    with pytest.raises(TimeoutError):
        await execute_local_ollama_provider_request(
            settings=settings(),
            request=provider_request(),
            approved_model="qwen3:4b",
            transport=httpx.MockTransport(handler),
            timeouts=OllamaExecutionTimeouts(overall_seconds=0.01),
        )
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "thinking", "expected", "reasoning_present"),
    [
        ("Final answer.", None, "Final answer.", False),
        ("<think>private trace</think>\nFinal answer.", None, "Final answer.", True),
        ("prefilled reasoning\n</think>\nFinal answer.", None, "Final answer.", True),
        ("Final answer.", "private trace", "Final answer.", True),
    ],
)
async def test_final_answer_and_reasoning_are_separated(
    content: str,
    thinking: str | None,
    expected: str,
    reasoning_present: bool,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(request, content=content, thinking=thinking)

    result = await execute_local_ollama_provider_request(
        settings=settings(),
        request=provider_request(),
        approved_model="qwen3:4b",
        transport=httpx.MockTransport(handler),
    )

    assert result.content == expected
    assert result.reasoning_present is reasoning_present
    assert "private trace" not in result.content


@pytest.mark.asyncio
async def test_alternate_reasoning_field_is_diagnostic_only() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"content": "Final answer.", "reasoning": "private trace"}},
            request=request,
        )

    result = await execute_local_ollama_provider_request(
        settings=settings(),
        request=provider_request(),
        approved_model="qwen3:4b",
        transport=httpx.MockTransport(handler),
    )

    assert result.content == "Final answer."
    assert result.reasoning_present is True
    assert "private trace" not in result.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    ["", "   ", "<think>reasoning only</think>", "<think>unfinished reasoning"],
)
async def test_reasoning_only_or_missing_final_answer_is_rejected(content: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(request, content=content, thinking="private trace")

    with pytest.raises(ValueError, match="final answer|thinking block"):
        await execute_local_ollama_provider_request(
            settings=settings(),
            request=provider_request(),
            approved_model="qwen3:4b",
            transport=httpx.MockTransport(handler),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{}, {"message": {}}, {"message": {"content": 42}}])
async def test_malformed_ollama_schema_is_rejected(payload: dict[str, object]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    with pytest.raises(TypeError, match="message content"):
        await execute_local_ollama_provider_request(
            settings=settings(),
            request=provider_request(),
            approved_model="qwen3:4b",
            transport=httpx.MockTransport(handler),
        )


@pytest.mark.asyncio
async def test_malformed_json_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json", request=request)

    with pytest.raises(json.JSONDecodeError):
        await execute_local_ollama_provider_request(
            settings=settings(),
            request=provider_request(),
            approved_model="qwen3:4b",
            transport=httpx.MockTransport(handler),
        )


@pytest.mark.asyncio
async def test_malformed_reasoning_metadata_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"content": "Final answer.", "thinking": {"private": True}}},
            request=request,
        )

    with pytest.raises(ValueError, match="thinking metadata"):
        await execute_local_ollama_provider_request(
            settings=settings(),
            request=provider_request(),
            approved_model="qwen3:4b",
            transport=httpx.MockTransport(handler),
        )


def test_execution_boundary_has_no_db_tool_memory_or_authority_dependencies() -> None:
    from app.domain.model_context import execution

    source = inspect.getsource(execution)
    forbidden_dependencies = (
        "app.infrastructure.db",
        "app.domain.ingest",
        "app.domain.authority",
        "app.domain.bridge",
        "app.mcp_server",
    )
    assert not any(dependency in source for dependency in forbidden_dependencies)


def test_options_and_timeouts_reject_unbounded_values() -> None:
    with pytest.raises(ValueError):
        OllamaContinuityOptions(num_ctx=0)
    with pytest.raises(ValueError):
        OllamaContinuityOptions(num_predict=0)
    with pytest.raises(ValueError):
        OllamaExecutionTimeouts(connect_seconds=0)
    with pytest.raises(ValueError):
        OllamaExecutionTimeouts(overall_seconds=0)
