"""Opt-in, non-persistent execution for succession-aware local conversations."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Protocol

import httpx

from app.domain.agent_succession.contracts import RenderedSuccessionContext
from app.domain.model_context.contracts import ModelContextModel, ProviderChatRequest
from app.domain.model_context.ollama import (
    OllamaContinuityOptions,
    OllamaHttpClient,
    build_ollama_succession_request_for_settings,
    ollama_chat_payload,
    parse_ollama_chat_response,
    submit_ollama_succession_request,
)
from app.domain.model_context.policy import is_local_inference_endpoint_url


class OllamaExecutionSettingsLike(Protocol):
    """The existing configuration fields needed by this narrow execution seam."""

    ollama_base_url: str
    ollama_model: str
    ollama_api_key: str


class ModelExecutionResult(ModelContextModel):
    """An unpersisted response produced by the currently configured model."""

    provider: Literal["ollama"] = "ollama"
    model: str
    content: str
    request: ProviderChatRequest
    reasoning_present: bool = False
    http_status: int | None = None
    elapsed_ms: int | None = None
    inherited_context_was_attributed: Literal[True] = True
    persisted: Literal[False] = False


@dataclass(frozen=True)
class OllamaExecutionTimeouts:
    """Bounded transport and overall limits for a one-shot local request."""

    connect_seconds: float = 5.0
    read_seconds: float = 60.0
    write_seconds: float = 10.0
    pool_seconds: float = 5.0
    overall_seconds: float = 75.0

    def __post_init__(self) -> None:
        for name, value in (
            ("connect_seconds", self.connect_seconds),
            ("read_seconds", self.read_seconds),
            ("write_seconds", self.write_seconds),
            ("pool_seconds", self.pool_seconds),
            ("overall_seconds", self.overall_seconds),
        ):
            if value <= 0:
                raise ValueError(f"Ollama {name} must be positive")


def _ollama_headers(settings: OllamaExecutionSettingsLike) -> dict[str, str] | None:
    api_key = (getattr(settings, "ollama_api_key", "") or "").strip()
    return {"Authorization": f"Bearer {api_key}"} if api_key else None


def _configured_base_url(settings: OllamaExecutionSettingsLike) -> str:
    base_url = (getattr(settings, "ollama_base_url", "") or "").strip()
    if not base_url:
        raise ValueError("An Ollama base URL must be supplied by configuration")
    return base_url


async def execute_local_ollama_provider_request(
    *,
    settings: OllamaExecutionSettingsLike,
    request: ProviderChatRequest,
    approved_model: str,
    options: OllamaContinuityOptions | None = None,
    timeouts: OllamaExecutionTimeouts | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ModelExecutionResult:
    """Execute one certified provider request through a local-only Ollama boundary.

    The request is already the certified Phase 2B artifact. This function does
    not rebuild messages, grant tools, retry, persist, or invoke any Recalium
    service. The optional transport exists for socket-free certification.
    """
    base_url = _configured_base_url(settings)
    if not is_local_inference_endpoint_url(base_url):
        raise PermissionError("Local continuity execution forbids nonlocal Ollama endpoints")

    expected_model = (approved_model or "").strip()
    if not expected_model:
        raise ValueError("An approved model must be supplied for local continuity execution")
    if not request.model.strip():
        raise ValueError("The provider request model must not be blank")
    if request.model != expected_model:
        raise ValueError("The provider request model does not match the approved model")

    selected_options = options or OllamaContinuityOptions()
    selected_timeouts = timeouts or OllamaExecutionTimeouts()
    timeout = httpx.Timeout(
        connect=selected_timeouts.connect_seconds,
        read=selected_timeouts.read_seconds,
        write=selected_timeouts.write_seconds,
        pool=selected_timeouts.pool_seconds,
    )
    started = perf_counter()
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
        trust_env=False,
        transport=transport,
    ) as client:
        async with asyncio.timeout(selected_timeouts.overall_seconds):
            response = await client.post(
                f"{base_url.rstrip('/')}/api/chat",
                json=ollama_chat_payload(request, options=selected_options),
                headers=_ollama_headers(settings),
            )
            response.raise_for_status()
            payload = response.json()

    parsed = parse_ollama_chat_response(payload)
    elapsed_ms = max(0, round((perf_counter() - started) * 1000))
    return ModelExecutionResult(
        model=request.model,
        content=parsed.content,
        request=request,
        reasoning_present=parsed.reasoning_present,
        http_status=response.status_code,
        elapsed_ms=elapsed_ms,
    )


async def execute_ollama_succession_conversation(
    *,
    settings: OllamaExecutionSettingsLike,
    user_prompt: str,
    succession_context: RenderedSuccessionContext,
    client: OllamaHttpClient,
    allow_external: bool = False,
) -> ModelExecutionResult:
    """Execute an attributed succession context through Ollama's chat endpoint.

    The caller supplies configuration and a transport.  This domain seam owns no
    database session and does not record either the envelope or the response.
    """
    base_url = _configured_base_url(settings)
    if not allow_external and not is_local_inference_endpoint_url(base_url):
        raise PermissionError("Policy forbids remote Ollama processing")

    request = build_ollama_succession_request_for_settings(
        settings,
        user_prompt=user_prompt,
        succession_context=succession_context,
    )
    response = await submit_ollama_succession_request(
        client,
        base_url,
        request,
        headers=_ollama_headers(settings),
    )
    message = response.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama chat response has no usable message content")
    return ModelExecutionResult(model=request.model, content=content, request=request)
