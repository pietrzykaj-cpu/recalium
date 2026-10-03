"""Opt-in, non-persistent execution for succession-aware local conversations."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Literal, Protocol

import httpx

from app.domain.agent_succession.contracts import RenderedSuccessionContext
from app.domain.model_context.contracts import ModelContextModel, ProviderChatRequest
from app.domain.model_context.ollama import (
    OllamaCompletionDiagnostics,
    OllamaHttpClient,
    build_ollama_succession_request_for_settings,
    ollama_chat_payload,
    parse_ollama_chat_response,
    submit_ollama_succession_request,
)
from app.domain.model_context.policy import is_local_inference_endpoint_url
from app.domain.model_context.profiles import LocalModelRole, resolve_local_model_profile
from app.infrastructure.local_inference import get_local_inference_coordinator


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
    ollama_diagnostics: OllamaCompletionDiagnostics | None = None
    inherited_context_was_attributed: Literal[True] = True
    persisted: Literal[False] = False


@dataclass(frozen=True)
class OllamaExecutionTimeouts:
    """Fixed transport and watchdog limits for a one-shot local request."""

    connect_seconds: float = 5.0
    read_seconds: float = 120.0
    write_seconds: float = 10.0
    pool_seconds: float = 5.0
    overall_seconds: float = 135.0

    def __post_init__(self) -> None:
        for name, value, certified_value in (
            ("connect_seconds", self.connect_seconds, 5.0),
            ("read_seconds", self.read_seconds, 120.0),
            ("write_seconds", self.write_seconds, 10.0),
            ("pool_seconds", self.pool_seconds, 5.0),
            ("overall_seconds", self.overall_seconds, 135.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or float(value) != certified_value
            ):
                raise ValueError(
                    f"Ollama {name} must remain at the certified value "
                    f"{certified_value:g}"
                )


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
    role: LocalModelRole,
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

    profile = resolve_local_model_profile(role)
    if not request.model.strip():
        raise ValueError("The provider request model must not be blank")
    if request.model != profile.model:
        raise ValueError("The provider request model does not match the approved profile")

    timeout = httpx.Timeout(
        connect=profile.connect_seconds,
        read=profile.read_seconds,
        write=profile.write_seconds,
        pool=profile.pool_seconds,
    )
    started = perf_counter()
    headers = _ollama_headers(settings)
    coordinator = get_local_inference_coordinator()
    async with coordinator.acquire(
        base_url=base_url,
        expected_model=profile.model,
        expected_digest=profile.model_digest,
        headers=headers,
    ):
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        ) as client:
            async with asyncio.timeout(profile.overall_seconds):
                response = await client.post(
                    f"{base_url.rstrip('/')}/api/chat",
                    json=ollama_chat_payload(request, profile=profile),
                    headers=headers,
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
        ollama_diagnostics=parsed.diagnostics,
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
