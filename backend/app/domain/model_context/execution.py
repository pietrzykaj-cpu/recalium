"""Opt-in, non-persistent execution for succession-aware local conversations."""
from __future__ import annotations

from typing import Protocol

from app.domain.agent_succession.contracts import RenderedSuccessionContext
from app.domain.model_context.contracts import ModelContextModel, ProviderChatRequest
from app.domain.model_context.ollama import (
    OllamaHttpClient,
    build_ollama_succession_request_for_settings,
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

    provider: str = "ollama"
    model: str
    content: str
    request: ProviderChatRequest
    inherited_context_was_attributed: bool = True
    persisted: bool = False


def _ollama_headers(settings: OllamaExecutionSettingsLike) -> dict[str, str] | None:
    api_key = (getattr(settings, "ollama_api_key", "") or "").strip()
    return {"Authorization": f"Bearer {api_key}"} if api_key else None


def _configured_base_url(settings: OllamaExecutionSettingsLike) -> str:
    base_url = (getattr(settings, "ollama_base_url", "") or "").strip()
    if not base_url:
        raise ValueError("An Ollama base URL must be supplied by configuration")
    return base_url


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
