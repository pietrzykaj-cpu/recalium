"""Ollama mapping for a rendered AgentSuccessionEnvelope context segment."""
from __future__ import annotations

from typing import Any, Protocol

from app.domain.agent_succession.contracts import RenderedSuccessionContext
from app.domain.model_context.contracts import ContextSegment, ProviderChatRequest, ProviderMessage


class OllamaHttpClient(Protocol):
    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> Any: ...


class OllamaSettingsLike(Protocol):
    ollama_model: str


def build_ollama_succession_request(
    *,
    model: str | None,
    user_prompt: str,
    succession_context: RenderedSuccessionContext,
) -> ProviderChatRequest:
    """Map a succession context to Ollama's native chat message shape.

    The rendered inheritance stays in a separately tracked system segment; the
    caller's prompt is retained as an independent user message.
    """
    selected_model = (model or "").strip()
    if not selected_model:
        raise ValueError("An Ollama model must be supplied by the caller/configuration")
    segment = ContextSegment(
        kind="agent_succession_envelope",
        content=succession_context.text,
        succession=succession_context,
    )
    return ProviderChatRequest(
        model=selected_model,
        context_segments=[segment],
        messages=[
            ProviderMessage(role="system", content=segment.content),
            ProviderMessage(role="user", content=user_prompt),
        ],
    )


def build_ollama_succession_request_for_settings(
    settings: OllamaSettingsLike,
    *,
    user_prompt: str,
    succession_context: RenderedSuccessionContext,
) -> ProviderChatRequest:
    """Use the existing configured local Ollama model without reading settings here."""
    return build_ollama_succession_request(
        model=settings.ollama_model,
        user_prompt=user_prompt,
        succession_context=succession_context,
    )


def ollama_chat_payload(request: ProviderChatRequest) -> dict[str, Any]:
    """Return the native `/api/chat` payload used by the existing local policy."""
    return {
        "model": request.model,
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
        "messages": [message.model_dump() for message in request.messages],
    }


async def submit_ollama_succession_request(
    client: OllamaHttpClient,
    base_url: str,
    request: ProviderChatRequest,
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Submit through an injected client; callers own settings, policy, and transport."""
    response = await client.post(
        f"{base_url.rstrip('/')}/api/chat",
        json=ollama_chat_payload(request),
        headers=headers,
    )
    response.raise_for_status()
    return response.json()
