"""Ollama request mapping and response parsing for local model-context execution."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
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


@dataclass(frozen=True)
class OllamaContinuityOptions:
    """Fixed deterministic options for the certified continuity executor."""

    temperature: float = 0
    seed: int = 20260915
    num_ctx: int = 4096
    num_predict: int = 256
    keep_alive: str = "0s"
    think: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not isfinite(float(self.temperature))
            or float(self.temperature) != 0.0
        ):
            raise ValueError("Ollama temperature must remain at the certified value 0")
        if type(self.seed) is not int or self.seed != 20260915:
            raise ValueError("Ollama seed must remain at the certified value 20260915")
        if type(self.num_ctx) is not int or self.num_ctx != 4096:
            raise ValueError("Ollama num_ctx must remain at the certified value 4096")
        if type(self.num_predict) is not int or self.num_predict != 256:
            raise ValueError("Ollama num_predict must remain at the certified value 256")
        if self.keep_alive != "0s":
            raise ValueError("Ollama keep_alive must remain at the certified value '0s'")
        if self.think is not False:
            raise ValueError("Ollama think must remain at the certified value false")


@dataclass(frozen=True)
class ParsedOllamaMessage:
    """Final assistant content plus non-content reasoning presence metadata."""

    content: str
    reasoning_present: bool


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


def ollama_chat_payload(
    request: ProviderChatRequest,
    *,
    options: OllamaContinuityOptions | None = None,
) -> dict[str, Any]:
    """Return a native `/api/chat` payload, with bounded Phase 2C options when supplied."""
    payload: dict[str, Any] = {
        "model": request.model,
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
        "messages": [message.model_dump() for message in request.messages],
    }
    if options is not None:
        payload.update(
            {
                "think": options.think,
                "keep_alive": options.keep_alive,
                "options": {
                    "temperature": options.temperature,
                    "seed": options.seed,
                    "num_ctx": options.num_ctx,
                    "num_predict": options.num_predict,
                },
            }
        )
    return payload


def ollama_final_content(raw: str) -> tuple[str, bool]:
    """Separate explicit thinking markup from a deterministic final answer.

    The behavior intentionally matches the hardened worker boundary. It does
    not guess from ordinary prose or strip protocol-token literals inside JSON.
    An orphan closing tag is treated as a Qwen template boundary only when it
    occupies its own line.
    """
    content = raw.strip()
    reasoning_present = False
    while content:
        if content.startswith("<think>"):
            end = content.find("</think>", len("<think>"))
            if end < 0:
                raise ValueError("Ollama returned an unfinished thinking block")
            reasoning_present = True
            content = content[end + len("</think>") :].strip()
            continue
        try:
            json.loads(content)
        except json.JSONDecodeError:
            closing = re.search(r"(?m)^[ \t]*</think>[ \t]*(?:\r?\n|$)", content)
            if closing:
                reasoning_present = True
                content = content[closing.end() :].strip()
                continue
        break
    if not content:
        raise ValueError("Ollama returned no final answer")
    return content, reasoning_present


def parse_ollama_chat_response(payload: Any) -> ParsedOllamaMessage:
    """Validate an Ollama chat response without exposing reasoning as content."""
    if not isinstance(payload, Mapping):
        raise TypeError("Ollama chat response is not an object")
    message = payload.get("message")
    if not isinstance(message, Mapping):
        raise TypeError("Ollama chat response has no usable message content")
    raw_content = message.get("content")
    if not isinstance(raw_content, str):
        raise TypeError("Ollama chat response has no usable message content")

    reasoning_present = False
    for field in ("thinking", "reasoning"):
        value = message.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Ollama chat response has invalid {field} metadata")
        reasoning_present = reasoning_present or bool(value and value.strip())

    content, embedded_reasoning = ollama_final_content(raw_content)
    return ParsedOllamaMessage(
        content=content,
        reasoning_present=reasoning_present or embedded_reasoning,
    )


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
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError("Ollama chat response is not an object")
    return payload
