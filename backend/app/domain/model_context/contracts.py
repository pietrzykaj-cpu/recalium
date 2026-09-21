"""Provider-neutral message and context-segment contracts."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.agent_succession.contracts import RenderedSuccessionContext


class ModelContextModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ContextSegment(ModelContextModel):
    kind: Literal["agent_succession_envelope"]
    content: str
    succession: RenderedSuccessionContext


class ProviderMessage(ModelContextModel):
    role: Literal["system", "user"]
    content: str


class ProviderChatRequest(ModelContextModel):
    model: str
    messages: list[ProviderMessage]
    context_segments: list[ContextSegment] = Field(default_factory=list)
