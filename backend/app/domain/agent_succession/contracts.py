"""Provider-neutral, non-persistent contracts for successor-agent handoff."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.context_packets.contracts import ContextPacket
from app.domain.retrieval.diagnostics import RetrievalDiagnostics


class SuccessionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Predecessor(SuccessionModel):
    id: str
    provider: str | None = None
    model: str | None = None
    version: str | None = None
    interaction_ids: list[str] = Field(default_factory=list)


class AttributedRecord(SuccessionModel):
    id: str
    content: str
    predecessor_ids: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    uncertainty: str | None = None
    conflict: bool = False


class SuccessorAnnotation(SuccessionModel):
    kind: Literal["disagreement", "reinterpretation", "limitation"]
    target_predecessor_id: str | None = None
    target_record_id: str | None = None
    rationale: str


class CurrentAgent(SuccessionModel):
    provider: str | None = None
    model: str | None = None
    version: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    session_id: str | None = None
    session_timestamp: datetime | None = None
    annotations: list[SuccessorAnnotation] = Field(default_factory=list)
    may_disagree_or_reinterpret: Literal[True] = True
    inherited_evidence_is_not_personal_memory: Literal[True] = True


class InheritedLayer(SuccessionModel):
    context_packet: ContextPacket
    predecessors: list[Predecessor] = Field(default_factory=list)
    interaction_history: list[AttributedRecord] = Field(default_factory=list)
    decisions: list[AttributedRecord] = Field(default_factory=list)
    relationship_context: list[AttributedRecord] = Field(default_factory=list)
    retrieval_diagnostics: RetrievalDiagnostics | None = None


class AgentSuccessionEnvelope(SuccessionModel):
    schema_version: Literal["recalium.agent-succession-envelope.v1"] = "recalium.agent-succession-envelope.v1"
    generated_at: datetime
    continuity_principle: Literal["Continuity of history is allowed. Continuity of identity must not be faked."] = "Continuity of history is allowed. Continuity of identity must not be faked."
    inherited: InheritedLayer
    current_agent: CurrentAgent


class RenderedSuccessionContext(SuccessionModel):
    text: str
    max_chars: int = Field(ge=1)
    truncated: bool
    included_memory_ids: list[str] = Field(default_factory=list)
    omitted_memory_count: int = Field(ge=0)
