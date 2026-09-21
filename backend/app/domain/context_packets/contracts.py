"""Provider-neutral ContextPacket v1 contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PacketModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PacketQuery(PacketModel):
    text: str
    mode_used: str


class PacketSource(PacketModel):
    archive_id: str
    system: str
    captured_at: str


class PacketRetrieval(PacketModel):
    final_score: float
    relevance_rank: int = Field(ge=1)
    lexical_score: float | None = None
    lexical_rank: int | None = None
    semantic_score: float | None = None
    semantic_rank: int | None = None
    hybrid_score: float | None = None
    rrf_score: float | None = None
    link_type: str | None = None


class SelectedEvidence(PacketModel):
    memory_id: str
    memory_type: Literal["canonical", "fact", "summary", "excerpt"]
    content: str
    evidence_class: Literal["evidence_from_prior_interactions"] = "evidence_from_prior_interactions"
    source: PacketSource
    provenance: dict[str, Any]
    retrieval: PacketRetrieval
    conflict_label: str | None = None
    source_fact_id: str | None = None


class ExcludedEvidence(PacketModel):
    memory_id: str
    memory_type: str
    relevance_rank: int = Field(ge=1)
    reason: Literal["context_budget_exceeded"]
    estimated_tokens: int = Field(ge=0)


class PacketBudget(PacketModel):
    unit: Literal["characters", "estimated_tokens"]
    limit: int = Field(ge=0)
    used: int = Field(ge=0)
    estimator: str
    upstream_unit: Literal["characters"] = "characters"
    upstream_limit: int = Field(ge=0)
    upstream_used: int = Field(ge=0)
    upstream_trimming_reason: str


class ModelContext(PacketModel):
    provider: str | None = None
    model: str | None = None


class PacketIntegrity(PacketModel):
    algorithm: Literal["sha256"] = "sha256"
    packet_digest: str


class ContextPacket(PacketModel):
    schema_version: Literal["recalium.context-packet.v1"] = "recalium.context-packet.v1"
    packet_id: str
    generated_at: datetime
    evidence_label: Literal["evidence_from_prior_interactions"] = "evidence_from_prior_interactions"
    attribution_notice: str
    query: PacketQuery
    selected: list[SelectedEvidence]
    excluded: list[ExcludedEvidence]
    unresolved_questions: list[str]
    flags: list[str]
    warnings: list[str]
    budget: PacketBudget
    model_context: ModelContext | None = None
    integrity: PacketIntegrity
