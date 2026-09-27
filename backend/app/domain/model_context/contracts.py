"""Provider-neutral message and continuity-consumption contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain.agent_succession.contracts import (
    CurrentAgent,
    Predecessor,
    RenderedSuccessionContext,
)


class ModelContextModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConsumptionAuthorityRecord(ModelContextModel):
    id: str
    record_kind: str
    content: str
    compact_provenance: tuple[str, ...] = ()


class ConsumptionAuthorityState(ModelContextModel):
    authority_key: str
    status: Literal["current", "empty", "ambiguous"]
    current_record: ConsumptionAuthorityRecord | None = None
    competing_records: tuple[ConsumptionAuthorityRecord, ...] = ()
    state_digest: str


class ConsumptionEvidence(ModelContextModel):
    memory_id: str
    memory_type: Literal["canonical", "fact", "summary", "excerpt"]
    content: str
    attribution_label: Literal["evidence_from_prior_interactions"]
    source_archive_id: str
    source_system: str
    captured_at: str
    conflict_label: str | None = None
    compact_provenance: tuple[str, ...] = ()


class FullProvenanceEntry(ModelContextModel):
    kind: Literal["authority", "memory"]
    record_id: str
    authority_key: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class RetrievalMetadataEntry(ModelContextModel):
    memory_id: str
    retrieval: dict[str, Any]


class AuthorityDigestEntry(ModelContextModel):
    authority_key: str
    state_digest: str


class ContinuityStructuredContext(ModelContextModel):
    authority_states: tuple[ConsumptionAuthorityState, ...]
    supporting_memory: tuple[ConsumptionEvidence, ...]
    unresolved_questions: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    critical_warnings: tuple[str, ...] = ()
    predecessors: tuple[Predecessor, ...] = ()
    current_agent: CurrentAgent
    attribution_notice: str
    continuity_principle: str


class ContinuityDiagnostics(ModelContextModel):
    authority_state_digests: tuple[AuthorityDigestEntry, ...]
    packet_digest: str
    source_excluded_memory_ids: tuple[str, ...] = ()
    consumption_omitted_memory_ids: tuple[str, ...] = ()
    full_provenance: tuple[FullProvenanceEntry, ...] = ()
    historical_authority_records: tuple[dict[str, Any], ...] = ()
    retrieval_metadata: tuple[RetrievalMetadataEntry, ...] = ()
    retrieval_diagnostics: dict[str, Any] | None = None
    source_budget: dict[str, Any]
    handoff_provenance: dict[str, Any] | None = None
    packet_integrity: dict[str, Any]


class ContinuityConsumptionBudget(ModelContextModel):
    max_model_chars: int = Field(ge=1)
    mandatory_chars: int = Field(ge=0)
    rendered_chars: int = Field(ge=0)
    supporting_memory_available: int = Field(ge=0)
    supporting_memory_included: int = Field(ge=0)
    supporting_memory_omitted: int = Field(ge=0)


class ContinuityCapabilities(ModelContextModel):
    """Operational permissions are deliberately absent from Phase 2A."""

    allowed_tools: tuple[str, ...] = ()
    authority_proposal_allowed: Literal[False] = False
    authority_mutation_allowed: Literal[False] = False
    permissions_are_not_implied: Literal[True] = True

    @field_validator("allowed_tools")
    @classmethod
    def reject_tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value:
            raise ValueError("Phase 2A continuity consumption cannot grant tools")
        return value


class ConsumptionIntegrity(ModelContextModel):
    algorithm: Literal["sha256"] = "sha256"
    consumption_digest: str


class ContinuityConsumptionPayload(ModelContextModel):
    schema_version: Literal["recalium.continuity-consumption.v1"] = (
        "recalium.continuity-consumption.v1"
    )
    system_instructions: tuple[str, ...]
    continuity: ContinuityStructuredContext
    rendered_context: str
    diagnostics: ContinuityDiagnostics
    budget: ContinuityConsumptionBudget
    capabilities: ContinuityCapabilities = Field(default_factory=ContinuityCapabilities)
    integrity: ConsumptionIntegrity


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
