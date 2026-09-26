"""Versioned, bounded client inputs. No client-supplied authorization fields."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.agent_succession.contracts import CurrentAgent, Predecessor

ProjectId = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")]


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SourceMetadata(StrictInput):
    author_kind: Literal["human", "model", "mixed", "unknown"] = "unknown"
    model_label: str | None = Field(default=None, max_length=128)
    conversation_id: str | None = Field(default=None, max_length=256)
    source_name: str | None = Field(default=None, max_length=256)


class IngestInput(StrictInput):
    project_id: ProjectId | None = None
    destination: Literal["private", "shared"] | None = None
    space_id: ProjectId | None = None
    content: str = Field(min_length=10, max_length=100000)
    source_metadata: SourceMetadata
    idempotency_key: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def destination_contract(self):
        if (self.project_id is None) == (self.destination is None):
            raise ValueError("Choose destination or legacy project_id")
        if self.space_id is not None and self.destination != "shared":
            raise ValueError("space_id only narrows an explicit shared destination")
        return self


class RetrieveInput(StrictInput):
    project_id: ProjectId | None = None
    space_ids: list[ProjectId] | None = Field(default=None, max_length=100)
    query: str = Field(min_length=1, max_length=2000)
    mode: Literal["keyword", "semantic", "hybrid"] = "hybrid"
    budget: int = Field(default=2000, ge=1, le=20000)
    limit: int = Field(default=20, ge=1, le=50)

    @model_validator(mode="after")
    def scope_contract(self):
        if self.project_id is not None and self.space_ids is not None:
            raise ValueError("Choose space_ids or legacy project_id")
        return self


class ContextPacketInput(RetrieveInput):
    """Opt-in packet request; the inherited retrieval contract is unchanged."""

    token_budget: int = Field(default=500, ge=0, le=50000)
    current_provider: str | None = Field(default=None, max_length=128)
    current_model: str | None = Field(default=None, max_length=128)
    include_diagnostics: bool = False


class StatusInput(StrictInput):
    project_id: ProjectId | None = None
    archive_id: UUID


class CurrentAuthorityInput(StrictInput):
    """Read-only query for one persisted authority scope."""

    space_id: ProjectId
    workstream_id: str = Field(min_length=1, max_length=128)
    authority_key: str = Field(min_length=1, max_length=255)
    include_historical: bool = False
    include_provenance: bool = False
    include_competing: bool = True

class ContinuityHandoffInput(StrictInput):
    """Read-only authority + retrieval + succession handoff request."""
    space_id: ProjectId
    workstream_id: str = Field(min_length=1, max_length=128)
    authority_keys: list[str] = Field(min_length=1, max_length=32)
    query: str = Field(min_length=1, max_length=2000)
    mode: Literal["keyword", "semantic", "hybrid"] = "hybrid"
    budget: int = Field(default=2000, ge=1, le=20000)
    limit: int = Field(default=20, ge=1, le=50)
    token_budget: int = Field(default=500, ge=0, le=50000)
    render_max_chars: int = Field(default=8000, ge=512, le=50000)
    current_agent: CurrentAgent = Field(default_factory=CurrentAgent)
    predecessors: list[Predecessor] = Field(default_factory=list, max_length=32)
    include_historical: bool = False
    include_provenance: bool = False
    include_competing: bool = True
    include_diagnostics: bool = False

    @model_validator(mode="after")
    def normalize_authority_keys(self):
        keys = [key.strip() for key in self.authority_keys]
        if any(not key or len(key) > 255 for key in keys):
            raise ValueError("authority_keys must be non-empty and at most 255 characters")
        if len(set(keys)) != len(keys):
            raise ValueError("authority_keys must be unique")
        self.authority_keys = sorted(keys)
        return self
