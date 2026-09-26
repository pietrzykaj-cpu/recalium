"""Provider-neutral, non-persistent authority contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AuthorityModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


LifecycleStatus = Literal["proposed", "active", "withdrawn", "disputed"]
AuthorityEdgeType = Literal["supersedes"]
AuthorityResultStatus = Literal["current", "empty", "ambiguous"]


class AuthorityRecord(AuthorityModel):
    """Explicit decision/state candidate; currentness is never stored here."""

    id: str = Field(min_length=1)
    space_id: str = Field(min_length=1)
    workstream_id: str = Field(min_length=1)
    authority_key: str = Field(min_length=1)
    record_kind: str = Field(min_length=1)
    content: str
    lifecycle_status: LifecycleStatus = "proposed"
    created_at: datetime
    created_by: str = Field(min_length=1)
    provenance: dict[str, Any] = Field(default_factory=dict)
    withdrawal_reason: str | None = None
    dispute_reason: str | None = None


class AuthorityEdge(AuthorityModel):
    """successor_record_id supersedes predecessor_record_id."""

    successor_record_id: str = Field(min_length=1)
    predecessor_record_id: str = Field(min_length=1)
    edge_type: AuthorityEdgeType = "supersedes"
    created_at: datetime
    created_by: str = Field(min_length=1)
    reason: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class AuthorityStateResult(AuthorityModel):
    """Deterministic current-state evaluation for one authority scope."""

    status: AuthorityResultStatus
    space_id: str
    workstream_id: str
    authority_key: str
    current_record: AuthorityRecord | None = None
    eligible_records: list[AuthorityRecord] = Field(default_factory=list)
    competing_records: list[AuthorityRecord] = Field(default_factory=list)
    historical_records: list[AuthorityRecord] = Field(default_factory=list)
    superseded_record_ids: list[str] = Field(default_factory=list)
