"""Transport-neutral contracts for guarded authority proposals and mutations."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .contracts import AuthorityModel, AuthorityResultStatus

AuthorityMutationOperation = Literal[
    "activate_authority_record",
    "supersede_authority_record",
    "withdraw_authority_record",
    "mark_authority_disputed",
]
AuthorityWriteOperation = Literal[
    "create_authority_proposal",
    "activate_authority_record",
    "supersede_authority_record",
    "withdraw_authority_record",
    "mark_authority_disputed",
]
AuthorityWriteOutcome = Literal["applied", "preview", "idempotent_replay"]


class AuthorityScope(AuthorityModel):
    space_id: str = Field(min_length=1, max_length=128)
    workstream_id: str = Field(min_length=1, max_length=128)
    authority_key: str = Field(min_length=1, max_length=255)


class AuthorityActorContext(AuthorityModel):
    """Trusted caller context; never construct this from request JSON."""

    requesting_client_id: str = Field(min_length=1, max_length=64)
    authorizing_client_id: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def phase_one_has_no_delegation(self):
        if self.requesting_client_id != self.authorizing_client_id:
            raise ValueError("Authority delegation is not supported in Phase 1E-A")
        return self


class CreateAuthorityProposal(AuthorityModel):
    scope: AuthorityScope
    record_id: str = Field(min_length=1)
    record_kind: str = Field(min_length=1, max_length=64)
    content: str
    provenance: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = Field(min_length=1, max_length=128)


class GuardedAuthorityMutation(AuthorityModel):
    scope: AuthorityScope
    expected_state_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    idempotency_key: str = Field(min_length=1, max_length=128)
    confirm: bool = False
    dry_run: bool = False
    reason: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def confirmation_mode(self):
        if self.confirm and self.dry_run:
            raise ValueError("Choose confirm or dry_run, not both")
        return self


class ActivateAuthorityRecord(GuardedAuthorityMutation):
    record_id: str = Field(min_length=1)


class SupersedeAuthorityRecord(GuardedAuthorityMutation):
    successor_record_id: str = Field(min_length=1)
    predecessor_record_id: str = Field(min_length=1)


class WithdrawAuthorityRecord(GuardedAuthorityMutation):
    record_id: str = Field(min_length=1)


class MarkAuthorityDisputed(GuardedAuthorityMutation):
    record_id: str = Field(min_length=1)


class AuthorityStateSummary(AuthorityModel):
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: AuthorityResultStatus
    current_record_id: str | None = None
    eligible_record_ids: list[str] = Field(default_factory=list)
    competing_record_ids: list[str] = Field(default_factory=list)
    historical_record_ids: list[str] = Field(default_factory=list)
    superseded_record_ids: list[str] = Field(default_factory=list)


class AuthorityWriteResult(AuthorityModel):
    operation_id: str
    operation: AuthorityWriteOperation
    outcome: AuthorityWriteOutcome
    scope: AuthorityScope
    affected_record_ids: list[str] = Field(default_factory=list)
    affected_edge_ids: list[str] = Field(default_factory=list)
    before: AuthorityStateSummary
    after: AuthorityStateSummary
    warnings: list[str] = Field(default_factory=list)
    audit_event_id: str | None = None
    idempotent_replay: bool = False
