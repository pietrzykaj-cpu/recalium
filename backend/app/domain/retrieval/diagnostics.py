"""Typed, non-persistent diagnostics for retrieval audit/debugging."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChannelObservation(BaseModel):
    score: float
    rank: int
    rrf_contribution: float


class RepresentativeDecision(BaseModel):
    memory_id: str
    memory_type: str
    reason: str


class MetadataMergeDiagnostic(BaseModel):
    merged_keys: list[str] = Field(default_factory=list)
    conflict_paths: list[str] = Field(default_factory=list)
    discarded: list[str] = Field(default_factory=list)
    unresolved_before: dict[str, list[Any]] = Field(default_factory=dict)
    unresolved_after: list[Any] = Field(default_factory=list)
    conflicts_before: dict[str, Any] = Field(default_factory=dict)
    conflict_after: Any = None


class ExclusionDiagnostic(BaseModel):
    stage: str
    reason: str
    observable: bool = True


class CandidateDiagnostic(BaseModel):
    fusion_key: str
    representative_id: str
    channels: list[str]
    lexical: ChannelObservation | None = None
    semantic: ChannelObservation | None = None
    fused_score: float | None = None
    final_rank: int | None = None
    representative: RepresentativeDecision
    metadata: MetadataMergeDiagnostic = Field(default_factory=MetadataMergeDiagnostic)
    duplicate_count: int = 1
    fused_memory_ids: list[str] = Field(default_factory=list)
    exclusion: ExclusionDiagnostic | None = None
    character_count: int | None = None


class UnknownExclusion(BaseModel):
    stage: str
    reason: str
    observable: Literal[False] = False
    memory_id: None = None


class RetrievalDiagnostics(BaseModel):
    version: Literal["retrieval-diagnostics-v1"] = "retrieval-diagnostics-v1"
    generated_at: datetime
    mode: str
    filters: dict[str, Any] = Field(default_factory=dict)
    memory_space_ids: list[str] = Field(default_factory=list)
    thresholds: dict[str, float | int] = Field(default_factory=dict)
    candidates: list[CandidateDiagnostic] = Field(default_factory=list)
    unknown_exclusions: list[UnknownExclusion] = Field(default_factory=list)


class RetrievalDiagnosticsCollector:
    """Mutable request-scoped collector; ``snapshot`` returns the typed sidecar."""

    def __init__(self, mode: str, filters: dict | None = None, memory_space_ids: list[str] | None = None):
        self.mode = mode
        self.filters = deepcopy(filters or {})
        self.memory_space_ids = list(memory_space_ids or [])
        self.thresholds: dict[str, float | int] = {}
        self._records: dict[str, CandidateDiagnostic] = {}
        self._representatives: dict[str, str] = {}
        self._unknown: list[UnknownExclusion] = []

    def record_fusion(self, key: str, candidates: list[tuple[str, int, dict, float]], representative: dict,
                      score: float, reason: str, metadata: MetadataMergeDiagnostic) -> None:
        channels: list[str] = []
        ids: list[str] = []
        lexical = semantic = None
        for channel, rank, candidate, contribution in candidates:
            if channel not in channels:
                channels.append(channel)
            if candidate["id"] not in ids:
                ids.append(candidate["id"])
            observation = ChannelObservation(score=float(candidate["score"]), rank=rank, rrf_contribution=contribution)
            if channel == "lexical": lexical = observation
            elif channel == "semantic": semantic = observation
        record = CandidateDiagnostic(
            fusion_key=key, representative_id=representative["id"], channels=channels,
            lexical=lexical, semantic=semantic, fused_score=score,
            representative=RepresentativeDecision(memory_id=representative["id"], memory_type=representative["type"], reason=reason),
            metadata=metadata, duplicate_count=len(candidates), fused_memory_ids=ids,
            character_count=len(representative.get("content", "")),
        )
        self._records[key] = record
        self._representatives[representative["id"]] = key

    def mark_rank(self, representative_id: str, rank: int) -> None:
        record = self.for_representative(representative_id)
        if record: record.final_rank = rank

    def exclude(self, representative_id: str, stage: str, reason: str) -> None:
        record = self.for_representative(representative_id)
        if record: record.exclusion = ExclusionDiagnostic(stage=stage, reason=reason)

    def record_unknown_exclusion(self, stage: str, reason: str) -> None:
        self._unknown.append(UnknownExclusion(stage=stage, reason=reason))

    def for_representative(self, memory_id: str) -> CandidateDiagnostic | None:
        key = self._representatives.get(memory_id)
        return self._records.get(key) if key else None

    def snapshot(self) -> RetrievalDiagnostics:
        return RetrievalDiagnostics(
            generated_at=datetime.now(timezone.utc), mode=self.mode, filters=self.filters,
            memory_space_ids=self.memory_space_ids, thresholds=self.thresholds,
            candidates=list(self._records.values()), unknown_exclusions=self._unknown,
        )
