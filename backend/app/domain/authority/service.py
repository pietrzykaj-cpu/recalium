"""Pure in-memory authority graph operations and currentness evaluation."""

from __future__ import annotations

from datetime import datetime

from .contracts import AuthorityEdge, AuthorityRecord, AuthorityStateResult


class AuthorityValidationError(ValueError):
    """Raised when an authority graph invariant is violated."""


def _scope(record: AuthorityRecord) -> tuple[str, str, str]:
    return record.space_id, record.workstream_id, record.authority_key


def _would_cycle(successor_id: str, predecessor_id: str, edges: list[AuthorityEdge]) -> bool:
    predecessors: dict[str, set[str]] = {}
    for edge in edges:
        predecessors.setdefault(edge.successor_record_id, set()).add(edge.predecessor_record_id)
    pending = [predecessor_id]
    visited: set[str] = set()
    while pending:
        node = pending.pop()
        if node == successor_id:
            return True
        if node in visited:
            continue
        visited.add(node)
        pending.extend(predecessors.get(node, ()))
    return False


def validate_supersession_edge(
    successor: AuthorityRecord,
    predecessor: AuthorityRecord,
    edges: list[AuthorityEdge] | tuple[AuthorityEdge, ...] = (),
) -> None:
    if successor.id == predecessor.id:
        raise AuthorityValidationError("A record cannot supersede itself")
    if _scope(successor) != _scope(predecessor):
        raise AuthorityValidationError("Supersession must remain in one authority scope")
    if any(
        edge.successor_record_id == successor.id
        and edge.predecessor_record_id == predecessor.id
        for edge in edges
    ):
        raise AuthorityValidationError("Duplicate supersession edge")
    if _would_cycle(successor.id, predecessor.id, list(edges)):
        raise AuthorityValidationError("Supersession cycle is not allowed")


def evaluate_current_state(
    records: list[AuthorityRecord] | tuple[AuthorityRecord, ...],
    edges: list[AuthorityEdge] | tuple[AuthorityEdge, ...],
    *,
    space_id: str,
    workstream_id: str,
    authority_key: str,
) -> AuthorityStateResult:
    """Derive currentness without timestamps, ranking, or insertion order."""
    scoped = sorted(
        (r for r in records if _scope(r) == (space_id, workstream_id, authority_key)),
        key=lambda r: r.id,
    )
    ids = {r.id for r in scoped}
    superseded = sorted({
        e.predecessor_record_id
        for e in edges
        if e.predecessor_record_id in ids and e.successor_record_id in ids
    })
    blocked = set(superseded)
    eligible = [r for r in scoped if r.lifecycle_status == "active" and r.id not in blocked]
    if len(eligible) == 1:
        status, current, competing = "current", eligible[0], []
    elif not eligible:
        status, current, competing = "empty", None, []
    else:
        status, current, competing = "ambiguous", None, list(eligible)
    return AuthorityStateResult(
        status=status,
        space_id=space_id,
        workstream_id=workstream_id,
        authority_key=authority_key,
        current_record=current,
        eligible_records=eligible,
        competing_records=competing,
        historical_records=[r for r in scoped if r.id in blocked],
        superseded_record_ids=superseded,
    )


class AuthorityGraph:
    """Small mutable in-memory aggregate for pure tests and future adapters."""

    def __init__(self) -> None:
        self._records: dict[str, AuthorityRecord] = {}
        self._edges: list[AuthorityEdge] = []

    @property
    def records(self) -> tuple[AuthorityRecord, ...]:
        return tuple(self._records.values())

    @property
    def edges(self) -> tuple[AuthorityEdge, ...]:
        return tuple(self._edges)

    def create_record(self, record: AuthorityRecord) -> AuthorityRecord:
        if record.id in self._records:
            raise AuthorityValidationError(f"Record already exists: {record.id}")
        self._records[record.id] = record
        return record

    def _get(self, record_id: str) -> AuthorityRecord:
        try:
            return self._records[record_id]
        except KeyError as exc:
            raise AuthorityValidationError(f"Unknown authority record: {record_id}") from exc

    def activate_record(self, record_id: str) -> AuthorityRecord:
        record = self._get(record_id)
        if record.lifecycle_status in {"withdrawn", "disputed"}:
            raise AuthorityValidationError("Withdrawn or disputed records cannot be activated")
        updated = record.model_copy(update={"lifecycle_status": "active"})
        self._records[record_id] = updated
        return updated

    def supersede(
        self,
        successor_id: str,
        predecessor_id: str,
        *,
        created_at: datetime,
        created_by: str,
        reason: str | None = None,
        provenance: dict | None = None,
    ) -> AuthorityEdge:
        successor = self._get(successor_id)
        predecessor = self._get(predecessor_id)
        validate_supersession_edge(successor, predecessor, self._edges)
        if successor.lifecycle_status in {"withdrawn", "disputed"}:
            raise AuthorityValidationError("Successor must be proposed or active")
        edge = AuthorityEdge(
            successor_record_id=successor_id,
            predecessor_record_id=predecessor_id,
            created_at=created_at,
            created_by=created_by,
            reason=reason,
            provenance=provenance or {},
        )
        self._records[successor_id] = successor.model_copy(update={"lifecycle_status": "active"})
        self._edges.append(edge)
        return edge

    def withdraw(self, record_id: str, *, reason: str | None = None) -> AuthorityRecord:
        record = self._get(record_id)
        updated = record.model_copy(update={"lifecycle_status": "withdrawn", "withdrawal_reason": reason})
        self._records[record_id] = updated
        return updated

    def mark_disputed(self, record_id: str, *, reason: str | None = None) -> AuthorityRecord:
        record = self._get(record_id)
        updated = record.model_copy(update={"lifecycle_status": "disputed", "dispute_reason": reason})
        self._records[record_id] = updated
        return updated

    def evaluate(self, *, space_id: str, workstream_id: str, authority_key: str) -> AuthorityStateResult:
        return evaluate_current_state(
            self.records,
            self.edges,
            space_id=space_id,
            workstream_id=workstream_id,
            authority_key=authority_key,
        )
