"""Pure Authority Phase 1A tests; no database/provider/filesystem access."""

from datetime import datetime, timezone

import pytest

from app.domain.authority.contracts import AuthorityRecord
from app.domain.authority.service import AuthorityGraph, AuthorityValidationError


NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)


def rec(record_id: str, *, status: str = "proposed", space: str = "recalium", workstream: str = "bootstrap", key: str = "storage") -> AuthorityRecord:
    return AuthorityRecord(
        id=record_id, space_id=space, workstream_id=workstream, authority_key=key,
        record_kind="decision", content=f"decision {record_id}", lifecycle_status=status,
        created_at=NOW, created_by="test", provenance={"source": f"synthetic-{record_id}"},
    )


def graph(*records: AuthorityRecord) -> AuthorityGraph:
    g = AuthorityGraph()
    for item in records:
        g.create_record(item)
    return g


def state(g: AuthorityGraph):
    return g.evaluate(space_id="recalium", workstream_id="bootstrap", authority_key="storage")


def test_one_active_is_current_and_provenance_survives():
    result = state(graph(rec("a", status="active")))
    assert result.status == "current"
    assert result.current_record.provenance["source"] == "synthetic-a"


def test_supersession_makes_successor_current_and_predecessor_historical():
    g = graph(rec("a", status="active"), rec("b"))
    g.supersede("b", "a", created_at=NOW, created_by="test")
    result = state(g)
    assert result.current_record.id == "b"
    assert [r.id for r in result.historical_records] == ["a"]


@pytest.mark.parametrize("transition", ["withdraw", "mark_disputed"])
def test_successor_never_revives_predecessor(transition):
    g = graph(rec("a", status="active"), rec("b"))
    g.supersede("b", "a", created_at=NOW, created_by="test")
    getattr(g, transition)("b", reason="synthetic")
    assert state(g).status == "empty"


def test_competitors_are_ambiguous_and_proposed_is_ignored():
    result = state(graph(rec("a", status="active"), rec("b", status="active")))
    assert result.status == "ambiguous"
    assert [r.id for r in result.competing_records] == ["a", "b"]
    assert state(graph(rec("a", status="active"), rec("b"))).current_record.id == "a"


def test_withdrawal_and_dispute_without_replacement_are_empty():
    g = graph(rec("a", status="active"))
    g.withdraw("a", reason="closed")
    assert state(g).status == "empty"
    g = graph(rec("b", status="active"))
    g.mark_disputed("b", reason="evidence")
    assert state(g).status == "empty"


def test_timestamp_and_insertion_order_never_select_authority():
    older = rec("z", status="active").model_copy(update={"created_at": NOW.replace(day=1)})
    newer = rec("a", status="active")
    assert state(graph(older, newer)).model_dump() == state(graph(newer, older)).model_dump()
    assert state(graph(older, newer)).status == "ambiguous"
    assert [r.id for r in state(graph(older, newer)).competing_records] == ["a", "z"]


def test_self_duplicate_and_unknown_edges_are_rejected():
    g = graph(rec("a", status="active"), rec("b"))
    with pytest.raises(AuthorityValidationError):
        g.supersede("a", "a", created_at=NOW, created_by="test")
    g.supersede("b", "a", created_at=NOW, created_by="test")
    with pytest.raises(AuthorityValidationError):
        g.supersede("b", "a", created_at=NOW, created_by="test")
    with pytest.raises(AuthorityValidationError):
        g.supersede("missing", "a", created_at=NOW, created_by="test")


def test_cycles_and_cross_scope_edges_are_rejected():
    g = graph(rec("a", status="active"), rec("b"), rec("c"))
    g.supersede("b", "a", created_at=NOW, created_by="test")
    g.supersede("c", "b", created_at=NOW, created_by="test")
    with pytest.raises(AuthorityValidationError):
        g.supersede("a", "c", created_at=NOW, created_by="test")
    with pytest.raises(AuthorityValidationError):
        graph(rec("a", status="active"), rec("x", space="other")).supersede("x", "a", created_at=NOW, created_by="test")
    with pytest.raises(AuthorityValidationError):
        graph(rec("a", status="active"), rec("y", workstream="other")).supersede("y", "a", created_at=NOW, created_by="test")


def test_reversal_requires_new_record_and_results_are_stable():
    g = graph(rec("a", status="active"), rec("b"))
    g.supersede("b", "a", created_at=NOW, created_by="test")
    g.withdraw("b", reason="reversed")
    assert state(g).status == "empty"
    g.create_record(rec("c"))
    g.supersede("c", "b", created_at=NOW, created_by="test")
    assert state(g).current_record.id == "c"
    assert state(g).model_dump() == state(g).model_dump()
