"""Pure canonical authority-state digest tests."""

from datetime import UTC, datetime

from app.domain.authority.contracts import AuthorityEdge, AuthorityRecord
from app.domain.authority.mutation_contracts import AuthorityScope
from app.domain.authority.mutations import authority_state_digest

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
SCOPE = AuthorityScope(space_id="shared", workstream_id="bootstrap", authority_key="storage")


def record(record_id: str, *, status: str = "active", content: str = "SQLite"):
    return AuthorityRecord(
        id=record_id,
        space_id=SCOPE.space_id,
        workstream_id=SCOPE.workstream_id,
        authority_key=SCOPE.authority_key,
        record_kind="decision",
        content=content,
        lifecycle_status=status,
        created_at=NOW,
        created_by="synthetic-client",
        provenance={"source": "synthetic", "nested": {"b": 2, "a": 1}},
    )


def edge(successor: str, predecessor: str):
    return AuthorityEdge(
        successor_record_id=successor,
        predecessor_record_id=predecessor,
        created_at=NOW,
        created_by="synthetic-client",
        reason="replacement",
        provenance={"z": 2, "a": 1},
    )


def digest(records, edges=()):
    return authority_state_digest(records, edges, scope=SCOPE)


def test_digest_ignores_record_edge_and_json_key_order():
    first = record("00000000-0000-0000-0000-000000000001")
    second = record("00000000-0000-0000-0000-000000000002", status="proposed")
    link = edge(second.id, first.id)
    reordered_second = second.model_copy(
        update={"provenance": {"nested": {"a": 1, "b": 2}, "source": "synthetic"}}
    )
    assert digest([first, second], [link]) == digest([reordered_second, first], [link])


def test_digest_is_timestamp_independent_and_platform_pinned():
    item = record("00000000-0000-0000-0000-000000000001")
    later = item.model_copy(update={"created_at": NOW.replace(day=27)})
    assert digest([item]) == digest([later])
    assert digest([item]) == "b8c4df696b35886e66c759fc1e9d0b84f5d50d9f14aba6ccef0e2ccdc322dc95"


def test_material_authority_changes_change_digest():
    item = record("00000000-0000-0000-0000-000000000001")
    variants = [
        item.model_copy(update={"content": "PostgreSQL"}),
        item.model_copy(update={"lifecycle_status": "withdrawn", "withdrawal_reason": "closed"}),
        item.model_copy(update={"record_kind": "constraint"}),
        item.model_copy(update={"provenance": {"source": "different"}}),
    ]
    assert all(digest([item]) != digest([variant]) for variant in variants)


def test_material_edge_changes_change_digest():
    first = record("00000000-0000-0000-0000-000000000001")
    second = record("00000000-0000-0000-0000-000000000002")
    link = edge(second.id, first.id)
    changed_reason = link.model_copy(update={"reason": "different"})
    reversed_link = link.model_copy(
        update={
            "successor_record_id": first.id,
            "predecessor_record_id": second.id,
        }
    )
    assert digest([first, second], [link]) != digest([first, second], [changed_reason])
    assert digest([first, second], [link]) != digest([first, second], [reversed_link])
