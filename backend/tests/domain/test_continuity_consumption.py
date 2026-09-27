"""Pure tests for the provider-neutral continuity-consumption boundary."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from app.domain.agent_succession.contracts import CurrentAgent, Predecessor
from app.domain.agent_succession.service import build_agent_succession_envelope
from app.domain.context_packets.service import build_context_packet
from app.domain.model_context.continuity import (
    ContinuityBudgetExceeded,
    build_continuity_consumption_payload,
    canonical_consumption_json,
)
from app.domain.model_context.contracts import ContinuityCapabilities
from app.domain.retrieval.service import RetrievalItem, RetrievalResponse

NOW = datetime(2026, 9, 27, 9, 0, tzinfo=UTC)


def authority_record(
    key: str,
    record_id: str,
    content: str,
    *,
    provenance: dict | None = None,
) -> dict:
    return {
        "id": record_id,
        "space_id": "shared",
        "workstream_id": "bootstrap",
        "authority_key": key,
        "record_kind": "decision",
        "content": content,
        "lifecycle_status": "active",
        "created_at": NOW.isoformat(),
        "created_by": "synthetic-author",
        "provenance": provenance if provenance is not None else {"source": f"source-{record_id}"},
    }


def authority_state(
    key: str,
    status: str,
    *,
    content: str | None = None,
    candidates: tuple[dict, ...] = (),
    provenance: dict | None = None,
) -> dict:
    current = (
        authority_record(key, f"{key}-current", content or f"Current {key}", provenance=provenance)
        if status == "current"
        else None
    )
    return {
        "status": status,
        "space_id": "shared",
        "workstream_id": "bootstrap",
        "authority_key": key,
        "current_record": current,
        "competing_records": list(candidates),
        "historical_records": [authority_record(key, f"{key}-historical", f"Historical {key}")],
        "superseded_record_ids": [f"{key}-historical"],
        "state_digest": f"digest-{key}",
        "deterministic": {
            "currentness": "graph_derived",
            "source": "persisted_authority_records",
        },
    }


def memory(
    memory_id: str,
    content: str,
    *,
    conflict_label: str | None = None,
    provenance: dict | None = None,
) -> RetrievalItem:
    return RetrievalItem(
        id=memory_id,
        type="fact",
        content=content,
        score=0.9,
        source_id=f"archive-{memory_id}",
        source_system="synthetic",
        captured_at=NOW.isoformat(),
        conflict_label=conflict_label,
        provenance=provenance or {"source_metadata": {"source_name": f"note-{memory_id}"}},
    )


def handoff(
    *,
    authority_results: list[dict] | None = None,
    memories: list[RetrievalItem] | None = None,
    unresolved_questions: list[str] | None = None,
    flags: list[str] | None = None,
    packet_warnings: list[str] | None = None,
    handoff_warnings: list[str] | None = None,
    current_agent: CurrentAgent | None = None,
    predecessors: list[Predecessor] | None = None,
) -> dict:
    response = RetrievalResponse(
        query="What should happen next?",
        retrieval_mode="hybrid",
        budget_used=sum(len(item.content) for item in memories or []),
        budget_limit=20_000,
        trimming_reason="result_exhausted",
        items=memories or [],
    )
    packet = build_context_packet(response, generated_at=NOW)
    packet = packet.model_copy(
        update={
            "unresolved_questions": unresolved_questions or [],
            "flags": flags or [],
            "warnings": packet_warnings or [],
        }
    )
    agent = current_agent or CurrentAgent(
        provider="local",
        model="synthetic",
        capabilities=["text"],
        limitations=["no network"],
        session_id="fresh-session",
    )
    prior = predecessors or [Predecessor(id="predecessor-1", provider="synthetic")]
    envelope = build_agent_succession_envelope(
        packet,
        current_agent=agent,
        predecessors=prior,
        generated_at=NOW,
    )
    results = authority_results or [authority_state("next_action", "current")]
    return {
        "authority_results": results,
        "context_packet": packet.model_dump(mode="json"),
        "included_memory_ids": [item.memory_id for item in packet.selected],
        "excluded_memory_ids": [item.memory_id for item in packet.excluded],
        "unresolved_questions": list(packet.unresolved_questions),
        "flags": list(packet.flags),
        "retrieval_diagnostics": {"version": "synthetic-diagnostics-v1"},
        "provenance": {
            "authority_source": "persisted_authority_records",
            "memory_source": "ordinary_authorized_retrieval",
        },
        "packet_integrity": packet.integrity.model_dump(mode="json"),
        "predecessors": [item.model_dump(mode="json") for item in prior],
        "current_agent": agent.model_dump(mode="json"),
        "succession_envelope": envelope.model_dump(mode="json"),
        "rendered_handoff": {
            "text": "legacy rendered view",
            "max_chars": 8_000,
            "truncated": False,
            "included_memory_ids": [item.memory_id for item in packet.selected],
            "omitted_memory_count": 0,
        },
        "warnings": handoff_warnings or [],
    }


def test_current_authority_and_supporting_memory_are_structurally_separate() -> None:
    payload = build_continuity_consumption_payload(
        handoff(memories=[memory("m1", "Supporting evidence only.")])
    )

    state = payload.continuity.authority_states[0]
    assert state.status == "current"
    assert state.current_record is not None
    assert state.current_record.content == "Current next_action"
    assert payload.continuity.supporting_memory[0].content == "Supporting evidence only."
    assert "SUPPORTING MEMORY — NON-AUTHORITATIVE" in payload.rendered_context
    assert "CURRENT — authoritative" in payload.rendered_context
    assert payload.capabilities.allowed_tools == ()


def test_ambiguous_and_empty_authority_never_invent_a_winner_or_promote_memory() -> None:
    candidates = (
        authority_record("database", "candidate-b", "Use PostgreSQL."),
        authority_record("database", "candidate-a", "Use SQLite."),
    )
    payload = build_continuity_consumption_payload(
        handoff(
            authority_results=[
                authority_state("missing", "empty"),
                authority_state("database", "ambiguous", candidates=candidates),
            ],
            memories=[memory("m1", "A note suggests SQLite.")],
        )
    )

    ambiguous, empty = payload.continuity.authority_states
    assert ambiguous.authority_key == "database"
    assert ambiguous.current_record is None
    assert [record.id for record in ambiguous.competing_records] == [
        "candidate-a",
        "candidate-b",
    ]
    assert empty.authority_key == "missing"
    assert empty.current_record is None
    assert (
        "must not choose, merge, rank, summarize into a winner, or infer one"
        in payload.rendered_context
    )
    assert "supporting memory must not be promoted" in payload.rendered_context


def test_authority_memory_conflict_remains_visible_and_authority_governs() -> None:
    payload = build_continuity_consumption_payload(
        handoff(
            authority_results=[authority_state("database", "current", content="Use SQLite.")],
            memories=[memory("m1", "Use PostgreSQL.", conflict_label="contradiction")],
        )
    )

    assert payload.continuity.authority_states[0].current_record.content == "Use SQLite."
    assert payload.continuity.supporting_memory[0].conflict_label == "contradiction"
    assert "Current authority governs present project action" in payload.rendered_context
    assert "supporting_memory_conflict:m1" in payload.continuity.critical_warnings


def test_questions_blockers_warnings_and_limitations_survive_structurally_and_in_text() -> None:
    payload = build_continuity_consumption_payload(
        handoff(
            unresolved_questions=["Which migration path remains unresolved?"],
            flags=["blocker: approval required"],
            packet_warnings=[
                "malformed_provenance:m1",
                "upstream_retrieval_budget_excluded_unreported_candidates",
            ],
            handoff_warnings=["custom_client_warning"],
            current_agent=CurrentAgent(
                provider="local",
                model="synthetic",
                capabilities=["text"],
                limitations=["no filesystem", "no network"],
            ),
        )
    )

    assert payload.continuity.unresolved_questions == ("Which migration path remains unresolved?",)
    assert payload.continuity.flags == ("blocker: approval required",)
    assert "custom_client_warning" in payload.continuity.warnings
    assert "malformed_provenance:m1" in payload.continuity.critical_warnings
    assert (
        "upstream_retrieval_budget_excluded_unreported_candidates"
        in payload.continuity.critical_warnings
    )
    assert "current_agent_limitation:no filesystem" in payload.continuity.critical_warnings
    assert "Which migration path remains unresolved?" in payload.rendered_context
    assert "blocker: approval required" in payload.rendered_context
    assert "no filesystem" in payload.rendered_context


def test_supporting_memory_is_reduced_as_whole_records_and_reported() -> None:
    source = handoff(
        memories=[
            memory("short", "short evidence"),
            memory("long", "x" * 3_000),
        ]
    )
    generous = build_continuity_consumption_payload(source, max_model_chars=10_000)
    constrained = build_continuity_consumption_payload(
        source,
        max_model_chars=generous.budget.mandatory_chars + 500,
    )

    assert [item.memory_id for item in constrained.continuity.supporting_memory] == ["short"]
    assert constrained.diagnostics.consumption_omitted_memory_ids == ("long",)
    assert "continuity_supporting_memory_reduced" in constrained.continuity.critical_warnings
    assert "x" * 100 not in constrained.rendered_context
    assert constrained.budget.rendered_chars <= constrained.budget.max_model_chars


def test_mandatory_authority_and_critical_warning_overflow_fail_closed() -> None:
    with pytest.raises(ContinuityBudgetExceeded) as authority_error:
        build_continuity_consumption_payload(
            handoff(
                authority_results=[authority_state("oversized", "current", content="a" * 4_000)]
            ),
            max_model_chars=1_000,
        )
    assert authority_error.value.code == "mandatory_context_exceeds_budget"
    assert authority_error.value.required_chars > authority_error.value.max_chars

    with pytest.raises(ContinuityBudgetExceeded) as warning_error:
        build_continuity_consumption_payload(
            handoff(handoff_warnings=["authority_ambiguous:" + "w" * 4_000]),
            max_model_chars=1_000,
        )
    assert warning_error.value.code == "mandatory_context_exceeds_budget"


def test_compact_prompt_provenance_and_full_client_provenance_are_separate() -> None:
    source = handoff(
        authority_results=[
            authority_state(
                "database",
                "current",
                provenance={
                    "source": "architecture-note",
                    "archive_id": "archive-authority",
                    "private_internal_note": "client-only-detail",
                },
            )
        ],
        memories=[
            memory(
                "m1",
                "Supporting evidence.",
                provenance={
                    "source_metadata": {"source_name": "handoff-note"},
                    "private_internal_note": "memory-client-only-detail",
                },
            )
        ],
    )
    payload = build_continuity_consumption_payload(source)

    assert "archive=archive-m1" in payload.rendered_context
    assert "source=architecture-note" in payload.rendered_context
    assert "client-only-detail" not in payload.rendered_context
    authority_provenance = next(
        entry for entry in payload.diagnostics.full_provenance if entry.kind == "authority"
    )
    memory_provenance = next(
        entry for entry in payload.diagnostics.full_provenance if entry.kind == "memory"
    )
    assert authority_provenance.provenance["private_internal_note"] == "client-only-detail"
    assert memory_provenance.provenance["private_internal_note"] == "memory-client-only-detail"


def test_identity_and_descriptive_capabilities_do_not_become_permissions() -> None:
    payload = build_continuity_consumption_payload(
        handoff(
            memories=[memory("m1", "I chose the earlier design.")],
            current_agent=CurrentAgent(
                provider="local",
                model="fresh-model",
                capabilities=["text", "can_mutate_authority"],
                limitations=["no shell"],
                session_id="fresh-session",
            ),
            predecessors=[Predecessor(id="old-session", provider="other", model="old-model")],
        )
    )

    assert payload.continuity.current_agent.capabilities == [
        "can_mutate_authority",
        "text",
    ]
    assert payload.capabilities.allowed_tools == ()
    assert payload.capabilities.authority_proposal_allowed is False
    assert payload.capabilities.authority_mutation_allowed is False
    assert (
        "Project continuity may persist; model/session identity does not carry over"
        in payload.rendered_context
    )
    assert "I chose the earlier design." in payload.rendered_context
    assert "I remember" not in payload.rendered_context
    assert "old-session" in payload.rendered_context

    with pytest.raises(ValueError, match="cannot grant tools"):
        ContinuityCapabilities(allowed_tools=("supersede_authority_record",))


def test_canonical_result_ignores_semantically_irrelevant_input_order() -> None:
    candidates = (
        authority_record("database", "candidate-b", "Use PostgreSQL."),
        authority_record("database", "candidate-a", "Use SQLite."),
    )
    first_input = handoff(
        authority_results=[
            authority_state("z-key", "current"),
            authority_state("database", "ambiguous", candidates=candidates),
        ],
        handoff_warnings=["warning-z", "warning-a"],
    )
    second_input = deepcopy(first_input)
    second_input["authority_results"].reverse()
    second_input["authority_results"][0]["competing_records"].reverse()
    second_input["warnings"].reverse()

    first = build_continuity_consumption_payload(first_input)
    second = build_continuity_consumption_payload(second_input)

    assert first == second
    assert canonical_consumption_json(first) == canonical_consumption_json(second)
    assert first.integrity.consumption_digest == second.integrity.consumption_digest


def test_adapter_has_no_network_database_tool_or_authority_side_effects(monkeypatch) -> None:
    network = Mock(side_effect=AssertionError("network call"))
    database = Mock(side_effect=AssertionError("database call"))
    monkeypatch.setattr("socket.create_connection", network)
    monkeypatch.setattr("sqlalchemy.ext.asyncio.create_async_engine", database)
    source = handoff(memories=[memory("m1", "Evidence")])
    before = deepcopy(source)

    payload = build_continuity_consumption_payload(source)

    assert payload.capabilities.allowed_tools == ()
    assert source == before
    network.assert_not_called()
    database.assert_not_called()


def test_rendered_and_structured_authority_statuses_agree_and_all_warnings_survive() -> None:
    source = handoff(
        authority_results=[
            authority_state("a-current", "current"),
            authority_state(
                "b-ambiguous",
                "ambiguous",
                candidates=(
                    authority_record("b-ambiguous", "b1", "Candidate one"),
                    authority_record("b-ambiguous", "b2", "Candidate two"),
                ),
            ),
            authority_state("c-empty", "empty"),
        ],
        handoff_warnings=["custom_diagnostic_only"],
    )
    payload = build_continuity_consumption_payload(source)

    assert [state.status for state in payload.continuity.authority_states] == [
        "current",
        "ambiguous",
        "empty",
    ]
    assert "a-current: CURRENT" in payload.rendered_context
    assert "b-ambiguous: AMBIGUOUS" in payload.rendered_context
    assert "c-empty: EMPTY" in payload.rendered_context
    assert set(source["warnings"]).issubset(payload.continuity.warnings)
    assert "custom_diagnostic_only" not in payload.continuity.critical_warnings
