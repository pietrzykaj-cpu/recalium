"""Pure AgentSuccessionEnvelope v1 tests; no provider, DB, or filesystem access."""
from __future__ import annotations

from datetime import datetime, timezone

from app.domain.agent_succession.contracts import (
    CurrentAgent, Predecessor, SuccessorAnnotation,
)
from app.domain.agent_succession.service import (
    build_agent_succession_envelope, render_agent_succession_context,
)
from app.domain.context_packets.service import build_context_packet
from app.domain.retrieval.service import RetrievalItem, RetrievalResponse


NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


def packet():
    response = RetrievalResponse(
        query="What did we decide?", retrieval_mode="hybrid", budget_used=8,
        budget_limit=2000, trimming_reason="result_exhausted", degraded_mode=False,
        items=[RetrievalItem(
            id="memory-1", type="fact", content="We chose local-first storage.", score=.9,
            source_id="archive-1", source_system="synthetic", captured_at="2026-09-14T00:00:00Z",
            conflict_label="needs_review", provenance={"source_metadata": {"conversation_id": "thread-1"}},
        )],
    )
    return build_context_packet(response, generated_at=NOW)


def current() -> CurrentAgent:
    return CurrentAgent(provider="ollama", model="qwen3:4b", version="local", capabilities=["text"], limitations=["no web"])


def test_single_predecessor() -> None:
    envelope = build_agent_succession_envelope(packet(), current_agent=current(), predecessors=[Predecessor(id="p1", provider="openai", model="gpt")], generated_at=NOW)
    assert envelope.inherited.context_packet.packet_id == packet().packet_id
    assert envelope.inherited.predecessors[0].id == "p1"


def test_multiple_predecessors() -> None:
    envelope = build_agent_succession_envelope(packet(), current_agent=current(), predecessors=[Predecessor(id="p1"), Predecessor(id="p2", provider="anthropic")], generated_at=NOW)
    assert [record.id for record in envelope.inherited.predecessors] == ["p1", "p2"]


def test_missing_predecessor_metadata_is_explicitly_unknown() -> None:
    envelope = build_agent_succession_envelope(packet(), current_agent=current(), predecessors=[Predecessor(id="unknown")], generated_at=NOW)
    assert envelope.inherited.predecessors[0].provider is None


def test_conflicting_predecessor_conclusions_are_preserved() -> None:
    envelope = build_agent_succession_envelope(packet(), current_agent=current(), predecessors=[Predecessor(id="a"), Predecessor(id="b")], decisions=[{"id": "d1", "content": "Use SQLite.", "predecessor_ids": ["a"], "conflict": True}, {"id": "d2", "content": "Use PostgreSQL.", "predecessor_ids": ["b"], "conflict": True}], generated_at=NOW)
    assert [record.content for record in envelope.inherited.decisions] == ["Use SQLite.", "Use PostgreSQL."]
    assert all(record.conflict for record in envelope.inherited.decisions)


def test_unresolved_questions_are_preserved() -> None:
    source = packet().model_copy(update={"unresolved_questions": ["Which model next?"]})
    envelope = build_agent_succession_envelope(source, current_agent=current(), generated_at=NOW)
    assert envelope.inherited.context_packet.unresolved_questions == ["Which model next?"]


def test_provenance_is_preserved() -> None:
    envelope = build_agent_succession_envelope(packet(), current_agent=current(), decisions=[{"id": "d1", "content": "Keep the source.", "provenance": {"archive_id": "archive-1"}}], generated_at=NOW)
    assert envelope.inherited.context_packet.selected[0].source.archive_id == "archive-1"
    assert envelope.inherited.decisions[0].provenance == {"archive_id": "archive-1"}


def test_current_model_metadata_and_disagreement_are_separate() -> None:
    envelope = build_agent_succession_envelope(packet(), current_agent=current(), predecessors=[Predecessor(id="p1")], annotations=[SuccessorAnnotation(kind="disagreement", target_predecessor_id="p1", rationale="The evidence is incomplete.")], generated_at=NOW)
    assert envelope.current_agent.model == "qwen3:4b"
    assert envelope.current_agent.annotations[0].target_predecessor_id == "p1"


def test_rendering_is_deterministic() -> None:
    envelope = build_agent_succession_envelope(packet(), current_agent=current(), predecessors=[Predecessor(id="p1", provider="openai", model="gpt")], generated_at=NOW)
    assert render_agent_succession_context(envelope, max_chars=2000) == render_agent_succession_context(envelope, max_chars=2000)


def test_renderer_does_not_turn_inherited_evidence_into_current_first_person_memory() -> None:
    source = packet().model_copy(update={"selected": [packet().selected[0].model_copy(update={"content": "I chose a local database."})]})
    envelope = build_agent_succession_envelope(source, current_agent=current(), generated_at=NOW)
    rendered = render_agent_succession_context(envelope, max_chars=2000).text
    assert "I remember" not in rendered
    assert "my prior experience" not in rendered
    assert "verbatim inherited evidence" in rendered
    assert "I chose a local database." in rendered


def test_context_packet_compatibility() -> None:
    envelope = build_agent_succession_envelope(packet(), current_agent=current(), generated_at=NOW)
    assert envelope.inherited.context_packet.schema_version == "recalium.context-packet.v1"


def test_empty_minimal_inheritance() -> None:
    empty = build_context_packet(RetrievalResponse(query="", retrieval_mode="keyword", budget_used=0, budget_limit=0, trimming_reason="result_exhausted", items=[]), generated_at=NOW)
    envelope = build_agent_succession_envelope(empty, current_agent=current(), generated_at=NOW)
    assert envelope.inherited.predecessors == []
    assert "No selected inherited evidence" in render_agent_succession_context(envelope, max_chars=500).text


def test_context_budget_truncation_preserves_boundaries_and_provenance() -> None:
    source = packet().model_copy(update={"selected": [packet().selected[0].model_copy(update={"content": "x" * 1000})]})
    envelope = build_agent_succession_envelope(source, current_agent=current(), predecessors=[Predecessor(id="p1")], generated_at=NOW)
    rendered = render_agent_succession_context(envelope, max_chars=420)
    assert rendered.truncated is True
    assert "INHERITED EVIDENCE" in rendered.text
    assert "CURRENT AGENT" in rendered.text
    assert "archive-1" in rendered.text
    assert rendered.included_memory_ids == []
