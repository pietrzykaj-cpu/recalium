"""Pure ContextPacket v1 contract and builder tests (no database access)."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict

import pytest

from app.domain.context_packets.service import build_context_packet
from app.domain.retrieval.diagnostics import RetrievalDiagnosticsCollector
from app.domain.retrieval.service import RetrievalItem, RetrievalRequest, RetrievalResponse, _merge_rrf


NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


def item(
    memory_id: str,
    content: str,
    *,
    score: float = 0.5,
    provenance: object | None = None,
) -> RetrievalItem:
    return RetrievalItem(
        id=memory_id,
        type="fact",
        content=content,
        score=score,
        source_id=f"source-{memory_id}",
        source_system="chatgpt_import",
        captured_at="2026-09-13T10:00:00+00:00",
        conflict_label=None,
        provenance=provenance,  # type: ignore[arg-type]
    )


def response(*items: RetrievalItem, mode: str = "hybrid") -> RetrievalResponse:
    return RetrievalResponse(
        query="What did we decide?",
        retrieval_mode=mode,
        budget_used=sum(len(record.content) for record in items),
        budget_limit=2000,
        trimming_reason="result_exhausted",
        items=list(items),
        degraded_mode=False,
    )


def test_deterministic_packet_shape_and_identity() -> None:
    retrieval = response(item("one", "A prior decision."))

    first = build_context_packet(retrieval, generated_at=NOW)
    second = build_context_packet(retrieval, generated_at=NOW)

    assert first == second
    assert first.schema_version == "recalium.context-packet.v1"
    assert first.packet_id == second.packet_id
    assert first.generated_at == NOW
    assert first.evidence_label == "evidence_from_prior_interactions"
    assert first.attribution_notice.startswith("Inherited records are attributed evidence")
    assert first.model_context is None


def test_score_and_provenance_preservation() -> None:
    provenance = {
        "source_metadata": {"conversation_id": "thread-7"},
        "retrieval": {
            "lexical_score": 0.82,
            "semantic_score": 0.74,
            "rrf_score": 0.031,
        },
        "source_excerpt": "verbatim excerpt",
    }
    packet = build_context_packet(
        response(item("one", "Decision", score=0.91, provenance=provenance)),
        generated_at=NOW,
    )

    selected = packet.selected[0]
    assert selected.provenance == provenance
    assert selected.source.archive_id == "source-one"
    assert selected.source.captured_at == "2026-09-13T10:00:00+00:00"
    assert selected.retrieval.final_score == 0.91
    assert selected.retrieval.hybrid_score == 0.91
    assert selected.retrieval.lexical_score == 0.82
    assert selected.retrieval.semantic_score == 0.74
    assert selected.retrieval.rrf_score == 0.031
    assert selected.retrieval.relevance_rank == 1


def test_context_packet_consumes_diagnostics_but_does_not_require_them() -> None:
    diagnostics = RetrievalDiagnosticsCollector(mode="hybrid")
    lexical = {
        "id": "fts", "type": "excerpt", "content": "evidence", "score": 0.9,
        "source_id": "shared", "source_system": "test",
        "captured_at": "2026-09-14T00:00:00+00:00", "provenance": {},
    }
    semantic = {
        "id": "vector", "type": "summary", "content": "evidence", "score": 0.7,
        "source_id": "shared", "source_system": "test",
        "captured_at": "2026-09-14T00:00:00+00:00", "provenance": {},
    }
    fused = _merge_rrf([lexical], [semantic], diagnostics=diagnostics)
    retrieval = response(
        RetrievalItem(
            id=fused[0]["id"], type=fused[0]["type"], content=fused[0]["content"],
            score=fused[0]["score"], source_id=fused[0]["source_id"],
            source_system=fused[0]["source_system"], captured_at=fused[0]["captured_at"],
            conflict_label=fused[0].get("conflict_label"), provenance=fused[0]["provenance"],
        )
    )

    plain = build_context_packet(retrieval, generated_at=NOW)
    enriched = build_context_packet(
        retrieval, diagnostics=diagnostics.snapshot(), generated_at=NOW,
    )

    assert plain.selected[0].retrieval.lexical_score is None
    assert enriched.selected[0].retrieval.lexical_score == 0.9
    assert enriched.selected[0].retrieval.semantic_score == 0.7
    assert enriched.selected[0].retrieval.rrf_score == retrieval.items[0].score
    assert [record.memory_id for record in plain.selected] == [
        record.memory_id for record in enriched.selected
    ]


def test_token_budget_trimming_is_whole_record_and_logs_exclusion() -> None:
    retrieval = response(item("one", "12345678"), item("two", "abcdefghijkl"))
    packet = build_context_packet(
        retrieval,
        token_budget=2,
        chars_per_token=4,
        generated_at=NOW,
    )

    assert [record.memory_id for record in packet.selected] == ["one"]
    assert packet.budget.unit == "estimated_tokens"
    assert packet.budget.limit == 2
    assert packet.budget.used == 2
    assert packet.budget.estimator == "character_ratio:4"
    assert packet.excluded[0].memory_id == "two"
    assert packet.excluded[0].reason == "context_budget_exceeded"


def test_upstream_exclusion_is_explicit_when_retrieval_was_already_trimmed() -> None:
    retrieval = response(item("one", "kept"))
    retrieval.trimming_reason = "budget_met"
    packet = build_context_packet(retrieval, generated_at=NOW)

    assert packet.excluded == []
    assert "upstream_retrieval_budget_excluded_unreported_candidates" in packet.warnings


def test_empty_retrieval_builds_valid_empty_packet() -> None:
    packet = build_context_packet(response(), generated_at=NOW)

    assert packet.selected == []
    assert packet.excluded == []
    assert packet.unresolved_questions == []
    assert packet.flags == []
    assert packet.budget.used == 0


def test_malformed_optional_metadata_is_ignored_and_flagged() -> None:
    malformed = item("bad", "Still usable", provenance="not-a-mapping")
    packet = build_context_packet(response(malformed), generated_at=NOW)

    assert packet.selected[0].provenance == {}
    assert packet.unresolved_questions == []
    assert packet.flags == []
    assert "malformed_provenance:bad" in packet.warnings


def test_existing_retrieval_object_remains_unchanged() -> None:
    provenance = {
        "source_metadata": {
            "unresolved_questions": ["Which deployment?"],
            "flags": ["needs_review"],
        }
    }
    retrieval = response(item("one", "Evidence", provenance=provenance))
    before_items = list(retrieval.items)
    before_provenance = dict(provenance)

    packet = build_context_packet(
        retrieval,
        provider="ollama",
        model="qwen3:4b",
        generated_at=NOW,
    )

    assert retrieval.items == before_items
    assert provenance == before_provenance
    assert packet.unresolved_questions == ["Which deployment?"]
    assert packet.flags == ["needs_review"]
    assert packet.model_context is not None
    assert packet.model_context.provider == "ollama"
    assert packet.model_context.model == "qwen3:4b"


@pytest.mark.asyncio
async def test_real_hybrid_pipeline_feeds_context_packet_without_database(monkeypatch) -> None:
    """Exercise real RRF/filter/budget code with only the SQL edges replaced."""
    from app.domain.retrieval import service as retrieval_service

    class NestedTransaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    class MemorySession:
        def __init__(self):
            self.added = []
            self.flushed = False

        def begin_nested(self):
            return NestedTransaction()

        def add(self, value):
            self.added.append(value)

        async def flush(self):
            self.flushed = True

    keyword = [
        {
            "id": "canonical-1",
            "type": "canonical",
            "content": "Keep this decision.",
            "score": 0.99,
            "source_id": "archive-canonical",
            "source_system": "canonical",
            "captured_at": "2026-09-01T00:00:00+00:00",
            "conflict_label": None,
            "provenance": {"derivation_method": "fts_retrieval"},
        },
        {
            "id": "fts-shared",
            "type": "excerpt",
            "content": "A" * 160,
            "score": 0.90,
            "source_id": "archive-shared",
            "source_system": "chatgpt_import",
            "captured_at": "2026-09-02T00:00:00+00:00",
            "conflict_label": None,
            "provenance": {
                "derivation_method": "fts_retrieval",
                "source_metadata": {"unresolved_questions": ["Which host?"]},
            },
        },
        {
            "id": "fact-conflict",
            "type": "fact",
            "content": "Conflicting evidence.",
            "score": 0.80,
            "source_id": "archive-conflict",
            "source_system": "claude_import",
            "captured_at": "2026-09-03T00:00:00+00:00",
            "conflict_label": "contradiction",
            "provenance": {},
        },
    ]
    semantic = [
        {
            "id": "embedding-shared",
            "type": "summary",
            "content": "Shared source summary.",
            "score": 0.95,
            "source_id": "archive-shared",
            "source_system": "chatgpt_import",
            "captured_at": "2026-09-02T00:00:00+00:00",
            "conflict_label": None,
            "provenance": {"derivation_method": "semantic_retrieval"},
        },
        {
            "id": "embedding-long",
            "type": "summary",
            "content": "L" * 240,
            "score": 0.70,
            "source_id": "archive-long",
            "source_system": "bridge",
            "captured_at": "2026-09-04T00:00:00+00:00",
            "conflict_label": None,
        },
        {
            "id": "embedding-short",
            "type": "summary",
            "content": "Short later evidence.",
            "score": 0.60,
            "source_id": "archive-short",
            "source_system": "bridge",
            "captured_at": "2026-09-05T00:00:00+00:00",
            "conflict_label": None,
            "provenance": {},
        },
    ]

    async def keyword_candidates(*args, **kwargs):
        return keyword

    async def semantic_candidates(*args, **kwargs):
        return semantic, False

    async def no_link_expansion(*args, **kwargs):
        return []

    monkeypatch.setattr(retrieval_service, "_keyword_candidates", keyword_candidates)
    monkeypatch.setattr(retrieval_service, "_semantic_candidates", semantic_candidates)
    monkeypatch.setattr(retrieval_service, "_traverse_links", no_link_expansion)
    retrieval_service.invalidate_cache()
    session = MemorySession()

    retrieval = await retrieval_service.retrieve(
        session,
        RetrievalRequest(query="continuity", mode="hybrid", budget=1000),
    )
    before = asdict(retrieval)
    packet = build_context_packet(
        retrieval,
        token_budget=20,
        chars_per_token=4,
        generated_at=NOW,
    )

    assert session.flushed is True
    assert len(session.added) == 1
    assert retrieval.retrieval_mode == "hybrid"
    assert len([record for record in retrieval.items if record.source_id == "archive-shared"]) == 1
    assert retrieval.items[0].type == "canonical"
    assert retrieval.items[1].type == "fact"
    assert retrieval.items[1].conflict_label == "contradiction"
    assert asdict(retrieval) == before

    selected_ids = [record.memory_id for record in packet.selected]
    assert selected_ids == ["canonical-1", "fact-conflict", "embedding-shared"]
    assert [record.memory_id for record in packet.excluded] == [
        "embedding-long",
        "embedding-short",
    ]
    assert packet.selected[2].retrieval.hybrid_score == retrieval.items[2].score
    assert packet.selected[2].retrieval.lexical_score is None
    assert packet.selected[2].retrieval.semantic_score is None
    # Fusion preserves useful metadata from the non-representative lexical hit.
    assert packet.unresolved_questions == ["Which host?"]
