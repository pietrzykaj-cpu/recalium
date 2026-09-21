"""RetrievalDiagnostics v1 tests; all SQL boundaries remain out of process."""

from __future__ import annotations

from dataclasses import asdict
from app.domain.retrieval.diagnostics import RetrievalDiagnosticsCollector
from app.domain.retrieval.service import (
    RetrievalItem,
    RetrievalResponse,
    _merge_rrf,
    apply_budget_trimming,
)


def candidate(
    memory_id: str,
    channel: str,
    *,
    source_id: str | None = None,
    score: float = 0.8,
    content: str = "evidence",
    provenance: dict | None = None,
    conflict_label: str | None = None,
) -> dict:
    return {
        "id": memory_id,
        "type": "excerpt" if channel == "lexical" else "summary",
        "content": content,
        "score": score,
        "source_id": source_id or f"source-{memory_id}",
        "source_system": "test",
        "captured_at": "2026-09-14T00:00:00+00:00",
        "conflict_label": conflict_label,
        "provenance": provenance or {},
    }


def response_from(candidates: list[dict]) -> RetrievalResponse:
    return RetrievalResponse(
        query="continuity",
        retrieval_mode="hybrid",
        budget_used=sum(len(item["content"]) for item in candidates),
        budget_limit=2000,
        trimming_reason="result_exhausted",
        items=[
            RetrievalItem(
                id=item["id"],
                type=item["type"],
                content=item["content"],
                score=item["score"],
                source_id=item["source_id"],
                source_system=item["source_system"],
                captured_at=item["captured_at"],
                conflict_label=item.get("conflict_label"),
                provenance=item.get("provenance", {}),
            )
            for item in candidates
        ],
    )


def test_lexical_only_hit_records_score_rank_and_channel() -> None:
    diagnostics = RetrievalDiagnosticsCollector(mode="hybrid")
    fused = _merge_rrf([candidate("lex", "lexical", score=0.91)], [], diagnostics=diagnostics)

    record = diagnostics.for_representative(fused[0]["id"])
    assert record is not None
    assert record.channels == ["lexical"]
    assert record.lexical.score == 0.91
    assert record.lexical.rank == 1
    assert record.semantic is None


def test_semantic_only_hit_records_score_rank_and_channel() -> None:
    diagnostics = RetrievalDiagnosticsCollector(mode="hybrid")
    fused = _merge_rrf([], [candidate("sem", "semantic", score=0.73)], diagnostics=diagnostics)

    record = diagnostics.for_representative(fused[0]["id"])
    assert record is not None
    assert record.channels == ["semantic"]
    assert record.semantic.score == 0.73
    assert record.semantic.rank == 1
    assert record.lexical is None


def test_same_memory_from_both_channels_fuses_without_duplicate() -> None:
    diagnostics = RetrievalDiagnosticsCollector(mode="hybrid")
    lexical = candidate("fts", "lexical", source_id="shared")
    semantic = candidate("vector", "semantic", source_id="shared")

    fused = _merge_rrf([lexical], [semantic], diagnostics=diagnostics)

    assert len(fused) == 1
    record = diagnostics.for_representative(fused[0]["id"])
    assert record.channels == ["lexical", "semantic"]
    assert record.duplicate_count == 2
    assert record.representative.memory_id == "vector"
    assert record.representative.reason == "higher_memory_type_priority"


def test_conflicting_metadata_is_recorded_and_representative_value_is_stable() -> None:
    diagnostics = RetrievalDiagnosticsCollector(mode="hybrid")
    lexical = candidate(
        "fts", "lexical", source_id="shared", provenance={"source": {"author": "Anna"}}
    )
    semantic = candidate(
        "vector", "semantic", source_id="shared", provenance={"source": {"author": "Model"}}
    )

    fused = _merge_rrf([lexical], [semantic], diagnostics=diagnostics)

    assert fused[0]["provenance"]["source"]["author"] == "Model"
    record = diagnostics.for_representative("vector")
    assert "source.author" in record.metadata.conflict_paths
    assert record.metadata.discarded == []


def test_unresolved_questions_are_preserved_across_fusion() -> None:
    diagnostics = RetrievalDiagnosticsCollector(mode="hybrid")
    lexical = candidate(
        "fts",
        "lexical",
        source_id="shared",
        provenance={"source_metadata": {"unresolved_questions": ["Which host?"]}},
    )
    semantic = candidate(
        "vector",
        "semantic",
        source_id="shared",
        provenance={"source_metadata": {"unresolved_questions": ["Which model?"]}},
    )

    fused = _merge_rrf([lexical], [semantic], diagnostics=diagnostics)

    assert fused[0]["provenance"]["source_metadata"]["unresolved_questions"] == [
        "Which model?",
        "Which host?",
    ]
    record = diagnostics.for_representative("vector")
    assert record.metadata.unresolved_after == ["Which model?", "Which host?"]


def test_metadata_merge_is_deterministic() -> None:
    lexical = candidate(
        "fts", "lexical", source_id="shared", provenance={"flags": ["a"], "z": 1}
    )
    semantic = candidate(
        "vector", "semantic", source_id="shared", provenance={"flags": ["b"], "a": 2}
    )
    outputs = []
    snapshots = []
    for _ in range(2):
        diagnostics = RetrievalDiagnosticsCollector(mode="hybrid")
        outputs.append(_merge_rrf([lexical], [semantic], diagnostics=diagnostics))
        snapshot = diagnostics.snapshot().model_dump(mode="json")
        snapshot.pop("generated_at")
        snapshots.append(snapshot)
    assert outputs[0] == outputs[1]
    assert snapshots[0] == snapshots[1]


def test_threshold_and_filter_configuration_are_explicit() -> None:
    diagnostics = RetrievalDiagnosticsCollector(
        mode="hybrid",
        filters={"source_system": "chatgpt"},
        memory_space_ids=["private"],
    )
    lexical = [candidate(f"item-{rank}", "lexical") for rank in range(1, 27)]
    _merge_rrf(lexical, [], diagnostics=diagnostics)
    diagnostics.record_unknown_exclusion("sql_candidate_limit", "not_observable_at_this_stage")

    snapshot = diagnostics.snapshot()
    assert snapshot.filters == {"source_system": "chatgpt"}
    assert snapshot.memory_space_ids == ["private"]
    assert snapshot.thresholds["rrf_minimum"] > 0
    assert any(
        item.exclusion is not None and item.exclusion.reason == "rrf_below_threshold"
        for item in snapshot.candidates
    )


def test_priority_and_budget_trimming_are_diagnosed() -> None:
    diagnostics = RetrievalDiagnosticsCollector(mode="hybrid")
    fused = _merge_rrf(
        [candidate("large", "lexical", content="x" * 50)],
        [candidate("small", "semantic", content="small")],
        diagnostics=diagnostics,
    )
    items = response_from(fused).items

    selected, used, reason = apply_budget_trimming(items, 10, diagnostics=diagnostics)

    assert [item.id for item in selected] == ["small"]
    assert used == 5
    assert reason == "result_exhausted"
    large = diagnostics.for_representative("large")
    assert large.exclusion.reason == "character_budget_exceeded"
    assert large.exclusion.stage == "priority_budget_trimming"


def test_duplicate_fusion_records_all_source_ids() -> None:
    diagnostics = RetrievalDiagnosticsCollector(mode="hybrid")
    fused = _merge_rrf(
        [candidate("fts", "lexical", source_id="shared")],
        [candidate("vector", "semantic", source_id="shared")],
        diagnostics=diagnostics,
    )
    record = diagnostics.for_representative(fused[0]["id"])
    assert record.fused_memory_ids == ["fts", "vector"]


def test_unobservable_exclusions_are_truthfully_unknown() -> None:
    diagnostics = RetrievalDiagnosticsCollector(mode="hybrid")
    diagnostics.record_unknown_exclusion("lexical_sql_limit", "not_observable_at_this_stage")
    unknown = diagnostics.snapshot().unknown_exclusions[0]
    assert unknown.observable is False
    assert unknown.memory_id is None
    assert unknown.reason == "not_observable_at_this_stage"


def test_diagnostics_do_not_change_fused_results_or_ranking() -> None:
    lexical = [candidate("a", "lexical", score=0.9), candidate("b", "lexical", score=0.8)]
    semantic = [candidate("a-sem", "semantic", source_id="source-a", score=0.7)]
    ordinary = _merge_rrf(lexical, semantic)
    diagnostics = RetrievalDiagnosticsCollector(mode="hybrid")
    observed = _merge_rrf(lexical, semantic, diagnostics=diagnostics)
    assert observed == ordinary
