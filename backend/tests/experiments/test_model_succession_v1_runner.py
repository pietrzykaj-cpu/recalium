"""Synthetic-only tests for the Model Succession Experiment v1 runner."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from experiments.model_succession_v1 import runner


def manifest():
    return runner.load_manifest()


def write_variant(tmp_path: Path, mutate) -> Path:
    raw = json.loads(runner.DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_frozen_manifest_loads_deterministically_and_verifies_hash() -> None:
    first, first_digest = manifest()
    second, second_digest = manifest()
    assert first == second
    assert first_digest == second_digest == runner.EXPECTED_MANIFEST_SHA256
    with pytest.raises(ValueError, match="hash"):
        runner.load_manifest(runner.DEFAULT_MANIFEST_PATH, expected_sha256="0" * 64)


def test_a_b_c_derivation_has_no_a_inheritance_and_exact_b_c_record_parity() -> None:
    source, digest = manifest()
    materials, parity = runner.build_conditions(source, digest, task=source.tasks[0])
    assert materials["A"].exposed_records == ()
    assert "Aurora-17" not in materials["A"].request.messages[0].content
    assert parity.parity_established
    assert [record.record_id for record in parity.b_records] == [f"E{n:02d}" for n in range(1, 17)]
    assert parity.b_records == parity.c_records
    assert materials["B"].request.messages[-1].content == materials["C"].request.messages[-1].content == source.tasks[0].prompt
    assert "AGENT SUCCESSION ENVELOPE" not in materials["B"].request.messages[0].content
    assert "AGENT SUCCESSION ENVELOPE" in materials["C"].request.messages[0].content
    assert parity.b_metrics.characters != parity.c_metrics.characters


def test_parity_failure_is_detected_before_trial_creation(monkeypatch) -> None:
    source, digest = manifest()
    original = runner._exposed
    monkeypatch.setattr(runner, "_exposed", lambda _: original(source)[:-1])
    # The test deliberately simulates a C-only exposure failure by replacing the renderer output.
    monkeypatch.setattr(runner, "_successor_context", lambda _: "too short")
    with pytest.raises(ValueError, match="Information parity"):
        runner.build_conditions(source, digest, task=source.tasks[0])


def test_seeded_schedule_is_reproducible_and_opaque() -> None:
    source, digest = manifest()
    first = runner.build_schedule(source, digest, seed=91, replicates=1, blind_salt="scorer-secret")
    second = runner.build_schedule(source, digest, seed=91, replicates=1, blind_salt="scorer-secret")
    changed = runner.build_schedule(source, digest, seed=92, replicates=1, blind_salt="scorer-secret")
    assert [(trial.trial_id, trial.blind_id) for trial in first] == [(trial.trial_id, trial.blind_id) for trial in second]
    assert [trial.trial_id for trial in first] != [trial.trial_id for trial in changed]
    assert len(first) == 14 * 3
    assert all(trial.condition not in trial.trial_id and trial.condition not in trial.blind_id for trial in first)


def test_manifest_rejects_malformed_missing_duplicate_and_unknown_record_references(tmp_path: Path) -> None:
    malformed = tmp_path / "bad.json"
    malformed.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        runner.load_manifest(malformed, expected_sha256="__TO_BE_FILLED__")
    missing = write_variant(tmp_path, lambda raw: raw["records"].pop())
    with pytest.raises(ValueError, match="frozen E01"):
        runner.load_manifest(missing, expected_sha256="__TO_BE_FILLED__")
    duplicate = write_variant(tmp_path, lambda raw: raw["records"].append(dict(raw["records"][0])))
    with pytest.raises(ValueError, match="unique"):
        runner.load_manifest(duplicate, expected_sha256="__TO_BE_FILLED__")
    unknown = write_variant(tmp_path, lambda raw: raw["records"][0].update({"predecessor_id": "missing"}))
    with pytest.raises(ValueError, match="unknown predecessors"):
        runner.load_manifest(unknown, expected_sha256="__TO_BE_FILLED__")


def test_conflicting_outdated_and_uncertain_evidence_survive_b_and_c() -> None:
    source, digest = manifest()
    materials, parity = runner.build_conditions(source, digest, task=source.tasks[6])
    b_context = materials["B"].request.messages[0].content
    c_context = materials["C"].request.messages[0].content
    for evidence in ("record_id=E08", "uncertainty=meaning unresolved", "record_id=E09", "conflict=true", "record_id=E11", "superseded_by:E12"):
        assert evidence in b_context
        assert evidence in c_context
    assert parity.parity_established


class RecordingExecutor:
    def __init__(self) -> None:
        self.request_ids: list[int] = []

    async def execute(self, request):
        self.request_ids.append(id(request))
        return f"mock:{request.messages[-1].content}"


@pytest.mark.asyncio
async def test_trials_are_isolated_capture_raw_output_and_only_write_explicit_experiment_artifact(tmp_path: Path, monkeypatch) -> None:
    source, digest = manifest()
    task = source.tasks[0]
    materials, parity = runner.build_conditions(source, digest, task=task)
    schedule = [
        runner.Trial("trial-a", "blind-a", "A", task.id, task.prompt, 1, materials["A"].request),
        runner.Trial("trial-b", "blind-b", "B", task.id, task.prompt, 1, materials["B"].request),
    ]
    executor = RecordingExecutor()
    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    results = await runner.execute_trials(schedule, executor, now=now)
    assert executor.request_ids[0] != executor.request_ids[1]
    assert [result.raw_output for result in results] == [f"mock:{task.prompt}", f"mock:{task.prompt}"]
    assert all(result.error is None for result in results)
    results_root = tmp_path / "experiment-results"
    monkeypatch.setattr(runner, "RESULTS_ROOT", results_root)
    target = runner.write_results(results_root, manifest_digest=digest, seed=1, parity=parity, results=results)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["kind"] == "experiment_only_model_succession_results.v1"
    assert payload["results"][0]["raw_output"] == f"mock:{task.prompt}"
    with pytest.raises(ValueError, match="results area"):
        runner.write_results(tmp_path / "not-results", manifest_digest=digest, seed=1, parity=parity, results=results)


def test_native_payload_uses_same_deterministic_ollama_settings() -> None:
    source, digest = manifest()
    trial = runner.build_schedule(source, digest, seed=7, replicates=1, blind_salt="blind")[0]
    payload = runner.native_ollama_payload(trial)
    assert payload["model"] == "qwen3:4b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"] == {"temperature": 0}
