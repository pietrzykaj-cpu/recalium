"""Experiment v1.1 apparatus: one executable authority, synthetic-safe validation."""
from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from experiments.model_succession_v1 import runner
from experiments.model_succession_v1.launcher import SerialTrialLauncher, TrialCheckpoint

ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "spec.v1.1.json"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


@dataclass(frozen=True)
class ExperimentSpec:
    raw: dict[str, Any]
    sha256: str

    @property
    def design(self) -> dict[str, Any]: return self.raw["design"]
    @property
    def execution(self) -> dict[str, Any]: return self.raw["execution"]


def load_spec(path: Path = SPEC_PATH) -> ExperimentSpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "recalium.model-succession-experiment-spec.v1.1":
        raise ValueError("unsupported experiment specification")
    spec = ExperimentSpec(raw=raw, sha256=canonical_sha256(raw))
    validate_spec(spec)
    return spec


def validate_spec(spec: ExperimentSpec) -> None:
    d, e = spec.design, spec.execution
    if len(d["task_ids"]) != 14 or set(d["conditions"]) != {"A", "B", "C"}:
        raise ValueError("v1.1 requires the frozen 14 tasks and A/B/C conditions")
    expected = len(d["task_ids"]) * len(d["conditions"]) * d["replicates_per_task_condition"]
    if d["replicates_per_task_condition"] != 10 or d["expected_total_responses"] != expected:
        raise ValueError("specification expected-trial invariant failed")
    if e["concurrency"] != 1 or e["automatic_retries"] or e["checkpoint_policy"] != "append-only-one-record-per-frozen-trial":
        raise ValueError("execution policy is incompatible with v1.1")


def build_frozen_schedule(spec: ExperimentSpec, *, blind_salt: str) -> list[runner.Trial]:
    manifest, digest = runner.load_manifest(expected_sha256=spec.raw["fixture"]["manifest_canonical_sha256"])
    schedule = runner.build_schedule(manifest, digest, seed=spec.raw["randomization"]["schedule_seed"], replicates=spec.design["replicates_per_task_condition"], blind_salt=blind_salt)
    validate_schedule(spec, schedule)
    return schedule


def validate_schedule(spec: ExperimentSpec, schedule: list[runner.Trial]) -> None:
    d = spec.design
    if len(schedule) != d["expected_total_responses"]:
        raise ValueError("generated schedule length does not equal specification expected total")
    if len({x.trial_id for x in schedule}) != len(schedule) or len({x.blind_id for x in schedule}) != len(schedule):
        raise ValueError("trial or blind IDs are not unique")
    tuples = [(x.task_id, x.condition, x.replicate) for x in schedule]
    if len(set(tuples)) != len(schedule):
        raise ValueError("duplicate task-condition-replicate tuple")
    expected = {(t, c, r) for t in d["task_ids"] for c in d["conditions"] for r in range(1, 11)}
    if set(tuples) != expected:
        raise ValueError("missing task-condition-replicate tuple")
    if Counter(x.condition for x in schedule) != Counter({"A": 140, "B": 140, "C": 140}):
        raise ValueError("condition totals violate specification")


def schedule_document(spec: ExperimentSpec, schedule: list[runner.Trial]) -> dict[str, Any]:
    rows = [{"order": i, "trial_id": x.trial_id, "blind_id": x.blind_id, "task_id": x.task_id, "condition": x.condition, "replicate": x.replicate} for i, x in enumerate(schedule, 1)]
    document = {"schema_version": spec.raw["output"]["schedule_schema"], "spec_sha256": spec.sha256, "schedule_generation_version": "v1.1.0", "schedule": rows}
    document["schedule_sha256"] = canonical_sha256(document)
    return document


def freeze_schedule(path: Path, spec: ExperimentSpec, schedule: list[runner.Trial]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError("frozen schedule already exists; regeneration refused")
    document = schedule_document(spec, schedule)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    return document


class SyntheticTransport(Protocol):
    async def post(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class DeterministicSyntheticTransport:
    """No HTTP and no model: deterministic payload-level stand-in."""
    async def post(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"message": {"content": "synthetic:" + _sha(json.dumps(payload, sort_keys=True).encode())[:16]}}


async def execute_frozen_synthetic(schedule: list[runner.Trial], launcher: SerialTrialLauncher) -> list[TrialCheckpoint]:
    completed: list[TrialCheckpoint] = []
    seen: set[str] = set()
    for trial in schedule:
        if trial.trial_id in seen:
            raise ValueError("frozen schedule attempted a duplicate trial")
        seen.add(trial.trial_id)
        completed.append(await launcher.execute_once(trial))
    return completed


def validate_result_completeness(spec: ExperimentSpec, schedule: list[runner.Trial], records: list[TrialCheckpoint]) -> None:
    if len(records) != spec.design["expected_total_responses"]:
        raise ValueError("incomplete result set; scoring refused")
    if len({x.trial_id for x in records}) != len(records):
        raise ValueError("duplicate result checkpoint; scoring refused")
    if {x.trial_id for x in records} != {x.trial_id for x in schedule}:
        raise ValueError("result IDs do not match frozen schedule; scoring refused")


def remaining_frozen_trials(schedule: list[runner.Trial], checkpoints: list[TrialCheckpoint]) -> list[runner.Trial]:
    """Resume only immutable schedule suffixes; reject unknown or duplicate records."""
    scheduled = {trial.trial_id for trial in schedule}
    completed = [record.trial_id for record in checkpoints]
    if len(completed) != len(set(completed)):
        raise ValueError("duplicate checkpoint would overwrite a completed frozen trial")
    if not set(completed) <= scheduled:
        raise ValueError("checkpoint is not part of the frozen schedule")
    expected_prefix = [trial.trial_id for trial in schedule[:len(completed)]]
    if completed != expected_prefix:
        raise ValueError("resume requires the exact completed schedule prefix")
    return schedule[len(completed):]


class FrozenCheckpointWriter:
    """Append-only checkpoint ledger for a single frozen schedule; no silent overwrite."""
    def __init__(self, path: Path, schedule: list[runner.Trial]) -> None:
        self.path, self._scheduled = path, {trial.trial_id for trial in schedule}

    def __call__(self, record: TrialCheckpoint) -> None:
        existing: list[dict[str, Any]] = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
        existing_ids = {entry["trial_id"] for entry in existing}
        if record.trial_id not in self._scheduled or record.trial_id in existing_ids:
            raise ValueError("checkpoint is unknown or already durably recorded")
        existing.append(asdict(record))
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.path)
