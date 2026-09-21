from __future__ import annotations

import asyncio
import json

import pytest

from experiments.model_succession_v1 import v11
from experiments.model_succession_v1.launcher import ExecutionControls, SerialTrialLauncher


def test_authoritative_spec_and_complete_schedule_invariants() -> None:
    spec = v11.load_spec()
    schedule = v11.build_frozen_schedule(spec, blind_salt="synthetic-scorer-salt")
    assert len(schedule) == 420
    assert {x.condition for x in schedule} == {"A", "B", "C"}
    assert len({x.trial_id for x in schedule}) == len({x.blind_id for x in schedule}) == 420
    assert all(sum(x.task_id == task and x.condition == cond for x in schedule) == 10 for task in spec.design["task_ids"] for cond in spec.design["conditions"])


def test_yesterday_replicates_one_mismatch_is_refused_before_inference(monkeypatch) -> None:
    spec = v11.load_spec()
    original = v11.runner.build_schedule
    def one_replicate(*args, **kwargs):
        kwargs["replicates"] = 1
        return original(*args, **kwargs)
    monkeypatch.setattr(v11.runner, "build_schedule", one_replicate)
    with pytest.raises(ValueError, match="schedule length"):
        v11.build_frozen_schedule(spec, blind_salt="synthetic-scorer-salt")


def test_frozen_schedule_cannot_be_silently_regenerated(tmp_path) -> None:
    spec = v11.load_spec(); schedule = v11.build_frozen_schedule(spec, blind_salt="synthetic-scorer-salt")
    path = tmp_path / "schedule.json"
    document = v11.freeze_schedule(path, spec, schedule)
    assert document["schedule_sha256"]
    with pytest.raises(FileExistsError): v11.freeze_schedule(path, spec, schedule)


@pytest.mark.asyncio
async def test_all_420_synthetic_trials_follow_serial_launcher_and_complete() -> None:
    spec = v11.load_spec(); schedule = v11.build_frozen_schedule(spec, blind_salt="synthetic-scorer-salt")
    checkpoints = []
    launcher = SerialTrialLauncher(v11.DeterministicSyntheticTransport(), controls=ExecutionControls(), checkpoint=checkpoints.append)
    records = await v11.execute_frozen_synthetic(schedule, launcher)
    assert len(records) == len(checkpoints) == 420
    assert all(record.error is None and record.raw_output.startswith("synthetic:") for record in records)
    v11.validate_result_completeness(spec, schedule, records)


@pytest.mark.asyncio
async def test_resume_uses_only_the_uncompleted_frozen_suffix_and_writer_refuses_overwrite(tmp_path) -> None:
    spec = v11.load_spec(); schedule = v11.build_frozen_schedule(spec, blind_salt="synthetic-scorer-salt")
    writer = v11.FrozenCheckpointWriter(tmp_path / "checkpoints.json", schedule)
    launcher = SerialTrialLauncher(v11.DeterministicSyntheticTransport(), controls=ExecutionControls(), checkpoint=writer)
    first = await launcher.execute_once(schedule[0])
    assert v11.remaining_frozen_trials(schedule, [first])[0].trial_id == schedule[1].trial_id
    with pytest.raises(ValueError, match="already durably"):
        writer(first)


def test_scoring_preflight_refuses_the_historical_42_response_shape() -> None:
    spec = v11.load_spec(); schedule = v11.build_frozen_schedule(spec, blind_salt="synthetic-scorer-salt")
    with pytest.raises(ValueError, match="incomplete"):
        v11.validate_result_completeness(spec, schedule, [])
