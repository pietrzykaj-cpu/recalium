"""Regression coverage for the experiment-only serial Ollama launcher."""
from __future__ import annotations

import asyncio
import json

import pytest

from experiments.model_succession_v1 import runner
from experiments.model_succession_v1.diagnostic import build_synthetic_diagnostic_trials, run_synthetic_diagnostics
from experiments.model_succession_v1.launcher import (
    ExecutionControls,
    SerialTrialLauncher,
    append_checkpoint_json,
)


def trial():
    manifest, digest = runner.load_manifest()
    return runner.build_schedule(manifest, digest, seed=1, replicates=1, blind_salt="diagnostic")[0]


class RecordingTransport:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response or {"message": {"content": "synthetic reply"}}
        self.error = error
        self.payloads: list[dict] = []

    async def post(self, payload: dict):
        self.payloads.append(payload)
        if self.error:
            raise self.error
        return self.response


def test_historical_wrong_boundary_reproduces_run2_attribute_error() -> None:
    scheduled = trial()
    with pytest.raises(AttributeError, match="ProviderChatRequest.*request"):
        runner.native_ollama_payload(scheduled.request)


@pytest.mark.asyncio
async def test_corrected_boundary_maps_request_once_and_checkpoints_one_response() -> None:
    scheduled = trial()
    transport = RecordingTransport()
    checkpoints = []
    launcher = SerialTrialLauncher(transport, controls=ExecutionControls(), checkpoint=checkpoints.append)

    record = await launcher.execute_once(scheduled)

    assert record.raw_output == "synthetic reply"
    assert record.error is None
    assert checkpoints == [record]
    assert len(transport.payloads) == 1
    assert transport.payloads[0]["model"] == scheduled.request.model
    assert transport.payloads[0]["options"] == {
        "temperature": 0, "seed": 20260915, "num_ctx": 4096, "num_predict": 512,
    }
    assert transport.payloads[0]["keep_alive"] == "0s"


@pytest.mark.asyncio
async def test_no_retry_and_exactly_one_checkpoint_on_transport_error() -> None:
    transport = RecordingTransport(error=ConnectionError("synthetic offline"))
    checkpoints = []
    launcher = SerialTrialLauncher(transport, controls=ExecutionControls(), checkpoint=checkpoints.append)

    record = await launcher.execute_once(trial())

    assert record.raw_output is None
    assert record.error == "ConnectionError: synthetic offline"
    assert len(transport.payloads) == 1
    assert checkpoints == [record]


@pytest.mark.asyncio
async def test_concurrent_attempt_is_rejected_and_first_attempt_still_checkpoints() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingTransport:
        async def post(self, payload: dict):
            entered.set()
            await release.wait()
            return {"message": {"content": "done"}}

    checkpoints = []
    launcher = SerialTrialLauncher(BlockingTransport(), controls=ExecutionControls(), checkpoint=checkpoints.append)
    first = asyncio.create_task(launcher.execute_once(trial()))
    await entered.wait()
    with pytest.raises(RuntimeError, match="one active trial"):
        await launcher.execute_once(trial())
    release.set()
    assert (await first).raw_output == "done"
    assert len(checkpoints) == 1


@pytest.mark.asyncio
async def test_json_checkpoint_writer_preserves_each_synthetic_response(tmp_path) -> None:
    path = tmp_path / "diagnostic-checkpoints.json"
    transport = RecordingTransport()
    launcher = SerialTrialLauncher(transport, controls=ExecutionControls(), checkpoint=append_checkpoint_json(path))
    await launcher.execute_once(trial())

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["raw_output"] == "synthetic reply"


@pytest.mark.asyncio
async def test_a_like_b_like_and_real_succession_diagnostics_are_synthetic_and_checkpointed(tmp_path) -> None:
    trials = build_synthetic_diagnostic_trials()
    assert [trial.condition for trial in trials] == ["A", "B", "C"]
    assert all("Aurora-17" not in trial.request.messages[0].content for trial in trials)
    assert "INHERITED EVIDENCE" not in trials[0].request.messages[0].content
    assert "Background context" in trials[1].request.messages[0].content
    assert "INHERITED EVIDENCE" in trials[2].request.messages[0].content

    transport = RecordingTransport()
    results = await run_synthetic_diagnostics(tmp_path / "checkpoints.json", transport)

    assert len(results) == len(transport.payloads) == 3
    assert all(result.error is None and result.raw_output == "synthetic reply" for result in results)
    assert len(json.loads((tmp_path / "checkpoints.json").read_text(encoding="utf-8"))) == 3
