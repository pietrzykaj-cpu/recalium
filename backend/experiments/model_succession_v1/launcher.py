"""Serial, experiment-only Ollama launcher with per-trial checkpoints.

This module owns execution controls only.  It never constructs conditions, reads a
database, or persists Recalium memories.  Scientific inputs remain in the frozen
manifest/runner and are supplied as already-built ``Trial`` objects.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Awaitable, Callable, Protocol

import httpx

from experiments.model_succession_v1 import runner
from experiments.model_succession_v1.runner import Trial


class OllamaPayloadTransport(Protocol):
    async def post(self, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ExecutionControls:
    base_url: str = "http://127.0.0.1:11434"
    connect_timeout_seconds: float = 10.0
    http_read_timeout_seconds: float = 180.0
    watchdog_seconds: float = 210.0
    num_ctx: int = 4096
    num_predict: int = 512
    seed: int = 20260915
    keep_alive: str = "0s"


@dataclass(frozen=True)
class TrialCheckpoint:
    trial_id: str
    blind_id: str
    request_sha256: str
    raw_output: str | None
    error: str | None
    completed_at: str
    latency_seconds: float


class LocalOllamaTransport:
    """One loopback HTTP submission per attempt; no retry or shared chat state."""

    def __init__(self, controls: ExecutionControls) -> None:
        self._controls = controls

    async def post(self, payload: dict[str, Any]) -> dict[str, Any]:
        timeout = httpx.Timeout(
            self._controls.http_read_timeout_seconds,
            connect=self._controls.connect_timeout_seconds,
        )
        async with httpx.AsyncClient(timeout=timeout, trust_env=False, follow_redirects=False) as client:
            response = await client.post(f"{self._controls.base_url.rstrip('/')}/api/chat", json=payload)
        response.raise_for_status()
        parsed = response.json()
        if not isinstance(parsed, dict):
            raise ValueError("Ollama response must be a JSON object")
        return parsed


class SerialTrialLauncher:
    """Execute a supplied trial at most once and checkpoint its outcome exactly once."""

    def __init__(
        self,
        transport: OllamaPayloadTransport,
        *,
        controls: ExecutionControls,
        checkpoint: Callable[[TrialCheckpoint], None],
    ) -> None:
        self._transport = transport
        self._controls = controls
        self._checkpoint = checkpoint
        self._active = False

    @staticmethod
    def _request_sha256(trial: Trial) -> str:
        raw = json.dumps(trial.request.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _payload(self, trial: Trial) -> dict[str, Any]:
        payload = runner.native_ollama_payload_for_request(trial.request)
        payload["stream"] = False
        payload["think"] = False
        payload["keep_alive"] = self._controls.keep_alive
        payload.setdefault("options", {}).update({
            "temperature": 0,
            "seed": self._controls.seed,
            "num_ctx": self._controls.num_ctx,
            "num_predict": self._controls.num_predict,
        })
        return payload

    async def execute_once(self, trial: Trial) -> TrialCheckpoint:
        if self._active:
            raise RuntimeError("SerialTrialLauncher permits only one active trial")
        self._active = True
        started = perf_counter()
        output: str | None = None
        error: str | None = None
        try:
            payload = self._payload(trial)
            response = await asyncio.wait_for(
                self._transport.post(payload), timeout=self._controls.watchdog_seconds,
            )
            message = response.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str):
                raise ValueError("Ollama response has no text content")
            output = content
        except asyncio.TimeoutError:
            error = f"TimeoutError: per-trial watchdog exceeded {self._controls.watchdog_seconds:g} seconds"
        except Exception as exc:  # Preserve every technical outcome; do not retry.
            error = f"{type(exc).__name__}: {exc}"
        finally:
            self._active = False
        checkpoint = TrialCheckpoint(
            trial_id=trial.trial_id,
            blind_id=trial.blind_id,
            request_sha256=self._request_sha256(trial),
            raw_output=output,
            error=error,
            completed_at=datetime.now(timezone.utc).isoformat(),
            latency_seconds=round(perf_counter() - started, 6),
        )
        self._checkpoint(checkpoint)
        return checkpoint


def append_checkpoint_json(path: Path) -> Callable[[TrialCheckpoint], None]:
    """Return an atomic, experiment-artifact-only checkpoint writer."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint(record: TrialCheckpoint) -> None:
        existing: list[dict[str, Any]] = []
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
        existing.append(asdict(record))
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(existing, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    return checkpoint
