"""Pure, experiment-only A/B/C runner; intentionally outside production Recalium."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.domain.agent_succession.contracts import CurrentAgent, Predecessor
from app.domain.agent_succession.service import build_agent_succession_envelope, render_agent_succession_context
from app.domain.context_packets.service import build_context_packet
from app.domain.model_context.contracts import ProviderChatRequest, ProviderMessage
from app.domain.model_context.ollama import ollama_chat_payload
from app.domain.retrieval.service import RetrievalItem, RetrievalResponse


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = ROOT / "manifest.v1.json"
RESULTS_ROOT = ROOT / "results"
MANIFEST_SCHEMA = "recalium.model-succession-experiment-manifest.v1"
EXPECTED_MANIFEST_SHA256 = "f74c5e8a908e9523e094cf7da21ce341fee15947d07397bd3b8f4c03411fdf26"
CONDITIONS = ("A", "B", "C")
REQUIRED_RECORD_IDS = frozenset(f"E{number:02d}" for number in range(1, 17))
REQUIRED_TASK_IDS = frozenset(f"T{number:02d}" for number in range(1, 15))


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SyntheticRecord(ManifestModel):
    id: str
    category: str
    predecessor_id: str
    content: str
    provenance: dict[str, Any]
    uncertainty: str | None = None
    conflict: bool = False
    unresolved_questions: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class ExperimentTask(ManifestModel):
    id: str
    prompt: str


class ManifestSettings(ManifestModel):
    stream: Literal[False] = False
    think: Literal[False] = False
    temperature: Literal[0] = 0
    max_context_chars: int = Field(ge=200)
    chars_per_token_estimate: int = Field(ge=1)


class ExperimentManifest(ManifestModel):
    schema_version: Literal["recalium.model-succession-experiment-manifest.v1"]
    manifest_version: str
    experiment_id: str
    generated_at: datetime
    current_agent: dict[str, Any]
    predecessors: list[Predecessor]
    records: list[SyntheticRecord]
    tasks: list[ExperimentTask]
    settings: ManifestSettings

    @model_validator(mode="after")
    def unique_ids_and_references(self) -> "ExperimentManifest":
        record_ids = [record.id for record in self.records]
        task_ids = [task.id for task in self.tasks]
        predecessor_ids = {entry.id for entry in self.predecessors}
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("Synthetic record IDs must be unique")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Task IDs must be unique")
        if set(record_ids) != REQUIRED_RECORD_IDS:
            raise ValueError("Manifest must contain exactly the frozen E01–E16 record set")
        if set(task_ids) != REQUIRED_TASK_IDS:
            raise ValueError("Manifest must contain exactly the frozen T01–T14 task set")
        unknown = sorted({record.predecessor_id for record in self.records} - predecessor_ids)
        if unknown:
            raise ValueError(f"Records reference unknown predecessors: {', '.join(unknown)}")
        if not self.records or not self.tasks:
            raise ValueError("Manifest requires records and tasks")
        return self


@dataclass(frozen=True)
class ExposedRecord:
    record_id: str
    record_digest: str
    category: str
    predecessor_id: str


@dataclass(frozen=True)
class PromptMetrics:
    characters: int
    estimated_tokens: int
    sha256: str


@dataclass(frozen=True)
class ParityReport:
    manifest_sha256: str
    b_records: tuple[ExposedRecord, ...]
    c_records: tuple[ExposedRecord, ...]
    parity_established: bool
    failures: tuple[str, ...]
    b_metrics: PromptMetrics
    c_metrics: PromptMetrics


@dataclass(frozen=True)
class ConditionMaterial:
    condition: Literal["A", "B", "C"]
    request: ProviderChatRequest
    exposed_records: tuple[ExposedRecord, ...]
    rendered_context: str | None


@dataclass(frozen=True)
class Trial:
    trial_id: str
    blind_id: str
    condition: Literal["A", "B", "C"]
    task_id: str
    user_prompt: str
    replicate: int
    request: ProviderChatRequest


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    blind_id: str
    condition: Literal["A", "B", "C"]
    task_id: str
    replicate: int
    request_sha256: str
    raw_output: str | None
    error: str | None
    completed_at: str


class TrialExecutor(Protocol):
    async def execute(self, request: ProviderChatRequest) -> str: ...


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _sha(value: Any) -> str:
    payload = value if isinstance(value, str) else _canonical_json(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _record_digest(record: SyntheticRecord) -> str:
    return _sha(record.model_dump(mode="json"))


def manifest_sha256(raw_manifest: dict[str, Any]) -> str:
    """Digest canonical parsed manifest content, independent of file whitespace."""
    return _sha(raw_manifest)


def load_manifest(
    path: Path = DEFAULT_MANIFEST_PATH,
    *,
    expected_sha256: str = EXPECTED_MANIFEST_SHA256,
) -> tuple[ExperimentManifest, str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load experiment manifest: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Experiment manifest must be a JSON object")
    digest = manifest_sha256(raw)
    if expected_sha256 != "__TO_BE_FILLED__" and digest != expected_sha256:
        raise ValueError("Experiment manifest hash does not match the frozen expected hash")
    try:
        manifest = ExperimentManifest.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Malformed experiment manifest: {exc}") from exc
    return manifest, digest


def _exposed(manifest: ExperimentManifest) -> tuple[ExposedRecord, ...]:
    return tuple(
        ExposedRecord(record.id, _record_digest(record), record.category, record.predecessor_id)
        for record in manifest.records
    )


def _evidence_text(record: SyntheticRecord) -> str:
    """One canonical evidence atom used verbatim by both B and C."""
    fields = [
        f"record_id={record.id}",
        f"category={record.category}",
        f"predecessor={record.predecessor_id}",
        f"content={record.content}",
        f"provenance={_canonical_json(record.provenance)}",
    ]
    if record.uncertainty is not None:
        fields.append(f"uncertainty={record.uncertainty}")
    if record.conflict:
        fields.append("conflict=true")
    if record.unresolved_questions:
        fields.append(f"unresolved_questions={_canonical_json(record.unresolved_questions)}")
    if record.flags:
        fields.append(f"flags={_canonical_json(record.flags)}")
    return " | ".join(fields)


def _metrics(text: str, chars_per_token: int) -> PromptMetrics:
    return PromptMetrics(len(text), (len(text) + chars_per_token - 1) // chars_per_token, _sha(text))


def _current_agent(manifest: ExperimentManifest) -> CurrentAgent:
    return CurrentAgent.model_validate(manifest.current_agent)


def _ordinary_context(manifest: ExperimentManifest) -> str:
    return "\n".join([
        "Background context for the current task. Use it when relevant. If it is insufficient or conflicting, say so.",
        *[f"- {_evidence_text(record)}" for record in manifest.records],
    ])


def _successor_context(manifest: ExperimentManifest) -> str:
    items = [
        RetrievalItem(
            id=record.id,
            type="fact",
            content=_evidence_text(record),
            score=1.0,
            source_id=str(record.provenance["source_id"]),
            source_system="synthetic-experiment-manifest",
            captured_at=str(record.provenance["captured_at"]),
            conflict_label="conflicting" if record.conflict else None,
            provenance={
                "category": record.category,
                "predecessor_id": record.predecessor_id,
                "source_metadata": {
                    "unresolved_questions": record.unresolved_questions,
                    "flags": record.flags,
                    "uncertainty": record.uncertainty,
                    "conflict": record.conflict,
                },
                "manifest_provenance": record.provenance,
            },
        )
        for record in manifest.records
    ]
    response = RetrievalResponse(
        query="Model Succession Experiment v1 frozen evidence",
        retrieval_mode="hybrid",
        budget_used=sum(len(item.content) for item in items),
        budget_limit=sum(len(item.content) for item in items),
        trimming_reason="result_exhausted",
        items=items,
    )
    packet = build_context_packet(
        response,
        token_budget=manifest.settings.max_context_chars,
        chars_per_token=1,
        provider=_current_agent(manifest).provider,
        model=_current_agent(manifest).model,
        generated_at=manifest.generated_at,
    )
    decisions = [
        {
            "id": record.id,
            "content": record.content,
            "predecessor_ids": [record.predecessor_id],
            "provenance": record.provenance,
            "uncertainty": record.uncertainty,
            "conflict": record.conflict,
        }
        for record in manifest.records
        if record.category in {"explicit_decision", "predecessor_interpretation_wrong", "conflicting_evidence"}
    ]
    relationship_context = [
        {
            "id": record.id,
            "content": record.content,
            "predecessor_ids": [record.predecessor_id],
            "provenance": record.provenance,
        }
        for record in manifest.records
        if record.category == "relationship_context"
    ]
    envelope = build_agent_succession_envelope(
        packet,
        current_agent=_current_agent(manifest),
        predecessors=manifest.predecessors,
        decisions=decisions,
        relationship_context=relationship_context,
        generated_at=manifest.generated_at,
    )
    rendered = render_agent_succession_context(envelope, max_chars=manifest.settings.max_context_chars)
    if rendered.truncated or tuple(rendered.included_memory_ids) != tuple(record.id for record in manifest.records):
        raise ValueError("C cannot expose every frozen record inside the declared context budget")
    return rendered.text


def build_conditions(manifest: ExperimentManifest, manifest_digest: str, *, task: ExperimentTask) -> tuple[dict[str, ConditionMaterial], ParityReport]:
    """Derive A/B/C from exactly one manifest; raise before execution on parity failure."""
    model = _current_agent(manifest).model
    if not model:
        raise ValueError("Experiment manifest current agent must declare a model")
    fresh_system = "Answer the current user task directly and accurately."
    ordinary = _ordinary_context(manifest)
    succession = _successor_context(manifest)
    exposed = _exposed(manifest)
    materials: dict[str, ConditionMaterial] = {
        "A": ConditionMaterial("A", ProviderChatRequest(model=model, messages=[ProviderMessage(role="system", content=fresh_system), ProviderMessage(role="user", content=task.prompt)]), (), None),
        "B": ConditionMaterial("B", ProviderChatRequest(model=model, messages=[ProviderMessage(role="system", content=ordinary), ProviderMessage(role="user", content=task.prompt)]), exposed, ordinary),
        "C": ConditionMaterial("C", ProviderChatRequest(model=model, messages=[ProviderMessage(role="system", content=succession), ProviderMessage(role="user", content=task.prompt)]), exposed, succession),
    }
    failures: list[str] = []
    if materials["A"].exposed_records:
        failures.append("A unexpectedly exposes inherited records")
    if materials["B"].exposed_records != materials["C"].exposed_records:
        failures.append("B/C record ID, order, digest, category, or predecessor parity failed")
    for record in manifest.records:
        atom = _evidence_text(record)
        if atom not in ordinary:
            failures.append(f"B does not expose canonical evidence atom {record.id}")
        if atom not in succession:
            failures.append(f"C does not expose canonical evidence atom {record.id}")
    if "AGENT SUCCESSION ENVELOPE" in ordinary or "inherited evidence" in ordinary.lower():
        failures.append("B contains Recalium succession wording")
    if any(material.request.messages[-1].content != task.prompt for material in materials.values()):
        failures.append("Conditions do not expose the identical user task")
    report = ParityReport(
        manifest_sha256=manifest_digest,
        b_records=materials["B"].exposed_records,
        c_records=materials["C"].exposed_records,
        parity_established=not failures,
        failures=tuple(failures),
        b_metrics=_metrics(ordinary, manifest.settings.chars_per_token_estimate),
        c_metrics=_metrics(succession, manifest.settings.chars_per_token_estimate),
    )
    if not report.parity_established:
        raise ValueError("Information parity is not established: " + "; ".join(report.failures))
    return materials, report


def build_schedule(manifest: ExperimentManifest, manifest_digest: str, *, seed: int, replicates: int, blind_salt: str) -> list[Trial]:
    if replicates < 1:
        raise ValueError("replicates must be at least one")
    if not blind_salt:
        raise ValueError("blind_salt is required for opaque blinded IDs")
    entries: list[tuple[ExperimentTask, str, int]] = [
        (task, condition, replicate)
        for task in manifest.tasks
        for condition in CONDITIONS
        for replicate in range(1, replicates + 1)
    ]
    random.Random(seed).shuffle(entries)
    trials: list[Trial] = []
    for ordinal, (task, condition, replicate) in enumerate(entries, start=1):
        materials, _ = build_conditions(manifest, manifest_digest, task=task)
        stable = f"{manifest_digest}:{seed}:{ordinal}:{task.id}:{condition}:{replicate}"
        trials.append(Trial(
            trial_id="trial-" + _sha(stable)[:20],
            blind_id="blind-" + _sha(f"{blind_salt}:{stable}")[:20],
            condition=condition, task_id=task.id, user_prompt=task.prompt,
            replicate=replicate, request=materials[condition].request,
        ))
    return trials


async def execute_trials(trials: list[Trial], executor: TrialExecutor, *, now: datetime | None = None) -> list[TrialResult]:
    """Execute isolated requests through an injected test/live transport; never persist."""
    results: list[TrialResult] = []
    for trial in trials:
        output: str | None = None
        error: str | None = None
        try:
            output = await executor.execute(trial.request)
            if not isinstance(output, str):
                raise ValueError("Trial executor returned a non-text output")
        except Exception as exc:  # results must preserve experimental failures
            error = f"{type(exc).__name__}: {exc}"
        completed = now or datetime.now(timezone.utc)
        results.append(TrialResult(
            trial_id=trial.trial_id, blind_id=trial.blind_id, condition=trial.condition,
            task_id=trial.task_id, replicate=trial.replicate,
            request_sha256=_sha(trial.request.model_dump(mode="json")), raw_output=output,
            error=error, completed_at=completed.isoformat(),
        ))
    return results


def write_results(results_dir: Path, *, manifest_digest: str, seed: int, parity: ParityReport, results: list[TrialResult]) -> Path:
    """Explicitly write experiment artifacts only; no automatic result persistence exists."""
    root = RESULTS_ROOT.resolve()
    candidate = results_dir.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Experiment results must remain under the experiment-only results area")
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / f"run-{manifest_digest[:12]}-seed-{seed}.json"
    payload = {
        "kind": "experiment_only_model_succession_results.v1",
        "manifest_sha256": manifest_digest,
        "seed": seed,
        "parity": asdict(parity),
        "results": [asdict(result) for result in results],
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return target


def native_ollama_payload(trial: Trial) -> dict[str, Any]:
    """Expose the same deterministic local Ollama payload used by production adapter tests."""
    return native_ollama_payload_for_request(trial.request)


def native_ollama_payload_for_request(request: ProviderChatRequest) -> dict[str, Any]:
    """Map an already-constructed request without requiring experiment trial metadata.

    Launchers execute a ``ProviderChatRequest`` after schedule construction.  Keeping
    this boundary explicit prevents the historical mistake of passing that request to
    ``native_ollama_payload()``, which intentionally accepts a ``Trial``.
    """
    return ollama_chat_payload(request)
