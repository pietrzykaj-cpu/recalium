"""Synthetic-only launcher diagnostics; intentionally unrelated to Experiment v1 fixtures."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.domain.agent_succession.contracts import CurrentAgent, Predecessor
from app.domain.agent_succession.service import build_agent_succession_envelope, render_agent_succession_context
from app.domain.context_packets.service import build_context_packet
from app.domain.model_context.contracts import ProviderChatRequest, ProviderMessage
from app.domain.retrieval.service import RetrievalItem, RetrievalResponse
from experiments.model_succession_v1.launcher import (
    ExecutionControls,
    OllamaPayloadTransport,
    SerialTrialLauncher,
    TrialCheckpoint,
    append_checkpoint_json,
)
from experiments.model_succession_v1.runner import Trial


DIAGNOSTIC_ITEM = "A synthetic quartz moth rests beside a cedar basin."
DIAGNOSTIC_TIME = datetime(2026, 9, 15, tzinfo=timezone.utc)


def build_synthetic_diagnostic_trials() -> list[Trial]:
    """Return A-like/B-like/C-like requests without loading Experiment v1 material."""
    fresh = ProviderChatRequest(
        model="qwen3:4b",
        messages=[
            ProviderMessage(role="system", content="This is a synthetic launcher diagnostic. Answer the user directly."),
            ProviderMessage(role="user", content="Reply with a brief greeting to a quartz moth."),
        ],
    )
    ordinary = ProviderChatRequest(
        model="qwen3:4b",
        messages=[
            ProviderMessage(
                role="system",
                content=f"Background context for the current task: {DIAGNOSTIC_ITEM} Use this context when relevant.",
            ),
            ProviderMessage(role="user", content="Where does the synthetic quartz moth rest?"),
        ],
    )
    item = RetrievalItem(
        id="diagnostic-quartz-moth",
        type="fact",
        content=DIAGNOSTIC_ITEM,
        score=1.0,
        source_id="diagnostic-source",
        source_system="launcher-diagnostic",
        captured_at="2026-09-15T00:00:00Z",
        conflict_label=None,
        provenance={"synthetic": True, "purpose": "launcher-boundary-diagnostic"},
    )
    packet = build_context_packet(
        RetrievalResponse(
            query="synthetic quartz moth",
            retrieval_mode="hybrid",
            budget_used=len(item.content),
            budget_limit=500,
            trimming_reason="result_exhausted",
            items=[item],
        ),
        generated_at=DIAGNOSTIC_TIME,
    )
    envelope = build_agent_succession_envelope(
        packet,
        current_agent=CurrentAgent(provider="ollama", model="qwen3:4b"),
        predecessors=[Predecessor(id="diagnostic-source", provider="synthetic", model="diagnostic")],
        generated_at=DIAGNOSTIC_TIME,
    )
    succession = ProviderChatRequest(
        model="qwen3:4b",
        messages=[
            ProviderMessage(role="system", content=render_agent_succession_context(envelope, max_chars=2000).text),
            ProviderMessage(
                role="user",
                content="According to the attributed diagnostic evidence, where does the quartz moth rest?",
            ),
        ],
    )
    requests = [("A", fresh), ("B", ordinary), ("C", succession)]
    return [
        Trial(
            trial_id=f"diagnostic-{condition.lower()}",
            blind_id=f"diagnostic-blind-{condition.lower()}",
            condition=condition,
            task_id=f"diagnostic-{condition.lower()}",
            user_prompt=request.messages[-1].content,
            replicate=1,
            request=request,
        )
        for condition, request in requests
    ]


async def run_synthetic_diagnostics(
    output_path: Path,
    transport: OllamaPayloadTransport,
    *,
    controls: ExecutionControls = ExecutionControls(),
) -> list[TrialCheckpoint]:
    """Submit exactly three serial synthetic requests and checkpoint each once."""
    launcher = SerialTrialLauncher(
        transport,
        controls=controls,
        checkpoint=append_checkpoint_json(output_path),
    )
    results: list[TrialCheckpoint] = []
    for trial in build_synthetic_diagnostic_trials():
        results.append(await launcher.execute_once(trial))
    return results
