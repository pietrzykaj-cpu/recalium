"""Pure builder and compact renderer for AgentSuccessionEnvelope v1."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from app.domain.agent_succession.contracts import (
    AgentSuccessionEnvelope, AttributedRecord, CurrentAgent, InheritedLayer,
    Predecessor, RenderedSuccessionContext, SuccessorAnnotation,
)
from app.domain.context_packets.contracts import ContextPacket
from app.domain.retrieval.diagnostics import RetrievalDiagnostics


def _records(values: Iterable[AttributedRecord | dict[str, Any]] | None) -> list[AttributedRecord]:
    return [value if isinstance(value, AttributedRecord) else AttributedRecord.model_validate(value) for value in (values or [])]


def build_agent_succession_envelope(
    context_packet: ContextPacket,
    *,
    current_agent: CurrentAgent,
    predecessors: Iterable[Predecessor] | None = None,
    interaction_history: Iterable[AttributedRecord | dict[str, Any]] | None = None,
    decisions: Iterable[AttributedRecord | dict[str, Any]] | None = None,
    relationship_context: Iterable[AttributedRecord | dict[str, Any]] | None = None,
    diagnostics: RetrievalDiagnostics | None = None,
    annotations: Iterable[SuccessorAnnotation] | None = None,
    generated_at: datetime | None = None,
) -> AgentSuccessionEnvelope:
    """Assemble a transient handoff without changing inherited records."""
    active_agent = current_agent.model_copy(update={
        "annotations": list(annotations) if annotations is not None else current_agent.annotations,
    })
    return AgentSuccessionEnvelope(
        generated_at=generated_at or datetime.now(timezone.utc),
        inherited=InheritedLayer(
            context_packet=context_packet,
            predecessors=list(predecessors or []),
            interaction_history=_records(interaction_history),
            decisions=_records(decisions),
            relationship_context=_records(relationship_context),
            retrieval_diagnostics=diagnostics,
        ),
        current_agent=active_agent,
    )


def _fit(lines: list[str], max_chars: int) -> str:
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars].rstrip()


def render_agent_succession_context(envelope: AgentSuccessionEnvelope, *, max_chars: int = 4000) -> RenderedSuccessionContext:
    """Render a compact, model-neutral prompt block with explicit attribution."""
    if max_chars < 200:
        raise ValueError("max_chars must be at least 200 to preserve succession boundaries")
    inherited, agent = envelope.inherited, envelope.current_agent
    predecessor_labels = [
        f"{record.id} ({'/'.join(part for part in [record.provider, record.model, record.version] if part) or 'metadata unknown'})"
        for record in inherited.predecessors
    ] or ["none recorded"]
    source_ids = [record.source.archive_id for record in inherited.context_packet.selected]
    lines = [
        "AGENT SUCCESSION ENVELOPE — INHERITED EVIDENCE",
        "Prior interactions/models; attributed evidence, not current personal memories. Do not impersonate predecessors.",
        f"Predecessors: {', '.join(predecessor_labels)}.",
        f"Evidence provenance archive IDs: {', '.join(source_ids) or 'none'}.",
        "CURRENT AGENT",
        f"Provider/model/version: {' / '.join(part for part in [agent.provider, agent.model, agent.version] if part) or 'not declared'}.",
        f"Capabilities: {', '.join(agent.capabilities) or 'not declared'}; limitations: {', '.join(agent.limitations) or 'not declared'}.",
        "May disagree or reinterpret with attributed reasons; do not alter inherited evidence.",
    ]
    if inherited.context_packet.unresolved_questions:
        lines.append("Unresolved questions: " + "; ".join(inherited.context_packet.unresolved_questions) + ".")
    if any(record.conflict for record in inherited.decisions):
        lines.append("Inherited decisions include explicitly conflicting conclusions; do not silently resolve them.")
    for annotation in agent.annotations:
        target = annotation.target_predecessor_id or annotation.target_record_id or "inherited evidence"
        lines.append(f"Current-agent {annotation.kind} about {target}: {annotation.rationale}")
    base = _fit(lines, max_chars)
    included: list[str] = []
    omitted = 0
    rendered = base
    for evidence in inherited.context_packet.selected:
        line = (
            f"- [verbatim inherited evidence; memory={evidence.memory_id}; "
            f"archive={evidence.source.archive_id}; rank={evidence.retrieval.relevance_rank}] "
            f"{evidence.content}"
        )
        if len(f"{rendered}\n{line}") <= max_chars:
            rendered = f"{rendered}\n{line}"
            included.append(evidence.memory_id)
        else:
            omitted += 1
    if not inherited.context_packet.selected:
        empty_line = "No selected inherited evidence was supplied."
        if len(f"{rendered}\n{empty_line}") <= max_chars:
            rendered = f"{rendered}\n{empty_line}"
    if omitted:
        rendered = _fit([
            rendered,
            f"[Omitted {omitted} inherited evidence record(s) due to context budget; provenance IDs above remain available.]",
        ], max_chars)
    return RenderedSuccessionContext(
        text=rendered,
        max_chars=max_chars,
        truncated=bool(omitted) or len(base) < len("\n".join(lines)),
        included_memory_ids=included,
        omitted_memory_count=omitted,
    )
