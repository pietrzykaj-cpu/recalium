"""Pure builder and compact renderer for AgentSuccessionEnvelope v1."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.domain.agent_succession.contracts import (
    AgentSuccessionEnvelope,
    AttributedRecord,
    CurrentAgent,
    InheritedLayer,
    Predecessor,
    RenderedSuccessionContext,
    SuccessorAnnotation,
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


MANDATORY_RENDER_OVERFLOW_NOTICE = (
    "Mandatory rendered continuity exceeded render_max_chars; use structured authority."
)


@dataclass(frozen=True)
class _AtomicRenderSection:
    """Renderer-owned complete section; optional sections are never sliced."""

    text: str
    reducible: bool = False
    memory_id: str | None = None


def _evidence_block(evidence, label: Callable[[str], str], *, encode_values: bool) -> str:
    header = (
        f"- [verbatim inherited evidence; memory={label(evidence.memory_id)}; "
        f"archive={label(evidence.source.archive_id)}; rank={evidence.retrieval.relevance_rank}] "
    )
    if encode_values:
        from app.domain.model_context.encoding import render_body_lines

        return header.rstrip() + "\n" + "\n".join(render_body_lines(evidence.content))
    return header + evidence.content


def _strict_sections(lines, agent, evidence, label, prose, *, encode_values):
    """Expose atomic sections in display order, separating optional capabilities.

    The shared base lines retain ownership of framing/provenance/identity and
    all unresolved/conflicting/annotation lines. Default legacy layout is not
    changed; only strict layout separates capabilities from limitations.
    """
    sections = [
        *(_AtomicRenderSection(line) for line in lines[:3]),
        _AtomicRenderSection(lines[3], reducible=True),
        *(_AtomicRenderSection(line) for line in lines[4:6]),
        _AtomicRenderSection(
            f"Capabilities: {', '.join(prose(part) for part in agent.capabilities) or 'not declared'}.",
            reducible=True,
        ),
        _AtomicRenderSection(
            f"Limitations: {', '.join(prose(part) for part in agent.limitations) or 'not declared'}.",
        ),
        *(_AtomicRenderSection(line) for line in lines[7:]),
        *(_AtomicRenderSection(
            _evidence_block(item, label, encode_values=encode_values),
            reducible=True, memory_id=item.memory_id,
        ) for item in evidence),
    ]
    if not evidence:
        sections.append(_AtomicRenderSection("No selected inherited evidence was supplied."))
    return sections


def _render_strict_sections(sections, *, max_chars, prefix_text):
    """Measure actual composed characters; no _fit/rstrip/final slice.

    Priority is mandatory sections, then ranked memory blocks, then other
    optional material (aggregate provenance, capabilities). A rejected item
    does not prevent a later smaller whole item from fitting. Every proposal
    includes its exact atomic omission notice before being accepted.
    """
    if max_chars < len(MANDATORY_RENDER_OVERFLOW_NOTICE):
        raise ValueError("strict budget must fit the fixed mandatory-overflow notice")
    selected_count = sum(section.memory_id is not None for section in sections)

    def compose(indices):
        missing = [section for i, section in enumerate(sections) if i not in indices]
        omitted = sum(section.memory_id is not None for section in missing)
        metadata = len(missing) - omitted
        parts = [section.text for i, section in enumerate(sections) if i in indices]
        if missing:
            parts.append(
                f"[Omitted {omitted} inherited evidence record(s) and {metadata} optional section(s) "
                "due to context budget; structured handoff remains complete.]"
            )
        return prefix_text + "\n".join(parts), omitted, bool(missing)

    all_indices = set(range(len(sections)))
    full, _, _ = compose(all_indices)
    if len(full) <= max_chars:
        return RenderedSuccessionContext(
            text=full, max_chars=max_chars, truncated=False,
            included_memory_ids=[section.memory_id for section in sections if section.memory_id is not None],
            omitted_memory_count=0,
        )
    included = {i for i, section in enumerate(sections) if not section.reducible}
    mandatory, _, _ = compose(included)
    if len(mandatory) > max_chars:
        return RenderedSuccessionContext(
            text=MANDATORY_RENDER_OVERFLOW_NOTICE, max_chars=max_chars, truncated=True,
            included_memory_ids=[], omitted_memory_count=selected_count,
        )
    optional_indices = [i for i, section in enumerate(sections) if section.reducible]
    optional_indices.sort(key=lambda i: sections[i].memory_id is None)
    for i in optional_indices:
        proposal, _, _ = compose(included | {i})
        if len(proposal) <= max_chars:
            included.add(i)
    text, omitted, truncated = compose(included)
    return RenderedSuccessionContext(
        text=text, max_chars=max_chars, truncated=truncated,
        included_memory_ids=[
            section.memory_id for i, section in enumerate(sections)
            if i in included and section.memory_id is not None
        ],
        omitted_memory_count=omitted,
    )


def render_agent_succession_context(
    envelope: AgentSuccessionEnvelope, *, max_chars: int = 4000, encode_values: bool = False,
    strict: bool = False, prefix_text: str = "",
) -> RenderedSuccessionContext:
    """Render attributed context; opt-in encoding contains untrusted Bridge values.

    Default rendering remains the legacy byte format for existing consumers.
    Budgets count the representation actually emitted, including encoded bodies.
    Strict mode composes atomic sections plus a trusted, already-encoded prefix
    within the full budget, or returns the field-level mandatory-overflow notice.
    """
    if not strict and prefix_text:
        raise ValueError("prefix_text requires strict rendering")
    if not strict and max_chars < 200:
        raise ValueError("max_chars must be at least 200 to preserve succession boundaries")
    label: Callable[[str], str]
    prose: Callable[[str], str]
    if encode_values:
        # Keep codec/package loading out of the default legacy import path.
        from app.domain.model_context.encoding import encode_label, encode_prose

        label, prose = encode_label, encode_prose
    else:
        label = prose = lambda value: value
    inherited, agent = envelope.inherited, envelope.current_agent
    predecessor_labels = [
        f"{label(record.id)} ({'/'.join(label(part) for part in [record.provider, record.model, record.version] if part) or 'metadata unknown'})"
        for record in inherited.predecessors
    ] or ["none recorded"]
    source_ids = [label(record.source.archive_id) for record in inherited.context_packet.selected]
    lines = [
        "AGENT SUCCESSION ENVELOPE — INHERITED EVIDENCE",
        "Prior interactions/models; attributed evidence, not current personal memories. Do not impersonate predecessors.",
        f"Predecessors: {', '.join(predecessor_labels)}.",
        f"Evidence provenance archive IDs: {', '.join(source_ids) or 'none'}.",
        "CURRENT AGENT",
        f"Provider/model/version: {' / '.join(label(part) for part in [agent.provider, agent.model, agent.version] if part) or 'not declared'}.",
        f"Capabilities: {', '.join(prose(part) for part in agent.capabilities) or 'not declared'}; limitations: {', '.join(prose(part) for part in agent.limitations) or 'not declared'}.",
        "May disagree or reinterpret with attributed reasons; do not alter inherited evidence.",
    ]
    if inherited.context_packet.unresolved_questions:
        lines.append("Unresolved questions: " + "; ".join(prose(part) for part in inherited.context_packet.unresolved_questions) + ".")
    if any(record.conflict for record in inherited.decisions):
        lines.append("Inherited decisions include explicitly conflicting conclusions; do not silently resolve them.")
    for annotation in agent.annotations:
        target = annotation.target_predecessor_id or annotation.target_record_id or "inherited evidence"
        lines.append(f"Current-agent {annotation.kind} about {label(target)}: {prose(annotation.rationale)}")
    if strict:
        sections = _strict_sections(
            lines, agent, inherited.context_packet.selected, label, prose, encode_values=encode_values,
        )
        return _render_strict_sections(sections, max_chars=max_chars, prefix_text=prefix_text)
    base = _fit(lines, max_chars)
    included: list[str] = []
    omitted = 0
    rendered = base
    for evidence in inherited.context_packet.selected:
        line = _evidence_block(evidence, label, encode_values=encode_values)
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
