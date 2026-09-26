"""Three memory operations with authenticated, server-resolved space access."""

import hashlib
import json
from datetime import datetime, timezone
from dataclasses import asdict

from sqlalchemy import select, text

from app.domain.archive.models import RawArchiveItem
from app.domain.audit.models import AuditEvent
from app.domain.bridge.access import permitted_spaces
from app.domain.authority.repository import evaluate_scope
from app.domain.bridge.contracts import ContextPacketInput, ContinuityHandoffInput, CurrentAuthorityInput, IngestInput, RetrieveInput, StatusInput
from app.domain.bridge.models import (
    BridgeAliasReceipt,
    BridgeArchive,
    BridgeBinding,
    BridgeClient,
    BridgeReceipt,
)
from app.domain.canonical_memory.models import CanonicalMemoryItem
from app.domain.context_packets.contracts import ContextPacket
from app.domain.context_packets.service import build_context_packet
from app.domain.agent_succession.contracts import AttributedRecord
from app.domain.agent_succession.service import build_agent_succession_envelope, render_agent_succession_context
from app.domain.derived_memory.models import Fact, Summary
from app.domain.ingest.service import ingest_text_content
from app.domain.jobs.models import Job
from app.domain.retrieval.diagnostics import RetrievalDiagnostics
from app.domain.retrieval.service import (
    RetrievalFilters,
    RetrievalItem,
    RetrievalRequest,
    RetrievalResponse,
    retrieve,
)


class BridgeError(Exception):
    def __init__(self, status, code):
        self.status, self.code = status, code
        super().__init__(code)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def audit(session, actor, operation, project, outcome, **details):
    session.add(
        AuditEvent(
            event_type="bridge_access",
            actor=actor,
            operation_metadata=dict(
                operation=operation, project_id=project, outcome=outcome, **details
            ),
        )
    )


async def _lock_receipt(session, *parts):
    value = int.from_bytes(
        hashlib.sha256(json.dumps(parts).encode()).digest()[:8], "big", signed=True
    )
    await session.execute(text("SELECT pg_advisory_xact_lock(:lock)"), {"lock": value})


async def _destination(session, actor, req):
    if req.project_id is not None:
        return req.project_id, None
    if req.space_id is not None:
        return req.space_id, None
    await _lock_receipt(session, actor, "alias", req.destination, digest(req.idempotency_key))
    pinned = await session.get(
        BridgeAliasReceipt, (actor, req.destination, digest(req.idempotency_key))
    )
    if pinned:
        return pinned.project_id, pinned
    binding = await session.get(BridgeBinding, (actor, req.destination))
    if binding is None:
        raise BridgeError(403, "destination_unavailable")
    return binding.project_id, None


async def execute(session, authorization, operation, request):
    actor = "unauthenticated"
    resolved = []
    destination = None
    try:
        scheme, _, credential = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not 32 <= len(credential) <= 256:
            raise BridgeError(401, "authentication_required")
        principal = (
            await session.execute(
                select(BridgeClient)
                .where(
                    BridgeClient.credential_digest == digest(credential),
                    BridgeClient.active.is_(True),
                )
                .with_for_update(read=True)
            )
        ).scalar_one_or_none()
        if principal is None:
            raise BridgeError(401, "authentication_required")
        actor = principal.id
        if operation == "ingest_memory":
            destination, pinned = await _destination(session, actor, request)
            spaces = await permitted_spaces(session, actor, [destination], write=True)
            if destination not in spaces:
                raise BridgeError(403, "permission_denied")
            space = spaces[destination]
            if request.destination is not None and request.destination != space.kind:
                raise BridgeError(403, "destination_unavailable")
            resolved = [destination]
            response = await _ingest(
                session, actor, request, destination, pinned=pinned is not None
            )
            if request.destination is not None and request.space_id is None and pinned is None:
                session.add(
                    BridgeAliasReceipt(
                        client_id=actor,
                        destination=request.destination,
                        request_digest=digest(request.idempotency_key),
                        project_id=destination,
                    )
                )
            response["memory_space"] = _space_label(space)
        else:
            requested = (
                [request.space_id]
                if operation in ("get_current_authority", "build_continuity_handoff")
                else (
                    [request.project_id]
                    if request.project_id is not None
                    else getattr(request, "space_ids", None)
                )
            )
            spaces = await permitted_spaces(session, actor, requested)
            if requested is not None and not set(requested).issubset(spaces):
                raise BridgeError(403, "permission_denied")
            resolved = sorted(spaces)
            if operation == "retrieve_memory":
                response = await _retrieve(session, actor, request, spaces)
            elif operation == "build_context_packet":
                response = await _context_packet(session, actor, request, spaces)
            elif operation == "get_ingest_status":
                response = await _status(session, request, spaces)
            elif operation == "get_current_authority":
                response = await _current_authority(session, request, spaces)
            elif operation == "build_continuity_handoff":
                response = await _continuity_handoff(session, actor, request, spaces)
            else:
                raise BridgeError(400, "unknown_operation")
        audit(
            session,
            actor,
            operation,
            destination or getattr(request, "project_id", None) or getattr(request, "space_id", None),
            "allowed",
            searched_space_ids=(
                resolved
                if operation in ("retrieve_memory", "build_context_packet", "build_continuity_handoff")
                else []
            ),
            resolved_destination=destination,
            result_count=len(response.get("items", response.get("selected", []))),
            replay=response.get("idempotent_replay", False),
        )
        await session.commit()
        return response
    except BridgeError as exc:
        await session.rollback()
        audit(session, actor, operation, None, "denied", code=exc.code)
        await session.commit()
        raise
    except Exception:
        await session.rollback()
        audit(session, actor, operation, None, "error")
        await session.commit()
        raise


def _space_label(space):
    return {"id": space.id, "kind": space.kind}


async def _ingest(session, actor, req: IngestInput, destination, *, pinned=False):
    request_digest = digest(req.idempotency_key)
    # Same canonical payload as v1: aliases cannot change receipt meaning.
    payload = {
        "project_id": destination,
        "content": req.content,
        "source_metadata": req.source_metadata.model_dump(),
    }
    fingerprint = digest(json.dumps(payload, sort_keys=True))
    await _lock_receipt(session, actor, "space", destination, request_digest)
    receipt = await session.get(BridgeReceipt, (actor, destination, request_digest))
    if pinned and receipt is None:
        raise BridgeError(409, "source_unavailable")
    if receipt:
        if receipt.payload_digest != fingerprint:
            raise BridgeError(409, "idempotency_conflict")
        archive = await session.get(RawArchiveItem, receipt.archive_id)
        assignment = await session.get(BridgeArchive, receipt.archive_id)
        if (
            archive is None
            or archive.deleted_at is not None
            or assignment is None
            or assignment.project_id != destination
        ):
            raise BridgeError(409, "source_unavailable")
        archive_id = receipt.archive_id
    else:
        result = await ingest_text_content(
            session,
            req.content,
            actor=actor,
            source_type="bridge",
            source_name=req.source_metadata.source_name,
            extra_metadata={
                "client_identity": actor,
                "source_metadata": req.source_metadata.model_dump(),
                "project_hint": destination,
                "processing_mode": "local_only",
                "import_method": "memory_bridge_v1",
            },
            commit=False,
        )
        archive_id = result.archive_ids[0]
        session.add(BridgeArchive(archive_id=archive_id, project_id=destination, client_id=actor))
        session.add(
            BridgeReceipt(
                client_id=actor,
                project_id=destination,
                request_digest=request_digest,
                payload_digest=fingerprint,
                archive_id=archive_id,
            )
        )
    return {
        "status": "accepted",
        "archive_id": str(archive_id),
        "project_id": destination,
        "client_identity": actor,
        "idempotent_replay": receipt is not None,
    }


async def _processing(session, item):
    """Read recorded processing provenance; never infer it from the search engine."""
    processing = {"method": None, "model": None}
    if item["type"] == "canonical":
        cm = (
            await session.execute(
                select(CanonicalMemoryItem).where(
                    CanonicalMemoryItem.id == item["id"],
                    CanonicalMemoryItem.raw_archive_id == item["source_id"],
                    CanonicalMemoryItem.source_status == "active",
                    CanonicalMemoryItem.status == "active",
                )
            )
        ).scalar_one_or_none()
        if cm is None:
            raise BridgeError(403, "source_scope_violation")
        if cm.fact_id is None:
            return {"method": "canonical_promotion", "model": None}
        fact_id = cm.fact_id
    elif item["type"] == "fact":
        fact_id = item["id"]
    else:
        fact_id = None
    if fact_id is not None:
        fact = (
            await session.execute(
                select(Fact).where(
                    Fact.id == fact_id,
                    Fact.raw_archive_id == item["source_id"],
                    Fact.source_status == "active",
                    Fact.review_status == "active",
                )
            )
        ).scalar_one_or_none()
        if fact is None:
            raise BridgeError(403, "source_scope_violation")
        processing = {"method": fact.derivation_method, "model": fact.derivation_model}
    elif item["type"] == "summary":
        summary = (
            await session.execute(
                select(Summary)
                .where(
                    Summary.raw_archive_id == item["source_id"],
                    Summary.source_status == "active",
                    Summary.summary_text == item["content"],
                )
                .order_by(Summary.created_at, Summary.id)
                .limit(1)
            )
        ).scalar_one_or_none()
        if summary is None:
            raise BridgeError(403, "source_scope_violation")
        processing = {"method": summary.derivation_method, "model": summary.model_used}
    return processing



def _authority_record_payload(record, *, include_provenance):
    payload = record.model_dump(mode="json")
    if not include_provenance:
        payload.pop("provenance", None)
    return payload


async def _current_authority(session, req: CurrentAuthorityInput, spaces):
    if req.space_id not in spaces:
        raise BridgeError(403, "permission_denied")
    result = await evaluate_scope(
        session,
        space_id=req.space_id,
        workstream_id=req.workstream_id,
        authority_key=req.authority_key,
    )
    return {
        "status": result.status,
        "space_id": result.space_id,
        "workstream_id": result.workstream_id,
        "authority_key": result.authority_key,
        "current_record": (
            _authority_record_payload(result.current_record, include_provenance=req.include_provenance)
            if result.current_record is not None else None
        ),
        "competing_records": (
            [_authority_record_payload(r, include_provenance=req.include_provenance) for r in result.competing_records]
            if req.include_competing else []
        ),
        "historical_records": (
            [_authority_record_payload(r, include_provenance=req.include_provenance) for r in result.historical_records]
            if req.include_historical else []
        ),
        "superseded_record_ids": result.superseded_record_ids if req.include_historical else [],
        "deterministic": {"currentness": "graph_derived", "source": "persisted_authority_records"},
    }


def _stable_unique(values):
    return list(dict.fromkeys(value for value in values if value))


def _iter_authority_records(result, name):
    value = result.get(name)
    if name == "current_record":
        return [value] if isinstance(value, dict) else []
    return value if isinstance(value, list) else []


def _snapshot_time(authority_results):
    values = []
    for result in authority_results:
        for name in ("current_record", "competing_records", "historical_records"):
            for record in _iter_authority_records(result, name):
                value = record.get("created_at")
                if value:
                    try:
                        values.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
                    except (TypeError, ValueError):
                        pass
    return max(values, default=datetime(1970, 1, 1, tzinfo=timezone.utc))


def _authority_decisions(authority_results, *, include_historical):
    records = []
    seen = set()
    for result in authority_results:
        buckets = ["current_record", "competing_records"]
        if include_historical:
            buckets.append("historical_records")
        for bucket in buckets:
            for record in _iter_authority_records(result, bucket):
                if record["id"] in seen:
                    continue
                seen.add(record["id"])
                records.append(AttributedRecord(
                    id=record["id"],
                    content=record["content"],
                    predecessor_ids=list(record.get("superseded_record_ids") or []),
                    provenance=record.get("provenance") or {},
                    conflict=result["status"] == "ambiguous",
                    uncertainty=("historical_or_superseded" if bucket == "historical_records" else None),
                ))
    return records

def _authority_lines(authority_results, *, include_historical):
    lines = ["AUTHORITATIVE CURRENT STATE"]
    for result in authority_results:
        key = result["authority_key"]
        status = result["status"].upper()
        if result["status"] == "current":
            current = result["current_record"]
            lines.append(f"- {key}: CURRENT record {current['id']}: {current['content']}")
        elif result["status"] == "ambiguous":
            ids = ", ".join(record["id"] for record in result.get("competing_records") or [])
            lines.append(f"- {key}: AMBIGUOUS; competing records: {ids or 'unavailable'}; do not guess.")
        else:
            lines.append(f"- {key}: {status}; no authoritative current record.")
        if include_historical and result.get("historical_records"):
            ids = ", ".join(record["id"] for record in result["historical_records"])
            lines.append(f"  Historical/superseded records: {ids}.")
    return lines


async def _continuity_handoff(session, actor, req: ContinuityHandoffInput, spaces):
    """Assemble an authorized, transient continuity handoff without persistence or providers."""
    if req.space_id not in spaces:
        raise BridgeError(403, "permission_denied")
    authority_results = []
    for key in req.authority_keys:
        authority_results.append(await _current_authority(
            session,
            CurrentAuthorityInput(
                space_id=req.space_id,
                workstream_id=req.workstream_id,
                authority_key=key,
                include_historical=req.include_historical,
                include_provenance=req.include_provenance,
                include_competing=req.include_competing,
            ),
            spaces,
        ))
    packet_request = ContextPacketInput(
        space_ids=[req.space_id],
        query=req.query,
        mode=req.mode,
        budget=req.budget,
        limit=req.limit,
        token_budget=req.token_budget,
        current_provider=req.current_agent.provider,
        current_model=req.current_agent.model,
        include_diagnostics=req.include_diagnostics,
    )
    packet_payload = await _context_packet(
        session,
        actor,
        packet_request,
        spaces,
        generated_at=_snapshot_time(authority_results),
    )
    diagnostics_payload = packet_payload.pop("retrieval_diagnostics", None)
    if diagnostics_payload is not None:
        diagnostics_payload["generated_at"] = _snapshot_time(authority_results).isoformat()
    packet = ContextPacket.model_validate(packet_payload)
    envelope = build_agent_succession_envelope(
        packet,
        current_agent=req.current_agent,
        predecessors=req.predecessors,
        decisions=_authority_decisions(authority_results, include_historical=req.include_historical),
        diagnostics=(RetrievalDiagnostics.model_validate(diagnostics_payload) if diagnostics_payload is not None else None),
        generated_at=packet.generated_at,
    )
    base_budget = max(200, req.render_max_chars - 1200)
    rendered_base = render_agent_succession_context(envelope, max_chars=base_budget)
    lines = _authority_lines(authority_results, include_historical=req.include_historical)
    lines.extend([
        "SUPPORTING MEMORY (NON-AUTHORITATIVE)",
        rendered_base.text,
    ])
    rendered_text = "\n".join(lines)
    if len(rendered_text) > req.render_max_chars:
        rendered_text = rendered_text[:req.render_max_chars].rstrip()
    rendered = rendered_base.model_copy(update={
        "text": rendered_text,
        "max_chars": req.render_max_chars,
        "truncated": rendered_base.truncated or len(rendered_text) < len("\n".join(lines)),
    })
    warnings = []
    for result in authority_results:
        if result["status"] == "ambiguous":
            warnings.append(f"authority_ambiguous:{result['authority_key']}")
        elif result["status"] == "empty":
            warnings.append(f"authority_empty:{result['authority_key']}")
    warnings.extend(packet.warnings)
    warnings.extend("unresolved_question_present" for _ in packet.unresolved_questions)
    warnings.extend(f"supporting_memory_excluded:{item.memory_id}" for item in packet.excluded)
    if not packet.selected:
        warnings.append("no_supporting_memory")
    provenance = None
    if req.include_provenance:
        provenance = {
            "space_id": req.space_id,
            "workstream_id": req.workstream_id,
            "authority_keys": list(req.authority_keys),
            "searched_space_ids": [req.space_id],
            "authority_source": "persisted_authority_records",
            "memory_source": "ordinary_authorized_retrieval",
        }
    return {
        "authority_results": authority_results,
        "context_packet": packet.model_dump(mode="json"),
        "included_memory_ids": [item.memory_id for item in packet.selected],
        "excluded_memory_ids": [item.memory_id for item in packet.excluded],
        "unresolved_questions": list(packet.unresolved_questions),
        "flags": list(packet.flags),
        "retrieval_diagnostics": diagnostics_payload,
        "provenance": provenance,
        "packet_integrity": packet.integrity.model_dump(mode="json"),
        "predecessors": [item.model_dump(mode="json") for item in req.predecessors],
        "current_agent": req.current_agent.model_dump(mode="json"),
        "succession_envelope": envelope.model_dump(mode="json"),
        "rendered_handoff": rendered.model_dump(mode="json"),
        "warnings": _stable_unique(warnings),
    }

async def _retrieve(session, actor, req: RetrieveInput, spaces, diagnostics=None):
    if spaces:
        response = asdict(
            await retrieve(
                session,
                RetrievalRequest(
                    query=req.query,
                    mode=req.mode,
                    budget=req.budget,
                    limit=req.limit,
                    actor=actor,
                    filters=RetrievalFilters(bridge_project_ids=tuple(sorted(spaces))),
                ),
                diagnostics=diagnostics,
            )
        )
    else:
        response = {
            "query": req.query,
            "retrieval_mode": req.mode,
            "budget_used": 0,
            "budget_limit": req.budget,
            "trimming_reason": "result_exhausted",
            "items": [],
            "degraded_mode": False,
        }
    for item in response["items"]:
        row = (
            await session.execute(
                select(BridgeArchive, RawArchiveItem)
                .join(RawArchiveItem, RawArchiveItem.id == BridgeArchive.archive_id)
                .where(
                    BridgeArchive.archive_id == item["source_id"],
                    BridgeArchive.project_id.in_(spaces),
                    RawArchiveItem.deleted_at.is_(None),
                )
                .with_for_update(read=True, of=(BridgeArchive, RawArchiveItem))
            )
        ).first()
        if row is None:
            raise BridgeError(403, "source_scope_violation")
        assignment, archive = row
        if item.get("source_fact_id"):
            source = (
                await session.execute(
                    select(Fact.id)
                    .join(BridgeArchive, BridgeArchive.archive_id == Fact.raw_archive_id)
                    .join(RawArchiveItem, RawArchiveItem.id == Fact.raw_archive_id)
                    .where(
                        Fact.id == item["source_fact_id"],
                        BridgeArchive.project_id.in_(spaces),
                        Fact.source_status == "active",
                        Fact.review_status == "active",
                        RawArchiveItem.deleted_at.is_(None),
                    )
                )
            ).scalar_one_or_none()
            if source is None:
                raise BridgeError(403, "source_scope_violation")
        previous = item["provenance"]
        item["provenance"] = {
            "authenticated_client": assignment.client_id,
            "source_metadata": (archive.metadata_json or {}).get("source_metadata", {}),
            "project_id": assignment.project_id,
            "processing": await _processing(session, item),
            "retrieval": {
                "method": previous.get("derivation_method"),
                "model": previous.get("derivation_model"),
            },
            "source_excerpt": previous.get("source_excerpt", ""),
        }
        item["memory_space"] = _space_label(spaces[assignment.project_id])
    response["searched_space_ids"] = sorted(spaces)
    response["project_id"] = req.project_id
    return response


async def _status(session, req: StatusInput, spaces):
    row = (
        await session.execute(
            select(RawArchiveItem, BridgeArchive)
            .join(BridgeArchive, BridgeArchive.archive_id == RawArchiveItem.id)
            .where(
                RawArchiveItem.id == req.archive_id,
                BridgeArchive.project_id.in_(spaces),
                RawArchiveItem.deleted_at.is_(None),
            )
        )
    ).first()
    if row is None:
        raise BridgeError(404, "not_found")
    archive, assignment = row
    jobs = (
        (
            await session.execute(
                select(Job).where(Job.raw_archive_id == archive.id).order_by(Job.created_at)
            )
        )
        .scalars()
        .all()
    )
    return {
        "archive_id": str(archive.id),
        "project_id": assignment.project_id,
        "memory_space": _space_label(spaces[assignment.project_id]),
        "jobs": [
            {
                "job_id": str(j.id),
                "status": j.status,
                "attempts": j.attempts,
                "has_error": bool(j.error_message),
            }
            for j in jobs
        ],
    }


async def _context_packet(session, actor, req: ContextPacketInput, spaces, generated_at=None):
    """Retrieve, authorize, enrich provenance, then build a transient packet."""
    diagnostics = None
    if req.include_diagnostics:
        from app.domain.retrieval.diagnostics import RetrievalDiagnosticsCollector
        diagnostics = RetrievalDiagnosticsCollector(
            mode=req.mode,
            filters={},
            memory_space_ids=sorted(spaces),
        )
    if diagnostics is None:
        enriched = await _retrieve(session, actor, req, spaces)
    else:
        enriched = await _retrieve(session, actor, req, spaces, diagnostics=diagnostics)
    retrieval = RetrievalResponse(
        query=enriched["query"],
        retrieval_mode=enriched["retrieval_mode"],
        budget_used=enriched["budget_used"],
        budget_limit=enriched["budget_limit"],
        trimming_reason=enriched["trimming_reason"],
        degraded_mode=enriched["degraded_mode"],
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
                source_fact_id=item.get("source_fact_id"),
                link_type=item.get("link_type"),
            )
            for item in enriched["items"]
        ],
    )
    diagnostic_snapshot = diagnostics.snapshot() if diagnostics else None
    packet = build_context_packet(
        retrieval,
        token_budget=req.token_budget,
        provider=req.current_provider,
        model=req.current_model,
        diagnostics=diagnostic_snapshot,
        generated_at=generated_at,
    )
    result = packet.model_dump(mode="json")
    if diagnostic_snapshot:
        result["retrieval_diagnostics"] = diagnostic_snapshot.model_dump(mode="json")
    return result
