"""Short fenced coordinator transactions; no model/provider work in this module."""
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import uuid
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert
from app.domain.archive.models import RawArchiveItem, Tombstone
from app.domain.jobs.models import Job
from app.domain.derived_memory.models import Fact, Summary, Embedding
from app.domain.derivations.models import ArchiveDerivationHead as Head, DerivationGeneration as Generation, DerivationStageCompletion as Completion, DerivationFtsProjection as Projection
from app.domain.derivations.visibility import enabled, HISTORY

class DerivationConflict(Exception):
    """Bounded ownership/lifecycle conflict; never expose raw provider errors."""

@dataclass(frozen=True)
class Claim:
    job_id: uuid.UUID
    archive_id: uuid.UUID
    generation_id: uuid.UUID
    attempts: int
    claimed_at: datetime
    policy_intent: tuple | None = None

def require_enabled():
    if not enabled():
        raise DerivationConflict("Managed derivation is dormant")

@asynccontextmanager
async def transaction(session):
    # Callers must snapshot and finish previous prepare/audit work before this.
    await session.rollback()
    try:
        yield
        await session.commit()
    except BaseException:
        await session.rollback()
        raise

async def lock_archives_and_heads(session, archive_ids):
    ids = sorted(set(archive_ids), key=str)
    sources = (await session.scalars(select(RawArchiveItem).where(
        RawArchiveItem.id.in_(ids)).order_by(RawArchiveItem.id).with_for_update()
        .execution_options(populate_existing=True, autoflush=False))).all()
    for aid in ids:
        await session.execute(insert(Head).values(raw_archive_id=aid).on_conflict_do_nothing())
    heads = (await session.scalars(select(Head).where(Head.raw_archive_id.in_(ids))
        .order_by(Head.raw_archive_id).with_for_update()
        .execution_options(populate_existing=True, autoflush=False))).all()
    return {r.id: r for r in sources}, {r.raw_archive_id: r for r in heads}

async def source_live(session, source):
    if source is None or source.deleted_at is not None:
        return False
    tombstone = await session.scalar(select(Tombstone.id).where(
        or_source_tombstone(source)).limit(1))
    return tombstone is None

def or_source_tombstone(source):
    from sqlalchemy import or_
    return or_(Tombstone.source_id == source.id, Tombstone.content_hash == source.content_hash)

async def context(session, claim):
    require_enabled()
    sources, heads = await lock_archives_and_heads(session, [claim.archive_id])
    job = await session.scalar(select(Job).where(Job.id == claim.job_id).with_for_update()
        .execution_options(populate_existing=True, autoflush=False))
    # Lock predecessor and candidate in deterministic order for publication too.
    generations = (await session.scalars(select(Generation).where(
        Generation.raw_archive_id == claim.archive_id).order_by(Generation.id)
        .with_for_update().execution_options(populate_existing=True, autoflush=False))).all()
    gen = next((g for g in generations if g.id == claim.generation_id), None)
    source, head = sources.get(claim.archive_id), heads.get(claim.archive_id)
    if (job is None or gen is None or head is None or not await source_live(session, source)
        or job.raw_archive_id != claim.archive_id or job.generation_id != claim.generation_id
        or job.status != "claimed" or job.attempts != claim.attempts or job.claimed_at != claim.claimed_at
        or gen.original_job_id != job.id or gen.source_content_hash != source.content_hash
        or (claim.policy_intent is not None and claim.policy_intent != (
            (source.metadata_json or {}).get('processing_mode'),
            (source.metadata_json or {}).get('sensitivity_hint')))
        or gen.contract_version != 1
        or gen.status not in ("building", "prepared", "active")
        or (gen.status == "active" and head.active_generation_id != gen.id)
        or (gen.status != "active" and head.candidate_generation_id != gen.id)):
        raise DerivationConflict("Derivation ownership or source changed")
    return source, head, job, gen, generations

async def admit_generation(session, job_id, *, manual=False):
    require_enabled()
    # Only scalar IDs escape the unlocked lookup; never cached status validation.
    aid = await session.scalar(select(Job.raw_archive_id).where(Job.id == job_id)
        .execution_options(autoflush=False))
    await session.rollback()
    if aid is None:
        return None
    result = None
    async with transaction(session):
        sources, heads = await lock_archives_and_heads(session, [aid])
        source, head = sources.get(aid), heads[aid]
        owners = (await session.scalars(select(Generation.original_job_id).where(
            Generation.id.in_([g for g in (head.candidate_generation_id, head.active_generation_id) if g])))).all()
        job_ids = sorted({job_id, *owners}, key=str)
        jobs = (await session.scalars(select(Job).where(Job.id.in_(job_ids))
            .order_by(Job.id).with_for_update().execution_options(
                populate_existing=True, autoflush=False))).all()
        job = next((j for j in jobs if j.id == job_id), None)
        gens = (await session.scalars(select(Generation).where(
            Generation.raw_archive_id == aid).order_by(Generation.id).with_for_update()
            .execution_options(populate_existing=True, autoflush=False))).all()
        candidate = next((g for g in gens if g.id == head.candidate_generation_id), None)
        if job is None or job.raw_archive_id != aid or not await source_live(session, source):
            raise DerivationConflict("Job or source is unavailable")
        if manual and job.status not in {"failed", "retryable_failed", "pending_provider"}:
            raise DerivationConflict("Job cannot be reprocessed in its current state")
        if not manual and job.status != "claimed":
            raise DerivationConflict("Job is not claimed")
        if any(j.id != job_id and j.status == 'claimed' for j in jobs):
            raise DerivationConflict('Another archive owner is in flight')
        candidate_owner = next((j for j in jobs if candidate and j.id == candidate.original_job_id), None)
        if candidate and candidate_owner and candidate_owner.status == "failed":
            candidate.status = "failed"
            head.candidate_generation_id = None
            candidate = None
            await session.flush()
        if not manual and job.generation_id is not None:
            result = next((g for g in gens if g.id == job.generation_id), None)
            if result is None or result.status not in {"building", "prepared", "active"}:
                raise DerivationConflict("Existing generation is not resumable")
            if result.status == "active" and head.active_generation_id != result.id:
                raise DerivationConflict("Published generation is no longer current")
            if result.status != "active" and head.candidate_generation_id != result.id:
                raise DerivationConflict("Candidate is no longer admitted")
        else:
            if candidate:
                if not manual or candidate.original_job_id != job.id:
                    raise DerivationConflict("Archive already has an open candidate")
                candidate.status = "abandoned"
                head.candidate_generation_id = None
                await session.flush()
            result = Generation(raw_archive_id=aid, original_job_id=job.id,
                sequence=max((g.sequence for g in gens), default=0)+1,
                creation_reason="manual_reprocess" if manual else "automatic_processing",
                source_content_hash=source.content_hash, predecessor_generation_id=head.active_generation_id)
            session.add(result)
            await session.flush()
            job.generation_id = result.id
            head.candidate_generation_id = result.id
            if manual:
                job.status, job.attempts, job.error_message = "pending", 0, None
                job.updated_at = datetime.now(timezone.utc)
    return result

async def stage(session, generation_id, name):
    return await session.get(Completion, (generation_id, name), populate_existing=True)

async def persist_extraction(session, claim, facts_data, *, outcome="succeeded"):
    require_enabled()
    async with transaction(session):
        _, _, _, gen, _ = await context(session, claim)
        completion = await stage(session, gen.id, "extraction")
        if completion is None:
            if gen.status != "building" or outcome not in {"succeeded", "policy_skipped"}:
                raise DerivationConflict("Extraction is not writable")
            from app.domain.derived_memory.service import write_facts, write_tags, upsert_tag_names
            facts = await write_facts(session, claim.archive_id, facts_data,
                                      commit=False, generation_id=gen.id)
            tag_batches = [(data.get("tags") or []) + [
                    "entity:"+e.strip().lower() for e in data.get("entities") or [] if e.strip()]
                for data in facts_data]
            tag_ids = await upsert_tag_names(session, [tag for batch in tag_batches for tag in batch])
            for fact, tags in zip(facts, tag_batches):
                await write_tags(session, fact.id, tags, commit=False, prepared_tag_ids=tag_ids)
            session.add(Completion(generation_id=gen.id, stage_name="extraction",
                outcome=outcome, fact_count=len(facts)))
            await session.flush()
        facts = (await session.scalars(select(Fact).where(Fact.generation_id == gen.id)
            .order_by(Fact.id).execution_options(**HISTORY))).all()
    return facts

async def persist_summary(session, claim, summary_text, *, outcome="succeeded",
                          model_used="unknown", method="llm_summarization",
                          partial=None):
    require_enabled()
    async with transaction(session):
        _, _, job, gen, _ = await context(session, claim)
        completion = await stage(session, gen.id, "summary")
        if completion is None:
            if gen.status != "building":
                raise DerivationConflict("Summary is not writable")
            audit_id = category = exception_class = None
            if outcome == "partial_failure":
                if job.attempts < job.max_attempts or partial is None:
                    raise DerivationConflict("Partial summary requires final-attempt evidence")
                from app.domain.audit.models import AuditEvent
                category, exception_class, provider = partial
                if category not in {"provider_transport", "provider_response", "response_parsing"}:
                    raise DerivationConflict("Partial summary category is not allowlisted")
                audit = AuditEvent(event_type="summary_failed", raw_archive_id=claim.archive_id,
                    actor="pipeline_worker", operation_metadata={
                        "job_id": str(job.id), "stage":"summary", "attempt": job.attempts,
                        "max_attempts":job.max_attempts, "final_attempt":True,
                        "failure_category":category, "exception_class":exception_class,
                        "provider":provider if provider in {"ollama","openai","anthropic"} else None,
                        "continuing_to_extraction":True})
                session.add(audit)
                await session.flush()
                audit_id = audit.id
            if summary_text:
                session.add(Summary(raw_archive_id=claim.archive_id, generation_id=gen.id,
                    summary_text=summary_text, model_used=model_used, derivation_method=method))
            session.add(Completion(generation_id=gen.id, stage_name="summary", outcome=outcome,
                audit_event_id=audit_id, failure_category=category, exception_class=exception_class))
    return await stage(session, claim.generation_id, "summary")

async def prepare_fts(session, claim, text_content, projection_kind):
    require_enabled()
    async with transaction(session):
        _, _, _, gen, _ = await context(session, claim)
        row = await session.scalar(select(Projection).where(Projection.generation_id == gen.id))
        if row is None:
            if gen.status != "building" or await stage(session, gen.id, "summary") is None or await stage(session, gen.id, "extraction") is None:
                raise DerivationConflict("Required stages not complete")
            row = Projection(raw_archive_id=claim.archive_id, generation_id=gen.id,
                text_content=text_content, projection_kind=projection_kind)
            session.add(row)
            gen.status = "prepared"
            await session.flush()
        elif row.source_status != "active":
            raise DerivationConflict("Projection is suppressed")
    return row

async def publish_generation(session, claim):
    require_enabled()
    changed = False
    async with transaction(session):
        _, head, _, gen, gens = await context(session, claim)
        projection = await session.scalar(select(Projection).where(Projection.generation_id == gen.id))
        summary, extraction = await stage(session, gen.id, "summary"), await stage(session, gen.id, "extraction")
        fact_count = await session.scalar(select(func.count(Fact.id)).where(
            Fact.generation_id == gen.id).execution_options(**HISTORY))
        summary_id = await session.scalar(select(Summary.id).where(
            Summary.generation_id == gen.id).execution_options(**HISTORY))
        if (summary is None or extraction is None or extraction.fact_count != fact_count
            or (summary.outcome == 'succeeded') != (summary_id is not None)):
            raise DerivationConflict("Required durable outputs are inconsistent")
        if gen.status == "active":
            if head.legacy_retired_at is None or projection is None or projection.source_status != "active":
                raise DerivationConflict("Published state is inconsistent")
        else:
            if (gen.status != "prepared" or summary is None or extraction is None or projection is None
                or projection.source_status != "active" or head.active_generation_id != gen.predecessor_generation_id):
                raise DerivationConflict("Candidate is not publishable")
            stamp = datetime.now(timezone.utc)
            previous = next((g for g in gens if g.id == head.active_generation_id), None)
            if previous:
                previous.status, previous.superseded_at = "superseded", stamp
                await session.flush()  # release partial active uniqueness before new active
            head.active_generation_id, head.candidate_generation_id = gen.id, None
            if head.legacy_retired_at is None:
                head.legacy_retired_at = stamp
            gen.status, gen.published_at = "active", stamp
            changed = True
    return changed

async def snapshot(session, claim):
    """Immutable inputs/checkpoint metadata, with all prepare locks released."""
    from copy import deepcopy
    async with transaction(session):
        source, _, job, gen, _ = await context(session, claim)
        summary = await session.scalar(select(Summary).where(Summary.generation_id == gen.id).execution_options(**HISTORY))
        embedding = await session.scalar(select(Embedding).where(Embedding.generation_id == gen.id).execution_options(**HISTORY))
        summary_stage = await stage(session, gen.id, "summary")
        extract_stage = await stage(session, gen.id, "extraction")
        data = dict(raw=source.raw_content, metadata=deepcopy(source.metadata_json or {}),
            status=gen.status, max_attempts=job.max_attempts,
            summary=summary.summary_text if summary else None,
            summary_stage=(summary_stage.outcome, summary_stage.failure_category, summary_stage.exception_class) if summary_stage else None,
            extraction_complete=extract_stage is not None,
            embedding=[float(v) for v in embedding.embedding] if embedding else None)
    return data

async def persist_embedding(session, claim, vector):
    async with transaction(session):
        _, _, _, gen, _ = await context(session, claim)
        if gen.status != "active":
            raise DerivationConflict("Enrichment requires active generation")
        if await stage(session, gen.id, "embedding") is None:
            from app.domain.derived_memory.service import ACTIVE_EMBEDDING_MODEL
            session.add(Embedding(raw_archive_id=claim.archive_id, generation_id=gen.id,
                embedding=vector, embedding_model=ACTIVE_EMBEDDING_MODEL))
            session.add(Completion(generation_id=gen.id, stage_name="embedding", outcome="succeeded"))

async def write_policy(session, claim, decision, policy, permissions):
    from app.domain.audit.models import AuditEvent
    async with transaction(session):
        source, _, _, _, _ = await context(session, claim)
        source.metadata_json = {**(source.metadata_json or {}), "data_class": decision.category}
        session.add(AuditEvent(event_type="sensitivity_gate", raw_archive_id=claim.archive_id,
            actor="pipeline_worker", operation_metadata={"category":decision.category,
                "confidence":round(decision.confidence,4), "blocked":decision.blocked, "method":decision.method}))
        allowed = {v["provider"] for v in permissions.values() if v["allowed"]}
        session.add(AuditEvent(event_type="policy_decision", raw_archive_id=claim.archive_id,
            actor="pipeline_worker", operation_metadata={"allow_external":policy.allow_external,
                "processing_mode":policy.processing_mode, "sensitivity_hint":policy.sensitivity_hint,
                "data_class":decision.category, "provider":next(iter(allowed)) if len(allowed)==1 else None,
                "operations":permissions,"reason":policy.reason}))

async def raw_baseline(session, claim):
    """Initial/no-provider baseline support only, never a bulk rebuild."""
    async with transaction(session):
        source, head, _, _, _ = await context(session, claim)
        if head.active_generation_id is not None or head.legacy_retired_at is not None:
            return
        row = await session.scalar(select(Projection).where(
            Projection.raw_archive_id == claim.archive_id, Projection.generation_id.is_(None)))
        if row is None:
            session.add(Projection(raw_archive_id=claim.archive_id, generation_id=None,
                projection_kind="raw_fallback", text_content=source.raw_content[:10000]))
        elif row.source_status != "active":
            raise DerivationConflict("Baseline is suppressed")

async def finish_claim(session, claim, *, pending_provider=False, partial_marker=None):
    async with transaction(session):
        _, _, job, gen, _ = await context(session, claim)
        if pending_provider:
            if gen.status != "building":
                raise DerivationConflict("Only unfinished generation may await provider")
            job.status = "pending_provider"
            job.error_message = "No LLM provider configured. Add an OpenAI, Anthropic, or Ollama key in Settings."
        else:
            if gen.status != "active":
                raise DerivationConflict("Completion requires published generation")
            job.status = "completed"
            job.completed_at = datetime.now(timezone.utc)
            job.error_message = "partial: "+partial_marker if partial_marker else None
        job.updated_at = datetime.now(timezone.utc)

async def fail_current_claim(session, job_id, attempts, claimed_at, *, error, retryable=True, generation_id=None):
    """Job-only failure transaction: no archive lock after a job lock."""
    from app.domain.jobs.service import fail_job
    await session.rollback()
    job = await session.scalar(select(Job).where(Job.id == job_id).with_for_update()
        .execution_options(populate_existing=True, autoflush=False))
    if (job is None or job.status != "claimed" or job.attempts != attempts or job.claimed_at != claimed_at
        or (generation_id is not None and job.generation_id != generation_id)):
        await session.rollback()
        return
    await fail_job(session, job, error=error, retryable=retryable)
