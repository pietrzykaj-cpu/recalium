"""Short fenced coordinator transactions; no model/provider work in this module."""
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import uuid
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert
from app.domain.archive.models import RawArchiveItem, Tombstone
from app.domain.jobs.models import Job
from app.domain.derived_memory.models import Fact, Summary, Embedding
from app.domain.derivations.models import ArchiveDerivationHead as Head, DerivationGeneration as Generation, DerivationStageCompletion as Completion, DerivationFtsProjection as Projection
from app.domain.derivations.visibility import enabled, HISTORY

class ConflictReason(str, Enum):
    CLAIM_LOST = "claim_lost"
    JOB_GONE = "job_gone"
    ALREADY_PUBLISHED = "already_published"
    SOURCE_REMOVED = "source_removed"
    SOURCE_CHANGED = "source_changed"
    POLICY_CHANGED = "policy_changed"
    GENERATION_RETIRED = "generation_retired"
    STATE_INCONSISTENT = "state_inconsistent"
    OUTPUTS_INCONSISTENT = "outputs_inconsistent"
    INVALID_PHASE = "invalid_phase"
    UNSUPPORTED_CONTRACT = "unsupported_contract"
    MANAGED_DISABLED = "managed_disabled"


class ConflictDisposition(str, Enum):
    STALE = "stale_noop"
    REPLAY = "published_replay"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    INVARIANT = "invariant"


CONFLICT_DISPOSITIONS = {
    ConflictReason.CLAIM_LOST: ConflictDisposition.STALE,
    ConflictReason.JOB_GONE: ConflictDisposition.STALE,
    ConflictReason.ALREADY_PUBLISHED: ConflictDisposition.REPLAY,
    ConflictReason.SOURCE_REMOVED: ConflictDisposition.CANCELLED,
    ConflictReason.SOURCE_CHANGED: ConflictDisposition.BLOCKED,
    ConflictReason.POLICY_CHANGED: ConflictDisposition.BLOCKED,
    ConflictReason.GENERATION_RETIRED: ConflictDisposition.BLOCKED,
    ConflictReason.STATE_INCONSISTENT: ConflictDisposition.INVARIANT,
    ConflictReason.OUTPUTS_INCONSISTENT: ConflictDisposition.INVARIANT,
    ConflictReason.INVALID_PHASE: ConflictDisposition.INVARIANT,
    ConflictReason.UNSUPPORTED_CONTRACT: ConflictDisposition.INVARIANT,
    ConflictReason.MANAGED_DISABLED: ConflictDisposition.INVARIANT,
}


class DerivationConflict(Exception):
    """Static reason is policy; messages are never parsed or persisted."""
    def __init__(self, reason, message=None):
        self.reason = reason
        super().__init__((message or reason.value) if isinstance(reason, ConflictReason)
                         else "Unknown derivation conflict")


class CompletedReprocessMismatch(Exception):
    """Manual CAS failure only; not a worker coordination outcome."""
    def __init__(self, job):
        self.current_generation_id = job.generation_id
        self.current_status = job.status if job.status in {
            "pending", "claimed", "completed", "failed", "retryable_failed", "pending_provider"
        } else "unsupported"
        super().__init__("expected_generation_mismatch")

@dataclass(frozen=True)
class Claim:
    job_id: uuid.UUID
    archive_id: uuid.UUID
    generation_id: uuid.UUID | None
    attempts: int
    claimed_at: datetime
    policy_intent: tuple | None = None

def require_enabled():
    if not enabled():
        raise DerivationConflict(ConflictReason.MANAGED_DISABLED, "Managed derivation is dormant")

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

async def locked_state(session, archive_id, job_id, *, create_head=False):
    """Existing-row classification uses archives -> heads -> jobs -> generations.

    Only admission may create a head; any later fencing rejection rolls it back.
    """
    from sqlalchemy import or_
    fresh = {"populate_existing": True, "autoflush": False}
    source = await session.scalar(select(RawArchiveItem).where(
        RawArchiveItem.id == archive_id).with_for_update().execution_options(**fresh))
    if create_head and source is not None:
        await session.execute(insert(Head).values(raw_archive_id=archive_id).on_conflict_do_nothing())
    head = await session.scalar(select(Head).where(Head.raw_archive_id == archive_id)
        .with_for_update().execution_options(**fresh))
    jobs = (await session.scalars(select(Job).where(or_(
        Job.raw_archive_id == archive_id, Job.id == job_id)).order_by(Job.id)
        .with_for_update().execution_options(**fresh))).all()
    generations = (await session.scalars(select(Generation).where(
        Generation.raw_archive_id == archive_id).order_by(Generation.id)
        .with_for_update().execution_options(**fresh))).all()
    job = next((j for j in jobs if j.id == job_id), None)
    return source, head, job, jobs, generations


def owns_claim(job, claim, *, generation=True):
    return (job is not None and job.id == claim.job_id and job.raw_archive_id == claim.archive_id
        and job.status == "claimed" and job.attempts == claim.attempts
        and claim.claimed_at is not None and job.claimed_at == claim.claimed_at
        and (not generation or claim.generation_id is None or job.generation_id == claim.generation_id))


async def state_reason(session, claim, source, head, job, jobs, gen):
    # Source removal outranks M3's abandoned generation; stale ownership outranks all.
    if job is None:
        return ConflictReason.JOB_GONE
    if not owns_claim(job, claim):
        return ConflictReason.CLAIM_LOST
    if not await source_live(session, source):
        return ConflictReason.SOURCE_REMOVED
    if not enabled():
        return ConflictReason.MANAGED_DISABLED
    if len(jobs) != 1:
        return ConflictReason.STATE_INCONSISTENT
    if gen is None or head is None or gen.original_job_id != job.id:
        return ConflictReason.STATE_INCONSISTENT
    if gen.source_content_hash != source.content_hash:
        return ConflictReason.SOURCE_CHANGED
    if claim.policy_intent is not None and claim.policy_intent != (
        (source.metadata_json or {}).get("processing_mode"),
        (source.metadata_json or {}).get("sensitivity_hint")):
        return ConflictReason.POLICY_CHANGED
    if gen.contract_version != 1:
        return ConflictReason.UNSUPPORTED_CONTRACT
    if gen.status in {"superseded", "failed", "abandoned"}:
        return ConflictReason.GENERATION_RETIRED
    if (gen.status not in {"building", "prepared", "active"}
        or (gen.status == "active" and head.active_generation_id != gen.id)
        or (gen.status != "active" and head.candidate_generation_id != gen.id)):
        return ConflictReason.STATE_INCONSISTENT
    return None


async def context(session, claim):
    require_enabled()
    source, head, job, jobs, generations = await locked_state(session, claim.archive_id, claim.job_id)
    gen = next((g for g in generations if g.id == claim.generation_id), None)
    reason = await state_reason(session, claim, source, head, job, jobs, gen)
    if reason is not None:
        raise DerivationConflict(reason)
    return source, head, job, gen, generations


async def _new_generation(session, source, job, generations, predecessor, *, manual):
    """Shared identity/succession construction; head assignment is the caller's duty."""
    generation = Generation(raw_archive_id=source.id, original_job_id=job.id,
        sequence=max((g.sequence for g in generations), default=0)+1,
        creation_reason="manual_reprocess" if manual else "automatic_processing",
        source_content_hash=source.content_hash, predecessor_generation_id=predecessor)
    session.add(generation)
    await session.flush()
    job.generation_id = generation.id
    if manual:
        job.status, job.attempts, job.error_message = "pending", 0, None
        job.updated_at = datetime.now(timezone.utc)
    return generation


async def admit_completed_generation(session, job_id, expected_generation):
    """Explicit completed CAS with ordinary candidate attachment, never publication.

    Existing archive/head locking creates a blank head when needed. Candidate,
    generation and job reset commit together; active/retirement stay unchanged.
    Existing PR18 manual reset remains separate.
    """
    require_enabled()
    if expected_generation != "legacy" and not isinstance(expected_generation, uuid.UUID):
        raise DerivationConflict(ConflictReason.INVALID_PHASE)
    aid = await session.scalar(select(Job.raw_archive_id).where(Job.id == job_id)
        .execution_options(autoflush=False))
    await session.rollback()
    if aid is None:
        return None
    async with transaction(session):
        source, head, job, jobs, gens = await locked_state(session, aid, job_id, create_head=True)
        if job is None:
            # A lost job must not commit the newly inserted blank head on 404.
            await session.rollback()
            return None
        if job.raw_archive_id != aid:
            raise DerivationConflict(ConflictReason.STATE_INCONSISTENT)
        expected = None if expected_generation == "legacy" else expected_generation
        # Check before status: a lost-ack replay is fenced even after claim/finish.
        if job.generation_id != expected:
            raise CompletedReprocessMismatch(job)
        if job.status != "completed":
            raise DerivationConflict(ConflictReason.INVALID_PHASE)
        require_enabled()
        if not await source_live(session, source):
            raise DerivationConflict(ConflictReason.SOURCE_REMOVED)
        if head is None or len(jobs) != 1 or any(g.original_job_id != job.id for g in gens):
            raise DerivationConflict(ConflictReason.STATE_INCONSISTENT)
        if job.generation_id is None:
            if gens or (head and (head.active_generation_id is not None
                    or head.candidate_generation_id is not None or head.legacy_retired_at is not None)):
                raise DerivationConflict(ConflictReason.STATE_INCONSISTENT)
            predecessor = None  # Never invent legacy derivation provenance.
        else:
            gen = next((g for g in gens if g.id == job.generation_id), None)
            if (gen is None or head is None or gen.status != "active"
                or head.active_generation_id != gen.id or head.candidate_generation_id is not None
                or gen.contract_version != 1 or gen.source_content_hash != source.content_hash
                or any(g.status in {"building", "prepared"} for g in gens)
                or sum(g.status == "active" for g in gens) != 1):
                raise DerivationConflict(ConflictReason.STATE_INCONSISTENT)
            await validate_publication(session, head, gen)
            predecessor = gen.id
        result = await _new_generation(session, source, job, gens, predecessor, manual=True)
        head.candidate_generation_id = result.id
        job.completed_at = None
    return result

async def admit_generation(session, job_id, *, manual=False, expected_claim=None, expected_generation=None):
    require_enabled()
    if not manual and (expected_claim is None or expected_claim.job_id != job_id):
        raise DerivationConflict(ConflictReason.INVALID_PHASE, "Automatic admission requires captured claim")
    # Only scalar IDs escape the unlocked lookup; never cached status validation.
    aid = await session.scalar(select(Job.raw_archive_id).where(Job.id == job_id)
        .execution_options(autoflush=False))
    await session.rollback()
    if aid is None:
        return None
    if not manual and aid != expected_claim.archive_id:
        raise DerivationConflict(ConflictReason.CLAIM_LOST)
    result = None
    async with transaction(session):
        source, head, job, jobs, gens = await locked_state(session, aid, job_id, create_head=True)
        if job is None:
            raise DerivationConflict(ConflictReason.JOB_GONE)
        if not manual and not owns_claim(job, expected_claim, generation=False):
            raise DerivationConflict(ConflictReason.CLAIM_LOST)
        if job.raw_archive_id != aid:
            raise DerivationConflict(ConflictReason.CLAIM_LOST)
        if manual and expected_generation is not None:
            expected = None if expected_generation == "legacy" else expected_generation
            if job.generation_id != expected:
                raise CompletedReprocessMismatch(job)
        if not await source_live(session, source):
            raise DerivationConflict(ConflictReason.SOURCE_REMOVED)
        if manual and job.status not in {"failed", "retryable_failed", "pending_provider"}:
            raise DerivationConflict(ConflictReason.INVALID_PHASE, "Job cannot be reprocessed in its current state")
        # Application paths create one processing job/archive. Legitimate multi-job
        # ownership would require a separately designed coordination contract.
        if len(jobs) != 1 or head is None:
            raise DerivationConflict(ConflictReason.STATE_INCONSISTENT)
        candidate = next((g for g in gens if g.id == head.candidate_generation_id), None)
        if any(g.original_job_id != job_id for g in gens
               if g.id in {head.active_generation_id, head.candidate_generation_id}):
            raise DerivationConflict(ConflictReason.STATE_INCONSISTENT)
        candidate_owner = next((j for j in jobs if candidate and j.id == candidate.original_job_id), None)
        if candidate and candidate_owner and candidate_owner.status == "failed":
            candidate.status = "failed"
            head.candidate_generation_id = None
            candidate = None
            await session.flush()
        if not manual and job.generation_id is not None:
            result = next((g for g in gens if g.id == job.generation_id), None)
            if result is None or result.status not in {"building", "prepared", "active"}:
                raise DerivationConflict(ConflictReason.GENERATION_RETIRED, "Existing generation is not resumable")
            if result.status == "active" and head.active_generation_id != result.id:
                raise DerivationConflict(ConflictReason.STATE_INCONSISTENT, "Published generation is no longer current")
            if result.status != "active" and head.candidate_generation_id != result.id:
                raise DerivationConflict(ConflictReason.STATE_INCONSISTENT, "Candidate is no longer admitted")
        else:
            if candidate:
                if not manual or candidate.original_job_id != job.id:
                    raise DerivationConflict(ConflictReason.STATE_INCONSISTENT, "Archive already has an open candidate")
                candidate.status = "abandoned"
                head.candidate_generation_id = None
                await session.flush()
            result = await _new_generation(session, source, job, gens, head.active_generation_id, manual=manual)
            head.candidate_generation_id = result.id
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
                raise DerivationConflict(ConflictReason.INVALID_PHASE, "Extraction is not writable")
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
                raise DerivationConflict(ConflictReason.INVALID_PHASE, "Summary is not writable")
            audit_id = category = exception_class = None
            if outcome == "partial_failure":
                if job.attempts < job.max_attempts or partial is None:
                    raise DerivationConflict(ConflictReason.INVALID_PHASE, "Partial summary requires final-attempt evidence")
                from app.domain.audit.models import AuditEvent
                category, exception_class, provider = partial
                if category not in {"provider_transport", "provider_response", "response_parsing"}:
                    raise DerivationConflict(ConflictReason.INVALID_PHASE, "Partial summary category is not allowlisted")
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
                raise DerivationConflict(ConflictReason.INVALID_PHASE, "Required stages not complete")
            row = Projection(raw_archive_id=claim.archive_id, generation_id=gen.id,
                text_content=text_content, projection_kind=projection_kind)
            session.add(row)
            gen.status = "prepared"
            await session.flush()
        elif row.source_status != "active":
            raise DerivationConflict(ConflictReason.OUTPUTS_INCONSISTENT, "Projection is suppressed")
    return row

async def validate_publication(session, head, gen):
    """One durable definition, shared by publication and managed replay resolution.

    Writers hold archive/head/job/generation locks. The read-only cutover checker
    instead uses one coherent REPEATABLE READ snapshot. This does not mutate state.
    Optional enrichment is deliberately not a publication prerequisite.
    """
    projection = await session.scalar(select(Projection).where(Projection.generation_id == gen.id))
    summary, extraction = await stage(session, gen.id, "summary"), await stage(session, gen.id, "extraction")
    fact_count = await session.scalar(select(func.count(Fact.id)).where(
        Fact.generation_id == gen.id).execution_options(**HISTORY))
    summary_id = await session.scalar(select(Summary.id).where(
        Summary.generation_id == gen.id).execution_options(**HISTORY))
    if (summary is None or extraction is None or extraction.fact_count != fact_count
        or (summary.outcome == 'succeeded') != (summary_id is not None)
        or projection is None or projection.source_status != "active"
        or projection.raw_archive_id != gen.raw_archive_id):
        raise DerivationConflict(ConflictReason.OUTPUTS_INCONSISTENT, "Required durable outputs are inconsistent")
    if gen.status == "active":
        if (head.active_generation_id != gen.id or head.legacy_retired_at is None
            or gen.published_at is None or gen.superseded_at is not None):
            raise DerivationConflict(ConflictReason.STATE_INCONSISTENT, "Published state is inconsistent")
    elif gen.status != "prepared" or head.active_generation_id != gen.predecessor_generation_id:
        raise DerivationConflict(ConflictReason.INVALID_PHASE, "Candidate is not publishable")


async def resolve_conflict(session, claim, conflict):
    """Resolve expected outcomes once, using fresh locked state, never messages.

    No attempt refunds or retry scheduling. Missing heads are not created here.
    Unknown codes and resolver/DB failures escape to unchanged PR #14 recovery.
    """
    if not isinstance(conflict.reason, ConflictReason) or conflict.reason not in CONFLICT_DISPOSITIONS:
        raise conflict
    async with transaction(session):
        source, head, job, jobs, generations = await locked_state(session, claim.archive_id, claim.job_id)
        if not owns_claim(job, claim):
            return ConflictDisposition.STALE
        gen = next((g for g in generations if g.id == job.generation_id), None)
        reason = await state_reason(session, claim, source, head, job, jobs, gen)
        if reason is None:
            reason = conflict.reason
            # A stale/removed/obsolete assertion is not evidence when the fresh
            # durable state contradicts it. Never silently complete such a claim.
            if CONFLICT_DISPOSITIONS[reason] in {
                ConflictDisposition.STALE, ConflictDisposition.CANCELLED, ConflictDisposition.BLOCKED
            }:
                reason = ConflictReason.STATE_INCONSISTENT
        if reason == ConflictReason.ALREADY_PUBLISHED:
            if gen.status != "active":
                reason = ConflictReason.STATE_INCONSISTENT
            else:
                try:
                    await validate_publication(session, head, gen)
                except DerivationConflict as invalid:
                    reason = invalid.reason
                else:
                    return ConflictDisposition.REPLAY
        disposition = CONFLICT_DISPOSITIONS[reason]
        prefix = {ConflictDisposition.CANCELLED: "derivation_cancelled",
                  ConflictDisposition.BLOCKED: "derivation_blocked",
                  ConflictDisposition.INVARIANT: "derivation_invariant"}[disposition]
        detail = "owner_conflict" if reason == ConflictReason.STATE_INCONSISTENT and (
            len(jobs) != 1 or any(g.original_job_id != job.id for g in generations
                if head and g.id in {head.active_generation_id, head.candidate_generation_id})) else reason.value
        job.status, job.error_message = "failed", prefix + ":" + detail
        job.updated_at = datetime.now(timezone.utc)
    return disposition


async def publish_generation(session, claim):
    require_enabled()
    changed = False
    async with transaction(session):
        _, head, _, gen, gens = await context(session, claim)
        await validate_publication(session, head, gen)
        if gen.status != "active":
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
            raise DerivationConflict(ConflictReason.INVALID_PHASE, "Enrichment requires active generation")
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
            raise DerivationConflict(ConflictReason.OUTPUTS_INCONSISTENT, "Baseline is suppressed")

async def finish_claim(session, claim, *, pending_provider=False, partial_marker=None):
    async with transaction(session):
        _, _, job, gen, _ = await context(session, claim)
        if pending_provider:
            if gen.status != "building":
                raise DerivationConflict(ConflictReason.INVALID_PHASE, "Only unfinished generation may await provider")
            job.status = "pending_provider"
            job.error_message = "No LLM provider configured. Add an OpenAI, Anthropic, or Ollama key in Settings."
        else:
            if gen.status != "active":
                raise DerivationConflict(ConflictReason.INVALID_PHASE, "Completion requires published generation")
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
