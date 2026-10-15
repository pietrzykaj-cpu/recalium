"""M4 expected outcomes; synthetic DB and offline provider spies only."""
import asyncio
from datetime import timedelta, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
import uuid

import pytest
from sqlalchemy import text, event
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.domain.derivations import service as g
from app.domain.derivations.models import DerivationGeneration as Gen, ArchiveDerivationHead as Head
from app.domain.jobs.models import Job
from app.domain.archive.models import RawArchiveItem, Tombstone
from app.infrastructure.settings import get_settings
from app.worker import managed_dispatcher as managed
from tests.domain.test_derivation_boundaries import build
from tests.domain.test_derivation_generations import admit_claimed
from tests.worker.test_managed_dispatcher import spies, job_row

pytestmark = pytest.mark.asyncio


async def test_every_reason_has_explicit_disposition(db_session, spies):
    assert set(g.CONFLICT_DISPOSITIONS) == set(g.ConflictReason)


async def test_stale_admission_does_not_bind_newer_claim(db_session, spies):
    job = await job_row(db_session)
    claim = g.Claim(job.id, job.raw_archive_id, None, job.attempts, job.claimed_at)
    job.attempts += 1
    job.claimed_at += timedelta(seconds=1)
    await db_session.commit()
    with pytest.raises(g.DerivationConflict) as rejected:
        await g.admit_generation(db_session, claim.job_id, expected_claim=claim)
    assert rejected.value.reason == g.ConflictReason.CLAIM_LOST
    assert await db_session.scalar(text("SELECT count(*) FROM derivation_generations")) == 0
    assert await db_session.scalar(text("SELECT count(*) FROM archive_derivation_heads")) == 0
    assert await db_session.scalar(text("SELECT count(*) FROM jobs WHERE generation_id IS NOT NULL")) == 0


@pytest.mark.parametrize("reason", list(g.ConflictReason))
async def test_each_reason_resolves_with_explicit_fresh_state(db_session, spies, monkeypatch, reason):
    job = await job_row(db_session)
    jid, aid, attempts, at = job.id, job.raw_archive_id, job.attempts, job.claimed_at
    if reason == g.ConflictReason.JOB_GONE:
        claim = g.Claim(jid, aid, None, attempts, at)
        await db_session.delete(job)
    elif reason == g.ConflictReason.ALREADY_PUBLISHED:
        claim, _ = await build(db_session, job)
    else:
        generation = await admit_claimed(db_session, jid)
        claim = g.Claim(jid, aid, generation.id, attempts, at, (None, None))
        source = await db_session.get(RawArchiveItem, aid)
        if reason == g.ConflictReason.CLAIM_LOST:
            job = await db_session.get(Job, jid)
            job.attempts += 1
            job.claimed_at += timedelta(seconds=1)
        elif reason == g.ConflictReason.SOURCE_REMOVED:
            source.deleted_at = datetime.now(timezone.utc)
        elif reason == g.ConflictReason.SOURCE_CHANGED:
            source.content_hash = uuid.uuid4().hex
        elif reason == g.ConflictReason.POLICY_CHANGED:
            source.metadata_json = {"processing_mode": "local_only"}
        elif reason == g.ConflictReason.GENERATION_RETIRED:
            generation.status = "abandoned"
            (await db_session.get(Head, aid)).candidate_generation_id = None
        elif reason == g.ConflictReason.UNSUPPORTED_CONTRACT:
            generation.contract_version = 2
        elif reason == g.ConflictReason.MANAGED_DISABLED:
            monkeypatch.setattr(get_settings(), "generation_idempotency_enabled", False)
    await db_session.commit()
    outcome = await g.resolve_conflict(db_session, claim, g.DerivationConflict(reason, "PRIVATE DETAIL"))
    assert outcome == g.CONFLICT_DISPOSITIONS[reason]
    current = await db_session.get(Job, jid, populate_existing=True)
    if reason == g.ConflictReason.JOB_GONE:
        assert current is None
        assert await db_session.scalar(text("SELECT count(*) FROM archive_derivation_heads")) == 0
    elif outcome == g.ConflictDisposition.STALE:
        assert current.status == "claimed" and current.attempts == attempts + 1
        assert current.error_message is None
    elif outcome == g.ConflictDisposition.REPLAY:
        assert current.status == "claimed" and current.attempts == attempts
    else:
        prefix = {g.ConflictDisposition.CANCELLED: "derivation_cancelled",
                  g.ConflictDisposition.BLOCKED: "derivation_blocked",
                  g.ConflictDisposition.INVARIANT: "derivation_invariant"}[outcome]
        assert current.status == "failed" and current.error_message == f"{prefix}:{reason.value}"
        assert current.attempts == attempts and current.claimed_at == at


async def test_missing_admission_result_is_explicit_stale_noop(db_session, spies, monkeypatch):
    job = await job_row(db_session)
    token = SimpleNamespace(id=job.id, raw_archive_id=job.raw_archive_id,
        generation_id=None, attempts=job.attempts, claimed_at=job.claimed_at)
    await db_session.delete(job)
    await db_session.commit()
    observed = AsyncMock(wraps=g.resolve_conflict)
    monkeypatch.setattr(g, "resolve_conflict", observed)
    await managed.dispatch_managed_job(db_session, token)
    assert observed.await_args.args[2].reason == g.ConflictReason.JOB_GONE
    assert await db_session.scalar(text("SELECT count(*) FROM archive_derivation_heads")) == 0
    assert not any(spy.await_count for spy in spies)


async def test_contradicted_source_removed_reason_fails_closed(db_session, spies):
    job = await job_row(db_session)
    claim, _ = await build(db_session, job)
    # Truly published/current, with live source and unchanged captured ownership:
    # the real publication checker accepts this same durable state.
    assert await g.publish_generation(db_session, claim) is False
    outcome = await g.resolve_conflict(db_session, claim, g.DerivationConflict(
        g.ConflictReason.SOURCE_REMOVED))
    assert outcome == g.ConflictDisposition.INVARIANT
    current = await db_session.get(Job, claim.job_id, populate_existing=True)
    assert current.status == "failed"
    assert current.error_message == "derivation_invariant:state_inconsistent"
    assert current.attempts == claim.attempts and current.claimed_at == claim.claimed_at
    source = await db_session.get(RawArchiveItem, claim.archive_id)
    generation = await db_session.get(Gen, claim.generation_id)
    head = await db_session.get(Head, claim.archive_id)
    assert source.deleted_at is None and generation.status == "active"
    assert head.active_generation_id == claim.generation_id


async def test_automatic_admission_requires_token(db_session, spies):
    job = await job_row(db_session)
    with pytest.raises(g.DerivationConflict) as rejected:
        await g.admit_generation(db_session, job.id)
    assert rejected.value.reason == g.ConflictReason.INVALID_PHASE
    assert await db_session.scalar(text("SELECT count(*) FROM derivation_generations")) == 0


async def test_two_session_reclaim_fences_admission_and_resolver(db_session, test_engine, spies):
    job = await job_row(db_session)
    claim = g.Claim(job.id, job.raw_archive_id, None, job.attempts, job.claimed_at)
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    await db_session.rollback()
    async with factory() as newer:
        row = await newer.get(Job, claim.job_id)
        row.attempts += 1
        row.claimed_at += timedelta(seconds=1)
        await newer.commit()
    with pytest.raises(g.DerivationConflict) as rejected:
        await g.admit_generation(db_session, claim.job_id, expected_claim=claim)
    assert await g.resolve_conflict(db_session, claim, rejected.value) == g.ConflictDisposition.STALE
    current = await db_session.get(Job, claim.job_id, populate_existing=True)
    assert current.status == "claimed" and current.attempts == 2 and current.generation_id is None
    assert await db_session.scalar(text("SELECT count(*) FROM derivation_generations")) == 0
    assert await db_session.scalar(text("SELECT count(*) FROM archive_derivation_heads")) == 0


async def test_second_job_is_owner_invariant_not_retry(db_session, spies):
    first = await job_row(db_session)
    jid, aid, at = first.id, first.raw_archive_id, first.claimed_at
    db_session.add(Job(raw_archive_id=aid, job_type="extract", status="pending"))
    await db_session.commit()
    token = SimpleNamespace(id=jid, raw_archive_id=aid, generation_id=None, attempts=1, claimed_at=at)
    await managed.dispatch_managed_job(db_session, token)
    current = await db_session.get(Job, jid, populate_existing=True)
    assert current.status == "failed" and current.error_message == "derivation_invariant:owner_conflict"
    assert current.attempts == 1
    assert await db_session.scalar(text("SELECT count(*) FROM archive_derivation_heads")) == 0
    assert await db_session.scalar(text("SELECT count(*) FROM derivation_generations")) == 0


@pytest.mark.parametrize("damage", [None, "facts", "summary", "projection", "retirement", "publication", "stages", "head"])
async def test_published_replay_shares_real_checker_and_keeps_embedding_fillable(
    db_session, spies, monkeypatch, damage,
):
    from app.domain.derived_memory import service as writers
    job = await job_row(db_session)
    jid = job.id
    claim, _ = await build(db_session, job)
    statements = {
        "facts": "UPDATE derivation_stage_completions SET fact_count=99 WHERE stage_name='extraction'",
        "summary": "DELETE FROM summaries",
        "projection": "UPDATE derivation_fts_projections SET source_status='source_removed'",
        # Corrupt-looking state is created only in this disposable test database.
        "retirement": "ALTER TABLE archive_derivation_heads DISABLE TRIGGER keep_legacy_retired; UPDATE archive_derivation_heads SET legacy_retired_at=NULL; ALTER TABLE archive_derivation_heads ENABLE TRIGGER keep_legacy_retired",
        "publication": "UPDATE derivation_generations SET published_at=NULL",
        "stages": "DELETE FROM derivation_stage_completions WHERE stage_name='summary'",
        "head": "UPDATE archive_derivation_heads SET active_generation_id=NULL",
    }
    if damage:
        for statement in statements[damage].split("; "):
            await db_session.execute(text(statement))
        await db_session.commit()
        with pytest.raises(g.DerivationConflict):
            await g.publish_generation(db_session, claim)
    checker = AsyncMock(wraps=g.validate_publication)
    monkeypatch.setattr(g, "validate_publication", checker)
    async def embed(raw):
        assert not db_session.in_transaction()
        return [1.0] + [0.0] * 383
    embedding = AsyncMock(side_effect=embed)
    monkeypatch.setattr(writers, "embed_text", embedding)
    await managed.dispatch_managed_job(db_session, await db_session.get(Job, jid))
    current = await db_session.get(Job, jid, populate_existing=True)
    assert current.attempts == 1
    if damage:
        assert current.status == "failed" and current.error_message.startswith("derivation_invariant:")
        embedding.assert_not_awaited()
    else:
        assert current.status == "completed"
        checker.assert_awaited_once()
        embedding.assert_awaited_once()
        assert await db_session.scalar(text("SELECT count(*) FROM embeddings")) == 1
    assert spies[1].await_count == spies[2].await_count == 0


@pytest.mark.parametrize("phase", ["gate", "summary", "extraction", "embedding_call", "embedding", "conflict", "links", "inner_links", "finalization"])
async def test_typed_stage_conflict_reaches_resolver_once(db_session, spies, monkeypatch, phase):
    from app.domain.derived_memory import service as writers
    from app.domain import conflict_detection
    from app.worker import managed_links
    from app.worker import dispatcher as legacy
    job = await job_row(db_session)
    jid = job.id
    async def conflict(*args, **kwargs):
        raise g.DerivationConflict(g.ConflictReason.INVALID_PHASE)
    target, name = {
        "extraction": (g, "persist_extraction"),
        "gate": (legacy._gate, "classify_async"),
        "summary": (legacy, "_run_summarize_job"),
        "embedding_call": (writers, "embed_text"),
        "embedding": (g, "persist_embedding"),
        "conflict": (conflict_detection, "detect_and_group_duplicates"),
        "links": (managed_links, "run_managed_links"),
        "inner_links": (legacy, "_classify_link_pair"),
        "finalization": (g, "finish_claim"),
    }[phase]
    async def embed(raw):
        assert not db_session.in_transaction()
        return [1.0] + [0.0] * 383
    monkeypatch.setattr(writers, "embed_text", embed)
    if phase == "inner_links":
        from app.domain.derived_memory.models import Embedding
        source = RawArchiveItem(source_type="test", raw_content="target", content_hash=uuid.uuid4().hex)
        db_session.add(source)
        await db_session.commit()
        aid = source.id
        await writers.write_facts(db_session, aid, [{"fact_text": "target"}])
        db_session.add(Embedding(raw_archive_id=aid, embedding=[1.0] + [0.0] * 383,
                                embedding_model=writers.ACTIVE_EMBEDDING_MODEL))
        await db_session.commit()
    monkeypatch.setattr(target, name, conflict)
    resolver = AsyncMock(wraps=g.resolve_conflict)
    monkeypatch.setattr(g, "resolve_conflict", resolver)
    await managed.dispatch_managed_job(db_session, job)
    resolver.assert_awaited_once()
    current = await db_session.get(Job, jid, populate_existing=True)
    assert current.status == "failed" and current.error_message == "derivation_invariant:invalid_phase"
    assert current.attempts == 1


async def test_deletion_during_embedding_resolves_source_before_m3_abandoned(
    db_session, spies, monkeypatch, tmp_path,
):
    from app.domain.archive.service import cascade_delete_archive_item
    from app.domain.derived_memory import service as writers
    job = await job_row(db_session)
    jid, aid, at = job.id, job.raw_archive_id, job.claimed_at
    async def delete_during_embedding(raw):
        assert not db_session.in_transaction()
        await cascade_delete_archive_item(db_session, aid, backup_dir=str(tmp_path))
        return [1.0] + [0.0] * 383
    monkeypatch.setattr(writers, "embed_text", delete_during_embedding)
    await managed.dispatch_managed_job(db_session, job)
    current = await db_session.get(Job, jid, populate_existing=True)
    generation = await db_session.get(Gen, current.generation_id)
    assert current.status == "failed" and current.error_message == "derivation_cancelled:source_removed"
    assert current.attempts == 1 and current.claimed_at == at
    assert generation.status == "abandoned" and generation.published_at is not None
    assert generation.superseded_at is None
    assert await db_session.scalar(text("SELECT count(*) FROM embeddings")) == 0


async def test_resolver_lost_ack_is_idempotent(db_session, spies, monkeypatch):
    job = await job_row(db_session)
    generation = await admit_claimed(db_session, job.id)
    job = await db_session.get(Job, generation.original_job_id)
    claim = g.Claim(job.id, job.raw_archive_id, generation.id, job.attempts, job.claimed_at)
    real_commit = db_session.commit
    async def lost_ack():
        await real_commit()
        raise RuntimeError("synthetic lost ack")
    with monkeypatch.context() as patch:
        patch.setattr(db_session, "commit", lost_ack)
        with pytest.raises(RuntimeError):
            await g.resolve_conflict(db_session, claim, g.DerivationConflict(g.ConflictReason.INVALID_PHASE))
    current = await db_session.get(Job, claim.job_id, populate_existing=True)
    original = (current.status, current.error_message, current.attempts, current.updated_at)
    assert await g.resolve_conflict(db_session, claim, g.DerivationConflict(g.ConflictReason.INVALID_PHASE)) == g.ConflictDisposition.STALE
    current = await db_session.get(Job, claim.job_id, populate_existing=True)
    assert (current.status, current.error_message, current.attempts, current.updated_at) == original


async def test_resolver_lock_order_and_dirty_cache_are_fresh(db_session, test_engine, spies):
    job = await job_row(db_session)
    generation = await admit_claimed(db_session, job.id)
    job = await db_session.get(Job, generation.original_job_id)
    claim = g.Claim(job.id, job.raw_archive_id, generation.id, job.attempts, job.claimed_at)
    job.attempts = 999
    statements = []
    def capture(conn, cursor, sql, parameters, context, executemany):
        if "FOR UPDATE" in sql:
            statements.append(sql)
    event.listen(test_engine.sync_engine, "before_cursor_execute", capture)
    try:
        await g.resolve_conflict(db_session, claim, g.DerivationConflict(g.ConflictReason.INVALID_PHASE))
    finally:
        event.remove(test_engine.sync_engine, "before_cursor_execute", capture)
    assert len(statements) == 4
    assert all(f"FROM {name}" in sql for name, sql in zip(
        ["raw_archive", "archive_derivation_heads", "jobs", "derivation_generations"], statements))
    assert "ORDER BY jobs.id" in statements[2]
    current = await db_session.get(Job, claim.job_id, populate_existing=True)
    assert current.status == "failed" and current.attempts == 1


@pytest.mark.parametrize("failure", ["unknown", "unmapped", "resolver", "database"])
async def test_unexpected_conflict_or_resolver_failure_reaches_unchanged_pr14(
    db_session, test_engine, spies, monkeypatch, failure,
):
    from app.domain.jobs import service as jobs
    from app.infrastructure import db
    from app.worker.loop import worker_loop
    from app.domain.retrieval import service as retrieval
    job = await job_row(db_session)
    jid = job.id
    job.status, job.attempts, job.claimed_at = "pending", 0, None
    await db_session.commit()
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    monkeypatch.setattr(db, "get_session_factory", lambda: factory)
    async def fail_admission(*args, **kwargs):
        raise g.DerivationConflict("unmapped" if failure == "unknown" else g.ConflictReason.INVALID_PHASE)
    monkeypatch.setattr(g, "admit_generation", fail_admission)
    if failure == "unmapped":
        monkeypatch.delitem(g.CONFLICT_DISPOSITIONS, g.ConflictReason.INVALID_PHASE)
    elif failure == "resolver":
        monkeypatch.setattr(g, "resolve_conflict", AsyncMock(side_effect=RuntimeError("synthetic resolver failure")))
    elif failure == "database":
        async def broken_state(session, *args, **kwargs):
            await session.execute(text("SELECT * FROM nonexistent_m4_fixture"))
        monkeypatch.setattr(g, "locked_state", broken_state)
    recovery = AsyncMock(wraps=jobs.recover_claimed_job_after_dispatch_failure)
    monkeypatch.setattr(jobs, "recover_claimed_job_after_dispatch_failure", recovery)
    monkeypatch.setattr(retrieval, "notify_cache_invalidation", AsyncMock())
    async def stop(seconds):
        raise asyncio.CancelledError()
    monkeypatch.setattr(asyncio, "sleep", stop)
    with pytest.raises(asyncio.CancelledError):
        await worker_loop()
    recovery.assert_awaited_once()
    current = await db_session.get(Job, jid, populate_existing=True)
    assert current.status == "retryable_failed" and current.attempts == 1
    assert current.error_message.startswith("worker_dispatch_unhandled:")
    assert "synthetic" not in current.error_message and "unmapped" not in current.error_message


@pytest.mark.parametrize("exception", [asyncio.CancelledError, SystemExit])
async def test_base_exception_propagates_not_resolved(db_session, spies, monkeypatch, exception):
    job = await job_row(db_session)
    jid = job.id
    monkeypatch.setattr(g, "admit_generation", AsyncMock(side_effect=exception()))
    resolver = AsyncMock()
    monkeypatch.setattr(g, "resolve_conflict", resolver)
    with pytest.raises(exception):
        await managed.dispatch_managed_job(db_session, job)
    resolver.assert_not_awaited()
    assert (await db_session.get(Job, jid)).status == "claimed"


async def test_r2_off_never_enters_managed_resolver(db_session, spies, monkeypatch):
    from app.worker import dispatcher as d
    job = await job_row(db_session)
    jid = job.id
    monkeypatch.setattr(get_settings(), "generation_idempotency_enabled", False)
    resolver = AsyncMock(side_effect=AssertionError("R2 OFF resolver unreachable"))
    admission = AsyncMock(side_effect=AssertionError("R2 OFF admission unreachable"))
    monkeypatch.setattr(g, "resolve_conflict", resolver)
    monkeypatch.setattr(g, "admit_generation", admission)
    # Legacy provider spies have their own prepare-session transaction semantics.
    spies[0].side_effect = None
    from app.domain.policy.gate import SensitivityDecision
    spies[0].return_value = SensitivityDecision(category="general", confidence=.9, blocked=False, method="test")
    spies[1].side_effect, spies[1].return_value = None, "summary"
    spies[2].side_effect, spies[2].return_value = None, []
    await d.dispatch_job(db_session, job)
    assert (await db_session.get(Job, jid)).status == "completed"
    admission.assert_not_awaited()
    resolver.assert_not_awaited()


async def test_tombstone_only_beats_retired_generation(db_session, spies):
    job = await job_row(db_session)
    generation = await admit_claimed(db_session, job.id)
    job = await db_session.get(Job, generation.original_job_id)
    claim = g.Claim(job.id, job.raw_archive_id, generation.id, job.attempts, job.claimed_at)
    source = await db_session.get(RawArchiveItem, claim.archive_id)
    generation.status = "abandoned"
    (await db_session.get(Head, claim.archive_id)).candidate_generation_id = None
    db_session.add(Tombstone(source_id=claim.archive_id, content_hash=source.content_hash))
    await db_session.commit()
    assert await g.resolve_conflict(db_session, claim, g.DerivationConflict(
        g.ConflictReason.GENERATION_RETIRED)) == g.ConflictDisposition.CANCELLED
    current = await db_session.get(Job, claim.job_id, populate_existing=True)
    assert current.error_message == "derivation_cancelled:source_removed" and current.attempts == 1


async def test_resolver_precommit_failure_rolls_back_and_keeps_attempt(db_session, spies, monkeypatch):
    job = await job_row(db_session)
    generation = await admit_claimed(db_session, job.id)
    job = await db_session.get(Job, generation.original_job_id)
    claim = g.Claim(job.id, job.raw_archive_id, generation.id, job.attempts, job.claimed_at)
    with monkeypatch.context() as patch:
        patch.setattr(db_session, "commit", AsyncMock(side_effect=RuntimeError("synthetic precommit failure")))
        with pytest.raises(RuntimeError):
            await g.resolve_conflict(db_session, claim, g.DerivationConflict(g.ConflictReason.INVALID_PHASE))
    current = await db_session.get(Job, claim.job_id, populate_existing=True)
    assert current.status == "claimed" and current.error_message is None and current.attempts == 1


@pytest.mark.parametrize("field", ["job_id", "archive_id", "attempts", "claimed_at"])
async def test_admission_fences_each_captured_field(db_session, spies, field):
    from dataclasses import replace
    job = await job_row(db_session)
    claim = g.Claim(job.id, job.raw_archive_id, None, job.attempts, job.claimed_at)
    bad = {"job_id": uuid.uuid4(), "archive_id": uuid.uuid4(), "attempts": 99,
           "claimed_at": job.claimed_at + timedelta(seconds=1)}
    with pytest.raises(g.DerivationConflict):
        await g.admit_generation(db_session, job.id, expected_claim=replace(claim, **{field: bad[field]}))
    assert await db_session.scalar(text("SELECT count(*) FROM archive_derivation_heads")) == 0
    assert await db_session.scalar(text("SELECT count(*) FROM derivation_generations")) == 0
