"""Explicit completed reprocess CAS; synthetic PostgreSQL, no provider calls."""
import asyncio
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select, text, event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.domain.derivations import service as g
from app.domain.derivations.models import ArchiveDerivationHead as Head, DerivationGeneration as Gen, DerivationFtsProjection as Projection, DerivationStageCompletion as Completion
from app.domain.derived_memory.models import Fact, Summary
from app.domain.jobs.models import Job
from app.domain.jobs.service import reprocess_job, ReprocessConflict, claim_next_job
from tests.domain.test_derivation_generations import claimed, managed, admit_claimed
from tests.domain.test_derivation_boundaries import build


async def completed(session, published=False):
    source, job = await claimed(session)
    aid, jid = source.id, job.id
    if published:
        claim, _ = await build(session, job)
        await g.finish_claim(session, claim)
    else:
        session.add(Fact(raw_archive_id=aid, fact_text='PRIVATE_OLD_FACT', source_span='PRIVATE_SPAN',
                         confidence_tier='low', derivation_method='synthetic', derivation_model='synthetic'))
        session.add(Projection(raw_archive_id=aid, generation_id=None, projection_kind='raw_fallback', text_content=source.raw_content))
        job.status = 'completed'
        job.completed_at = datetime.now(timezone.utc)
        await session.commit()
    job = await session.get(Job, jid, populate_existing=True)
    job.error_message = 'PRIVATE_JOB_ERROR'
    await session.commit()
    return aid, jid, job.generation_id


async def state(session):
    result = {}
    for model in (Job, Gen, Head, Fact, Summary, Projection, Completion):
        table = model.__table__
        result[table.name] = [dict(row) for row in (await session.execute(select(table).order_by(*table.primary_key.columns))).mappings()]
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize('blank_head', [False, True])
async def test_legacy_completed_candidate_then_normal_fenced_worker(db_session, managed, blank_head):
    aid, jid, _ = await completed(db_session)
    if blank_head:
        db_session.add(Head(raw_archive_id=aid))
        await db_session.commit()
    before = await state(db_session)
    job = await reprocess_job(db_session, jid, expected_generation='legacy')
    gid = job.generation_id
    assert job.status == 'pending' and job.attempts == 0 and job.error_message is None and job.completed_at is None
    unchanged = set(before['jobs'][0]) - {'generation_id', 'status', 'attempts', 'error_message', 'completed_at', 'updated_at'}
    after_job = (await state(db_session))['jobs'][0]
    assert {k: after_job[k] for k in unchanged} == {k: before['jobs'][0][k] for k in unchanged}
    gen = await db_session.get(Gen, gid)
    assert gen.creation_reason == 'manual_reprocess' and gen.status == 'building' and gen.predecessor_generation_id is None
    head = await db_session.get(Head, aid)
    assert head is not None and head.active_generation_id is None
    assert head.candidate_generation_id == gid and head.legacy_retired_at is None
    requested_head = (await state(db_session))['archive_derivation_heads']
    assert (await state(db_session))['facts'] == before['facts']
    assert (await state(db_session))['derivation_fts_projections'] == before['derivation_fts_projections']
    assert len((await db_session.scalars(select(Fact))).all()) == 1
    await db_session.rollback()
    worker_job = await claim_next_job(db_session)
    admitted = await admit_claimed(db_session, worker_job.id)
    assert admitted.id == gid
    assert (await state(db_session))['archive_derivation_heads'] == requested_head
    assert len((await db_session.scalars(select(Gen))).all()) == 1
    head = await db_session.get(Head, aid)
    assert head.candidate_generation_id == gid and head.active_generation_id is None and head.legacy_retired_at is None
    assert [f.fact_text for f in (await db_session.scalars(select(Fact))).all()] == ['PRIVATE_OLD_FACT']
    claim, _ = await build(db_session, worker_job, 'synthetic replacement')
    assert claim.generation_id == gid
    assert [f.fact_text for f in (await db_session.scalars(select(Fact))).all()] == ['synthetic replacement']
    assert (await db_session.get(Head, aid)).legacy_retired_at is not None
    assert len((await db_session.scalars(select(Fact).execution_options(**g.HISTORY))).all()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('published', [False, True])
async def test_replay_conflicts_without_second_generation_or_reset(db_session, managed, published):
    aid, jid, old = await completed(db_session, published)
    expected = old if published else 'legacy'
    first = await reprocess_job(db_session, jid, expected_generation=expected)
    gid = first.generation_id
    assert (await db_session.get(Head, aid)).candidate_generation_id == gid
    before = await state(db_session)
    with pytest.raises(ReprocessConflict) as exc:
        await reprocess_job(db_session, jid, expected_generation=expected)
    assert exc.value.reason == 'expected_generation_mismatch'
    assert exc.value.current_generation_id == gid and exc.value.current_status == 'pending'
    assert await state(db_session) == before


@pytest.mark.asyncio
async def test_managed_successor_preserves_active_until_publication(db_session, managed):
    aid, jid, old = await completed(db_session, True)
    before = await state(db_session)
    reset = await reprocess_job(db_session, jid, expected_generation=old)
    gid = reset.generation_id
    new = await db_session.get(Gen, gid)
    assert new.predecessor_generation_id == old and new.sequence == 2 and new.status == 'building'
    assert (await db_session.get(Gen, old)).status == 'active'
    expected_head = {**before['archive_derivation_heads'][0], 'candidate_generation_id': gid}
    assert (await state(db_session))['archive_derivation_heads'] == [expected_head]
    assert [row for row in (await state(db_session))['derivation_generations'] if row['id'] == old] == before['derivation_generations']
    assert (await state(db_session))['facts'] == before['facts']
    await db_session.rollback()
    worker_job = await claim_next_job(db_session)
    claim, _ = await build(db_session, worker_job, 'new successor')
    assert claim.generation_id == gid
    assert (await db_session.get(Head, aid)).active_generation_id == gid
    assert (await db_session.get(Gen, old)).status == 'superseded'
    assert len((await db_session.scalars(select(Gen))).all()) == 2


@pytest.mark.asyncio
async def test_stale_managed_expected_preserves_every_row(db_session, managed):
    _, jid, _ = await completed(db_session, True)
    before = await state(db_session)
    with pytest.raises(ReprocessConflict) as exc:
        await reprocess_job(db_session, jid, expected_generation=uuid.uuid4())
    assert exc.value.reason == 'expected_generation_mismatch'
    assert await state(db_session) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('expected', ['legacy', uuid.UUID(int=1)])
async def test_completed_off_rejected_without_mutation(db_session, expected):
    _, jid, _ = await completed(db_session)
    before = await state(db_session)
    with pytest.raises(ReprocessConflict) as exc:
        await reprocess_job(db_session, jid, expected_generation=expected)
    assert exc.value.reason == 'generation_reprocess_disabled'
    assert await state(db_session) == before


@pytest.mark.asyncio
async def test_completed_missing_precondition_not_legacy(db_session, managed):
    _, jid, _ = await completed(db_session)
    before = await state(db_session)
    with pytest.raises(ReprocessConflict):
        await reprocess_job(db_session, jid)
    assert await state(db_session) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('privacy', ['deleted', 'source_id', 'hash'])
async def test_completed_source_privacy_fails_closed(db_session, managed, privacy):
    from app.domain.archive.models import RawArchiveItem, Tombstone
    aid, jid, _ = await completed(db_session)
    source = await db_session.get(RawArchiveItem, aid)
    if privacy == 'deleted':
        source.deleted_at = datetime.now(timezone.utc)
    else:
        db_session.add(Tombstone(source_id=aid if privacy == 'source_id' else uuid.uuid4(),
            content_hash='unrelated-hash' if privacy == 'source_id' else source.content_hash))
    await db_session.commit()
    before = await state(db_session)
    with pytest.raises(ReprocessConflict):
        await reprocess_job(db_session, jid, expected_generation='legacy')
    assert await state(db_session) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('invalid', ['abandoned', 'headless', 'outputs', 'source_changed'])
async def test_invalid_managed_current_rejected(db_session, managed, invalid):
    from app.domain.archive.models import RawArchiveItem
    aid, jid, gid = await completed(db_session, True)
    if invalid == 'abandoned':
        (await db_session.get(Gen, gid)).status = 'abandoned'
        (await db_session.get(Head, aid)).active_generation_id = None
    elif invalid == 'headless':
        (await db_session.get(Head, aid)).active_generation_id = None
    elif invalid == 'outputs':
        await db_session.execute(text("DELETE FROM derivation_stage_completions WHERE generation_id=:gid AND stage_name='extraction'"), {'gid':gid})
    else:
        (await db_session.get(RawArchiveItem, aid)).content_hash = 'changed'
    await db_session.commit()
    before = await state(db_session)
    with pytest.raises(ReprocessConflict):
        await reprocess_job(db_session, jid, expected_generation=gid)
    assert await state(db_session) == before


@pytest.mark.asyncio
async def test_cross_archive_job_generation_corruption_fails_closed(db_session, managed):
    _, first, _ = await completed(db_session)
    _, other, _ = await completed(db_session)
    different = await reprocess_job(db_session, other, expected_generation='legacy')
    gid = different.generation_id
    await db_session.execute(text("SET LOCAL session_replication_role='replica'"))
    await db_session.execute(text('UPDATE jobs SET generation_id=:gid WHERE id=:jid'), {'gid':gid,'jid':first})
    await db_session.commit()
    before = await state(db_session)
    with pytest.raises(ReprocessConflict):
        await reprocess_job(db_session, first, expected_generation=gid)
    assert await state(db_session) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('head_state', ['absent', 'blank', 'active'])
async def test_transaction_failure_rolls_back_and_retry_can_succeed(db_session, managed, monkeypatch, head_state):
    aid, jid, old = await completed(db_session, head_state == 'active')
    expected = old if head_state == 'active' else 'legacy'
    if head_state == 'blank':
        db_session.add(Head(raw_archive_id=aid))
        await db_session.commit()
    before = await state(db_session)
    async def fail_commit():
        raise RuntimeError('synthetic precommit failure')
    with monkeypatch.context() as patch:
        patch.setattr(db_session, 'commit', fail_commit)
        with pytest.raises(RuntimeError):
            await reprocess_job(db_session, jid, expected_generation=expected)
    assert await state(db_session) == before
    result = await reprocess_job(db_session, jid, expected_generation=expected)
    assert result.status == 'pending'
    assert (await db_session.get(Head, aid)).candidate_generation_id == result.generation_id


@pytest.mark.asyncio
async def test_job_gone_before_locked_read_does_not_commit_blank_head(db_session, test_engine, managed, monkeypatch):
    aid, jid, _ = await completed(db_session)
    original = g.locked_state
    async def remove_before_lock(session, *args, **kwargs):
        async with async_sessionmaker(test_engine)() as other:
            await other.execute(text('DELETE FROM jobs WHERE id=:jid'), {'jid': jid})
            await other.commit()
        return await original(session, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(g, 'locked_state', remove_before_lock)
        assert await reprocess_job(db_session, jid, expected_generation='legacy') is None
    assert await db_session.get(Job, jid) is None
    assert await db_session.get(Head, aid) is None
    assert not (await db_session.scalars(select(Gen))).all()


@pytest.mark.asyncio
@pytest.mark.parametrize('published', [False, True])
async def test_two_session_same_expected_serializes_once(db_session, test_engine, managed, monkeypatch, published):
    aid, jid, old = await completed(db_session, published)
    expected = old if published else 'legacy'
    await db_session.rollback()
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    locked, release = asyncio.Event(), asyncio.Event()
    original = g.locked_state
    observer_engine = create_async_engine(test_engine.url, pool_size=1, max_overflow=0)
    async with factory() as first, factory() as second, async_sessionmaker(observer_engine)() as observer:
        async def pause(session, *args, **kwargs):
            result = await original(session, *args, **kwargs)
            if session is first:
                locked.set()
                await release.wait()
            return result
        with monkeypatch.context() as patch:
            patch.setattr(g, 'locked_state', pause)
            one = asyncio.create_task(reprocess_job(first, jid, expected_generation=expected))
            await asyncio.wait_for(locked.wait(), 5)
            two = asyncio.create_task(reprocess_job(second, jid, expected_generation=expected))
            blocked = False
            try:
                for _ in range(100):
                    blocked = bool(await observer.scalar(text("SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() AND pid<>pg_backend_pid() AND cardinality(pg_blocking_pids(pid))>0")))
                    await observer.rollback()
                    if blocked: break
                    await asyncio.sleep(0.01)
            finally:
                release.set()
                results = await asyncio.gather(one, two, return_exceptions=True)
            assert blocked
            assert sum(isinstance(r, ReprocessConflict) for r in results) == 1
            assert sum(isinstance(r, Job) for r in results) == 1
    await observer_engine.dispose()
    assert len((await db_session.scalars(select(Gen))).all()) == (2 if published else 1)
    head = await db_session.get(Head, aid)
    winner = next(r for r in results if isinstance(r, Job))
    assert head.candidate_generation_id == winner.generation_id
    assert head.active_generation_id == old and (published or head.legacy_retired_at is None)
    assert await db_session.scalar(select(g.func.count(Gen.id)).where(Gen.status.in_(['building', 'prepared']))) == 1


@pytest.mark.asyncio
async def test_lost_ack_after_worker_failure_still_cas_fenced(db_session, managed):
    _, jid, _ = await completed(db_session)
    first=await reprocess_job(db_session,jid,expected_generation='legacy')
    first.status='failed'
    await db_session.commit()
    before=await state(db_session)
    with pytest.raises(ReprocessConflict) as exc:
        await reprocess_job(db_session,jid,expected_generation='legacy')
    assert exc.value.reason == 'expected_generation_mismatch'
    assert await state(db_session) == before


@pytest.mark.asyncio
async def test_dirty_cache_cannot_override_completed_binding(db_session, managed):
    _, jid, _ = await completed(db_session)
    first = await reprocess_job(db_session, jid, expected_generation='legacy')
    gid = first.generation_id
    first.generation_id, first.status = None, 'completed'
    with pytest.raises(ReprocessConflict) as exc:
        await reprocess_job(db_session, jid, expected_generation='legacy')
    assert exc.value.current_generation_id == gid and exc.value.current_status == 'pending'


@pytest.mark.asyncio
async def test_real_commit_ack_loss_fresh_session_cannot_repeat(db_session, test_engine, managed, monkeypatch):
    _, jid, _ = await completed(db_session)
    original = db_session.commit
    async def lost_ack():
        await original()
        raise RuntimeError('synthetic acknowledgement loss')
    with monkeypatch.context() as patch:
        patch.setattr(db_session, 'commit', lost_ack)
        with pytest.raises(RuntimeError, match='acknowledgement loss'):
            await reprocess_job(db_session, jid, expected_generation='legacy')
    async with async_sessionmaker(test_engine, expire_on_commit=False)() as fresh:
        before = await state(fresh)
        gid = before['jobs'][0]['generation_id']
        assert gid is not None and len(before['derivation_generations']) == 1
        assert before['archive_derivation_heads'][0]['candidate_generation_id'] == gid
        assert before['archive_derivation_heads'][0]['active_generation_id'] is None
        assert before['archive_derivation_heads'][0]['legacy_retired_at'] is None
        with pytest.raises(ReprocessConflict) as exc:
            await reprocess_job(fresh, jid, expected_generation='legacy')
        assert exc.value.current_generation_id == gid
        assert await state(fresh) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('published', [False, True])
async def test_completed_request_lock_order(db_session, test_engine, managed, published):
    _, jid, gid = await completed(db_session, published)
    statements = []
    phases = []
    def record(connection, cursor, statement, parameters, context, executemany):
        if 'FOR UPDATE' in statement:
            statements.append(statement)
        if 'FOR UPDATE' in statement or 'INSERT INTO archive_derivation_heads' in statement:
            phases.append(statement)
    event.listen(test_engine.sync_engine, 'before_cursor_execute', record)
    try:
        await reprocess_job(db_session, jid, expected_generation=gid if published else 'legacy')
    finally:
        event.remove(test_engine.sync_engine, 'before_cursor_execute', record)
    assert len(statements) == 4
    for statement, table in zip(statements, ('raw_archive', 'archive_derivation_heads', 'jobs', 'derivation_generations')):
        assert f'FROM {table}' in statement
    assert 'ORDER BY jobs.id' in statements[2]
    assert 'ORDER BY derivation_generations.id' in statements[3]
    assert len(phases) == 5
    assert 'FROM raw_archive' in phases[0] and 'FOR UPDATE' in phases[0]
    assert 'INSERT INTO archive_derivation_heads' in phases[1]
    assert 'FROM archive_derivation_heads' in phases[2] and 'FOR UPDATE' in phases[2]


@pytest.mark.asyncio
async def test_legacy_retired_head_does_not_revive_fallback(db_session, managed):
    aid, jid, _ = await completed(db_session)
    db_session.add(Head(raw_archive_id=aid, legacy_retired_at=datetime.now(timezone.utc)))
    await db_session.commit()
    before = await state(db_session)
    with pytest.raises(ReprocessConflict):
        await reprocess_job(db_session, jid, expected_generation='legacy')
    assert await state(db_session) == before


@pytest.mark.asyncio
async def test_deletion_between_request_and_worker_never_resurrects(db_session, managed, tmp_path):
    from app.domain.archive.service import cascade_delete_archive_item
    aid, jid, _ = await completed(db_session)
    result = await reprocess_job(db_session, jid, expected_generation='legacy')
    gid = result.generation_id
    await cascade_delete_archive_item(db_session, aid, backup_dir=str(tmp_path))
    before = await state(db_session)
    # Privacy withdrawal abandons the attached candidate; reprocess cannot revive it.
    assert (await db_session.get(Gen, gid)).status == 'abandoned'
    head = await db_session.get(Head, aid)
    assert head.active_generation_id is None and head.candidate_generation_id is None
    with pytest.raises(ReprocessConflict):
        await reprocess_job(db_session, jid, expected_generation=gid)
    assert await state(db_session) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['failed', 'retryable_failed', 'pending_provider'])
@pytest.mark.parametrize('explicit', [False, True])
async def test_pr18_failure_reset_of_attached_candidate(db_session, managed, status, explicit):
    aid, jid, _ = await completed(db_session)
    first = await reprocess_job(db_session, jid, expected_generation='legacy')
    gid = first.generation_id
    assert (await db_session.get(Head, aid)).candidate_generation_id == gid
    # A crash/provider boundary leaves the existing ordinary candidate lifecycle.
    first.status = status
    await db_session.commit()
    result = await reprocess_job(db_session, jid, **({'expected_generation': gid} if explicit else {}))
    assert result.status == 'pending' and result.generation_id != gid
    assert (await db_session.get(Gen, gid)).status == ('failed' if status == 'failed' else 'abandoned')
    head = await db_session.get(Head, aid)
    assert head.candidate_generation_id == result.generation_id
    assert head.active_generation_id is None and head.legacy_retired_at is None
    assert len((await db_session.scalars(select(Gen))).all()) == 2
