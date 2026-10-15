"""PostgreSQL lifecycle/constraint/privacy boundaries, with synthetic records only."""
import asyncio
from datetime import datetime, timezone
import uuid
import pytest
from sqlalchemy import select, text, event
from sqlalchemy.exc import IntegrityError, DBAPIError
from sqlalchemy.ext.asyncio import async_sessionmaker
from app.domain.jobs.models import Job
from app.domain.derivations import service as g
from app.domain.derivations.models import DerivationGeneration as Gen, ArchiveDerivationHead as Head, DerivationFtsProjection as Fts
from app.domain.derived_memory.models import Fact
from app.domain.derivations.visibility import HISTORY
from tests.domain.test_derivation_generations import admit_claimed, claimed, managed

async def build(session, job, content='alpha managed', publish=True):
    jid, aid, attempt, at = job.id, job.raw_archive_id, job.attempts, job.claimed_at
    gen = await admit_claimed(session, jid)
    claim = g.Claim(jid, aid, gen.id, attempt, at)
    await g.persist_summary(session, claim, content)
    facts = await g.persist_extraction(session, claim, [{'fact_text':content,'tags':['visible'],'entities':['Anna']}])
    fid = facts[0].id
    await g.prepare_fts(session, claim, content, 'summary')
    if publish:
        await g.publish_generation(session, claim)
    return claim, fid

@pytest.mark.asyncio
@pytest.mark.parametrize('status',['failed','retryable_failed','pending_provider'])
async def test_manual_new_generation_supersedes_without_transfer(db_session, managed, status):
    from app.domain.jobs.service import reprocess_job, claim_next_job
    source, job = await claimed(db_session)
    jid, aid = job.id, source.id
    old, oldfid = await build(db_session, job)
    stamp = (await db_session.get(Head, aid)).legacy_retired_at
    job = await db_session.get(Job, jid)
    job.status = status
    await db_session.commit()
    reset = await reprocess_job(db_session, jid)
    newgid = reset.generation_id
    assert newgid != old.generation_id and reset.attempts == 0 and reset.status == 'pending'
    current = await claim_next_job(db_session)
    new, newfid = await build(db_session, current, 'beta new')
    assert new.generation_id == newgid and newfid != oldfid
    assert (await db_session.get(Gen, old.generation_id)).status == 'superseded'
    assert (await db_session.get(Head, aid)).legacy_retired_at == stamp
    assert [f.id for f in (await db_session.scalars(select(Fact))).all()] == [newfid]
    assert len((await db_session.scalars(select(Fact).execution_options(**HISTORY))).all()) == 2

@pytest.mark.asyncio
async def test_open_partial_unique_and_terminal_release(db_session, managed):
    source, job = await claimed(db_session)
    aid, jid = source.id, job.id
    gen = await admit_claimed(db_session, jid)
    gid = gen.id
    for status in ['building','prepared']:
        async with db_session.begin_nested():
            with pytest.raises(IntegrityError):
                async with db_session.begin_nested():
                    db_session.add(Gen(raw_archive_id=aid,original_job_id=jid,sequence=2,
                        creation_reason='manual_reprocess',source_content_hash='synthetic',status=status))
                    await db_session.flush()
    for status in ['failed','abandoned']:
        current = await db_session.get(Gen, gid)
        current.status = status
        await db_session.commit()
        async with db_session.begin_nested():
            contender = Gen(raw_archive_id=aid,original_job_id=jid,sequence=2,
                creation_reason='manual_reprocess',source_content_hash='synthetic')
            db_session.add(contender)
            await db_session.flush()
            await db_session.delete(contender)
        await db_session.commit()

@pytest.mark.asyncio
async def test_projection_uniqueness_and_archive_provenance(db_session, managed):
    source, job = await claimed(db_session)
    aid, jid, at = source.id, job.id, job.claimed_at
    gen = await admit_claimed(db_session, jid)
    gid = gen.id
    claim = g.Claim(jid, aid, gid, 1, at)
    await g.raw_baseline(db_session, claim)
    await g.raw_baseline(db_session, claim)
    assert await db_session.scalar(text('SELECT count(*) FROM derivation_fts_projections')) == 1
    other_source, _ = await claimed(db_session)
    other_aid = other_source.id
    for row in [Fts(raw_archive_id=aid,projection_kind='raw_fallback',text_content='duplicate'),
                Fact(raw_archive_id=other_aid,generation_id=gid,fact_text='wrong source',
                    source_span='',confidence_tier='low',derivation_method='test',derivation_model='test')]:
        with pytest.raises(IntegrityError):
            async with db_session.begin_nested():
                db_session.add(row)
                await db_session.flush()
    await g.persist_summary(db_session,claim,None,outcome='empty')
    await g.persist_extraction(db_session,claim,[])
    await g.prepare_fts(db_session,claim,'alpha','raw_fallback')
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(Fts(raw_archive_id=aid,generation_id=gid,projection_kind='summary',text_content='duplicate'))
            await db_session.flush()

@pytest.mark.asyncio
async def test_all_generation_erasure_and_permanent_retirement(db_session, managed, monkeypatch):
    from app.domain.derived_memory.service import write_facts, write_fts_entry
    from app.domain.archive import service as archive
    from app.domain.jobs.service import reprocess_job, claim_next_job
    from app.domain.derivations.privacy import assert_restore_safe
    source, job = await claimed(db_session)
    aid, jid = source.id, job.id
    await write_facts(db_session, aid, [{'fact_text':'alpha legacy'}])
    await write_fts_entry(db_session, aid, 'alpha legacy')
    job=await db_session.get(Job,jid)
    attempt,at=job.attempts,job.claimed_at
    gen=await admit_claimed(db_session,jid)
    await g.raw_baseline(db_session,g.Claim(jid,aid,gen.id,attempt,at))
    first, _ = await build(db_session, await db_session.get(Job,jid))
    job = await db_session.get(Job,jid)
    job.status = 'retryable_failed'
    await db_session.commit()
    await reprocess_job(db_session,jid)
    nextjob = await claim_next_job(db_session)
    second, _ = await build(db_session,nextjob,'alpha second')
    job = await db_session.get(Job,jid)
    job.status = 'retryable_failed'
    await db_session.commit()
    await reprocess_job(db_session,jid)
    nextjob = await claim_next_job(db_session)
    candidate, _ = await build(db_session,nextjob,'alpha private',publish=False)
    stamp = (await db_session.get(Head,aid)).legacy_retired_at
    with pytest.raises(RuntimeError, match='certified restore'):
        await assert_restore_safe(db_session)
    for stmt in ['UPDATE archive_derivation_heads SET legacy_retired_at=NULL',
                 'DELETE FROM archive_derivation_heads']:
        with pytest.raises(DBAPIError):
            async with db_session.begin_nested():
                await db_session.execute(text(stmt))
    monkeypatch.setattr(archive,'append_tombstone_ledger',lambda *a,**k:None)
    await archive.cascade_delete_archive_item(db_session,aid)
    head = await db_session.get(Head,aid,populate_existing=True)
    assert head.legacy_retired_at == stamp and head.active_generation_id is None and head.candidate_generation_id is None
    for table, column in [('facts','fact_text'),('summaries','summary_text'),('fts_entries','text_content'),('derivation_fts_projections','text_content')]:
        rows = (await db_session.execute(text(f'SELECT {column},source_status FROM {table}'))).all()
        assert rows and all(r[0] == archive.REDACTION_MARKER and r[1] == 'source_removed' for r in rows)
    assert not await db_session.scalar(text('SELECT count(*) FROM derivation_fts_projections WHERE search_vector IS NOT NULL'))
    assert (await db_session.get(Gen,candidate.generation_id,populate_existing=True)).status == 'abandoned'
    with pytest.raises(g.DerivationConflict):
        await g.publish_generation(db_session,candidate)
    with pytest.raises(g.DerivationConflict):
        await g.raw_baseline(db_session,second)

@pytest.mark.asyncio
async def test_stale_claim_cannot_finish_or_persist(db_session, managed):
    source, job = await claimed(db_session)
    jid, aid, at = job.id, source.id, job.claimed_at
    gen = await admit_claimed(db_session,jid)
    claim = g.Claim(jid,aid,gen.id,1,at)
    job = await db_session.get(Job,jid)
    job.attempts = 2
    await db_session.commit()
    with pytest.raises(g.DerivationConflict):
        await g.persist_extraction(db_session,claim,[{'fact_text':'stale'}])
    assert not await db_session.scalar(text('SELECT count(*) FROM facts'))

@pytest.mark.asyncio
async def test_lock_order_and_cross_archive_tag_race(db_session, test_engine, managed):
    first, job1 = await claimed(db_session)
    aid1,jid1,at1 = first.id,job1.id,job1.claimed_at
    second,job2 = await claimed(db_session)
    aid2,jid2,at2 = second.id,job2.id,job2.claimed_at
    locked = []
    def capture(conn,cursor,statement,parameters,context,executemany):
        if 'FOR UPDATE' in statement:
            locked.append(statement)
    event.listen(test_engine.sync_engine,'before_cursor_execute',capture)
    try:
        gen1 = await admit_claimed(db_session,jid1)
    finally:
        event.remove(test_engine.sync_engine,'before_cursor_execute',capture)
    assert len(locked) == 4
    assert all(f'FROM {name}' in query for name,query in zip(
        ['raw_archive','archive_derivation_heads','jobs','derivation_generations'],locked))
    gid1=gen1.id
    gen2 = await admit_claimed(db_session,jid2)
    c1,c2 = g.Claim(jid1,aid1,gid1,1,at1), g.Claim(jid2,aid2,gen2.id,1,at2)
    factory=async_sessionmaker(test_engine,expire_on_commit=False)
    async with factory() as left, factory() as right:
        await asyncio.wait_for(asyncio.gather(
            g.persist_extraction(left,c1,[{'fact_text':'first','tags':['z']},{'fact_text':'second','tags':['a']}]),
            g.persist_extraction(right,c2,[{'fact_text':'third','tags':['a']},{'fact_text':'fourth','tags':['z']}]),
        ),10)
    assert await db_session.scalar(text('SELECT count(*) FROM tags')) == 2
    assert await db_session.scalar(text('SELECT count(*) FROM fact_tags')) == 4

@pytest.mark.asyncio
async def test_restore_guard_dormant_and_retired_even_after_flag_off(db_session,managed,monkeypatch):
    from app.domain.derivations.privacy import assert_restore_safe
    from app.infrastructure.settings import get_settings
    with monkeypatch.context() as off:
        off.setattr(get_settings(),'generation_idempotency_enabled',False)
        await assert_restore_safe(db_session)
    with pytest.raises(RuntimeError,match='certified restore'):
        await assert_restore_safe(db_session)
    source,job=await claimed(db_session)
    await build(db_session,job)
    with monkeypatch.context() as off:
        off.setattr(get_settings(),'generation_idempotency_enabled',False)
        with pytest.raises(RuntimeError,match='certified restore'):
            await assert_restore_safe(db_session)

@pytest.mark.asyncio
async def test_dirty_cached_job_cannot_override_locked_source_status(db_session,managed):
    source,job=await claimed(db_session)
    aid,jid,at=source.id,job.id,job.claimed_at
    other,_=await claimed(db_session)
    other_aid=other.id
    job=await db_session.get(Job,jid)
    # Capture valid dispatch ownership BEFORE dirtying the identity map. Do not
    # use admit_claimed(): its populate_existing refresh would erase this proof.
    token=g.Claim(jid,aid,job.generation_id,job.attempts,at)
    source_hash=source.content_hash
    job.raw_archive_id=other_aid
    job.status='failed'
    job.attempts=99
    source.deleted_at=datetime.now(timezone.utc)
    source.content_hash='dirty-cache-only'
    assert job in db_session.dirty and source in db_session.dirty
    gen=await g.admit_generation(db_session,jid,expected_claim=token)
    assert gen.raw_archive_id==aid
    assert gen.source_content_hash==source_hash
    assert source.deleted_at is None and source.content_hash==source_hash
    current=await db_session.get(Job,jid,populate_existing=True)
    assert current.raw_archive_id==aid and current.generation_id==gen.id
    assert current.status=='claimed' and current.attempts==1 and current.claimed_at==at
    assert await db_session.scalar(text('SELECT count(*) FROM archive_derivation_heads'))==1
    current.status='failed'  # Cached eligibility is not manual authorization.
    with pytest.raises(g.DerivationConflict):
        await g.admit_generation(db_session,jid,manual=True)
    assert (await db_session.get(Job,jid,populate_existing=True)).status=='claimed'

@pytest.mark.asyncio
async def test_manual_two_session_claim_wins_reprocess_boundary(db_session,test_engine,managed):
    from app.domain.jobs.service import reprocess_job,ReprocessConflict
    source,job=await claimed(db_session)
    jid=job.id
    job.status='failed'
    await db_session.commit()
    factory=async_sessionmaker(test_engine,expire_on_commit=False)
    arrived=asyncio.Event()
    def capture(conn,cursor,statement,parameters,context,executemany):
        if 'FROM jobs' in statement and 'FOR UPDATE' in statement:
            arrived.set()
    async with factory() as worker, factory() as manual:
        locked=await worker.scalar(select(Job).where(Job.id==jid).with_for_update())
        stamp=datetime.now(timezone.utc)
        locked.status='claimed'
        locked.claimed_at=stamp
        await worker.flush()
        event.listen(test_engine.sync_engine,'before_cursor_execute',capture)
        task=asyncio.create_task(reprocess_job(manual,jid))
        try:
            await asyncio.wait_for(arrived.wait(),10)
            assert not task.done()
            await worker.commit()
            with pytest.raises(ReprocessConflict):
                await asyncio.wait_for(task,10)
        finally:
            event.remove(test_engine.sync_engine,'before_cursor_execute',capture)
            if not task.done():
                task.cancel()
    current=await db_session.get(Job,jid,populate_existing=True)
    assert current.status=='claimed' and current.claimed_at==stamp and current.attempts==1
    assert await db_session.scalar(text('SELECT count(*) FROM derivation_generations'))==0

@pytest.mark.asyncio
async def test_terminal_owner_reconciled_before_replacement(db_session,managed):
    from app.domain.jobs.service import fail_job,reprocess_job
    _,job=await claimed(db_session)
    jid=job.id
    gen=await admit_claimed(db_session,jid)
    oldgid=gen.id
    current=await db_session.get(Job,jid)
    # Simulate PR #14's job-only terminal commit and a crash before closure.
    await fail_job(db_session,current,error='synthetic terminal',retryable=False)
    assert (await db_session.get(Gen,oldgid)).status=='building'
    new=await reprocess_job(db_session,jid)
    assert new.generation_id != oldgid
    assert (await db_session.get(Gen,oldgid,populate_existing=True)).status=='failed'
    assert await db_session.scalar(text("SELECT count(*) FROM derivation_generations WHERE status IN ('building','prepared')"))==1

@pytest.mark.asyncio
async def test_two_session_publication_one_switch(db_session,test_engine,managed):
    _,job=await claimed(db_session)
    claim,_=await build(db_session,job,publish=False)
    factory=async_sessionmaker(test_engine,expire_on_commit=False)
    async with factory() as left, factory() as right:
        result=await asyncio.gather(g.publish_generation(left,claim),g.publish_generation(right,claim))
    assert sorted(result)==[False,True]
    assert await db_session.scalar(text("SELECT count(*) FROM derivation_generations WHERE status='active'"))==1
    assert await db_session.scalar(text('SELECT count(*) FROM derivation_fts_projections'))==1

@pytest.mark.asyncio
@pytest.mark.parametrize('phase',['fts','publication'])
async def test_precommit_failure_preserves_previous_publication(db_session,managed,monkeypatch,phase):
    from app.domain.jobs.service import reprocess_job,claim_next_job
    _,job=await claimed(db_session)
    jid=job.id
    old,oldfid=await build(db_session,job,'alpha previous')
    job=await db_session.get(Job,jid)
    job.status='retryable_failed'
    await db_session.commit()
    await reprocess_job(db_session,jid)
    job=await claim_next_job(db_session)
    aid,gid,attempt,at=job.raw_archive_id,job.generation_id,job.attempts,job.claimed_at
    claim=g.Claim(jid,aid,gid,attempt,at)
    await g.persist_summary(db_session,claim,None,outcome='empty')
    newfacts=await g.persist_extraction(db_session,claim,[{'fact_text':'alpha replacement'}])
    newfid=newfacts[0].id
    if phase=='publication':
        await g.prepare_fts(db_session,claim,'alpha replacement','raw_fallback')
    async def before_commit():
        raise RuntimeError('synthetic failure before commit')
    with monkeypatch.context() as patch:
        patch.setattr(db_session,'commit',before_commit)
        with pytest.raises(RuntimeError):
            if phase=='fts':
                await g.prepare_fts(db_session,claim,'alpha replacement','raw_fallback')
            else:
                await g.publish_generation(db_session,claim)
    head=await db_session.get(Head,aid,populate_existing=True)
    assert head.active_generation_id==old.generation_id and head.candidate_generation_id==gid
    assert (await db_session.get(Gen,gid)).status==('building' if phase=='fts' else 'prepared')
    assert [f.id for f in (await db_session.scalars(select(Fact))).all()]==[oldfid]
    await g.prepare_fts(db_session,claim,'alpha replacement','raw_fallback')
    await g.publish_generation(db_session,claim)
    assert [f.id for f in (await db_session.scalars(select(Fact))).all()]==[newfid]

@pytest.mark.asyncio
async def test_publication_rechecks_required_output_consistency(db_session,managed):
    _,job=await claimed(db_session)
    claim,_=await build(db_session,job,publish=False)
    await db_session.execute(text("UPDATE derivation_stage_completions SET fact_count=2 WHERE stage_name='extraction'"))
    await db_session.commit()
    with pytest.raises(g.DerivationConflict,match='inconsistent'):
        await g.publish_generation(db_session,claim)
    assert (await db_session.get(Head,claim.archive_id)).active_generation_id is None
    assert (await db_session.get(Gen,claim.generation_id)).status=='prepared'

@pytest.mark.asyncio
@pytest.mark.parametrize('r2_on', [False, True])
async def test_bundle_tombstones_lock_all_archives_before_suppression(
    db_session, test_engine, managed, monkeypatch, r2_on,
):
    from app.api.routes import portability
    from app.infrastructure.settings import get_settings
    from app.domain.archive.tombstones import REDACTION_MARKER

    sources = []
    for _ in range(2):
        source, job = await claimed(db_session)
        aid, content_hash = source.id, source.content_hash
        claim, _ = await build(db_session, job)
        stamp = (await db_session.get(Head, aid)).legacy_retired_at
        sources.append((aid, content_hash, claim.generation_id, stamp))
    unrelated, _ = await claimed(db_session)
    unrelated_id, unrelated_text = unrelated.id, unrelated.raw_content
    # Bundle order is deliberately opposite the required archive lock order.
    sources.sort(key=lambda record: str(record[0]), reverse=True)
    bundle = [{'content_hash': record[1]} for record in sources]
    bundle += [None, {}, dict(bundle[0])]  # Ignore empty entries; replay is unchanged.
    expected_ids = {record[0] for record in sources}
    expected_hashes = {record[1] for record in sources}
    locked = []
    suppressed = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        if 'FOR UPDATE' in statement:
            locked.append((statement, parameters))

    original_suppress = portability._suppress_derived
    async def suppress(session, archive_id):
        # No archive/head alternation: ALL archive rows, then ALL heads, then
        # generation rows have already been locked before the first suppression.
        if not suppressed:
            assert len(locked) == 3
            for name, (sql, _) in zip(
                ['raw_archive', 'archive_derivation_heads', 'derivation_generations'], locked,
            ):
                assert f'FROM {name}' in sql and f'ORDER BY {name}.' in sql
            assert 'raw_archive.content_hash IN' in locked[0][0]
            assert set(locked[0][1]) == expected_hashes
            assert set(locked[1][1]) == expected_ids
        suppressed.append(uuid.UUID(archive_id))
        await original_suppress(session, archive_id)

    monkeypatch.setattr(get_settings(), 'generation_idempotency_enabled', r2_on)
    monkeypatch.setattr(portability, '_suppress_derived', suppress)
    monkeypatch.setattr(portability, 'append_tombstone_ledger', lambda *a, **k: None)
    event.listen(test_engine.sync_engine, 'before_cursor_execute', capture)
    hashes = set()
    try:
        assert await portability._apply_bundle_tombstones(db_session, bundle, hashes) == 2
        await db_session.commit()
    finally:
        event.remove(test_engine.sync_engine, 'before_cursor_execute', capture)
    assert hashes == expected_hashes
    assert suppressed == [record[0] for record in sources]
    for aid, _, gid, stamp in sources:
        head = await db_session.get(Head, aid, populate_existing=True)
        assert head.active_generation_id is None and head.legacy_retired_at == stamp
        # M3: clearing the serving head also terminally withdraws its generation.
        assert (await db_session.get(Gen, gid, populate_existing=True)).status == 'abandoned'
        source = await db_session.get(type(unrelated), aid, populate_existing=True)
        assert source.deleted_at is not None and source.raw_content == REDACTION_MARKER
        projection = (await db_session.execute(text('SELECT text_content,source_status,search_vector '
            'FROM derivation_fts_projections WHERE raw_archive_id=:id'), {'id': aid})).one()
        assert tuple(projection) == (REDACTION_MARKER, 'source_removed', None)
    untouched = await db_session.get(type(unrelated), unrelated_id, populate_existing=True)
    assert untouched.deleted_at is None and untouched.raw_content == unrelated_text
