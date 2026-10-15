"""Isolated generation durability; all providers are stubbed."""
import uuid
from datetime import datetime, timezone
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker
from app.domain.archive.models import RawArchiveItem
from app.domain.jobs.models import Job
from app.infrastructure.settings import get_settings

@pytest.fixture
def managed(monkeypatch):
    monkeypatch.setattr(get_settings(), "generation_idempotency_enabled", True)

async def claimed(session):
    source = RawArchiveItem(source_type="test", raw_content="unique alpha source", content_hash=uuid.uuid4().hex)
    session.add(source)
    await session.flush()
    job = Job(raw_archive_id=source.id, job_type="extract", status="claimed", attempts=1,
              claimed_at=datetime.now(timezone.utc))
    session.add(job)
    await session.commit()
    return source, job

async def admit_claimed(session, job_id):
    """Capture durable test ownership explicitly, as dispatch must do."""
    from app.domain.derivations import service as g
    job = await session.get(Job, job_id, populate_existing=True,
                            execution_options={"autoflush": False})
    token = g.Claim(job.id, job.raw_archive_id, job.generation_id, job.attempts, job.claimed_at)
    return await g.admit_generation(session, job_id, expected_claim=token)

@pytest.mark.asyncio
async def test_dormant_admission(db_session):
    from app.domain.derivations.service import admit_generation, DerivationConflict
    _, job = await claimed(db_session)
    with pytest.raises(DerivationConflict):
        await admit_claimed(db_session, job.id)

@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [[], [{"fact_text": "alpha assertion", "source_span": "alpha", "tags": ["one"], "entities": ["Anna"]}]])
async def test_replay_stable_ids_and_atomic_publication(db_session, managed, batch):
    from app.domain.derivations.service import admit_generation, Claim, persist_extraction, persist_summary, prepare_fts, publish_generation
    from app.domain.derivations.models import ArchiveDerivationHead, DerivationFtsProjection, DerivationStageCompletion
    source, job = await claimed(db_session)
    jid, aid, attempts, at = job.id, source.id, job.attempts, job.claimed_at
    generation = await admit_claimed(db_session, jid)
    gid = generation.id
    claim = Claim(jid, aid, gid, attempts, at)
    await persist_summary(db_session, claim, None, outcome="empty")
    first = await persist_extraction(db_session, claim, batch)
    ids = [f.id for f in first]
    second = await persist_extraction(db_session, claim, [{"fact_text": "losing output"}])
    assert [f.id for f in second] == ids
    projection = await prepare_fts(db_session, claim, "alpha searchable", "raw_fallback")
    pid = projection.id
    assert await publish_generation(db_session, claim) is True
    assert await publish_generation(db_session, claim) is False
    head = await db_session.get(ArchiveDerivationHead, aid)
    assert head.active_generation_id == gid and head.legacy_retired_at is not None
    assert (await db_session.get(DerivationFtsProjection, pid)).generation_id == gid
    completion = await db_session.get(DerivationStageCompletion, (gid, "extraction"))
    assert completion.fact_count == len(batch)
    await db_session.rollback()

@pytest.mark.asyncio
async def test_two_session_extraction_completion(db_session, test_engine, managed):
    import asyncio
    from app.domain.derivations.service import admit_generation, Claim, persist_extraction
    source, job = await claimed(db_session)
    jid, aid, attempt, at = job.id, source.id, job.attempts, job.claimed_at
    generation = await admit_claimed(db_session, jid)
    claim = Claim(jid, aid, generation.id, attempt, at)
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as left, factory() as right:
        results = await asyncio.gather(
            persist_extraction(left, claim, [{"fact_text": "winner one"}]),
            persist_extraction(right, claim, [{"fact_text": "winner two"}]),
        )
        assert [f.id for f in results[0]] == [f.id for f in results[1]]
    assert (await db_session.execute(text("SELECT count(*) FROM facts"))).scalar_one() == 1

@pytest.mark.asyncio
async def test_candidates_history_and_retired_headless(db_session, managed):
    from sqlalchemy import select
    from app.domain.derived_memory.service import write_facts, write_fts_entry
    from app.domain.derived_memory.models import Fact
    from app.domain.derivations.service import admit_generation, Claim, persist_summary, persist_extraction, prepare_fts, publish_generation
    from app.domain.derivations.models import ArchiveDerivationHead
    from app.domain.retrieval.service import _keyword_candidates
    source, job = await claimed(db_session)
    aid, jid, attempt, at = source.id, job.id, job.attempts, job.claimed_at
    legacy = await write_facts(db_session, aid, [{'fact_text':'alpha legacy'}])
    legacy_id = legacy[0].id
    await write_fts_entry(db_session, aid, 'alpha legacy')
    assert [f.id for f in (await db_session.scalars(select(Fact))).all()] == [legacy_id]
    gen = await admit_claimed(db_session, jid)
    claim = Claim(jid, aid, gen.id, attempt, at)
    await persist_summary(db_session, claim, None, outcome='empty')
    new = await persist_extraction(db_session, claim, [{'fact_text':'alpha managed'}])
    new_id = new[0].id
    await prepare_fts(db_session, claim, 'alpha managed', 'raw_fallback')
    assert [f.id for f in (await db_session.scalars(select(Fact))).all()] == [legacy_id]
    await publish_generation(db_session, claim)
    assert [f.id for f in (await db_session.scalars(select(Fact))).all()] == [new_id]
    head = await db_session.get(ArchiveDerivationHead, aid)
    head.active_generation_id = None
    await db_session.commit()
    assert not (await db_session.scalars(select(Fact))).all()
    assert not await _keyword_candidates(db_session, 'alpha')

@pytest.mark.asyncio
async def test_two_jobs_cannot_admit_two_candidates(db_session, test_engine, managed):
    import asyncio
    from app.domain.derivations.service import admit_generation, DerivationConflict
    source, first = await claimed(db_session)
    first_id = first.id
    second = Job(raw_archive_id=source.id, job_type='extract', status='claimed', attempts=1,
                 claimed_at=datetime.now(timezone.utc))
    db_session.add(second)
    await db_session.commit()
    second_id = second.id
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as left, factory() as right:
        results = await asyncio.gather(admit_claimed(left, first_id), admit_claimed(right, second_id), return_exceptions=True)
    assert all(isinstance(r, DerivationConflict) for r in results)
    assert all(r.reason.value == 'state_inconsistent' for r in results)
    assert (await db_session.execute(text('SELECT count(*) FROM derivation_generations'))).scalar_one() == 0

@pytest.mark.asyncio
async def test_extraction_rollback_has_no_partial_batch(db_session, managed, monkeypatch):
    from app.domain.derivations.service import admit_generation, Claim, persist_extraction
    import app.domain.derived_memory.service as writers
    source, job = await claimed(db_session)
    aid, jid, attempt, at = source.id, job.id, job.attempts, job.claimed_at
    gen = await admit_claimed(db_session, jid)
    claim = Claim(jid, aid, gen.id, attempt, at)
    async def fail_tags(*args, **kwargs):
        raise RuntimeError('synthetic tag persistence failure')
    monkeypatch.setattr(writers, 'write_tags', fail_tags)
    with pytest.raises(RuntimeError):
        await persist_extraction(db_session, claim, [{'fact_text':'alpha', 'tags':['tag']}])
    for table in ('facts','fact_tags','derivation_stage_completions'):
        assert (await db_session.execute(text(f'SELECT count(*) FROM {table}'))).scalar_one() == 0

@pytest.mark.asyncio
async def test_completed_reprocess_still_blocked_managed(db_session, managed):
    from app.domain.jobs.service import reprocess_job, ReprocessConflict
    _, job = await claimed(db_session)
    jid = job.id
    job.status = 'completed'
    await db_session.commit()
    with pytest.raises(ReprocessConflict):
        await reprocess_job(db_session, jid)
    assert (await db_session.get(Job, jid)).status == 'completed'

@pytest.mark.asyncio
async def test_lost_publication_ack_is_idempotent(db_session, managed, monkeypatch):
    from app.domain.derivations.service import admit_generation, Claim, persist_summary, persist_extraction, prepare_fts, publish_generation
    from app.domain.derivations.models import ArchiveDerivationHead
    source, job = await claimed(db_session)
    aid, jid, attempt, at = source.id, job.id, job.attempts, job.claimed_at
    gen = await admit_claimed(db_session, jid)
    gid = gen.id
    claim = Claim(jid, aid, gid, attempt, at)
    await persist_summary(db_session, claim, None, outcome='empty')
    await persist_extraction(db_session, claim, [])
    await prepare_fts(db_session, claim, 'alpha', 'raw_fallback')
    real_commit = db_session.commit
    async def uncertain_commit():
        await real_commit()
        raise RuntimeError('synthetic lost acknowledgement')
    with monkeypatch.context() as patch:
        patch.setattr(db_session, 'commit', uncertain_commit)
        with pytest.raises(RuntimeError):
            await publish_generation(db_session, claim)
    retired = (await db_session.get(ArchiveDerivationHead, aid)).legacy_retired_at
    assert await publish_generation(db_session, claim) is False
    assert (await db_session.get(ArchiveDerivationHead, aid)).legacy_retired_at == retired

@pytest.mark.asyncio
async def test_extraction_commit_ack_loss_fresh_session_reuses_exact_ids(db_session,test_engine,managed,monkeypatch):
    from app.domain.derivations import service as g
    from app.domain.derived_memory.models import Fact
    from app.domain.derivations.visibility import HISTORY
    from sqlalchemy import select
    source,job=await claimed(db_session)
    aid,jid,attempt,at=source.id,job.id,job.attempts,job.claimed_at
    gen=await admit_claimed(db_session,jid)
    claim=g.Claim(jid,aid,gen.id,attempt,at)
    real_commit=db_session.commit
    async def uncertain_commit():
        await real_commit()
        raise RuntimeError('synthetic extraction acknowledgement loss')
    with monkeypatch.context() as patch:
        patch.setattr(db_session,'commit',uncertain_commit)
        with pytest.raises(RuntimeError):
            await g.persist_extraction(db_session,claim,[{'fact_text':'winning batch','tags':['one']}])
    ids=(await db_session.scalars(select(Fact.id).execution_options(**HISTORY))).all()
    await db_session.rollback()
    factory=async_sessionmaker(test_engine,expire_on_commit=False)
    async with factory() as retry:
        reused=await g.persist_extraction(retry,claim,[{'fact_text':'discard recomputation'}])
        assert [f.id for f in reused]==ids
    assert await db_session.scalar(text('SELECT count(*) FROM facts'))==1
    assert await db_session.scalar(text('SELECT count(*) FROM fact_tags'))==1

def test_missing_cutover_setting_is_not_on(monkeypatch):
    from types import SimpleNamespace
    from app.domain.derivations import visibility
    monkeypatch.setattr(visibility,'get_settings',lambda:SimpleNamespace())
    assert visibility.enabled() is False and visibility.current_sql('f')==''
