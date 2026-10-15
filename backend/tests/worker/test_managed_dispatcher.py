"""Managed pipeline tests; every model function is an offline spy."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock
import uuid
import pytest
from sqlalchemy import text, select
from app.domain.jobs.models import Job
from app.domain.archive.models import RawArchiveItem
from app.infrastructure.settings import get_settings

@pytest.fixture
def spies(monkeypatch, db_session):
    from app.worker import dispatcher as d
    from app.domain.policy.gate import SensitivityDecision
    import app.domain.derived_memory.service as writers
    monkeypatch.setattr(get_settings(), "generation_idempotency_enabled", True)
    async def gate(raw):
        assert not db_session.in_transaction()
        return SensitivityDecision(category="general", confidence=0.9, blocked=False, method="test")
    async def summary(raw, **kwargs):
        assert not db_session.in_transaction()
        return "alpha summary"
    async def extract(raw, **kwargs):
        assert not db_session.in_transaction()
        return [{"fact_text":"alpha fact","source_span":"alpha","tags":["alpha"],"entities":["Anna"]}]
    async def no_embedding(raw):
        assert not db_session.in_transaction()
        raise RuntimeError("synthetic unavailable optional embedding")
    gate_spy, summary_spy, extract_spy = AsyncMock(side_effect=gate), AsyncMock(side_effect=summary), AsyncMock(side_effect=extract)
    monkeypatch.setattr(d._gate,"classify_async",gate_spy)
    monkeypatch.setattr(d,"_run_summarize_job",summary_spy)
    monkeypatch.setattr(d,"_run_extract_job",extract_spy)
    monkeypatch.setattr(d,"_has_llm_provider",lambda:True)
    monkeypatch.setattr(d,"_resolve_provider",lambda value:"openai")
    monkeypatch.setattr(d,"_provider_allowed",lambda provider,allow_external:allow_external)
    monkeypatch.setattr(d,"_provider_name",lambda:"synthetic")
    monkeypatch.setattr(d,"_resolve_model",lambda *args:"synthetic")
    monkeypatch.setattr(writers,"embed_text",no_embedding)
    return gate_spy,summary_spy,extract_spy

async def job_row(session, attempts=1):
    source=RawArchiveItem(source_type="test",raw_content="alpha source",content_hash=uuid.uuid4().hex)
    session.add(source)
    await session.flush()
    job=Job(raw_archive_id=source.id,job_type="extract",status="claimed",attempts=attempts,
            claimed_at=datetime.now(timezone.utc))
    session.add(job)
    await session.commit()
    return job

@pytest.mark.asyncio
async def test_managed_success_and_optional_failure(db_session,spies):
    from app.worker.dispatcher import dispatch_job
    job=await job_row(db_session)
    jid=job.id
    await dispatch_job(db_session,job)
    final=await db_session.get(Job,jid,populate_existing=True)
    assert final.status=="completed"
    assert final.generation_id is not None
    assert spies[1].await_count==spies[2].await_count==1
    assert (await db_session.execute(text("SELECT count(*) FROM facts"))).scalar_one()==1
    assert (await db_session.execute(text("SELECT count(*) FROM fact_tags"))).scalar_one()==2

@pytest.mark.asyncio
@pytest.mark.parametrize("crash_point",["after_extract","after_fts","after_publication"])
async def test_managed_resume_avoids_model_replay(db_session,spies,monkeypatch,crash_point):
    from app.worker.dispatcher import dispatch_job
    from app.domain.derivations import service as g
    from app.domain.jobs.service import recover_claimed_job_after_dispatch_failure,claim_next_job
    job=await job_row(db_session)
    jid=job.id
    name={"after_extract":"prepare_fts","after_fts":"publish_generation","after_publication":"finish_claim"}[crash_point]
    original=getattr(g,name)
    async def crash(*args,**kwargs):
        if crash_point=="after_publication":
            raise RuntimeError("synthetic after publication")
        raise RuntimeError("synthetic crash before next stage")
    with monkeypatch.context() as patch:
        patch.setattr(g,name,crash)
        with pytest.raises(RuntimeError):
            await dispatch_job(db_session,job)
    await db_session.rollback()
    current=await db_session.get(Job,jid,populate_existing=True)
    gid=current.generation_id
    assert await recover_claimed_job_after_dispatch_failure(db_session,jid,current.attempts,
        current.claimed_at,RuntimeError('synthetic crash'))=='recovered'
    next_job=await claim_next_job(db_session)
    assert next_job.id==jid and next_job.generation_id==gid
    await dispatch_job(db_session,next_job)
    assert (await db_session.get(Job,jid,populate_existing=True)).status=="completed"
    assert spies[1].await_count==spies[2].await_count==1
    assert (await db_session.execute(text("SELECT count(*) FROM facts"))).scalar_one()==1
    assert (await db_session.execute(text("SELECT count(*) FROM derivation_generations"))).scalar_one()==1
    assert (await db_session.execute(text("SELECT count(*) FROM derivation_fts_projections WHERE generation_id IS NOT NULL"))).scalar_one()==1

@pytest.mark.asyncio
async def test_final_summary_partial_preserves_bounded_marker(db_session,spies):
    from app.worker import dispatcher as d
    job=await job_row(db_session,3)
    jid=job.id
    exc=RuntimeError("raw provider detail MUST NOT be persisted")
    d._mark_summary_output_failure(exc,"provider_response")
    spies[1].side_effect=exc
    await d.dispatch_job(db_session,job)
    final=await db_session.get(Job,jid,populate_existing=True)
    assert final.status=="completed"
    assert final.error_message=="partial: summary_failed(category=provider_response, exception_class=RuntimeError)"
    assert "raw provider" not in final.error_message

@pytest.mark.asyncio
async def test_missing_provider_then_same_generation_reactivation(db_session,spies,monkeypatch):
    from app.worker import dispatcher as d
    from app.domain.jobs.service import reactivate_pending_provider_jobs,claim_next_job
    job=await job_row(db_session)
    jid=job.id
    with monkeypatch.context() as patch:
        patch.setattr(d,"_has_llm_provider",lambda:False)
        await d.dispatch_job(db_session,job)
    current=await db_session.get(Job,jid,populate_existing=True)
    gid=current.generation_id
    assert current.status=="pending_provider"
    assert spies[1].await_count==spies[2].await_count==0
    await reactivate_pending_provider_jobs(db_session)
    next_job=await claim_next_job(db_session)
    assert next_job.generation_id==gid
    await d.dispatch_job(db_session,next_job)
    assert (await db_session.get(Job,jid,populate_existing=True)).status=="completed"
    assert (await db_session.execute(text("SELECT count(*) FROM derivation_generations"))).scalar_one()==1

@pytest.mark.asyncio
@pytest.mark.parametrize('failure',['before_persistence','during_persistence'])
async def test_extraction_failure_retries_same_generation_atomic_batch(db_session,spies,monkeypatch,failure):
    from app.worker import dispatcher as d
    from app.domain.jobs.service import claim_next_job
    import app.domain.derived_memory.service as writers
    job=await job_row(db_session)
    jid=job.id
    with monkeypatch.context() as patch:
        if failure=='before_persistence':
            original=spies[2].side_effect
            spies[2].side_effect=RuntimeError('synthetic before persistence')
        else:
            async def fail(*args,**kwargs):
                raise RuntimeError('synthetic persistence rollback')
            patch.setattr(writers,'write_tags',fail)
        await d.dispatch_job(db_session,job)
    if failure=='before_persistence':
        spies[2].side_effect=original
    current=await db_session.get(Job,jid,populate_existing=True)
    gid=current.generation_id
    assert current.status=='retryable_failed'
    assert await db_session.scalar(text('SELECT count(*) FROM facts'))==0
    assert await db_session.scalar(text("SELECT count(*) FROM derivation_stage_completions WHERE stage_name='extraction'"))==0
    await db_session.rollback()
    nextjob=await claim_next_job(db_session)
    await d.dispatch_job(db_session,nextjob)
    assert (await db_session.get(Job,jid,populate_existing=True)).generation_id==gid
    assert spies[1].await_count==1 and spies[2].await_count==2
    assert await db_session.scalar(text('SELECT count(*) FROM facts'))==1
    assert await db_session.scalar(text('SELECT count(*) FROM derivation_generations'))==1

@pytest.mark.asyncio
@pytest.mark.parametrize('attempt,origin',[(1,True),(3,False)])
async def test_managed_summary_fails_closed_before_partial_eligibility(db_session,spies,attempt,origin):
    from app.worker import dispatcher as d
    job=await job_row(db_session,attempt)
    jid=job.id
    exc=RuntimeError('synthetic summary failure')
    if origin:
        d._mark_summary_output_failure(exc,'provider_response')
    spies[1].side_effect=exc
    await d.dispatch_job(db_session,job)
    assert (await db_session.get(Job,jid,populate_existing=True)).status==('retryable_failed' if attempt==1 else 'failed')
    assert spies[2].await_count==0
    assert await db_session.scalar(text("SELECT count(*) FROM derivation_stage_completions"))==0

@pytest.mark.asyncio
@pytest.mark.parametrize('link_failure',[False,True])
async def test_optional_enrichment_model_calls_outside_transactions(db_session,spies,monkeypatch,link_failure):
    from app.worker import dispatcher as d
    from app.domain.derived_memory import service as writers
    from app.domain.derived_memory.models import Embedding
    vector=[1.0]+[0.0]*383
    source=RawArchiveItem(source_type='test',raw_content='alpha target',content_hash=uuid.uuid4().hex)
    db_session.add(source)
    await db_session.commit()
    aid=source.id
    facts=await writers.write_facts(db_session,aid,[{'fact_text':'alpha target','source_span':'alpha'}])
    await writers.write_tags(db_session,facts[0].id,['entity:anna'])
    db_session.add(Embedding(raw_archive_id=aid,embedding=vector,embedding_model=writers.ACTIVE_EMBEDDING_MODEL))
    await db_session.commit()
    async def embed(raw):
        assert not db_session.in_transaction()
        return vector
    async def classify(left,right):
        assert not db_session.in_transaction()
        if link_failure:
            raise RuntimeError('raw classification detail MUST NOT persist')
        return 'supports'
    classification=AsyncMock(side_effect=classify)
    monkeypatch.setattr(writers,'embed_text',embed)
    monkeypatch.setattr(d,'_classify_link_pair',classification)
    job=await job_row(db_session)
    jid=job.id
    await d.dispatch_job(db_session,job)
    assert (await db_session.get(Job,jid,populate_existing=True)).status=='completed'
    assert classification.await_count==1
    kinds=set((await db_session.scalars(text('SELECT link_type FROM memory_links'))).all())
    assert {'related','entity'} <= kinds
    assert ('supports' in kinds) is not link_failure
    errors=(await db_session.scalars(text("SELECT operation_metadata FROM audit_events WHERE event_type='link_detection_error'"))).all()
    if link_failure:
        assert len(errors)==1 and errors[0]['exception_class']=='RuntimeError'
        assert 'raw classification' not in str(errors)
    else:
        assert not errors
