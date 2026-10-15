"""R2 reader closure, E1 authorization, and coherent head snapshots."""
import uuid
import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker
from app.infrastructure.settings import get_settings
from app.domain.derivations import service as g
from app.domain.derivations.models import ArchiveDerivationHead as Head
from app.domain.derived_memory.models import Fact, Embedding, ConflictGroup
from tests.domain.test_derivation_generations import admit_claimed, claimed, managed
from tests.domain.test_derivation_boundaries import build

async def versions(session):
    from app.domain.jobs.models import Job
    from app.domain.jobs.service import reprocess_job, claim_next_job
    from app.domain.derived_memory.service import write_facts, write_tags, write_fts_entry
    from app.domain.canonical_memory.service import promote_fact_to_canonical
    source,job = await claimed(session)
    aid,jid = source.id,job.id
    legacy = await write_facts(session,aid,[{'fact_text':'alpha legacy'}])
    lid = legacy[0].id
    await write_tags(session,lid,['legacy-only'])
    await write_fts_entry(session,aid,'alpha old slot')
    first,oldid = await build(session, await session.get(Job,jid),'alpha previous')
    canonical = await promote_fact_to_canonical(session,oldid,'alpha curated','test',confirmed=True)
    cid = canonical.id
    # create_all's FetchedValue is a marker; the migration supplies the generated
    # canonical column. Seed its test-schema projection just like existing tests.
    await session.execute(text("UPDATE canonical_memory SET search_vector=to_tsvector('english',content) WHERE id=:id"),{'id':cid})
    for content,publish in [('alpha current',True),('alpha candidate',False)]:
        job=await session.get(Job,jid)
        job.status='retryable_failed'
        await session.commit()
        await reprocess_job(session,jid)
        nextjob=await claim_next_job(session)
        claim,fid = await build(session,nextjob,content,publish)
        if publish:
            active,currentid=claim,fid
        else:
            candidate,candidateid=claim,fid
    await write_tags(session,candidateid,['private-only'])
    return aid,lid,oldid,currentid,candidateid,cid,active,candidate

@pytest.mark.asyncio
async def test_api_mcp_tags_links_current_history_and_candidate(client,db_session,test_engine,managed,monkeypatch):
    from app.domain.derived_memory.service import write_links
    from app.mcp_server import server as mcp
    aid,lid,oldid,fid,candidateid,_,_,_ = await versions(db_session)
    source, _ = await claimed(db_session)
    target = Fact(raw_archive_id=source.id,fact_text='alpha other',source_span='',
        confidence_tier='low',derivation_method='test',derivation_model='test')
    db_session.add(target)
    await db_session.commit()
    tid = target.id
    await write_links(db_session,[dict(source_fact_id=fid,target_fact_id=t,link_type='related',created_by='test') for t in [oldid,candidateid,tid]])
    response = (await client.get('/api/facts')).json()
    assert {row['id'] for row in response['facts']} == {str(fid),str(tid)}
    assert (await client.get(f'/api/facts/{candidateid}')).status_code == 404
    history = (await client.get(f'/api/facts/{oldid}')).json()
    assert history['generation_visibility']=='history'
    current = (await client.get(f'/api/facts/{fid}')).json()
    assert current['generation_visibility']=='current'
    assert (await client.get(f'/api/facts/{oldid}/tags')).status_code == 404
    tags = (await client.get('/api/tags')).json()['tags']
    assert {t['name'] for t in tags} == {'visible','entity:anna'}
    tagid = next(t['id'] for t in tags if t['name']=='visible')
    tagged = (await client.get(f'/api/tags/{tagid}/facts')).json()
    assert [f['id'] for f in tagged['facts']] == [str(fid)]
    for direction in ['outgoing','both']:
        links = (await client.get(f'/api/facts/{fid}/links?direction={direction}')).json()
        assert [row['other_fact_id'] for row in links['links']]==[str(tid)]
    monkeypatch.setattr(mcp,'get_session_factory',lambda:async_sessionmaker(test_engine,expire_on_commit=False))
    assert {t['name'] for t in (await mcp.list_tags())['tags']} == {'visible','entity:anna'}
    assert (await mcp.get_fact_links(str(oldid)))['error']['code'] == 'not_found'
    assert [link['other_fact_id'] for link in (await mcp.get_fact_links(str(fid)))['links']] == [str(tid)]

@pytest.mark.asyncio
async def test_keyword_vector_health_canonical_and_review_closure(db_session,managed,monkeypatch):
    from app.domain.retrieval.service import retrieve, RetrievalRequest, _keyword_candidates
    from app.domain.derived_memory import service as writers
    from app.domain.review_queue.models import ReviewQueueItem
    from app.domain.review_queue.service import list_pending_review_items, resolve_review_item, ReviewItemStaleError
    from app.domain.derivations.visibility import HISTORY
    aid,lid,oldid,fid,candidateid,cid,active,candidate = await versions(db_session)
    vector=[1.0]+[0.0]*383
    db_session.add_all([Embedding(raw_archive_id=aid,generation_id=claim.generation_id,embedding=vector,
        embedding_model=writers.ACTIVE_EMBEDDING_MODEL) for claim in [active,candidate]])
    db_session.add(Embedding(raw_archive_id=aid,embedding=vector,embedding_model='legacy'))
    group=ConflictGroup(group_type='duplicate')
    db_session.add(group)
    await db_session.flush()
    for id in [oldid,fid]:
        fact=await db_session.scalar(select(Fact).where(Fact.id==id).execution_options(**HISTORY))
        fact.conflict_group_id=group.id
    item=ReviewQueueItem(conflict_group_id=group.id,item_type='duplicate')
    db_session.add(item)
    await db_session.commit()
    itemid=item.id
    assert not await list_pending_review_items(db_session)
    await db_session.rollback()
    with pytest.raises(ReviewItemStaleError):
        await resolve_review_item(db_session,itemid,'keep','test')
    await db_session.rollback()
    health=await writers.embedding_model_health(db_session)
    assert health['active_model_count']==1 and health['stale_count']==0
    await db_session.rollback()
    async def embed(query):
        assert not db_session.in_transaction()
        return vector
    monkeypatch.setattr(writers,'embed_text',embed)
    for mode in ['keyword','semantic','hybrid']:
        result=await retrieve(db_session,RetrievalRequest(query='alpha',mode=mode,budget=10000))
        contents=[i.content for i in result.items]
        assert all(not any(hidden in content for hidden in ['alpha legacy','alpha candidate','alpha previous','alpha old slot']) for content in contents)
        assert 'alpha current' in contents
        if mode!='semantic':
            assert 'alpha curated' in contents
        await db_session.rollback()
    head=await db_session.get(Head,aid)
    head.active_generation_id=None
    await db_session.commit()
    rows=await _keyword_candidates(db_session,'alpha')
    assert {r['type'] for r in rows}=={'canonical'}

@pytest.mark.asyncio
async def test_off_old_fts_on_new_baseline_then_atomic_switch(db_session,managed,monkeypatch):
    from app.domain.derived_memory.service import write_fts_entry
    from app.domain.retrieval.service import _keyword_candidates
    source,job=await claimed(db_session)
    aid,jid,at = source.id,job.id,job.claimed_at
    await write_fts_entry(db_session,aid,'oldsourceword')
    gen=await admit_claimed(db_session,jid)
    claim=g.Claim(jid,aid,gen.id,1,at)
    await g.raw_baseline(db_session,claim)
    assert not await _keyword_candidates(db_session,'oldsourceword')
    assert await _keyword_candidates(db_session,'alpha')
    with monkeypatch.context() as off:
        off.setattr(get_settings(),'generation_idempotency_enabled',False)
        assert await _keyword_candidates(db_session,'oldsourceword')
        assert not await _keyword_candidates(db_session,'alpha')
        with pytest.raises(g.DerivationConflict):
            await g.publish_generation(db_session,claim)
    await g.persist_summary(db_session,claim,'newsourceword')
    await g.persist_extraction(db_session,claim,[])
    await g.prepare_fts(db_session,claim,'newsourceword','summary')
    assert not await _keyword_candidates(db_session,'newsourceword')
    await g.publish_generation(db_session,claim)
    assert await _keyword_candidates(db_session,'newsourceword')
    assert not await _keyword_candidates(db_session,'alpha')

@pytest.mark.asyncio
async def test_repeatable_reader_snapshot_does_not_mix_heads(db_session,test_engine,managed):
    from app.domain.derivations.readers import begin_reader_snapshot
    source,job=await claimed(db_session)
    aid,jid=source.id,job.id
    claim,fid=await build(db_session,job,publish=False)
    factory=async_sessionmaker(test_engine,expire_on_commit=False)
    async with factory() as reader, factory() as writer:
        await begin_reader_snapshot(reader)
        assert not (await reader.scalars(select(Fact))).all()
        await g.publish_generation(writer,claim)
        assert not (await reader.scalars(select(Fact))).all()
        await reader.rollback()
        await begin_reader_snapshot(reader)
        assert [f.id for f in (await reader.scalars(select(Fact))).all()]==[fid]

@pytest.mark.asyncio
@pytest.mark.parametrize('revoke',['client','grant'])
async def test_bridge_reauthorizes_after_unlocked_query_preparation(db_session,test_engine,managed,monkeypatch,revoke):
    from app.domain.bridge import service as bridge
    from app.domain.bridge.models import BridgeClient,BridgeProject,BridgeGrant
    from app.domain.bridge.contracts import RetrieveInput
    from app.domain.derived_memory import service as writers
    credential='synthetic-'+'x'*40
    db_session.add(BridgeClient(id='reader',credential_digest=bridge.digest(credential)))
    db_session.add(BridgeProject(id='shared'))
    await db_session.flush()
    db_session.add(BridgeGrant(client_id='reader',project_id='shared',can_read=True))
    await db_session.commit()
    factory=async_sessionmaker(test_engine,expire_on_commit=False)
    async def prepare(query):
        assert not db_session.in_transaction()
        async with factory() as other:
            await other.execute(text("SET LOCAL lock_timeout='1s'"))
            sql=("UPDATE bridge_clients SET active=false WHERE id='reader'" if revoke=='client'
                 else "UPDATE bridge_grants SET can_read=false WHERE client_id='reader'")
            await other.execute(text(sql))
            await other.commit()
        return [1.0]+[0.0]*383
    monkeypatch.setattr(writers,'embed_text',prepare)
    with pytest.raises(bridge.BridgeError) as rejected:
        await bridge.execute(db_session,'Bearer '+credential,'retrieve_memory',RetrieveInput(project_id='shared',query='alpha',mode='hybrid'))
    assert rejected.value.status == (401 if revoke=='client' else 403)
    assert rejected.value.code in {'authentication_required','permission_denied'}

@pytest.mark.asyncio
@pytest.mark.parametrize('operation',['retrieve_memory','build_context_packet','build_continuity_handoff'])
async def test_bridge_current_handoff_and_curated_history(db_session,managed,monkeypatch,operation):
    from app.domain.bridge import service as bridge
    from app.domain.bridge.models import BridgeClient,BridgeProject,BridgeGrant,BridgeArchive
    from app.domain.bridge.contracts import RetrieveInput,ContextPacketInput,ContinuityHandoffInput
    from app.domain.agent_succession.contracts import CurrentAgent
    from app.domain.derived_memory import service as writers
    aid,*_ = await versions(db_session)
    credential='synthetic-'+'x'*40
    db_session.add(BridgeClient(id='reader',credential_digest=bridge.digest(credential)))
    db_session.add(BridgeProject(id='shared'))
    await db_session.flush()
    db_session.add(BridgeGrant(client_id='reader',project_id='shared',can_read=True))
    db_session.add(BridgeArchive(archive_id=aid,project_id='shared',client_id='reader'))
    await db_session.commit()
    async def embed(query):
        assert not db_session.in_transaction()
        raise RuntimeError('synthetic optional vector unavailable')
    monkeypatch.setattr(writers,'embed_text',embed)
    if operation=='retrieve_memory':
        request=RetrieveInput(project_id='shared',query='alpha',mode='hybrid')
    elif operation=='build_context_packet':
        request=ContextPacketInput(space_ids=['shared'],query='alpha',mode='hybrid')
    else:
        request=ContinuityHandoffInput(space_id='shared',workstream_id='test',authority_keys=['test'],
            query='alpha',mode='hybrid',current_agent=CurrentAgent(provider='test',model='test'))
    result=await bridge.execute(db_session,'Bearer '+credential,operation,request)
    if operation=='retrieve_memory':
        contents=[i['content'] for i in result['items']]
    else:
        packet=result if operation=='build_context_packet' else result['context_packet']
        contents=[i['content'] for i in packet['selected']]
    assert 'alpha current' in contents and 'alpha curated' in contents
    assert not any(hidden in contents for hidden in ['alpha previous','alpha candidate','alpha legacy'])
