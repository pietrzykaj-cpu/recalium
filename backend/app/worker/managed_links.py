"""Current-only links: snapshot, compute without locks, then fenced flush."""
from sqlalchemy import select, text
from app.domain.derived_memory.models import Fact, Embedding
from app.domain.derivations.visibility import current_sql
from app.domain.derivations import service as g

async def run_managed_links(session, claim, *, allow_external):
    from app.worker import dispatcher as legacy
    from app.domain.derived_memory.service import write_links
    async with g.transaction(session):
        await g.context(session, claim)
        facts = (await session.scalars(select(Fact).where(
            Fact.raw_archive_id == claim.archive_id, Fact.review_status == "active",
            Fact.source_status == "active").order_by(Fact.id))).all()
        source_texts = {f.id:f.fact_text for f in facts}
        source_ids = list(source_texts)
        embedding = await session.scalar(select(Embedding).where(
            Embedding.generation_id == claim.generation_id, Embedding.source_status == "active"))
        target_ids = []
        if embedding is not None:
            target_ids = list((await session.scalars(text(f"""
                SELECT f.id FROM facts f JOIN embeddings e ON e.raw_archive_id=f.raw_archive_id
                JOIN raw_archive ra ON ra.id=f.raw_archive_id
                WHERE f.source_status='active' AND f.review_status='active' AND e.source_status='active'
                AND e.embedding_model=:model AND ra.deleted_at IS NULL AND f.raw_archive_id<>:aid
                {current_sql('f')} {current_sql('e')}
                ORDER BY e.embedding <=> CAST(:vec AS vector), f.id LIMIT 5
            """), {"aid":claim.archive_id,"vec":str([float(v) for v in embedding.embedding]), "model":embedding.embedding_model})).all())
        target_facts = (await session.scalars(select(Fact).where(Fact.id.in_(target_ids),
            Fact.source_status=="active",Fact.review_status=="active"))).all()
        target_texts = {f.id:f.fact_text for f in target_facts}
        entity_pairs = (await session.execute(text(f"""
            SELECT DISTINCT sf.id AS sid, tf.id AS tid, t.name
            FROM facts sf JOIN fact_tags sft ON sft.fact_id=sf.id JOIN tags t ON t.id=sft.tag_id
            JOIN fact_tags tft ON tft.tag_id=t.id JOIN facts tf ON tf.id=tft.fact_id
            JOIN raw_archive tra ON tra.id=tf.raw_archive_id
            WHERE sf.raw_archive_id=:aid AND tf.raw_archive_id<>:aid
            AND t.name LIKE 'entity:%' AND sf.source_status='active' AND tf.source_status='active'
            AND sf.review_status='active' AND tf.review_status='active' AND tra.deleted_at IS NULL
            {current_sql('sf')} {current_sql('tf')} ORDER BY sid,tid,t.name LIMIT 200
        """), {"aid":claim.archive_id})).all()
        all_ids = set(source_ids) | set(target_ids) | {p.tid for p in entity_pairs}
        all_facts = (await session.scalars(select(Fact).where(Fact.id.in_(all_ids),
            Fact.source_status=="active",Fact.review_status=="active"))).all()
        expected = {f.id:(f.raw_archive_id,f.fact_text) for f in all_facts}
    data = [{"source_fact_id":s,"target_fact_id":t,"link_type":"related",
             "confidence":0.7,"created_by":"pipeline_semantic","entity_name":None}
            for s in source_ids for t in target_ids]
    errors = []
    if allow_external and target_ids and legacy._has_llm_provider():
        for sid in source_ids[:3]:
            for tid in target_ids[:5]:
                try:
                    kind = await legacy._classify_link_pair(source_texts[sid],target_texts[tid])
                except g.DerivationConflict:
                    raise
                except Exception as exc:
                    errors.append((sid, tid, legacy._bounded_exception_class(exc)))
                    continue
                if kind and kind != "unrelated":
                    data.append(dict(source_fact_id=sid,target_fact_id=tid,link_type=kind,
                        confidence=0.8,created_by="pipeline_llm",entity_name=None))
    data += [dict(source_fact_id=p.sid,target_fact_id=p.tid,link_type="entity",
                  confidence=1.0,created_by="pipeline_entity",entity_name=p.name[len("entity:"):])
             for p in entity_pairs]
    if not data:
        return
    async with g.transaction(session):
        sources,_ = await g.lock_archives_and_heads(session, {a for a,_ in expected.values()})
        await g.context(session, claim)
        if not all([await g.source_live(session,s) for s in sources.values()]):
            return
        rows = (await session.scalars(select(Fact).where(Fact.id.in_(all_ids),
            Fact.source_status=="active",Fact.review_status=="active"))).all()
        valid = {f.id for f in rows if expected.get(f.id)==(f.raw_archive_id,f.fact_text)}
        await write_links(session,[d for d in data if d["source_fact_id"] in valid and d["target_fact_id"] in valid], commit=False)
        from app.domain.audit.models import AuditEvent
        for sid, tid, exception_class in errors:
            if sid in valid and tid in valid:
                session.add(AuditEvent(event_type='link_detection_error', raw_archive_id=claim.archive_id,
                    actor='pipeline_worker', operation_metadata={'pass':'llm_typed',
                        'source_fact_id':str(sid), 'target_fact_id':str(tid),
                        'exception_class':exception_class}))
