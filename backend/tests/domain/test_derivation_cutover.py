"""Synthetic PostgreSQL only: OFF baseline preparation and fail-closed coverage."""
import hashlib
import json
import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.domain.archive.models import RawArchiveItem, Tombstone
from app.domain.derivations.models import ArchiveDerivationHead as Head, DerivationFtsProjection as Projection
from app.domain.jobs.models import Job
from app.infrastructure.settings import get_settings

pytestmark = pytest.mark.asyncio


@pytest.fixture
def ready_local_assets(monkeypatch):
    # These publication/privacy tests do not load models. Locality readiness
    # itself is validated separately using real static file checks.
    from app.domain.derivations import service as g
    async def ready():
        return None
    monkeypatch.setattr(g, 'require_local_model_ready_async', ready)


@pytest_asyncio.fixture
async def schema(db_session, monkeypatch):
    monkeypatch.setattr(get_settings(), 'generation_idempotency_enabled', False)
    await db_session.execute(text('CREATE TABLE IF NOT EXISTS alembic_version (version_num varchar(32) NOT NULL PRIMARY KEY)'))
    await db_session.execute(text('DELETE FROM alembic_version'))
    await db_session.execute(text("INSERT INTO alembic_version VALUES ('0013')"))
    await db_session.commit()


async def archive(session, content='alpha baseline evidence', status='completed'):
    row = RawArchiveItem(source_type='test', raw_content=content,
                         content_hash=hashlib.sha256(content.encode()).hexdigest())
    session.add(row)
    await session.flush()
    session.add(Job(raw_archive_id=row.id, job_type='extract', status=status))
    await session.commit()
    return row.id


async def report(session, kind='plan', **kwargs):
    from app.domain.derivations.cutover import inspect_population
    await session.rollback()
    return await inspect_population(session, kind=kind, **kwargs)


async def test_off_prepare_is_baseline_only_and_replay_stable(db_session, schema):
    from app.domain.derivations.cutover import prepare
    aid = await archive(db_session, 'alpha ' * 2000)
    plan = await report(db_session)
    assert plan['ok']
    first = await prepare(db_session, plan)
    second = await prepare(db_session, plan)
    assert first['created'] == 1 and second['reused'] == 1
    rows = (await db_session.scalars(select(Projection))).all()
    assert len(rows) == 1 and rows[0].generation_id is None
    assert rows[0].raw_archive_id == aid and rows[0].projection_kind == 'raw_fallback'
    assert rows[0].text_content == ('alpha ' * 2000)[:10000]
    assert rows[0].search_vector is not None
    for table in ('derivation_generations', 'archive_derivation_heads', 'derivation_stage_completions', 'fts_entries'):
        assert await db_session.scalar(text(f'SELECT count(*) FROM {table}')) == 0
    assert (await report(db_session, 'check'))['ok']
    assert 'alpha' not in json.dumps(plan)


async def test_prepare_refuses_on(db_session, schema, monkeypatch):
    from app.domain.derivations.cutover import prepare, CutoverError
    await archive(db_session)
    plan = await report(db_session)
    monkeypatch.setattr(get_settings(), 'generation_idempotency_enabled', True)
    with pytest.raises(CutoverError, match='r2_must_be_off'):
        await prepare(db_session, plan)


@pytest.mark.parametrize('status', ['claimed', 'pending', 'retryable_failed', 'pending_provider', 'failed', 'unsupported'])
async def test_unsafe_job_states_block(db_session, schema, status):
    await archive(db_session, status=status)
    result = await report(db_session, 'check')
    assert not result['ok'] and 'job_not_completed' in result['archives'][0]['reasons']


async def test_hash_only_tombstone_and_deleted_no_resurrection(db_session, schema):
    from app.domain.derivations.cutover import prepare, CutoverError
    aid = await archive(db_session)
    plan = await report(db_session)
    source = await db_session.get(RawArchiveItem, aid)
    db_session.add(Tombstone(source_id=uuid.uuid4(), content_hash=source.content_hash))
    await db_session.commit()
    with pytest.raises(CutoverError):
        await prepare(db_session, plan)
    result = await report(db_session, 'check')
    assert result['counts']['D'] == 1 and not result['ok']
    assert 'tombstone_reapply_required' in result['archives'][0]['reasons']
    assert await db_session.scalar(text('SELECT count(*) FROM derivation_fts_projections')) == 0


@pytest.mark.parametrize('change', ['raw', 'hash', 'metadata', 'head', 'job'])
async def test_plan_drift_requires_replan(db_session, schema, change):
    from app.domain.derivations.cutover import prepare, CutoverError
    aid = await archive(db_session)
    plan = await report(db_session)
    source = await db_session.get(RawArchiveItem, aid)
    if change == 'raw':
        source.raw_content += ' changed'
    elif change == 'hash':
        source.content_hash = 'f' * 64
    elif change == 'metadata':
        source.metadata_json = {'sensitivity_hint': 'private'}
    elif change == 'head':
        db_session.add(Head(raw_archive_id=aid))
    else:
        job = await db_session.scalar(select(Job))
        job.attempts += 1
    await db_session.commit()
    with pytest.raises(CutoverError, match='replan_required'):
        await prepare(db_session, plan)


@pytest.mark.parametrize('condition,reason', [
    ('retired', 'retired_headless'), ('suppressed', 'baseline_suppressed'),
    ('stale', 'baseline_stale'), ('missing', 'baseline_missing'),
])
async def test_coverage_anomalies(db_session, schema, condition, reason):
    aid = await archive(db_session)
    if condition == 'retired':
        db_session.add(Head(raw_archive_id=aid, legacy_retired_at=datetime.now(timezone.utc)))
    elif condition != 'missing':
        db_session.add(Projection(raw_archive_id=aid, projection_kind='raw_fallback',
            text_content='stale' if condition == 'stale' else 'alpha baseline evidence',
            source_status='source_removed' if condition == 'suppressed' else 'active'))
    await db_session.commit()
    result = await report(db_session, 'check')
    assert not result['ok'] and reason in result['archives'][0]['reasons']


async def test_partial_batch_resume_and_lost_ack(db_session, test_engine, schema, monkeypatch):
    from app.domain.derivations import cutover as tool
    for content in ('one evidence', 'two evidence', 'three evidence'):
        await archive(db_session, content)
    plan = await report(db_session)
    original = tool.load_state
    committed = []
    async def fail_second(session, aid, **kwargs):
        if kwargs.get('locked'):
            committed.append(aid)
            if len(committed) == 2:
                raise tool.CutoverError('synthetic_interruption')
        return await original(session, aid, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(tool, 'load_state', fail_second)
        with pytest.raises(tool.CutoverError):
            await tool.prepare(db_session, plan)
    first_id = await db_session.scalar(select(Projection.id))
    assert first_id is not None
    await db_session.rollback()
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as retry:
        result = await tool.prepare(retry, plan)
        assert result['reused'] == 1 and result['created'] == 2
        # Simulate lost batch acknowledgement: discard result, rerun in new session.
    async with factory() as retry:
        assert (await tool.prepare(retry, plan))['reused'] == 3
        assert first_id in (await retry.scalars(select(Projection.id))).all()


async def test_open_candidate_and_duplicate_job_block(db_session, schema):
    from app.domain.derivations.models import DerivationGeneration as Generation
    aid = await archive(db_session)
    job = await db_session.scalar(select(Job))
    source = await db_session.get(RawArchiveItem, aid)
    generation = Generation(raw_archive_id=aid, original_job_id=job.id, sequence=1,
        creation_reason='automatic_processing', source_content_hash=source.content_hash)
    db_session.add(generation)
    db_session.add(Job(raw_archive_id=aid, job_type='extract', status='completed'))
    await db_session.commit()
    result = await report(db_session, 'check')
    assert {'open_candidate', 'job_cardinality'} <= set(result['archives'][0]['reasons'])


async def test_orphan_job_and_partition_proof_fail_closed(db_session, schema, monkeypatch):
    from app.domain.derivations import cutover as tool
    aid = await archive(db_session)
    async with db_session.begin():
        # Synthetic corruption only: real FK normally prevents orphan creation.
        await db_session.execute(text('SET LOCAL session_replication_role=replica'))
        await db_session.execute(text('UPDATE jobs SET raw_archive_id=:id'), {'id': uuid.uuid4()})
    result = await report(db_session, 'check')
    assert not result['ok'] and 'orphan_jobs' in result['blockers']
    assert result['orphan_jobs'] == 1
    monkeypatch.setattr(tool, 'partition_valid', lambda *a: False)
    result = await report(db_session, 'check')
    assert 'partition_mismatch' in result['blockers']


async def test_active_class_calls_real_shared_publication_validator(db_session, schema, monkeypatch, ready_local_assets):
    from app.domain.derivations import service as g
    from tests.domain.test_derivation_generations import claimed
    from tests.domain.test_derivation_boundaries import build
    monkeypatch.setattr(get_settings(), 'generation_idempotency_enabled', True)
    source, job = await claimed(db_session)
    claim, _ = await build(db_session, job)
    job = await db_session.get(Job, claim.job_id)
    job.status = 'completed'
    await db_session.commit()
    original = g.validate_publication
    calls = []
    async def tracked(session, head, generation):
        calls.append(generation.id)
        assert await session.scalar(text('SHOW transaction_read_only')) == 'on'
        return await original(session, head, generation)
    monkeypatch.setattr(g, 'validate_publication', tracked)
    result = await report(db_session, 'check')
    assert result['ok'] and result['counts']['B'] == 1 and calls == [claim.generation_id]
    await db_session.execute(text("DELETE FROM derivation_stage_completions WHERE stage_name='extraction'"))
    await db_session.commit()
    result = await report(db_session, 'check')
    assert not result['ok'] and 'publication_invalid' in result['archives'][0]['reasons']


async def test_full_population_snapshot_read_only_and_quiescence(db_session, test_engine, schema):
    from app.domain.derivations.cutover import prepare
    await archive(db_session)
    plan = await report(db_session)
    await prepare(db_session, plan)
    # A second live PG client, even idle in a transaction, blocks cutover.
    async with async_sessionmaker(test_engine)() as other:
        await other.execute(text('SELECT 1'))
        result = await report(db_session, 'check', cutover=True)
        assert not result['ok'] and 'other_database_clients' in result['blockers']
        assert result['snapshot'] == {'isolation': 'repeatable read', 'read_only': 'on'}
    await archive(db_session, 'new unplanned evidence')
    result = await report(db_session, 'check')
    assert result['total_archives'] == 2 and not result['ok']
    assert sum(result['counts'].values()) == 2
    assert len({r['archive_id'] for r in result['archives']}) == 2


@pytest.mark.parametrize('first', ['prepare', 'delete'])
@pytest.mark.parametrize('existing', [False, True])
async def test_deletion_prepare_race_both_orders(db_session, test_engine, schema, monkeypatch, first, existing):
    import asyncio
    from app.domain.derivations import cutover as tool
    from app.domain.archive import service as deletion
    from sqlalchemy.ext.asyncio import create_async_engine
    from app.domain.archive.tombstones import REDACTION_MARKER
    aid = await archive(db_session)
    plan = await report(db_session)
    if existing:
        await tool.prepare(db_session, plan)
    monkeypatch.setattr(deletion, 'append_tombstone_ledger', lambda *a: None)
    locked, release = asyncio.Event(), asyncio.Event()
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as writer, factory() as remover:
        if first == 'prepare':
            original = tool.load_state
            async def paused(session, target, **kw):
                state = await original(session, target, **kw)
                if kw.get('locked'):
                    locked.set()
                    await release.wait()
                return state
            monkeypatch.setattr(tool, 'load_state', paused)
            leading = asyncio.create_task(tool.prepare(writer, plan))
            await asyncio.wait_for(locked.wait(), 5)
            following = asyncio.create_task(deletion.cascade_delete_archive_item(remover, aid))
        else:
            original = deletion._suppress_derived
            async def paused(session, target):
                locked.set()
                await release.wait()
                await original(session, target)
            monkeypatch.setattr(deletion, '_suppress_derived', paused)
            leading = asyncio.create_task(deletion.cascade_delete_archive_item(remover, aid))
            await asyncio.wait_for(locked.wait(), 5)
            following = asyncio.create_task(tool.prepare(writer, plan))
        # Observe a real PostgreSQL lock wait, not elapsed-time scheduling luck.
        observer = create_async_engine(get_settings().database_url)
        try:
            async with observer.connect() as connection:
                async with asyncio.timeout(5):
                    while not await connection.scalar(text('''SELECT EXISTS (
                        SELECT 1 FROM pg_stat_activity WHERE datname=current_database()
                        AND cardinality(pg_blocking_pids(pid)) > 0)''')):
                        await asyncio.sleep(0.01)
                assert not following.done()
        finally:
            release.set()
            results = await asyncio.gather(leading, following, return_exceptions=True)
            await observer.dispose()
        assert not isinstance(results[0], BaseException)
        if first == 'delete':
            assert isinstance(results[1], tool.CutoverError)
        else:
            assert results[1] is None
    result = await report(db_session, 'check')
    assert result['counts']['D'] == 1 and result['ok']
    rows = (await db_session.scalars(select(Projection))).all()
    assert all(r.text_content == REDACTION_MARKER and r.source_status == 'source_removed'
               and r.search_vector is None for r in rows)
    await db_session.rollback()
    # Fresh deleted-source plan is a skip, never a reactivation.
    fresh = await report(db_session)
    assert (await tool.prepare(db_session, fresh))['skipped'] == 1


async def test_artifact_bounds_tampering_and_readonly_enforcement(db_session, schema, monkeypatch):
    from app.domain.derivations import cutover as tool
    await archive(db_session, content='PRIVATE-PAYLOAD-DO-NOT-EXPORT')
    plan = await report(db_session)
    assert 'PRIVATE-PAYLOAD' not in tool.bounded_json(plan) + tool.human_report(plan)
    altered = json.loads(json.dumps(plan))
    altered['archives'].clear()
    with pytest.raises(tool.CutoverError, match='invalid_plan'):
        await tool.prepare(db_session, altered)
    async with tool.readonly_snapshot(db_session):
        with pytest.raises(Exception, match='read.only'):
            await db_session.execute(text("UPDATE jobs SET status='pending'"))
    monkeypatch.setattr(tool, 'MAX_ARTIFACT_BYTES', 1)
    with pytest.raises(tool.CutoverError, match='artifact_limit_exceeded'):
        await report(db_session)


async def test_real_commit_ack_loss_fresh_session_reuses_identity(db_session, test_engine, schema):
    from sqlalchemy import event
    from app.domain.derivations.cutover import prepare
    await archive(db_session)
    plan = await report(db_session)
    commits = []
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as writer:
        def lose_ack(session):
            commits.append(1)
            # First commit closes read-only preflight; second commits the baseline.
            if len(commits) == 2:
                raise RuntimeError('synthetic lost commit acknowledgement')
        event.listen(writer.sync_session, 'after_commit', lose_ack)
        try:
            with pytest.raises(RuntimeError, match='lost commit acknowledgement'):
                await prepare(writer, plan)
        finally:
            event.remove(writer.sync_session, 'after_commit', lose_ack)
    identity = await db_session.scalar(select(Projection.id))
    assert identity is not None
    await db_session.rollback()
    async with factory() as retry:
        assert (await prepare(retry, plan))['reused'] == 1
        assert await retry.scalar(select(Projection.id)) == identity


async def test_certificate_is_one_coherent_population_snapshot(db_session, test_engine, schema, monkeypatch):
    from app.domain.derivations import cutover as tool
    aid = await archive(db_session)
    plan = await report(db_session)
    await tool.prepare(db_session, plan)
    original = tool.classify
    changed = []
    async def concurrent_change(session, state, **kw):
        if not changed:
            changed.append(True)
            async with async_sessionmaker(test_engine)() as other:
                await other.execute(text("UPDATE raw_archive SET raw_content='changed evidence' WHERE id=:id"), {'id': aid})
                await other.commit()
        return await original(session, state, **kw)
    with monkeypatch.context() as patch:
        patch.setattr(tool, 'classify', concurrent_change)
        assert (await report(db_session, 'check'))['ok']
    fresh = await report(db_session, 'check')
    assert not fresh['ok'] and 'baseline_stale' in fresh['archives'][0]['reasons']


async def test_cross_archive_derived_binding_corruption_blocks(db_session, schema):
    from app.domain.derivations.models import DerivationGeneration as Generation
    first = await archive(db_session, 'first source')
    second = await archive(db_session, 'second source')
    job = await db_session.scalar(select(Job).where(Job.raw_archive_id == first))
    generation = Generation(raw_archive_id=first, original_job_id=job.id, sequence=1,
        creation_reason='automatic_processing', source_content_hash='synthetic', status='failed')
    db_session.add(generation)
    await db_session.commit()
    gid = generation.id
    async with db_session.begin():
        await db_session.execute(text('SET LOCAL session_replication_role=replica'))
        await db_session.execute(text('''INSERT INTO facts
            (id,raw_archive_id,generation_id,fact_text,source_span,confidence_tier,
             derivation_method,derivation_model,source_status,review_status,created_at)
            VALUES (:id,:aid,:gid,'synthetic','synthetic','low','test','test','active','active',now())'''),
            {'id': uuid.uuid4(), 'aid': second, 'gid': gid})
    result = await report(db_session, 'check')
    assert not result['ok'] and 'derived_generation_binding' in result['blockers']


async def test_real_partition_duplicate_and_population_cap(db_session, schema, monkeypatch):
    from app.domain.derivations import cutover as tool
    await archive(db_session, 'first source')
    await archive(db_session, 'second source')
    original = tool.classify
    seen = []
    async def duplicate(session, state, **kw):
        row = await original(session, state, **kw)
        seen.append(row['archive_id'])
        row['archive_id'] = seen[0]
        return row
    with monkeypatch.context() as patch:
        patch.setattr(tool, 'classify', duplicate)
        result = await report(db_session)
        assert not result['ok'] and 'partition_mismatch' in result['blockers']
    monkeypatch.setattr(tool, 'MAX_ARCHIVES', 1)
    with pytest.raises(tool.CutoverError, match='population_limit_exceeded'):
        await report(db_session)


async def test_cli_plan_prepare_resume_check_roundtrip(db_session, schema, tmp_path, capsys):
    from types import SimpleNamespace
    from app.domain.derivations.cutover_cli import run
    await archive(db_session, 'private synthetic CLI source')
    assert await run(SimpleNamespace(action='plan', cutover=False)) == 0
    captured = capsys.readouterr()
    plan = json.loads(captured.out)
    assert 'private synthetic CLI source' not in captured.out + captured.err
    location = tmp_path / 'plan.json'
    location.write_text(json.dumps(plan))
    assert await run(SimpleNamespace(action='prepare', plan=str(location))) == 0
    assert json.loads(capsys.readouterr().out)['created'] == 1
    assert await run(SimpleNamespace(action='prepare', plan=str(location))) == 0
    assert json.loads(capsys.readouterr().out)['reused'] == 1
    assert await run(SimpleNamespace(action='check', cutover=False)) == 0
    certificate = json.loads(capsys.readouterr().out)
    assert certificate['ok'] and certificate['counts']['A'] == 1


async def test_schema_drift_fails_before_write(db_session, schema):
    from app.domain.derivations.cutover import CutoverError, prepare
    await archive(db_session)
    plan = await report(db_session)
    await db_session.execute(text("UPDATE alembic_version SET version_num='0012'"))
    await db_session.commit()
    with pytest.raises(CutoverError, match='schema_mismatch'):
        await report(db_session, 'check')
    with pytest.raises(CutoverError, match='replan_required'):
        await prepare(db_session, plan)
    assert await db_session.scalar(text('SELECT count(*) FROM derivation_fts_projections')) == 0


async def test_baseline_only_readers_and_window_a_rollback(db_session, client, schema, monkeypatch):
    from app.domain.derivations.cutover import prepare
    from app.domain.derived_memory import service as writers
    from app.domain.derived_memory.models import Embedding, ConflictGroup
    from app.domain.review_queue.models import ReviewQueueItem
    from app.domain.review_queue.service import list_pending_review_items
    from app.domain.retrieval.service import _keyword_candidates, _semantic_candidates
    from app.domain.derivations.readers import prepared_query
    aid = await archive(db_session, 'baselineword raw evidence')
    facts = await writers.write_facts(db_session, aid, [
        {'fact_text': 'legacyfactword first'}, {'fact_text': 'legacyfactword second'}])
    fids = [f.id for f in facts]
    await writers.write_tags(db_session, fids[0], ['legacy-visible'])
    await writers.write_links(db_session, [dict(source_fact_id=fids[0], target_fact_id=fids[1],
        link_type='related', created_by='test')])
    await writers.write_fts_entry(db_session, aid, 'oldslotword legacy projection')
    vector = [1.0] + [0.0] * 383
    db_session.add(Embedding(raw_archive_id=aid, embedding=vector, embedding_model=writers.ACTIVE_EMBEDDING_MODEL))
    group = ConflictGroup(group_type='duplicate')
    db_session.add(group)
    await db_session.flush()
    for fact in facts:
        fact.conflict_group_id = group.id
    review = ReviewQueueItem(conflict_group_id=group.id, item_type='duplicate')
    db_session.add(review)
    await db_session.commit()
    rid = review.id

    async def visible():
        await db_session.rollback()
        fact_ids = sorted(f['id'] for f in (await client.get('/api/facts')).json()['facts'])
        tags = (await client.get('/api/tags')).json()['tags']
        links = (await client.get(f'/api/facts/{fids[0]}/links')).json()['links']
        item_ids = [i.id for i in await list_pending_review_items(db_session)]
        await db_session.rollback()
        return fact_ids, [(t['name'], t['fact_count']) for t in tags], [l['other_fact_id'] for l in links], item_ids

    before = await visible()
    assert before == (sorted(map(str, fids)), [('legacy-visible', 1)], [str(fids[1])], [rid])
    old = await _keyword_candidates(db_session, 'oldslotword')
    old_ids = [r['id'] for r in old]
    assert old_ids and not await _keyword_candidates(db_session, 'baselineword')
    plan = await report(db_session)
    await prepare(db_session, plan)
    monkeypatch.setattr(get_settings(), 'generation_idempotency_enabled', True)
    assert await visible() == before
    assert await _keyword_candidates(db_session, 'baselineword')
    assert not await _keyword_candidates(db_session, 'oldslotword')
    token = prepared_query.set(('synthetic query', vector))
    try:
        candidates, degraded = await _semantic_candidates(db_session, 'synthetic query')
    finally:
        prepared_query.reset(token)
    assert candidates and not degraded
    assert {r['source_id'] for r in candidates} == {str(aid)}
    for query in ('SELECT count(*) FROM derivation_generations',
                  'SELECT count(*) FROM archive_derivation_heads WHERE legacy_retired_at IS NOT NULL',
                  'SELECT count(*) FROM facts WHERE generation_id IS NOT NULL',
                  'SELECT count(*) FROM summaries WHERE generation_id IS NOT NULL',
                  'SELECT count(*) FROM embeddings WHERE generation_id IS NOT NULL',
                  "SELECT count(*) FROM jobs WHERE status='claimed'"):
        assert await db_session.scalar(text(query)) == 0
    monkeypatch.setattr(get_settings(), 'generation_idempotency_enabled', False)
    assert await visible() == before
    assert [r['id'] for r in await _keyword_candidates(db_session, 'oldslotword')] == old_ids
    assert not await _keyword_candidates(db_session, 'baselineword')


@pytest.mark.parametrize('managed_generation', [False, True], ids=['legacy-off', 'published-managed'])
async def test_class_d_real_derived_privacy_and_plaintext_rejection(
    db_session, schema, monkeypatch, tmp_path, managed_generation, ready_local_assets,
):
    """N2: real nonempty derived rows, real deletion, then deliberate corruption.

    Every row/column assertion is non-vacuous. In the managed case, the corrupted
    legacy fact is hidden by normal current-reader rules; certification must still
    inspect all generations, rather than mistaking invisibility for erasure.
    """
    from app.domain.archive import service as deletion
    from app.domain.archive.tombstones import REDACTION_MARKER
    from app.domain.canonical_memory.service import promote_fact_to_canonical
    from app.domain.derivations import service as generations
    from app.domain.derivations.cutover import prepare
    from app.domain.derived_memory import service as writers
    from app.domain.derived_memory.models import Embedding, ConflictGroup
    from app.domain.review_queue.models import ReviewQueueItem
    from tests.domain.test_derivation_boundaries import build

    aid = await archive(db_session, 'alpha source span confidential evidence')
    facts = await writers.write_facts(db_session, aid, [{
        'fact_text': 'ordinary private fact', 'source_span': 'alpha source span',
        'confidence_tier': 'high', 'derivation_method': 'test', 'derivation_model': 'synthetic',
    }])
    fid = facts[0].id
    await writers.write_summary(db_session, aid, 'private summary evidence', 'synthetic', 'test')
    await writers.write_fts_entry(db_session, aid, 'private legacy lexical evidence')
    db_session.add(Embedding(raw_archive_id=aid, embedding=[1.0] + [0.0] * 383,
        embedding_model=writers.ACTIVE_EMBEDDING_MODEL))
    group = ConflictGroup(group_type='duplicate')
    db_session.add(group)
    await db_session.flush()
    facts[0].conflict_group_id = group.id
    review = ReviewQueueItem(conflict_group_id=group.id, item_type='duplicate')
    db_session.add(review)
    canonical = await promote_fact_to_canonical(
        db_session, fid, 'retained curated evidence', 'test', confirmed=True)
    cid, group_id = canonical.id, group.id
    await db_session.commit()

    # Exercise the real OFF preparation path, not just an empty derived archive.
    plan = await report(db_session)
    assert plan['ok']
    assert (await prepare(db_session, plan))['created'] == 1
    if managed_generation:
        # Synthetic claim fixture only; no route/reprocess guard is bypassed in code.
        monkeypatch.setattr(get_settings(), 'generation_idempotency_enabled', True)
        job = await db_session.scalar(select(Job).where(Job.raw_archive_id == aid))
        job.status, job.attempts = 'claimed', 1
        job.claimed_at = datetime.now(timezone.utc)
        await db_session.commit()
        claim, _ = await build(db_session, job, 'private published evidence')
        await generations.persist_embedding(db_session, claim, [1.0] + [0.0] * 383)
        await generations.finish_claim(db_session, claim)

    async def rows(table, columns):
        return (await db_session.execute(text(
            f'SELECT {columns} FROM {table} WHERE raw_archive_id=:aid'), {'aid': aid})).all()

    # Pin every precondition so absence of seeded data cannot make privacy pass.
    assert await db_session.scalar(select(RawArchiveItem.raw_content).where(RawArchiveItem.id == aid)) == 'alpha source span confidential evidence'
    original_fact = (await db_session.execute(text(
        'SELECT fact_text,source_span,source_status FROM facts WHERE id=:fid'), {'fid': fid})).one()
    assert tuple(original_fact) == ('ordinary private fact', 'alpha source span', 'active')
    for table, column in (('summaries', 'summary_text'), ('fts_entries', 'text_content'),
                          ('derivation_fts_projections', 'text_content')):
        before = await rows(table, column + ',source_status')
        assert before and all(value != REDACTION_MARKER and status == 'active' for value, status in before)
    assert all(status == 'active' for status, in await rows('embeddings', 'source_status'))
    assert len(await rows('embeddings', 'id')) == (2 if managed_generation else 1)
    assert len(await rows('derivation_fts_projections', 'id')) == (2 if managed_generation else 1)
    assert await db_session.scalar(text('SELECT search_vector IS NOT NULL FROM fts_entries WHERE raw_archive_id=:aid'), {'aid': aid})
    if managed_generation:
        assert await db_session.scalar(text('SELECT count(*) FROM derivation_generations WHERE status=\'active\'')) == 1
        assert await db_session.scalar(select(Head.active_generation_id).where(Head.raw_archive_id == aid)) == claim.generation_id
    derived_ids = {table: {row[0] for row in await rows(table, 'id')}
                   for table in ('facts', 'summaries', 'fts_entries', 'embeddings', 'derivation_fts_projections')}
    assert all(derived_ids.values())

    await db_session.rollback()
    # Only the external ledger file sink is redirected into disposable test /tmp.
    # Suppression/erasure/head lifecycle and DB tombstone paths are NOT mocked.
    await deletion.cascade_delete_archive_item(db_session, aid, backup_dir=str(tmp_path))
    source = (await db_session.execute(text(
        'SELECT raw_content,deleted_at FROM raw_archive WHERE id=:aid'), {'aid': aid})).one()
    assert source.raw_content == REDACTION_MARKER and source.deleted_at is not None
    for table, expected_ids in derived_ids.items():
        assert {row[0] for row in await rows(table, 'id')} == expected_ids
    for table, columns in (
        ('facts', 'fact_text,source_span'), ('summaries', 'summary_text'),
        ('fts_entries', 'text_content'), ('derivation_fts_projections', 'text_content'),
    ):
        after = await rows(table, columns + ',source_status')
        assert after and all(all(value == REDACTION_MARKER for value in row[:-1])
                             and row[-1] == 'source_removed' for row in after)
    assert all(status == 'source_removed' for status, in await rows('embeddings', 'source_status'))
    for table in ('fts_entries', 'derivation_fts_projections'):
        vectors = await rows(table, 'search_vector')
        assert vectors and all(vector is None for vector, in vectors)
    retained = (await db_session.execute(text(
        'SELECT content,source_status,status FROM canonical_memory WHERE id=:id'), {'id': cid})).one()
    assert tuple(retained) == ('retained curated evidence', 'source_removed', 'required_review')
    for table, key in (('conflict_groups', 'id'), ('review_queue_items', 'conflict_group_id')):
        statuses = (await db_session.scalars(text(
            f'SELECT source_status FROM {table} WHERE {key}=:id'), {'id': group_id})).all()
        assert statuses and all(status == 'source_removed' for status in statuses)
    heads = await rows('archive_derivation_heads', 'active_generation_id,candidate_generation_id')
    assert all(tuple(head) == (None, None) for head in heads)
    if managed_generation:
        assert len(heads) == 1
        states = await rows('derivation_generations', 'status,published_at')
        assert len(states) == 1 and states[0].status == 'abandoned' and states[0].published_at is not None
    assert await db_session.scalar(text("SELECT count(*) FROM derivation_generations WHERE status IN ('active','building','prepared')")) == 0
    assert await db_session.scalar(text("SELECT count(*) FROM derivation_fts_projections WHERE source_status='active'")) == 0
    assert await db_session.scalar(select(Tombstone.id).where(Tombstone.source_id == aid)) is not None

    positive = await report(db_session, 'check')
    assert positive['ok'] and positive['counts'] == {'A': 0, 'B': 0, 'C': 0, 'D': 1}
    assert positive['archives'][0]['archive_id'] == str(aid)
    assert positive['archives'][0]['reasons'] == []

    for column in ('fact_text', 'source_span'):
        # Corrupt only this ordinary legacy fact column; leave it suppressed, so
        # a status-only or current-reader-only validator would falsely pass.
        await db_session.execute(text(f'UPDATE facts SET {column}=:plaintext WHERE id=:id'),
            {'plaintext': 'synthetic leaked plaintext', 'id': fid})
        await db_session.commit()
        negative = await report(db_session, 'check')
        assert not negative['ok'] and negative['counts']['D'] == 1
        assert negative['archives'][0]['reasons'] == ['privacy_not_suppressed']
        await db_session.execute(text(f'UPDATE facts SET {column}=:marker WHERE id=:id'),
            {'marker': REDACTION_MARKER, 'id': fid})
        await db_session.commit()
        assert (await report(db_session, 'check'))['ok']
