"""Explicit, one-shot OFF baseline tooling. Never called by app startup/worker.

Reports are bounded evidence, not authorization. Every write rechecks durable
controls under archive-first locks; every certificate rescans the full population.
No provider, legacy FTS writer, admission or migration dependency belongs here.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
import uuid

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import select, text

from app.domain.archive.models import RawArchiveItem, Tombstone
from app.domain.archive.tombstones import REDACTION_MARKER
from app.domain.derivations import service as coordinator
from app.domain.derivations.models import (
    ArchiveDerivationHead as Head, DerivationGeneration as Generation,
    DerivationFtsProjection as Projection,
)
from app.domain.derivations.visibility import enabled
from app.domain.jobs.models import Job

FORMAT = 'recalium-baseline-v1'
MAX_ARCHIVES = 10000
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
CLASSES = ('A', 'B', 'C', 'D')


class CutoverError(Exception):
    """Only static codes; never source text, credentials or database messages."""


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)


def software():
    # Binds evidence to both this policy and the actual shared publication check.
    return digest(Path(__file__).read_text() + Path(coordinator.__file__).read_text())


def deployed_head():
    backend = Path(__file__).resolve().parents[3]
    config = Config(str(backend / 'alembic.ini'))
    config.set_main_option('script_location', str(backend / 'migrations'))
    return ScriptDirectory.from_config(config).get_current_head()


def require_idle(session):
    if session.in_transaction() or session.new or session.dirty or session.deleted:
        raise CutoverError('dedicated_idle_session_required')


@asynccontextmanager
async def readonly_snapshot(session):
    require_idle(session)
    session.expire_all()
    async with session.begin():
        await session.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY'))
        yield


async def load_state(session, aid, *, locked=False):
    fresh = {'populate_existing': True, 'autoflush': False}

    async def rows(model, criterion, order):
        stmt = select(model).where(criterion).order_by(order).execution_options(**fresh)
        if locked:
            stmt = stmt.with_for_update()
        return list((await session.scalars(stmt)).all())

    # Never use lock_archives_and_heads: it intentionally CREATES missing heads.
    sources = await rows(RawArchiveItem, RawArchiveItem.id == aid, RawArchiveItem.id)
    if not sources:
        raise CutoverError('replan_required')
    source = sources[0]
    heads = await rows(Head, Head.raw_archive_id == aid, Head.raw_archive_id)
    jobs = await rows(Job, Job.raw_archive_id == aid, Job.id)
    generations = await rows(Generation, Generation.raw_archive_id == aid, Generation.id)
    tombstones = list((await session.scalars(select(Tombstone).where(
        coordinator.or_source_tombstone(source)).order_by(Tombstone.id)
        .execution_options(**fresh))).all())
    projections = await rows(Projection, Projection.raw_archive_id == aid, Projection.id)
    return source, heads[0] if heads else None, jobs, generations, tombstones, projections


def control_fingerprint(state):
    source, head, jobs, generations, tombstones, _ = state
    # No projection state here: valid inserts must remain resumable after commit
    # acknowledgement loss. Projection correctness is independently revalidated.
    return digest(canonical({
        'source': [source.id, source.content_hash, digest(source.raw_content),
                   source.deleted_at, digest(canonical(source.metadata_json))],
        'head': None if head is None else [head.active_generation_id,
                 head.candidate_generation_id, head.legacy_retired_at],
        'jobs': [[j.id, j.raw_archive_id, j.generation_id, j.status, j.attempts,
                  j.max_attempts, j.claimed_at, j.completed_at, j.updated_at] for j in jobs],
        'generations': [[g.id, g.raw_archive_id, g.original_job_id, g.status,
                         g.source_content_hash, g.contract_version, g.predecessor_generation_id,
                         g.published_at, g.superseded_at] for g in generations],
        'tombstones': [[t.id, t.source_id, t.content_hash, t.removed_at] for t in tombstones],
    }))


async def vector_valid(session, projection):
    return bool(await session.scalar(text('''SELECT search_vector IS NOT NULL
        AND search_vector = to_tsvector('english', text_content)
        FROM derivation_fts_projections WHERE id=:id'''), {'id': projection.id}))


async def privacy_valid(session, state):
    source, head, _, generations, _, projections = state
    if (source.raw_content != REDACTION_MARKER
        or (head and (head.active_generation_id or head.candidate_generation_id))
        or any(g.status in {'active', 'building', 'prepared'} for g in generations)
        or any(p.source_status != 'source_removed' or p.text_content != REDACTION_MARKER
               or p.search_vector is not None for p in projections)):
        return False
    # Deliberately all generations, no current-reader filters. Canonical/curated
    # text is governed separately by PRIV-02 and is not erased by this tooling.
    for table, columns in (
        ('facts', ('fact_text', 'source_span')), ('summaries', ('summary_text',)),
        ('fts_entries', ('text_content',)), ('embeddings', ()),
    ):
        bad = "source_status <> 'source_removed'"
        for column in columns:
            bad += f' OR {column} IS DISTINCT FROM :marker'
        if table == 'fts_entries':
            bad += ' OR search_vector IS NOT NULL'
        if await session.scalar(text(f'SELECT count(*) FROM {table} WHERE raw_archive_id=:aid AND ({bad})'),
            {'aid': source.id, 'marker': REDACTION_MARKER}):
            return False
    if await session.scalar(text('''SELECT count(*) FROM canonical_memory
        WHERE raw_archive_id=:aid AND (source_status <> 'source_removed' OR status <> 'required_review')'''),
        {'aid': source.id}):
        return False
    for table in ('conflict_groups', 'review_queue_items'):
        key = 'id' if table == 'conflict_groups' else 'conflict_group_id'
        if await session.scalar(text(f'''SELECT count(*) FROM {table}
            WHERE {key} IN (SELECT conflict_group_id FROM facts WHERE raw_archive_id=:aid)
            AND source_status <> 'source_removed' '''), {'aid': source.id}):
            return False
    return True


async def classify(session, state, *, coverage):
    source, head, jobs, generations, tombstones, projections = state
    reasons = []
    removed = source.deleted_at is not None or bool(tombstones)
    active_id = head.active_generation_id if head else None
    retired = head.legacy_retired_at is not None if head else False
    category = 'D' if removed else 'B' if active_id else 'C' if retired else 'A'
    if len(jobs) != 1:
        reasons.append('job_cardinality')
    if any(j.status != 'completed' for j in jobs):
        reasons.append('job_not_completed')
    by_id = {g.id: g for g in generations}
    if any(j.generation_id is not None and (j.generation_id not in by_id
           or by_id[j.generation_id].original_job_id != j.id) for j in jobs):
        reasons.append('job_generation_binding')
    if any(g.original_job_id not in {j.id for j in jobs} or g.raw_archive_id != source.id for g in generations):
        reasons.append('generation_owner_binding')
    if any(g.predecessor_generation_id is not None and g.predecessor_generation_id not in by_id
           for g in generations):
        reasons.append('generation_predecessor_binding')
    if any(g.status in {'building', 'prepared'} for g in generations) or (head and head.candidate_generation_id):
        reasons.append('open_candidate')
    if any((g.status == 'active') != (g.id == active_id) for g in generations):
        reasons.append('active_head_mismatch')
    baseline = [p for p in projections if p.generation_id is None]
    if category == 'D':
        if tombstones and source.deleted_at is None:
            reasons.append('tombstone_reapply_required')
        if not await privacy_valid(session, state):
            reasons.append('privacy_not_suppressed')
    elif category == 'C':
        reasons.append('retired_headless')
    elif category == 'A':
        if any(g.status == 'active' for g in generations):
            reasons.append('headless_active_generation')
        if len(baseline) > 1:
            reasons.append('baseline_cardinality')
        if not baseline and coverage:
            reasons.append('baseline_missing')
        for row in baseline:
            if row.source_status != 'active':
                reasons.append('baseline_suppressed')
            if row.projection_kind != 'raw_fallback' or row.text_content != source.raw_content[:10000]:
                reasons.append('baseline_stale')
            if not await vector_valid(session, row):
                reasons.append('baseline_vector_invalid')
    else:
        generation = by_id.get(active_id)
        if (generation is None or generation.status != 'active'
            or generation.source_content_hash != source.content_hash
            or generation.contract_version != 1
            or len(jobs) != 1 or jobs[0].id != generation.original_job_id
            or jobs[0].generation_id != generation.id):
            reasons.append('active_binding_invalid')
        else:
            try:
                # Same durable checker as actual publication and M4 replay.
                # A coherent READ ONLY snapshot replaces writer locks here.
                await coordinator.validate_publication(session, head, generation)
            except coordinator.DerivationConflict:
                reasons.append('publication_invalid')
            active_projections = [p for p in projections if p.generation_id == generation.id]
            if len(active_projections) != 1 or not await vector_valid(session, active_projections[0]):
                reasons.append('active_vector_invalid')
    return {
        'archive_id': str(source.id), 'class': category,
        'reasons': sorted(set(reasons)), 'control_hash': control_fingerprint(state),
        'source_hash': digest(source.raw_content), 'content_hash_digest': digest(source.content_hash),
        'baseline_text_hash': digest(source.raw_content[:10000]),
    }


def partition_valid(rows, population):
    ids = [r['archive_id'] for r in rows]
    return (len(ids) == len(set(ids)) == len(population)
        and set(ids) == {str(a) for a in population}
        and all(r['class'] in CLASSES for r in rows))


async def schema_state(session):
    current = []
    if await session.scalar(text("SELECT to_regclass('alembic_version')")):
        current = list((await session.scalars(text('SELECT version_num FROM alembic_version ORDER BY version_num'))).all())
    expected = deployed_head()
    return {'deployed_head': expected, 'current': current}, current == [expected]


async def inspect_population(session, *, kind='plan', cutover=False):
    if kind not in {'plan', 'check'}:
        raise CutoverError('invalid_action')
    async with readonly_snapshot(session):
        schema, schema_ok = await schema_state(session)
        if not schema_ok:
            raise CutoverError('schema_mismatch')
        population = list((await session.scalars(select(RawArchiveItem.id)
            .order_by(RawArchiveItem.id).limit(MAX_ARCHIVES + 1))).all())
        if len(population) > MAX_ARCHIVES:
            raise CutoverError('population_limit_exceeded')
        rows = [await classify(session, await load_state(session, aid), coverage=kind == 'check')
                for aid in population]
        blockers = []
        if not partition_valid(rows, population):
            blockers.append('partition_mismatch')
        orphan_count = await session.scalar(text('''SELECT count(*) FROM jobs j
            LEFT JOIN raw_archive r ON r.id=j.raw_archive_id WHERE r.id IS NULL'''))
        if orphan_count:
            blockers.append('orphan_jobs')
        # Check orphan/control references even if database constraints were bypassed.
        for table in ('archive_derivation_heads', 'derivation_generations', 'derivation_fts_projections'):
            if await session.scalar(text(f'''SELECT count(*) FROM {table} d
                LEFT JOIN raw_archive r ON r.id=d.raw_archive_id WHERE r.id IS NULL''')):
                blockers.append('orphan_derivation_state')
        for table in ('facts', 'summaries', 'embeddings', 'derivation_fts_projections', 'jobs'):
            if await session.scalar(text(f'''SELECT count(*) FROM {table} d
                LEFT JOIN derivation_generations g ON g.id=d.generation_id
                WHERE d.generation_id IS NOT NULL AND
                (g.id IS NULL OR g.raw_archive_id<>d.raw_archive_id)''')):
                blockers.append('derived_generation_binding')
        if await session.scalar(text('''SELECT count(*) FROM derivation_stage_completions c
            LEFT JOIN derivation_generations g ON g.id=c.generation_id WHERE g.id IS NULL''')):
            blockers.append('orphan_stage_completion')
        other_clients = await session.scalar(text('''SELECT count(*) FROM pg_stat_activity
            WHERE datname=current_database() AND pid<>pg_backend_pid()
            AND (backend_type='client backend' OR backend_type IS NULL)'''))
        if cutover and other_clients:
            blockers.append('other_database_clients')
        counts = {category: sum(r['class'] == category for r in rows) for category in CLASSES}
        result = {
            'format': FORMAT, 'kind': kind, 'software_hash': software(), 'schema': schema,
            'postgres_version_num': await session.scalar(text("SELECT current_setting('server_version_num')")),
            'snapshot': {'isolation': await session.scalar(text('SHOW transaction_isolation')),
                         'read_only': await session.scalar(text('SHOW transaction_read_only'))},
            'job_contract': 'exactly_one_completed_per_archive',
            'cutover': cutover, 'other_database_clients': other_clients,
            'total_archives': len(population), 'counts': counts,
            'orphan_jobs': orphan_count, 'blockers': sorted(set(blockers)), 'archives': rows,
            'ok': not blockers and not any(r['reasons'] for r in rows),
        }
        result['id'] = digest(canonical(result))
        bounded_json(result)
        return result


def bounded_json(artifact):
    encoded = canonical(artifact)
    if len(encoded.encode('utf-8')) > MAX_ARTIFACT_BYTES:
        raise CutoverError('artifact_limit_exceeded')
    return encoded


def validate_plan(plan):
    try:
        recorded = plan['id']
        body = {k: v for k, v in plan.items() if k != 'id'}
        if (plan['format'] != FORMAT or plan['kind'] != 'plan' or not plan['ok']
            or plan['software_hash'] != software() or digest(canonical(body)) != recorded
            or len(plan['archives']) > MAX_ARCHIVES
            or not partition_valid(plan['archives'], [uuid.UUID(r['archive_id']) for r in plan['archives']])
            or sum(plan['counts'].values()) != plan['total_archives']
            or len(plan['archives']) != plan['total_archives']):
            raise CutoverError('invalid_plan')
        bounded_json(plan)
    except (KeyError, TypeError, ValueError):
        raise CutoverError('invalid_plan') from None


async def prepare(session, plan):
    """Resume by rerunning the same plan; baseline rows, not artifact cursors, win.

    Per-archive commits, not exactly-once batch execution. A partial failure leaves
    already committed baselines intact and safe to reuse after replan/resume.
    """
    if enabled():
        raise CutoverError('r2_must_be_off')
    require_idle(session)
    validate_plan(plan)
    # Recheck the complete population/global blockers too. The saved artifact is
    # never permission to ignore a newly introduced orphan/control anomaly.
    try:
        fresh = await inspect_population(session, kind='plan')
    except CutoverError:
        raise CutoverError('replan_required') from None
    if (not fresh['ok'] or fresh['schema'] != plan['schema']
        or {r['archive_id']: r['control_hash'] for r in fresh['archives']} != {
            r['archive_id']: r['control_hash'] for r in plan['archives']}):
        raise CutoverError('replan_required')
    result = {'format': FORMAT, 'kind': 'prepare', 'plan_id': plan['id'], 'created': 0, 'reused': 0, 'skipped': 0}
    for expected in sorted(plan['archives'], key=lambda r: r['archive_id']):
        aid = uuid.UUID(expected['archive_id'])
        session.expire_all()
        async with session.begin():
            if enabled():
                raise CutoverError('r2_must_be_off')
            state = await load_state(session, aid, locked=True)
            row = await classify(session, state, coverage=False)
            if row['control_hash'] != expected['control_hash'] or row['reasons']:
                raise CutoverError('replan_required')
            if row['class'] != 'A':
                action = 'skipped'
            else:
                baseline = [p for p in state[5] if p.generation_id is None]
                if baseline:
                    action = 'reused'
                else:
                    session.add(Projection(raw_archive_id=aid, generation_id=None,
                        projection_kind='raw_fallback', text_content=state[0].raw_content[:10000]))
                    await session.flush()
                    action = 'created'
        result[action] += 1
    return result


def human_report(artifact):
    """Human format is generated solely from the allowlisted evidence report."""
    lines = [f"{FORMAT} {artifact['kind']}", f"Evidence ID: {artifact.get('id', artifact.get('plan_id'))}"]
    if artifact['kind'] == 'prepare':
        lines.append(f"Created={artifact['created']} reused={artifact['reused']} skipped={artifact['skipped']}")
    else:
        lines.extend([f"OK={artifact['ok']} total={artifact['total_archives']} classes={canonical(artifact['counts'])}",
                      f"Cutover={artifact['cutover']} other_database_clients={artifact['other_database_clients']}",
                      f"Software={artifact['software_hash']} schema={canonical(artifact['schema'])}",
                      f"PostgreSQL={artifact['postgres_version_num']} snapshot={canonical(artifact['snapshot'])}",
                      f"Job contract={artifact['job_contract']}",
                      f"Blockers={canonical(artifact['blockers'])}"])
        lines.extend(f"{r['archive_id']} {r['class']} {','.join(r['reasons']) or 'ok'}" for r in artifact['archives'])
    return '\n'.join(lines) + '\n'
