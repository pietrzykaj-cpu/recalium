"""M3 privacy withdrawal only; synthetic PostgreSQL, no model/provider calls."""
import asyncio
from datetime import datetime, timezone

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.domain.archive import service as archive
from app.domain.archive.models import RawArchiveItem
from app.domain.archive.tombstones import REDACTION_MARKER
from app.domain.derivations import privacy, service as g
from app.domain.derivations.models import (
    ArchiveDerivationHead as Head, DerivationGeneration as Gen,
)
from app.domain.derived_memory.models import Fact
from app.domain.derivations.visibility import HISTORY
from app.infrastructure.settings import get_settings
from tests.domain.test_derivation_generations import claimed, managed
from tests.domain.test_derivation_boundaries import build

pytestmark = pytest.mark.asyncio


async def history(session, aid):
    """Compare retained generation metadata and completion records exactly."""
    generations = [dict(r) for r in (await session.execute(
        Gen.__table__.select().where(Gen.raw_archive_id == aid).order_by(Gen.id)
    )).mappings()]
    completions = (await session.execute(text("""
        SELECT c.* FROM derivation_stage_completions c
        JOIN derivation_generations g ON g.id=c.generation_id
        WHERE g.raw_archive_id=:aid ORDER BY c.generation_id,c.stage_name
    """), {"aid": aid})).mappings().all()
    return generations, [dict(r) for r in completions]


async def seed_history(session, candidate_status):
    from app.domain.jobs.service import reprocess_job, claim_next_job
    from app.domain.derived_memory.service import write_facts, write_fts_entry
    source, job = await claimed(session)
    aid, jid, content_hash = source.id, job.id, source.content_hash
    await write_facts(session, aid, [{"fact_text": "legacy private evidence"}])
    await write_fts_entry(session, aid, "legacy private evidence")
    first = await g.admit_generation(session, jid)
    job = await session.get(type(job), jid)
    await g.raw_baseline(session, g.Claim(jid, aid, first.id, job.attempts, job.claimed_at))
    published = []
    for label in ("old private evidence", "current private evidence"):
        job = await session.get(type(job), jid)
        claim, fid = await build(session, job, label)
        published.append((claim, fid))
        job = await session.get(type(job), jid)
        job.status = "retryable_failed"
        await session.commit()
        await reprocess_job(session, jid)
        job = await claim_next_job(session)
        assert job.id == jid
    gen = await g.admit_generation(session, jid)
    job = await session.get(type(job), jid)
    candidate = g.Claim(jid, aid, gen.id, job.attempts, job.claimed_at)
    await g.persist_summary(session, candidate, "candidate private evidence")
    facts = await g.persist_extraction(session, candidate, [{"fact_text": "candidate private evidence"}])
    candidate_fid = facts[0].id
    if candidate_status == "prepared":
        await g.prepare_fts(session, candidate, "candidate private evidence", "summary")
    for sequence, status in ((4, "failed"), (5, "abandoned")):
        session.add(Gen(raw_archive_id=aid, original_job_id=jid, sequence=sequence,
                        creation_reason="automatic_processing", source_content_hash=content_hash,
                        status=status))
    await session.commit()
    return aid, content_hash, published, candidate, candidate_fid


async def assert_withdrawn(session, aid, before, retired):
    after, completions = await history(session, aid)
    old, old_completions = before
    assert [r["id"] for r in after] == [r["id"] for r in old]
    for previous, current in zip(old, after):
        expected = "abandoned" if previous["status"] in {"building", "prepared", "active"} else previous["status"]
        assert current["status"] == expected
        assert {k: v for k, v in current.items() if k != "status"} == {
            k: v for k, v in previous.items() if k != "status"}
    assert completions == old_completions
    head = await session.get(Head, aid, populate_existing=True)
    assert head.active_generation_id is None and head.candidate_generation_id is None
    assert head.legacy_retired_at == retired
    assert await session.scalar(text("""
        SELECT count(*) FROM derivation_generations
        WHERE raw_archive_id=:aid AND status IN ('building','prepared','active')
    """), {"aid": aid}) == 0
    for table, column in (("facts", "fact_text"), ("summaries", "summary_text"),
                          ("fts_entries", "text_content"), ("derivation_fts_projections", "text_content")):
        rows = (await session.execute(text(
            f"SELECT {column},source_status FROM {table} WHERE raw_archive_id=:aid"
        ), {"aid": aid})).all()
        assert rows and all(tuple(row) == (REDACTION_MARKER, "source_removed") for row in rows)
    assert await session.scalar(text("""
        SELECT count(*) FROM derivation_fts_projections
        WHERE raw_archive_id=:aid AND search_vector IS NOT NULL
    """), {"aid": aid}) == 0
    source = await session.get(RawArchiveItem, aid, populate_existing=True)
    assert source.deleted_at is not None and source.raw_content == REDACTION_MARKER


@pytest.mark.parametrize("r2_on", [False, True])
@pytest.mark.parametrize("candidate_status", ["building", "prepared"])
@pytest.mark.parametrize("entry", ["cascade", "reconcile", "reapply", "bundle"])
async def test_all_privacy_entry_points_withdraw_without_rewriting_history(
    db_session, managed, monkeypatch, tmp_path, r2_on, candidate_status, entry,
):
    from app.api.routes import portability
    aid, content_hash, published, candidate, _ = await seed_history(db_session, candidate_status)
    before = await history(db_session, aid)
    retired = (await db_session.get(Head, aid)).legacy_retired_at
    # Every approved source status is represented, and active was truly published.
    assert {r["status"] for r in before[0]} == {
        "superseded", "active", candidate_status, "failed", "abandoned"}
    active = next(r for r in before[0] if r["status"] == "active")
    assert active["published_at"] is not None and active["superseded_at"] is None
    monkeypatch.setattr(get_settings(), "generation_idempotency_enabled", r2_on)
    monkeypatch.setattr(portability, "append_tombstone_ledger", lambda *a, **k: None)
    if entry == "cascade":
        await archive.cascade_delete_archive_item(db_session, aid, backup_dir=str(tmp_path))
    elif entry == "reconcile":
        source = await db_session.get(RawArchiveItem, aid)
        source.deleted_at = datetime.now(timezone.utc)
        await db_session.commit()
        assert await archive.suppress_new_derivations_if_deleted(db_session, aid)
    elif entry == "reapply":
        monkeypatch.setattr(archive, "read_tombstone_ledger", lambda *a: [{"content_hash": content_hash}])
        assert (await archive.reapply_tombstones(db_session, str(tmp_path)))["reapplied"] == 1
    else:
        hashes = set()
        assert await portability._apply_bundle_tombstones(db_session, [{"content_hash": content_hash}], hashes) == 1
        await db_session.commit()
        assert hashes == {content_hash}
    await assert_withdrawn(db_session, aid, before, retired)
    # Reconciliation is the normal idempotent replay for an already deleted source.
    assert await archive.suppress_new_derivations_if_deleted(db_session, aid)
    await assert_withdrawn(db_session, aid, before, retired)
    if r2_on:
        from app.domain.retrieval.service import _keyword_candidates
        assert not (await db_session.scalars(select(Fact))).all()
        assert not await _keyword_candidates(db_session, "private")
        with pytest.raises(g.DerivationConflict):
            await g.publish_generation(db_session, candidate)
    # History bypasses current selection, never erasure; the original IDs survive.
    rows = (await db_session.scalars(select(Fact).execution_options(**HISTORY, populate_existing=True))).all()
    assert {fid for _, fid in published} <= {f.id for f in rows}
    assert all(f.fact_text == REDACTION_MARKER for f in rows)


@pytest.mark.parametrize("lost_ack", [False, True])
async def test_delete_rollback_or_lost_ack_preserves_real_timestamps(
    db_session, managed, monkeypatch, tmp_path, lost_ack,
):
    aid, _, _, _, _ = await seed_history(db_session, "prepared")
    before = await history(db_session, aid)
    head = await db_session.get(Head, aid)
    original_head = (head.active_generation_id, head.candidate_generation_id, head.legacy_retired_at)
    tables = ("raw_archive", "jobs", "facts", "summaries", "fts_entries",
              "derivation_fts_projections", "archive_derivation_heads",
              "derivation_generations", "derivation_stage_completions", "tombstones", "audit_events")
    async def snapshot_rows():
        return {table: (await db_session.scalars(text(
            f"SELECT row_to_json(t)::text FROM {table} t ORDER BY 1"))).all() for table in tables}
    original_rows = await snapshot_rows()
    commit = db_session.commit
    async def interrupt():
        await db_session.flush()
        if lost_ack:
            await commit()
        raise RuntimeError("synthetic acknowledgement loss" if lost_ack else "synthetic precommit failure")
    with monkeypatch.context() as patch:
        patch.setattr(db_session, "commit", interrupt)
        with pytest.raises(RuntimeError):
            await archive.cascade_delete_archive_item(db_session, aid, backup_dir=str(tmp_path))
    await db_session.rollback()
    if lost_ack:
        assert await archive.suppress_new_derivations_if_deleted(db_session, aid)
        await assert_withdrawn(db_session, aid, before, original_head[2])
    else:
        assert await snapshot_rows() == original_rows
        assert await history(db_session, aid) == before
        head = await db_session.get(Head, aid, populate_existing=True)
        assert (head.active_generation_id, head.candidate_generation_id, head.legacy_retired_at) == original_head
        source = await db_session.get(RawArchiveItem, aid, populate_existing=True)
        assert source.deleted_at is None and source.raw_content != REDACTION_MARKER
        assert await db_session.scalar(text("SELECT count(*) FROM tombstones")) == 0
        assert await db_session.scalar(text("SELECT count(*) FROM facts WHERE source_status='source_removed'")) == 0


@pytest.mark.parametrize("first", ["publish", "delete"])
async def test_publication_and_deletion_serialize_both_orders(
    db_session, test_engine, managed, monkeypatch, tmp_path, first,
):
    source, job = await claimed(db_session)
    aid = source.id
    claim, _ = await build(db_session, job, "race private evidence", publish=False)
    before = await history(db_session, aid)
    await db_session.rollback()
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    flushed, release = asyncio.Event(), asyncio.Event()
    committed_generation = []
    async with factory() as publisher, factory() as deleter:
        holder = publisher if first == "publish" else deleter
        waiter = deleter if first == "publish" else publisher
        holder_pid = await holder.scalar(text("SELECT pg_backend_pid()"))
        waiter_pid = await waiter.scalar(text("SELECT pg_backend_pid()"))
        await holder.rollback()
        await waiter.rollback()
        commit = holder.commit
        async def hold_before_commit():
            await holder.flush()
            committed_generation.extend((await history(holder, aid))[0])
            flushed.set()
            await release.wait()
            await commit()
        monkeypatch.setattr(holder, "commit", hold_before_commit)
        async def publish():
            return await g.publish_generation(publisher, claim)
        async def delete():
            await archive.cascade_delete_archive_item(deleter, aid, backup_dir=str(tmp_path))
        leading = asyncio.create_task(publish() if first == "publish" else delete())
        trailing = None
        try:
            await asyncio.wait_for(flushed.wait(), 10)
            trailing = asyncio.create_task(delete() if first == "publish" else publish())
            async with asyncio.timeout(10):
                while True:
                    blockers = await holder.scalar(text("SELECT pg_blocking_pids(:pid)"), {"pid": waiter_pid})
                    if holder_pid in blockers:
                        break
                    assert not trailing.done(), "Contender must wait for the archive lock"
                    await asyncio.sleep(0)  # scheduling yield, not a timing assumption
            release.set()
            await asyncio.wait_for(leading, 10)
            if first == "delete":
                with pytest.raises(g.DerivationConflict):
                    await asyncio.wait_for(trailing, 10)
            else:
                await asyncio.wait_for(trailing, 10)
        finally:
            release.set()
            for task in (leading, trailing):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(*(t for t in (leading, trailing) if t is not None), return_exceptions=True)
    after, completions = await history(db_session, aid)
    assert after[0]["status"] == "abandoned"
    assert after[0]["superseded_at"] is None
    assert after[0]["published_at"] == committed_generation[0]["published_at"]
    assert (after[0]["published_at"] is not None) == (first == "publish")
    assert completions == before[1]
    head = await db_session.get(Head, aid, populate_existing=True)
    assert head.active_generation_id is None and head.candidate_generation_id is None
    assert (head.legacy_retired_at is not None) == (first == "publish")
    assert await db_session.scalar(text("SELECT count(*) FROM facts WHERE source_status='active'")) == 0
    assert await db_session.scalar(text("SELECT count(*) FROM derivation_generations WHERE status IN ('active','building','prepared')")) == 0


async def test_explicit_history_returns_redaction_not_withdrawn_plaintext(client, db_session, managed, tmp_path):
    from app.domain.canonical_memory.models import CanonicalMemoryItem
    from app.domain.canonical_memory.service import promote_fact_to_canonical, SourceRemovedError
    aid, _, published, _, candidate_fid = await seed_history(db_session, "prepared")
    curated = await promote_fact_to_canonical(
        db_session, published[0][1], "separately curated content", "test_user", confirmed=True)
    curated_id, curated_provenance = curated.id, curated.provenance_note
    await archive.cascade_delete_archive_item(db_session, aid, backup_dir=str(tmp_path))
    retained = await db_session.get(CanonicalMemoryItem, curated_id, populate_existing=True)
    assert retained.content == "separately curated content"
    assert retained.provenance_note == curated_provenance
    assert retained.source_status == "source_removed" and retained.status == "required_review"
    for _, fid in published:
        response = await client.get(f"/api/facts/{fid}")
        assert response.status_code == 200
        assert response.json()["fact_text"] == REDACTION_MARKER
        assert response.json()["source_status"] == "source_removed"
        assert response.json()["generation_visibility"] == "history"
    assert (await client.get(f"/api/facts/{candidate_fid}")).status_code == 404
    assert (await client.get("/api/facts")).json()["facts"] == []
    with pytest.raises(SourceRemovedError):
        await promote_fact_to_canonical(
            db_session, published[1][1], "must not resurrect", "test_user", confirmed=True)
    with pytest.raises(RuntimeError, match="certified restore"):
        await privacy.assert_restore_safe(db_session)
