"""Slice 2A: managed conflict/review replay identity. Synthetic DB and offline spies only.

One detection result per (trigger generation, detection contract version); replay writes nothing;
zero-link runs create no group/item; membership history is immutable.
"""
import asyncio
import uuid

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.domain import conflict_detection as cd
from app.domain.archive.models import RawArchiveItem
from app.domain.derivations import service as g
from app.domain.derived_memory import service as writers
from app.domain.derived_memory.models import Embedding
from app.domain.jobs.models import Job
from app.domain.review_queue.models import ReviewQueueItem
from app.domain.review_queue.service import dismiss_review_item, resolve_review_item
from app.worker import managed_dispatcher as managed
from tests.domain.test_derivation_generations import admit_claimed
from tests.worker.test_managed_dispatcher import job_row, spies  # noqa: F401 (fixture)

pytestmark = pytest.mark.asyncio

VEC = [1.0] + [0.0] * 383
FAR = [0.0, 1.0] + [0.0] * 382
TABLES = ("conflict_groups", "review_queue_items", "conflict_detection_runs",
          "conflict_group_members", "archive_derivation_heads")


async def counts(session):
    out = {t: await session.scalar(text(f"SELECT count(*) FROM {t}")) for t in TABLES}
    out["conflict_audit"] = await session.scalar(text(
        "SELECT count(*) FROM audit_events WHERE event_type='conflict_detected'"))
    out["linked_facts"] = await session.scalar(text(
        "SELECT count(*) FROM facts WHERE conflict_group_id IS NOT NULL"))
    return out


async def runs(session):
    return (await session.execute(text(
        "SELECT generation_id, raw_archive_id, contract_version, outcome, conflict_group_id, audit_event_id "
        "FROM conflict_detection_runs ORDER BY completed_at, contract_version"))).mappings().all()


@pytest.fixture
def embed(monkeypatch, db_session, spies):  # noqa: F811
    from app.worker import dispatcher as legacy
    vectors = {"value": VEC}

    async def fake(raw):
        assert not db_session.in_transaction()
        return vectors["value"]

    async def no_typed_link(*args, **kwargs):
        return None

    monkeypatch.setattr(writers, "embed_text", fake)
    monkeypatch.setattr(legacy, "_classify_link_pair", no_typed_link)
    return vectors


async def legacy_partner(session, facts=("beta fact", "beta second"), vector=VEC):
    source = RawArchiveItem(source_type="test", raw_content="beta source", content_hash=uuid.uuid4().hex)
    session.add(source)
    await session.commit()
    aid = source.id
    if facts:
        await writers.write_facts(session, aid, [{"fact_text": t} for t in facts])
    session.add(Embedding(raw_archive_id=aid, embedding=vector, embedding_model=writers.ACTIVE_EMBEDDING_MODEL))
    await session.commit()
    return aid


async def publish_for(session, job, facts=("alpha fact", "alpha second")):
    jid, aid, attempt, at = job.id, job.raw_archive_id, job.attempts, job.claimed_at
    gen = await admit_claimed(session, jid)
    claim = g.Claim(jid, aid, gen.id, attempt, at)
    await g.persist_summary(session, claim, "alpha summary")
    await g.persist_extraction(session, claim, [{"fact_text": t} for t in facts])
    await g.prepare_fts(session, claim, "alpha summary", "summary")
    await g.publish_generation(session, claim)
    return claim


async def published(session, facts=("alpha fact", "alpha second"), vector=None):
    claim = await publish_for(session, await job_row(session), facts)
    if vector is not None:
        await g.persist_embedding(session, claim, vector)
    return claim


async def detect(session, claim):
    row = await session.scalar(select(Embedding).where(Embedding.generation_id == claim.generation_id)
                               .execution_options(populate_existing=True))
    eid, vec = row.id, [float(v) for v in row.embedding]
    await session.rollback()
    return await cd.detect_and_group_duplicates(
        session, raw_archive_id=claim.archive_id, embedding_id=eid, embedding=vec,
        generation_id=claim.generation_id)


async def enrich_crash_before_finish(session, claim, monkeypatch):
    async def crash(*args, **kwargs):
        raise RuntimeError("synthetic crash after enrichment")
    with monkeypatch.context() as patch:
        patch.setattr(g, "finish_claim", crash)
        with pytest.raises(RuntimeError):
            await managed.enrich_and_finish(session, claim, allow_external=False)
    await session.rollback()


async def replay(session, claim):
    job = await session.get(Job, claim.job_id, populate_existing=True)
    await managed.dispatch_managed_job(session, job)
    job = await session.get(Job, claim.job_id, populate_existing=True)
    assert job.status == "completed"


async def only_item(session):
    return (await session.scalars(select(ReviewQueueItem).execution_options(populate_existing=True))).one()


async def test_t1_published_replay_after_keep_writes_nothing(db_session, spies, embed, monkeypatch):  # noqa: F811
    await legacy_partner(db_session)
    claim = await published(db_session)
    await enrich_crash_before_finish(db_session, claim, monkeypatch)
    first = await counts(db_session)
    assert (first["conflict_groups"], first["review_queue_items"], first["conflict_detection_runs"],
            first["conflict_audit"], first["conflict_group_members"], first["linked_facts"]) == (1, 1, 1, 1, 4, 4)
    item = await only_item(db_session)
    item_id = item.id
    await resolve_review_item(db_session, item_id, resolution_note="", resolved_by="test", action="keep")
    await db_session.commit()
    before = await counts(db_session)
    assert before["linked_facts"] == 0 and before["conflict_group_members"] == 4
    await replay(db_session, claim)
    assert await counts(db_session) == before
    assert (await only_item(db_session)).status == "resolved"
    assert [r["outcome"] for r in await runs(db_session)] == ["grouped"]


async def test_t2_published_replay_after_dismiss_writes_nothing(db_session, spies, embed, monkeypatch):  # noqa: F811
    await legacy_partner(db_session)
    claim = await published(db_session)
    await enrich_crash_before_finish(db_session, claim, monkeypatch)
    item = await only_item(db_session)
    await dismiss_review_item(db_session, item.id)
    await db_session.commit()
    before = await counts(db_session)
    await replay(db_session, claim)
    assert await counts(db_session) == before
    assert (await only_item(db_session)).status == "dismissed"


async def test_t3_published_replay_while_pending_writes_nothing(db_session, spies, embed, monkeypatch):  # noqa: F811
    await legacy_partner(db_session)
    claim = await published(db_session)
    await enrich_crash_before_finish(db_session, claim, monkeypatch)
    before = await counts(db_session)
    assert before["review_queue_items"] == 1
    await replay(db_session, claim)
    assert await counts(db_session) == before
    assert (await only_item(db_session)).status == "pending"


async def test_t4_fault_after_members_rolls_back_then_one_rerun(db_session, spies, embed, monkeypatch):  # noqa: F811
    from app.domain.review_queue import service as review
    await legacy_partner(db_session)
    claim = await published(db_session, vector=VEC)
    before = await counts(db_session)

    async def boom(*args, **kwargs):
        raise RuntimeError("synthetic fault after members, before commit")
    with monkeypatch.context() as patch:
        patch.setattr(review, "materialize_review_item", boom)
        with pytest.raises(RuntimeError):
            await detect(db_session, claim)
    await db_session.rollback()
    assert await counts(db_session) == before
    await detect(db_session, claim)
    after = await counts(db_session)
    assert (after["conflict_groups"], after["review_queue_items"], after["conflict_detection_runs"],
            after["conflict_group_members"], after["conflict_audit"]) == (1, 1, 1, 4, 1)
    await detect(db_session, claim)
    assert await counts(db_session) == after


async def test_t5_concurrent_same_generation_creates_exactly_one(db_session, test_engine, spies, embed):  # noqa: F811
    await legacy_partner(db_session)
    claim = await published(db_session, vector=VEC)
    await db_session.rollback()
    maker = async_sessionmaker(test_engine, expire_on_commit=False)
    async with maker() as first, maker() as second:
        results = await asyncio.gather(detect(first, claim), detect(second, claim), return_exceptions=True)
    assert not [r for r in results if isinstance(r, BaseException)]
    after = await counts(db_session)
    assert (after["conflict_groups"], after["review_queue_items"], after["conflict_detection_runs"],
            after["conflict_group_members"], after["conflict_audit"]) == (1, 1, 1, 4, 1)


async def test_t5b_primary_key_backstop_rolls_back_loser(db_session, test_engine, spies, embed, monkeypatch):  # noqa: F811
    from sqlalchemy.exc import IntegrityError
    await legacy_partner(db_session)
    claim = await published(db_session, vector=VEC)
    await db_session.rollback()

    async def never_recorded(*args, **kwargs):
        return None
    monkeypatch.setattr(cd, "_recorded_run", never_recorded)
    maker = async_sessionmaker(test_engine, expire_on_commit=False)
    async with maker() as first, maker() as second:
        results = await asyncio.gather(detect(first, claim), detect(second, claim), return_exceptions=True)
    errors = [r for r in results if isinstance(r, BaseException)]
    assert len(errors) == 1 and isinstance(errors[0], IntegrityError)
    after = await counts(db_session)
    assert (after["conflict_groups"], after["review_queue_items"], after["conflict_detection_runs"],
            after["conflict_group_members"], after["conflict_audit"]) == (1, 1, 1, 4, 1)


async def test_t6_partial_and_zero_link(db_session, spies, embed):  # noqa: F811
    partner = await legacy_partner(db_session, facts=("beta fact",))
    a = await published(db_session, facts=("alpha fact",), vector=VEC)
    await detect(db_session, a)
    b = await published(db_session, facts=("gamma fact",), vector=VEC)
    await detect(db_session, b)
    rows = await runs(db_session)
    assert [r["outcome"] for r in rows] == ["grouped", "grouped"]
    member_archives = (await db_session.execute(text(
        "SELECT raw_archive_id FROM conflict_group_members WHERE conflict_group_id=:g"),
        {"g": rows[1]["conflict_group_id"]})).scalars().all()
    assert member_archives == [b.archive_id]
    before = await counts(db_session)
    c = await published(db_session, facts=(), vector=VEC)
    await detect(db_session, c)
    after = await counts(db_session)
    assert [r["outcome"] for r in await runs(db_session)] == ["grouped", "grouped", "nothing_to_link"]
    for key in ("conflict_groups", "review_queue_items", "conflict_group_members", "conflict_audit", "linked_facts"):
        assert after[key] == before[key], key
    assert partner is not None


async def _reprocess(session, claim, facts=("alpha renewed",)):
    from app.domain.jobs.service import claim_next_job, reprocess_job
    job = await session.get(Job, claim.job_id, populate_existing=True)
    job.status = "failed"
    await session.commit()
    await reprocess_job(session, claim.job_id)
    current = await claim_next_job(session)
    assert current.id == claim.job_id
    new = await publish_for(session, current, facts)
    assert new.generation_id != claim.generation_id
    return new


async def test_t7_new_generation_is_reviewed_again_old_rows_untouched(db_session, spies, embed):  # noqa: F811
    await legacy_partner(db_session)
    old = await published(db_session, vector=VEC)
    await detect(db_session, old)
    item = await only_item(db_session)
    old_item_id = item.id
    await resolve_review_item(db_session, item.id, resolution_note="", resolved_by="test", action="keep")
    await db_session.commit()
    old_members = (await db_session.execute(text(
        "SELECT conflict_group_id, fact_id, raw_archive_id, generation_id FROM conflict_group_members ORDER BY fact_id"))).all()
    old_runs = await runs(db_session)
    new = await _reprocess(db_session, old)
    await g.persist_embedding(db_session, new, VEC)
    await detect(db_session, new)
    rows = await runs(db_session)
    assert rows[0] == old_runs[0] and len(rows) == 2
    assert rows[1]["generation_id"] == new.generation_id and rows[1]["outcome"] == "grouped"
    items = {i.id: i.status for i in (await db_session.scalars(select(ReviewQueueItem))).all()}
    assert items[old_item_id] == "resolved" and sorted(items.values()) == ["pending", "resolved"]
    members = (await db_session.execute(text(
        "SELECT conflict_group_id, fact_id, raw_archive_id, generation_id FROM conflict_group_members "
        "WHERE conflict_group_id=:g ORDER BY fact_id"), {"g": old_runs[0]["conflict_group_id"]})).all()
    assert members == old_members


async def test_t8_legacy_partner_members_have_null_generation(db_session, spies, embed):  # noqa: F811
    partner = await legacy_partner(db_session)
    claim = await published(db_session, vector=VEC)
    await detect(db_session, claim)
    rows = await runs(db_session)
    assert len(rows) == 1 and rows[0]["generation_id"] == claim.generation_id
    assert rows[0]["raw_archive_id"] == claim.archive_id and rows[0]["contract_version"] == 1
    members = (await db_session.execute(text(
        "SELECT raw_archive_id, generation_id FROM conflict_group_members"))).all()
    assert sorted((str(a), str(gid)) for a, gid in members) == sorted(
        [(str(partner), "None")] * 2 + [(str(claim.archive_id), str(claim.generation_id))] * 2)


async def test_t9_archive_deletion_keeps_history_and_erasure_valid(db_session, spies, embed, tmp_path):  # noqa: F811
    from app.domain.archive.service import cascade_delete_archive_item
    from app.domain.derivations import cutover
    partner = await legacy_partner(db_session)
    claim = await published(db_session, vector=VEC)
    await detect(db_session, claim)
    group_id = (await runs(db_session))[0]["conflict_group_id"]
    await cascade_delete_archive_item(db_session, partner, backup_dir=str(tmp_path))
    await db_session.commit()
    status = await db_session.scalar(text("SELECT source_status::text FROM conflict_groups WHERE id=:g"), {"g": group_id})
    item_status = await db_session.scalar(text("SELECT source_status::text FROM review_queue_items"))
    assert status == "source_removed" and item_status == "source_removed"
    assert await db_session.scalar(text("SELECT count(*) FROM conflict_group_members")) == 4
    assert await db_session.scalar(text("SELECT count(*) FROM conflict_detection_runs")) == 1
    assert await cutover.privacy_valid(db_session, await cutover.load_state(db_session, partner))


async def test_t10_non_active_generation_writes_nothing(db_session, spies, embed, monkeypatch):  # noqa: F811
    await legacy_partner(db_session)
    old = await published(db_session, vector=VEC)
    await detect(db_session, old)
    item = await only_item(db_session)
    await resolve_review_item(db_session, item.id, resolution_note="", resolved_by="test", action="keep")
    await db_session.commit()
    new = await _reprocess(db_session, old)
    old_embedding = await db_session.scalar(text("SELECT id FROM embeddings WHERE generation_id=:g"),
                                            {"g": old.generation_id})
    before = await counts(db_session)
    monkeypatch.setattr(cd, "DETECTION_CONTRACT_VERSION", 2)
    await db_session.rollback()
    result = await cd.detect_and_group_duplicates(
        db_session, raw_archive_id=old.archive_id, embedding_id=old_embedding, embedding=VEC,
        generation_id=old.generation_id)
    assert result is None
    assert await counts(db_session) == before
    assert new.generation_id is not None


async def test_t11_no_candidates_recorded_and_replay_writes_nothing(db_session, spies, embed):  # noqa: F811
    await legacy_partner(db_session, vector=FAR)
    claim = await published(db_session, vector=VEC)
    await detect(db_session, claim)
    rows = await runs(db_session)
    assert [(r["outcome"], r["conflict_group_id"], r["audit_event_id"]) for r in rows] == [("no_candidates", None, None)]
    after = await counts(db_session)
    assert (after["conflict_groups"], after["review_queue_items"], after["conflict_audit"]) == (0, 0, 0)
    await detect(db_session, claim)
    assert await counts(db_session) == after


async def test_t12_members_equal_linked_facts(db_session, spies, embed):  # noqa: F811
    await legacy_partner(db_session)
    claim = await published(db_session, vector=VEC)
    await detect(db_session, claim)
    group_id = (await runs(db_session))[0]["conflict_group_id"]
    linked = (await db_session.execute(text(
        "SELECT id, raw_archive_id, generation_id FROM facts WHERE conflict_group_id=:g ORDER BY id"),
        {"g": group_id})).all()
    members = (await db_session.execute(text(
        "SELECT fact_id, raw_archive_id, generation_id FROM conflict_group_members "
        "WHERE conflict_group_id=:g ORDER BY fact_id"), {"g": group_id})).all()
    assert len(linked) == 4 and [tuple(r) for r in members] == [tuple(r) for r in linked]
    run = (await runs(db_session))[0]
    audit = await db_session.scalar(text(
        "SELECT operation_metadata->>'conflict_group_id' FROM audit_events WHERE id=:a"), {"a": run["audit_event_id"]})
    assert audit == str(group_id)


async def test_t13_contract_version_bump_allows_exactly_one_new_run(db_session, spies, embed, monkeypatch):  # noqa: F811
    await legacy_partner(db_session)
    claim = await published(db_session, vector=VEC)
    await detect(db_session, claim)
    monkeypatch.setattr(cd, "DETECTION_CONTRACT_VERSION", 2)
    await detect(db_session, claim)
    after = await counts(db_session)
    await detect(db_session, claim)
    assert await counts(db_session) == after
    rows = await runs(db_session)
    assert [(r["contract_version"], r["outcome"]) for r in rows] == [(1, "grouped"), (2, "nothing_to_link")]
    assert after["conflict_groups"] == after["review_queue_items"] == 1
