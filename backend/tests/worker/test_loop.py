"""Worker loop tests — PIPE-01, PIPE-04.

Tests will FAIL (RED) until app.worker.loop and app.domain.jobs.service are created.
"""
from __future__ import annotations

import uuid

import pytest

# These imports will fail until plans 03-04 create the modules.
# That is expected — tests are stubs until GREEN.
pytest.importorskip("app.worker.loop", reason="worker.loop not yet implemented")
pytest.importorskip("app.domain.jobs.service", reason="jobs.service not yet implemented")

from app.worker.loop import worker_loop  # noqa: E402
from app.domain.jobs.service import claim_next_job, complete_job, fail_job  # noqa: E402


async def _make_archive_item(session):
    """Helper: insert a minimal RawArchiveItem to satisfy FK constraint on jobs.raw_archive_id."""
    from app.domain.archive.models import RawArchiveItem
    import hashlib
    content = "test content"
    item = RawArchiveItem(
        id=uuid.uuid4(),
        source_type="test",
        raw_content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
    )
    session.add(item)
    await session.flush()
    return item


async def test_worker_claims_and_completes_job(db_session_phase2):
    """PIPE-01: Worker claims a pending job and transitions it to completed."""
    # Arrange: create a pending job
    from app.domain.jobs.models import Job
    archive_item = await _make_archive_item(db_session_phase2)
    job = Job(
        id=uuid.uuid4(),
        job_type="process_archive_item",
        raw_archive_id=archive_item.id,
        status="pending",
    )
    db_session_phase2.add(job)
    await db_session_phase2.commit()

    # Act: claim the job — poll until we get our job (other jobs may exist from prior tests)
    claimed = None
    for _ in range(20):  # at most 20 polls to drain prior-test jobs
        result = await claim_next_job(db_session_phase2)
        if result is None:
            break
        if result.id == job.id:
            claimed = result
            break

    # Assert: our specific job is claimed
    assert claimed is not None, "Our newly created job was never claimed"
    assert claimed.status == "claimed"
    assert claimed.id == job.id


async def test_job_retried_on_retryable_failed(db_session_phase2):
    """PIPE-04: Job in retryable_failed is re-claimed on next poll cycle."""
    from app.domain.jobs.models import Job
    archive_item = await _make_archive_item(db_session_phase2)
    job = Job(
        id=uuid.uuid4(),
        job_type="process_archive_item",
        raw_archive_id=archive_item.id,
        status="retryable_failed",
        attempts=1,
        max_attempts=3,
    )
    db_session_phase2.add(job)
    await db_session_phase2.commit()

    claimed = await claim_next_job(db_session_phase2)
    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.attempts == 2


async def test_job_not_retried_after_max_attempts(db_session_phase2):
    """PIPE-04: Job at max_attempts is not claimed (terminal failed)."""
    from app.domain.jobs.models import Job
    archive_item = await _make_archive_item(db_session_phase2)
    job = Job(
        id=uuid.uuid4(),
        job_type="process_archive_item",
        raw_archive_id=archive_item.id,
        status="retryable_failed",
        attempts=3,
        max_attempts=3,
    )
    db_session_phase2.add(job)
    await db_session_phase2.commit()

    claimed = await claim_next_job(db_session_phase2)
    # No job claimed — at max attempts
    assert claimed is None


async def test_stale_claimed_jobs_reset_on_startup(db_session_phase2):
    """PIPE-04: Stale claimed jobs (older than 10 min) are re-queued on worker startup."""
    from app.domain.jobs.service import reset_stale_jobs
    from app.domain.jobs.models import Job
    from datetime import datetime, timezone, timedelta

    archive_item = await _make_archive_item(db_session_phase2)
    stale_job = Job(
        id=uuid.uuid4(),
        job_type="process_archive_item",
        raw_archive_id=archive_item.id,
        status="claimed",
        attempts=1,
        claimed_at=datetime.now(timezone.utc) - timedelta(minutes=15),
    )
    db_session_phase2.add(stale_job)
    await db_session_phase2.commit()

    await reset_stale_jobs(db_session_phase2)
    await db_session_phase2.refresh(stale_job)

    assert stale_job.status == "pending"
    assert stale_job.attempts == 0


async def test_worker_loop_persists_status_transitions(db_session_phase2, monkeypatch):
    """Regression (F19): status transitions made during dispatch must persist.

    The loop must run claim_next_job and dispatch_job in the SAME session:
    complete_job mutates the claimed Job instance, which is only attached to
    the session that loaded it. With a second session the commit persists
    nothing and the job stays 'claimed' forever (then gets stale-reset and
    reprocessed on every restart).
    """
    import asyncio
    from datetime import datetime, timezone

    from sqlalchemy import select

    import app.worker.dispatcher as dispatcher_module
    from app.domain.jobs.models import Job
    from app.domain.jobs.service import complete_job

    archive_item = await _make_archive_item(db_session_phase2)
    job = Job(
        id=uuid.uuid4(),
        job_type="process_archive_item",
        raw_archive_id=archive_item.id,
        status="pending",
    )
    db_session_phase2.add(job)
    await db_session_phase2.commit()
    job_id = job.id

    async def fake_dispatch(session, claimed_job):
        # Mimic real dispatch: terminal transition via jobs.service helpers
        await complete_job(session, claimed_job)

    monkeypatch.setattr(dispatcher_module, "dispatch_job", fake_dispatch)

    loop_task = asyncio.create_task(worker_loop())
    try:
        deadline = asyncio.get_event_loop().time() + 15
        status = None
        while asyncio.get_event_loop().time() < deadline:
            # Read through an independent session — what the DB actually holds
            result = await db_session_phase2.execute(
                select(Job.status).where(Job.id == job_id)
            )
            db_session_phase2.expire_all()
            status = result.scalar_one_or_none()
            if status == "completed":
                break
            await asyncio.sleep(0.5)
    finally:
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

    assert status == "completed", (
        f"Job status in DB is {status!r} after dispatch — status transition "
        "was not persisted (claim and dispatch must share one session)"
    )


@pytest.mark.parametrize("mode", ["dispatch", "recovery_db", "success", "cache", "cancel", "system_exit", "import"])
async def test_dispatch_recovery_boundary_and_next_poll(db_session_phase2, test_engine, monkeypatch, caplog, mode):
    import asyncio
    import builtins
    from unittest.mock import AsyncMock
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from app.domain.jobs.models import Job
    from app.domain.jobs import service
    from app.infrastructure import db
    from app.worker import dispatcher
    from app.domain.retrieval import service as retrieval

    session = db_session_phase2
    archive = await _make_archive_item(session)
    first = Job(id=uuid.uuid4(), job_type="process_archive_item", raw_archive_id=archive.id,
                status="pending", max_attempts=3)
    second = Job(id=uuid.uuid4(), job_type="process_archive_item", raw_archive_id=archive.id,
                 status="pending")
    session.add_all([first, second])
    await session.commit()
    ids = (first.id, second.id)
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    monkeypatch.setattr(db, "get_session_factory", lambda: factory)
    reset = AsyncMock(wraps=service.reset_stale_jobs)
    recovery = AsyncMock(wraps=service.recover_claimed_job_after_dispatch_failure)
    if mode == "recovery_db":
        recovery.side_effect = RuntimeError("synthetic recovery DB failure")
    monkeypatch.setattr(service, "reset_stale_jobs", reset)
    monkeypatch.setattr(service, "recover_claimed_job_after_dispatch_failure", recovery)
    calls, sleeps = [], []

    async def dispatch(active, claimed):
        calls.append(claimed.id)
        if len(calls) == 1:
            if mode in {"dispatch", "recovery_db"}:
                raise ValueError("PRIVATE-CANARY" * 1000)
            if mode == "cancel":
                raise asyncio.CancelledError()
            if mode == "system_exit":
                raise SystemExit(7)
        await service.complete_job(active, claimed)

    async def sleep(seconds):
        sleeps.append(seconds)
        if mode != "dispatch" or seconds == 2:
            raise asyncio.CancelledError()

    monkeypatch.setattr(dispatcher, "dispatch_job", dispatch)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    invalidate = AsyncMock()
    if mode == "cache":
        invalidate.side_effect = RuntimeError("synthetic cache failure")
    monkeypatch.setattr(retrieval, "notify_cache_invalidation", invalidate)
    if mode == "import":
        original_import = builtins.__import__
        def fail_dispatch_import(name, *args, **kwargs):
            if name == "app.worker.dispatcher":
                raise ImportError("synthetic import failure")
            return original_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", fail_dispatch_import)

    with pytest.raises(SystemExit if mode == "system_exit" else asyncio.CancelledError):
        await worker_loop()
    reset.assert_awaited_once()
    if mode in {"dispatch", "recovery_db", "import"}:
        recovery.assert_awaited_once()
        assert sleeps[0] == 5
    else:
        recovery.assert_not_awaited()
    async with factory() as observer:
        saved = (await observer.execute(select(Job).where(Job.id == ids[0]))).scalar_one()
        if mode == "dispatch":
            assert calls == [ids[0], ids[0], ids[1]]
            assert saved.status == "completed" and saved.attempts == 2
            assert saved.error_message == "worker_dispatch_unhandled:ValueError"
            assert invalidate.await_count == 2
        elif mode in {"success", "cache"}:
            assert saved.status == "completed"
            assert invalidate.await_count >= 1
        elif mode == "import":
            assert saved.status == "retryable_failed" and saved.attempts == 1
            assert saved.error_message == "worker_dispatch_unhandled:ImportError"
        else:
            assert saved.status == "claimed" and saved.attempts == 1
            if mode == "recovery_db":
                assert "Claimed-job recovery failed" in caplog.text
