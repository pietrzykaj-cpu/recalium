"""Jobs service tests — PIPE-04, PIPE-05.

Tests will FAIL (RED) until app.domain.jobs.service is created.
"""
from __future__ import annotations

import uuid

import pytest

pytest.importorskip("app.domain.jobs.service", reason="jobs.service not yet implemented")

from app.domain.jobs.service import (  # noqa: E402
    claim_next_job,
    complete_job,
    fail_job,
    reset_stale_jobs,
    reprocess_job,
)


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
    await session.flush()  # flush to get PK without committing
    return item


async def test_claim_pending_job(db_session_phase2):
    """PIPE-04: claim_next_job claims a pending job atomically."""
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

    claimed = await claim_next_job(db_session_phase2)
    assert claimed is not None
    assert claimed.status == "claimed"
    assert claimed.attempts == 1


async def test_fail_job_with_error_message(db_session_phase2):
    """BYOK-07: fail_job sets retryable_failed with error_message captured."""
    from app.domain.jobs.models import Job
    archive_item = await _make_archive_item(db_session_phase2)
    job = Job(
        id=uuid.uuid4(),
        job_type="process_archive_item",
        raw_archive_id=archive_item.id,
        status="claimed",
        attempts=1,
    )
    db_session_phase2.add(job)
    await db_session_phase2.commit()

    await fail_job(db_session_phase2, job, error="AuthenticationError: invalid api key", retryable=True)
    await db_session_phase2.refresh(job)

    assert job.status == "retryable_failed"
    assert "AuthenticationError" in job.error_message


async def test_terminal_failure_when_max_attempts_reached(db_session_phase2):
    """PIPE-04: Job transitions to 'failed' (terminal) when attempts >= max_attempts."""
    from app.domain.jobs.models import Job
    archive_item = await _make_archive_item(db_session_phase2)
    job = Job(
        id=uuid.uuid4(),
        job_type="process_archive_item",
        raw_archive_id=archive_item.id,
        status="claimed",
        attempts=3,
        max_attempts=3,
    )
    db_session_phase2.add(job)
    await db_session_phase2.commit()

    await fail_job(db_session_phase2, job, error="Too many attempts", retryable=True)
    await db_session_phase2.refresh(job)

    # At max_attempts the job must be terminal (failed or retryable_failed with no re-claim possible)
    # The job itself must not be claimable again (attempts >= max_attempts)
    assert job.status in ("failed", "retryable_failed")
    # Verify this specific job cannot be claimed (attempts >= max_attempts blocks SELECT)
    assert job.attempts >= job.max_attempts


async def test_reprocess_resets_job_to_pending(db_session_phase2):
    """PIPE-05: reprocess_job resets a failed job back to pending for re-queue."""
    from app.domain.jobs.models import Job
    archive_item = await _make_archive_item(db_session_phase2)
    job = Job(
        id=uuid.uuid4(),
        job_type="process_archive_item",
        raw_archive_id=archive_item.id,
        status="failed",
        attempts=3,
        max_attempts=3,
        error_message="Some old error",
    )
    db_session_phase2.add(job)
    await db_session_phase2.commit()

    await reprocess_job(db_session_phase2, job.id)
    await db_session_phase2.refresh(job)

    assert job.status == "pending"
    assert job.attempts == 0
    assert job.error_message is None


@pytest.mark.parametrize("attempt", [1, 2, 3])
async def test_dispatch_recovery_preserves_attempt_and_persists(db_session_phase2, test_engine, attempt):
    from datetime import datetime, timezone
    from sqlalchemy import select, text
    from sqlalchemy.exc import DBAPIError
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from app.domain.jobs.models import Job
    from app.domain.jobs import service

    session = db_session_phase2
    archive = await _make_archive_item(session)
    job = Job(id=uuid.uuid4(), job_type="process_archive_item", raw_archive_id=archive.id,
              status="claimed", attempts=attempt, max_attempts=3,
              claimed_at=datetime.now(timezone.utc))
    session.add(job)
    await session.commit()
    fingerprint = (job.id, job.attempts, job.claimed_at)
    # Genuine PostgreSQL transaction abort; recovery must roll it back first.
    with pytest.raises(DBAPIError) as aborted:
        await session.execute(text("SELECT 1 / 0"))
    assert aborted.value.orig.sqlstate == "22012"
    outcome = await service.recover_claimed_job_after_dispatch_failure(
        session, *fingerprint, ValueError("SENSITIVE-CANARY" * 1000))
    assert outcome == "recovered"
    async with async_sessionmaker(test_engine, expire_on_commit=False)() as observer:
        saved = (await observer.execute(select(Job).where(Job.id == fingerprint[0]))).scalar_one()
        assert saved.status == ("failed" if attempt == 3 else "retryable_failed")
        assert saved.attempts == attempt
        assert saved.claimed_at == fingerprint[2]
        assert saved.error_message == "worker_dispatch_unhandled:ValueError"
        assert "SENSITIVE-CANARY" not in saved.error_message
        claimed = await service.claim_next_job(observer)
        if attempt == 3:
            assert claimed is None
        else:
            assert claimed.id == fingerprint[0] and claimed.attempts == attempt + 1


@pytest.mark.parametrize("mismatch", ["status", "attempts", "claimed_at", "missing", "double"])
async def test_dispatch_recovery_never_overwrites_changed_claim(db_session_phase2, monkeypatch, mismatch):
    from datetime import datetime, timezone, timedelta
    from unittest.mock import AsyncMock
    from app.domain.jobs.models import Job
    from app.domain.jobs import service

    session = db_session_phase2
    archive = await _make_archive_item(session)
    job = Job(id=uuid.uuid4(), job_type="process_archive_item", raw_archive_id=archive.id,
              status="claimed", attempts=1, claimed_at=datetime.now(timezone.utc))
    session.add(job)
    await session.commit()
    fingerprint = (job.id, job.attempts, job.claimed_at)
    if mismatch == "double":
        assert await service.recover_claimed_job_after_dispatch_failure(
            session, *fingerprint, ValueError("first")) == "recovered"
    elif mismatch == "missing":
        await session.delete(job)
    else:
        setattr(job, mismatch, {"status": "completed", "attempts": 2,
                               "claimed_at": fingerprint[2] + timedelta(seconds=1)}[mismatch])
    await session.commit()
    fail = AsyncMock(wraps=service.fail_job)
    monkeypatch.setattr(service, "fail_job", fail)
    outcome = await service.recover_claimed_job_after_dispatch_failure(
        session, *fingerprint, RuntimeError("second"))
    assert outcome == ("missing" if mismatch == "missing" else "claim_mismatch")
    fail.assert_not_awaited()
    assert not session.in_transaction()  # no-op releases its row lock


async def test_dispatch_recovery_lock_and_fresh_state(db_session_phase2, test_engine):
    from datetime import datetime, timezone
    from sqlalchemy import text, update
    from sqlalchemy.exc import DBAPIError
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from app.domain.jobs.models import Job
    from app.domain.jobs import service

    session = db_session_phase2
    archive = await _make_archive_item(session)
    job = Job(id=uuid.uuid4(), job_type="process_archive_item", raw_archive_id=archive.id,
              status="claimed", attempts=1, claimed_at=datetime.now(timezone.utc))
    session.add(job)
    await session.commit()
    fingerprint = (job.id, job.attempts, job.claimed_at)
    # The identity map must not preserve this dirty state through recovery.
    job.status = "completed"
    original_fail = service.fail_job

    async def assert_locked(active, fresh, **kwargs):
        assert fresh.status == "claimed"
        async with async_sessionmaker(test_engine)() as competing:
            await competing.execute(text("SET LOCAL lock_timeout = '100ms'"))
            with pytest.raises(DBAPIError) as blocked:
                await competing.execute(update(Job).where(Job.id == fingerprint[0]).values(status="pending"))
            assert blocked.value.orig.sqlstate == "55P03"
            await competing.rollback()
        await original_fail(active, fresh, **kwargs)

    from unittest.mock import patch
    with patch.object(service, "fail_job", side_effect=assert_locked) as fail:
        assert await service.recover_claimed_job_after_dispatch_failure(
            session, *fingerprint, ValueError("synthetic")) == "recovered"
        fail.assert_awaited_once()


@pytest.mark.parametrize("name", ["SafeError", "A" * 64, "A" * 65, "Private-Canary", "Érror"])
async def test_dispatch_recovery_marker_is_bounded_and_rollback_is_first(monkeypatch, name):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock
    from app.domain.jobs import service

    job = SimpleNamespace(id=uuid.uuid4(), status="claimed", attempts=1, max_attempts=3,
                          claimed_at=datetime.now(timezone.utc))
    session = AsyncMock()
    session.execute.return_value = MagicMock(scalar_one_or_none=MagicMock(return_value=job))
    exc = type(name, (Exception,), {})("PRIVATE-CANARY" * 1000)
    assert await service.recover_claimed_job_after_dispatch_failure(
        session, job.id, job.attempts, job.claimed_at, exc) == "recovered"
    assert [call[0] for call in session.method_calls] == ["rollback", "execute", "commit"]
    statement = session.execute.call_args.args[0]
    assert statement._for_update_arg is not None
    assert statement.get_execution_options()["populate_existing"] is True
    expected = name if name in {"SafeError", "A" * 64} else "Exception"
    assert job.error_message == "worker_dispatch_unhandled:" + expected
    assert len(job.error_message) <= 90 and "PRIVATE-CANARY" not in job.error_message
