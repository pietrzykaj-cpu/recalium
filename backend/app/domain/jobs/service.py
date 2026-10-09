"""Jobs domain service — PostgreSQL job queue operations.

All status transitions happen here. Worker loop and API routes call these functions.
SECURITY: Never reads or stores API keys — keys are read from settings at dispatch time.
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone, timedelta
from typing import Literal

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.jobs.models import Job

logger = logging.getLogger(__name__)

# How long a claimed job can sit before being considered stale (worker crashed mid-job)
_STALE_THRESHOLD_MINUTES = 10


async def claim_next_job(session: AsyncSession) -> Job | None:
    """Atomically claim the next available job using SKIP LOCKED.

    Eligible: status in ('pending', 'retryable_failed') AND attempts < max_attempts.
    Returns None if no eligible job is available.

    CRITICAL: .with_for_update(skip_locked=True) ensures no double-claim race.
    The lock and status update happen in the same transaction.
    """
    stmt = (
        select(Job)
        .where(Job.status.in_(["pending", "retryable_failed"]))
        .where(Job.attempts < Job.max_attempts)
        .order_by(Job.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    result = await session.execute(stmt)
    job = result.scalar_one_or_none()

    if job is None:
        return None

    job.status = "claimed"
    job.attempts += 1
    job.claimed_at = datetime.now(timezone.utc)
    job.updated_at = datetime.now(timezone.utc)
    await session.commit()

    logger.debug("Claimed job %s (type=%s, attempts=%d)", job.id, job.job_type, job.attempts)
    return job


async def complete_job(session: AsyncSession, job: Job) -> None:
    """Mark job as completed successfully."""
    job.status = "completed"
    job.completed_at = datetime.now(timezone.utc)
    job.updated_at = datetime.now(timezone.utc)
    await session.commit()
    logger.debug("Completed job %s", job.id)


async def fail_job(
    session: AsyncSession,
    job: Job,
    error: str,
    retryable: bool = True,
) -> None:
    """Mark job as failed.

    If retryable=True AND attempts < max_attempts: status='retryable_failed' (will be retried).
    If retryable=False OR attempts >= max_attempts: status='failed' (terminal).

    BYOK-07: error_message is always set — no silent drops.
    """
    if retryable and job.attempts < job.max_attempts:
        job.status = "retryable_failed"
    else:
        job.status = "failed"

    job.error_message = error
    job.updated_at = datetime.now(timezone.utc)
    await session.commit()

    logger.warning(
        "Job %s failed (status=%s, attempts=%d/%d): %s",
        job.id, job.status, job.attempts, job.max_attempts, error[:200],
    )


async def recover_claimed_job_after_dispatch_failure(
    session: AsyncSession,
    job_id: uuid.UUID,
    attempts: int,
    claimed_at: datetime | None,
    exc: Exception,
) -> Literal["recovered", "missing", "claim_mismatch"]:
    """Fail only the escaped dispatch's claim; preserve its consumed attempt."""
    await session.rollback()
    result = await session.execute(
        select(Job)
        .where(Job.id == job_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    job = result.scalar_one_or_none()
    if job is None:
        await session.rollback()
        return "missing"
    if job.status != "claimed" or job.attempts != attempts or job.claimed_at != claimed_at:
        await session.rollback()
        return "claim_mismatch"

    # Match the dispatcher's bounded class convention without importing the
    # worker here: recovery must also work when the dispatcher import failed.
    name = type(exc).__name__
    name = name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) else "Exception"
    await fail_job(session, job, error=f"worker_dispatch_unhandled:{name}", retryable=True)
    return "recovered"


async def set_pending_provider(session: AsyncSession, job: Job, reason: str) -> None:
    """Mark job as pending_provider (blocked on missing API key — NOT a failure).

    BYOK-07/08: pending_provider is amber badge, not red. These jobs are re-activated
    when user adds a provider key in the settings route.
    """
    job.status = "pending_provider"
    job.error_message = reason
    job.updated_at = datetime.now(timezone.utc)
    await session.commit()
    logger.info("Job %s pending provider: %s", job.id, reason)


async def reset_stale_jobs(session: AsyncSession) -> int:
    """Re-queue jobs that were claimed but never completed (worker crashed mid-job).

    Called once on worker startup before polling begins.
    Jobs claimed more than STALE_THRESHOLD_MINUTES ago are reset to pending.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=_STALE_THRESHOLD_MINUTES)
    stmt = (
        update(Job)
        .where(Job.status == "claimed")
        .where(Job.claimed_at < cutoff)
        .values(
            status="pending",
            attempts=Job.attempts - 1,
            updated_at=datetime.now(timezone.utc),
        )
    )
    result = await session.execute(stmt)
    await session.commit()

    count = result.rowcount
    if count > 0:
        logger.warning("Reset %d stale claimed jobs to pending on worker startup", count)
    return count


async def reprocess_job(session: AsyncSession, job_id: uuid.UUID) -> Job | None:
    """Reset a failed/terminal job back to pending for manual re-queue.

    PIPE-05: Used by POST /api/jobs/{job_id}/reprocess endpoint.
    Resets attempts to 0 so it gets full max_attempts retries.
    """
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        return None

    job.status = "pending"
    job.attempts = 0
    job.error_message = None
    job.updated_at = datetime.now(timezone.utc)
    await session.commit()

    logger.info("Reprocessing job %s", job.id)
    return job


async def reactivate_pending_provider_jobs(session: AsyncSession) -> int:
    """Re-queue all pending_provider jobs when a new provider key is configured.

    Called from settings route when a key is saved/validated.
    Pitfall 9 prevention: jobs don't stay blocked forever when user adds a key.
    """
    stmt = (
        update(Job)
        .where(Job.status == "pending_provider")
        .values(
            status="pending",
            error_message=None,
            updated_at=datetime.now(timezone.utc),
        )
    )
    result = await session.execute(stmt)
    await session.commit()
    count = result.rowcount
    if count > 0:
        logger.info("Reactivated %d pending_provider jobs after key configuration", count)
    return count
