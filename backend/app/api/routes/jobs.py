"""Jobs routes — POST /api/jobs/{job_id}/reprocess."""
from __future__ import annotations

import logging
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.jobs.service import ReprocessConflict, reprocess_job
from app.infrastructure.db import get_session
from app.infrastructure.local_model import LocalModelNotReady

logger = logging.getLogger(__name__)
router = APIRouter()


class ReprocessResponse(BaseModel):
    status: str
    job_id: str
    generation_id: str | None = None


class ReprocessRequest(BaseModel):
    """An explicit NULL sentinel, never omitted/null-means-legacy."""
    model_config = ConfigDict(extra="forbid")
    expected_generation: Literal["legacy"] | uuid.UUID


@router.post("/{job_id}/reprocess", response_model=ReprocessResponse, response_model_exclude_none=True)
async def reprocess_job_endpoint(
    job_id: uuid.UUID,
    body: ReprocessRequest | None = None,
    session: AsyncSession = Depends(get_session),
) -> ReprocessResponse:
    """POST /api/jobs/{job_id}/reprocess — reset an eligible job to pending.

    PIPE-05: Existing no-body calls allow failed/retryable_failed/pending_provider.
    Resets attempts to 0 so job gets full max_attempts retries.
    Returns 404 if not found, or 409 for a currently ineligible status.
    Explicit expected_generation additionally permits completed new-generation CAS
    while R2 is ON, and fences optional failure-state requests by the same binding.
    Any explicit fence while OFF returns disabled 409, never a legacy reset.
    """
    try:
        job = await reprocess_job(session, job_id,
            expected_generation=body.expected_generation if body is not None else None)
    except LocalModelNotReady:
        raise HTTPException(status_code=503, detail="local_model_not_ready") from None
    except ReprocessConflict as exc:
        detail = "Job cannot be reprocessed in its current state"
        if exc.reason == "expected_generation_mismatch":
            detail = {"reason": exc.reason,
                "current_generation": str(exc.current_generation_id) if exc.current_generation_id is not None else "legacy",
                "current_status": exc.current_status}
        elif exc.reason == "generation_reprocess_disabled":
            detail = "Generation-aware reprocess is disabled"
        raise HTTPException(
            status_code=409, detail=detail
        ) from exc
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return ReprocessResponse(status=job.status, job_id=str(job.id),
        generation_id=str(job.generation_id) if body is not None and job.generation_id is not None else None)
