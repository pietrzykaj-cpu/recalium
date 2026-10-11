"""Jobs routes — POST /api/jobs/{job_id}/reprocess."""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.jobs.service import ReprocessConflict, reprocess_job
from app.infrastructure.db import get_session

logger = logging.getLogger(__name__)
router = APIRouter()


class ReprocessResponse(BaseModel):
    status: str
    job_id: str


@router.post("/{job_id}/reprocess", response_model=ReprocessResponse)
async def reprocess_job_endpoint(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> ReprocessResponse:
    """POST /api/jobs/{job_id}/reprocess — reset an eligible job to pending.

    PIPE-05: Allow failed, retryable_failed, or pending_provider jobs only.
    Resets attempts to 0 so job gets full max_attempts retries.
    Returns 404 if not found, or 409 for a currently ineligible status.
    Completed jobs stay blocked until generation-aware reprocessing exists.
    """
    try:
        job = await reprocess_job(session, job_id)
    except ReprocessConflict as exc:
        raise HTTPException(
            status_code=409, detail="Job cannot be reprocessed in its current state"
        ) from exc
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return ReprocessResponse(status=job.status, job_id=str(job.id))
