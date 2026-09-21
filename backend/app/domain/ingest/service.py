"""Ingest domain service.

Sequence:
1. Parse payload (detect format, normalize)
2. Persist raw_archive item
3. Persist audit_event (synchronously)
4. Enqueue job stub (status="pending", type="pending_pipeline")
5. Return IngestResult

Atomicity: steps 2-4 are in a single DB transaction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.archive.models import RawArchiveItem
from app.domain.audit.models import AuditEvent
from app.domain.ingest.parsers import ParsedIngest, detect_and_parse
from app.domain.jobs.models import Job

logger = logging.getLogger(__name__)


class IdempotencyConflictError(ValueError):
    """Raised when an idempotency key is reused with different content."""


@dataclass
class IngestResult:
    item_count: int
    archive_ids: list[UUID]
    idempotent_replay: bool = False


async def ingest_text_content(
    session: AsyncSession,
    content: str,
    source_name: str | None = None,
    actor: str = "user_ui",
    source_type: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
    commit: bool = True,
) -> IngestResult:
    """Ingest plain text or JSON content (paste mode)."""
    content = content.strip()
    if len(content) < 10:
        raise ValueError(f"Content too short to ingest (got {len(content)} chars, need ≥ 10)")

    parsed = detect_and_parse(content=content, filename=None, source_name=source_name)
    if source_type:
        parsed.source_type = source_type
    parsed.metadata = {
        **(parsed.metadata or {}),
        **(extra_metadata or {}),
    } or None

    if idempotency_key:
        existing = (await session.execute(
            text("""
                SELECT id::text AS id, conversation_count, content_hash
                FROM raw_archive
                WHERE metadata_json ->> 'idempotency_key' = :idempotency_key
                  AND deleted_at IS NULL
                LIMIT 1
            """),
            {"idempotency_key": idempotency_key},
        )).mappings().first()
        if existing is not None:
            if existing["content_hash"] != parsed.content_hash:
                raise IdempotencyConflictError(
                    "idempotency key was already used with different content"
                )
            return IngestResult(
                item_count=int(existing["conversation_count"] or 1),
                archive_ids=[UUID(existing["id"])],
                idempotent_replay=True,
            )

    return await _persist_ingest(session=session, parsed=parsed, actor=actor, commit=commit)


async def ingest_file_content(
    session: AsyncSession,
    filename: str,
    content: str,
    source_name: str | None = None,
    actor: str = "user_ui",
) -> IngestResult:
    """Ingest content from an uploaded file (.json, .txt, .md)."""
    allowed_extensions = {".json", ".txt", ".md"}
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in allowed_extensions:
        raise ValueError(
            f"Unsupported file type: {filename!r}. "
            f"Allowed: {', '.join(sorted(allowed_extensions))}"
        )

    content = content.strip()
    if not content:
        raise ValueError(f"File {filename!r} is empty")

    parsed = detect_and_parse(
        content=content,
        filename=filename,
        source_name=source_name or filename,
    )
    return await _persist_ingest(session=session, parsed=parsed, actor=actor)


async def _persist_ingest(
    session: AsyncSession,
    parsed: ParsedIngest,
    actor: str,
    commit: bool = True,
) -> IngestResult:
    """Persist raw_archive item + audit_event + job stub in a single transaction."""
    archive_item = RawArchiveItem(
        source_type=parsed.source_type,
        source_name=parsed.source_name,
        raw_content=parsed.raw_content,
        content_hash=parsed.content_hash,
        conversation_count=parsed.conversation_count,
        metadata_json=parsed.metadata,
    )
    session.add(archive_item)
    await session.flush()  # Get archive_item.id without committing

    audit_event = AuditEvent(
        event_type="ingest",
        raw_archive_id=archive_item.id,
        actor=actor,
        operation_metadata={
            "source_type": parsed.source_type,
            "conversation_count": parsed.conversation_count,
            "content_hash": parsed.content_hash,
        },
    )
    session.add(audit_event)

    job = Job(
        job_type="pending_pipeline",
        raw_archive_id=archive_item.id,
        status="pending",
    )
    session.add(job)

    if commit:
        await session.commit()
    else:
        await session.flush()

    logger.info(
        f"Ingested: id={archive_item.id} type={parsed.source_type} "
        f"conversations={parsed.conversation_count}"
    )

    return IngestResult(
        item_count=parsed.conversation_count,
        archive_ids=[archive_item.id],
    )
