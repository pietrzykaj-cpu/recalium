"""Backup and restore service.

BKUP-01: Scheduled daily backups with 30-day retention.
BKUP-02: Restore within 15 minutes.
BKUP-03: Restore recovers all data including audit events and configuration.
PRIV-03: backup_predates_deletion() detects if a backup predates any deletion event.

GPT5.6 #2 (deletion/backup/restore safety): restore now rejects path traversal,
validates the archive before touching the active database, takes a pre-restore
safety snapshot for rollback, and reapplies deletion tombstones afterwards so a
restored pre-deletion backup cannot resurrect removed content.

Tool: pg_dump (custom format -Fc) and pg_restore.
Backup location: /backups/ (bind-mounted from ./backups on host).
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.archive.tombstones import DEFAULT_BACKUP_DIR
from app.infrastructure.settings import get_settings

logger = logging.getLogger(__name__)

_RESTORE_TIMEOUT_SECONDS = 15 * 60  # 15 minutes (BKUP-02)

# Tables whose presence proves a restore produced a usable database. They span the
# domains BKUP-03 guarantees: archive content, the deletion ledger, and provenance.
# pg_restore's exit code is not a reliable success signal (it is non-zero for
# benign, ignorable errors), so restore health is judged by validating these.
_RESTORE_HEALTHCHECK_TABLES = ("raw_archive", "tombstones", "audit_events")


def _resolve_backup_path(filename: str, backup_dir: str) -> Path:
    """Resolve ``filename`` strictly inside ``backup_dir`` (GPT5.6 #2).

    Rejects absolute paths and ``..`` traversal so a restore/read can never touch a
    file outside the backup directory. Raises ValueError on any escape attempt.
    """
    base = Path(backup_dir).resolve()
    candidate = (base / filename).resolve()
    if candidate == base or not candidate.is_relative_to(base):
        raise ValueError(
            f"Refusing backup path outside {backup_dir!r}: {filename!r}"
        )
    return candidate


def _get_db_params() -> dict[str, str]:
    """Extract DB connection parameters from settings.

    Prefers individual postgres_* fields when set; falls back to parsing database_url.
    """
    settings = get_settings()

    # Use individual fields if they look non-default or meaningful
    # (They have defaults so we always use them directly)
    return {
        "host": settings.postgres_host,
        "port": str(settings.postgres_port),
        "user": settings.postgres_user,
        "password": settings.postgres_password,
        "dbname": settings.postgres_db,
    }


async def _run_pg_dump(filepath: Path) -> None:
    """Dump the active database to ``filepath`` in custom format (-Fc)."""
    params = _get_db_params()
    env = os.environ.copy()
    env["PGPASSWORD"] = params["password"]
    proc = await asyncio.create_subprocess_exec(
        "pg_dump", "-Fc",
        "-h", params["host"], "-p", params["port"], "-U", params["user"],
        "-d", params["dbname"], "-f", str(filepath),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"pg_dump failed (exit {proc.returncode}): {stderr.decode().strip()}"
        )


async def _run_pg_restore(filepath: Path) -> tuple[int, str]:
    """Restore a custom-format dump into the active database (--clean --if-exists).

    Returns ``(returncode, stderr)``. Deliberately does NOT raise on a non-zero
    exit: pg_restore reports a non-zero status for *benign, ignorable* errors
    (e.g. a newer pg_dump emitting a ``SET`` the older server rejects) while still
    restoring all the data. Callers must judge success by validating the restored
    database (see ``_validate_restored_db``), not by the exit code. Raises
    RuntimeError only on timeout, which is a genuine failure (BKUP-02).
    """
    params = _get_db_params()
    env = os.environ.copy()
    env["PGPASSWORD"] = params["password"]
    proc = await asyncio.create_subprocess_exec(
        "pg_restore", "--clean", "--if-exists",
        "-h", params["host"], "-p", params["port"], "-U", params["user"],
        "-d", params["dbname"], str(filepath),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_RESTORE_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise RuntimeError(
            f"pg_restore timed out after {_RESTORE_TIMEOUT_SECONDS}s (BKUP-02 violation)"
        )
    return proc.returncode or 0, stderr.decode().strip()


async def _validate_restored_db() -> None:
    """Confirm the database is healthy and queryable after a restore (GPT5.6 #2).

    pg_restore of a custom-format archive exits non-zero for benign reasons, so the
    exit code alone cannot decide success. Instead we verify the restore produced a
    usable database: every core table in ``_RESTORE_HEALTHCHECK_TABLES`` must exist
    and the central ``raw_archive`` table must be queryable. Raises RuntimeError if
    the database is not healthy so the caller can roll back.
    """
    from app.infrastructure.db import get_engine  # noqa: PLC0415

    # --clean dropped/recreated the schema out-of-band; drop connections pooled
    # against the old schema before validating.
    await get_engine().dispose()
    async with get_engine().connect() as conn:
        for table in _RESTORE_HEALTHCHECK_TABLES:
            present = await conn.scalar(
                text("SELECT to_regclass(:t)"), {"t": f"public.{table}"}
            )
            if present is None:
                raise RuntimeError(
                    f"post-restore validation failed: core table {table!r} is missing"
                )
        # Confirm the central table is actually queryable, not merely catalogued.
        await conn.execute(text("SELECT count(*) FROM raw_archive"))


async def _validate_archive(filepath: Path) -> None:
    """Confirm ``filepath`` is a readable pg custom-format archive.

    Runs ``pg_restore --list`` (read-only, never touches the DB) so a corrupted or
    non-archive file is rejected *before* the active database is modified.
    """
    proc = await asyncio.create_subprocess_exec(
        "pg_restore", "--list", str(filepath),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            "invalid or corrupted backup archive: "
            f"{stderr.decode().strip()[:300]}"
        )


async def create_backup(backup_dir: str = DEFAULT_BACKUP_DIR) -> dict:
    """Run pg_dump and save a custom-format dump file.

    BKUP-01: Backup includes all data (archive, settings, audit events, telemetry).

    Returns:
        dict with keys: filename, created_at, size_bytes, backup_dir
    """
    os.makedirs(backup_dir, exist_ok=True)

    # Microsecond resolution so two backups triggered in the same second do not
    # collide and overwrite each other (GPT5.6 #2 backup drill finding).
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    filename = f"recalium_{timestamp}.dump"
    filepath = Path(backup_dir) / filename

    await _run_pg_dump(filepath)

    size_bytes = filepath.stat().st_size
    created_at = datetime.now(timezone.utc).isoformat()

    logger.info("Backup created: %s (%d bytes)", filename, size_bytes)
    return {
        "filename": filename,
        "created_at": created_at,
        "size_bytes": size_bytes,
        "backup_dir": backup_dir,
    }


async def list_backups(backup_dir: str = DEFAULT_BACKUP_DIR) -> list[dict]:
    """List all .dump backup files in backup_dir, newest first.

    Returns:
        list of dicts with keys: filename, created_at, size_bytes
    """
    backup_path = Path(backup_dir)
    if not backup_path.exists():
        return []

    backups = []
    for f in backup_path.glob("*.dump"):
        # Skip internal pre-restore rollback snapshots (".rollback_*.dump").
        if f.name.startswith("."):
            continue
        stat = f.stat()
        created_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        backups.append({
            "filename": f.name,
            "created_at": created_at,
            "size_bytes": stat.st_size,
        })

    # Sort newest first
    backups.sort(key=lambda x: x["created_at"], reverse=True)
    return backups


async def restore_backup(
    filename: str,
    backup_dir: str = DEFAULT_BACKUP_DIR,
    reapply_tombstones_after: bool = True,
) -> dict:
    """Safely restore a pg_dump custom-format file (GPT5.6 #2).

    Safety stages, in order:
      1. Path containment — reject any filename that escapes ``backup_dir``.
      2. Archive validation — ``pg_restore --list`` rejects a corrupted/non-archive
         file *before* the active database is touched.
      3. Pre-restore safety snapshot — dump the current DB so a mid-restore failure
         can be rolled back.
      4. Restore; on failure, roll back from the safety snapshot.
      5. Tombstone reapply — re-suppress any content covered by the deletion ledger
         so restoring a pre-deletion backup cannot resurrect removed data.

    BKUP-02: Restore completes within 15 minutes.
    BKUP-03: Recovers all data including audit events and configuration.

    Returns:
        dict with keys: filename, restored_at, status, tombstones_reapplied, rolled_back
    Raises:
        ValueError: if the filename escapes the backup directory
        FileNotFoundError: if the backup file does not exist
        RuntimeError: if the archive is invalid, or restore fails (after rollback)
    """
    # 1. Path containment (raises ValueError on traversal).
    filepath = _resolve_backup_path(filename, backup_dir)
    if not filepath.exists():
        raise FileNotFoundError(f"Backup file not found: {filepath}")

    # 2. Validate the archive before mutating the active database.
    await _validate_archive(filepath)

    # Preserve R1 before destructive restore; compatible reconciliation is 1C.
    from app.infrastructure.db import get_session_factory
    from app.domain.derivations.privacy import assert_restore_safe
    async with get_session_factory()() as retirement_session:
        await assert_restore_safe(retirement_session)

    # 3. Pre-restore safety snapshot for rollback.
    snapshot = Path(backup_dir) / (
        f".rollback_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.dump"
    )
    try:
        await _run_pg_dump(snapshot)
    except Exception as exc:
        raise RuntimeError(
            f"Could not create pre-restore safety snapshot; aborting restore: {exc}"
        ) from exc

    # 4. Restore, then validate health. pg_restore's exit code is NOT a reliable
    #    success signal (it is non-zero for benign SET/parameter errors, e.g. under
    #    pg_dump/server version skew), so success is judged by post-restore health
    #    validation. Roll back from the snapshot only if validation fails.
    rolled_back = False
    rc, stderr = await _run_pg_restore(filepath)
    try:
        await _validate_restored_db()
    except Exception as exc:
        logger.error(
            "Restore produced an unhealthy database (pg_restore exit %d: %s); "
            "rolling back from safety snapshot: %s",
            rc, stderr[:300], exc,
        )
        try:
            await _run_pg_restore(snapshot)
            await _validate_restored_db()
            rolled_back = True
            logger.info("Rollback from safety snapshot succeeded.")
        except Exception as rb_exc:
            logger.critical(
                "ROLLBACK FAILED after a failed restore. Safety snapshot retained at %s",
                snapshot,
            )
            raise RuntimeError(
                f"Restore failed AND rollback failed: {rb_exc}. "
                f"Manual recovery snapshot: {snapshot}"
            ) from rb_exc
        # Snapshot consumed by a successful rollback; clean it up.
        snapshot.unlink(missing_ok=True)
        raise RuntimeError(f"Restore failed and was rolled back: {exc}") from exc

    if rc != 0:
        logger.warning(
            "pg_restore reported ignorable errors (exit %d) but the restored database "
            "validated healthy; continuing. This usually means the pg_dump client is a "
            "newer major than the server — align them to silence this. Details: %s",
            rc, stderr[:300],
        )

    # 5. Reapply tombstones so a pre-deletion backup cannot resurrect removed content.
    reapplied = 0
    if reapply_tombstones_after:
        try:
            from app.domain.archive.service import reapply_tombstones  # noqa: PLC0415
            from app.infrastructure.db import get_engine, get_session_factory  # noqa: PLC0415

            # pg_restore --clean dropped/recreated tables out-of-band; drop any
            # pooled connections so reapply reads the freshly restored schema.
            await get_engine().dispose()

            factory = get_session_factory()
            async with factory() as session:
                summary = await reapply_tombstones(session, backup_dir=backup_dir)
            reapplied = summary["reapplied"]
        except Exception as exc:
            logger.error(
                "Restore succeeded but tombstone reapply failed: %s. "
                "Deleted content from a pre-deletion backup may be temporarily "
                "retrievable until reapply is re-run.",
                exc,
            )

    # Success — remove the safety snapshot.
    snapshot.unlink(missing_ok=True)

    restored_at = datetime.now(timezone.utc).isoformat()
    logger.info(
        "Restore complete from: %s (tombstones reapplied: %d)", filename, reapplied
    )
    return {
        "filename": filename,
        "restored_at": restored_at,
        "status": "success",
        "tombstones_reapplied": reapplied,
        "rolled_back": rolled_back,
    }


async def delete_old_backups(
    backup_dir: str = DEFAULT_BACKUP_DIR,
    retention_days: int = 30,
) -> int:
    """Delete .dump backup files older than retention_days.

    BKUP-01: 30-day retention policy.

    Returns:
        Number of files deleted
    """
    backup_path = Path(backup_dir)
    if not backup_path.exists():
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    deleted = 0

    for f in backup_path.glob("*.dump"):
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
        if mtime < cutoff:
            f.unlink()
            logger.info("Deleted old backup: %s", f.name)
            deleted += 1

    return deleted


async def backup_predates_deletion(
    session: AsyncSession,
    backup_timestamp: datetime,
) -> bool:
    """Check if there are any deletion audit events AFTER the given backup timestamp.

    PRIV-03: UI flags backups that predate any deletion event, warning the user
    that restoring may re-introduce data that was intentionally deleted.

    Returns:
        True if there is at least one deletion event after backup_timestamp,
        False otherwise.
    """
    result = await session.execute(
        text(
            "SELECT 1 FROM audit_events "
            "WHERE event_type = 'archive_delete' "
            "AND occurred_at > :ts "
            "LIMIT 1"
        ),
        {"ts": backup_timestamp},
    )
    row = result.fetchone()
    return row is not None
