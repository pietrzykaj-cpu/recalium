"""Additive derivation identities; legacy rows are never attributed by migration."""
from datetime import datetime, timezone
import uuid

from sqlalchemy import CheckConstraint, Computed, ForeignKey, ForeignKeyConstraint, Index, Integer, String, Text, TIMESTAMP, UniqueConstraint, text, event, DDL, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column
from app.infrastructure.db import Base

def now():
    return datetime.now(timezone.utc)

def generation_fk(table):
    return ForeignKeyConstraint(
        ["generation_id", "raw_archive_id"],
        ["derivation_generations.id", "derivation_generations.raw_archive_id"],
        name=f"fk_{table}_generation_archive", use_alter=True,
    )

class DerivationGeneration(Base):
    __tablename__ = "derivation_generations"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    raw_archive_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("raw_archive.id"), nullable=False)
    original_job_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    creation_reason: Mapped[str] = mapped_column(String(32), nullable=False)
    contract_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    source_content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="building")
    predecessor_generation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, default=now)
    published_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    superseded_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    __table_args__ = (
        UniqueConstraint("id", "raw_archive_id", name="uq_generation_archive"),
        UniqueConstraint("raw_archive_id", "sequence", name="uq_generation_sequence"),
        ForeignKeyConstraint(["original_job_id", "raw_archive_id"], ["jobs.id", "jobs.raw_archive_id"], name="fk_generation_owner_archive", use_alter=True),
        ForeignKeyConstraint(["predecessor_generation_id", "raw_archive_id"], ["derivation_generations.id", "derivation_generations.raw_archive_id"], name="fk_generation_predecessor_archive"),
        CheckConstraint("sequence > 0 AND contract_version > 0", name="ck_generation_order"),
        CheckConstraint("creation_reason IN ('automatic_processing','manual_reprocess')", name="ck_generation_reason"),
        CheckConstraint("status IN ('building','prepared','active','superseded','failed','abandoned')", name="ck_generation_status"),
        Index("uq_generation_open", "raw_archive_id", unique=True, postgresql_where=text("status IN ('building','prepared')")),
        Index("uq_generation_active", "raw_archive_id", unique=True, postgresql_where=text("status = 'active'")),
    )

class ArchiveDerivationHead(Base):
    __tablename__ = "archive_derivation_heads"
    raw_archive_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("raw_archive.id"), primary_key=True)
    active_generation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    candidate_generation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    legacy_retired_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    __table_args__ = tuple(
        ForeignKeyConstraint([column, "raw_archive_id"], ["derivation_generations.id", "derivation_generations.raw_archive_id"], name=f"fk_head_{column}")
        for column in ("active_generation_id", "candidate_generation_id")
    )

class DerivationStageCompletion(Base):
    __tablename__ = "derivation_stage_completions"
    generation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("derivation_generations.id"), primary_key=True)
    stage_name: Mapped[str] = mapped_column(String(16), primary_key=True)
    outcome: Mapped[str] = mapped_column(String(16), nullable=False)
    fact_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completed_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, default=now)
    audit_event_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("audit_events.id"), nullable=True)
    failure_category: Mapped[str | None] = mapped_column(String(32), nullable=True)
    exception_class: Mapped[str | None] = mapped_column(String(64), nullable=True)
    __table_args__ = (
        CheckConstraint("stage_name IN ('summary','extraction','embedding')", name="ck_completion_stage"),
        CheckConstraint("outcome IN ('succeeded','empty','policy_skipped','partial_failure')", name="ck_completion_outcome"),
        CheckConstraint("(stage_name = 'extraction' AND fact_count IS NOT NULL AND fact_count >= 0 AND outcome IN ('succeeded','policy_skipped')) OR (stage_name <> 'extraction' AND fact_count IS NULL)", name="ck_completion_fact_count"),
        CheckConstraint("outcome <> 'partial_failure' OR (stage_name = 'summary' AND audit_event_id IS NOT NULL AND failure_category IS NOT NULL AND exception_class IS NOT NULL)", name="ck_completion_partial"),
    )

class DerivationFtsProjection(Base):
    __tablename__ = "derivation_fts_projections"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    raw_archive_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("raw_archive.id"), nullable=False)
    generation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    projection_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    text_content: Mapped[str] = mapped_column(Text, nullable=False)
    source_status: Mapped[str] = mapped_column(SAEnum('active','source_removed',name='source_status',create_type=False), nullable=False, default="active")
    search_vector: Mapped[str | None] = mapped_column(TSVECTOR, Computed("CASE WHEN source_status = 'active' THEN to_tsvector('english', text_content) ELSE NULL END", persisted=True))
    __table_args__ = (
        generation_fk("derivation_fts_projections"),
        CheckConstraint("projection_kind IN ('summary','raw_fallback')", name="ck_projection_kind"),
        CheckConstraint("source_status IN ('active','source_removed')", name="ck_projection_source"),
        CheckConstraint("generation_id IS NOT NULL OR projection_kind = 'raw_fallback'", name="ck_baseline_kind"),
        Index("uq_fts_baseline", "raw_archive_id", unique=True, postgresql_where=text("generation_id IS NULL")),
        Index("uq_fts_generation", "generation_id", unique=True, postgresql_where=text("generation_id IS NOT NULL")),
        Index("ix_derivation_fts_vector", "search_vector", postgresql_using="gin"),
    )

# The database backstop also applies to direct SQL and source/head cleanup.
# TRUNCATE/drop are test/admin operations, not ordinary lifecycle transitions.
event.listen(ArchiveDerivationHead.__table__, "after_create", DDL("""
CREATE OR REPLACE FUNCTION preserve_legacy_retirement() RETURNS trigger AS $$
BEGIN
  IF OLD.legacy_retired_at IS NOT NULL THEN
    IF TG_OP = 'DELETE' THEN
      RAISE EXCEPTION 'Retired derivation head cannot be deleted';
    END IF;
    IF NEW.legacy_retired_at IS DISTINCT FROM OLD.legacy_retired_at THEN
      RAISE EXCEPTION 'Legacy retirement is permanent';
    END IF;
  END IF;
  IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
  RETURN NEW;
END; $$ LANGUAGE plpgsql
""").execute_if(dialect="postgresql"))
event.listen(ArchiveDerivationHead.__table__, "after_create", DDL("""
CREATE TRIGGER keep_legacy_retired BEFORE UPDATE OR DELETE ON archive_derivation_heads
FOR EACH ROW EXECUTE FUNCTION preserve_legacy_retirement()
""").execute_if(dialect="postgresql"))
