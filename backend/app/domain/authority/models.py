"""SQLAlchemy persistence models for authority records and supersession edges."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, ForeignKey, Index, JSON, String, Text, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db import Base


class AuthorityRecordRow(Base):
    __tablename__ = "authority_records"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    space_id: Mapped[str] = mapped_column(String(128), nullable=False)
    workstream_id: Mapped[str] = mapped_column(String(128), nullable=False)
    authority_key: Mapped[str] = mapped_column(String(255), nullable=False)
    record_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    lifecycle_status: Mapped[str] = mapped_column(String(16), nullable=False, default="proposed")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    provenance: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    withdrawal_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    dispute_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "lifecycle_status IN ('proposed','active','withdrawn','disputed')",
            name="ck_authority_record_lifecycle",
        ),
        Index(
            "ix_authority_records_scope",
            "space_id",
            "workstream_id",
            "authority_key",
        ),
    )


class AuthorityEdgeRow(Base):
    __tablename__ = "authority_edges"

    successor_record_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("authority_records.id", ondelete="CASCADE"),
        primary_key=True,
    )
    predecessor_record_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("authority_records.id", ondelete="CASCADE"),
        primary_key=True,
    )
    edge_type: Mapped[str] = mapped_column(String(32), primary_key=True, default="supersedes")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    provenance: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    __table_args__ = (
        CheckConstraint("edge_type = 'supersedes'", name="ck_authority_edge_type"),
        Index("ix_authority_edges_predecessor", "predecessor_record_id"),
    )
