"""Bridge ownership is explicit; legacy archive metadata grants no access."""

import uuid

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db import Base


class BridgeClient(Base):
    __tablename__ = "bridge_clients"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    credential_digest: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class BridgeProject(Base):
    __tablename__ = "bridge_projects"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(
        String(16), nullable=False, default="shared", server_default="shared"
    )
    owner_client_id: Mapped[str | None] = mapped_column(
        ForeignKey("bridge_clients.id"), nullable=True
    )
    __table_args__ = (
        CheckConstraint(
            "(kind = 'shared' AND owner_client_id IS NULL) OR (kind = 'private' AND owner_client_id IS NOT NULL)",
            name="ck_bridge_space_owner",
        ),
    )


class BridgeGrant(Base):
    __tablename__ = "bridge_grants"
    client_id: Mapped[str] = mapped_column(ForeignKey("bridge_clients.id"), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("bridge_projects.id"), primary_key=True)
    can_read: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    can_write: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class BridgeArchive(Base):
    __tablename__ = "bridge_archives"
    archive_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("raw_archive.id", ondelete="CASCADE"), primary_key=True
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey("bridge_projects.id"), nullable=False, index=True
    )
    client_id: Mapped[str] = mapped_column(ForeignKey("bridge_clients.id"), nullable=False)


class BridgeReceipt(Base):
    __tablename__ = "bridge_receipts"
    client_id: Mapped[str] = mapped_column(ForeignKey("bridge_clients.id"), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("bridge_projects.id"), primary_key=True)
    request_digest: Mapped[str] = mapped_column(String(64), primary_key=True)
    payload_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    archive_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("raw_archive.id", ondelete="CASCADE"), nullable=False
    )


class BridgeBinding(Base):
    """Operator-managed aliases, never client-controlled authority."""

    __tablename__ = "bridge_bindings"
    client_id: Mapped[str] = mapped_column(ForeignKey("bridge_clients.id"), primary_key=True)
    destination: Mapped[str] = mapped_column(String(16), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("bridge_projects.id"), nullable=False)
    __table_args__ = (
        CheckConstraint(
            "destination IN ('private', 'shared')", name="ck_bridge_binding_destination"
        ),
    )


class BridgeAliasReceipt(Base):
    """Pins an alias retry to the first committed resolution, even after rebinding."""

    __tablename__ = "bridge_alias_receipts"
    client_id: Mapped[str] = mapped_column(ForeignKey("bridge_clients.id"), primary_key=True)
    destination: Mapped[str] = mapped_column(String(16), primary_key=True)
    request_digest: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("bridge_projects.id"), nullable=False)
    __table_args__ = (
        CheckConstraint("destination IN ('private', 'shared')", name="ck_bridge_alias_destination"),
    )
