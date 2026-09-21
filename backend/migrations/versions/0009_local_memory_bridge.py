"""Add explicit local bridge identities, grants, assignments and receipts.

No existing archive is assigned or modified by this migration.
"""

from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""CREATE TABLE bridge_clients (
        id VARCHAR(64) PRIMARY KEY, credential_digest VARCHAR(64) NOT NULL UNIQUE,
        active BOOLEAN NOT NULL DEFAULT true)""")
    op.execute("CREATE TABLE bridge_projects (id VARCHAR(64) PRIMARY KEY)")
    op.execute("""CREATE TABLE bridge_grants (
        client_id VARCHAR(64) REFERENCES bridge_clients(id) NOT NULL,
        project_id VARCHAR(64) REFERENCES bridge_projects(id) NOT NULL,
        can_read BOOLEAN NOT NULL DEFAULT false, can_write BOOLEAN NOT NULL DEFAULT false,
        PRIMARY KEY (client_id, project_id))""")
    op.execute("""CREATE TABLE bridge_archives (
        archive_id UUID PRIMARY KEY REFERENCES raw_archive(id) ON DELETE CASCADE,
        project_id VARCHAR(64) NOT NULL REFERENCES bridge_projects(id),
        client_id VARCHAR(64) NOT NULL REFERENCES bridge_clients(id))""")
    op.execute("CREATE INDEX ix_bridge_archives_project_id ON bridge_archives(project_id)")
    op.execute("""CREATE TABLE bridge_receipts (
        client_id VARCHAR(64) NOT NULL REFERENCES bridge_clients(id),
        project_id VARCHAR(64) NOT NULL REFERENCES bridge_projects(id),
        request_digest VARCHAR(64) NOT NULL, payload_digest VARCHAR(64) NOT NULL,
        archive_id UUID NOT NULL REFERENCES raw_archive(id) ON DELETE CASCADE,
        PRIMARY KEY (client_id, project_id, request_digest))""")


def downgrade():
    for name in (
        "bridge_receipts",
        "bridge_archives",
        "bridge_grants",
        "bridge_projects",
        "bridge_clients",
    ):
        op.drop_table(name)
