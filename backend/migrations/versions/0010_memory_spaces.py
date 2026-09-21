"""Explicit space kinds, private owners, destination bindings and pinned retries.

Existing bridge projects remain shared; no archive assignment or memory is changed.
"""

from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE bridge_projects ADD COLUMN kind VARCHAR(16) NOT NULL DEFAULT 'shared'")
    op.execute(
        "ALTER TABLE bridge_projects ADD COLUMN owner_client_id VARCHAR(64) REFERENCES bridge_clients(id)"
    )
    op.execute(
        "ALTER TABLE bridge_projects ADD CONSTRAINT ck_bridge_space_owner CHECK ((kind='shared' AND owner_client_id IS NULL) OR (kind='private' AND owner_client_id IS NOT NULL))"
    )
    op.execute("""CREATE TABLE bridge_bindings (
        client_id VARCHAR(64) REFERENCES bridge_clients(id) NOT NULL,
        destination VARCHAR(16) NOT NULL CHECK (destination IN ('private','shared')),
        project_id VARCHAR(64) REFERENCES bridge_projects(id) NOT NULL,
        PRIMARY KEY(client_id,destination))""")
    op.execute("""CREATE TABLE bridge_alias_receipts (
        client_id VARCHAR(64) REFERENCES bridge_clients(id) NOT NULL,
        destination VARCHAR(16) NOT NULL CHECK (destination IN ('private','shared')),
        request_digest VARCHAR(64) NOT NULL,
        project_id VARCHAR(64) REFERENCES bridge_projects(id) NOT NULL,
        PRIMARY KEY(client_id,destination,request_digest))""")


def downgrade():
    op.drop_table("bridge_alias_receipts")
    op.drop_table("bridge_bindings")
    op.execute("ALTER TABLE bridge_projects DROP CONSTRAINT ck_bridge_space_owner")
    op.drop_column("bridge_projects", "owner_client_id")
    op.drop_column("bridge_projects", "kind")
