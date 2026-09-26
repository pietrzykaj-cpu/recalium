"""Add durable authority records and supersession edges.

No existing memory is promoted or backfilled; authority remains unknown until
an explicit authority record is created.
"""

from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""CREATE TABLE authority_records (
        id UUID PRIMARY KEY,
        space_id VARCHAR(128) NOT NULL,
        workstream_id VARCHAR(128) NOT NULL,
        authority_key VARCHAR(255) NOT NULL,
        record_kind VARCHAR(64) NOT NULL,
        content TEXT NOT NULL,
        lifecycle_status VARCHAR(16) NOT NULL
            CHECK (lifecycle_status IN ('proposed','active','withdrawn','disputed')),
        created_at TIMESTAMPTZ NOT NULL,
        created_by VARCHAR(128) NOT NULL,
        provenance JSON NOT NULL DEFAULT '{}'::json,
        withdrawal_reason TEXT NULL,
        dispute_reason TEXT NULL
    )""")
    op.execute(
        "CREATE INDEX ix_authority_records_scope "
        "ON authority_records(space_id, workstream_id, authority_key)"
    )
    op.execute("""CREATE TABLE authority_edges (
        successor_record_id UUID NOT NULL REFERENCES authority_records(id) ON DELETE CASCADE,
        predecessor_record_id UUID NOT NULL REFERENCES authority_records(id) ON DELETE CASCADE,
        edge_type VARCHAR(32) NOT NULL DEFAULT 'supersedes'
            CHECK (edge_type = 'supersedes'),
        created_at TIMESTAMPTZ NOT NULL,
        created_by VARCHAR(128) NOT NULL,
        reason TEXT NULL,
        provenance JSON NOT NULL DEFAULT '{}'::json,
        PRIMARY KEY (successor_record_id, predecessor_record_id, edge_type)
    )""")
    op.execute(
        "CREATE INDEX ix_authority_edges_predecessor "
        "ON authority_edges(predecessor_record_id)"
    )


def downgrade():
    op.drop_index("ix_authority_edges_predecessor", table_name="authority_edges")
    op.drop_table("authority_edges")
    op.drop_index("ix_authority_records_scope", table_name="authority_records")
    op.drop_table("authority_records")
