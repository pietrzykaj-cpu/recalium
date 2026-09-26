"""Add guarded authority-write capabilities and idempotency receipts.

Existing grants retain their ordinary permissions but gain no authority-write
capability. Existing authority records and edges are not changed.
"""

from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "ALTER TABLE bridge_grants ADD COLUMN "
        "can_propose_authority BOOLEAN NOT NULL DEFAULT false"
    )
    op.execute(
        "ALTER TABLE bridge_grants ADD COLUMN "
        "can_mutate_authority BOOLEAN NOT NULL DEFAULT false"
    )
    op.execute("""CREATE TABLE authority_mutation_receipts (
        client_id VARCHAR(64) NOT NULL REFERENCES bridge_clients(id),
        request_digest VARCHAR(64) NOT NULL,
        payload_digest VARCHAR(64) NOT NULL,
        operation_id UUID NOT NULL,
        operation_type VARCHAR(64) NOT NULL,
        result_json JSON NOT NULL,
        audit_event_id UUID NOT NULL REFERENCES audit_events(id),
        created_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (client_id, request_digest)
    )""")
    op.execute(
        "CREATE UNIQUE INDEX ix_authority_mutation_receipts_operation "
        "ON authority_mutation_receipts(operation_id)"
    )


def downgrade():
    op.drop_index(
        "ix_authority_mutation_receipts_operation",
        table_name="authority_mutation_receipts",
    )
    op.drop_table("authority_mutation_receipts")
    op.drop_column("bridge_grants", "can_mutate_authority")
    op.drop_column("bridge_grants", "can_propose_authority")
