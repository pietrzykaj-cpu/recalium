"""Slice 2A conflict/review replay identity: detection runs + immutable group membership. No backfill."""
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None

def upgrade():
    # Immutable migration SQL; do not import evolving application ORM metadata.
    op.execute("""CREATE TABLE conflict_detection_runs (
      generation_id UUID NOT NULL, raw_archive_id UUID NOT NULL, contract_version INTEGER NOT NULL,
      outcome VARCHAR(16) NOT NULL, conflict_group_id UUID REFERENCES conflict_groups(id),
      audit_event_id UUID REFERENCES audit_events(id), completed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY(generation_id, contract_version),
      CONSTRAINT fk_detection_generation_archive FOREIGN KEY(generation_id, raw_archive_id) REFERENCES derivation_generations(id, raw_archive_id),
      CONSTRAINT ck_detection_version CHECK(contract_version > 0),
      CONSTRAINT ck_detection_outcome CHECK(outcome IN ('grouped','no_candidates','nothing_to_link')),
      CONSTRAINT ck_detection_grouped CHECK((outcome = 'grouped') = (conflict_group_id IS NOT NULL AND audit_event_id IS NOT NULL))
    )""")
    op.execute("CREATE UNIQUE INDEX uq_detection_group ON conflict_detection_runs(conflict_group_id) WHERE conflict_group_id IS NOT NULL")
    op.execute("""CREATE TABLE conflict_group_members (
      conflict_group_id UUID NOT NULL REFERENCES conflict_groups(id), fact_id UUID NOT NULL REFERENCES facts(id),
      raw_archive_id UUID NOT NULL REFERENCES raw_archive(id), generation_id UUID,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY(conflict_group_id, fact_id)
    )""")
    op.execute("CREATE INDEX ix_conflict_group_members_fact ON conflict_group_members(fact_id)")

def downgrade():
    raise RuntimeError("Destructive conflict-identity history downgrade requires separate reviewed authorization")
