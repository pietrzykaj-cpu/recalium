"""Dormant derivation identities and one generation-keyed FTS source. No backfill."""
from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None

def upgrade():
    # Immutable migration SQL; do not import evolving application ORM metadata.
    op.execute("ALTER TABLE jobs ADD CONSTRAINT uq_job_archive UNIQUE(id, raw_archive_id)")
    op.execute("""CREATE TABLE derivation_generations (
      id UUID PRIMARY KEY, raw_archive_id UUID NOT NULL REFERENCES raw_archive(id),
      original_job_id UUID NOT NULL, sequence INTEGER NOT NULL, creation_reason VARCHAR(32) NOT NULL,
      contract_version INTEGER NOT NULL DEFAULT 1, source_content_hash VARCHAR(64) NOT NULL,
      status VARCHAR(16) NOT NULL DEFAULT 'building', predecessor_generation_id UUID,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(), published_at TIMESTAMPTZ, superseded_at TIMESTAMPTZ,
      CONSTRAINT uq_generation_archive UNIQUE(id,raw_archive_id),
      CONSTRAINT uq_generation_sequence UNIQUE(raw_archive_id,sequence),
      CONSTRAINT fk_generation_owner_archive FOREIGN KEY(original_job_id,raw_archive_id) REFERENCES jobs(id,raw_archive_id),
      CONSTRAINT fk_generation_predecessor_archive FOREIGN KEY(predecessor_generation_id,raw_archive_id) REFERENCES derivation_generations(id,raw_archive_id),
      CONSTRAINT ck_generation_order CHECK(sequence > 0 AND contract_version > 0),
      CONSTRAINT ck_generation_reason CHECK(creation_reason IN ('automatic_processing','manual_reprocess')),
      CONSTRAINT ck_generation_status CHECK(status IN ('building','prepared','active','superseded','failed','abandoned'))
    )""")
    op.execute("CREATE UNIQUE INDEX uq_generation_open ON derivation_generations(raw_archive_id) WHERE status IN ('building','prepared')")
    op.execute("CREATE UNIQUE INDEX uq_generation_active ON derivation_generations(raw_archive_id) WHERE status = 'active'")
    op.execute("""CREATE TABLE archive_derivation_heads (
      raw_archive_id UUID PRIMARY KEY REFERENCES raw_archive(id),
      active_generation_id UUID, candidate_generation_id UUID, legacy_retired_at TIMESTAMPTZ,
      CONSTRAINT fk_head_active_generation_id FOREIGN KEY(active_generation_id,raw_archive_id) REFERENCES derivation_generations(id,raw_archive_id),
      CONSTRAINT fk_head_candidate_generation_id FOREIGN KEY(candidate_generation_id,raw_archive_id) REFERENCES derivation_generations(id,raw_archive_id)
    )""")
    op.execute("""CREATE OR REPLACE FUNCTION preserve_legacy_retirement() RETURNS trigger AS $$
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
END; $$ LANGUAGE plpgsql""")
    op.execute("""CREATE TRIGGER keep_legacy_retired BEFORE UPDATE OR DELETE ON archive_derivation_heads
      FOR EACH ROW EXECUTE FUNCTION preserve_legacy_retirement()""")
    op.execute("""CREATE TABLE derivation_stage_completions (
      generation_id UUID REFERENCES derivation_generations(id), stage_name VARCHAR(16),
      outcome VARCHAR(16) NOT NULL, fact_count INTEGER, completed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      audit_event_id UUID REFERENCES audit_events(id), failure_category VARCHAR(32), exception_class VARCHAR(64),
      PRIMARY KEY(generation_id,stage_name),
      CONSTRAINT ck_completion_stage CHECK(stage_name IN ('summary','extraction','embedding')),
      CONSTRAINT ck_completion_outcome CHECK(outcome IN ('succeeded','empty','policy_skipped','partial_failure')),
      CONSTRAINT ck_completion_fact_count CHECK((stage_name='extraction' AND fact_count IS NOT NULL AND fact_count >= 0 AND outcome IN ('succeeded','policy_skipped')) OR (stage_name<>'extraction' AND fact_count IS NULL)),
      CONSTRAINT ck_completion_partial CHECK(outcome<>'partial_failure' OR (stage_name='summary' AND audit_event_id IS NOT NULL AND failure_category IS NOT NULL AND exception_class IS NOT NULL))
    )""")
    op.execute("""CREATE TABLE derivation_fts_projections (
      id UUID PRIMARY KEY, raw_archive_id UUID NOT NULL REFERENCES raw_archive(id), generation_id UUID,
      projection_kind VARCHAR(16) NOT NULL, text_content TEXT NOT NULL,
      source_status source_status NOT NULL DEFAULT 'active',
      search_vector TSVECTOR GENERATED ALWAYS AS (
        CASE WHEN source_status='active' THEN to_tsvector('english',text_content) ELSE NULL END
      ) STORED,
      CONSTRAINT fk_derivation_fts_projections_generation_archive FOREIGN KEY(generation_id,raw_archive_id) REFERENCES derivation_generations(id,raw_archive_id),
      CONSTRAINT ck_projection_kind CHECK(projection_kind IN ('summary','raw_fallback')),
      CONSTRAINT ck_projection_source CHECK(source_status IN ('active','source_removed')),
      CONSTRAINT ck_baseline_kind CHECK(generation_id IS NOT NULL OR projection_kind='raw_fallback')
    )""")
    op.execute("CREATE UNIQUE INDEX uq_fts_baseline ON derivation_fts_projections(raw_archive_id) WHERE generation_id IS NULL")
    op.execute("CREATE UNIQUE INDEX uq_fts_generation ON derivation_fts_projections(generation_id) WHERE generation_id IS NOT NULL")
    op.execute("CREATE INDEX ix_derivation_fts_vector ON derivation_fts_projections USING gin(search_vector)")
    for table in ("jobs","summaries","facts","embeddings"):
        op.execute(f"ALTER TABLE {table} ADD COLUMN generation_id UUID")
        op.execute(f"ALTER TABLE {table} ADD CONSTRAINT fk_{table}_generation_archive FOREIGN KEY(generation_id,raw_archive_id) REFERENCES derivation_generations(id,raw_archive_id)")
        op.execute(f"CREATE INDEX ix_{table}_generation_id ON {table}(generation_id)")
    for table, prefix in (("summaries","summary"),("embeddings","embedding")):
        op.execute(f"ALTER TABLE {table} ADD CONSTRAINT uq_{prefix}_generation UNIQUE(generation_id)")

def downgrade():
    raise RuntimeError("Destructive generation-history downgrade requires separate reviewed authorization")
