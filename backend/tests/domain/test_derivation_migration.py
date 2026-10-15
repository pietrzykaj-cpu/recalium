"""Additive migration executed in an isolated schema of the disposable DB only."""
import importlib.util
from pathlib import Path
import uuid
import pytest
from sqlalchemy import text
from alembic.migration import MigrationContext
from alembic.operations import Operations

@pytest.mark.asyncio
@pytest.mark.parametrize('seed_legacy',[False,True])
async def test_migration_chain_preserves_legacy_and_constraints(test_engine,seed_legacy):
    schema = "migration_" + uuid.uuid4().hex
    async with test_engine.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        await connection.execute(text(f'SET LOCAL search_path TO "{schema}", public'))
        def migrate(sync):
            context=MigrationContext.configure(sync)
            with Operations.context(context):
                paths=sorted(Path("migrations/versions").glob("*.py"))
                for path in paths:
                    spec=importlib.util.spec_from_file_location("migration_"+path.stem,path)
                    module=importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if module.revision=="0013" and seed_legacy:
                        aid,jid,fid,f1,f2 = [str(uuid.uuid4()) for _ in range(5)]
                        sync.execute(text("""INSERT INTO raw_archive(id,source_type,raw_content,content_hash)
                            VALUES(:id,'test','legacy alpha','honest-unattributed')"""),{"id":aid})
                        sync.execute(text("""INSERT INTO jobs(id,job_type,raw_archive_id,status)
                            VALUES(:id,'extract',:aid,'failed')"""),{"id":jid,"aid":aid})
                        sync.execute(text("""INSERT INTO facts(id,raw_archive_id,fact_text,source_span,
                            confidence_tier,derivation_method,derivation_model) VALUES
                            (:id,:aid,'legacy alpha','alpha','low','test','test')"""),{"id":fid,"aid":aid})
                        for rid in (f1,f2):
                            sync.execute(text("""INSERT INTO fts_entries(id,raw_archive_id,text_content)
                                VALUES(:id,:aid,'legacy alpha')"""),{"id":rid,"aid":aid})
                    module.upgrade()
                if seed_legacy:
                    assert sync.execute(text("SELECT id FROM facts")).scalar_one()==uuid.UUID(fid)
                    assert sync.execute(text("SELECT generation_id FROM facts")).scalar_one() is None
                else:
                    assert sync.execute(text("SELECT count(*) FROM facts")).scalar_one()==0
                    aid=str(uuid.uuid4())
                    sync.execute(text("""INSERT INTO raw_archive(id,source_type,raw_content,content_hash)
                        VALUES(:id,'test','empty-fixture alpha','honest-new-source')"""),{'id':aid})
                assert sync.execute(text("SELECT count(*) FROM fts_entries")).scalar_one()==(2 if seed_legacy else 0)
                assert sync.execute(text("SELECT count(*) FROM derivation_generations")).scalar_one()==0
                assert sync.execute(text("SELECT count(*) FROM archive_derivation_heads")).scalar_one()==0
                pid=str(uuid.uuid4())
                sync.execute(text("""INSERT INTO derivation_fts_projections(id,raw_archive_id,projection_kind,text_content)
                    VALUES(:id,:aid,'raw_fallback','alpha')"""),{"id":pid,"aid":aid})
                assert sync.execute(text("SELECT search_vector @@ to_tsquery('english','alpha') FROM derivation_fts_projections")).scalar_one()
                sync.execute(text("UPDATE derivation_fts_projections SET source_status='source_removed',text_content='[redacted:deleted]'"))
                assert sync.execute(text("SELECT search_vector FROM derivation_fts_projections")).scalar_one() is None
                indexes=set(sync.execute(text("SELECT indexname FROM pg_indexes WHERE schemaname=:schema"),{"schema":schema}).scalars())
                assert {"uq_generation_open","uq_generation_active","uq_fts_baseline","uq_fts_generation"} <= indexes
                assert int(sync.execute(text("SHOW server_version_num")).scalar_one()) >= 160000
        try:
            await connection.run_sync(migrate)
        finally:
            await connection.execute(text("SET LOCAL search_path TO public"))
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
