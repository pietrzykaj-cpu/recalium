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


SLICE2A_TABLES = ("conflict_detection_runs", "conflict_group_members")


def _slice2a_shape(sync, schema, table):
    sync.execute(text(f'SET LOCAL search_path TO "{schema}"'))
    columns = sync.execute(text("""SELECT column_name, data_type, character_maximum_length, is_nullable, column_default
        FROM information_schema.columns WHERE table_schema=:s AND table_name=:t ORDER BY column_name"""),
        {"s": schema, "t": table}).all()
    constraints = sync.execute(text("""SELECT c.conname, c.contype, pg_get_constraintdef(c.oid)
        FROM pg_constraint c JOIN pg_class r ON r.oid=c.conrelid JOIN pg_namespace n ON n.oid=r.relnamespace
        WHERE n.nspname=:s AND r.relname=:t ORDER BY c.conname"""), {"s": schema, "t": table}).all()
    indexes = sync.execute(text("""SELECT indexname, replace(indexdef, :prefix, '') FROM pg_indexes
        WHERE schemaname=:s AND tablename=:t ORDER BY indexname"""),
        {"s": schema, "t": table, "prefix": f"{schema}."}).all()
    return [tuple(r) for r in columns], [tuple(r) for r in constraints], [tuple(r) for r in indexes]


@pytest.mark.asyncio
@pytest.mark.parametrize('seed_legacy', [False, True])
async def test_slice2a_0014_objects_and_orm_parity(test_engine, seed_legacy):
    """Slice 2A: additive 0014 in an isolated schema; DDL must equal the ORM (test DB uses create_all)."""
    schema = "migration_" + uuid.uuid4().hex
    async with test_engine.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        await connection.execute(text(f'SET LOCAL search_path TO "{schema}", public'))
        def migrate(sync):
            context = MigrationContext.configure(sync)
            with Operations.context(context):
                for path in sorted(Path("migrations/versions").glob("*.py")):
                    spec = importlib.util.spec_from_file_location("migration_" + path.stem, path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if module.revision == "0014" and seed_legacy:
                        aid, fid, gid = [str(uuid.uuid4()) for _ in range(3)]
                        sync.execute(text("""INSERT INTO raw_archive(id,source_type,raw_content,content_hash)
                            VALUES(:id,'test','legacy alpha','honest-unattributed')"""), {"id": aid})
                        sync.execute(text("""INSERT INTO facts(id,raw_archive_id,fact_text,source_span,
                            confidence_tier,derivation_method,derivation_model) VALUES
                            (:id,:aid,'legacy alpha','alpha','low','test','test')"""), {"id": fid, "aid": aid})
                        sync.execute(text("INSERT INTO conflict_groups(id,group_type,source_status) "
                                          "VALUES(:id,'duplicate','active')"), {"id": gid})
                        sync.execute(text("UPDATE facts SET conflict_group_id=:g"), {"g": gid})
                    module.upgrade()
                    if module.revision == "0014":
                        with pytest.raises(RuntimeError):
                            module.downgrade()
            assert sync.execute(text("SELECT count(*) FROM conflict_detection_runs")).scalar_one() == 0
            assert sync.execute(text("SELECT count(*) FROM conflict_group_members")).scalar_one() == 0
            if seed_legacy:
                assert sync.execute(text("SELECT count(*) FROM facts WHERE conflict_group_id IS NOT NULL")).scalar_one() == 1
            migrated = {t: _slice2a_shape(sync, schema, t) for t in SLICE2A_TABLES}
            orm = {t: _slice2a_shape(sync, "public", t) for t in SLICE2A_TABLES}
            sync.execute(text(f'SET LOCAL search_path TO "{schema}", public'))
            names = {c[0] for t in SLICE2A_TABLES for c in migrated[t][1]} | {i[0] for t in SLICE2A_TABLES for i in migrated[t][2]}
            assert {"conflict_detection_runs_pkey", "fk_detection_generation_archive", "ck_detection_version",
                    "ck_detection_outcome", "ck_detection_grouped", "uq_detection_group",
                    "conflict_group_members_pkey", "ix_conflict_group_members_fact"} <= names
            for table in SLICE2A_TABLES:
                assert migrated[table] == orm[table], table
        try:
            await connection.run_sync(migrate)
        finally:
            await connection.execute(text("SET LOCAL search_path TO public"))
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
