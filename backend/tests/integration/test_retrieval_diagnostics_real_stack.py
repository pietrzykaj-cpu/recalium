"""Real PostgreSQL/pgvector validation for RetrievalDiagnostics v1.

Runs only against an explicitly supplied, disposable database whose name starts
with ``recalium_diag_``. It never uses the repository's normal test/production URL.
"""
from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime, timezone
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.domain.context_packets.service import build_context_packet
from app.domain.retrieval.diagnostics import RetrievalDiagnosticsCollector
from app.domain.retrieval.service import RetrievalFilters, RetrievalRequest, retrieve


DATABASE_URL = os.environ.get("RECALIUM_DIAGNOSTICS_TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="disposable diagnostics database not supplied")


def _guard_disposable_url() -> None:
    parsed = urlparse(DATABASE_URL.replace("postgresql+asyncpg", "postgresql"))
    assert parsed.hostname in {"127.0.0.1", "localhost"}
    assert parsed.path.lstrip("/").startswith("recalium_diag_")


SCHEMA = """
DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TYPE source_status AS ENUM ('active', 'source_removed');
CREATE TABLE raw_archive (
  id uuid PRIMARY KEY, source_type varchar(64) NOT NULL, source_name varchar(255),
  source_uri text, raw_content text NOT NULL, content_hash varchar(64) NOT NULL,
  conversation_count integer NOT NULL DEFAULT 1, ingested_at timestamptz NOT NULL,
  deleted_at timestamptz, metadata_json jsonb
);
CREATE TABLE bridge_archives (
  archive_id uuid NOT NULL REFERENCES raw_archive(id), project_id text NOT NULL,
  client_id text NOT NULL DEFAULT 'synthetic', PRIMARY KEY (archive_id, project_id)
);
CREATE TABLE fts_entries (
  id uuid PRIMARY KEY, raw_archive_id uuid NOT NULL REFERENCES raw_archive(id),
  text_content text NOT NULL, search_vector tsvector NOT NULL,
  source_status source_status NOT NULL DEFAULT 'active'
);
CREATE TABLE embeddings (
  id uuid PRIMARY KEY, raw_archive_id uuid NOT NULL REFERENCES raw_archive(id),
  embedding vector(3) NOT NULL, embedding_model text NOT NULL,
  source_status source_status NOT NULL DEFAULT 'active'
);
CREATE TABLE summaries (
  id uuid PRIMARY KEY, raw_archive_id uuid NOT NULL REFERENCES raw_archive(id),
  summary_text text NOT NULL, model_used varchar(128) NOT NULL,
  derivation_method varchar(64) NOT NULL, source_status source_status NOT NULL DEFAULT 'active',
  created_at timestamptz NOT NULL
);
CREATE TABLE canonical_memory (
  id uuid PRIMARY KEY, raw_archive_id uuid, fact_id uuid, content text NOT NULL,
  search_vector tsvector, source_status source_status NOT NULL DEFAULT 'active',
  status text NOT NULL DEFAULT 'active', created_at timestamptz NOT NULL
);
CREATE TABLE facts (
  id uuid PRIMARY KEY, raw_archive_id uuid NOT NULL REFERENCES raw_archive(id),
  fact_text text NOT NULL, source_span text NOT NULL, search_vector tsvector,
  source_status source_status NOT NULL DEFAULT 'active', review_status text NOT NULL DEFAULT 'active'
);
CREATE TABLE memory_links (
  id uuid PRIMARY KEY, source_fact_id uuid NOT NULL REFERENCES facts(id),
  target_fact_id uuid NOT NULL REFERENCES facts(id), link_type text NOT NULL,
  confidence double precision NOT NULL
);
"""


CORPUS = [
    # name, uuid suffix, text, vector, project, metadata
    ("lexical", "01", "continuity continuity continuity memory memory", "[0,1,0]", "space-a", {}),
    ("semantic", "02", "orchid vessel carries context", "[0.999,0.02,0]", "space-a", {}),
    ("both", "03", "continuity memory continuity", "[1,0,0]", "space-a", {}),
    ("near_duplicate", "04", "continuity memory", "[0.98,0.02,0]", "space-a", {}),
    ("conflict", "05", "continuity memory claim is disputed", "[0.80,0.20,0]", "space-a", {"conflict_label": "contradiction"}),
    ("unresolved", "06", "continuity memory open question", "[0.75,0.25,0]", "space-a", {"source_metadata": {"unresolved_questions": ["Which model succeeds it?"]}}),
    ("filtered", "07", "continuity continuity memory", "[1,0,0]", "space-b", {}),
    ("sql_hidden", "08", "continuity " + "separation " * 100 + "memory", "[-1,0,0]", "space-a", {}),
    ("long", "09", "continuity memory " + "long-evidence " * 100, "[0.55,0.45,0]", "space-a", {}),
]


class RetrievalSession:
    """Delegate SQL to the disposable DB while keeping audit writes in memory."""
    def __init__(self, session: AsyncSession): self.session = session
    async def execute(self, *args, **kwargs): return await self.session.execute(*args, **kwargs)
    def begin_nested(self): return self.session.begin_nested()
    def add(self, value): self.audit = value
    async def flush(self): pass


@pytest.mark.asyncio
async def test_real_fts_pgvector_diagnostics_and_context_packet(monkeypatch) -> None:
    _guard_disposable_url()
    engine = create_async_engine(DATABASE_URL)
    try:
        async with engine.begin() as connection:
            for statement in SCHEMA.split(";\n"):
                if statement.strip(): await connection.execute(text(statement))
            for index, (name, suffix, content, vector, project, metadata) in enumerate(CORPUS, start=1):
                archive_id = f"00000000-0000-0000-0000-{int(suffix):012d}"
                fts_id = f"10000000-0000-0000-0000-{int(suffix):012d}"
                embedding_id = f"20000000-0000-0000-0000-{int(suffix):012d}"
                summary_id = f"30000000-0000-0000-0000-{int(suffix):012d}"
                params = {"aid": archive_id, "fid": fts_id, "eid": embedding_id, "sid": summary_id,
                          "content": content, "metadata": __import__("json").dumps(metadata), "vector": vector,
                          "project": project, "captured": datetime(2026, 9, index, tzinfo=timezone.utc)}
                statements = [
                    "INSERT INTO raw_archive (id, source_type, raw_content, content_hash, ingested_at, metadata_json) VALUES (:aid, 'synthetic', :content, :fid, :captured, CAST(:metadata AS jsonb))",
                    "INSERT INTO bridge_archives VALUES (:aid, :project, 'synthetic')",
                    "INSERT INTO fts_entries VALUES (:fid, :aid, :content, to_tsvector('english', :content), 'active')",
                    "INSERT INTO embeddings VALUES (:eid, :aid, CAST(:vector AS vector), 'all-MiniLM-L6-v2', 'active')",
                    "INSERT INTO summaries VALUES (:sid, :aid, :content, 'synthetic-embedder', 'synthetic_summary', 'active', :captured)",
                ]
                for statement in statements: await connection.execute(text(statement), params)

        async def query_embedding(_: str) -> list[float]: return [1.0, 0.0, 0.0]
        monkeypatch.setattr("app.domain.derived_memory.service.embed_text", query_embedding)
        request = RetrievalRequest(
            query="continuity memory", mode="hybrid", budget=350, limit=20,
            filters=RetrievalFilters(bridge_project_ids=("space-a",)), actor="synthetic_test",
        )
        async with AsyncSession(engine) as sql_session:
            diagnostics = RetrievalDiagnosticsCollector(mode="hybrid")
            diagnosed = await retrieve(RetrievalSession(sql_session), request, diagnostics=diagnostics)
            snapshot = diagnostics.snapshot()

        # A second request without diagnostics must preserve the ordinary response contract/results.
        async with AsyncSession(engine) as sql_session:
            ordinary = await retrieve(RetrievalSession(sql_session), request)
        assert asdict(diagnosed) == asdict(ordinary)

        records = {record.representative_id: record for record in snapshot.candidates}
        both = next(record for record in records.values() if len(record.channels) == 2 and "000000000003" in record.fusion_key)
        lexical = next(record for record in records.values() if "000000000001" in record.fusion_key)
        semantic = next(record for record in records.values() if "000000000002" in record.fusion_key)
        assert both.lexical is not None and both.semantic is not None
        assert both.lexical.score > 0 and both.semantic.score > 0.99
        assert both.fused_score == pytest.approx(both.lexical.rrf_contribution + both.semantic.rrf_contribution)
        assert lexical.lexical.rank == 3 and lexical.semantic.rank > 5
        assert lexical.lexical.score > lexical.semantic.score
        assert semantic.channels == ["semantic"] and semantic.semantic.rank == 2
        assert [item.source_id[-12:] for item in diagnosed.items] == [
            "000000000003", "000000000004", "000000000006", "000000000005",
            "000000000001", "000000000002",
        ]
        assert snapshot.memory_space_ids == ["space-a"]
        assert snapshot.unknown_exclusions == []
        assert not any("000000000007" in record.fusion_key for record in snapshot.candidates)
        assert any(record.exclusion and record.exclusion.reason == "character_budget_exceeded" for record in snapshot.candidates)

        # Fill a deliberately small SQL window. Later rows are unknowable by
        # identity, so diagnostics report only a channel-level blind spot.
        monkeypatch.setattr("app.domain.retrieval.service.RRF_CANDIDATES_PER_MODE", 6)
        async with AsyncSession(engine) as sql_session:
            limited_diagnostics = RetrievalDiagnosticsCollector(mode="hybrid")
            await retrieve(RetrievalSession(sql_session), request, diagnostics=limited_diagnostics)
            limited_snapshot = limited_diagnostics.snapshot()
        assert {entry.stage for entry in limited_snapshot.unknown_exclusions} == {"lexical_sql_limit", "semantic_sql_limit"}
        assert all(entry.observable is False and entry.memory_id is None for entry in limited_snapshot.unknown_exclusions)
        assert len(limited_snapshot.candidates) < len(snapshot.candidates)
        monkeypatch.setattr("app.domain.retrieval.service.RRF_CANDIDATES_PER_MODE", 50)

        # Exercise the real post-retrieval bridge provenance enrichment before
        # ContextPacket assembly. This is where archive source metadata enters
        # the current deployed path.
        from app.domain.bridge.contracts import RetrieveInput
        from app.domain.bridge.service import _retrieve as bridge_retrieve
        async with AsyncSession(engine) as sql_session:
            enriched = await bridge_retrieve(
                RetrievalSession(sql_session), "synthetic_test",
                RetrieveInput(query="continuity memory", mode="hybrid", budget=350),
                {"space-a": SimpleNamespace(id="space-a", kind="shared")},
            )
        unresolved_item = next(item for item in enriched["items"] if item["source_id"].endswith("000000000006"))
        assert unresolved_item["provenance"]["source_metadata"]["unresolved_questions"] == ["Which model succeeds it?"]

        from app.domain.bridge.contracts import ContextPacketInput
        from app.domain.bridge.service import _context_packet as bridge_context_packet
        async with AsyncSession(engine) as sql_session:
            bridge_packet = await bridge_context_packet(
                RetrievalSession(sql_session), "synthetic_test",
                ContextPacketInput(query="continuity memory", mode="hybrid", budget=350, token_budget=80, include_diagnostics=True),
                {"space-a": SimpleNamespace(id="space-a", kind="shared")},
            )
        assert bridge_packet["unresolved_questions"] == ["Which model succeeds it?"]
        assert bridge_packet["retrieval_diagnostics"]["version"] == "retrieval-diagnostics-v1"
        # The current candidate SQL hard-codes direct conflict labels to null;
        # preserve this observed limitation rather than inventing a label.
        conflict_item = next(item for item in enriched["items"] if item["source_id"].endswith("000000000005"))
        assert conflict_item["conflict_label"] is None

        fixed_time = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
        packet_one = build_context_packet(diagnosed, diagnostics=snapshot, token_budget=80, generated_at=fixed_time)
        packet_two = build_context_packet(diagnosed, diagnostics=snapshot, token_budget=80, generated_at=fixed_time)
        assert packet_one.model_dump(mode="json") == packet_two.model_dump(mode="json")
        assert packet_one.integrity.packet_digest == packet_two.integrity.packet_digest
        assert any(item.retrieval.lexical_score is not None for item in packet_one.selected)
        assert any(item.retrieval.semantic_score is not None for item in packet_one.selected)
    finally:
        await engine.dispose()
