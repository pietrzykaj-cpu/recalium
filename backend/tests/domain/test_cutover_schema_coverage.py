"""Test-only review guard for the cutover tooling (O1, cutover schema-head micro-slice).

The cutover schema check compares the exact DB/code Alembic head. It no longer pins one
reviewed schema, so this test carries that review pressure instead: every ORM table must be
explicitly classified for the cutover tooling, with a concrete reason. A new or unclassified
table fails until a human reviews whether cutover plan/check/privacy logic must change.

Classifications are current review assertions, not permanent architectural truth. No DB
access; only ORM model modules and the cutover module are imported.
"""
import importlib
import importlib.util
import pkgutil
import re
from pathlib import Path

from sqlalchemy import Column, MetaData, String, Table

CUTOVER_CHECKED = "cutover_checked"
IDS_ONLY = "ids_only_no_cutover_action"
DERIVED_TEXT_GAP = "derived_text_not_cutover_checked"
NOT_ARCHIVE_DERIVED = "not_archive_derived"
CATEGORIES = {CUTOVER_CHECKED, IDS_ONLY, DERIVED_TEXT_GAP, NOT_ARCHIVE_DERIVED}

CLASSIFICATION = {
    # Read or checked by cutover classification, privacy, binding or orphan checks.
    "raw_archive": (CUTOVER_CHECKED, "population source; content/deletion/tombstone classification"),
    "archive_derivation_heads": (CUTOVER_CHECKED, "head/active/candidate/retirement classification; orphan check"),
    "derivation_generations": (CUTOVER_CHECKED, "generation status/owner/predecessor binding; orphan check"),
    "derivation_fts_projections": (CUTOVER_CHECKED, "baseline/active projection, vector validity, privacy erasure"),
    "derivation_stage_completions": (CUTOVER_CHECKED, "orphan stage-completion blocker"),
    "jobs": (CUTOVER_CHECKED, "exactly-one-completed job contract; orphan jobs; generation binding"),
    "tombstones": (CUTOVER_CHECKED, "source-id/content-hash tombstone classification (class D)"),
    "facts": (CUTOVER_CHECKED, "privacy erasure of fact_text/source_span; generation binding"),
    "summaries": (CUTOVER_CHECKED, "privacy erasure of summary_text; generation binding"),
    "embeddings": (CUTOVER_CHECKED, "privacy suppression; generation binding"),
    "fts_entries": (CUTOVER_CHECKED, "legacy FTS privacy erasure and vector clearing"),
    "canonical_memory": (CUTOVER_CHECKED, "PRIV-02 source_removed + required_review status check"),
    "conflict_groups": (CUTOVER_CHECKED, "privacy check: groups reached via facts.conflict_group_id must be source_removed"),
    "review_queue_items": (CUTOVER_CHECKED, "privacy check: items of those groups must be source_removed"),
    # Archive-related, ids/labels only: nothing for cutover to erase or bind.
    "fact_tags": (IDS_ONLY, "fact_id/tag_id/assigned_by/created_at only; tag text lives in tags"),
    "bridge_archives": (IDS_ONLY, "archive membership ids/labels (archive_id, project_id, client_id); no derived text"),
    # Explicit known gap: archive-derived text that cutover does not check (pre-existing; not changed here).
    "memory_links": (DERIVED_TEXT_GAP, "entity_name derived from extraction; not in cutover privacy checks"),
    "tags": (DERIVED_TEXT_GAP, "tag name text derived from extraction (e.g. entity names); not in cutover privacy checks"),
    # Outside cutover scope: operator, config, audit, telemetry, Bridge identity and authority data.
    "audit_events": (NOT_ARCHIVE_DERIVED, "operational audit log; not archive derivation output"),
    "settings": (NOT_ARCHIVE_DERIVED, "application configuration"),
    "telemetry": (NOT_ARCHIVE_DERIVED, "operational telemetry"),
    "authority_records": (NOT_ARCHIVE_DERIVED, "Bridge authority records (client-written), not archive derivations"),
    "authority_edges": (NOT_ARCHIVE_DERIVED, "authority record relationships"),
    "authority_mutation_receipts": (NOT_ARCHIVE_DERIVED, "authority write idempotency receipts"),
    "bridge_projects": (NOT_ARCHIVE_DERIVED, "Bridge memory-space definitions"),
    "bridge_clients": (NOT_ARCHIVE_DERIVED, "Bridge client identities (credential digests)"),
    "bridge_grants": (NOT_ARCHIVE_DERIVED, "Bridge access grants"),
    "bridge_bindings": (NOT_ARCHIVE_DERIVED, "Bridge destination bindings"),
    "bridge_receipts": (NOT_ARCHIVE_DERIVED, "Bridge ingest idempotency receipts"),
    "bridge_alias_receipts": (NOT_ARCHIVE_DERIVED, "Bridge alias idempotency receipts"),
    # Forward-declared (Slice 2A, planned): ids/timestamps only.
    "conflict_detection_runs": (IDS_ONLY, "Slice 2A (planned): ids/timestamps only"),
    "conflict_group_members": (IDS_ONLY, "Slice 2A (planned): ids/timestamps only"),
}
FORWARD_DECLARED = {"conflict_detection_runs", "conflict_group_members"}


def orm_metadata():
    import app.domain as domain
    for info in pkgutil.iter_modules(domain.__path__):
        name = f"app.domain.{info.name}.models"
        if info.ispkg and importlib.util.find_spec(name) is not None:
            importlib.import_module(name)
    from app.infrastructure.db import Base
    return Base.metadata


def coverage_problems(table_names, classification=CLASSIFICATION, forward=FORWARD_DECLARED):
    names = set(table_names)
    return {
        "unclassified": sorted(names - set(classification)),
        "unknown_not_forward_declared": sorted(set(classification) - names - set(forward)),
        "invalid_entries": sorted(t for t, (category, reason) in classification.items()
                                  if category not in CATEGORIES or not reason.strip()),
    }


def test_every_orm_table_is_explicitly_classified_for_cutover():
    problems = coverage_problems(orm_metadata().tables)
    assert problems["unclassified"] == [], (
        "New/unclassified table(s) need cutover review (plan/check/privacy) and an explicit "
        f"classification: {problems['unclassified']}")
    assert problems["unknown_not_forward_declared"] == []
    assert problems["invalid_entries"] == []


def test_cutover_checked_tables_are_really_referenced_by_cutover():
    from app.domain.derivations import cutover
    source = Path(cutover.__file__).read_text(encoding="utf-8")
    imported = {getattr(value, "__tablename__", None) for value in vars(cutover).values()}
    for table, (category, _) in CLASSIFICATION.items():
        if category == CUTOVER_CHECKED:
            assert table in imported or re.search(rf"\b{table}\b", source), table


def test_guard_fails_on_unreviewed_or_unknown_table():
    copy = MetaData()
    for table in orm_metadata().tables.values():
        table.to_metadata(copy)
    Table("zz_unreviewed_derived_table", copy, Column("id", String, primary_key=True))
    assert coverage_problems(copy.tables)["unclassified"] == ["zz_unreviewed_derived_table"]
    ghost = dict(CLASSIFICATION, ghost_table=(IDS_ONLY, "not a real table"))
    assert coverage_problems(orm_metadata().tables, ghost)["unknown_not_forward_declared"] == ["ghost_table"]
    blank = dict(CLASSIFICATION, raw_archive=(CUTOVER_CHECKED, " "))
    assert coverage_problems(orm_metadata().tables, blank)["invalid_entries"] == ["raw_archive"]
