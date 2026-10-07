"""Offline startup-guard tests; run with --noconftest to exclude DB fixtures.

Compile the real function without importing the application or its lifespan.
Its model imports receive inert modules; SQLAlchemy metadata stays in memory.
"""
from __future__ import annotations

import ast
import builtins
import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import Column, MetaData, String, Table
from sqlalchemy.orm import DeclarativeBase


BACKEND = Path(__file__).resolve().parents[1]
MODEL_IMPORTS = {
    f"app.domain.{name}.models"
    for name in (
        "archive", "settings", "jobs", "audit", "derived_memory",
        "canonical_memory", "review_queue", "telemetry",
    )
}


def _namespace(base):
    db = ModuleType("app.infrastructure.db")
    db.Base = base
    inert_app = ModuleType("app")

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "app.infrastructure.db":
            return db
        if name in MODEL_IMPORTS:
            return inert_app
        if name == "app" or name.startswith("app."):
            raise AssertionError(f"Unexpected application import: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    return {
        "__builtins__": dict(vars(builtins), __import__=safe_import),
        "logger": logging.getLogger("schema_guard_test"),
        "__name__": "schema_guard_test",
    }


def _guard(metadata):
    base = type("MetadataOnlyBase", (), {"metadata": metadata})
    source = BACKEND / "app" / "main.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_assert_no_keys_in_schema"
    )
    namespace = _namespace(base)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_assert_no_keys_in_schema"]


def _metadata(table_name, *columns):
    metadata = MetaData()
    Table(table_name, metadata, *(Column(name, String) for name in columns))
    return metadata


def test_exact_authority_scope_identifier_allowed():
    _guard(_metadata("authority_records", "authority_key"))()


@pytest.mark.parametrize("column", ["api_key", "api_secret", "api_token", "api_password"])
def test_credential_suffixes_rejected(column):
    with pytest.raises(RuntimeError, match=f"some_table.{column}"):
        _guard(_metadata("some_table", column))()


def test_same_column_on_other_table_rejected():
    with pytest.raises(RuntimeError, match="other_table.authority_key"):
        _guard(_metadata("other_table", "authority_key"))()


@pytest.mark.parametrize("table_name", ["authority_records", "other_table"])
@pytest.mark.parametrize("column", [
    "api_key", "api_secret", "api_token", "api_password", "AUTHORITY_KEY",
    "authority_key_secret", "authority_key_token", "authority_key_password",
])
def test_exemption_does_not_hide_other_forbidden_columns(table_name, column):
    with pytest.raises(RuntimeError, match=f"{table_name}.{column}"):
        _guard(_metadata(table_name, column))()


@pytest.mark.parametrize("forbidden", ["key", "secret", "token", "password"])
@pytest.mark.parametrize("allowed", ["fingerprint", "configured", "validation_status", "validated_at"])
def test_existing_allowed_suffix_behavior_unchanged(forbidden, allowed):
    _guard(_metadata("settings", f"api_{forbidden}_{allowed}"))()


def test_allowed_identifier_does_not_suppress_another_tables_violation():
    metadata = _metadata("authority_records", "authority_key")
    Table("settings", metadata, Column("api_key", String))
    with pytest.raises(RuntimeError, match="settings.api_key") as error:
        _guard(metadata)()
    assert "authority_records.authority_key" not in str(error.value)


def test_real_authority_model_metadata_allowed_without_database(monkeypatch):
    class Base(DeclarativeBase):
        pass

    source = BACKEND / "app" / "domain" / "authority" / "models.py"
    module = ModuleType("schema_guard_authority_model_test")
    module.__dict__.update(_namespace(Base))
    module.__dict__["__name__"] = module.__name__ = "schema_guard_authority_model_test"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), module.__dict__)
    assert "authority_key" in Base.metadata.tables["authority_records"].columns
    _guard(Base.metadata)()
