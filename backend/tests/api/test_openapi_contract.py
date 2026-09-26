"""Contract checks for the runtime OpenAPI schema and deterministic exporter."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from app.main import create_app


def _schema() -> dict:
    return create_app().openapi()


def test_runtime_schema_has_core_api_surface_and_excludes_sidecars() -> None:
    paths = _schema()["paths"]
    for path in ("/api/health", "/api/ingest", "/api/archive", "/api/search", "/api/retrieve"):
        assert path in paths
    assert not any(path.startswith("/bridge") for path in paths)
    assert not any(path.startswith("/mcp") for path in paths)


def test_runtime_schema_metadata_and_security_are_truthful() -> None:
    schema = _schema()
    assert schema["info"]["title"] == "Recalium API"
    assert schema["info"]["version"] == "0.1.0"
    # Authentication is enforced by middleware, so do not claim blanket OpenAPI security.
    assert schema.get("security") in (None, [])


def test_runtime_schema_normalizes_deterministically() -> None:
    first = json.dumps(_schema(), sort_keys=True, separators=(",", ":"))
    second = json.dumps(_schema(), sort_keys=True, separators=(",", ":"))
    assert first == second


def test_exporter_matches_runtime_schema(tmp_path: Path) -> None:
    output = tmp_path / "openapi.json"
    script = Path(__file__).resolve().parents[2] / "scripts" / "export_openapi.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    result = subprocess.run(
        [sys.executable, str(script), str(output)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    exported_bytes = output.read_bytes()
    assert b"\r\n" not in exported_bytes
    assert b"\n" in exported_bytes
    exported = json.loads(output.read_text(encoding="utf-8"))
    assert json.dumps(exported, sort_keys=True, separators=(",", ":")) == json.dumps(
        _schema(), sort_keys=True, separators=(",", ":")
    )
