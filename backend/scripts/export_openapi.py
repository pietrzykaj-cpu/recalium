"""Export the authoritative FastAPI OpenAPI schema to an explicit path.

This is a deterministic build/test artifact. The running application's /openapi.json
remains the authoritative schema source.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Recalium's runtime OpenAPI schema")
    parser.add_argument("output", type=Path, help="destination JSON path")
    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parents[1]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    try:
        from app.main import create_app
        schema = create_app().openapi()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except Exception as exc:
        parser.error(f"could not build OpenAPI schema: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
