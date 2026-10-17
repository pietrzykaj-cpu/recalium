"""Static configuration plumbing; no production dotenv or services."""
from pathlib import Path
import yaml


def test_compose_durable_read_only_assets_and_offline_defaults():
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load((root / "docker-compose.yml").read_text())
    app = config["services"]["recalium-app"]
    env = app["environment"]
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert "1110a243fdf4706b3f48f1d95db1a4f5529b4d41" in env["LOCAL_MODEL_ASSETS_DIR"]
    mount = next(v for v in app["volumes"] if isinstance(v, dict) and v["target"] == "/app/local-models")
    assert mount["type"] == "bind" and mount["read_only"] is True
    assert mount["bind"]["create_host_path"] is False
    assert mount["source"] == "${LOCAL_MODEL_ASSETS_HOST_DIR:-./data/local-models}"
