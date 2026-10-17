"""Static locality contract; synthetic files only, no Hub/model acquisition."""
import hashlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.infrastructure import local_model as lm


@pytest.fixture
def assets(tmp_path, monkeypatch):
    root = tmp_path / lm.MODEL_REVISION
    root.mkdir()
    data = {"modules.json": b"synthetic modules", "model.safetensors": b"synthetic weights"}
    manifest = {}
    for name, content in data.items():
        (root / name).write_bytes(content)
        if name == "modules.json":
            digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            manifest[name] = (len(content), "git-sha1", digest)
        else:
            manifest[name] = (len(content), "sha256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(lm, "ASSETS", manifest)
    monkeypatch.setattr(lm, "_check_packages", lambda: None)
    monkeypatch.setattr(lm, "get_settings", lambda: SimpleNamespace(local_model_assets_dir=str(root)))
    return root


def test_exact_contract_ready_without_importing_model(assets, monkeypatch):
    monkeypatch.setattr(lm, "_construct", Mock(side_effect=AssertionError("no model in readiness")))
    assert lm.require_local_model_ready() == assets.resolve()


@pytest.mark.parametrize("damage", ["missing_root", "missing_file", "corrupt", "wrong_revision", "extra_file", "escaping_symlink"])
def test_reject_missing_corrupt_wrong_revision_and_extra(assets, damage, tmp_path):
    if damage == "missing_root":
        for path in assets.iterdir():
            path.unlink()
        assets.rmdir()
    elif damage == "missing_file":
        (assets / "modules.json").unlink()
    elif damage == "corrupt":
        # Same size corruption must not pass a size-only check.
        (assets / "model.safetensors").write_bytes(b"x" * len(b"synthetic weights"))
    elif damage == "wrong_revision":
        assets.rename(assets.parent / "unapproved")
    elif damage == "extra_file":
        (assets / "adapter_config.json").write_text("{}")
    else:
        target = tmp_path / "private"
        target.write_bytes(b"synthetic modules")
        (assets / "modules.json").unlink()
        (assets / "modules.json").symlink_to(target)
    with pytest.raises(lm.LocalModelNotReady) as exc:
        lm.require_local_model_ready()
    assert str(exc.value) == "local_model_not_ready"
    assert str(tmp_path) not in repr(exc.value)


def test_git_blob_digest_is_not_plain_sha1(assets, monkeypatch):
    data = (assets / "modules.json").read_bytes()
    monkeypatch.setitem(lm.ASSETS, "modules.json", (len(data), "git-sha1", hashlib.sha1(data).hexdigest()))
    with pytest.raises(lm.LocalModelNotReady):
        lm.require_local_model_ready()


def test_constructor_local_only_and_two_consumers_share_contract(assets, monkeypatch):
    from app.domain.policy import gate
    from app.domain.derived_memory import service
    factory = Mock()
    factory.return_value.encode.return_value = [1.0]
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=factory))
    monkeypatch.setattr(gate, "_gate_model", None)
    monkeypatch.setattr(gate, "_prototype_embeddings", None)
    monkeypatch.setattr(service, "_embed_model", None)
    monkeypatch.setattr(service, "_SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(service, "_SentenceTransformer", None)
    assert gate._get_gate_model() is factory.return_value
    assert service._get_embed_model() is factory.return_value
    assert factory.call_count == 2
    for call in factory.call_args_list:
        assert call.args == (str(assets.resolve()),)
        assert call.kwargs == {"local_files_only": True, "trust_remote_code": False, "revision": lm.MODEL_REVISION}
    assert service.ACTIVE_EMBEDDING_MODEL == "all-MiniLM-L6-v2"


def test_missing_never_calls_constructor(assets, monkeypatch):
    factory = Mock(side_effect=AssertionError("remote fallback"))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=factory))
    (assets / "modules.json").unlink()
    with pytest.raises(lm.LocalModelNotReady):
        lm.load_local_model()
    factory.assert_not_called()


def test_constructor_exception_is_static(assets, monkeypatch):
    factory = Mock(side_effect=RuntimeError("SECRET /cache/path https://private/token"))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=factory))
    with pytest.raises(lm.LocalModelNotReady) as exc:
        lm.load_local_model()
    assert str(exc.value) == "local_model_not_ready"
    assert exc.value.__cause__ is None


def test_package_mismatch_is_bounded(monkeypatch):
    monkeypatch.setattr(lm.metadata, "version", lambda _: "unapproved")
    with pytest.raises(lm.LocalModelNotReady):
        lm._check_packages()
