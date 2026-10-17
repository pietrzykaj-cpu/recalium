"""One prospective, offline MiniLM contract. No model import/inference in readiness.

Approved public file identities were captured from the verified Hub snapshot.
Small files use Hub Git-blob identities; safetensors uses its LFS SHA-256.
Assets must be provisioned separately, immutable/read-only, never downloaded here.
"""
from __future__ import annotations

import asyncio
import hashlib
from importlib import metadata
import os
from pathlib import Path

from app.infrastructure.settings import get_settings

# Before lazy ML/Hub imports, including native use. Explicit local-only arguments
# below remain mandatory; these flags are not a process-wide network sandbox.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ASSETS = {
    "1_Pooling/config.json": (190, "git-sha1", "d1514c3162bbe87b343f565fadc62e6c06f04f03"),
    "README.md": (10502, "git-sha1", "44af2e3b0fa3a0b6239e48422972bf755f28fde0"),
    "config.json": (612, "git-sha1", "72b987fd805cfa2b58c4c8c952b274a11bfd5a00"),
    "config_sentence_transformers.json": (116, "git-sha1", "fd1b291129c607e5d49799f87cb219b27f98acdf"),
    "model.safetensors": (90868376, "sha256", "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"),
    "modules.json": (349, "git-sha1", "952a9b81c0bfd99800fabf352f69c7ccd46c5e43"),
    "sentence_bert_config.json": (53, "git-sha1", "59d594003bf59880a884c574bf88ef7555bb0202"),
    "special_tokens_map.json": (112, "git-sha1", "e7b0375001f109a6b8873d756ad4f7bbb15fbaa5"),
    "tokenizer.json": (466247, "git-sha1", "cb202bfe2e3c98645018a6d12f182a434c9d3e02"),
    "tokenizer_config.json": (350, "git-sha1", "c79f2b6a0cea6f4b564fed1938984bace9d30ff0"),
    "vocab.txt": (231508, "git-sha1", "fb140275c155a9c7c5a3b3e0e77a9e839594a938"),
}
PACKAGES = {"sentence-transformers": "5.3.0", "transformers": "5.16.1",
            "huggingface-hub": "1.30.0", "tokenizers": "0.23.2", "safetensors": "0.8.0",
            "torch": "2.14.0+cpu"}


class LocalModelNotReady(Exception):
    """Only this static marker crosses API/job/log boundaries."""
    def __init__(self):
        super().__init__("local_model_not_ready")


def _check_packages() -> None:
    try:
        if any(metadata.version(name) != version for name, version in PACKAGES.items()):
            raise LocalModelNotReady()
    except metadata.PackageNotFoundError:
        raise LocalModelNotReady() from None


def require_local_model_ready() -> Path:
    """Validate the entire immutable approved snapshot, without ML or network.

    Do not cache readiness: check every admission and every first construction.
    Mount the snapshot read-only; validation is not a concurrent asset writer lock.
    """
    try:
        root = Path(get_settings().local_model_assets_dir)
        if not root.is_absolute() or root.name != MODEL_REVISION:
            raise LocalModelNotReady()
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise LocalModelNotReady()
        _check_packages()
        files = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                # Permit only links contained in the provisioned snapshot.
                path.resolve(strict=True).relative_to(root)
                if path.is_dir():
                    raise LocalModelNotReady()
            if path.is_file():
                files.add(path.relative_to(root).as_posix())
            elif not path.is_dir():
                raise LocalModelNotReady()
        if files != set(ASSETS):
            raise LocalModelNotReady()
        for name, (size, algorithm, expected) in ASSETS.items():
            path = root / name
            if path.stat().st_size != size:
                raise LocalModelNotReady()
            digest = hashlib.sha1() if algorithm == "git-sha1" else hashlib.sha256()
            if algorithm == "git-sha1":
                digest.update(f"blob {size}\0".encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise LocalModelNotReady()
        return root
    except (OSError, ValueError, RuntimeError):
        raise LocalModelNotReady() from None


async def require_local_model_ready_async() -> Path:
    """Filesystem hashing off-loop; caller must end DB transactions first."""
    return await asyncio.to_thread(require_local_model_ready)


def _construct(path: Path, factory=None):
    if factory is None:
        from sentence_transformers import SentenceTransformer
        factory = SentenceTransformer
    return factory(str(path), local_files_only=True, trust_remote_code=False, revision=MODEL_REVISION)


def load_local_model(*, factory=None):
    path = require_local_model_ready()
    try:
        return _construct(path, factory)
    except Exception:
        # Do not persist exception messages, URLs, local paths or loader internals.
        raise LocalModelNotReady() from None
