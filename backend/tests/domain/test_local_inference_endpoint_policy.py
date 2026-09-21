"""Provider-neutral local-inference endpoint policy tests."""
from __future__ import annotations

import pytest

from app.domain.model_context.policy import is_local_inference_endpoint_url


@pytest.mark.parametrize("url,local", [
    ("http://host.docker.internal:11434", True),
    ("http://localhost:11434/", True), ("http://127.0.0.1:11434", True),
    ("http://[::1]:11434", True), ("https://ollama.example.com", False),
    ("http://192.168.1.20:11434", False), ("http://localhost.evil.test", False),
    ("http://localhost@evil.test", False), ("http://evil.test@localhost", False),
    ("http://localhost:99999", False), ("file://localhost", False),
    ("http://localhost/proxy", False), ("http://[", False), (None, False),
])
def test_local_endpoint_boundary(url, local):
    assert is_local_inference_endpoint_url(url) is local
