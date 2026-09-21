"""Provider-neutral local inference endpoint policy."""
from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlsplit


def is_local_inference_endpoint_url(url: str | None) -> bool:
    """Allow only loopback endpoints and Docker Desktop's host gateway.

    This policy deliberately accepts an endpoint URL rather than a provider-specific
    settings object so local execution seams can share one bounded trust decision.
    """
    try:
        parsed = urlsplit(url or "")
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            return False
        # Accessing port also rejects malformed and out-of-range ports.
        parsed.port
        if parsed.hostname in {"localhost", "host.docker.internal"}:
            return True
        return ip_address(parsed.hostname).is_loopback
    except ValueError:
        return False
