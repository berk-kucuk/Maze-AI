"""Shared HTTP helper with retry/backoff for transient failures.

Backends make one blocking request per turn; a dropped connection or a
momentary 5xx/429 shouldn't surface as a hard error when a quick retry would
succeed. This wraps ``requests`` with exponential backoff on the errors that
are actually worth retrying.
"""

from __future__ import annotations

import time

import requests

# Status codes that usually clear up on a retry (rate limit + gateway/5xx).
_RETRY_STATUS = {429, 500, 502, 503, 504}


def request_with_retry(
    method: str,
    url: str,
    *,
    attempts: int = 3,
    backoff: float = 0.8,
    **kwargs,
) -> requests.Response:
    """Perform an HTTP request, retrying transient failures.

    Retries on connection errors/timeouts and on retryable status codes, with
    exponential backoff (``backoff * 2**i`` seconds). The last response or
    exception is returned/raised so callers keep their existing status handling.
    Streaming responses (``stream=True``) are returned on the first success
    without inspecting the status body.
    """
    attempts = max(1, attempts)
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            resp = requests.request(method, url, **kwargs)
        except requests.RequestException as exc:
            last_exc = exc
            if i == attempts - 1:
                raise
        else:
            # Don't retry good responses, non-retryable errors, or the last try.
            if resp.status_code not in _RETRY_STATUS or i == attempts - 1:
                return resp
        time.sleep(backoff * (2 ** i))
    # Unreachable in practice (loop either returns or raises), but keeps mypy
    # and the type checker happy.
    if last_exc is not None:
        raise last_exc
    raise requests.RequestException("request failed")
