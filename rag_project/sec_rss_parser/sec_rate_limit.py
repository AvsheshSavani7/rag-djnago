"""
Simple process-wide rate limiter for SEC (sec.gov) HTTP requests.

The SEC browse-edgar endpoints are documented as having a limit of
around 10 requests per second per IP. This module centralizes throttling
so that all callers share the same budget and we stay comfortably below
that limit.
"""

import os
import time
import threading
from urllib.parse import urlparse

# Default minimum interval between SEC requests (in seconds).
# 0.2s ≈ 5 requests/second. You can override via SEC_MIN_REQ_INTERVAL env var.
_DEFAULT_INTERVAL = 0.2
_MIN_INTERVAL = float(os.environ.get("SEC_MIN_REQ_INTERVAL", _DEFAULT_INTERVAL))

_SEC_HOST = "www.sec.gov"
_lock = threading.Lock()
_last_call = 0.0


def rate_limited_get(session_or_requests, url, *args, **kwargs):
    """
    Wrapper around <session_or_requests>.get that enforces a global
    per-process rate limit for sec.gov requests.

    - If the URL host is not www.sec.gov, the call is forwarded unmodified.
    - For sec.gov URLs, calls are spaced by at least _MIN_INTERVAL seconds.
    """
    parsed = urlparse(url)
    if parsed.netloc and parsed.netloc.lower() != _SEC_HOST:
        return session_or_requests.get(url, *args, **kwargs)

    global _last_call
    with _lock:
        now = time.monotonic()
        wait = _MIN_INTERVAL - (now - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()

    return session_or_requests.get(url, *args, **kwargs)

