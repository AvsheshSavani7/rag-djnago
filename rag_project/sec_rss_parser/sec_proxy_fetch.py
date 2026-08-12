"""
Sticky residential proxy GET for SEC Archives document fetches.

Uses the manual proxy list file (residential-rotational proxy.txt). Each attempt
picks a random sticky session; failed sessions are deprioritized temporarily.
Max 5 attempts — no rotating-gateway fallback.

On total failure: send an exception email (existing N8N path) then raise so
callers do not silently skip.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROXY_FILE = os.path.join(BASE_DIR, "residential-rotational proxy.txt")
PROXY_FILE = os.environ.get("SEC_PROXY_LIST_FILE", DEFAULT_PROXY_FILE)

MAX_ATTEMPTS = int(os.environ.get("SEC_PROXY_MAX_ATTEMPTS", "5"))
TIMEOUT = float(os.environ.get("SEC_PROXY_TIMEOUT", "12"))
BACKOFF_BASE = float(os.environ.get("SEC_PROXY_BACKOFF", "0.3"))
DEPRIORITIZE_SEC = float(os.environ.get("SEC_PROXY_DEPRIORITIZE_SEC", "60"))

DEFAULT_HEADERS = {
    "User-Agent": "MNA-Finder/1.0 (https://teqnodux.com; contact: ashish.kachadiya@teqnodux.com)",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class ProxyFetchError(RuntimeError):
    """Raised when all sticky-proxy attempts fail."""

    def __init__(self, message: str, *, url: str, attempts: List[Dict[str, Any]]):
        super().__init__(message)
        self.url = url
        self.attempts = attempts


class ProxyPool:
    """Thread-safe sticky proxy pool with temporary deprioritization."""

    def __init__(self, proxies: List[str]):
        self._all = list(proxies)
        self._lock = threading.Lock()
        self._cooldown_until: Dict[str, float] = {}

    def __len__(self) -> int:
        return len(self._all)

    @staticmethod
    def label_for(line: str) -> str:
        if "session-" in line:
            return "session-" + line.split("session-", 1)[1].split("_", 1)[0]
        return line.split("@", 1)[-1]

    def pick(self) -> Tuple[str, Dict[str, str], str]:
        now = time.time()
        with self._lock:
            eligible = [
                p for p in self._all
                if self._cooldown_until.get(self.label_for(p), 0) <= now
            ]
            if not eligible:
                eligible = self._all
            line = random.choice(eligible)
            label = self.label_for(line)
            url = f"http://{line}"
            return line, {"http": url, "https": url}, label

    def deprioritize(self, label: str) -> None:
        until = time.time() + DEPRIORITIZE_SEC
        with self._lock:
            self._cooldown_until[label] = until
        logger.warning("sec_proxy_fetch: deprioritized %s for %.0fs", label, DEPRIORITIZE_SEC)


_pool_lock = threading.Lock()
_pool: Optional[ProxyPool] = None
_pool_mtime: Optional[float] = None


def _load_proxy_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip()]
    if not lines:
        raise RuntimeError(f"No proxies found in {path}")
    return lines


def get_proxy_pool(force_reload: bool = False) -> ProxyPool:
    """Lazy-load / optionally reload the sticky proxy pool from disk."""
    global _pool, _pool_mtime
    path = PROXY_FILE
    mtime = os.path.getmtime(path) if os.path.exists(path) else None
    with _pool_lock:
        if (
            _pool is None
            or force_reload
            or (mtime is not None and _pool_mtime != mtime)
        ):
            lines = _load_proxy_lines(path)
            _pool = ProxyPool(lines)
            _pool_mtime = mtime
            logger.info(
                "sec_proxy_fetch: loaded %d sticky proxies from %s",
                len(lines),
                os.path.basename(path),
            )
        return _pool


def _send_fail_email(
    url: str,
    attempts: List[Dict[str, Any]],
    last_exc: BaseException,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        from core.exception_email import send_exception_email

        ctx = {
            "url": url,
            "max_attempts": MAX_ATTEMPTS,
            "attempt_log": attempts,
        }
        if context:
            ctx.update(context)
        send_exception_email(
            pipeline="sec_proxy_fetch",
            error_message=f"SEC proxy fetch failed after {MAX_ATTEMPTS} attempts: {type(last_exc).__name__}: {last_exc}",
            context=ctx,
            exception=last_exc,
            email_type="sec_proxy_fetch_failed",
        )
    except Exception:
        logger.exception("sec_proxy_fetch: failed to send failure email for %s", url)


def proxy_get(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    context: Optional[Dict[str, Any]] = None,
    session: Optional[requests.Session] = None,
) -> requests.Response:
    """
    GET ``url`` through the sticky proxy list (max MAX_ATTEMPTS).

    On total failure: emails via send_exception_email, then raises ProxyFetchError.
    """
    pool = get_proxy_pool()
    hdrs = headers or DEFAULT_HEADERS
    to = TIMEOUT if timeout is None else timeout
    getter = session.get if session is not None else requests.get

    attempt_log: List[Dict[str, Any]] = []
    last_exc: Optional[BaseException] = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        _, proxy_dict, label = pool.pick()
        try:
            resp = getter(url, headers=hdrs, proxies=proxy_dict, timeout=to)
            resp.raise_for_status()
            logger.info(
                "sec_proxy_fetch: attempt %d/%d via %s -> %s | %s",
                attempt, MAX_ATTEMPTS, label, resp.status_code, url[:120],
            )
            return resp
        except Exception as e:  # noqa: BLE001
            last_exc = e
            pool.deprioritize(label)
            attempt_log.append({
                "attempt": attempt,
                "proxy": label,
                "error": type(e).__name__,
                "detail": str(e)[:300],
            })
            logger.warning(
                "sec_proxy_fetch: attempt %d/%d via %s FAILED: %s | %s",
                attempt, MAX_ATTEMPTS, label, type(e).__name__, url[:120],
            )
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_BASE * attempt)

    assert last_exc is not None
    _send_fail_email(url, attempt_log, last_exc, context=context)
    raise ProxyFetchError(
        f"SEC proxy fetch failed after {MAX_ATTEMPTS} attempts for {url}",
        url=url,
        attempts=attempt_log,
    ) from last_exc
