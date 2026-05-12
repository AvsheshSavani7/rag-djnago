"""Anthropic Messages API errors: one-shot diagnostic logging + billing detection (parallel-safe)."""

import threading

_lock = threading.Lock()
_logged_diagnostic = False
_billing_blocked = threading.Event()
_billing_detail = ""

_CREDIT_MARKERS = (
    "credit balance is too low",
    "too low to access the anthropic api",
)


def record_anthropic_http_error(response) -> None:
    """
    Inspect every failed API response. Sets a process-wide flag when Anthropic
    reports exhausted / insufficient credits (HTTP 400 with their standard JSON).
    """
    global _billing_detail
    if response is None:
        return
    try:
        raw = response.text or ""
        blob = raw.lower()
    except Exception:
        return
    if not any(m in blob for m in _CREDIT_MARKERS):
        return
    with _lock:
        if not _billing_detail:
            _billing_detail = raw[:2000]
    _billing_blocked.set()


def raise_if_anthropic_billing_blocked() -> None:
    """Abort immediately when billing/credit errors were detected."""
    if not _billing_blocked.is_set():
        return
    hint = (
        "Anthropic returned a billing/credit error for this API key. "
        "The Console UI can disagree briefly (wrong workspace, spend cap, prepaid vs usage billing). "
        "Confirm Plans & billing for the workspace that created this key, and contact Anthropic with "
        "request_id from the diagnostic JSON if needed."
    )
    raise RuntimeError(f"{hint}\n\nLast API error body:\n{_billing_detail}")


def log_anthropic_error_response(response, context: str) -> None:
    """
    Print response body once per process for HTTP errors (e.g. 400 invalid model).
    Safe when many threads hit the same failure.
    """
    global _logged_diagnostic
    if response is None:
        return
    with _lock:
        if _logged_diagnostic:
            return
        _logged_diagnostic = True
    try:
        body = (response.text or "")[:8000]
    except Exception as exc:
        body = f"<could not read body: {exc}>"
    print(
        f"  [Anthropic diagnostic — {context}] "
        f"HTTP {response.status_code} response body:\n{body}"
    )


def handle_anthropic_http_error(response, context: str) -> None:
    """Record billing state, print one diagnostic, then raise if credits are blocked."""
    record_anthropic_http_error(response)
    log_anthropic_error_response(response, context)
    raise_if_anthropic_billing_blocked()
