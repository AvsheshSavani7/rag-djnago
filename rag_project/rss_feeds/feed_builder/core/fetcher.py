import logging
import os
import time
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

try:
    from curl_cffi import requests as cffi_requests  # type: ignore
    _CURL_CFFI_AVAILABLE = True
except ImportError:
    _CURL_CFFI_AVAILABLE = False

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# Sites known to soft-block / WAF plain requests (hang, drop, or 403).
# When fetch_mode is auto/requests, these hosts skip straight to cffi + residential proxy.
BOT_PROTECTED_HINTS = (
    "businesswire.com",
    "bloomberg.com",
    "globenewswire.com",
)

# Residential proxy — read from env; falls back to reference credentials.
_PROXY_HOST     = os.environ.get("RESIDENTIAL_PROXY_HOST", "108.59.242.138")
_PROXY_PORT     = os.environ.get("RESIDENTIAL_PROXY_PORT", "46885")
_PROXY_USERNAME = os.environ.get("RESIDENTIAL_PROXY_USERNAME", "GSenAgrfKhuNWkd")
_PROXY_PASSWORD = os.environ.get("RESIDENTIAL_PROXY_PASSWORD", "8lmVa5yl0pKp9MI")

_PROXY_URL  = f"http://{_PROXY_USERNAME}:{_PROXY_PASSWORD}@{_PROXY_HOST}:{_PROXY_PORT}"
_PROXY_DICT = {"http": _PROXY_URL, "https": _PROXY_URL}


class FetchBlockedError(Exception):
    """Raised when a site blocks automated fetching (403/Access Denied)."""


def _is_access_denied(body: str) -> bool:
    lower = (body or "").lower()
    return "access denied" in lower or "edgesuite.net" in lower


def _is_akamai_blocked(body: str) -> bool:
    return "Reference #" in (body or "") and "edgesuite.net" in (body or "")


def _fetch_with_cffi(url: str, headers: dict, timeout: int, proxies: Optional[dict] = None) -> Optional[str]:
    """Fetch using curl_cffi with Chrome TLS impersonation."""
    if not _CURL_CFFI_AVAILABLE:
        return None
    try:
        resp = cffi_requests.get(
            url,
            headers=headers,
            proxies=proxies,
            impersonate="chrome124",
            timeout=timeout,
            allow_redirects=True,
        )
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        logger.warning("curl_cffi fetch failed for %s (proxy=%s): %s", url, bool(proxies), exc)
        return None


def _fetch_with_cloudscraper(url: str, headers: dict, timeout: int) -> Optional[requests.Response]:
    try:
        import cloudscraper

        scraper = cloudscraper.create_scraper()
        return scraper.get(url, headers=headers, timeout=timeout)
    except Exception as exc:
        logger.warning("cloudscraper fetch failed for %s: %s", url, exc)
        return None


def _fetch_with_playwright(url: str, timeout: int) -> Optional[str]:
    try:
        from rss_feeds.merger_news_classifier import fetch_html_from_url

        return fetch_html_from_url(url, timeout=timeout)
    except Exception as exc:
        logger.warning("playwright fetch failed for %s: %s", url, exc)
        return None


def _needs_cffi_proxy(url: str) -> bool:
    lower = (url or "").lower()
    return any(hint in lower for hint in BOT_PROTECTED_HINTS)


def _blocked_message(url: str, status_code: int) -> str:
    host = url.split("/")[2] if "://" in url else url
    base = (
        f"HTTP {status_code} Forbidden — {host} blocks automated/bot requests "
        f"(likely Akamai or similar WAF)."
    )
    if "businesswire.com" in url.lower():
        base += (
            " BusinessWire listing pages often cannot be fetched from a server. "
            "Workarounds: use a BusinessWire RSS feed URL if available, "
            "configure the feed from a machine/IP that is allowed, "
            "or use Playwright on a local desktop with a real browser session."
        )
    return base


def _try_bot_bypass(url: str, headers: dict, timeout: int) -> Optional[str]:
    """
    Try multiple strategies to bypass bot/WAF protection.

    Order:
      1. curl_cffi + residential proxy  (best: Chrome TLS fingerprint + residential IP)
      2. curl_cffi direct               (Chrome TLS fingerprint, no proxy)
      3. requests + residential proxy   (residential IP only)
      4. cloudscraper                   (JS-challenge bypass)
      5. playwright                     (real browser, last resort)
    """
    # Strategy 1 & 2: curl_cffi
    for label, proxies in [
        ("curl_cffi + residential proxy", _PROXY_DICT),
        ("curl_cffi direct", None),
    ]:
        html = _fetch_with_cffi(url, headers, timeout, proxies=proxies)
        if html and not _is_akamai_blocked(html) and not _is_access_denied(html):
            logger.info("Bot bypass succeeded via %s for %s", label, url)
            return html
        if html:
            logger.warning("Bot bypass blocked via %s for %s (Akamai/access-denied)", label, url)
        time.sleep(1)

    # Strategy 3: requests + residential proxy
    try:
        resp = requests.get(
            url,
            headers=headers,
            proxies=_PROXY_DICT,
            timeout=timeout,
            verify=False,
            allow_redirects=True,
        )
        if resp.status_code == 200 and not _is_akamai_blocked(resp.text) and not _is_access_denied(resp.text):
            logger.info("Bot bypass succeeded via requests + residential proxy for %s", url)
            return resp.text
    except Exception as exc:
        logger.warning("requests + proxy failed for %s: %s", url, exc)

    # Strategy 4: cloudscraper
    cs_resp = _fetch_with_cloudscraper(url, headers, timeout)
    if cs_resp is not None and cs_resp.status_code == 200 and not _is_access_denied(cs_resp.text):
        logger.info("Bot bypass succeeded via cloudscraper for %s", url)
        return cs_resp.text

    # Strategy 5: playwright
    html = _fetch_with_playwright(url, timeout=timeout)
    if html and not _is_access_denied(html):
        logger.info("Bot bypass succeeded via playwright for %s", url)
        return html

    return None


def fetch_url(
    url: str,
    timeout: int = 30,
    headers: Optional[dict] = None,
    fetch_mode: str = "auto",
) -> Tuple[str, str, int]:
    """
    Fetch a URL and return (body_text, content_type, status_code).

    fetch_mode values:
      auto        — requests → bot-bypass on 403/429/503 or timeout/connection errors
      requests    — same as auto (legacy value stored in existing feed configs)
      cffi_proxy  — skip initial requests call; go straight to curl_cffi + residential proxy.
                    Use for known Akamai-protected sites (e.g. BusinessWire, GlobeNewswire).
      playwright  — skip everything; use Playwright directly

    Hosts in BOT_PROTECTED_HINTS auto-upgrade auto/requests → cffi_proxy so they skip
    the failing plain-requests attempt (other feeds are unchanged).
    """
    merged_headers = {**BROWSER_HEADERS, **(headers or {})}

    effective_mode = fetch_mode
    if fetch_mode in ("auto", "requests") and _needs_cffi_proxy(url):
        effective_mode = "cffi_proxy"
        logger.info("Using cffi_proxy for bot-protected host: %s", url)

    if effective_mode == "playwright":
        html = _fetch_with_playwright(url, timeout=timeout)
        if not html or _is_access_denied(html):
            raise FetchBlockedError(_blocked_message(url, 403))
        return html, "text/html", 200

    if effective_mode == "cffi_proxy":
        html = _fetch_with_cffi(url, merged_headers, timeout, proxies=_PROXY_DICT)
        if html and not _is_akamai_blocked(html) and not _is_access_denied(html):
            return html, "text/html", 200
        # cffi+proxy failed — fall through to full bypass chain as safety net
        html = _try_bot_bypass(url, merged_headers, timeout)
        if html:
            return html, "text/html", 200
        raise FetchBlockedError(_blocked_message(url, 403))

    try:
        response = requests.get(url, timeout=timeout, headers=merged_headers)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        # Soft-blocks often hang or drop instead of returning 403 — fall back to bypass.
        logger.warning(
            "Direct fetch failed for %s (%s); trying bot bypass", url, exc
        )
        html = _try_bot_bypass(url, merged_headers, timeout)
        if html:
            return html, "text/html", 200
        raise

    if response.status_code in (403, 429, 503) and fetch_mode in ("auto", "requests"):
        html = _try_bot_bypass(url, merged_headers, timeout)
        if html:
            return html, "text/html", 200

    if response.status_code == 403 or _is_access_denied(response.text):
        raise FetchBlockedError(_blocked_message(url, response.status_code))

    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    response.encoding = response.encoding or "utf-8"
    return response.text, content_type, response.status_code
