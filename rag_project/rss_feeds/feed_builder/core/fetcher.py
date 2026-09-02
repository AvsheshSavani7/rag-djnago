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

try:
    from curl_cffi import CurlHttpVersion  # type: ignore
    _CURL_HTTP_1_1 = CurlHttpVersion.V1_1
except Exception:
    _CURL_HTTP_1_1 = None

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
_PROXY_HOST = os.environ.get("RESIDENTIAL_PROXY_HOST", "108.59.242.138")
_PROXY_PORT = os.environ.get("RESIDENTIAL_PROXY_PORT", "46885")
_PROXY_USERNAME = os.environ.get("RESIDENTIAL_PROXY_USERNAME", "GSenAgrfKhuNWkd")
_PROXY_PASSWORD = os.environ.get("RESIDENTIAL_PROXY_PASSWORD", "8lmVa5yl0pKp9MI")

_PROXY_URL = f"http://{_PROXY_USERNAME}:{_PROXY_PASSWORD}@{_PROXY_HOST}:{_PROXY_PORT}"
_PROXY_DICT = {"http": _PROXY_URL, "https": _PROXY_URL}

_PROXY_RETRIES = 3
_RETRYABLE_STATUS = {500, 502, 503, 504}
_RETRYABLE_ERROR_TOKENS = (
    "connect tunnel failed",
    "curl: (56)",
    "http error 500",
    "http error 502",
    "http error 503",
    "http error 504",
    "response 500",
    "502 bad gateway",
    "503 service",
    "504 gateway",
)


class FetchBlockedError(Exception):
    """Raised when a site blocks automated fetching (403/Access Denied)."""


def _is_access_denied(body: str) -> bool:
    lower = (body or "").lower()
    return "access denied" in lower or "edgesuite.net" in lower


def _is_akamai_blocked(body: str) -> bool:
    return "Reference #" in (body or "") and "edgesuite.net" in (body or "")


def _is_businesswire_unavailable_page(body: str) -> bool:
    """BusinessWire WAF often returns HTTP 200 with a 'Page Unavailable' interstitial."""
    lower = (body or "").lower()
    return (
        "please be advised that this page is unavailable" in lower
        or (
            "<title>page unavailable</title>" in lower
            and "websupport@businesswire.com" in lower
        )
    )


def _usable_html(html: Optional[str]) -> bool:
    return (
        bool(html)
        and not _is_akamai_blocked(html)
        and not _is_access_denied(html)
        and not _is_businesswire_unavailable_page(html)
    )


def _is_retryable_proxy_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(token in msg for token in _RETRYABLE_ERROR_TOKENS)


def _backoff_sleep(attempt: int) -> None:
    time.sleep(min(2 ** attempt, 8))


def _fetch_with_cffi(
    url: str,
    headers: dict,
    timeout: int,
    proxies: Optional[dict] = None,
    retries: int = 1,
    http_version=None,
) -> Optional[str]:
    """Fetch using curl_cffi with Chrome TLS impersonation.

    Retries only proxy/gateway failures (CONNECT 500, HTTP 502/503/504).
    Does not retry a sticky WAF 403.
    """
    if not _CURL_CFFI_AVAILABLE:
        return None
    attempts = max(1, retries) if proxies else 1
    extra = {}
    if http_version is not None:
        extra["http_version"] = http_version

    for attempt in range(attempts):
        try:
            resp = cffi_requests.get(
                url,
                headers=headers,
                proxies=proxies,
                impersonate="chrome124",
                timeout=timeout,
                allow_redirects=True,
                **extra,
            )
            if resp.status_code == 200:
                return resp.text
            logger.warning(
                "curl_cffi fetch failed for %s (proxy=%s): HTTP Error %s:",
                url,
                bool(proxies),
                resp.status_code,
            )
            if (
                proxies
                and resp.status_code in _RETRYABLE_STATUS
                and attempt < attempts - 1
            ):
                logger.info(
                    "Retrying curl_cffi+proxy after HTTP %s (attempt %s/%s)",
                    resp.status_code,
                    attempt + 2,
                    attempts,
                )
                _backoff_sleep(attempt)
                continue
            return None
        except Exception as exc:
            logger.warning(
                "curl_cffi fetch failed for %s (proxy=%s): %s",
                url,
                bool(proxies),
                exc,
            )
            if proxies and _is_retryable_proxy_error(exc) and attempt < attempts - 1:
                logger.info(
                    "Retrying curl_cffi+proxy after %s (attempt %s/%s)",
                    exc,
                    attempt + 2,
                    attempts,
                )
                _backoff_sleep(attempt)
                continue
            return None
    return None


def _fetch_with_cffi_http1(
    url: str, headers: dict, timeout: int, proxies: Optional[dict] = None
) -> Optional[str]:
    """One extra curl_cffi attempt over HTTP/1.1 (Akamai often RST_STREAMs HTTP/2)."""
    if _CURL_HTTP_1_1 is None:
        return None
    logger.info("Retrying curl_cffi over HTTP/1.1 for %s (proxy=%s)", url, bool(proxies))
    return _fetch_with_cffi(
        url,
        headers,
        timeout,
        proxies=proxies,
        retries=1,
        http_version=_CURL_HTTP_1_1,
    )


def _fetch_with_cloudscraper(url: str, headers: dict, timeout: int) -> Optional[requests.Response]:
    try:
        import cloudscraper

        scraper = cloudscraper.create_scraper()
        return scraper.get(url, headers=headers, timeout=timeout)
    except Exception as exc:
        logger.warning("cloudscraper fetch failed for %s: %s", url, exc)
        return None


def _playwright_launch_kwargs(use_proxy: bool) -> dict:
    kwargs = {
        "headless": True,
        "args": [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
            "--disable-http2",
        ],
    }
    if use_proxy:
        kwargs["proxy"] = {
            "server": f"http://{_PROXY_HOST}:{_PROXY_PORT}",
            "username": _PROXY_USERNAME,
            "password": _PROXY_PASSWORD,
        }
    return kwargs


def fetch_html_with_playwright(url: str, timeout: int = 30) -> Optional[str]:
    """Headless Chromium fetch with HTTP/2 disabled.

    Residential proxy is used only for bot-protected hosts (BusinessWire,
    GlobeNewswire, Bloomberg) so other Playwright callers keep a direct path.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning("Playwright not installed; cannot fetch %s", url)
        return None

    timeout_ms = min(max(timeout, 1) * 1000, 60000)
    ua = BROWSER_HEADERS["User-Agent"]
    try_proxy = _needs_cffi_proxy(url)

    def _goto(playwright, use_proxy: bool) -> str:
        browser = playwright.chromium.launch(**_playwright_launch_kwargs(use_proxy))
        try:
            context = browser.new_context(
                user_agent=ua,
                locale="en-US",
                viewport={"width": 1280, "height": 800},
            )
            page = context.new_page()
            page.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            page.wait_for_timeout(2000)
            return page.content()
        finally:
            browser.close()

    try:
        with sync_playwright() as playwright:
            if try_proxy:
                try:
                    html = _goto(playwright, use_proxy=True)
                    logger.info("Playwright fetch succeeded via proxy for %s", url)
                    return html
                except Exception as exc:
                    logger.warning("Playwright+proxy failed for %s: %s", url, exc)
            html = _goto(playwright, use_proxy=False)
            logger.info("Playwright fetch succeeded without proxy for %s", url)
            return html
    except Exception as exc:
        logger.warning("Playwright fetch failed for %s: %s", url, exc)
        return None


def _fetch_with_playwright(url: str, timeout: int) -> Optional[str]:
    return fetch_html_with_playwright(url, timeout=timeout)


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


def _try_bot_bypass(
    url: str,
    headers: dict,
    timeout: int,
    skip_cffi_proxy: bool = False,
) -> Optional[str]:
    """
    Try multiple strategies to bypass bot/WAF protection.

    Order:
      1. curl_cffi + residential proxy  (skipped if that path already returned 403)
      2. curl_cffi direct
      3. curl_cffi HTTP/1.1 (proxy, then direct)
      4. requests + residential proxy
      5. cloudscraper
      6. playwright (HTTP/2 disabled, proxy then direct)

    Jina Reader is intentionally not used here. Listing scans (RSS/HTML
    newsrooms) would treat Jina's HTML/markdown as a successful fetch and
    then fail to parse items. Article summarization still uses Jina in
    PRNewswire_summary.
    """
    cffi_attempts = []
    if not skip_cffi_proxy:
        cffi_attempts.append(("curl_cffi + residential proxy", _PROXY_DICT))
    cffi_attempts.append(("curl_cffi direct", None))

    for label, proxies in cffi_attempts:
        html = _fetch_with_cffi(url, headers, timeout, proxies=proxies, retries=1)
        if _usable_html(html):
            logger.info("Bot bypass succeeded via %s for %s", label, url)
            return html
        if html:
            logger.warning("Bot bypass blocked via %s for %s (Akamai/access-denied)", label, url)
        time.sleep(1)

    for label, proxies in (
        ("curl_cffi HTTP/1.1 + proxy", _PROXY_DICT),
        ("curl_cffi HTTP/1.1 direct", None),
    ):
        html = _fetch_with_cffi_http1(url, headers, timeout, proxies=proxies)
        if _usable_html(html):
            logger.info("Bot bypass succeeded via %s for %s", label, url)
            return html

    try:
        resp = requests.get(
            url,
            headers=headers,
            proxies=_PROXY_DICT,
            timeout=timeout,
            verify=False,
            allow_redirects=True,
        )
        if resp.status_code == 200 and _usable_html(resp.text):
            logger.info("Bot bypass succeeded via requests + residential proxy for %s", url)
            return resp.text
        logger.warning(
            "requests + proxy returned HTTP %s for %s",
            resp.status_code,
            url,
        )
    except Exception as exc:
        logger.warning("requests + proxy failed for %s: %s", url, exc)

    cs_resp = _fetch_with_cloudscraper(url, headers, timeout)
    if (
        cs_resp is not None
        and cs_resp.status_code == 200
        and _usable_html(cs_resp.text)
    ):
        logger.info("Bot bypass succeeded via cloudscraper for %s", url)
        return cs_resp.text
    if cs_resp is not None:
        logger.warning("cloudscraper returned HTTP %s for %s", cs_resp.status_code, url)

    html = _fetch_with_playwright(url, timeout=timeout)
    if _usable_html(html):
        logger.info("Bot bypass succeeded via playwright for %s", url)
        return html
    if html:
        logger.warning("Playwright returned blocked/error HTML for %s", url)

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
        if not _usable_html(html):
            raise FetchBlockedError(_blocked_message(url, 403))
        return html, "text/html", 200

    if effective_mode == "cffi_proxy":
        html = _fetch_with_cffi(
            url, merged_headers, timeout, proxies=_PROXY_DICT, retries=_PROXY_RETRIES
        )
        if _usable_html(html):
            return html, "text/html", 200
        # Sticky 403: do not immediately repeat the same proxy call in the bypass chain.
        html = _try_bot_bypass(
            url, merged_headers, timeout, skip_cffi_proxy=True
        )
        if html:
            return html, "text/html", 200
        raise FetchBlockedError(_blocked_message(url, 403))

    try:
        response = requests.get(url, timeout=timeout, headers=merged_headers)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
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
