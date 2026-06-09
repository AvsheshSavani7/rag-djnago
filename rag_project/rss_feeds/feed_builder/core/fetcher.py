import logging
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# Sites known to use Akamai / heavy bot protection on listing pages.
BOT_PROTECTED_HINTS = (
    "businesswire.com",
    "bloomberg.com",
)


class FetchBlockedError(Exception):
    """Raised when a site blocks automated fetching (403/Access Denied)."""


def _is_access_denied(body: str) -> bool:
    lower = (body or "").lower()
    return "access denied" in lower or "edgesuite.net" in lower


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


def fetch_url(
    url: str,
    timeout: int = 30,
    headers: Optional[dict] = None,
    fetch_mode: str = "auto",
) -> Tuple[str, str, int]:
    """
    Fetch a URL and return (body_text, content_type, status_code).

    fetch_mode: auto | requests | playwright
      auto = requests → cloudscraper (403) → playwright (403)
    """
    merged_headers = {**BROWSER_HEADERS, **(headers or {})}

    if fetch_mode == "playwright":
        html = _fetch_with_playwright(url, timeout=timeout)
        if not html or _is_access_denied(html):
            raise FetchBlockedError(_blocked_message(url, 403))
        return html, "text/html", 200

    response = requests.get(url, timeout=timeout, headers=merged_headers)

    if response.status_code in (403, 429, 503) and fetch_mode in ("auto", "requests"):
        cs_resp = _fetch_with_cloudscraper(url, merged_headers, timeout)
        if cs_resp is not None and cs_resp.status_code == 200 and not _is_access_denied(cs_resp.text):
            response = cs_resp
        elif fetch_mode == "auto":
            html = _fetch_with_playwright(url, timeout=timeout)
            if html and not _is_access_denied(html):
                return html, "text/html", 200

    if response.status_code == 403 or _is_access_denied(response.text):
        raise FetchBlockedError(_blocked_message(url, response.status_code))

    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    response.encoding = response.encoding or "utf-8"
    return response.text, content_type, response.status_code
