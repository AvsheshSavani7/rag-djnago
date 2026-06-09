import re
from typing import Optional, Tuple
from urllib.parse import urljoin

from bs4 import BeautifulSoup

RSS_CONTENT_TYPES = (
    "application/rss+xml",
    "application/atom+xml",
    "application/xml",
    "text/xml",
)


def _looks_like_feed_xml(text: str) -> bool:
    snippet = text[:4000].lower()
    return "<rss" in snippet or "<feed" in snippet or "<rdf:rdf" in snippet


def detect_source_type(
    body: str,
    content_type: str = "",
    page_url: str = "",
) -> Tuple[str, Optional[str]]:
    """
    Return (source_type, resolved_url).
    source_type is 'rss' or 'html'.
    For HTML pages with <link rel="alternate" type="application/rss+xml">,
  returns rss with the discovered feed URL.
    """
    ct = (content_type or "").lower()
    if any(token in ct for token in RSS_CONTENT_TYPES) and _looks_like_feed_xml(body):
        return "rss", page_url

    if _looks_like_feed_xml(body):
        return "rss", page_url

    if "<html" in body.lower() or "<body" in body.lower():
        soup = BeautifulSoup(body, "html.parser")
        for link in soup.find_all("link", rel=True):
            rel = " ".join(link.get("rel", [])).lower()
            link_type = (link.get("type") or "").lower()
            href = link.get("href")
            if not href:
                continue
            if "alternate" in rel and (
                "rss" in link_type or "atom" in link_type or "xml" in link_type
            ):
                return "rss", urljoin(page_url, href)

    return "html", page_url
