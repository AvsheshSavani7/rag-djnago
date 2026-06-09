import hashlib
import re
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


STRIP_QUERY_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
}


def sanitize_http_url(value: Optional[str], max_length: int = 2000) -> Optional[str]:
    """Keep only http(s) URLs suitable for MongoEngine URLField (rejects data: URIs, author names, etc.)."""
    if not value:
        return None
    url = str(value).strip()
    if not url or url.lower().startswith("data:"):
        return None
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    return url[:max_length]


def normalize_url(url: str, strip_tracking: bool = True) -> str:
    if not url:
        return ""

    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"

    query = parsed.query
    if strip_tracking and query:
        pairs = [
            (k, v)
            for k, v in parse_qsl(query, keep_blank_values=True)
            if k.lower() not in STRIP_QUERY_PARAMS
        ]
        query = urlencode(pairs)

    normalized = urlunparse((scheme, netloc, path, "", query, ""))
    return normalized.lower()


def build_url_hash(detail_url: str) -> str:
    normalized = normalize_url(detail_url)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_dedupe_key(source_id: str, detail_url: str) -> str:
    return f"{source_id}:{build_url_hash(detail_url)}"


def slugify_source_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "feed"
