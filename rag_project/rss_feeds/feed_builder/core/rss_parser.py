from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import feedparser
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
import re

from .dedupe import sanitize_http_url

# Output field -> feedparser entry keys tried in order (when element_map is empty).
DEFAULT_ELEMENT_CHAINS: Dict[str, List[str]] = {
    "title": ["title"],
    "detail_url": ["link", "id"],
    "published_at": ["published", "updated", "created"],
    "description": ["summary", "description", "subtitle", "content"],
    "author": ["author", "authors"],
    "image": ["media_thumbnail", "media_content", "enclosures"],
    "guid": ["id", "guid"],
}

OUTPUT_FIELDS = list(DEFAULT_ELEMENT_CHAINS.keys())

ElementMapValue = Union[str, List[str]]


def _normalize_chain(value: ElementMapValue) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if value:
        return [str(value).strip()]
    return []


def resolve_element_chains(element_map: Optional[Dict[str, ElementMapValue]] = None) -> Dict[str, List[str]]:
    """Merge per-feed element_map with defaults."""
    element_map = element_map or {}
    chains: Dict[str, List[str]] = {}
    for field, defaults in DEFAULT_ELEMENT_CHAINS.items():
        override = element_map.get(field)
        if override:
            chains[field] = _normalize_chain(override)
        else:
            chains[field] = list(defaults)
    return chains


def list_rss_field_options(entry: dict) -> List[str]:
    """Keys available on a feedparser entry for the mapping UI."""
    options = set(DEFAULT_ELEMENT_CHAINS.keys())
    for key in entry:
        if entry.get(key) not in (None, "", [], {}):
            options.add(key)
    return sorted(options)


def infer_element_map(entries: List[dict]) -> Dict[str, str]:
    """Pick the first working RSS key per output field from the first entry."""
    if not entries:
        return {}
    entry = entries[0]
    inferred: Dict[str, str] = {}
    for field, chain in DEFAULT_ELEMENT_CHAINS.items():
        for key in chain:
            if _extract_raw_value(entry, key) is not None:
                inferred[field] = key
                break
    return inferred


def parse_rss_content(
    content: str,
    source_url: str = "",
    limit: Optional[int] = None,
    element_map: Optional[Dict[str, ElementMapValue]] = None,
) -> List[Dict[str, Any]]:
    feed = feedparser.parse(content)
    chains = resolve_element_chains(element_map)

    items: List[Dict[str, Any]] = []
    for entry in feed.entries:
        detail_url = _extract_field(
            entry, "detail_url", chains.get("detail_url", []))
        if not detail_url:
            continue

        item = {
            "title": _extract_field(entry, "title", chains.get("title", [])),
            "detail_url": detail_url,
            "published_at": _extract_field(entry, "published_at", chains.get("published_at", [])),
            "description": _extract_field(entry, "description", chains.get("description", [])),
            "author": _extract_field(entry, "author", chains.get("author", [])),
            "image": _extract_field(entry, "image", chains.get("image", [])),
            "guid": _extract_field(entry, "guid", chains.get("guid", [])),
            "raw_data": dict(entry),
        }
        items.append(item)

        if limit and len(items) >= limit:
            break

    return items


def _extract_field(entry: dict, field: str, keys: List[str]) -> Optional[str]:
    for key in keys:
        value = _extract_raw_value(entry, key)
        if value is None:
            continue
        if field == "published_at":
            return _parse_date_value(value, entry, key)
        if field == "description":
            return _html_to_plain_text(str(value))
        if field == "image":
            return sanitize_http_url(str(value))
        if field in ("title", "detail_url", "guid", "author"):
            return str(value).strip() or None
        return str(value).strip() or None
    return None


def _extract_raw_value(entry: dict, key: str) -> Any:
    if key == "content":
        blocks = entry.get("content")
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, dict) and block.get("value"):
                    return block.get("value")
        return None

    if key == "authors":
        authors = entry.get("authors") or []
        if authors and isinstance(authors[0], dict):
            name = authors[0].get("name")
            return name if name else None
        return None

    if key == "media_thumbnail":
        media = entry.get("media_thumbnail") or []
        if isinstance(media, list) and media:
            return media[0].get("url")
        return None

    if key == "media_content":
        media = entry.get("media_content") or []
        if isinstance(media, list) and media:
            return media[0].get("url")
        return None

    if key == "enclosures":
        for enc in entry.get("enclosures") or []:
            if enc.get("type", "").startswith("image") and enc.get("href"):
                return enc["href"]
        return None

    value = entry.get(key)
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, (list, dict)) and not value:
        return None
    return value


def _parse_date_value(value: Any, entry: dict, key: str) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        try:
            return date_parser.parse(value).isoformat()
        except (ValueError, TypeError, OverflowError):
            pass

    if key == "published" and entry.get("published_parsed"):
        return _parsed_tuple_to_iso(entry["published_parsed"])
    if key == "updated" and entry.get("updated_parsed"):
        return _parsed_tuple_to_iso(entry["updated_parsed"])
    if key == "created" and entry.get("created_parsed"):
        return _parsed_tuple_to_iso(entry["created_parsed"])

    for parsed_key in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed = entry.get(parsed_key)
        if parsed:
            iso = _parsed_tuple_to_iso(parsed)
            if iso:
                return iso
    return None


def _parsed_tuple_to_iso(parsed: Any) -> Optional[str]:
    try:
        return datetime(*parsed[:6]).isoformat()
    except (TypeError, ValueError):
        return None


def _html_to_plain_text(value: str) -> str:
    """RSS descriptions often contain HTML (e.g. GlobeNewswire <p><b>...</b>)."""
    text = (value or "").strip()
    if not text:
        return ""
    if "<" not in text or ">" not in text:
        return re.sub(r"\s+", " ", text).strip()

    soup = BeautifulSoup(text, "html.parser")
    plain = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", plain).strip()
