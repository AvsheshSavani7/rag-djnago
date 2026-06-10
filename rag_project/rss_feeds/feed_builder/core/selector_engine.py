from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag
from dateutil import parser as date_parser

from .dedupe import sanitize_http_url
from .field_inferrer import clean_title_text


def _select_text(container: Tag, selector: Optional[str]) -> Optional[str]:
    if not selector:
        return None
    el = container.select_one(selector)
    if not el:
        return None
    return el.get_text(" ", strip=True) or None


def _select_title(container: Tag, selector: Optional[str]) -> Optional[str]:
    if not selector:
        return None
    el = container.select_one(selector)
    if not el:
        return None

    date_el = el.select_one("small, time, .date, .timestamp")
    date_text = date_el.get_text(" ", strip=True) if date_el else None
    full_text = el.get_text(" ", strip=True)
    return clean_title_text(full_text, date_text)


def _select_attr(container: Tag, selector: Optional[str], attr: str) -> Optional[str]:
    if not selector:
        return None
    el = container.select_one(selector)
    if not el:
        return None
    value = el.get(attr)
    if value:
        return str(value).strip()
    return None


def _select_link(container: Tag, selector: Optional[str], base_url: str) -> Optional[str]:
    if not selector:
        link = container.find("a", href=True)
        if link:
            return urljoin(base_url, link["href"])
        return None

    el = container.select_one(selector)
    if not el:
        return None

    if el.name == "a" and el.get("href"):
        return urljoin(base_url, el["href"])

    nested = el.find("a", href=True)
    if nested:
        return urljoin(base_url, nested["href"])

    for attr in ("href", "data-href"):
        if el.get(attr):
            return urljoin(base_url, el[attr])

    return None


def _parse_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        dt = date_parser.parse(value, fuzzy=True)
        if dt.tzinfo is None:
            return dt.isoformat()
        return dt.astimezone().isoformat()
    except (ValueError, TypeError, OverflowError):
        return None


def _select_published_at(container: Tag, selector: Optional[str]) -> Optional[str]:
    """Prefer <time> inside the matched block; fall back to raw text if parse fails."""
    if not selector:
        return None
    el = container.select_one(selector)
    if not el:
        return None

    time_el = el if el.name == "time" else el.find("time")
    if time_el:
        raw = (time_el.get("datetime") or "").strip() or time_el.get_text(" ", strip=True)
    else:
        raw = (el.get("datetime") or "").strip() or el.get_text(" ", strip=True)

    if not raw:
        return None

    parsed = _parse_date(raw)
    if parsed:
        return parsed

    # Keep human-readable date in preview when parser fails (e.g. French month names).
    for line in raw.splitlines():
        line = line.strip()
        if line:
            parsed_line = _parse_date(line)
            return parsed_line or line
    return raw


def _passes_url_rules(detail_url: str, url_rules: Optional[Dict]) -> bool:
    if not detail_url:
        return False

    rules = url_rules or {}
    must_contain = rules.get("must_contain") or []
    exclude = rules.get("exclude") or []

    lower_url = detail_url.lower()
    for token in must_contain:
        if token.lower() not in lower_url:
            return False

    for token in exclude:
        if token.lower() in lower_url:
            return False

    return True


def extract_html_items(
    html: str,
    base_url: str,
    selectors: Dict[str, str],
    url_rules: Optional[Dict] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    container_selector = selectors.get("container") or "body"
    try:
        containers = soup.select(container_selector)
    except Exception:
        # Selector contains unsupported pseudo-classes or syntax (e.g. Tailwind :flex).
        containers = soup.select("body")

    items: List[Dict[str, Any]] = []
    seen_urls = set()

    for container in containers:
        detail_url = _select_link(container, selectors.get("detail_url"), base_url)
        if not detail_url or detail_url in seen_urls:
            continue
        if not _passes_url_rules(detail_url, url_rules):
            continue

        title = _select_title(container, selectors.get("title"))
        if not title:
            title = _select_text(container, selectors.get("detail_url") or "a")
            date_for_strip = _select_text(container, selectors.get("published_at"))
            title = clean_title_text(title, date_for_strip)

        description = _select_text(container, selectors.get("description"))
        author = _select_text(container, selectors.get("author"))
        image = _select_attr(container, selectors.get("image"), "src")
        if not image:
            image = _select_attr(container, selectors.get("image"), "data-src")
        image = sanitize_http_url(image)

        item = {
            "title": title,
            "detail_url": detail_url,
            "published_at": _select_published_at(container, selectors.get("published_at")),
            "description": description,
            "author": author,
            "image": image,
            "raw_data": {
                "container_html": str(container)[:500],
            },
        }
        items.append(item)
        seen_urls.add(detail_url)

        if limit and len(items) >= limit:
            break

    return items
