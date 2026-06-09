from typing import Dict, List, Optional, Tuple

from .block_detector import suggest_container_selectors
from .fetcher import fetch_url
from .field_inferrer import infer_field_selectors
import feedparser

from .rss_parser import infer_element_map, list_rss_field_options, parse_rss_content
from .selector_engine import extract_html_items
from .type_detector import detect_source_type


def _rss_analysis_extras(body: str, element_map: Optional[Dict] = None) -> Dict:
    feed = feedparser.parse(body)
    entries = feed.entries or []
    inferred = infer_element_map(entries)
    field_options = list_rss_field_options(entries[0]) if entries else []
    return {
        "inferred_element_map": inferred,
        "rss_field_options": field_options,
        "preview_items": parse_rss_content(
            body,
            limit=25,
            element_map=element_map or inferred,
        ),
    }


def analyze_html_body(html: str, source_url: str, from_upload: bool = False) -> Dict:
    """Analyze HTML already loaded (e.g. saved from browser when fetch is blocked)."""
    source_type, resolved_url = detect_source_type(html, "text/html", source_url)
    result = {
        "input_url": source_url,
        "source_type": source_type,
        "resolved_url": resolved_url or source_url,
        "status_code": 200,
        "content_type": "text/html",
        "html": html if source_type == "html" else None,
        "rss_content": html if source_type == "rss" else None,
        "container_suggestions": [],
        "preview_items": [],
        "loaded_from_upload": from_upload,
    }
    if source_type == "rss":
        extras = _rss_analysis_extras(html)
        result.update(extras)
    else:
        result["container_suggestions"] = suggest_container_selectors(html)
        if result["container_suggestions"]:
            default_selector = result["container_suggestions"][0]["selector"]
            inferred = infer_field_selectors(html, default_selector)
            result["inferred_selectors"] = inferred
            result["preview_items"] = extract_html_items(
                html=html,
                base_url=source_url,
                selectors=inferred,
                limit=25,
            )
    return result


def analyze_url(url: str) -> Dict:
    """
    Fetch URL, detect type, and return analysis payload for the builder UI.
    """
    body, content_type, status_code = fetch_url(url)
    source_type, resolved_url = detect_source_type(body, content_type, url)

    result = {
        "input_url": url,
        "source_type": source_type,
        "resolved_url": resolved_url or url,
        "status_code": status_code,
        "content_type": content_type,
        "html": body if source_type == "html" else None,
        "rss_content": body if source_type == "rss" else None,
        "container_suggestions": [],
        "preview_items": [],
    }

    if source_type == "rss":
        feed_url = resolved_url or url
        if feed_url != url:
            body, content_type, status_code = fetch_url(feed_url)
            result["rss_content"] = body
            result["resolved_url"] = feed_url
        extras = _rss_analysis_extras(body)
        result.update(extras)
    else:
        result["container_suggestions"] = suggest_container_selectors(body)
        if result["container_suggestions"]:
            default_selector = result["container_suggestions"][0]["selector"]
            inferred = infer_field_selectors(body, default_selector)
            result["inferred_selectors"] = inferred
            result["preview_items"] = extract_html_items(
                html=body,
                base_url=url,
                selectors=inferred,
                limit=25,
            )

    return result


def preview_html_extraction(
    html: str,
    base_url: str,
    selectors: Dict[str, str],
    url_rules: Optional[Dict] = None,
    limit: int = 25,
) -> List[dict]:
    return extract_html_items(
        html=html,
        base_url=base_url,
        selectors=selectors,
        url_rules=url_rules or {},
        limit=limit,
    )
