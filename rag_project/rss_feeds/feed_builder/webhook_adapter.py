"""
Build RSS.app-shaped webhook payloads from feed_builder scan results.

GO LIVE: scanner calls process_feed_builder_newswire_articles() which uses this module.
source_name in news_source_configs must equal feed.title keys in
FEED_TITLE_DISPLAY_NAMES / FEED_TITLE_DISPLAY_NAME_2.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

def _format_date_published(value: Any) -> str:
    """ISO 8601 string required by FeedItemCreateSerializer."""
    if value is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    if isinstance(value, str) and value.strip():
        normalized = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        except ValueError:
            pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def article_link_to_webhook_item(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map news_article_links row → process_webhook_payload items_new entry."""
    from rss_feeds.services import _normalize_thumbnail

    author = (row.get("author") or "").strip()
    return {
        "url": row["detail_url"],
        "title": ((row.get("title") or "Untitled").strip())[:500],
        "description_text": ((row.get("description") or "")[:4000]),
        "thumbnail": _normalize_thumbnail(row.get("image")),
        "date_published": _format_date_published(row.get("published_at")),
        "authors": [{"name": author}] if author else [],
    }


def build_feed_builder_webhook_payload(
    feed_config: Dict[str, Any],
    new_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    One payload per newswire (all new items in items_new).

    feed_config: news_source_configs document (source_id, source_name, source_url, …)
    new_items: rows from save_if_new / news_article_links
    """
    source_id = feed_config["source_id"]
    source_url = feed_config["source_url"]
    source_name = feed_config.get("source_name") or source_id

    return {
        "id": f"feed-builder-{source_id}-{datetime.now(timezone.utc).isoformat()}",
        "type": "feed_update",
        "feed": {
            "id": source_id,
            "title": source_name,
            "source_url": source_url,
            # No RSS.app — listing URL used for feed upsert when GO LIVE source_url flow is enabled.
            "rss_feed_url": source_url,
            "description": "",
            "icon": "",
        },
        "data": {
            "items_new": [article_link_to_webhook_item(row) for row in new_items],
            "items_changed": [],
        },
    }
