from datetime import datetime
from typing import Dict, List, Optional

from rss_feeds.models import NewsSourceConfig


def list_feeds(active_only: bool = False) -> List[dict]:
    qs = NewsSourceConfig.objects
    if active_only:
        qs = qs.filter(is_active=True)
    return [doc.to_dict() for doc in qs.order_by("-updated_at")]


def get_feed(source_id: str) -> Optional[dict]:
    doc = NewsSourceConfig.objects(source_id=source_id).first()
    return doc.to_dict() if doc else None


def get_feed_by_url(source_url: str) -> Optional[dict]:
    doc = NewsSourceConfig.objects(source_url=source_url).first()
    return doc.to_dict() if doc else None


def save_feed(feed: dict) -> dict:
    """
    Create or update config. Upserts by source_url (then updates source_id/name/etc).
    """
    source_url = feed["source_url"]
    existing = NewsSourceConfig.objects(source_url=source_url).first()

    if existing:
        doc = existing
    else:
        doc = NewsSourceConfig(source_url=source_url)

    doc.source_id = feed["source_id"]
    doc.source_name = feed["source_name"]
    doc.source_type = feed.get("source_type", "html")
    doc.fetch_mode = feed.get("fetch_mode", "requests")
    doc.selectors = feed.get("selectors") or {}
    doc.element_map = feed.get("element_map") or {}
    doc.url_rules = feed.get("url_rules") or {}
    doc.is_active = feed.get("is_active", True)
    doc.poll_interval_minutes = int(feed.get("poll_interval_minutes") or 10)
    doc.save()
    return doc.to_dict()


def delete_feed(source_id: str) -> bool:
    doc = NewsSourceConfig.objects(source_id=source_id).first()
    if not doc:
        return False
    doc.delete()
    return True


def set_feed_active(source_id: str, is_active: bool) -> Optional[dict]:
    doc = NewsSourceConfig.objects(source_id=source_id).first()
    if not doc:
        return None
    doc.is_active = is_active
    doc.save()
    return doc.to_dict()


def update_feed_runtime(source_id: str, **fields) -> Optional[dict]:
    doc = NewsSourceConfig.objects(source_id=source_id).first()
    if not doc:
        return None

    for key, value in fields.items():
        if key.endswith("_at") and isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass
        if hasattr(doc, key):
            setattr(doc, key, value)

    doc.save()
    return doc.to_dict()
