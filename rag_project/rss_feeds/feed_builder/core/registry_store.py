from datetime import datetime, timezone
from typing import Dict, List, Optional

from .dedupe import slugify_source_id
from .json_store import load_list, save_list, update_list
from .paths import FEEDS_REGISTRY_PATH


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_feeds(active_only: bool = False) -> List[dict]:
    feeds = load_list(FEEDS_REGISTRY_PATH)
    if active_only:
        return [f for f in feeds if f.get("is_active", True)]
    return feeds


def get_feed(source_id: str) -> Optional[dict]:
    for feed in list_feeds():
        if feed.get("source_id") == source_id:
            return feed
    return None


def save_feed(feed: dict) -> dict:
    source_id = feed.get("source_id") or slugify_source_id(feed.get("source_name", "feed"))
    feed["source_id"] = source_id
    now = _utc_now_iso()

    def mutator(rows: List[dict]) -> List[dict]:
        updated = False
        for idx, row in enumerate(rows):
            if row.get("source_id") == source_id:
                feed.setdefault("created_at", row.get("created_at", now))
                feed["updated_at"] = now
                rows[idx] = feed
                updated = True
                break
        if not updated:
            feed.setdefault("created_at", now)
            feed["updated_at"] = now
            rows.append(feed)
        return rows

    update_list(FEEDS_REGISTRY_PATH, mutator)
    return feed


def delete_feed(source_id: str) -> bool:
    feeds = list_feeds()
    new_feeds = [f for f in feeds if f.get("source_id") != source_id]
    if len(new_feeds) == len(feeds):
        return False
    save_list(FEEDS_REGISTRY_PATH, new_feeds)
    return True


def set_feed_active(source_id: str, is_active: bool) -> Optional[dict]:
    feed = get_feed(source_id)
    if not feed:
        return None
    feed["is_active"] = is_active
    return save_feed(feed)


def update_feed_runtime(source_id: str, **fields) -> Optional[dict]:
    feed = get_feed(source_id)
    if not feed:
        return None
    feed.update(fields)
    return save_feed(feed)
