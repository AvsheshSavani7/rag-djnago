from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .dedupe import build_dedupe_key, build_url_hash
from .json_store import load_list, update_list
from .paths import ARTICLE_LINKS_PATH


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_articles(
    is_processed: Optional[bool] = None,
    source_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[dict]:
    rows = load_list(ARTICLE_LINKS_PATH)
    if source_id:
        rows = [r for r in rows if r.get("source_id") == source_id]
    if is_processed is not None:
        rows = [r for r in rows if r.get("is_processed") is is_processed]
    rows = sorted(rows, key=lambda r: r.get("first_seen_at", ""), reverse=True)
    if limit:
        return rows[:limit]
    return rows


def save_if_new(item: dict) -> Tuple[bool, dict]:
    """
    Insert article if dedupe_key is new; otherwise update last_seen_at.
    Returns (is_new, record).
    """
    now = _utc_now_iso()
    detail_url = item.get("detail_url")
    source_id = item["source_id"]

    dedupe_key = build_dedupe_key(source_id, detail_url)
    url_hash = build_url_hash(detail_url)

    def mutator(rows: List[dict]) -> List[dict]:
        for row in rows:
            if row.get("dedupe_key") == dedupe_key:
                row["last_seen_at"] = now
                item["_result"] = (False, row)
                return rows

        record = {
            "source_id": source_id,
            "source_name": item.get("source_name"),
            "source_type": item.get("source_type"),
            "source_url": item.get("source_url"),
            "title": item.get("title"),
            "detail_url": detail_url,
            "published_at": item.get("published_at"),
            "author": item.get("author"),
            "image": item.get("image"),
            "guid": item.get("guid"),
            "url_hash": url_hash,
            "dedupe_key": dedupe_key,
            "is_processed": False,
            "processed_at": None,
            "first_seen_at": now,
            "last_seen_at": now,
            "raw_data": item.get("raw_data") or {},
        }
        rows.append(record)
        item["_result"] = (True, record)
        return rows

    update_list(ARTICLE_LINKS_PATH, mutator)
    return item.pop("_result")


def mark_processed(dedupe_key: str) -> bool:
    now = _utc_now_iso()

    def mutator(rows: List[dict]) -> List[dict]:
        for row in rows:
            if row.get("dedupe_key") == dedupe_key:
                row["is_processed"] = True
                row["processed_at"] = now
                return rows
        return rows

    before = load_list(ARTICLE_LINKS_PATH)
    update_list(ARTICLE_LINKS_PATH, mutator)
    after = load_list(ARTICLE_LINKS_PATH)
    return before != after


def count_articles(is_processed: Optional[bool] = None) -> int:
    return len(list_articles(is_processed=is_processed))
