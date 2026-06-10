from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from dateutil import parser as date_parser

from rss_feeds.models import NewsArticleLink

from .dedupe import build_dedupe_key, build_url_hash, sanitize_http_url


def _parse_dt(value) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return date_parser.parse(str(value))
    except (ValueError, TypeError, OverflowError):
        return None


def is_new_link(item: dict) -> bool:
    detail_url = item.get("detail_url")
    if not detail_url:
        return False
    dedupe_key = build_dedupe_key(item["source_id"], detail_url)
    return NewsArticleLink.objects(dedupe_key=dedupe_key).first() is None


def save_if_new(item: dict) -> Tuple[bool, Optional[dict]]:
    detail_url = sanitize_http_url(item.get("detail_url"))
    if not detail_url:
        return False, None

    source_id = item["source_id"]
    dedupe_key = build_dedupe_key(source_id, detail_url)
    url_hash = build_url_hash(detail_url)
    now = datetime.now(timezone.utc)
    image_url = sanitize_http_url(item.get("image"))
    source_url = sanitize_http_url(item.get("source_url")) or item.get("source_url")

    existing = NewsArticleLink.objects(dedupe_key=dedupe_key).first()
    if existing:
        existing.last_seen_at = now
        existing.save()
        return False, _doc_to_dict(existing)

    doc = NewsArticleLink(
        source_id=source_id,
        source_name=item.get("source_name"),
        source_type=item.get("source_type"),
        source_url=source_url,
        title=item.get("title"),
        detail_url=detail_url,
        published_at=_parse_dt(item.get("published_at")),
        description=(item.get("description") or "")[:4000] or None,
        author=item.get("author"),
        image=image_url,
        guid=item.get("guid"),
        url_hash=url_hash,
        dedupe_key=dedupe_key,
        is_processed=False,
        first_seen_at=now,
        last_seen_at=now,
        raw_data=item.get("raw_data") or {},
    )
    doc.save()
    return True, _doc_to_dict(doc)


def mark_articles_processed(dedupe_keys: List[str]) -> int:
    """Mark news_article_links rows processed after production pipeline runs (GO LIVE flow)."""
    if not dedupe_keys:
        return 0
    now = datetime.now(timezone.utc)
    updated = 0
    for key in dedupe_keys:
        doc = NewsArticleLink.objects(dedupe_key=key).first()
        if not doc:
            continue
        doc.is_processed = True
        doc.processed_at = now
        doc.save()
        updated += 1
    return updated


def _doc_to_dict(doc: NewsArticleLink) -> dict:
    return {
        "source_id": doc.source_id,
        "source_name": doc.source_name,
        "title": doc.title,
        "detail_url": doc.detail_url,
        "published_at": doc.published_at,
        "description": doc.description,
        "author": doc.author,
        "image": doc.image,
        "is_processed": doc.is_processed,
        "dedupe_key": doc.dedupe_key,
        "first_seen_at": doc.first_seen_at.isoformat() if doc.first_seen_at else None,
    }
