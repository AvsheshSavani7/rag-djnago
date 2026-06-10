import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from .fetcher import fetch_url
from .mongo_article_store import is_new_link, save_if_new
from .mongo_config_store import list_feeds, update_feed_runtime
from .rss_parser import parse_rss_content
from .selector_engine import extract_html_items

logger = logging.getLogger(__name__)


_IST = timezone(timedelta(hours=5, minutes=30))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ist_now() -> str:
    return datetime.now(_IST).strftime("%Y-%m-%d %I:%M:%S %p IST")


def _title_slug(title: str, max_chars: int = 20) -> str:
    """Return a filesystem-safe slug of the first `max_chars` chars of a title."""
    slug = re.sub(r"[^\w\s-]", "", (title or "").lower())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug[:max_chars] or "untitled"


def _published_ist(published_at: str) -> str:
    """Convert an ISO published_at string to a readable IST label, or return as-is."""
    if not published_at:
        return "-"
    try:
        dt = datetime.fromisoformat(published_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_IST).strftime("%Y-%m-%d %I:%M:%S %p IST")
    except Exception:
        return published_at


def validate_feed_config(feed: dict) -> Optional[str]:
    """Return an error message when config is not ready to scan."""
    if not feed.get("source_url"):
        return "missing source_url"
    if feed.get("source_type") == "html" and not (feed.get("selectors") or {}).get("container"):
        return "HTML feed missing selectors.container — build and save config in Streamlit first"
    return None


def extract_items_from_feed(feed: dict, limit: Optional[int] = None) -> List[dict]:
    source_type = feed.get("source_type", "html")
    source_url = feed.get("source_url", "")
    fetch_mode = feed.get("fetch_mode") or "auto"

    if source_type == "rss":
        body, _, _ = fetch_url(source_url, fetch_mode=fetch_mode)
        raw_items = parse_rss_content(
            body,
            source_url=source_url,
            limit=limit,
            element_map=feed.get("element_map") or {},
        )
    else:
        body, _, _ = fetch_url(source_url, fetch_mode=fetch_mode)
        raw_items = extract_html_items(
            html=body,
            base_url=source_url,
            selectors=feed.get("selectors") or {},
            url_rules=feed.get("url_rules") or {},
            limit=limit,
        )

    enriched = []
    for item in raw_items:
        enriched.append(
            {
                **item,
                "source_id": feed["source_id"],
                "source_name": feed.get("source_name"),
                "source_type": source_type,
                "source_url": source_url,
            }
        )
    return enriched


def scan_feed(feed: dict, *, dry_run: bool = False) -> Dict:
    source_id = feed["source_id"]

    try:
        from core.pipeline_logger import start_pipeline, RSS
        start_pipeline(RSS, accession=source_id, doc_type="NEWSWIRE")
    except Exception:
        pass  # pipeline context is optional; don't let it block scanning

    result = {
        "source_id": source_id,
        "source_name": feed.get("source_name"),
        "source_type": feed.get("source_type"),
        "found": 0,
        "new": 0,
        "skipped": 0,
        "new_items": [],
        "error": None,
    }

    config_error = validate_feed_config(feed)
    if config_error:
        result["error"] = config_error
        update_feed_runtime(
            source_id,
            last_checked_at=_utc_now_iso(),
            last_error=config_error,
            consecutive_failures=int(
                feed.get("consecutive_failures") or 0) + 1,
        )
        return result

    try:
        items = extract_items_from_feed(feed)
        result["found"] = len(items)

        for item in items:
            if dry_run:
                if is_new_link(item):
                    result["new"] += 1
                    entry = {
                        "title": item.get("title"),
                        "detail_url": item.get("detail_url"),
                        "published_at": item.get("published_at"),
                    }
                    result["new_items"].append(entry)
                    try:
                        from core.pipeline_logger import start_pipeline, RSS
                        start_pipeline(
                            RSS,
                            accession=_title_slug(entry["title"]),
                            doc_type="NEWSWIRE",
                        )
                    except Exception:
                        pass
                    logger.info(
                        "[dry-run] NEW item | source=%s | title=%s | url=%s | published=%s | scanned=%s",
                        source_id,
                        entry["title"],
                        entry["detail_url"],
                        _published_ist(entry["published_at"]),
                        _ist_now(),
                    )
                continue

            try:
                is_new, saved = save_if_new(item)
            except Exception as exc:
                result["skipped"] += 1
                logger.warning(
                    "Skip item for %s (%s): %s",
                    source_id,
                    item.get("detail_url"),
                    exc,
                )
                continue
            if not saved and not is_new:
                result["skipped"] += 1
                continue
            if is_new:
                result["new"] += 1
                result["new_items"].append(saved)
                _item = saved or item
                try:
                    from core.pipeline_logger import start_pipeline, RSS
                    start_pipeline(
                        RSS,
                        accession=_title_slug(_item.get("title")),
                        doc_type="NEWSWIRE",
                    )
                except Exception:
                    pass
                logger.info(
                    "NEW item saved | source=%s | title=%s | url=%s | published=%s | scanned=%s",
                    source_id,
                    _item.get("title"),
                    _item.get("detail_url"),
                    _published_ist(_item.get("published_at")),
                    _ist_now(),
                )

        if not dry_run and result["new_items"]:
            # ===== DEBUG: digest email (N8N_WEBHOOK_ONLY_ME) so you can see what was found =====
            try:
                from rss_feeds.services import send_feed_builder_newswire_test_email

                result["email_sent"] = send_feed_builder_newswire_test_email(
                    source_id=source_id,
                    source_name=feed.get("source_name"),
                    source_url=feed.get("source_url", ""),
                    new_items=result["new_items"],
                )
            except Exception as exc:
                logger.warning(
                    "Feed builder debug email failed for %s: %s", source_id, exc
                )
                result["email_sent"] = False
            # ===== END DEBUG =====

            # ===== GO LIVE (production): merger classify, save feed_items, production emails =====
            # Replaces RSS.app webhook entirely.
            # source_url upsert is also active in RSSFeedService.create_or_update_feed (services.py).
            try:
                from rss_feeds.services import process_feed_builder_newswire_articles

                result["pipeline_result"] = process_feed_builder_newswire_articles(
                    feed_config=feed,
                    new_items=result["new_items"],
                )
                result["pipeline_ok"] = bool(
                    (result.get("pipeline_result") or {}).get("success")
                )
            except Exception as exc:
                logger.exception(
                    "Feed builder production pipeline failed for %s", source_id
                )
                result["pipeline_ok"] = False
                result["pipeline_error"] = str(exc)
            # ===== END GO LIVE =====

        logger.info(
            "Scan complete | source=%s | found=%d | new=%d | skipped=%d | at=%s",
            source_id,
            result["found"],
            result["new"],
            result.get("skipped", 0),
            _ist_now(),
        )

        if not dry_run:
            update_feed_runtime(
                source_id,
                last_checked_at=_utc_now_iso(),
                last_success_at=_utc_now_iso(),
                last_error=None,
                consecutive_failures=0,
            )
    except Exception as exc:
        logger.exception("Feed scan failed for %s", source_id)
        result["error"] = str(exc)
        failures = int(feed.get("consecutive_failures") or 0) + 1
        update_feed_runtime(
            source_id,
            last_checked_at=_utc_now_iso(),
            last_error=str(exc),
            consecutive_failures=failures,
        )

    return result


def scan_feed_by_id(source_id: str, *, dry_run: bool = False) -> Dict:
    from .mongo_config_store import get_feed

    feed = get_feed(source_id)
    if not feed:
        return {
            "source_id": source_id,
            "source_name": None,
            "found": 0,
            "new": 0,
            "new_items": [],
            "error": f"unknown source_id: {source_id}",
        }
    return scan_feed(feed, dry_run=dry_run)


_SCAN_MAX_WORKERS = 5  # keep low to avoid overwhelming proxy / rate limits


def _scan_feed_task(feed: dict, dry_run: bool) -> Dict:
    """Wrapper run in each thread: log before delegating to scan_feed."""
    logger.info(
        "Scanning %s (%s) — %s",
        feed.get("source_id"),
        feed.get("source_type"),
        feed.get("source_url"),
    )
    return scan_feed(feed, dry_run=dry_run)


def scan_all_feeds(
    active_only: bool = True,
    *,
    dry_run: bool = False,
    source_ids: Optional[List[str]] = None,
) -> Dict:
    feeds = list_feeds(active_only=active_only)
    if source_ids:
        wanted = set(source_ids)
        feeds = [f for f in feeds if f.get("source_id") in wanted]

    summary = {
        "scanned_at": _utc_now_iso(),
        "feeds_scanned": 0,
        "feeds_total": len(feeds),
        "total_found": 0,
        "total_new": 0,
        "results": [],
        "errors": [],
    }
    _lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=_SCAN_MAX_WORKERS) as executor:
        futures = {
            executor.submit(_scan_feed_task, feed, dry_run): feed
            for feed in feeds
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                feed = futures[future]
                result = {
                    "source_id": feed.get("source_id"),
                    "source_name": feed.get("source_name"),
                    "source_type": feed.get("source_type"),
                    "found": 0,
                    "new": 0,
                    "skipped": 0,
                    "new_items": [],
                    "error": str(exc),
                }
                logger.exception(
                    "Unhandled error scanning %s", feed.get("source_id")
                )

            with _lock:
                summary["feeds_scanned"] += 1
                summary["total_found"] += result["found"]
                summary["total_new"] += result["new"]
                summary["results"].append(result)
                if result["error"]:
                    summary["errors"].append(
                        {"source_id": result["source_id"], "error": result["error"]}
                    )

    if not dry_run and summary["errors"]:
        try:
            from rss_feeds.services import send_feed_builder_pipeline_error_email
            send_feed_builder_pipeline_error_email(summary["errors"])
        except Exception as exc:
            logger.warning("Pipeline error email failed: %s", exc)

    return summary
