import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .fetcher import fetch_url
from .mongo_article_store import is_new_link, save_if_new
from .mongo_config_store import list_feeds, update_feed_runtime
from .rss_parser import parse_rss_content
from .selector_engine import extract_html_items

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    if source_type == "rss":
        body, _, _ = fetch_url(source_url)
        raw_items = parse_rss_content(
            body,
            source_url=source_url,
            limit=limit,
            element_map=feed.get("element_map") or {},
        )
    else:
        body, _, _ = fetch_url(source_url)
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
                    result["new_items"].append(
                        {
                            "title": item.get("title"),
                            "detail_url": item.get("detail_url"),
                            "published_at": item.get("published_at"),
                        }
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

        if not dry_run and result["new_items"]:
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
                    "Feed builder test email failed for %s: %s", source_id, exc
                )
                result["email_sent"] = False

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

    for feed in feeds:
        logger.info(
            "Scanning %s (%s) — %s",
            feed.get("source_id"),
            feed.get("source_type"),
            feed.get("source_url"),
        )
        result = scan_feed(feed, dry_run=dry_run)
        summary["feeds_scanned"] += 1
        summary["total_found"] += result["found"]
        summary["total_new"] += result["new"]
        summary["results"].append(result)
        if result["error"]:
            summary["errors"].append(
                {"source_id": result["source_id"], "error": result["error"]}
            )

    return summary
