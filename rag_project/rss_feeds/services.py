from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import os
import requests

from .models import Feed, FeedItem, Author
from .serializers import FeedItemCreateSerializer
from .websocket_service import RSSWebSocketService
from .email_templates import (
    generate_rss_feed_item_email_html,
    generate_rss_feed_item_email_html_flow2,
    generate_feed_builder_newswire_email_html,
    feed_builder_display_name,
    rss_subject_uses_client_webhook,
    FEED_TITLE_DISPLAY_NAMES,
    FEED_TITLE_DISPLAY_NAME_2,
)
from .merger_news_classifier import (
    get_deals_record_string,
    resolve_rss_item_flow,
    classify_feed_item_by_title_description,
    get_deal_info_for_email,
)
from core.exception_email import send_exception_email
from core.pipeline_logger import RSS
from .rss_error_collector import RSSArticleErrorRegistry, record_rss_error
from sec_rss_parser.sec_summarizers.filing_router import route_and_summarize
from sec_rss_parser.utils_8k import get_deal_tickers
from sec_rss_parser.email_service.email_dispatch_service import send_direct_email, send_report_email

logger = logging.getLogger(__name__)

# Substrings in route_and_summarize failures that mean "article isn't M&A" — log only, no error email.
_NON_MA_SUMMARY_SKIP_MARKERS = (
    "not a merger",
    "not an m&a",
    "no m&a transaction",
    "financing round",
    "series b extension",
    "does not match the content",
    "does not relate to any provided deal context",
    "this press release describes",
)


def _is_skippable_non_ma_summary_error(exc: BaseException) -> bool:
    """True when summarizer refused or skipped a non-M&A / deal-mismatch article."""
    try:
        from sec_rss_parser.sec_summarizers.PRNewswire_summary import (
            NotMergerPressReleaseError,
        )
        if isinstance(exc, NotMergerPressReleaseError):
            return True
    except ImportError:
        pass

    msg = str(exc).lower()
    if "summarize: empty response" in msg:
        return True
    if "summarize: invalid json:" in msg:
        return any(marker in msg for marker in _NON_MA_SUMMARY_SKIP_MARKERS)
    return False


# N8N webhook for RSS feed update emails (testing – same as sec_rss_parser)
N8N_WEBHOOK_ONLY_ME = os.environ.get(
    "N8N_WEBHOOK_ONLY_ME",
    "https://n8n.arbintel.cloud/webhook/d50502ea-6746-4d4b-8dfe-fb7bd71e0a1f"
)
N8N_WEKHOOK_INTERNAL_WITH_JOSH = os.environ.get(
    "N8N_WEKHOOK_INTERNAL_WITH_JOSH",
    "https://n8n.arbintel.cloud/webhook/b3007d21-6845-47b5-aece-7b26583758bc"
)

N8N_WEBHOOK_SEND_TO_ALL = os.environ.get(
    "N8N_WEBHOOK_SEND_TO_ALL",
    "https://n8n.arbintel.cloud/webhook/3ff1b0ea-7114-4dda-940e-95ce81e08017",
)

# Temporary: email on feed_builder scan new articles (disable after testing).
FEED_BUILDER_SCAN_TEST_EMAILS = os.environ.get(
    "FEED_BUILDER_SCAN_TEST_EMAILS", "true"
).lower() in ("1", "true", "yes")


def _parse_date_published(value: Any) -> datetime:
    """Parse date_published from RSS.app webhook (ISO 8601 string) to datetime."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        # RSS.app sends e.g. "2024-09-12T17:52:59.000Z"
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    raise ValueError(f"Invalid date_published: {value!r}")


def _normalize_thumbnail(value: Any) -> Optional[str]:
    """Return a valid thumbnail URL or None. URLField rejects empty string."""
    if value is None:
        return None
    s = (value or "").strip()
    if not s:
        return None
    if s.startswith("http://") or s.startswith("https://"):
        return s
    return None


def _send_rss_feed_email_via_webhook(
    webhook_url: str,
    subject: str,
    html_email: str,
    feed_title: str,
    items_count: int,
    feed_source_url: str = "",
) -> bool:
    """Send RSS feed update email via N8N webhook (same payload shape as sec_rss_parser)."""
    try:
        payload = {
            "subject": subject,
            "html": html_email,
            "feed_title": feed_title,
            "items_count": items_count,
            "feed_source_url": feed_source_url,
        }
        logger.info(
            "Sending RSS feed update email via webhook: %s", webhook_url)
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        logger.info(
            "RSS feed update email sent successfully (status=%s)", response.status_code
        )
        return True
    except requests.exceptions.RequestException as e:
        logger.warning(
            "Failed to send RSS feed update email via webhook: %s", e)
        if hasattr(e, "response") and e.response is not None:
            logger.warning(
                "Webhook response: %s %s",
                getattr(e.response, "status_code", ""),
                (e.response.text[:200] if getattr(
                    e.response, "text", None) else ""),
            )
        return False


def send_feed_builder_newswire_test_email(
    *,
    source_id: str,
    source_name: Optional[str],
    source_url: str,
    new_items: List[Dict[str, Any]],
) -> bool:
    """
    Testing-only: send one email per newswire when feed_builder scan finds new articles.

    Uses N8N_WEBHOOK_ONLY_ME. Does not modify or replace RSS.app webhook email flow.
    Set FEED_BUILDER_SCAN_TEST_EMAILS=false to disable without code changes.
    """
    if not FEED_BUILDER_SCAN_TEST_EMAILS:
        logger.debug(
            "Feed builder test emails disabled (FEED_BUILDER_SCAN_TEST_EMAILS=false)"
        )
        return False
    if not new_items:
        return False

    try:
        subject, html_email = generate_feed_builder_newswire_email_html(
            source_id=source_id,
            source_name=source_name,
            source_url=source_url or "",
            new_items=new_items,
        )
        display_name = feed_builder_display_name(source_name, source_id)
        logger.info(
            "Sending feed builder test email for %s (%s new item(s)) via N8N_WEBHOOK_ONLY_ME",
            source_id,
            len(new_items),
        )
        return _send_rss_feed_email_via_webhook(
            N8N_WEBHOOK_ONLY_ME,
            subject=subject,
            html_email=html_email,
            feed_title=display_name,
            items_count=len(new_items),
            feed_source_url=source_url or "",
        )
    except Exception as exc:
        logger.warning(
            "Feed builder test email failed for %s: %s", source_id, exc
        )
        return False


def send_feed_builder_pipeline_error_email(errors: List[Dict[str, Any]]) -> bool:
    """
    Send a single digest email after scan_all_feeds completes with one or more errors.
    `errors` is the summary["errors"] list: [{"source_id": ..., "error": ...}, ...]
    """
    if not errors:
        return False

    rows_html = "".join(
        f"<tr>"
        f"<td style='padding:6px 12px;border:1px solid #ddd'><b>{e['source_id']}</b></td>"
        f"<td style='padding:6px 12px;border:1px solid #ddd;color:#c0392b'>{e['error']}</td>"
        f"</tr>"
        for e in errors
    )
    html_email = (
        "<h2 style='color:#c0392b'>Feed Builder Pipeline — "
        f"{len(errors)} Feed(s) Failed</h2>"
        "<table style='border-collapse:collapse;width:100%'>"
        "<tr style='background:#f2f2f2'>"
        "<th style='padding:6px 12px;border:1px solid #ddd;text-align:left'>Source ID</th>"
        "<th style='padding:6px 12px;border:1px solid #ddd;text-align:left'>Error</th>"
        "</tr>"
        f"{rows_html}"
        "</table>"
    )
    subject = f"[Feed Builder] {len(errors)} feed(s) failed"

    logger.warning(
        "Feed builder pipeline finished with %d error(s) — sending digest email",
        len(errors),
    )
    return _send_rss_feed_email_via_webhook(
        N8N_WEBHOOK_ONLY_ME,
        subject=subject,
        html_email=html_email,
        feed_title="Feed Builder Pipeline",
        items_count=len(errors),
    )


def process_feed_builder_newswire_articles(
    *,
    feed_config: Dict[str, Any],
    new_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    GO LIVE flow: feed_builder scan → same pipeline as RSS.app process_webhook_payload.

    Called from scanner when NEW FLOW block is uncommented.
    source_name must match FEED_TITLE_DISPLAY_NAMES / FEED_TITLE_DISPLAY_NAME_2 keys.
    """
    from rss_feeds.feed_builder.webhook_adapter import build_feed_builder_webhook_payload

    if not new_items:
        return {"success": True, "skipped": True, "reason": "no new items"}

    payload = build_feed_builder_webhook_payload(feed_config, new_items)
    result = RSSFeedService.process_webhook_payload(payload)

    if result.get("success"):
        from rss_feeds.feed_builder.core.mongo_article_store import mark_articles_processed

        dedupe_keys = [item["dedupe_key"]
                       for item in new_items if item.get("dedupe_key")]
        mark_articles_processed(dedupe_keys)

    return result


class RSSFeedService:
    """Service class for RSS feed operations"""

    @staticmethod
    def create_or_update_feed(feed_data: Dict) -> Feed:
        """
        Create or update a feed based on webhook data

        Args:
            feed_data: Dictionary containing feed information

        Returns:
            Feed: The created or updated feed object
        """
        try:
            # ===== ACTIVE (RSS.app): upsert feeds by rss_feed_url =====
            # existing_feed = Feed.objects(
            #     rss_feed_url=feed_data.get('rss_feed_url')).first()
            # ===== END ACTIVE =====

            # ===== GO LIVE (feed builder): upsert by source_url, fallback to rss_feed_url =====
            source_url = (feed_data.get("source_url") or "").strip()
            existing_feed = None
            if source_url:
                existing_feed = Feed.objects(source_url=source_url).first()
            # if not existing_feed and feed_data.get("rss_feed_url"):
            #     existing_feed = Feed.objects(
            #         rss_feed_url=feed_data.get("rss_feed_url")
            #     ).first()
            # ===== END GO LIVE =====

            if existing_feed:
                # Update existing feed
                existing_feed.title = feed_data.get(
                    'title', existing_feed.title)
                existing_feed.source_url = feed_data.get(
                    'source_url', existing_feed.source_url)
                if "description" in feed_data:
                    existing_feed.description = feed_data.get(
                        'description', existing_feed.description)
                else:
                    existing_feed.description = None
                if "icon" in feed_data:
                    existing_feed.icon = _normalize_thumbnail(
                        feed_data.get("icon"))
                existing_feed.source = feed_data.get(
                    'source', existing_feed.source)
                existing_feed.save()
                logger.info(f"Updated existing feed: {existing_feed.title}")

                # Emit WebSocket notification for feed update
                import asyncio
                import threading

                def emit_websocket():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(
                            RSSWebSocketService.emit_feed_update(existing_feed, 'updated'))
                        loop.close()
                    except Exception as e:
                        logger.warning(
                            f"Could not emit WebSocket notification: {str(e)}")

                thread = threading.Thread(target=emit_websocket)
                thread.daemon = True
                thread.start()

                return existing_feed
            else:
                # Create new feed
                feed = Feed(
                    title=feed_data.get('title'),
                    source_url=feed_data.get('source_url'),
                    rss_feed_url=feed_data.get('rss_feed_url'),
                    description=feed_data.get('description', ''),
                    icon=_normalize_thumbnail(feed_data.get('icon')),
                    source=feed_data.get('source', '')
                )
                feed.save()
                logger.info(f"Created new feed: {feed.title}")

                # Emit WebSocket notification for new feed
                import asyncio
                import threading

                def emit_websocket():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(
                            RSSWebSocketService.emit_feed_update(feed, 'created'))
                        loop.close()
                    except Exception as e:
                        logger.warning(
                            f"Could not emit WebSocket notification: {str(e)}")

                thread = threading.Thread(target=emit_websocket)
                thread.daemon = True
                thread.start()

                return feed

        except Exception as e:
            logger.error(f"Error creating/updating feed: {str(e)}")
            raise

    @staticmethod
    def create_feed_items(feed_id: str, items_data: List[Dict]) -> List[FeedItem]:
        """
        Create feed items from webhook data

        Args:
            feed_id: The ID of the parent feed
            items_data: List of feed item dictionaries

        Returns:
            List[FeedItem]: List of created feed items
        """
        created_items = []

        try:
            for item_data in items_data:
                # Normalize thumbnail so empty/invalid URLs don't fail URLField validation
                item_data = dict(item_data)
                item_data["thumbnail"] = _normalize_thumbnail(
                    item_data.get("thumbnail"))

                # Validate item data
                serializer = FeedItemCreateSerializer(data=item_data)
                if not serializer.is_valid():
                    logger.warning(
                        f"Invalid feed item data: {serializer.errors}")
                    continue

                # Check if item already exists (by URL)
                existing_item = FeedItem.objects(url=item_data['url']).first()
                if existing_item:
                    logger.info(
                        f"Feed item already exists: {item_data['url']}")
                    continue

                # Create authors
                authors = []
                if 'authors' in item_data and item_data['authors']:
                    for author_data in item_data['authors']:
                        author = Author(name=author_data.get('name', ''))
                        authors.append(author)

                # Create feed item (thumbnail None when missing/invalid; URLField rejects '')
                feed_item = FeedItem(
                    url=item_data['url'],
                    title=item_data['title'],
                    description_text=item_data.get('description_text') or '',
                    thumbnail=item_data.get('thumbnail'),
                    date_published=_parse_date_published(
                        item_data['date_published']),
                    authors=authors,
                    rss_feed_id=feed_id,
                    deal_id=item_data.get('deal_id') or None,
                    l1_headline=item_data.get('l1_headline') or None,
                    l2_brief=item_data.get('l2_brief') or None,
                    l3_detailed=item_data.get('l3_detailed') or None,
                    s3_docx_url=item_data.get('s3_docx_url') or None,
                    s3_json_url=item_data.get('s3_json_url') or None,
                )
                feed_item.save()
                created_items.append(feed_item)
                logger.info(f"Created feed item: {feed_item.title}")

            logger.info(
                f"Successfully created {len(created_items)} feed items")
            return created_items

        except Exception as e:
            logger.error(f"Error creating feed items: {str(e)}")
            raise

    @staticmethod
    def process_webhook_payload(payload: Dict) -> Dict:
        """
        Process webhook payload and save to database

        Args:
            payload: Webhook payload dictionary

        Returns:
            Dict: Processing results
        """
        try:
            # Extract feed data
            feed_data = payload.get('feed', {})
            data = payload.get('data', {})
            items_new = data.get('items_new', [])
            logger.info(f"items_new: {items_new}")
            logger.info(f"feed_data: {feed_data}")
            logger.info(f"data: {data}")

            # Create or update feed
            feed = RSSFeedService.create_or_update_feed(feed_data)

            feed_title_str = feed_data.get("title") or feed.title
            feed_source_url_str = feed_data.get(
                "source_url") or getattr(feed, "source_url", "") or ""

            # Use 3-prompt merger flow only for feeds in FEED_TITLE_DISPLAY_NAMES; otherwise old way (save all, email all)
            use_merger_flow = feed_title_str in FEED_TITLE_DISPLAY_NAMES

            use_merger_flow_2 = feed_title_str in FEED_TITLE_DISPLAY_NAME_2

            if use_merger_flow:
                # New process: 3-prompt flow, save and email only merger-related items, attach deal_id
                # AI summary (route_and_summarize) is only run for items that pass this filter.
                deals_record_string = get_deals_record_string()
                flow_results = []
                error_registry = RSSArticleErrorRegistry(
                    flow="merger",
                    feed_title=feed_title_str,
                )
                # Run merger classifier on the original webhook items (items_new)
                for item in items_new:
                    # Set pipeline context per RSS item
                    from core.pipeline_logger import start_pipeline, RSS
                    import re as _re
                    _raw = (item.get("url") or "").rstrip(
                        "/").rsplit("/", 1)[-1]
                    _item_id = _re.sub(
                        r"[^\w\-]", "-", _raw)[:50] or "rss-item"
                    start_pipeline(RSS, accession=_item_id, doc_type="RSS")

                    collector = error_registry.get_collector(
                        item.get("url"), item.get("title"))
                    token = error_registry.activate(collector)
                    try:
                        result = resolve_rss_item_flow(
                            item, deals_record_string)
                        flow_results.append((item, result))
                    except Exception as e:
                        logger.exception(
                            "RSS item flow failed for %s", item.get("url"))
                        record_rss_error(
                            step="resolve_rss_item_flow",
                            message=f"RSS item flow failed: {e}",
                            exception=e,
                            module="rss_feeds.services.process_webhook_payload",
                            feed_title=feed_title_str,
                            article_url=item.get("url"),
                            article_title=item.get("title"),
                        )
                        flow_results.append(
                            (item, {"skip_email": True, "deal_id": None, "deal_info": None, "email_note": None}))
                    finally:
                        error_registry.deactivate(token)

                # Log how many items passed the merger filter (only these get AI summary)
                n_total = len(flow_results)
                n_skipped = sum(
                    1 for _, r in flow_results if r.get("skip_email"))
                n_not_merger = sum(1 for _, r in flow_results if r.get(
                    "email_note") == "not_merger_related")
                n_passed = n_total - n_skipped - n_not_merger
                logger.info(
                    "Merger flow: %s items total, %s skipped, %s not_merger_related, %s passed (will get AI summary if route_and_summarize succeeds)",
                    n_total, n_skipped, n_not_merger, n_passed,
                )

                items_to_save = []
                email_items: List[tuple[Dict, Dict]] = []
                for item, result in flow_results:
                    if result.get("skip_email"):
                        continue
                    if result.get("email_note") == "not_merger_related":
                        continue
                    item_with_deal = dict(item)
                    if result.get("deal_id"):
                        item_with_deal["deal_id"] = result["deal_id"]

                    # Try to generate SEC/press-release summary via filing_router before saving
                    url = item_with_deal.get("url")
                    if url:
                        summary_collector = error_registry.get_collector(
                            url, item_with_deal.get("title"))
                        summary_token = error_registry.activate(
                            summary_collector)
                        try:
                            logger.info(
                                "Calling route_and_summarize for merger-related item: %s", url
                            )
                            _deal_id = item_with_deal.get("deal_id")
                            _deal_tickers = get_deal_tickers(_deal_id)
                            _deal_context = {
                                "primary_ticker":  _deal_tickers.get("ticker"),
                                "target_ticker":   _deal_tickers.get("target_ticker"),
                                "target_name":     _deal_tickers.get("target_name"),
                                "acquirer_ticker": _deal_tickers.get("acquirer_ticker"),
                                "acquirer_name":   _deal_tickers.get("acquirer_name"),
                            } if any(_deal_tickers.values()) else None
                            summary = route_and_summarize(
                                url, deal_context=_deal_context)
                            if isinstance(summary, dict) and summary.get("skipped"):
                                logger.info(
                                    "Skipped AI summary (not M&A press release) for %s: %s",
                                    url,
                                    (summary.get("skip_reason") or "")[:300],
                                )
                            else:
                                s3_docx_url = summary.get(
                                    "s3_docx_url") or summary.get("s3_url")
                                if s3_docx_url:
                                    item_with_deal["l1_headline"] = summary.get(
                                        "L1_headline")
                                    item_with_deal["l2_brief"] = summary.get(
                                        "L2_brief")
                                    item_with_deal["l3_detailed"] = summary.get(
                                        "L3_detailed") or None
                                    item_with_deal["s3_docx_url"] = s3_docx_url
                                    item_with_deal["s3_json_url"] = summary.get(
                                        "s3_json_url")
                                    logger.info(
                                        "AI summary attached for %s (L1: %s)",
                                        url,
                                        (summary.get("L1_headline") or "")[
                                            :60],
                                    )
                                else:
                                    logger.warning(
                                        "route_and_summarize returned no S3 docx URL for %s (keys: %s)",
                                        url,
                                        list(summary.keys()) if isinstance(
                                            summary, dict) else type(summary).__name__,
                                    )
                        except Exception as e:
                            if _is_skippable_non_ma_summary_error(e):
                                logger.info(
                                    "Skipped AI summary (not M&A / deal mismatch) for %s: %s",
                                    url,
                                    str(e)[:300],
                                )
                            else:
                                logger.exception(
                                    "route_and_summarize failed for %s", url)
                                record_rss_error(
                                    step="route_and_summarize",
                                    message=f"route_and_summarize failed: {e}",
                                    exception=e,
                                    module="rss_feeds.services.process_webhook_payload",
                                    flow="merger",
                                    feed_title=feed_title_str,
                                    article_url=url,
                                    article_title=item_with_deal.get("title"),
                                )
                        finally:
                            error_registry.deactivate(summary_token)

                    items_to_save.append(item_with_deal)
                    email_items.append((item_with_deal, result))
                created_items = RSSFeedService.create_feed_items(
                    str(feed.id), items_to_save)
                logger.debug(
                    "Webhook created_items count (merger flow): %s", len(created_items))

                if created_items:
                    import asyncio
                    import threading

                    def emit_websocket():
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(
                                RSSWebSocketService.emit_new_feed_items(created_items, feed))
                            loop.close()
                        except Exception as e:
                            logger.warning(
                                f"Could not emit WebSocket notification: {str(e)}")

                    thread = threading.Thread(target=emit_websocket)
                    thread.daemon = True
                    thread.start()

                # For emails, use the enriched items (with summaries) when available
                for item_with_deal, result in email_items:
                    if result.get("skip_email"):
                        continue
                    try:
                        subject, html_email, report_type = generate_rss_feed_item_email_html(
                            feed_data,
                            item_with_deal,
                            deal_info=result.get("deal_info"),
                            email_note=result.get("email_note"),
                            match_details=result.get("match_details"),
                        )
                        if "Net Asset Value(s)" in subject:
                            logger.info(
                                "Skipping email — subject contains 'Net Asset Value(s)': %s", subject
                            )
                            continue
                        # webhook_url = (
                        #     N8N_WEBHOOK_SEND_TO_ALL
                        #     if rss_subject_uses_client_webhook(subject)
                        #     else N8N_WEKHOOK_INTERNAL_WITH_JOSH
                        # )
                        # _send_rss_feed_email_via_webhook(
                        #     webhook_url,
                        #     subject=subject,
                        #     html_email=html_email,
                        #     feed_title=feed_title_str,
                        #     items_count=1,
                        #     feed_source_url=feed_source_url_str
                        # )
                        if report_type:
                            send_report_email(
                                report_type=report_type,
                                payload={
                                    "subject": subject,
                                    "html": html_email,
                                    "feed_title": feed_title_str,
                                    "items_count": 1,
                                    "feed_source_url": feed_source_url_str,
                                }
                            )
                    except Exception as e:
                        logger.warning(
                            "Could not generate/send RSS feed item email: %s", e
                        )

                error_registry.flush_all()
            elif use_merger_flow_2:
                # Flow 2: no save. For each item, ask LLM if title/description mention a deal we follow;
                # if yes, send one email per matching item with deal_id. Do not save any items.
                deals_record_string = get_deals_record_string()
                created_items = []
                error_registry = RSSArticleErrorRegistry(
                    flow="merger_flow_2",
                    feed_title=feed_title_str,
                )
                for item in items_new:
                    # Set pipeline context per RSS item
                    from core.pipeline_logger import start_pipeline, RSS
                    import re as _re
                    _raw = (item.get("url") or "").rstrip(
                        "/").rsplit("/", 1)[-1]
                    _item_id = _re.sub(
                        r"[^\w\-]", "-", _raw)[:50] or "rss-item"
                    start_pipeline(RSS, accession=_item_id,
                                   doc_type="RSS_FLOW2")

                    title = item.get("title") or ""
                    description = item.get("description_text") or ""
                    collector = error_registry.get_collector(
                        item.get("url"), item.get("title"))
                    token = error_registry.activate(collector)
                    try:
                        result = classify_feed_item_by_title_description(
                            deals_record_string, title, description
                        )
                        logger.info(
                            f"classify_feed_item_by_title_description result: {result}")
                    except Exception as e:
                        logger.exception(
                            "classify_feed_item_by_title_description failed for item %s",
                            item.get("url"),
                        )
                        record_rss_error(
                            step="classify_feed_item_by_title_description",
                            message=(
                                "classify_feed_item_by_title_description failed: "
                                f"{e}"
                            ),
                            exception=e,
                            module="rss_feeds.services.process_webhook_payload",
                            feed_title=feed_title_str,
                            article_url=item.get("url"),
                            article_title=title,
                        )
                        continue
                    finally:
                        error_registry.deactivate(token)
                    if not result.get("match") or not result.get("deal_id"):
                        continue
                    deal_id = result["deal_id"]
                    deal_info = get_deal_info_for_email(deal_id)
                    if deal_info:
                        logger.info(f"deal_info: {deal_info}")
                        deal_info["in_db"] = True
                    item_with_deal = dict(item)
                    item_with_deal["deal_id"] = deal_id
                    try:
                        subject, html_email, report_type = generate_rss_feed_item_email_html_flow2(
                            feed_data,
                            item_with_deal,
                            deal_info=deal_info,
                        )
                        # _send_rss_feed_email_via_webhook(
                        #     N8N_WEKHOOK_INTERNAL_WITH_JOSH,
                        #     subject=subject,
                        #     html_email=html_email,
                        #     feed_title=feed_title_str,
                        #     items_count=1,
                        #     feed_source_url=feed_source_url_str
                        # )

                        if report_type:
                            send_report_email(
                                report_type=report_type,
                                payload={
                                    "subject": subject,
                                    "html": html_email,
                                    "feed_title": feed_title_str,
                                    "items_count": 1,
                                    "feed_source_url": feed_source_url_str,
                                }
                            )
                    except Exception as e:
                        logger.error(
                            "Could not generate/send RSS feed item email (flow 2): %s", e
                        )

                error_registry.flush_all()
            else:
                # Old way: save all items, send email for every item (no deal logic)
                items_with_summaries = []
                error_registry = RSSArticleErrorRegistry(
                    flow="legacy",
                    feed_title=feed_title_str,
                )
                for item in items_new:
                    item_data = dict(item)
                    url = item_data.get("url")
                    if url:
                        collector = error_registry.get_collector(
                            url, item_data.get("title"))
                        token = error_registry.activate(collector)
                        try:
                            summary = route_and_summarize(url)
                            s3_docx_url = summary.get(
                                "s3_docx_url") or summary.get("s3_url")
                            if s3_docx_url:
                                item_data["l1_headline"] = summary.get(
                                    "L1_headline")
                                item_data["l2_brief"] = summary.get(
                                    "L2_brief")
                                item_data["l3_detailed"] = summary.get(
                                    "L3_detailed") or None
                                item_data["s3_docx_url"] = s3_docx_url
                                item_data["s3_json_url"] = summary.get(
                                    "s3_json_url")
                            else:
                                logger.error(
                                    "route_and_summarize returned no S3 docx URL for %s", url
                                )
                        except Exception as e:
                            logger.exception(
                                "route_and_summarize failed for %s", url)
                            record_rss_error(
                                step="route_and_summarize",
                                message=f"route_and_summarize failed: {e}",
                                exception=e,
                                module="rss_feeds.services.process_webhook_payload",
                                flow="legacy",
                                feed_title=feed_title_str,
                                article_url=url,
                                article_title=item_data.get("title"),
                            )
                        finally:
                            error_registry.deactivate(token)
                    items_with_summaries.append(item_data)

                created_items = RSSFeedService.create_feed_items(
                    str(feed.id), items_with_summaries)
                logger.debug("Webhook created_items count: %s",
                             len(created_items))

                if created_items:
                    import asyncio
                    import threading

                    def emit_websocket():
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(
                                RSSWebSocketService.emit_new_feed_items(created_items, feed))
                            loop.close()
                        except Exception as e:
                            logger.error(
                                f"Could not emit WebSocket notification: {str(e)}")

                    thread = threading.Thread(target=emit_websocket)
                    thread.daemon = True
                    thread.start()

                for item in items_new:
                    try:
                        subject, html_email, report_type = generate_rss_feed_item_email_html(
                            feed_data, item
                        )
                        # _send_rss_feed_email_via_webhook(
                        #     N8N_WEKHOOK_INTERNAL_WITH_JOSH,
                        #     subject=subject,
                        #     html_email=html_email,
                        #     feed_title=feed_title_str,
                        #     items_count=1,
                        #     feed_source_url=feed_source_url_str
                        # )

                        send_report_email(
                            report_type="other_newswire",
                            payload={
                                "subject": subject,
                                "html": html_email,
                                "feed_title": feed_title_str,
                                "items_count": 1,
                                "feed_source_url": feed_source_url_str,
                            }
                        )
                    except Exception as e:
                        logger.error(
                            "Could not generate/send RSS feed item email: %s", e
                        )

                error_registry.flush_all()

            return {
                'success': True,
                'feed_id': str(feed.id),
                'feed_title': feed.title,
                'items_created': len(created_items),
                'total_items_received': len(items_new)
            }

        except Exception as e:
            logger.exception("Error processing webhook payload")
            send_exception_email(
                pipeline=RSS,
                error_message=f"Error processing webhook payload: {e}",
                context={
                    "module": "rss_feeds.services.process_webhook_payload",
                    "feed_title": (
                        (payload.get("feed") or {}).get("title")
                        if isinstance(payload, dict)
                        else None
                    ),
                    "items_received": len(
                        ((payload.get("data") or {}).get("items_new") or [])
                        if isinstance(payload, dict)
                        else []
                    ),
                },
                exception=e,
                email_type="rss_webhook_error",
            )
            return {
                'success': False,
                'error': str(e)
            }

    @staticmethod
    def get_feed_by_id(feed_id: str) -> Optional[Feed]:
        """Get feed by ID"""
        try:
            return Feed.objects(id=feed_id).first()
        except Exception as e:
            logger.error(f"Error getting feed by ID: {str(e)}")
            return None

    @staticmethod
    def get_all_feeds(limit: int = 1000) -> List[Feed]:
        """Get all feeds with optional limit"""
        try:
            return list(Feed.objects.all())
        except Exception as e:
            logger.error(f"Error getting all feeds: {str(e)}")
            return []

    @staticmethod
    def get_feed_items(feed_id: str, limit: int = 50) -> List[FeedItem]:
        """Get feed items for a specific feed"""
        try:
            return list(FeedItem.objects(rss_feed_id=feed_id).order_by('-date_published'))
        except Exception as e:
            logger.error(f"Error getting feed items: {str(e)}")
            return []

    @staticmethod
    def get_recent_feed_items_with_source(limit: int = 100) -> List[Dict]:
        """Get recent feed items across all feeds with source field from parent feeds"""
        try:
            # Get recent feed items
            feed_items = FeedItem.objects.all().order_by('-date_published')

            # Get all unique feed IDs to fetch feed information efficiently
            feed_ids = set(item.rss_feed_id for item in feed_items)
            feeds = {str(feed.id): feed for feed in Feed.objects(
                id__in=feed_ids)}

            # Convert to list of dictionaries and add source field
            items_with_source = []
            for item in feed_items:
                feed = feeds.get(item.rss_feed_id)
                item_dict = {
                    'id': str(item.id),
                    'url': item.url,
                    'title': item.title,
                    'description_text': item.description_text,
                    'thumbnail': item.thumbnail,
                    'date_published': item.date_published,
                    'authors': [{'name': author.name} for author in item.authors] if item.authors else [],
                    'rss_feed_id': item.rss_feed_id,
                    'created_at': item.created_at,
                    'updated_at': item.updated_at,
                    'source': feed.source if feed else None
                }
                items_with_source.append(item_dict)

            return items_with_source
        except Exception as e:
            logger.error(
                f"Error getting recent feed items with source: {str(e)}")
            return []

    @staticmethod
    def get_feed_items_with_source(feed_id: str, limit: int = 50) -> List[Dict]:
        """Get feed items for a specific feed with source field from parent feed"""
        try:
            # Get the feed to extract source information
            feed = Feed.objects(id=feed_id).first()
            if not feed:
                logger.warning(f"Feed not found: {feed_id}")
                return []

            # Get feed items
            feed_items = FeedItem.objects(rss_feed_id=feed_id).order_by(
                '-date_published')

            # Convert to list of dictionaries and add source field
            items_with_source = []
            for item in feed_items:
                item_dict = {
                    'id': str(item.id),
                    'url': item.url,
                    'title': item.title,
                    'description_text': item.description_text,
                    'thumbnail': item.thumbnail,
                    'date_published': item.date_published,
                    'authors': [{'name': author.name} for author in item.authors] if item.authors else [],
                    'rss_feed_id': item.rss_feed_id,
                    'created_at': item.created_at,
                    'updated_at': item.updated_at,
                    'source': feed.source
                }
                items_with_source.append(item_dict)

            return items_with_source
        except Exception as e:
            logger.error(f"Error getting feed items with source: {str(e)}")
            return []

    @staticmethod
    def get_recent_feed_items(limit: int = 100) -> List[FeedItem]:
        """Get recent feed items across all feeds"""
        try:
            return list(FeedItem.objects.all().order_by('-date_published'))
        except Exception as e:
            logger.error(f"Error getting recent feed items: {str(e)}")
            return []

    @staticmethod
    def delete_feed(feed_id: str) -> bool:
        """Delete a feed and all its items"""
        try:
            feed = Feed.objects(id=feed_id).first()
            if not feed:
                return False

            # Delete all feed items
            FeedItem.objects(rss_feed_id=feed_id).delete()

            # Delete the feed
            feed.delete()

            logger.info(f"Deleted feed and all items: {feed_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting feed: {str(e)}")
            return False
