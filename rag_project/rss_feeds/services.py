from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import requests

from .models import Feed, FeedItem, Author
from .serializers import FeedItemCreateSerializer
from .websocket_service import RSSWebSocketService
from .email_templates import generate_rss_feed_item_email_html, FEED_TITLE_DISPLAY_NAMES
from .merger_news_classifier import (
    get_deals_record_string,
    resolve_rss_item_flow,
)

logger = logging.getLogger(__name__)

# N8N webhook for RSS feed update emails (testing – same as sec_rss_parser)
N8N_WEBHOOK_URL_FOR_TESTING_ME = (
    "https://n8n-xwx1.onrender.com/webhook/80830c6d-ff5b-45e3-9ef3-a061db1fbf0c"
)
N8N_WEBHOOK_URL_FOR_TESTING = (
    "https://n8n-xwx1.onrender.com/webhook/b3007d21-6845-47b5-aece-7b26583758bc"
)


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
            # Check if feed already exists
            existing_feed = Feed.objects(
                rss_feed_url=feed_data.get('rss_feed_url')).first()

            if existing_feed:
                # Update existing feed
                existing_feed.title = feed_data.get(
                    'title', existing_feed.title)
                existing_feed.source_url = feed_data.get(
                    'source_url', existing_feed.source_url)
                existing_feed.description = feed_data.get(
                    'description', existing_feed.description)
                existing_feed.icon = feed_data.get('icon', existing_feed.icon)
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
                    icon=feed_data.get('icon', ''),
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

            if use_merger_flow:
                # New process: 3-prompt flow, save and email only merger-related items, attach deal_id
                deals_record_string = get_deals_record_string()
                flow_results = []
                for item in items_new:
                    try:
                        result = resolve_rss_item_flow(
                            item, deals_record_string)
                        flow_results.append((item, result))
                    except Exception as e:
                        logger.warning(
                            "RSS item flow failed for %s: %s", item.get("url"), e)
                        flow_results.append(
                            (item, {"skip_email": True, "deal_id": None, "deal_info": None, "email_note": None}))

                items_to_save = []
                for item, result in flow_results:
                    if result.get("skip_email"):
                        continue
                    if result.get("email_note") == "not_merger_related":
                        continue
                    item_with_deal = dict(item)
                    if result.get("deal_id"):
                        item_with_deal["deal_id"] = result["deal_id"]
                    items_to_save.append(item_with_deal)
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

                for item, result in flow_results:
                    if result.get("skip_email"):
                        continue
                    try:
                        subject, html_email = generate_rss_feed_item_email_html(
                            feed_data,
                            item,
                            deal_info=result.get("deal_info"),
                            email_note=result.get("email_note"),
                        )
                        _send_rss_feed_email_via_webhook(
                            N8N_WEBHOOK_URL_FOR_TESTING,
                            subject=subject,
                            html_email=html_email,
                            feed_title=feed_title_str,
                            items_count=1,
                            feed_source_url=feed_source_url_str
                        )
                    except Exception as e:
                        logger.warning(
                            "Could not generate/send RSS feed item email: %s", e
                        )
            else:
                # Old way: save all items, send email for every item (no deal logic)
                created_items = RSSFeedService.create_feed_items(
                    str(feed.id), items_new)
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
                            logger.warning(
                                f"Could not emit WebSocket notification: {str(e)}")

                    thread = threading.Thread(target=emit_websocket)
                    thread.daemon = True
                    thread.start()

                for item in items_new:
                    try:
                        subject, html_email = generate_rss_feed_item_email_html(
                            feed_data, item
                        )
                        _send_rss_feed_email_via_webhook(
                            N8N_WEBHOOK_URL_FOR_TESTING,
                            subject=subject,
                            html_email=html_email,
                            feed_title=feed_title_str,
                            items_count=1,
                            feed_source_url=feed_source_url_str
                        )
                    except Exception as e:
                        logger.warning(
                            "Could not generate/send RSS feed item email: %s", e
                        )

            return {
                'success': True,
                'feed_id': str(feed.id),
                'feed_title': feed.title,
                'items_created': len(created_items),
                'total_items_received': len(items_new)
            }

        except Exception as e:
            logger.error(f"Error processing webhook payload: {str(e)}")
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
