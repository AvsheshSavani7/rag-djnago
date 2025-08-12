from datetime import datetime
from typing import Dict, List, Optional
from .models import Feed, FeedItem, Author
from .serializers import FeedItemCreateSerializer
from .websocket_service import RSSWebSocketService
import logging

logger = logging.getLogger(__name__)


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
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(
                            RSSWebSocketService.emit_feed_update(existing_feed, 'updated'))
                    else:
                        asyncio.run(RSSWebSocketService.emit_feed_update(
                            existing_feed, 'updated'))
                except Exception as e:
                    logger.warning(
                        f"Could not emit WebSocket notification: {str(e)}")

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
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(
                            RSSWebSocketService.emit_feed_update(feed, 'created'))
                    else:
                        asyncio.run(
                            RSSWebSocketService.emit_feed_update(feed, 'created'))
                except Exception as e:
                    logger.warning(
                        f"Could not emit WebSocket notification: {str(e)}")

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

                # Create feed item
                feed_item = FeedItem(
                    url=item_data['url'],
                    title=item_data['title'],
                    description_text=item_data.get('description_text', ''),
                    thumbnail=item_data.get('thumbnail', ''),
                    date_published=item_data['date_published'],
                    authors=authors,
                    rss_feed_id=feed_id
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

            # Create or update feed
            feed = RSSFeedService.create_or_update_feed(feed_data)

            # Create feed items
            created_items = RSSFeedService.create_feed_items(
                str(feed.id), items_new)

            # Emit WebSocket notification for new feed items
            if created_items:
                # Use background task for WebSocket emission
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Schedule the emit in the background
                        asyncio.create_task(
                            RSSWebSocketService.emit_new_feed_items(created_items, feed))
                    else:
                        # Run in new event loop
                        asyncio.run(RSSWebSocketService.emit_new_feed_items(
                            created_items, feed))
                except Exception as e:
                    logger.warning(
                        f"Could not emit WebSocket notification: {str(e)}")

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
    def get_all_feeds(limit: int = 100) -> List[Feed]:
        """Get all feeds with optional limit"""
        try:
            return list(Feed.objects.all().limit(limit))
        except Exception as e:
            logger.error(f"Error getting all feeds: {str(e)}")
            return []

    @staticmethod
    def get_feed_items(feed_id: str, limit: int = 50) -> List[FeedItem]:
        """Get feed items for a specific feed"""
        try:
            return list(FeedItem.objects(rss_feed_id=feed_id).order_by('-date_published').limit(limit))
        except Exception as e:
            logger.error(f"Error getting feed items: {str(e)}")
            return []

    @staticmethod
    def get_recent_feed_items(limit: int = 100) -> List[FeedItem]:
        """Get recent feed items across all feeds"""
        try:
            return list(FeedItem.objects.all().order_by('-date_published').limit(limit))
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
