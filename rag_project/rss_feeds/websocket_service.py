from datetime import datetime
import socketio
import json
import logging
from typing import Dict, List
from .models import FeedItem, Feed
from .serializers import FeedItemSerializer

logger = logging.getLogger(__name__)

# Create Socket.IO server instance with proper CORS configuration
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=[
        '*',  # Allow all origins for development
        'http://localhost:3000',
        'http://localhost:5500',
        'http://127.0.0.1:3000',
        'http://127.0.0.1:5500',
        'http://localhost:8000',
        'http://localhost:8080',
        'http://127.0.0.1:8000',
        'https://rag-summary-fe.onrender.com'
    ],
    logger=True,
    engineio_logger=True
)


class RSSWebSocketService:
    """Service for handling WebSocket connections and real-time RSS feed updates"""

    @staticmethod
    async def emit_new_feed_items(feed_items: List[FeedItem], feed: Feed):
        """
        Emit new feed items to all connected clients

        Args:
            feed_items: List of new feed items
            feed: The parent feed
        """
        try:
            # Serialize feed items and add source field
            items_data = []
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
                items_data.append(item_dict)

            # Prepare notification payload
            notification = {
                'type': 'new_feed_items',
                'feed': {
                    'id': str(feed.id),
                    'title': feed.title,
                    'source': feed.source
                },
                'items': items_data,
                'count': len(feed_items)
            }

            # Emit to all connected clients
            await sio.emit('rss_update', notification)

            logger.info(
                f"Emitted {len(feed_items)} new feed items to connected clients")

        except Exception as e:
            logger.error(f"Error emitting feed items: {str(e)}")

    @staticmethod
    async def emit_feed_update(feed: Feed, action: str = 'updated'):
        """
        Emit feed update notification

        Args:
            feed: The feed that was updated
            action: The action performed (created, updated, deleted)
        """
        try:
            notification = {
                'type': 'feed_update',
                'action': action,
                'feed': {
                    'id': str(feed.id),
                    'title': feed.title,
                    'source': feed.source,
                    'description': feed.description
                }
            }

            await sio.emit('rss_update', notification)
            logger.info(
                f"Emitted feed {action} notification for feed: {feed.title}")

        except Exception as e:
            logger.error(f"Error emitting feed update: {str(e)}")

    @staticmethod
    async def emit_error(error_message: str, error_type: str = 'general'):
        """
        Emit error notification to clients

        Args:
            error_message: Error message
            error_type: Type of error
        """
        try:
            notification = {
                'type': 'error',
                'error_type': error_type,
                'message': error_message,
                'timestamp': str(datetime.utcnow())
            }

            await sio.emit('rss_update', notification)
            logger.error(f"Emitted error notification: {error_message}")

        except Exception as e:
            logger.error(f"Error emitting error notification: {str(e)}")


# Socket.IO event handlers
@sio.event
async def connect(sid, environ, auth=None):
    """Handle client connection"""
    logger.info(f"Client connected: {sid}")
    # Send welcome message
    await sio.emit('connected', {
        'message': 'Connected to RSS feed updates',
        'sid': sid
    }, room=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {sid}")


@sio.event
async def join_feed(sid, data):
    """Handle client joining specific feed room"""
    try:
        feed_id = data.get('feed_id')
        if feed_id:
            room_name = f'feed_{feed_id}'
            await sio.enter_room(sid, room_name)
            logger.info(f"Client {sid} joined feed room: {room_name}")

            await sio.emit('joined_feed', {
                'feed_id': feed_id,
                'message': f'Joined feed room: {feed_id}'
            }, room=sid)
        else:
            await sio.emit('error', {
                'message': 'feed_id is required'
            }, room=sid)

    except Exception as e:
        logger.error(f"Error joining feed room: {str(e)}")
        await sio.emit('error', {
            'message': f'Error joining feed room: {str(e)}'
        }, room=sid)


@sio.event
async def leave_feed(sid, data):
    """Handle client leaving specific feed room"""
    try:
        feed_id = data.get('feed_id')
        if feed_id:
            room_name = f'feed_{feed_id}'
            await sio.leave_room(sid, room_name)
            logger.info(f"Client {sid} left feed room: {room_name}")

            await sio.emit('left_feed', {
                'feed_id': feed_id,
                'message': f'Left feed room: {feed_id}'
            }, room=sid)
        else:
            await sio.emit('error', {
                'message': 'feed_id is required'
            }, room=sid)

    except Exception as e:
        logger.error(f"Error leaving feed room: {str(e)}")
        await sio.emit('error', {
            'message': f'Error leaving feed room: {str(e)}'
        }, room=sid)


@sio.event
async def get_recent_items(sid, data):
    """Handle client request for recent feed items"""
    try:
        limit = data.get('limit', 10)
        feed_id = data.get('feed_id')

        if feed_id:
            # Get items for specific feed with source
            from .services import RSSFeedService
            items_data = RSSFeedService.get_feed_items_with_source(
                feed_id, limit)
        else:
            # Get recent items from all feeds with source
            from .services import RSSFeedService
            items_data = RSSFeedService.get_recent_feed_items_with_source(
                limit)

        await sio.emit('recent_items', {
            'items': items_data,
            'count': len(items_data),
            'feed_id': feed_id
        }, room=sid)

        logger.info(f"Sent {len(items_data)} recent items to client {sid}")

    except Exception as e:
        logger.error(f"Error getting recent items: {str(e)}")
        await sio.emit('error', {
            'message': f'Error getting recent items: {str(e)}'
        }, room=sid)
