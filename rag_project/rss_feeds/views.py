from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from django.http import JsonResponse
from .serializers import (
    FeedSerializer,
    FeedItemSerializer,
    FeedItemWithSourceSerializer,
    FeedWithItemsSerializer,
    WebhookPayloadSerializer
)
from .services import RSSFeedService
from .models import Feed, FeedItem
import logging

logger = logging.getLogger(__name__)


class WebhookView(APIView):
    """
    Webhook endpoint for receiving RSS feed updates
    """
    permission_classes = [AllowAny]

    def post(self, request, format=None):
        """
        Process webhook payload from RSS service
        """
        try:
            # Validate webhook payload
            serializer = WebhookPayloadSerializer(data=request.data)
            if not serializer.is_valid():
                logger.error(f"Invalid webhook payload: {serializer.errors}")
                return Response({
                    'success': False,
                    'error': 'Invalid webhook payload',
                    'details': serializer.errors
                }, status=status.HTTP_400_BAD_REQUEST)

            # Process the webhook payload
            result = RSSFeedService.process_webhook_payload(request.data)

            if result['success']:
                logger.info(f"Webhook processed successfully: {result}")
                return Response({
                    'success': True,
                    'message': 'Webhook processed successfully',
                    'data': result
                }, status=status.HTTP_200_OK)
            else:
                logger.error(f"Webhook processing failed: {result['error']}")
                return Response({
                    'success': False,
                    'error': result['error']
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except Exception as e:
            logger.error(f"Unexpected error in webhook: {str(e)}")
            return Response({
                'success': False,
                'error': 'Internal server error'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FeedListView(APIView):
    """
    List all RSS feeds
    """
    permission_classes = [AllowAny]

    def get(self, request, format=None):
        """
        Get all feeds with optional limit
        """
        try:
            limit = int(request.query_params.get('limit', 100))
            feeds = RSSFeedService.get_all_feeds(limit=limit)

            serializer = FeedSerializer(feeds, many=True)
            return Response({
                'success': True,
                'count': len(feeds),
                'feeds': serializer.data
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error getting feeds: {str(e)}")
            return Response({
                'success': False,
                'error': 'Failed to retrieve feeds'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FeedDetailView(APIView):
    """
    Get, update, or delete a specific feed
    """

    def get(self, request, feed_id, format=None):
        """
        Get a specific feed by ID
        """
        try:
            feed = RSSFeedService.get_feed_by_id(feed_id)
            if not feed:
                return Response({
                    'success': False,
                    'error': 'Feed not found'
                }, status=status.HTTP_404_NOT_FOUND)

            # Check if user wants feed items included
            include_items = request.query_params.get(
                'include_items', 'false').lower() == 'true'

            if include_items:
                feed_items_with_source = RSSFeedService.get_feed_items_with_source(
                    feed_id)
                feed_data = FeedSerializer(feed).data
                feed_data['feed_items'] = FeedItemWithSourceSerializer(
                    feed_items_with_source, many=True).data
                return Response({
                    'success': True,
                    'feed': feed_data
                }, status=status.HTTP_200_OK)
            else:
                serializer = FeedSerializer(feed)
                return Response({
                    'success': True,
                    'feed': serializer.data
                }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error getting feed: {str(e)}")
            return Response({
                'success': False,
                'error': 'Failed to retrieve feed'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def delete(self, request, feed_id, format=None):
        """
        Delete a feed and all its items
        """
        try:
            success = RSSFeedService.delete_feed(feed_id)
            if success:
                return Response({
                    'success': True,
                    'message': 'Feed deleted successfully'
                }, status=status.HTTP_200_OK)
            else:
                return Response({
                    'success': False,
                    'error': 'Feed not found or could not be deleted'
                }, status=status.HTTP_404_NOT_FOUND)

        except Exception as e:
            logger.error(f"Error deleting feed: {str(e)}")
            return Response({
                'success': False,
                'error': 'Failed to delete feed'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FeedItemsView(APIView):
    """
    Get feed items for a specific feed
    """

    def get(self, request, feed_id, format=None):
        """
        Get feed items for a specific feed
        """
        try:
            limit = int(request.query_params.get('limit', 50))
            feed_items_with_source = RSSFeedService.get_feed_items_with_source(
                feed_id, limit=limit)

            serializer = FeedItemWithSourceSerializer(
                feed_items_with_source, many=True)
            return Response({
                'success': True,
                'feed_id': feed_id,
                'count': len(feed_items_with_source),
                'feed_items': serializer.data
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error getting feed items: {str(e)}")
            return Response({
                'success': False,
                'error': 'Failed to retrieve feed items'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class RecentFeedItemsView(APIView):
    """
    Get recent feed items across all feeds
    """
    permission_classes = [AllowAny]

    def get(self, request, format=None):
        """
        Get recent feed items from all feeds
        """
        try:
            limit = int(request.query_params.get('limit', 100))
            feed_items_with_source = RSSFeedService.get_recent_feed_items_with_source(
                limit=limit)

            serializer = FeedItemWithSourceSerializer(
                feed_items_with_source, many=True)
            return Response({
                'success': True,
                'count': len(feed_items_with_source),
                'feed_items': serializer.data
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error getting recent feed items: {str(e)}")
            return Response({
                'success': False,
                'error': 'Failed to retrieve recent feed items'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FeedItemDetailView(APIView):
    """
    Get a specific feed item by ID
    """

    def get(self, request, item_id, format=None):
        """
        Get a specific feed item by ID
        """
        try:
            feed_item = FeedItem.objects(id=item_id).first()
            if not feed_item:
                return Response({
                    'success': False,
                    'error': 'Feed item not found'
                }, status=status.HTTP_404_NOT_FOUND)

            serializer = FeedItemSerializer(feed_item)
            return Response({
                'success': True,
                'feed_item': serializer.data
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error getting feed item: {str(e)}")
            return Response({
                'success': False,
                'error': 'Failed to retrieve feed item'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
