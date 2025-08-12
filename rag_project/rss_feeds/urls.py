from django.urls import path
from .views import (
    WebhookView,
    FeedListView,
    FeedDetailView,
    FeedItemsView,
    RecentFeedItemsView,
    FeedItemDetailView
)
from .websocket_service import socket_app

app_name = 'rss_feeds'

urlpatterns = [
    # Webhook endpoint for receiving RSS feed updates
    path('webhook/', WebhookView.as_view(), name='webhook'),

    # Feed management endpoints
    path('feeds/', FeedListView.as_view(), name='feed-list'),
    path('feeds/<str:feed_id>/', FeedDetailView.as_view(), name='feed-detail'),
    path('feeds/<str:feed_id>/items/',
         FeedItemsView.as_view(), name='feed-items'),

    # Feed items endpoints
    path('items/', RecentFeedItemsView.as_view(), name='recent-items'),
    path('items/<str:item_id>/', FeedItemDetailView.as_view(), name='item-detail'),

    # WebSocket endpoint
    path('socket.io/', socket_app, name='socketio'),
]
