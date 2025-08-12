"""
ASGI config for rag_project project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os
from django.core.asgi import get_asgi_application
import socketio
from rss_feeds.websocket_service import sio

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rag_project.settings")

# Get Django ASGI application
django_asgi_app = get_asgi_application()

# Create Socket.IO ASGI application that handles both WebSocket and HTTP
application = socketio.ASGIApp(
    sio,
    other_asgi_app=django_asgi_app,
    socketio_path="socket.io"
)
