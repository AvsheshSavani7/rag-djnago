#!/usr/bin/env python
"""
Test script for RSS feed WebSocket functionality
"""
import asyncio
import socketio
import requests
import json


async def test_websocket():
    """Test WebSocket connection and events"""

    sio = socketio.AsyncClient()

    @sio.event
    async def connect():
        print("✅ Connected to WebSocket server")
        await sio.emit('get_recent_items', {'limit': 5})

    @sio.event
    async def rss_update(data):
        print(f"📡 RSS Update: {data['type']} - {data.get('count', 0)} items")

    @sio.event
    async def recent_items(data):
        print(f"📋 Recent items: {data['count']} items received")

    try:
        await sio.connect('http://localhost:8000/api/rss/socket.io/', namespaces=['/rss'])
        await asyncio.sleep(10)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        await sio.disconnect()


def test_webhook():
    """Test webhook with WebSocket notifications"""

    webhook_data = {
        "id": "test_websocket_event",
        "type": "feed_update",
        "feed": {
            "id": "test_feed_websocket",
            "title": "Test Feed for WebSocket",
            "source_url": "https://example.com/test",
            "rss_feed_url": "https://rss.app/feeds/test_websocket.xml",
            "description": "Test feed for WebSocket functionality",
            "icon": "https://example.com/icon.png"
        },
        "data": {
            "items_new": [
                {
                    "url": "https://example.com/test-article-1",
                    "title": "Test Article 1 - WebSocket Test",
                    "description_text": "This is a test article for WebSocket functionality",
                    "thumbnail": "https://example.com/thumb1.jpg",
                    "date_published": "2025-01-01T12:00:00.000Z",
                    "authors": [{"name": "Test Author"}]
                }
            ],
            "items_changed": []
        }
    }

    try:
        response = requests.post(
            'http://localhost:8000/api/rss/webhook/',
            json=webhook_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )

        print(f"📥 Webhook response: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(
                f"✅ Webhook successful: {result['data']['items_created']} items created")
        else:
            print(f"❌ Webhook failed: {response.text}")

    except Exception as e:
        print(f"❌ Error sending webhook: {str(e)}")


if __name__ == "__main__":
    print("🧪 Testing WebSocket functionality...")
    test_webhook()
    asyncio.run(test_websocket())
