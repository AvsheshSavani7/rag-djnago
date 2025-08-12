# RSS Feeds Module

This module provides functionality to receive and store RSS feed data via webhooks and manage feed items in MongoDB.

## Features

- **Webhook Endpoint**: Receive RSS feed updates from external services
- **Real-time WebSocket Notifications**: Send new feed items to frontend in real-time
- **Feed Management**: Create, update, and manage RSS feeds
- **Feed Items**: Store and retrieve individual feed items
- **MongoDB Integration**: Uses mongoengine for MongoDB operations
- **REST API**: Full REST API for managing feeds and items
- **Admin Interface**: Django admin interface for data management

## Database Schema

### Feed Collection
```javascript
{
  "_id": ObjectId,
  "title": String,           // Feed title
  "source_url": String,      // Original source URL
  "rss_feed_url": String,    // RSS feed URL
  "description": String,     // Feed description
  "icon": String,           // Feed icon URL
  "source": String,         // Feed source name
  "created_at": DateTime,   // Creation timestamp
  "updated_at": DateTime,   // Last update timestamp
  "__v": Number            // Version field
}
```

### Feed Items Collection
```javascript
{
  "_id": ObjectId,
  "url": String,            // Item URL
  "title": String,          // Item title
  "description_text": String, // Item description
  "thumbnail": String,      // Thumbnail URL
  "date_published": DateTime, // Publication date
  "authors": [              // Array of authors
    {
      "name": String
    }
  ],
  "rss_feed_id": String,    // Reference to parent feed
  "created_at": DateTime,   // Creation timestamp
  "updated_at": DateTime,   // Last update timestamp
  "__v": Number            // Version field
}
```

## API Endpoints

### Webhook
- **POST** `/api/rss/webhook/` - Receive RSS feed updates

### Feeds
- **GET** `/api/rss/feeds/` - List all feeds
- **GET** `/api/rss/feeds/{feed_id}/` - Get specific feed
- **GET** `/api/rss/feeds/{feed_id}/?include_items=true` - Get feed with items
- **DELETE** `/api/rss/feeds/{feed_id}/` - Delete feed and all items

### Feed Items
- **GET** `/api/rss/feeds/{feed_id}/items/` - Get items for specific feed
- **GET** `/api/rss/items/` - Get recent items across all feeds
- **GET** `/api/rss/items/{item_id}/` - Get specific feed item

### WebSocket
- **WebSocket** `/api/rss/socket.io/` - Real-time RSS feed updates

## Webhook Payload Format

The webhook expects the following JSON structure:

```json
{
  "id": "webhook_event_id",
  "type": "feed_update",
  "feed": {
    "id": "feed_id",
    "title": "Feed Title",
    "source_url": "https://example.com/source",
    "rss_feed_url": "https://rss.app/feeds/feed_id.xml",
    "description": "Feed description",
    "icon": "https://example.com/icon.png"
  },
  "data": {
    "items_new": [
      {
        "url": "https://example.com/article",
        "title": "Article Title",
        "description_text": "Article description",
        "thumbnail": "https://example.com/thumbnail.jpg",
        "date_published": "2025-01-01T00:00:00.000Z",
        "authors": [
          {
            "name": "Author Name"
          }
        ]
      }
    ],
    "items_changed": []
  }
}
```

## Usage Examples

### Testing the Webhook

```python
import requests
import json

webhook_url = "http://localhost:8000/api/rss/webhook/"
test_data = {
    "id": "test_event",
    "type": "feed_update",
    "feed": {
        "id": "test_feed",
        "title": "Test Feed",
        "source_url": "https://example.com",
        "rss_feed_url": "https://rss.app/feeds/test.xml",
        "description": "Test feed description"
    },
    "data": {
        "items_new": [
            {
                "url": "https://example.com/article",
                "title": "Test Article",
                "description_text": "Test description",
                "date_published": "2025-01-01T00:00:00.000Z",
                "authors": [{"name": "Test Author"}]
            }
        ],
        "items_changed": []
    }
}

response = requests.post(webhook_url, json=test_data)
print(response.json())
```

### Getting All Feeds

```python
import requests

response = requests.get("http://localhost:8000/api/rss/feeds/")
feeds = response.json()
print(f"Found {feeds['count']} feeds")
```

### Getting Recent Items

```python
import requests

response = requests.get("http://localhost:8000/api/rss/items/?limit=10")
items = response.json()
print(f"Found {items['count']} recent items")
```

## Configuration

1. Add `rss_feeds` to `INSTALLED_APPS` in `settings.py`
2. Include RSS feed URLs in main URL configuration
3. Ensure MongoDB connection is configured

## WebSocket Events

### Client to Server Events
- `connect` - Connect to WebSocket server
- `disconnect` - Disconnect from WebSocket server
- `join_feed` - Join a specific feed room (data: `{feed_id: "string"}`)
- `leave_feed` - Leave a specific feed room (data: `{feed_id: "string"}`)
- `get_recent_items` - Get recent feed items (data: `{limit: number, feed_id?: "string"}`)

### Server to Client Events
- `connected` - Connection confirmation
- `rss_update` - RSS feed updates (types: `new_feed_items`, `feed_update`, `error`)
- `joined_feed` - Confirmation of joining feed room
- `left_feed` - Confirmation of leaving feed room
- `recent_items` - Response with recent feed items
- `error` - Error notifications

### Example WebSocket Usage (JavaScript)

```javascript
// Connect to WebSocket
const socket = io('http://localhost:8000/api/rss/socket.io/', {
    namespace: '/rss',
    transports: ['websocket', 'polling']
});

// Listen for RSS updates
socket.on('rss_update', function(data) {
    if (data.type === 'new_feed_items') {
        console.log(`New ${data.count} items from ${data.feed.title}`);
        // Handle new feed items
        data.items.forEach(item => {
            // Update UI with new item
        });
    }
});

// Join specific feed room
socket.emit('join_feed', { feed_id: 'feed_id_here' });

// Get recent items
socket.emit('get_recent_items', { limit: 10 });
```

## Testing

Run the test script to verify functionality:

```bash
cd rag_project
python test_webhook.py
```

### WebSocket Testing

1. Start the Django server:
```bash
python manage.py runserver
```

2. Open the WebSocket example page:
```
http://localhost:8000/api/rss/static/rss_websocket_example.html
```

3. Connect to WebSocket and test real-time updates

### Live Deployment

The RSS feed system is deployed at:
- **Webhook URL**: `https://rag-django-sq2f.onrender.com/api/rss/webhook/`
- **API Base URL**: `https://rag-django-sq2f.onrender.com/api/rss/`
- **WebSocket URL**: `https://rag-django-sq2f.onrender.com/api/rss/socket.io/`

### Testing Live Deployment

```bash
# Test webhook
curl -X POST https://rag-django-sq2f.onrender.com/api/rss/webhook/ \
  -H "Content-Type: application/json" \
  -d '{"id":"test","type":"feed_update","feed":{"id":"test","title":"Test","source_url":"https://example.com","rss_feed_url":"https://rss.app/feeds/test.xml","description":"Test","icon":"https://example.com/icon.png"},"data":{"items_new":[],"items_changed":[]}}'

# Test API
curl https://rag-django-sq2f.onrender.com/api/rss/feeds/
curl https://rag-django-sq2f.onrender.com/api/rss/items/
```

## Admin Interface

Access the Django admin interface to manage feeds and items:

1. Create a superuser: `python manage.py createsuperuser`
2. Access admin at: `http://localhost:8000/admin/`
3. Navigate to "RSS Feeds" section

## Error Handling

The system includes comprehensive error handling:

- Invalid webhook payloads are rejected with detailed error messages
- Duplicate feed items are automatically skipped
- Database errors are logged and handled gracefully
- All API endpoints return consistent error responses

## Security

- Webhook endpoint allows anonymous access (required for external services)
- Other endpoints require authentication
- Input validation on all endpoints
- SQL injection protection through mongoengine
