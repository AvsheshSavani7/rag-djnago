from mongoengine import (
    Document,
    StringField,
    URLField,
    DateTimeField,
    ListField,
    ReferenceField,
    EmbeddedDocumentField,
    EmbeddedDocument,
    IntField,
    DictField,
    BooleanField,
)
from datetime import datetime
import uuid


class Author(EmbeddedDocument):
    """Embedded document for author information"""
    name = StringField(required=True, max_length=255)


class FeedItem(Document):
    """Model for RSS feed items"""

    # Feed item information
    url = URLField(required=True, max_length=2000)
    title = StringField(required=True, max_length=500)
    description_text = StringField(required=False, max_length=4000)
    thumbnail = URLField(required=False, max_length=1000)
    date_published = DateTimeField(required=True)
    authors = ListField(EmbeddedDocumentField(
        Author), required=False, default=[])

    # Optional AI summary fields (from sec_rss_parser.sec_summarizers.filing_router.route_and_summarize)
    l1_headline = StringField(required=False, max_length=500, null=True)
    l2_brief = StringField(required=False, max_length=4000, null=True)
    l3_detailed = DictField(required=False, null=True)  # full JSON blob
    s3_docx_url = URLField(required=False, max_length=1000, null=True)
    s3_json_url = URLField(required=False, max_length=1000, null=True)

    # Reference to parent feed
    rss_feed_id = StringField(required=True, max_length=50)

    # Optional link to deal (when article is merger-related and matched or created)
    deal_id = StringField(required=False, max_length=50, null=True)

    # Timestamps
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    # MongoDB-specific field
    v_version = IntField(default=0, db_field="__v")

    meta = {
        'collection': 'feed_items',
        'ordering': ['-date_published', '-created_at'],
        'indexes': [
            'rss_feed_id',
            'date_published',
            'url',
            ('rss_feed_id', 'date_published'),
        ]
    }

    def __str__(self):
        return f"Feed Item: {self.title[:50]}... (ID: {str(self.id)})"

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)


class Feed(Document):
    """Model for RSS feeds"""

    # Feed information
    title = StringField(required=True, max_length=255)
    source_url = URLField(required=True, max_length=1000)
    rss_feed_url = URLField(required=True, max_length=1000)
    description = StringField(required=False, max_length=500)
    icon = URLField(required=False, max_length=1000)
    source = StringField(required=False, max_length=100)

    # Timestamps
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    # MongoDB-specific field
    v_version = IntField(default=0, db_field="__v")

    meta = {
        'collection': 'feeds',
        'ordering': ['-created_at'],
        'indexes': [
            'title',
            'source_url',
            'rss_feed_url',
        ]
    }

    def __str__(self):
        return f"Feed: {self.title} (ID: {str(self.id)})"

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)

    def get_feed_items(self, limit=50):
        """Get feed items for this feed"""
        return FeedItem.objects(rss_feed_id=str(self.id)).order_by('-date_published').limit(limit)


class NewsSourceConfig(Document):
    """Feed builder source config. HTML uses selectors; RSS uses element_map. Upsert by source_url."""

    source_id = StringField(required=True, max_length=100, unique=True)
    source_name = StringField(required=True, max_length=255)
    source_type = StringField(required=True, max_length=20)  # rss | html
    source_url = URLField(required=True, max_length=2000, unique=True)

    fetch_mode = StringField(default="requests", max_length=20)
    selectors = DictField(default=dict)  # HTML: CSS selectors
    element_map = DictField(default=dict)  # RSS: feedparser field names
    url_rules = DictField(default=dict)

    is_active = BooleanField(default=True)
    poll_interval_minutes = IntField(default=10)

    last_checked_at = DateTimeField(null=True)
    last_success_at = DateTimeField(null=True)
    last_error = StringField(null=True)
    consecutive_failures = IntField(default=0)

    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    v_version = IntField(default=0, db_field="__v")

    meta = {
        "collection": "news_source_configs",
        "ordering": ["-updated_at"],
        "indexes": [
            "source_id",
            "source_url",
            "is_active",
        ],
    }

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "fetch_mode": self.fetch_mode,
            "selectors": self.selectors or {},
            "element_map": self.element_map or {},
            "url_rules": self.url_rules or {},
            "is_active": self.is_active,
            "poll_interval_minutes": self.poll_interval_minutes,
            "last_checked_at": self.last_checked_at.isoformat() if self.last_checked_at else None,
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
            "last_error": self.last_error,
            "consecutive_failures": self.consecutive_failures or 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class NewsArticleLink(Document):
    """Discovered article URLs from feed builder scanner (is_processed=False until pipeline runs)."""

    source_id = StringField(required=True, max_length=100)
    source_name = StringField(max_length=255)
    source_type = StringField(max_length=20)
    source_url = URLField(max_length=2000)

    title = StringField(max_length=2000, null=True)
    detail_url = URLField(required=True, max_length=2000)
    published_at = DateTimeField(null=True)
    description = StringField(max_length=4000, null=True)
    author = StringField(max_length=500, null=True)
    image = URLField(max_length=2000, null=True)
    guid = StringField(max_length=500, null=True)

    url_hash = StringField(required=True, max_length=64)
    dedupe_key = StringField(required=True, max_length=255, unique=True)

    is_processed = BooleanField(default=False)
    processed_at = DateTimeField(null=True)

    first_seen_at = DateTimeField(default=datetime.utcnow)
    last_seen_at = DateTimeField(null=True)
    raw_data = DictField(default=dict)

    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    v_version = IntField(default=0, db_field="__v")

    meta = {
        "collection": "news_article_links",
        "ordering": ["-first_seen_at"],
        "indexes": [
            "source_id",
            "url_hash",
            "dedupe_key",
            "is_processed",
            ("source_id", "is_processed"),
        ],
    }

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)
