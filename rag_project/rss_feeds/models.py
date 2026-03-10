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
)
from datetime import datetime
import uuid


class Author(EmbeddedDocument):
    """Embedded document for author information"""
    name = StringField(required=True, max_length=255)


class FeedItem(Document):
    """Model for RSS feed items"""

    # Feed item information
    url = URLField(required=True, max_length=1000)
    title = StringField(required=True, max_length=500)
    description_text = StringField(required=False, max_length=2000)
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
