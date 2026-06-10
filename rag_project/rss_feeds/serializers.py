from rest_framework import serializers
from .models import Feed, FeedItem, Author


class AuthorSerializer(serializers.Serializer):
    """Serializer for Author embedded document"""
    name = serializers.CharField(max_length=255)


class FeedItemSerializer(serializers.Serializer):
    """Serializer for FeedItem model"""
    id = serializers.CharField(read_only=True)
    url = serializers.URLField(max_length=1000)
    title = serializers.CharField(max_length=500)
    description_text = serializers.CharField(
        max_length=2000, required=False, allow_blank=True)
    thumbnail = serializers.URLField(
        max_length=1000, required=False, allow_blank=True)
    date_published = serializers.DateTimeField()
    authors = AuthorSerializer(many=True, required=False)
    rss_feed_id = serializers.CharField(max_length=50)
    # Optional AI summary fields
    l1_headline = serializers.CharField(
        max_length=500, required=False, allow_blank=True, allow_null=True
    )
    l2_brief = serializers.CharField(
        max_length=4000, required=False, allow_blank=True, allow_null=True
    )
    l3_detailed = serializers.JSONField(required=False, allow_null=True)
    s3_docx_url = serializers.URLField(
        max_length=1000, required=False, allow_blank=True, allow_null=True
    )
    s3_json_url = serializers.URLField(
        max_length=1000, required=False, allow_blank=True, allow_null=True
    )
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)


class FeedItemWithSourceSerializer(FeedItemSerializer):
    """Serializer for FeedItem model with source field from parent feed"""
    source = serializers.CharField(
        max_length=100, required=False, allow_blank=True, allow_null=True)


class FeedSerializer(serializers.Serializer):
    """Serializer for Feed model"""
    id = serializers.CharField(read_only=True)
    title = serializers.CharField(max_length=255)
    source_url = serializers.URLField(max_length=1000)
    rss_feed_url = serializers.URLField(max_length=1000)
    description = serializers.CharField(
        max_length=500, required=False, allow_blank=True, allow_null=True)
    icon = serializers.URLField(
        max_length=1000, required=False, allow_blank=True, allow_null=True)
    source = serializers.CharField(
        max_length=100, required=False, allow_blank=True, allow_null=True)
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)


class FeedWithItemsSerializer(FeedSerializer):
    """Serializer for Feed model with feed items"""
    feed_items = FeedItemSerializer(many=True, read_only=True)


class WebhookPayloadSerializer(serializers.Serializer):
    """Serializer for webhook payload validation"""
    id = serializers.CharField()
    type = serializers.CharField()
    feed = serializers.DictField()
    data = serializers.DictField()

    def validate_feed(self, value):
        """Validate feed data"""
        required_fields = ['id', 'title', 'source_url', 'rss_feed_url']
        for field in required_fields:
            if field not in value:
                raise serializers.ValidationError(
                    f"Feed data missing required field: {field}")
        return value

    def validate_data(self, value):
        """Validate data field"""
        if 'items_new' not in value:
            raise serializers.ValidationError(
                "Data must contain 'items_new' field")
        return value


class FeedItemCreateSerializer(serializers.Serializer):
    """Serializer for creating feed items from webhook data"""
    url = serializers.URLField(max_length=2000)
    title = serializers.CharField(max_length=500)
    description_text = serializers.CharField(
        max_length=4000, required=False, allow_blank=True, allow_null=True)
    thumbnail = serializers.URLField(
        max_length=1000, required=False, allow_blank=True, allow_null=True)
    date_published = serializers.DateTimeField()
    authors = AuthorSerializer(many=True, required=False, allow_null=True)
    # Optional AI summary fields attached before saving
    l1_headline = serializers.CharField(
        max_length=500, required=False, allow_blank=True, allow_null=True
    )
    l2_brief = serializers.CharField(
        max_length=4000, required=False, allow_blank=True, allow_null=True
    )
    l3_detailed = serializers.JSONField(required=False, allow_null=True)
    s3_docx_url = serializers.URLField(
        max_length=1000, required=False, allow_blank=True, allow_null=True
    )
    s3_json_url = serializers.URLField(
        max_length=1000, required=False, allow_blank=True, allow_null=True
    )
