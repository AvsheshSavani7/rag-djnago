from rest_framework import serializers
from .models import SECFiling, SECFeedStatus, XBRLFile


class XBRLFileSerializer(serializers.Serializer):
    """Serializer for XBRL files"""
    sequence = serializers.IntegerField()
    file = serializers.CharField(max_length=255)
    type = serializers.CharField(max_length=50)
    size = serializers.IntegerField()
    description = serializers.CharField(max_length=255)
    inlineXBRL = serializers.BooleanField(default=False)
    url = serializers.URLField(max_length=1000)


class SECFilingSerializer(serializers.Serializer):
    """Serializer for SEC filings"""
    id = serializers.CharField(read_only=True)
    title = serializers.CharField(max_length=500)
    link = serializers.URLField(max_length=1000)
    guid = serializers.URLField(max_length=1000)
    description = serializers.CharField(max_length=50)
    pubDate = serializers.DateTimeField()

    # Enclosure information
    enclosure_url = serializers.URLField(max_length=1000)
    enclosure_length = serializers.IntegerField()
    enclosure_type = serializers.CharField(max_length=100)

    # EDGAR specific fields
    company_name = serializers.CharField(max_length=255)
    form_type = serializers.CharField(max_length=50)
    filing_date = serializers.DateTimeField()
    cik_number = serializers.CharField(max_length=20)
    accession_number = serializers.CharField(max_length=50)
    file_number = serializers.CharField(
        max_length=50, required=False, allow_null=True)
    acceptance_datetime_utc = serializers.DateTimeField()
    period = serializers.CharField(
        max_length=20, required=False, allow_null=True)
    fiscal_year_end = serializers.CharField(
        max_length=10, required=False, allow_null=True)
    assigned_sic = serializers.IntegerField(required=False, allow_null=True)

    # XBRL files
    xbrl_files = XBRLFileSerializer(many=True, required=False)

    # Processing flags
    has_htm_files = serializers.BooleanField(default=False)
    processed = serializers.BooleanField(default=False)

    # GPT Analysis fields
    is_new_deal = serializers.BooleanField(required=False, allow_null=True)
    following = serializers.BooleanField(default=False)

    # Timestamps
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)


class SECFeedStatusSerializer(serializers.Serializer):
    """Serializer for SEC feed status"""
    id = serializers.CharField(read_only=True)
    feed_url = serializers.URLField(max_length=1000)
    last_fetch_time = serializers.DateTimeField()
    last_build_date = serializers.DateTimeField()
    total_items_processed = serializers.IntegerField()
    new_items_found = serializers.IntegerField()
    error_message = serializers.CharField(
        max_length=1000, required=False, allow_null=True)
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)


class SECFilingListSerializer(serializers.Serializer):
    """Serializer for listing SEC filings with filters"""
    id = serializers.CharField(read_only=True)
    title = serializers.CharField(read_only=True)
    link = serializers.URLField(read_only=True)
    guid = serializers.URLField(read_only=True)
    description = serializers.CharField(read_only=True)
    pubDate = serializers.DateTimeField(read_only=True)
    enclosure_url = serializers.URLField(read_only=True)
    enclosure_length = serializers.IntegerField(read_only=True)
    enclosure_type = serializers.CharField(read_only=True)
    company_name = serializers.CharField(read_only=True)
    form_type = serializers.CharField(read_only=True)
    filing_date = serializers.DateTimeField(read_only=True)
    cik_number = serializers.CharField(read_only=True)
    accession_number = serializers.CharField(read_only=True)
    file_number = serializers.CharField(read_only=True)
    acceptance_datetime_utc = serializers.DateTimeField(read_only=True)
    period = serializers.CharField(read_only=True)
    fiscal_year_end = serializers.CharField(read_only=True)
    assigned_sic = serializers.IntegerField(read_only=True)
    xbrl_files = XBRLFileSerializer(many=True, read_only=True)

    has_htm_files = serializers.BooleanField(read_only=True)
    processed = serializers.BooleanField(read_only=True)

    # GPT Analysis fields
    is_new_deal = serializers.BooleanField(read_only=True)
    following = serializers.BooleanField(read_only=True)

    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)


class SECFilingDetailSerializer(SECFilingSerializer):
    """Detailed serializer for SEC filings including all fields"""
    pass
