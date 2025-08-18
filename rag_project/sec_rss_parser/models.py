from mongoengine import Document, StringField, DateTimeField, IntField, ListField, DictField, BooleanField, URLField
from datetime import datetime
import uuid


def generate_object_id():
    return str(uuid.uuid4())


class XBRLFile(Document):
    """Model for XBRL files within SEC filings"""
    sequence = IntField(required=True)
    file = StringField(required=True, max_length=255)
    type = StringField(required=True, max_length=50)
    size = IntField(required=True)
    description = StringField(required=True, max_length=255)
    inlineXBRL = BooleanField(default=False)
    url = URLField(required=True, max_length=1000)

    meta = {
        'collection': 'xbrl_files',
        'indexes': ['sequence', 'type', 'url']
    }


class SECFiling(Document):
    """Model for SEC RSS feed filings"""
    _id = StringField(primary_key=True, default=generate_object_id)

    # Basic RSS item fields
    title = StringField(required=True, max_length=500)
    link = URLField(required=True, max_length=1000)
    guid = URLField(required=True, max_length=1000)
    # Form type (8-K, 10-Q, etc.)
    description = StringField(required=True, max_length=50)
    pubDate = DateTimeField(required=False, null=True)

    # Enclosure information
    enclosure_url = URLField(required=False, max_length=1000, null=True)
    enclosure_length = IntField(required=False, null=True)
    enclosure_type = StringField(required=False, max_length=100, null=True)

    # EDGAR specific fields
    company_name = StringField(required=True, max_length=255)
    form_type = StringField(required=True, max_length=50)
    filing_date = DateTimeField(required=False, null=True)
    cik_number = StringField(required=True, max_length=20)
    accession_number = StringField(required=True, max_length=50)
    file_number = StringField(required=False, max_length=50, null=True)
    acceptance_datetime_utc = DateTimeField(required=False, null=True)
    period = StringField(required=False, max_length=20, null=True)
    fiscal_year_end = StringField(required=False, max_length=10, null=True)
    assigned_sic = IntField(required=False, null=True)

    # XBRL files
    xbrl_files = ListField(DictField(), required=False, default=[])

    # Processing flags
    has_htm_files = BooleanField(default=False)
    processed = BooleanField(default=False)

    # GPT Analysis fields
    # True=new deal, False=amendment, None=not analyzed
    is_new_deal = BooleanField(required=False, null=True)
    # Always False for new deals, can be True for amendments
    following = BooleanField(default=False)

    # Timestamps
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    # MongoDB-specific field
    v_version = IntField(default=0, db_field="__v")

    meta = {
        'collection': 'sec_filings',
        'indexes': [
            'cik_number',
            'form_type',
            'filing_date',
            'accession_number',
            'has_htm_files',
            'processed'
        ]
    }

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.company_name} - {self.form_type} - {self.filing_date}"


class SECFeedStatus(Document):
    """Model to track RSS feed processing status"""
    _id = StringField(primary_key=True, default=generate_object_id)

    feed_url = URLField(required=True, max_length=1000)
    last_fetch_time = DateTimeField(required=True)
    last_build_date = DateTimeField(required=True)
    total_items_processed = IntField(default=0)
    new_items_found = IntField(default=0)
    error_message = StringField(required=False, max_length=1000)

    # Timestamps
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    # MongoDB-specific field
    v_version = IntField(default=0, db_field="__v")

    meta = {
        'collection': 'sec_feed_status',
        'indexes': ['feed_url', 'last_fetch_time']
    }

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)

    def __str__(self):
        return f"SEC Feed Status - {self.last_fetch_time}"
