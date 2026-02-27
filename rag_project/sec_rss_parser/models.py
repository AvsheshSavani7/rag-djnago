from mongoengine import Document, StringField, DateTimeField, IntField, ListField, DictField, BooleanField, URLField, DynamicField
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
    accession_number = StringField(required=True, max_length=50, unique=True)
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
    document_kind = StringField(required=False, null=True)
    # Always False for new deals, can be True for amendments
    following = BooleanField(default=False)
    # Processing status: "Not Started", "In Progress", "Fail", "Completed"
    following_status = StringField(default="Not Started", max_length=20)

    company_details = DynamicField(required=False, null=True)
    target_ticker = StringField(required=False, max_length=20, null=True)
    acquirer_ticker = StringField(required=False, max_length=20, null=True)
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


class LastCronJob(Document):
    """Model to store last cron job execution information"""
    _id = StringField(primary_key=True, default=generate_object_id)

    job_name = StringField(required=True, max_length=100)
    last_build_date = StringField(required=True, max_length=255)

    # Timestamps
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    # MongoDB-specific field
    v_version = IntField(default=0, db_field="__v")

    meta = {
        'collection': 'last-cron-job',
        'indexes': ['job_name']
    }

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)

    def __str__(self):
        return f"LastCronJob - {self.job_name} - {self.last_build_date}"


class AccessionLookedUp(Document):
    """Model to track accession numbers that have been looked up"""
    _id = StringField(primary_key=True, default=generate_object_id)

    accession_number = StringField(required=True, unique=True, max_length=50)

    # Timestamps
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    # MongoDB-specific field
    v_version = IntField(default=0, db_field="__v")

    meta = {
        'collection': 'accession_lookedup',
        'indexes': ['accession_number']
    }

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)

    def __str__(self):
        return f"AccessionLookedUp - {self.accession_number}"


class EightKSummary(Document):
    """Summary of an 8-K document (main 8-K filing document)."""
    _id = StringField(primary_key=True, default=generate_object_id)

    accession_number = StringField(required=True, max_length=50)
    company_name = StringField(required=False, max_length=255, null=True)
    cik_number = StringField(required=False, max_length=20, null=True)
    sec_document_url = URLField(required=True, max_length=1000)
    s3_docx_url = URLField(required=False, max_length=1000, null=True)
    s3_json_url = URLField(required=False, max_length=1000, null=True)

    ticker = StringField(required=False, max_length=20, null=True)
    filing_date = StringField(required=False, max_length=20, null=True)
    items_reported = ListField(StringField(max_length=50), default=[])

    deal_id = StringField(required=False, max_length=50, null=True)
    one_line_summary = StringField(required=False, max_length=1000, null=True)

    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': '8k_summary',
        'indexes': ['accession_number', 'cik_number', 'created_at', 'deal_id'],
    }

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)

    def __str__(self):
        return f"8K Summary - {self.accession_number} - {self.ticker or 'N/A'}"


class Ex99_1Summary(Document):
    """Summary of an EX-99.1 document (exhibit to 8-K)."""
    _id = StringField(primary_key=True, default=generate_object_id)

    accession_number = StringField(required=True, max_length=50)
    company_name = StringField(required=False, max_length=255, null=True)
    cik_number = StringField(required=False, max_length=20, null=True)
    sec_document_url = URLField(required=True, max_length=1000)
    s3_docx_url = URLField(required=False, max_length=1000, null=True)
    s3_json_url = URLField(required=False, max_length=1000, null=True)

    ticker = StringField(required=False, max_length=20, null=True)
    filing_date = StringField(required=False, max_length=20, null=True)
    items_reported = ListField(StringField(max_length=50), default=[])

    deal_id = StringField(required=False, max_length=50, null=True)
    one_line_summary = StringField(required=False, max_length=1000, null=True)

    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': '99_1_summary',
        'indexes': ['accession_number', 'cik_number', 'created_at', 'deal_id'],
    }

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)

    def __str__(self):
        return f"99.1 Summary - {self.accession_number} - {self.ticker or 'N/A'}"


class TenKTenQSummary(Document):
    """Summary record for 10-K/10-Q filings (tracking for processing)."""
    _id = StringField(primary_key=True, default=generate_object_id)

    sec_document_url = URLField(required=True, max_length=1000)
    deal_id = StringField(required=False, max_length=50, null=True)
    s3_json_url = URLField(required=False, max_length=1000, null=True)
    s3_docx_url = URLField(required=False, max_length=1000, null=True)

    cik_number = StringField(required=True, max_length=20)
    accession_number = StringField(required=True, max_length=50, unique=True)
    filing_date = StringField(required=False, max_length=20, null=True)

    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    form_type = StringField(required=False, max_length=50, null=True)

    meta = {
        'collection': '10k_10Q_Summary',
        'indexes': ['accession_number', 'cik_number', 'deal_id', 'filing_date'],
    }

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)

    def __str__(self):
        return f"10K/10Q Summary - {self.accession_number} - {self.cik_number}"
