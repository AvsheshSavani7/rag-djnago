from mongoengine import (
    Document,
    StringField,
    DateTimeField,
    IntField,
    ListField,
    DictField,
    BooleanField,
    URLField,
    DynamicField,
    ObjectIdField,
)
from datetime import datetime
import uuid
from bson import ObjectId


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


class AccessionProcessingLock(Document):
    """
    Distributed lock per accession_number to prevent concurrent processing
    across parallel workers/flows.
    """
    _id = StringField(primary_key=True, default=generate_object_id)

    accession_number = StringField(required=True, unique=True, max_length=50)
    owner_id = StringField(required=True, max_length=100)
    expires_at = DateTimeField(required=True)

    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    v_version = IntField(default=0, db_field="__v")

    meta = {
        'collection': 'accession_processing_lock',
        'indexes': [
            {'fields': ['accession_number'], 'unique': True},
            # TTL index: delete document when expires_at is reached
            {'fields': ['expires_at'], 'expireAfterSeconds': 0},
        ]
    }

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)

    def __str__(self):
        return f"AccessionProcessingLock - {self.accession_number} - {self.owner_id}"


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


class DealDmaSummary(Document):
    """
    Store DMA (8-K) summary generation details for a deal.

    Requirements:
    - `deal_id` is unique
    - if record exists, update it (upsert)
    - maintain `createdAt` / `updatedAt` timestamps
    """

    deal_id = ObjectIdField(required=True, unique=True)

    summary_docx_url = URLField(required=False, null=True, max_length=1000)
    summary_status = StringField(required=True, max_length=20)
    summary_using = StringField(required=False, null=True, max_length=100)

    createdAt = DateTimeField(default=datetime.utcnow)
    updatedAt = DateTimeField(default=datetime.utcnow)

    meta = {
        "collection": "deal_dma_summary",
        "indexes": [
            "deal_id",  # unique
            "-createdAt",
        ],
    }

    def save(self, *args, **kwargs):
        self.updatedAt = datetime.utcnow()
        return super().save(*args, **kwargs)

    @classmethod
    def save_or_update(
        cls,
        *,
        deal_id,
        summary_status: str,
        summary_docx_url=None,
        summary_using=None,
    ):
        now = datetime.utcnow()

        if isinstance(deal_id, str):
            deal_id = ObjectId(deal_id)

        existing = cls.objects(deal_id=deal_id).first()
        if existing:
            existing.summary_status = summary_status
            existing.summary_docx_url = summary_docx_url
            existing.summary_using = summary_using
            return existing.save()

        record = cls(
            deal_id=deal_id,
            summary_status=summary_status,
            summary_docx_url=summary_docx_url,
            summary_using=summary_using,
            createdAt=now,
            updatedAt=now,
        )
        return record.save()


class SECFilingSummary(Document):
    """
    Unified summary collection for all SEC form types (8-K, 10-K/10-Q, proxy).
    One document per filing; only the nested object for that form_type is set, others are null.
    """
    _id = StringField(primary_key=True, default=generate_object_id)

    accession_number = StringField(required=False, max_length=50, null=True)
    cik_number = StringField(required=False, max_length=20, null=True)
    sec_document_url = StringField(required=True, max_length=2000)
    # Normalized date for filtering (8-K: MM/DD/YY, 10-K/10-Q: YYYY-MM-DD, proxy: YYYY-MM-DD)
    filing_date = DateTimeField(required=False, null=True)
    deal_id = StringField(required=False, max_length=50, null=True)

    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    form_type = StringField(required=True, max_length=50)
    items_reported = ListField(StringField(max_length=50), default=[])
    L1_headline = StringField(required=False, max_length=1000, null=True)
    L2_brief = StringField(required=False, max_length=1000, null=True)
    L3_detailed = DictField(required=False, null=True, default={})
    s3_docx_url = StringField(required=False, max_length=2000, null=True)
    s3_json_url = StringField(required=False, max_length=2000, null=True)

    # EX-99.1 exhibit summary: { items_reported, L1_headline, L2_brief, L3_detailed, s3_docx_url, s3_json_url }
    # When set, parent-level summary fields are typically null.
    ex99_1 = DictField(null=True, db_field="99_1")

    # When form_type is proxy (DEFM14A, DEF 14A, etc.): only proxy is set
    # Proxy node schema (all fields stored as dict):
    # {
    #   "proxy_parsing_status": "pending|processing|completed|failed",
    #   "empty_percentage": float (0-100),
    #   "processing_state": {
    #     "pdf_created": bool,
    #     "toc_found": bool,
    #     "toc_extracted": bool,
    #     "sections_extracted": bool,
    #     "empty_percentage": float,
    #     "iteration_count": int
    #   },
    #   "s3_urls": {
    #     "pdf_url": str|None,
    #     "toc_pdf_url": str|None,
    #     "toc_json_url": str|None,
    #     "sections_json_url": str|None
    #   },
    #   "pinecone_processing_status": "pending|processing|completed|failed",
    #   "pinecone_processed_at": datetime|None,
    #   "pinecone_error_message": str|None,
    #   "summary_generation_status": "pending|processing|completed|failed",
    #   "summary_docx_url": str|None,
    #   "summary_generated_at": datetime|None,
    #   "agent_response": str|None,
    #   "error_message": str|None,
    #   "completed_at": datetime|None,
    #   "total_sections": int,
    #   "empty_sections": int,
    #   "iteration_count": int
    #   "comparison": {   # Proxy comparison pipeline (S3 + MongoDB only)
    #     "cache": {
    #       "status": "pending|building|ready|error",
    #       "form_type": str,
    #       "form_family": str,
    #       "filing_date": str,
    #       "priority_facts_url": str,   # S3 URL (summary_json/proxy_comp_...)
    #       "topic_blocks_url": str,
    #       "sections_url": str,
    #       "error": str|None
    #     },
    #     "result": {
    #       "status": "pending|complete|error",
    #       "changes_json_url": str,
    #       "change_txt_url": str,
    #       "change_docx_url": str,
    #       "tier1_changes": int,
    #       "tier2_changes": int,
    #       "completed_at": str (ISO datetime)
    #     }
    #   }
    # }
    proxy = DictField(null=True)

    # When form_type is 10-K or 10-Q: only ten_k_ten_q is set
    ten_k_ten_q = DictField(null=True)

    # When form_type is 8-K: only 8_k is set. Exhibit 99.1 is inside 8_k.filings[], never a separate top-level record.
    eight_k = DictField(null=True, db_field="8_k")

    # Future expansion; must always be present, null for now.
    other_filings = DictField(null=True, db_field="other_filings")

    meta = {
        'collection': 'sec_filing_summary',
        # Uses Deal_DB_New (MONGODB_CONNECTION_STRING_NEW)
        'indexes': [
            'accession_number',
            'cik_number',
            'deal_id',
            'filing_date',
            'form_type',
            'sec_document_url',
        ],
    }

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)

    def __str__(self):
        return f"SECFilingSummary - {self.form_type} - {self.accession_number or self.sec_document_url[:50]}"
