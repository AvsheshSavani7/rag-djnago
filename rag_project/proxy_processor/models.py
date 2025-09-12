from django.db import models
from mongoengine import Document, StringField, DateTimeField, FloatField, DictField, ListField, BooleanField
from datetime import datetime


class ProxyDocument(Document):
    """
    Model to store proxy document information and processing results.
    """

    # Basic document information
    cik_number = StringField(required=True, max_length=20)
    company_name = StringField(required=True, max_length=200)
    sec_filling_id = StringField(required=True, max_length=50)
    filing_date = StringField(required=True, max_length=20)
    form_type = StringField(required=True, max_length=20)
    proxy_sec_url = StringField(required=True, max_length=500)

    # Deal information
    # if CIK match with deal table CIK then get deal_id or null
    deal_id = StringField(max_length=50, null=True)

    # Processing information
    # AWS URL folder name: proxy-parse-jsons/
    proxy_parsing_status = StringField(max_length=20, default='pending', choices=[
        'pending', 'processing', 'completed', 'failed'
    ])

    # Processing results (from agentic SEC processor)
    empty_percentage = FloatField(default=100.0)

    # S3 URLs for cloud storage
    s3_urls = DictField(default=dict)

    # Processing state
    processing_state = DictField(default=dict)

    # Timestamps
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    completed_at = DateTimeField()

    # Error information
    error_message = StringField(max_length=1000)

    # Processing statistics
    total_sections = FloatField(default=0)
    empty_sections = FloatField(default=0)
    iteration_count = FloatField(default=0)

    # Agent response
    agent_response = StringField(max_length=10000)

    meta = {
        'collection': 'proxy_documents',
        'indexes': [
            'cik_number',
            'sec_filling_id',
            'proxy_parsing_status',
            'created_at',
            'company_name',
            'deal_id',
            'proxy_sec_url'
        ]
    }

    def __str__(self):
        return f"Proxy Document: {self.company_name} - {self.proxy_parsing_status}"

    def save(self, *args, **kwargs):
        """Override save to update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()
        super().save(*args, **kwargs)


class ProxyProcessingLog(Document):
    """
    Model to store detailed processing logs for debugging and monitoring.
    """

    # Reference to the proxy document
    proxy_document_id = StringField(required=True, max_length=100)

    # Log information
    level = StringField(max_length=20, choices=[
        'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    ])
    message = StringField(required=True, max_length=2000)
    module = StringField(max_length=100)

    # Timestamp
    timestamp = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'proxy_processing_logs',
        'indexes': [
            'proxy_document_id',
            'level',
            'timestamp'
        ]
    }

    def __str__(self):
        return f"Log: {self.level} - {self.message[:50]}... (Proxy: {self.proxy_document_id})"
