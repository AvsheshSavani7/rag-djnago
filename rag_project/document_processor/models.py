from mongoengine import (
    Document,
    StringField,
    URLField,
    DateTimeField,
    BooleanField,
    IntField,
    # DictField,
    DynamicField,
    ReferenceField,
    ListField,
)
from datetime import datetime
import json


class ProcessingJob(Document):
    """Model to track document processing jobs"""

    EMBEDDING_STATUS_CHOICES = ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED')
    SUMMARY_STATUS_CHOICES = ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED')

    # Deal information
    cik = StringField(max_length=20, required=False, null=True)
    acquire_name = StringField(max_length=255, required=False, null=True)
    target_name = StringField(max_length=255, required=False, null=True)
    announce_date = DateTimeField(required=False, null=True)
    embedding_status = StringField(
        max_length=20,
        choices=EMBEDDING_STATUS_CHOICES,
        default='PENDING'
    )
    summary_status = StringField(
        max_length=20,
        choices=SUMMARY_STATUS_CHOICES,
        default='PENDING'
    )
    sec_filing_id = StringField(max_length=100, required=False, null=True)

    # Processing fields
    file_url = URLField(max_length=1000, required=True)
    pdf_url = URLField(max_length=1000, required=True)
    parsed_json_url = URLField(max_length=1000, required=False, null=True)
    flattened_json_url = URLField(max_length=1000, required=False, null=True)
    summary_docx_url = URLField(max_length=1000, required=False, null=True)
    sec_url = URLField(max_length=1000, required=False, null=True)

    # Schema parsing results
    # schema_results = DictField(null=True)
    schema_results = DynamicField(null=True)
    schema_processing_completed = BooleanField(default=False)
    schema_processing_timestamp = DateTimeField(required=False, null=True)

    # Twitter search processing flags
    RF1_approach_done = BooleanField(default=False)
    RF2_approach_done = BooleanField(default=False)
    RF3_approach_done = BooleanField(default=False)

    # Twitter handles information
    # Store Twitter handles for companies and subsidiaries
    twitter_details = DynamicField(default=[])

    # Error information
    error_message = StringField(required=False, null=True)

    # Timestamps
    createdAt = DateTimeField(default=datetime.utcnow)
    updatedAt = DateTimeField(default=datetime.utcnow)

    # MongoDB-specific field
    v_version = IntField(default=0, db_field="__v")

    meta = {
        'collection': 'deals',
        'ordering': ['-createdAt']
    }

    def __getattr__(self, name):
        """Override __getattr__ to handle schema_results conversion"""
        if name == 'schema_results':
            value = super().__getattribute__('schema_results')
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return {}
            return value or {}
        return super().__getattr__(name)

    def __str__(self):
        return f"Deal: {self.acquire_name}/{self.target_name} (ID: {str(self.id)})"

    def update_embedding_status(self, status, error_message=None):
        """Update the embedding status of the job"""
        self.embedding_status = status
        if error_message:
            self.error_message = error_message
        self.updatedAt = datetime.utcnow()

        print(f"Schema results: {self.schema_results}")

        # if self.schema_results is not None and not isinstance(
        #         self.schema_results, dict
        #     ):
        #     return el

        self.save()
        print(f"Updated embedding status to {status}")
        return self

    def save_json_to_db(self, results, error_message=None):
        """Save schema results and mark processing complete"""
        self.schema_results = results
        self.schema_processing_completed = True
        self.schema_processing_timestamp = datetime.utcnow()
        if error_message:
            self.error_message = error_message
        self.updatedAt = datetime.utcnow()
        self.save()
        return self

    def upsert_json_to_db(self, new_results, error_message=None):
        """
        Update the schema results by merging with existing data (upsert).
        For each section in new_results, if it already exists in schema_results,
        completely replace the section data with new data.
        For new sections, add them to the existing schema_results.
        """
        # Convert schema_results to dict if it's a string
        current_results = {}
        if isinstance(self.schema_results, str):
            try:
                current_results = json.loads(self.schema_results)
            except json.JSONDecodeError:
                current_results = {}
        elif self.schema_results:
            current_results = self.schema_results

        # Merge new results with existing data
        for section_name, section_data in new_results.items():
            # Replace entire section data if section exists, otherwise add it
            current_results[section_name] = section_data

        # Store the updated results
        self.schema_results = current_results

        # Update processing status
        self.schema_processing_completed = True
        self.schema_processing_timestamp = datetime.now()
        self.save()
        return self


class SearchQuery(Document):
    """Model to store Twitter search queries"""

    # Search query information
    search_query = StringField(max_length=1000, required=True)
    deal_id = StringField(max_length=50, required=True)
    approach = StringField(max_length=10, default="RF1")
    # Stores company combination data
    combination = DynamicField(required=True)
    # Total number of tweets found for this search query
    total_tweets = IntField(default=0)

    # Timestamps
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'search_queries',
        'ordering': ['-created_at'],
        'indexes': [
            'deal_id',
            'approach',
            'created_at'
        ]
    }

    def __str__(self):
        return f"SearchQuery: {self.search_query[:50]}... (Deal: {self.deal_id})"


class Tweet(Document):
    """Model to store individual tweets"""

    # Reference to search query
    search_query_id = ReferenceField(SearchQuery, required=True)

    # Tweet data
    tweet = DynamicField(required=True)  # Stores the complete tweet object

    # Timestamps
    created_at = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'tweets',
        'ordering': ['-created_at'],
        'indexes': [
            'search_query_id',
            'created_at'
        ]
    }

    def __str__(self):
        tweet_text = ""
        if isinstance(self.tweet, dict) and 'text' in self.tweet:
            tweet_text = self.tweet['text'][:50]
        return f"Tweet: {tweet_text}... (Query: {self.search_query_id.id})"


class CompanyProducts(Document):
    """Model to store company product lists extracted by GPT"""

    # Company and deal information
    deal_id = StringField(max_length=50, required=True)
    company = StringField(max_length=255, required=True)
    company_type = StringField(max_length=20, choices=[
                               'target', 'acquire'], required=True)

    # Product information
    # Structured product data with categories and descriptions
    products = DynamicField()  # Store the complete structured JSON data

    # Processing metadata
    gpt_model_used = StringField(max_length=50, default="gpt-4.1")
    extraction_timestamp = DateTimeField(default=datetime.utcnow)
    processing_status = StringField(
        max_length=20, choices=['pending', 'completed', 'failed'], default='pending')

    # Timestamps
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'company_products',
        'ordering': ['-created_at'],
        'indexes': [
            'deal_id',
            'company_type',
            'company',
            'created_at'
        ]
    }

    def __str__(self):
        return f"CompanyProducts: {self.company} ({self.company_type}) - {len(self.products)} products (Deal: {self.deal_id})"


class CompetitiveAnalysis(Document):
    """Model to store competitive product analysis results"""

    # Deal and company references
    deal_id = StringField(max_length=50, required=True)
    target_company_products = ReferenceField(CompanyProducts, required=True)
    acquire_company_products = ReferenceField(CompanyProducts, required=True)

    # Competitive analysis results
    # Array of competitive product pairs
    competitive_pairs = ListField(DynamicField())
    # Example structure: [{"target_product": "Product A", "acquire_product": "Product 1", "competition_score": 0.85, "analysis": "..."}]

    # Analysis metadata
    gpt_model_used = StringField(max_length=50, default="gpt-4.1")
    analysis_timestamp = DateTimeField(default=datetime.utcnow)
    processing_status = StringField(
        max_length=20, choices=['pending', 'completed', 'failed'], default='pending')

    # Timestamps
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'competitive_products',
        'ordering': ['-created_at'],
        'indexes': [
            'deal_id',
            'created_at',
            'processing_status'
        ]
    }

    def __str__(self):
        return f"CompetitiveAnalysis: Deal {self.deal_id} - {len(self.competitive_pairs)} pairs"
