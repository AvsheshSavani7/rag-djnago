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
    GUNSHOT_approach_done = BooleanField(default=False)

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
        if error_message:
            self.error_message = error_message
        self.updatedAt = datetime.utcnow()
        self.save()
        return self


class SearchQuery(Document):
    """Model to store Twitter search queries"""

    # Search query information
    # Increased from 1000 to 5000 to handle all products
    search_query = StringField(max_length=5000, required=True)
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
    approach = StringField(max_length=40, default="")

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


class Followers(Document):
    """Model to store company followers data (chunked to handle large datasets)"""

    # Deal and company information
    deal_id = StringField(max_length=50, required=True)
    # Twitter handle without @
    company_handle = StringField(max_length=100, required=True)
    company_name = StringField(max_length=255, required=True)

    # Followers data (chunked)
    followers = ListField(DynamicField())  # Array of follower objects
    chunk_index = IntField(default=0)  # Which chunk this is (0, 1, 2, etc.)
    chunk_size = IntField(default=1000)  # Number of followers per chunk

    # Processing metadata
    approach = StringField(max_length=20, default="GUNSHOT")
    total_followers = IntField(default=0)
    processing_status = StringField(
        max_length=20, choices=['pending', 'completed', 'failed'], default='pending')

    # Timestamps
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'followers',
        'ordering': ['-created_at'],
        'indexes': [
            'deal_id',
            'company_handle',
            'company_name',
            'approach',
            'created_at',
            # Compound index for efficient chunking
            ('deal_id', 'company_handle', 'chunk_index')
        ]
    }

    def __str__(self):
        return f"Followers: {self.company_name} (@{self.company_handle}) - Chunk {self.chunk_index} - {len(self.followers)} followers (Deal: {self.deal_id})"


class FollowersMetadata(Document):
    """Model to store metadata about follower collections"""
    deal_id = StringField(max_length=50, required=True)
    company_handle = StringField(max_length=100, required=True)
    company_name = StringField(max_length=255, required=True)

    # Metadata
    total_followers = IntField(default=0)
    total_chunks = IntField(default=0)
    approach = StringField(max_length=20, default="GUNSHOT")
    processing_status = StringField(
        max_length=20, choices=['pending', 'completed', 'failed'], default='pending')

    # Timestamps
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'followers_metadata',
        'ordering': ['-created_at'],
        'indexes': [
            'deal_id', 'company_handle', 'approach', 'created_at'
        ]
    }

    def __str__(self):
        return f"Followers Metadata: {self.company_name} (@{self.company_handle}) - {self.total_followers} followers in {self.total_chunks} chunks (Deal: {self.deal_id})"


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
    gpt_model_used = StringField(max_length=50, default="gpt-4.1-mini")
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
    gpt_model_used = StringField(max_length=50, default="gpt-4.1-mini")
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


class HighValueFollowers(Document):
    """Model to store high-value followers with GPT analysis scores"""

    # Deal and company information
    deal_id = StringField(max_length=50, required=True)
    company_name = StringField(max_length=255, required=True)
    company_handle = StringField(max_length=100, required=True)

    # Follower information (spread from original follower object)
    follower_id = StringField(max_length=50, required=True)
    name = StringField(max_length=255, required=False)
    screen_name = StringField(max_length=100, required=False)
    description = StringField(max_length=1000, required=False)
    location = StringField(max_length=255, required=False)
    followers_count = IntField(default=0)
    statuses_count = IntField(default=0)
    protected = BooleanField(default=False)
    verified = BooleanField(default=False)
    created_at_twitter = DateTimeField(required=False, null=True)

    # GPT Analysis results
    overall_score = IntField(required=True)  # 0-10 score
    # Explanation for the score
    reason = StringField(max_length=500, required=False)
    # List of key indicators from bio
    key_indicators = ListField(StringField(), default=[])
    analysis_timestamp = DateTimeField(default=datetime.utcnow)
    gpt_model_used = StringField(max_length=50, default="gpt-3.5-turbo")

    # Processing metadata
    processing_status = StringField(
        max_length=20, choices=['pending', 'completed', 'failed'], default='completed')
    approach = StringField(max_length=20, default="GUNSHOT")

    # Timestamps
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'high_value_followers',
        'ordering': ['-overall_score', '-created_at'],
        'indexes': [
            'deal_id',
            'company_handle',
            'company_name',
            'overall_score',
            'follower_id',
            'screen_name',
            'approach',
            'created_at',
            # Compound indexes for efficient queries
            ('deal_id', 'company_handle'),
            ('deal_id', 'overall_score'),
            ('company_handle', 'overall_score')
        ]
    }

    def __str__(self):
        return f"HighValueFollower: @{self.screen_name} - Score: {self.overall_score} - {self.company_name} (Deal: {self.deal_id})"


class RedditPost(Document):
    """Model to store individual Reddit posts with unique constraints per deal"""

    # Deal and Reddit identification
    deal_id = StringField(max_length=50, required=True)
    reddit_id = StringField(max_length=50, required=True)
    search_query = StringField(max_length=1000, required=True)

    # Complete Reddit post data
    # Stores the complete Reddit post object
    post = DynamicField(required=True)

    # Processing metadata
    # e.g., "Product A vs Product B"
    competition = StringField(max_length=500, required=False)
    approach = StringField(max_length=20, default="REDDIT_SCRAPER")

    # Timestamps
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'reddit_posts',
        'ordering': ['-created_at'],
        'indexes': [
            'deal_id',
            'reddit_id',
            'search_query',
            'competition',
            'created_at',
            # Compound unique index to ensure unique posts per deal
            ('deal_id', 'reddit_id')
        ]
    }

    def __str__(self):
        post_title = ""
        if isinstance(self.post, dict) and 'title' in self.post:
            post_title = self.post['title'][:50]
        return f"RedditPost: {post_title}... (Deal: {self.deal_id}, Reddit ID: {self.reddit_id})"

    @classmethod
    def save_unique_post(cls, deal_id: str, reddit_id: str, search_query: str,
                         post_data: dict, competition: str = None, approach: str = "REDDIT_SCRAPER"):
        """
        Save a Reddit post only if it doesn't already exist for this deal.

        Args:
            deal_id (str): Deal ID
            reddit_id (str): Reddit post ID
            search_query (str): Search query used to find this post
            post_data (dict): Complete Reddit post data
            competition (str): Competition pair (optional)
            approach (str): Processing approach (default: "REDDIT_SCRAPER")

        Returns:
            tuple: (saved_post, is_new) - RedditPost object and boolean indicating if it was newly created
        """
        # Check if post already exists for this deal
        existing_post = cls.objects(
            deal_id=deal_id, reddit_id=reddit_id).first()

        if existing_post:
            # Post already exists, return existing post
            return existing_post, False

        # Create new post
        new_post = cls(
            deal_id=deal_id,
            reddit_id=reddit_id,
            search_query=search_query,
            post=post_data,
            competition=competition,
            approach=approach
        )
        new_post.save()

        return new_post, True
