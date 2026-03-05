from rest_framework import serializers
from .models import ProcessingJob, SearchQuery, Tweet, CompanyProducts, CompetitiveAnalysis, RedditPost


class ProcessingJobSerializer(serializers.Serializer):
    id = serializers.CharField(read_only=True)
    cik = serializers.CharField(required=False, allow_null=True)
    acquirer_cik = serializers.CharField(required=False, allow_null=True)
    acquire_name = serializers.CharField(required=False, allow_null=True)
    target_name = serializers.CharField(required=False, allow_null=True)
    announce_date = serializers.DateTimeField(
        format="%Y-%m-%d", required=False, allow_null=True)
    embedding_status = serializers.CharField()
    summary_status = serializers.CharField()
    summary_using = serializers.CharField(required=False, allow_null=True)
    file_url = serializers.URLField()
    parsed_json_url = serializers.URLField(required=False, allow_null=True)
    flattened_json_url = serializers.URLField(required=False, allow_null=True)
    error_message = serializers.CharField(required=False, allow_null=True)
    createdAt = serializers.DateTimeField(format="%Y-%m-%d %H:%M:%S")
    updatedAt = serializers.DateTimeField(format="%Y-%m-%d %H:%M:%S")
    brazil = serializers.JSONField(required=False, allow_null=True)
    samr_public = serializers.JSONField(required=False, allow_null=True)
    samr_conditional = serializers.JSONField(required=False, allow_null=True)
    samr_unconditional = serializers.JSONField(required=False, allow_null=True)
    uk_cma_cases = serializers.JSONField(required=False, allow_null=True)
    german_scrap = serializers.JSONField(required=False, allow_null=True)
    ec_cases = serializers.JSONField(required=False, allow_null=True)
    accc_cases = serializers.JSONField(required=False, allow_null=True)

    ftc_early_termination = serializers.JSONField(
        required=False, allow_null=True)
    canada_competition_bureau_cases = serializers.JSONField(
        required=False, allow_null=True)
    nz_cases = serializers.JSONField(required=False, allow_null=True)
    fs_ec_cases = serializers.JSONField(required=False, allow_null=True)
    target_ticker = serializers.CharField(required=False, allow_null=True)
    acquirer_ticker = serializers.CharField(required=False, allow_null=True)
    deal_status = serializers.JSONField(required=False, allow_null=True)

    schema_results = serializers.JSONField(required=False, allow_null=True)
    schema_processing_completed = serializers.BooleanField()
    schema_processing_timestamp = serializers.DateTimeField(
        required=False, allow_null=True)
    RF1_approach_done = serializers.BooleanField()
    RF2_approach_done = serializers.BooleanField()
    RF3_approach_done = serializers.BooleanField()
    GUNSHOT_approach_done = serializers.BooleanField()
    twitter_details = serializers.JSONField(required=False, allow_null=True)

    parent_aliases = serializers.ListField(child=serializers.CharField())
    target_aliases = serializers.ListField(child=serializers.CharField())

    summary_docx_url = serializers.URLField(required=False, allow_null=True)
    sec_url = serializers.URLField(required=False, allow_null=True)
    sec_filing_id = serializers.CharField(required=False, allow_null=True)


class SearchQuerySerializer(serializers.Serializer):
    """Serializer for SearchQuery model"""
    id = serializers.CharField(read_only=True)
    search_query = serializers.CharField()
    deal_id = serializers.CharField()
    approach = serializers.CharField()
    combination = serializers.JSONField()
    total_tweets = serializers.IntegerField()
    created_at = serializers.DateTimeField(
        format="%Y-%m-%d %H:%M:%S", read_only=True)
    updated_at = serializers.DateTimeField(
        format="%Y-%m-%d %H:%M:%S", read_only=True)


class TweetSerializer(serializers.Serializer):
    """Serializer for Tweet model"""
    id = serializers.CharField(read_only=True)
    search_query_id = serializers.CharField()
    tweet = serializers.JSONField()
    tweet_created_at = serializers.DateTimeField(
        format="%Y-%m-%d %H:%M:%S", required=False, allow_null=True)
    created_at = serializers.DateTimeField(
        format="%Y-%m-%d %H:%M:%S", read_only=True)
    approach = serializers.CharField()
    search_query_info = serializers.JSONField()


class CompanyProductsSerializer(serializers.Serializer):
    """Serializer for CompanyProducts model"""
    id = serializers.CharField(read_only=True)
    deal_id = serializers.CharField()
    company = serializers.CharField()
    company_type = serializers.ChoiceField(choices=['target', 'acquire'])
    products = serializers.ListField(child=serializers.CharField())
    gpt_model_used = serializers.CharField()
    extraction_timestamp = serializers.DateTimeField(
        format="%Y-%m-%d %H:%M:%S", read_only=True)
    processing_status = serializers.ChoiceField(
        choices=['pending', 'completed', 'failed'])
    created_at = serializers.DateTimeField(
        format="%Y-%m-%d %H:%M:%S", read_only=True)
    updated_at = serializers.DateTimeField(
        format="%Y-%m-%d %H:%M:%S", read_only=True)


class CompetitiveAnalysisSerializer(serializers.Serializer):
    """Serializer for CompetitiveAnalysis model"""
    id = serializers.CharField(read_only=True)
    deal_id = serializers.CharField()
    target_company_products = serializers.CharField()  # Reference ID
    acquire_company_products = serializers.CharField()  # Reference ID
    competitive_pairs = serializers.JSONField()
    gpt_model_used = serializers.CharField()
    analysis_timestamp = serializers.DateTimeField(
        format="%Y-%m-%d %H:%M:%S", read_only=True)
    processing_status = serializers.ChoiceField(
        choices=['pending', 'completed', 'failed'])
    created_at = serializers.DateTimeField(
        format="%Y-%m-%d %H:%M:%S", read_only=True)
    updated_at = serializers.DateTimeField(
        format="%Y-%m-%d %H:%M:%S", read_only=True)


class HighValueFollowersSerializer(serializers.Serializer):
    """Serializer for HighValueFollowers model"""
    id = serializers.CharField(read_only=True)
    deal_id = serializers.CharField()
    company_name = serializers.CharField()
    company_handle = serializers.CharField()
    follower_id = serializers.CharField()
    name = serializers.CharField(required=False, allow_null=True)
    screen_name = serializers.CharField(required=False, allow_null=True)
    description = serializers.CharField(required=False, allow_null=True)
    location = serializers.CharField(required=False, allow_null=True)
    followers_count = serializers.IntegerField()
    statuses_count = serializers.IntegerField()
    protected = serializers.BooleanField()
    verified = serializers.BooleanField()
    created_at_twitter = serializers.DateTimeField(
        format="%Y-%m-%d %H:%M:%S", required=False, allow_null=True)
    overall_score = serializers.IntegerField()
    reason = serializers.CharField(required=False, allow_null=True)
    key_indicators = serializers.ListField(child=serializers.CharField())
    analysis_timestamp = serializers.DateTimeField(
        format="%Y-%m-%d %H:%M:%S", read_only=True)
    gpt_model_used = serializers.CharField()
    processing_status = serializers.CharField()
    approach = serializers.CharField()
    created_at = serializers.DateTimeField(
        format="%Y-%m-%d %H:%M:%S", read_only=True)
    updated_at = serializers.DateTimeField(
        format="%Y-%m-%d %H:%M:%S", read_only=True)


class FileProcessRequestSerializer(serializers.Serializer):
    """Serializer for file processing request"""
    file_url = serializers.URLField(
        required=True, help_text="URL to the JSON file to process")
    cik = serializers.CharField(required=False, max_length=20)
    acquirer_cik = serializers.CharField(required=False, max_length=20)
    acquire_name = serializers.CharField(required=False, max_length=255)
    target_name = serializers.CharField(required=False, max_length=255)
    announce_date = serializers.DateField(required=False)
    parsed_json_url = serializers.URLField(required=False)
    flattened_json_url = serializers.URLField(required=False)


class DocumentProcessRequestSerializer(serializers.Serializer):
    """Serializer for document processing request"""
    # Option 1: Process data from S3
    input_key = serializers.CharField(required=False,
                                      help_text="S3 key for input file")

    # Option 2: Process data directly
    input_data = serializers.JSONField(required=False,
                                       help_text="Document data to process")

    output_key = serializers.CharField(required=False,
                                       help_text="Custom S3 key for output file")

    def validate(self, data):
        """Validate that either input_key or input_data is provided"""
        if not data.get('input_key') and not data.get('input_data'):
            raise serializers.ValidationError(
                "Either input_key or input_data must be provided"
            )
        return data


class DocumentListRequestSerializer(serializers.Serializer):
    """Serializer for listing documents"""
    prefix = serializers.CharField(required=False, default="",
                                   help_text="S3 prefix to filter files")


class DocumentListResponseSerializer(serializers.Serializer):
    """Serializer for document list response"""
    files = serializers.ListField(
        child=serializers.CharField(),
        help_text="List of S3 keys"
    )


class RedditPostSerializer(serializers.Serializer):
    """Serializer for RedditPost model"""
    id = serializers.CharField(read_only=True)
    deal_id = serializers.CharField()
    reddit_id = serializers.CharField()
    search_query = serializers.CharField()
    post = serializers.JSONField()
    competition = serializers.CharField(required=False, allow_null=True)
    approach = serializers.CharField()
    created_at = serializers.DateTimeField(
        format="%Y-%m-%d %H:%M:%S", read_only=True)
    updated_at = serializers.DateTimeField(
        format="%Y-%m-%d %H:%M:%S", read_only=True)
