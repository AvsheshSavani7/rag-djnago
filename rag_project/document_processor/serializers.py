from rest_framework import serializers
from .models import ProcessingJob, SearchQuery, Tweet, CompanyProducts, CompetitiveAnalysis


class ProcessingJobSerializer(serializers.Serializer):
    id = serializers.CharField(read_only=True)
    cik = serializers.CharField(required=False, allow_null=True)
    acquire_name = serializers.CharField(required=False, allow_null=True)
    target_name = serializers.CharField(required=False, allow_null=True)
    announce_date = serializers.DateTimeField(
        format="%Y-%m-%d", required=False, allow_null=True)
    embedding_status = serializers.CharField()
    summary_status = serializers.CharField()
    file_url = serializers.URLField()
    parsed_json_url = serializers.URLField(required=False, allow_null=True)
    flattened_json_url = serializers.URLField(required=False, allow_null=True)
    error_message = serializers.CharField(required=False, allow_null=True)
    createdAt = serializers.DateTimeField(format="%Y-%m-%d %H:%M:%S")
    updatedAt = serializers.DateTimeField(format="%Y-%m-%d %H:%M:%S")
    schema_results = serializers.JSONField(required=False, allow_null=True)
    schema_processing_completed = serializers.BooleanField()
    schema_processing_timestamp = serializers.DateTimeField(
        required=False, allow_null=True)
    RF1_approach_done = serializers.BooleanField()
    RF2_approach_done = serializers.BooleanField()
    RF3_approach_done = serializers.BooleanField()
    twitter_details = serializers.JSONField(required=False, allow_null=True)
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
    created_at = serializers.DateTimeField(
        format="%Y-%m-%d %H:%M:%S", read_only=True)


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


class FileProcessRequestSerializer(serializers.Serializer):
    """Serializer for file processing request"""
    file_url = serializers.URLField(
        required=True, help_text="URL to the JSON file to process")
    cik = serializers.CharField(required=False, max_length=20)
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
