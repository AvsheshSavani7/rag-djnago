from rest_framework import serializers
from .models import ProxyDocument


class ProxyDocumentSerializer(serializers.Serializer):
    """
    Serializer for Proxy documents.
    """

    id = serializers.CharField(read_only=True)

    # Basic document information
    cik_number = serializers.CharField(read_only=True)
    company_name = serializers.CharField(read_only=True)
    sec_filling_id = serializers.CharField(read_only=True)
    filing_date = serializers.CharField(read_only=True)
    form_type = serializers.CharField(read_only=True)
    proxy_sec_url = serializers.URLField(read_only=True)

    # Deal information
    deal_id = serializers.CharField(read_only=True)

    # Processing information
    proxy_parsing_status = serializers.CharField(read_only=True)

    # Processing results
    empty_percentage = serializers.FloatField(read_only=True)

    # S3 URLs for cloud storage
    s3_urls = serializers.DictField(read_only=True)

    # Processing state
    processing_state = serializers.DictField(read_only=True)

    # Timestamps
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)
    completed_at = serializers.DateTimeField(read_only=True)

    # Error information
    error_message = serializers.CharField(read_only=True)

    # Processing statistics
    total_sections = serializers.FloatField(read_only=True)
    empty_sections = serializers.FloatField(read_only=True)
    iteration_count = serializers.FloatField(read_only=True)

    # Agent response
    agent_response = serializers.CharField(read_only=True)

    # Pinecone processing information
    pinecone_processing_status = serializers.CharField(read_only=True)
    pinecone_processed_at = serializers.DateTimeField(read_only=True)
    pinecone_error_message = serializers.CharField(read_only=True)

    # Summary document information
    summary_docx_url = serializers.URLField(read_only=True, allow_null=True)
    summary_generation_status = serializers.CharField(read_only=True)
    summary_generated_at = serializers.DateTimeField(read_only=True)
    summary_error_message = serializers.CharField(read_only=True)

    def create(self, validated_data):
        """
        Create a new Proxy document.
        """
        document = ProxyDocument(**validated_data)
        document.save()
        return document


class ProxyProcessingRequestSerializer(serializers.Serializer):
    """
    Serializer for Proxy processing requests.
    """

    # Required fields from user input
    cik_number = serializers.CharField(
        required=True, max_length=20, help_text="CIK number of the company")
    company_name = serializers.CharField(
        required=True, max_length=200, help_text="Name of the company")
    sec_filling_id = serializers.CharField(
        required=True, max_length=50, help_text="SEC filing ID")
    filing_date = serializers.CharField(
        required=True, max_length=20, help_text="Filing date")
    form_type = serializers.CharField(
        required=True, max_length=20, help_text="Form type (e.g., DEF 14A)")
    proxy_sec_url = serializers.URLField(
        required=True, help_text="URL of the SEC proxy document")

    # Optional fields
    deal_id = serializers.CharField(
        required=False, max_length=50, allow_blank=True, help_text="Deal ID if CIK matches with deal table")

    def validate_proxy_sec_url(self, value):
        """
        Validate that the URL is a valid SEC document URL.
        """
        if not value.startswith('https://www.sec.gov/'):
            raise serializers.ValidationError(
                "URL must be a valid SEC document URL")
        return value

    def validate_cik_number(self, value):
        """
        Validate CIK number format.
        """
        if not value.isdigit() or len(value) != 10:
            raise serializers.ValidationError(
                "CIK number must be a 10-digit number")
        return value
