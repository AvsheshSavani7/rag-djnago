from rest_framework import serializers
from .models import ProcessingJob


class ProcessingJobSerializer(serializers.ModelSerializer):
    """Serializer for processing jobs"""
    announce_date = serializers.DateTimeField(
        format="%Y-%m-%d", required=False)
    createdAt = serializers.DateTimeField(
        format="%Y-%m-%d %H:%M:%S", read_only=True)
    updatedAt = serializers.DateTimeField(
        format="%Y-%m-%d %H:%M:%S", read_only=True)
    id = serializers.CharField(source='_id', read_only=True)

    class Meta:
        model = ProcessingJob
        fields = [
            'id', 'cik', 'acquire_name', 'target_name', 'announce_date',
            'embedding_status', 'file_url', 'parsed_json_url',
            'flattened_json_url', 'error_message', 'createdAt', 'updatedAt'
        ]
        read_only_fields = [
            'id', 'error_message', 'createdAt', 'updatedAt'
        ]


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
