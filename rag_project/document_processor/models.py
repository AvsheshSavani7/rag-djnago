from mongoengine import (
    Document,
    StringField,
    URLField,
    DateTimeField,
    BooleanField,
    IntField,
    # DictField,
    DynamicField,
)
from datetime import datetime
import json


class ProcessingJob(Document):
    """Model to track document processing jobs"""

    EMBEDDING_STATUS_CHOICES = ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED')

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
