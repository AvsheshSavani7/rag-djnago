from mongoengine import (
    Document,
    StringField,
    URLField,
    DateTimeField,
    BooleanField,
    IntField,
    DictField,
)
from datetime import datetime


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

    # Schema parsing results
    schema_results = DictField(null=True)
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

    def __str__(self):
        return f"Deal: {self.acquire_name}/{self.target_name} (ID: {str(self.id)})"

    def update_embedding_status(self, status, error_message=None):
        """Update the embedding status of the job"""
        self.embedding_status = status
        if error_message:
            self.error_message = error_message
        self.updatedAt = datetime.utcnow()
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
        if not self.schema_results:
            # If no existing data, just save the new results
            self.schema_results = new_results
        else:
            # Merge new results with existing data
            for section_name, section_data in new_results.items():
                # Replace entire section data if section exists, otherwise add it
                self.schema_results[section_name] = section_data

        # Update processing status
        self.schema_processing_completed = True
        self.schema_processing_timestamp = datetime.now()
        self.save()
        return self

