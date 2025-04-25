from django.db import models
import uuid
from djongo.models import ObjectIdField
from datetime import datetime


class ProcessingJob(models.Model):
    """Model to track document processing jobs"""
    EMBEDDING_STATUS_CHOICES = (
        ('PENDING', 'Pending'),
        ('PROCESSING', 'Processing'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    )

    _id = ObjectIdField(primary_key=True)

    # Deal information
    cik = models.CharField(max_length=20, blank=True, null=True)
    acquire_name = models.CharField(max_length=255, blank=True, null=True)
    target_name = models.CharField(max_length=255, blank=True, null=True)
    announce_date = models.DateTimeField(blank=True, null=True)
    embedding_status = models.CharField(
        max_length=20, choices=EMBEDDING_STATUS_CHOICES, default='PENDING')

    # Processing fields
    file_url = models.URLField(
        max_length=1000, help_text="URL to the JSON file to process")
    parsed_json_url = models.URLField(
        max_length=1000, blank=True, null=True, help_text="URL to the parsed JSON file")
    flattened_json_url = models.URLField(
        max_length=1000, blank=True, null=True, help_text="URL to the flattened JSON file")

    # Error information
    error_message = models.TextField(null=True, blank=True)

    # Timestamps
    createdAt = models.DateTimeField(auto_now_add=True, db_column="createdAt")
    updatedAt = models.DateTimeField(auto_now=True, db_column="updatedAt")

    # MongoDB-specific fields
    v_version = models.IntegerField(default=0, db_column="__v")

    def __str__(self):
        return f"Deal: {self.acquire_name}/{self.target_name} (ID: {self._id})"

    def update_embedding_status(self, status, error_message=None):
        """Update the embedding status of the job"""
        self.embedding_status = status
        if error_message:
            self.error_message = error_message
        self.save()
        return self

    class Meta:
        db_table = "deals"
        ordering = ['-createdAt']
