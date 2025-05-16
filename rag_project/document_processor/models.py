from django.db import models
import uuid
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId
from django.conf import settings

# Connect to MongoDB directly using pymongo
mongo_client = MongoClient(settings.MONGODB_URI)
mongo_db = mongo_client[settings.MONGODB_NAME]
deals_collection = mongo_db['deals']


class ProcessingJob:
    """Model to track document processing jobs without Django ORM"""
    EMBEDDING_STATUS_CHOICES = (
        ('PENDING', 'Pending'),
        ('PROCESSING', 'Processing'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    )

    def __init__(self, **kwargs):
        # MongoDB document ID
        self._id = kwargs.get('_id')

        # Deal information
        self.cik = kwargs.get('cik')
        self.acquire_name = kwargs.get('acquire_name')
        self.target_name = kwargs.get('target_name')
        self.announce_date = kwargs.get('announce_date')
        self.embedding_status = kwargs.get('embedding_status', 'PENDING')

        # Processing fields
        self.file_url = kwargs.get('file_url', '')
        self.parsed_json_url = kwargs.get('parsed_json_url')
        self.flattened_json_url = kwargs.get('flattened_json_url')

        # Schema fields
        self.schema_results = kwargs.get('schema_results')
        self.schema_processing_completed = kwargs.get(
            'schema_processing_completed', False)
        self.schema_processing_timestamp = kwargs.get(
            'schema_processing_timestamp')

        # Error information
        self.error_message = kwargs.get('error_message')

        # Timestamps
        self.createdAt = kwargs.get('createdAt', datetime.now())
        self.updatedAt = kwargs.get('updatedAt', datetime.now())

        # Version
        self.v_version = kwargs.get('__v', 0)

    def __str__(self):
        return f"Deal: {self.acquire_name}/{self.target_name} (ID: {self._id})"

    def save(self):
        """Save document to MongoDB"""
        mongo_data = {
            'cik': self.cik,
            'acquire_name': self.acquire_name,
            'target_name': self.target_name,
            'announce_date': self.announce_date,
            'embedding_status': self.embedding_status,
            'file_url': self.file_url,
            'parsed_json_url': self.parsed_json_url,
            'flattened_json_url': self.flattened_json_url,
            'schema_results': self.schema_results,
            'schema_processing_completed': self.schema_processing_completed,
            'schema_processing_timestamp': self.schema_processing_timestamp,
            'error_message': self.error_message,
            'createdAt': self.createdAt,
            'updatedAt': self.updatedAt,
            '__v': self.v_version,
        }

        # Update updatedAt timestamp
        self.updatedAt = datetime.now()
        mongo_data['updatedAt'] = self.updatedAt

        # If we already have an ObjectId, update the record
        if self._id:
            deals_collection.update_one(
                {'_id': ObjectId(self._id)},
                {'$set': mongo_data}
            )
        else:
            # Insert new record and store the ObjectId
            result = deals_collection.insert_one(mongo_data)
            self._id = result.inserted_id

        return self

    def update_embedding_status(self, status, error_message=None):
        """Update the embedding status of the job"""
        self.embedding_status = status
        if error_message:
            self.error_message = error_message
        self.save()
        return self

    def save_json_to_db(self, results, error_message=None):
        """Update schema results"""
        self.schema_results = results
        self.schema_processing_completed = True
        self.schema_processing_timestamp = datetime.now()
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

    @classmethod
    def find_by_id(cls, object_id):
        """Find a document by ID"""
        document = deals_collection.find_one({'_id': ObjectId(object_id)})
        if not document:
            return None

        # Convert MongoDB _id to string for easier handling
        document['_id'] = str(document['_id'])
        return cls(**document)

    @classmethod
    def find_all(cls):
        """Find all documents"""
        documents = deals_collection.find()
        result = []
        for doc in documents:
            # Convert MongoDB _id to string for easier handling
            doc['_id'] = str(doc['_id'])
            result.append(cls(**doc))
        return result

    @classmethod
    def find_by_query(cls, query=None, sort=None, limit=None):
        """Find documents by query"""
        query = query or {}
        cursor = deals_collection.find(query)

        if sort:
            cursor = cursor.sort(sort)

        if limit:
            cursor = cursor.limit(limit)

        result = []
        for doc in cursor:
            # Convert MongoDB _id to string for easier handling
            doc['_id'] = str(doc['_id'])
            result.append(cls(**doc))
        return result
