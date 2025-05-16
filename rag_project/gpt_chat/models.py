from django.db import models
from bson import ObjectId
from datetime import datetime
from django.conf import settings
from pymongo import MongoClient, DESCENDING

# Connect to MongoDB directly using pymongo
mongo_client = MongoClient(settings.MONGODB_URI)
mongo_db = mongo_client[settings.MONGODB_NAME]
threads_collection = mongo_db['threads']

# Create an index on user_id field
threads_collection.create_index([("user_id", DESCENDING)])


def generate_object_id():
    return str(ObjectId())


class Thread:
    """Thread model using pymongo directly instead of Django ORM"""

    def __init__(self, **kwargs):
        self._id = kwargs.get('_id')
        self.user_id = kwargs.get('user_id')
        self.openai_thread_id = kwargs.get('openai_thread_id')
        self.name = kwargs.get('name')
        self.created_at = kwargs.get('created_at', datetime.now())

    def __str__(self):
        return f"Thread {self.name} ({self._id})"

    def save(self):
        """Save thread to MongoDB"""
        thread_data = {
            'user_id': self.user_id,
            'openai_thread_id': self.openai_thread_id,
            'name': self.name,
            'created_at': self.created_at
        }

        # If we already have an ObjectId, update the record
        if self._id:
            threads_collection.update_one(
                {'_id': ObjectId(self._id) if isinstance(
                    self._id, str) else self._id},
                {'$set': thread_data}
            )
        else:
            # Insert new record and store the ObjectId
            result = threads_collection.insert_one(thread_data)
            self._id = result.inserted_id

        return self

    @classmethod
    def find_by_id(cls, thread_id):
        """Find a thread by ID"""
        document = threads_collection.find_one({'_id': ObjectId(thread_id)})
        if not document:
            return None

        # Convert MongoDB _id to string for easier handling
        document['_id'] = str(document['_id'])
        return cls(**document)

    @classmethod
    def find_by_user_id(cls, user_id):
        """Find threads by user_id"""
        documents = threads_collection.find({'user_id': user_id})

        result = []
        for doc in documents:
            # Convert MongoDB _id to string for easier handling
            doc['_id'] = str(doc['_id'])
            result.append(cls(**doc))

        return result

    @classmethod
    def find_all(cls):
        """Find all threads"""
        documents = threads_collection.find()

        result = []
        for doc in documents:
            # Convert MongoDB _id to string for easier handling
            doc['_id'] = str(doc['_id'])
            result.append(cls(**doc))

        return result

    @classmethod
    def find_by_query(cls, query=None, sort=None, limit=None):
        """Find threads by query"""
        query = query or {}
        cursor = threads_collection.find(query)

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

    @classmethod
    def delete_by_id(cls, thread_id):
        """Delete a thread by ID"""
        threads_collection.delete_one({
            '_id': ObjectId(thread_id) if isinstance(thread_id, str) else thread_id
        })
        return True
