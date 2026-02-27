from mongoengine import (
    Document,
    StringField,
    DateTimeField,
)
from bson import ObjectId
from datetime import datetime


def generate_object_id():
    return str(ObjectId())


class Thread(Document):
    _id = StringField(primary_key=True, default=generate_object_id, max_length=24)
    user_id = StringField(required=True, max_length=100)
    openai_thread_id = StringField(required=True, max_length=100)
    name = StringField(required=True, max_length=255)
    created_at = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'threads',
        'indexes': [
            'user_id'
        ],
        'ordering': ['-created_at']
    }

    def __str__(self):
        return f"Thread {self.name} ({self._id})"