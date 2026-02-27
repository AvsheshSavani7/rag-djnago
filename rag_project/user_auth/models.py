from mongoengine import Document, StringField, DateTimeField
from datetime import datetime
from bson import ObjectId


def generate_object_id():
    return str(ObjectId())


class User(Document):
    _id = StringField(primary_key=True, default=generate_object_id)
    email = StringField(required=True, unique=True)
    password = StringField(required=True, max_length=128)
    role = StringField(default='admin', max_length=50)
    createdAt = DateTimeField(default=datetime.utcnow)
    updatedAt = DateTimeField(default=datetime.utcnow)
    last_login = DateTimeField(default=None)

    meta = {
        'collection': 'users',
        'indexes': ['email'],
    }

    def save(self, *args, **kwargs):
        self.updatedAt = datetime.utcnow()
        return super().save(*args, **kwargs)

    def __str__(self):
        return self.email
    
    @property
    def is_authenticated(self):
        return True

    @property
    def is_anonymous(self):
        return False
