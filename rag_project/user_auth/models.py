from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.utils import timezone
from bson import ObjectId
import uuid
from datetime import datetime
from django.conf import settings
from pymongo import MongoClient
import hashlib

# Connect to MongoDB directly using pymongo
mongo_client = MongoClient(settings.MONGODB_URI)
mongo_db = mongo_client[settings.MONGODB_NAME]
users_collection = mongo_db['users']


def generate_object_id():
    return str(ObjectId())


class UserManager:
    """User manager for pymongo-based User model"""

    @staticmethod
    def create_user(email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')

        # Normalize email (lowercase domain part)
        email_parts = email.split('@')
        if len(email_parts) == 2:
            email = f"{email_parts[0]}@{email_parts[1].lower()}"

        # Check if user already exists
        existing_user = users_collection.find_one({'email': email})
        if existing_user:
            raise ValueError('User with this email already exists')

        # Hash the password
        hashed_password = hashlib.sha256(
            password.encode()).hexdigest() if password else None

        # Create user data
        user_data = {
            '_id': generate_object_id(),
            'email': email,
            'password': hashed_password,
            'role': extra_fields.get('role', 'admin'),
            'createdAt': datetime.now(),
            'updatedAt': datetime.now(),
        }

        # Insert into MongoDB
        users_collection.insert_one(user_data)

        # Return User instance
        return User(**user_data)


class User:
    """User model using pymongo directly instead of Django ORM"""

    # Required for Django authentication
    is_anonymous = False
    is_authenticated = True

    # Manager for class methods
    objects = UserManager()

    # Fields required by Django auth
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    def __init__(self, **kwargs):
        self._id = kwargs.get('_id')
        self.email = kwargs.get('email')
        self.password = kwargs.get('password')
        self.role = kwargs.get('role', 'admin')
        self.createdAt = kwargs.get('createdAt', datetime.now())
        self.updatedAt = kwargs.get('updatedAt', datetime.now())
        # Required for Django auth
        self.is_active = True
        self.is_staff = kwargs.get('role') == 'admin'
        self.is_superuser = kwargs.get('role') == 'admin'
        # Additional attributes needed for JWT
        self.is_anonymous = False
        self.is_authenticated = True

    def __str__(self):
        return self.email

    def save(self):
        """Save user to MongoDB"""
        user_data = {
            'email': self.email,
            'password': self.password,
            'role': self.role,
            'updatedAt': datetime.now(),
        }

        # If we already have an ObjectId, update the record
        if self._id:
            users_collection.update_one(
                {'_id': ObjectId(self._id) if isinstance(
                    self._id, str) else self._id},
                {'$set': user_data}
            )
        else:
            # Insert new record and store the ObjectId
            user_data['_id'] = generate_object_id()
            user_data['createdAt'] = datetime.now()
            result = users_collection.insert_one(user_data)
            self._id = result.inserted_id

        return self

    def set_password(self, raw_password):
        """Set the password for the user"""
        self.password = hashlib.sha256(
            raw_password.encode()).hexdigest() if raw_password else None
        self.save()

    def check_password(self, raw_password):
        """Check if the provided password matches the stored hash"""
        return self.password == hashlib.sha256(raw_password.encode()).hexdigest()

    @classmethod
    def find_by_id(cls, user_id):
        """Find a user by ID"""
        document = users_collection.find_one({'_id': ObjectId(user_id)})
        if not document:
            return None

        # Convert MongoDB _id to string for easier handling
        document['_id'] = str(document['_id'])
        return cls(**document)

    @classmethod
    def find_by_email(cls, email):
        """Find a user by email"""
        document = users_collection.find_one({'email': email})
        if not document:
            return None

        # Convert MongoDB _id to string for easier handling
        document['_id'] = str(document['_id'])
        return cls(**document)

    @classmethod
    def find_all(cls):
        """Find all users"""
        documents = users_collection.find()
        result = []
        for doc in documents:
            # Convert MongoDB _id to string for easier handling
            doc['_id'] = str(doc['_id'])
            result.append(cls(**doc))
        return result

    # Required for Django auth
    def get_username(self):
        return self.email

    # Methods required for JWT token generation
    @property
    def id(self):
        """Return _id as id for JWT token"""
        return self._id

    def get_all_permissions(self, obj=None):
        """Return an empty set of permissions"""
        return set()

    def has_perm(self, perm, obj=None):
        """Check if user has specific permission"""
        return True if self.role == 'admin' else False

    def has_perms(self, perm_list, obj=None):
        """Check if user has multiple permissions"""
        return all(self.has_perm(perm, obj) for perm in perm_list)

    def has_module_perms(self, app_label):
        """Check if user has permissions for a module"""
        return True if self.role == 'admin' else False
