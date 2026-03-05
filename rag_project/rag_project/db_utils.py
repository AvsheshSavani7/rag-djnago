"""
Dual MongoDB connection helpers.

The project uses two MongoDB databases:
- default (alias "default"): existing DB — MONGODB_CONNECTION_STRING / MONGODB_NAME
- new_db (alias "new_db"): new DB for new flows — MONGODB_CONNECTION_STRING_NEW / MONGODB_NAME_NEW

Usage:

1) MongoEngine models
   - Existing models (no meta) use the default DB. Leave them unchanged.
   - New models that should live in the new DB add to the class:
       meta = {"db_alias": "new_db"}
   - Query the old DB from new code:
       from document_processor.models import ProcessingJob
       ProcessingJob.objects(...)   # uses default DB
       # or explicitly:
       ProcessingJob.objects.using("default").filter(...)
   - Query the new DB:
       MyNewModel.objects.using("new_db").filter(...)

2) Raw PyMongo (e.g. scripts, backups)
   - get_default_db() -> (db, client) for the old DB
   - get_new_db() -> (db, client) for the new DB (raises if MONGODB_CONNECTION_STRING_NEW not set)
"""
import os
from pymongo import MongoClient


def get_default_db():
    """Return (db, client) for the default/old MongoDB. Uses MONGODB_CONNECTION_STRING."""
    connection_string = os.getenv("MONGODB_CONNECTION_STRING")
    db_name = os.getenv("MONGODB_NAME", "Deal_DB")
    if not connection_string:
        raise ValueError("MONGODB_CONNECTION_STRING not found in environment variables")
    client = MongoClient(connection_string)
    return client[db_name], client


def get_new_db():
    """Return (db, client) for the new MongoDB. Uses MONGODB_CONNECTION_STRING_NEW."""
    connection_string = os.getenv("MONGODB_CONNECTION_STRING_NEW")
    db_name = os.getenv("MONGODB_NAME_NEW", "Deal_DB_New")
    if not connection_string:
        raise ValueError(
            "MONGODB_CONNECTION_STRING_NEW not found in environment variables"
        )
    client = MongoClient(connection_string)
    return client[db_name], client
