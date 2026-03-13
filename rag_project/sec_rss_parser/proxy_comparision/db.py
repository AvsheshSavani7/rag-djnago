"""
db.py — MongoDB-backed access to sec_filing_summary for proxy comparison.
Uses pymongo only (no Django/MongoEngine). Collection: sec_filing_summary.
Env: MONGODB_CONNECTION_STRING, MONGODB_NAME (optional; default Deal_DB).
Use the same database where sec_filing_summary records live.
"""

import os
from datetime import datetime, timezone
from typing import Any, Optional

# Use raw pymongo so this package runs without Django
try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None  # type: ignore

try:
    from bson import ObjectId
except ImportError:
    ObjectId = None  # type: ignore

COLLECTION_NAME = "sec_filing_summary"


def _id_filter(record_id: str) -> dict:
    """Build query filter for _id. SECFilingSummary uses string _id; support ObjectId for 24-char hex."""
    if not record_id:
        return {"_id": None}
    # String _id (UUID, etc.) — normal case for sec_filing_summary
    if len(record_id) != 24:
        return {"_id": record_id}
    # 24-char id: match both string and ObjectId so we find the doc either way
    if ObjectId:
        try:
            return {"_id": {"$in": [record_id, ObjectId(record_id)]}}
        except Exception:
            pass
    return {"_id": record_id}


def _normalize_record(record: dict) -> dict:
    """Ensure proxy.comparison.cache and proxy.comparison.result exist; _id as string."""
    if record is None:
        return None
    r = dict(record)
    if "_id" in r and not isinstance(r["_id"], str):
        r["_id"] = str(r["_id"])
    if "proxy" not in r:
        r["proxy"] = {}
    if "comparison" not in r["proxy"]:
        r["proxy"]["comparison"] = {}
    if "cache" not in r["proxy"]["comparison"]:
        r["proxy"]["comparison"]["cache"] = {"status": "pending"}
    if "result" not in r["proxy"]["comparison"]:
        r["proxy"]["comparison"]["result"] = {"status": "pending"}
    return r


class ProxyDB:
    """MongoDB-backed database for proxy filing records (sec_filing_summary collection)."""

    def __init__(self):
        if MongoClient is None:
            raise ValueError("pymongo is required. Install with: pip install pymongo")
        connection_string = os.environ.get("MONGODB_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("MONGODB_CONNECTION_STRING is not set in environment")
        db_name = os.environ.get("MONGODB_NAME", "Deal_DB")
        self._client = MongoClient(connection_string)
        self._db = self._client[db_name]
        self._coll = self._db[COLLECTION_NAME]

    def get_by_id(self, record_id: str) -> Optional[dict]:
        """Return the sec_filing_summary document with the given _id, or None.
        _id is normalized to string. proxy.comparison.cache/result are ensured.
        """
        doc = self._coll.find_one(_id_filter(record_id))
        return _normalize_record(doc) if doc else None

    @property
    def db_and_collection(self) -> str:
        """e.g. 'Deal_DB_New.sec_filing_summary' — so you can verify you're querying the same place."""
        return f"{self._db.name}.{self._coll.name}"

    def ensure_record_exists(self, record: dict) -> None:
        """Ensure a document with this record's _id exists in sec_filing_summary.
        If it does not exist, insert a minimal document (so set_cache_status/set_result_status later will update it).
        Uses $setOnInsert so existing documents are never overwritten.
        """
        if not record:
            return
        record_id = record.get("_id")
        if not record_id:
            return
        record_id = str(record_id)
        set_on_insert = {
            "_id": record_id,
            "sec_document_url": record.get("sec_document_url") or "",
            "deal_id": record.get("deal_id"),
            "form_type": record.get("form_type") or "PROXY",
            "accession_number": record.get("accession_number"),
            "cik_number": record.get("cik_number"),
        }
        # Remove None values so we don't store nulls for optional fields
        set_on_insert = {k: v for k, v in set_on_insert.items() if v is not None}
        # Use exact _id for upsert so inserted doc has _id = record_id (string)
        self._coll.update_one(
            {"_id": record_id},
            {"$setOnInsert": set_on_insert},
            upsert=True,
        )

    def set_cache_status(self, record_id: str, **fields: Any) -> None:
        """Update proxy.comparison.cache and updated_at for the document."""
        update = {"$set": {"updated_at": datetime.now(timezone.utc)}}
        for k, v in fields.items():
            update["$set"][f"proxy.comparison.cache.{k}"] = v
        r = self._coll.update_one(_id_filter(record_id), update)
        if r.matched_count == 0:
            import warnings
            warnings.warn(
                f"set_cache_status: no document matched _id={record_id!r} in {self.db_and_collection}. "
                "Check that the record exists in this DB."
            )

    def set_result_status(self, record_id: str, **fields: Any) -> None:
        """Update proxy.comparison.result and updated_at for the document."""
        update = {"$set": {"updated_at": datetime.now(timezone.utc)}}
        for k, v in fields.items():
            update["$set"][f"proxy.comparison.result.{k}"] = v
        r = self._coll.update_one(_id_filter(record_id), update)
        if r.matched_count == 0:
            import warnings
            warnings.warn(
                f"set_result_status: no document matched _id={record_id!r} in {self.db_and_collection}. "
                "Check that the record exists in this DB."
            )
