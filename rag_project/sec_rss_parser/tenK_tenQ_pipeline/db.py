"""FilingDB: JSON-backed mock database for SEC filing records."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class FilingDB:
    """
    JSON-file backed mock database that mirrors the MongoDB schema for SEC filings.

    Each record schema:
    {
        "_id": "uuid-string",
        "sec_document_url": "https://www.sec.gov/...",
        "deal_id": "...",
        "cik_number": null,
        "accession_number": null,
        "filing_date": null,
        "period_date": null,
        "filing_type": null,
        "label": null,
        "processed": false,
        "processed_at": null,
        "local_json_path": null,
        "local_docx_path": null,
        "local_comparison_json_path": null,
        "local_redline_docx_path": null,
        "local_client_report_docx_path": null,
        "local_exec_summary_docx_path": null,
        "created_at": "ISO datetime",
        "updated_at": "ISO datetime"
    }
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._records: list[dict] = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_by_deal_id(self, deal_id: str) -> list[dict]:
        """Return all records for a given deal_id."""
        return [r for r in self._records if r.get("deal_id") == deal_id]

    def get_by_url(self, url: str) -> Optional[dict]:
        """Return the record matching sec_document_url, or None."""
        for r in self._records:
            if r.get("sec_document_url") == url:
                return r
        return None

    def insert(self, record: dict) -> dict:
        """Insert a new record (adds _id, created_at, updated_at if missing)."""
        now = _now_iso()
        record.setdefault("_id", str(uuid.uuid4()))
        record.setdefault("created_at", now)
        record.setdefault("updated_at", now)
        self._records.append(record)
        self._save()
        return record

    def update(self, record_id: str, fields: dict) -> None:
        """Update fields on the record with the given _id."""
        for r in self._records:
            if r.get("_id") == record_id:
                r.update(fields)
                r["updated_at"] = _now_iso()
                self._save()
                return
        raise KeyError(f"No record with _id={record_id!r}")

    def upsert_by_url(self, url: str, fields: dict) -> dict:
        """
        If a record with sec_document_url == url exists, update it with fields.
        Otherwise insert a new record with those fields + url.
        Returns the (possibly newly created) record.
        """
        existing = self.get_by_url(url)
        if existing:
            self.update(existing["_id"], fields)
            return self.get_by_url(url)  # re-fetch after update
        else:
            new_record = {
                "sec_document_url": url,
                "processed": False,
                "processed_at": None,
                "local_json_path": None,
                "local_docx_path": None,
                "local_comparison_json_path": None,
                "local_redline_docx_path": None,
                "local_client_report_docx_path": None,
                "local_exec_summary_docx_path": None,
            }
            new_record.update(fields)
            return self.insert(new_record)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> list[dict]:
        if self.db_path.exists():
            try:
                return json.loads(self.db_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save(self) -> None:
        """Write atomically via a temp file."""
        tmp = self.db_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._records, indent=2), encoding="utf-8")
        tmp.replace(self.db_path)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
