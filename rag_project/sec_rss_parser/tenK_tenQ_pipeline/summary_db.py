"""
MongoDB-backed summary DB for 10-K/10-Q pipeline (Option A).
Uses SECFilingSummary collection; one doc per filing with ten_k_ten_q payload.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..models import SECFilingSummary

from .sec_fetcher import detect_filing_metadata, parse_sec_document_url


# Default ten_k_ten_q payload for new records
def _default_ten_k_ten_q() -> dict:
    return {
        "processed": False,
        "processed_at": None,
        "s3_json_url": None,
        "s3_docx_url": None,
        "s3_comparison_json_url": None,
        "s3_redline_docx_url": None,
        "s3_client_report_docx_url": None,
        "s3_change_report_docx_url": None,
        "s3_exec_summary_docx_url": None,
        "label": None,
        "period_date": None,
        "filing_type": None,
    }


def _doc_to_record(doc: SECFilingSummary) -> dict:
    """Convert MongoEngine doc to dict shape expected by orchestrator (id, ten_k_ten_q, etc.)."""
    if doc is None:
        return None
    tq = doc.ten_k_ten_q or {}
    return {
        "_id": str(doc.id),
        "id": str(doc.id),
        "sec_document_url": doc.sec_document_url,
        "accession_number": doc.accession_number,
        "cik_number": doc.cik_number,
        "deal_id": doc.deal_id,
        "form_type": doc.form_type,
        "filing_date": doc.filing_date,
        "processed": tq.get("processed"),
        "s3_json_url": tq.get("s3_json_url"),
        "s3_docx_url": tq.get("s3_docx_url"),
        # orchestrator may still look for this
        "local_json_path": tq.get("s3_json_url"),
        "local_docx_path": tq.get("s3_docx_url"),
        "local_comparison_json_path": tq.get("s3_comparison_json_url"),
        "local_redline_docx_path": tq.get("s3_redline_docx_url"),
        "local_client_report_docx_path": tq.get("s3_client_report_docx_url"),
        "local_change_report_docx_path": tq.get("s3_change_report_docx_url"),
        "local_exec_summary_docx_path": tq.get("s3_exec_summary_docx_url"),
        "label": tq.get("label"),
        "period_date": tq.get("period_date"),
        "filing_type": tq.get("filing_type"),
        "ten_k_ten_q": tq,
    }


_TEN_K_TEN_Q_FORM_TYPES = ["10-K", "10-Q", "10-K/A"]


def _find_doc_by_url_or_accession(url: str) -> Optional[SECFilingSummary]:
    """
    Find SECFilingSummary by exact sec_document_url, or by accession parsed from url.

    Dual-filer filings (e.g. parent + subsidiary) may use different CIK paths for the
    same accession; accession fallback keeps reads aligned with upsert_by_url.
    """
    if not url:
        return None
    doc = SECFilingSummary.objects(sec_document_url=url).first()
    if doc:
        return doc
    _, acc = parse_sec_document_url(url)
    if not acc:
        return None
    return SECFilingSummary.objects(
        accession_number=acc,
        form_type__in=_TEN_K_TEN_Q_FORM_TYPES,
    ).first()


class SummaryDB:
    """
    MongoDB-backed DB using SECFilingSummary.
    Records are keyed by sec_document_url or (accession_number, form_type).
    """

    def get_by_url(self, url: str) -> Optional[dict]:
        """Return the record matching sec_document_url or accession from url, or None."""
        return _doc_to_record(_find_doc_by_url_or_accession(url))

    def get_by_deal_id(self, deal_id: str) -> List[dict]:
        """Return all 10-K/10-Q summary records for the given deal_id."""
        docs = SECFilingSummary.objects(
            deal_id=deal_id,
            form_type__in=_TEN_K_TEN_Q_FORM_TYPES,
        ).all()
        return [_doc_to_record(d) for d in docs]

    def get_by_deal_id_and_cik(self, deal_id: str, cik_number: str) -> List[dict]:
        """Return all 10-K/10-Q summary records for the given deal_id and cik_number."""
        cik_stripped = (cik_number or "").lstrip("0") or "0"
        cik_padded = cik_stripped.zfill(10)
        docs = SECFilingSummary.objects(
            # deal_id=deal_id,
            cik_number__in=[cik_stripped, cik_padded],
            form_type__in=_TEN_K_TEN_Q_FORM_TYPES,
        ).all()
        return [_doc_to_record(d) for d in docs]

    def upsert_by_url(self, url: str, fields: dict) -> dict:
        """
        If a record with sec_document_url == url exists, update it with fields (merged into ten_k_ten_q).
        Otherwise, if a record with the same accession_number exists (e.g. index
        URL vs document URL for the same filing), reuse that record.
        Otherwise create a new SECFilingSummary with ten_k_ten_q stub.
        Returns the record as dict.
        """
        doc = _find_doc_by_url_or_accession(url)

        if doc:
            tq = doc.ten_k_ten_q or _default_ten_k_ten_q()
            for k, v in fields.items():
                if k in (
                    "period_date", "filing_type", "label",
                    "processed", "processed_at",
                    "s3_json_url", "s3_docx_url",
                    "s3_comparison_json_url", "s3_redline_docx_url",
                    "s3_client_report_docx_url", "s3_change_report_docx_url",
                    "s3_exec_summary_docx_url",
                ):
                    tq[k] = v
                elif k == "deal_id":
                    doc.deal_id = v
            doc.ten_k_ten_q = tq
            doc.save()
            return _doc_to_record(doc)

        # Create new
        cik_number, accession_number = parse_sec_document_url(url)
        period_date, filing_type = detect_filing_metadata(url)
        form_type = filing_type if filing_type in (
            "10-K", "10-Q", "10-K/A") else "10-Q"  # fallback
        if not accession_number:
            accession_number = url.strip(
                "/").split("/")[-2] if "/" in url else None
        if not cik_number:
            import re
            m = re.search(r"/edgar/data/(\d+)/", url)
            cik_number = m.group(1).zfill(10) if m else None

        tq = _default_ten_k_ten_q()
        tq["period_date"] = period_date
        tq["filing_type"] = filing_type
        if fields.get("label"):
            tq["label"] = fields["label"]

        new_doc = SECFilingSummary(
            accession_number=accession_number,
            cik_number=cik_number,
            sec_document_url=url,
            filing_date=None,
            deal_id=fields.get("deal_id"),
            form_type=form_type,
            ten_k_ten_q=tq,
            proxy=None,
            eight_k=None,
            other_filings=None,
        )
        new_doc.save()
        return _doc_to_record(new_doc)

    def update(self, record_id: str, fields: dict) -> None:
        """
        Update the SECFilingSummary with id == record_id.
        Merges ten_k_ten_q fields (s3_*_url, processed, processed_at, label, period_date, filing_type).
        """
        doc = SECFilingSummary.objects(pk=record_id).first()
        if not doc:
            raise KeyError(f"No SECFilingSummary with id={record_id!r}")

        tq = doc.ten_k_ten_q or _default_ten_k_ten_q()
        for k, v in fields.items():
            if k in (
                "period_date", "filing_type", "label",
                "processed", "processed_at",
                "s3_json_url", "s3_docx_url",
                "s3_comparison_json_url", "s3_redline_docx_url",
                "s3_client_report_docx_url", "s3_change_report_docx_url",
                "s3_exec_summary_docx_url",
            ):
                tq[k] = v
            elif k == "deal_id":
                doc.deal_id = v
        doc.ten_k_ten_q = tq
        doc.save()
