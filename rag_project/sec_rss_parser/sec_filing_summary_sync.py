"""
Sync ProxyDocument state to SECFilingSummary (proxy node).

When proxy_processor updates a ProxyDocument, this module updates the unified
sec_filing_summary collection so the proxy node reflects the same state.
"""
import re
import logging
from datetime import datetime

from sec_rss_parser.models import SECFilingSummary

logger = logging.getLogger(__name__)


def _parse_filing_date(value):
    """Parse filing_date string to datetime (midnight UTC). Returns None if unparseable."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    # YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", value)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    # MM/DD/YY
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", value)
    if m:
        try:
            y = int(m.group(3))
            if y < 100:
                y += 2000 if y < 50 else 1900
            return datetime(y, int(m.group(1)), int(m.group(2)))
        except ValueError:
            pass
    return None


def _proxy_doc_to_proxy_payload(proxy_doc):
    """Build the proxy dict for SECFilingSummary from a ProxyDocument."""
    s3_urls = getattr(proxy_doc, "s3_urls", None) or {}
    return {
        "proxy_parsing_status": getattr(proxy_doc, "proxy_parsing_status", None) or "pending",
        "empty_percentage": getattr(proxy_doc, "empty_percentage", None),
        "processing_state": getattr(proxy_doc, "processing_state", None) or {},
        "s3_urls": {
            "pdf_url": s3_urls.get("pdf_url"),
            "toc_pdf_url": s3_urls.get("toc_pdf_url"),
            "toc_json_url": s3_urls.get("toc_json_url"),
            "sections_json_url": s3_urls.get("sections_json_url"),
        },
        "pinecone_processing_status": getattr(proxy_doc, "pinecone_processing_status", None),
        "pinecone_processed_at": getattr(proxy_doc, "pinecone_processed_at", None),
        "pinecone_error_message": getattr(proxy_doc, "pinecone_error_message", None),
        "summary_generation_status": getattr(proxy_doc, "summary_generation_status", None),
        "summary_docx_url": getattr(proxy_doc, "summary_docx_url", None),
        "summary_generated_at": getattr(proxy_doc, "summary_generated_at", None),
        "summary_error_message": getattr(proxy_doc, "summary_error_message", None),
        "agent_response": getattr(proxy_doc, "agent_response", None),
        "error_message": getattr(proxy_doc, "error_message", None),
        "completed_at": getattr(proxy_doc, "completed_at", None),
        "total_sections": getattr(proxy_doc, "total_sections", None),
        "empty_sections": getattr(proxy_doc, "empty_sections", None),
        "iteration_count": getattr(proxy_doc, "iteration_count", None),
    }


def sync_proxy_document_to_sec_filing_summary(proxy_doc):
    """
    Sync a ProxyDocument's state to the unified SECFilingSummary (proxy node).

    - Finds SECFilingSummary by accession_number (or sec_filling_id) + form_type, or by sec_document_url.
    - Updates proxy, filing_date, deal_id, updated_at; creates the document if missing.

    Args:
        proxy_doc: ProxyDocument instance (from proxy_processor.models)
    """
    if proxy_doc is None:
        return
    accession = getattr(proxy_doc, "accession_number", None) or getattr(proxy_doc, "sec_filling_id", None)
    form_type = getattr(proxy_doc, "form_type", None)
    sec_document_url = getattr(proxy_doc, "proxy_sec_url", None)
    if not form_type or not sec_document_url:
        logger.warning("sync_proxy_document_to_sec_filing_summary: missing form_type or proxy_sec_url")
        return

    proxy_payload = _proxy_doc_to_proxy_payload(proxy_doc)
    filing_date = _parse_filing_date(getattr(proxy_doc, "filing_date", None))
    deal_id = getattr(proxy_doc, "deal_id", None)
    cik_number = getattr(proxy_doc, "cik_number", None) or ""
    company_name = getattr(proxy_doc, "company_name", None) or ""

    existing = None
    if accession:
        existing = SECFilingSummary.objects(
            accession_number=accession,
            form_type=form_type,
        ).first()
    if not existing and sec_document_url:
        existing = SECFilingSummary.objects(
            sec_document_url=sec_document_url,
            form_type=form_type,
        ).first()

    if existing:
        existing.proxy = proxy_payload
        if filing_date is not None:
            existing.filing_date = filing_date
        if deal_id is not None:
            existing.deal_id = deal_id
        existing.save()
        logger.info(
            "Updated SECFilingSummary proxy for %s %s",
            form_type,
            accession or sec_document_url[:50],
        )
    else:
        SECFilingSummary(
            form_type=form_type,
            accession_number=accession,
            cik_number=cik_number,
            sec_document_url=sec_document_url,
            filing_date=filing_date,
            deal_id=deal_id,
            proxy=proxy_payload,
        ).save()
        logger.info(
            "Created SECFilingSummary proxy for %s %s",
            form_type,
            accession or sec_document_url[:50],
        )
