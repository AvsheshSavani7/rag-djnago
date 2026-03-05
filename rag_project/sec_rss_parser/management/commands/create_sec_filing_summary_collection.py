"""
Create the unified sec_filing_summary collection and optionally migrate from existing collections.

Schema: one document per filing; only the nested object for that form_type is set:
  - form_type 8-K -> 8_k populated (Exhibit 99.1 only inside 8_k.filings[]; never separate top-level record).
  - form_type 10-K / 10-Q -> ten_k_ten_q populated, proxy and 8_k null
  - form_type proxy (DEFM14A, DEF 14A, etc.) -> proxy populated, 8_k and ten_k_ten_q null

8-K rules:
  - Same accession_number for 8-K and its Exhibit 99.1; one document per accession.
  - 8_k.filings: [] if only 8-K (no exhibit); otherwise entries with filing_date, filing_url, s3_*, exhibit_type (null for main 8-K, "EX_99.1" for exhibit).
  - Other_filings: always present, null for now (future expansion).

filing_date is stored as MongoDB date type.

Usage:
  python manage.py create_sec_filing_summary_collection
  python manage.py create_sec_filing_summary_collection --migrate
  python manage.py create_sec_filing_summary_collection --migrate --dry-run
"""
from django.core.management.base import BaseCommand
from datetime import datetime
import re

from sec_rss_parser.models import (
    SECFilingSummary,
    EightKSummary,
    Ex99_1Summary,
    TenKTenQSummary,
)
from proxy_processor.models import ProxyDocument


def parse_filing_date(value, source="auto"):
    """
    Parse filing_date string to datetime (midnight UTC) for MongoDB date storage.
    - 8-K: MM/DD/YY
    - 10-K/10-Q: YYYY/MM/DD or YYYY-MM-DD
    - proxy: YYYY-MM-DD
    Returns None if value is falsy or unparseable.
    """
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None

    # YYYY-MM-DD (proxy or 10-K/10-Q)
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", value)
    if m:
        try:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return datetime(y, mo, d, 0, 0, 0, 0)
        except (ValueError, TypeError):
            pass

    # YYYY/MM/DD (10-K/10-Q)
    m = re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})$", value)
    if m:
        try:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return datetime(y, mo, d, 0, 0, 0, 0)
        except (ValueError, TypeError):
            pass

    # MM/DD/YY (8-K): 2-digit year 00-50 -> 2000-2050, 51-99 -> 1951-1999
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2})$", value)
    if m:
        try:
            mo, d, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
            year = (2000 + yy) if yy <= 50 else (1900 + yy)
            return datetime(year, mo, d, 0, 0, 0, 0)
        except (ValueError, TypeError):
            pass

    return None


class Command(BaseCommand):
    help = (
        "Create the sec_filing_summary collection and indexes. "
        "Use --migrate to copy from 8k_summary, 99_1_summary, 10k_10Q_Summary, proxy_documents."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--migrate",
            action="store_true",
            help="Populate sec_filing_summary from existing summary collections.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="With --migrate: only report what would be done, do not insert.",
        )

    def handle(self, *args, **options):
        self.stdout.write(
            "Creating sec_filing_summary collection and indexes...")
        try:
            SECFilingSummary.ensure_indexes()
            self.stdout.write(self.style.SUCCESS(
                "Collection and indexes created."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"ensure_indexes failed: {e}"))
            return

        if not options["migrate"]:
            return

        dry_run = options["dry_run"]
        if dry_run:
            self.stdout.write(self.style.WARNING(
                "DRY RUN — no documents will be inserted."))

        migrated = 0

        # 8-K + EX-99.1 merged by accession_number (one doc per accession; Exhibit 99.1 only in 8_k.filings[])
        count_8k = 0
        self.stdout.write("Migrating 8k_summary + 99_1_summary (merged by accession)...")
        for payload in _build_merged_8k_documents():
            if payload and not dry_run:
                SECFilingSummary(**payload).save()
            if payload:
                count_8k += 1
        migrated += count_8k
        self.stdout.write(self.style.SUCCESS(
            f"  8-K (merged): {count_8k} documents"))

        # 10-K / 10-Q
        count_10 = 0
        self.stdout.write("Migrating 10k_10Q_Summary (TenKTenQSummary)...")
        for doc in TenKTenQSummary.objects.all():
            payload = _ten_k_ten_q_to_unified(doc)
            if payload and not dry_run:
                SECFilingSummary(**payload).save()
            if payload:
                count_10 += 1
        migrated += count_10
        self.stdout.write(self.style.SUCCESS(
            f"  10k_10Q_Summary: {count_10} documents"))

        # Proxy
        count_proxy = 0
        self.stdout.write("Migrating proxy_documents (ProxyDocument)...")
        for doc in ProxyDocument.objects.all():
            payload = _proxy_to_unified(doc)
            if payload and not dry_run:
                SECFilingSummary(**payload).save()
            if payload:
                count_proxy += 1
        migrated += count_proxy
        self.stdout.write(self.style.SUCCESS(
            f"  proxy_documents: {count_proxy} documents"))

        self.stdout.write(
            self.style.SUCCESS(
                f"Migration done. Total documents: {migrated}" +
                (" (dry-run)" if dry_run else "")
            )
        )


def _build_merged_8k_documents():
    """
    Yield one SECFilingSummary payload per accession_number. 8-K and Exhibit 99.1 share the same
    accession; Exhibit 99.1 is only inside 8_k.filings[], never a separate top-level record.
    """
    # Group by accession_number
    by_acc_8k = {}
    for doc in EightKSummary.objects.all():
        acc = (doc.accession_number or "").strip()
        if acc:
            by_acc_8k[acc] = doc  # at most one 8-K per accession

    by_acc_99_1 = {}
    for doc in Ex99_1Summary.objects.all():
        acc = (doc.accession_number or "").strip()
        if acc:
            by_acc_99_1.setdefault(acc, []).append(doc)

    all_accessions = set(by_acc_8k) | set(by_acc_99_1)

    for accession_number in sorted(all_accessions):
        main_8k = by_acc_8k.get(accession_number)
        exhibits_99_1 = by_acc_99_1.get(accession_number) or []

        # Top-level and 8_k summary from main 8-K if present, else from first EX-99.1
        if main_8k:
            source = main_8k
            cik_number = main_8k.cik_number
            sec_document_url = main_8k.sec_document_url
            filing_date = parse_filing_date(getattr(main_8k, "filing_date", None) or None)
            deal_id = main_8k.deal_id
            created_at = main_8k.created_at
            updated_at = main_8k.updated_at
            one_line_summary = main_8k.one_line_summary
            items_reported = list(main_8k.items_reported) if main_8k.items_reported else []
            s3_docx_url = main_8k.s3_docx_url
            s3_json_url = main_8k.s3_json_url
        elif exhibits_99_1:
            source = exhibits_99_1[0]
            cik_number = source.cik_number
            sec_document_url = source.sec_document_url
            filing_date = parse_filing_date(getattr(source, "filing_date", None) or None)
            deal_id = source.deal_id
            created_at = source.created_at
            updated_at = source.updated_at
            one_line_summary = source.one_line_summary
            items_reported = list(source.items_reported) if source.items_reported else []
            s3_docx_url = source.s3_docx_url
            s3_json_url = source.s3_json_url
        else:
            continue

        # Build 8_k.filings: only Exhibit 99.1 entries. The main 8-K is already represented by
        # top-level 8_k (one_line_summary, items_reported, s3_docx_url, s3_json_url); do not
        # add a duplicate entry with exhibit_type null.
        filings = []
        for ex in exhibits_99_1:
            filings.append({
                "filing_date": parse_filing_date(getattr(ex, "filing_date", None) or None),
                "filing_url": ex.sec_document_url,
                "s3_docx_url": ex.s3_docx_url,
                "s3_json_url": ex.s3_json_url,
                "exhibit_type": "EX_99.1",
            })

        yield {
            "accession_number": accession_number,
            "cik_number": cik_number,
            "sec_document_url": sec_document_url,
            "filing_date": filing_date,
            "deal_id": deal_id,
            "created_at": created_at,
            "updated_at": updated_at,
            "form_type": "8-K",
            "proxy": None,
            "ten_k_ten_q": None,
            "eight_k": {
                "one_line_summary": one_line_summary,
                "items_reported": items_reported,
                "s3_docx_url": s3_docx_url,
                "s3_json_url": s3_json_url,
                "filings": filings,
            },
            "other_filings": None,
        }


def _ten_k_ten_q_to_unified(doc):
    """Build SECFilingSummary payload from TenKTenQSummary. ten_k_ten_q set."""
    return {
        "accession_number": doc.accession_number,
        "cik_number": doc.cik_number,
        "sec_document_url": doc.sec_document_url,
        "filing_date": parse_filing_date(doc.filing_date),
        "deal_id": doc.deal_id,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
        "form_type": doc.form_type or "10-K",
        "proxy": None,
        "eight_k": None,
        "other_filings": None,
        "ten_k_ten_q": {
            "processed": bool(doc.s3_json_url or doc.s3_docx_url),
            "processed_at": doc.updated_at,
            "s3_json_url": doc.s3_json_url,
            "s3_docx_url": doc.s3_docx_url,
            "s3_comparison_json_url": None,
            "s3_redline_docx_url": None,
            "s3_client_report_docx_url": None,
            "s3_exec_summary_docx_url": None,
            "label": None,
        },
    }


def _proxy_to_unified(doc):
    """Build SECFilingSummary payload from ProxyDocument. proxy set; sec_document_url = proxy_sec_url."""
    s3_urls = doc.s3_urls or {}
    return {
        "accession_number": getattr(doc, "accession_number", None) or doc.sec_filling_id,
        "cik_number": doc.cik_number,
        "sec_document_url": doc.proxy_sec_url,
        "filing_date": parse_filing_date(doc.filing_date),
        "deal_id": doc.deal_id,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
        "form_type": doc.form_type,
        "eight_k": None,
        "ten_k_ten_q": None,
        "other_filings": None,
        "proxy": {
            "proxy_parsing_status": doc.proxy_parsing_status,
            "empty_percentage": doc.empty_percentage,
            "processing_state": doc.processing_state or {},
            "s3_urls": {
                "pdf_url": s3_urls.get("pdf_url"),
                "toc_pdf_url": s3_urls.get("toc_pdf_url"),
                "toc_json_url": s3_urls.get("toc_json_url"),
                "sections_json_url": s3_urls.get("sections_json_url"),
            },
            "pinecone_processing_status": doc.pinecone_processing_status,
            "pinecone_processed_at": doc.pinecone_processed_at,
            "pinecone_error_message": getattr(doc, "pinecone_error_message", None),
        },
    }
