"""
Management command to test the proxy summary V2 flow (Pinecone chunks → merger background → Q&A → DOCX → S3).

Usage:
  # Full flow for a given filing summary ID
  python manage.py test_proxy_summary_v2 --filing_summary_id=7f1ff5e6-aedc-4a85-a087-b81a72d8e881

  # Only test fetching background chunks from Pinecone (no LLM, no DOCX, no S3)
  python manage.py test_proxy_summary_v2 --filing_summary_id=7f1ff5e6-aedc-4a85-a087-b81a72d8e881 --chunks-only

  # Full flow but skip uploading to S3 (DOCX is created and path printed, then deleted)
  python manage.py test_proxy_summary_v2 --filing_summary_id=7f1ff5e6-aedc-4a85-a087-b81a72d8e881 --no-upload

  # List recent proxy SECFilingSummary docs (to pick an ID)
  python manage.py test_proxy_summary_v2 --list --limit=10
"""
import os

from django.core.management.base import BaseCommand

from sec_rss_parser.models import SECFilingSummary
from sec_rss_parser.proxy_summary_service_v2 import ProxySummaryServiceV2


def _resolve_questions_file():
    """Same resolution as proxy_processor_helper.generate_proxy_summary_v2."""
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "..", "proxy_processor", "quetions.json"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
            __file__)))), "rag_project", "proxy_processor", "quetions.json"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


class Command(BaseCommand):
    help = "Test proxy summary V2 flow: Pinecone chunks → merger background → Q&A → DOCX → S3."

    def add_arguments(self, parser):
        parser.add_argument(
            "--filing_summary_id",
            type=str,
            default=None,
            help="SECFilingSummary document ID (UUID). Required unless --list.",
        )
        parser.add_argument(
            "--chunks-only",
            action="store_true",
            help="Only fetch background chunks from Pinecone and print length; no LLM, DOCX, or S3.",
        )
        parser.add_argument(
            "--no-upload",
            action="store_true",
            help="Run full flow but do not upload DOCX to S3 (create and print path, then delete).",
        )
        parser.add_argument(
            "--list",
            action="store_true",
            help="List recent SECFilingSummary docs with proxy (form_type proxy); then exit.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=10,
            help="When using --list, max number of records to show (default 10).",
        )

    def handle(self, *args, **options):
        if options["list"]:
            self._list_proxy_summaries(options["limit"])
            return

        filing_summary_id = (options.get("filing_summary_id") or "").strip()
        if not filing_summary_id:
            self.stdout.write(self.style.ERROR(
                "--filing_summary_id is required (or use --list to see IDs)."))
            return

        # Optional: resolve from DB to validate and show doc info
        filing_summary = SECFilingSummary.objects(
            _id=filing_summary_id).first()
        if not filing_summary:
            self.stdout.write(
                self.style.WARNING(
                    f"No SECFilingSummary found with _id={filing_summary_id}; continuing with ID only.")
            )
        else:
            self.stdout.write(
                f"SECFilingSummary: form_type={filing_summary.form_type}, "
                f"accession={getattr(filing_summary, 'accession_number', 'N/A')}, "
                f"deal_id={getattr(filing_summary, 'deal_id', 'N/A')}"
            )

        service = ProxySummaryServiceV2()

        if options["chunks_only"]:
            self._run_chunks_only(service, filing_summary_id)
            return

        if options["no_upload"]:
            self._run_full_no_upload(service, filing_summary_id)
            return

        self._run_full(service, filing_summary_id)

    def _run_chunks_only(self, service: ProxySummaryServiceV2, sec_filing_summary_id: str):
        """Only test get_background_chunks_by_filing_id."""
        self.stdout.write(
            "Running chunks-only: get_background_chunks_by_filing_id(...)")
        document_text = service.get_background_chunks_by_filing_id(
            sec_filing_summary_id)
        if not document_text:
            self.stdout.write(self.style.ERROR(
                "No background chunks returned (see logs for Pinecone diagnostic)."))
            return
        self.stdout.write(self.style.SUCCESS(
            f"Chunks fetched: document length = {len(document_text):,} characters"))
        self.stdout.write("First 500 chars:")
        self.stdout.write(document_text[:500])
        if len(document_text) > 500:
            self.stdout.write("...")

    def _run_full_no_upload(self, service: ProxySummaryServiceV2, sec_filing_summary_id: str):
        """Run full flow but skip S3 upload by temporarily overriding upload_file."""
        questions_file = _resolve_questions_file()
        if questions_file:
            self.stdout.write(f"Using questions file: {questions_file}")
        else:
            self.stdout.write(self.style.WARNING(
                "No questions file found; Q&A section will be empty."))

        # Run normal generate_summary_document; we'll intercept S3 upload
        original_upload = service.s3_service.upload_file

        def no_upload_upload_file(file_path, s3_key, content_type=None):
            self.stdout.write(self.style.SUCCESS(
                f"[no-upload] Would upload: {file_path} -> s3_key={s3_key}"))
            # Return a fake URL so the rest of the flow doesn't break
            return f"file://localhost{file_path}"

        service.s3_service.upload_file = no_upload_upload_file
        try:
            result = service.generate_summary_document(
                sec_filing_summary_id=sec_filing_summary_id,
                questions_file=questions_file,
            )
        finally:
            service.s3_service.upload_file = original_upload

        if result.get("success"):
            self.stdout.write(self.style.SUCCESS(
                f"Summary generation succeeded (no S3 upload). docx_url={result.get('docx_url')}"))
        else:
            self.stdout.write(self.style.ERROR(
                f"Summary generation failed: {result.get('error', 'Unknown error')}"))

    def _run_full(self, service: ProxySummaryServiceV2, sec_filing_summary_id: str):
        """Run full flow including S3 upload."""
        questions_file = _resolve_questions_file()
        if questions_file:
            self.stdout.write(f"Using questions file: {questions_file}")
        else:
            self.stdout.write(self.style.WARNING(
                "No questions file found; Q&A section will be empty."))

        result = service.generate_summary_document(
            sec_filing_summary_id=sec_filing_summary_id,
            questions_file=questions_file,
        )

        if result.get("success"):
            self.stdout.write(self.style.SUCCESS(
                f"Summary generated and uploaded: {result.get('docx_url')}"))
            self.stdout.write(f"S3 key: {result.get('s3_key')}")
        else:
            self.stdout.write(self.style.ERROR(
                f"Summary generation failed: {result.get('error', 'Unknown error')}"))

    def _list_proxy_summaries(self, limit: int):
        """List recent SECFilingSummary docs that have proxy (proxy form types)."""
        # Prefer docs that have proxy and optionally pinecone completed
        cursor = (
            SECFilingSummary.objects(form_type__in=[
                                     "DEFM14A", "DEF 14A", "DEFM14C", "PREM14A", "PREM14C", "S-4", "F-4", "S-4/A", "F-4/A"])
            .order_by("-updated_at")
            .limit(limit)
        )
        rows = list(cursor)
        if not rows:
            self.stdout.write(self.style.WARNING(
                "No proxy SECFilingSummary documents found."))
            return
        self.stdout.write(self.style.SUCCESS(
            f"Recent proxy SECFilingSummary (limit={limit}):"))
        for doc in rows:
            proxy = doc.proxy or {}
            pc_status = proxy.get("pinecone_processing_status", "N/A")
            summary_status = proxy.get("summary_generation_status", "N/A")
            self.stdout.write(
                f"  _id={doc.id}  form_type={doc.form_type}  accession={doc.accession_number or 'N/A'}  "
                f"pinecone={pc_status}  summary={summary_status}"
            )


# python manage.py test_proxy_summary_v2 --filing_summary_id=7f1ff5e6-aedc-4a85-a087-b81a72d8e881 --chunks-only
