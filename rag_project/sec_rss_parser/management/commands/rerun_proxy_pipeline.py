"""
Re-run the proxy background pipeline for an existing SECFilingSummary.

Only resets/updates the `proxy` subdocument. Does NOT modify L1_headline, L2_brief,
L3_detailed, s3_docx_url, s3_json_url, or other top-level filing summary fields.

Usage:
  # Full pipeline: agentic scrape -> Pinecone -> background summary (async, returns immediately)
  python manage.py rerun_proxy_pipeline --filing_summary_id=bee3e70c-045d-46bc-93e5-22364b7c1110

  # Same pipeline, blocking until finished (local debugging; can take a long time)
  python manage.py rerun_proxy_pipeline --filing_summary_id=bee3e70c-045d-46bc-93e5-22364b7c1110 --sync

  # Re-embed sections only (requires proxy.s3_urls.sections_json_url from a prior scrape)
  python manage.py rerun_proxy_pipeline --filing_summary_id=... --step=pinecone

  # Re-generate background summary DOCX only (requires Pinecone chunks)
  python manage.py rerun_proxy_pipeline --filing_summary_id=... --step=summary

  # Show proxy node status without running anything
  python manage.py rerun_proxy_pipeline --filing_summary_id=... --status-only
"""
from django.core.management.base import BaseCommand, CommandError

from sec_rss_parser.models import SECFilingSummary
from sec_rss_parser.proxy_processor_helper import rerun_proxy_pipeline


class Command(BaseCommand):
    help = (
        "Re-run proxy pipeline (scrape/Pinecone/background summary) for an existing "
        "SECFilingSummary; only updates the proxy subdocument."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--filing_summary_id",
            type=str,
            required=True,
            help="SECFilingSummary document ID (UUID).",
        )
        parser.add_argument(
            "--step",
            type=str,
            choices=("all", "pinecone", "summary"),
            default="all",
            help=(
                "Pipeline step: all (default) = agentic scrape + Pinecone + summary; "
                "pinecone = re-embed from sections_json_url; summary = background DOCX only."
            ),
        )
        parser.add_argument(
            "--sync",
            action="store_true",
            help="Run in the foreground until the step completes (slow for step=all).",
        )
        parser.add_argument(
            "--status-only",
            action="store_true",
            help="Print proxy subdocument status and exit without running the pipeline.",
        )

    def handle(self, *args, **options):
        filing_summary_id = (options["filing_summary_id"] or "").strip()
        if not filing_summary_id:
            raise CommandError("--filing_summary_id is required")

        filing_summary = SECFilingSummary.objects(_id=filing_summary_id).first()
        if not filing_summary:
            raise CommandError(
                f"No SECFilingSummary found with _id={filing_summary_id}")

        if options["status_only"]:
            self._print_status(filing_summary)
            return

        self.stdout.write(
            f"SECFilingSummary: form_type={filing_summary.form_type} "
            f"accession={filing_summary.accession_number or 'N/A'} "
            f"deal_id={filing_summary.deal_id or 'N/A'}"
        )
        self.stdout.write(
            f"sec_document_url: {filing_summary.sec_document_url or 'MISSING'}"
        )
        self.stdout.write(
            "Note: L1/L2/L3 and top-level s3_docx_url/s3_json_url are NOT modified."
        )

        proxy_before = filing_summary.proxy or {}
        self.stdout.write(
            f"Proxy before: parsing={proxy_before.get('proxy_parsing_status')} "
            f"pinecone={proxy_before.get('pinecone_processing_status')} "
            f"summary={proxy_before.get('summary_generation_status')}"
        )

        try:
            result = rerun_proxy_pipeline(
                filing_summary_id=filing_summary_id,
                sync=options["sync"],
                step=options["step"],
            )
        except ValueError as exc:
            raise CommandError(str(exc)) from exc

        mode = result.get("mode", "async")
        step = result.get("step", options["step"])

        if mode == "async":
            self.stdout.write(
                self.style.SUCCESS(
                    f"Started proxy pipeline step={step} in background. "
                    f"Watch logs or re-run with --status-only."
                )
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    f"Completed proxy pipeline step={step} (sync)."
                )
            )

        self.stdout.write(f"Result: {result}")

        filing_summary.reload()
        self._print_status(filing_summary)

    def _print_status(self, filing_summary):
        p = filing_summary.proxy or {}
        s3 = p.get("s3_urls") or {}
        self.stdout.write(self.style.SUCCESS("Proxy subdocument status:"))
        self.stdout.write(f"  proxy_parsing_status: {p.get('proxy_parsing_status')}")
        self.stdout.write(f"  empty_percentage: {p.get('empty_percentage')}")
        err = p.get("error_message") or ""
        if err:
            self.stdout.write(f"  error_message: {err[:300]}{'...' if len(err) > 300 else ''}")
        self.stdout.write(f"  sections_json_url: {s3.get('sections_json_url')}")
        self.stdout.write(
            f"  pinecone_processing_status: {p.get('pinecone_processing_status')}"
        )
        if p.get("pinecone_error_message"):
            self.stdout.write(
                f"  pinecone_error_message: {p.get('pinecone_error_message')}"
            )
        self.stdout.write(
            f"  summary_generation_status: {p.get('summary_generation_status')}"
        )
        self.stdout.write(f"  summary_docx_url: {p.get('summary_docx_url')}")
