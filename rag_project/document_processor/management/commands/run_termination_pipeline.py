"""
Django management command to run the termination analysis pipeline.

Usage:
    python manage.py run_termination_pipeline --deal-id <deal_id> --sec-url <url>
    python manage.py run_termination_pipeline --deal-id <deal_id>

Examples:
    # With explicit sec_url
    python manage.py run_termination_pipeline \
        --deal-id 69c668f5cef98f44b7207caa \
        --sec-url "https://www.sec.gov/Archives/edgar/data/1465740/000110465926035663/tm269980d1_ex2-1.htm"

    # Auto-fetch sec_url from the deal record
    python manage.py run_termination_pipeline --deal-id 69c668f5cef98f44b7207caa
"""

from django.core.management.base import BaseCommand, CommandError
from document_processor.services import DocumentProcessingService


class Command(BaseCommand):
    help = "Run the S3-based termination analysis pipeline for a deal"

    def add_arguments(self, parser):
        parser.add_argument(
            "--deal-id",
            type=str,
            required=True,
            help="Deal ID from the deals collection (ProcessingJob _id)",
        )
        parser.add_argument(
            "--sec-url",
            type=str,
            default=None,
            help="SEC EDGAR URL for the EX-2.1 document. If omitted, fetched from the deal record.",
        )
        parser.add_argument(
            "--deal-name",
            type=str,
            default=None,
            help="Deal name for dashboard header. If omitted, built from deal record.",
        )

    def handle(self, *args, **options):
        deal_id = options["deal_id"]
        sec_url = options.get("sec_url")
        deal_name = options.get("deal_name") or ""

        job = self._fetch_job(deal_id)

        if not sec_url:
            if not job.sec_url:
                raise CommandError(
                    f"Deal {deal_id} has no sec_url. Provide --sec-url explicitly."
                )
            sec_url = job.sec_url

        if not deal_name:
            acquirer = getattr(job, "acquire_name", "") or ""
            target = getattr(job, "target_name", "") or ""
            if acquirer and target:
                deal_name = f"{acquirer} / {target}"
            elif target:
                deal_name = target

        accession = DocumentProcessingService._extract_accession_from_url(
            sec_url)
        if not accession:
            raise CommandError(
                f"Could not extract accession number from URL: {sec_url}"
            )

        self.stdout.write("=" * 80)
        self.stdout.write(self.style.SUCCESS("TERMINATION ANALYSIS PIPELINE"))
        self.stdout.write("=" * 80)
        self.stdout.write(f"  deal_id:          {deal_id}")
        self.stdout.write(f"  deal_name:        {deal_name or '(none)'}")
        self.stdout.write(f"  sec_url:          {sec_url}")
        self.stdout.write(f"  accession_number: {accession}")
        self.stdout.write(f"  doc_type:         2.1")
        self.stdout.write("=" * 80)

        svc = DocumentProcessingService()
        svc._run_termination_analysis_pipeline(deal_id, sec_url, deal_name=deal_name)

        self.stdout.write(self.style.SUCCESS("\nDone."))
        self._print_record(accession)

    def _fetch_job(self, deal_id: str):
        from bson import ObjectId
        from document_processor.models import ProcessingJob

        try:
            return ProcessingJob.objects.get(id=ObjectId(deal_id))
        except Exception:
            raise CommandError(f"Deal not found: {deal_id}")

    def _print_record(self, accession: str):
        try:
            from sec_rss_parser.models import TerminationAnalysis

            record = TerminationAnalysis.objects(
                accession_number=accession).first()
            if not record:
                self.stdout.write("  (No MongoDB record found)")
                return

            self.stdout.write("\n" + "=" * 80)
            self.stdout.write("MongoDB termination_analysis record:")
            self.stdout.write("=" * 80)
            for field in [
                "deal_id", "sec_url", "accession_number", "doc_type",
                "full_json", "triggers_json", "fees_json",
                "triggers_raw_json", "triggers_8k_json", "fees_8k_json",
                "classification_json", "summary_csv",
                "assessment_json", "provision_checks_json",
                "dashboard_html",
                "created_at", "updated_at",
            ]:
                val = getattr(record, field, None)
                if val:
                    self.stdout.write(f"  {field:<25}: {val}")
        except Exception as e:
            self.stdout.write(f"  (Could not print record: {e})")
