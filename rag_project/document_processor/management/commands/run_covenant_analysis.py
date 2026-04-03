"""
Django management command to run the covenant analysis pipeline.

Usage:
    python manage.py run_covenant_analysis \
        --url "https://www.sec.gov/Archives/edgar/data/875045/000119312526134889/d14986dex21.htm" \
        --accession "0001193125-26-134889"

    # Optional: pass deal-id and deal-name
    python manage.py run_covenant_analysis \
        --url "https://www.sec.gov/..." \
        --accession "0001193125-26-134889" \
        --deal-id "69cbae4f640784d45bf14678" \
        --deal-name "Acquirer / Target"
"""

import sys
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Run the S3-based covenant analysis pipeline"

    def add_arguments(self, parser):
        parser.add_argument(
            "--url", type=str, required=True,
            help="SEC EDGAR URL for the document",
        )
        parser.add_argument(
            "--accession", type=str, required=True,
            help="SEC accession number (e.g. 0001193125-26-134889)",
        )
        parser.add_argument(
            "--deal-id", type=str, default="",
            help="Deal ID from deals collection (optional)",
        )
        parser.add_argument(
            "--deal-name", type=str, default="",
            help="Deal name for dashboard header (optional)",
        )

    def handle(self, *args, **options):
        url = options["url"]
        accession = options["accession"]
        deal_id = options.get("deal_id") or ""
        deal_name = options.get("deal_name") or ""

        self.stdout.write("=" * 80)
        self.stdout.write(self.style.SUCCESS("COVENANT ANALYSIS PIPELINE"))
        self.stdout.write("=" * 80)
        self.stdout.write(f"  url:              {url}")
        self.stdout.write(f"  accession_number: {accession}")
        self.stdout.write(f"  deal_id:          {deal_id or '(none)'}")
        self.stdout.write(f"  deal_name:        {deal_name or '(none)'}")
        self.stdout.write("=" * 80)

        _covenant_dir = Path(__file__).resolve().parent.parent.parent.parent / "sec_rss_parser" / "Covenenat Project Feb 2026"
        if str(_covenant_dir) not in sys.path:
            sys.path.insert(0, str(_covenant_dir))

        from covenant_pipeline import run_covenant_pipeline_s3

        run_covenant_pipeline_s3(
            url=url,
            accession_number=accession,
            deal_id=deal_id,
            deal_name=deal_name,
        )

        self.stdout.write(self.style.SUCCESS("\nDone."))

        try:
            from sec_rss_parser.models import CovenantAnalysis
            record = CovenantAnalysis.objects(accession_number=accession).first()
            if record:
                self.stdout.write("\n" + "=" * 80)
                self.stdout.write("MongoDB covenant_analysis record:")
                self.stdout.write("=" * 80)
                for field in [
                    "deal_id", "sec_url", "accession_number",
                    "full_json", "covenants_json", "individual_clauses_json",
                    "classification_json", "summary_csv",
                    "assessment_json",
                    "benchmark_comparison_json", "benchmark_summary_csv",
                    "specific_provisions_json",
                    "dashboard_html", "created_at", "updated_at",
                ]:
                    val = getattr(record, field, None)
                    if val:
                        self.stdout.write(f"  {field:<30}: {val}")
        except Exception as e:
            self.stdout.write(f"  (Could not print record: {e})")
