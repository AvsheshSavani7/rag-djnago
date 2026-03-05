"""
Management command to test the 10-K/10-Q summary pipeline (MongoDB + S3).

Usage:
  python manage.py test_10k_10q_pipeline --deal_id=68f1d30397173821e21c541c
  python manage.py test_10k_10q_pipeline --deal_id=68f1d30397173821e21c541c --urls="https://www.sec.gov/Archives/edgar/data/794619/000079461925000107/amwd-20250731.htm,https://www.sec.gov/Archives/edgar/data/794619/000079461925000115/amwd-20251031.htm"
  python manage.py test_10k_10q_pipeline --deal_id=YOUR_DEAL_ID --urls=URL1,URL2 --env=.env

Ticker is resolved from deal_id via ProcessingJob.target_ticker.
"""
from pathlib import Path

from django.core.management.base import BaseCommand

from sec_rss_parser.models import SECFilingSummary
from sec_rss_parser.tenK_tenQ_pipeline.config import DEFAULT_SEC_URLS
from sec_rss_parser.tenK_tenQ_pipeline.orchestrator import run_pipeline


class Command(BaseCommand):
    help = "Test the 10-K/10-Q summary pipeline (MongoDB + S3). Runs run_pipeline(urls, deal_id)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--deal_id",
            type=str,
            required=True,
            help="Deal ID (ProcessingJob id). Ticker is resolved from this deal.",
        )
        parser.add_argument(
            "--urls",
            type=str,
            default=None,
            help="Comma-separated SEC document URLs. If omitted, uses DEFAULT_SEC_URLS from config.",
        )
        parser.add_argument(
            "--env",
            type=str,
            default=".env",
            help="Path to .env file for API keys (default: .env).",
        )
        parser.add_argument(
            "--skip-assessment",
            action="store_true",
            help="Skip the assessment step (faster run).",
        )

    def handle(self, *args, **options):
        deal_id = options["deal_id"].strip()
        env_path = Path(options["env"]) if options["env"] else None
        skip_assessment = options["skip_assessment"]

        if options["urls"]:
            urls = [u.strip() for u in options["urls"].split(",") if u.strip()]
        else:
            urls = list(DEFAULT_SEC_URLS)

        if not urls:
            self.stdout.write(self.style.ERROR("No URLs provided and DEFAULT_SEC_URLS is empty."))
            return

        self.stdout.write(f"Deal ID: {deal_id}")
        self.stdout.write(f"URLs: {len(urls)}")
        for u in urls:
            self.stdout.write(f"  - {u[:80]}...")
        self.stdout.write("")

        try:
            result = run_pipeline(
                urls=urls,
                deal_id=deal_id,
                env_path=env_path if env_path and env_path.exists() else None,
                skip_assessment=skip_assessment,
            )
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Pipeline failed: {e}"))
            raise

        self.stdout.write(self.style.SUCCESS("\n=== PIPELINE RESULT ==="))
        self.stdout.write(f"Processed:  {len(result['processed'])}")
        self.stdout.write(f"Skipped:    {len(result['skipped'])}")
        self.stdout.write("Comparison outputs (S3 URLs):")
        for k, v in (result.get("comparison_outputs") or {}).items():
            self.stdout.write(f"  {k}: {v or 'N/A'}")

        records = list(
            SECFilingSummary.objects(
                deal_id=deal_id,
                form_type__in=["10-K", "10-Q"],
            )
        )
        self.stdout.write(self.style.SUCCESS(f"\n=== SECFilingSummary records ({len(records)} total) ==="))
        for r in records:
            tq = r.ten_k_ten_q or {}
            self.stdout.write(
                f"  [{tq.get('label', '?')}] processed={tq.get('processed')} "
                f"s3_json={(tq.get('s3_json_url') or 'none')[:70]}..."
            )
