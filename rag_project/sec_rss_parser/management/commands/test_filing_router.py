"""
Management command to test the SEC filing router (detect type → summarizer → result).

Usage:
  python manage.py test_filing_router
  python manage.py test_filing_router --url="https://www.sec.gov/Archives/edgar/data/2087587/000121390026023748/ea0258819-09.htm"
"""
from django.core.management.base import BaseCommand

from sec_rss_parser.sec_summarizers.filing_router import FILING_URL, route_and_summarize


class Command(BaseCommand):
    help = "Test the SEC filing router: detect filing type, run summarizer, print result."

    def add_arguments(self, parser):
        parser.add_argument(
            "--url",
            type=str,
            default=FILING_URL,
            help="SEC filing URL to summarize (default: FILING_URL from filing_router).",
        )

    def handle(self, *args, **options):
        url = (options["url"] or "").strip()
        if not url:
            self.stdout.write(self.style.ERROR("No URL provided. Use --url=..."))
            return

        self.stdout.write(f"URL: {url}\n")
        try:
            result = route_and_summarize(url)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Router failed: {e}"))
            raise

        self.stdout.write(self.style.SUCCESS("\n=== RESULT ==="))
        if isinstance(result, dict):
            for key in ("s3_docx_path", "s3_json_path"):
                if key in result:
                    self.stdout.write(f"  {key}: {result[key]}")
            self.stdout.write(f"  (keys: {list(result.keys())})")
        else:
            self.stdout.write(f"  {result}")
        self.stdout.write(self.style.SUCCESS("Done."))
