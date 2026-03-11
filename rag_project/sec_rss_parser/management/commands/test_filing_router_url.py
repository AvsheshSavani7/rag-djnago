"""
Management command to test the SEC filing router by passing a URL.

Usage:
  python manage.py test_filing_router_url --url="https://www.sec.gov/Archives/edgar/data/..."
  python manage.py test_filing_router_url -u "https://www.sec.gov/..."
"""
from django.core.management.base import BaseCommand

from sec_rss_parser.sec_summarizers.filing_router import (
    detect_from_url,
    route_and_summarize,
)


class Command(BaseCommand):
    help = "Test the SEC filing router with a URL: detect type, run summarizer, print result."

    def add_arguments(self, parser):
        parser.add_argument(
            "--url",
            "-u",
            type=str,
            required=True,
            help="SEC filing URL to test (required).",
        )

    def handle(self, *args, **options):
        url = (options["url"] or "").strip()
        if not url:
            self.stdout.write(self.style.ERROR("No URL provided. Use --url=... or -u ..."))
            return

        self.stdout.write(f"Testing filing router with URL:\n  {url}\n")

        # Quick URL-based detection (no network)
        detected = detect_from_url(url)
        if detected:
            self.stdout.write(self.style.SUCCESS(f"URL pattern detected: {detected}"))
        else:
            self.stdout.write("URL pattern: no match (will classify from content if needed).")

        self.stdout.write("")
        try:
            result = route_and_summarize(url)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Router failed: {e}"))
            raise

        self.stdout.write(self.style.SUCCESS("\n=== RESULT ==="))
        if isinstance(result, dict):
            for key, value in result.items():
                self.stdout.write(f"  {key}: {value}")
        else:
            self.stdout.write(f"  {result}")
        self.stdout.write(self.style.SUCCESS("Done."))
