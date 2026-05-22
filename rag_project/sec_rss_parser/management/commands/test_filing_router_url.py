"""
Management command to test the SEC filing router by passing a URL.

Usage:
  python manage.py test_filing_router_url --url="https://www.sec.gov/Archives/edgar/data/..."
  python manage.py test_filing_router_url -u "https://www.sec.gov/..."

  # With deal context (fetched from DB):
  python manage.py test_filing_router_url --url="..." --deal-id="<ObjectId>" --cik="<CIK>"

  # With hardcoded dummy context (no DB needed):
  python manage.py test_filing_router_url --url="..." --dummy-context
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
        parser.add_argument(
            "--deal-id",
            type=str,
            default=None,
            help="Deal ObjectId — fetches real target/acquirer names and tickers from DB.",
        )
        parser.add_argument(
            "--cik",
            type=str,
            default=None,
            help="CIK of the filing company — used with --deal-id to resolve primary ticker.",
        )
        parser.add_argument(
            "--dummy-context",
            action="store_true",
            default=False,
            help="Inject hardcoded dummy deal context (no DB needed) to verify prompt injection.",
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

        # Build deal_context ────────────────────────────────────────────────
        deal_context = None

        if options["dummy_context"]:
            deal_context = {
                "primary_ticker":  "DUMMY",
                "target_ticker":   "DUMMY",
                "target_name":     "Dummy Target Corp",
                "acquirer_ticker": "ACQR",
                "acquirer_name":   "Dummy Acquirer Inc",
            }
            self.stdout.write(self.style.WARNING(
                f"\nUsing DUMMY deal context: {deal_context}"
            ))

        elif options["deal_id"]:
            from sec_rss_parser.utils_8k import get_deal_tickers
            deal_id = options["deal_id"].strip()
            cik = (options["cik"] or "").strip() or None
            deal_tickers = get_deal_tickers(deal_id, cik)
            if any(deal_tickers.values()):
                deal_context = {
                    "primary_ticker":  deal_tickers.get("ticker"),
                    "target_ticker":   deal_tickers.get("target_ticker"),
                    "target_name":     deal_tickers.get("target_name"),
                    "acquirer_ticker": deal_tickers.get("acquirer_ticker"),
                    "acquirer_name":   deal_tickers.get("acquirer_name"),
                }
                self.stdout.write(self.style.SUCCESS(
                    f"\nDeal context from DB: {deal_context}"
                ))
            else:
                self.stdout.write(self.style.WARNING(
                    f"\nNo deal data found for deal_id={deal_id} — running without deal context."
                ))

        self.stdout.write("")
        try:
            result = route_and_summarize(url, deal_context=deal_context)
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
