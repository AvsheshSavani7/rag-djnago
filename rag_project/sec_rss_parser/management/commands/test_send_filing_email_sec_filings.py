"""
Test the two SEC form filings follow-up branches inside _send_filing_email:

1. ex21_merger + 8-K: fetch all SEC forms (1 year before filing_date), send to testing webhook
2. standard + 10-K/10-Q: use deal announce date (or LLM fallback), fetch filtered forms, send to testing webhook

Uses dummy item_data and optional mock deal. Run from project root (rag_project):

  python manage.py test_send_filing_email_sec_filings --ex21
  python manage.py test_send_filing_email_sec_filings --10k
  python manage.py test_send_filing_email_sec_filings --10q
  python manage.py test_send_filing_email_sec_filings --ex21 --10k
"""
from datetime import datetime, timedelta
from types import SimpleNamespace

from django.core.management.base import BaseCommand

from sec_rss_parser.services import SECFeedProcessor


# Dummy CIK that returns real SEC data (Apple Inc.)
DUMMY_CIK = "320193"
DUMMY_COMPANY = "Apple Inc. (Test)"


def dummy_item_data_ex21(cik=DUMMY_CIK, company_name=DUMMY_COMPANY, filing_date=None):
    """Dummy item_data for 8-K EX-2.1 branch."""
    if filing_date is None:
        filing_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    return {
        "form_type": "8-K",
        "cik_number": cik,
        "filing_date": filing_date,
        "company_name": company_name,
        "link": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=" + cik,
        "accession_number": "0000320193-24-000042",
        "xbrl_files": [],
        "company_details": {},
    }


def dummy_item_data_10k_10q(form_type="10-K", cik=DUMMY_CIK, company_name=DUMMY_COMPANY):
    """Dummy item_data for 10-K/10-Q standard branch."""
    return {
        "form_type": form_type,
        "cik_number": cik,
        "filing_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        "company_name": company_name,
        "link": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=" + cik,
        "accession_number": "0000320193-23-000105",
        "xbrl_files": [],
        "company_details": {},
    }


def dummy_matched_deal(announce_date=None, target_name="Test Target", acquire_name="Test Acquirer", sec_url=None):
    """Mock deal for 10-K/10-Q branch (announce_date optional to test LLM path)."""
    if announce_date is None:
        announce_date = datetime(2024, 6, 1)
    if sec_url is None:
        sec_url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000042/aapl-20240201.htm"
    return SimpleNamespace(
        announce_date=announce_date,
        target_name=target_name,
        acquire_name=acquire_name,
        sec_url=sec_url,
    )


class Command(BaseCommand):
    help = (
        "Test SEC form filings follow-up: ex21_merger (8-K) and/or standard (10-K/10-Q) with dummy data."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--ex21",
            action="store_true",
            help="Test branch: email_type=ex21_merger, form_type=8-K (fetch all forms, 1yr from filing_date)",
        )
        parser.add_argument(
            "--10k",
            action="store_true",
            help="Test branch: email_type=standard, form_type=10-K (deal announce date, form filter 10-K)",
        )
        parser.add_argument(
            "--10q",
            action="store_true",
            help="Test branch: email_type=standard, form_type=10-Q (deal announce date, form filter 10-Q)",
        )
        parser.add_argument(
            "--cik",
            type=str,
            default=DUMMY_CIK,
            help=f"CIK to use (default: {DUMMY_CIK})",
        )

    def handle(self, *args, **options):
        run_ex21 = options["ex21"]
        run_10k = options["10k"]
        run_10q = options["10q"]
        cik = (options["cik"] or DUMMY_CIK).strip()

        if not (run_ex21 or run_10k or run_10q):
            self.stdout.write(
                self.style.WARNING("Choose at least one: --ex21, --10k, or --10q")
            )
            self.stdout.write("  Example: python manage.py test_send_filing_email_sec_filings --ex21 --10k")
            return

        processor = SECFeedProcessor()

        # ---- Branch 1: ex21_merger + 8-K ----
        if run_ex21:
            self.stdout.write("\n--- Testing ex21_merger + 8-K ---")
            item_data = dummy_item_data_ex21(cik=cik)
            self.stdout.write(
                f"  item_data: form_type={item_data['form_type']}, cik={cik}, company={item_data['company_name']}"
            )
            try:
                processor._send_filing_email(
                    item_data,
                    "ex21_merger",
                    matched_deal=None,
                    filing=None,
                )
                self.stdout.write(self.style.SUCCESS("  ex21_merger branch completed (main email + SEC filings email)."))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  ex21_merger failed: {e}"))

        # ---- Branch 2: standard + 10-K ----
        if run_10k:
            self.stdout.write("\n--- Testing standard + 10-K ---")
            item_data = dummy_item_data_10k_10q(form_type="10-K", cik=cik)
            matched_deal = dummy_matched_deal()
            self.stdout.write(
                f"  item_data: form_type=10-K, cik={cik}, deal announce_date={matched_deal.announce_date}"
            )
            try:
                processor._send_filing_email(
                    item_data,
                    "standard",
                    matched_deal=matched_deal,
                    filing=None,
                )
                self.stdout.write(self.style.SUCCESS("  standard + 10-K branch completed (main email + SEC filings email)."))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  standard + 10-K failed: {e}"))

        # ---- Branch 3: standard + 10-Q ----
        if run_10q:
            self.stdout.write("\n--- Testing standard + 10-Q ---")
            item_data = dummy_item_data_10k_10q(form_type="10-Q", cik=cik)
            matched_deal = dummy_matched_deal()
            self.stdout.write(
                f"  item_data: form_type=10-Q, cik={cik}, deal announce_date={matched_deal.announce_date}"
            )
            try:
                processor._send_filing_email(
                    item_data,
                    "standard",
                    matched_deal=matched_deal,
                    filing=None,
                )
                self.stdout.write(self.style.SUCCESS("  standard + 10-Q branch completed (main email + SEC filings email)."))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  standard + 10-Q failed: {e}"))

        self.stdout.write("")
