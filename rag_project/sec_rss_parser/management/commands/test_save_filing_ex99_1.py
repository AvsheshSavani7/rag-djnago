"""
Management command to test save_filing with the EX-99.1 flow.

Uses a mock item_data based on the Trump Media & Technology Group Corp. 8-K filing:
- Accession: 0001140361-26-005064
- Form: 8-K
- Item 7.01: Regulation FD Disclosure
- Item 9.01: Financial Statements and Exhibits
- Has EX-99.1 exhibit
"""
from datetime import datetime
from django.core.management.base import BaseCommand
from sec_rss_parser.services import SECFeedProcessor, SECRSSParser
import logging

logger = logging.getLogger(__name__)


def get_mock_item_data_ex99_1(use_real_urls: bool = True) -> dict:
    """
    Build mock item_data from the Trump Media 8-K Atom entry.
    Mirrors structure from parse_atom_entry + fetch_and_parse_html.
    """
    # Real EX-99.1 URL from the actual filing index page
    ex99_1_url = (
        "https://www.sec.gov/Archives/edgar/data/1451505/000110465926011776/tm265550d1_ex99-1.htm"
        if use_real_urls
        else "https://www.sec.gov/Archives/edgar/data/1849635/000114036126005064/mock_ex99-1.htm"
    )

    return {
        # From parse_atom_entry
        "title": "8-K - Transocean Ltd. (0001849635) (Filer)",
        "link": "https://www.sec.gov/Archives/edgar/data/1451505/000110465926011776/0001104659-26-011776-index.html",
        "guid": "urn:tag:sec.gov,2008:accession-number=0001104659-26-011776",
        "description": "Filed: 2026-02-12 AccNo: 0001104659-26-011776 Size: 246 KB Item 7.01 Item 9.01",
        "pubDate": "2026-02-12T17:28:04-05:00",
        "form_type": "8-K",
        "accession_number": "0001104659-26-011776",
        # From fetch_and_parse_html (mock)
        "filing_date": "2026-02-12",
        "acceptance_datetime_utc": "2026-02-12T17:28:04-05:00",
        "period": "2026-02-12",
        "company_name": "Transocean Ltd.",
        "cik_number": "0001849635",
        "file_number": "001-40779",
        "ein": "854293042",
        "state_of_incorp": "FL",
        "fiscal_year_end": "1231",
        "assigned_sic": 7370,
        "xbrl_files": [
            {
                "sequence": 1,
                "file": "tm265550d1_8k.htm",
                "type": "8-K",
                "size": 37878,
                "description": "8-K",
                "url": "https://www.sec.gov/Archives/edgar/data/1451505/000110465926011776/tm265550d1_8k.htm",
                "doc_type": "8-K",
            },
            {
                "sequence": 2,
                "file": "tm265550d1_ex99-1.htm",
                "type": "EX-99.1",
                "size": 14389,
                "description": "EXHIBIT 99.1",
                "url": ex99_1_url,
                "doc_type": "EX-99.1",
            },
        ],
        "has_ex21": False,
        "has_ex99_1": True,
    }


class Command(BaseCommand):
    help = "Test save_filing with EX-99.1 flow using mock item_data (Trump Media 8-K)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Build mock item_data and log it; do not call save_filing",
        )
        parser.add_argument(
            "--fetch-real",
            action="store_true",
            help="Fetch real HTML from SEC to get actual xbrl_files instead of mock",
        )
        parser.add_argument(
            "--mock-urls",
            action="store_true",
            help="Use mock (non-downloadable) URLs in xbrl_files (for dry-run only)",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        fetch_real = options["fetch_real"]
        mock_urls = options["mock_urls"]

        self.stdout.write("Building item_data for EX-99.1 flow test...")

        if fetch_real:
            # Fetch real HTML to get actual xbrl_files, has_ex21, has_ex99_1
            parser = SECRSSParser(form_type="8-K")
            html_url = "https://www.sec.gov/Archives/edgar/data/1849635/000114036126005064/0001140361-26-005064-index.htm"
            self.stdout.write(f"Fetching HTML from: {html_url}")
            html_data = parser.fetch_and_parse_html(
                html_url, form_type_from_feed="8-K")
            if not html_data:
                self.stdout.write(self.style.ERROR(
                    "Failed to fetch/parse HTML"))
                return

            item_data = get_mock_item_data_ex99_1(use_real_urls=True)
            item_data.update(html_data)
            self.stdout.write("Using real xbrl_files from SEC")
        else:
            item_data = get_mock_item_data_ex99_1(use_real_urls=not mock_urls)

        self.stdout.write(f"  form_type: {item_data.get('form_type')}")
        self.stdout.write(
            f"  accession_number: {item_data.get('accession_number')}")
        self.stdout.write(f"  company_name: {item_data.get('company_name')}")
        self.stdout.write(f"  has_ex21: {item_data.get('has_ex21')}")
        self.stdout.write(f"  has_ex99_1: {item_data.get('has_ex99_1')}")
        ex99_files = [
            f for f in item_data.get("xbrl_files", [])
            if "EX-99.1" in f.get("type", "") or "EX-99.1" in f.get("description", "")
        ]
        self.stdout.write(f"  EX-99.1 files in xbrl_files: {len(ex99_files)}")
        for f in ex99_files:
            self.stdout.write(f"    -> {f.get('url')}")

        if dry_run:
            self.stdout.write(self.style.WARNING(
                "\nDry run - not calling save_filing"))
            return

        self.stdout.write(
            "\nCalling save_filing (will download EX-99.1, call GPT, then save)...")
        processor = SECFeedProcessor()

        print(f"item_data: {item_data}")
        result = processor.save_filing(item_data)

        if result:
            self.stdout.write(self.style.SUCCESS(
                "save_filing returned True (filing saved)"))
        else:
            self.stdout.write(self.style.WARNING(
                "save_filing returned False (not saved - may already exist or filter)"))
