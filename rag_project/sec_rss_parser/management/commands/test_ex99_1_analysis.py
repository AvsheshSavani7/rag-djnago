"""
Management command to test EX-99.1 document analysis pipeline.

Calls:
  - download_htm_file(url)
  - extract_document_pages(html_content)
  - analyze_ex99_1_document_with_gpt(document_text, company_name)

Prints the analysis result. URL and company name are configurable via variables or CLI.
"""
import json
import logging
from django.core.management.base import BaseCommand
from sec_rss_parser.document_analyzer import SECDocumentAnalyzer

logger = logging.getLogger(__name__)

# Default values – override via --url and --company or edit here
EX99_1_URL = "https://www.sec.gov/Archives/edgar/data/1522727/000152272726000010/usacq42025ex991.htm"
COMPANY_NAME = "USA Compression Partners, LP"


class Command(BaseCommand):
    help = "Test EX-99.1 analysis: download HTM, extract pages, run GPT analysis, print result"

    def add_arguments(self, parser):
        parser.add_argument(
            "--url",
            type=str,
            default=EX99_1_URL,
            help="EX-99.1 HTM URL to download and analyze",
        )
        parser.add_argument(
            "--company",
            type=str,
            default=COMPANY_NAME,
            help="Company name for GPT context",
        )
        parser.add_argument(
            "--max-pages",
            type=int,
            default=15,
            help="Max pages to extract (chars ≈ 3000 * max_pages). Default: 5",
        )

    def handle(self, *args, **options):
        url = options["url"]
        company_name = options["company"]
        max_pages = options["max_pages"]

        self.stdout.write(f"EX-99.1 URL: {url}")
        self.stdout.write(f"Company name: {company_name}")
        self.stdout.write("")

        analyzer = SECDocumentAnalyzer()

        # 1. Download HTM
        self.stdout.write("Downloading HTM file...")
        html_content = analyzer.download_htm_file(url)
        if not html_content:
            self.stdout.write(self.style.ERROR("Failed to download HTM file"))
            return
        self.stdout.write(self.style.SUCCESS(
            f"Downloaded {len(html_content)} chars"))

        # 2. Extract document pages
        self.stdout.write(
            f"Extracting document pages (max_pages={max_pages})...")
        document_text = analyzer.extract_document_pages(
            html_content, max_pages=max_pages)
        if not document_text:
            self.stdout.write(self.style.ERROR(
                "Failed to extract document text"))
            return
        self.stdout.write(self.style.SUCCESS(
            f"Extracted {len(document_text)} chars"))

        # 3. Analyze with GPT
        self.stdout.write("Running GPT analysis...")
        analysis = analyzer.analyze_ex99_1_document_with_gpt(
            document_text, company_name)

        # 4. Print result
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("--- Analysis result ---"))
        self.stdout.write(json.dumps(analysis, indent=2))
