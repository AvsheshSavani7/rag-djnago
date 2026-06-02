"""
Test the RSS merger news 3-prompt flow: Prompt 1 (merger? self-announce?),
Prompt 2 (match deal_id), Prompt 3 (extract deal; save only if self-announce).

Usage:
  python manage.py test_merger_news_classifier --dry-run
  python manage.py test_merger_news_classifier --url "https://example.com/article"
  python manage.py test_merger_news_classifier --output-html /tmp/email.html
"""
import os

from django.core.management.base import BaseCommand

from rss_feeds.merger_news_classifier import (
    get_deals_record_string,
    resolve_rss_item_flow,
)
from rss_feeds.email_templates import generate_rss_feed_item_email_html


DEFAULT_TEST_URL = (
    "https://www.globenewswire.com/news-release/2026/02/26/3246187/0/en/ContextLogic-Completes-907-5-Million-Acquisition-of-US-Salt-Marking-Transformation-into-Business-Ownership-Platform.html"
)


class Command(BaseCommand):
    help = (
        "Test RSS merger news 3-prompt flow: merger check, match deal_id, extract/save deal; generate email."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--url",
            type=str,
            default=None,
            help="Article URL to test. If omitted, uses DEFAULT_TEST_URL defined in this script.",
        )
        parser.add_argument(
            "--title",
            type=str,
            default=None,
            help="Override article title for the test item.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Do not create new deal in DB (Prompt 3 extraction only; no save).",
        )
        parser.add_argument(
            "--output-html",
            type=str,
            default=None,
            help="Write generated email HTML to this file path.",
        )
        parser.add_argument(
            "--no-fetch",
            action="store_true",
            help="Skip fetching HTML (not recommended).",
        )

    def handle(self, *args, **options):
        url = (options.get("url") or "").strip() or DEFAULT_TEST_URL
        title_override = (options.get("title") or "").strip()
        dry_run = options.get("dry_run", False)
        output_html_path = (options.get("output_html") or "").strip() or None
        no_fetch = options.get("no_fetch", False)

        self.stdout.write(self.style.SUCCESS(
            "Testing RSS merger news (3-prompt flow)"))
        self.stdout.write("=" * 60)
        self.stdout.write(f"URL: {url}")
        if title_override:
            self.stdout.write(f"Title override: {title_override}")
        if dry_run:
            self.stdout.write(self.style.WARNING(
                "Dry run: will not create new deal in DB"))
        self.stdout.write("")

        if not os.environ.get("OPENAI_API_KEY_NEWSWIRE"):
            self.stdout.write(
                self.style.ERROR(
                    "OPENAI_API_KEY_NEWSWIRE not set; flow will be skipped.")
            )

        item = {
            "url": url,
            "title": title_override or "Test article",
            "description_text": "",
            "date_published": "",
            "authors": [],
        }
        if no_fetch:
            # Flow will still fetch; we can't skip fetch inside resolve_rss_item_flow easily
            pass

        self.stdout.write("Loading deals from DB...")
        deals_record_string = get_deals_record_string()
        lines = [ln for ln in deals_record_string.splitlines() if ln.strip()]
        self.stdout.write(f"  {len(lines)} deal(s) in DB")
        self.stdout.write("")

        self.stdout.write(
            "Running 3-prompt flow (Prompt 1 → 2 → 3)...")
        result = resolve_rss_item_flow(
            item, deals_record_string, dry_run=dry_run)

        self.stdout.write(self.style.SUCCESS(
            "  Prompt 1: deal we follow? → match + deal_id or false"))
        self.stdout.write(self.style.SUCCESS(
            "  Prompt 2: self-announce new merger? + extract deal fields"))
        self.stdout.write(self.style.SUCCESS(
            "  Prompt 3: US listed & market cap > $100M? → create deal if true"))
        self.stdout.write("")

        skip = result.get("skip_email", True)
        if skip:
            self.stdout.write(self.style.WARNING(
                "  Result: skip (no match, or not self-announce). No save, no email."))
        else:
            self.stdout.write(self.style.SUCCESS("  Result: merger-related."))
            self.stdout.write(
                f"  deal_id: {result.get('deal_id') or '(none)'}")
            self.stdout.write(
                f"  email_note: {result.get('email_note') or '(none)'}")
            if result.get("match_details"):
                md = result["match_details"]
                self.stdout.write(
                    f"  matched_side: {md.get('matched_side') or '(none)'}")
                keywords = md.get('match_keywords') or []
                if keywords:
                    self.stdout.write(
                        f"  match_keywords: {keywords}")
            if result.get("deal_info"):
                d = result["deal_info"]
                self.stdout.write(
                    f"  Deal: {d.get('acquire_name')} / {d.get('target_name')}")
        self.stdout.write("")

        feed_data = {"title": "Scantox Acquires DuplexSeq™ Nonclinical Genomics Safety Business from TwinStrand Biosciences",
                     "source_url": "https://www.prnewswire.com/news-releases/scantox-acquires-duplexseq-nonclinical-genomics-safety-business-from-twinstrand-biosciences-302698141.html", "description_text": "/PRNewswire/ -- Scantox Group today announced it has acquired the nonclinical genomic safety business of TwinStrand Biosciences, Inc. through a technology...", "date_published": "2026-02-26T09:04:00.000+00:00"}
        subject, html_email = generate_rss_feed_item_email_html(
            feed_data,
            item,
            deal_info=result.get("deal_info"),
            email_note=result.get("email_note"),
            match_details=result.get("match_details"),
        )
        self.stdout.write("Generated email:")
        self.stdout.write(f"  Subject: {subject}")
        self.stdout.write(f"  HTML length: {len(html_email)} chars")
        with open("email.html", "w", encoding="utf-8") as f:
            f.write(html_email)

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Done."))
