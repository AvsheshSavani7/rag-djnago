"""
Management command to generate SEC filing L1/L2/L3 summaries locally.

NO MongoDB writes. NO emails. NO S3 uploads.
All outputs saved to local files only.

Usage:
  python manage.py generate_local_summary --url="https://www.sec.gov/Archives/edgar/data/915912/000110465926083113/0001104659-26-083113-index.htm"
  python manage.py generate_local_summary -u "https://www.sec.gov/..." --output-dir="./my_summaries"

  # With deal context (optional):
  python manage.py generate_local_summary --url="..." --target="Target Corp" --acquirer="Acquirer Inc"

Examples:
  # DEFM14A proxy
  python manage.py generate_local_summary -u "https://www.sec.gov/Archives/edgar/data/915912/000110465926083113/tm2619865-1_defm14a.htm"

  # S-4 registration
  python manage.py generate_local_summary -u "https://www.sec.gov/Archives/edgar/data/1234567/000119312526123456/d123456ds4.htm"

  # 8-K with custom output directory
  python manage.py generate_local_summary -u "https://..." --output-dir="/tmp/summaries"
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from django.core.management.base import BaseCommand

# Monkey-patch S3 upload functions to save locally instead


def _patch_s3_functions():
    """Replace S3 upload functions with local file saves"""
    import sec_rss_parser.sec_summarizers.s3_utils as s3_utils

    original_upload_json = s3_utils.upload_json
    original_upload_docx_bytes = s3_utils.upload_docx_bytes

    # Store output directory in a way the patched functions can access it
    _patch_s3_functions._output_dir = None

    def local_upload_json(data, filename):
        """Save JSON to local file instead of S3"""
        output_dir = _patch_s3_functions._output_dir or Path.cwd() / "local_summaries"
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

        return str(filepath), f"file://{filepath.absolute()}"

    def local_upload_docx_bytes(data: bytes, filename: str):
        """Save DOCX bytes to local file instead of S3 (all summarizers use upload_docx_bytes)"""
        output_dir = _patch_s3_functions._output_dir or Path.cwd() / "local_summaries"
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename
        with open(filepath, 'wb') as f:
            f.write(data)

        return str(filepath), f"file://{filepath.absolute()}"

    # Apply patches
    s3_utils.upload_json = local_upload_json
    s3_utils.upload_docx_bytes = local_upload_docx_bytes

    return original_upload_json, original_upload_docx_bytes


class Command(BaseCommand):
    help = "Generate SEC filing L1/L2/L3 summaries locally (no DB, no email, no S3)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--url",
            "-u",
            type=str,
            required=True,
            help="SEC filing URL (required). Can be any SEC.gov URL or press release.",
        )
        parser.add_argument(
            "--output-dir",
            "-o",
            type=str,
            default=None,
            help="Local output directory for JSON and DOCX files (default: ./local_summaries)",
        )
        parser.add_argument(
            "--target",
            type=str,
            default=None,
            help="Target company name (optional, for deal context)",
        )
        parser.add_argument(
            "--target-ticker",
            type=str,
            default=None,
            help="Target ticker symbol (optional, for deal context)",
        )
        parser.add_argument(
            "--acquirer",
            type=str,
            default=None,
            help="Acquirer company name (optional, for deal context)",
        )
        parser.add_argument(
            "--acquirer-ticker",
            type=str,
            default=None,
            help="Acquirer ticker symbol (optional, for deal context)",
        )
        parser.add_argument(
            "--no-docx",
            action="store_true",
            default=False,
            help="Skip DOCX generation, only create JSON",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            default=False,
            help="Show detailed progress logs",
        )

    def handle(self, *args, **options):
        url = (options["url"] or "").strip()
        if not url:
            self.stdout.write(self.style.ERROR(
                "❌ No URL provided. Use --url=... or -u ..."))
            return

        # Set up output directory
        output_dir = options["output_dir"]
        if output_dir:
            output_dir = Path(output_dir).resolve()
        else:
            output_dir = Path.cwd() / "local_summaries"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Apply S3 patches to save locally
        _patch_s3_functions._output_dir = output_dir
        original_upload_json, original_upload_docx_bytes = _patch_s3_functions()

        self.stdout.write(self.style.SUCCESS("=" * 70))
        self.stdout.write(self.style.SUCCESS(
            "🚀 Local SEC Filing Summary Generator"))
        self.stdout.write(self.style.SUCCESS("=" * 70))
        self.stdout.write(f"📄 URL: {url}")
        self.stdout.write(f"📁 Output directory: {output_dir}")
        self.stdout.write("")

        # Build deal context if provided
        deal_context = None
        if any([options["target"], options["acquirer"], options["target_ticker"], options["acquirer_ticker"]]):
            deal_context = {}
            if options["target"]:
                deal_context["target_name"] = options["target"]
            if options["target_ticker"]:
                deal_context["target_ticker"] = options["target_ticker"]
                # Default to target
                deal_context["primary_ticker"] = options["target_ticker"]
            if options["acquirer"]:
                deal_context["acquirer_name"] = options["acquirer"]
            if options["acquirer_ticker"]:
                deal_context["acquirer_ticker"] = options["acquirer_ticker"]

            self.stdout.write(self.style.WARNING("📌 Using deal context:"))
            for key, value in deal_context.items():
                self.stdout.write(f"   {key}: {value}")
            self.stdout.write("")

        # Import router (do this after patching)
        from sec_rss_parser.sec_summarizers.filing_router import route_and_summarize

        # Run the summarizer
        try:
            self.stdout.write(self.style.SUCCESS(
                "🔍 Starting summary generation..."))
            self.stdout.write("")

            result = route_and_summarize(url, deal_context=deal_context)

            if isinstance(result, dict) and result.get("skipped"):
                skip_type = result.get("skip_type") or "unknown"
                skip_reason = result.get("skip_reason") or "no reason given"
                self.stdout.write("")
                self.stdout.write(self.style.WARNING("=" * 70))
                self.stdout.write(self.style.WARNING(
                    f"⚠️  SUMMARY SKIPPED ({skip_type})"))
                self.stdout.write(self.style.WARNING("=" * 70))
                self.stdout.write(skip_reason)
                self.stdout.write("")
                self.stdout.write(
                    "No database writes. No emails sent.")
                return

            self.stdout.write("")
            self.stdout.write(self.style.SUCCESS("=" * 70))
            self.stdout.write(self.style.SUCCESS(
                "✅ SUMMARY GENERATED SUCCESSFULLY"))
            self.stdout.write(self.style.SUCCESS("=" * 70))

            # Display key information
            if isinstance(result, dict):
                self.stdout.write(self.style.SUCCESS("\n📊 Summary Contents:"))
                self.stdout.write("")

                # Show filing metadata
                filing_type = result.get("filing_type", "N/A")
                self.stdout.write(f"  Form Type:      {filing_type}")

                if result.get("target"):
                    self.stdout.write(
                        f"  Target:         {result.get('target')} ({result.get('target_ticker', 'N/A')})")
                if result.get("acquirer"):
                    self.stdout.write(
                        f"  Acquirer:       {result.get('acquirer')} ({result.get('acquirer_ticker', 'N/A')})")
                if result.get("filing_date"):
                    self.stdout.write(
                        f"  Filing Date:    {result.get('filing_date')}")

                self.stdout.write("")

                # Show L1 headline
                if result.get("L1_headline"):
                    self.stdout.write(self.style.WARNING("📌 L1 Headline:"))
                    self.stdout.write(f"  {result.get('L1_headline')}")
                    self.stdout.write("")

                # Show L2 brief
                if result.get("L2_brief"):
                    self.stdout.write(self.style.WARNING("📝 L2 Brief:"))
                    # Wrap text at 70 characters
                    brief = result.get("L2_brief")
                    words = brief.split()
                    line = "  "
                    for word in words:
                        if len(line) + len(word) + 1 > 72:
                            self.stdout.write(line)
                            line = "  " + word
                        else:
                            line += (" " if line != "  " else "") + word
                    if line.strip():
                        self.stdout.write(line)
                    self.stdout.write("")

                # Show file locations
                self.stdout.write(self.style.SUCCESS("💾 Output Files:"))
                if result.get("s3_json_path"):
                    json_path = Path(result["s3_json_path"])
                    self.stdout.write(f"  JSON: {json_path.name}")
                    self.stdout.write(f"        {json_path.absolute()}")

                if result.get("s3_docx_path"):
                    docx_path = Path(result["s3_docx_path"])
                    self.stdout.write(f"  DOCX: {docx_path.name}")
                    self.stdout.write(f"        {docx_path.absolute()}")

                self.stdout.write("")

                # Save a simplified summary to console log
                console_log_path = output_dir / \
                    f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(console_log_path, 'w', encoding='utf-8') as f:
                    f.write(
                        f"SEC Filing Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 70 + "\n\n")
                    f.write(f"URL: {url}\n\n")
                    f.write(f"Form Type: {filing_type}\n")
                    if result.get("target"):
                        f.write(
                            f"Target: {result.get('target')} ({result.get('target_ticker', 'N/A')})\n")
                    if result.get("acquirer"):
                        f.write(
                            f"Acquirer: {result.get('acquirer')} ({result.get('acquirer_ticker', 'N/A')})\n")
                    f.write(
                        f"Filing Date: {result.get('filing_date', 'N/A')}\n\n")
                    f.write("L1 HEADLINE\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"{result.get('L1_headline', 'N/A')}\n\n")
                    f.write("L2 BRIEF\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"{result.get('L2_brief', 'N/A')}\n\n")
                    f.write("L3 DETAILED\n")
                    f.write("-" * 70 + "\n")
                    f.write(json.dumps(result.get(
                        'L3_detailed', {}), indent=2, default=str))

                self.stdout.write(
                    f"  TXT:  summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                self.stdout.write(f"        {console_log_path.absolute()}")

            else:
                self.stdout.write(f"Result: {result}")

            self.stdout.write("")
            self.stdout.write(self.style.SUCCESS(
                "🎉 Done! No database writes. No emails sent."))
            self.stdout.write(self.style.SUCCESS("=" * 70))

        except Exception as e:
            self.stdout.write("")
            self.stdout.write(self.style.ERROR("=" * 70))
            self.stdout.write(self.style.ERROR(
                f"❌ Summary generation failed: {e}"))
            self.stdout.write(self.style.ERROR("=" * 70))

            if options["verbose"]:
                import traceback
                self.stdout.write("")
                self.stdout.write(self.style.ERROR("Full traceback:"))
                traceback.print_exc()

            raise

        finally:
            # Restore original S3 functions
            import sec_rss_parser.sec_summarizers.s3_utils as s3_utils
            s3_utils.upload_json = original_upload_json
            s3_utils.upload_docx_bytes = original_upload_docx_bytes
