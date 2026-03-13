"""
Management command to test `_llm_form_affects_deal` from `fetch_sec_feed_by_deal_cik`.

Usage:
  python manage.py test_form_affects_deal

All inputs are hard-coded inside this command so you can quickly check
the LLM + web search behavior without passing CLI arguments.
"""

from django.core.management.base import BaseCommand

from sec_rss_parser.fetch_sec_feed_by_deal_cik import _llm_form_affects_deal


class Command(BaseCommand):
    help = "Call _llm_form_affects_deal with static test values and print the result."

    def handle(self, *args, **options):
        # Static test inputs (edit these as needed for experimentation)
        target_name = "E.W. SCRIPPS Co"
        acquirer_name = "Sinclair, Inc."
        sec_url = "https://www.sec.gov/Archives/edgar/data/1971213/000202354026000003/xslF345X05/primary_doc.xml"
        form_type = "4"

        self.stdout.write(self.style.NOTICE(
            "Testing _llm_form_affects_deal with:"))
        self.stdout.write(f"  Target   : {target_name}")
        self.stdout.write(f"  Acquirer : {acquirer_name}")
        self.stdout.write(f"  SEC URL  : {sec_url}")
        self.stdout.write(f"  Form type: {form_type}")
        self.stdout.write("")

        try:
            result = _llm_form_affects_deal(
                target_name=target_name,
                acquirer_name=acquirer_name,
                sec_url=sec_url,
                form_type=form_type,
            )
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Call failed: {e}"))
            return

        self.stdout.write(self.style.SUCCESS("=== LLM RESULT ==="))
        self.stdout.write(f"form_affects_deal: {result!r}")
        self.stdout.write(self.style.SUCCESS("Done."))
