"""
Reconcile the daily feed JSON against EDGAR's daily index (completeness backstop).

Merges any filing present in master.{YYYYMMDD}.idx but missing from
feed_YYYYMMDD.json, recovering filings the live collector may have missed during
downtime. Safe to run repeatedly (dedup by accession). Run it a couple of times a
day and/or after any collector outage.

Usage:
    python manage.py reconcile_sec_daily_index                 # today (America/New_York)
    python manage.py reconcile_sec_daily_index --date 20260720 # a specific day
    python manage.py reconcile_sec_daily_index --tracked-only  # only tracked deal CIKs
"""

from datetime import datetime

from django.core.management.base import BaseCommand

from sec_rss_parser.sec_daily_index import reconcile_into_feed
from sec_rss_parser.sec_feed_daily_store import SEC_FEED_TZ, default_feed_dir, feed_now


class Command(BaseCommand):
    help = "Merge missing filings from EDGAR daily index into the daily feed JSON."

    def add_arguments(self, parser):
        parser.add_argument("--feed-dir", default=default_feed_dir())
        parser.add_argument(
            "--date",
            help="SEC feed day as YYYYMMDD in America/New_York (default: today)",
        )
        parser.add_argument(
            "--tracked-only",
            action="store_true",
            help="Only merge filings whose CIK belongs to an open/unknown deal",
        )

    def handle(self, *args, **opts):
        if opts.get("date"):
            day = datetime.strptime(opts["date"], "%Y%m%d").replace(tzinfo=SEC_FEED_TZ)
        else:
            day = feed_now()

        tracked_ciks = None
        if opts["tracked_only"]:
            from sec_rss_parser.fetch_sec_feed_by_deal_cik import build_tracked_ciks_map
            tracked_ciks = build_tracked_ciks_map()
            self.stdout.write(f"Tracked CIKs: {len(tracked_ciks)}")

        added, parsed, new_accs = reconcile_into_feed(
            opts["feed_dir"],
            day=day,
            tracked_ciks=tracked_ciks,
        )
        msg = (
            f"Reconcile {day.strftime('%Y-%m-%d')} ({SEC_FEED_TZ}) | "
            f"parsed={parsed} | added={added}"
        )
        if new_accs:
            msg += f"\nAdded accessions: {', '.join(new_accs)}"
        self.stdout.write(self.style.SUCCESS(msg))
