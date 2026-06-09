from django.core.management.base import BaseCommand

from rss_feeds.feed_builder.core.preview import analyze_url
from rss_feeds.feed_builder.core.mongo_config_store import get_feed
from rss_feeds.feed_builder.core.scanner import extract_items_from_feed


class Command(BaseCommand):
    help = "Test a feed_builder source by URL or saved source_id."

    def add_arguments(self, parser):
        parser.add_argument("--url", type=str, help="Listing or RSS URL to test.")
        parser.add_argument("--source-id", type=str, help="Saved feed source_id.")
        parser.add_argument("--limit", type=int, default=10)

    def handle(self, *args, **options):
        limit = options["limit"]

        if options["source_id"]:
            feed = get_feed(options["source_id"])
            if not feed:
                self.stderr.write(self.style.ERROR(f"Unknown source_id: {options['source_id']}"))
                return
            items = extract_items_from_feed(feed, limit=limit)
            self.stdout.write(
                self.style.SUCCESS(
                    f"{feed['source_name']} ({feed['source_type']}): {len(items)} item(s)"
                )
            )
        elif options["url"]:
            analysis = analyze_url(options["url"])
            self.stdout.write(
                self.style.SUCCESS(
                    f"Detected {analysis['source_type']} (resolved: {analysis['resolved_url']})"
                )
            )
            items = analysis["preview_items"][:limit]
        else:
            self.stderr.write(self.style.ERROR("Provide --url or --source-id"))
            return

        for idx, item in enumerate(items, start=1):
            self.stdout.write(
                f"{idx}. {item.get('title') or '(no title)'}\n   {item.get('detail_url')}"
            )
