from django.core.management.base import BaseCommand

from rss_feeds.feed_builder.core.scanner import scan_all_feeds


class Command(BaseCommand):
    help = (
        "Scan news_source_configs and save new article URLs to news_article_links "
        "(is_processed=false)."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--include-inactive",
            action="store_true",
            help="Scan feeds even when is_active is false.",
        )
        parser.add_argument(
            "--source-id",
            action="append",
            dest="source_ids",
            metavar="ID",
            help="Scan only this source_id (repeatable).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Fetch and parse only — do not write to MongoDB.",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        source_ids = options.get("source_ids")

        summary = scan_all_feeds(
            active_only=not options["include_inactive"],
            dry_run=dry_run,
            source_ids=source_ids,
        )

        prefix = "[dry-run] " if dry_run else ""
        self.stdout.write(
            self.style.SUCCESS(
                f"{prefix}Scanned {summary['feeds_scanned']}/{summary['feeds_total']} feed(s): "
                f"{summary['total_found']} item(s) found, "
                f"{summary['total_new']} new URL(s)."
            )
        )

        for result in summary["results"]:
            line = (
                f"  - {result['source_id']} ({result.get('source_type') or '?'}) "
                f"found={result['found']} new={result['new']}"
            )
            if result.get("error"):
                self.stdout.write(self.style.ERROR(
                    f"{line} error={result['error']}"))
            else:
                self.stdout.write(line)

        if summary["errors"]:
            self.stdout.write(self.style.WARNING(
                f"{len(summary['errors'])} feed(s) failed."))
