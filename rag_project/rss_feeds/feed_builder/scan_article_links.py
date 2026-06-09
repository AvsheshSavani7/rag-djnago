#!/usr/bin/env python3
"""
Scan saved newswire configs and insert newly discovered articles into MongoDB.

Reads:  news_source_configs
Writes: news_article_links (is_processed=false for new rows)

Run from rag_project/:

  python rss_feeds/feed_builder/scan_article_links.py
  python rss_feeds/feed_builder/scan_article_links.py --source-id globenewswire_press
  python rss_feeds/feed_builder/scan_article_links.py --dry-run
  python rss_feeds/feed_builder/scan_article_links.py --include-inactive

Equivalent Django command:

  python manage.py scan_news_sources
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_RAG_PROJECT = Path(__file__).resolve().parents[2]
if str(_RAG_PROJECT) not in sys.path:
    sys.path.insert(0, str(_RAG_PROJECT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rag_project.settings")
import django  # noqa: E402

django.setup()

from rss_feeds.feed_builder.core.scanner import scan_all_feeds  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan news_source_configs and save new article links to news_article_links.",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        dest="source_ids",
        metavar="ID",
        help="Scan only this source_id (repeatable). Default: all active feeds.",
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Include feeds where is_active=false.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse only — do not write to news_article_links.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full summary JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log each new article URL.",
    )
    return parser


def _print_summary(summary: dict, *, verbose: bool = False) -> int:
    print(
        f"Scanned {summary['feeds_scanned']}/{summary['feeds_total']} feed(s) "
        f"at {summary['scanned_at']}"
    )
    print(
        f"Found {summary['total_found']} listing item(s), "
        f"{summary['total_new']} new (would save / saved)."
    )

    for result in summary.get("results", []):
        skipped = result.get("skipped") or 0
        line = (
            f"  - {result.get('source_id')} "
            f"({result.get('source_type') or '?'}) "
            f"found={result.get('found')} new={result.get('new')}"
        )
        if skipped:
            line += f" skipped={skipped}"
        if result.get("error"):
            print(f"{line} ERROR: {result['error']}")
        else:
            print(line)
            if verbose and result.get("new_items"):
                for item in result["new_items"][:20]:
                    title = (item.get("title") or "(no title)")[:80]
                    print(f"      + {title}")
                    print(f"        {item.get('detail_url')}")

    if summary.get("errors"):
        print(f"\n{len(summary['errors'])} feed(s) failed.")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    summary = scan_all_feeds(
        active_only=not args.include_inactive,
        dry_run=args.dry_run,
        source_ids=args.source_ids,
    )

    if args.json:
        print(json.dumps(summary, indent=2, default=str))

    return _print_summary(summary, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
