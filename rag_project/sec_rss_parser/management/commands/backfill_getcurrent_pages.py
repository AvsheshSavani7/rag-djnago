"""
Morning getcurrent deep-page backfill (pages start=100..1900).

Why:
  After midnight the live collector only watches start=0&count=100. SEC often
  surfaces prior-day / late-sorted filings deeper in the rolling window by
  ~5–7 AM ET. Those are missing from yesterday's master.idx (published earlier)
  and never appear on page 1 while buried — so evening reconcile cannot catch
  them either.

This command (keep collector + evening reconcile unchanged):
  1. Fetch getcurrent Atom pages start=100,200,…,1900 (skip start=0).
  2. Skip rows already present in today's OR previous 4 days' feed JSON.
  3. Optionally skip AccessionLookedUp (default on).
  4. Append remaining rows into TODAY's feed JSON for B/B2/B3 to pick up.
  5. Write a dedicated JSONL audit log AND a pipeline .log (logs UI).

Usage:
    python manage.py backfill_getcurrent_pages
    python manage.py backfill_getcurrent_pages --dry-run
    python manage.py backfill_getcurrent_pages --lookback-days 4

Cron (America/New_York), e.g.:
    TZ=America/New_York
    0 1-23 * * * cd /app/rag_project && python manage.py backfill_getcurrent_pages
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from django.conf import settings
from django.core.management.base import BaseCommand

from core.pipeline_logger import SEC_FEED_BACKFILL, start_pipeline
from sec_rss_parser.models import AccessionLookedUp
from sec_rss_parser.sec_feed_collector import (
    DEFAULT_HEADERS,
    build_session,
    parse_global_all_forms,
)
from sec_rss_parser.sec_feed_daily_store import (
    FEED_SOURCE_GETCURRENT_BACKFILL,
    SEC_FEED_TZ,
    append_feed_items,
    default_feed_dir,
    feed_now,
    feed_path_for_date,
    load_daily_feed,
)
from sec_rss_parser.sec_feed_item_utils import make_feed_item_key
from sec_rss_parser.sec_rate_limit import rate_limited_get
from sec_rss_parser.utils_8k import normalize_cik

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 4

GETCURRENT_URL_TEMPLATE = (
    "https://www.sec.gov/cgi-bin/browse-edgar?"
    "action=getcurrent&CIK=&type=&company=&dateb=&owner=include&"
    "start={start}&count={count}&output=atom"
)


def _backfill_log_path(feed_dir: str, day: Optional[datetime] = None) -> Path:
    """
    Dedicated JSONL log (separate from pipeline daily logs).

    Prefer LOG_ROOT/sec_feed_backfill/daily/YYYY-MM-DD/getcurrent_backfill.jsonl
    so it lands on the same host volume as other Django logs in docker.
    Fallback: {feed_dir}/backfill_logs/backfill_YYYYMMDD.jsonl
    """
    d = day or feed_now()
    date_str = d.astimezone(SEC_FEED_TZ).strftime("%Y-%m-%d")
    date_compact = d.astimezone(SEC_FEED_TZ).strftime("%Y%m%d")

    log_root = getattr(settings, "LOG_ROOT",
                       None) or os.environ.get("LOG_ROOT")
    if log_root:
        folder = Path(log_root) / "sec_feed_backfill" / "daily" / date_str
    else:
        folder = Path(feed_dir) / "backfill_logs"
    folder.mkdir(parents=True, exist_ok=True)

    if log_root:
        return folder / "getcurrent_backfill.jsonl"
    return folder / f"backfill_{date_compact}.jsonl"


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_known_keys(feed_dir: str, days: List[datetime]) -> Set[str]:
    """Union of cik|accession keys from the given feed days."""
    known: Set[str] = set()
    for day in days:
        path, data = load_daily_feed(feed_dir, day=day)
        items = data.get("items") or {}
        if not isinstance(items, dict):
            continue
        for key in items.keys():
            if key:
                known.add(key)
        # Also normalize from record fields in case of legacy keys
        for key, raw in items.items():
            if not isinstance(raw, dict):
                continue
            acc = (raw.get("accession_number") or "").strip()
            cik = normalize_cik(raw.get("cik_number") or "")
            if acc and cik:
                known.add(make_feed_item_key(cik, acc))
        logger.info(
            "backfill_getcurrent: loaded %d key(s) from %s",
            len(items),
            os.path.basename(path),
        )
    return known


def _feed_days_for_lookback(today: datetime, lookback_days: int) -> List[datetime]:
    """Today plus the previous N calendar days in SEC feed TZ."""
    n = max(0, int(lookback_days))
    return [today - timedelta(days=i) for i in range(n + 1)]


def _fetch_page(session, start: int, count: int) -> Optional[str]:
    url = GETCURRENT_URL_TEMPLATE.format(start=start, count=count)
    try:
        resp = rate_limited_get(
            session, url, headers=DEFAULT_HEADERS, timeout=45
        )
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.warning(
            "backfill_getcurrent: fetch failed start=%d: %s", start, e
        )
        return None


def run_backfill(
    feed_dir: str,
    start: int = 100,
    end: int = 2000,
    step: int = 100,
    count: int = 100,
    skip_looked_up: bool = True,
    dry_run: bool = False,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> Dict[str, Any]:
    """
    Scan getcurrent pages [start, end) and merge missing rows into today's feed.

    Dedupes against today's feed JSON plus the previous ``lookback_days`` files
    (default 4: e.g. Sunday also sees Sat/Fri/Thu/Wed).

    Returns a summary dict (also written to JSONL + pipeline .log).
    """
    start_pipeline(SEC_FEED_BACKFILL, doc_type="BACKFILL")
    today = feed_now()
    lookback_days = max(0, int(lookback_days))
    feed_days = _feed_days_for_lookback(today, lookback_days)
    log_path = _backfill_log_path(feed_dir, day=today)
    run_ts = today.replace(microsecond=0).isoformat()

    known = _load_known_keys(feed_dir, feed_days)
    logger.info(
        "backfill_getcurrent: lookback_days=%d known_keys=%d days=%s",
        lookback_days,
        len(known),
        ",".join(d.astimezone(SEC_FEED_TZ).strftime("%Y%m%d")
                 for d in feed_days),
    )
    session = build_session()

    to_add: List[Dict[str, Any]] = []
    seen_this_run: Set[str] = set()
    pages_ok = 0
    pages_fail = 0
    considered = 0
    skipped_known = 0
    skipped_looked_up = 0
    skipped_incomplete = 0
    page_hits: List[Dict[str, Any]] = []  # for per-record page_start logging

    # start inclusive, end exclusive-ish: start=100..1900 when end=2000 step=100
    page_starts = list(range(start, end, step))
    for page_start in page_starts:
        raw = _fetch_page(session, page_start, count)
        if not raw:
            pages_fail += 1
            continue
        pages_ok += 1
        items = parse_global_all_forms(raw)
        for item in items:
            considered += 1
            acc = (item.get("accession_number") or "").strip()
            cik = normalize_cik(item.get("cik_number") or "")
            if not acc or not cik:
                skipped_incomplete += 1
                continue
            key = make_feed_item_key(cik, acc)
            if key in known or key in seen_this_run:
                skipped_known += 1
                continue
            if skip_looked_up and AccessionLookedUp.objects(
                accession_number=acc
            ).first():
                skipped_looked_up += 1
                known.add(key)  # don't re-check same acc this run
                continue
            seen_this_run.add(key)
            item["source"] = FEED_SOURCE_GETCURRENT_BACKFILL
            to_add.append(item)
            page_hits.append({
                "accession_number": acc,
                "cik_number": cik,
                "form_type": item.get("form_type"),
                "company_name": item.get("company_name"),
                "title": item.get("title"),
                "link": item.get("link"),
                "page_start": page_start,
            })

    added = 0
    new_accs: List[str] = []
    if to_add and not dry_run:
        added, new_accs = append_feed_items(feed_dir, to_add, day=today)
    elif to_add and dry_run:
        added = len(to_add)
        new_accs = [
            (i.get("accession_number") or "").strip() for i in to_add
        ]

    today_feed = os.path.basename(feed_path_for_date(feed_dir, day=today))
    summary = {
        "event": "run_summary",
        "ts": run_ts,
        "timezone": str(SEC_FEED_TZ),
        "dry_run": dry_run,
        "lookback_days": lookback_days,
        "known_keys": len(known),
        "start": start,
        "end": end,
        "step": step,
        "count": count,
        "pages_ok": pages_ok,
        "pages_fail": pages_fail,
        "considered": considered,
        "skipped_known": skipped_known,
        "skipped_looked_up": skipped_looked_up,
        "skipped_incomplete": skipped_incomplete,
        "candidates": len(to_add),
        "added": added,
        "feed_file": today_feed,
        "log_file": str(log_path),
        "new_accessions": new_accs[:50],
    }

    # JSONL audit + pipeline .log (frontend logs_api).
    _append_jsonl(log_path, summary)
    start_pipeline(SEC_FEED_BACKFILL, doc_type="BACKFILL")
    logger.info(
        "backfill_getcurrent: done | lookback_days=%d known_keys=%d "
        "pages_ok=%d fail=%d considered=%d skipped_known=%d "
        "skipped_looked_up=%d added=%d dry_run=%s log=%s",
        lookback_days,
        len(known),
        pages_ok,
        pages_fail,
        considered,
        skipped_known,
        skipped_looked_up,
        added,
        dry_run,
        log_path,
    )
    if added and page_hits:
        by_acc = {}
        for hit in page_hits:
            a = hit["accession_number"]
            if a not in by_acc:
                by_acc[a] = hit
        for acc in new_accs:
            hit = by_acc.get(acc) or {"accession_number": acc}
            record = {
                "event": "record_added",
                "ts": run_ts,
                "dry_run": dry_run,
                "reason": "missing_from_lookback_window",
                "source": FEED_SOURCE_GETCURRENT_BACKFILL,
                "feed_file": today_feed,
                "page_start": hit.get("page_start"),
                "accession_number": hit.get("accession_number") or acc,
                "cik_number": hit.get("cik_number"),
                "form_type": hit.get("form_type"),
                "company_name": hit.get("company_name"),
                "title": hit.get("title"),
                "link": hit.get("link"),
            }
            _append_jsonl(log_path, record)
            start_pipeline(
                SEC_FEED_BACKFILL,
                accession=record["accession_number"],
                doc_type=(record.get("form_type") or "BACKFILL"),
            )
            logger.info(
                "record_added | accession=%s cik=%s form=%s company=%s "
                "page_start=%s feed=%s dry_run=%s",
                record["accession_number"],
                record.get("cik_number") or "-",
                record.get("form_type") or "-",
                record.get("company_name") or "-",
                record.get("page_start"),
                today_feed,
                dry_run,
            )

    return summary


class Command(BaseCommand):
    help = (
        "Backfill today's SEC feed JSON from getcurrent pages start=100..1900 "
        "(skip page 0). Dedupes against today + previous 4 days of feed JSON. "
        "Writes JSONL audit + pipeline log (sec_feed_backfill) for the logs UI."
    )

    def add_arguments(self, parser):
        parser.add_argument("--feed-dir", default=default_feed_dir())
        parser.add_argument(
            "--start",
            type=int,
            default=100,
            help="First getcurrent start offset (default 100; skip page 0)",
        )
        parser.add_argument(
            "--end",
            type=int,
            default=2000,
            help="End offset exclusive for range() (default 2000 → last start=1900)",
        )
        parser.add_argument("--step", type=int, default=100)
        parser.add_argument("--count", type=int, default=100)
        parser.add_argument(
            "--include-looked-up",
            action="store_true",
            help="Do NOT skip AccessionLookedUp (default: skip already processed)",
        )
        parser.add_argument(
            "--lookback-days",
            type=int,
            default=DEFAULT_LOOKBACK_DAYS,
            help=(
                "Previous calendar days of feed JSON to treat as already known "
                f"(default {DEFAULT_LOOKBACK_DAYS}; plus today). "
                "Covers a Sunday run seeing Friday's file."
            ),
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Fetch and report adds without writing the feed JSON",
        )

    def handle(self, *args, **opts):
        summary = run_backfill(
            feed_dir=opts["feed_dir"],
            start=opts["start"],
            end=opts["end"],
            step=opts["step"],
            count=opts["count"],
            skip_looked_up=not opts["include_looked_up"],
            dry_run=opts["dry_run"],
            lookback_days=opts["lookback_days"],
        )
        msg = (
            f"getcurrent backfill | lookback_days={summary['lookback_days']} "
            f"known_keys={summary['known_keys']} "
            f"pages_ok={summary['pages_ok']} "
            f"fail={summary['pages_fail']} considered={summary['considered']} "
            f"skipped_known={summary['skipped_known']} "
            f"skipped_looked_up={summary['skipped_looked_up']} "
            f"added={summary['added']} dry_run={summary['dry_run']}\n"
            f"log: {summary['log_file']}"
        )
        if summary.get("new_accessions"):
            preview = ", ".join(summary["new_accessions"][:15])
            more = len(summary["new_accessions"]) - 15
            if more > 0:
                preview += f" ... (+{more} more)"
            msg += f"\nadded accessions: {preview}"
        self.stdout.write(self.style.SUCCESS(msg))
