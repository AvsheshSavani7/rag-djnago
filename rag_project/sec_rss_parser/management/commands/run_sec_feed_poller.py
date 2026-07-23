"""
Run the SEC feed poller (collector + processor + reconcile) in ONE process.

All stages run as threads in the same process so they share the single
process-wide SEC rate limiter (sec_rate_limit.py). Running them as separate
processes would give each its own budget and can exceed SEC's ~10 req/s limit.

    Stage A (collector): polls global getcurrent every ~1s → feed_YYYYMMDD.json
    Stage B (processor): scheduler + worker queue → process_items()
    Reconcile (Stage C): merges EDGAR daily index into feed JSON (downtime backstop)
        - 05:55 ET → yesterday (final index published ~10 PM prior evening)
        - 23:30 ET → today (after SEC publishes today's master.idx ~10 PM ET)

Usage:
    python manage.py run_sec_feed_poller
    python manage.py run_sec_feed_poller --no-collector      # processor only
    python manage.py run_sec_feed_poller --no-processor      # collector only
    python manage.py run_sec_feed_poller --no-reconcile      # disable scheduled reconcile
    python manage.py run_sec_feed_poller --reconcile-schedule "05:55:yesterday,23:30:today"

IMPORTANT: run exactly ONE instance (one container/replica). Two collectors
double SEC load and duplicate work.
"""

import argparse
import logging
import signal
import threading
from datetime import timedelta
from typing import Any, Dict, List, Tuple

from django.core.management.base import BaseCommand

from core.pipeline_logger import SEC_FEED_POLLER, start_pipeline
from sec_rss_parser.sec_feed_collector import (
    COLLECTOR_INTERVAL_SEC,
    run_collector_loop,
)
from sec_rss_parser.sec_feed_daily_store import (
    SEC_FEED_TZ,
    default_feed_dir,
    feed_now,
)
from sec_rss_parser.sec_feed_processor import (
    PROCESSOR_INTERVAL_SEC,
    run_processor_loop,
)

logger = logging.getLogger(__name__)

# SEC publishes master.{YYYYMMDD}.idx each evening (~10 PM ET). Morning slot
# backfills yesterday; late-evening slot backfills today after the index exists.
DEFAULT_RECONCILE_SCHEDULE = "23:30:today"


def _parse_reconcile_schedule(spec: str) -> List[Dict[str, Any]]:
    """
    Parse 'HH:MM:day,HH:MM:day,...' into reconcile slots.

    day is 'today' or 'yesterday' (feed calendar day in America/New_York).
    """
    slots = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        pieces = part.split(":")
        if len(pieces) == 2:
            hh, mm = pieces
            day_key = "today"
        elif len(pieces) == 3:
            hh, mm, day_key = pieces
        else:
            raise ValueError(
                f"Invalid reconcile slot {part!r}; use HH:MM:yesterday or HH:MM:today"
            )
        day_key = day_key.strip().lower()
        if day_key not in ("today", "yesterday"):
            raise ValueError(f"Invalid reconcile day {day_key!r}; use today or yesterday")
        slots.append({
            "hour": int(hh),
            "minute": int(mm),
            "day_key": day_key,
            "day_offset": -1 if day_key == "yesterday" else 0,
        })
    return sorted(slots, key=lambda s: (s["hour"], s["minute"]))


def _next_scheduled_slot(now, slots: List[Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
    """Next (datetime, slot) pair in SEC feed timezone."""
    candidates = []
    for slot in slots:
        t = now.replace(
            hour=slot["hour"],
            minute=slot["minute"],
            second=0,
            microsecond=0,
        )
        if t <= now:
            t += timedelta(days=1)
        candidates.append((t, slot))
    return min(candidates, key=lambda pair: pair[0])


def _run_reconcile_once(feed_dir, day_offset: int = 0, day_label: str = "today"):
    """Reconcile a feed day against EDGAR's daily index (day_offset: 0=today, -1=yesterday)."""
    from sec_rss_parser.fetch_sec_feed_by_deal_cik import build_tracked_ciks_map
    from sec_rss_parser.sec_daily_index import reconcile_into_feed

    start_pipeline(SEC_FEED_POLLER, doc_type="reconcile")
    tracked = build_tracked_ciks_map()
    day = feed_now() + timedelta(days=day_offset)
    added, parsed, new_accs = reconcile_into_feed(
        feed_dir, day=day, tracked_ciks=tracked
    )
    missed_preview = ""
    if new_accs:
        preview = ", ".join(new_accs[:10])
        if len(new_accs) > 10:
            preview += f" ... (+{len(new_accs) - 10} more)"
        missed_preview = f" | missed: {preview}"
    logger.info(
        "sec_feed reconcile: complete | target=%s (%s) | day=%s | tracked_ciks=%d | parsed=%d | added=%d%s",
        day_label,
        SEC_FEED_TZ,
        day.strftime("%Y-%m-%d"),
        len(tracked),
        parsed,
        added,
        missed_preview,
    )


def run_reconcile_scheduler(feed_dir, stop_event, slots: List[Dict[str, Any]]):
    start_pipeline(SEC_FEED_POLLER, doc_type="reconcile")
    schedule_desc = ",".join(
        f"{s['hour']:02d}:{s['minute']:02d}:{s['day_key']}" for s in slots
    )
    logger.info(
        "sec_feed reconcile scheduler: starting | schedule(%s)=%s",
        SEC_FEED_TZ,
        schedule_desc,
    )
    while not stop_event.is_set():
        now = feed_now()
        nxt, slot = _next_scheduled_slot(now, slots)
        wait_s = max(1.0, (nxt - now).total_seconds())
        logger.info(
            "sec_feed reconcile: next run at %s %s → %s (in %.0f min)",
            nxt.strftime("%Y-%m-%d %H:%M"),
            SEC_FEED_TZ,
            slot["day_key"],
            wait_s / 60.0,
        )
        if stop_event.wait(wait_s):
            break
        try:
            _run_reconcile_once(
                feed_dir,
                day_offset=slot["day_offset"],
                day_label=slot["day_key"],
            )
        except Exception:
            logger.exception("sec_feed reconcile: run failed")
    logger.info("sec_feed reconcile scheduler: stopped")


class Command(BaseCommand):
    help = "Run SEC feed collector + processor + reconcile in one process (shared rate limiter)."

    def add_arguments(self, parser):
        parser.add_argument("--feed-dir", default=default_feed_dir())
        parser.add_argument("--collector-interval", type=float, default=COLLECTOR_INTERVAL_SEC)
        parser.add_argument("--processor-interval", type=float, default=PROCESSOR_INTERVAL_SEC)
        parser.add_argument("--no-collector", action="store_true", help="Do not run Stage A")
        parser.add_argument("--no-processor", action="store_true", help="Do not run Stage B")
        parser.add_argument("--no-reconcile", action="store_true", help="Do not run scheduled reconcile")
        parser.add_argument(
            "--reconcile-schedule",
            default=DEFAULT_RECONCILE_SCHEDULE,
            help=(
                "Comma-separated HH:MM:today|yesterday in SEC feed TZ "
                "(default: %(default)s)"
            ),
        )
        # Back-compat alias
        parser.add_argument(
            "--reconcile-times",
            default=None,
            help=argparse.SUPPRESS,
        )

    def handle(self, *args, **opts):
        feed_dir = opts["feed_dir"]
        stop_event = threading.Event()

        def _shutdown(signum, _frame):
            self.stdout.write(self.style.WARNING(f"Received signal {signum}, shutting down..."))
            stop_event.set()

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        self.stdout.write(self.style.SUCCESS(f"SEC feed poller starting | feed_dir={feed_dir}"))

        threads = []
        if not opts["no_collector"]:
            threads.append(threading.Thread(
                target=run_collector_loop,
                kwargs={
                    "feed_dir": feed_dir,
                    "interval": opts["collector_interval"],
                    "stop_event": stop_event,
                },
                name="sec-feed-collector",
                daemon=True,
            ))

        if not opts["no_reconcile"]:
            schedule_spec = opts["reconcile_schedule"]
            if opts.get("reconcile_times"):
                # Legacy: times only → all target today
                legacy_times = opts["reconcile_times"].split(",")
                schedule_spec = ",".join(
                    f"{t.strip()}:today" for t in legacy_times if t.strip()
                )
            slots = _parse_reconcile_schedule(schedule_spec)
            threads.append(threading.Thread(
                target=run_reconcile_scheduler,
                kwargs={"feed_dir": feed_dir, "stop_event": stop_event, "slots": slots},
                name="sec-feed-reconcile",
                daemon=True,
            ))

        for t in threads:
            t.start()

        try:
            if not opts["no_processor"]:
                run_processor_loop(
                    feed_dir,
                    interval=opts["processor_interval"],
                    stop_event=stop_event,
                )
            else:
                while not stop_event.is_set():
                    stop_event.wait(1.0)
        finally:
            stop_event.set()
            for t in threads:
                t.join(timeout=10)

        self.stdout.write(self.style.SUCCESS("SEC feed poller stopped"))
