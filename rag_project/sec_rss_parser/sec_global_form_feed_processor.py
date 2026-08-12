"""
SEC S-4/F-4 feed processor — Stage B sibling from daily JSON.

Scheduler (every ~15s): scan feed JSON for S-4/F-4 (/A) rows that are not
owned by a tracked deal CIK, then enqueue.

Worker pool (default 20): each worker runs process_one_global_form_item
(LLM name-match → summary + proxy pipeline).

Does NOT call SEC getcurrent type=S-4/F-4 — discovery comes from the collector JSON.
"""

import logging
import os
import threading
import time

from core.pipeline_logger import GLOBAL_FORM_FEED, start_pipeline
from sec_rss_parser.sec_global_form_feed_work_queue import (
    GLOBAL_FORM_FEED_WORKERS,
    SecGlobalFormFeedWorkQueue,
)

logger = logging.getLogger(__name__)

GLOBAL_FORM_PROCESSOR_INTERVAL_SEC = float(
    os.environ.get("SEC_FEED_GLOBAL_FORM_INTERVAL_SEC", "15.0")
)


def run_global_form_feed_processor_loop(
    feed_dir, interval=None, stop_event=None, num_workers=None, dry_run=False
):
    """
    Discovery scheduler + worker pool for S-4/F-4 items from the daily feed JSON.

    Intended to run as a daemon thread alongside the all-filings / 8-K processors
    inside run_sec_feed_poller.
    """
    interval = (
        GLOBAL_FORM_PROCESSOR_INTERVAL_SEC if interval is None else interval
    )
    stop_event = stop_event or threading.Event()
    workers_n = (
        GLOBAL_FORM_FEED_WORKERS if num_workers is None else num_workers
    )

    work = SecGlobalFormFeedWorkQueue(
        feed_dir, num_workers=workers_n, dry_run=dry_run
    )
    worker_threads = work.start_workers(stop_event)

    logger.info(
        "sec_global_form_feed: starting | scheduler=%.1fs | workers=%d | "
        "feed_dir=%s | dry_run=%s",
        interval,
        workers_n,
        feed_dir,
        dry_run,
    )

    try:
        while not stop_event.is_set():
            tick = time.monotonic()
            start_pipeline(GLOBAL_FORM_FEED, doc_type="S4-scheduler")
            try:
                work.discover_and_enqueue()
            except Exception:
                logger.exception("sec_global_form_feed: discover/enqueue failed")
            elapsed = time.monotonic() - tick
            remaining = interval - elapsed
            if remaining > 0:
                stop_event.wait(remaining)
    finally:
        stop_event.set()
        for t in worker_threads:
            t.join(timeout=30)
        logger.info("sec_global_form_feed: stopped | %s", work.stats())
