"""
SEC 8-K feed processor — Stage B sibling for form_type 8-K from daily JSON.

Scheduler (every ~10s): scan feed JSON for ALL 8-K / 8-K/A rows and enqueue.
Worker pool (default 20): each worker runs EightKFeedProcessor._process_single_item.

Does NOT call SEC getcurrent type=8-K — discovery comes from the collector JSON.
"""

import logging
import os
import threading
import time

from core.pipeline_logger import SEC_8K, start_pipeline
from sec_rss_parser.sec_8k_feed_work_queue import (
    EIGHT_K_FEED_WORKERS,
    Sec8KFeedWorkQueue,
)

logger = logging.getLogger(__name__)

EIGHT_K_PROCESSOR_INTERVAL_SEC = float(
    os.environ.get("SEC_FEED_8K_INTERVAL_SEC", "10.0")
)


def run_8k_feed_processor_loop(
    feed_dir, interval=None, stop_event=None, num_workers=None
):
    """
    Discovery scheduler + worker pool for 8-K items from the daily feed JSON.

    Intended to run as a daemon thread alongside the all-filings processor
    inside run_sec_feed_poller.
    """
    interval = EIGHT_K_PROCESSOR_INTERVAL_SEC if interval is None else interval
    stop_event = stop_event or threading.Event()
    workers_n = EIGHT_K_FEED_WORKERS if num_workers is None else num_workers

    work = Sec8KFeedWorkQueue(feed_dir, num_workers=workers_n)
    worker_threads = work.start_workers(stop_event)

    logger.info(
        "sec_8k_feed: starting | scheduler=%.1fs | workers=%d | feed_dir=%s",
        interval,
        workers_n,
        feed_dir,
    )

    try:
        while not stop_event.is_set():
            tick = time.monotonic()
            start_pipeline(SEC_8K, doc_type="8K-scheduler")
            try:
                work.discover_and_enqueue()
            except Exception:
                logger.exception("sec_8k_feed: discover/enqueue failed")
            elapsed = time.monotonic() - tick
            remaining = interval - elapsed
            if remaining > 0:
                stop_event.wait(remaining)
    finally:
        stop_event.set()
        for t in worker_threads:
            t.join(timeout=30)
        logger.info("sec_8k_feed: stopped | %s", work.stats())
