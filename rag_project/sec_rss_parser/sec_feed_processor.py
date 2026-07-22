"""
SEC feed processor (production) — Stage B.

Scheduler (every ~15s): scans the daily feed JSON, filters by tracked CIK, and
enqueues new accessions onto an in-process work queue.

Worker pool (default 3): each worker calls process_items([item]) for one filing
at a time so multiple summaries can run in parallel without overlapping ticks.

Dedup / locking:
  - session_done + reserved (in-memory, per process)
  - AccessionLookedUp + AccessionProcessingLock (MongoDB, inside process_items)
"""

import logging
import os
import threading
import time

from core.pipeline_logger import SEC_FEED_POLLER, start_pipeline
from sec_rss_parser.sec_feed_work_queue import (
    DEAL_CIK_REFRESH_SEC,
    PROCESSOR_WORKERS,
    SecFeedWorkQueue,
)

logger = logging.getLogger(__name__)

PROCESSOR_INTERVAL_SEC = float(os.environ.get("SEC_FEED_PROCESSOR_INTERVAL_SEC", "15.0"))


def run_processor_loop(feed_dir, interval=None, stop_event=None, num_workers=None):
    """
    Run the discovery scheduler on the main thread and a pool of worker threads
    that drain the queue. Discovery fires every `interval` seconds regardless of
    how long workers are busy.
    """
    interval = PROCESSOR_INTERVAL_SEC if interval is None else interval
    stop_event = stop_event or threading.Event()
    workers_n = PROCESSOR_WORKERS if num_workers is None else num_workers

    work = SecFeedWorkQueue(feed_dir, num_workers=workers_n)
    work.maybe_refresh_ciks(force=True)
    worker_threads = work.start_workers(stop_event)

    logger.info(
        "sec_feed_processor: starting | scheduler=%.1fs | workers=%d | "
        "cik_refresh=%.1fs | feed_dir=%s",
        interval,
        workers_n,
        DEAL_CIK_REFRESH_SEC,
        feed_dir,
    )

    try:
        while not stop_event.is_set():
            tick = time.monotonic()
            start_pipeline(SEC_FEED_POLLER, doc_type="scheduler")
            try:
                work.discover_and_enqueue()
            except Exception:
                logger.exception("sec_feed_processor: discover/enqueue failed")
            elapsed = time.monotonic() - tick
            remaining = interval - elapsed
            if remaining > 0:
                stop_event.wait(remaining)
    finally:
        stop_event.set()
        for t in worker_threads:
            t.join(timeout=30)
        logger.info("sec_feed_processor: stopped | %s", work.stats())
