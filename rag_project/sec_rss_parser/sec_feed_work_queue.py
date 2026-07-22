"""
Work queue for SEC feed processor (Stage B).

Scheduler (every ~15s): scan feed JSON → filter → enqueue new accessions.
Worker pool: dequeue one item at a time → process_items([item]).

Dedup layers (no double-processing, no missed retries):
  1. session_done     — terminal for this process (excluded forms + AccessionLookedUp)
  2. reserved         — in queue or currently being processed by a worker
  3. AccessionLookedUp — checked at enqueue time (Mongo terminal)
  4. AccessionProcessingLock — inside process_items per accession
"""

import logging
import os
import queue
import threading
import time
from typing import Any, Dict, List, Set, Tuple

from django import db as django_db

from core.pipeline_logger import SEC_FEED_POLLER, start_pipeline
from sec_rss_parser.models import AccessionLookedUp
from sec_rss_parser.sec_feed_daily_store import (
    get_feed_days_to_process,
    is_within_midnight_grace_window,
)
from sec_rss_parser.fetch_sec_feed_by_deal_cik import (
    build_items_from_daily_feed,
    build_tracked_ciks_map,
    process_items,
)

logger = logging.getLogger(__name__)

PROCESSOR_WORKERS = int(os.environ.get("SEC_FEED_PROCESSOR_WORKERS", "3"))


def _accession_preview(accessions, limit=10):
    if not accessions:
        return ""
    preview = ", ".join(accessions[:limit])
    if len(accessions) > limit:
        preview += f" ... (+{len(accessions) - limit} more)"
    return preview

DEAL_CIK_REFRESH_SEC = float(os.environ.get("SEC_FEED_DEAL_CIK_REFRESH_SEC", "60.0"))


class SecFeedWorkQueue:
    def __init__(self, feed_dir: str, num_workers: int = PROCESSOR_WORKERS):
        self.feed_dir = feed_dir
        self.num_workers = max(1, num_workers)
        self._queue: queue.Queue = queue.Queue()
        self._state_lock = threading.Lock()
        self.session_done: Set[str] = set()
        # Accession in queue or being processed — prevents duplicate enqueue.
        self.reserved: Set[str] = set()
        # Subset of reserved still waiting in the queue (not yet picked up by a worker).
        self._waiting_accessions: Set[str] = set()
        self.tracked_ciks: Dict[str, str] = {}
        self.last_cik_refresh = 0.0

    def maybe_refresh_ciks(self, force: bool = False) -> int:
        now = time.monotonic()
        if force or (now - self.last_cik_refresh) >= DEAL_CIK_REFRESH_SEC:
            self.tracked_ciks = build_tracked_ciks_map()
            self.last_cik_refresh = now
            logger.info(
                "sec_feed_processor: refreshed tracked CIKs=%d",
                len(self.tracked_ciks),
            )
        return len(self.tracked_ciks)

    def _try_reserve(self, accession: str) -> bool:
        with self._state_lock:
            if accession in self.reserved:
                return False
            self.reserved.add(accession)
            return True

    def _release_reservation(self, accession: str) -> None:
        with self._state_lock:
            self.reserved.discard(accession)

    def _mark_session_done(self, accession: str) -> None:
        with self._state_lock:
            self.session_done.add(accession)

    def discover_and_enqueue(self) -> Tuple[int, int]:
        """Scan feed file(s), enqueue new items. Returns (enqueued, terminal_skips)."""
        self.maybe_refresh_ciks()
        if is_within_midnight_grace_window():
            logger.info(
                "sec_feed_processor: midnight grace — also scanning yesterday's feed"
            )

        enqueued = 0
        terminal_count = 0
        newly_enqueued: List[str] = []
        with self._state_lock:
            skip_snapshot = set(self.session_done)

        for day, label in get_feed_days_to_process():
            items, terminal_skips = build_items_from_daily_feed(
                self.feed_dir,
                self.tracked_ciks,
                day=day,
                skip_accessions=skip_snapshot,
            )
            for acc in terminal_skips:
                self._mark_session_done(acc)
                skip_snapshot.add(acc)
                terminal_count += 1

            for item in items:
                acc = (item.get("accession_number") or "").strip()
                if not acc or acc in skip_snapshot:
                    continue
                if AccessionLookedUp.objects(accession_number=acc).first():
                    self._mark_session_done(acc)
                    skip_snapshot.add(acc)
                    continue
                if not self._try_reserve(acc):
                    continue
                item["_feed_label"] = label
                with self._state_lock:
                    self._waiting_accessions.add(acc)
                self._queue.put(item)
                newly_enqueued.append(acc)
                enqueued += 1

        st = self.stats()
        pending_preview = (
            _accession_preview(st["waiting_accessions"])
            if st["waiting_accessions"]
            else "-"
        )
        new_preview = (
            _accession_preview(newly_enqueued) if newly_enqueued else "-"
        )
        logger.info(
            "sec_feed_processor: scheduler tick | waiting=%d | in_flight=%d | "
            "queue≈%d | session_done=%d | enqueued=+%d | pending: %s | new: %s",
            st["waiting"],
            st["in_flight"],
            st["queue_size"],
            st["session_done"],
            enqueued,
            pending_preview,
            new_preview,
        )
        return enqueued, terminal_count

    def worker_loop(self, worker_id: int, stop_event: threading.Event) -> None:
        """Drain the queue; one accession per process_items() call."""
        start_pipeline(SEC_FEED_POLLER, doc_type="processor")
        logger.info("sec_feed_processor: worker-%d started", worker_id)
        while not stop_event.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            acc = (item.get("accession_number") or "").strip()
            label = item.pop("_feed_label", "today")
            with self._state_lock:
                self._waiting_accessions.discard(acc)
            try:
                django_db.close_old_connections()
                start_pipeline(SEC_FEED_POLLER, doc_type="processor")
                logger.info(
                    "sec_feed_processor[worker-%d][%s]: processing %s",
                    worker_id,
                    label,
                    acc,
                )
                process_items([item])
                if acc and AccessionLookedUp.objects(accession_number=acc).first():
                    self._mark_session_done(acc)
            except Exception:
                logger.exception(
                    "sec_feed_processor[worker-%d]: process_items failed for %s",
                    worker_id,
                    acc,
                )
            finally:
                if acc:
                    self._release_reservation(acc)
                self._queue.task_done()

        logger.info("sec_feed_processor: worker-%d stopped", worker_id)

    def start_workers(self, stop_event: threading.Event) -> list:
        threads = []
        for i in range(self.num_workers):
            t = threading.Thread(
                target=self.worker_loop,
                kwargs={"worker_id": i + 1, "stop_event": stop_event},
                name=f"sec-feed-worker-{i + 1}",
                daemon=True,
            )
            t.start()
            threads.append(t)
        return threads

    def stats(self) -> Dict[str, Any]:
        with self._state_lock:
            waiting_accessions = sorted(self._waiting_accessions)
            waiting = len(waiting_accessions)
            reserved = len(self.reserved)
            return {
                "queue_size": self._queue.qsize(),
                "reserved": reserved,
                "waiting": waiting,
                "in_flight": max(0, reserved - waiting),
                "waiting_accessions": waiting_accessions,
                "session_done": len(self.session_done),
                "workers": self.num_workers,
            }
