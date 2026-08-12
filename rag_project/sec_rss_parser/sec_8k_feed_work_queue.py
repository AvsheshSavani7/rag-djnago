"""
Work queue for 8-K items from the daily feed JSON.

Scheduler (every ~10s): scan feed JSON for ALL form_type 8-K / 8-K/A rows
(no deal-CIK filter) → enqueue new accessions.

Worker pool: dequeue one item → EightKFeedProcessor._process_single_item().

Dedup layers (enqueue only if ALL clear):
  1. session_done              — finished this process
  2. reserved (in-memory queue) — already queued or in-flight in this process
  3. AccessionLookedUp         — terminal done in Mongo
  4. AccessionProcessingLock   — another worker/process currently holds the lock
  5. SECFiling (legacy)        — already materialized; backfill LookedUp and skip
     (lock is acquired again inside _process_single_item as belt-and-suspenders)
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Any, Dict, List, Set, Tuple

from django import db as django_db

from core.pipeline_logger import SEC_8K, start_pipeline
from sec_rss_parser.accession_lock import mark_accession_processed
from sec_rss_parser.fetch_sec_feed_by_deal_cik import build_8k_items_from_daily_feed
from sec_rss_parser.models import (
    AccessionLookedUp,
    AccessionProcessingLock,
    SECFiling,
)

logger = logging.getLogger(__name__)

EIGHT_K_FEED_WORKERS = int(os.environ.get("SEC_FEED_8K_WORKERS", "20"))


def _accession_preview(accessions, limit=10):
    if not accessions:
        return ""
    preview = ", ".join(accessions[:limit])
    if len(accessions) > limit:
        preview += f" ... (+{len(accessions) - limit} more)"
    return preview


class Sec8KFeedWorkQueue:
    """Per-accession 8-K queue fed from daily JSON (not live getcurrent)."""

    def __init__(self, feed_dir: str, num_workers: int = EIGHT_K_FEED_WORKERS):
        self.feed_dir = feed_dir
        self.num_workers = max(1, num_workers)
        self._queue: queue.Queue = queue.Queue()
        self._state_lock = threading.Lock()
        self.session_done: Set[str] = set()
        self.reserved: Set[str] = set()
        self._waiting_accessions: Set[str] = set()
        # One processor instance per worker is created in worker_loop.

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

    def _is_already_terminal_or_locked(self, accession: str) -> str | None:
        """
        Return a skip reason if this accession must not be enqueued, else None.

        Reasons: looked_up | processing_lock | sec_filing
        """
        if AccessionLookedUp.objects(accession_number=accession).first():
            return "looked_up"
        if AccessionProcessingLock.objects(accession_number=accession).first():
            return "processing_lock"
        # Legacy: filing already saved but LookedUp missing — match old 8-K filter.
        if SECFiling.objects(accession_number=accession).first():
            mark_accession_processed(accession)
            return "sec_filing"
        return None

    def discover_and_enqueue(self) -> Tuple[int, int]:
        """Scan today's feed for 8-K rows; enqueue only if not done/locked/queued."""
        enqueued = 0
        skipped_terminal = 0
        newly_enqueued: List[str] = []
        with self._state_lock:
            skip_snapshot = set(self.session_done)
            reserved_snapshot = set(self.reserved)

        items = build_8k_items_from_daily_feed(
            self.feed_dir,
            day=None,
            skip_accessions=skip_snapshot,
        )
        for item in items:
            acc = (item.get("accession_number") or "").strip()
            if not acc or acc in skip_snapshot:
                continue
            # Already in this process queue / in-flight.
            if acc in reserved_snapshot:
                continue

            reason = self._is_already_terminal_or_locked(acc)
            if reason:
                # LookedUp / SECFiling → permanent for this process.
                # processing_lock → do NOT session_done (retry after lock expires).
                if reason != "processing_lock":
                    self._mark_session_done(acc)
                    skip_snapshot.add(acc)
                skipped_terminal += 1
                continue

            if not self._try_reserve(acc):
                continue
            reserved_snapshot.add(acc)
            item["_feed_label"] = "today"
            with self._state_lock:
                self._waiting_accessions.add(acc)
            self._queue.put(item)
            newly_enqueued.append(acc)
            enqueued += 1

        st = self.stats()
        logger.info(
            "sec_8k_feed: scheduler tick | waiting=%d | in_flight=%d | "
            "queue≈%d | session_done=%d | enqueued=+%d | skipped_done_or_locked=%d | new: %s",
            st["waiting"],
            st["in_flight"],
            st["queue_size"],
            st["session_done"],
            enqueued,
            skipped_terminal,
            _accession_preview(newly_enqueued) if newly_enqueued else "-",
        )
        return enqueued, skipped_terminal

    def worker_loop(self, worker_id: int, stop_event: threading.Event) -> None:
        from sec_rss_parser.process_feed_8k import EightKFeedProcessor

        processor = EightKFeedProcessor()
        start_pipeline(SEC_8K, doc_type="8K")
        logger.info("sec_8k_feed: worker-%d started", worker_id)

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
                start_pipeline(SEC_8K, accession=acc, doc_type="8K")
                logger.info(
                    "sec_8k_feed[worker-%d][%s]: processing %s",
                    worker_id, label, acc,
                )
                processor._process_single_item(item)
                if acc and AccessionLookedUp.objects(accession_number=acc).first():
                    self._mark_session_done(acc)
            except Exception:
                logger.exception(
                    "sec_8k_feed[worker-%d]: _process_single_item failed for %s",
                    worker_id, acc,
                )
            finally:
                if acc:
                    self._release_reservation(acc)
                self._queue.task_done()

        logger.info("sec_8k_feed: worker-%d stopped", worker_id)

    def start_workers(self, stop_event: threading.Event) -> list:
        threads = []
        for i in range(self.num_workers):
            t = threading.Thread(
                target=self.worker_loop,
                kwargs={"worker_id": i + 1, "stop_event": stop_event},
                name=f"sec-8k-feed-worker-{i + 1}",
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
