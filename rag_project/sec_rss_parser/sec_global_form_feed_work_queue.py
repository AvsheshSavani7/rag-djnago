"""
Work queue for S-4/F-4 items from the daily feed JSON.

Scheduler (every ~15s): scan feed JSON for GLOBAL_FORM_TYPES rows that are
NOT already owned by a tracked deal CIK → enqueue new accessions.

Worker pool: dequeue one item → process_one_global_form_item() (LLM match +
proxy pipeline).

Dedup layers (enqueue only if ALL clear):
  1. session_done              — finished this process
  2. reserved (in-memory queue) — already queued or in-flight in this process
  3. AccessionLookedUp         — terminal done in Mongo
  4. AccessionProcessingLock   — another worker/process currently holds the lock

SECFiling alone is NOT terminal here: a partial run may have created the filing
before LookedUp; skipping on SECFiling would freeze incomplete proxy work.

CIK-tracked accessions are filtered out in build_global_form_items_from_daily_feed
(not session_done — CIK/all-filings owns them; may become eligible if tracking ends).

If open_deals is empty, discover skips the tick (no enqueue / no LookedUp burn).
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

from django import db as django_db

from core.pipeline_logger import GLOBAL_FORM_FEED, start_pipeline
from sec_rss_parser.fetch_sec_global_form_type_feed import (
    _get_open_deals_for_matching,
    build_global_form_items_from_daily_feed,
    process_one_global_form_item,
)
from sec_rss_parser.models import (
    AccessionLookedUp,
    AccessionProcessingLock,
)

logger = logging.getLogger(__name__)

GLOBAL_FORM_FEED_WORKERS = int(os.environ.get("SEC_FEED_GLOBAL_FORM_WORKERS", "20"))


def _accession_preview(accessions, limit=10):
    if not accessions:
        return ""
    preview = ", ".join(accessions[:limit])
    if len(accessions) > limit:
        preview += f" ... (+{len(accessions) - limit} more)"
    return preview


class SecGlobalFormFeedWorkQueue:
    """Per-accession S-4/F-4 queue fed from daily JSON (not live getcurrent)."""

    def __init__(
        self,
        feed_dir: str,
        num_workers: int = GLOBAL_FORM_FEED_WORKERS,
        dry_run: bool = False,
    ):
        self.feed_dir = feed_dir
        self.num_workers = max(1, num_workers)
        self.dry_run = dry_run
        self._queue: queue.Queue = queue.Queue()
        self._state_lock = threading.Lock()
        self.session_done: Set[str] = set()
        self.reserved: Set[str] = set()
        self._waiting_accessions: Set[str] = set()
        self._open_deals: Optional[List[Dict[str, Any]]] = None
        self._open_deals_lock = threading.Lock()

    def refresh_open_deals(self) -> List[Dict[str, Any]]:
        deals = _get_open_deals_for_matching()
        with self._open_deals_lock:
            self._open_deals = deals
        return deals

    def get_open_deals(self) -> List[Dict[str, Any]]:
        with self._open_deals_lock:
            if self._open_deals is None:
                pass
            else:
                return self._open_deals
        return self.refresh_open_deals()

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
        if AccessionLookedUp.objects(accession_number=accession).first():
            return "looked_up"
        if AccessionProcessingLock.objects(accession_number=accession).first():
            return "processing_lock"
        return None

    def discover_and_enqueue(self) -> Tuple[int, int]:
        """Scan today's feed for S-4/F-4 rows; enqueue if not done/locked/queued."""
        enqueued = 0
        skipped_terminal = 0
        newly_enqueued: List[str] = []
        with self._state_lock:
            skip_snapshot = set(self.session_done)
            reserved_snapshot = set(self.reserved)

        open_deals = self.refresh_open_deals()
        if not open_deals:
            logger.warning(
                "sec_global_form_feed: open_deals empty — skipping discover tick "
                "(avoids mass LookedUp / wrong CIK ownership)"
            )
            return 0, 0

        items = build_global_form_items_from_daily_feed(
            feed_dir=self.feed_dir,
            day=None,
            skip_accessions=skip_snapshot,
            open_deals=open_deals,
        )
        for item in items:
            acc = (item.get("accession_number") or "").strip()
            if not acc or acc in skip_snapshot:
                continue
            if acc in reserved_snapshot:
                continue

            reason = self._is_already_terminal_or_locked(acc)
            if reason:
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
            "sec_global_form_feed: scheduler tick | waiting=%d | in_flight=%d | "
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
        start_pipeline(GLOBAL_FORM_FEED, doc_type="S4")
        logger.info("sec_global_form_feed: worker-%d started", worker_id)

        while not stop_event.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            acc = (item.get("accession_number") or "").strip()
            form_type = (item.get("form_type") or "S-4").strip().upper()
            label = item.pop("_feed_label", "today")
            with self._state_lock:
                self._waiting_accessions.discard(acc)
            try:
                django_db.close_old_connections()
                start_pipeline(
                    GLOBAL_FORM_FEED, accession=acc, doc_type=form_type
                )
                logger.info(
                    "sec_global_form_feed[worker-%d][%s]: processing %s (%s)",
                    worker_id, label, acc, form_type,
                )
                result = process_one_global_form_item(
                    item,
                    self.get_open_deals(),
                    dry_run=self.dry_run,
                )
                status = (result or {}).get("status")
                # Terminal for this process only on success / genuine no-match.
                # llm_error / failed / skipped_lock must remain re-enqueueable.
                if status in ("processed", "skipped_no_match") or (
                    acc and AccessionLookedUp.objects(accession_number=acc).first()
                ):
                    self._mark_session_done(acc)
            except Exception:
                logger.exception(
                    "sec_global_form_feed[worker-%d]: process failed for %s",
                    worker_id, acc,
                )
            finally:
                if acc:
                    self._release_reservation(acc)
                self._queue.task_done()

        logger.info("sec_global_form_feed: worker-%d stopped", worker_id)

    def start_workers(self, stop_event: threading.Event) -> list:
        threads = []
        for i in range(self.num_workers):
            t = threading.Thread(
                target=self.worker_loop,
                kwargs={"worker_id": i + 1, "stop_event": stop_event},
                name=f"sec-global-form-feed-worker-{i + 1}",
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
