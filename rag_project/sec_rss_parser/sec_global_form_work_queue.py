"""
In-process work queue for the global S-4 / F-4 form-type feed API.

n8n may trigger every ~15–20s; this module coalesces ticks so overlapping
calls do not spawn unbounded threads. A small worker pool (default 1, max 2
via SEC_GLOBAL_FORM_PROCESSOR_WORKERS) drains the queue and runs
run_fetch_global_form_type_feed().
"""

import logging
import os
import queue
import threading
from typing import Any, Dict

from django import db as django_db

from core.pipeline_logger import GLOBAL_FORM_FEED, start_pipeline

logger = logging.getLogger(__name__)

SEC_GLOBAL_FORM_PROCESSOR_WORKERS = max(
    1,
    min(2, int(os.environ.get("SEC_GLOBAL_FORM_PROCESSOR_WORKERS", "1"))),
)


class SecGlobalFormWorkQueue:
    """Singleton queue + worker pool for global S-4/F-4 feed processing."""

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self, num_workers: int = SEC_GLOBAL_FORM_PROCESSOR_WORKERS):
        self.num_workers = num_workers
        # At most one pending tick while workers are busy (coalesce n8n triggers).
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._workers_started = False
        self._start_lock = threading.Lock()
        self._active = 0
        self._state_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "SecGlobalFormWorkQueue":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(SEC_GLOBAL_FORM_PROCESSOR_WORKERS)
            return cls._instance

    def ensure_workers_started(self) -> None:
        with self._start_lock:
            if self._workers_started:
                return
            for worker_id in range(1, self.num_workers + 1):
                thread = threading.Thread(
                    target=self._worker_loop,
                    kwargs={"worker_id": worker_id},
                    name=f"sec-global-form-worker-{worker_id}",
                    daemon=True,
                )
                thread.start()
            self._workers_started = True
            logger.info(
                "sec_global_form_processor: worker pool started | workers=%d | queue_max=1",
                self.num_workers,
            )

    def enqueue_tick(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Queue one global form-type processor run.

        Returns status for the API:
          - queued: tick accepted (waiting or will run soon)
          - already_running: worker(s) busy and queue already has a pending tick
        """
        self.ensure_workers_started()
        payload = {"dry_run": bool(dry_run)}
        try:
            self._queue.put_nowait(payload)
            with self._state_lock:
                active = self._active
                pending = self._queue.qsize()
            return {
                "status": "queued",
                "active_workers": active,
                "pending_ticks": pending,
                "workers": self.num_workers,
                "dry_run": payload["dry_run"],
            }
        except queue.Full:
            with self._state_lock:
                active = self._active
            return {
                "status": "already_running",
                "active_workers": active,
                "pending_ticks": 1,
                "workers": self.num_workers,
                "dry_run": dry_run,
            }

    def stats(self) -> Dict[str, Any]:
        with self._state_lock:
            return {
                "workers": self.num_workers,
                "active_workers": self._active,
                "pending_ticks": self._queue.qsize(),
                "workers_started": self._workers_started,
            }

    def _worker_loop(self, worker_id: int) -> None:
        logger.info("sec_global_form_processor: worker-%d ready", worker_id)
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            dry_run = False
            if isinstance(item, dict):
                dry_run = bool(item.get("dry_run"))

            try:
                with self._state_lock:
                    self._active += 1
                django_db.close_old_connections()
                start_pipeline(GLOBAL_FORM_FEED, doc_type="S4")
                logger.info(
                    "sec_global_form_processor[worker-%d]: starting "
                    "run_fetch_global_form_type_feed dry_run=%s",
                    worker_id,
                    dry_run,
                )
                from sec_rss_parser.fetch_sec_global_form_type_feed import (
                    run_fetch_global_form_type_feed,
                )

                result = run_fetch_global_form_type_feed(dry_run=dry_run)
                logger.info(
                    "sec_global_form_processor[worker-%d]: completed | %s",
                    worker_id,
                    result,
                )
            except Exception:
                logger.exception(
                    "sec_global_form_processor[worker-%d]: "
                    "run_fetch_global_form_type_feed failed",
                    worker_id,
                )
            finally:
                with self._state_lock:
                    self._active -= 1
                self._queue.task_done()


def enqueue_global_form_processor_tick(
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Public entry for views / management commands."""
    return SecGlobalFormWorkQueue.get_instance().enqueue_tick(dry_run=dry_run)
