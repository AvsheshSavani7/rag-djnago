#!/usr/bin/env python3
"""
SEC feed processor (test script) — Stage B.

Reads today's feed JSON, loads deal CIKs from MongoDB (read-only), checks
AccessionLookedUp and AccessionProcessingLock (read-only), and LOGS what would
enter the real pipeline.
Does NOT call process_items, acquire/release locks, summary flow, or write to MongoDB.

Processor state (handled accessions + pipeline log) is stored in JSON only:
  sec_daily_feed/processor_state_YYYYMMDD.json

Usage:
    cd rag_project
    python sec_rss_parser/sec_feed_processor_test.py

    # Single evaluation pass then exit:
    python sec_rss_parser/sec_feed_processor_test.py --once

Env:
    SEC_FEED_PROCESSOR_INTERVAL_SEC   default 15.0
    SEC_FEED_DEAL_CIK_REFRESH_SEC     default 60.0
    SEC_FEED_DAILY_DIR                default sec_rss_parser/sec_daily_feed
    SEC_FEED_LOG_DIR                  default {SEC_FEED_DAILY_DIR}/logs
    SEC_FEED_MIDNIGHT_GRACE_MINUTES   default 30 (also scan yesterday 00:00–00:29 ET)
"""

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

import django

_rag_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _rag_root not in sys.path:
    sys.path.insert(0, _rag_root)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rag_project.settings")
django.setup()

from document_processor.models import ProcessingJob  # noqa: E402
from mongoengine.queryset.visitor import Q  # noqa: E402
from sec_rss_parser.models import AccessionLookedUp, AccessionProcessingLock  # noqa: E402
from sec_rss_parser.sec_feed_daily_store import (  # noqa: E402
    MIDNIGHT_GRACE_MINUTES,
    default_feed_dir,
    feed_path_for_date,
    get_feed_days_to_process,
    is_within_midnight_grace_window,
    load_daily_feed,
    load_processor_state,
    processor_state_path_for_date,
    record_processor_decisions,
)
from sec_rss_parser.sec_feed_test_logging import configure_test_logger, log_elapsed_ms  # noqa: E402
from sec_rss_parser.utils_8k import normalize_cik  # noqa: E402

LOG_PREFIX = "sec_feed_processor_test"

DEAL_STATUS_OPEN_OR_UNKNOWN = ["Open", "Unknown"]

# Keep in sync with fetch_sec_feed_by_deal_cik.py (avoid heavy import chain).
PROXY_FORM_TYPES = [
    "DEFM14A", "DEFM14C", "PREM14A", "PREM14C",
    "S-4", "F-4", "S-4/A", "F-4/A",
]
TEN_K_TEN_Q_FORM_TYPES = ["10-K", "10-Q", "10-K/A"]
EXCLUDED_FORM_TYPES = [
    "8-K", "4", "4/A", "144", "S-8", "S-8 POS",
    "SCHEDULE 13D/A", "SCHEDULE 13D", "SCHEDULE 13G", "SCHEDULE 13G/A",
]

# Option A: decisions that must stay eligible for retry are NOT persisted to
# handled_accessions. MongoDB (AccessionLookedUp + AccessionProcessingLock) is
# their only gate, so a failed pipeline run or a released lock is re-evaluated.
#   - WOULD_ENTER_PIPELINE: prod would acquire a lock + write AccessionLookedUp on
#     success; on failure neither is written, so it must retry next tick.
#   - SKIP_ALREADY_IN_MONGO_PROCESSING_LOCK: another worker holds a live lock; if
#     that worker fails the lock is released and this item must be retried.
NON_TERMINAL_ACTIONS = {
    "WOULD_ENTER_PIPELINE",
    "SKIP_ALREADY_IN_MONGO_PROCESSING_LOCK",
}

PROCESSOR_INTERVAL_SEC = float(os.environ.get("SEC_FEED_PROCESSOR_INTERVAL_SEC", "15.0"))
DEAL_CIK_REFRESH_SEC = float(os.environ.get("SEC_FEED_DEAL_CIK_REFRESH_SEC", "60.0"))

logger = logging.getLogger(__name__)
_stop = False


def get_open_or_unknown_deals():
    deal_status_filter = (
        Q(deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN)
        | Q(deal_status=None)
        | Q(deal_status__exists=False)
    )
    return list(
        ProcessingJob.objects(deal_status_filter).only(
            "id", "cik", "acquirer_cik", "target_name", "acquire_name",
        )
    )


def get_ciks_for_deal(deal):
    ciks = []
    for raw in (getattr(deal, "cik", None), getattr(deal, "acquirer_cik", None)):
        if raw and str(raw).strip():
            c = normalize_cik(raw)
            if c and c not in ciks:
                ciks.append(c)
    return ciks


def _route_label(form_type: str) -> str:
    ft = (form_type or "").strip().upper()
    if ft in PROXY_FORM_TYPES:
        return "proxy_pipeline"
    if ft in TEN_K_TEN_Q_FORM_TYPES:
        return "ten_k_ten_q_pipeline"
    return "other_sec_feed_pipeline"


def build_tracked_ciks_map() -> Dict[str, Dict[str, Any]]:
    """Read open deals from MongoDB and build CIK → deal metadata map."""
    tracked: Dict[str, Dict[str, Any]] = {}
    deals = get_open_or_unknown_deals()
    for deal in deals:
        deal_id = str(deal.id)
        target_cik = normalize_cik(getattr(deal, "cik", None) or "")
        acquirer_cik = normalize_cik(getattr(deal, "acquirer_cik", None) or "")
        target_name = getattr(deal, "target_name", None)
        acquirer_name = getattr(deal, "acquire_name", None)

        for cik in get_ciks_for_deal(deal):
            cik_n = normalize_cik(cik)
            if not cik_n:
                continue
            role = "acquirer" if cik_n == acquirer_cik and cik_n != target_cik else "target"
            if cik_n == target_cik:
                role = "target"
            elif cik_n == acquirer_cik:
                role = "acquirer"
            tracked[cik_n] = {
                "deal_id": deal_id,
                "role": role,
                "target_name": target_name,
                "acquirer_name": acquirer_name,
            }
    return tracked


def bulk_already_looked_up(accessions: List[str]) -> Set[str]:
    """Read-only Mongo check against AccessionLookedUp."""
    if not accessions:
        return set()
    found = AccessionLookedUp.objects(accession_number__in=accessions).only("accession_number")
    return {doc.accession_number for doc in found}


def bulk_active_processing_locks(accessions: List[str]) -> Dict[str, str]:
    """
    Read-only Mongo check for non-expired AccessionProcessingLock rows.

    Returns {accession_number: owner_id}. Does not acquire or release locks.
    """
    if not accessions:
        return {}
    now = datetime.utcnow()
    found = AccessionProcessingLock.objects(
        accession_number__in=accessions,
        expires_at__gt=now,
    ).only("accession_number", "owner_id")
    return {doc.accession_number: doc.owner_id for doc in found}


class ProcessorContext:
    def __init__(self, feed_dir: str):
        self.feed_dir = feed_dir
        self.tracked_ciks: Dict[str, Dict[str, Any]] = {}
        self.last_cik_refresh = 0.0
        # Option A: accessions already evaluated as non-terminal in THIS process
        # run (entered pipeline / held by a live lock). Not persisted to disk, so
        # a restart re-evaluates them (MongoDB is the real gate). Keeps us from
        # re-reading Mongo and re-logging the same item on every tick.
        self.session_non_terminal: Set[str] = set()

    def maybe_refresh_ciks(self, force: bool = False) -> int:
        now = time.monotonic()
        if force or (now - self.last_cik_refresh) >= DEAL_CIK_REFRESH_SEC:
            refresh_start = time.monotonic()
            self.tracked_ciks = build_tracked_ciks_map()
            self.last_cik_refresh = now
            log_elapsed_ms(
                logger,
                LOG_PREFIX,
                "mongo_refresh_deal_ciks",
                refresh_start,
                tracked_ciks=len(self.tracked_ciks),
            )
            logger.info(
                "%s refreshed deal CIKs from MongoDB | tracked_ciks=%d",
                LOG_PREFIX,
                len(self.tracked_ciks),
            )
        return len(self.tracked_ciks)


def evaluate_feed_for_day(
    ctx: ProcessorContext,
    feed_dir: str,
    day=None,
    feed_label: str = "today",
) -> Tuple[int, int, int]:
    """
    Evaluate new feed items for a single SEC feed day (ET) not yet in that day's processor_state.

    Returns (evaluated_count, would_pipeline_count, skipped_count).
    """
    day_start = time.monotonic()
    ctx.maybe_refresh_ciks()

    load_start = time.monotonic()
    feed_path, feed_data = load_daily_feed(feed_dir, day=day)
    state_path, state_data = load_processor_state(feed_dir, day=day)
    handled = set(state_data.get("handled_accessions") or [])
    log_elapsed_ms(
        logger,
        LOG_PREFIX,
        f"load_json[{feed_label}]",
        load_start,
        feed_items=len((feed_data.get("items") or {})),
    )

    items: Dict[str, Dict[str, Any]] = feed_data.get("items") or {}
    if not items:
        if feed_label == "yesterday_grace":
            logger.debug(
                "%s midnight grace: no items in yesterday feed (%s)",
                LOG_PREFIX,
                feed_path,
            )
        else:
            logger.info("%s no items in feed yet (%s)", LOG_PREFIX, feed_path)
        return 0, 0, 0

    pending_accs = [
        acc
        for acc in items
        if acc not in handled and acc not in ctx.session_non_terminal
    ]
    if not pending_accs:
        logger.debug(
            "%s [%s] feed has %d items, all already evaluated (%s)",
            LOG_PREFIX,
            feed_label,
            len(items),
            state_path,
        )
        return 0, 0, 0

    logger.info(
        "%s [%s] evaluating %d new accession(s) (feed total=%d, file=%s, tracked CIKs=%d)",
        LOG_PREFIX,
        feed_label,
        len(pending_accs),
        len(items),
        os.path.basename(feed_path),
        len(ctx.tracked_ciks),
    )

    # Bulk read Mongo for candidates that pass CIK + form filters first.
    filter_start = time.monotonic()
    candidates: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    decisions: List[Dict[str, Any]] = []
    skipped = 0

    for acc in pending_accs:
        item = items[acc]
        form_type = (item.get("form_type") or "").strip().upper()
        cik = normalize_cik(item.get("cik_number") or "")

        base = {
            "accession_number": acc,
            "form_type": form_type or None,
            "cik_number": cik or None,
            "title": item.get("title"),
            "link": item.get("link"),
        }

        if not cik or cik not in ctx.tracked_ciks:
            decisions.append({**base, "action": "SKIP_NO_CIK_MATCH"})
            skipped += 1
            continue

        if form_type in EXCLUDED_FORM_TYPES:
            decisions.append({
                **base,
                "action": "SKIP_EXCLUDED_FORM",
                "deal_id": ctx.tracked_ciks[cik]["deal_id"],
                "role": ctx.tracked_ciks[cik]["role"],
            })
            skipped += 1
            continue

        deal_meta = ctx.tracked_ciks[cik]
        candidates.append((acc, item, deal_meta))

    log_elapsed_ms(
        logger,
        LOG_PREFIX,
        f"memory_filter[{feed_label}]",
        filter_start,
        pending=len(pending_accs),
        candidates=len(candidates),
    )

    candidate_accs = [c[0] for c in candidates]
    mongo_start = time.monotonic()
    already_done = bulk_already_looked_up(candidate_accs)
    active_locks = bulk_active_processing_locks(candidate_accs)
    log_elapsed_ms(
        logger,
        LOG_PREFIX,
        f"mongo_read_lookup_lock[{feed_label}]",
        mongo_start,
        looked_up=len(already_done),
        active_locks=len(active_locks),
    )
    would_pipeline = 0

    for acc, item, deal_meta in candidates:
        form_type = (item.get("form_type") or "").strip().upper()
        cik = normalize_cik(item.get("cik_number") or "")
        route = _route_label(form_type)
        base = {
            "accession_number": acc,
            "form_type": form_type,
            "cik_number": cik,
            "title": item.get("title"),
            "link": item.get("link"),
            "deal_id": deal_meta["deal_id"],
            "role": deal_meta["role"],
        }

        if acc in already_done:
            decisions.append({**base, "action": "SKIP_ALREADY_IN_MONGO_ACCESSION_LOOKUP"})
            skipped += 1
            logger.info(
                "%s SKIP (Mongo AccessionLookedUp) | %s | form=%s | deal=%s",
                LOG_PREFIX,
                acc,
                form_type,
                deal_meta["deal_id"],
            )
            continue

        lock_owner = active_locks.get(acc)
        if lock_owner:
            decisions.append({
                **base,
                "action": "SKIP_ALREADY_IN_MONGO_PROCESSING_LOCK",
                "lock_owner_id": lock_owner,
            })
            skipped += 1
            logger.info(
                "%s SKIP (Mongo AccessionProcessingLock) | %s | form=%s | deal=%s | owner=%s",
                LOG_PREFIX,
                acc,
                form_type,
                deal_meta["deal_id"],
                lock_owner,
            )
            continue

        # Test mode: log only — would call process_items() in production.
        entry = {
            **base,
            "action": "WOULD_ENTER_PIPELINE",
            "route": route,
            "next_step": "process_items → fetch_and_parse_html_by_form_type → "
            + route,
        }
        if deal_meta["role"] == "acquirer":
            entry["note"] = (
                "Acquirer CIK — production flow also runs _llm_form_affects_deal gate"
            )
        decisions.append(entry)
        would_pipeline += 1

        logger.info(
            "%s >>> WOULD_ENTER_PIPELINE <<< | acc=%s | form=%s | cik=%s | deal=%s | role=%s | route=%s",
            LOG_PREFIX,
            acc,
            form_type,
            cik,
            deal_meta["deal_id"],
            deal_meta["role"],
            route,
        )
        logger.info(
            "%s     title: %s",
            LOG_PREFIX,
            (item.get("title") or "")[:120],
        )
        logger.info(
            "%s     link:  %s",
            LOG_PREFIX,
            item.get("link") or "",
        )
        if deal_meta["role"] == "acquirer":
            logger.info(
                "%s     note:  acquirer CIK — _llm_form_affects_deal would run before full processing",
                LOG_PREFIX,
            )

    # Option A: remember non-terminal accessions for THIS run only (not on disk),
    # so we don't re-read Mongo / re-log them every tick, but a restart re-checks.
    for d in decisions:
        if d.get("action") in NON_TERMINAL_ACTIONS:
            acc = (d.get("accession_number") or "").strip()
            if acc:
                ctx.session_non_terminal.add(acc)

    write_start = time.monotonic()
    record_processor_decisions(
        feed_dir,
        decisions,
        day=day,
        feed_label=feed_label,
        non_terminal_actions=NON_TERMINAL_ACTIONS,
    )
    log_elapsed_ms(
        logger,
        LOG_PREFIX,
        f"write_processor_state[{feed_label}]",
        write_start,
        decisions=len(decisions),
    )

    log_elapsed_ms(
        logger,
        LOG_PREFIX,
        f"evaluate_day_total[{feed_label}]",
        day_start,
        evaluated=len(decisions),
        would_pipeline=would_pipeline,
        skipped=skipped,
    )

    logger.info(
        "%s [%s] run complete | evaluated=%d | would_pipeline=%d | skipped=%d | state=%s",
        LOG_PREFIX,
        feed_label,
        len(decisions),
        would_pipeline,
        skipped,
        os.path.basename(processor_state_path_for_date(feed_dir, day)),
    )
    return len(decisions), would_pipeline, skipped


def evaluate_feed_items(
    ctx: ProcessorContext,
    feed_dir: str,
) -> Tuple[int, int, int]:
    """Evaluate today; during ET midnight grace, also drain yesterday's feed."""
    tick_start = time.monotonic()
    days = get_feed_days_to_process()
    if is_within_midnight_grace_window():
        logger.info(
            "%s midnight grace active — also scanning yesterday's feed (ET 00:00, first %d min)",
            LOG_PREFIX,
            MIDNIGHT_GRACE_MINUTES,
        )

    total_eval = 0
    total_would = 0
    total_skipped = 0
    for day, label in days:
        n, w, s = evaluate_feed_for_day(ctx, feed_dir, day=day, feed_label=label)
        total_eval += n
        total_would += w
        total_skipped += s

    if len(days) > 1:
        logger.info(
            "%s tick totals | evaluated=%d | would_pipeline=%d | skipped=%d",
            LOG_PREFIX,
            total_eval,
            total_would,
            total_skipped,
        )
    log_elapsed_ms(
        logger,
        LOG_PREFIX,
        "evaluate_tick_total",
        tick_start,
        evaluated=total_eval,
        would_pipeline=total_would,
        skipped=total_skipped,
        days_scanned=len(days),
    )
    return total_eval, total_would, total_skipped


def run_processor_loop(feed_dir: str, interval: float, run_once: bool = False) -> None:
    ctx = ProcessorContext(feed_dir)
    ctx.maybe_refresh_ciks(force=True)

    logger.info(
        "%s starting | interval=%.1fs | cik_refresh=%.1fs | feed_dir=%s",
        LOG_PREFIX,
        interval,
        DEAL_CIK_REFRESH_SEC,
        feed_dir,
    )
    logger.info("%s feed file today: %s", LOG_PREFIX, feed_path_for_date(feed_dir))
    logger.info(
        "%s state file today: %s",
        LOG_PREFIX,
        processor_state_path_for_date(feed_dir),
    )
    logger.info(
        "%s READ-ONLY MongoDB (AccessionLookedUp + AccessionProcessingLock) — "
        "no process_items, no lock acquire/release, no writes",
        LOG_PREFIX,
    )
    logger.info(
        "%s midnight grace: %d min ET (also scans yesterday's feed when active)",
        LOG_PREFIX,
        MIDNIGHT_GRACE_MINUTES,
    )

    while not _stop:
        tick = time.monotonic()
        try:
            evaluate_feed_items(ctx, feed_dir)
        except Exception:
            logger.exception("%s unexpected error in processor tick", LOG_PREFIX)
        if run_once:
            break
        loop_ms = (time.monotonic() - tick) * 1000.0
        logger.info(
            "%s timing | loop_wall | %.1f ms (incl. work, before sleep)",
            LOG_PREFIX,
            loop_ms,
        )
        elapsed = time.monotonic() - tick
        sleep_for = max(0.0, interval - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)

    logger.info("%s stopped", LOG_PREFIX)


def _handle_signal(signum, frame):
    global _stop
    logger.info("%s received signal %s, shutting down...", LOG_PREFIX, signum)
    _stop = True


def main():
    parser = argparse.ArgumentParser(description="SEC feed processor (test, read-only Mongo)")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one evaluation pass then exit",
    )
    parser.add_argument(
        "--feed-dir",
        default=default_feed_dir(),
        help="Directory for daily feed + processor state JSON",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=PROCESSOR_INTERVAL_SEC,
        help="Seconds between processor runs (default 15)",
    )
    parser.add_argument(
        "--refresh-ciks",
        action="store_true",
        help="Force refresh deal CIKs from Mongo on start",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    os.makedirs(args.feed_dir, exist_ok=True)
    log_path = configure_test_logger("processor", args.feed_dir)
    logger.info("%s log file: %s", LOG_PREFIX, log_path)
    run_processor_loop(args.feed_dir, args.interval, run_once=args.once)


if __name__ == "__main__":
    main()
