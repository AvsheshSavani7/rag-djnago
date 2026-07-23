"""
SEC feed collector (production) — Stage A.

Polls the global SEC `getcurrent` Atom feed on a short interval, dedupes by
CIK + accession (composite key), and appends new filings to
sec_daily_feed/feed_YYYYMMDD.json.

The collector NEVER touches MongoDB and never runs the pipeline — it only grows
the daily feed cache. It is designed to run as a background thread inside the
same process as the processor (see management/commands/run_sec_feed_poller.py)
so both share the single process-wide SEC rate limiter in sec_rate_limit.py.
"""

import logging
import os
import re
import threading
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.pipeline_logger import SEC_FEED_POLLER, start_pipeline
from sec_rss_parser.sec_feed_daily_store import (
    append_feed_items,
    feed_path_for_date,
)
from sec_rss_parser.sec_rate_limit import rate_limited_get
from sec_rss_parser.sec_feed_item_utils import parse_filing_role
from sec_rss_parser.utils_8k import (
    SECRSSParser,
    extract_accession_from_guid,
    normalize_cik,
)

logger = logging.getLogger(__name__)

GLOBAL_ALL_FORMS_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar?"
    "action=getcurrent&CIK=&type=&company=&dateb=&owner=include&"
    "start=0&count=100&output=atom"
)

DEFAULT_HEADERS = {
    "User-Agent": "MNA-Finder/1.0 (https://teqnodux.com; contact: ashish.kachadiya@teqnodux.com)",
    "Accept": "application/atom+xml, application/xml, text/xml, */*",
    "Referer": "https://www.sec.gov/",
}

COLLECTOR_INTERVAL_SEC = float(os.environ.get("SEC_FEED_COLLECTOR_INTERVAL_SEC", "1.0"))


def _cik_from_atom_title(title):
    if not title:
        return None
    m = re.search(r"\((\d{10})\)", title)
    if m:
        return m.group(1)
    m = re.search(r"\((\d+)\)", title)
    if m:
        return m.group(1).zfill(10)
    return None


def _company_name_from_title(title):
    if not title or " - " not in title:
        return None
    after = title.split(" - ", 1)[1].strip()
    m = re.match(r"(.+?)\s*\(\d", after)
    return m.group(1).strip() if m else None


def parse_global_all_forms(raw_xml):
    """Parse the all-forms Atom feed into item dicts, filling cik/company from title."""
    parser = SECRSSParser(form_type="8-K")
    parsed = parser.parse_rss_content(raw_xml) if raw_xml else []
    items = []
    for it in parsed:
        item = dict(it)
        title = item.get("title") or ""
        cik = _cik_from_atom_title(title)
        if cik:
            item["cik_number"] = normalize_cik(cik)
        company = _company_name_from_title(title)
        if company:
            item["company_name"] = company
        acc = item.get("accession_number") or extract_accession_from_guid(item.get("guid"))
        if acc:
            item["accession_number"] = acc
        role = parse_filing_role(title)
        if role:
            item["filing_role"] = role
        items.append(item)
    return items


def build_session():
    session = requests.Session()
    # read=0: do not retry on ReadTimeout (avoids duplicate load when SEC is slow).
    retry = Retry(
        total=3,
        read=0,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def _fetch_global_feed(session):
    try:
        resp = rate_limited_get(
            session,
            GLOBAL_ALL_FORMS_URL,
            headers=DEFAULT_HEADERS,
            timeout=45,
        )
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.warning("sec_feed_collector: SEC fetch failed: %s", e)
        return None


def collect_once(session, feed_dir):
    """One poll → parse → append new filings into today's feed file.

    Every filing returned by getcurrent is written to the current (America/
    New_York) day's feed file, regardless of its own filing/acceptance date.
    Records already handled on a prior day are harmlessly re-listed here but are
    skipped downstream by the dedup layers (AccessionLookedUp / session_done),
    so nothing is processed twice and late-surfacing filings are never dropped.
    """
    raw = _fetch_global_feed(session)
    if not raw:
        return 0
    items = parse_global_all_forms(raw)
    added, new_accs = append_feed_items(feed_dir, items)
    if added:
        logger.info(
            "sec_feed_collector: +%d new filing(s) → %s | %s",
            added,
            os.path.basename(feed_path_for_date(feed_dir)),
            ", ".join(new_accs[:5]) + ("..." if len(new_accs) > 5 else ""),
        )
    return added


def run_collector_loop(feed_dir, interval=None, stop_event=None):
    """
    Poll on a fixed cadence until stop_event is set.

    A slow SEC response naturally slows the cadence (no request overlap because
    the loop is synchronous) but never fires a second request before the first
    returns.
    """
    interval = COLLECTOR_INTERVAL_SEC if interval is None else interval
    stop_event = stop_event or threading.Event()
    session = build_session()
    # Route this thread's logs to the dedicated poller log
    # (logs/sec_feed_poller/daily/<date>/sec_feed_poller.log). Only new-filing
    # ticks emit a line; empty polls stay silent.
    start_pipeline(SEC_FEED_POLLER, doc_type="collector")
    logger.info(
        "sec_feed_collector: starting | interval=%.1fs | feed_dir=%s",
        interval,
        feed_dir,
    )
    while not stop_event.is_set():
        tick = time.monotonic()
        try:
            collect_once(session, feed_dir)
        except Exception:
            logger.exception("sec_feed_collector: unexpected error in tick")
        elapsed = time.monotonic() - tick
        remaining = interval - elapsed
        if remaining > 0:
            stop_event.wait(remaining)
    logger.info("sec_feed_collector: stopped")
