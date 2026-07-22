"""
EDGAR daily-index reconcile — completeness safety net.

The live `getcurrent` feed the collector polls is a rolling ~100-item window, so
filings that land while the collector is down are lost from it forever. The EDGAR
daily index is published each evening (~10 PM ET) as a static file with every
filing for that calendar day:

    https://www.sec.gov/Archives/edgar/daily-index/{YYYY}/QTR{q}/master.{YYYYMMDD}.idx

Reconcile downloads that file (one cheap GET), derives each accession, and merges
any accession missing from feed_YYYYMMDD.json. The processor then picks the added
filings up on its next tick exactly as if they had been caught live.
"""

import logging
import os
import re
from datetime import datetime

from sec_rss_parser.sec_feed_daily_store import append_feed_items, feed_now
from sec_rss_parser.sec_rate_limit import rate_limited_get
from sec_rss_parser.sec_feed_collector import DEFAULT_HEADERS, build_session
from sec_rss_parser.utils_8k import normalize_cik

logger = logging.getLogger(__name__)

SEC_BASE_URL = "https://www.sec.gov"
DAILY_INDEX_URL_TEMPLATE = (
    "https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{quarter}/master.{date}.idx"
)
# Row: CIK|Company Name|Form Type|Date Filed|Filename
_ACCESSION_RE = re.compile(r"(\d{10}-\d{2}-\d{6})")


def daily_index_url(day):
    quarter = (day.month - 1) // 3 + 1
    return DAILY_INDEX_URL_TEMPLATE.format(
        year=day.year,
        quarter=quarter,
        date=day.strftime("%Y%m%d"),
    )


def _index_url_from_accession(cik, accession):
    """Build the filing -index.htm URL the pipeline expects (not the raw .txt)."""
    acc_nodash = accession.replace("-", "")
    return (
        f"{SEC_BASE_URL}/Archives/edgar/data/{int(cik)}/{acc_nodash}/{accession}-index.htm"
    )


def parse_master_idx(text):
    """Parse master.idx pipe-delimited body into feed-record dicts."""
    records = []
    started = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if not started:
            # Data begins after the dashed separator line.
            if set(line) == {"-"}:
                started = True
            continue
        parts = line.split("|")
        if len(parts) != 5:
            continue
        cik_raw, company, form_type, date_filed, filename = parts
        m = _ACCESSION_RE.search(filename)
        if not m:
            continue
        accession = m.group(1)
        cik = normalize_cik(cik_raw)
        records.append({
            "accession_number": accession,
            "cik_number": cik,
            "form_type": (form_type or "").strip(),
            "company_name": (company or "").strip(),
            "date_filed": date_filed,
            "title": f"{form_type} - {company} ({cik}) (Filer)",
            "link": _index_url_from_accession(cik, accession),
            "guid": f"urn:tag:sec.gov,2008:accession-number={accession}",
        })
    return records


def fetch_daily_index(day, session=None):
    session = session or build_session()
    url = daily_index_url(day)
    try:
        resp = rate_limited_get(session, url, headers=DEFAULT_HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.warning("sec_daily_index: fetch failed for %s: %s", url, e)
        return None


def _accession_preview(accessions, limit=10):
    if not accessions:
        return ""
    preview = ", ".join(accessions[:limit])
    if len(accessions) > limit:
        preview += f" ... (+{len(accessions) - limit} more)"
    return preview


def reconcile_into_feed(feed_dir, day=None, session=None, tracked_ciks=None):
    """
    Merge any filing present in the daily index but missing from the feed JSON.

    If tracked_ciks is provided, only those CIKs are merged (keeps the feed lean);
    otherwise every filing for the day is merged and the processor filters later.

    Returns (added_count, parsed_count, new_accessions).
    """
    day = day or feed_now()
    text = fetch_daily_index(day, session=session)
    if not text:
        return 0, 0, []
    records = parse_master_idx(text)
    if tracked_ciks is not None:
        records = [r for r in records if r["cik_number"] in tracked_ciks]
    added, new_accs = append_feed_items(feed_dir, records, day=day)
    if added:
        logger.info(
            "sec_daily_index: reconcile %s | parsed=%d | added=%d missed | %s",
            day.strftime("%Y-%m-%d"),
            len(records),
            added,
            _accession_preview(new_accs),
        )
    else:
        logger.info(
            "sec_daily_index: reconcile %s | parsed=%d | added=0",
            day.strftime("%Y-%m-%d"),
            len(records),
        )
    return added, len(records), new_accs
