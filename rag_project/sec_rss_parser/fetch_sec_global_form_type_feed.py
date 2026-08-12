"""
Fetch SEC global Atom feed for S-4 and F-4 form types (no CIK filter).

Flow:
1. For each form type in GLOBAL_FORM_TYPES (S-4, F-4), fetch the global SEC feed.
2. Parse all items from the feed.
3. Group by accession_number (same accession may appear under different CIKs).
4. Filter:
   a. Skip accessions already in AccessionLookedUp.
   b. If ANY candidate CIK for an accession matches an open deal's cik /
      acquirer_cik, drop the whole accession (CIK flow owns it).
   c. Otherwise pick one row and continue.
5. For remaining items, ask LLM if the filing company matches any open deal
   by company name / alias.
6. If matched, inject deal_id + a discovery_note, then process through the
   same proxy pipeline used by fetch_sec_feed_by_deal_cik.py.

Legacy API ticks are coalesced via sec_global_form_work_queue (n8n path).
Production discovery now uses the collector daily JSON via
sec_global_form_feed_processor (same filters: LookedUp, drop if deal-CIK,
LLM match, proxy pipeline).

Usage:
    cd rag_project && python sec_rss_parser/fetch_sec_global_form_type_feed.py
"""

import os
import sys
import json
import re
import time
import logging
from datetime import datetime

import django

if __name__ == "__main__":
    _rag_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _rag_root not in sys.path:
        sys.path.insert(0, _rag_root)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rag_project.settings")
    django.setup()

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import openai
from mongoengine.queryset.visitor import Q

from document_processor.models import ProcessingJob
from sec_rss_parser.models import AccessionLookedUp, SECFilingSummary
from sec_rss_parser.services import DEAL_STATUS_OPEN_OR_UNKNOWN
from sec_rss_parser.utils_8k import (
    normalize_cik,
    extract_accession_from_guid,
    log_and_print,
)
from sec_rss_parser.sec_rate_limit import rate_limited_get
from sec_rss_parser.accession_lock import (
    acquire_accession_lock,
    mark_accession_processed,
    release_accession_lock,
)

from core.pipeline_logger import start_pipeline, GLOBAL_FORM_FEED

# Import pipeline functions from the existing CIK-based flow (no duplication)
from sec_rss_parser.fetch_sec_feed_by_deal_cik import (
    parse_atom_to_items,
    _cik_from_atom_title,
    _extract_cik_from_url,
    _accession_already_looked_up,
    fetch_and_parse_html_by_form_type,
    _ensure_sec_filing,
    _route_summarize_and_save,
    _handle_proxy_form_by_type,
    DEFAULT_HEADERS,
    PROXY_FORM_TYPES,
)

logger = logging.getLogger(__name__)

LOG_PREFIX = "global_form_type_feed: "

# Form types to monitor globally (subsidiary/new-company registrations for M&A)
GLOBAL_FORM_TYPES = ["S-4", "F-4", "S-4/A", "F-4/A"]

# Dry-run: process filings but send emails only via N8N_WEBHOOK_ONLY_ME (not all orgs)
GLOBAL_FORM_FEED_DRY_RUN_ENV = "GLOBAL_FORM_FEED_DRY_RUN"

GLOBAL_SEC_FEED_URL_TEMPLATE = (
    "https://www.sec.gov/cgi-bin/browse-edgar?"
    "action=getcurrent&CIK=&type={form_type}&company=&dateb=&"
    "owner=include&start=0&count=40&output=atom"
)

DISCOVERY_NOTE = (
    "Filing detected via global {form_type} monitoring — "
    "matched to this deal by company name / alias analysis "
    "(not via CIK tracking)."
)


# ---------------------------------------------------------------------------
# Deal data helpers
# ---------------------------------------------------------------------------

def _is_dry_run(dry_run=None):
    """True when dry_run param or GLOBAL_FORM_FEED_DRY_RUN env is set."""
    if dry_run is not None:
        return bool(dry_run)
    return os.environ.get(GLOBAL_FORM_FEED_DRY_RUN_ENV, "").lower() in (
        "1", "true", "yes"
    )


def _company_name_from_atom_title(title):
    """
    Extract company name from Atom title like:
    'S-4 - Kennedy-Wilson Holdings, Inc. (0001408100) (Filer)'
    """
    if not title or " - " not in title:
        return None
    after_form = title.split(" - ", 1)[1].strip()
    m = re.match(r"(.+?)\s*\(\d", after_form)
    if m:
        name = m.group(1).strip()
        return name or None
    return None


def _normalize_llm_deal_id_answer(answer):
    """Strip quotes / deal_id= prefix from LLM response."""
    if not answer:
        return ""
    text = answer.strip().strip('"').strip("'")
    if text.upper().startswith("DEAL_ID="):
        text = text.split("=", 1)[1].strip()
    return text


def _get_open_deals_for_matching():
    """
    Return a lightweight list of open deal dicts for LLM name-matching.
    Only fetches fields needed for matching.
    """
    deal_status_filter = (
        Q(deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN)
        | Q(deal_status=None)
        | Q(deal_status__exists=False)
    )
    deals = ProcessingJob.objects(deal_status_filter).only(
        "id", "cik", "acquirer_cik",
        "acquire_name", "target_name",
        "parent_aliases", "target_aliases",
    )
    result = []
    for d in deals:
        result.append({
            "deal_id": str(d.id),
            "cik": normalize_cik(d.cik) if d.cik else None,
            "acquirer_cik": normalize_cik(d.acquirer_cik) if d.acquirer_cik else None,
            "acquire_name": d.acquire_name or "",
            "target_name": d.target_name or "",
            "parent_aliases": list(d.parent_aliases or []),
            "target_aliases": list(d.target_aliases or []),
        })
    return result


def _cik_is_tracked_by_deal(cik_number, open_deals):
    """
    Returns True if this CIK is already tracked as target or acquirer on any open deal.
    Pass pre-fetched open_deals list to avoid repeated DB queries per item.
    """
    if not cik_number:
        return False
    cik_n = normalize_cik(cik_number)
    for d in open_deals:
        if (d.get("cik") and d["cik"] == cik_n) or \
           (d.get("acquirer_cik") and d["acquirer_cik"] == cik_n):
            return True
    return False


# ---------------------------------------------------------------------------
# LLM name-match
# ---------------------------------------------------------------------------

def _build_deals_text(open_deals):
    """Compact text representation of all deals for the LLM prompt."""
    lines = []
    for d in open_deals:
        names = []
        if d["target_name"]:
            names.append(d["target_name"])
        names.extend(d.get("target_aliases") or [])
        if d["acquire_name"]:
            names.append(d["acquire_name"])
        names.extend(d.get("parent_aliases") or [])
        all_names = ", ".join(n for n in names if n)
        lines.append(f'- deal_id={d["deal_id"]} | {all_names}')
    return "\n".join(lines)


def _llm_match_filing_to_deal(filing_company_name, filing_cik, form_type, open_deals):
    """
    Ask LLM whether the filing company matches any open deal.

    Returns (status, deal_id_or_None):
      - ("match", deal_id)     confident match
      - ("no_match", None)     LLM said NONE / invalid deal_id (safe to LookedUp)
      - ("error", None)        missing inputs, API failure, unexpected format
                               (do NOT mark LookedUp — allow retry)

    Uses a single LLM call with all deal names batched — no per-deal calls.
    """
    if not open_deals:
        return "error", None
    if not filing_company_name:
        return "error", None

    deals_text = _build_deals_text(open_deals)
    prompt = f"""You are a financial data assistant helping identify M&A deal matches.

A company just filed an SEC {form_type} form (a registration statement typically filed in M&A transactions).

Filing company name: {filing_company_name}
Filing CIK: {filing_cik or 'unknown'}

Open M&A deals (deal_id | company names and aliases):
{deals_text}

Task:
Does the filing company match any of the deals listed above as the TARGET or ACQUIRER (including aliases)?

Rules:
- Return ONLY the deal_id of the matching deal if confident (e.g. "6943af1457e6884d5fc7ea68")
- Return NONE if there is no clear match
- Do not guess; only return a match if the company name clearly corresponds to a deal party
- Partial name matches or subsidiary names that clearly belong to a deal party are acceptable

Answer (deal_id or NONE):"""

    try:
        client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY_SEC_FILING"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            temperature=0,
        )
        answer = _normalize_llm_deal_id_answer(
            response.choices[0].message.content or ""
        )
        log_and_print(
            f"{LOG_PREFIX} LLM match for '{filing_company_name}': {answer}"
        )
        if answer.upper() == "NONE" or not answer:
            return "no_match", None
        valid_deal_ids = {d["deal_id"] for d in open_deals}
        if len(answer) == 24 and all(c in "0123456789abcdefABCDEF" for c in answer):
            if answer not in valid_deal_ids:
                log_and_print(
                    f"{LOG_PREFIX} LLM returned deal_id not in open deals: {answer}",
                    "warning",
                )
                # Hallucinated / stale id — treat as no match (terminal).
                return "no_match", None
            return "match", answer
        log_and_print(
            f"{LOG_PREFIX} LLM returned unexpected format: {answer[:80]}", "warning"
        )
        return "error", None
    except Exception as e:
        log_and_print(
            f"{LOG_PREFIX} LLM match failed for '{filing_company_name}': {e}", "error"
        )
        return "error", None


# ---------------------------------------------------------------------------
# Feed fetch & parse
# ---------------------------------------------------------------------------

def _fetch_global_feed(form_type, session):
    """Fetch the global SEC Atom feed for a given form type."""
    url = GLOBAL_SEC_FEED_URL_TEMPLATE.format(form_type=form_type)
    try:
        resp = rate_limited_get(
            session, url, headers=DEFAULT_HEADERS, timeout=45
        )
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        log_and_print(
            f"{LOG_PREFIX} Failed to fetch global feed for {form_type}: {e}", "warning"
        )
        return None


def _parse_global_feed(rss_content, form_type):
    """Parse Atom feed content into item dicts, extracting CIK from title."""
    items = parse_atom_to_items(rss_content, cik_number=None, deal_id=None)
    result = []
    for item in items:
        cik = _cik_from_atom_title(item.get("title"))
        if cik:
            item["cik_number"] = normalize_cik(cik)
        company_name = _company_name_from_atom_title(item.get("title"))
        if company_name:
            item["company_name"] = company_name
        # Force form_type from feed level (Atom entries sometimes omit it)
        if not item.get("form_type"):
            item["form_type"] = form_type
        result.append(item)
    return result


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def _item_cik(item_data):
    """CIK from filing link or item fields."""
    link = item_data.get("link") or ""
    cik = _extract_cik_from_url(link) or item_data.get("cik_number")
    return normalize_cik(cik) if cik else None


def _group_items_by_accession(items):
    """
    Group feed rows by accession_number, preserving first-seen order.
    Returns (ordered_accession_list, {accession: [item, ...]}).
    """
    by_accession = {}
    order = []
    for item_data in items:
        form_type = (item_data.get("form_type") or "").strip().upper()
        if form_type not in GLOBAL_FORM_TYPES:
            continue
        if not item_data.get("link"):
            continue
        acc = item_data.get("accession_number") or extract_accession_from_guid(
            item_data.get("guid")
        )
        if not acc:
            continue
        if acc not in by_accession:
            by_accession[acc] = []
            order.append(acc)
        by_accession[acc].append(item_data)
    return order, by_accession


def _pick_item_for_accession(accession_number, candidates, open_deals):
    """
    If any candidate CIK matches an open deal, drop the whole accession
    (CIK-based flow owns it). Otherwise pick the first feed row.
    Returns (item_or_None, reason, tracked_cik_or_None).
    """
    for item_data in candidates:
        cik = _item_cik(item_data)
        if _cik_is_tracked_by_deal(cik, open_deals):
            log_and_print(
                f"{LOG_PREFIX} ⏭️ Accession {accession_number}: CIK {cik} "
                f"tracked in open deal — dropping all {len(candidates)} "
                f"row(s) (CIK flow will handle)"
            )
            return None, "cik_tracked", cik

    chosen = candidates[0]
    if len(candidates) > 1:
        log_and_print(
            f"{LOG_PREFIX} Accession {accession_number}: no tracked CIK "
            f"among {len(candidates)} rows — using first "
            f"(cik={_item_cik(chosen)})"
        )
    return chosen, "first", None


def _accession_has_summary(accession_number):
    """True if any SECFilingSummary already exists for this accession."""
    if not accession_number:
        return False
    return bool(
        SECFilingSummary.objects(accession_number=accession_number).first()
    )


def process_one_global_form_item(item_data, open_deals, dry_run=False):
    """
    Process a single S-4/F-4 item already selected by accession filters:
    LookedUp / CIK-tracked drops happen before enqueue.

    Returns dict with keys: status, accession, error (optional).
    status:
      processed | skipped_no_match | skipped_lock | llm_error | failed

    LookedUp rules (aligned with legacy behavior around emails):
      - genuine LLM no_match → LookedUp
      - full success → LookedUp
      - proxy fail after a summary already exists / was sent → LookedUp
        (no retry that would re-send summary email)
      - LLM/API errors, HTML fail, proxy fail with no summary → no LookedUp
        (retry allowed)
    """
    dry_run = _is_dry_run(dry_run)
    acc = item_data.get("accession_number") or extract_accession_from_guid(
        item_data.get("guid")
    )
    link = item_data.get("link")
    form_type = (item_data.get("form_type") or "").strip().upper()
    filing_cik = _item_cik(item_data)

    start_pipeline(GLOBAL_FORM_FEED, accession=acc, doc_type=form_type)

    if not open_deals:
        log_and_print(
            f"{LOG_PREFIX} ⏭️ Skipping {acc}: no open deals loaded "
            f"(will retry; not marking LookedUp)",
            "warning",
        )
        return {
            "status": "llm_error",
            "accession": acc,
            "error": "open_deals empty",
        }

    lock_owner = None
    if acc:
        lock_owner = acquire_accession_lock(acc, source="global_form_type_feed")
        if not lock_owner:
            log_and_print(
                f"{LOG_PREFIX} ⏭️ Skipping {acc} "
                f"(in-progress by another worker or already finalized)",
                "warning",
            )
            return {"status": "skipped_lock", "accession": acc}

    try:
        company_name = (
            item_data.get("company_name")
            or _company_name_from_atom_title(item_data.get("title"))
            or item_data.get("title")
        )
        if company_name and not item_data.get("company_name"):
            item_data["company_name"] = company_name

        match_status, matched_deal_id = _llm_match_filing_to_deal(
            filing_company_name=company_name,
            filing_cik=filing_cik,
            form_type=form_type,
            open_deals=open_deals,
        )
        if match_status == "error":
            log_and_print(
                f"{LOG_PREFIX} ⏭️ LLM match error for '{company_name}' "
                f"({form_type}) — will retry (not marking LookedUp)",
                "warning",
            )
            return {
                "status": "llm_error",
                "accession": acc,
                "error": "llm_match_error",
            }
        if match_status == "no_match" or not matched_deal_id:
            log_and_print(
                f"{LOG_PREFIX} ⏭️ No deal match for '{company_name}' ({form_type})"
            )
            if acc:
                mark_accession_processed(acc)
            return {"status": "skipped_no_match", "accession": acc}

        log_and_print(
            f"{LOG_PREFIX} ✅ Matched '{company_name}' ({form_type}) "
            f"→ deal_id={matched_deal_id}"
        )

        item_data["deal_id"] = matched_deal_id
        item_data["cik_number"] = filing_cik or item_data.get("cik_number")
        item_data["discovery_note"] = DISCOVERY_NOTE.format(form_type=form_type)
        if dry_run:
            item_data["email_dry_run"] = True

        html_data = fetch_and_parse_html_by_form_type(
            link, form_type_from_feed=form_type
        )
        if not html_data:
            log_and_print(
                f"{LOG_PREFIX} ❌ Failed to parse HTML for {acc}", "error"
            )
            return {
                "status": "failed",
                "accession": acc,
                "error": "Failed to parse HTML",
            }

        item_data.update(html_data)
        item_data["deal_id"] = matched_deal_id
        item_data["cik_number"] = filing_cik or html_data.get("cik_number")
        item_data["discovery_note"] = DISCOVERY_NOTE.format(form_type=form_type)
        if dry_run:
            item_data["email_dry_run"] = True

        filing, _ = _ensure_sec_filing(item_data)

        # Skip re-summarize/email if a summary already exists (avoids duplicate
        # production emails on retry after a prior partial success).
        summary_already = _accession_has_summary(acc)
        if summary_already:
            log_and_print(
                f"{LOG_PREFIX} Summary already exists for {acc} — "
                f"skipping re-summarize/email"
            )
        else:
            try:
                _route_summarize_and_save(item_data, html_data)
            except Exception as summary_e:
                log_and_print(
                    f"{LOG_PREFIX} ⚠️ Route summary failed (continuing): {summary_e}",
                    "warning",
                )
            summary_already = _accession_has_summary(acc)

        if dry_run:
            log_and_print(
                f"{LOG_PREFIX} dry_run=True — skipping proxy pipeline "
                f"(avoids org-wide proxy emails during testing)"
            )
        else:
            try:
                _handle_proxy_form_by_type(item_data, html_data, filing)
            except Exception as proxy_e:
                # Legacy: once a summary exists/was sent, mark LookedUp even if
                # proxy fails — do not retry and re-email.
                if summary_already or _accession_has_summary(acc):
                    log_and_print(
                        f"{LOG_PREFIX} ⚠️ Proxy pipeline failed: {proxy_e} — "
                        f"summary already exists, marking LookedUp (no retry)",
                        "warning",
                    )
                    if acc:
                        mark_accession_processed(acc)
                    return {
                        "status": "processed",
                        "accession": acc,
                        "error": str(proxy_e),
                    }
                log_and_print(
                    f"{LOG_PREFIX} ⚠️ Proxy pipeline failed: {proxy_e} — "
                    f"no summary yet, will retry (not marking LookedUp)",
                    "warning",
                )
                return {
                    "status": "failed",
                    "accession": acc,
                    "error": str(proxy_e),
                }

        # Proxy succeeded (or dry_run) — same as legacy mark-after-pipeline.
        if acc:
            mark_accession_processed(acc)
        return {"status": "processed", "accession": acc}

    except Exception as e:
        log_and_print(
            f"{LOG_PREFIX} ❌ Unexpected error for {acc}: {e}", "error"
        )
        return {"status": "failed", "accession": acc, "error": str(e)}
    finally:
        if acc and lock_owner:
            release_accession_lock(acc, lock_owner)


def build_global_form_items_from_daily_feed(
    feed_dir=None, day=None, skip_accessions=None, open_deals=None
):
    """
    Build S-4/F-4 work items from the collector's daily feed JSON.

    Applies the same accession filters as the legacy global getcurrent path:
    - group by accession
    - drop if any candidate CIK is on an open deal (CIK/all-filings owns it)
    - otherwise pick the first feed row

    LookedUp / lock checks happen at enqueue time.
    """
    from sec_rss_parser.sec_feed_daily_store import default_feed_dir, load_daily_feed
    from sec_rss_parser.sec_feed_item_utils import group_feed_records_by_accession

    skip = skip_accessions or set()
    feed_dir = feed_dir or default_feed_dir()
    _, feed_data = load_daily_feed(feed_dir, day=day)
    groups = group_feed_records_by_accession(feed_data.get("items") or {})

    if open_deals is None:
        open_deals = _get_open_deals_for_matching()

    # Without open deals we cannot safely filter cik_tracked vs LLM candidates.
    # Returning [] avoids enqueueing (and permanently burning) filings.
    if not open_deals:
        log_and_print(
            f"{LOG_PREFIX} Daily feed: open_deals empty — "
            f"skipping S-4/F-4 candidate build (will retry next tick)",
            "warning",
        )
        return []

    out = []
    skipped_cik = 0
    considered = 0
    for acc, records in groups.items():
        if not acc or acc in skip:
            continue
        candidates = [
            r for r in records
            if (r.get("form_type") or "").strip().upper() in GLOBAL_FORM_TYPES
            and r.get("link")
        ]
        if not candidates:
            continue
        considered += 1
        item_data, pick_reason, _tracked = _pick_item_for_accession(
            acc, candidates, open_deals
        )
        if pick_reason == "cik_tracked":
            skipped_cik += 1
            continue
        if not item_data:
            continue
        out.append({
            "accession_number": acc,
            "cik_number": _item_cik(item_data) or item_data.get("cik_number"),
            "form_type": (item_data.get("form_type") or "").strip().upper(),
            "title": item_data.get("title"),
            "link": item_data.get("link"),
            "guid": item_data.get("guid"),
            "company_name": item_data.get("company_name"),
            "filing_role": item_data.get("filing_role"),
        })

    log_and_print(
        f"{LOG_PREFIX} Daily feed: {len(out)} S-4/F-4 candidate(s) "
        f"(skipped_cik_tracked={skipped_cik}, "
        f"accessions_considered={considered})"
    )
    return out


def _process_global_items(items, open_deals, dry_run=False):
    """
    Filter and process global-feed items:
    1. Group by accession (same accession may have multiple CIK rows).
    2. Skip already-looked-up accessions.
    3. If any CIK for an accession is tracked by an open deal, drop it
       (do not mark looked_up — CIK flow owns it).
    4. Otherwise pick one row, LLM name-match, process via proxy pipeline.
    """
    processed = 0
    skipped_accession = 0
    skipped_cik = 0
    skipped_no_match = 0
    errors = []

    order, by_accession = _group_items_by_accession(items)
    log_and_print(
        f"{LOG_PREFIX} Unique accessions to consider: {len(order)} "
        f"(from {len(items)} feed rows)"
    )

    for idx, acc in enumerate(order):
        if idx > 0 and idx % 10 == 0:
            time.sleep(0.5)

        candidates = by_accession[acc]

        if _accession_already_looked_up(acc):
            log_and_print(
                f"{LOG_PREFIX} ⏭️ Skipping already looked up: {acc}"
            )
            skipped_accession += 1
            continue

        item_data, pick_reason, _tracked_cik = _pick_item_for_accession(
            acc, candidates, open_deals
        )
        if pick_reason == "cik_tracked":
            skipped_cik += 1
            continue

        result = process_one_global_form_item(
            item_data, open_deals, dry_run=dry_run
        )
        status = result.get("status")
        if status == "processed":
            processed += 1
        elif status == "skipped_no_match":
            skipped_no_match += 1
        elif status in ("failed", "llm_error"):
            errors.append({
                "accession": result.get("accession"),
                "message": result.get("error") or status,
            })

    return {
        "processed": processed,
        "skipped_accession": skipped_accession,
        "skipped_cik_tracked": skipped_cik,
        "skipped_no_llm_match": skipped_no_match,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_fetch_global_form_type_feed(dry_run=False):
    """
    Main entry point: fetch global S-4 / F-4 feeds, match against open deals,
    and process matched filings through the proxy pipeline.

    dry_run: when True, summary emails go to N8N_WEBHOOK_ONLY_ME only and
    proxy pipeline is skipped. Defaults to GLOBAL_FORM_FEED_DRY_RUN env var.
    """
    dry_run = _is_dry_run(dry_run)
    if dry_run:
        log_and_print(
            f"{LOG_PREFIX} dry_run=True — emails via N8N_WEBHOOK_ONLY_ME only"
        )

    session = requests.Session()
    retry = Retry(total=3, read=0, backoff_factor=1,
                  status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))

    # Load open deals once — shared across all form types and items
    log_and_print(f"{LOG_PREFIX} Loading open deals for matching...")
    open_deals = _get_open_deals_for_matching()
    log_and_print(f"{LOG_PREFIX} Loaded {len(open_deals)} open deals")

    if not open_deals:
        log_and_print(
            f"{LOG_PREFIX} No open deals — aborting run "
            f"(avoids mass LookedUp / wrong CIK ownership)",
            "warning",
        )
        return {
            "form_types": GLOBAL_FORM_TYPES,
            "items_count": 0,
            "processed": 0,
            "errors": [{"message": "open_deals empty"}],
        }

    all_items = []
    for form_type in GLOBAL_FORM_TYPES:
        log_and_print(f"{LOG_PREFIX} Fetching global feed for {form_type}...")
        raw = _fetch_global_feed(form_type, session)
        if not raw:
            log_and_print(
                f"{LOG_PREFIX} ⚠️ Empty feed response for {form_type}", "warning"
            )
            continue
        items = _parse_global_feed(raw, form_type)
        log_and_print(
            f"{LOG_PREFIX} Parsed {len(items)} items for {form_type}"
        )
        all_items.extend(items)
        time.sleep(1)  # SEC rate-limit courtesy pause between form types

    log_and_print(
        f"{LOG_PREFIX} Total items across all form types: {len(all_items)}"
    )

    if not all_items:
        return {
            "form_types": GLOBAL_FORM_TYPES,
            "items_count": 0,
            "processed": 0,
            "errors": [],
        }

    result = _process_global_items(all_items, open_deals, dry_run=dry_run)
    log_and_print(
        f"{LOG_PREFIX} Done — processed={result['processed']}, "
        f"dry_run={dry_run}, "
        f"skipped_accession={result['skipped_accession']}, "
        f"skipped_cik_tracked={result['skipped_cik_tracked']}, "
        f"skipped_no_llm_match={result['skipped_no_llm_match']}, "
        f"errors={len(result['errors'])}"
    )
    return {
        "form_types": GLOBAL_FORM_TYPES,
        "items_count": len(all_items),
        "dry_run": dry_run,
        **result,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    output = run_fetch_global_form_type_feed()
    print(json.dumps(output, indent=2, default=str))
