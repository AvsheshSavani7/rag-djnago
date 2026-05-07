"""
Fetch SEC RSS/Atom feed per deal CIK, then process items.

Flow:
1. Fetch all deals from MongoDB where deal_status is Open or Unknown.
2. For each deal, iterate by target CIK and acquirer CIK (if both available, do both).
3. For each CIK, call SEC browse-edgar URL and parse the Atom feed.
4. Filter items: skip if accession already in AccessionLookedUp; find unique items by accession.
5. Iterate items; for each, branch by form_type:
   - 8-K: SKIPPED here — handled entirely by process_feed_8k.py (which has full EX-2.1 flow).
   - PROXY_FORM_TYPES: process via proxy_processor_helper.process_sec_document_for_filing_summary(), 
     which creates/updates SECFilingSummary.proxy directly (no ProxyDocument, no sync).
   - TEN_K_TEN_Q_FORM_TYPES: save to sec_filing_summary.ten_k_ten_q (minimal record).
   - Other: generate summary (no email), save to sec_filing_summary.other_filings.
   - change per second fetch 7 to 5

Usage:
    cd rag_project && python sec_rss_parser/fetch_sec_feed_by_deal_cik.py
"""

import os
import sys
import time
import json
import re
import tempfile
import logging
from datetime import datetime, timedelta
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import openai

import django

if __name__ == "__main__":
    _rag_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _rag_root not in sys.path:
        sys.path.insert(0, _rag_root)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rag_project.settings")
    django.setup()

from bson import ObjectId
from document_processor.models import ProcessingJob
from mongoengine.queryset.visitor import Q
from sec_rss_parser.utils_8k import (
    SECRSSParser,
    get_ticker_for_deal_and_cik,
    normalize_cik,
    extract_accession_from_guid,
    build_full_sec_url,
    find_file_by_type,
    parse_filing_date,
    log_and_print,
    send_webhook_notification,
)
from sec_rss_parser.services import (
    DEAL_STATUS_OPEN_OR_UNKNOWN,
    N8N_WEBHOOK_URL_8K_SUMMARY,
    send_summary_email_via_webhook,
)
from sec_rss_parser.sec_Last_Year import print_filings as fetch_sec_filings
from sec_rss_parser.email_templates import (
    generate_item_5_02_one_year_filings_email_html,
    generate_proxy_comparison_summary_email_html,
)
from sec_rss_parser.models import (
    AccessionLookedUp,
    SECFiling,
    SECFilingSummary,
)
from sec_rss_parser.Eight_k_summary import summarize_8k_filing
from sec_rss_parser.sec_summarizers.filing_router import route_and_summarize
from sec_rss_parser.proxy_processor_helper import process_sec_document_for_filing_summary
from sec_rss_parser.proxy_comparision.orchestrator import run_comparison
from sec_rss_parser.sec_rate_limit import rate_limited_get
from sec_rss_parser.accession_lock import (
    acquire_accession_lock,
    mark_accession_processed,
    release_accession_lock,
)

logger = logging.getLogger(__name__)


PROXY_FORM_TYPES = ["DEFM14A", "DEFM14C", "PREM14A",
                    "PREM14C", "S-4", "F-4", "S-4/A", "F-4/A"]
TEN_K_TEN_Q_FORM_TYPES = ["10-K", "10-Q", "10-K/A"]
EXCLUDED_FORM_TYPES = ["8-K", "4", "144", "S-8", "S-8 POS"]

LOG_PREFIX = "form by cik: "

SEC_FEED_URL_TEMPLATE = (
    "https://www.sec.gov/cgi-bin/browse-edgar?"
    "action=getcurrent&CIK={cik}&type=&company=&dateb=&owner=include&start=0&count=100&output=atom"
)
DEFAULT_HEADERS = {
    "User-Agent": "MNA-Finder/1.0 (https://teqnodux.com; contact: ashish.kachadiya@teqnodux.com)",
    "Accept": "application/atom+xml, application/xml, text/xml, */*",
    "Referer": "https://www.sec.gov/",
}
N8N_WEBHOOK_URL_8K_SUMMARY = os.environ.get(
    "N8N_WEBHOOK_URL_8K_SUMMARY",
    "https://n8n-xwx1.onrender.com/webhook/b3007d21-6845-47b5-aece-7b26583758bc",
)
SEC_BASE_URL = "https://www.sec.gov"
CIK_LENGTH = 10


def _extract_cik_from_url(url):
    """Extract filer CIK from SEC filing URL like /Archives/edgar/data/{CIK}/..."""
    if not url:
        return None
    m = re.search(r"/Archives/edgar/data/(\d+)", url)
    if m:
        return m.group(1).zfill(CIK_LENGTH)
    m = re.search(r"[?&]CIK=(\d+)", url, re.IGNORECASE)
    if m:
        return m.group(1).zfill(CIK_LENGTH)
    return None


def _extract_company_info(company_info):
    """Extract company_name and cik_number from SEC companyInfo div."""
    result = {"company_name": None, "cik_number": None}
    if not company_info:
        return result
    company_name_elem = company_info.find("span", class_="companyName")
    if company_name_elem:
        company_text = company_name_elem.get_text()
        m = re.match(r"^([^(]+)", company_text)
        if m:
            result["company_name"] = m.group(1).strip()
        cik_m = re.search(r"CIK[:\s]+(\d+)", company_text)
        if cik_m:
            result["cik_number"] = cik_m.group(1).zfill(CIK_LENGTH)
    return result


def _extract_filing_dates(soup):
    """Extract filing_date, acceptance_datetime_utc, period from infoHead/info divs."""
    filing_date = None
    acceptance_datetime_utc = None
    period = None
    for info_head in soup.find_all("div", class_="infoHead"):
        head_text = info_head.get_text()
        info_elem = info_head.find_next_sibling("div", class_="info")
        if not info_elem:
            continue
        text = info_elem.get_text().strip()
        if "Filing Date" in head_text:
            try:
                filing_date = datetime.strptime(text, "%Y-%m-%d")
            except ValueError:
                pass
        elif "Accepted" in head_text:
            acceptance_datetime_utc = text
        elif "Period of Report" in head_text:
            period = text
    return filing_date, acceptance_datetime_utc, period


# Document extensions to extract from SEC index (htm, html, xml e.g. Form 4)
_DOC_EXTENSIONS = (".htm", ".html", ".xml")


def _extract_xbrl_files_by_form_type(soup, form_type_from_feed):
    """
    Extract .htm/.html/.xml files from document table, filtered by form_type:
    - If form_type is 8-K: only include 8-K and EX-99.1 (not EX-2.1).
    - Otherwise: only include files where doc_type equals form_type.
    """
    xbrl_files = []
    table = soup.find("table", class_="tableFile")
    if not table:
        return xbrl_files
    form_type = (form_type_from_feed or "").strip().upper()
    rows = table.find_all("tr")[1:]
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        seq = cells[0].get_text().strip()
        description = cells[1].get_text().strip()
        doc_link = cells[2].find("a")
        doc_type = (cells[3].get_text().strip() or "").upper()
        size_text = cells[4].get_text().strip() if len(cells) > 4 else "0"
        if not doc_link:
            continue
        doc_url = doc_link.get("href", "")
        if not doc_url.startswith("http"):
            doc_url = urljoin(SEC_BASE_URL, doc_url)
        if not doc_url or not any(
            doc_url.lower().endswith(ext) for ext in _DOC_EXTENSIONS
        ):
            continue
        size = 0
        if size_text:
            sm = re.search(r"(\d+)", size_text.replace(",", ""))
            if sm:
                size = int(sm.group(1))
        file_data = {
            "sequence": int(seq) if seq.isdigit() else 0,
            "file": doc_link.get_text().strip(),
            "type": doc_type,
            "size": size,
            "description": description,
            "url": doc_url,
        }
        # Filter by form type
        if form_type == "8-K":
            # Only 8-K or EX-99.1 (not EX-2.1)
            if "EX-99.1" in doc_type or "EX-99.1" in description:
                xbrl_files.append(file_data)
            elif "8-K" in doc_type or "8-K" in description:
                xbrl_files.append(file_data)
        else:
            # Only file type equals form_type (e.g. DEFM14A, 10-K, 10-Q)
            if not form_type:
                xbrl_files.append(file_data)
            else:
                # Match doc_type to form_type (normalize for 10-K/10-Q vs DEFM14A etc.)
                doc_type_norm = doc_type.strip()
                form_norm = form_type.strip()
                if doc_type_norm == form_norm or form_norm in doc_type_norm or doc_type_norm in form_norm:
                    xbrl_files.append(file_data)
    return xbrl_files


def fetch_and_parse_html_by_form_type(html_url, form_type_from_feed=None):
    """
    Fetch filing index HTML and parse; form-type-specific file list.
    - If form_type is 8-K: xbrl_files only contains 8-K and EX-99.1.
    - Otherwise: xbrl_files only contains files where type equals form_type.
    Does not use services.py.
    """
    try:
        resp = rate_limited_get(
            requests,
            html_url,
            headers=DEFAULT_HEADERS,
            timeout=45,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        company_info = soup.find("div", class_="companyInfo")
        form_type = form_type_from_feed
        if not form_type:
            ident_info = company_info.find(
                "p", class_="identInfo") if company_info else None
            if ident_info:
                ident_text = ident_info.get_text()
                tm = re.search(r"Type[:\s]+([A-Z0-9\s\-/]+)", ident_text)
                if tm:
                    form_type = tm.group(1).strip()
            if not form_type and soup.find("title"):
                title_match = re.search(
                    r"([A-Z0-9\s\-/]+)\s+-\s+",
                    soup.find("title").get_text(),
                )
                if title_match:
                    form_type = title_match.group(
                        1).strip().split(" - ")[0].strip()
            form_type = (form_type or "").strip().upper().replace(" ", "")
        else:
            form_type = form_type.strip().upper()
        accession_number = None
        sec_num = soup.find("div", {"id": "secNum"})
        if sec_num:
            acc_m = re.search(r"(\d{10}-\d{2}-\d{6})", sec_num.get_text())
            if acc_m:
                accession_number = acc_m.group(1)
        if not accession_number:
            url_m = re.search(r"/(\d{10}-\d{2}-\d{6})", html_url)
            if url_m:
                accession_number = url_m.group(1)
        filing_date, acceptance_datetime_utc, period = _extract_filing_dates(
            soup)
        company_data = _extract_company_info(company_info)
        xbrl_files = _extract_xbrl_files_by_form_type(soup, form_type)
        if not form_type or not accession_number:
            log_and_print(
                f"{LOG_PREFIX} :fetch_and_parse_html_by_form_type: Missing form_type or accession: form_type={form_type}, accession={accession_number}",
                "warning",
            )
            return None
        # For 8-K build xbrl_files in same shape as downstream (type = 8-K or EX-99.1)
        if form_type == "8-K":
            xbrl_for_return = []
            for f in xbrl_files:
                doc_type = f.get("type", "")
                desc = f.get("description", "")
                document_type = None
                if "EX-99.1" in doc_type or "EX-99.1" in desc:
                    document_type = "EX-99.1"
                elif "8-K" in doc_type or "8-K" in desc:
                    document_type = "8-K"
                if document_type:
                    url_full = build_full_sec_url(f.get("url")) or f.get("url")
                    if url_full and "ix?doc=/" in url_full:
                        url_full = url_full.replace("ix?doc=/", "", 1)
                    xbrl_for_return.append({
                        "type": document_type,
                        "url": url_full,
                        "description": desc or doc_type,
                        "file": f.get("file", ""),
                        "size": f.get("size", 0),
                        "sequence": f.get("sequence", 0),
                    })
            has_ex99_1 = any("EX-99.1" in (e.get("type") or "")
                             for e in xbrl_for_return)
            has_8k_document = any("8-K" in (e.get("type") or "")
                                  for e in xbrl_for_return)
            return {
                "form_type": form_type,
                "accession_number": accession_number,
                "filing_date": filing_date,
                "acceptance_datetime_utc": acceptance_datetime_utc,
                "period": period,
                **company_data,
                "xbrl_files": xbrl_for_return,
                "has_ex99_1": has_ex99_1,
                "has_8k_document": has_8k_document,
            }
        # Non-8-K: xbrl_files already filtered to form_type
        return {
            "form_type": form_type,
            "accession_number": accession_number,
            "filing_date": filing_date,
            "acceptance_datetime_utc": acceptance_datetime_utc,
            "period": period,
            **company_data,
            "xbrl_files": xbrl_files,
        }
    except Exception as e:
        log_and_print(
            f"{LOG_PREFIX} :fetch_and_parse_html_by_form_type: Error in fetch_and_parse_html_by_form_type: {e}", "error")
        return None


def get_open_or_unknown_deals():
    deal_status_filter = (
        Q(deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN)
        | Q(deal_status=None)
        | Q(deal_status__exists=False)
    )
    return list(ProcessingJob.objects(deal_status_filter).only("id", "cik", "acquirer_cik"))


def get_ciks_for_deal(deal):
    ciks = []
    for raw in (getattr(deal, "cik", None), getattr(deal, "acquirer_cik", None)):
        if raw and str(raw).strip():
            c = normalize_cik(raw)
            if c and c not in ciks:
                ciks.append(c)
    return ciks


def fetch_feed_for_cik(cik, session, headers=None):
    url = SEC_FEED_URL_TEMPLATE.format(cik=cik)
    try:
        resp = rate_limited_get(
            session, url, headers=headers or DEFAULT_HEADERS, timeout=45
        )
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.warning("Fetch for CIK %s failed: %s", cik, e)
        return None


def parse_atom_to_items(rss_content, cik_number=None, deal_id=None):
    parser = SECRSSParser(form_type="8-K")
    items = parser.parse_rss_content(rss_content) if rss_content else []
    result = []
    for it in items:
        item = dict(it)
        if cik_number is not None:
            item["cik_number"] = cik_number
        if deal_id is not None:
            item["deal_id"] = deal_id
        result.append(item)
    return result


def _accession_already_looked_up(accession_number):
    if not accession_number:
        return False
    return AccessionLookedUp.objects(accession_number=accession_number).first() is not None


def _filter_unique_items(items):
    """Skip items whose accession is already in AccessionLookedUp; return unique by accession."""
    seen = set()
    unique = []
    for item in items:
        acc = item.get("accession_number") or extract_accession_from_guid(
            item.get("guid"))
        if not acc:
            unique.append(item)
            continue
        if _accession_already_looked_up(acc):
            log_and_print(
                f"{LOG_PREFIX} :_filter_unique_items: ⏭️ Skipping already looked up: {acc}")
            continue
        if acc in seen:
            continue
        seen.add(acc)
        unique.append(item)
    return unique


def _deal_id_for_cik(cik_number):
    if not cik_number:
        return None
    cik_n = normalize_cik(cik_number)
    deal_status_filter = (
        Q(deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN)
        | Q(deal_status=None)
        | Q(deal_status__exists=False)
    )
    deal = ProcessingJob.objects(
        Q(cik=cik_n) & deal_status_filter
    ).only("id").first()
    if deal:
        return str(deal.id)
    deal = ProcessingJob.objects(
        Q(acquirer_cik=cik_n) & deal_status_filter
    ).only("id").first()
    if deal:
        return str(deal.id)
    return None


def _llm_form_affects_deal(target_name, acquirer_name, sec_url, form_type=None):
    """
    Ask LLM whether this SEC form affects the deal. Returns True/False or None on error.
    Used only when the filing CIK is the acquirer.
    """
    try:
        import openai

        client = openai.OpenAI()
        form_label = form_type or "this SEC form"

        prompt = f"""You are evaluating whether an SEC filing by the acquirer/parent is materially related to a specific M&A deal.

Deal parties:
- Target: {target_name or 'Unknown'}
- Acquirer / Parent: {acquirer_name or 'Unknown'}

SEC filing URL: {sec_url}
Form type: {form_label}

Important context:
- This filing was made under the acquirer/parent's CIK.
- Many filings under the acquirer/parent's CIK are unrelated to the acquisition.
- Return YES only if the filing is specifically related to the acquisition transaction or provides a material transaction-related update.
-Return NO for routine or unrelated parent/acquirer filings, even if they were filed under the acquirer’s CIK.

Examples of YES:
- merger announcement
- acquisition update
- transaction financing
- shareholder approval related to the deal
- amendment to merger agreement
- closing announcement
- termination of the transaction
- regulatory approval or material transaction condition update

Examples of NO:
- insider trading Form 4 filings not tied to the transaction
- routine governance updates
- unrelated earnings filings
- unrelated securities offerings
- general business updates with no meaningful connection to the deal

Question:
Is this filing specifically related to, or does it meaningfully affect, the acquisition of {target_name or 'Unknown'} by {acquirer_name or 'Unknown'}?

Answer with exactly one word: YES or NO."""

        response = client.responses.create(
            model="gpt-5.2",
            tools=[{"type": "web_search"}],
            input=prompt,
            reasoning={"effort": "medium"}

        )

        # print(f"response: {response}")

        text = (response.output_text or "").strip().upper()

        if text == "YES":
            return True
        if text == "NO":
            return False

        log_and_print(
            f"{LOG_PREFIX} :_llm_form_affects_deal: Unexpected response: {text[:100]}",
            "warning",
        )
        return None

    except Exception as e:
        log_and_print(
            f"{LOG_PREFIX} :_llm_form_affects_deal: LLM call failed: {e}",
            "error",
        )
        logger.exception("_llm_form_affects_deal")
        return None


def _filing_date_for_summary(item_filing_date, result_filing_date):
    if item_filing_date:
        if isinstance(item_filing_date, datetime):
            return item_filing_date
        p = parse_filing_date(item_filing_date)
        if p:
            return p
    if result_filing_date:
        p = parse_filing_date(result_filing_date)
        if p:
            return p
    return None


# SECFiling model limits (from sec_rss_parser.models)
SEC_FILING_DESCRIPTION_MAX_LENGTH = 50
SEC_FILING_GUID_MAX_LENGTH = 1000


def _ensure_sec_filing(item_data):
    """Create SECFiling if not exists; return (filing_or_none, created)."""
    acc = item_data.get("accession_number")
    if not acc:
        return None, False
    existing = SECFiling.objects(accession_number=acc).first()
    if existing:
        return existing, False
    allowed = {
        "title", "link", "guid", "description", "pubDate",
        "company_name", "form_type", "filing_date", "cik_number",
        "accession_number", "file_number", "acceptance_datetime_utc",
        "period", "fiscal_year_end", "assigned_sic", "xbrl_files",
        "has_htm_files", "processed", "following", "following_status",
    }
    payload = {k: item_data.get(k)
               for k in allowed if item_data.get(k) is not None}
    if not payload.get("title"):
        payload["title"] = item_data.get(
            "title") or f"{item_data.get('form_type', '')} - {acc}"
    if not payload.get("link"):
        payload["link"] = item_data.get("link") or ""
    # guid must be a URL (http/https); Atom often has URN like urn:tag:sec.gov,2008:accession-number=...
    guid_raw = payload.get("guid") or item_data.get("guid") or payload["link"]
    if guid_raw.startswith("urn:"):
        payload["guid"] = payload["link"]
    else:
        payload["guid"] = guid_raw[:SEC_FILING_GUID_MAX_LENGTH] if guid_raw else payload["link"]
    # description max 50 chars
    desc_raw = payload.get("description") or item_data.get("description") or ""
    payload["description"] = (
        desc_raw[:SEC_FILING_DESCRIPTION_MAX_LENGTH]) if desc_raw else ""
    if not payload.get("company_name"):
        payload["company_name"] = item_data.get("company_name") or "Unknown"
    if not payload.get("form_type"):
        payload["form_type"] = item_data.get("form_type") or "8-K"
    if not payload.get("cik_number"):
        payload["cik_number"] = item_data.get("cik_number") or ""
    if not payload.get("accession_number"):
        payload["accession_number"] = acc
    try:
        filing = SECFiling(**payload)
        filing.save()
        return filing, True
    except Exception as e:
        log_and_print(
            f"{LOG_PREFIX} :_ensure_sec_filing: Failed to create SECFiling for {acc}: {e}", "error")
        return None, False


# Form types that always use _process_proxy_item (no comparison lookup).
PROXY_FORM_ALWAYS_STANDALONE = ["S-4", "F-4"]

# For comparison path: form_type -> list of previous form_types to look up (by deal_id).
PROXY_FORM_PREVIOUS_LOOKUP = {
    "PREM14A": ["PREM14A"],
    "PREM14C": ["PREM14C"],
    "DEFM14A": ["PREM14A", "S-4", "S-4/A", "F-4", "F-4/A"],
    "DEFM14C": ["PREM14C"],
    "S-4/A": ["S-4", "S-4/A"],
    "F-4/A": ["F-4", "F-4/A"],
}


def _get_previous_proxy_summary(deal_id, form_types, exclude_accession_number):
    """Return the latest SECFilingSummary for this deal_id and form_type in list, excluding current accession."""
    if not deal_id or not form_types:
        return None
    q = SECFilingSummary.objects(deal_id=deal_id, form_type__in=form_types)
    if exclude_accession_number:
        q = q.filter(accession_number__ne=exclude_accession_number)
    return q.order_by("-created_at").first()


def _get_current_proxy_summary(deal_id, accession_number, form_type):
    """Return SECFilingSummary for the current filing (same deal_id and accession_number)."""
    if not accession_number:
        return None
    doc = SECFilingSummary.objects(
        deal_id=deal_id, accession_number=accession_number
    ).first()
    if doc:
        return doc
    return SECFilingSummary.objects(
        accession_number=accession_number, form_type=form_type
    ).first()


def _sec_filing_summary_to_comparison_record(doc):
    """Convert SECFilingSummary doc (or dict) to record shape for run_comparison."""
    if doc is None:
        return None
    if hasattr(doc, "id"):
        _id = str(doc.id)
        sec_document_url = doc.sec_document_url
        deal_id = doc.deal_id
        form_type = doc.form_type
        accession_number = doc.accession_number
        cik_number = doc.cik_number
        # Preserve proxy nested dict if present (orchestrator uses it for cache)
        proxy = getattr(doc, "proxy", None)
    else:
        _id = str(doc.get("_id", doc.get("id", "")))
        sec_document_url = doc.get("sec_document_url", "")
        deal_id = doc.get("deal_id")
        form_type = doc.get("form_type", "")
        accession_number = doc.get("accession_number")
        cik_number = doc.get("cik_number")
        proxy = doc.get("proxy")
    out = {
        "_id": _id,
        "sec_document_url": sec_document_url or "",
        "deal_id": deal_id,
        "form_type": form_type or "PROXY",
        "accession_number": accession_number,
        "cik_number": cik_number,
    }
    if proxy is not None:
        out["proxy"] = proxy
    return out


def _send_proxy_comparison_email(
    company_name,
    form_type,
    deal_id,
    cik_number,
    result_from_orchestrator,
):
    """Send proxy comparison summary email after run_comparison returns. Uses email_templates."""
    try:
        ticker = get_ticker_for_deal_and_cik(deal_id, cik_number)
        subject, html = generate_proxy_comparison_summary_email_html(
            company_name=company_name,
            form_type=form_type,
            ticker=ticker or "",
            label=company_name or ticker,
            deal_id=deal_id,
            cik_number=cik_number,
            past_record_id=result_from_orchestrator.get("past_id"),
            latest_record_id=result_from_orchestrator.get("latest_id"),
            change_docx_url=result_from_orchestrator.get("change_docx_url"),
            change_txt_url=result_from_orchestrator.get("change_txt_url"),
            changes_json_url=result_from_orchestrator.get("changes_json_url"),
            tier1_changes=result_from_orchestrator.get("tier1_changes"),
            tier2_changes=result_from_orchestrator.get("tier2_changes"),
        )
        payload = {
            "subject": subject,
            "html": html,
            "company_name": company_name,
            "form_type": form_type,
            "email_type": "proxy_comparison_summary",
            "change_docx_url": result_from_orchestrator.get("change_docx_url"),
            "change_txt_url": result_from_orchestrator.get("change_txt_url"),
            "changes_json_url": result_from_orchestrator.get("changes_json_url"),
            "deal_id": deal_id,
            "cik_number": cik_number,
            "past_record_id": result_from_orchestrator.get("past_id"),
            "latest_record_id": result_from_orchestrator.get("deal_id"),
        }
        send_webhook_notification(
            N8N_WEBHOOK_URL_8K_SUMMARY, payload, "proxy comparison summary email"
        )
        log_and_print(
            f"{LOG_PREFIX} :_send_proxy_comparison_email: ✅ Proxy comparison email sent for {form_type}"
        )
    except Exception as e:
        log_and_print(
            f"{LOG_PREFIX} :_send_proxy_comparison_email: ❌ Failed to send proxy comparison email: {e}",
            "error",
        )
        logger.exception(
            f"{LOG_PREFIX} :_send_proxy_comparison_email: error={e}"
        )


def _handle_proxy_form_by_type(item_data, html_data, filing):
    """
    Route proxy forms: run comparison (orchestrator + email) when a previous filing exists,
    otherwise run _process_proxy_item (which sends email in its own flow).

    - S-4, F-4: always _process_proxy_item.
    - PREM14A (lookup previous PREM14A), PREM14C (lookup previous PREM14C), DEFM14A, DEFM14C,
      S-4/A, F-4/A: lookup previous by deal_id + form_type list; if found -> run_comparison
      then send proxy comparison email; if not found -> _process_proxy_item.
    """
    form_type = (
        (item_data.get("form_type") or html_data.get(
            "form_type") or "").strip().upper()
    )
    if form_type not in PROXY_FORM_TYPES:
        logger.info(
            f"{LOG_PREFIX} :_handle_proxy_form_by_type: form_type={form_type} not in PROXY_FORM_TYPES"
        )
        return

    deal_id = item_data.get("deal_id") or _deal_id_for_cik(
        item_data.get("cik_number") or html_data.get("cik_number")
    )
    accession_number = (
        item_data.get("accession_number") or html_data.get("accession_number")
    )
    company_name = (
        html_data.get("company_name") or item_data.get("company_name") or ""
    )
    cik_number = item_data.get("cik_number") or html_data.get("cik_number")

    # Always standalone: no previous lookup, just process proxy item (email sent in that flow).
    if form_type in PROXY_FORM_ALWAYS_STANDALONE:
        logger.info(
            f"{LOG_PREFIX} :_handle_proxy_form_by_type: form_type={form_type} using _process_proxy_item (standalone)"
        )
        _process_proxy_item(item_data, html_data, filing)
        return

    # Comparison path: need previous form_type list for this form_type.
    previous_form_types = PROXY_FORM_PREVIOUS_LOOKUP.get(form_type)
    if not previous_form_types:
        logger.info(
            f"{LOG_PREFIX} :_handle_proxy_form_by_type: form_type={form_type} no lookup map, using _process_proxy_item"
        )
        _process_proxy_item(item_data, html_data, filing)
        return

    previous_doc = _get_previous_proxy_summary(
        deal_id, previous_form_types, exclude_accession_number=accession_number
    )
    if not previous_doc:
        logger.info(
            f"{LOG_PREFIX} :_handle_proxy_form_by_type: form_type={form_type} no previous filing found, using _process_proxy_item"
        )
        _process_proxy_item(item_data, html_data, filing)
        return

    # Both current and past are in DB: fetch current by deal_id + accession_number.
    current_doc = _get_current_proxy_summary(
        deal_id, accession_number, form_type)
    if not current_doc:
        log_and_print(
            f"{LOG_PREFIX} :_handle_proxy_form_by_type: current summary not found in DB (deal_id={deal_id}, accession={accession_number}), falling back to _process_proxy_item",
            "warning",
        )
        _process_proxy_item(item_data, html_data, filing)
        return

    # Convert DB docs to record dicts (same shape as test_runner / run_comparison expects).
    current_record = _sec_filing_summary_to_comparison_record(current_doc)
    past_record = _sec_filing_summary_to_comparison_record(previous_doc)
    if not current_record or not past_record:
        _process_proxy_item(item_data, html_data, filing)
        return

    logger.info(
        f"{LOG_PREFIX} :_handle_proxy_form_by_type: form_type={form_type} running comparison (current={current_record.get('_id')}, past={past_record.get('_id')})"
    )
    try:
        result = run_comparison(
            latest_doc_record=current_record,
            past_doc_record=past_record,
        )
        if result and result.get("status") == "complete":
            _send_proxy_comparison_email(
                company_name=company_name,
                form_type=form_type,
                deal_id=deal_id,
                cik_number=cik_number,
                result_from_orchestrator={
                    **result,
                    "past_id": past_record.get("_id"),
                    "latest_id": current_record.get("_id"),
                },
            )
        else:
            log_and_print(
                f"{LOG_PREFIX} :_handle_proxy_form_by_type: run_comparison did not return status=complete: {result}",
                "warning",
            )
    except Exception as e:
        log_and_print(
            f"{LOG_PREFIX} :_handle_proxy_form_by_type: ❌ run_comparison failed: {e}",
            "error",
        )
        logger.exception(
            f"{LOG_PREFIX} :_handle_proxy_form_by_type: run_comparison error={e}"
        )
        # _process_proxy_item(item_data, html_data, filing)


def _process_proxy_item(item_data, html_data, filing):
    """Process proxy form: start proxy document processing (summary + email happen async)."""
    logger.info(f"{LOG_PREFIX} :_process_proxy_item: item_data={item_data}")
    logger.info(f"{LOG_PREFIX} :_process_proxy_item: html_data={html_data}")
    logger.info(f"{LOG_PREFIX} :_process_proxy_item: filing={filing}")
    form_type = item_data.get("form_type") or html_data.get("form_type")
    if form_type not in PROXY_FORM_TYPES:
        logger.info(
            f"{LOG_PREFIX} :_process_proxy_item: form_type={form_type} not in PROXY_FORM_TYPES")
        return
    cik_number = item_data.get("cik_number") or html_data.get("cik_number")
    if not cik_number:
        logger.info(
            f"{LOG_PREFIX} :_process_proxy_item: cik_number={cik_number} not found")
        return
    xbrl_files = html_data.get(
        "xbrl_files") or item_data.get("xbrl_files") or []
    proxy_file = find_file_by_type(xbrl_files, PROXY_FORM_TYPES)
    if not proxy_file or not proxy_file.get("url"):
        log_and_print(
            f"{LOG_PREFIX} :_process_proxy_item: ⚠️ No proxy HTM file for {item_data.get('company_name')}", "warning")
        return
    proxy_sec_url = build_full_sec_url(proxy_file.get("url"))
    logger.info(
        f"{LOG_PREFIX} :_process_proxy_item: proxy_sec_url={proxy_sec_url}")
    if not proxy_sec_url:
        logger.info(
            f"{LOG_PREFIX} :_process_proxy_item: proxy_sec_url not found")
        return
    filing_date = html_data.get("filing_date") or item_data.get("filing_date")
    if isinstance(filing_date, datetime):
        logger.info(
            f"{LOG_PREFIX} :_process_proxy_item: filing_date={filing_date}")

        filing_date = filing_date.strftime("%Y-%m-%d")
    else:
        logger.info(
            f"{LOG_PREFIX} :_process_proxy_item: filing_date={filing_date}")
        filing_date = str(filing_date) if filing_date else ""
    logger.info(
        f"{LOG_PREFIX} :_process_proxy_item: filing_date={filing_date}")
    sec_filling_id = str(filing.id) if filing else None
    if not sec_filling_id:
        logger.info(
            f"{LOG_PREFIX} :_process_proxy_item: sec_filling_id not found")
        log_and_print(
            f"{LOG_PREFIX} :_process_proxy_item: ⚠️ No sec_filling_id for proxy", "warning")
        return
    company_name = html_data.get(
        "company_name") or item_data.get("company_name") or ""
    deal_id = item_data.get("deal_id") or _deal_id_for_cik(cik_number)
    accession_number = item_data.get(
        "accession_number") or html_data.get("accession_number")
    logger.info(
        f"{LOG_PREFIX} :_process_proxy_item: accession_number={accession_number}")
    logger.info(
        f"{LOG_PREFIX} :_process_proxy_item: company_name={company_name}")
    logger.info(f"{LOG_PREFIX} :_process_proxy_item: deal_id={deal_id}")
    logger.info(
        f"{LOG_PREFIX} :_process_proxy_item: sec_filling_id={sec_filling_id}")
    logger.info(
        f"{LOG_PREFIX} :_process_proxy_item: filing_date={filing_date}")
    logger.info(f"{LOG_PREFIX} :_process_proxy_item: form_type={form_type}")
    logger.info(
        f"{LOG_PREFIX} :_process_proxy_item: proxy_sec_url={proxy_sec_url}")
    result = process_sec_document_for_filing_summary(
        cik_number=cik_number,
        company_name=company_name,
        sec_filling_id=sec_filling_id,
        filing_date=filing_date,
        form_type=form_type,
        proxy_sec_url=proxy_sec_url,
        deal_id=deal_id,
        accession_number=accession_number,
    )
    if result:
        log_and_print(
            f"{LOG_PREFIX} :_process_proxy_item: ✅ Proxy processing started: {result.get('sec_filing_summary_id')}")
    else:
        log_and_print("❌ Failed to start proxy processing", "error")


def _process_ten_k_ten_q_item(item_data, html_data, filing):
    """
    Fetch and save 10-K/10-Q filings from SEC API.

    This function:
    1. Fetches all 10-K/10-Q filings from SEC API (from announce date or 1 year ago)
    2. Saves each filing to SECFilingSummary.ten_k_ten_q (including current filing)
    3. Sends email with all filings

    Note: We don't save the current filing separately because it will be included
    in the SEC API results when we fetch filings from the time period.
    """
    logger.info(
        f"{LOG_PREFIX} :_process_ten_k_ten_q_item: item_data={item_data}")
    logger.info(
        f"{LOG_PREFIX} :_process_ten_k_ten_q_item: html_data={html_data}")
    logger.info(f"{LOG_PREFIX} :_process_ten_k_ten_q_item: filing={filing}")
    form_type = item_data.get("form_type") or html_data.get("form_type")
    cik_number = item_data.get("cik_number") or html_data.get("cik_number")
    company_name = item_data.get("company_name") or html_data.get(
        "company_name") or "Unknown Company"
    deal_id = item_data.get("deal_id") or _deal_id_for_cik(cik_number)
    logger.info(f"{LOG_PREFIX} :_process_ten_k_ten_q_item: deal_id={deal_id}")
    logger.info(
        f"{LOG_PREFIX} :_process_ten_k_ten_q_item: cik_number={cik_number}")
    logger.info(
        f"{LOG_PREFIX} :_process_ten_k_ten_q_item: company_name={company_name}")
    logger.info(
        f"{LOG_PREFIX} :_process_ten_k_ten_q_item: form_type={form_type}")

    # Fetch and save all 10-K/10-Q filings from SEC API (including current filing)
    try:
        from sec_rss_parser.utils_10k_10q import fetch_and_save_additional_10k_10q_filings

        log_and_print(
            f"{LOG_PREFIX} :_process_ten_k_ten_q_item: 🔍 Fetching 10-K/10-Q filings for CIK {cik_number}...")

        result = fetch_and_save_additional_10k_10q_filings(
            cik_number=cik_number,
            company_name=company_name,
            form_type=form_type,
            announce_date=None,  # Will be determined by the helper
            deal_id=deal_id,
            matched_deal=None,  # Could be passed if available
            item_data=item_data,
        )

        if result.get('success'):
            log_and_print(
                f"{LOG_PREFIX} :_process_ten_k_ten_q_item: ✅ 10-K/10-Q processing completed: "
                f"{result.get('filings_count', 0)} filings found, "
                f"{result.get('saved_count', 0)} new records saved to sec_filing_summary.ten_k_ten_q")
        else:
            log_and_print(
                f"{LOG_PREFIX} :_process_ten_k_ten_q_item: ⚠️ 10-K/10-Q processing failed: {result.get('error', 'Unknown error')}", "warning")
    except Exception as e:
        log_and_print(
            f"{LOG_PREFIX} :_process_ten_k_ten_q_item: ❌ Error fetching 10-K/10-Q filings: {e}", "error")


def _normalize_sec_url(url):
    """Build full SEC URL and strip ix?doc=/ prefix if present."""
    if not url:
        return None
    u = build_full_sec_url(url) or url
    if "ix?doc=/" in u:
        u = u.replace("ix?doc=/", "", 1)
    return u


def _pick_single_doc_url_for_form(xbrl_files, form_type, link):
    """
    For non-8-K form types, pick exactly one document URL when multiple xbrl_files match.
    Priority is by the index 'file' extension (not URL): 1) .htm  2) .html  3) .xml.
    """
    form_norm = (form_type or "").strip().upper()
    exts = _DOC_EXTENSIONS
    ext_priority = (".htm", ".html", ".xml")

    def priority_key(item):
        _, filename = item
        f_lower = (filename or "").lower()
        for i, ext in enumerate(ext_priority):
            if f_lower.endswith(ext):
                return i
        return len(ext_priority)

    candidates = []
    for f in xbrl_files or []:
        url = f.get("url")
        if not url or not any(url.lower().endswith(ext) for ext in exts):
            continue
        doc_type = (f.get("type") or "").upper()
        desc = (f.get("description") or "").upper()
        if not form_norm or form_norm in doc_type or form_norm in desc or doc_type in form_norm:
            candidates.append(f)
    if not candidates:
        return _normalize_sec_url(link) if link else None
    seen = set()
    unique = []  # list of (normalized_url, file)
    for f in candidates:
        u = _normalize_sec_url(f.get("url"))
        if u and u not in seen:
            seen.add(u)
            unique.append((u, f.get("file") or ""))
    if not unique:
        return _normalize_sec_url(link) if link else None
    if len(unique) == 1:
        return unique[0][0]
    return min(unique, key=priority_key)[0]


def _route_summarize_and_save(item_data, html_data):
    """
    For any form type: generate summary via route_and_summarize(url), save to SECFilingSummary
    with same 8-K/99.1 logic as process_feed_8k, and send a separate summary email.
    """
    form_type = (item_data.get("form_type") or html_data.get(
        "form_type") or "").strip().upper()
    if not form_type:
        form_type = "OTHER"
    accession_number = item_data.get(
        "accession_number") or html_data.get("accession_number")
    cik_number = item_data.get("cik_number") or html_data.get("cik_number")
    deal_id = item_data.get("deal_id") or _deal_id_for_cik(cik_number)
    company_name = item_data.get(
        "company_name") or html_data.get("company_name") or ""
    link = item_data.get("link") or ""
    xbrl_files = html_data.get(
        "xbrl_files") or item_data.get("xbrl_files") or []

    # Build list of (url, is_ex99). Only 8-K has 99.1 (EX-99.1 exhibit); other form types = parent-level only.
    urls_to_summarize = []
    if form_type == "8-K":
        file_8k = find_file_by_type(xbrl_files, "8-K")
        file_ex99 = find_file_by_type(xbrl_files, "EX-99.1")
        if file_8k and file_8k.get("url"):
            url_8k = _normalize_sec_url(file_8k.get("url"))
            if url_8k:
                urls_to_summarize.append((url_8k, False))
        if file_ex99 and file_ex99.get("url"):
            url_ex99 = _normalize_sec_url(file_ex99.get("url"))
            if url_ex99:
                urls_to_summarize.append((url_ex99, True))
    else:
        # Any other form type: single document, parent-level only (no 99.1 node).
        # If multiple xbrl_files match (e.g. primary_doc.html + primary_doc.xml), pick one URL only.
        sec_url = _pick_single_doc_url_for_form(xbrl_files, form_type, link)
        if sec_url:
            urls_to_summarize.append((sec_url, False))

    if not urls_to_summarize:
        log_and_print(
            f"{LOG_PREFIX} :_route_summarize_and_save: ⚠️ No document URL found for {form_type}", "warning")
        return

    for url, is_ex99 in urls_to_summarize:
        try:
            log_and_print(
                f"{LOG_PREFIX} :_route_summarize_and_save: 📝 Generating summary for {form_type}: {url[:80]}...")
            result = route_and_summarize(url)
            logger.info(
                f"{LOG_PREFIX} :_route_summarize_and_save: result={result}")
            s3_docx_url = result.get("s3_docx_url") or result.get("s3_url")
            if not s3_docx_url:
                log_and_print(
                    f"{LOG_PREFIX} :_route_summarize_and_save: ⚠️ No S3 docx URL returned for {url[:60]}...", "warning")
                continue
            log_and_print(
                f"{LOG_PREFIX} :_route_summarize_and_save: ✅ Summary uploaded to S3: {s3_docx_url[:80]}...")

            filing_dt = _filing_date_for_summary(
                html_data.get("filing_date") or item_data.get("filing_date"),
                result.get("filing_date"),
            )
            summary_kind = "EX-99.1" if is_ex99 else form_type
            doc_form_type = "8-K" if is_ex99 else form_type

            if is_ex99:
                ex99_1_payload = {
                    "items_reported": result.get("items_reported") or [],
                    "L1_headline": result.get("L1_headline"),
                    "L2_brief": result.get("L2_brief"),
                    "L3_detailed": result.get("L3_detailed") or {},
                    "s3_docx_url": result.get("s3_docx_url") or result.get("s3_url"),
                    "s3_json_url": result.get("s3_json_url"),
                }
                existing = SECFilingSummary.objects(
                    accession_number=accession_number, form_type="8-K"
                ).first()
                url_8k_main = None
                for fe in (html_data.get("filing_array") or item_data.get("filing_array") or []):
                    if (fe.get("document_type") or "").upper() == "8-K" and fe.get("url"):
                        url_8k_main = _normalize_sec_url(fe.get("url"))
                        break
                sec_document_url = url_8k_main or url
                if existing:
                    existing.sec_document_url = sec_document_url
                    existing.filing_date = filing_dt
                    existing.deal_id = deal_id
                    existing.ex99_1 = ex99_1_payload
                    existing.save()
                else:
                    SECFilingSummary(
                        form_type="8-K",
                        accession_number=accession_number,
                        cik_number=cik_number,
                        sec_document_url=sec_document_url,
                        filing_date=filing_dt,
                        deal_id=deal_id,
                        items_reported=[],
                        L1_headline=None,
                        L2_brief=None,
                        L3_detailed=None,
                        s3_docx_url=None,
                        s3_json_url=None,
                        ex99_1=ex99_1_payload,
                    ).save()
                log_and_print(
                    f"{LOG_PREFIX} :_route_summarize_and_save: 💾 EX-99.1 summary saved (99_1 node)")
            else:
                existing = SECFilingSummary.objects(
                    accession_number=accession_number, form_type=doc_form_type
                ).first()
                if existing:
                    existing.sec_document_url = url
                    existing.filing_date = filing_dt
                    existing.deal_id = deal_id
                    existing.items_reported = result.get(
                        "items_reported") or []
                    existing.L1_headline = result.get("L1_headline")
                    existing.L2_brief = result.get("L2_brief")
                    existing.L3_detailed = result.get("L3_detailed") or {}
                    existing.s3_docx_url = result.get(
                        "s3_docx_url") or result.get("s3_url")
                    existing.s3_json_url = result.get("s3_json_url")
                    existing.save()
                else:
                    SECFilingSummary(
                        form_type=doc_form_type,
                        accession_number=accession_number,
                        cik_number=cik_number,
                        sec_document_url=url,
                        filing_date=filing_dt,
                        deal_id=deal_id,
                        items_reported=result.get("items_reported") or [],
                        L1_headline=result.get("L1_headline"),
                        L2_brief=result.get("L2_brief"),
                        L3_detailed=result.get("L3_detailed") or {},
                        s3_docx_url=result.get(
                            "s3_docx_url") or result.get("s3_url"),
                        s3_json_url=result.get("s3_json_url"),
                    ).save()
                log_and_print(
                    f"{LOG_PREFIX} :_route_summarize_and_save: 💾 Summary saved (parent-level) for {doc_form_type}")

                # If 8-K with Item 5.02 (or 5.02): fetch one-year filings and send separate email
                # items_reported = result.get("items_reported") or []
                # _has_item_502 = any(
                #     "Item 5.02" in str(i) or str(i).strip() == "5.02"
                #     for i in items_reported
                # )
                # if form_type == "8-K" and doc_form_type == "8-K" and _has_item_502 and cik_number:
                #     try:
                #         start_date = (datetime.now() -
                #                       timedelta(days=365)).strftime("%Y-%m-%d")
                #         filings = fetch_sec_filings(
                #             str(cik_number), start_date=start_date)
                #         ticker_item502 = get_ticker_for_deal_and_cik(
                #             deal_id, cik_number)
                #         sec_subject, sec_html = generate_item_5_02_one_year_filings_email_html(
                #             company_name,
                #             filings,
                #             trigger_accession_number=accession_number,
                #             trigger_filing_date=filing_dt,
                #             cik_number=cik_number,
                #             ticker=ticker_item502,
                #         )
                #         payload = {
                #             "subject": sec_subject,
                #             "html": sec_html,
                #             "company_name": company_name,
                #             "email_type": "item_5_02_one_year_filings",
                #         }
                #         send_webhook_notification(
                #             N8N_WEBHOOK_URL_8K_SUMMARY, payload, "Item 5.02 one-year filings email"
                #         )
                #         log_and_print(
                #             f"{LOG_PREFIX} :_route_summarize_and_save: 📤 Sent Item 5.02 one-year filings email: {len(filings)} filings for {company_name}"
                #         )
                #     except Exception as item502_e:
                #         log_and_print(
                #             f"{LOG_PREFIX} :_route_summarize_and_save: ❌ Item 5.02 one-year filings email failed: {item502_e}",
                #             "error",
                #         )

            # Send email with the generated summary (doc link + L1 headline)
            log_and_print(
                f"{LOG_PREFIX} :_route_summarize_and_save: 📧 Sending summary email for {summary_kind}...")
            ticker = get_ticker_for_deal_and_cik(deal_id, cik_number)

            # Deal match: target vs acquirer for "(target)" or "(acquirer)" beside company name; acquirer-only LLM
            matched_cik_label = None
            form_affects_deal = None
            email_company_name = company_name
            if deal_id and cik_number:
                try:
                    deal = ProcessingJob.objects(id=ObjectId(deal_id)).only(
                        "cik", "acquirer_cik", "target_name", "acquire_name"
                    ).first()
                    cik_n = normalize_cik(cik_number)
                    if deal and cik_n:
                        if normalize_cik(deal.acquirer_cik) == cik_n:
                            matched_cik_label = "(acquirer)"
                            email_company_name = deal.acquire_name or company_name
                            form_affects_deal = _llm_form_affects_deal(
                                target_name=deal.target_name or "",
                                acquirer_name=deal.acquire_name or "",
                                sec_url=link or url,
                                form_type="8-K (EX-99.1)" if is_ex99 else doc_form_type,
                            )
                        elif normalize_cik(deal.cik) == cik_n:
                            matched_cik_label = "(target)"
                            email_company_name = deal.target_name or company_name
                except Exception as deal_e:
                    log_and_print(
                        f"{LOG_PREFIX} :_route_summarize_and_save: Deal lookup for match/LLM: {deal_e}", "warning")

            try:
                send_summary_email_via_webhook(
                    summary_doc_url=s3_docx_url,
                    company_name=email_company_name,
                    form_type="8-K (EX-99.1)" if is_ex99 else doc_form_type,
                    cik_number=cik_number or "",
                    sec_url=link or url,
                    accession_number=accession_number or "",
                    summary_kind=summary_kind,
                    l1_headline=result.get("L1_headline"),
                    l2_brief=result.get("L2_brief"),
                    l3_detailed=result.get("L3_detailed"),
                    ticker=ticker,
                    filing_date=filing_dt,
                    matched_cik_label=matched_cik_label,
                    form_affects_deal=form_affects_deal,
                )
                log_and_print(
                    f"{LOG_PREFIX} :_route_summarize_and_save: ✅ Summary email sent for {summary_kind}")
            except Exception as email_e:
                log_and_print(
                    f"{LOG_PREFIX} :_route_summarize_and_save: ❌ Summary email failed: {email_e}", "error")
        except Exception as e:
            log_and_print(
                f"{LOG_PREFIX} :_route_summarize_and_save: ❌ Summary failed for {url[:60]}...: {e}", "error")
            logger.exception(
                f"{LOG_PREFIX} :_route_summarize_and_save: url={url[:80]} error={e}")


def process_items(items):
    """
    For each item: skip 8-K (handled by process_feed_8k.py), check accession lookup,
    fetch HTML by form type, create SECFiling if needed, then branch by form_type.
    """
    unique = _filter_unique_items(items)
    logger.info(f"{LOG_PREFIX} :process_items: unique={len(unique)}")
    logger.info(f"{LOG_PREFIX} :process_items: unique={unique}")
    processed = 0
    errors = []
    for idx, item_data in enumerate(unique):
        lock_owner = None
        if idx > 0 and idx % 10 == 0:
            time.sleep(0.5)
        link = item_data.get("link")
        if not link:
            continue
        # Skip 8-K form type entirely — handled by process_feed_8k.py which has
        # the full EX-2.1 qualification flow (document_kind, us_listed, market_cap).
        feed_form_type = (item_data.get("form_type") or "").strip().upper()
        if feed_form_type in EXCLUDED_FORM_TYPES:
            log_and_print(
                f"{LOG_PREFIX} :process_items: ⏭️ Skipping excluded form type {feed_form_type}: {item_data.get('title', 'N/A')[:80]}")
            continue
        acc = item_data.get("accession_number") or extract_accession_from_guid(
            item_data.get("guid"))
        # Skip if already looked up; do not add to lookup here so read timeouts/failures can retry next run.
        if acc and AccessionLookedUp.objects(accession_number=acc).first():
            log_and_print(
                f"{LOG_PREFIX} :process_items: ⏭️ Skipping {acc} (already looked up)", "warning")
            continue
        if acc:
            lock_owner = acquire_accession_lock(acc, source="fetch_by_cik")
            if not lock_owner:
                log_and_print(
                    f"{LOG_PREFIX} :process_items: ⏭️ Skipping {acc} (in-progress by another worker or already finalized)",
                    "warning",
                )
                continue
        html_data = fetch_and_parse_html_by_form_type(
            link, form_type_from_feed=item_data.get("form_type")
        )
        logger.info(f"{LOG_PREFIX} :process_items: html_data={html_data}")
        if not html_data:
            errors.append({"accession": item_data.get(
                "accession_number"), "message": "Failed to parse HTML"})
            continue

           # Preserve deal CIK (from parse_atom_to_items) before HTML overwrites it.
        # We use deal_cik for saving SECFilingSummary and email; html_data has filer CIK.
        # deal_cik = item_data.get("cik_number")
        # Always use filer CIK extracted from the filing URL.
        item_data.update(html_data)
        # if deal_cik is not None:
        #     item_data["cik_number"] = deal_cik
        item_data["cik_number"] = _extract_cik_from_url(
            link) or item_data.get("cik_number")
        item_data["deal_id"] = item_data.get(
            "deal_id") or _deal_id_for_cik(item_data.get("cik_number"))
        filing, _ = _ensure_sec_filing(item_data)
        form_type = (item_data.get("form_type") or "").strip().upper()
        logger.info(f"{LOG_PREFIX} :process_items: form_type={form_type}")
        if not form_type:
            form_type = (html_data.get("form_type") or "").strip().upper()
        # Route-and-summarize for any form type: save to DB (8-K/99.1 logic) and send summary email
        try:
            _route_summarize_and_save(item_data, html_data)
        except Exception as summary_e:
            log_and_print(
                f"{LOG_PREFIX} :process_items: ⚠️ Route summary failed (continuing): {summary_e}", "warning")
            logger.exception(
                f"{LOG_PREFIX} :process_items: _route_summarize_and_save error={summary_e}")
        try:
            if form_type in PROXY_FORM_TYPES:
                logger.info(
                    f"{LOG_PREFIX} :process_items: form_type={form_type} handling proxy (comparison or standalone)")
                _handle_proxy_form_by_type(item_data, html_data, filing)
            elif form_type in TEN_K_TEN_Q_FORM_TYPES:
                logger.info(
                    f"{LOG_PREFIX} :process_items: form_type={form_type} processing 10-K/10-Q item")
                _process_ten_k_ten_q_item(item_data, html_data, filing)

            logger.info(f"{LOG_PREFIX} :process_items: accession_number={acc}")
            processed += 1
            # Only add to lookup after successful processing so read timeouts/failures can retry next run
            if acc:
                mark_accession_processed(acc)
        except Exception as e:
            errors.append({"accession": item_data.get(
                "accession_number"), "message": str(e)})
            log_and_print(
                f"{LOG_PREFIX} :process_items: ❌ Error processing item: {e}", "error")
        finally:
            if acc and lock_owner:
                release_accession_lock(acc, lock_owner)
    return {"processed": processed, "errors": errors}


def _cik_from_atom_title(title):
    """Extract CIK from Atom entry title like '8-K - Kennedy-Wilson Holdings, Inc. (0001408100) (Filer)'."""
    if not title:
        return None
    m = re.search(r"\((\d{10})\)", title)
    if m:
        return m.group(1)
    m = re.search(r"\((\d+)\)", title)
    if m:
        return m.group(1).zfill(10)
    return None


def run_fetch_sec_feed_by_deal_cik(
    output_json_path=None,
    limit_deals=None,
    process_items_flow=True,
    rss_file=None,
    rss_content=None,
):
    """
    Fetch deals -> CIKs -> feed per CIK -> parse to items.
    If process_items_flow=True, filter unique items and process each (proxy/8-K/10-K/other).

    For development: pass rss_file (path to XML) or rss_content (string) to use demo/local
    RSS instead of fetching from SEC. Example: rss_file="sec_rss_parser/rss copy.xml"
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    all_items = []
    errors = []
    feed_fetches = 0
    deals_processed = 0

    if rss_file or rss_content:
        # Development: use provided RSS content (file or string)
        if rss_content is None:
            rss_path = rss_file
            if not os.path.isabs(rss_path):
                base = os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__)))
                rss_path = os.path.join(base, rss_path)
            try:
                with open(rss_path, "r", encoding="utf-8", errors="replace") as f:
                    rss_content = f.read()
            except Exception as e:
                errors.append({"message": f"Failed to read RSS file: {e}"})
                return {"deals_processed": 0, "feed_fetches": 0, "items_count": 0, "errors": errors}
        if rss_content:
            items = parse_atom_to_items(
                rss_content, cik_number=None, deal_id=None)
            for it in items:
                cik = _cik_from_atom_title(it.get("title"))
                if cik:
                    it["cik_number"] = normalize_cik(cik)
                    it["deal_id"] = _deal_id_for_cik(it["cik_number"])
                all_items.append(it)
            feed_fetches = 1
        out = {
            "deals_processed": 0,
            "feed_fetches": feed_fetches,
            "items_count": len(all_items),
            "errors": errors,
            "source": "rss_file" if rss_file else "rss_content",
        }
    else:
        session = requests.Session()
        # read=0: do not retry on ReadTimeoutError (avoids triple load when SEC is slow)
        retry = Retry(total=3, read=0, backoff_factor=1,
                      status_forcelist=[429, 500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.mount("http://", HTTPAdapter(max_retries=retry))

        deals = get_open_or_unknown_deals()
        logger.info(
            f"{LOG_PREFIX} :run_fetch_sec_feed_by_deal_cik: deals={len(deals)}")
        if limit_deals is not None:
            deals = deals[:limit_deals]
        deals_processed = len(deals)

        for deal in deals:
            deal_id = str(deal.id)
            ciks = get_ciks_for_deal(deal)
            if not ciks:
                continue
            for cik in ciks:
                logger.info(
                    f"{LOG_PREFIX} :run_fetch_sec_feed_by_deal_cik: cik={cik}")
                raw = fetch_feed_for_cik(cik, session)

                feed_fetches += 1
                if feed_fetches % 7 == 0:
                    time.sleep(1)
                if not raw:
                    errors.append({"cik": cik, "deal_id": deal_id,
                                  "message": "Failed to fetch feed"})
                    continue
                try:
                    items = parse_atom_to_items(
                        raw, cik_number=cik, deal_id=deal_id)
                    logger.info(
                        f"{LOG_PREFIX} :run_fetch_sec_feed_by_deal_cik: items={len(items)}")
                    logger.info(
                        f"{LOG_PREFIX} :run_fetch_sec_feed_by_deal_cik: items={items}")
                    all_items.extend(items)
                except Exception as e:
                    errors.append(
                        {"cik": cik, "deal_id": deal_id, "message": str(e)})

        out = {
            "deals_processed": deals_processed,
            "feed_fetches": feed_fetches,
            "items_count": len(all_items),
            "errors": errors,
        }

    if process_items_flow and all_items:
        logger.info(
            f"{LOG_PREFIX} :run_fetch_sec_feed_by_deal_cik: process_items_flow=True all_items={len(all_items)}")
        result = process_items(all_items)
        logger.info(
            f"{LOG_PREFIX} :run_fetch_sec_feed_by_deal_cik: result={result}")
        out["items_processed"] = result["processed"]
        out["processing_errors"] = result["errors"]
    else:
        logger.info(
            f"{LOG_PREFIX} :run_fetch_sec_feed_by_deal_cik: process_items_flow=False all_items={len(all_items)}")
        out["items"] = all_items

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info("Wrote output to %s", output_json_path)

    return out


# For development: set to a path (relative to rag_project) to use demo RSS instead of live SEC.
DEMO_RSS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "rss copy.xml",
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Use demo RSS when file exists, or when USE_DEMO_RSS=1 (for development)
    use_demo = os.path.isfile(DEMO_RSS_FILE) or os.environ.get(
        "USE_DEMO_RSS", "").lower() in ("1", "true", "yes")
    result = run_fetch_sec_feed_by_deal_cik(
        output_json_path=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "sec_feed_by_deal_cik_output.json",
        ),
        limit_deals=5,
        process_items_flow=True,
        rss_file=DEMO_RSS_FILE if use_demo else None,
    )
    if result.get("source"):
        print("Source:", result["source"])
    print("Deals processed:", result["deals_processed"])
    print("Feed fetches:", result["feed_fetches"])
    print("Items count:", result["items_count"])
    print("Items processed:", result.get("items_processed", "N/A"))
    print("Errors:", len(result["errors"]))
    print("Processing errors:", len(result.get("processing_errors", [])))
