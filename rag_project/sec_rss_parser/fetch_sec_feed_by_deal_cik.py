"""
Fetch SEC RSS/Atom feed per deal CIK, then process items.

Flow:
1. Fetch all deals from MongoDB where deal_status is Open or Unknown.
2. For each deal, iterate by target CIK and acquirer CIK (if both available, do both).
3. For each CIK, call SEC browse-edgar URL and parse the Atom feed.
4. Filter items: skip if accession already in AccessionLookedUp; find unique items by accession.
5. Iterate items; for each, branch by form_type:
   - PROXY_SUMMARY_FORM_TYPES: process via proxy_processor_helper.process_sec_document_for_filing_summary(), 
     which creates/updates SECFilingSummary.proxy directly (no ProxyDocument, no sync).
   - 8-K: generate summary for 8-K and EX-99.1 (if present), send emails, save to sec_filing_summary.eight_k.
   - TEN_K_TEN_Q_FORM_TYPES: save to sec_filing_summary.ten_k_ten_q (minimal record).
   - Other: generate summary (no email), save to sec_filing_summary.other_filings.

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
from datetime import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

import django

if __name__ == "__main__":
    _rag_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _rag_root not in sys.path:
        sys.path.insert(0, _rag_root)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rag_project.settings")
    django.setup()

from document_processor.models import ProcessingJob
from sec_rss_parser.utils_8k import (
    SECRSSParser,
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
    send_summary_email_via_webhook,
)
from sec_rss_parser.models import (
    AccessionLookedUp,
    SECFiling,
    SECFilingSummary,
)
from sec_rss_parser.Eight_k_summary import summarize_8k_filing
from sec_rss_parser.proxy_processor_helper import process_sec_document_for_filing_summary

logger = logging.getLogger(__name__)

PROXY_SUMMARY_FORM_TYPES = [
    "DEFM14A", "DEFM14C", "PREM14A", "PREM14C", "S-4", "F-4", "S-4/A", "F-4/A",
]
PROXY_FORM_TYPES = ["DEFM14A", "DEFM14C", "PREM14A", "PREM14C", "S-4", "F-4"]
TEN_K_TEN_Q_FORM_TYPES = ["10-K", "10-Q"]

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


def _extract_xbrl_files_by_form_type(soup, form_type_from_feed):
    """
    Extract .htm files from document table, filtered by form_type:
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
        if not doc_url.endswith(".htm"):
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
        resp = requests.get(
            html_url,
            headers=DEFAULT_HEADERS,
            timeout=30,
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
                f"Missing form_type or accession: form_type={form_type}, accession={accession_number}",
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
            f"Error in fetch_and_parse_html_by_form_type: {e}", "error")
        return None


def get_open_or_unknown_deals():
    return list(
        ProcessingJob.objects(deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN).only(
            "id", "cik", "acquirer_cik"
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


def fetch_feed_for_cik(cik, session, headers=None):
    url = SEC_FEED_URL_TEMPLATE.format(cik=cik)
    for attempt in range(3):
        try:
            resp = session.get(
                url, headers=headers or DEFAULT_HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.warning(
                "Fetch attempt %s for CIK %s failed: %s", attempt + 1, cik, e)
            if attempt == 2:
                return None
            time.sleep(5)
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
            log_and_print(f"⏭️ Skipping already looked up: {acc}")
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
    deal = ProcessingJob.objects(
        cik=cik_n, deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN
    ).only("id").first()
    if deal:
        return str(deal.id)
    deal = ProcessingJob.objects(
        acquirer_cik=cik_n, deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN
    ).only("id").first()
    if deal:
        return str(deal.id)
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
        log_and_print(f"Failed to create SECFiling for {acc}: {e}", "error")
        return None, False


def _process_proxy_item(item_data, html_data, filing):
    """Process proxy form: start proxy document processing (summary + email happen async)."""
    form_type = item_data.get("form_type") or html_data.get("form_type")
    if form_type not in PROXY_FORM_TYPES:
        return
    cik_number = item_data.get("cik_number") or html_data.get("cik_number")
    if not cik_number:
        return
    xbrl_files = html_data.get(
        "xbrl_files") or item_data.get("xbrl_files") or []
    proxy_file = find_file_by_type(xbrl_files, PROXY_FORM_TYPES)
    if not proxy_file or not proxy_file.get("url"):
        log_and_print(
            f"⚠️ No proxy HTM file for {item_data.get('company_name')}", "warning")
        return
    proxy_sec_url = build_full_sec_url(proxy_file.get("url"))
    if not proxy_sec_url:
        return
    filing_date = html_data.get("filing_date") or item_data.get("filing_date")
    if isinstance(filing_date, datetime):
        filing_date = filing_date.strftime("%Y-%m-%d")
    else:
        filing_date = str(filing_date) if filing_date else ""
    sec_filling_id = str(filing.id) if filing else None
    if not sec_filling_id:
        log_and_print("⚠️ No sec_filling_id for proxy", "warning")
        return
    company_name = html_data.get(
        "company_name") or item_data.get("company_name") or ""
    deal_id = item_data.get("deal_id") or _deal_id_for_cik(cik_number)
    accession_number = item_data.get(
        "accession_number") or html_data.get("accession_number")
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
            f"✅ Proxy processing started: {result.get('sec_filing_summary_id')}")
    else:
        log_and_print("❌ Failed to start proxy processing", "error")


def _process_8k_item(item_data, html_data):
    """
    Generate 8-K and EX-99.1 summaries, save to sec_filing_summary.eight_k, send emails.
    - 8-K main document: summarize, save as main 8-K record, send email.
    - EX-99.1 (if present): summarize, append to eight_k.filings[], send email.
    """
    xbrl_files = html_data.get(
        "xbrl_files") or item_data.get("xbrl_files") or []
    accession_number = item_data.get(
        "accession_number") or html_data.get("accession_number")
    deal_id = item_data.get("deal_id") or _deal_id_for_cik(
        item_data.get("cik_number"))
    cik_number = item_data.get("cik_number") or html_data.get("cik_number")
    company_name = item_data.get(
        "company_name") or html_data.get("company_name") or ""
    link = item_data.get("link") or ""

    # Find 8-K and EX-99.1 files
    file_8k = find_file_by_type(xbrl_files, "8-K")
    file_ex99 = find_file_by_type(xbrl_files, "EX-99.1")

    url_8k = None
    if file_8k and file_8k.get("url"):
        url_8k = build_full_sec_url(file_8k.get("url")) or file_8k.get("url")
        if url_8k and "ix?doc=/" in url_8k:
            url_8k = url_8k.replace("ix?doc=/", "", 1)

    url_ex99 = None
    if file_ex99 and file_ex99.get("url"):
        url_ex99 = build_full_sec_url(
            file_ex99.get("url")) or file_ex99.get("url")
        if url_ex99 and "ix?doc=/" in url_ex99:
            url_ex99 = url_ex99.replace("ix?doc=/", "", 1)

    if not url_8k and not url_ex99:
        log_and_print("⚠️ No 8-K or EX-99.1 document URL found", "warning")
        return

    output_dir = tempfile.mkdtemp()
    filing_dt = None
    eight_k_payload = {
        "one_line_summary": None,
        "items_reported": [],
        "s3_docx_url": None,
        "s3_json_url": None,
        "filings": [],
    }

    # --- Process main 8-K document ---
    if url_8k:
        log_and_print(f"📝 Generating 8-K summary: {url_8k}")
        try:
            result_8k = summarize_8k_filing(
                url_8k, output_dir, upload_to_s3=True, s3_folder="8k", verbose=False
            )
            if result_8k.get("s3_url"):
                log_and_print(
                    f"✅ 8-K summary uploaded to S3: {result_8k['s3_url']}")
                filing_dt = _filing_date_for_summary(
                    html_data.get("filing_date") or item_data.get(
                        "filing_date"),
                    result_8k.get("filing_date"),
                )
                eight_k_payload["one_line_summary"] = result_8k.get(
                    "L1_headline")
                eight_k_payload["items_reported"] = result_8k.get(
                    "items_reported") or []
                eight_k_payload["s3_docx_url"] = result_8k.get("s3_url")
                eight_k_payload["s3_json_url"] = result_8k.get("s3_json_url")
                # Send 8-K summary email
                try:
                    send_summary_email_via_webhook(
                        summary_doc_url=result_8k.get("s3_url"),
                        company_name=company_name,
                        form_type="8-K",
                        cik_number=cik_number or "",
                        sec_url=link or url_8k,
                        accession_number=accession_number or "",
                        summary_kind="8-K",
                        l1_headline=result_8k.get("L1_headline"),
                    )
                    log_and_print("✅ 8-K summary email sent")
                except Exception as e:
                    log_and_print(f"❌ 8-K summary email failed: {e}", "error")
            else:
                log_and_print(
                    "⚠️ 8-K summary did not return S3 URL", "warning")
        except Exception as e:
            log_and_print(f"❌ 8-K summary failed: {e}", "error")

    # --- Process EX-99.1 document (if present) ---
    if url_ex99:
        log_and_print(f"📝 Generating EX-99.1 summary: {url_ex99}")
        try:
            result_99 = summarize_8k_filing(
                url_ex99, output_dir, upload_to_s3=True, s3_folder="99_1", verbose=False
            )
            if result_99.get("s3_url"):
                log_and_print(
                    f"✅ EX-99.1 summary uploaded to S3: {result_99['s3_url']}")
                ex99_filing_dt = _filing_date_for_summary(
                    html_data.get("filing_date") or item_data.get(
                        "filing_date"),
                    result_99.get("filing_date"),
                )
                if not filing_dt:
                    filing_dt = ex99_filing_dt
                ex99_entry = {
                    "filing_date": ex99_filing_dt,
                    "filing_url": url_ex99,
                    "s3_docx_url": result_99.get("s3_url"),
                    "s3_json_url": result_99.get("s3_json_url"),
                    "exhibit_type": "EX_99.1",
                }
                eight_k_payload["filings"].append(ex99_entry)
                # Send EX-99.1 summary email
                try:
                    send_summary_email_via_webhook(
                        summary_doc_url=result_99.get("s3_url"),
                        company_name=company_name,
                        form_type="8-K (EX-99.1)",
                        cik_number=cik_number or "",
                        sec_url=link or url_ex99,
                        accession_number=accession_number or "",
                        summary_kind="EX-99.1",
                        l1_headline=result_99.get("L1_headline"),
                    )
                    log_and_print("✅ EX-99.1 summary email sent")
                except Exception as e:
                    log_and_print(
                        f"❌ EX-99.1 summary email failed: {e}", "error")
            else:
                log_and_print(
                    "⚠️ EX-99.1 summary did not return S3 URL", "warning")
        except Exception as e:
            log_and_print(f"❌ EX-99.1 summary failed: {e}", "error")

    # --- Save to sec_filing_summary ---
    if not eight_k_payload["s3_docx_url"] and not eight_k_payload["filings"]:
        log_and_print(
            "⚠️ No 8-K or EX-99.1 summary generated, skipping DB save", "warning")
        return

    existing = SECFilingSummary.objects(
        accession_number=accession_number, form_type="8-K"
    ).first()
    if existing:
        existing.sec_document_url = url_8k or url_ex99
        existing.filing_date = filing_dt
        existing.deal_id = deal_id
        # Merge filings if existing has some
        if existing.eight_k and existing.eight_k.get("filings"):
            existing_filings = list(existing.eight_k.get("filings") or [])
            existing_filings.extend(eight_k_payload["filings"])
            eight_k_payload["filings"] = existing_filings
        existing.eight_k = eight_k_payload
        existing.save()
    else:
        SECFilingSummary(
            form_type="8-K",
            accession_number=accession_number,
            cik_number=cik_number,
            sec_document_url=url_8k or url_ex99,
            filing_date=filing_dt,
            deal_id=deal_id,
            eight_k=eight_k_payload,
        ).save()
    log_and_print("💾 8-K summary saved to sec_filing_summary")


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
    form_type = item_data.get("form_type") or html_data.get("form_type")
    cik_number = item_data.get("cik_number") or html_data.get("cik_number")
    company_name = item_data.get("company_name") or html_data.get(
        "company_name") or "Unknown Company"
    deal_id = item_data.get("deal_id") or _deal_id_for_cik(cik_number)

    # Fetch and save all 10-K/10-Q filings from SEC API (including current filing)
    try:
        from .utils_10k_10q import fetch_and_save_additional_10k_10q_filings

        log_and_print(f"🔍 Fetching 10-K/10-Q filings for CIK {cik_number}...")

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
                f"✅ 10-K/10-Q processing completed: "
                f"{result.get('filings_count', 0)} filings found, "
                f"{result.get('saved_count', 0)} new records saved to sec_filing_summary.ten_k_ten_q"
            )
        else:
            log_and_print(
                f"⚠️ 10-K/10-Q processing failed: {result.get('error', 'Unknown error')}",
                "warning"
            )
    except Exception as e:
        log_and_print(
            f"❌ Error fetching 10-K/10-Q filings: {e}",
            "error"
        )


def _process_other_filing_item(item_data, html_data, filing):
    """
    Generate summary for other filings (same way as 8-K/EX-99.1) and save to
    sec_filing_summary.other_filings. No email is sent.
    """
    form_type = item_data.get("form_type") or html_data.get(
        "form_type") or "OTHER"
    accession_number = item_data.get(
        "accession_number") or html_data.get("accession_number")
    cik_number = item_data.get("cik_number") or html_data.get("cik_number")
    sec_url = item_data.get("link") or ""
    xbrl_files = html_data.get(
        "xbrl_files") or item_data.get("xbrl_files") or []

    # Find the document file matching the form type
    doc_file = find_file_by_type(xbrl_files, form_type)
    if doc_file and doc_file.get("url"):
        sec_url = build_full_sec_url(
            doc_file.get("url")) or doc_file.get("url")
    elif xbrl_files and xbrl_files[0].get("url"):
        sec_url = build_full_sec_url(
            xbrl_files[0].get("url")) or xbrl_files[0].get("url")

    if sec_url and "ix?doc=/" in sec_url:
        sec_url = sec_url.replace("ix?doc=/", "", 1)

    filing_dt = _filing_date_for_summary(
        html_data.get("filing_date"), item_data.get("filing_date")
    )
    deal_id = item_data.get("deal_id") or _deal_id_for_cik(cik_number)

    # Initialize payload with pending status
    other_payload = {
        "summary_status": "pending",
        "form_type": form_type,
        "s3_docx_url": None,
        "s3_json_url": None,
        "one_line_summary": None,
    }

    # --- Generate summary (same as 8-K/EX-99.1) but NO email ---
    if sec_url:
        log_and_print(f"📝 Generating summary for {form_type}: {sec_url}")
        output_dir = tempfile.mkdtemp()
        try:
            result = summarize_8k_filing(
                sec_url,
                output_dir,
                upload_to_s3=True,
                s3_folder="other_filings",
                verbose=False,
            )
            if result.get("s3_url"):
                log_and_print(
                    f"✅ {form_type} summary uploaded to S3: {result['s3_url']}")
                other_payload["summary_status"] = "completed"
                other_payload["s3_docx_url"] = result.get("s3_url")
                other_payload["s3_json_url"] = result.get("s3_json_url")
                other_payload["one_line_summary"] = result.get("L1_headline")
            else:
                log_and_print(
                    f"⚠️ {form_type} summary did not return S3 URL", "warning")
                other_payload["summary_status"] = "failed"
        except Exception as e:
            log_and_print(f"❌ {form_type} summary failed: {e}", "error")
            other_payload["summary_status"] = "failed"
    else:
        log_and_print(f"⚠️ No document URL found for {form_type}", "warning")

    # --- Save to sec_filing_summary ---
    existing = SECFilingSummary.objects(
        accession_number=accession_number, form_type=form_type
    ).first()
    if existing:
        existing.sec_document_url = sec_url
        existing.filing_date = filing_dt
        existing.deal_id = deal_id
        existing.other_filings = other_payload
        existing.save()
    else:
        SECFilingSummary(
            form_type=form_type,
            accession_number=accession_number,
            cik_number=cik_number,
            sec_document_url=sec_url,
            filing_date=filing_dt,
            deal_id=deal_id,
            other_filings=other_payload,
        ).save()
    log_and_print(
        f"💾 Other filing ({form_type}) saved to sec_filing_summary.other_filings")


def process_items(items):
    """
    For each item: check accession lookup, fetch HTML by form type, create SECFiling if needed, then branch by form_type.
    Uses fetch_and_parse_html_by_form_type: for 8-K only 8-K/EX-99.1 files; otherwise only files matching form_type.
    """
    unique = _filter_unique_items(items)
    processed = 0
    errors = []
    for idx, item_data in enumerate(unique):
        if idx > 0 and idx % 10 == 0:
            time.sleep(0.5)
        link = item_data.get("link")
        if not link:
            continue
        html_data = fetch_and_parse_html_by_form_type(
            link, form_type_from_feed=item_data.get("form_type")
        )
        if not html_data:
            errors.append({"accession": item_data.get(
                "accession_number"), "message": "Failed to parse HTML"})
            continue
        item_data.update(html_data)
        item_data["deal_id"] = item_data.get(
            "deal_id") or _deal_id_for_cik(item_data.get("cik_number"))
        filing, _ = _ensure_sec_filing(item_data)
        form_type = (item_data.get("form_type") or "").strip().upper()
        if not form_type:
            form_type = (html_data.get("form_type") or "").strip().upper()
        try:
            if form_type in PROXY_SUMMARY_FORM_TYPES:
                _process_proxy_item(item_data, html_data, filing)
            elif form_type == "8-K":
                _process_8k_item(item_data, html_data)
            elif form_type in TEN_K_TEN_Q_FORM_TYPES:
                _process_ten_k_ten_q_item(item_data, html_data, filing)
            else:
                _process_other_filing_item(item_data, html_data, filing)
            acc = item_data.get("accession_number")
            if acc:
                try:
                    AccessionLookedUp(accession_number=acc).save()
                except Exception:
                    pass
            processed += 1
        except Exception as e:
            errors.append({"accession": item_data.get(
                "accession_number"), "message": str(e)})
            log_and_print(f"❌ Error processing item: {e}", "error")
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
        retry = Retry(total=3, backoff_factor=1,
                      status_forcelist=[429, 500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.mount("http://", HTTPAdapter(max_retries=retry))

        deals = get_open_or_unknown_deals()
        if limit_deals is not None:
            deals = deals[:limit_deals]
        deals_processed = len(deals)

        for deal in deals:
            deal_id = str(deal.id)
            ciks = get_ciks_for_deal(deal)
            if not ciks:
                continue
            for cik in ciks:
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
        result = process_items(all_items)
        out["items_processed"] = result["processed"]
        out["processing_errors"] = result["errors"]
    else:
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
