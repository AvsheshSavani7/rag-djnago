"""SEC filing fetcher: resolve URLs, fetch HTML, detect metadata, make labels."""

import re
from datetime import datetime
from typing import Tuple

import requests

from .config import SEC_HEADERS


def parse_sec_document_url(url: str) -> Tuple[str, str]:
    """
    Extract cik_number and accession_number from an SEC document URL.
    URL pattern: https://www.sec.gov/Archives/edgar/data/{cik}/{acc_raw}/...
    Returns (cik_number, accession_number) with accession in SEC format NNNNNNNNNN-NN-NNNNNN.
    Returns (None, None) if URL does not match.
    """
    if not url:
        return (None, None)
    # Match /edgar/data/1234567/000123456789012345/ or similar
    m = re.search(r"/edgar/data/(\d+)/(\d+)/", url)
    if not m:
        return (None, None)
    cik = m.group(1).lstrip("0") or "0"
    acc_raw = m.group(2)
    if len(acc_raw) >= 18:
        # SEC format: NNNNNNNNNN-NN-NNNNNN (10-2-6)
        acc_dashed = f"{acc_raw[:10]}-{acc_raw[10:12]}-{acc_raw[12:18]}"
    else:
        acc_dashed = acc_raw
    return (cik, acc_dashed)


def resolve_sec_url(url: str) -> str:
    """Resolve inline-XBRL viewer URLs to the raw document URL."""
    if "ix?doc=" in url or "ix?doc%3D" in url:
        if "ix?doc=" in url:
            doc_path = url.split("ix?doc=")[1].split("&")[0]
        else:
            doc_path = url.split("ix%3Fdoc%3D")[1].split("&")[0]
        if not doc_path.startswith("/"):
            doc_path = "/" + doc_path
        return f"https://www.sec.gov{doc_path}"
    return url


def fetch_sec_filing(url: str) -> str:
    """Fetch the HTML content of a SEC filing. Returns raw HTML string."""
    resolved_url = resolve_sec_url(url)
    print(f"  Fetching: {resolved_url}")
    response = requests.get(resolved_url, headers=SEC_HEADERS, timeout=60)
    response.raise_for_status()
    print(f"  Response: {response.status_code} ({len(response.text):,} chars)")
    return response.text


def detect_filing_metadata(url: str, html: str = "") -> Tuple[str, str]:
    """
    Extract period date and filing type from URL and/or HTML content.
    Returns (period_date, filing_type).
    """
    # Try URL pattern first: ticker-YYYYMMDD.htm, YYYYMMDDx10k.htm, or YYYYMMDD_10k.htm
    date_match = re.search(
        r'(\d{4})(\d{2})(\d{2})(?:[x_]10[kq])?\.htm', url, re.IGNORECASE)
    if date_match:
        period_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
    else:
        period_date = "unknown"

    # Layer 1: Explicit match in URL (check amendment before base form)
    url_lower = url.lower()
    if "10-k/a" in url_lower or "10ka" in url_lower:
        filing_type = "10-K/A"
    elif "10-k" in url_lower or "10k" in url_lower:
        filing_type = "10-K"
    elif "10-q" in url_lower or "10q" in url_lower:
        filing_type = "10-Q"

    # Layer 2: Period date heuristic — reliable for standard calendar fiscal years
    elif period_date != "unknown":
        try:
            month = int(period_date.split("-")[1])
            day = int(period_date.split("-")[2])
            if month == 12 and day == 31:
                filing_type = "10-K"
            elif month in (3, 6, 9) and day in (30, 31):
                filing_type = "10-Q"
            else:
                filing_type = "unknown"   # non-standard fiscal year → fall to HTML
        except (IndexError, ValueError):
            filing_type = "unknown"

    else:
        filing_type = "unknown"

    # Layer 3 & 4: HTML scan — only reached for non-standard fiscal years (e.g. AMWD April 30)
    if filing_type == "unknown" and html:
        html_lower = html.lower()

        # Layer 3: strict — both form declaration AND report type must agree
        # Check amendment before base form
        if re.search(r'form\s+10-k/a\b', html_lower) and "annual report" in html_lower[:50000]:
            filing_type = "10-K/A"
        elif re.search(r'form\s+10-q\b', html_lower) and "quarterly report" in html_lower[:50000]:
            filing_type = "10-Q"
        elif re.search(r'form\s+10-k\b', html_lower) and "annual report" in html_lower[:50000]:
            filing_type = "10-K"

        # Layer 4: loose fallback — either signal alone
        elif re.search(r'form\s+10-k/a\b', html_lower) or "amendment" in html_lower[:50000] and "10-k" in html_lower[:50000]:
            filing_type = "10-K/A"
        elif re.search(r'form\s+10-q\b', html_lower) or "quarterly report" in html_lower[:50000]:
            filing_type = "10-Q"
        elif re.search(r'form\s+10-k\b', html_lower) or "annual report" in html_lower[:50000]:
            filing_type = "10-K"
        elif "10-k/a" in html_lower[:50000]:
            filing_type = "10-K/A"
        elif "10-q" in html_lower[:50000]:
            filing_type = "10-Q"
        elif "10-k" in html_lower[:50000]:
            filing_type = "10-K"
        else:
            filing_type = "unknown"

    return period_date, filing_type


def make_filing_label(period_date: str, filing_type: str) -> str:
    """Generate human-readable label like 'FY24 10-K' or 'Q1 2025 10-Q'."""
    if period_date == "unknown":
        return filing_type or "Unknown"

    try:
        dt = datetime.strptime(period_date, "%Y-%m-%d")
        if filing_type == "10-K/A":
            return f"FY{dt.strftime('%y')} 10-K/A"
        elif filing_type == "10-K":
            return f"FY{dt.strftime('%y')} 10-K"
        elif filing_type == "10-Q":
            month = dt.month
            if month <= 3:
                q = "Q1"
            elif month <= 6:
                q = "Q2"
            elif month <= 9:
                q = "Q3"
            else:
                q = "Q4"
            return f"{q} {dt.year} 10-Q"
        else:
            return f"{period_date} {filing_type}"
    except ValueError:
        return f"{period_date} {filing_type}"
