"""
sec_fetcher.py — SEC filing fetch and form type detection.
"""

import requests
from .config import SEC_HEADERS, get_form_label, get_form_family


def fetch_html(url: str) -> str:
    """Fetch SEC filing HTML."""
    r = requests.get(url, headers=SEC_HEADERS, timeout=60)
    r.raise_for_status()
    return r.text


def guess_form_type(url: str, html: str = "") -> str:
    """Detect filing type from SEC URL filename, with HTML content fallback."""
    fname = url.split("/")[-1].lower()
    # Order matters: more specific patterns first

    if "defm14c" in fname:
        return "DEFM14C"
    if "defm14a" in fname or "defm14" in fname:
        return "DEFM14A"
    if "defa14c" in fname:
        return "DEFA14C"
    if "defa14a" in fname or "defa14" in fname:
        return "DEFA14A"
    if "prer14a" in fname or "prer14" in fname:
        return "PREM14A/A"
    if "prem14c" in fname:
        return "PREM14C"
    if "prem14a" in fname or "prem14" in fname:
        return "PREM14A"
    if "sc14d9" in fname or "14d9" in fname or "14d-9" in fname:
        return "SC 14D-9"
    if "sctot" in fname or "sc_to_t" in fname or "sc-to-t" in fname:
        return "SC TO-T"
    if "sctoi" in fname or "sc_to_i" in fname or "sc-to-i" in fname:
        return "SC TO-I"
    if "f4a" in fname:
        return "F-4/A"
    if "f4" in fname:
        return "F-4"
    if "s4a" in fname:
        return "S-4/A"
    if "s4" in fname:
        return "S-4"

     # Fallback: check HTML content for form type (handles generic filenames like ea0283916-02.htm)
    if html:
        header = html[:3000].upper()
        if "DEFM14A" in header or "DEFM 14A" in header:
            return "DEFM14A"
        if "DEFM14C" in header or "DEFM 14C" in header:
            return "DEFM14C"
        if "DEFA14A" in header or "DEFA 14A" in header:
            return "DEFA14A"
        if "DEFA14C" in header or "DEFA 14C" in header:
            return "DEFA14C"
        if "PRER14A" in header or "PRER 14A" in header:
            return "PREM14A/A"
        if "PREM14A" in header or "PREM 14A" in header:
            return "PREM14A"
        if "PREM14C" in header or "PREM 14C" in header:
            return "PREM14C"
        if "SC 14D-9" in header or "SC14D9" in header:
            return "SC 14D-9"
        if "S-4/A" in header or "S4/A" in header:
            return "S-4/A"
        if "S-4" in header:
            return "S-4"
        if "F-4/A" in header or "F4/A" in header:
            return "F-4/A"
        if "F-4" in header:
            return "F-4"
    return "PROXY"
