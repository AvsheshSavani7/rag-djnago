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


def guess_form_type(url: str) -> str:
    """Detect filing type from SEC URL filename."""
    fname = url.split("/")[-1].lower()
    # Order matters: specific 14C variants before broad 14-catch-all
    if "defm14c" in fname:
        return "DEFM14C"
    if "defm14a" in fname or "defm14" in fname:
        return "DEFM14A"
    if "defa14c" in fname:
        return "DEFA14C"
    if "defa14a" in fname or "defa14" in fname:
        return "DEFA14A"
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
    return "PROXY"
