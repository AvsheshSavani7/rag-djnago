"""
Helpers for SEC daily feed JSON (CIK|accession composite keys, role parsing).
"""

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from sec_rss_parser.utils_8k import normalize_cik

FEED_ITEM_KEY_SEP = "|"
_ROLE_RE = re.compile(
    r"\((Issuer|Reporting|Subject|Filed by|Filer)\)\s*$",
    re.IGNORECASE,
)


def parse_filing_role(title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    match = _ROLE_RE.search(title.strip())
    return match.group(1) if match else None


def make_feed_item_key(cik: str, accession: str) -> str:
    return f"{normalize_cik(cik)}|{accession.strip()}"


def split_feed_item_key(key: str) -> Tuple[Optional[str], Optional[str]]:
    if FEED_ITEM_KEY_SEP not in key:
        return None, key if "-" in key else None
    cik, accession = key.split(FEED_ITEM_KEY_SEP, 1)
    return cik or None, accession or None


def iter_feed_records(items: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield feed records with accession_number and cik_number populated."""
    for key, raw in (items or {}).items():
        if not isinstance(raw, dict):
            continue
        record = dict(raw)
        key_cik, key_acc = split_feed_item_key(key)
        acc = (record.get("accession_number") or key_acc or "").strip()
        cik = normalize_cik(record.get("cik_number") or key_cik or "")
        if acc:
            record["accession_number"] = acc
        if cik:
            record["cik_number"] = cik
        if acc:
            yield record


def feed_accessions_present(items: Dict[str, Any]) -> Set[str]:
    return {
        rec["accession_number"]
        for rec in iter_feed_records(items)
        if rec.get("accession_number")
    }


def group_feed_records_by_accession(items: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in iter_feed_records(items):
        acc = record.get("accession_number")
        if acc:
            groups[acc].append(record)
    return dict(groups)


def pick_first_tracked_record(
    records: List[Dict[str, Any]],
    tracked_ciks: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """First feed row (in file order) whose CIK is tracked; None if no match."""
    for rec in records:
        if normalize_cik(rec.get("cik_number") or "") in tracked_ciks:
            return rec
    return None
