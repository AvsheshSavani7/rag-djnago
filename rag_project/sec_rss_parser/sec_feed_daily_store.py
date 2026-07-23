"""
Shared daily JSON store for SEC feed poller (production + test scripts).

Feed file names (feed_YYYYMMDD.json) use the SEC filing calendar day in
America/New_York (US Eastern, DST-aware) so collector, processor, and
daily-index reconcile all target the same date as master.{YYYYMMDD}.idx.

No MongoDB — file I/O only.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from sec_rss_parser.sec_feed_item_utils import make_feed_item_key, parse_filing_role
from sec_rss_parser.utils_8k import normalize_cik

# SEC filing day + reconcile schedule timezone (matches EDGAR daily-index dates).
SEC_FEED_TZ = ZoneInfo(os.environ.get("SEC_FEED_TIMEZONE", "America/New_York"))

# Minutes after ET midnight when processor also drains previous day's feed.
MIDNIGHT_GRACE_MINUTES = int(os.environ.get("SEC_FEED_MIDNIGHT_GRACE_MINUTES", "30"))


def feed_now() -> datetime:
    """Current time in the SEC feed timezone (America/New_York by default)."""
    return datetime.now(SEC_FEED_TZ)


def _as_feed_tz(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=SEC_FEED_TZ)
    return dt.astimezone(SEC_FEED_TZ)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _feed_tz_now_iso() -> str:
    """ISO timestamp in SEC feed TZ (America/New_York — EST/EDT per DST)."""
    return feed_now().replace(microsecond=0).isoformat()


def default_feed_dir() -> str:
    return os.environ.get(
        "SEC_FEED_DAILY_DIR",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "sec_daily_feed",
        ),
    )


def feed_path_for_date(feed_dir: str, day: Optional[datetime] = None) -> str:
    d = _as_feed_tz(day or feed_now()).strftime("%Y%m%d")
    return os.path.join(feed_dir, f"feed_{d}.json")


def processor_state_path_for_date(feed_dir: str, day: Optional[datetime] = None) -> str:
    d = _as_feed_tz(day or feed_now()).strftime("%Y%m%d")
    return os.path.join(feed_dir, f"processor_state_{d}.json")


def is_within_midnight_grace_window(
    now: Optional[datetime] = None,
    grace_minutes: Optional[int] = None,
) -> bool:
    """
    True during the first N minutes after ET midnight (default 30: 00:00–00:29).
    """
    t = _as_feed_tz(now or feed_now())
    grace = grace_minutes if grace_minutes is not None else MIDNIGHT_GRACE_MINUTES
    return t.hour == 0 and t.minute < max(1, grace)


def get_feed_days_to_process(
    now: Optional[datetime] = None,
    grace_minutes: Optional[int] = None,
) -> List[Tuple[Optional[datetime], str]]:
    """
    (day, label) pairs for processor runs.

    day=None means today's SEC feed date (America/New_York). During midnight
    grace, yesterday is included first.
    """
    t = _as_feed_tz(now or feed_now())
    days: List[Tuple[Optional[datetime], str]] = [(None, "today")]
    if is_within_midnight_grace_window(t, grace_minutes=grace_minutes):
        days.insert(0, (t - timedelta(days=1), "yesterday_grace"))
    return days


def _empty_feed(date_str: str) -> Dict[str, Any]:
    return {
        "date": date_str,
        "updated_at": _feed_tz_now_iso(),
        "schema_version": 2,
        "items": {},
    }


def _empty_processor_state(date_str: str) -> Dict[str, Any]:
    return {
        "date": date_str,
        "updated_at": _feed_tz_now_iso(),
        "handled_accessions": [],
        "pipeline_log": [],
    }


def load_json(path: str, default_factory) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return default_factory()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return default_factory()
        return data
    except (json.JSONDecodeError, OSError):
        return default_factory()


def atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data["updated_at"] = _feed_tz_now_iso()
    fd, tmp_path = tempfile.mkstemp(
        suffix=".json",
        prefix=os.path.basename(path) + ".",
        dir=os.path.dirname(path) or ".",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_daily_feed(feed_dir: str, day: Optional[datetime] = None) -> Tuple[str, Dict[str, Any]]:
    path = feed_path_for_date(feed_dir, day)
    date_str = _as_feed_tz(day or feed_now()).strftime("%Y-%m-%d")
    data = load_json(path, lambda: _empty_feed(date_str))
    if "items" not in data or not isinstance(data["items"], dict):
        data["items"] = {}
    if "date" not in data:
        data["date"] = date_str
    if not data.get("schema_version"):
        data["schema_version"] = 1
    return path, data


def load_processor_state(feed_dir: str, day: Optional[datetime] = None) -> Tuple[str, Dict[str, Any]]:
    path = processor_state_path_for_date(feed_dir, day)
    date_str = _as_feed_tz(day or feed_now()).strftime("%Y-%m-%d")
    data = load_json(path, lambda: _empty_processor_state(date_str))
    if "handled_accessions" not in data or not isinstance(data["handled_accessions"], list):
        data["handled_accessions"] = []
    if "pipeline_log" not in data or not isinstance(data["pipeline_log"], list):
        data["pipeline_log"] = []
    if "date" not in data:
        data["date"] = date_str
    return path, data


def append_feed_items(
    feed_dir: str,
    new_items: List[Dict[str, Any]],
    day: Optional[datetime] = None,
) -> Tuple[int, List[str]]:
    """
    Merge new_items into today's feed JSON keyed by ``cik|accession_number``.

    Multiple SEC Atom entries for the same filing (Issuer + Reporting, etc.)
    are stored as separate rows. Stage B groups by accession when processing.

    Returns (count_added, list_of_new_accession_numbers for logging).
    """
    path, data = load_daily_feed(feed_dir, day)
    items = data["items"]
    added = 0
    new_accessions: List[str] = []

    now = _feed_tz_now_iso()
    for raw in new_items:
        acc = (raw.get("accession_number") or "").strip()
        cik = normalize_cik(raw.get("cik_number") or "")
        if not acc or not cik:
            continue
        item_key = make_feed_item_key(cik, acc)
        if item_key in items:
            continue
        record = dict(raw)
        record["accession_number"] = acc
        record["cik_number"] = cik
        role = record.get("filing_role") or parse_filing_role(record.get("title"))
        if role:
            record["filing_role"] = role
        record["first_seen"] = now
        items[item_key] = record
        added += 1
        new_accessions.append(acc)

    if added:
        data["schema_version"] = 2
        atomic_write_json(path, data)
    return added, new_accessions


def record_processor_decisions(
    feed_dir: str,
    decisions: List[Dict[str, Any]],
    day: Optional[datetime] = None,
    feed_label: Optional[str] = None,
    non_terminal_actions: Optional[set] = None,
) -> None:
    """Append processor decisions to processor_state JSON (no Mongo writes).

    Every decision is appended to pipeline_log for visibility. Only *terminal*
    decisions are added to handled_accessions. Decisions whose action is listed
    in non_terminal_actions (e.g. entered pipeline, or held by a live lock) are
    intentionally NOT persisted as handled, so they stay eligible for retry —
    MongoDB (AccessionLookedUp + AccessionProcessingLock) remains their sole gate.
    """
    if not decisions:
        return
    non_terminal = non_terminal_actions or set()
    path, data = load_processor_state(feed_dir, day)
    handled = set(data["handled_accessions"])
    for d in decisions:
        acc = (d.get("accession_number") or "").strip()
        if not acc:
            continue
        entry = dict(d)
        entry["logged_at"] = _utc_now_iso()
        if feed_label:
            entry["feed_label"] = feed_label
        data["pipeline_log"].append(entry)
        if d.get("action") not in non_terminal:
            handled.add(acc)
    data["handled_accessions"] = sorted(handled)
    atomic_write_json(path, data)
