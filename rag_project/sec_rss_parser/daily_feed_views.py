"""
API views for SEC daily feed JSON files (collector output).

Endpoints:
    GET /api/sec/daily-feed/dates/
        List available feed_YYYYMMDD.json files.

    GET /api/sec/daily-feed/
    GET /api/sec/daily-feed/<YYYYMMDD>/
        View feed JSON (paginated) or download the raw file.

    Query params (detail):
        download=1       — attachment download of the raw JSON file
        tracked_only=1   — only filings whose CIK is on an open/unknown deal
        limit            — max items returned (default 100, max 500)
        offset           — pagination offset (default 0)
        summary_only=1   — metadata only, no items array
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from django.http import FileResponse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from sec_rss_parser.fetch_sec_feed_by_deal_cik import build_tracked_ciks_map
from sec_rss_parser.sec_feed_daily_store import (
    SEC_FEED_TZ,
    default_feed_dir,
    feed_now,
    load_daily_feed,
)

logger = logging.getLogger(__name__)

_FEED_DATE_RE = re.compile(r"^feed_(\d{8})\.json$")
_DATE_KEY_RE = re.compile(r"^\d{8}$")

DEFAULT_LIMIT = 100
MAX_LIMIT = 500


def _parse_date_key(date_key: Optional[str]) -> Optional[datetime]:
    if not date_key:
        return feed_now()
    if not _DATE_KEY_RE.match(date_key):
        return None
    return datetime.strptime(date_key, "%Y%m%d").replace(tzinfo=SEC_FEED_TZ)


def _wants_download(request) -> bool:
    return request.query_params.get("download", "").lower() in ("1", "true")


def _wants_tracked_only(request) -> bool:
    return request.query_params.get("tracked_only", "").lower() in ("1", "true")


def _wants_summary_only(request) -> bool:
    return request.query_params.get("summary_only", "").lower() in ("1", "true")


def _pagination(request) -> Tuple[int, int]:
    try:
        limit = int(request.query_params.get("limit", DEFAULT_LIMIT))
    except (TypeError, ValueError):
        limit = DEFAULT_LIMIT
    try:
        offset = int(request.query_params.get("offset", 0))
    except (TypeError, ValueError):
        offset = 0
    limit = max(1, min(limit, MAX_LIMIT))
    offset = max(0, offset)
    return limit, offset


def list_feed_files(feed_dir: str) -> List[Dict[str, Any]]:
    root = Path(feed_dir)
    if not root.is_dir():
        return []

    results: List[Dict[str, Any]] = []
    for path in sorted(root.glob("feed_*.json"), reverse=True):
        match = _FEED_DATE_RE.match(path.name)
        if not match:
            continue
        date_key = match.group(1)
        item_count = 0
        updated_at = None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                items = data.get("items")
                if isinstance(items, dict):
                    item_count = len(items)
                updated_at = data.get("updated_at")
        except (json.JSONDecodeError, OSError):
            pass

        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = 0

        results.append({
            "date": f"{date_key[:4]}-{date_key[4:6]}-{date_key[6:8]}",
            "date_key": date_key,
            "filename": path.name,
            "size_bytes": size_bytes,
            "item_count": item_count,
            "updated_at": updated_at,
        })
    return results


def _items_to_list(
    items: Dict[str, Any],
    tracked_ciks: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for accession, record in items.items():
        if not isinstance(record, dict):
            continue
        cik = (record.get("cik_number") or "").strip()
        if tracked_ciks is not None and cik not in tracked_ciks:
            continue
        row = dict(record)
        row["accession_number"] = accession
        if tracked_ciks is not None and cik in tracked_ciks:
            row["deal_id"] = tracked_ciks[cik]
        rows.append(row)
    rows.sort(key=lambda row: row.get("first_seen") or "", reverse=True)
    return rows


class DailyFeedDateListView(APIView):
    """GET /api/sec/daily-feed/dates/ — list available daily feed files."""

    def get(self, request):
        feed_dir = default_feed_dir()
        return Response({
            "success": True,
            "timezone": str(SEC_FEED_TZ),
            "today_date_key": feed_now().strftime("%Y%m%d"),
            "feed_dir": feed_dir,
            "count": len(list_feed_files(feed_dir)),
            "dates": list_feed_files(feed_dir),
        })


class DailyFeedDetailView(APIView):
    """
    GET /api/sec/daily-feed/ or /api/sec/daily-feed/<YYYYMMDD>/

    View paginated feed items or download the raw JSON file.
    """

    def get(self, request, date: Optional[str] = None):
        day = _parse_date_key(date)
        if day is None:
            return Response(
                {"success": False, "error": "Invalid date; use YYYYMMDD (America/New_York feed day)"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        feed_dir = default_feed_dir()
        path, data = load_daily_feed(feed_dir, day=day)
        date_key = day.strftime("%Y%m%d")
        exists = os.path.isfile(path)

        if _wants_download(request):
            if not exists:
                return Response(
                    {
                        "success": False,
                        "error": "Feed file not found",
                        "date_key": date_key,
                        "filename": os.path.basename(path),
                    },
                    status=status.HTTP_404_NOT_FOUND,
                )
            return FileResponse(
                open(path, "rb"),
                as_attachment=True,
                filename=os.path.basename(path),
                content_type="application/json",
            )

        items = data.get("items") or {}
        if not isinstance(items, dict):
            items = {}

        tracked_ciks = None
        tracked_only = _wants_tracked_only(request)
        if tracked_only:
            tracked_ciks = build_tracked_ciks_map()

        all_rows = _items_to_list(items, tracked_ciks=tracked_ciks)
        limit, offset = _pagination(request)
        page = all_rows[offset: offset + limit]

        payload: Dict[str, Any] = {
            "success": True,
            "date": data.get("date") or day.strftime("%Y-%m-%d"),
            "date_key": date_key,
            "timezone": str(SEC_FEED_TZ),
            "filename": os.path.basename(path),
            "exists": exists,
            "size_bytes": os.path.getsize(path) if exists else 0,
            "updated_at": data.get("updated_at"),
            "item_count": len(all_rows),
            "total_item_count": len(items),
            "tracked_only": tracked_only,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < len(all_rows),
        }

        if not _wants_summary_only(request):
            payload["items"] = page

        return Response(payload)
