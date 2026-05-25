"""
Logs API views — read pipeline logs and per-accession trace files.

Endpoints:
    GET /api/logs/pipelines/
        List all pipeline folders with sizes.

    GET /api/logs/search/?accession=XXX&level=ERROR&run_id=a8f91c&search=text&tail=500
        Search across ALL pipeline logs simultaneously.

    GET /api/logs/<pipeline>/stream/
        Read rolling pipeline log with optional filters.
        Query params: tail, level, accession, run_id, search

    GET /api/logs/<pipeline>/traces/
        List dates that have per-accession trace files.

    GET /api/logs/<pipeline>/traces/<date>/
        List per-accession trace files for a pipeline+date.

    GET /api/logs/<pipeline>/traces/<date>/<filename>/
        Read full contents of one per-accession trace file.
"""

import re
import os
from pathlib import Path

from django.conf import settings
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from core.log_reader import (
    list_pipelines,
    list_trace_dates,
    list_trace_files,
    read_rolling_log,
    read_trace_file,
    search_all_pipelines,
)

LOG_ROOT = getattr(settings, "LOG_ROOT", str(Path(settings.BASE_DIR) / "logs"))


_SAFE_RE = re.compile(r"^[\w\-\.]+$")
_UNSAFE_FILENAME_RE = re.compile(r'(\.\.)|[/\\]|\x00')


def _safe_segment(value: str) -> bool:
    """Return True if value is a safe pipeline name / date (strict allowlist)."""
    return bool(_SAFE_RE.match(value))


def _safe_filename(value: str) -> bool:
    """Return True if value is safe to use as a filename.

    Allows commas and other characters that appear in RSS-derived accession
    slugs. Only blocks path-traversal characters (.. / \\ null-byte).
    """
    if not value:
        return False
    return not bool(_UNSAFE_FILENAME_RE.search(value))


class PipelineListView(APIView):
    """
    GET /api/logs/pipelines/

    Response:
        {
            "log_root": "/var/log/rag",
            "pipelines": [
                {"pipeline": "sec_8k", "size_bytes": 172032, "last_modified": "..."},
                ...
            ]
        }
    """
    permission_classes = [AllowAny]

    def get(self, request):
        return Response({
            "log_root": LOG_ROOT,
            "pipelines": list_pipelines(LOG_ROOT),
        })


class GlobalSearchView(APIView):
    """
    GET /api/logs/search/

    Search across ALL pipeline rolling logs at once.

    Query params (all optional):
        accession  — substring match on the accession field
        run_id     — exact match on run_id
        level      — INFO / WARNING / ERROR / DEBUG
        search     — case-insensitive substring in the full log line
        tail       — max lines per pipeline to scan (default 500, max 5000)

    Response:
        {
            "total_matched": 14,
            "results": {
                "sec_8k":  { "total_matched": 12, "lines": [...] },
                "ex21":    { "total_matched": 2,  "lines": [...] }
            }
        }
    """
    permission_classes = [AllowAny]

    def get(self, request):
        tail = min(int(request.query_params.get("tail", 500)), 5000)
        return Response(
            search_all_pipelines(
                log_root=LOG_ROOT,
                accession=request.query_params.get("accession") or None,
                run_id=request.query_params.get("run_id") or None,
                level=request.query_params.get("level") or None,
                search=request.query_params.get("search") or None,
                tail=tail,
            )
        )


class PipelineLogStreamView(APIView):
    """
    GET /api/logs/<pipeline>/stream/

    Read the rolling log for a specific pipeline.

    Query params (all optional):
        tail             — last N matched lines to return (default 200, max 4000)
        level            — INFO / WARNING / ERROR / DEBUG
        accession        — substring match on accession field
        run_id           — exact run_id match
        search           — case-insensitive substring in the full log line
        include_rotated  — true/1 to also read .log.1 .log.2 … backup files

    Response:
        {
            "pipeline": "sec_8k",
            "tail": 200,
            "total_matched": 47,
            "file_size_bytes": 172032,
            "last_modified": "2026-05-25T11:30:00Z",
            "lines": [
                {
                    "ts": "2026-05-25 11:30:42",
                    "level": "INFO",
                    "pipeline": "sec_8k",
                    "run_id": "a8f91c",
                    "accession": "0001193125-26-126362",
                    "doc_type": "8K",
                    "module": "sec_rss_parser.process_feed_8k:625",
                    "message": "Processing started"
                },
                ...
            ]
        }
    """
    permission_classes = [AllowAny]

    def get(self, request, pipeline):
        if not _safe_segment(pipeline):
            return Response({"error": "Invalid pipeline name"}, status=status.HTTP_400_BAD_REQUEST)

        tail = min(int(request.query_params.get("tail", 200)), 4000)
        include_rotated = request.query_params.get(
            "include_rotated", "").lower() in ("true", "1")
        result = read_rolling_log(
            log_root=LOG_ROOT,
            pipeline=pipeline,
            tail=tail,
            level=request.query_params.get("level") or None,
            accession=request.query_params.get("accession") or None,
            run_id=request.query_params.get("run_id") or None,
            search=request.query_params.get("search") or None,
            include_rotated=include_rotated,
        )
        if "error" in result:
            return Response(result, status=status.HTTP_404_NOT_FOUND)
        return Response(result)


class TraceDateListView(APIView):
    """
    GET /api/logs/<pipeline>/traces/

    Response:
        {
            "pipeline": "sec_8k",
            "dates": ["2026-05-25", "2026-05-24", ...]
        }
    """
    permission_classes = [AllowAny]

    def get(self, request, pipeline):
        if not _safe_segment(pipeline):
            return Response({"error": "Invalid pipeline name"}, status=status.HTTP_400_BAD_REQUEST)

        return Response({
            "pipeline": pipeline,
            "dates": list_trace_dates(LOG_ROOT, pipeline),
        })


class TraceFileListView(APIView):
    """
    GET /api/logs/<pipeline>/traces/<date>/

    Response:
        {
            "pipeline": "sec_8k",
            "date": "2026-05-25",
            "count": 3,
            "files": [
                {
                    "filename": "0001193125-26-126362_EX21_a8f91c.log",
                    "accession": "0001193125-26-126362",
                    "doc_type": "EX21",
                    "run_id": "a8f91c",
                    "size_bytes": 4210,
                    "last_modified": "2026-05-25T11:31:00Z"
                },
                ...
            ]
        }
    """
    permission_classes = [AllowAny]

    def get(self, request, pipeline, date):
        if not _safe_segment(pipeline) or not _safe_segment(date):
            return Response({"error": "Invalid path segment"}, status=status.HTTP_400_BAD_REQUEST)

        files = list_trace_files(LOG_ROOT, pipeline, date)
        return Response({
            "pipeline": pipeline,
            "date": date,
            "count": len(files),
            "files": files,
        })


class TraceFileDetailView(APIView):
    """
    GET /api/logs/<pipeline>/traces/<date>/<filename>/

    Returns the full contents of one per-accession trace file.

    Response:
        {
            "pipeline": "sec_8k",
            "date": "2026-05-25",
            "filename": "0001193125-26-126362_EX21_a8f91c.log",
            "total_lines": 47,
            "lines": [
                {"ts": "...", "level": "INFO", "accession": "...", "message": "..."},
                ...
            ]
        }
    """
    permission_classes = [AllowAny]

    def get(self, request, pipeline, date, filename):
        if not _safe_segment(pipeline) or not _safe_segment(date) or not _safe_filename(filename):
            return Response({"error": "Invalid path segment"}, status=status.HTTP_400_BAD_REQUEST)

        result = read_trace_file(LOG_ROOT, pipeline, date, filename)
        if "error" in result:
            return Response(result, status=status.HTTP_404_NOT_FOUND)
        return Response(result)
