"""
Pure-Python log reading utilities for the Logs API.
No Django imports — safe to use anywhere.
"""

import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

_IST = timezone(timedelta(hours=5, minutes=30))

# Matches lines produced by PIPELINE_FORMATTER in dynamic_pipeline_handler.py
# Supports both 24h (legacy) and 12h AM/PM (current) timestamp formats.
LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} (?:\d{2}:\d{2}:\d{2}|\d{1,2}:\d{2}:\d{2} (?:AM|PM)))"
    r" \| (?P<level>\w+)\s*"
    r" \| pipeline=(?P<pipeline>\S+)"
    r" \| run_id=(?P<run_id>\S+)"
    r" \| accession=(?P<accession>\S+)"
    r" \| doc_type=(?P<doc_type>\S+)"
    r" \| (?P<module>[^|]+?)\s*"
    r" \| (?P<message>.*)$"
)


def _parse_line(raw: str) -> dict | None:
    """Parse one formatted log line into a dict. Returns None if unrecognised."""
    m = LOG_LINE_RE.match(raw.rstrip())
    if not m:
        return None
    return m.groupdict()


def _file_meta(path: Path) -> dict:
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "last_modified": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def _today_ist() -> str:
    return datetime.now(tz=_IST).strftime("%Y-%m-%d")


def _daily_dir(log_root: str, pipeline: str, date: str = None) -> Path:
    return Path(log_root) / pipeline / "daily" / (date or _today_ist())


def active_log_path(log_root: str, pipeline: str, date: str = None) -> Path:
    """Path to today's (or given date's) active log: daily/{date}/{pipeline}.log."""
    d = date or _today_ist()
    return _daily_dir(log_root, pipeline, d) / f"{pipeline}.log"


def _resolve_active_log(log_root: str, pipeline: str, date: str = None) -> Path | None:
    """Return active log path, falling back to legacy flat layout if needed."""
    path = active_log_path(log_root, pipeline, date)
    if path.exists():
        return path
    legacy = Path(log_root) / pipeline / f"{pipeline}.log"
    if legacy.exists():
        return legacy
    return None


def _rotation_index(filename: str, pipeline: str) -> int:
    """Return rotation index: app.log → 0, app.log.3 → 3."""
    prefix = f"{pipeline}.log."
    if filename == f"{pipeline}.log":
        return 0
    if filename.startswith(prefix):
        suffix = filename[len(prefix):]
        if suffix.isdigit():
            return int(suffix)
    return 999


def _list_rotation_files_in_dir(day_dir: Path, pipeline: str) -> list:
    """List app.log + app.log.N in one daily folder, active first."""
    files = []
    active = day_dir / f"{pipeline}.log"
    if active.exists():
        meta = _file_meta(active)
        meta.update({"filename": active.name, "rotation": 0})
        files.append(meta)

    rotated = sorted(
        day_dir.glob(f"{pipeline}.log.*"),
        key=lambda f: _rotation_index(f.name, pipeline),
    )
    for p in rotated:
        idx = _rotation_index(p.name, pipeline)
        if idx == 999:
            continue
        meta = _file_meta(p)
        meta.update({"filename": p.name, "rotation": idx})
        files.append(meta)
    return files


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def list_pipelines(log_root: str) -> list:
    """Return metadata for every pipeline folder that has a daily or legacy log file."""
    root = Path(log_root)
    if not root.exists():
        return []
    result = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        pipeline = folder.name
        log_file = _resolve_active_log(log_root, pipeline)
        if log_file is None:
            continue
        meta = _file_meta(log_file)
        meta["pipeline"] = pipeline
        result.append(meta)
    return result


def read_file_raw(path: Path) -> dict:
    """Return the full content of a log file as a single string. No parsing."""
    if not path.exists():
        return {"error": f"File not found: {path.name}"}
    return {"content": path.read_text(encoding="utf-8", errors="replace")}


def _filter_lines(raw_lines: list, level: str, accession: str, run_id: str, search: str) -> list:
    """Apply AND filters to a list of raw log lines, return parsed dicts."""
    matched = []
    for raw in raw_lines:
        parsed = _parse_line(raw)
        if parsed is None:
            continue
        if level and parsed["level"] != level.upper():
            continue
        if accession and accession not in parsed["accession"]:
            continue
        if run_id and parsed["run_id"] != run_id:
            continue
        if search and search.lower() not in raw.lower():
            continue
        matched.append(parsed)
    return matched


def read_rolling_log(
    log_root: str,
    pipeline: str,
    tail: int = 200,
    level: str = None,
    accession: str = None,
    run_id: str = None,
    search: str = None,
) -> dict:
    """
    Read the active rolling log ({pipeline}.log) and return the last `tail` matched lines.

    Filters (all ANDed):
        level     — exact level match (INFO / WARNING / ERROR / DEBUG)
        accession — substring match inside the accession field
        run_id    — exact run_id match
        search    — case-insensitive substring in the raw line
    """
    log_file = _resolve_active_log(log_root, pipeline)
    if log_file is None:
        return {"error": f"No log file found for pipeline '{pipeline}'"}

    meta = _file_meta(log_file)
    raw_lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    matched = _filter_lines(raw_lines, level, accession, run_id, search)

    tail = max(1, min(tail, 5000))
    return {
        "pipeline": pipeline,
        "tail": tail,
        "total_matched": len(matched),
        "file_size_bytes": meta["size_bytes"],
        "last_modified": meta["last_modified"],
        "lines": matched[-tail:],
    }


def list_daily_dates(log_root: str, pipeline: str) -> list:
    """Return IST dates (most recent first) under logs/{pipeline}/daily/."""
    daily_dir = Path(log_root) / pipeline / "daily"
    if not daily_dir.exists():
        return []
    return sorted(
        [d.name for d in daily_dir.iterdir() if d.is_dir()],
        reverse=True,
    )


def list_daily_files(log_root: str, pipeline: str, date: str) -> list:
    """
    Return rotation files for one day: {pipeline}.log, {pipeline}.log.1, ...
    under logs/{pipeline}/daily/{date}/.
    """
    day_dir = _daily_dir(log_root, pipeline, date)
    if not day_dir.exists():
        return []
    return _list_rotation_files_in_dir(day_dir, pipeline)


def list_rotated_files(log_root: str, pipeline: str) -> list:
    """Legacy alias — returns files for today only. Prefer list_daily_dates + list_daily_files."""
    return list_daily_files(log_root, pipeline, _today_ist())


def read_daily_file(
    log_root: str,
    pipeline: str,
    date: str,
    filename: str,
    tail: int = 500,
    level: str = None,
    accession: str = None,
    run_id: str = None,
    search: str = None,
) -> dict:
    """Read a specific daily rotation file and return the last `tail` matched lines."""
    path = _daily_dir(log_root, pipeline, date) / filename
    if not path.exists():
        return {"error": f"Log file '{filename}' not found for {pipeline}/{date}"}

    rotation = _rotation_index(filename, pipeline)
    meta = _file_meta(path)
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    matched = _filter_lines(raw_lines, level, accession, run_id, search)

    tail = max(1, min(tail, 5000))
    return {
        "pipeline": pipeline,
        "date": date,
        "filename": filename,
        "rotation": rotation,
        "tail": tail,
        "total_matched": len(matched),
        "size_bytes": meta["size_bytes"],
        "last_modified": meta["last_modified"],
        "lines": matched[-tail:],
    }


def read_rotated_file(
    log_root: str,
    pipeline: str,
    filename: str,
    tail: int = 500,
    level: str = None,
    accession: str = None,
    run_id: str = None,
    search: str = None,
    date: str = None,
) -> dict:
    """Read a rotation file. Uses today's daily folder if date is omitted."""
    return read_daily_file(
        log_root=log_root,
        pipeline=pipeline,
        date=date or _today_ist(),
        filename=filename,
        tail=tail,
        level=level,
        accession=accession,
        run_id=run_id,
        search=search,
    )


def list_trace_dates(log_root: str, pipeline: str) -> list:
    """Return dates (most recent first) that have trace files for a pipeline."""
    traces_dir = Path(log_root) / pipeline / "traces"
    if not traces_dir.exists():
        return []
    return sorted(
        [d.name for d in traces_dir.iterdir() if d.is_dir()],
        reverse=True,
    )


def list_trace_files(log_root: str, pipeline: str, date: str) -> list:
    """
    Return metadata for every per-accession trace file under
    logs/{pipeline}/traces/{date}/.
    """
    date_dir = Path(log_root) / pipeline / "traces" / date
    if not date_dir.exists():
        return []

    files = []
    for f in sorted(date_dir.iterdir()):
        if not f.is_file() or not f.name.endswith(".log"):
            continue
        # Expected filename: {accession}_{doc_type}_{run_id}.log
        stem = f.stem
        parts = stem.rsplit("_", 2)
        meta = _file_meta(f)
        meta.update({
            "filename": f.name,
            "accession": parts[0] if len(parts) >= 1 else stem,
            "doc_type":  parts[1] if len(parts) >= 2 else "-",
            "run_id":    parts[2] if len(parts) == 3 else "-",
        })
        files.append(meta)
    return files


def read_trace_file(log_root: str, pipeline: str, date: str, filename: str) -> dict:
    """Read a single per-accession trace file and return all parsed lines."""
    path = Path(log_root) / pipeline / "traces" / date / filename
    if not path.exists():
        return {"error": "Trace file not found"}

    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    parsed = []
    for raw in raw_lines:
        row = _parse_line(raw)
        parsed.append(row if row is not None else {"raw": raw})

    return {
        "pipeline": pipeline,
        "date": date,
        "filename": filename,
        "total_lines": len(parsed),
        "lines": parsed,
    }


def search_all_pipelines(
    log_root: str,
    accession: str = None,
    run_id: str = None,
    level: str = None,
    search: str = None,
    tail: int = 500,
) -> dict:
    """
    Search across ALL pipeline rolling logs simultaneously.
    Returns results grouped by pipeline.
    """
    root = Path(log_root)
    if not root.exists():
        return {"results": {}, "total_matched": 0}

    results = {}
    total = 0
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        pipeline = folder.name
        if _resolve_active_log(log_root, pipeline) is None:
            continue
        data = read_rolling_log(
            log_root=log_root,
            pipeline=folder.name,
            tail=tail,
            level=level,
            accession=accession,
            run_id=run_id,
            search=search,
        )
        if data.get("total_matched", 0) > 0:
            results[folder.name] = data
            total += data["total_matched"]

    return {"results": results, "total_matched": total}
