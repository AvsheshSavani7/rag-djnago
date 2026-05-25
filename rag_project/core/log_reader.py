"""
Pure-Python log reading utilities for the Logs API.
No Django imports — safe to use anywhere.
"""

import re
from datetime import datetime, timezone
from pathlib import Path

# Matches lines produced by PIPELINE_FORMATTER in dynamic_pipeline_handler.py
LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
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


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def list_pipelines(log_root: str) -> list:
    """Return metadata for every pipeline folder that has a rolling log file."""
    root = Path(log_root)
    if not root.exists():
        return []
    result = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        log_file = folder / f"{folder.name}.log"
        if log_file.exists():
            meta = _file_meta(log_file)
            meta["pipeline"] = folder.name
            result.append(meta)
    return result


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
    log_file = Path(log_root) / pipeline / f"{pipeline}.log"
    if not log_file.exists():
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


def list_rotated_files(log_root: str, pipeline: str) -> list:
    """
    Return metadata for every rotation file under logs/{pipeline}/, newest first.

    Active file  : {pipeline}.log       → rotation index 0
    Rotated files: {pipeline}.log.1     → index 1 (most recent backup)
                   {pipeline}.log.2     → index 2
                   ...
    """
    pipeline_dir = Path(log_root) / pipeline
    if not pipeline_dir.exists():
        return []

    files = []
    active = pipeline_dir / f"{pipeline}.log"
    if active.exists():
        meta = _file_meta(active)
        meta.update({"filename": active.name, "rotation": 0})
        files.append(meta)

    for p in sorted(
        pipeline_dir.glob(f"{pipeline}.log.*"),
        key=lambda f: int(f.suffix.lstrip(".")) if f.suffix.lstrip(".").isdigit() else 999,
    ):
        idx = int(p.suffix.lstrip(".")) if p.suffix.lstrip(".").isdigit() else 999
        meta = _file_meta(p)
        meta.update({"filename": p.name, "rotation": idx})
        files.append(meta)

    return files


def read_rotated_file(
    log_root: str,
    pipeline: str,
    filename: str,
    tail: int = 500,
    level: str = None,
    accession: str = None,
    run_id: str = None,
    search: str = None,
) -> dict:
    """
    Read a specific rotation file and return the last `tail` matched lines.
    filename must be exactly {pipeline}.log or {pipeline}.log.N
    """
    path = Path(log_root) / pipeline / filename
    if not path.exists():
        return {"error": f"Rotated log file '{filename}' not found for pipeline '{pipeline}'"}

    suffix = path.suffix.lstrip(".")
    rotation = int(suffix) if suffix.isdigit() else 0

    meta = _file_meta(path)
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    matched = _filter_lines(raw_lines, level, accession, run_id, search)

    tail = max(1, min(tail, 5000))
    return {
        "pipeline": pipeline,
        "filename": filename,
        "rotation": rotation,
        "tail": tail,
        "total_matched": len(matched),
        "size_bytes": meta["size_bytes"],
        "last_modified": meta["last_modified"],
        "lines": matched[-tail:],
    }


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
        log_file = folder / f"{folder.name}.log"
        if not log_file.exists():
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
