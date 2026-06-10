import logging
import logging.handlers
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

_lock = threading.Lock()
# (pipeline, date) → RotatingFileHandler
_pipeline_handlers: dict = {}
_trace_handlers: dict = {}   # absolute file path → FileHandler

MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
# No practical cap within a day; old date folders removed by VPS cron after 7 days
BACKUP_COUNT = 9999

_IST = timezone(timedelta(hours=5, minutes=30))


def _today_ist() -> str:
    return datetime.now(tz=_IST).strftime("%Y-%m-%d")


class _ISTFormatter(logging.Formatter):
    """Logging formatter that always stamps times in IST (UTC+5:30)."""

    def formatTime(self, record: logging.LogRecord, datefmt: str = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=_IST)
        return dt.strftime(datefmt or "%Y-%m-%d %I:%M:%S %p")


PIPELINE_FORMATTER = _ISTFormatter(
    fmt=(
        "{asctime} | {levelname:<5} | pipeline={pipeline} | run_id={run_id} | "
        "accession={accession} | doc_type={doc_type} | {name}:{lineno} | {message}"
    ),
    datefmt="%Y-%m-%d %I:%M:%S %p",
    style="{",
)


def _get_pipeline_handler(log_root: str, pipeline: str) -> logging.Handler:
    """Return (cached) RotatingFileHandler for logs/{pipeline}/daily/{date}/{pipeline}.log."""
    today = _today_ist()
    key = (pipeline, today)
    if key not in _pipeline_handlers:
        folder = Path(log_root) / pipeline / "daily" / today
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError):
            # Fall back to a local logs/ dir beside manage.py when the
            # configured log_root is not writable (e.g. local dev without /var/log/rag).
            import django.conf as _dc
            fallback_root = getattr(_dc.settings, "BASE_DIR", Path(__file__).resolve().parents[1]) / "logs"
            folder = Path(fallback_root) / pipeline / "daily" / today
            folder.mkdir(parents=True, exist_ok=True)
        h = logging.handlers.RotatingFileHandler(
            folder / f"{pipeline}.log",
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        h.setFormatter(PIPELINE_FORMATTER)
        _pipeline_handlers[key] = h
    return _pipeline_handlers[key]


def _get_trace_handler(trace_path: Path) -> logging.Handler:
    """Return (cached) FileHandler for a per-accession trace file."""
    key = str(trace_path)
    if key not in _trace_handlers:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        h = logging.FileHandler(trace_path, encoding="utf-8")
        h.setFormatter(PIPELINE_FORMATTER)
        _trace_handlers[key] = h
    return _trace_handlers[key]


class DynamicPipelineHandler(logging.Handler):
    """
    Single logging handler that routes every record to two destinations:

    1. logs/{pipeline}/daily/{YYYY-MM-DD}/{pipeline}.log
       Size-based rotation (10 MB) within each IST day — no file-count cap per day.
       Whole daily/ folders older than 7 days are deleted by VPS cron.

    2. logs/{pipeline}/traces/{YYYY-MM-DD}/{accession}_{doc_type}_{run_id}.log
       Per-accession trace file — only created when accession context is set.
       Deleted by server cron after 30 days (see LOGGING_SYSTEM.md).

    VPS cron (7-day daily cleanup):
        find /opt/apps/logs/django/*/daily -mindepth 1 -maxdepth 1 -type d -mtime +7 -exec rm -rf {} +
    """

    def __init__(self, log_root: str):
        super().__init__()
        self._log_root = log_root

    def emit(self, record: logging.LogRecord):
        pipeline = getattr(record, "pipeline",  "app")
        accession = getattr(record, "accession", "-")
        doc_type = getattr(record, "doc_type",  "UNKNOWN")
        run_id = getattr(record, "run_id",    "-")

        with _lock:
            # 1. Always write to daily rolling pipeline log
            try:
                ph = _get_pipeline_handler(self._log_root, pipeline)
                ph.emit(record)
            except Exception:
                self.handleError(record)

            # 2. Write to per-accession trace file only when context is set
            if accession and accession != "-":
                try:
                    today = _today_ist()
                    safe_acc = accession.replace("/", "-")
                    filename = f"{safe_acc}_{doc_type}_{run_id}.log"
                    trace_path = (
                        Path(self._log_root) / pipeline /
                        "traces" / today / filename
                    )
                    th = _get_trace_handler(trace_path)
                    th.emit(record)
                except Exception:
                    self.handleError(record)
