import logging
import logging.handlers
import threading
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()
_pipeline_handlers: dict = {}   # pipeline name → RotatingFileHandler
_trace_handlers: dict    = {}   # absolute file path → FileHandler

MAX_BYTES    = 5 * 1024 * 1024  # 5 MB per rolling log file
BACKUP_COUNT = 20               # keep up to 20 rotations (~100 MB per pipeline)

PIPELINE_FORMATTER = logging.Formatter(
    fmt=(
        "{asctime} | {levelname:<5} | pipeline={pipeline} | run_id={run_id} | "
        "accession={accession} | doc_type={doc_type} | {name}:{lineno} | {message}"
    ),
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
)


def _get_pipeline_handler(log_root: str, pipeline: str) -> logging.Handler:
    """Return (cached) RotatingFileHandler for logs/{pipeline}/{pipeline}.log."""
    if pipeline not in _pipeline_handlers:
        folder = Path(log_root) / pipeline
        folder.mkdir(parents=True, exist_ok=True)
        h = logging.handlers.RotatingFileHandler(
            folder / f"{pipeline}.log",
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        h.setFormatter(PIPELINE_FORMATTER)
        _pipeline_handlers[pipeline] = h
    return _pipeline_handlers[pipeline]


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

    1. logs/{pipeline}/{pipeline}.log
       Rolling file, max 5 MB, last 20 rotations kept.
       Contains all accessions for that pipeline (easy to tail / monitor).

    2. logs/{pipeline}/traces/{YYYY-MM-DD}/{accession}_{doc_type}_{run_id}.log
       Per-accession trace file — only created when accession context is set.
       Deleted by server cron after 30 days (see LOGGING_SYSTEM.md).

    Configure in Django LOGGING as:
        "pipeline_file": {
            "()": "core.dynamic_pipeline_handler.DynamicPipelineHandler",
            "log_root": LOG_ROOT,
            "filters": ["pipeline_context"],
        }
    """

    def __init__(self, log_root: str):
        super().__init__()
        self._log_root = log_root

    def emit(self, record: logging.LogRecord):
        pipeline  = getattr(record, "pipeline",  "app")
        accession = getattr(record, "accession", "-")
        doc_type  = getattr(record, "doc_type",  "UNKNOWN")
        run_id    = getattr(record, "run_id",    "-")

        with _lock:
            # 1. Always write to rolling pipeline log
            try:
                ph = _get_pipeline_handler(self._log_root, pipeline)
                ph.emit(record)
            except Exception:
                self.handleError(record)

            # 2. Write to per-accession trace file only when context is set
            if accession and accession != "-":
                try:
                    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
                    safe_acc = accession.replace("/", "-")
                    filename = f"{safe_acc}_{doc_type}_{run_id}.log"
                    trace_path = (
                        Path(self._log_root) / pipeline / "traces" / today / filename
                    )
                    th = _get_trace_handler(trace_path)
                    th.emit(record)
                except Exception:
                    self.handleError(record)
