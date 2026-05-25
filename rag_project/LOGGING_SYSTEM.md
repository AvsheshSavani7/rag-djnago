# Logging System — Design & Implementation Plan

## Overview

A structured, pipeline-aware logging system that makes every log line traceable by
`pipeline`, `run_id`, `accession_number`, and `doc_type`. Supports both a rolling
pipeline log (for monitoring) and per-accession trace files (for debugging individual
filings).

**Technology:** Python `stdlib logging` + `contextvars` (no third-party logging library)  
**Infrastructure:** Docker volume mount on Hostinger VPS, rotating files + cron cleanup  
**Impact:** ~120 new lines, ~10 lines changed. Zero changes to existing `logger.info()` calls.

---

## Problem Being Solved

Multiple threads run simultaneously for different accession numbers inside the same
process. Log lines from different accessions interleave, making it impossible to trace
the full story of a single filing.

**Before:**
```
INFO | Fetching HTML for 0001193125-26-126362
INFO | Fetching HTML for 0001104659-26-055221
INFO | EX-2.1 found           ← which accession?
INFO | S3 upload complete      ← which accession?
ERROR | Summary failed         ← which accession?
```

**After:**
```
INFO | pipeline=sec_8k | run_id=a8f91c | accession=0001193125-26-126362 | doc_type=8K  | Fetching HTML
INFO | pipeline=sec_8k | run_id=b3c82e | accession=0001104659-26-055221 | doc_type=8K  | Fetching HTML
INFO | pipeline=ex21   | run_id=a8f91c | accession=0001193125-26-126362 | doc_type=EX21 | EX-2.1 found
INFO | pipeline=ex21   | run_id=a8f91c | accession=0001193125-26-126362 | doc_type=EX21 | S3 upload complete
ERROR| pipeline=dma    | run_id=a8f91c | accession=0001193125-26-126362 | doc_type=EX21 | Summary failed
```

---

## Why `contextvars` (not `threading.local`)

When `_process_single_item` sets context for accession `ABC-123` and spawns a child
thread (`process_8k_document_async`), that child thread automatically inherits a
**copy** of the parent's context. No need to pass accession as a parameter to spawned
functions.

```
_process_single_item (accession=ABC-123)
    │  set_pipeline_context(accession="ABC-123")
    ├─ logger.info("Fetch HTML")         → stamped: accession=ABC-123  ✓
    └─ spawns: process_8k_document_async ← copy of context inherited
          ├─ logger.info("S3 upload")    → stamped: accession=ABC-123  ✓
          └─ spawns: generate_8k_summary ← copy inherited again
                └─ logger.info("Done")   → stamped: accession=ABC-123  ✓
```

`threading.local` would NOT propagate to child threads. `contextvars` does.

---

## Log Format

```
2026-05-25 17:10:42 | INFO  | pipeline=sec_8k | run_id=a8f91c | accession=0001193125-26-126362 | doc_type=EX21 | sec_rss_parser.process_feed_8k:625 | Pipeline started
```

| Field | Example | Source |
|-------|---------|--------|
| `asctime` | `2026-05-25 17:10:42` | Auto |
| `levelname` | `INFO` | Auto |
| `pipeline` | `sec_8k` | `ContextVar` |
| `run_id` | `a8f91c` | `uuid4().hex[:6]` per item |
| `accession` | `0001193125-26-126362` | `ContextVar` |
| `doc_type` | `EX21` | `ContextVar` |
| `name:lineno` | `process_feed_8k:625` | Auto (Python module + line) |
| `message` | `Pipeline started` | `logger.info(...)` call |

---

## Log Storage Structure

```
/var/log/rag/                                ← Docker volume mount (host: /opt/apps/django-app/logs)
│
├── sec_8k/
│   ├── sec_8k.log                           ← rolling, 5 MB max, all accessions
│   ├── sec_8k.log.1                         ← rotated
│   └── traces/
│       └── 2026-05-25/
│           ├── 0001193125-26-126362_8K_a8f91c.log       ← one accession, 8-K doc
│           ├── 0001193125-26-126362_EX21_a8f91c.log     ← same accession, EX-2.1 doc
│           └── 0001104659-26-055221_8K_b3c82e.log
│
├── sec_feed/
│   ├── sec_feed.log
│   └── traces/2026-05-25/
│
├── ex21/
│   ├── ex21.log
│   └── traces/2026-05-25/
│
├── sec_summary/
│   └── sec_summary.log
│
├── proxy/
│   ├── proxy.log
│   └── traces/2026-05-25/
│
├── proxy_comparison/
│   └── proxy_comparison.log
│
├── ten_k_ten_q/
│   ├── ten_k_ten_q.log
│   └── traces/2026-05-25/
│
├── covenant/
│   └── covenant.log
│
├── termination/
│   └── termination.log
│
├── dma/
│   ├── dma.log
│   └── traces/2026-05-25/
│
├── dma_summary/
│   ├── dma_summary.log
│   └── traces/2026-05-25/
│
├── mae/
│   └── mae.log
│
├── regeneration/
│   └── regeneration.log
│
├── rss/
│   ├── rss.log
│   └── traces/2026-05-25/
│
├── email/
│   └── email.log
│
└── app/
    └── app.log                              ← Django HTTP errors, unclassified
```

**Rules:**
- Rolling files: max 5 MB, keep last 20 rotations = max ~100 MB per pipeline
- Trace files: no size rotation (tiny per-accession files, usually 50–500 lines)
- Subfolders created automatically on first use (no manual mkdir needed)
- Old trace folders deleted by cron after 30 days

---

## Pipeline → Log Folder Mapping

| Pipeline | Log folder | Entry point file | doc_type values |
|----------|-----------|-----------------|-----------------|
| 8-K feed ingest | `sec_8k` | `process_feed_8k.py` | `8K`, `EX21`, `EX991` |
| SEC feed by CIK | `sec_feed` | `fetch_sec_feed_by_deal_cik.py` | form type e.g. `DEFM14A` |
| EX-2.1 document flow | `ex21` | `services.process_8k_document_async` | `EX21` |
| SEC summary (router) | `sec_summary` | `filing_router.route_and_summarize` | form type |
| 8-K post-DMA summary | `dma_summary` | `services.generate_8k_summary_async` | `EX21` |
| Proxy summary | `proxy` | `proxy_processor_helper.py` | proxy form type |
| Proxy comparison | `proxy_comparison` | `proxy_comparision/orchestrator.py` | proxy form type |
| 10-K / 10-Q | `ten_k_ten_q` | `tenK_tenQ_pipeline/orchestrator.py` | `10K`, `10Q` |
| Covenant analysis | `covenant` | `covenant_pipeline.py` (triggered from `document_processor`) | `EX21` |
| Termination analysis | `termination` | `termination_pipeline.py` (triggered from `document_processor`) | `EX21` |
| DMA document processing | `dma` | `document_processor/services.process_document` | `EX21` |
| DMA summary generation | `dma_summary` | `SummaryGenerationService.generate_summary_engine` | `EX21` |
| DMA extraction | `dma` | `summary_processor/dma_summary_processor.py` | `EX21` |
| Press release extraction | `sec_summary` | `summary_processor/press_release_processor.py` | `EX991` |
| MAE pipeline | `mae` | `MAE/run_full_pipeline.py` | `EX21` |
| Regeneration API | `regeneration` | `regeneration_pipeline.py` | varies |
| RSS newswire | `rss` | `rss_feeds/services.py` | `RSS` |
| Email sending | `email` | `utils_8k.send_webhook_notification` + others | varies |

---

## Summary Generator → Pipeline Mapping

| Summarizer | Pipeline log | AI Model |
|-----------|-------------|---------|
| `Eight_k_summary.summarize_8k_filing` | `sec_8k` | Claude Opus |
| `sec_summarizers/8k_summary.py` | `sec_summary` | Claude Opus |
| `sec_summarizers/991_summary.py` | `sec_summary` | Claude Opus |
| `sec_summarizers/filing_router.route_and_summarize` | `sec_summary` | Claude (varies by form type) |
| `sec_summarizers/PRNewswire_summary.py` | `rss` (when from RSS feed) or `sec_summary` | Claude Opus |
| `services.generate_8k_summary_async` + `SummaryGenerationService.generate_summary_engine` | `dma_summary` | GPT-5.2 (OpenAI) |
| `proxy_summary_service_v2.ProxySummaryServiceV2` | `proxy` | Claude + OpenAI |
| `proxy_comparision/report_writer.generate_full_summary` | `proxy_comparison` | Claude Sonnet/Opus |
| `tenK_tenQ_pipeline` (prefilter → score → assess → report) | `ten_k_ten_q` | Haiku → Sonnet → Sonnet |
| `summary_processor/dma_summary_processor.extract_from_dma_summary` | `dma` | Claude Haiku |
| `summary_processor/press_release_processor.extract_from_press_release` | `sec_summary` | Claude Haiku |

---

## Files to Create

### `rag_project/core/__init__.py`
Empty file — makes `core` a Python package.

### `rag_project/core/logging_context.py`
```python
from contextvars import ContextVar

_pipeline  = ContextVar("pipeline",  default="app")
_run_id    = ContextVar("run_id",    default="-")
_accession = ContextVar("accession", default="-")
_doc_type  = ContextVar("doc_type",  default="UNKNOWN")

def set_pipeline_context(pipeline: str, run_id: str = "-",
                         accession: str = "-", doc_type: str = "UNKNOWN"):
    _pipeline.set(pipeline)
    _run_id.set(run_id)
    _accession.set(accession)
    _doc_type.set(doc_type)

def get_pipeline()  -> str: return _pipeline.get()
def get_run_id()    -> str: return _run_id.get()
def get_accession() -> str: return _accession.get()
def get_doc_type()  -> str: return _doc_type.get()
```

### `rag_project/core/logging_filters.py`
```python
import logging
from core.logging_context import get_pipeline, get_run_id, get_accession, get_doc_type

class PipelineContextFilter(logging.Filter):
    """Stamps every LogRecord with pipeline context. No changes to logger.info() calls needed."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.pipeline  = get_pipeline()
        record.run_id    = get_run_id()
        record.accession = get_accession()
        record.doc_type  = get_doc_type()
        return True
```

### `rag_project/core/dynamic_pipeline_handler.py`
```python
import logging
import logging.handlers
import threading
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()
_pipeline_handlers: dict = {}
_trace_handlers: dict    = {}

PIPELINE_FORMATTER = logging.Formatter(
    fmt=(
        "{asctime} | {levelname:<5} | pipeline={pipeline} | run_id={run_id} | "
        "accession={accession} | doc_type={doc_type} | {name}:{lineno} | {message}"
    ),
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
)

MAX_BYTES    = 5 * 1024 * 1024
BACKUP_COUNT = 20


def _get_pipeline_handler(log_root: str, pipeline: str) -> logging.Handler:
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
    key = str(trace_path)
    if key not in _trace_handlers:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        h = logging.FileHandler(trace_path, encoding="utf-8")
        h.setFormatter(PIPELINE_FORMATTER)
        _trace_handlers[key] = h
    return _trace_handlers[key]


class DynamicPipelineHandler(logging.Handler):
    """
    Routes each record to:
      1. logs/{pipeline}/{pipeline}.log           — rotating, 5 MB
      2. logs/{pipeline}/traces/{date}/{accession}_{doc_type}_{run_id}.log
         — per-accession trace file, deleted by cron after 30 days
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
            ph = _get_pipeline_handler(self._log_root, pipeline)
            ph.emit(record)

            # 2. Write to per-accession trace file when context is set
            if accession and accession != "-":
                today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
                safe_acc = accession.replace("/", "-")
                filename = f"{safe_acc}_{doc_type}_{run_id}.log"
                trace_path = (
                    Path(self._log_root) / pipeline / "traces" / today / filename
                )
                th = _get_trace_handler(trace_path)
                th.emit(record)
```

### `rag_project/core/pipeline_logger.py`
```python
import uuid
from core.logging_context import set_pipeline_context

# Pipeline name constants — use these instead of plain strings
SEC_8K        = "sec_8k"
SEC_FEED      = "sec_feed"
EX21          = "ex21"
SEC_SUMMARY   = "sec_summary"
PROXY         = "proxy"
PROXY_COMP    = "proxy_comparison"
TEN_K_TEN_Q   = "ten_k_ten_q"
COVENANT      = "covenant"
TERMINATION   = "termination"
DMA           = "dma"
DMA_SUMMARY   = "dma_summary"
MAE           = "mae"
REGENERATION  = "regeneration"
RSS           = "rss"
EMAIL         = "email"


def start_pipeline(pipeline: str, accession: str = None,
                   doc_type: str = "UNKNOWN") -> str:
    """
    Call once at the start of processing each accession/item.
    All log lines in the current thread AND any threads spawned from it
    will automatically carry these context values.

    Returns run_id (6-char hex) so you can reference it in your own log lines.

    Usage:
        run_id = start_pipeline(SEC_8K, accession=accession_number, doc_type="8K")
        logger.info("Processing started, run_id=%s", run_id)
    """
    run_id = uuid.uuid4().hex[:6]
    set_pipeline_context(
        pipeline=pipeline,
        run_id=run_id,
        accession=accession or "-",
        doc_type=doc_type,
    )
    return run_id
```

---

## Files to Modify

### `rag_project/rag_project/settings.py`

**Replace** the entire `LOGGING = { ... }` block (lines ~251–289) with:

```python
import os

LOG_ROOT = os.environ.get("LOG_ROOT", str(BASE_DIR / "logs"))

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "pipeline_context": {
            "()": "core.logging_filters.PipelineContextFilter",
        }
    },
    "formatters": {
        "pipeline": {
            "()": "logging.Formatter",
            "format": (
                "{asctime} | {levelname:<5} | pipeline={pipeline} | run_id={run_id} | "
                "accession={accession} | doc_type={doc_type} | {name}:{lineno} | {message}"
            ),
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "style": "{",
        },
        "console": {
            "format": "{levelname:<5} | {name}:{lineno} | {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "console",
            "filters": ["pipeline_context"],
        },
        "pipeline_file": {
            "()": "core.dynamic_pipeline_handler.DynamicPipelineHandler",
            "log_root": LOG_ROOT,
            "filters": ["pipeline_context"],
        },
    },
    "root": {
        "handlers": ["console", "pipeline_file"],
        "level": "INFO",
    },
    "loggers": {
        "django":         {"handlers": ["console"], "level": "WARNING", "propagate": False},
        "django.request": {"handlers": ["console"], "level": "ERROR",   "propagate": False},
        "django.db":      {"handlers": [],          "level": "WARNING", "propagate": False},
        "mongoengine":    {"handlers": [],          "level": "WARNING", "propagate": False},
        "botocore":       {"handlers": [],          "level": "WARNING", "propagate": False},
        "urllib3":        {"handlers": [],          "level": "WARNING", "propagate": False},
        "anthropic":      {"handlers": [],          "level": "WARNING", "propagate": False},
        "openai":         {"handlers": [],          "level": "WARNING", "propagate": False},
    },
}
```

### `rag_project/sec_rss_parser/process_feed_8k.py`

**In `_process_single_item` (line ~604)** — add 3 lines after `accession_number` is computed:

```python
def _process_single_item(self, item_data):
    accession_number = item_data.get(
        'accession_number') or extract_accession_from_guid(item_data.get('guid'))

    # ── NEW: set pipeline context so all log lines in this thread + spawned threads carry it ──
    from core.pipeline_logger import start_pipeline, SEC_8K
    start_pipeline(SEC_8K, accession=accession_number, doc_type="8K")
    # ──────────────────────────────────────────────────────────────────────────────────────────

    lock_owner = None
    # ... rest unchanged
```

**When EX-2.1 is detected** (the point where `process_8k_document_helper` is called) — update context:

```python
# Add this immediately before the process_8k_document_helper call
from core.logging_context import set_pipeline_context, get_run_id
set_pipeline_context(pipeline="ex21", run_id=get_run_id(),
                     accession=accession_number, doc_type="EX21")
```

### `rag_project/sec_rss_parser/fetch_sec_feed_by_deal_cik.py`

**In `process_items` (line ~1285)** — add after `acc` is computed inside the for loop:

```python
acc = item_data.get("accession_number") or extract_accession_from_guid(
    item_data.get("guid"))

# ── NEW: set pipeline context per item ──
from core.pipeline_logger import start_pipeline
feed_form = (item_data.get("form_type") or "").strip().upper()
if feed_form in PROXY_FORM_TYPES:
    _pipeline_name = "proxy"
elif feed_form in TEN_K_TEN_Q_FORM_TYPES:
    _pipeline_name = "ten_k_ten_q"
else:
    _pipeline_name = "sec_feed"
start_pipeline(_pipeline_name, accession=acc, doc_type=feed_form or "UNKNOWN")
# ─────────────────────────────────────────
```

### `rag_project/rss_feeds/services.py`

**In `process_webhook_payload`** — add at the top of the `for item in items_new:` loop (line ~309):

```python
for item in items_new:
    # ── NEW: set pipeline context per RSS item ──
    from core.pipeline_logger import start_pipeline, RSS
    _item_id = (item.get("url") or "")[-40:].replace("/", "-")
    start_pipeline(RSS, accession=_item_id, doc_type="RSS")
    # ─────────────────────────────────────────────
    try:
        result = resolve_rss_item_flow(item, deals_record_string)
```

### `docker-compose.yml`

Add `volumes` and `LOG_ROOT` env var:

```yaml
services:
  django-app:
    build: .
    container_name: django_app
    restart: unless-stopped
    env_file:
      - .env
    ports:
      - "8000"
    volumes:                                      # ADD
      - /opt/apps/django-app/logs:/var/log/rag   # ADD
    environment:
      - PORT=8000
      - LOG_ROOT=/var/log/rag                    # ADD
```

### `.env`

Add one line:

```
LOG_ROOT=/var/log/rag
```

---

## VPS Setup (run once on Hostinger server)

```bash
# 1. Create log root on host (subfolders auto-created by the handler)
mkdir -p /opt/apps/django-app/logs
chmod 777 /opt/apps/django-app/logs

# 2. 30-day cleanup cron
cat > /etc/cron.d/rag-log-cleanup << 'EOF'
# Delete log files older than 30 days daily at 2 AM
0 2 * * * root find /opt/apps/django-app/logs -type f -name "*.log*" -mtime +30 -delete && find /opt/apps/django-app/logs -type d -empty -mtime +30 -delete
EOF
chmod 644 /etc/cron.d/rag-log-cleanup
```

---

## Useful Commands After Deployment

```bash
# Trace a single accession across all pipelines
grep -r "accession=0001193125-26-126362" /var/log/rag/ | sort

# Open the per-accession trace file directly
cat /var/log/rag/sec_8k/traces/2026-05-25/0001193125-26-126362_EX21_a8f91c.log

# Watch live 8-K processing
tail -f /var/log/rag/sec_8k/sec_8k.log

# Watch live EX-2.1 + DMA processing
tail -f /var/log/rag/ex21/ex21.log /var/log/rag/dma/dma.log

# See all ERRORs today across all pipelines
grep "ERROR" /var/log/rag/*/  *.log | sort | tail -100

# Count accessions processed today per pipeline
for f in /var/log/rag/*/  *.log; do echo "$f: $(grep -c 'step=start' $f 2>/dev/null || echo 0)"; done

# See proxy comparison activity
cat /var/log/rag/proxy_comparison/proxy_comparison.log | tail -200

# See 10-K/10-Q pipeline activity
tail -f /var/log/rag/ten_k_ten_q/ten_k_ten_q.log
```

---

## Implementation Checklist

### Phase 1 — Core infrastructure
- [ ] Create `rag_project/core/__init__.py`
- [ ] Create `rag_project/core/logging_context.py`
- [ ] Create `rag_project/core/logging_filters.py`
- [ ] Create `rag_project/core/dynamic_pipeline_handler.py`
- [ ] Create `rag_project/core/pipeline_logger.py`
- [ ] Update `rag_project/rag_project/settings.py` — replace LOGGING block
- [ ] Add `LOG_ROOT=/var/log/rag` to `.env`
- [ ] Update `docker-compose.yml` — add volumes + LOG_ROOT env

### Phase 2 — Entry point wiring (highest value, 3 files)
- [ ] `sec_rss_parser/process_feed_8k.py` — `_process_single_item`: add `start_pipeline(SEC_8K, ...)`
- [ ] `sec_rss_parser/process_feed_8k.py` — EX-2.1 detection point: update context to `ex21`
- [ ] `sec_rss_parser/fetch_sec_feed_by_deal_cik.py` — `process_items` loop: add `start_pipeline`
- [ ] `rss_feeds/services.py` — `process_webhook_payload` loop: add `start_pipeline(RSS, ...)`

### Phase 3 — Secondary pipelines (can be done incrementally)
- [ ] `sec_rss_parser/services.py` — `generate_8k_summary_async`: add `start_pipeline(DMA_SUMMARY, ...)`
- [ ] `sec_rss_parser/proxy_processor_helper.py` — `process_sec_document_for_filing_summary`: add `start_pipeline(PROXY, ...)`
- [ ] `sec_rss_parser/proxy_comparision/orchestrator.py` — `run_comparison`: add `start_pipeline(PROXY_COMP, ...)`
- [ ] `sec_rss_parser/tenK_tenQ_pipeline/orchestrator.py` — `run_pipeline`: add `start_pipeline(TEN_K_TEN_Q, ...)`
- [ ] `document_processor/services.py` — `process_document`: add `start_pipeline(DMA, ...)`
- [ ] `document_processor/services.py` — `generate_summary_engine`: add `start_pipeline(DMA_SUMMARY, ...)`
- [ ] `document_processor/services.py` — `_run_covenant_analysis_pipeline`: add `start_pipeline(COVENANT, ...)`
- [ ] `document_processor/services.py` — `_run_termination_analysis_pipeline`: add `start_pipeline(TERMINATION, ...)`
- [ ] `document_processor/MAE/run_full_pipeline.py` — `run_pipeline_for_deal_id`: add `start_pipeline(MAE, ...)`

### Phase 4 — VPS setup
- [ ] SSH to Hostinger VPS
- [ ] Run `mkdir -p /opt/apps/django-app/logs && chmod 777 /opt/apps/django-app/logs`
- [ ] Create `/etc/cron.d/rag-log-cleanup` with 30-day cleanup command
- [ ] `docker compose down && docker compose up -d` to apply volume mount

---

## Notes

- **`log_and_print()` functions** in `services.py` and `utils_8k.py` do not need to change — they call `logger.info()` which goes through the same filter chain.
- **`print()` calls** in pipeline files will NOT appear in log files (they go to stdout/docker logs). Replace gradually with `logger.info()` as needed.
- **`Eight_k_summary.py`** vs **`sec_summarizers/8k_summary.py`** are two separate files with similar names. The former is used by `process_feed_8k.py`; the latter is used by `filing_router`. Both will write to `sec_8k` and `sec_summary` respectively based on which pipeline calls them.
- **Local development**: `LOG_ROOT` defaults to `BASE_DIR/logs` (inside the repo). Add `logs/` to `.gitignore`.
- **Thread safety**: The `_lock` in `DynamicPipelineHandler` protects handler creation. Individual `RotatingFileHandler` and `FileHandler` instances are thread-safe by default in Python's `logging` module.

---

---

# Logs API

HTTP API that exposes the pipeline log files so the admin frontend can browse,
filter, and read logs without SSH access.

## Files Created

| File | Purpose |
|------|---------|
| `rag_project/core/log_reader.py` | Pure-Python file-reading logic — no Django imports |
| `rag_project/logs_api/__init__.py` | Package marker |
| `rag_project/logs_api/apps.py` | Django app config |
| `rag_project/logs_api/views.py` | 6 API view classes |
| `rag_project/logs_api/urls.py` | URL routing |

## Files Modified

| File | Change |
|------|--------|
| `rag_project/rag_project/settings.py` | Added `"logs_api"` to `INSTALLED_APPS` |
| `rag_project/rag_project/urls.py` | Added `path('api/logs/', include('logs_api.urls'))` |

---

## Architecture

```
Frontend (React admin)
    │
    │  HTTP GET
    ▼
logs_api/views.py           ← Django REST views (6 endpoints)
    │
    ▼
core/log_reader.py          ← Pure Python: read/parse/filter log files
    │
    ▼
/var/log/rag/               ← Docker volume (host: /opt/apps/django-app/logs)
    ├── sec_8k/sec_8k.log
    ├── sec_8k/traces/2026-05-25/0001193125-26-126362_EX21_a8f91c.log
    └── ...
```

`log_reader.py` uses a regex to parse every line produced by `PIPELINE_FORMATTER`:

```
2026-05-25 17:10:42 | INFO  | pipeline=sec_8k | run_id=a8f91c | accession=0001193125-26-126362 | doc_type=EX21 | sec_rss_parser.process_feed_8k:625 | Processing started
```

---

## API Endpoints

### 1. List all pipelines
```
GET /api/logs/pipelines/
```

Returns every pipeline folder that has a rolling log file, with file sizes.

**Response:**
```json
{
  "log_root": "/var/log/rag",
  "pipelines": [
    {"pipeline": "app",    "size_bytes": 172032, "last_modified": "2026-05-25T06:07:17Z"},
    {"pipeline": "sec_8k", "size_bytes": 48210,  "last_modified": "2026-05-25T11:30:00Z"},
    {"pipeline": "ex21",   "size_bytes": 12500,  "last_modified": "2026-05-25T11:31:00Z"}
  ]
}
```

---

### 2. Search across ALL pipelines
```
GET /api/logs/search/?accession=XXX&level=ERROR&run_id=a8f91c&search=text&tail=500
```

Searches every pipeline rolling log simultaneously. Best for tracking one accession
end-to-end across `sec_8k → ex21 → dma → dma_summary`.

**Query params (all optional):**

| Param | Example | Description |
|-------|---------|-------------|
| `accession` | `0001193125-26-126362` | Substring match on accession field |
| `run_id` | `a8f91c` | Exact run_id match |
| `level` | `ERROR` | INFO / WARNING / ERROR / DEBUG |
| `search` | `S3+upload` | Case-insensitive substring in full line |
| `tail` | `500` | Max lines per pipeline to scan (default 500, max 5000) |

**Response:**
```json
{
  "total_matched": 14,
  "results": {
    "sec_8k": {
      "total_matched": 3,
      "lines": [{"ts": "...", "level": "INFO", "accession": "0001193125-26-126362", ...}]
    },
    "ex21": {
      "total_matched": 8,
      "lines": [...]
    },
    "dma_summary": {
      "total_matched": 3,
      "lines": [...]
    }
  }
}
```

---

### 3. Read rolling pipeline log (with filters)
```
GET /api/logs/<pipeline>/stream/?tail=200&level=ERROR&accession=XXX&run_id=a8f91c&search=text
```

Read the rolling log for a single pipeline. All query params are optional and ANDed.

**Query params:**

| Param | Default | Max | Description |
|-------|---------|-----|-------------|
| `tail` | `200` | `2000` | Last N matched lines to return |
| `level` | — | — | INFO / WARNING / ERROR / DEBUG |
| `accession` | — | — | Substring match on accession field |
| `run_id` | — | — | Exact match |
| `search` | — | — | Case-insensitive substring in full line |

**Response:**
```json
{
  "pipeline": "sec_8k",
  "tail": 200,
  "total_matched": 47,
  "file_size_bytes": 48210,
  "last_modified": "2026-05-25T11:30:00Z",
  "lines": [
    {
      "ts":        "2026-05-25 11:30:42",
      "level":     "INFO",
      "pipeline":  "sec_8k",
      "run_id":    "a8f91c",
      "accession": "0001193125-26-126362",
      "doc_type":  "8K",
      "module":    "sec_rss_parser.process_feed_8k:625",
      "message":   "Processing started"
    }
  ]
}
```

---

### 4. List trace dates for a pipeline
```
GET /api/logs/<pipeline>/traces/
```

Returns the list of dates that have per-accession trace files, most recent first.

**Response:**
```json
{
  "pipeline": "sec_8k",
  "dates": ["2026-05-25", "2026-05-24", "2026-05-23"]
}
```

---

### 5. List trace files for a pipeline+date
```
GET /api/logs/<pipeline>/traces/<date>/
```

Returns one entry per trace file under `logs/{pipeline}/traces/{date}/`.

**Response:**
```json
{
  "pipeline": "sec_8k",
  "date": "2026-05-25",
  "count": 3,
  "files": [
    {
      "filename":      "0001193125-26-126362_EX21_a8f91c.log",
      "accession":     "0001193125-26-126362",
      "doc_type":      "EX21",
      "run_id":        "a8f91c",
      "size_bytes":    4210,
      "last_modified": "2026-05-25T11:31:00Z"
    },
    {
      "filename":      "0001104659-26-055221_8K_b3c82e.log",
      "accession":     "0001104659-26-055221",
      "doc_type":      "8K",
      "run_id":        "b3c82e",
      "size_bytes":    1850,
      "last_modified": "2026-05-25T11:28:00Z"
    }
  ]
}
```

---

### 6. Read one trace file (full accession story)
```
GET /api/logs/<pipeline>/traces/<date>/<filename>/
```

Returns every log line for that single accession, in order. This is the
complete timeline: `sec_8k → ex21 → dma → dma_summary` in one view.

**Response:**
```json
{
  "pipeline":    "sec_8k",
  "date":        "2026-05-25",
  "filename":    "0001193125-26-126362_EX21_a8f91c.log",
  "total_lines": 47,
  "lines": [
    {"ts": "2026-05-25 11:30:42", "level": "INFO",  "pipeline": "sec_8k",    "run_id": "a8f91c", "accession": "0001193125-26-126362", "doc_type": "8K",   "module": "process_feed_8k:625",   "message": "Processing started"},
    {"ts": "2026-05-25 11:30:44", "level": "INFO",  "pipeline": "sec_8k",    "run_id": "a8f91c", "accession": "0001193125-26-126362", "doc_type": "8K",   "module": "process_feed_8k:668",   "message": "Fetching HTML"},
    {"ts": "2026-05-25 11:30:47", "level": "INFO",  "pipeline": "ex21",      "run_id": "a8f91c", "accession": "0001193125-26-126362", "doc_type": "EX21", "module": "process_feed_8k:792",   "message": "EX-2.1 found"},
    {"ts": "2026-05-25 11:30:49", "level": "INFO",  "pipeline": "ex21",      "run_id": "a8f91c", "accession": "0001193125-26-126362", "doc_type": "EX21", "module": "services:910",           "message": "8-K document processing started"},
    {"ts": "2026-05-25 11:31:05", "level": "INFO",  "pipeline": "dma",       "run_id": "a8f91c", "accession": "0001193125-26-126362", "doc_type": "EX21", "module": "document_processor:279", "message": "Processing document"},
    {"ts": "2026-05-25 11:31:22", "level": "INFO",  "pipeline": "dma",       "run_id": "a8f91c", "accession": "0001193125-26-126362", "doc_type": "EX21", "module": "document_processor:567", "message": "Embedding completed"},
    {"ts": "2026-05-25 11:31:25", "level": "INFO",  "pipeline": "dma_summary","run_id": "a8f91c", "accession": "0001193125-26-126362", "doc_type": "EX21", "module": "services:427",           "message": "Summary generated"},
    {"ts": "2026-05-25 11:31:26", "level": "INFO",  "pipeline": "dma_summary","run_id": "a8f91c", "accession": "0001193125-26-126362", "doc_type": "EX21", "module": "services:452",           "message": "Email sent"}
  ]
}
```

---

## Security Notes

- All URL segments (`pipeline`, `date`, `filename`) are validated against
  `^[\w\-\.]+$` to prevent path traversal attacks.
- `AllowAny` permission is used — add JWT auth if the admin frontend requires it
  (wrap views with `IsAuthenticated` from `rest_framework.permissions`).
- Log files may contain company names, accession numbers, and error messages.
  Do not expose this API publicly without authentication.

---

## Suggested Frontend Usage (React admin)

```
Sidebar
└── Logs
    ├── [Pipelines list]  ← GET /api/logs/pipelines/
    │     sec_8k  172 KB  last: 11:30
    │     ex21     12 KB  last: 11:31
    │     proxy     8 KB  last: 10:45
    │     ...
    │
    ├── 🔍 Search box     ← GET /api/logs/search/?accession=XXX
    │     accession / run_id / keyword
    │
    └── [Pipeline detail]  (click a pipeline)
          ├── Rolling log viewer  ← GET /api/logs/sec_8k/stream/?tail=200&level=ERROR
          │     Filters: Level ▼  Accession [____]  Search [____]
          │     Auto-refresh every 10s
          │
          └── Trace browser
                [Date picker]    ← GET /api/logs/sec_8k/traces/
                [File list]      ← GET /api/logs/sec_8k/traces/2026-05-25/
                [Trace viewer]   ← GET /api/logs/sec_8k/traces/2026-05-25/filename.log/
                  Shows full timeline of one accession in a table
```

---

## Quick Test (from VPS or local)

```bash
# After deploy: docker compose down && docker compose up -d

# List all pipeline folders
curl https://django.arbintel.cloud/api/logs/pipelines/

# Search everywhere for one accession (tracks it across all pipelines)
curl "https://django.arbintel.cloud/api/logs/search/?accession=0001193125-26-126362"

# Get last 50 errors in sec_8k
curl "https://django.arbintel.cloud/api/logs/sec_8k/stream/?level=ERROR&tail=50"

# Get all app-level logs (filtering/skipping activity)
curl "https://django.arbintel.cloud/api/logs/app/stream/?tail=100"

# List dates with traces in sec_8k
curl "https://django.arbintel.cloud/api/logs/sec_8k/traces/"

# List trace files for today
curl "https://django.arbintel.cloud/api/logs/sec_8k/traces/2026-05-25/"

# Read full story of one accession
curl "https://django.arbintel.cloud/api/logs/sec_8k/traces/2026-05-25/0001193125-26-126362_EX21_a8f91c.log/"
```
