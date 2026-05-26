# Logging System — Reference

Structured, pipeline-aware logging for the Django RAG backend. Every log line is
traceable by `pipeline`, `run_id`, `accession`, and `doc_type`. Supports daily
rolling pipeline logs (for monitoring) and per-accession trace files (for debugging
individual filings).

**Technology:** Python `stdlib logging` + `contextvars`  
**Timezone:** All log timestamps are **IST (UTC+5:30)** with **12-hour AM/PM** format  
**Infrastructure:** Docker volume on Hostinger VPS + cron cleanup on host

---

## Problem Being Solved

Multiple threads process different accessions simultaneously. Without context, log
lines interleave and you cannot tell which accession a message belongs to.

**After (every line stamped):**
```
2026-05-26 09:45:11 AM | INFO  | pipeline=sec_8k | run_id=a8f91c | accession=0001193125-26-126362 | doc_type=8K   | process_feed_8k:625 | Processing started
2026-05-26 09:45:47 AM | INFO  | pipeline=ex21   | run_id=a8f91c | accession=0001193125-26-126362 | doc_type=EX21 | services:910        | EX-2.1 found
2026-05-26 09:46:05 AM | ERROR | pipeline=dma    | run_id=a8f91c | accession=0001193125-26-126362 | doc_type=EX21 | services:427        | Summary failed
```

---

## Why `contextvars`

Child threads inherit the parent's pipeline context automatically. No need to pass
accession as a parameter to spawned functions. `threading.local` would **not**
propagate to child threads.

Call `start_pipeline()` once per accession/item at the entry point:

```python
from core.pipeline_logger import start_pipeline, SEC_8K

run_id = start_pipeline(SEC_8K, accession=accession_number, doc_type="8K")
logger.info("Processing started")
```

---

## Log Format

```
2026-05-26 09:45:11 AM | INFO  | pipeline=sec_8k | run_id=a8f91c | accession=0001193125-26-126362 | doc_type=EX21 | sec_rss_parser.process_feed_8k:625 | Pipeline started
```

| Field | Example | Source |
|-------|---------|--------|
| `asctime` | `2026-05-26 09:45:11 AM` | IST, 12-hour AM/PM (`_ISTFormatter`) |
| `levelname` | `INFO` | Auto |
| `pipeline` | `sec_8k` | `ContextVar` |
| `run_id` | `a8f91c` | `uuid4().hex[:6]` per item |
| `accession` | `0001193125-26-126362` | `ContextVar` (RSS uses URL slug) |
| `doc_type` | `EX21` | `ContextVar` |
| `name:lineno` | `process_feed_8k:625` | Auto |
| `message` | `Pipeline started` | `logger.info(...)` |

---

## Log Storage Structure

```
Host:  /opt/apps/logs/django/
Container: /var/log/rag/

/opt/apps/logs/django/
│
├── app/
│   ├── daily/                               ← daily rolling logs (IST date folders)
│   │   ├── 2026-05-26/
│   │   │   ├── app.log                      ← active file for today
│   │   │   ├── app.log.1                    ← rotated at 10 MB
│   │   │   ├── app.log.2
│   │   │   └── ...                          ← unlimited rotations per day
│   │   ├── 2026-05-25/
│   │   │   ├── app.log
│   │   │   └── app.log.1
│   │   └── ...                              ← kept 7 days, then deleted by cron
│   └── traces/                              ← per-accession (unchanged)
│       └── 2026-05-26/
│           └── nsic-psychiatry-in-kri-c4352466_RSS_a8f91c.log
│
├── rss/
│   ├── daily/
│   │   └── 2026-05-26/
│   │       ├── rss.log
│   │       └── rss.log.1
│   └── traces/
│       └── 2026-05-26/
│
├── sec_8k/
│   ├── daily/2026-05-26/sec_8k.log
│   └── traces/2026-05-26/0001193125-26-126362_8K_a8f91c.log
│
└── ... (one folder per pipeline)
```

### Two parallel write destinations

Every `logger.info()` call goes through `DynamicPipelineHandler.emit()`:

```
logger.info(...)
        │
        ▼
DynamicPipelineHandler
        │
        ├──→ {pipeline}/daily/{IST-date}/{pipeline}.log     (rolling, all accessions mixed)
        │
        └──→ {pipeline}/traces/{IST-date}/{accession}_{doc_type}_{run_id}.log
             (only when accession context is set — one accession per file)
```

---

## Retention & Rotation Rules

| Type | Location | Max file size | File count cap | Retention |
|------|----------|---------------|----------------|-----------|
| **Daily rolling log** | `{pipeline}/daily/{date}/{pipeline}.log` | **10 MB** per file | **Unlimited** per day (`.1`, `.2`, …) | **7 days** (whole date folder deleted by cron) |
| **Trace file** | `{pipeline}/traces/{date}/{accession}_{doc_type}_{run_id}.log` | No rotation (small files) | One file per accession run | **30 days** (cron) |

- At **midnight IST**, logging switches to a new `daily/{new-date}/` folder automatically.
- Handler cache key is `(pipeline, date)` — new `RotatingFileHandler` per pipeline per day.
- `BACKUP_COUNT = 9999` in code = no practical rotation limit within a single day.

---

## Pipeline → Log Folder Mapping

| Pipeline | Log folder | Entry point | doc_type values |
|----------|-----------|-------------|-----------------|
| 8-K feed ingest | `sec_8k` | `process_feed_8k.py` | `8K`, `EX21`, `EX991` |
| SEC feed by CIK | `sec_feed` | `fetch_sec_feed_by_deal_cik.py` | form type e.g. `DEFM14A` |
| EX-2.1 document flow | `ex21` | `services.process_8k_document_async` | `EX21` |
| SEC summary (router) | `sec_summary` | `filing_router.route_and_summarize` | form type |
| 8-K post-DMA summary | `dma_summary` | `services.generate_8k_summary_async` | `EX21` |
| Proxy summary | `proxy` | `proxy_processor_helper.py` | proxy form type |
| Proxy comparison | `proxy_comparison` | `proxy_comparision/orchestrator.py` | proxy form type |
| 10-K / 10-Q | `ten_k_ten_q` | `tenK_tenQ_pipeline/orchestrator.py` | `10K`, `10Q` |
| Covenant analysis | `covenant` | `document_processor/services.py` | `EX21` |
| Termination analysis | `termination` | `document_processor/services.py` | `EX21` |
| DMA document processing | `dma` | `document_processor/services.py` | `EX21` |
| DMA summary generation | `dma_summary` | `SummaryGenerationService` | `EX21` |
| MAE pipeline | `mae` | `MAE/run_full_pipeline.py` | `EX21` |
| RSS newswire | `rss` | `rss_feeds/services.py` | `RSS`, `RSS_FLOW2` |
| Unclassified / pre-context | `app` | Default when `start_pipeline()` not called | `UNKNOWN` |

**RSS accession IDs:** Derived from the last URL path segment, sanitized
(non-word chars → `-`, max 50 chars). Example:
`nsic-psychiatry-in-kristinehamn-c4352466`

---

## Core Files

| File | Purpose |
|------|---------|
| `core/logging_context.py` | `ContextVar` storage + getters/setters |
| `core/logging_filters.py` | `PipelineContextFilter` — stamps every `LogRecord` |
| `core/dynamic_pipeline_handler.py` | Routes logs to `daily/` + `traces/`; IST formatter |
| `core/pipeline_logger.py` | Pipeline name constants + `start_pipeline()` |
| `core/log_reader.py` | Pure-Python log reading for the API |
| `logs_api/` | Django REST API (`views.py`, `urls.py`) |

Configured in `rag_project/settings.py` → `LOGGING` dict + `LOG_ROOT`.

---

## Docker & Environment

### `docker-compose.yml`

```yaml
volumes:
  # Host path: /opt/apps/logs/django  →  container path: /var/log/rag
  - /opt/apps/logs/django:/var/log/rag
environment:
  - LOG_ROOT=/var/log/rag
```

### `.env`

```
LOG_ROOT=/var/log/rag
```

### Local development

`LOG_ROOT` defaults to `BASE_DIR/logs`. Add `logs/` to `.gitignore`.

---

## VPS Setup (run once)

```bash
# Create log root on host
mkdir -p /opt/apps/logs/django
chmod 777 /opt/apps/logs/django

# Deploy and restart
cd /opt/apps/django-app
docker compose down && docker compose up -d
```

Subfolders (`daily/`, `traces/`) are created automatically on first log write.

---

## Cron Jobs (Hostinger VPS)

Create `/etc/cron.d/rag-log-cleanup`:

```bash
# RAG log cleanup — runs daily at 2 AM IST (server time may be UTC; adjust hour if needed)
#
# 1. Delete daily/ date folders older than 7 days
# 2. Delete trace .log files older than 30 days
# 3. Remove empty trace date directories

0 2 * * * root for dir in /opt/apps/logs/django/*/daily; do [ -d "$dir" ] && find "$dir" -mindepth 1 -maxdepth 1 -type d -mtime +7 -exec rm -rf {} +; done
5 2 * * * root find /opt/apps/logs/django/*/traces -type f -name "*.log" -mtime +30 -delete
10 2 * * * root find /opt/apps/logs/django/*/traces -type d -empty -mtime +30 -delete
```

Apply:

```bash
chmod 644 /etc/cron.d/rag-log-cleanup
```

### Why the loop for daily cleanup?

This command **fails silently-safe** when no `daily/` folder exists yet:

```bash
# BAD — errors if no daily/ folders exist yet:
find /opt/apps/logs/django/*/daily -mindepth 1 ...

# GOOD — skips pipelines without daily/:
for dir in /opt/apps/logs/django/*/daily; do
  [ -d "$dir" ] && find "$dir" -mindepth 1 -maxdepth 1 -type d -mtime +7 -exec rm -rf {} +
done
```

### Retention summary

| What | Cron rule | Effect |
|------|-----------|--------|
| `daily/2026-05-19/` folder | `-mtime +7` on date dirs | Entire day deleted (all `app.log`, `.1`, `.2`, …) |
| `traces/.../file.log` | `-mtime +30` on files | Individual trace files deleted |
| Empty trace dirs | `-type d -empty -mtime +30` | Cleanup after file deletion |

---

## Migrating Old Logs (pre-daily layout)

After deploying the new code, old flat files may still exist at the pipeline root:

```
/opt/apps/logs/django/app/app.log
/opt/apps/logs/django/app/app.log.1
...
```

New logs write to `app/daily/{today}/app.log`. Migrate old files **once** after deploy:

```bash
LOG_ROOT=/opt/apps/logs/django
ARCHIVE_DATE=$(TZ=Asia/Kolkata date +%Y-%m-%d)

for pipeline in app rss sec_8k; do   # add other pipeline names as needed
  dir="$LOG_ROOT/$pipeline"
  [ -d "$dir" ] || continue
  mkdir -p "$dir/daily/$ARCHIVE_DATE"
  for f in "$dir/${pipeline}.log"*; do
    [ -f "$f" ] || continue
    echo "Moving $f -> $dir/daily/$ARCHIVE_DATE/"
    mv "$f" "$dir/daily/$ARCHIVE_DATE/"
  done
done
```

Verify:

```bash
ls -la /opt/apps/logs/django/app/daily/$(TZ=Asia/Kolkata date +%Y-%m-%d)/
```

**Note:** Old files in one archive folder may span multiple calendar days. For
perfect day-splitting you would need a script that parses line timestamps.

Legacy flat `{pipeline}.log` at pipeline root is still readable by the API as a
fallback until removed.

---

## Useful VPS Commands

```bash
LOG=/opt/apps/logs/django
TODAY=$(TZ=Asia/Kolkata date +%Y-%m-%d)

# List pipeline folders
ls -lah $LOG/

# Watch today's live app log
tail -f $LOG/app/daily/$TODAY/app.log

# List rotation files for today
ls -lah $LOG/app/daily/$TODAY/

# List all daily dates for app
ls $LOG/app/daily/

# Trace a single accession across all files
grep -r "accession=0001193125-26-126362" $LOG/ | sort

# Read one accession trace file
cat $LOG/sec_8k/traces/$TODAY/0001193125-26-126362_EX21_a8f91c.log

# All ERRORs in today's app log
grep "ERROR" $LOG/app/daily/$TODAY/app.log | tail -50

# Disk usage per pipeline
du -sh $LOG/*/
```

---

## Logs API

HTTP API for the admin frontend. Base URL: `https://django.arbintel.cloud/api/logs/`

### Architecture

```
React admin  →  logs_api/views.py  →  core/log_reader.py  →  /var/log/rag/
```

### Endpoint summary

| Method | URL | Purpose |
|--------|-----|---------|
| GET | `/api/logs/pipelines/` | List all pipelines with active log metadata |
| GET | `/api/logs/search/` | Search today's active log across all pipelines |
| GET | `/api/logs/<pipeline>/stream/` | Read **today's** active daily log (structured JSON) |
| GET | `/api/logs/<pipeline>/rotated/` | List **dates** under `daily/` |
| GET | `/api/logs/<pipeline>/rotated/<date>/` | List rotation files for one day |
| GET | `/api/logs/<pipeline>/rotated/<date>/<filename>/` | Read one daily rotation file |
| GET | `/api/logs/<pipeline>/traces/` | List trace dates |
| GET | `/api/logs/<pipeline>/traces/<date>/` | List trace files for a date |
| GET | `/api/logs/<pipeline>/traces/<date>/<filename>/` | Read one trace file |

### Query params (stream, rotated file, search)

| Param | Description |
|-------|-------------|
| `tail` | Return last N **matched** lines (search scans full file first) |
| `level` | Filter: INFO / WARNING / ERROR / DEBUG |
| `accession` | Substring match on accession field |
| `run_id` | Exact run_id match |
| `search` | Case-insensitive substring in full raw line |
| `raw=1` or `raw=true` | Return `{ "content": "..." }` — full file as plain string |

**Important:** Use `raw=1`, **not** `format=raw`. Django REST Framework reserves
`format` for content negotiation and returns 404 for unknown formats.

**With `raw=1`:** All filters (`tail`, `search`, etc.) are ignored — full file returned.

### Examples

```bash
BASE=https://django.arbintel.cloud/api/logs

# List pipelines
curl "$BASE/pipelines/"

# Today's live app log (structured)
curl "$BASE/app/stream/?tail=200&level=ERROR"

# Today's live app log (plain text)
curl "$BASE/app/stream/?raw=1"

# List daily dates
curl "$BASE/app/rotated/"

# List files for one day
curl "$BASE/app/rotated/2026-05-26/"

# Read a rotated file (plain text)
curl "$BASE/app/rotated/2026-05-26/app.log.3/?raw=1"

# Search accession across all pipelines (today's active logs only)
curl "$BASE/search/?accession=0001193125-26-126362&tail=500"

# Trace files
curl "$BASE/sec_8k/traces/"
curl "$BASE/sec_8k/traces/2026-05-26/"
curl "$BASE/rss/traces/2026-05-26/nsic-psychiatry-in-kri-c4352466_RSS_a8f91c.log/?raw=1"
```

### Response formats

**Structured (default):**
```json
{
  "pipeline": "app",
  "date": "2026-05-26",
  "filename": "app.log",
  "total_matched": 3026,
  "lines": [
    {
      "ts": "2026-05-26 09:45:11 AM",
      "level": "INFO",
      "pipeline": "app",
      "run_id": "-",
      "accession": "-",
      "doc_type": "UNKNOWN",
      "module": "sec_rss_parser.process_feed_8k:625",
      "message": "Processing started"
    }
  ]
}
```

**Raw (`?raw=1`):**
```json
{
  "content": "2026-05-26 09:45:11 AM | INFO  | pipeline=app | ...\n..."
}
```

### Suggested frontend layout

```
Logs
├── Pipeline list          ← GET /pipelines/
├── Search                 ← GET /search/?accession=...
└── Pipeline detail
    ├── Live stream        ← GET /{pipeline}/stream/?tail=200
    ├── Daily browser
    │   ├── Date list      ← GET /{pipeline}/rotated/
    │   ├── File list      ← GET /{pipeline}/rotated/{date}/
    │   └── File viewer    ← GET /{pipeline}/rotated/{date}/{filename}/?raw=1
    └── Trace browser
        ├── Date list      ← GET /{pipeline}/traces/
        ├── File list      ← GET /{pipeline}/traces/{date}/
        └── Trace viewer   ← GET /{pipeline}/traces/{date}/{filename}/?raw=1
```

### Security

- URL segments validated: pipeline/date use strict allowlist; filenames block `..`, `/`, `\`.
- `AllowAny` by default — add auth (`IsAuthenticated`) before exposing publicly.
- Do not commit log files or secrets.

---

## Entry Points Wired with `start_pipeline()`

| File | Where |
|------|-------|
| `sec_rss_parser/process_feed_8k.py` | `_process_single_item`, EX-2.1 detection |
| `sec_rss_parser/fetch_sec_feed_by_deal_cik.py` | `process_items` loop |
| `rss_feeds/services.py` | RSS webhook item loops (flow 1 + flow 2) |
| `sec_rss_parser/services.py` | 8-K summary, document processing |
| `sec_rss_parser/proxy_processor_helper.py` | Proxy summary |
| `sec_rss_parser/proxy_comparision/orchestrator.py` | Proxy comparison |
| `sec_rss_parser/tenK_tenQ_pipeline/orchestrator.py` | 10-K/10-Q |
| `document_processor/services.py` | DMA, summary, covenant, termination |
| `document_processor/MAE/run_full_pipeline.py` | MAE |

Lines logged **before** `start_pipeline()` is called use `pipeline=app`, `accession=-`.

---

## Notes

- **`log_and_print()`** in `services.py` / `utils_8k.py` — no changes needed; uses `logger.info()`.
- **`print()`** calls go to stdout/Docker logs only, not log files.
- **Local dev:** logs at `rag_project/logs/` (gitignored).
- **Thread safety:** `_lock` in `DynamicPipelineHandler` protects handler creation.
- **DRF `format` param:** Reserved — always use `raw=1` for plain-text responses.
- **Sentry (optional):** Not yet integrated — see team docs for exception alerting setup.

---

## Change History

| Date | Change |
|------|--------|
| 2026-05 | Initial pipeline logging with `contextvars` + trace files |
| 2026-05 | Logs API (`logs_api` app) |
| 2026-05 | IST timestamps with AM/PM format |
| 2026-05 | Daily folder layout: `{pipeline}/daily/{date}/` with unlimited daily rotations |
| 2026-05 | 7-day daily retention + 30-day trace retention (cron) |
| 2026-05 | API: two-level rotated browser (date → files), `raw=1` plain-text mode |
| 2026-05 | Log volume path: `/opt/apps/logs/django` on host |
