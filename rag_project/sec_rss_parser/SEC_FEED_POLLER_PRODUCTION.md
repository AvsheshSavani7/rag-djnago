# SEC Feed Poller — Production Workflow

Two-stage SEC filing poller that replaces the old per-CIK polling in
`fetch_sec_feed_by_deal_cik.py`. It polls the SEC global feed once, caches every
filing to a daily JSON file, then filters that cache down to tracked deal CIKs
and runs the **unchanged** `process_items()` pipeline.

---

## 1. Why this design

The old flow made one `getcurrent&CIK={cik}` request **per tracked CIK** every
run (~2 × number of deals), consuming most of the SEC rate-limit budget before
any document was even fetched.

The new flow makes **one** global request per second regardless of deal count,
then filters locally at zero SEC/DB cost — freeing almost the entire rate budget
for the document + summary pipeline.

---

## 2. Architecture (one process, multiple threads)

```
                    ┌─────────────────────────────────────────────────────────┐
                    │        run_sec_feed_poller  (single OS process)          │
                    │                                                          │
  SEC getcurrent ──▶│  Stage A — Collector thread        (~1s poll)             │
  (all forms, 100)  │    poll → parse → dedup → append                         │
                    │                         │                                │
                    │                         ▼                                │
                    │              feed_YYYYMMDD.json  (daily cache, ET day)   │
                    │                         │                                │
                    │         ┌───────────────┴───────────────┐                │
                    │         ▼                               ▼                │
                    │  Stage B — Scheduler (main thread)   Stage C — Reconcile │
                    │    every ~15s: scan feed → enqueue     thread (scheduled) │
                    │                         │              05:55 → yesterday  │
                    │                         ▼              23:30 → today      │
                    │              Work queue + N workers                      │
                    │    each worker: process_items([1 item])                  │
                    └─────────────────────────┬───────────────────────────────┘
                                              ▼
                    existing pipeline (proxy / 10-K-10-Q /
                    route-and-summarize / emails) — UNCHANGED
```

| Thread | Role |
|--------|------|
| **Collector** (daemon) | Stage A — global `getcurrent` → feed JSON |
| **Scheduler** (main) | Stage B — discover + enqueue every ~15s |
| **Workers** (daemon, N) | Stage B — `process_items([item])` in parallel |
| **Reconcile** (daemon) | Stage C — merge `master.{date}.idx` into feed JSON |

**All stages run in the same process on purpose** — the SEC rate limiter
(`sec_rate_limit.py`) is process-local. Separate processes would each get their
own budget and could exceed SEC's ~10 req/s limit.

**Run exactly ONE `sec-feed-poller` container.** Two pollers double SEC load and
duplicate work.

---

## 3. Files

| File | Role |
|------|------|
| `sec_feed_collector.py` | Stage A: poll global feed, write `feed_YYYYMMDD.json`. No Mongo. |
| `sec_feed_daily_store.py` | Atomic JSON read/write, ET day keys, midnight-grace helpers. |
| `sec_feed_work_queue.py` | Stage B: queue, `reserved` dedup, worker pool, `discover_and_enqueue()`. |
| `sec_feed_processor.py` | Stage B entry: starts scheduler (main) + workers. |
| `fetch_sec_feed_by_deal_cik.py` | `process_items` (unchanged) + `build_tracked_ciks_map` / `build_items_from_daily_feed`. |
| `sec_daily_index.py` | Stage C: fetch/parse `master.{YYYYMMDD}.idx`, merge into feed JSON. |
| `sec_rate_limit.py` | Shared process-wide SEC throttle. |
| `management/commands/run_sec_feed_poller.py` | Entry point: collector + processor + reconcile threads. |
| `management/commands/reconcile_sec_daily_index.py` | Manual reconcile (one-off / past days). |

Test-only counterparts (`sec_feed_collector_test.py`, `sec_feed_processor_test.py`,
`sec_feed_test_logging.py`) stay for local, read-only, no-pipeline experiments.

---

## 4. Data flow, step by step

### Stage A — Collector (`collect_once`)

1. `GET getcurrent` (all forms, `count=100`) through the shared rate limiter.
2. Parse the Atom feed → per-item `form_type`, `accession_number`, `cik`, `title`, `link`.
3. `append_feed_items()` merges new rows into `feed_YYYYMMDD.json`, **keyed by
   accession** (re-seen filings are no-ops). Writes are atomic (temp file + rename).
4. Sleep until the next tick. A slow SEC response just slows cadence — the loop is
   synchronous, so a second request never starts before the first returns.

Daily rollover is automatic: the filename uses the current **America/New_York**
date (same calendar day as EDGAR's `master.{YYYYMMDD}.idx`).

### Stage B — Processor (scheduler + work queue)

1. **Scheduler** (main thread, every `SEC_FEED_PROCESSOR_INTERVAL_SEC`, default 15s):
   refreshes CIKs, scans feed JSON, filters, enqueues new accessions. Runs on
   schedule **even while workers are busy** (unlike the old blocking `process_tick`).
2. **Work queue** (`sec_feed_work_queue.py`): thread-safe `queue.Queue` + `reserved`
   set so the same accession is never enqueued twice while queued or in-flight.
3. **Worker pool** (`SEC_FEED_PROCESSOR_WORKERS`, default **3** in code,
   **5** in `docker-compose.yml`): each worker calls `process_items([item])` —
   one filing at a time, up to N in parallel.
4. `build_items_from_daily_feed()` returns tracked CIK + non-excluded items;
   excluded forms (8-K, Form 4, 13D, …) go straight to `session_done` (terminal).
5. Downstream `process_items()` unchanged: `AccessionLookedUp`, lock, pipeline,
   `mark_accession_processed`.

During the first **30 minutes** after US Eastern midnight, the scheduler also
scans **yesterday's** feed (`get_feed_days_to_process`).

**Example:** tick 1 enqueues A–G; 5 workers process A,B,C,D,E in parallel; F,G
wait in queue. Tick 2 (+15s) enqueues only H,I (F,G already `reserved`).

### Stage C — Reconcile (scheduled thread)

See §6. Merges EDGAR daily index into feed JSON; processor queue picks up new rows
on the next scheduler tick.

---

## 5. Dedup & retry — Option A

MongoDB is the source of truth; the processor keeps **in-memory** `session_done`
and `reserved` sets (never written to disk).

| Layer | Purpose |
|-------|---------|
| `reserved` | Accession in queue or being processed — skip on enqueue |
| `session_done` | Terminal this run (excluded forms + `AccessionLookedUp`) |
| `AccessionProcessingLock` | One pipeline run per accession (in `process_items`) |
| `AccessionLookedUp` | Permanent terminal across restarts |

| Outcome | Remembered? | Re-tried? |
|---------|-------------|-----------|
| CIK not tracked | No | Re-checked every scheduler tick |
| Tracked CIK, excluded form (e.g. 8-K) | `session_done` | Never — **8-K handled by `process_feed_8k` / n8n** |
| In queue or processing | `reserved` | Not re-enqueued |
| Pipeline success (`AccessionLookedUp`) | `session_done` | Never |
| Pipeline failed (no looked-up) | Released from `reserved` | **Re-enqueued on next scheduler tick** |

On process restart `session_done` is empty and MongoDB re-gates everything, so
nothing is double-processed.

---

## 6. Downtime completeness — reconcile

`getcurrent` is a rolling ~100-item window (~30–50s of filings at ~2–3/s). If the
collector is down longer than that, those filings vanish from the live feed.

The EDGAR **daily index** (`master.{YYYYMMDD}.idx`) is published **each evening
(~10 PM ET)** with every filing for that calendar day. `reconcile_into_feed()`
downloads it (one GET), derives each accession, and merges any missing accession
into the feed JSON. The processor queue then handles them on the next scheduler tick.

**SEC does not publish a complete same-day index during market hours** — the live
collector is the intraday source; reconcile is the end-of-day backstop.

**Automatic** (built into `run_sec_feed_poller`, tracked CIKs only):

| Time (ET) | Index target | Why |
|-----------|--------------|-----|
| **05:55** | **Yesterday** | Final index from prior evening; late filings + collector gaps |
| **23:30** | **Today** | After today's index exists (~10 PM ET) |

```bash
python manage.py run_sec_feed_poller --reconcile-schedule "05:55:yesterday,23:30:today"
python manage.py run_sec_feed_poller --no-reconcile   # disable built-in reconcile
```

**Manual** (any time / past day):

```bash
python manage.py reconcile_sec_daily_index                 # today, all CIKs
python manage.py reconcile_sec_daily_index --tracked-only  # only deal CIKs (lean)
python manage.py reconcile_sec_daily_index --date 20260720 # specific ET day
```

> Do **not** add a host cron for reconcile if the poller runs with reconcile enabled
> — you would duplicate work. Use manual reconcile only for one-off backfill.

---

## 7. Logging

Collector, scheduler, workers, and reconcile log to pipeline `sec_feed_poller`
via `DynamicPipelineHandler`:

```
/opt/apps/logs/django/sec_feed_poller/daily/<YYYY-MM-DD>/sec_feed_poller.log
```

(Inside container: `/var/log/rag/sec_feed_poller/...` — rotates at 10 MB.)

| `doc_type` | Source |
|------------|--------|
| `collector` | Stage A — only logs ticks with **new** filings |
| `scheduler` | Stage B — **every ~15s** queue snapshot (`waiting`, `in_flight`, `pending` accessions, `new` this tick) |
| `processor` | Stage B — worker `process_items` lines |
| `reconcile` | Stage C — daily index merge; logs **missed accession numbers** when `added > 0` |

Downstream pipeline work logs to its own pipelines (`proxy`, `ten_k_ten_q`,
`sec_feed`). Log folder dates use **IST** (existing logging system); feed file
dates use **America/New_York**.

**Scheduler tick example (every 15s):**
```
sec_feed_processor: scheduler tick | waiting=2 | in_flight=3 | queue≈2 | session_done=142 | enqueued=+1 | pending: 0001234567-26-000001, 0001234567-26-000002 | new: 0001234567-26-000003
```

**Reconcile example (when missed filings are merged):**
```
sec_daily_index: reconcile 2026-07-22 | parsed=12 | added=2 missed | 0001234567-26-000001, 0001234567-26-000002
```

---

## 8. Environment variables

| Variable | Default (code) | Production (`docker-compose`) | Meaning |
|----------|----------------|-------------------------------|---------|
| `SEC_FEED_DAILY_DIR` | `sec_rss_parser/sec_daily_feed` | same (mounted volume) | Feed JSON directory |
| `SEC_FEED_COLLECTOR_INTERVAL_SEC` | `1.0` | `1.0` | Collector poll cadence |
| `SEC_FEED_PROCESSOR_INTERVAL_SEC` | `15.0` | `15.0` | Scheduler enqueue cadence |
| `SEC_FEED_PROCESSOR_WORKERS` | `3` | **`5`** | Parallel `process_items` workers |
| `SEC_FEED_DEAL_CIK_REFRESH_SEC` | `60.0` | `60.0` | CIK map refresh from Mongo |
| `SEC_FEED_TIMEZONE` | `America/New_York` | `America/New_York` | Feed day + reconcile schedule |
| `SEC_FEED_MIDNIGHT_GRACE_MINUTES` | `30` | `30` | Also scan yesterday's feed after ET midnight |
| `SEC_MIN_REQ_INTERVAL` | `0.2` | `0.2` | Min seconds between SEC requests (~5/s) |
| `ACCESSION_LOCK_TTL_SECONDS` | `900` | from `.env` | Lock TTL during long summaries (15 min) |
| `LOG_ROOT` | `logs/` | `/var/log/rag` | Pipeline log root |

---

## 9. Running it

### Local

```bash
cd rag_project
python manage.py run_sec_feed_poller
python manage.py run_sec_feed_poller --no-collector   # processor + reconcile only
python manage.py run_sec_feed_poller --no-processor   # collector + reconcile only
python manage.py run_sec_feed_poller --no-reconcile   # no Stage C
```

### Docker (production)

`sec-feed-poller` in `docker-compose.yml` reuses `rag-django:latest` (built by
`django-app`), runs `python manage.py run_sec_feed_poller`, no HTTP port.

Deploy via GitHub Actions (`vps/main` push) → `deploy.sh` →
`docker compose up -d --build`.

```bash
# First time on VPS:
mkdir -p /opt/apps/logs/django /opt/apps/sec_daily_feed
chmod 777 /opt/apps/logs/django /opt/apps/sec_daily_feed

# Normal deploy:
cd /opt/apps/django-app && ./deploy.sh

# Verify:
docker compose ps
docker compose logs -f sec-feed-poller
ls -la /opt/apps/sec_daily_feed/

# Optional one-off backfill after cutover:
docker compose exec -T sec-feed-poller \
  python manage.py reconcile_sec_daily_index --tracked-only

# Rollback (stop new poller, re-enable old n8n CIK webhook):
docker compose stop sec-feed-poller
```

---

## 10. Operational rules & gotchas

1. **Exactly one poller instance** — one `sec_feed_poller` container only.
2. **Disable the old CIK trigger** — unpublish n8n / cron calling
   `fetch-feed-by-deal-cik/` or it competes for SEC budget and locks.
3. **Keep 8-K flow running** — poller **excludes** 8-K; `process_feed_8k` (n8n
   `process-feed`) still handles 8-K / EX-2.1 / EX-99.1.
4. **One container for A + B + C** — do not split collector and processor into
   separate containers (shared rate limiter).
5. **Feed + logs must persist** — host volumes `/opt/apps/sec_daily_feed` and
   `/opt/apps/logs/django`.
6. **`link` must be an index URL** — collector stores `-index.htm`; reconcile
   derives the same shape from the idx `.txt` path.
7. **Intraday gaps** — collector covers live filings; reconcile covers end-of-day
   gaps after ~10 PM ET. Long collector outage → wait for 23:30 reconcile or run
   manual `reconcile_sec_daily_index --date YYYYMMDD`.
8. **Worker count** — 5 workers increase throughput but also LLM/API load; reduce
   `SEC_FEED_PROCESSOR_WORKERS` if you see OOM or rate-limit errors.
