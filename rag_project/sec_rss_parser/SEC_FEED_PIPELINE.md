# SEC Feed Pipeline — Collector to Processor

How the live SEC filing system works end-to-end: discovery, queues, processing, and which file owns each piece.

**Entry point (one process):**

```bash
python manage.py run_sec_feed_poller
```

Run **exactly one** instance. All stages share one process-wide listing rate limiter and the same daily feed JSON.

---

## Big picture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     run_sec_feed_poller (ONE process)                    │
├──────────────┬──────────────┬──────────────┬──────────────┬─────────────┤
│  Stage A     │  Stage B     │  Stage B2    │  Stage B3    │  Stage C    │
│  Collector   │  All-filings │  8-K queue   │  S-4/F-4     │  Reconcile  │
│  ~1s         │  ~15s / 20w  │  ~10s / 20w  │  ~15s / 20w  │  scheduled  │
└──────┬───────┴──────┬───────┴──────┬───────┴──────┬───────┴──────┬──────┘
       │              │              │              │              │
       ▼              ▼              ▼              ▼              ▼
  getcurrent     tracked CIKs    ALL 8-Ks      non-tracked     EDGAR daily
  all forms      (not 8-K)       from JSON     S-4/F-4 + LLM   master.idx
       │              │              │              │              │
       └──────────────┴──────────────┴──────────────┴──────────────┘
                              │
                    feed_YYYYMMDD.json
                    (sec_daily_feed/)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     rate_limited_get    proxy_get()     Mongo locks
     (listing only)      (Archives HTML) LookedUp / Lock
```

**Why this design?**

| Problem | Solution |
|--------|----------|
| SEC ~10 req/s per IP on listing | One collector + shared `rate_limited_get` |
| Archives doc fetches also rate-limited | Sticky residential proxies (`sec_proxy_fetch`) |
| Don’t miss filings when down | Daily JSON cache + evening reconcile |
| Don’t double-process | `AccessionLookedUp` + processing locks + in-memory `reserved` |
| Different form rules | Separate queues: all-filings / 8-K / global S-4/F-4 |

---

## Stage A — Collector (discovery only)

**Job:** Poll SEC `getcurrent` (all forms), append new rows into today’s JSON.  
**Does not** touch Mongo or run pipelines.

| File | Need |
|------|------|
| `sec_feed_collector.py` | Loop (~1s): fetch all-forms Atom → parse → append to daily JSON |
| `sec_feed_daily_store.py` | Read/write `feed_YYYYMMDD.json`; keys = `cik\|accession`; timezone America/New_York |
| `sec_feed_item_utils.py` | Feed key helpers, role parse, group-by-accession |
| `sec_rate_limit.py` | Process-wide throttle for `www.sec.gov` listing GETs (`SEC_MIN_REQ_INTERVAL`) |
| `utils_8k.py` (`SECRSSParser`) | Atom parse shared by collector |

**Output:** `sec_rss_parser/sec_daily_feed/feed_YYYYMMDD.json`  
Every filing seen that calendar day (ET), regardless of form type.

**Flags:** `--no-collector` · env `SEC_FEED_COLLECTOR_INTERVAL_SEC` (default `1.0`)

---

## Stage B — All-filings processor (tracked deal CIKs)

**Job:** Every ~15s, scan today’s JSON for filings whose CIK is on an open/unknown deal. Enqueue one item per accession. Workers call `process_items()`.

**Skips here:**

- `8-K` / `8-K/A` → Stage B2  
- Forms 4, 144, S-8, Schedule 13D/G, etc. → terminal skip  
- Already `AccessionLookedUp` / reserved / in-flight  

Tracked **S-4 / F-4** for deal CIKs are handled **here** (not in Stage B3).

| File | Need |
|------|------|
| `sec_feed_processor.py` | Scheduler loop + starts Stage B worker pool |
| `sec_feed_work_queue.py` | Discover/enqueue; `session_done` / `reserved`; N workers |
| `fetch_sec_feed_by_deal_cik.py` | `build_tracked_ciks_map`, `build_items_from_daily_feed`, `process_items`, proxy/10-K/summary helpers |
| `accession_lock.py` | Acquire/release lock; `mark_accession_processed` → `AccessionLookedUp` |

**Pipeline per item (simplified):** lock → fetch index HTML (proxy) → `SECFiling` → L1/L2/L3 summary email → form-specific path (proxy comparison / 10-K / etc.) → LookedUp.

**Flags:** `--no-processor` · `SEC_FEED_PROCESSOR_INTERVAL_SEC` · `SEC_FEED_PROCESSOR_WORKERS` · `SEC_FEED_DEAL_CIK_REFRESH_SEC`

---

## Stage B2 — 8-K from JSON

**Job:** Every ~10s, take **all** `8-K` / `8-K/A` rows from the daily JSON (no deal-CIK filter). Prefer deal CIK when multi-CIK rows share one accession; otherwise first row. Workers run `EightKFeedProcessor._process_single_item`.

| File | Need |
|------|------|
| `sec_8k_feed_processor.py` | Scheduler loop for 8-K |
| `sec_8k_feed_work_queue.py` | Per-accession queue + dedupe (LookedUp / lock / reserved) |
| `fetch_sec_feed_by_deal_cik.py` → `build_8k_items_from_daily_feed` | Build 8-K work items from JSON |
| `process_feed_8k.py` | EX-2.1 / EX-99.1 / deal vs non-deal 8-K logic, emails, GPT |

**Does not** call `getcurrent?type=8-K`. Discovery is the collector JSON only.

**Flags:** `--no-8k` · `SEC_FEED_8K_INTERVAL_SEC` · `SEC_FEED_8K_WORKERS`

**Cutover:** stop n8n `/process-feed/` so you don’t double-run 8-K.

---

## Stage B3 — Global S-4 / F-4 (non-tracked + LLM match)

**Job:** Every ~15s, find S-4/F-4(/A) in JSON that are **not** owned by a tracked deal CIK. LLM name/alias match → if the filing CIK has more than `SEC_GLOBAL_FORM_MAX_PRIOR_FILINGS` EDGAR filings (data.sec.gov RSS), LookedUp and stop; otherwise inject `deal_id` + `discovery_note` → same summary + proxy pipeline as CIK flow.

**Filter contrast vs 8-K:**

| | 8-K (B2) | Global S-4/F-4 (B3) |
|--|----------|---------------------|
| Tracked CIK present | Prefer that row; still process | **Drop** accession (Stage B owns it) |
| Match to deal | Inside 8-K processor | LLM company-name match |

| File | Need |
|------|------|
| `sec_global_form_feed_processor.py` | Scheduler loop for S-4/F-4 |
| `sec_global_form_feed_work_queue.py` | Per-accession queue + dedupe |
| `fetch_sec_global_form_type_feed.py` | `build_global_form_items_from_daily_feed`, `process_one_global_form_item`, LLM match, LookedUp rules |
| `fetch_sec_feed_by_deal_cik.py` | Shared HTML fetch, summarize, `_handle_proxy_form_by_type` |

**LookedUp behavior (B3):**

- Genuine LLM `NONE` → LookedUp  
- LLM match but filing CIK has **more than** `SEC_GLOBAL_FORM_MAX_PRIOR_FILINGS` RSS items (default 10) → LookedUp  
- Success → LookedUp  
- Proxy fail **after** summary exists → LookedUp (no re-email retry)  
- HTML / LLM API error / CIK-history RSS fail / proxy fail with **no** summary → no LookedUp (retry)  
- Empty open deals → skip tick (do not burn accessions)  
- `SEC_GLOBAL_FORM_MAX_PRIOR_FILINGS=0` disables the prior-filing gate  

**Flags:** `--no-global-form` · `SEC_FEED_GLOBAL_FORM_INTERVAL_SEC` · `SEC_FEED_GLOBAL_FORM_WORKERS`

**Cutover:** stop n8n `/fetch-global-form-type-feed/`. Legacy tick coalescer (`sec_global_form_work_queue.py`) remains for API only — not used by the poller JSON path.

---

## Stage C — Reconcile (completeness backstop)

**Job:** On a schedule (default ~23:30 ET), download EDGAR `master.YYYYMMDD.idx` and merge any missing accessions into the feed JSON. New rows get `"source": "daily_index"`. Processor queues pick them up on next tick.

| File | Need |
|------|------|
| `sec_daily_index.py` | Download/parse daily index; append missing rows |
| `run_sec_feed_poller.py` | Reconcile scheduler thread |

**Why:** `getcurrent` is a rolling window (~100 items). Outages drop filings from live discovery; the daily index has the full day.

**Flags:** `--no-reconcile` · `--reconcile-schedule`

---

## Morning getcurrent deep-page backfill (optional cron)

**Job:** Separate from the poller. Scans `getcurrent` pages `start=100..1900` (skips page 0), dedupes against **today + previous 4 days** of feed JSON (`--lookback-days 4`), appends misses into **today’s** feed with `"source": "getcurrent_backfill"`. Covers a Sunday run still seeing Friday Form 4s on getcurrent. Catches late-sorted / prior-day filings that never hit page 1 and were missing from yesterday’s `master.idx`. Live collector rows omit `source`.

| File | Need |
|------|------|
| `management/commands/backfill_getcurrent_pages.py` | Single self-contained command + dedicated JSONL add log |

```bash
python manage.py backfill_getcurrent_pages
python manage.py backfill_getcurrent_pages --dry-run
python manage.py backfill_getcurrent_pages --lookback-days 4
```

**Logs:**
- Pipeline (frontend): `{LOG_ROOT}/sec_feed_backfill/daily/YYYY-MM-DD/sec_feed_backfill.log`
- JSONL audit: `{LOG_ROOT}/sec_feed_backfill/daily/YYYY-MM-DD/getcurrent_backfill.jsonl`

**Suggested cron (ET):** `0 1-23 * * *` with `CRON_TZ=America/New_York` (hourly 1 AM–11 PM; skip midnight). Daily, not weekdays-only, so Saturday deep pages are not missed.

---

## Shared infrastructure

| File | Need |
|------|------|
| `management/commands/run_sec_feed_poller.py` | Orchestrates A + B + B2 + B3 + C in one process |
| `sec_proxy_fetch.py` | Sticky proxy list for **Archives** HTML/doc GETs; max 5 attempts; fail email |
| `residential-rotational proxy.txt` | Manual sticky session list (refresh by hand) |
| `sec_rate_limit.py` | Listing-only throttle (getcurrent / daily index) |
| `accession_lock.py` | Cross-worker / cross-queue lock + LookedUp helpers |
| `models.py` | `AccessionLookedUp`, `AccessionProcessingLock`, `SECFiling`, `SECFilingSummary`, … |
| `core/pipeline_logger.py` | Structured pipeline tags (`sec_feed`, `sec_8k`, `global_form_feed`, …) |

### Listing vs document fetch

```
getcurrent / daily-index  →  rate_limited_get  (direct IP, ~5/s)
Archives index + exhibits →  proxy_get         (sticky residential list)
```

---

## Dedup layers (all processor queues)

1. **`session_done`** — in-memory; finished this process lifetime  
2. **`reserved`** — queued or in-flight in this process  
3. **`AccessionLookedUp`** — Mongo terminal “done”  
4. **`AccessionProcessingLock`** — another worker holds the accession (retry after TTL; do not session_done)  

Stage B / B2 / B3 do **not** share `reserved`, but they share Mongo locks + LookedUp, so the same accession is not fully processed twice.

---

## Ownership cheat sheet

| Filing | Who processes it? |
|--------|-------------------|
| Any form, CIK on open deal (except 8-K, excluded forms) | Stage B |
| 8-K / 8-K/A (any CIK) | Stage B2 |
| S-4/F-4, CIK on open deal | Stage B (CIK path) |
| S-4/F-4, CIK **not** on open deal | Stage B3 (LLM match only if name hits a deal) |
| Form 4 / 144 / S-8 / 13D/G | Skipped (terminal) |

---

## Env vars (typical compose)

| Variable | Role | Typical |
|----------|------|---------|
| `SEC_FEED_DAILY_DIR` | Feed JSON directory | `.../sec_daily_feed` |
| `SEC_FEED_TIMEZONE` | Feed day boundary | `America/New_York` |
| `SEC_FEED_COLLECTOR_INTERVAL_SEC` | Stage A | `1.0` |
| `SEC_FEED_PROCESSOR_INTERVAL_SEC` | Stage B | `15.0` |
| `SEC_FEED_PROCESSOR_WORKERS` | Stage B workers | `20` |
| `SEC_FEED_8K_INTERVAL_SEC` | Stage B2 | `10.0` |
| `SEC_FEED_8K_WORKERS` | Stage B2 workers | `20` |
| `SEC_FEED_GLOBAL_FORM_INTERVAL_SEC` | Stage B3 | `15.0` |
| `SEC_FEED_GLOBAL_FORM_WORKERS` | Stage B3 workers | `20` |
| `SEC_GLOBAL_FORM_MAX_PRIOR_FILINGS` | B3 skip if CIK RSS count > this (`0` = off) | `10` |
| `SEC_GLOBAL_FORM_CIK_RSS_COUNT` | `count=` on data.sec.gov CIK RSS | `40` |
| `SEC_FEED_DEAL_CIK_REFRESH_SEC` | Refresh tracked CIK map | `60` |
| `SEC_MIN_REQ_INTERVAL` | Listing throttle | `0.2` |
| `SEC_PROXY_MAX_ATTEMPTS` | Doc fetch retries | `5` |
| `SEC_PROXY_TIMEOUT` | Proxy GET timeout | `12` |
| `SEC_PROXY_LIST_FILE` | Sticky list path | default beside `sec_proxy_fetch.py` |

---

## Related / legacy (not the JSON poller path)

| File | Role |
|------|------|
| `sec_8k_work_queue.py` | Old n8n coalesced full `run_8k_processor` ticks |
| `sec_global_form_work_queue.py` | Old n8n coalesced full global getcurrent runs |
| `views.py` API endpoints | Still callable; stop n8n on cutover to avoid double work |
| `*_test.py` / `test_*_fetch.py` | Dry-run / proxy load tests — not production stages |

---

## Quick start checklist

1. Ensure sticky proxy list file is present and current.  
2. Start one `run_sec_feed_poller` container/process.  
3. Stop n8n hits to `/process-feed/` and `/fetch-global-form-type-feed/`.  
4. Confirm feed file grows: `sec_daily_feed/feed_YYYYMMDD.json`.  
5. Watch logs for `sec_feed_collector`, `sec_feed_processor`, `sec_8k_feed`, `sec_global_form_feed`.  

Disable one stage without stopping others: `--no-collector`, `--no-processor`, `--no-8k`, `--no-global-form`, `--no-reconcile`.
