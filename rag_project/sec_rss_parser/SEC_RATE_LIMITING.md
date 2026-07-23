### SEC rate limiting design (sec_rss_parser)

**Goal:** Stay safely under SEC’s `browse-edgar` rate limits (≈10 requests/second per IP) for all RSS/feed and index-HTML calls made by this service, without adding Redis or external coordination.

---

### 1. Core mechanism: `sec_rate_limit.py`

- **File**: `sec_rate_limit.py`
- **Function**: `rate_limited_get(session_or_requests, url, *args, **kwargs)`
- **Behavior**:
  - Parses the URL and checks `netloc == "www.sec.gov"`.
  - For non‑SEC URLs, just calls `.get(url, ...)` directly (no throttling).
  - For SEC URLs, enforces a **minimum interval** between calls in this process:
    - Default: `_MIN_INTERVAL = 0.2` seconds → max ≈ **5 req/sec per process**.
    - Configurable via env var: `SEC_MIN_REQ_INTERVAL` (float seconds).
  - Thread-safe: uses a module‑level `threading.Lock` around the timing logic.

This provides a **process‑wide token bucket–like throttle** for all SEC GETs routed through `rate_limited_get`.

---

### 2. Where it is used

Currently wired into the two main feed/index paths that were hitting limits:

**A. 8‑K processor (`process_feed_8k.py` via `utils_8k.py`)**

- **File**: `utils_8k.py`
- **Class**: `SECRSSParser`
- **Changes**:
  - `fetch_rss_feed()`:
    - Before: manual retry loop (`for attempt in range(max_retries)`) + `requests.Session` with `Retry(total=3, status_forcelist=[429, ...])`.
    - Now: single call via
      ```python
      response = rate_limited_get(
          self.session, self.feed_url, headers=self.headers, timeout=45
      )
      ```
    - Manual outer loop removed; retries are handled only by `urllib3.Retry`.
  - `fetch_and_parse_html(html_url, ...)`:
    - Before: `time.sleep(2)` then `self.session.get(html_url, ...)`.
    - Now:
      ```python
      response = rate_limited_get(
          self.session, html_url, headers=self.headers, timeout=45
      )
      ```

**B. Deal‑CIK feed processor (`fetch_sec_feed_by_deal_cik.py`)**

- **File**: `fetch_sec_feed_by_deal_cik.py`
- **Functions**:
  - `fetch_feed_for_cik(cik, session, headers=None)`:
    - Before: manual `for attempt in range(3)` retry loop around `session.get(...)` **plus** a `Session` with `Retry(total=3, status_forcelist=[429, ...])`.
    - Now:
      ```python
        resp = rate_limited_get(
            session, url, headers=headers or DEFAULT_HEADERS, timeout=45
        )
      ```
    - Manual loop removed; only `Retry` on the adapter handles 429/5xx.
  - `fetch_and_parse_html_by_form_type(html_url, ...)`:
    - Before: `resp = requests.get(html_url, headers=DEFAULT_HEADERS, timeout=30)`.
    - Now:
      ```python
      resp = rate_limited_get(
          requests,
          html_url,
          headers=DEFAULT_HEADERS,
          timeout=45,
      )
      ```

As a result, **all SEC RSS/feed + index‑HTML calls from these two scripts** share the same ~5 rps budget within a single Python process.

---

### 3. 8-K `process-feed` worker queue (`sec_8k_work_queue.py`)

n8n may call `/api/sec/process-feed/` every ~45s. The view **does not** spawn a new thread per request.

- **Queue:** `maxsize=1` — at most one pending tick while workers are busy.
- **Workers:** `SEC_8K_PROCESSOR_WORKERS` (default **1**, max **2**).
- **API response:** `status: queued` or `already_running` (tick coalesced).
- **Trace logs:** `LOG_TRACE_HANDLER_CACHE` (default 500) caps open trace `FileHandler`s in `dynamic_pipeline_handler.py`.

---

### 4. Why this fixes the original problem

Original symptoms:

- `process_feed_8k.py` ran every 45 second, `fetch_sec_feed_by_deal_cik.py` every 3 minutes, over ~150 CIKs.
- Each path had:
  - A `requests` session with `Retry(total=3, backoff_factor=1, status_forcelist=[429, ...])`
  - **Plus** a manual retry loop (`for attempt in range(3)`).
- When SEC started returning 429s, the nested retries multiplied calls and kept retrying, quickly re‑triggering the rate limit.

Fixes introduced:

- **Single retry layer per request**:
  - Manual outer loops removed.
  - Only `urllib3.Retry` handles transient 429/5xx, with backoff.
- **Global throttle below SEC’s cap**:
  - `rate_limited_get` enforces ≈5 rps per process across:
    - CIK Atom feed fetches,
    - Filing index HTML fetches (both jobs),
    - 8‑K feed fetches.

This combination greatly reduces the probability of sustained 429s from SEC due to these scripts, and prevents self‑amplifying retry storms after the service has been running for hours.

---

### 5. Read timeouts and retries

SEC sometimes responds slowly; the client can hit **ReadTimeoutError** (previously read timeout=30s, now 45s). By default, `urllib3.Retry` also retries on read errors, so one slow request became 3 attempts and produced "Retrying (Retry(total=1, ...)) after connection broken by 'ReadTimeoutError'" in logs.

To avoid that:

- **`read=0`** is set on all SEC `Retry` instances (`utils_8k.SECRSSParser._create_session` and `fetch_sec_feed_by_deal_cik.run_fetch_sec_feed_by_deal_cik`). We still retry on **429 / 5xx** (status_forcelist); we **do not** retry on read timeouts.
- **Request timeout** for SEC GETs was increased from **30s to 45s** everywhere we call `rate_limited_get` (feed + index HTML), so SEC has more time to respond before we give up.

Result: a slow SEC response either succeeds within 45s or fails once (no retry storm), and 429/5xx continue to be retried with backoff.

**AccessionLookedUp and failures:** We only add an accession to `AccessionLookedUp` **after** the item has been successfully processed (HTML fetched, filing/summary logic run). If the request fails (e.g. read timeout) or we never get `html_data`, we do **not** add to the lookup, so the next run will retry that accession instead of skipping it forever.

---

### 6. Configuration & tuning

- **Env var**: `SEC_MIN_REQ_INTERVAL`
  - Default: `0.2` (seconds) if unset.
  - Example settings:
    - `0.1` → up to ~10 rps (closer to SEC’s documented limit; less margin).
    - `0.25` → up to ~4 rps (more conservative).
- Change requires a process restart to take effect.

Recommendation: keep the default `0.2` unless you have strong evidence that you need higher throughput and SEC is not rate‑limiting at that level.

---

### 7. Limitations / future improvements

- **Per‑process only**:
  - The limiter is in‑memory. If you run multiple processes/containers from the same IP, each enforces its own rps, and the combined traffic could still exceed SEC’s global limit.
  - To coordinate across processes, you’d need a shared store (e.g. Redis) or another external mechanism; currently **not used** by design.

- **Other SEC callers**:
  - Several other modules also call `sec.gov` directly (`Eight_k_summary.py`, `sec_summarizers/fetch_utils.py`, etc.).
  - They are not yet routed through `rate_limited_get`, but they are typically lower volume and per‑filing.
  - Future hardening: import `rate_limited_get` in those modules and wrap their `requests.get` calls to share the same throttle.

Use this file as the reference when:

- Adjusting the SEC call rate, or
- Debugging future SEC 429/rate‑limit issues, or
- Extending rate limiting to other SEC HTTP callers under `sec_rss_parser`.

---

### 8. Concurrency lock and crash edge cases

To prevent duplicate processing when `process_feed_8k.py` and `fetch_sec_feed_by_deal_cik.py` hit the same accession at nearly the same time, we added a Mongo lock:

- **Model**: `AccessionProcessingLock` (`accession_number` unique, TTL on `expires_at`)
- **Helpers**: `acquire_accession_lock`, `release_accession_lock`, `mark_accession_processed`

Flow per accession:

1. If already in `AccessionLookedUp` -> skip.
2. Try to acquire lock:
   - success -> process
   - duplicate lock -> skip (another worker is processing)
3. On success, mark accession in `AccessionLookedUp`.
4. Release lock in `finally`.

Crash edge case handling:

- If a worker crashes, lock eventually expires (TTL), so the system can recover.
- On stale-lock recovery, helper checks if `SECFilingSummary` or `SECFiling` already exists:
  - if yes -> mark `AccessionLookedUp` and skip reprocessing (reduces duplicate side effects).
  - if no -> delete stale lock and allow retry.

Residual risk (cannot be fully eliminated without idempotent side effects):

- If crash happens **after external side effect (e.g., email sent)** but **before any DB artifact / looked_up marker is written**, a later retry can still duplicate that side effect.
- Full protection requires explicit idempotency keys for side-effect producers (email/webhook layer) keyed by `accession_number` + `email_type`.

