# API: Fetch SEC Feed by Deal CIK

## Overview

**Endpoint:** `GET` or `POST` `/api/sec/fetch-feed-by-deal-cik/`

**Purpose:** Start a **background job** that fetches SEC (Securities and Exchange Commission) RSS/Atom feeds per deal CIK, then processes each filing by form type (proxy, 8-K, 10-K/10-Q, other). The API returns immediately with a “processing started” response; the actual work runs in a daemon thread.

**View:** `FetchSECFeedByDealCIKView` (sec_rss_parser.views)  
**Handler:** `run_fetch_sec_feed_by_deal_cik()` in `sec_rss_parser/fetch_sec_feed_by_deal_cik.py`

---

## Base URL

- **Local:** `http://localhost:8001/api/sec/fetch-feed-by-deal-cik/`  
  (or whatever host/port your app uses; e.g. 8000)
- **Production:** `https://<your-domain>/api/sec/fetch-feed-by-deal-cik/`

---

## Request

### Method

- **GET** – parameters via query string  
- **POST** – parameters via query params or JSON body (both supported)

### Parameters

| Parameter       | Type   | Required | Description |
|----------------|--------|----------|-------------|
| `limit_deals`  | int    | No       | Max number of deals to process. If omitted, all open/unknown deals are processed. |
| `use_demo`     | bool   | No       | If `true`, `1`, or `yes`: use local demo RSS file (`sec_rss_parser/rss copy.xml`) instead of calling the live SEC API. Default: `false`. |
| `output_path`  | string | No       | Full path for the JSON output file. Default: `sec_feed_by_deal_cik_output.json` under the app directory. |

### Example: Live SEC feeds (real API)

```bash
# Process up to 10 deals using live SEC.gov feeds
curl "http://localhost:8001/api/sec/fetch-feed-by-deal-cik/?limit_deals=10"
```

```bash
# Process all open/unknown deals (no limit)
curl "http://localhost:8001/api/sec/fetch-feed-by-deal-cik/"
```

### Example: Demo mode (local RSS file)

```bash
# Use demo RSS file, limit to 5 deals
curl "http://localhost:8001/api/sec/fetch-feed-by-deal-cik/?use_demo=true&limit_deals=5"
```

### Example: Custom output path

```bash
curl "http://localhost:8001/api/sec/fetch-feed-by-deal-cik/?limit_deals=10&output_path=/path/to/output.json"
```

### Example: POST with JSON body

```bash
curl -X POST "http://localhost:8001/api/sec/fetch-feed-by-deal-cik/" \
  -H "Content-Type: application/json" \
  -d '{"limit_deals": 5, "use_demo": false}'
```

---

## Response

### Success (200 OK)

```json
{
  "success": true,
  "message": "SEC feed by deal CIK processing started in background",
  "status": "processing",
  "limit_deals": 10,
  "use_demo": false,
  "output_path": "/absolute/path/to/sec_feed_by_deal_cik_output.json"
}
```

- **`limit_deals`** – value used (or `null` if not set).  
- **`use_demo`** – whether demo RSS file was requested.  
- **`output_path`** – path where the run will write its JSON summary (when provided).

### Error (500)

```json
{
  "error": "Failed to start processing"
}
```

---

## Background Processing Flow

Once the request returns, the following runs in a background thread.

### 1. Deal and CIK source

- **Live mode** (`use_demo` false or omitted):
  - Load deals from MongoDB: `ProcessingJob` with `deal_status` in **Open** or **Unknown**.
  - For each deal, collect CIKs from `cik` and `acquirer_cik` (normalized to 10-digit).
  - Optionally cap the number of deals with `limit_deals`.

- **Demo mode** (`use_demo=true`):
  - Read RSS/Atom from `sec_rss_parser/rss copy.xml`.
  - Parse entries and infer CIK from Atom entry titles; match to deals by CIK for `deal_id`.

### 2. Fetching SEC feed (live only)

- For each CIK, call the **live SEC Edgar feed**:
  - URL:  
    `https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&CIK={cik}&type=&company=&dateb=&owner=include&start=0&count=100&output=atom`
  - Headers: `User-Agent`, `Accept`, `Referer` (SEC-friendly).
  - Retries: up to 3 attempts with backoff; 2–5 second delays to respect SEC rate limits.

### 3. Item filtering

- Parse Atom XML into “items” (one per filing).
- Skip items whose **accession number** is already in `AccessionLookedUp`.
- Deduplicate by accession; only new, unique accessions are processed.

### 4. Per-item processing

For each item, the code fetches the filing index HTML from SEC, then branches by **form type**:

| Form types | Handling |
|------------|----------|
| **Proxy (DEFM14A, DEFM14C, PREM14A, PREM14C, S-4, F-4, S-4/A, F-4/A)** | `proxy_processor_helper.process_sec_document_for_filing_summary()` – creates/updates `SECFilingSummary.proxy`; summary/email can be async. |
| **8-K** | 8-K and EX-99.1 summaries via `Eight_k_summary.summarize_8k_filing()`; upload to S3; send emails via webhook; save to `SECFilingSummary.eight_k`. |
| **10-K / 10-Q** | `utils_10k_10q.fetch_and_save_additional_10k_10q_filings()` – fetch from SEC API, save to `SECFilingSummary.ten_k_ten_q`. |
| **Other** | Summary generated (no email), saved to `SECFilingSummary.other_filings`. |

- **SECFiling** is created if the accession does not already exist.
- Processed accessions are recorded in **AccessionLookedUp** so they are skipped on future runs.

### 5. Output file

If `output_path` (or the default) is set, the run writes a JSON file with:

- `deals_processed`, `feed_fetches`, `items_count`
- `items_processed`, `processing_errors` (when processing runs)
- `errors` (e.g. feed fetch failures)
- Optional `items` (when processing is disabled)
- Optional `source`: `"rss_file"` or `"rss_content"` in demo mode

---

## Dependencies and Data

- **MongoDB:** `ProcessingJob` (deals, `cik`, `acquirer_cik`, `deal_status`), `AccessionLookedUp`, `SECFiling`, `SECFilingSummary`.
- **External:** Live SEC Edgar (browse-edgar Atom feed and filing index HTML), S3 (8-K/EX-99.1/other summaries), n8n webhook for 8-K summary emails (`N8N_WEBHOOK_URL_8K_SUMMARY`).
- **Env:** `N8N_WEBHOOK_URL_8K_SUMMARY` (optional); no auth required for the API itself (view uses `AllowAny`).

---

## Calling the real live API

To use **only** the live SEC API (no demo file):

1. Do **not** pass `use_demo` or set it to `false`/`0`.
2. Call the endpoint as in the “Live SEC feeds” examples above.

Example for 10 deals, live only:

```bash
curl "http://localhost:8001/api/sec/fetch-feed-by-deal-cik/?limit_deals=10"
```

Or with POST:

```bash
curl -X POST "http://localhost:8001/api/sec/fetch-feed-by-deal-cik/" \
  -H "Content-Type: application/json" \
  -d '{"limit_deals": 10}'
```

The backend will then fetch from `https://www.sec.gov/cgi-bin/browse-edgar?...&CIK={cik}&...&output=atom` for each deal CIK and process items as described above.
