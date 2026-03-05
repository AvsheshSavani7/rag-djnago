# 10-K/10-Q Summary Pipeline: Migration to S3 + MongoDB

## Goal

- **DB:** Use MongoDB `sec_filing_summary` collection (and its schema) instead of local `filings_db.json`.
- **Files:** Store all generated artifacts in S3 under folder **`10K_10Q`**; store only S3 URLs in the DB (no local paths).
- **Entry point:** `run_pipeline(urls, deal_id)` — only URLs and `deal_id` required. **Ticker is not passed:** it is resolved from `deal_id` via `ProcessingJob.target_ticker` (or `target_name` fallback) inside the orchestrator.

---

## Current Flow (summary)

| Step | Where | What |
|------|--------|------|
| 1 | `test_10k_10Q_pipeline.py` | Calls `run_pipeline(deal_id, ticker, urls, output_dir, db_path, env_path)`. |
| 2 | `orchestrator.run_pipeline` | Loads `FilingDB(db_path)` (JSON file), creates `output_dir`. |
| 3 | Orchestrator | For each URL: `db.upsert_by_url(url, {deal_id})`; then detect metadata (period_date, filing_type, label) and update record. |
| 4 | Orchestrator | Deal context: load from `output_dir / f"{ticker}_deal_context.json"` or fetch via Perplexity and save there. |
| 5 | Orchestrator | For each URL with `processed != True`: fetch HTML → parse → score → assess → write `*_excerpts.json` and `*_fulsome_report.docx` to `output_dir`; update DB with `local_json_path`, `local_docx_path`, `processed`, etc. |
| 6 | Orchestrator | Comparison: `db.get_by_deal_id(deal_id)` → filter processed, sort by period_date; load excerpts via `load_excerpts(Path(record["local_json_path"]))`; generate redline, client_report, comparison.json, exec_summary to `output_dir`; update newest record with `local_*_path` for those four files. |
| 7 | Return | `{ processed, skipped, comparison_outputs }` with Path values. |

**Files that touch DB or paths:**

- **`tenK_tenQ_pipeline/db.py`** — `FilingDB`: JSON load/save, `get_by_url`, `get_by_deal_id`, `upsert_by_url`, `update`. All references to `local_*_path` and `_id`.
- **`tenK_tenQ_pipeline/orchestrator.py`** — Uses `FilingDB`, `output_dir` for all outputs and deal context cache; passes paths to `generate_excerpts_json`, `generate_single_filing_report_fulsome`, `load_excerpts`, and docx_builder functions; updates DB with path strings.
- **`tenK_tenQ_pipeline/excerpts.py`** — `generate_excerpts_json(..., output_path: Path)` writes JSON to file; `load_excerpts(filepath: Path, ...)` reads JSON from file. No DB.
- **`tenK_tenQ_pipeline/docx_builder.py`** — All generators take `Path` and write DOCX to that path. No DB.
- **`tenK_tenQ_pipeline/deal_context.py`** — Fetches by `ticker`; `DealContext.save(path)` / `load(path)` are file-based.
- **`tenK_tenQ_pipeline/sec_fetcher.py`** — `detect_filing_metadata(url, html)` → period_date, filing_type; no DB/paths except URL.
- **`test_10k_10Q_pipeline.py`** — Passes `output_dir`, `db_path`, `env_path`; prints result and reads `filings_db.json` for verification.

---

## Target Flow (after migration)

| Step | What |
|------|------|
| 1 | Call `run_pipeline(urls, deal_id, ticker=None, env_path=None, ...)`. If `ticker` is None, resolve from `ProcessingJob.objects(id=deal_id).first().target_ticker` (optional fallback: keep ticker required or derive from first URL / company). |
| 2 | **DB:** Use MongoDB via `SECFilingSummary` (sec_rss_parser.models). Implement a thin “summary DB” layer: get by `sec_document_url` (or by `accession_number` + `form_type`), upsert by URL (create doc if not exists), update by doc id. Records are SECFilingSummary docs with `form_type` in ["10-K","10-Q"] and `ten_k_ten_q` payload. |
| 3 | **URL → document:** From each SEC URL derive `cik_number` and `accession_number` (and `form_type` from existing `detect_filing_metadata`). If no SECFilingSummary exists for that URL/accession+form, create one with `ten_k_ten_q = { processed: False, ... }` (same shape as in `utils_10k_10q.py`). |
| 4 | **Deal context:** Keep file-based cache for simplicity: e.g. temp dir or a single local path (e.g. `output_dir` defaulting to `tempfile.mkdtemp()` or a fixed dir). So we still have a small “working dir” for deal_context JSON and for **temporary** generated files before upload. |
| 5 | **Process each filing:** Generate excerpts JSON and fulsome DOCX into **temp/local files** → upload to S3 under **`10K_10Q/`** (e.g. `10K_10Q/{accession_or_label}/excerpts.json`, `10K_10Q/{accession_or_label}/fulsome_report.docx`) → update that filing’s `ten_k_ten_q` with `s3_json_url`, `s3_docx_url`, `processed=True`, `processed_at`, `label`. |
| 6 | **Comparison:** Load “processed” filings for `deal_id` from MongoDB; for each record that has `s3_json_url`, **download that JSON to a temp file** (or add a `load_excerpts_from_url` helper that fetches and parses) and call existing `load_excerpts(temp_path)`. Generate comparison outputs to temp files → upload to S3 under `10K_10Q/` (e.g. `10K_10Q/comparison/{deal_id}_{timestamp}_redline.docx`, etc.) → update the **newest** filing’s `ten_k_ten_q` with `s3_comparison_json_url`, `s3_redline_docx_url`, `s3_client_report_docx_url`, `s3_exec_summary_docx_url`. |
| 7 | Return `{ processed, skipped, comparison_outputs }` where `comparison_outputs` values are **S3 URLs** (or Paths to temp files if you want to keep local copies for tests). |

---

## Schema Alignment: filings_db.json → SECFilingSummary.ten_k_ten_q

**Current JSON record (filings_db.json):**

- `_id`, `sec_document_url`, `deal_id`, `cik_number`, `accession_number`, `filing_date`, `period_date`, `filing_type`, `label`
- `processed`, `processed_at`
- `local_json_path`, `local_docx_path`
- `local_comparison_json_path`, `local_redline_docx_path`, `local_client_report_docx_path`, `local_exec_summary_docx_path`
- `created_at`, `updated_at`

**SECFilingSummary (top-level):**

- `_id`, `accession_number`, `cik_number`, `sec_document_url`, `filing_date`, `deal_id`, `form_type`, `created_at`, `updated_at`
- `proxy`, `ten_k_ten_q`, `eight_k`, `other_filings`

**ten_k_ten_q payload (already in use in utils_10k_10q / 10K_10Q_PROCESSING.md):**

- `processed`, `processed_at`
- `s3_json_url`, `s3_docx_url`
- `s3_comparison_json_url`, `s3_redline_docx_url`, `s3_client_report_docx_url`, `s3_exec_summary_docx_url`
- `label` (optional)

So the mapping is:

- **Without DB:** All `local_*_path` fields → **S3 URLs** stored in `ten_k_ten_q` as `s3_*_url`.
- **Lookup key:** By `sec_document_url` or by `(accession_number, form_type)`.
- **Metadata:** `period_date`, `filing_type`, `label` can live in `ten_k_ten_q` (e.g. add `period_date`, `filing_type` there if you want them queryable without re-parsing; otherwise they can be derived from excerpts metadata).

---

## Required Code Changes (by file)

### 1. **New: `tenK_tenQ_pipeline/summary_db.py` (or replace `db.py`)**

- **Option A:** New module `summary_db.py` that uses `SECFilingSummary`:
  - `get_by_url(url)` → query `SECFilingSummary.objects(sec_document_url=url).first()` (or by accession+form_type after parsing URL).
  - `get_by_deal_id(deal_id)` → `SECFilingSummary.objects(deal_id=deal_id, form_type__in=["10-K","10-Q"]).all()`.
  - `upsert_by_url(url, fields)` → parse URL for CIK/accession; get or create SECFilingSummary; if create, set `ten_k_ten_q = { processed: False, ... }`, `form_type` from detect_filing_metadata or URL.
  - `update(record_id, fields)` → update by MongoEngine doc: either by `_id` or by `accession_number`+`form_type`; write updates into `ten_k_ten_q` (and top-level `filing_date`, `accession_number`, `cik_number` if needed).
- **Option B:** Refactor existing `db.py` to a backend interface and implement `FilingDBJson` (current) and `FilingDBMongo` (SECFilingSummary). Orchestrator then takes the backend; test script uses Mongo when running in “real” mode.

Recommendation: **Option A** — single implementation for MongoDB; remove dependency on JSON file for production.

- **URL parsing:** Add helper (e.g. in `sec_fetcher.py` or `summary_db.py`): `parse_sec_document_url(url) -> (cik_number, accession_number)`. Pattern: `https://www.sec.gov/Archives/edgar/data/{cik}/{acc_raw}/...`. Accession with dashes: 18-char `acc_raw` → `f"{acc_raw[:10]}-{acc_raw[10:12]}-{acc_raw[12:]}"` (SEC format NNNNNNNNNN-NN-NNNNNN).

### 2. **New: S3 upload helper under folder `10K_10Q`**

- Reuse existing pattern (e.g. `Eight_k_summary._upload_to_s3`, `_upload_json_to_s3`) or add a small `tenK_tenQ_pipeline/s3_utils.py` that:
  - Uses same env: `AWS_S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`.
  - Prefix: **`10K_10Q/`**.
  - Functions: e.g. `upload_file_to_s3(local_path, s3_key) -> s3_url`, `upload_json_to_s3(data, s3_key) -> s3_url`. Keys could be e.g. `10K_10Q/{accession}/{filename}` or `10K_10Q/{deal_id}/{label}/{filename}` so they are unique and readable.

### 3. **`tenK_tenQ_pipeline/orchestrator.py`**

- **Signature:** `run_pipeline(urls, deal_id, ticker=None, output_dir=None, env_path=None, ...)`.
  - If `ticker` is None: try `ProcessingJob.objects(id=deal_id).first().target_ticker` (requires Django/app setup and `document_processor.models.ProcessingJob`). If still None, raise or use a fallback (e.g. “UNKNOWN”).
  - If `output_dir` is None: use `tempfile.mkdtemp()` (or a config default) for deal context cache and temp files.
- **DB:** Replace `FilingDB(db_path)` with the new MongoDB summary layer (e.g. `from .summary_db import SummaryDB` and `db = SummaryDB()`).
- **Reading “record”:** Use `record` from MongoDB; `record["_id"]` for updates; for “processed” check `record.ten_k_ten_q.get("processed")`; for paths use `record.ten_k_ten_q.get("s3_json_url")`, etc.
- **After generating excerpts JSON and fulsome DOCX:** Upload to S3 (folder `10K_10Q`), then update DB with `s3_json_url`, `s3_docx_url`, `processed`, `processed_at`, `label` (and optionally `period_date`, `filing_type` in `ten_k_ten_q`).
- **Comparison:** When loading excerpts for prior/newest, if record has `s3_json_url`, download that JSON to a temp file (e.g. `requests.get(s3_url)` or boto3 get_object → write to tempfile), then call `load_excerpts(Path(temp_path), threshold)`. After generating comparison outputs, upload each to S3 and update newest doc’s `ten_k_ten_q` with the four S3 URLs.
- **Return:** `comparison_outputs` can be dict of S3 URL strings (no Path objects).

### 4. **`tenK_tenQ_pipeline/excerpts.py`**

- **No change required** if we keep writing to a temp path and then uploading. Optionally add `load_excerpts_from_url(url)` that fetches JSON and parses in memory (so we don’t need to write temp file for S3 JSON), then reuse the same parsing logic as `load_excerpts`. This avoids temp files for comparison step.

### 5. **`tenK_tenQ_pipeline/docx_builder.py`**

- No change: still accepts `Path` and writes DOCX. Orchestrator will pass a temp path, then upload the file to S3.

### 6. **`tenK_tenQ_pipeline/deal_context.py`**

- No change to API. Orchestrator passes a path under `output_dir` (or temp dir) for cache file.

### 7. **`tenK_tenQ_pipeline/sec_fetcher.py`**

- Add `parse_sec_document_url(url) -> (cik_number, accession_number)` as above (or in `summary_db.py`). Used when creating/upserting SECFilingSummary from URL.

### 8. **`test_10k_10Q_pipeline.py`**

- Call `run_pipeline(urls=URLS, deal_id=DEAL_ID)` (and optionally `ticker=TICKER`, `env_path=ENV_PATH`).
- Remove `output_dir`, `db_path` from the call (or keep as optional overrides).
- For “verify DB”: instead of reading `filings_db.json`, query MongoDB: e.g. `SECFilingSummary.objects(deal_id=DEAL_ID, form_type__in=["10-K","10-Q"])` and print `ten_k_ten_q.processed`, `ten_k_ten_q.s3_json_url`, etc.

---

## Potential Issues and Decisions

| Issue | Recommendation |
|-------|----------------|
| **Django/DB setup** | Orchestrator (and test script) must run with Django setup and MongoDB so that `SECFilingSummary` and `ProcessingJob` are available. Test script should set `DJANGO_SETTINGS_MODULE` and `django.setup()` if not already. |
| **Existing SECFilingSummary docs** | `utils_10k_10q.fetch_and_save_additional_10k_10q_filings` already creates SECFilingSummary with `ten_k_ten_q = { processed: False, ... }`. The new pipeline will **update** those same docs (by `sec_document_url` or accession+form_type) when generating summaries. So no duplicate docs. |
| **Uniqueness** | SECFilingSummary does not enforce unique `sec_document_url` in the schema; indexes include `accession_number`, `form_type`. Prefer lookup/upsert by `(accession_number, form_type)` to avoid duplicates when the same filing is referenced by slightly different URLs. |
| **Ticker optional** | Making `ticker` optional and resolving from `deal_id` requires a DB round-trip and ties the pipeline to `ProcessingJob`. Alternatively keep `ticker` as a required argument for the standalone script so the script stays decoupled from deal DB. |
| **S3 key design** | Use a consistent key scheme under `10K_10Q/`, e.g. `10K_10Q/{accession_number}/{filename}` for per-filing artifacts and `10K_10Q/comparison/{deal_id}_{timestamp}_{name}.{ext}` for comparison outputs, to avoid overwrites and keep debugging easy. |
| **Download vs stream for load_excerpts** | For comparison, we need excerpt JSON. Easiest: download S3 JSON to a temp file and call `load_excerpts(path)`. Alternative: implement `load_excerpts_from_url(url)` that `requests.get(url)` and parses JSON; then no temp file. |
| **Deal context cache** | Keeping deal context in a local file (under temp or a fixed dir) is simplest. No need to put it in S3 unless you want to share across runs/machines. |

---

## Summary

- **Replace** local JSON DB with MongoDB `sec_filing_summary` and a small summary-DB layer that uses `SECFilingSummary` and the existing `ten_k_ten_q` payload.
- **Replace** all local file paths with S3 uploads under folder **`10K_10Q`** and store only S3 URLs in `ten_k_ten_q` (no DB schema change beyond using existing fields).
- **Simplify** entry point to `run_pipeline(urls, deal_id, ticker=None, ...)`; resolve `ticker` from `deal_id` when not provided if you want minimal arguments.
- **Keep** generating to temp/local paths and then uploading to S3 so that `excerpts.py` and `docx_builder.py` stay unchanged; add URL→(cik, accession) parsing and S3 helpers; add logic to download S3 JSON to temp path (or add load_excerpts_from_url) for the comparison step.

If you confirm this plan, the next step is to implement the changes in the order above (summary_db + URL parsing → S3 helpers → orchestrator → test script).
