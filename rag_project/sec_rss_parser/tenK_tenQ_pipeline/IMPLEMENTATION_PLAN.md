# 10-K/10-Q Pipeline: S3 + MongoDB Implementation Plan

Use this file to track progress. Steps are ordered; complete each before moving to the next.

---

## Step 1: URL parsing helper

**Goal:** From an SEC document URL, derive `cik_number` and `accession_number` (SEC format with dashes).

**File:** `tenK_tenQ_pipeline/sec_fetcher.py`

- [x] Add function `parse_sec_document_url(url: str) -> tuple[str, str]` returning `(cik_number, accession_number)`.
- [ ] URL pattern: `https://www.sec.gov/Archives/edgar/data/{cik}/{acc_raw}/...` (acc_raw is 18 chars, no dashes).
- [ ] Accession with dashes: `{acc_raw[:10]}-{acc_raw[10:12]}-{acc_raw[12:]}` (SEC format NNNNNNNNNN-NN-NNNNNN).
- [x] Return `(None, None)` or raise if URL does not match.

**Dependency:** None.

---

## Step 2: summary_db.py (Option A — MongoDB only)

**Goal:** Replace JSON FilingDB with a MongoDB layer using `SECFilingSummary`.

**File:** `tenK_tenQ_pipeline/summary_db.py` (new)

- [x] Implement `SummaryDB` class (no constructor args, or optional for testing).
- [x] `get_by_url(url: str)` → `SECFilingSummary.objects(sec_document_url=url).first()`. Return doc as dict-like (or MongoEngine doc) so orchestrator can use `record.ten_k_ten_q`, `record.id`, etc. If returning doc, convert to a simple dict for `_id` / `ten_k_ten_q` access (or use doc.id and doc.ten_k_ten_q).
- [x] `get_by_deal_id(deal_id: str)` → `SECFilingSummary.objects(deal_id=deal_id, form_type__in=["10-K", "10-Q"]).all()`. Return list of docs (or list of dicts with `_id`, `ten_k_ten_q`, `sec_document_url`, etc.).
- [x] `upsert_by_url(url: str, fields: dict)`:
  - Parse URL with `parse_sec_document_url`; get `form_type` from `detect_filing_metadata(url)` (no HTML).
  - If `SECFilingSummary.objects(sec_document_url=url).first()` exists, update it with `fields` (merge into `ten_k_ten_q` and top-level where applicable), then return.
  - Else create new `SECFilingSummary`: `accession_number`, `cik_number`, `sec_document_url`, `deal_id` from fields, `form_type`, `filing_date` optional, `ten_k_ten_q` = `{ "processed": False, ... }`, `proxy`/`eight_k`/`other_filings` = None.
  - Return the doc (or its dict).
- [x] `update(record_id, fields)` → Find by `id=record_id`; update `ten_k_ten_q` with fields; record_id is the document `id` (str).

**Imports:** `sec_rss_parser.models.SECFilingSummary`, `sec_fetcher.parse_sec_document_url`, `sec_fetcher.detect_filing_metadata`. Handle optional Django/app setup (import inside function if needed to avoid circular import).

**Dependency:** Step 1 done.

---

## Step 3: S3 upload helpers (folder 10K_10Q)

**Goal:** Upload files/JSON to S3 under prefix `10K_10Q/` and return S3 URL.

**File:** `tenK_tenQ_pipeline/s3_utils.py` (new)

- [x] Use same env as rest of app: `AWS_S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` (default `us-east-1`).
- [x] Define prefix `S3_PREFIX = "10K_10Q"`.
- [x] `upload_file(local_path, s3_key_suffix, content_type=None) -> str`: full key = `10K_10Q/{s3_key_suffix}`; upload file; return S3 URL.
- [x] `upload_json(data, s3_key_suffix) -> str`: same; upload JSON bytes; return S3 URL.
- [x] Key convention: per-filing `{accession_number}/excerpts.json` etc.; comparison `comparison/{deal_id}_{timestamp}_{name}.{ext}`.

**Dependency:** None (can mirror Eight_k_summary S3 helpers).

---

## Step 4: Resolve ticker from deal_id

**Goal:** Remove `ticker` from the public API; resolve it inside the orchestrator when needed.

**File:** `tenK_tenQ_pipeline/orchestrator.py` (later in Step 5) or a small helper.

- [x] Add helper `_get_ticker_for_deal(deal_id: str) -> str`: `ProcessingJob.objects(id=deal_id).first()`; return `target_ticker` (or `target_name` fallback) or raise ValueError if not found.
- [x] Use in `run_pipeline`: at start, `ticker = _get_ticker_for_deal(deal_id)`.

**Dependency:** Django + document_processor.models.ProcessingJob.

---

## Step 5: Orchestrator changes

**Goal:** Use SummaryDB, S3, temp dir, and only `urls` + `deal_id` (no ticker in signature).

**File:** `tenK_tenQ_pipeline/orchestrator.py`

- [x] **Signature:** `run_pipeline(urls, deal_id, output_dir=None, env_path=None, ...)`. No `ticker` or `db_path`.
- [x] Resolve ticker: `ticker = _get_ticker_for_deal(deal_id)`.
- [x] If `output_dir` is None: `output_dir = Path(tempfile.mkdtemp())`.
- [x] **DB:** Use `SummaryDB()`.
- [x] **Record shape:** SummaryDB returns dict with `_id`, `processed`, `s3_json_url`, etc. (see summary_db._doc_to_record).
- [x] **Deal context cache path:** `output_dir / f"{deal_id}_deal_context.json"`.
- [x] **After generating excerpts + DOCX:** Upload to S3 via s3_utils; update record with s3_json_url, s3_docx_url, processed, etc.
- [x] **Comparison:** Use `load_excerpts_from_url(record["s3_json_url"], ...)` (no temp file).
- [x] **Comparison outputs:** Generate to temp; upload to S3; update newest record's ten_k_ten_q with four S3 URLs.
- [x] **Return:** comparison_outputs values are S3 URL strings.

**Dependency:** Steps 1, 2, 3, 4.

---

## Step 6: Excerpts — optional load from URL

**Goal:** Avoid writing S3 JSON to temp file for comparison if we add a URL loader.

**File:** `tenK_tenQ_pipeline/excerpts.py`

- [x] Add `load_excerpts_from_url(url, threshold, source_label=None)`: fetch URL, parse JSON, reuse _parse_excerpts_data.
- [x] Orchestrator uses `load_excerpts_from_url(record["s3_json_url"], threshold, source_label=record.get("label"))`.

**Dependency:** None (can be done anytime).

---

## Step 7: Test script update

**Goal:** Call new API and verify via MongoDB.

**File:** `sec_rss_parser/test_10k_10Q_pipeline.py`

- [x] Call `run_pipeline(urls=URLS, deal_id=DEAL_ID, env_path=ENV_PATH)` (no ticker, output_dir, db_path).
- [x] Do not read `filings_db.json`; verify via `SECFilingSummary.objects(deal_id=..., form_type__in=["10-K","10-Q"])`.
- [x] Django setup when run as __main__ (path + DJANGO_SETTINGS_MODULE + django.setup()).

**Dependency:** Steps 1–5.

---

## Step 8: Config / docs

- [x] config.py unchanged.
- [x] MIGRATION_S3_MONGO.md / IMPLEMENTATION_PLAN.md state: entry `run_pipeline(urls, deal_id)`; ticker from `deal_id`; DB `sec_filing_summary`; S3 `10K_10Q/`.

---

## Completion checklist

- [x] Step 1 — URL parsing
- [x] Step 2 — summary_db.py
- [x] Step 3 — s3_utils.py
- [x] Step 4 — Ticker from deal_id
- [x] Step 5 — Orchestrator
- [x] Step 6 — load_excerpts_from_url
- [x] Step 7 — Test script
- [x] Step 8 — Config/docs

---

## If you get stuck

- **Import errors:** Ensure `run_pipeline` is invoked in a context where Django is set up and `sec_rss_parser` (and `document_processor`) are on the path. For standalone script, set `sys.path` and `DJANGO_SETTINGS_MODULE` then `django.setup()` before importing orchestrator.
- **SECFilingSummary not found:** Use `from sec_rss_parser.models import SECFilingSummary` when running from `rag_project` as cwd.
- **S3 upload fails:** Check `AWS_S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`; ensure bucket and keys have write access.
- **ProcessingJob.target_ticker None:** Deal may not have ticker set; either set it in DB or allow optional fallback (e.g. use "UNKNOWN" and still fetch deal context, or skip deal context and use minimal scoring).
