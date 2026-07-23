# process-feed API & run_8k_processor – Step-by-Step Execution & Email Flow

This document describes the execution flow for the **process-feed** API and **run_8k_processor**, including when each document type (8-K, EX-2.1, EX-99.1) is processed, which conditions trigger emails, and which email template is used.

---

## 1. API Entry: `process-feed/`

| Step | What happens |
|------|----------------|
| **1.1** | Request hits `ProcessSECFeedView` (GET or POST). |
| **1.2** | Optional `form_type` may be read from query params (GET) or body (POST). Currently **not used** for branching. |
| **1.3** | Request enqueues one 8-K tick on the in-process worker pool (`sec_8k_work_queue.py`). Returns immediately with `status: queued` or `already_running` (coalesced). |
| **1.4** | Worker(s) call `run_8k_processor()` — default **1 worker**, configurable to **2** via `SEC_8K_PROCESSOR_WORKERS`. |

**Code:** `sec_rss_parser/views.py` → `ProcessSECFeedView.process_feed_request()` → `run_8k_processor()`.

---

## 2. run_8k_processor() Entry Point

| Step | What happens |
|------|----------------|
| **2.1** | `run_8k_processor(rss_content=None, rss_file=None)` is called. |
| **2.2** | If `rss_file` is set: RSS is read from that file (testing). If `rss_content` is set: that string is used (testing). Otherwise the live SEC 8-K feed is fetched inside `processor.run()`. |
| **2.3** | `EightKFeedProcessor()` is instantiated; `processor.run(rss_content=..., rss_file=...)` is invoked. |
| **2.4** | Return value is the dict returned by `processor.run()` (e.g. `success`, `total_items`, `new_items`, `ex21_processed`, `ex99_processed`, etc.). |

**Code:** `sec_rss_parser/process_feed_8k.py` → `run_8k_processor()` → `EightKFeedProcessor().run()`.

---

## 3. EightKFeedProcessor.run() – High-Level Steps

| Step | What happens |
|------|----------------|
| **3.1** | **RSS source:** Use `rss_file` / `rss_content` if provided; else fetch live 8-K feed via `SECRSSParser(form_type='8-K')`. |
| **3.2** | **Parse feed:** `parser.parse_rss_content(rss_content)` → list of `items` (each item = one 8-K filing from the feed). |
| **3.3** | **Filter by accession:** `_filter_unique_items(items)` → only items whose `accession_number` is **not** in `AccessionLookedUp` and **not** in `SECFiling`. New accessions are bulk-inserted into `AccessionLookedUp`. |
| **3.4** | **Process each item:** For each `item_data` in `unique_items`, call `_process_single_item(item_data)`. |
| **3.5** | **Summary:** Log and return counts (processed, ex21, ex99, summary_8k, summary_ex99, skipped, errors). |

---

## 4. _process_single_item() – Per-Filing Flow

For each 8-K item:

| Step | What happens |
|------|----------------|
| **4.1** | If no `link` (HTML URL): skip; increment `skipped_count`. |
| **4.2** | **Fetch filing details:** `parser.fetch_and_parse_html(html_url, form_type_from_feed='8-K')` → `html_data` (includes `filing_array`, `form_type`, `cik_number`, etc.). |
| **4.3** | Merge `html_data` into `item_data`. **filing_array** is built from the SEC document table: each entry has `document_type` in **{'8-K', 'EX-2.1', 'EX-99.1'}** (from file `type`/`description`: EX-2.1, EX-99.1, or 8-K). |
| **4.4** | If no `filing_array`: skip (“No 8-K / EX-2.1 / EX-99.1 .htm documents”); increment `skipped_count`. |
| **4.5** | **CIK check:** `_check_cik_matches_deal(cik_number)` → `(cik_matches_deal, deal_id)`. Stored in `item_data['cik_matches_deal']` and `item_data['deal_id']`. |
| **4.6** | **Save filing once:** `_save_filing(item_data)` → create `SECFiling` (if not exists). |
| **4.7** | **Loop over filing_array:** For each `filing` in `filing_array`, read `doc_type = filing.get('document_type')`: |

---

## 5. Branch by document_type (8-K / EX-2.1 / EX-99.1)

Values of `document_type` come from the SEC filing’s document table (parsed in `utils_8k.fetch_and_parse_html`):

- **`8-K`** – main 8-K document (type/description contains “8-K”).
- **`EX-2.1`** – exhibit (type/description contains “EX-2.1”).
- **`EX-99.1`** – exhibit (type/description contains “EX-99.1”).

For each entry in `filing_array`:

| document_type | Handler | What runs |
|---------------|---------|-----------|
| **8-K** | `_process_8k_document(item_data, filing)` | See § 6. |
| **EX-99.1** | `_process_ex99_filing(item_data, filing)` | See § 7. |
| **EX-2.1** | `_process_ex21_filing(item_data, filing)` | See § 8. |

Other document types in the table are not added to `filing_array` (only 8-K, EX-2.1, EX-99.1 with `.htm` are).

---

## 6. When value is **8-K** – _process_8k_document

| Step | Condition | What happens |
|------|-----------|--------------|
| 6.1 | No 8-K URL in `filing_entry` | Return; no email. |
| 6.2 | **`cik_matches_deal` is True** | Generate 8-K summary (`summarize_8k_filing`), save to `SECFilingSummary.eight_k`, then **send email**. |
| 6.3 | **`cik_matches_deal` is False** | Run GPT analysis (`document_analyzer.analyze_ex99_1_filing`). Then: |
| 6.4 | All of **is_merger_related**, **is_target_us_listed**, **is_target_market_cap_greater_than_100m** are True | **Send email.** |
| 6.5 | Any of the above is False | **Do not send email** (log: “8-K main email skipped (need is_merger_related, is_target_us_listed, and market cap > $100M)”). |

### 6. When email is sent for 8-K

| Scenario | Email template | Webhook | Email type |
|----------|----------------|---------|------------|
| CIK matches deal (summary path) | `generate_8k_99_1_summary_email_html(..., summary_kind='8-K')` | `N8N_WEBHOOK_URL_8K_SUMMARY` | 8-K summary document |
| No deal match but GPT criteria met | `generate_8k_document_email_html(...)` | `N8N_WEBHOOK_URL_8K_SUMMARY` | `8k_gpt` |

- **Template (summary):** `email_templates.generate_8k_99_1_summary_email_html` – “New 8-K Summary Document – …”, link to S3 summary doc, optional L1 headline.
- **Template (GPT):** `email_templates.generate_8k_document_email_html` – “8-K – {company_name}”, main 8-K document analyzed for M&A relevance, confidence, reasoning, US listed, market cap.

---

## 7. When value is **EX-99.1** – _process_ex99_filing

| Step | Condition | What happens |
|------|-----------|--------------|
| 7.1 | No EX-99.1 URL in `filing_entry` | Return; no email. |
| 7.2 | **`cik_matches_deal` is True** | Generate EX-99.1 summary, append to `SECFilingSummary.eight_k.filings[]`, then **send email**. |
| 7.3 | **`cik_matches_deal` is False** | Run GPT (`analyze_ex99_1_filing`). Then: |
| 7.4 | All of **is_merger_related**, **is_target_us_listed**, **is_target_market_cap_greater_than_100m** are True | **Send email.** |
| 7.5 | Any of the above is False | **Do not send email** (log: “EX-99.1 main email skipped (need …)”). |

### 7. When email is sent for EX-99.1

| Scenario | Email template | Webhook | Email type |
|----------|----------------|---------|------------|
| CIK matches deal (summary path) | `generate_8k_99_1_summary_email_html(..., summary_kind='EX-99.1')` | `N8N_WEBHOOK_URL_8K_SUMMARY` | EX-99.1 summary document |
| No deal match but GPT criteria met | `generate_ex99_1_merger_email_html(...)` | `N8N_WEBHOOK_URL_8K_SUMMARY` | `ex99_1_merger` |

- **Template (summary):** Same builder as 8-K summary but with `summary_kind='EX-99.1'` and form_type `'8-K (EX-99.1)'`.
- **Template (merger):** `email_templates.generate_ex99_1_merger_email_html` – “EX-99.1 M&A-Related Press Release – {company_name}”, confidence, reasoning, US listed, market cap.

---

## 8. When value is **EX-2.1** – _process_ex21_filing

| Step | Condition | What happens |
|------|-----------|--------------|
| 8.1 | No EX-2.1 URL in `filing_entry` | Return; no email. |
| 8.2 | **Always** | Analyze filing (`document_analyzer.analyze_filing`) → `document_kind`, `company_details` (e.g. `is_target_us_listed`, `is_target_market_cap_greater_than_100m`). Update `SECFiling` with `document_kind` and `company_details`. |
| 8.3 | **Always** | **Send EX-2.1 email** (no gate on US listed / market cap for this first email). |
| 8.4 | **is_target_us_listed and is_target_market_cap_greater_than_100m** both True | Send **historical 8-K filings email** (last 1 year); call **8-K document helper** (Node API); increment `ex21_processed_count`. |
| 8.5 | Otherwise | No historical email, no helper; log “Not qualified for 8-K EX-2.1 document processing”. |

### 8. When email is sent for EX-2.1

| Email | When | Email template | Webhook (chosen by) | Email type |
|-------|------|----------------|----------------------|------------|
| **EX-2.1 filing** | Always (once per EX-2.1 document) | `generate_filing_email_html(...)` | If US listed and market cap > $100M → `N8N_WEBHOOK_URL_FILING`, else → `N8N_WEBHOOK_URL_8K_SUMMARY` | `ex21_merger` |
| **Historical 8-K filings** | Only when EX-2.1 is “qualified” (US listed + market cap > $100M) | `generate_sec_filings_email_html(company_name, filings, form_type="8-K(EX-2.1)")` | `N8N_WEBHOOK_URL_8K_SUMMARY` | `sec_filings_last_year` |

- **EX-2.1 template:** `email_templates.generate_filing_email_html` – “SEC Filing – 8-K – {company_name}”, accession, dates, company details (target/acquirer, US listed, market cap), document table.
- **Historical template:** `email_templates.generate_sec_filings_email_html` – “SEC Form Filings (8-K(EX-2.1)) – {company_name}”, table of filings (date, form, document link).

---

## 9. Email Templates Reference (email_templates.py)

| Function | Subject / purpose | Used for |
|----------|-------------------|----------|
| `generate_filing_email_html(filing_data, doc_files)` | “SEC Filing – {form_type} – {company_name}”; company details + doc table | EX-2.1 filing email |
| `generate_8k_document_email_html(filing_data, doc_files)` | “8-K – {company_name}”; main 8-K, confidence, merger-related, reasoning | 8-K GPT path (no deal match, criteria met) |
| `generate_ex99_1_merger_email_html(filing_data, doc_files)` | “EX-99.1 M&A-Related Press Release – {company_name}” | EX-99.1 merger path (no deal match, criteria met) |
| `generate_8k_99_1_summary_email_html(..., summary_kind, l1_headline=...)` | “New {summary_kind} Summary Document – …”; link to S3 summary doc | 8-K summary and EX-99.1 summary (when CIK matches deal) |
| `generate_sec_filings_email_html(company_name, filings, form_type)` | “SEC Form Filings ({form_type}) – {company_name}”; table of filings | Historical 8-K filings (EX-2.1 qualified) |

---

## 10. Quick Reference: When Do We Send Email?

| Document type | Condition to send | Template |
|---------------|-------------------|----------|
| **8-K** | CIK matches deal → always (summary email). Else: only if is_merger_related AND is_target_us_listed AND market_cap > $100M (GPT email). | Summary: `generate_8k_99_1_summary_email_html` (8-K). GPT: `generate_8k_document_email_html`. |
| **EX-99.1** | CIK matches deal → always (summary email). Else: only if is_merger_related AND is_target_us_listed AND market_cap > $100M. | Summary: `generate_8k_99_1_summary_email_html` (EX-99.1). Merger: `generate_ex99_1_merger_email_html`. |
| **EX-2.1** | **Always** one email per EX-2.1 document. **If** US listed and market cap > $100M: **additionally** historical 8-K filings email. | EX-2.1: `generate_filing_email_html`. Historical: `generate_sec_filings_email_html` (“8-K(EX-2.1)”). |

---

## 11. Webhooks

- **N8N_WEBHOOK_URL_8K_SUMMARY** – 8-K/EX-99.1 summaries, 8-K GPT email, EX-99.1 merger email, EX-2.1 email when not “qualified”, historical 8-K email.
- **N8N_WEBHOOK_URL_FILING** – EX-2.1 email only when **is_target_us_listed** and **is_target_market_cap_greater_than_100m** (qualified EX-2.1).

All emails are sent via `send_webhook_notification(webhook_url, payload, label)` (no direct SMTP from this flow).
