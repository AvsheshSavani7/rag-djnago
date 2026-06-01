# Email subject lines (updated formats)

This document lists **only** email types whose subject lines were updated in the recent subject-standardization work. It maps each subject pattern to the **template function** and the **code path that sends** the email.

**Deal label** in subjects is always the deal **target** ticker when available (`ProcessingJob.target_ticker` or GPT/extracted `target_ticker`), otherwise target name, otherwise `Unknown`.

**Parent vs Target** means who filed the document relative to the deal:

| Label in subject | Meaning |
|------------------|---------|
| `Parent …` / `Parent 2.1 …` | Filer CIK matches the deal **acquirer** (`matched_cik_label == "(acquirer)"`) |
| `Target …` / `Target 2.1 …` | Filer CIK matches the deal **target** (`matched_cik_label == "(target)"`) |
| `{10-digit CIK} …` | Filer CIK does not match target or acquirer on the deal record |

For EX-2.1 **filing alert** emails (no DB deal yet), Parent/Target is derived from GPT `company_details` (`target_cik` / `acquirer_cik`) vs the 8-K filer CIK, with default **Target** if unclear.

---

## 1. L1 / L2 / L3 summary (8-K, EX-99.1, periodic forms)

**Subject builder:** `_build_l123_summary_email_subject` in `sec_rss_parser/email_templates.py`  
**HTML + subject:** `generate_8k_99_1_summary_email_html`  
**Send helper:** `send_summary_email_via_webhook` in `sec_rss_parser/services.py`

### Subject format

```
{target_ticker}: Parent {form_type} [- {L1}] [SSM]
{target_ticker}: Target {form_type} [- {L1}] [SSM]
{target_ticker}: {filer_cik} {form_type} [- {L1}] [SSM]
```

- `{form_type}` comes from `summary_kind` / `form_type` (e.g. `8-K`, `EX-99.1`, `DEFM14A`, `10-Q`).
- `{L1}` is optional: first ~120 characters of L1 headline; leading `+` stripped. Omitted if no L1.

### Examples

- `UDMY: Target 8-K - Company announces merger agreement [SSM]`
- `ACME: Parent 10-Q - Q1 results beat estimates [SSM]`

### Who calls it (active flows)

| Caller | File | When |
|--------|------|------|
| `_route_summarize_and_save` | `fetch_sec_feed_by_deal_cik.py` | After AI summary for routed SEC items (8-K body, EX-99.1, proxy/periodic types in allow list) |
| `_send_8k_summary_email` | `process_feed_8k.py` | 8-K L123 summary when `cik_matches_deal` (combined or 8-K only) |
| `_send_ex99_summary_email` | `process_feed_8k.py` | EX-99.1 L123 summary when `cik_matches_deal` |
| Legacy `services.py` | `services.py` | Older RSS save path still calling `send_summary_email_via_webhook` (less used vs `process_feed_8k`) |

**Webhook:** `N8N_WEBHOOK_URL_8K_SUMMARY_L123` when `summary_kind` is in `ALLOW_EMAIL_TO_CLIENT_FORM`; else `N8N_WEBHOOK_URL_8K_SUMMARY`.

---

## 2. 10-K / 10-Q redline comparison

**Subject builder:** `_build_10k_10q_comparison_email_subject` in `sec_rss_parser/email_templates.py`  
**HTML + subject:** `generate_10k_10q_comparison_summary_email_html`

### Subject format

```
{target_ticker}: Parent {form_type} Comparison [SCM]
{target_ticker}: Target {form_type} Comparison [SCM]
{target_ticker}: {filer_cik} {form_type} Comparison [SCM]
```

- `{form_type}` is the **newest** filing in the comparison run (e.g. `10-K`, `10-Q`, `10-K/A`). Defaults to `10-K` if unknown.

### Example

- `UDMY: Target 10-Q Comparison [SCM]`

### Who calls it

| Caller | File | When |
|--------|------|------|
| `run_pipeline` (email step) | `tenK_tenQ_pipeline/orchestrator.py` | After 10-K/10-Q comparison DOCX is built and saved |

Triggered from `fetch_sec_feed_by_deal_cik._process_ten_k_ten_q_item` → `utils_10k_10q.fetch_and_save_additional_10k_10q_filings` → orchestrator.

---

## 3. Proxy comparison (change report)

**Subject builder:** same `_build_10k_10q_comparison_email_subject` (reused)  
**HTML + subject:** `generate_proxy_comparison_summary_email_html` in `sec_rss_parser/email_templates.py`

### Subject format

Same as §2, with `{form_type}` = proxy form (e.g. `DEFM14A`, `S-4`).

### Example

- `UDMY: Target DEFM14A Comparison [SCM]`

### Who calls it

| Caller | File | When |
|--------|------|------|
| `_send_proxy_comparison_email` | `fetch_sec_feed_by_deal_cik.py` | After `proxy_comparision.run_comparison` for amendment proxy types |

---

## 4. Proxy background summary (5-Q + merger background DOCX)

**Subject builder:** `_build_proxy_background_summary_email_subject` in `sec_rss_parser/email_templates.py`  
**HTML + subject:** `generate_summary_email_html` in `sec_rss_parser/proxy_processor_helper.py` (imports the builder)

### Subject format

```
{target_ticker}: Parent {form_type} Background Summary [SBM]
{target_ticker}: Target {form_type} Background Summary [SBM]
{target_ticker}: {filer_cik} {form_type} Background Summary [SBM]
```

### Example

- `UDMY: Target DEFM14A Background Summary [SBM]`

### Who calls it

| Caller | File | When |
|--------|------|------|
| `send_summary_email_notification_v2` | `proxy_processor_helper.py` | After `ProxySummaryServiceV2.generate_summary_document` completes |
| Triggered from | `fetch_sec_feed_by_deal_cik._process_proxy_item` → `process_sec_document_for_filing_summary` → `generate_proxy_summary_v2` |

**Webhook:** `N8N_WEBHOOK_SEND_TO_ALL`

Deal context: `SECFilingSummary.deal_id` + `cik_number` → `get_deal_tickers` + `ProcessingJob` for `matched_cik_label`.

---

## 5. EX-2.1 DMA summary DOCX (post-processing)

**Subject builder:** `_build_ex21_dma_summary_email_subject` in `sec_rss_parser/email_templates.py`  
**HTML + subject:** `generate_8k_summary_email_html`

### Subject format

```
{target_ticker}: Parent 2.1 - DMA Summary [SNS]
{target_ticker}: Target 2.1 - DMA Summary [SNS]
```

(No `Form` word; fixed suffix `DMA Summary`.)

### Example

- `NL: Target 2.1 - DMA Summary [SNS]`

### Who calls it

| Caller | File | When |
|--------|------|------|
| `send_8k_summary_email` | `sec_rss_parser/services.py` | After `generate_8k_summary_async` finishes DOCX for a deal (`summary_kind="EX-2.1"`) |
| `regeneration_pipeline` | `document_processor/regeneration_pipeline.py` | Manual regenerate + optional email |

Triggered from `process_feed_8k._process_ex21_via_8k_helper` → `process_8k_document_helper` → async pipeline ( **after** the filing alert email in §6).

**Webhook:** `N8N_WEBHOOK_URL_8K_SUMMARY`

Ticker / Parent–Target: `deal_id` + filer `cik_number` via `get_deal_tickers` and `ProcessingJob`, same pattern as proxy summary.

---

## 6. EX-2.1 filing alert (SEC table + M&A block + document list)

Two templates; subject differs for **new** vs **existing deal in DB**.

**Subject helpers:** `_build_ex21_filing_alert_email_subject`, `_ex21_filing_subject_ticker`, `_ex21_filing_parent_or_target_role` in `sec_rss_parser/email_templates.py`

### 6a. New deal / no matched deal in DB

**HTML + subject:** `generate_filing_email_html`

```
{target_ticker}: Parent 2.1 - New Deal Announcement [SND]
{target_ticker}: Target 2.1 - New Deal Announcement [SND]
```

- Ticker / names from GPT `company_details` on `item_data` (set **before** email in `analyze_filing`).
- Parent/Target: `matched_cik_label` if present, else GPT `acquirer_cik` / `target_cik` vs 8-K filer CIK.

### 6b. Existing deal (CIK already on file)

**HTML + subject:** `generate_filing_email_with_deal_html`

```
{target_ticker}: Parent 2.1 - New Deal Details [SNDD]
{target_ticker}: Target 2.1 - New Deal Details [SNDD]
```

- Ticker from `deal_details` / `company_details`.
- Parent/Target: `item_data['matched_cik_label']` (set in `_process_single_item` from `ProcessingJob`) or GPT CIKs as fallback.

### Examples

- `NL: Target 2.1 - New Deal Announcement [SND]` — first-time EX-2.1, target files
- `NL: Parent 2.1 - New Deal Details [SNDD]` — acquirer files, deal already tracked

### Who calls it (primary flow)

| Caller | File | When |
|--------|------|------|
| `_send_ex21_email` | `process_feed_8k.py` | Qualified EX-2.1: `document_kind == Definitive Merger Agreement` and US listed and market cap > $100M |

Uses `generate_filing_email_with_deal_html` when `deal_details` is populated (`cik_matches_deal`); else `generate_filing_email_html`.

**Webhook:** `N8N_WEBHOOK_URL_FILING` if US listed + cap > $100M; else `N8N_WEBHOOK_URL_8K_SUMMARY`

**Order:** GPT `analyze_filing` → **this email** → historical 8-K email (unchanged subject) → `_process_ex21_via_8k_helper` → later §5 DMA summary email.

---

## 7. RSS / press news (merger flow 1)

**Template:** `generate_rss_feed_item_email_html` in `rss_feeds/email_templates.py`  
**Send:** `RSSFeedService.process_webhook_payload` merger branch in `rss_feeds/services.py`

### Subject formats

**Existing deal** (`email_note == "existing_deal"`, suffix from article side: NWB / NWA / NWT):

```
{target_ticker}: {feed_display_name} - {article_title} - [NWB]
{target_ticker}: {feed_display_name} - {article_title} - [NWA]
{target_ticker}: {feed_display_name} - {article_title} - [NWT]
```

**New self-announce, US listed + cap > $100M** (`NWNDWT`):

```
{target_ticker}: {feed_display_name} - New Deal Announcement - {article_title} [NWNDWT]
```

**New self-announce, not qualified** (`NWNDW/OT`) — **unchanged** legacy format:

```
[NWNDW/OT] {feed_display_name} : {article_title}
```

### Examples

- `NVA: PR News - Acme to acquire Beta Corp - [NWA]`
- `NVA: PR News - New Deal Announcement - Acme to acquire Beta Corp [NWNDWT]`

### Webhook routing

- Client (`N8N_WEBHOOK_SEND_TO_ALL`): subject ends with ` - [NWB]` or ` - [NWT]` (`rss_subject_uses_client_webhook`)
- Internal: all other subjects (including `NWA`, `NWNDWT`, `NWNDW/OT`)

Ticker: `deal_info.target_ticker` or `target_name` (`_rss_deal_subject_label`). Not from `ProcessingJob` at email time unless already in `deal_info`.

---

## Quick reference table

| Email type | Subject builder | Template entrypoint | Primary sender |
|------------|-----------------|---------------------|----------------|
| L123 8-K / EX-99.1 / form summary | `_build_l123_summary_email_subject` | `generate_8k_99_1_summary_email_html` | `send_summary_email_via_webhook` ← `fetch_sec_feed_by_deal_cik`, `process_feed_8k` |
| 10-K / 10-Q comparison | `_build_10k_10q_comparison_email_subject` | `generate_10k_10q_comparison_summary_email_html` | `tenK_tenQ_pipeline/orchestrator.py` |
| Proxy comparison | `_build_10k_10q_comparison_email_subject` | `generate_proxy_comparison_summary_email_html` | `fetch_sec_feed_by_deal_cik._send_proxy_comparison_email` |
| Proxy background summary | `_build_proxy_background_summary_email_subject` | `generate_summary_email_html` (helper) | `proxy_processor_helper.send_summary_email_notification_v2` |
| EX-2.1 DMA DOCX summary | `_build_ex21_dma_summary_email_subject` | `generate_8k_summary_email_html` | `services.send_8k_summary_email` (after EX-2.1 processing) |
| EX-2.1 filing alert (new) | `_build_ex21_filing_alert_email_subject` | `generate_filing_email_html` | `process_feed_8k._send_ex21_email` |
| EX-2.1 filing alert (existing deal) | `_build_ex21_filing_alert_email_subject` | `generate_filing_email_with_deal_html` | `process_feed_8k._send_ex21_email` |
| RSS existing deal | (inline in template) | `generate_rss_feed_item_email_html` | `rss_feeds/services.py` |
| RSS NWNDWT | (inline in template) | `generate_rss_feed_item_email_html` | `rss_feeds/services.py` |

---

## Source files

| File | Role |
|------|------|
| `sec_rss_parser/email_templates.py` | All `_build_*_email_subject` helpers; SEC HTML templates |
| `sec_rss_parser/proxy_processor_helper.py` | Proxy background email HTML; calls proxy subject builder |
| `sec_rss_parser/services.py` | `send_summary_email_via_webhook`, `send_8k_summary_email` |
| `sec_rss_parser/fetch_sec_feed_by_deal_cik.py` | L123, proxy comparison/background triggers |
| `sec_rss_parser/process_feed_8k.py` | L123 (deal CIK), EX-2.1 filing alert |
| `sec_rss_parser/tenK_tenQ_pipeline/orchestrator.py` | 10-K/10-Q comparison email |
| `rss_feeds/email_templates.py` | RSS news subjects |
| `rss_feeds/services.py` | RSS webhook send + client routing |

---

## Not in scope (subjects not changed in this pass)

- `generate_filing_email_html` legacy subject `SEC Filing – 8-K – {company}` — **replaced** for EX-2.1 qualified path (§6); may still appear only on unused `services.py` paths.
- `generate_sec_filings_email_html` (historical 8-K list after EX-2.1)
- `generate_8k_document_email_html`, `generate_ex99_1_merger_email_html`
- `generate_rss_feed_item_email_html_flow2` (regulatory feeds)
- Parsing success/error, press release extraction, DMA extraction notification emails
