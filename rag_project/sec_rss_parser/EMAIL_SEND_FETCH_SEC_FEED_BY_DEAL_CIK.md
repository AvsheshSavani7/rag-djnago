# Emails sent from `fetch_sec_feed_by_deal_cik.py`

This document lists **when** each email is sent, **subject/content**, and **which functions** generate and send them. It only covers code paths triggered by `sec_rss_parser/fetch_sec_feed_by_deal_cik.py`.

---

## 1. Summary emails (8-K, EX-99.1, and other form types) — **active path**

### When

- **Trigger:** For each feed item, `process_items()` calls `_route_summarize_and_save(item_data, html_data)` (no form-type skip in current code).
- **When email is sent:** After a summary is generated and uploaded to S3, for **each** document summarized:
  - **8-K:** One email for the main 8-K document; if EX-99.1 exists, a second email for EX-99.1.
  - **Other form types (e.g. DEFM14A, 10-K, 10-Q, OTHER):** One email per filing (single document).

### Call chain (this file only)

```
process_items()
  → _route_summarize_and_save(item_data, html_data)
      → route_and_summarize(url)   # sec_summarizers.filing_router
      → send_summary_email_via_webhook(...)   # services.send_summary_email_via_webhook
```

### Email generation and send (outside this file)

| Step | Module | Function |
|------|--------|----------|
| Build subject + HTML | `sec_rss_parser.email_templates` | `generate_8k_99_1_summary_email_html()` |
| Send to N8N | `sec_rss_parser.services` | `send_summary_email_via_webhook()` → `send_webhook_notification()` (from `utils_8k`) |

### Subject

- **Pattern:** `New {summary_kind} Summary Document – {form_type} – {company_name}`
- **Examples:**
  - 8-K: `New 8-K Summary Document – 8-K – Acme Corp.`
  - EX-99.1: `New EX-99.1 Summary Document – 8-K (EX-99.1) – Acme Corp.`
  - Other: `New DEFM14A Summary Document – DEFM14A – Acme Corp.` (if that form type is summarized)

### Payload (webhook)

- **Endpoint:** `N8N_WEBHOOK_URL_8K_SUMMARY` (env or default `https://n8n.arbintel.cloud/webhook/b3007d21-6845-47b5-aece-7b26583758bc`).
- **Keys:** `subject`, `html`, `company_name`, `form_type`, `summary_doc_url`, `accession_number`, `cik_number`, `sec_url`.

### Parameters passed to `send_summary_email_via_webhook`

- From `_route_summarize_and_save`:  
  `summary_doc_url`, `company_name`, `form_type` (or `"8-K (EX-99.1)"` for EX-99.1), `cik_number`, `sec_url`, `accession_number`, `summary_kind` (e.g. `"8-K"`, `"EX-99.1"`), `l1_headline`, `l2_brief`.

---

## 2. 8-K and EX-99.1 summary emails from `_process_8k_item` — **not used by main flow**

### When

- **Trigger:** Only if something calls `_process_8k_item(item_data, html_data)`.
- **Current flow:** `process_items()` does **not** call `_process_8k_item()`; it only calls `_route_summarize_and_save()` and then `_process_proxy_item()` or `_process_ten_k_ten_q_item()` by form type. So these emails are **not** sent in the main feed-by-CIK flow.

### Call chain (if ever used)

```
_process_8k_item(item_data, html_data)
  → summarize_8k_filing(url_8k, ...) or summarize_8k_filing(url_ex99, ...)   # Eight_k_summary
  → send_summary_email_via_webhook(...)   # same as above
```

### Email generation and send

- Same as in **§1**: `generate_8k_99_1_summary_email_html()` (email_templates) and `send_summary_email_via_webhook()` (services).

### Subject and payload

- Same pattern and webhook as **§1**; only the call site and arguments differ (e.g. `form_type="8-K"` or `"8-K (EX-99.1)"`, `summary_kind="8-K"` or `"EX-99.1"`).

---

## 3. Proxy summary email — **triggered from this file, sent elsewhere**

### When

- **Trigger:** For each item with `form_type` in `PROXY_FORM_TYPES` (e.g. DEFM14A, PREM14A, S-4, F-4), `process_items()` calls `_process_proxy_item()`, which calls `process_sec_document_for_filing_summary()` in `proxy_processor_helper`.
- **When email is sent:** Asynchronously, when the proxy pipeline has finished generating the summary (not in the same call as `process_items`).

### Call chain (this file → other module)

```
process_items()
  → _process_proxy_item(item_data, html_data, filing)
      → process_sec_document_for_filing_summary(...)   # proxy_processor_helper
          → (async pipeline; when summary is ready)
          → send_summary_email_notification_v2(filing_summary)   # proxy_processor_helper
```

### Email generation and send (outside this file)

| Step | Module | Function |
|------|--------|----------|
| Build subject + HTML | `sec_rss_parser.proxy_processor_helper` | `generate_summary_email_html()` |
| Send to N8N | `sec_rss_parser.proxy_processor_helper` | `send_summary_email_notification_v2()` |

- **Subject (from proxy_processor_helper):** `New Proxy Summary Document – {form_type} – {company_name}`.
- **Payload:** `subject`, `html`, `company_name`, `form_type`, `summary_doc_url`, `sec_filing_summary_id`.
- **Webhook:** Same N8N URL as above (hardcoded in proxy_processor_helper).

---

## 4. 10-K / 10-Q bulk email — **triggered from this file, sent elsewhere**

### When

- **Trigger:** For each item with `form_type` in `TEN_K_TEN_Q_FORM_TYPES` (`10-K`, `10-Q`), `process_items()` calls `_process_ten_k_ten_q_item()`, which calls `fetch_and_save_additional_10k_10q_filings()` in `utils_10k_10q`.
- **When email is sent:** After fetching and saving 10-K/10-Q filings; one email per company/run with **all** filings in the period (not per accession).

### Call chain (this file → other module)

```
process_items()
  → _process_ten_k_ten_q_item(item_data, html_data, filing)
      → fetch_and_save_additional_10k_10q_filings(...)   # utils_10k_10q
          → generate_sec_filings_email_html(company_name, filings, form_type)   # email_templates
          → send_webhook_notification(N8N_WEBHOOK_URL_10K_10Q, sec_payload, "email")   # utils_8k
```

### Email generation and send (outside this file)

| Step | Module | Function |
|------|--------|----------|
| Build subject + HTML | `sec_rss_parser.email_templates` | `generate_sec_filings_email_html()` |
| Send to N8N | `sec_rss_parser.utils_10k_10q` | `send_webhook_notification()` (from `utils_8k`) |

- **Payload:** `subject`, `html`, `company_name`, `email_type: 'sec_filings_last_year'`.
- **Webhook:** `N8N_WEBHOOK_URL_10K_10Q` in utils_10k_10q (same URL as 8K summary in practice).

---

## 5. Common functions used for email (from this file’s perspective)

| Purpose | Function | Module | Used by (in fetch_sec_feed_by_deal_cik) |
|--------|----------|--------|----------------------------------------|
| 8-K/EX-99.1 summary subject + HTML | `generate_8k_99_1_summary_email_html()` | `email_templates` | Only via `send_summary_email_via_webhook()` (services) — used by §1 and §2 |
| Send summary to N8N | `send_summary_email_via_webhook()` | `services` | `_route_summarize_and_save` (§1), `_process_8k_item` (§2) |
| Low-level webhook POST | `send_webhook_notification()` | `utils_8k` | Used by `send_summary_email_via_webhook()` in services (not called directly in this file) |

---

## 6. Quick reference: when does which email run?

| Form type | Email sent from this file’s flow? | Where subject/HTML is built | Where send happens |
|----------|-----------------------------------|----------------------------|---------------------|
| **8-K** (main doc) | Yes | `email_templates.generate_8k_99_1_summary_email_html` | `services.send_summary_email_via_webhook` (§1) |
| **8-K** (EX-99.1) | Yes | Same | Same (§1) |
| **Other** (e.g. SC 13G, 4) | Yes | Same | Same (§1) |
| **Proxy** (DEFM14A, etc.) | Triggered here; send elsewhere | `proxy_processor_helper.generate_summary_email_html` | `proxy_processor_helper.send_summary_email_notification_v2` (§3) |
| **10-K / 10-Q** | Triggered here; send elsewhere | `email_templates.generate_sec_filings_email_html` | `utils_10k_10q` via `send_webhook_notification` (§4) |

---

## 7. Webhook URL (this file)

- **Defined in this file:** `N8N_WEBHOOK_URL_8K_SUMMARY` (lines 86–89), from env or default.
- **Used in this file:** Only indirectly; this file calls `send_summary_email_via_webhook()` from `services`, which uses its own `N8N_WEBHOOK_URL_8K_SUMMARY` (same default URL in practice).
