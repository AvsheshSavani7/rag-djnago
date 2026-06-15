# 10-K/10-Q Processing Implementation

## Overview

This document describes the 10-K/10-Q processing flow used in `fetch_sec_feed_by_deal_cik.py`. The implementation saves records to `SECFilingSummary.ten_k_ten_q` (not the deprecated `TenKTenQSummary`). The **current filing is not saved separately**—it is included when we fetch all 10-K/10-Q filings from the SEC API for the company and save each.

## Architecture

### Files Created/Modified

1. **`utils_10k_10q.py`** – Helper utilities: announce-date resolution, SEC fetch, save to `SECFilingSummary`, email
2. **`fetch_sec_feed_by_deal_cik.py`** – Entry: `_process_ten_k_ten_q_item()` calls into `utils_10k_10q`
3. **`test_10k_10q_processor.py`** – Test script for 10-K/10-Q flow

### Nested / Related Files (for confirmation)

| File | Role |
|------|------|
| `fetch_sec_feed_by_deal_cik.py` | Main loop: filter by `TEN_K_TEN_Q_FORM_TYPES` ("10-K", "10-Q"), call `_process_ten_k_ten_q_item()`; uses `_filter_unique_items`, `fetch_and_parse_html_by_form_type`, `_ensure_sec_filing` |
| `utils_10k_10q.py` | `get_announce_date_for_10k_10q()`, `fetch_and_save_additional_10k_10q_filings()`; imports `sec_Last_Year.print_filings`, `utils_8k`, `email_templates`, `models.SECFilingSummary` |
| `sec_rss_parser/sec_Last_Year.py` | `print_filings(cik, start_date=None, form_types=None)` – SEC API fetch |
| `sec_rss_parser/email_templates.py` | `generate_sec_filings_email_html(company_name, filings, form_type)` |
| `sec_rss_parser/utils_8k.py` | `normalize_cik`, `parse_filing_date`, `send_webhook_notification`, `log_and_print` |
| `sec_rss_parser/services.py` | `_extract_announce_date_with_llm()` (used by `utils_10k_10q`) |
| `sec_rss_parser/models.py` | `SECFilingSummary` |
| `document_processor/models.py` | `ProcessingJob` (deal/announce date lookup) |

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ fetch_sec_feed_by_deal_cik.py                                    │
│ (main loop: form_type in TEN_K_TEN_Q_FORM_TYPES → 10-K, 10-Q)    │
│ _process_ten_k_ten_q_item(item_data, html_data, filing)          │
└────────────────────┬────────────────────────────────────────────┘
                      │
                      └─► fetch_and_save_additional_10k_10q_filings()  [utils_10k_10q.py]
                             │
                             ├─► a. Get announce date (if not provided)
                             │      - From matched_deal
                             │      - From DB lookup by CIK (ProcessingJob)
                             │      - From LLM + web search (services._extract_announce_date_with_llm)
                             │
                             ├─► b. Fetch ALL 10-K/10-Q from SEC API
                             │      sec_Last_Year.print_filings(cik, start_date, form_types=["10-K","10-Q"])
                             │      (start_date = announce_date or 1 year before today)
                             │
                             ├─► c. For each filing: if not already in SECFilingSummary
                             │      → create SECFilingSummary with ten_k_ten_q payload
                             │      (includes the filing that triggered this run)
                             │
                             └─► d. Send one email with all filings
                                    (generate_sec_filings_email_html → send_webhook_notification)
```

## Key Functions

### 1. `get_announce_date_for_10k_10q()`
**Location:** `utils_10k_10q.py`

**Purpose:** Extract announce date for 10-K/10-Q processing

**Logic:**
1. Try `matched_deal.announce_date`
2. Try DB lookup by CIK (as target or acquirer)
3. Try LLM + web search using company details
4. Return `None` if not found (will use 1 year before today)

**Returns:** `(announce_date, deal)` tuple

### 2. `fetch_and_save_additional_10k_10q_filings()`
**Location:** `utils_10k_10q.py`

**Purpose:** Fetch **all** 10-K/10-Q filings for the company from the SEC API (from announce date or 1 year ago), save each to `SECFilingSummary.ten_k_ten_q`, and send one email. The filing that triggered the run is included in this set, not saved separately.

**Parameters:**
- `cik_number`: Company CIK
- `company_name`: Company name
- `form_type`: "10-K" or "10-Q"
- `announce_date`: Optional; if not provided, resolved by `get_announce_date_for_10k_10q()`
- `deal_id`: Optional deal ID
- `matched_deal`: Optional matched deal object
- `item_data`: Optional item data dict (used for announce-date resolution and company name)

**Process:**
1. Resolve announce date via `get_announce_date_for_10k_10q()` if not provided.
2. Fetch all 10-K/10-Q filings from SEC API via `sec_Last_Year.print_filings(cik, start_date, form_types=["10-K","10-Q"])`.
3. For each filing: if no `SECFilingSummary` exists for that `accession_number` + `form_type`, create one with `ten_k_ten_q` payload.
4. Send one email with all filings using `generate_sec_filings_email_html()` and `send_webhook_notification()`.

**Returns:** Dict with `success`, `filings_count`, `saved_count`, and optional `error` or `message`.

### 3. `_process_ten_k_ten_q_item()`
**Location:** `fetch_sec_feed_by_deal_cik.py`

**Purpose:** Process a 10-K/10-Q item from the RSS feed (no separate save of the current filing).

**Process:**
1. Call `fetch_and_save_additional_10k_10q_filings()` with `announce_date=None` (helper resolves it). That function fetches **all** 10-K/10-Q filings from the SEC API for the CIK (from announce date or 1 year ago) and saves each to `SECFilingSummary.ten_k_ten_q`, so the current filing is included in that batch.
2. Log success/failure and counts (filings found, new records saved).

## Data Structure

### SECFilingSummary full schema (for future reference)

**Source:** `sec_rss_parser/models.py` — collection: `sec_filing_summary`, `db_alias`: `new_db`.

One document per filing; only the nested object for that `form_type` is set (`proxy`, `ten_k_ten_q`, `eight_k`, or `other_filings`); the others are `null`.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `_id` | string | yes (auto) | UUID primary key |
| `accession_number` | string | no | SEC accession number (max 50) |
| `cik_number` | string | no | Company CIK (max 20) |
| `sec_document_url` | string | yes | URL to SEC document (max 2000) |
| `filing_date` | datetime | no | Normalized date (8-K: MM/DD/YY; 10-K/10-Q, proxy: YYYY-MM-DD) |
| `deal_id` | string | no | Deal ID (max 50) |
| `form_type` | string | yes | e.g. `"10-K"`, `"10-Q"`, `"8-K"`, `"DEFM14A"`, … (max 50) |
| `created_at` | datetime | auto | Set on create |
| `updated_at` | datetime | auto | Updated on save |
| `proxy` | dict | no | Set when `form_type` is proxy (DEFM14A, DEF 14A, etc.); see below |
| `ten_k_ten_q` | dict | no | Set when `form_type` is 10-K or 10-Q; see below |
| `eight_k` | dict | no | Stored as `8_k` in DB. Set when `form_type` is 8-K |
| `other_filings` | dict | no | Future use; null for now |

**Indexes:** `accession_number`, `cik_number`, `deal_id`, `filing_date`, `form_type`, `sec_document_url`.

**Nested `proxy` schema (when form_type is proxy):**

- `proxy_parsing_status`: `"pending"` \| `"processing"` \| `"completed"` \| `"failed"`
- `empty_percentage`: float (0–100)
- `processing_state`: `{ "pdf_created", "toc_found", "toc_extracted", "sections_extracted", "empty_percentage", "iteration_count" }`
- `s3_urls`: `{ "pdf_url", "toc_pdf_url", "toc_json_url", "sections_json_url" }`
- `pinecone_processing_status`, `pinecone_processed_at`, `pinecone_error_message`
- `summary_generation_status`, `summary_docx_url`, `summary_generated_at`
- `agent_response`, `error_message`, `completed_at`, `total_sections`, `empty_sections`, `iteration_count`

**Nested `eight_k` schema:** (omitted here; see 8-K processing docs.)

### SECFilingSummary.ten_k_ten_q Schema

```python
{
    "processed": False,              # Will be True when processing is done
    "processed_at": None,            # Timestamp when processed
    "s3_json_url": None,            # S3 URL for JSON summary
    "s3_docx_url": None,            # S3 URL for DOCX summary
    "s3_comparison_json_url": None, # S3 URL for comparison JSON
    "s3_redline_docx_url": None,    # S3 URL for redline DOCX
    "s3_client_report_docx_url": None,  # S3 URL for client report
    "s3_exec_summary_docx_url": None,   # S3 URL for exec summary
    "label": None,                   # Optional label
}
```

**Note:** All fields start as `None` or `False`. They will be populated in future when 10-K/10-Q processing is implemented.

## Email Notification

### Webhook URL
#### N8N_WEBHOOK_INTERNAL
```python
N8N_WEBHOOK_URL_10K_10Q ="https://n8n.arbintel.cloud/webhook/80830c6d-ff5b-45e3-9ef3-a061db1fbf0c"
```

### Email Payload
```python
{
    'subject': 'SEC Filings for [Company Name]',
    'html': '<html>...</html>',
    'company_name': 'Company Name',
    'email_type': 'sec_filings_last_year',
}
```

### Email Template
Uses `generate_sec_filings_email_html()` from `email_templates.py`

## Comparison with services.py

| Feature | services.py | fetch_sec_feed_by_deal_cik.py | process_feed_8k.py |
|---------|-------------|--------------------------------|--------------------|
| Collection | `TenKTenQSummary` (deprecated) | `SECFilingSummary.ten_k_ten_q` | N/A (email only) |
| Announce Date | From matched_deal or DB or LLM | Same | N/A |
| Fetch Filings | `sec_Last_Year.print_filings()` | Same | Same |
| Email | ✅ Sent | ✅ Sent | ✅ Sent |
| Historical Filings Email | ✅ For 8-K (EX-2.1) | ✅ For 10-K/10-Q | ✅ For 8-K (EX-2.1) |
| Uniqueness Check | By accession_number | By accession_number + form_type | N/A |
| Additional Processing | None | None (placeholder for future) | 8-K document helper |

## Key Differences from services.py

1. **Collection**: Saves to `SECFilingSummary` instead of `TenKTenQSummary`
2. **Uniqueness**: Checks both `accession_number` AND `form_type` (more precise)
3. **Modular**: Separated into `utils_10k_10q.py` for reusability
4. **Future-Ready**: Placeholder fields for future processing (s3_json_url, s3_docx_url, etc.)

## Testing

### Run Test Script
```bash
cd rag_project
python test_10k_10q_processor.py
```

### Test Coverage
1. ✅ Announce date extraction (DB + LLM)
2. ✅ Fetching filings from SEC API
3. ✅ Saving to SECFilingSummary
4. ✅ Email sending
5. ✅ Error handling

### Manual Testing
```bash
# Test with demo RSS file
cd rag_project
python sec_rss_parser/fetch_sec_feed_by_deal_cik.py
```

## Historical 8-K Filings Email

### Added to process_feed_8k.py

When processing an 8-K filing with EX-2.1, the system now sends **TWO emails**:

1. **Main EX-2.1 Email** - Details about the current EX-2.1 filing
2. **Historical 8-K Email** - All 8-K filings from the last 1 year (for context)

This matches the behavior in `services.py` and provides users with:
- Immediate detail about the current merger agreement
- Historical context of recent material events (8-K filings)

### Implementation Details

**Function**: `_send_historical_8k_email()` in `process_feed_8k.py`

**Process**:
1. Get CIK and filing date from current item
2. Calculate start_date = filing_date - 365 days
3. Fetch all 8-K filings from SEC API using `sec_Last_Year.print_filings()`
4. Generate email with `generate_sec_filings_email_html()`
5. Send to N8N webhook

**Email Details**:
- Form type label: "8-K(EX-2.1)"
- Email type: `sec_filings_last_year`
- Webhook: `N8N_WEBHOOK_URL_8K_SUMMARY`

## Important Notes

### 1. Uniqueness Handling
- **Before reaching `_process_ten_k_ten_q_item()`**: Accession numbers are filtered (no duplicates from `AccessionLookedUp`).
- **Inside `fetch_and_save_additional_10k_10q_filings()`**: For each filing returned by the SEC API (including the one that triggered the run), we check `SECFilingSummary` by `accession_number` + `form_type`; only create a new record if none exists.

### 2. Future Processing
The `ten_k_ten_q` payload has placeholder fields for future processing:
- `s3_json_url`: Will store JSON summary URL
- `s3_docx_url`: Will store DOCX summary URL
- `s3_comparison_json_url`: Will store comparison JSON URL
- `s3_redline_docx_url`: Will store redline DOCX URL
- `s3_client_report_docx_url`: Will store client report URL
- `s3_exec_summary_docx_url`: Will store exec summary URL

### 3. Email Sending
- Emails are sent for ALL filings (current + additional from SEC API)
- Email includes filing date, form type, and direct links to SEC documents
- Email type is `sec_filings_last_year`

### 4. Error Handling
- All errors are logged but don't stop processing
- If announce date extraction fails, defaults to 1 year before today
- If email sending fails, processing continues
- If saving a filing fails, processing continues with next filing

## Dependencies

### Python Packages
- `requests`: For SEC API calls
- `mongoengine`: For database operations
- `django`: For Django ORM

### Internal Modules
- `sec_Last_Year.print_filings()`: Fetch filings from SEC API
- `utils_8k`: Common utilities (normalize_cik, parse_filing_date, etc.)
- `email_templates`: Email generation
- `models`: Database models (SECFilingSummary, ProcessingJob)

## Migration from TenKTenQSummary

If you have existing data in `TenKTenQSummary`, you can migrate it using:

```bash
cd rag_project
python manage.py create_sec_filing_summary_collection --dry-run
python manage.py create_sec_filing_summary_collection
```

This will migrate all `TenKTenQSummary` records to `SECFilingSummary.ten_k_ten_q`.

## Future Enhancements

1. **Processing Pipeline**: Implement actual 10-K/10-Q document processing
2. **Summary Generation**: Generate JSON and DOCX summaries
3. **Comparison**: Compare with previous filings
4. **Redline**: Generate redline documents
5. **Client Reports**: Generate client-specific reports
6. **Executive Summaries**: Generate executive summaries

## Troubleshooting

### Issue: No filings found
**Solution:** Check if announce_date is too recent or CIK is correct

### Issue: Duplicate filings
**Solution:** Check `AccessionLookedUp` collection for filtering

### Issue: Email not sent
**Solution:** Check N8N webhook URL and network connectivity

### Issue: LLM extraction fails
**Solution:** Check API keys and network connectivity

## Support

For questions or issues, check:
1. Logs in terminal output
2. MongoDB `SECFilingSummary` collection
3. N8N webhook logs
4. SEC API status (https://www.sec.gov/edgar/sec-api-documentation)
