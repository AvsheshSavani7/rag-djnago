# SEC Filing Workflow Charts

---

## `process_feed_8k.py`

> Entry point: `run_8k_processor()` → `EightKFeedProcessor.run()`
> Processes **only 8-K** filings from the global SEC RSS feed.
> **Sole handler for all 8-K filings** — `fetch_sec_feed_by_deal_cik.py` skips 8-K entirely.

```
run_8k_processor(rss_content?, rss_file?)
│
├─ Fetch RSS (live SEC feed, or rss_content / rss_file for testing)
│   └─ IF not rss_content → parser.set_feed_url('8-K') → parser.fetch_rss_feed()
│
├─ Parse RSS → items[]
│   └─ parser.parse_rss_content(rss_content)
│
├─ _filter_unique_items(items)
│   │
│   │  FOR each item:
│   │   ├─ field: accession_number (or extract from guid)
│   │   ├─ CHECK AccessionLookedUp.objects(accession_number=acc)
│   │   │   └─ IF exists → SKIP
│   │   ├─ CHECK SECFiling.objects(accession_number=acc)
│   │   │   └─ IF exists → SKIP (+ cache to AccessionLookedUp)
│   │   └─ CHECK duplicate in current batch (seen_accessions set)
│   │       └─ IF duplicate → SKIP
│   │
│   └─ RETURN unique_items[]
│
└─ FOR each unique item → _process_single_item(item_data)
```

---

### `_process_single_item(item_data)`

```
_process_single_item(item_data)
│
├─ field: accession_number (from item_data or extracted from guid)
│
├─ acquire_accession_lock(accession_number, source="process_feed_8k")
│   └─ IF lock_owner is None → SKIP (already in-progress or finalized)
│
├─ field: html_url = item_data.get('link')
│   └─ CONDITION: html_url is required → if missing, SKIP
│
├─ Fetch & parse HTML: parser.fetch_and_parse_html(html_url, form_type_from_feed='8-K')
│   └─ CONDITION: html_data is required → if None, ERROR + SKIP
│
├─ Merge: item_data.update(html_data)
├─ field: cik_number = _extract_cik_from_url(html_url) or item_data.get('cik_number')
├─ field: filing_array = item_data.get('filing_array', [])
│
├─ _check_cik_matches_deal(cik_number)
│   │  Queries ProcessingJob where:
│   │    (deal_status in ["Open","Unknown"]) OR (deal_status is None) OR (deal_status not exists)
│   │  Checks: cik (target) first, then acquirer_cik
│   │
│   ├─ SET item_data['deal_id'] = matched deal id or None
│   └─ SET item_data['cik_matches_deal'] = True/False
│
├─ Resolve deal party (if deal_id and cik_number):
│   ├─ IF cik == acquirer_cik → matched_cik_label = "(acquirer)", email_company_name = acquire_name
│   └─ IF cik == target cik   → matched_cik_label = "(target)",   email_company_name = target_name
│
├─ CONDITION: filing_array is required → if empty, SKIP
│
├─ _save_filing(item_data) → saves SECFiling to DB
│
├─ *** ROUTING DECISION ***
│   │
│   │  ex21_entries = [f for f in filing_array if f.document_type == 'EX-2.1']
│   │
│   ├─── IF ex21_entries is NOT empty (EX-2.1 PRESENT):
│   │    │
│   │    │  other_filings = [f for f in filing_array if f.document_type != 'EX-2.1']
│   │    │
│   │    └─ _process_ex21_filing(item_data, ex21_entries[0], other_filings)
│   │       (see EX-2.1 flow below)
│   │
│   └─── ELSE (NO EX-2.1):
│        │
│        └─ FOR each filing in filing_array:
│            ├─ IF document_type == '8-K'    → _process_8k_document(item_data, filing)
│            └─ IF document_type == 'EX-99.1' → _process_ex99_filing(item_data, filing)
│
├─ mark_accession_processed(accession_number)   ← only on success
│
└─ finally: release_accession_lock(accession_number, lock_owner)
```

---

### `_process_ex21_filing(item_data, filing_entry, other_filings)` — EX-2.1 Flow

```
_process_ex21_filing(item_data, filing_entry, other_filings=None)
│
├─ field: url_ex21 = filing_entry.get('url')
│   └─ CONDITION: url_ex21 is required → if missing, RETURN
│
├─ GPT ANALYSIS (MANDATORY — always runs regardless of cik_matches_deal):
│   │  data = deep_copy(item_data)
│   │  data['xbrl_files'] = [{ type: 'EX-2.1', url: url_ex21 }]
│   │  result = document_analyzer.analyze_filing(data)
│   │  item_data.update(result)
│   │
│   ├─ field: document_kind              ← from GPT result
│   ├─ field: company_details            ← from GPT result (dict)
│   │   ├─ company_details.is_target_us_listed
│   │   └─ company_details.is_target_market_cap_greater_than_100m
│   │
│   └─ Update SECFiling in DB with document_kind + company_details
│
├─ *** THREE MANDATORY CONDITIONS (all must be truthy): ***
│   │
│   │  1. document_kind == "Definitive Merger Agreement"   (exact string)
│   │  2. is_target_us_listed == truthy                    (from company_details)
│   │  3. is_target_market_cap_greater_than_100m == truthy (from company_details)
│   │
│   ├─── IF ALL THREE PASS (QUALIFIED):
│   │    │
│   │    ├─ 0. Fetch deal details (if cik_matches_deal + deal_id):
│   │    │      │  Query ProcessingJob by deal_id for:
│   │    │      │    target_name, acquire_name, cik, acquirer_cik,
│   │    │      │    deal_status, announce_date, target_ticker, acquirer_ticker
│   │    │      └─ Build deal_details dict (includes matched_cik_label)
│   │    │         → None if cik_matches_deal is False or query fails
│   │    │
│   │    ├─ 1. _send_ex21_email(item_data, company_details, filing_entry, deal_details)
│   │    │      │
│   │    │      ├─── IF deal_details is NOT None (CIK matches existing deal):
│   │    │      │    │  Generates email via generate_filing_email_with_deal_html()
│   │    │      │    │  Email includes:
│   │    │      │    │    - All standard filing info (form_type, accession, dates, CIK, etc.)
│   │    │      │    │    - "Existing Deal Match Found" banner (yellow)
│   │    │      │    │    - Deal Target, Deal Acquirer, Deal Status, Announce Date
│   │    │      │    │    - Tickers (target + acquirer), Deal CIKs, Filing CIK Role
│   │    │      │    │    - Company Details (M&A from GPT)
│   │    │      │    │  Subject: "SEC Filing – EX-2.1 – CompanyName – Existing Deal Match"
│   │    │      │    └─  Title: "8-K – CompanyName (Existing Deal: Target / Acquirer)"
│   │    │      │
│   │    │      └─── ELSE (no deal match):
│   │    │           │  Generates email via generate_filing_email_html() (standard)
│   │    │           └─  Subject: "SEC Filing – 8-K – CompanyName"
│   │    │
│   │    │      Webhook: N8N_WEBHOOK_URL_FILING (if us_listed + cap>100m)
│   │    │               else N8N_WEBHOOK_URL_8K_SUMMARY
│   │    │      email_type: 'ex21_merger'
│   │    │
│   │    ├─ 2. _send_historical_8k_email(item_data)
│   │    │      │  CONDITION: cik_number AND filing_date required → else SKIP
│   │    │      │  start_date = filing_date - 365 days
│   │    │      │  Fetches: fetch_sec_filings(cik, start_date, form_types=None)
│   │    │      │  Generates: generate_sec_filings_email_html(company_name, filings, "8-K(EX-2.1)")
│   │    │      └─ Webhook: N8N_WEBHOOK_URL_8K_SUMMARY, email_type: 'sec_filings_last_year'
│   │    │
│   │    ├─ 3. _process_ex21_via_8k_helper(item_data, filing)
│   │    │      │  ex21_url = build_full_sec_url(find_file_by_type(xbrl_files, 'EX-2.1'))
│   │    │      │  CONDITION: ex21_url required → else SKIP
│   │    │      └─ Calls: process_8k_document_helper(
│   │    │             cik_number, company_name, sec_filing_id,
│   │    │             filing_date, form_type='8-K', ex21_url,
│   │    │             item_data, company_details
│   │    │         )
│   │    │         → Hits Node API: deal/process-with-url
│   │    │
│   │    └─ 4. Process other_filings (if any):
│   │           │
│   │           └─ FOR each f in other_filings:
│   │               ├─ IF document_type == '8-K' AND url exists:
│   │               │   └─ _generate_8k_summary_and_send(item_data, url_8k)
│   │               │       ├─ route_and_summarize(url_8k) → S3 upload
│   │               │       ├─ Save to SECFilingSummary (parent-level: L1, L2, L3, s3_docx_url)
│   │               │       └─ _send_8k_summary_email()
│   │               │
│   │               └─ IF document_type == 'EX-99.1' AND url exists:
│   │                   ├─ _generate_ex99_summary(item_data, url_ex99)
│   │                   │   ├─ route_and_summarize(url_ex99) → S3 upload
│   │                   │   ├─ Save to SECFilingSummary.ex99_1 node
│   │                   │   └─ _send_ex99_summary_email()
│   │                   │
│   │                   └─ _extract_press_release_data(item_data, ex99_result, url_ex99)
│   │                       ├─ Builds summary_text from L1 + L2 + L3
│   │                       ├─ CONDITION: L1 or L2 must have content → else SKIP
│   │                       └─ extract_from_press_release(summary_text, deal_id, ...)
│   │                           → Saves to fo_press_release_extraction collection
│   │
│   └─── IF ANY CONDITION FAILS (NOT QUALIFIED):
│        │
│        └─ Logs "Not qualified for 8-K EX-2.1 document processing"
│           → NO email, NO Node API call, NO other_filings processing
│           → Only SECFiling DB record (with document_kind + company_details) was saved above
│
└─ processed_count += 1 (always, regardless of qualification)
```

---

### `_process_8k_document(item_data, filing_entry)` — 8-K Doc (NO EX-2.1 present)

```
_process_8k_document(item_data, filing_entry)
│
├─ field: url_8k = filing_entry.get('url')
│   └─ CONDITION: url_8k is required → if missing, RETURN
│
├─── IF item_data['cik_matches_deal'] == True:
│    │
│    └─ _generate_8k_summary_and_send(item_data, url_8k)
│        ├─ route_and_summarize(url_8k) → result
│        ├─ Save to SECFilingSummary (parent-level fields)
│        └─ _send_8k_summary_email()
│        └─ RETURN (no GPT)
│
└─── ELSE (cik does NOT match deal):
     │
     ├─ GPT ANALYSIS:
     │   result = document_analyzer.analyze_ex99_1_filing(data)
     │   ├─ field: is_merger_related
     │   ├─ field: confidence
     │   ├─ field: is_target_us_listed
     │   └─ field: is_target_market_cap_greater_than_100m
     │
     ├─ *** THREE MANDATORY CONDITIONS (all must be truthy): ***
     │   1. is_merger_related == truthy
     │   2. is_target_us_listed == truthy
     │   3. is_target_market_cap_greater_than_100m == truthy
     │
     ├─── IF ALL THREE PASS:
     │    └─ _send_8k_gpt_email(item_data, doc_files)
     │        ├─ generate_8k_document_email_html()
     │        └─ Webhook: N8N_WEBHOOK_URL_8K_SUMMARY, email_type: '8k_gpt'
     │
     └─── ELSE:
          └─ "8-K main email skipped (criteria not met)"
```

---

### `_process_ex99_filing(item_data, filing_entry)` — EX-99.1 Doc (NO EX-2.1 present)

```
_process_ex99_filing(item_data, filing_entry)
│
├─ field: url_ex99 = filing_entry.get('url')
│   └─ CONDITION: url_ex99 is required → if missing, RETURN
│
├─── IF item_data['cik_matches_deal'] == True:
│    │
│    └─ _generate_ex99_summary(item_data, url_ex99)
│        ├─ route_and_summarize(url_ex99) → result
│        ├─ Save to SECFilingSummary.ex99_1 node
│        └─ _send_ex99_summary_email()
│        └─ RETURN (no GPT)
│
└─── ELSE (cik does NOT match deal):
     │
     ├─ GPT ANALYSIS:
     │   result = document_analyzer.analyze_ex99_1_filing(data)
     │   ├─ field: is_merger_related
     │   ├─ field: confidence
     │   ├─ field: is_target_us_listed
     │   └─ field: is_target_market_cap_greater_than_100m
     │
     ├─ *** THREE MANDATORY CONDITIONS (all must be truthy): ***
     │   1. is_merger_related == truthy
     │   2. is_target_us_listed == truthy
     │   3. is_target_market_cap_greater_than_100m == truthy
     │
     ├─── IF ALL THREE PASS:
     │    └─ _send_ex99_email(item_data, filing_entry)
     │        ├─ generate_ex99_1_merger_email_html()
     │        └─ Webhook: N8N_WEBHOOK_URL_8K_SUMMARY, email_type: 'ex99_1_merger'
     │
     └─── ELSE:
          └─ "EX-99.1 main email skipped (criteria not met)"
```

---
---

## `fetch_sec_feed_by_deal_cik.py`

> Entry point: `run_fetch_sec_feed_by_deal_cik()`
> Fetches SEC filings **per deal CIK** (target + acquirer), processes **non-8-K form types** (PROXY, 10-K/10-Q, other).
> **8-K is SKIPPED entirely** — handled by `process_feed_8k.py` which has the full EX-2.1 flow.
> **Does NOT handle EX-2.1** — intentionally excludes it from xbrl_files extraction.

```
run_fetch_sec_feed_by_deal_cik(output_json_path?, limit_deals?, process_items_flow?, rss_file?, rss_content?)
│
├─── IF rss_file or rss_content (development mode):
│    │  Parse RSS → items[]
│    │  For each item: extract CIK from title, resolve deal_id
│    └─ all_items = items
│
└─── ELSE (production mode):
     │
     ├─ get_open_or_unknown_deals()
     │   └─ ProcessingJob where (deal_status in ["Open","Unknown"]) OR null/missing
     │      Returns: [{ id, cik, acquirer_cik }]
     │
     └─ FOR each deal:
         ├─ get_ciks_for_deal(deal) → [target_cik, acquirer_cik] (deduplicated)
         │
         └─ FOR each cik:
             ├─ fetch_feed_for_cik(cik, session)
             │   URL: sec.gov/cgi-bin/browse-edgar?CIK={cik}&count=100&output=atom
             │   Uses rate_limited_get() + retry (3 retries, no read retry)
             │   Rate limit: sleep(1) every 7 fetches
             │
             ├─ parse_atom_to_items(raw, cik_number=cik, deal_id=deal_id)
             │   └─ Sets cik_number and deal_id on each item
             │
             └─ all_items.extend(items)
│
├─── IF process_items_flow == True AND all_items not empty:
│    └─ process_items(all_items)
│
└─── ELSE: return raw items
```

---

### `process_items(items)`

```
process_items(items)
│
├─ _filter_unique_items(items)
│   └─ Skip if accession in AccessionLookedUp or duplicate in batch
│
└─ FOR each item_data (with rate-limit: sleep 0.5s every 10 items):
    │
    ├─ field: link = item_data.get('link')
    │   └─ CONDITION: link required → if missing, SKIP
    │
    ├─ *** 8-K SKIP (EARLY EXIT): ***
    │   │  feed_form_type = item_data.get('form_type').upper()
    │   │  IF feed_form_type == "8-K" → SKIP entirely
    │   │  Reason: 8-K handled by process_feed_8k.py (has full EX-2.1 flow)
    │   └─ No accession check, no HTML fetch, no processing
    │
    ├─ field: acc = accession_number (or extract from guid)
    │   └─ CHECK AccessionLookedUp again → if exists, SKIP
    │
    ├─ acquire_accession_lock(acc, source="fetch_by_cik")
    │   └─ IF lock_owner is None → SKIP
    │
    ├─ fetch_and_parse_html_by_form_type(link, form_type_from_feed)
    │   │  *** FOR non-8-K: only files matching form_type ***
    │   └─ CONDITION: html_data required → if None, ERROR + SKIP
    │
    ├─ Merge: item_data.update(html_data)
    ├─ field: cik_number = _extract_cik_from_url(link) or existing
    ├─ field: deal_id = existing or _deal_id_for_cik(cik_number)
    ├─ _ensure_sec_filing(item_data) → saves SECFiling if not exists
    ├─ field: form_type = item_data.form_type (uppercased)
    │
    ├─ STEP 1: _route_summarize_and_save(item_data, html_data)
    │   └─ (runs for all non-8-K form types — see chart below)
    │
    ├─ STEP 2: FORM-TYPE SPECIFIC ROUTING:
    │   │
    │   ├─── IF form_type in PROXY_FORM_TYPES:
    │   │    │  ["DEFM14A","DEFM14C","PREM14A","PREM14C","S-4","F-4","S-4/A","F-4/A"]
    │   │    └─ _handle_proxy_form_by_type(item_data, html_data, filing)
    │   │
    │   └─── ELIF form_type in TEN_K_TEN_Q_FORM_TYPES:
    │        │  ["10-K","10-Q","10-K/A"]
    │        └─ _process_ten_k_ten_q_item(item_data, html_data, filing)
    │
    │   (other form types: NO additional step 2 processing — only _route_summarize_and_save)
    │
    ├─ mark_accession_processed(acc)   ← only on success
    └─ finally: release_accession_lock(acc, lock_owner)
```

---

### `_route_summarize_and_save(item_data, html_data)` — Summary for non-8-K form types

```
_route_summarize_and_save(item_data, html_data)
│
├─ field: form_type (uppercased, fallback "OTHER")
├─ field: accession_number
├─ field: cik_number
├─ field: deal_id (from item_data or _deal_id_for_cik)
├─ field: company_name
├─ field: link
├─ field: xbrl_files (from html_data or item_data)
│
├─ BUILD urls_to_summarize[]:
│   │
│   │  (8-K items never reach here — skipped in process_items)
│   │
│   └─── Non-8-K form types:
│        └─ _pick_single_doc_url_for_form(xbrl_files, form_type, link)
│           Priority: .htm > .html > .xml
│           → single url (is_ex99=False)
│
├─ CONDITION: urls_to_summarize not empty → if empty, RETURN
│
└─ FOR each (url, is_ex99) in urls_to_summarize:
    │
    ├─ route_and_summarize(url) → result
    │   └─ field: s3_docx_url = result.s3_docx_url or result.s3_url
    │       └─ CONDITION: s3_docx_url required → if missing, CONTINUE to next
    │
    ├─ filing_dt = _filing_date_for_summary(...)
    │
    └─── Parent-level save:
         │  Save to SECFilingSummary (top-level fields):
         │    form_type = doc_form_type
         │    items_reported, L1_headline, L2_brief, L3_detailed, s3_docx_url, s3_json_url
         │
         └─ SEND SUMMARY EMAIL:
             │
             ├─ Resolve deal party:
             │   ├─ IF cik == acquirer_cik:
             │   │   ├─ matched_cik_label = "(acquirer)"
             │   │   ├─ email_company_name = acquire_name
             │   │   └─ *** LLM CHECK (acquirer only): ***
             │   │       _llm_form_affects_deal(target_name, acquirer_name, sec_url, form_type)
             │   │       → Uses GPT + web_search tool
             │   │       → Returns True (YES) / False (NO) / None (error)
             │   │       → Stored as form_affects_deal field in email
             │   │
             │   └─ IF cik == target cik:
             │       ├─ matched_cik_label = "(target)"
             │       └─ email_company_name = target_name
             │          (NO LLM check for target filings)
             │
             └─ send_summary_email_via_webhook(
                    summary_doc_url, company_name, form_type,
                    cik_number, sec_url, accession_number,
                    summary_kind, l1_headline, l2_brief, l3_detailed,
                    ticker, filing_date,
                    matched_cik_label, form_affects_deal
                )
```

---

### `fetch_and_parse_html_by_form_type()` — File filtering

```
fetch_and_parse_html_by_form_type(html_url, form_type_from_feed)
│
├─ Fetch HTML from SEC (rate_limited_get, timeout=45s)
├─ Parse with BeautifulSoup
│
├─ _extract_xbrl_files_by_form_type(soup, form_type):
│   │
│   │  (8-K never reaches here — skipped in process_items)
│   │
│   └─── Non-8-K form types:
│        └─ INCLUDE only files where doc_type matches form_type
│
└─ RETURN { form_type, accession_number, filing_date, xbrl_files, ... }
```

---

### `_handle_proxy_form_by_type()` — Proxy routing

```
_handle_proxy_form_by_type(item_data, html_data, filing)
│
├─ field: form_type (uppercased)
│   └─ CONDITION: must be in PROXY_FORM_TYPES → else RETURN
│
├─── IF form_type in ["S-4", "F-4"] (ALWAYS STANDALONE):
│    └─ _process_proxy_item() → process_sec_document_for_filing_summary()
│
└─── ELSE (comparison path):
     │
     ├─ Lookup map (PROXY_FORM_PREVIOUS_LOOKUP):
     │   ├─ "PREM14A" → look for previous ["PREM14A"]
     │   ├─ "PREM14C" → look for previous ["PREM14C"]
     │   ├─ "DEFM14A" → look for previous ["PREM14A","S-4","S-4/A","F-4","F-4/A"]
     │   ├─ "DEFM14C" → look for previous ["PREM14C"]
     │   ├─ "S-4/A"   → look for previous ["S-4","S-4/A"]
     │   └─ "F-4/A"   → look for previous ["F-4","F-4/A"]
     │
     ├─ _get_previous_proxy_summary(deal_id, form_types, exclude=accession)
     │   └─ IF no previous found → _process_proxy_item() (standalone)
     │
     ├─ _get_current_proxy_summary(deal_id, accession, form_type)
     │   └─ IF not found → _process_proxy_item() (standalone fallback)
     │
     └─ run_comparison(latest_doc_record, past_doc_record)
         ├─ IF status == "complete":
         │   └─ _send_proxy_comparison_email(company_name, form_type, deal_id, cik, result)
         └─ ELSE: log warning
```

---

## Key Differences Between the Two Files

| Aspect | `process_feed_8k.py` | `fetch_sec_feed_by_deal_cik.py` |
|---|---|---|
| **Source** | Global SEC 8-K RSS feed | Per-deal CIK Atom feed |
| **Form types** | 8-K only | All **except 8-K** (PROXY, 10-K/10-Q, other) |
| **8-K handling** | Full flow (GPT, emails, Node API) | **SKIPPED** — `form_type == "8-K"` → continue |
| **EX-2.1 handling** | Full flow (GPT analysis, 3 conditions, Node API) | N/A (8-K skipped) |
| **Deal-aware EX-2.1 email** | Yes — when `cik_matches_deal`, email includes deal details | N/A |
| **GPT for 8-K/EX-99.1** | Only when `cik_matches_deal == False` | N/A (8-K skipped) |
| **LLM deal relevance** | Not used | `_llm_form_affects_deal` for acquirer CIK filings |
| **Proxy comparison** | Not applicable | Yes — orchestrator + comparison email |
| **Press release extraction** | Yes (when EX-2.1 qualified + EX-99.1) | No |

---

## Mandatory Conditions Summary

### EX-2.1 Qualification (process_feed_8k.py only)

| # | Field | Condition | Source |
|---|---|---|---|
| 1 | `document_kind` | `== "Definitive Merger Agreement"` | `document_analyzer.analyze_filing()` |
| 2 | `company_details.is_target_us_listed` | `== truthy` | `document_analyzer.analyze_filing()` |
| 3 | `company_details.is_target_market_cap_greater_than_100m` | `== truthy` | `document_analyzer.analyze_filing()` |

**All 3 required for:** email + historical filings + Node API + other_filings processing.

### EX-2.1 Email Template Selection (process_feed_8k.py)

| Condition | Email Template | Subject Format |
|---|---|---|
| `cik_matches_deal == True` + `deal_details` fetched | `generate_filing_email_with_deal_html()` | "SEC Filing – 8-K – Company – Existing Deal Match" |
| `cik_matches_deal == False` or deal fetch fails | `generate_filing_email_html()` | "SEC Filing – 8-K – Company" |

**Deal-aware email additionally includes:** "Existing Deal Match Found" banner, Deal Target, Deal Acquirer, Deal Status, Announce Date, Tickers, Deal CIKs, Filing CIK Role (target/acquirer).

### 8-K / EX-99.1 Email (process_feed_8k.py, when cik_matches_deal is False)

| # | Field | Condition | Source |
|---|---|---|---|
| 1 | `is_merger_related` | `== truthy` | `document_analyzer.analyze_ex99_1_filing()` |
| 2 | `is_target_us_listed` | `== truthy` | `document_analyzer.analyze_ex99_1_filing()` |
| 3 | `is_target_market_cap_greater_than_100m` | `== truthy` | `document_analyzer.analyze_ex99_1_filing()` |

**All 3 required for:** sending the 8-K GPT email or EX-99.1 merger email.

### Preconditions (both files)

| Field | Condition | Consequence if fails |
|---|---|---|
| `form_type` (fetch_sec_feed_by_deal_cik.py) | Must NOT be "8-K" | Item skipped (handled by process_feed_8k.py) |
| `accession_number` | Must exist | Item skipped |
| `acquire_accession_lock()` | Must return owner | Item skipped (in-progress) |
| `link` (html_url) | Must exist | Item skipped |
| `html_data` | Must parse successfully | Item skipped (error) |
| `filing_array` | Must not be empty | Item skipped |
| Document URL (`url_8k`, `url_ex21`, `url_ex99`) | Must exist in filing_entry | That document skipped |
| `s3_docx_url` from `route_and_summarize()` | Must be returned | Summary not saved |
