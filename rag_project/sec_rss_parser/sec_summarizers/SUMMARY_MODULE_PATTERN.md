# SEC Summarizer Module Pattern

Use this as the **single source of truth** when creating or updating any summary module (6-K, 8-K, 99.1, sec_filing_summary, etc.). Apply the same conventions so behavior and structure stay consistent.

---

## 1. Imports (Django / package-safe)

- **Always use relative imports** so modules load correctly when invoked via `filing_router` (e.g. `sec_rss_parser.sec_summarizers.6k_summary`).

```python
from pathlib import Path
from ._naming import filing_uid

# Config: prefer relative; fallback for rare standalone run
try:
    from ._config import get_anthropic_api_key
except ImportError:
    from _config import get_anthropic_api_key

# Fetch text
from .fetch_utils import fetch_text   # inside fetch_filing_text()
# S3
from .s3_utils import upload_json, upload_docx_bytes   # where needed
```

- Do **not** use `from _naming import` or `from fetch_utils import` (absolute); they break when the module is loaded as part of the package.

---

## 2. API key

- Get the key once at module load:  
  `ANTHROPIC_API_KEY = get_anthropic_api_key()`
- Exit only when run as script and key is missing:

```python
if not ANTHROPIC_API_KEY and __name__ == "__main__":
    print("❌ ANTHROPIC_API_KEY not found. Set it in .env or Django settings (ANTHROPIC_API_KEY).")
    sys.exit(1)
```

- In `summarize()`, raise if key is missing (for library use):

```python
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set. Set it in .env or Django settings (ANTHROPIC_API_KEY).")
```

---

## 3. S3 upload (summary_docx / summary_json)

- **Folders:** `summary_docx/` and `summary_json/` (see `s3_utils.py`).
- **Upload functions return** `(s3_path, s3_url)`:
  - `upload_json(result, s3_key_suffix)` → `(path, url)`
  - `upload_docx_bytes(data, s3_key_suffix)` → `(path, url)`

### In `main()` after you have `result`:

1. **Unique id:**  
   `uid = filing_uid(FILING_URL)`

2. **Upload JSON** (key under `summary_json/`):

```python
from .s3_utils import upload_json
s3_json_path, s3_json_url = upload_json(result, f"{module_prefix}_summary_{uid}.json")
# e.g. 6k_summary_{uid}.json, 8k_summary_{uid}.json
```

3. **Build DOCX and upload** (key under `summary_docx/`):
   - Build the Word doc in memory (`io.BytesIO`), then call `upload_docx_bytes(buf.read(), s3_key_suffix)`.
   - `export_docx(s, s3_key_suffix)` should return `(path, url)`.

4. **Attach to result and return:**

```python
result["s3_docx_path"] = s3_docx_path
result["s3_docx_url"]  = s3_docx_url
result["s3_json_path"] = s3_json_path
result["s3_json_url"]  = s3_json_url
return result
```

- **Naming:**  
  - JSON: `{form}_summary_{uid}.json` (e.g. `6k_summary_26024354.json`).  
  - DOCX: `{Form}_Summary_{ticker}_{date}_{uid}.docx` (e.g. `6K_Summary_BABA_03-06-26_26024354.docx`). Sanitize ticker/date for filenames (`re.sub(r'[^\w\-\.]', '_', ticker)`, `date.replace("/", "-")`).

---

## 4. `export_docx(s, s3_key_suffix)`

- Build the doc with `python-docx` (same structure as 6k: title, meta, L1/L2/L3 sections).
- **Do not** save to a local file. Use in-memory:

```python
buf = io.BytesIO()
doc.save(buf)
buf.seek(0)
path, url = upload_docx_bytes(buf.read(), s3_key_suffix)
return path, url
```

- Ensure `import io` at top of file.

---

## 5. `main()` flow (common to all summaries)

1. `source = FILING_URL` (or argument; router sets `module.FILING_URL = url`).
2. `text = fetch_filing_text(source)` (uses `from .fetch_utils import fetch_text`).
3. `result = summarize(text)`.
4. `print_summary(result)` (optional).
5. `uid = filing_uid(FILING_URL)`.
6. Upload JSON → `(s3_json_path, s3_json_url)`.
7. Build DOCX suffix (ticker, date, uid), then `export_docx(result, docx_suffix)` → `(s3_docx_path, s3_docx_url)`.
8. Set `result["s3_docx_path"]`, `result["s3_docx_url"]`, `result["s3_json_path"]`, `result["s3_json_url"]`.
9. `return result`.

---

## 6. Reference implementation

- **6-K:** `6k_summary.py` is the reference. When changing other summary modules (8-K, 99.1, sec_filing_summary, etc.), align with 6k:
  - Relative imports (`._naming`, `._config`, `.fetch_utils`, `.s3_utils`).
  - Single API key from `get_anthropic_api_key()`, with `__name__ == "__main__"` check and `summarize()` guard.
  - S3: upload JSON and DOCX, add all four keys to `result`, return `result`.

---

## 7. Checklist for updating another summary (e.g. 8k, 991)

- [ ] Imports: `from ._naming import filing_uid`, `from ._config import get_anthropic_api_key`, `from .fetch_utils import fetch_text`, `from .s3_utils import upload_json, upload_docx_bytes`.
- [ ] `export_docx(s, s3_key_suffix)` builds doc in memory, calls `upload_docx_bytes`, returns `(path, url)`.
- [ ] `main()`: after `summarize()` and `print_summary()`, upload JSON then DOCX; set `s3_docx_path`, `s3_docx_url`, `s3_json_path`, `s3_json_url` on `result`; return `result`.
- [ ] JSON key: `{form}_summary_{uid}.json`.
- [ ] DOCX key: `{Form}_Summary_{safe_ticker}_{safe_date}_{uid}.docx`.
- [ ] No local file writes for JSON/DOCX; S3 only for these outputs.
