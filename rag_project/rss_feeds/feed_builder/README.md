# Feed Builder (R&D)

Local RSS.app-style feed builder and URL monitor. This module is separate from the existing webhook/RSS.app production flow.

## What it does

1. Fetch a source URL and detect `rss` vs `html`
2. For HTML pages: suggest container selectors, map title/link/date/author/image
3. Preview extracted article rows live
4. Save feed config to local JSON
5. Scan active feeds and store only **new** `detail_url` rows with `is_processed=false`

## MongoDB collections

- `news_source_configs` — feed source configs (upsert by `source_url`)
- `news_article_links` — discovered article URLs (`is_processed=false` until pipeline runs)

Legacy JSON files under `data/` are no longer used by the Streamlit app.

## Run Streamlit UI

From `rag_project/`:

```bash
# Main project deps first (if not already installed)
pip install -r ../requirements.txt

# Streamlit + feed builder extras
pip install -r requirements_streamlit.txt

streamlit run rss_feeds/feed_builder/app.py
```

## Scan configs → news_article_links

Standalone script (cron-friendly):

```bash
cd rag_project

# Scan all active feeds in news_source_configs
python rss_feeds/feed_builder/scan_article_links.py

# One newswire
python rss_feeds/feed_builder/scan_article_links.py --source-id globenewswire_press

# Preview without writing MongoDB
python rss_feeds/feed_builder/scan_article_links.py --dry-run --verbose

# JSON summary for logging
python rss_feeds/feed_builder/scan_article_links.py --json
```

Django management command (same logic):

```bash
python manage.py scan_news_sources
python manage.py scan_news_sources --source-id france_autorite
python manage.py scan_news_sources --dry-run
```

Each run:
1. Loads configs from `news_source_configs`
2. Fetches listing URL (RSS or HTML selectors)
3. Upserts only **new** `detail_url` rows into `news_article_links` with `is_processed=false`

## Other CLI

```bash
# Test a URL without saving
python manage.py test_news_source --url "https://www.ftc.gov/feeds/press-release.xml" --limit 5

# Test a saved feed
python manage.py test_news_source --source-id ftc_press_release --limit 5
```

## Suggested workflow

1. Open Streamlit → **Create Feed**
2. Paste URL → **Fetch & Detect**
3. In the visual builder panel:
   - **Auto mode**: click any article card (orange highlight = containers)
   - Or pick a suggested container → **Use suggested container + auto-infer fields**
   - Click **Apply selectors to Feed Builder**
4. Confirm rows in **Matching entries** preview
5. **Save Feed** → writes to `news_source_configs` (updates if URL exists)
6. **All Feeds** tab → view saved configs

Cron / management command `scan_news_sources` reads Mongo configs and writes new URLs to `news_article_links`.

## Notes

- HTML pages that require JavaScript may need `fetch_mode=playwright` (future phase).
- Webhook flow in `rss_feeds/services.py` is intentionally untouched.
