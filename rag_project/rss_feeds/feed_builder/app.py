"""
Streamlit Feed Builder — create/update news source configs in MongoDB.

Run from rag_project/:
  streamlit run rss_feeds/feed_builder/app.py
"""

from __future__ import annotations
from rss_feeds.feed_builder.core.rss_parser import OUTPUT_FIELDS, parse_rss_content
from rss_feeds.feed_builder.core.preview import (
    analyze_html_body,
    analyze_url,
    preview_html_extraction,
)
from rss_feeds.feed_builder.core.mongo_config_store import (
    delete_feed,
    get_feed_by_url,
    list_feeds,
    save_feed,
)
from rss_feeds.feed_builder.core.fetcher import FetchBlockedError
from rss_feeds.feed_builder.core.field_inferrer import infer_field_selectors
from rss_feeds.feed_builder.core.dedupe import slugify_source_id
import streamlit.components.v1 as components
import streamlit as st

import base64
import html as html_module
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Must run before any `rss_feeds` imports (Streamlit executes this file directly).
_RAG_PROJECT = Path(__file__).resolve().parents[2]
if str(_RAG_PROJECT) not in sys.path:
    sys.path.insert(0, str(_RAG_PROJECT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rag_project.settings")
import django  # noqa: E402

django.setup()


BUILDER_PANEL_PATH = Path(__file__).parent / \
    "components" / "rss_builder_panel.html"

st.set_page_config(page_title="Feed Builder", page_icon="📰", layout="wide")

st.markdown(
    "<style>.block-container { padding-top: 1rem; max-width: 1400px; }</style>",
    unsafe_allow_html=True,
)


def _init_session_state() -> None:
    defaults = {
        "analysis": None,
        "source_name": "",
        "source_id": "",
        "source_url": "",
        "selectors": {
            "container": "",
            "title": "",
            "detail_url": "a",
            "published_at": "",
            "description": "",
            "author": "",
            "image": "",
        },
        "url_rules": {"must_contain": [], "exclude": []},
        "element_map": {},
        "preview_items": [],
        "selector_widget_gen": 0,
        "rss_map_widget_gen": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_SELECTOR_FIELDS = (
    "container",
    "detail_url",
    "title",
    "published_at",
    "description",
    "image",
)


def _selector_widget_key(field: str) -> str:
    gen = st.session_state.get("selector_widget_gen", 0)
    return f"fb_sel_{gen}_{field}"


def _invalidate_selector_widgets() -> None:
    """New widget keys on next run — avoids Streamlit 'cannot modify after instantiate' errors."""
    st.session_state["selector_widget_gen"] = st.session_state.get(
        "selector_widget_gen", 0) + 1


def _rss_map_widget_key(field: str) -> str:
    gen = st.session_state.get("rss_map_widget_gen", 0)
    return f"rss_map_{gen}_{field}"


def _invalidate_rss_map_widgets() -> None:
    st.session_state["rss_map_widget_gen"] = st.session_state.get(
        "rss_map_widget_gen", 0) + 1


def _pull_selectors_from_widgets() -> None:
    sel = st.session_state.setdefault("selectors", {})
    for field in _SELECTOR_FIELDS:
        wkey = _selector_widget_key(field)
        if wkey in st.session_state:
            sel[field] = st.session_state[wkey]


def _apply_selectors_to_session(selectors: dict, *, invalidate_widgets: bool = False) -> None:
    st.session_state["selectors"] = dict(selectors or {})
    if invalidate_widgets:
        _invalidate_selector_widgets()


def _load_saved_config_into_session(existing: dict) -> None:
    """Restore source fields + selectors/element_map from MongoDB."""
    if not existing:
        return
    st.session_state["source_name"] = existing.get("source_name") or ""
    st.session_state["source_id"] = existing.get("source_id") or ""
    if existing.get("source_type") == "rss":
        st.session_state["element_map"] = dict(
            existing.get("element_map") or {})
        _invalidate_rss_map_widgets()
    elif existing.get("selectors"):
        _apply_selectors_to_session(existing.get(
            "selectors") or {}, invalidate_widgets=True)
    if existing.get("url_rules"):
        st.session_state["url_rules"] = dict(existing.get("url_rules") or {})


def _maybe_load_config_for_url(source_url: str) -> Optional[dict]:
    """Load saved Mongo config once when the source URL field changes."""
    if not source_url:
        return None
    if source_url == st.session_state.get("_config_loaded_for_url"):
        return get_feed_by_url(source_url)

    existing = get_feed_by_url(source_url)
    if existing:
        _load_saved_config_into_session(existing)
    st.session_state["_config_loaded_for_url"] = source_url
    return existing


def _resolve_html_selectors_after_fetch(
    analysis: dict,
    existing: Optional[dict],
    *,
    force_redetect: bool,
) -> None:
    """Keep manual/DB selectors on re-fetch unless user asked to re-detect."""
    saved = (existing or {}).get("selectors") or {}
    current = st.session_state.get("selectors") or {}

    if not force_redetect:
        if saved.get("container"):
            _apply_selectors_to_session(saved, invalidate_widgets=True)
            _refresh_preview(analysis)
            return
        if current.get("container"):
            _refresh_preview(analysis)
            return

    if analysis.get("inferred_selectors"):
        _apply_selectors_to_session(
            analysis["inferred_selectors"], invalidate_widgets=True)
    elif analysis.get("container_suggestions"):
        _apply_inferred_selectors(
            analysis, analysis["container_suggestions"][0]["selector"])
        return
    _refresh_preview(analysis)


def _sync_selectors_from_query() -> bool:
    qp = st.query_params
    if qp.get("fb_sync") != "1" or not qp.get("selectors"):
        return False
    try:
        synced = json.loads(qp.get("selectors"))
        _apply_selectors_to_session(synced, invalidate_widgets=True)
    except json.JSONDecodeError:
        return False
    st.query_params.clear()
    return True


RSS_FIELD_LABELS = {
    "title": "Title",
    "detail_url": "Link",
    "published_at": "Date",
    "description": "Description",
    "author": "Author",
    "image": "Image",
    "guid": "GUID",
}


def _refresh_rss_preview(analysis: dict) -> None:
    rss_body = analysis.get("rss_content") or ""
    st.session_state["preview_items"] = parse_rss_content(
        rss_body,
        source_url=analysis.get(
            "resolved_url") or analysis.get("input_url") or "",
        element_map=st.session_state.get("element_map") or {},
        limit=30,
    )


def _refresh_preview(analysis: dict) -> None:
    st.session_state["preview_items"] = preview_html_extraction(
        html=analysis["html"],
        base_url=analysis["input_url"],
        selectors=st.session_state["selectors"],
        url_rules=st.session_state.get("url_rules") or {},
        limit=30,
    )


def _apply_inferred_selectors(analysis: dict, container_selector: str) -> None:
    st.session_state["selectors"] = infer_field_selectors(
        analysis["html"], container_selector)
    _invalidate_selector_widgets()
    _refresh_preview(analysis)


def _render_builder_panel(analysis: dict) -> None:
    template = BUILDER_PANEL_PATH.read_text(encoding="utf-8")
    config = {
        "htmlB64": base64.b64encode((analysis.get("html") or "").encode("utf-8")).decode("ascii"),
        "baseUrl": analysis.get("input_url") or "",
        "selectors": st.session_state.get("selectors") or {},
    }
    config_json = json.dumps(config).replace("</", "<\\/")
    injected = template.replace("__FB_CONFIG_JSON__", config_json)
    components.html(injected, height=740, scrolling=False)


def _render_items_table(items: list, columns: list[str] | None = None) -> None:
    if not items:
        st.caption("No items to display.")
        return
    if columns is None:
        columns = ["title", "detail_url",
                   "published_at", "description", "image"]
    columns = [c for c in columns if any(c in item for item in items)]
    header = "".join(f"<th>{html_module.escape(c)}</th>" for c in columns)
    rows = []
    for item in items:
        cells = []
        for col in columns:
            val = str(item.get(col, "") or "")
            if col == "detail_url" and val:
                safe = html_module.escape(val)
                cells.append(
                    f'<td><a href="{safe}" target="_blank">{safe[:80]}...</a></td>')
            elif col == "image" and val:
                if val.startswith("data:"):
                    cells.append(
                        '<td><span style="color:#888;">embedded image</span></td>')
                else:
                    safe = html_module.escape(val)
                    cells.append(
                        f'<td><img src="{safe}" style="max-width:80px;max-height:50px;object-fit:cover;" /></td>'
                    )
            else:
                cells.append(f"<td>{html_module.escape(val[:120])}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        '<div style="overflow:auto;max-height:360px;"><table style="width:100%;font-size:13px;border-collapse:collapse;">'
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def page_create_feed() -> None:
    if flash := st.session_state.pop("_save_flash", None):
        st.success(flash)

    if _sync_selectors_from_query():
        if st.session_state.get("analysis"):
            _refresh_preview(st.session_state["analysis"])
        st.session_state["_apply_flash"] = "Selectors applied from visual builder."
        st.rerun()

    if apply_flash := st.session_state.pop("_apply_flash", None):
        st.success(apply_flash)

    st.header("Create Feed")
    st.caption(
        "Flow: **Fetch URL** → pick selectors in builder → **Apply selectors** (loads into form) → **Save Feed** (writes MongoDB)."
    )

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        source_name = st.text_input(
            "Source name",
            value=st.session_state.get("source_name", ""),
            placeholder="PR Newswire M&A",
        )
    with c2:
        source_id = st.text_input(
            "Source ID",
            value=st.session_state.get(
                "source_id") or slugify_source_id(source_name),
            help="Unique short key for this feed (e.g. calcalistech_zim). Used by cron + dedupe — not the URL.",
        )
    with c3:
        poll_interval = st.number_input(
            "Poll interval (min)", min_value=5, value=10, step=5)

    source_url = st.text_input(
        "Source URL",
        value=st.session_state.get("source_url", ""),
        placeholder="https://www.prnewswire.com/news-releases/...",
    )

    st.session_state["source_name"] = source_name
    st.session_state["source_id"] = source_id
    st.session_state["source_url"] = source_url

    existing = _maybe_load_config_for_url(source_url) if source_url else None
    if existing:
        st.info(
            f"URL already exists — loaded saved config `{existing['source_id']}`. "
            "Saving will **update** MongoDB."
        )

    force_redetect = st.checkbox(
        "Re-detect on fetch (overwrites saved manual selectors / element map)",
        value=False,
        key="force_redetect_selectors",
        help="Leave unchecked to keep MongoDB-saved selectors (HTML) or element_map (RSS) when re-fetching.",
    )

    if st.button("Fetch URL", type="primary"):
        if not source_url:
            st.error("Enter a source URL.")
        else:
            with st.spinner("Fetching live page..."):
                try:
                    existing = get_feed_by_url(source_url)
                    analysis = analyze_url(source_url)
                    st.session_state["analysis"] = analysis
                    st.session_state["preview_items"] = analysis.get(
                        "preview_items", [])
                    if analysis["source_type"] == "rss":
                        saved_map = (existing or {}).get("element_map") or {}
                        if not force_redetect and saved_map:
                            st.session_state["element_map"] = dict(saved_map)
                            _invalidate_rss_map_widgets()
                        else:
                            st.session_state["element_map"] = (
                                saved_map or analysis.get(
                                    "inferred_element_map") or {}
                            )
                        _refresh_rss_preview(analysis)
                    elif analysis["source_type"] == "html":
                        _resolve_html_selectors_after_fetch(
                            analysis,
                            existing,
                            force_redetect=force_redetect,
                        )
                except FetchBlockedError as exc:
                    st.error(str(exc))
                    st.warning(
                        "**BusinessWire workaround:** open the URL in Chrome → "
                        "`Save Page As…` (HTML) → upload below. "
                        "Build selectors locally, then save config. "
                        "Cron may still need Playwright or a licensed BW RSS feed."
                    )
                except Exception as exc:
                    st.error(f"Fetch failed: {exc}")

    with st.expander("Blocked site? Load saved HTML from your browser", expanded=False):
        st.caption(
            "For BusinessWire / Akamai-protected pages: save the listing page in Chrome, then upload here."
        )
        uploaded = st.file_uploader("Upload .html file", type=["html", "htm"])
        if st.button("Load uploaded HTML", disabled=not uploaded):
            if not source_url:
                st.error(
                    "Enter the Source URL first (used to resolve relative links).")
            elif uploaded:
                html_text = uploaded.read().decode("utf-8", errors="replace")
                analysis = analyze_html_body(
                    html_text, source_url, from_upload=True)
                st.session_state["analysis"] = analysis
                st.session_state["preview_items"] = analysis.get(
                    "preview_items", [])
                if analysis["source_type"] == "html":
                    existing_upload = get_feed_by_url(source_url)
                    _resolve_html_selectors_after_fetch(
                        analysis,
                        existing_upload,
                        force_redetect=force_redetect,
                    )
                st.success("HTML loaded from file.")
                st.rerun()

    analysis = st.session_state.get("analysis")
    if not analysis:
        st.info(
            "Fill in source fields and click **Fetch URL** (or upload saved HTML for blocked sites).")
        return

    loaded_via = " (from uploaded HTML)" if analysis.get(
        "loaded_from_upload") else ""
    st.success(
        f"Detected **{analysis['source_type'].upper()}** — "
        f"`{analysis.get('resolved_url', source_url)}`{loaded_via}"
    )

    if analysis["source_type"] == "rss":
        st.info(
            "RSS feed — map each output field to a feed XML element below. "
            "HTML sources use CSS **selectors** instead."
        )

        field_options = sorted(set(analysis.get("rss_field_options") or []))
        element_map = dict(st.session_state.get("element_map") or {})
        inferred = analysis.get("inferred_element_map") or {}

        with st.expander("RSS element mapping", expanded=True):
            st.caption(
                "Pick which feed XML element supplies each column (auto-detected from the first entry).")
            cols = st.columns(2)
            for idx, field in enumerate(OUTPUT_FIELDS):
                if field == "guid":
                    continue
                options = list(field_options)
                current = element_map.get(field) or inferred.get(field) or ""
                if current and current not in options:
                    options = [current] + options
                if not options:
                    continue
                pick_idx = options.index(current) if current in options else 0
                with cols[idx % 2]:
                    wkey = _rss_map_widget_key(field)
                    if wkey not in st.session_state:
                        st.session_state[wkey] = (
                            current if current in options else options[pick_idx]
                        )
                    element_map[field] = st.selectbox(
                        RSS_FIELD_LABELS[field],
                        options,
                        key=wkey,
                    )

            st.session_state["element_map"] = element_map
            if st.button("Refresh RSS preview"):
                _refresh_rss_preview(analysis)
                st.rerun()

        _render_items_table(st.session_state.get("preview_items", []))

        if st.button("Save Feed", type="primary"):
            element_map_to_save = dict(
                st.session_state.get("element_map") or {})
            if not source_name or not source_id:
                st.error("Source name and ID required.")
            else:
                doc = save_feed(
                    {
                        "source_id": source_id,
                        "source_name": source_name,
                        "source_type": "rss",
                        "source_url": analysis.get("resolved_url") or source_url,
                        "fetch_mode": "requests",
                        "selectors": {},
                        "element_map": element_map_to_save,
                        "url_rules": {},
                        "is_active": True,
                        "poll_interval_minutes": int(poll_interval),
                    }
                )
                st.session_state["element_map"] = dict(
                    doc.get("element_map") or element_map_to_save)
                st.session_state["_config_loaded_for_url"] = source_url
                st.session_state["_save_flash"] = (
                    f"Saved `{doc['source_id']}` to MongoDB. element_map: "
                    f"{json.dumps(element_map_to_save)}"
                )
                _invalidate_rss_map_widgets()
                st.rerun()
        return

    suggestions = analysis.get("container_suggestions") or []
    if suggestions:
        labels = [f"{s['selector']} ({s['count']})" for s in suggestions]
        pick = st.selectbox("Suggested containers", labels, index=0)
        if st.button("Use suggestion + auto-infer"):
            sel = suggestions[labels.index(pick)]["selector"]
            _apply_inferred_selectors(analysis, sel)
            st.session_state[
                "_apply_flash"] = f"Auto-inferred selectors for container `{sel}`."
            st.rerun()

    st.markdown(
        "**Step 2:** After tuning selectors in the panel, click **Apply selectors to Feed Builder** "
        "(required before Save — syncs manual edits into Streamlit)."
    )

    with st.expander("Fine-tune selectors", expanded=True):
        sel = st.session_state["selectors"]
        for field, label, default in (
            ("container", "Container", ""),
            ("detail_url", "Link", "a"),
            ("title", "Title", ""),
            ("published_at", "Date", ""),
            ("description", "Description", ""),
            ("image", "Image", ""),
        ):
            wkey = _selector_widget_key(field)
            if wkey not in st.session_state:
                st.session_state[wkey] = sel.get(field, default)
            st.text_input(label, key=wkey)
        _pull_selectors_from_widgets()

    _pull_selectors_from_widgets()
    _render_builder_panel(analysis)

    st.markdown(
        "**Step 3 — Final feed preview** (title, URL, date, description, image)")
    st.caption(
        "Uses selectors from **Apply selectors** or **Fine-tune** above (updates automatically).")
    _pull_selectors_from_widgets()
    _refresh_preview(analysis)
    _render_items_table(st.session_state.get("preview_items", []))
    st.caption(
        f"{len(st.session_state.get('preview_items', []))} items matched.")

    if st.button("Save Feed to MongoDB", type="primary"):
        _pull_selectors_from_widgets()
        selectors_to_save = dict(st.session_state.get("selectors") or {})
        if not source_name or not source_id:
            st.error("Source name and ID required.")
        elif not selectors_to_save.get("container"):
            st.error(
                "Set a container selector first (Apply selectors from panel or Fine-tune).")
        else:
            doc = save_feed(
                {
                    "source_id": source_id,
                    "source_name": source_name,
                    "source_type": "html",
                    "source_url": source_url,
                    "fetch_mode": "requests",
                    "selectors": selectors_to_save,
                    "element_map": {},
                    "url_rules": dict(st.session_state.get("url_rules") or {}),
                    "is_active": True,
                    "poll_interval_minutes": int(poll_interval),
                }
            )
            st.session_state["selectors"] = dict(
                doc.get("selectors") or selectors_to_save)
            st.session_state["_config_loaded_for_url"] = source_url
            st.session_state["_save_flash"] = (
                f"Saved `{doc['source_id']}` to MongoDB. Selectors: "
                f"{json.dumps(selectors_to_save)}"
            )
            _invalidate_selector_widgets()
            st.rerun()


def page_all_feeds() -> None:
    st.header("All Feeds")
    st.caption("Configs stored in MongoDB collection `news_source_configs`.")

    feeds = list_feeds()
    if not feeds:
        st.info("No feeds yet. Create one in the **Create Feed** tab.")
        return

    st.metric("Total feeds", len(feeds))

    for feed in feeds:
        active = "🟢" if feed.get("is_active", True) else "⚪"
        with st.expander(f"{active} {feed.get('source_name')} — `{feed.get('source_id')}`", expanded=False):
            st.markdown(f"**URL:** {feed.get('source_url')}")
            st.markdown(
                f"**Type:** {feed.get('source_type')} | **Poll:** {feed.get('poll_interval_minutes')} min")
            st.markdown(f"**Updated:** {feed.get('updated_at') or '—'}")
            if feed.get("source_type") == "rss" and feed.get("element_map"):
                st.markdown("**Element map (RSS):**")
                st.json(feed.get("element_map"))
            elif feed.get("selectors"):
                st.markdown("**Selectors (HTML):**")
                st.json(feed.get("selectors"))
            if st.button("Delete", key=f"del_{feed['source_id']}"):
                delete_feed(feed["source_id"])
                st.rerun()


def main() -> None:
    _init_session_state()
    st.title("📰 Feed Builder")
    st.caption(
        "Create source configs only. Scanning & article collection run via internal cron.")

    tab_create, tab_all = st.tabs(["Create Feed", "All Feeds"])
    with tab_create:
        page_create_feed()
    with tab_all:
        page_all_feeds()


if __name__ == "__main__":
    main()
