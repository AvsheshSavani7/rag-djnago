"""
RSS feed update email HTML builder.

Generates a single HTML email for RSS.app webhook payloads (feed + new items).
Supports optional deal_info to show linked deal (follow) or new deal details.
"""
from typing import Any, Dict, List, Optional


# Map raw feed titles to email display names (subject, heading, "From" line)
FEED_TITLE_DISPLAY_NAMES = {
    "Merger and Acquisition Breaking News and Press Releases": "Business wire",
    "All Acquisitions, Mergers and Takeovers News and Press Releases from PR Newswire": "PR News",
    "GlobeNewswire - Mergers and Acquisitions": "GlobeNewswire - Mergers and Acquisitions",
    "GlobeNewswire - Press Releases": "GlobeNewswire - Press Releases",
    "news.cision.com": "Cision News",
    "NASDAQ": "NASDAQ"
}

FEED_TITLE_DISPLAY_NAME_2 = {
    "Justice News": "Justice News",
    "Federal Trade Commission | Protecting America's Consumers": "Federal Trade Commission",
    "Latest news articles": "Netherlands ACM",
    "RSS CNMC": "CNMC",
    "Press releases | Autorité de la concurrence": "France Autorite"
}


def escape_html(text: Any) -> str:
    """Escape HTML special characters."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&#039;")
    return text


# Label for email_note: existing_deal | new_deal_in_db | new_deal_not_in_db | not_merger_related
EMAIL_NOTE_LABELS = {
    "existing_deal": "Existing deal related article",
    "new_deal_in_db": "New deal ",
    "new_deal_not_in_db": "Deal we not follow",
    "not_merger_related": "Merger related: false",
}


def _deal_info_block(
    deal_info: Dict[str, Any],
    email_note: Optional[str] = None,
    match_details: Optional[Dict[str, Any]] = None,
) -> str:
    """Render deal details block for the email body. Shows US listed and Market cap > $100M when present."""
    note_line = ""
    if email_note and email_note in EMAIL_NOTE_LABELS:
        note_line = f'<p style="margin:0 0 8px 0; font-size:12px; font-weight:bold; color:#2c5282;">{escape_html(EMAIL_NOTE_LABELS[email_note])}</p>'

    target = escape_html(deal_info.get("target_name") or "—")
    acquirer = escape_html(deal_info.get("acquire_name") or "—")
    cik = escape_html(deal_info.get("cik") or "—")
    acquirer_cik = escape_html(deal_info.get("acquirer_cik") or "—")
    sec_url = (deal_info.get("sec_url") or "").strip()
    announce = escape_html(deal_info.get("announce_date") or "—")
    deal_id = escape_html(deal_info.get("id") or "")

    sec_line = ""
    if sec_url:
        sec_line = f'<p style="margin:4px 0 0 0; font-size:12px;">SEC: <a href="{escape_html(sec_url)}" style="color:#4a90e2;" target="_blank">Filing</a></p>'

    us_listed_line = ""
    if "is_target_us_listed" in deal_info:
        val = deal_info["is_target_us_listed"]
        text = "Yes" if val else "No"
        color = "#28a745" if val else "#dc3545"
        us_listed_line = f'<p style="margin:4px 0 0 0; font-size:12px; color:#555;">Target US listed: <span style="color:{color}; font-weight:bold;">{escape_html(text)}</span></p>'

    market_cap_line = ""
    if "is_target_market_cap_gt_100m" in deal_info:
        val = deal_info["is_target_market_cap_gt_100m"]
        text = "Yes (&gt; $100M)" if val else "No (&lt; $100M)"
        color = "#28a745" if val else "#dc3545"
        market_cap_line = f'<p style="margin:4px 0 0 0; font-size:12px; color:#555;">Target market cap &gt; $100M: <span style="color:{color}; font-weight:bold;">{escape_html(text)}</span></p>'

    # Match details block (matched side + keywords)
    match_details_block = ""
    if match_details:
        matched_side = match_details.get("matched_side")
        match_keywords = match_details.get("match_keywords")
        keywords_list = match_keywords if isinstance(
            match_keywords, list) and match_keywords else []

        if matched_side or keywords_list:
            parts = []
            if matched_side:
                side_display = matched_side.capitalize()
                parts.append(
                    f'<p style="margin:0 0 4px 0; font-size:12px; color:#555;">Matched side: <strong style="color:#4a90e2;">{escape_html(side_display)}</strong></p>')
            if keywords_list:
                keywords_escaped = ", ".join(
                    escape_html(str(k)) for k in keywords_list)
                parts.append(
                    f'<p style="margin:4px 0 0 0; font-size:12px; color:#555; background-color:#f0f7ff; padding:8px; border-radius:3px;">Matched keywords: {keywords_escaped}</p>')

            match_details_block = f"""
      <div style="margin:8px 0 0 0; padding-top:8px; border-top:1px solid #e0e0e0;">
        <p style="margin:0 0 4px 0; font-size:11px; font-weight:bold; color:#666; text-transform:uppercase;">Match Evidence</p>
        {''.join(parts)}
      </div>"""

    return f"""
    <div style="margin:16px 0; padding:12px; background-color:#f8f9fa; border-left:4px solid #4a90e2; border-radius:4px;">
      {note_line}
      <p style="margin:0 0 6px 0; font-size:12px; font-weight:bold; color:#333;">Deal details</p>
      <p style="margin:0; font-size:12px; color:#555;">Target: <strong>{target}</strong> (CIK: {cik})</p>
      <p style="margin:4px 0 0 0; font-size:12px; color:#555;">Acquirer: <strong>{acquirer}</strong> (CIK: {acquirer_cik})</p>
      <p style="margin:4px 0 0 0; font-size:12px; color:#555;">Announce date: {announce}</p>
      {sec_line}
      {us_listed_line}
      {market_cap_line}
      {f'<p style="margin:4px 0 0 0; font-size:11px; color:#888;">Deal ID: {deal_id}</p>' if deal_id else ''}
      {match_details_block}
    </div>"""


def _render_l3_value(key: str, value: Any, level: int) -> str:
    """
    Render a single L3 key-value by type. Recursive for nested objects.
    - string → L2-style block with key as label
    - list of strings → key as label + <ul><li>...</li></ul>
    - list of objects or single dict → recurse with level+1 and indent
    """
    indent_px = level * 20
    margin_style = f"margin-left:{indent_px}px;" if indent_px else ""

    if value is None:
        return ""

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return ""
        return f"""
    <div style="margin-bottom:12px; padding:10px; background-color:#eef5ff; border-left:4px solid #0b5ed7; border-radius:4px; {margin_style}">
      <p style="margin:0 0 4px 0; font-size:11px; font-weight:bold; color:#0b5ed7; text-transform:uppercase; letter-spacing:0.5px;">{escape_html(key)}</p>
      <p style="margin:0; font-size:15px; font-weight:bold; color:#003366; line-height:1.5;">{escape_html(s)}</p>
    </div>
"""

    if isinstance(value, list):
        if not value:
            return ""
        if all(isinstance(item, str) for item in value):
            items_html = "".join(
                f'<li style="margin:4px 0; line-height:1.5; font-size:15px; font-weight:bold; color:#003366;">{escape_html(str(item).strip())}</li>'
                for item in value if str(item).strip()
            )
            if not items_html:
                return ""
            return f"""
    <div style="margin-bottom:12px; padding:10px; background-color:#eef5ff; border-left:4px solid #0b5ed7; border-radius:4px; {margin_style}">
      <p style="margin:0 0 6px 0; font-size:11px; font-weight:bold; color:#0b5ed7; text-transform:uppercase; letter-spacing:0.5px;">{escape_html(key)}</p>
      <ul style="margin:0; padding-left:20px;">{items_html}</ul>
    </div>
"""
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(_render_l3_detailed(item, level + 1))
            else:
                parts.append(_render_l3_value(key, str(item), level))
        if not parts:
            return ""
        key_label = f"""
    <div style="margin-bottom:6px; {margin_style}">
      <p style="margin:0; font-size:11px; font-weight:bold; color:#0b5ed7; text-transform:uppercase; letter-spacing:0.5px;">{escape_html(key)}</p>
    </div>
"""
        return key_label + "".join(parts)

    if isinstance(value, dict):
        return _render_l3_detailed(value, level + 1)

    return f"""
    <div style="margin-bottom:12px; padding:10px; background-color:#eef5ff; border-left:4px solid #0b5ed7; border-radius:4px; {margin_style}">
      <p style="margin:0 0 4px 0; font-size:11px; font-weight:bold; color:#0b5ed7; text-transform:uppercase; letter-spacing:0.5px;">{escape_html(key)}</p>
      <p style="margin:0; font-size:15px; font-weight:bold; color:#003366; line-height:1.5;">{escape_html(str(value))}</p>
    </div>
"""


def _render_l3_detailed(l3_data: Dict[str, Any], level: int = 0) -> str:
    """Recursively render L3 dict key-value pairs with type-based formatting and indent."""
    if not l3_data or not isinstance(l3_data, dict):
        return ""
    parts = []
    for k, v in l3_data.items():
        if k is None:
            continue
        key_str = str(k).strip()
        if not key_str:
            continue
        parts.append(_render_l3_value(key_str, v, level))
    return "".join(parts)


def _old_email_html(
    subject: str,
    feed_display_name_escaped: str,
    source_url: str,
    url: str,
    item_title: str,
    desc_escaped: str,
    date_pub: str,
    author_line: str,
    extra_content: str = "",
) -> str:
    """
    Base (old) email HTML: feed header, article link, description, date, authors, read more.
    Old email = call with extra_content="".
    New email = old email HTML + call with extra_content=deal_block (deal related info).
    """
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{escape_html(subject)}</title>
</head>
<body style="margin:0; padding:0; font-family:Arial,sans-serif; background-color:#f4f4f4;">
  <div style="max-width:700px; margin:20px auto; background-color:#ffffff; padding:30px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <h2 style="color:#333; margin-top:0; padding-bottom:16px; border-bottom:3px solid #4a90e2;">
      {feed_display_name_escaped}
    </h2>

    <p style="margin:8px 0; font-size:12px; color:#888;">
      From <strong>{feed_display_name_escaped}</strong>
      {f' · <a href="{escape_html(source_url)}" style="color:#4a90e2;" target="_blank">Source</a>' if source_url else ''}
    </p>

    <div style="margin:20px 0;">
      <a href="{escape_html(url)}" style="color:#4a90e2; text-decoration:none; font-weight:bold; font-size:18px;" target="_blank">{item_title}</a>
      <p style="margin:10px 0 0 0; font-size:14px; color:#555; line-height:1.5;">{desc_escaped}</p>
      <p style="margin:8px 0 0 0; font-size:12px; color:#888;">{escape_html(str(date_pub))}</p>
      {author_line}
    </div>
    {extra_content}

    <p style="margin-top:20px;">
      <a href="{escape_html(url)}" style="display:inline-block; background-color:#4a90e2; color:#fff; padding:10px 20px; text-decoration:none; border-radius:5px; font-size:14px;" target="_blank">Read more</a>
    </p>

  </div>
</body>
</html>
"""


def generate_rss_feed_item_email_html(
    feed_data: Dict[str, Any],
    item: Dict[str, Any],
    deal_info: Optional[Dict[str, Any]] = None,
    email_note: Optional[str] = None,
    match_details: Optional[Dict[str, Any]] = None,
) -> tuple:
    """
    Generate HTML email for a single RSS feed item (one email per item).

    Old email = old email HTML only (base).
    New email = old email HTML + deal related info (when deal_info is provided).

    Args:
        feed_data: Webhook feed object (title, source_url, description, icon).
        item: Single item (url, title, description_text, thumbnail, date_published, authors).
        deal_info: Optional dict with id, target_name, acquire_name, cik, acquirer_cik, sec_url, announce_date.
        email_note: "existing_deal" | "new_deal_in_db" | "new_deal_not_in_db" for deal block label.
        match_details: Optional dict with matched_side and match_keywords from Prompt 1.

    Returns:
        tuple: (subject, html_email) with subject "PR News : {item title}"
    """
    item_title = escape_html(item.get("title") or "Untitled")
    raw_feed_title = feed_data.get("title") or "RSS Feed"
    feed_display_name = FEED_TITLE_DISPLAY_NAMES.get(
        raw_feed_title, raw_feed_title
    )
    feed_display_name_escaped = escape_html(feed_display_name)
    subject = f"{feed_display_name} : {item_title}"

    url = item.get("url") or "#"
    desc = item.get("description_text") or ""
    desc_escaped = escape_html(desc)
    date_pub = item.get("date_published") or ""
    authors = item.get("authors") or []
    author_names = ", ".join(a.get("name", "")
                             for a in authors if a.get("name"))
    author_line = (
        f'<p style="margin:4px 0 0 0; font-size:12px; color:#888;">{escape_html(author_names)}</p>'
        if author_names
        else ""
    )
    source_url = feed_data.get("source_url") or ""

    # Optional summary block (from sec_rss_parser.sec_summarizers.filing_router)
    l1_headline = item.get("l1_headline")
    l2_brief = item.get("l2_brief")
    l3_detailed = item.get("l3_detailed")
    s3_docx_url = item.get("s3_docx_url")
    s3_json_url = item.get("s3_json_url")

    # L3: dict → recursive by type (string / list of strings / list of objects); str → legacy single block
    l3_block = ""
    if l3_detailed is not None:
        if isinstance(l3_detailed, dict):
            inner = (
                '<p style="margin:0 0 6px 0; font-size:11px; font-weight:bold; color:#0b5ed7; text-transform:uppercase;">L3 — Detailed</p>'
                + _render_l3_detailed(l3_detailed, 0)
            )
            l3_block = f'<div style="margin-top:12px;">{inner}</div>'
        elif isinstance(l3_detailed, str) and l3_detailed.strip():
            l3_block = f'<p style="margin:12px 0 0 0; font-size:15px; font-weight:bold; color:#003366; line-height:1.5;">{escape_html(l3_detailed.strip())}</p>'

    summary_block = ""
    if l1_headline or l2_brief or l3_detailed or s3_docx_url or s3_json_url:
        summary_parts: List[str] = []
        if l1_headline:
            summary_parts.append(
                '<div style="margin-bottom:16px; padding:12px; background-color:#eef5ff; border-left:4px solid #0b5ed7; border-radius:4px;">'
                '<p style="margin:0 0 6px 0; font-size:12px; font-weight:bold; color:#0b5ed7; text-transform:uppercase; letter-spacing:0.5px;">L1 — Headline</p>'
                f'<p style="margin:0; font-size:15px; font-weight:bold; color:#003366; line-height:1.5;">{escape_html(l1_headline)}</p>'
                '</div>'
            )
        if l2_brief:
            summary_parts.append(
                '<div style="margin-bottom:16px; padding:12px; background-color:#eef5ff; border-left:4px solid #0b5ed7; border-radius:4px;">'
                '<p style="margin:0 0 6px 0; font-size:12px; font-weight:bold; color:#0b5ed7; text-transform:uppercase; letter-spacing:0.5px;">L2 — Brief</p>'
                f'<p style="margin:0; font-size:15px; font-weight:bold; color:#003366; line-height:1.5;">{escape_html(l2_brief)}</p>'
                '</div>'
            )
        if l3_block:
            summary_parts.append(l3_block)
        link_bits = []
        if s3_docx_url:
            link_bits.append(
                f'<a href="{escape_html(s3_docx_url)}" style="color:#0b5ed7; text-decoration:none;" target="_blank" rel="noopener noreferrer">View DOCX summary</a>'
            )
        if s3_json_url:
            if link_bits:
                link_bits.append("&nbsp;·&nbsp;")
            link_bits.append(
                f'<a href="{escape_html(s3_json_url)}" style="color:#0b5ed7; text-decoration:none;" target="_blank" rel="noopener noreferrer">View JSON</a>'
            )
        if link_bits:
            summary_parts.append(
                f'<p style="margin:4px 0 0 0; font-size:12px; color:#555;">{"".join(link_bits)}</p>'
            )

        if summary_parts:
            summary_block = "".join(summary_parts)

    # Old email = old email HTML (no extra content)
    # New email = old email HTML + summary (if any) + deal related info or not_merger_related flag
    if email_note == "not_merger_related":
        note_label = EMAIL_NOTE_LABELS.get(
            "not_merger_related", "Merger related: false")
        note_block = (
            f'<div style="margin:16px 0; padding:12px; background-color:#f8f9fa; border-left:4px solid #6c757d; border-radius:4px;">'
            f'<p style="margin:0; font-size:12px; font-weight:bold; color:#555;">{escape_html(note_label)}</p>'
            f"</div>"
        )
        extra_content = summary_block + note_block
    elif deal_info:
        deal_block = _deal_info_block(deal_info, email_note, match_details)
        extra_content = summary_block + deal_block
    else:
        extra_content = summary_block
    html_email = _old_email_html(
        subject=subject,
        feed_display_name_escaped=feed_display_name_escaped,
        source_url=source_url,
        url=url,
        item_title=item_title,
        desc_escaped=desc_escaped,
        date_pub=date_pub,
        author_line=author_line,
        extra_content=extra_content,
    )

    return subject, html_email


# ─── Flow 2: Title/description deal match (no article fetch, no save). Separate template to avoid confusion. ───

def _deal_info_block_flow2(deal_info: Dict[str, Any]) -> str:
    """Render deal details block for Flow 2 emails. Deal info in clear HTML format."""
    target = escape_html(deal_info.get("target_name") or "—")
    acquirer = escape_html(deal_info.get("acquire_name") or "—")
    cik = escape_html(deal_info.get("cik") or "—")
    acquirer_cik = escape_html(deal_info.get("acquirer_cik") or "—")
    sec_url = (deal_info.get("sec_url") or "").strip()
    announce = escape_html(deal_info.get("announce_date") or "—")
    deal_id = escape_html(deal_info.get("id") or "")

    sec_line = ""
    if sec_url:
        sec_line = f'<p style="margin:4px 0 0 0; font-size:12px;">SEC: <a href="{escape_html(sec_url)}" style="color:#4a90e2;" target="_blank">Filing</a></p>'

    return f"""
    <div style="margin:16px 0; padding:12px; background-color:#f0f7ff; border-left:4px solid #2563eb; border-radius:4px;">
      <p style="margin:0 0 8px 0; font-size:11px; font-weight:bold; color:#1e40af; text-transform:uppercase; letter-spacing:0.5px;">Deal match (title/description)</p>
      <p style="margin:0 0 6px 0; font-size:12px; font-weight:bold; color:#333;">Deal details</p>
      <table style="font-size:12px; color:#555; border-collapse:collapse;">
        <tr><td style="padding:2px 8px 2px 0; vertical-align:top; font-weight:bold;">Target:</td><td>{target} (CIK: {cik})</td></tr>
        <tr><td style="padding:2px 8px 2px 0; vertical-align:top; font-weight:bold;">Acquirer:</td><td>{acquirer} (CIK: {acquirer_cik})</td></tr>
        <tr><td style="padding:2px 8px 2px 0; vertical-align:top; font-weight:bold;">Announce date:</td><td>{announce}</td></tr>
        {f'<tr><td style="padding:2px 8px 2px 0; vertical-align:top; font-weight:bold;">Deal ID:</td><td>{deal_id}</td></tr>' if deal_id else ''}
      </table>
      {sec_line}
    </div>"""


def _flow2_email_html(
    subject: str,
    feed_display_name_escaped: str,
    source_url: str,
    url: str,
    item_title: str,
    desc_escaped: str,
    date_pub: str,
    author_line: str,
    deal_block: str,
) -> str:
    """
    HTML layout for Flow 2 emails (title/description deal match only).
    Same structure as main RSS email but with Flow 2 badge and deal block; no AI summary.
    """
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{escape_html(subject)}</title>
</head>
<body style="margin:0; padding:0; font-family:Arial,sans-serif; background-color:#f4f4f4;">
  <div style="max-width:700px; margin:20px auto; background-color:#ffffff; padding:30px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <p style="margin:0 0 6px 0; font-size:10px; color:#2563eb; font-weight:bold; text-transform:uppercase;">Flow 2 · Deal match from feed</p>
    <h2 style="color:#333; margin-top:0; padding-bottom:16px; border-bottom:3px solid #2563eb;">
      {feed_display_name_escaped}
    </h2>

    <p style="margin:8px 0; font-size:12px; color:#888;">
      From <strong>{feed_display_name_escaped}</strong>
      {f' · <a href="{escape_html(source_url)}" style="color:#2563eb;" target="_blank">Source</a>' if source_url else ''}
    </p>

    <div style="margin:20px 0;">
      <a href="{escape_html(url)}" style="color:#2563eb; text-decoration:none; font-weight:bold; font-size:18px;" target="_blank">{item_title}</a>
      <p style="margin:10px 0 0 0; font-size:14px; color:#555; line-height:1.5;">{desc_escaped}</p>
      <p style="margin:8px 0 0 0; font-size:12px; color:#888;">{escape_html(str(date_pub))}</p>
      {author_line}
    </div>
    {deal_block}

    <p style="margin-top:20px;">
      <a href="{escape_html(url)}" style="display:inline-block; background-color:#2563eb; color:#fff; padding:10px 20px; text-decoration:none; border-radius:5px; font-size:14px;" target="_blank">Read more</a>
    </p>

  </div>
</body>
</html>
"""


def generate_rss_feed_item_email_html_flow2(
    feed_data: Dict[str, Any],
    item: Dict[str, Any],
    deal_info: Optional[Dict[str, Any]] = None,
) -> tuple:
    """
    Generate HTML email for Flow 2: deal match from title/description only (no article fetch, no save).

    Use this only for use_merger_flow_2 feeds (Justice News, FTC, ACM, CNMC, etc.).
    Deal info is rendered in HTML. Subject uses FEED_TITLE_DISPLAY_NAME_2.

    Args:
        feed_data: Webhook feed object (title, source_url, description).
        item: Single item (url, title, description_text, date_published, authors).
        deal_info: Dict with id, target_name, acquire_name, cik, acquirer_cik, sec_url, announce_date.

    Returns:
        tuple: (subject, html_email)
    """
    item_title = escape_html(item.get("title") or "Untitled")
    raw_feed_title = feed_data.get("title") or "RSS Feed"
    feed_display_name = FEED_TITLE_DISPLAY_NAME_2.get(
        raw_feed_title, raw_feed_title)
    feed_display_name_escaped = escape_html(feed_display_name)
    subject = f"{feed_display_name} : {item_title}"

    url = item.get("url") or "#"
    desc = item.get("description_text") or ""
    desc_escaped = escape_html(desc)
    date_pub = item.get("date_published") or ""
    authors = item.get("authors") or []
    author_names = ", ".join(a.get("name", "")
                             for a in authors if a.get("name"))
    author_line = (
        f'<p style="margin:4px 0 0 0; font-size:12px; color:#888;">{escape_html(author_names)}</p>'
        if author_names
        else ""
    )
    source_url = feed_data.get("source_url") or ""

    deal_block = _deal_info_block_flow2(deal_info) if deal_info else ""
    html_email = _flow2_email_html(
        subject=subject,
        feed_display_name_escaped=feed_display_name_escaped,
        source_url=source_url,
        url=url,
        item_title=item_title,
        desc_escaped=desc_escaped,
        date_pub=date_pub,
        author_line=author_line,
        deal_block=deal_block,
    )
    return subject, html_email
