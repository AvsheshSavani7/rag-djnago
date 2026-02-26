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
    "GlobeNewswire - Press Releases": "GlobeNewswire - Press Releases"

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


# Label for email_note: existing_deal | new_deal_in_db | new_deal_not_in_db
EMAIL_NOTE_LABELS = {
    "existing_deal": "Existing deal related article",
    "new_deal_in_db": "New deal ",
    "new_deal_not_in_db": "Deal we not follow",
}


def _deal_info_block(
    deal_info: Dict[str, Any],
    email_note: Optional[str] = None,
) -> str:
    """Render deal details block for the email body."""
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

    return f"""
    <div style="margin:16px 0; padding:12px; background-color:#f8f9fa; border-left:4px solid #4a90e2; border-radius:4px;">
      {note_line}
      <p style="margin:0 0 6px 0; font-size:12px; font-weight:bold; color:#333;">Deal details</p>
      <p style="margin:0; font-size:12px; color:#555;">Target: <strong>{target}</strong> (CIK: {cik})</p>
      <p style="margin:4px 0 0 0; font-size:12px; color:#555;">Acquirer: <strong>{acquirer}</strong> (CIK: {acquirer_cik})</p>
      <p style="margin:4px 0 0 0; font-size:12px; color:#555;">Announce date: {announce}</p>
      {sec_line}
      {f'<p style="margin:4px 0 0 0; font-size:11px; color:#888;">Deal ID: {deal_id}</p>' if deal_id else ''}
    </div>"""


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

    # Old email = old email HTML (no extra content)
    # New email = old email HTML + deal related info (extra_content = deal block)
    deal_block = _deal_info_block(deal_info, email_note) if deal_info else ""
    html_email = _old_email_html(
        subject=subject,
        feed_display_name_escaped=feed_display_name_escaped,
        source_url=source_url,
        url=url,
        item_title=item_title,
        desc_escaped=desc_escaped,
        date_pub=date_pub,
        author_line=author_line,
        extra_content=deal_block,
    )

    return subject, html_email
