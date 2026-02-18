"""
RSS feed update email HTML builder.

Generates a single HTML email for RSS.app webhook payloads (feed + new items).
"""
from typing import Any, Dict, List


# Map raw feed titles to email display names (subject, heading, "From" line)
FEED_TITLE_DISPLAY_NAMES = {
    "Merger and Acquisition Breaking News and Press Releases": "Business wire",
    "All Acquisitions, Mergers and Takeovers News and Press Releases from PR Newswire": "PR News",
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


def generate_rss_feed_item_email_html(
    feed_data: Dict[str, Any],
    item: Dict[str, Any],
) -> tuple:
    """
    Generate HTML email for a single RSS feed item (one email per item).

    Args:
        feed_data: Webhook feed object (title, source_url, description, icon).
        item: Single item (url, title, description_text, thumbnail, date_published, authors).

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

    html_email = f"""
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

    <p style="margin-top:20px;">
      <a href="{escape_html(url)}" style="display:inline-block; background-color:#4a90e2; color:#fff; padding:10px 20px; text-decoration:none; border-radius:5px; font-size:14px;" target="_blank">Read more</a>
    </p>

  
  </div>
</body>
</html>
"""
    return subject, html_email
