"""
RSS feed update email HTML builder.

Generates a single HTML email for RSS.app webhook payloads (feed + new items).
"""
from typing import Any, Dict, List


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


def generate_rss_feed_update_email_html(
    feed_data: Dict[str, Any],
    items_new: List[Dict[str, Any]],
) -> tuple:
    """
    Generate a single HTML email for an RSS feed webhook update.

    Args:
        feed_data: Webhook feed object (title, source_url, rss_feed_url, description, icon).
        items_new: List of new items (url, title, description_text, thumbnail, date_published, authors).

    Returns:
        tuple: (subject, html_email)
    """
    feed_title = escape_html(feed_data.get("title") or "RSS Feed")
    source_url = feed_data.get("source_url") or ""
    feed_description = escape_html((feed_data.get("description") or "")[:300])
    n = len(items_new)

    subject = f"RSS update: {feed_title} – {n} new item{'s' if n != 1 else ''}"

    items_html_parts = []
    for idx, item in enumerate(items_new):
        url = item.get("url") or "#"
        title = escape_html(item.get("title") or "Untitled")
        desc = (item.get("description_text") or "")[:200]
        if len((item.get("description_text") or "")) > 200:
            desc += "…"
        desc = escape_html(desc)
        date_pub = item.get("date_published") or ""
        authors = item.get("authors") or []
        author_names = ", ".join(a.get("name", "")
                                 for a in authors if a.get("name"))
        author_line = f"<p style=\"margin:4px 0 0 0; font-size:12px; color:#888;\">{escape_html(author_names)}</p>" if author_names else ""

        items_html_parts.append(f"""
    <tr style="background-color:{"#ffffff" if idx % 2 == 0 else "#f9f9f9"};">
      <td style="padding:12px; border:1px solid #e0e0e0; vertical-align:top;">
        <a href="{escape_html(url)}" style="color:#4a90e2; text-decoration:none; font-weight:bold; font-size:15px;" target="_blank">{title}</a>
        <p style="margin:6px 0 0 0; font-size:13px; color:#555; line-height:1.4;">{desc}</p>
        <p style="margin:4px 0 0 0; font-size:12px; color:#888;">{escape_html(str(date_pub))}</p>
        {author_line}
      </td>
    </tr>
""")

    items_table = "".join(items_html_parts) if items_html_parts else """
    <tr><td style="padding:12px; color:#666;">No new news in this update.</td></tr>
"""

    html_email = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{escape_html(subject)}</title>
</head>
<body style="margin:0; padding:0; font-family:Arial,sans-serif; background-color:#f4f4f4;">
  <div style="max-width:700px; margin:20px auto; background-color:#ffffff; padding:30px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <h2 style="color:#333; text-align:center; margin-top:0; padding-bottom:20px; border-bottom:3px solid #4a90e2;">
      RSS Feed Update
    </h2>

    <div style="margin-bottom:20px;">
      <p style="color:#333; font-size:16px; line-height:1.6;">
        <strong>{feed_title}</strong> has {n} new item{"s" if n != 1 else ""}.
      </p>
      <p style="margin:8px 0; color:#555; font-size:14px;">
        <strong>Source:</strong> <a href="{escape_html(source_url)}" style="color:#4a90e2; text-decoration:none;" target="_blank">{escape_html(source_url)}</a>
      </p>
      {f'<p style="margin:8px 0; color:#666; font-size:13px;">{feed_description}</p>' if feed_description else ''}
    </div>

    <table style="width:100%; border-collapse:collapse; margin-top:16px;">
      <thead>
        <tr style="background-color:#f5f5f5;">
          <th style="padding:10px; border:1px solid #ddd; text-align:left;">New news</th>
        </tr>
      </thead>
      <tbody>
{items_table}
      </tbody>
    </table>

    <p style="margin-top:24px; font-size:12px; color:#999; text-align:center;">
      This email was generated from an RSS.app webhook.
    </p>
  </div>
</body>
</html>
"""
    return subject, html_email


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
    subject = f"PR News : {item_title}"

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
    feed_title = escape_html(feed_data.get("title") or "RSS Feed")
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
      PR News
    </h2>

    <p style="margin:8px 0; font-size:12px; color:#888;">
      From <strong>{feed_title}</strong>
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
