"""
Shared exception email generation and delivery via N8N webhook.

Use `send_exception_email()` in exception handlers to notify on pipeline failures
with dynamic context, traceback, and optional log records.
"""
import json
import logging
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

N8N_WEBHOOK_URL_EXCEPTION = os.environ.get(
    "N8N_WEBHOOK_URL_EXCEPTION",
    "https://n8n.arbintel.cloud/webhook/80830c6d-ff5b-45e3-9ef3-a061db1fbf0c",
)

MAX_LOG_LINES = 200
MAX_LOG_CHARS = 20000
MAX_EXTRA_CHARS = 20000


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


def _format_context_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, indent=2, default=str)
        except Exception:
            return str(value)
    return str(value)


def _truncate_text(text: str, max_chars: int) -> Tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\n...[truncated]", True


def _prepare_log_records(log_records: Optional[List[Any]]) -> Tuple[str, int, bool]:
    logs = log_records or []
    if isinstance(logs, str):
        logs = [logs]

    truncated = False
    if len(logs) > MAX_LOG_LINES:
        truncated = True
        logs = logs[:MAX_LOG_LINES]

    logs_str = "\n".join(str(x) for x in logs)
    logs_str, char_truncated = _truncate_text(logs_str, MAX_LOG_CHARS)
    truncated = truncated or char_truncated
    return logs_str, len(log_records or []), truncated


def _serialize_extra_data(extra_data: Any) -> str:
    if extra_data is None:
        return ""
    try:
        if isinstance(extra_data, (dict, list)):
            extra_str = json.dumps(extra_data, indent=2, default=str)
        else:
            extra_str = str(extra_data)
    except Exception as ser_exc:
        extra_str = (
            "[could not serialize extra_data for email: "
            f"{ser_exc!s}]"
        )
    extra_str, _ = _truncate_text(extra_str, MAX_EXTRA_CHARS)
    return extra_str


def _subject_label(context: Optional[Dict[str, Any]], error_message: str) -> str:
    if not context:
        return (error_message or "Unknown error")[:80]

    for key in (
        "company_name",
        "feed_title",
        "article_url",
        "article_title",
        "url",
        "accession_number",
        "module",
    ):
        value = context.get(key)
        if value:
            return str(value)[:80]

    return (error_message or "Unknown error")[:80]


def _build_context_rows(context: Optional[Dict[str, Any]]) -> str:
    if not context:
        return ""

    rows = []
    for idx, (key, value) in enumerate(context.items()):
        bg = "#f9f9f9" if idx % 2 == 0 else "#ffffff"
        label = escape_html(str(key).replace("_", " ").title())
        formatted = _format_context_value(value)
        cell = escape_html(formatted)
        if formatted.startswith("http://") or formatted.startswith("https://"):
            cell = (
                f'<a href="{cell}" target="_blank" style="color:#4a90e2; '
                f'text-decoration:none;">{cell}</a>'
            )
        rows.append(
            f"""
      <tr style="background-color:{bg};">
        <td style="padding:8px; font-weight:bold; width:170px; color:#555;">{label}</td>
        <td style="padding:8px; color:#333; white-space:pre-wrap; word-break:break-word;">{cell}</td>
      </tr>"""
        )
    return "\n".join(rows)


def generate_exception_email_html(
    *,
    pipeline: str,
    error_message: str,
    context: Optional[Dict[str, Any]] = None,
    log_records: Optional[List[Any]] = None,
    extra_data: Any = None,
    exception: Optional[BaseException] = None,
) -> Tuple[str, str]:
    """
    Build subject and HTML for a pipeline exception email.

    Returns:
        Tuple of (subject, html_email).
    """
    pipeline_esc = escape_html(pipeline or "unknown")
    error_esc = escape_html(error_message or "Unknown error")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    label = escape_html(_subject_label(context, error_message))
    subject = f"❌ [{pipeline}] Error - {label} ({timestamp})"

    context_rows = _build_context_rows(context)
    context_table = ""
    if context_rows:
        context_table = f"""
    <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
      <tr style="background-color:#eef2f7;">
        <td colspan="2" style="padding:10px; font-weight:bold; color:#333;">Context</td>
      </tr>
{context_rows}
    </table>"""

    traceback_str = ""
    if exception is not None:
        traceback_str = "".join(
            traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
        )
    traceback_esc = escape_html(traceback_str)

    logs_str, logs_count, logs_truncated = _prepare_log_records(log_records)
    logs_esc = escape_html(logs_str)

    extra_str = _serialize_extra_data(extra_data)
    extra_esc = escape_html(extra_str)

    html_email = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Pipeline Exception</title>
</head>
<body style="margin:0; padding:0; font-family:Arial,sans-serif; background-color:#f4f4f4;">
  <div style="max-width:900px; margin:20px auto; background-color:#ffffff; padding:30px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <h2 style="color:#333; text-align:center; margin-top:0; padding-bottom:20px; border-bottom:3px solid #dc3545;">
      Pipeline Exception
    </h2>

    <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; width:170px; color:#555;">Pipeline</td>
        <td style="padding:8px; color:#333;">{pipeline_esc}</td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Timestamp (UTC)</td>
        <td style="padding:8px; color:#333;">{escape_html(timestamp)}</td>
      </tr>
    </table>

    {context_table}

    <h3 style="color:#333; margin-top:20px; margin-bottom:10px;">Error</h3>
    <pre style="white-space:pre-wrap; word-break:break-word; background:#fff7f7; border:1px solid #f1c0c0; padding:12px; border-radius:6px; font-size:12px;">{error_esc}</pre>

    <h3 style="color:#333; margin-top:20px; margin-bottom:10px;">
      Traceback
      <span style="color:#888; font-weight:normal;">{'' if traceback_esc else '(none)'}</span>
    </h3>
    <pre style="white-space:pre-wrap; word-break:break-word; background:#f7f7f7; border:1px solid #e6e6e6; padding:12px; border-radius:6px; font-size:12px;">{traceback_esc if traceback_esc else "No traceback available."}</pre>

    <h3 style="color:#333; margin-top:20px; margin-bottom:10px;">
      log_records / warnings
      <span style="color:#888; font-weight:normal;">({logs_count} entries{', truncated' if logs_truncated else ''})</span>
    </h3>
    <pre style="white-space:pre-wrap; word-break:break-word; background:#f7f7f7; border:1px solid #e6e6e6; padding:12px; border-radius:6px; font-size:12px;">{logs_esc if logs_esc else "No log_records available."}</pre>

    <h3 style="color:#333; margin-top:20px; margin-bottom:10px;">
      extra_data
      <span style="color:#888; font-weight:normal;">{'' if extra_esc else '(none)'}</span>
    </h3>
    <pre style="white-space:pre-wrap; word-break:break-word; background:#f7f7f7; border:1px solid #e6e6e6; padding:12px; border-radius:6px; font-size:12px;">{extra_esc if extra_esc else "No extra_data available."}</pre>

    <div style="margin-top:22px; padding-top:18px; border-top:1px solid #e0e0e0; text-align:center; color:#999; font-size:12px;">
      Sent via N8N webhook for debugging pipeline failures.
    </div>
  </div>
</body>
</html>
"""
    return subject, html_email


def send_exception_email(
    *,
    pipeline: str,
    error_message: str,
    context: Optional[Dict[str, Any]] = None,
    log_records: Optional[List[Any]] = None,
    extra_data: Any = None,
    exception: Optional[BaseException] = None,
    webhook_url: Optional[str] = None,
    email_type: str = "exception",
) -> bool:
    """
    Generate and send an exception email via N8N webhook.

    Never raises; returns True when the webhook call succeeds.
    """
    try:
        from sec_rss_parser.utils_8k import send_webhook_notification

        subject, html_email = generate_exception_email_html(
            pipeline=pipeline,
            error_message=error_message,
            context=context,
            log_records=log_records,
            extra_data=extra_data,
            exception=exception,
        )

        payload = {
            "subject": subject,
            "html": html_email,
            "email_type": email_type,
            "pipeline": pipeline,
            "error_message": error_message,
        }
        if context:
            payload["context"] = context

        send_webhook_notification(
            webhook_url or N8N_WEBHOOK_URL_EXCEPTION,
            payload,
            email_type,
        )
        return True
    except Exception as email_exc:
        logger.error(
            "Failed to send exception email via webhook: %s",
            email_exc,
            exc_info=True,
        )
        return False
