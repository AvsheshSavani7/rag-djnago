"""
SEC filing email HTML builders.

Centralized module for all email template generation.
Easier to maintain, debug, and update without touching services logic.
"""
from datetime import datetime


def escape_html(text):
    """Escape HTML special characters"""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&#039;")
    return text


def _doc_display_name(file):
    """Display name for document: 'file' key, or description, or filename from URL."""
    name = file.get('file') or file.get('description')
    if name:
        return str(name).strip()
    url = file.get('url', '')
    if url:
        return url.rstrip('/').split('/')[-1] or 'Document'
    return 'Document'


def build_doc_files_table(doc_files):
    """
    Build HTML table for document format files.
    Shared by filing and EX-99.1 merger emails.
    Uses fallbacks for missing sequence, file, size when only type/url/description are passed.
    """
    if not doc_files or len(doc_files) == 0:
        return "<p><em>No Document Format Files found.</em></p>"

    rows = []
    for idx, file in enumerate(doc_files):
        bg = "#ffffff" if idx % 2 == 0 else "#f9f9f9"
        seq = escape_html(file.get('sequence') if file.get(
            'sequence') not in (None, '') else str(idx + 1))
        description = escape_html(file.get('description', ''))
        doc_name = escape_html(_doc_display_name(file))
        doc_type = escape_html(file.get('type', ''))
        size_val = file.get('size')
        # Size can be int (bytes) from SEC table; show number or — if missing
        if size_val is None or size_val == '':
            size = '—'
        else:
            size = escape_html(str(size_val))

        doc_url = file.get('url', '')
        if doc_url:
            if not (doc_url.startswith('http://') or doc_url.startswith('https://')):
                doc_url = f"https://www.sec.gov{doc_url}"
            doc_name_html = f'<a href="{escape_html(doc_url)}" style="color:#4a90e2; text-decoration:none;" target="_blank">{doc_name}</a>'
        else:
            doc_name_html = doc_name

        rows.append(f"""
      <tr style="background-color:{bg};">
        <td style="padding:8px; border:1px solid #ddd;">{seq}</td>
        <td style="padding:8px; border:1px solid #ddd;">{description}</td>
        <td style="padding:8px; border:1px solid #ddd;">{doc_name_html}</td>
        <td style="padding:8px; border:1px solid #ddd;">{doc_type}</td>
        <td style="padding:8px; border:1px solid #ddd;">{size}</td>
      </tr>
""")
    rows_html = "".join(rows)
    return f"""
    <table style="width:100%; border-collapse:collapse; margin-top:10px;">
      <thead>
        <tr style="background-color:#f5f5f5;">
          <th style="padding:8px; border:1px solid #ddd; text-align:left;">Seq</th>
          <th style="padding:8px; border:1px solid #ddd; text-align:left;">Description</th>
          <th style="padding:8px; border:1px solid #ddd; text-align:left;">Document</th>
          <th style="padding:8px; border:1px solid #ddd; text-align:left;">Type</th>
          <th style="padding:8px; border:1px solid #ddd; text-align:left;">Size</th>
        </tr>
      </thead>
      <tbody>
{rows_html}
      </tbody>
    </table>
"""


def _build_company_details_rows(company_details):
    """Build HTML rows for company details (M&A info) section."""
    if not company_details:
        return ""

    html = """
      <tr>
        <td colspan="2" style="padding:12px 8px 8px 8px; font-weight:bold; color:#4a90e2; font-size:14px; border-top:2px solid #e0e0e0;">
          Company Details (M&A Information)
        </td>
      </tr>
"""
    target_name = company_details.get('target_name', '')
    if target_name:
        html += f"""
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">Target Company:</td>
        <td style="padding:8px; color:#333;">{escape_html(target_name)}</td>
      </tr>
"""
    target_cik = company_details.get('target_cik', '')
    if target_cik:
        html += f"""
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Target CIK:</td>
        <td style="padding:8px; color:#333;">{escape_html(target_cik)}</td>
      </tr>
"""
    acquirer_name = company_details.get('acquirer_name', '')
    if acquirer_name:
        html += f"""
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">Acquirer Company:</td>
        <td style="padding:8px; color:#333;">{escape_html(acquirer_name)}</td>
      </tr>
"""
    acquirer_cik = company_details.get('acquirer_cik', '')
    if acquirer_cik:
        html += f"""
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Acquirer CIK:</td>
        <td style="padding:8px; color:#333;">{escape_html(acquirer_cik)}</td>
      </tr>
"""
    is_us_listed = company_details.get('is_target_us_listed')
    if is_us_listed is not None:
        listed_text = "✓ Yes" if is_us_listed else "✗ No"
        listed_color = "#28a745" if is_us_listed else "#dc3545"
        html += f"""
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">Target US Listed:</td>
        <td style="padding:8px; color:{listed_color}; font-weight:bold;">{escape_html(listed_text)}</td>
      </tr>
"""
    is_cap_gt_100m = company_details.get(
        'is_target_market_cap_greater_than_100m')
    if is_cap_gt_100m is not None:
        cap_text = "✓ Yes (> $100M)" if is_cap_gt_100m else "✗ No (< $100M)"
        cap_color = "#28a745" if is_cap_gt_100m else "#dc3545"
        html += f"""
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Market Cap > $100M:</td>
        <td style="padding:8px; color:{cap_color}; font-weight:bold;">{escape_html(cap_text)}</td>
      </tr>
"""
    return html


def generate_filing_email_html(filing_data, doc_files):
    """Generate HTML email for SEC filing notification (8-K EX-2.1, etc.)."""
    form_type = filing_data.get('form_type', 'N/A')
    company_name = filing_data.get('company_name', 'Unknown Company')
    accession_no = filing_data.get('accession_number', 'N/A')
    filing_date = filing_data.get('filing_date', 'N/A')
    company_details = filing_data.get('company_details', None)
    if isinstance(filing_date, datetime):
        filing_date = filing_date.strftime('%Y-%m-%d')
    accepted_date = filing_data.get('acceptance_datetime_utc', 'N/A')
    if isinstance(accepted_date, datetime):
        accepted_date = accepted_date.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(accepted_date, str):
        try:
            dt = datetime.fromisoformat(accepted_date.replace('Z', '+00:00'))
            accepted_date = dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            pass
    period = filing_data.get('period', 'N/A')
    cik = filing_data.get('cik_number', 'N/A')
    filing_url = filing_data.get('link', '')
    documents_count = len(doc_files) if doc_files else 0

    doc_files_html = build_doc_files_table(doc_files)
    title_text = f"{form_type} – {company_name}" if form_type != 'N/A' and company_name != 'Unknown Company' else f"Filing #{accession_no}"
    subject = f"SEC Filing – {form_type} – {company_name}"

    company_details_html = ""
    if form_type == '8-K' and company_details:
        company_details_html = _build_company_details_rows(company_details)

    filing_url_html = ""
    if filing_url:
        filing_url_html = f"""
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Filing URL:</td>
        <td style="padding:8px;">
          <a href="{escape_html(filing_url)}" style="color:#4a90e2; text-decoration:none;" target="_blank">
            View Filing Detail Page
          </a>
        </td>
      </tr>
"""

    html_email = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{escape_html(subject)}</title>
</head>
<body style="margin:0; padding:0; font-family:Arial,sans-serif; background-color:#f4f4f4;">
  <div style="max-width:900px; margin:20px auto; background-color:#ffffff; padding:30px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <h2 style="color:#333; text-align:center; margin-top:0; padding-bottom:20px; border-bottom:3px solid #4a90e2;">
      {escape_html(title_text)}
    </h2>

    <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
      <tr>
        <td style="padding:8px; font-weight:bold; width:170px; color:#555;">Form Type:</td>
        <td style="padding:8px; color:#333;">{escape_html(form_type)}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">Accession No.:</td>
        <td style="padding:8px; color:#333;">{escape_html(accession_no)}</td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Filing Date:</td>
        <td style="padding:8px; color:#333;">{escape_html(filing_date)}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">Accepted:</td>
        <td style="padding:8px; color:#333;">{escape_html(accepted_date)}</td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Period of Report:</td>
        <td style="padding:8px; color:#333;">{escape_html(period)}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">Documents Count:</td>
        <td style="padding:8px; color:#333;">{escape_html(str(documents_count))}</td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Company:</td>
        <td style="padding:8px; color:#333;">{escape_html(company_name)}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">CIK:</td>
        <td style="padding:8px; color:#333;">{escape_html(cik)}</td>
      </tr>
{company_details_html}

{filing_url_html}
    </table>

    <h3 style="color:#333; margin-top:20px; margin-bottom:10px;">Document Format Files</h3>
    {doc_files_html}

    <div style="margin-top:30px; padding-top:20px; border-top:1px solid #e0e0e0; text-align:center; color:#999; font-size:12px;">
      <p>This is an automated email generated from SEC EDGAR filing detail pages.</p>
    </div>
  </div>
</body>
</html>
"""
    return subject, html_email


def generate_8k_document_email_html(filing_data, doc_files):
    """
    Generate HTML email for main 8-K document notification.
    Used when we analyze the primary 8-K document (not an exhibit).
    Subject/title: "8-K – {company_name}" so it is distinct from EX-99.1 emails.
    """
    form_type = filing_data.get('form_type', '8-K')
    company_name = filing_data.get('company_name', 'Unknown Company')
    accession_no = filing_data.get('accession_number', 'N/A')
    filing_date = filing_data.get('filing_date', 'N/A')
    if isinstance(filing_date, datetime):
        filing_date = filing_date.strftime('%Y-%m-%d')
    cik = filing_data.get('cik_number', 'N/A')
    filing_url = filing_data.get('link', '')
    confidence = filing_data.get(
        'confidence', 0) or filing_data.get('ex99_1_confidence', 0)
    is_merger_related = filing_data.get('is_merger_related', False)
    reasoning = filing_data.get(
        'reasoning', '') or filing_data.get('ex99_1_reasoning', '')
    is_target_us_listed = filing_data.get('is_target_us_listed')
    is_target_market_cap_greater_than_100m = filing_data.get(
        'is_target_market_cap_greater_than_100m')

    def _fmt_bool(val):
        if val is None:
            return 'N/A'
        return 'Yes' if val else 'No'

    # Subject: ticker (if from deal) else company_name : 8-K New Merger : filing_date
    ticker = (filing_data.get("ticker") or "").strip()
    label = ticker or company_name
    filing_date_str = filing_date if isinstance(filing_date, str) else (filing_date.strftime(
        "%Y-%m-%d") if isinstance(filing_date, datetime) else str(filing_date))
    subject = f"{label} : Form 8-K New Merger by {company_name} on [ {filing_date_str} ]"
    confidence_badge = f"<span style='background:#28a745;color:white;padding:2px 8px;border-radius:4px;'>{confidence}% confidence</span>" if confidence else ""
    doc_files_html = build_doc_files_table(doc_files)

    reasoning_html = ""
    if reasoning:
        reasoning_html = f"""
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555; vertical-align:top;">Reasoning:</td>
        <td style="padding:8px; color:#333;">{escape_html(reasoning)}</td>
      </tr>
"""

    filing_url_html = ""
    if filing_url:
        filing_url_html = f"""
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Filing URL:</td>
        <td style="padding:8px;">
          <a href="{escape_html(filing_url)}" style="color:#4a90e2; text-decoration:none;" target="_blank">View Filing</a>
        </td>
      </tr>
"""

    html_email = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{escape_html(subject)}</title>
</head>
<body style="margin:0; padding:0; font-family:Arial,sans-serif; background-color:#f4f4f4;">
  <div style="max-width:900px; margin:20px auto; background-color:#ffffff; padding:30px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <h2 style="color:#333; text-align:center; margin-top:0; padding-bottom:20px; border-bottom:3px solid #4a90e2;">
      8-K – {escape_html(company_name)}
    </h2>
    <p style="background:#e7f3ff; padding:12px; border-left:4px solid #4a90e2; margin:0 0 20px 0;">
      <strong>Main 8-K document analyzed for merger/acquisition relevance.</strong>
    </p>
    <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
      <tr>
        <td style="padding:8px; font-weight:bold; width:170px; color:#555;">Form Type:</td>
        <td style="padding:8px; color:#333;">{escape_html(form_type)}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">Accession No.:</td>
        <td style="padding:8px; color:#333;">{escape_html(accession_no)}</td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Filing Date:</td>
        <td style="padding:8px; color:#333;">{escape_html(filing_date)}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">Confidence:</td>
        <td style="padding:8px;">{confidence_badge}</td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Is Merger Related:</td>
        <td style="padding:8px; color:#333;">{escape_html(is_merger_related)}</td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Company:</td>
        <td style="padding:8px; color:#333;">{escape_html(company_name)}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">CIK:</td>
        <td style="padding:8px; color:#333;">{escape_html(cik)}</td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Target US Listed:</td>
        <td style="padding:8px; color:#333;">{escape_html(_fmt_bool(is_target_us_listed))}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">Target Market Cap &gt; $100M:</td>
        <td style="padding:8px; color:#333;">{escape_html(_fmt_bool(is_target_market_cap_greater_than_100m))}</td>
      </tr>
{reasoning_html}{filing_url_html}
    </table>

    <h3 style="color:#333; margin-top:20px; margin-bottom:10px;">Document Format Files</h3>
    {doc_files_html}

    <div style="margin-top:30px; padding-top:20px; border-top:1px solid #e0e0e0; text-align:center; color:#999; font-size:12px;">
      <p>8-K main document notification. For EX-99.1 exhibit emails, see separate "EX-99.1 M&A-Related Press Release" emails.</p>
    </div>
  </div>
</body>
</html>
"""
    return subject, html_email


def generate_ex99_1_merger_email_html(filing_data, doc_files):
    """
    Generate HTML email for 8-K EX-99.1 merger-related notification.
    Used when is_merger_related=True (EX-99.1 press release identified as M&A-related).
    Different from EX-2.1 Definitive Merger Agreement email.
    Subject/title: "EX-99.1 M&A-Related Press Release – {company_name}".
    """
    form_type = filing_data.get('form_type', '8-K')
    company_name = filing_data.get('company_name', 'Unknown Company')
    accession_no = filing_data.get('accession_number', 'N/A')
    filing_date = filing_data.get('filing_date', 'N/A')
    if isinstance(filing_date, datetime):
        filing_date = filing_date.strftime('%Y-%m-%d')
    cik = filing_data.get('cik_number', 'N/A')
    filing_url = filing_data.get('link', '')
    confidence = filing_data.get(
        'confidence', 0) or filing_data.get('ex99_1_confidence', 0)
    is_merger_related = filing_data.get('is_merger_related', False)
    reasoning = filing_data.get(
        'reasoning', '') or filing_data.get('ex99_1_reasoning', '')
    is_target_us_listed = filing_data.get('is_target_us_listed')
    is_target_market_cap_greater_than_100m = filing_data.get(
        'is_target_market_cap_greater_than_100m')

    def _fmt_bool(val):
        if val is None:
            return 'N/A'
        return 'Yes' if val else 'No'

    # Subject: ticker (if from deal) else company_name : EX-99.1 New Merger : filing_date
    ticker = (filing_data.get("ticker") or "").strip()
    label = ticker or company_name
    filing_date_str = filing_date if isinstance(filing_date, str) else (filing_date.strftime(
        "%Y-%m-%d") if isinstance(filing_date, datetime) else str(filing_date))
    subject = f"{label} : Form EX-99.1 New Merger by {company_name} on [ {filing_date_str} ]"
    confidence_badge = f"<span style='background:#28a745;color:white;padding:2px 8px;border-radius:4px;'>{confidence}% confidence</span>" if confidence else ""
    doc_files_html = build_doc_files_table(doc_files)

    reasoning_html = ""
    if reasoning:
        reasoning_html = f"""
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555; vertical-align:top;">Reasoning:</td>
        <td style="padding:8px; color:#333;">{escape_html(reasoning)}</td>
      </tr>
"""

    filing_url_html = ""
    if filing_url:
        filing_url_html = f"""
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Filing URL:</td>
        <td style="padding:8px;">
          <a href="{escape_html(filing_url)}" style="color:#4a90e2; text-decoration:none;" target="_blank">View Filing</a>
        </td>
      </tr>
"""

    html_email = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{escape_html(subject)}</title>
</head>
<body style="margin:0; padding:0; font-family:Arial,sans-serif; background-color:#f4f4f4;">
  <div style="max-width:900px; margin:20px auto; background-color:#ffffff; padding:30px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <h2 style="color:#333; text-align:center; margin-top:0; padding-bottom:20px; border-bottom:3px solid #ffc107;">
      EX-99.1 M&A-Related Press Release – {escape_html(company_name)}
    </h2>
    <p style="background:#fff3cd; padding:12px; border-left:4px solid #ffc107; margin:0 0 20px 0;">
      <strong>This 8-K filing contains an EX-99.1 exhibit identified as merger/acquisition-related.</strong>
    </p>
    <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
      <tr>
        <td style="padding:8px; font-weight:bold; width:170px; color:#555;">Form Type:</td>
        <td style="padding:8px; color:#333;">{escape_html(form_type)} (EX-99.1)</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">Accession No.:</td>
        <td style="padding:8px; color:#333;">{escape_html(accession_no)}</td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Filing Date:</td>
        <td style="padding:8px; color:#333;">{escape_html(filing_date)}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">Confidence:</td>
        <td style="padding:8px;">{confidence_badge}</td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Is Merger Related:</td>
        <td style="padding:8px; color:#333;">{escape_html(is_merger_related)}</td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Company:</td>
        <td style="padding:8px; color:#333;">{escape_html(company_name)}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">CIK:</td>
        <td style="padding:8px; color:#333;">{escape_html(cik)}</td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Target US Listed:</td>
        <td style="padding:8px; color:#333;">{escape_html(_fmt_bool(is_target_us_listed))}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">Target Market Cap &gt; $100M:</td>
        <td style="padding:8px; color:#333;">{escape_html(_fmt_bool(is_target_market_cap_greater_than_100m))}</td>
      </tr>
{reasoning_html}{filing_url_html}
    </table>

    <h3 style="color:#333; margin-top:20px; margin-bottom:10px;">Document Format Files</h3>
    {doc_files_html}

    <div style="margin-top:30px; padding-top:20px; border-top:1px solid #e0e0e0; text-align:center; color:#999; font-size:12px;">
      <p>8-K EX-99.1 merger-related notification. This is not a Definitive Merger Agreement (EX-2.1).</p>
    </div>
  </div>
</body>
</html>
"""
    return subject, html_email


def generate_8k_summary_email_html(company_name: str, form_type: str, summary_doc_url: str, cik_number: str, sec_url: str, accession_number: str, summary_kind: str = "8-K") -> tuple:
    """
    Generate HTML email for 8-K summary document notification.

    Args:
        company_name: Name of the company
        form_type: Form type (e.g., 8-K)
        summary_doc_url: URL of the generated summary document
        cik_number: CIK number
        sec_url: URL of the SEC filing
        accession_number: SEC accession number
    Returns:
        tuple: (subject, html_email)
    """
    subject = f"New {summary_kind} Summary Document – {form_type} – {company_name}"

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
      New {summary_kind} Summary Document
    </h2>

    <div style="margin-bottom:30px;">
      <p style="color:#333; font-size:16px; line-height:1.6;">
        The {summary_kind} summary document has been successfully generated for:
      </p>

      <div style="background-color:#f9f9f9; padding:15px; border-radius:5px; margin:20px 0;">
        <p style="margin:8px 0; color:#555;">
          <strong style="color:#333;">Company:</strong> {escape_html(company_name)}
        </p>
        <p style="margin:8px 0; color:#555;">
          <strong style="color:#333;">Form Type:</strong> {escape_html(form_type)}
        </p>
        <p style="margin:8px 0; color:#555;">
          <strong style="color:#333;">CIK Number:</strong> {escape_html(cik_number)}
        </p>
        <p style="margin:8px 0; color:#555;">
          <strong style="color:#333;">Accession Number:</strong> {escape_html(accession_number)}
        </p>
        <p style="margin:8px 0; color:#555;">
          <strong style="color:#333;">SEC URL:</strong> <a href="{escape_html(sec_url)}" style="color:#4a90e2; text-decoration:none;" target="_blank">{escape_html(sec_url)}</a>
        </p>
      </div>
    </div>

    <div style="text-align:center; margin:30px 0;">
      <a href="{escape_html(summary_doc_url)}"
         style="display:inline-block; background-color:#4a90e2; color:#ffffff; padding:15px 30px; text-decoration:none; border-radius:5px; font-size:16px; font-weight:bold; box-shadow:0 2px 4px rgba(0,0,0,0.2);">
        Download Summary Document
      </a>
    </div>

   
    
  </div>
</body>
</html>
"""
    return subject, html_email


def _render_l3_value(key: str, value, level: int) -> str:
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
    <div style="margin-bottom:16px; padding:12px; background-color:#f0f7ff; border-left:4px solid #4a90e2; border-radius:4px; {margin_style}">
      <p style="margin:0 0 6px 0; font-size:12px; font-weight:bold; color:#4a90e2; text-transform:uppercase; letter-spacing:0.5px;">{escape_html(key)}</p>
      <p style="margin:0; font-size:15px; font-weight:bold; color:#003366; line-height:1.5;">{escape_html(s)}</p>
    </div>
"""

    if isinstance(value, list):
        if not value:
            return ""
        # Check if list of strings
        if all(isinstance(item, str) for item in value):
            items_html = "".join(
                f"<li style=\"margin:4px 0; line-height:1.5; font-size:15px; font-weight:bold; color:#003366;\">{escape_html(str(item).strip())}</li>"
                for item in value if str(item).strip()
            )
            if not items_html:
                return ""
            return f"""
    <div style="margin-bottom:16px; padding:12px; background-color:#f0f7ff; border-left:4px solid #4a90e2; border-radius:4px; {margin_style}">
      <p style="margin:0 0 8px 0; font-size:12px; font-weight:bold; color:#4a90e2; text-transform:uppercase; letter-spacing:0.5px;">{escape_html(key)}</p>
      <ul style="margin:0; padding-left:20px;">{items_html}</ul>
    </div>
"""
        # List of objects (dicts): key as label, then recurse for each with indent
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(_render_l3_detailed(item, level + 1))
            else:
                parts.append(_render_l3_value(key, str(item), level))
        if not parts:
            return ""
        key_label = f"""
    <div style="margin-bottom:8px; {margin_style}">
      <p style="margin:0; font-size:12px; font-weight:bold; color:#4a90e2; text-transform:uppercase; letter-spacing:0.5px;">{escape_html(key)}</p>
    </div>
"""
        return key_label + "".join(parts)

    if isinstance(value, dict):
        return _render_l3_detailed(value, level + 1)

    # number, bool, etc.
    return f"""
    <div style="margin-bottom:16px; padding:12px; background-color:#f0f7ff; border-left:4px solid #4a90e2; border-radius:4px; {margin_style}">
      <p style="margin:0 0 6px 0; font-size:12px; font-weight:bold; color:#4a90e2; text-transform:uppercase; letter-spacing:0.5px;">{escape_html(key)}</p>
      <p style="margin:0; font-size:15px; font-weight:bold; color:#003366; line-height:1.5;">{escape_html(str(value))}</p>
    </div>
"""


def _render_l3_detailed(l3_data: dict, level: int = 0) -> str:
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


def generate_8k_99_1_summary_email_html(company_name: str, form_type: str, summary_doc_url: str, cik_number: str, sec_url: str, accession_number: str, summary_kind: str = "8-K", l1_headline: str = None, l2_brief: str = None, l3_detailed=None, ticker: str = None, filing_date=None, matched_cik_label: str = None, form_affects_deal: bool = None) -> tuple:
    """
    Generate HTML email for 8-K summary document notification.

    Args:
        company_name: Name of the company
        form_type: Form type (e.g., 8-K)
        summary_doc_url: URL of the generated summary document
        cik_number: CIK number
        sec_url: URL of the SEC filing
        accession_number: SEC accession number
        summary_kind: Summary type label (e.g. "8-K", "EX-99.1")
        l1_headline: Optional L1 headline from the summary doc (shown so user can see content without opening doc)
        l2_brief: Optional L2 brief from the summary doc (shown so user can see content without opening doc)
        l3_detailed: Optional L3 detailed: dict (key-value by type, recursively rendered) or str (legacy, shown as one block)
        ticker: Optional ticker (from deal); if present, used in subject instead of company_name
        filing_date: Optional filing date for subject (datetime or str, formatted as YYYY-MM-DD)
        matched_cik_label: Optional "(target)" or "(acquirer)" to show beside company name
        form_affects_deal: Optional bool; when True/False (acquirer filing), show "Affects deal: Yes/No"
    Returns:
        tuple: (subject, html_email)
    """
    # Subject: ticker (if from deal) else company_name : form_type Summary : filing_date
    form_type_subject = summary_kind or f"-{form_type}-"
    filing_date_str = ""
    if filing_date is not None:
        if isinstance(filing_date, datetime):
            filing_date_str = filing_date.strftime("%Y-%m-%d")
        else:
            filing_date_str = str(filing_date)[:10] if str(
                filing_date) else ""
    label = (ticker or "").strip() or (company_name or "Unknown")
    subject = f"{label} :Form {form_type_subject} Summary By {company_name} on [ {filing_date_str} ]"

    headline_block = ""
    if l1_headline and l1_headline.strip():
        headline_block = f"""
    <div style="margin-bottom:24px; padding:16px; background-color:#f0f7ff; border-left:4px solid #4a90e2; border-radius:4px;">
      <p style="margin:0 0 6px 0; font-size:12px; font-weight:bold; color:#4a90e2; text-transform:uppercase; letter-spacing:0.5px;">L1 — Headline</p>
      <p style="margin:0; font-size:15px; font-weight:bold; color:#003366; line-height:1.5;">{escape_html(l1_headline.strip())}</p>
    </div>
"""
    brief_block = ""
    if l2_brief and l2_brief.strip():
        brief_block = f"""
    <div style="margin-bottom:24px; padding:16px; background-color:#f0f7ff; border-left:4px solid #4a90e2; border-radius:4px;">
      <p style="margin:0 0 6px 0; font-size:12px; font-weight:bold; color:#4a90e2; text-transform:uppercase; letter-spacing:0.5px;">L2 — Brief</p>
      <p style="margin:0; font-size:15px; font-weight:bold; color:#003366; line-height:1.5;">{escape_html(l2_brief.strip())}</p>
    </div>
"""
    # L3: dict → recursive by type (string / list of strings / list of objects); str → legacy single block
    l3_block = ""
    if l3_detailed is not None:
        if isinstance(l3_detailed, dict):
            inner = (
                '<p style="margin:0 0 8px 0; font-size:12px; font-weight:bold; color:#4a90e2; text-transform:uppercase;">L3 — Detailed</p>'
                + _render_l3_detailed(l3_detailed, 0)
            )
            l3_block = f"""
    <div style="margin-bottom:24px; padding:16px; background-color:#f8fbff; border-left:4px solid #4a90e2; border-radius:4px;">
{inner}
    </div>
"""
        elif isinstance(l3_detailed, str) and l3_detailed.strip():
            l3_block = f"""
    <div style="margin-bottom:24px; padding:16px; background-color:#f0f7ff; border-left:4px solid #4a90e2; border-radius:4px;">
      <p style="margin:0 0 6px 0; font-size:12px; font-weight:bold; color:#4a90e2; text-transform:uppercase; letter-spacing:0.5px;">L3 — Detailed</p>
      <p style="margin:0; font-size:15px; font-weight:bold; color:#003366; line-height:1.5;">{escape_html(l3_detailed.strip())}</p>
    </div>
"""
    form_affects_deal_block = ""
    if form_affects_deal is not None:
        affects_text = "Yes" if form_affects_deal else "No"
        form_affects_deal_block = f"""
        <p style="margin:8px 0; color:#555;">
          <strong style="color:#333;">Affects deal:</strong> {escape_html(affects_text)}
        </p>
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
      New {form_type_subject} Summary Document
    </h2>

    <div style="margin-bottom:30px;">
      <p style="color:#333; font-size:16px; line-height:1.6;">
        The {form_type_subject} summary document has been successfully generated for:
      </p>

      <div style="background-color:#f9f9f9; padding:15px; border-radius:5px; margin:20px 0;">
        <p style="margin:8px 0; color:#555;">
          <strong style="color:#333;">Company:</strong> {escape_html(company_name)}{escape_html((matched_cik_label or "").strip())}
        </p>
        <p style="margin:8px 0; color:#555;">
          <strong style="color:#333;">Form Type:</strong> {escape_html(form_type)}
        </p>
        <p style="margin:8px 0; color:#555;">
          <strong style="color:#333;">CIK Number:</strong> {escape_html(cik_number)}
        </p>
        <p style="margin:8px 0; color:#555;">
          <strong style="color:#333;">Accession Number:</strong> {escape_html(accession_number)}
        </p>
        <p style="margin:8px 0; color:#555;">
          <strong style="color:#333;">SEC URL:</strong> <a href="{escape_html(sec_url)}" style="color:#4a90e2; text-decoration:none;" target="_blank">{escape_html(sec_url)}</a>
        </p>
{form_affects_deal_block}
      </div>
    </div>
{headline_block}{brief_block}{l3_block}
    <div style="text-align:center; margin:30px 0;">
      <a href="{escape_html(summary_doc_url)}"
         style="display:inline-block; background-color:#4a90e2; color:#ffffff; padding:15px 30px; text-decoration:none; border-radius:5px; font-size:16px; font-weight:bold; box-shadow:0 2px 4px rgba(0,0,0,0.2);">
        Download Summary Document
      </a>
    </div>

   
    
  </div>
</body>
</html>
"""
    return subject, html_email


def build_sec_filings_table(filings):
    """
    Build HTML table for SEC form filings list (e.g. from sec_Last_Year.print_filings).
    Each item: filing_date, form, accession_number, primary_document, url.
    """
    if not filings or len(filings) == 0:
        return "<p><em>No SEC filings found in the period.</em></p>"

    rows = []
    for idx, f in enumerate(filings):
        bg = "#ffffff" if idx % 2 == 0 else "#f9f9f9"
        filing_date = escape_html(f.get('filing_date', ''))
        form = escape_html(f.get('form', ''))
        accession = escape_html(f.get('accession_number', ''))
        primary_doc = escape_html(f.get('primary_document', ''))
        url = f.get('url', '')
        if url:
            link_html = f'<a href="{escape_html(url)}" style="color:#4a90e2; text-decoration:none;" target="_blank">{primary_doc or "Link"}</a>'
        else:
            link_html = primary_doc

        rows.append(f"""
      <tr style="background-color:{bg};">
        <td style="padding:8px; border:1px solid #ddd;">{filing_date}</td>
        <td style="padding:8px; border:1px solid #ddd;">{form}</td>
        <td style="padding:8px; border:1px solid #ddd;">{link_html}</td>
      </tr>
""")
    rows_html = "".join(rows)
    return f"""
    <table style="width:100%; border-collapse:collapse; margin-top:10px;">
      <thead>
        <tr style="background-color:#f5f5f5;">
          <th style="padding:8px; border:1px solid #ddd; text-align:left;">Filing Date</th>
          <th style="padding:8px; border:1px solid #ddd; text-align:left;">Form</th>
         
          <th style="padding:8px; border:1px solid #ddd; text-align:left;">Document</th>
        </tr>
      </thead>
      <tbody>
{rows_html}
      </tbody>
    </table>
"""


def generate_sec_filings_email_html(company_name, filings, form_type):
    """Generate full email HTML for SEC form filings (last year) table. Returns (subject, html).
    form_type identifies which email is for which (e.g. '8-K(EX-2.1)', '10-K', '10-Q').
    Subject: for 8-K(EX-2.1) -> company name : form_type All One Year Forms.;
    for 10-K/10-Q -> company name : form_type All 10k/10q since announcing date."""
    form_type_esc = escape_html(form_type)
    company_esc = escape_html(company_name or "Unknown Company")
    if form_type and "8-K" in str(form_type):
        subject = f"{company_esc} : {form_type_esc} All One Year Forms."
    elif form_type and str(form_type).upper() in ("10-K", "10-Q"):
        subject = f"{company_esc} : {form_type_esc} All 10k/10q since announcing date."
    else:
        subject = f"SEC Form Filings ({form_type_esc}) – {company_esc}"
    table_html = build_sec_filings_table(filings)
    html_email = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>SEC Filings</title></head>
<body style="font-family: Arial, sans-serif; margin: 20px;">
  <div style="max-width:900px;">
    <h2 style="color:#333;">SEC Form Filings – {escape_html(form_type)}</h2>
    <p style="color:#555;">Company: <strong>{escape_html(company_name)}</strong></p>
    {table_html}
  </div>
</body>
</html>
"""
    return subject, html_email


def generate_item_5_02_one_year_filings_email_html(company_name, filings, trigger_accession_number=None, trigger_filing_date=None, cik_number=None, ticker=None):
    """Generate email HTML for 8-K Item 5.02 trigger: one-year SEC filings for the company.
    Separate format from generate_sec_filings_email_html. Returns (subject, html)."""
    company_esc = escape_html(company_name or "Unknown Company")
    # Subject: same pattern as generate_8k_99_1_summary_email_html — {label} :Form {form_type_subject} Item 5.02 All Filing of last one year By {company_name}
    form_type_subject = "8-K"
    label = (ticker or "").strip() or (company_name or "Unknown")
    label_esc = escape_html(label)
    subject = f"{label_esc} :Form {form_type_subject} (Item 5.02) All Filing Of Last One Year By {company_esc}"
    table_html = build_sec_filings_table(filings)
    cik_display = (str(cik_number).zfill(10) if cik_number else "").strip()
    intro_parts = [
        f"<p style='color:#555;'>Company: <strong>{company_esc}</strong></p>",
    ]
    if cik_display:
        intro_parts.append(
            f"<p style='color:#555;'>CIK: <strong>{escape_html(cik_display)}</strong></p>"
        )

    if trigger_accession_number or trigger_filing_date:
        intro_parts.append(
            f"<p style='color:#666; font-size:0.9em;'>Triggering filing: {escape_html(trigger_accession_number or '')} {('filed ' + escape_html(str(trigger_filing_date))) if trigger_filing_date else ''}</p>"
        )
    intro_html = "\n    ".join(intro_parts)
    html_email = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>8-K (Item 5.02) – One Year SEC Filings</title></head>
<body style="font-family: Arial, sans-serif; margin: 20px;">
  <div style="max-width:900px;">
    <h2 style="color:#333;">8-K (Item 5.02) – One Year SEC Filings</h2>
    {intro_html}
    {table_html}
  </div>
</body>
</html>
"""
    return subject, html_email


def generate_10k_10q_comparison_summary_email_html(
    ticker: str,
    target_company: str,
    filing_labels: list,
    s3_comparison_json_url: str,
    s3_exec_summary_docx_url: str,
    s3_redline_docx_url: str = None,
    s3_client_report_docx_url: str = None,
):
    """Generate email HTML for 10-K/10-Q comparison final summary with JSON and DOCX links.
    Returns (subject, html). Used after orchestrator comparison run."""
    company_esc = escape_html(target_company or ticker or "Unknown Company")
    ticker_esc = escape_html(ticker or "")
    subject = f"{company_esc} : 10-K/10-Q Comparison Summary – {ticker_esc}" if ticker_esc else f"{company_esc} : 10-K/10-Q Comparison Summary"

    labels_line = ", ".join(escape_html(l or "")
                            for l in (filing_labels or [])[:10])
    if filing_labels and len(filing_labels) > 10:
        labels_line += " …"

    links = []
    if s3_exec_summary_docx_url:
        links.append(("Executive Summary (DOCX)", s3_exec_summary_docx_url))
    if s3_comparison_json_url:
        links.append(("Comparison data (JSON)", s3_comparison_json_url))
    if s3_redline_docx_url:
        links.append(("Redline report (DOCX)", s3_redline_docx_url))
    if s3_client_report_docx_url:
        links.append(("Client report (DOCX)", s3_client_report_docx_url))

    rows_html = "".join(
        f'<tr><td style="padding:8px; border:1px solid #ddd;"><a href="{escape_html(url)}" style="color:#4a90e2;" target="_blank">{escape_html(label)}</a></td></tr>'
        for label, url in links
    )
    links_table = f"""
    <table style="width:100%; border-collapse:collapse; margin-top:10px;">
      <thead><tr style="background-color:#f5f5f5;"><th style="padding:8px; border:1px solid #ddd; text-align:left;">Document</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
""" if rows_html else "<p><em>No links available.</em></p>"

    html_email = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>10-K/10-Q Comparison Summary</title></head>
<body style="font-family: Arial, sans-serif; margin: 20px;">
  <div style="max-width:900px;">
    <h2 style="color:#333;">10-K/10-Q Comparison Summary</h2>
    <p style="color:#555;">Company: <strong>{company_esc}</strong></p>
    <p style="color:#555;">Filings compared: <strong>{labels_line}</strong></p>
    <p style="color:#555;">Links to final summary and comparison data:</p>
    {links_table}
  </div>
</body>
</html>
"""
    return subject, html_email


def generate_proxy_comparison_summary_email_html(
    company_name: str,
    form_type: str,
    ticker: str,
    label: str,
    deal_id: str = None,
    cik_number: str = None,
    past_record_id: str = None,
    latest_record_id: str = None,
    change_docx_url: str = None,
    change_txt_url: str = None,
    changes_json_url: str = None,
    tier1_changes: int = None,
    tier2_changes: int = None,
):
    """Generate email HTML for proxy comparison (DEFM14A, S-4/A, etc.) with change report links.
    Returns (subject, html). Used after proxy_comparision orchestrator run_comparison."""
    company_esc = escape_html(company_name or "Unknown Company")
    ticker_esc = escape_html(ticker or "")
    form_esc = escape_html(form_type or "PROXY")
    label_esc = escape_html(label or "")
    subject = f"{label_esc} : ({form_esc}) - Proxy Comparison"

    changes_line = ""
    if tier1_changes is not None or tier2_changes is not None:
        t1 = tier1_changes if tier1_changes is not None else 0
        t2 = tier2_changes if tier2_changes is not None else 0
        changes_line = f"<p style='color:#555;'>Tier 1 changes: <strong>{t1}</strong> | Tier 2 changes: <strong>{t2}</strong></p>"

    links = []
    if change_docx_url:
        links.append(("Change report (DOCX)", change_docx_url))
    if change_txt_url:
        links.append(("Change report (TXT)", change_txt_url))
    if changes_json_url:
        links.append(("Changes data (JSON)", changes_json_url))

    rows_html = "".join(
        f'<tr><td style="padding:8px; border:1px solid #ddd;"><a href="{escape_html(url)}" style="color:#4a90e2;" target="_blank">{escape_html(label)}</a></td></tr>'
        for label, url in links
    )
    links_table = (
        f"""
    <table style="width:100%; border-collapse:collapse; margin-top:10px;">
      <thead><tr style="background-color:#f5f5f5;"><th style="padding:8px; border:1px solid #ddd; text-align:left;">Document</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
"""
        if rows_html
        else "<p><em>No links available.</em></p>"
    )

    html_email = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Proxy Comparison Summary</title></head>
<body style="font-family: Arial, sans-serif; margin: 20px;">
  <div style="max-width:900px;">
    <h2 style="color:#333;">Proxy Comparison Summary</h2>
    <p style="color:#555;">Company: <strong>{company_esc}</strong></p>
    <p style="color:#555;">Form type: <strong>{form_esc}</strong></p>
    <p style="color:#555;">Deal ID: <strong>{escape_html(deal_id or "")}</strong></p>
    <p style="color:#555;">CIK Number: <strong>{escape_html(cik_number or "")}</strong></p>
    <p style="color:#555;">Latest record ID: <strong>{escape_html(latest_record_id or "")}</strong></p>
    <p style="color:#555;">Past record ID: <strong>{escape_html(past_record_id or "")}</strong></p>
    {changes_line}
    <p style="color:#555;">Links to change report and comparison data:</p>
    {links_table}
  </div>
</body>
</html>
"""
    return subject, html_email


def generate_parsing_error_email_html(
    company_name: str,
    form_type: str,
    sec_filing_id: str,
    sec_url: str,
    accession_number: str,
    error_message: str,
    log_records: list,
):
    """
    Generate an email HTML payload for parsing/extraction failures.

    The main goal is to include the captured `log_records` (WARNING+ from the extraction worker)
    so you can debug without needing to access logs/files.
    """
    company_esc = escape_html(company_name or "Unknown Company")
    form_esc = escape_html(form_type or "8-K")
    accession_esc = escape_html(accession_number or "")
    sec_filing_id_esc = escape_html(sec_filing_id or "")
    sec_url_esc = escape_html(sec_url or "")

    error_esc = escape_html(error_message or "Unknown error")

    logs = log_records or []
    if isinstance(logs, str):
        logs = [logs]

    # Keep the email reasonably small; N8N/webhook payload size can be limited.
    max_log_lines = 200
    max_log_chars = 20000

    logs_str = "\n".join(str(x) for x in logs)
    truncated = False
    if len(logs) > max_log_lines:
        truncated = True
        logs_str = "\n".join(str(x) for x in logs[:max_log_lines])

    if len(logs_str) > max_log_chars:
        truncated = True
        logs_str = logs_str[:max_log_chars] + "\n...[truncated]"

    logs_count = len(logs)
    logs_esc = escape_html(logs_str)

    subject = f"❌ Parsing Error - {company_esc} ({form_esc})"

    html_email = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Parsing Error</title>
</head>
<body style="margin:0; padding:0; font-family:Arial,sans-serif; background-color:#f4f4f4;">
  <div style="max-width:900px; margin:20px auto; background-color:#ffffff; padding:30px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <h2 style="color:#333; text-align:center; margin-top:0; padding-bottom:20px; border-bottom:3px solid #dc3545;">
      Parsing / Extraction Error
    </h2>

    <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; width:170px; color:#555;">Company</td>
        <td style="padding:8px; color:#333;">{company_esc}</td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Form Type</td>
        <td style="padding:8px; color:#333;">{form_esc}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">SEC Filing ID</td>
        <td style="padding:8px; color:#333;">{sec_filing_id_esc}</td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Accession Number</td>
        <td style="padding:8px; color:#333;">{accession_esc}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">SEC URL</td>
        <td style="padding:8px; color:#333;">
          <a href="{sec_url_esc}" target="_blank" style="color:#4a90e2; text-decoration:none;">{sec_url_esc if sec_url_esc else "Link"}</a>
        </td>
      </tr>
    </table>

    <h3 style="color:#333; margin-top:20px; margin-bottom:10px;">Error</h3>
    <pre style="white-space:pre-wrap; word-break:break-word; background:#fff7f7; border:1px solid #f1c0c0; padding:12px; border-radius:6px; font-size:12px;">{error_esc}</pre>

    <h3 style="color:#333; margin-top:20px; margin-bottom:10px;">
      log_records / warnings
      <span style="color:#888; font-weight:normal;">({logs_count} entries{', truncated' if truncated else ''})</span>
    </h3>
    <pre style="white-space:pre-wrap; word-break:break-word; background:#f7f7f7; border:1px solid #e6e6e6; padding:12px; border-radius:6px; font-size:12px;">{logs_esc if logs_esc else "No log_records available."}</pre>

    <div style="margin-top:22px; padding-top:18px; border-top:1px solid #e0e0e0; text-align:center; color:#999; font-size:12px;">
      Sent via N8N webhook for debugging parsing failures.
    </div>
  </div>
</body>
</html>
"""
    return subject, html_email


def generate_parsing_success_email_html(
    company_name: str,
    form_type: str,
    sec_filing_id: str,
    sec_url: str,
    accession_number: str,
    parsed_json_url: str,
    deal_id: str,
    log_records: list,
):
    """
    Generate an email HTML payload for parsing/extraction success.

    Includes the same `log_records` list (warnings/errors captured during extraction)
    to make it easy to debug borderline extractions that still "succeed".
    """
    company_esc = escape_html(company_name or "Unknown Company")
    form_esc = escape_html(form_type or "8-K")
    accession_esc = escape_html(accession_number or "")
    sec_filing_id_esc = escape_html(sec_filing_id or "")
    sec_url_esc = escape_html(sec_url or "")
    parsed_json_url_esc = escape_html(parsed_json_url or "")
    deal_id_esc = escape_html(deal_id or "")

    logs = log_records or []
    if isinstance(logs, str):
        logs = [logs]

    # Keep the email reasonably small; N8N/webhook payload size can be limited.
    max_log_lines = 200
    max_log_chars = 20000

    logs_str = "\n".join(str(x) for x in logs)
    truncated = False
    if len(logs) > max_log_lines:
        truncated = True
        logs_str = "\n".join(str(x) for x in logs[:max_log_lines])

    if len(logs_str) > max_log_chars:
        truncated = True
        logs_str = logs_str[:max_log_chars] + "\n...[truncated]"

    logs_count = len(logs)
    logs_esc = escape_html(logs_str)

    subject = f"✅ Parsing Success - {company_esc} ({form_esc})"

    html_email = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Parsing Success</title>
</head>
<body style="margin:0; padding:0; font-family:Arial,sans-serif; background-color:#f4f4f4;">
  <div style="max-width:900px; margin:20px auto; background-color:#ffffff; padding:30px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <h2 style="color:#333; text-align:center; margin-top:0; padding-bottom:20px; border-bottom:3px solid #28a745;">
      Parsing / Extraction Success
    </h2>

    <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; width:170px; color:#555;">Company</td>
        <td style="padding:8px; color:#333;">{company_esc}</td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Form Type</td>
        <td style="padding:8px; color:#333;">{form_esc}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">SEC Filing ID</td>
        <td style="padding:8px; color:#333;">{sec_filing_id_esc}</td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Deal ID</td>
        <td style="padding:8px; color:#333;">{deal_id_esc}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">Accession Number</td>
        <td style="padding:8px; color:#333;">{accession_esc}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">SEC URL</td>
        <td style="padding:8px; color:#333;">
          <a href="{sec_url_esc}" target="_blank" style="color:#4a90e2; text-decoration:none;">{sec_url_esc if sec_url_esc else "Link"}</a>
        </td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Parsed JSON (S3)</td>
        <td style="padding:8px; color:#333;">
          <a href="{parsed_json_url_esc}" target="_blank" style="color:#4a90e2; text-decoration:none;">{parsed_json_url_esc if parsed_json_url_esc else "Link"}</a>
        </td>
      </tr>
    </table>

    <h3 style="color:#333; margin-top:20px; margin-bottom:10px;">log_records / warnings
      <span style="color:#888; font-weight:normal;">({logs_count} entries{', truncated' if truncated else ''})</span>
    </h3>
    <pre style="white-space:pre-wrap; word-break:break-word; background:#f7f7f7; border:1px solid #e6e6e6; padding:12px; border-radius:6px; font-size:12px;">{logs_esc if logs_esc else "No log_records available."}</pre>

    <div style="margin-top:22px; padding-top:18px; border-top:1px solid #e0e0e0; text-align:center; color:#999; font-size:12px;">
      Sent via N8N webhook for debugging parsing success.
    </div>
  </div>
</body>
</html>
"""
    return subject, html_email


def _build_extracted_data_table(extracted: dict, title: str = "Extracted Data") -> str:
    """Build HTML table for extracted data fields from press release or DMA extraction."""
    if not extracted or not isinstance(extracted, dict):
        return "<p><em>No extracted data available.</em></p>"

    rows = []
    idx = 0
    for key, value in extracted.items():
        if value is None:
            display_value = "—"
        elif isinstance(value, list):
            if not value:
                display_value = "—"
            else:
                items = [escape_html(str(v)) for v in value if v is not None]
                display_value = "<br>".join(items) if items else "—"
        elif isinstance(value, bool):
            display_value = "Yes" if value else "No"
        elif isinstance(value, (int, float)):
            display_value = escape_html(str(value))
        else:
            display_value = escape_html(str(value))

        bg = "#ffffff" if idx % 2 == 0 else "#f9f9f9"
        key_display = key.replace("_", " ").title()
        rows.append(f"""
      <tr style="background-color:{bg};">
        <td style="padding:10px; border:1px solid #ddd; font-weight:bold; color:#555; width:220px;">{escape_html(key_display)}</td>
        <td style="padding:10px; border:1px solid #ddd; color:#333;">{display_value}</td>
      </tr>
""")
        idx += 1

    rows_html = "".join(rows)
    return f"""
    <h3 style="color:#333; margin-top:20px; margin-bottom:10px;">{escape_html(title)}</h3>
    <table style="width:100%; border-collapse:collapse; margin-top:10px;">
      <tbody>
{rows_html}
      </tbody>
    </table>
"""


def _build_inconsistencies_table(inconsistencies: list) -> str:
    """Build HTML table for DMA vs Press Release inconsistencies."""
    if not inconsistencies:
        return ""

    rows = []
    for idx, item in enumerate(inconsistencies):
        bg = "#ffffff" if idx % 2 == 0 else "#fff3cd"
        field = escape_html(item.get('label', item.get('field', '')))
        dma_val = escape_html(item.get('dma_value', ''))
        pr_val = escape_html(item.get('pr_value', ''))
        rows.append(f"""
      <tr style="background-color:{bg};">
        <td style="padding:10px; border:1px solid #ddd; font-weight:bold; color:#555;">{field}</td>
        <td style="padding:10px; border:1px solid #ddd; color:#dc3545;">{dma_val}</td>
        <td style="padding:10px; border:1px solid #ddd; color:#28a745;">{pr_val}</td>
      </tr>
""")

    rows_html = "".join(rows)
    return f"""
    <h3 style="color:#dc3545; margin-top:30px; margin-bottom:10px;">⚠️ Inconsistencies (DMA vs Press Release)</h3>
    <table style="width:100%; border-collapse:collapse; margin-top:10px;">
      <thead>
        <tr style="background-color:#fff3cd;">
          <th style="padding:10px; border:1px solid #ddd; text-align:left;">Field</th>
          <th style="padding:10px; border:1px solid #ddd; text-align:left;">DMA Value</th>
          <th style="padding:10px; border:1px solid #ddd; text-align:left;">PR Value</th>
        </tr>
      </thead>
      <tbody>
{rows_html}
      </tbody>
    </table>
"""


def generate_press_release_extraction_email_html(
    company_name: str,
    deal_id: str,
    accession_number: str,
    extracted: dict,
    filing_date: str = None,
    cik_number: str = None,
    press_release_docx: str = None,
) -> tuple:
    """
    Generate HTML email for Press Release (EX-99.1) extraction notification.

    Args:
        company_name: Name of the company
        deal_id: Deal ID
        accession_number: SEC accession number
        extracted: Extracted structured data dict
        filing_date: Filing date
        cik_number: CIK number
        press_release_docx: S3 URL of the press release summary DOCX

    Returns:
        tuple: (subject, html_email)
    """
    company_esc = escape_html(company_name or "Unknown Company")
    deal_id_esc = escape_html(deal_id or "N/A")
    accession_esc = escape_html(accession_number or "N/A")
    filing_date_esc = escape_html(filing_date or "N/A")
    cik_esc = escape_html(cik_number or "N/A")

    subject = f"Press Release Extraction – {company_esc}"

    extracted_table = _build_extracted_data_table(extracted, "Extracted Deal Financial Data")

    docx_link_html = ""
    if press_release_docx:
        docx_link_html = f"""
    <div style="text-align:center; margin:20px 0;">
      <a href="{escape_html(press_release_docx)}"
         style="display:inline-block; background-color:#4a90e2; color:#ffffff; padding:12px 24px; text-decoration:none; border-radius:5px; font-size:14px; font-weight:bold;">
        Download Press Release Summary
      </a>
    </div>
"""

    html_email = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{escape_html(subject)}</title>
</head>
<body style="margin:0; padding:0; font-family:Arial,sans-serif; background-color:#f4f4f4;">
  <div style="max-width:900px; margin:20px auto; background-color:#ffffff; padding:30px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <h2 style="color:#333; text-align:center; margin-top:0; padding-bottom:20px; border-bottom:3px solid #28a745;">
      Press Release Extraction (EX-99.1)
    </h2>

    <div style="background-color:#d4edda; padding:15px; border-radius:5px; margin:20px 0; border-left:4px solid #28a745;">
      <p style="margin:0; color:#155724; font-size:14px;">
        <strong>Structured deal financial data has been extracted from the EX-99.1 Press Release summary.</strong>
      </p>
    </div>

    <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
      <tr style="background-color:#f9f9f9;">
        <td style="padding:10px; font-weight:bold; width:170px; color:#555;">Deal ID:</td>
        <td style="padding:10px; color:#333; font-family:monospace;">{deal_id_esc}</td>
      </tr>
      <tr>
        <td style="padding:10px; font-weight:bold; color:#555;">Company:</td>
        <td style="padding:10px; color:#333;">{company_esc}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:10px; font-weight:bold; color:#555;">Accession Number:</td>
        <td style="padding:10px; color:#333;">{accession_esc}</td>
      </tr>
      <tr>
        <td style="padding:10px; font-weight:bold; color:#555;">CIK:</td>
        <td style="padding:10px; color:#333;">{cik_esc}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:10px; font-weight:bold; color:#555;">Filing Date:</td>
        <td style="padding:10px; color:#333;">{filing_date_esc}</td>
      </tr>
    </table>

{extracted_table}
{docx_link_html}

    <div style="margin-top:30px; padding-top:20px; border-top:1px solid #e0e0e0; text-align:center; color:#999; font-size:12px;">
      <p>This is an automated extraction from EX-99.1 (Press Release) summary using Claude Haiku.</p>
    </div>
  </div>
</body>
</html>
"""
    return subject, html_email


def generate_dma_extraction_email_html(
    company_name: str,
    deal_id: str,
    accession_number: str,
    extracted: dict,
    inconsistencies: list = None,
    filing_date: str = None,
    cik_number: str = None,
    dma_summary_docx: str = None,
) -> tuple:
    """
    Generate HTML email for DMA (EX-2.1 Definitive Merger Agreement) extraction notification.

    Args:
        company_name: Name of the company
        deal_id: Deal ID
        accession_number: SEC accession number
        extracted: Extracted structured data dict
        inconsistencies: List of inconsistencies between DMA and Press Release
        filing_date: Filing date
        cik_number: CIK number
        dma_summary_docx: S3 URL of the DMA summary DOCX

    Returns:
        tuple: (subject, html_email)
    """
    company_esc = escape_html(company_name or "Unknown Company")
    deal_id_esc = escape_html(deal_id or "N/A")
    accession_esc = escape_html(accession_number or "N/A")
    filing_date_esc = escape_html(filing_date or "N/A")
    cik_esc = escape_html(cik_number or "N/A")

    subject = f"DMA Extraction – {company_esc}"

    extracted_table = _build_extracted_data_table(extracted, "Extracted DMA Data")
    inconsistencies_table = _build_inconsistencies_table(inconsistencies or [])

    docx_link_html = ""
    if dma_summary_docx:
        docx_link_html = f"""
    <div style="text-align:center; margin:20px 0;">
      <a href="{escape_html(dma_summary_docx)}"
         style="display:inline-block; background-color:#4a90e2; color:#ffffff; padding:12px 24px; text-decoration:none; border-radius:5px; font-size:14px; font-weight:bold;">
        Download DMA Summary
      </a>
    </div>
"""

    inconsistency_alert = ""
    if inconsistencies:
        inconsistency_alert = f"""
    <div style="background-color:#fff3cd; padding:15px; border-radius:5px; margin:20px 0; border-left:4px solid #ffc107;">
      <p style="margin:0; color:#856404; font-size:14px;">
        <strong>⚠️ {len(inconsistencies)} inconsistency(ies) detected between DMA and Press Release data.</strong>
      </p>
    </div>
"""

    html_email = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{escape_html(subject)}</title>
</head>
<body style="margin:0; padding:0; font-family:Arial,sans-serif; background-color:#f4f4f4;">
  <div style="max-width:900px; margin:20px auto; background-color:#ffffff; padding:30px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <h2 style="color:#333; text-align:center; margin-top:0; padding-bottom:20px; border-bottom:3px solid #4a90e2;">
      DMA Extraction (EX-2.1 Definitive Merger Agreement)
    </h2>

    <div style="background-color:#e7f3ff; padding:15px; border-radius:5px; margin:20px 0; border-left:4px solid #4a90e2;">
      <p style="margin:0; color:#004085; font-size:14px;">
        <strong>Structured deal data has been extracted from the EX-2.1 DMA summary.</strong>
      </p>
    </div>

{inconsistency_alert}

    <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
      <tr style="background-color:#f9f9f9;">
        <td style="padding:10px; font-weight:bold; width:170px; color:#555;">Deal ID:</td>
        <td style="padding:10px; color:#333; font-family:monospace;">{deal_id_esc}</td>
      </tr>
      <tr>
        <td style="padding:10px; font-weight:bold; color:#555;">Company:</td>
        <td style="padding:10px; color:#333;">{company_esc}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:10px; font-weight:bold; color:#555;">Accession Number:</td>
        <td style="padding:10px; color:#333;">{accession_esc}</td>
      </tr>
      <tr>
        <td style="padding:10px; font-weight:bold; color:#555;">CIK:</td>
        <td style="padding:10px; color:#333;">{cik_esc}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:10px; font-weight:bold; color:#555;">Filing Date:</td>
        <td style="padding:10px; color:#333;">{filing_date_esc}</td>
      </tr>
    </table>

{extracted_table}
{inconsistencies_table}
{docx_link_html}

    <div style="margin-top:30px; padding-top:20px; border-top:1px solid #e0e0e0; text-align:center; color:#999; font-size:12px;">
      <p>This is an automated extraction from EX-2.1 (DMA) summary using Claude Haiku.</p>
    </div>
  </div>
</body>
</html>
"""
    return subject, html_email
