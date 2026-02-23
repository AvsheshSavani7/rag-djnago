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


def build_doc_files_table(doc_files):
    """
    Build HTML table for document format files.
    Shared by filing and EX-99.1 merger emails.
    """
    if not doc_files or len(doc_files) == 0:
        return "<p><em>No Document Format Files found.</em></p>"

    rows = []
    for idx, file in enumerate(doc_files):
        bg = "#ffffff" if idx % 2 == 0 else "#f9f9f9"
        seq = escape_html(file.get('sequence', ''))
        description = escape_html(file.get('description', ''))
        doc_name = escape_html(file.get('file', ''))
        doc_type = escape_html(file.get('type', ''))
        size = escape_html(file.get('size', ''))

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


def generate_ex99_1_merger_email_html(filing_data, doc_files):
    """
    Generate HTML email for 8-K EX-99.1 merger-related notification.
    Used when is_merger_related=True (EX-99.1 press release identified as M&A-related).
    Different from EX-2.1 Definitive Merger Agreement email.
    """
    form_type = filing_data.get('form_type', '8-K')
    company_name = filing_data.get('company_name', 'Unknown Company')
    accession_no = filing_data.get('accession_number', 'N/A')
    filing_date = filing_data.get('filing_date', 'N/A')
    if isinstance(filing_date, datetime):
        filing_date = filing_date.strftime('%Y-%m-%d')
    cik = filing_data.get('cik_number', 'N/A')
    filing_url = filing_data.get('link', '')
    confidence = filing_data.get('ex99_1_confidence', 0)
    is_merger_related = filing_data.get('is_merger_related', False)
    reasoning = filing_data.get('ex99_1_reasoning', '')
    is_target_us_listed = filing_data.get('is_target_us_listed')
    is_target_market_cap_greater_than_100m = filing_data.get(
        'is_target_market_cap_greater_than_100m')

    def _fmt_bool(val):
        if val is None:
            return 'N/A'
        return 'Yes' if val else 'No'

    subject = f"8-K EX-99.1 M&A-Related – {form_type} – {company_name}"
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
    form_type identifies which email is for which (e.g. '8-K(EX-2.1)', '10-K', '10-Q')."""
    subject = f"SEC Form Filings ({escape_html(form_type)}) – {escape_html(company_name)}"
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
