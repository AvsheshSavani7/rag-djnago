"""
Quick test: parse proxy summary DOCX from S3, generate email HTML, save locally.
"""
from proxy_processor.proxy_docx_parser import parse_proxy_summary_docx
import sys
import os
import urllib.parse

sys.path.insert(0, os.path.dirname(__file__))


DOCX_URL = "https://rag-mna-doc.s3.amazonaws.com/proxy-summaries/943880ae-123b-468e-96fa-b36fad5010b9_summary.docx"
OUTPUT_FILE = os.path.join(os.path.dirname(
    __file__), "test_proxy_email_preview.html")


def escape_html(text):
    if text is None:
        return ""
    text = str(text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace('"', "&quot;").replace("'", "&#039;")
    return text


print(f"Fetching and parsing DOCX from:\n  {DOCX_URL}\n")
parsed = parse_proxy_summary_docx(DOCX_URL)
qa_items = parsed["qa_items"]
chronological_summary = parsed["chronological_summary"]

print(f"Parsed {len(qa_items)} Q&A items:")
for item in qa_items:
    print(f"  [{item['question_key']}] Q: {item['question'][:70]}...")
    print(f"           A: {item['answer'][:90]}...")

print(f"\nChronological Summary: {len(chronological_summary)} paragraph(s)")
for line in chronological_summary[:3]:
    print(f"  - {line[:90]}...")

# Build inline Q&A + Chronological HTML block
rows = ""
for item in qa_items:
    question = escape_html(item.get("question", ""))
    answer = escape_html(item.get("answer", ""))
    rows += f"""
      <div style="margin-bottom:20px;">
        <p style="margin:0 0 6px 0; font-size:14px; font-weight:bold; color:#4a90e2;">{question}</p>
        <p style="margin:0 0 0 14px; font-size:13px; line-height:1.6; color:#333;">
          <span style="font-weight:bold; margin-right:6px; color:#333;">+</span>{answer}
        </p>
      </div>"""

chrono_html = ""
if chronological_summary:
    chrono_rows = "".join(
        f'<p style="margin:0 0 0 14px; font-size:13px; line-height:1.6; color:#333;">'
        f'<span style="font-weight:bold; margin-right:6px; color:#333;">+</span>{escape_html(line)}</p>'
        for line in chronological_summary
    )
    chrono_html = f"""
      <div style="margin-top:24px; padding-top:18px; border-top:1px solid #e8e8e8;">
        <p style="margin:0 0 12px 0; font-size:14px; font-weight:bold; color:#4a90e2;">
          Chronological Summary
        </p>
        {chrono_rows}
      </div>"""

inline_html = f"""
    <div style="margin:30px 0; border-top:2px solid #e0e0e0; padding-top:20px;">
      <p style="margin:0 0 16px 0; font-size:15px; font-weight:bold; color:#333; text-decoration:underline;">
        Proxy Summary
      </p>
      {rows}
      {chrono_html}
    </div>"""

company_name = "Modiv Industrial, Inc."
form_type = "DEFM14A"
cik_number = "0001234567"
proxy_sec_url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001234567"
subject = f"New Proxy Summary Document – {form_type} – {company_name}"

html_email = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{escape_html(subject)}</title>
</head>
<body style="margin:0; padding:0; font-family:Arial,sans-serif; background-color:#f4f4f4;">
  <div style="max-width:700px; margin:20px auto; background-color:#ffffff; padding:30px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <h2 style="color:#333; text-align:center; margin-top:0; padding-bottom:20px; border-bottom:3px solid #4a90e2;">
      New Proxy Summary Document
    </h2>
    <div style="margin-bottom:30px;">
      <p style="color:#333; font-size:16px; line-height:1.6;">The proxy summary document has been successfully generated for:</p>
      <div style="background-color:#f9f9f9; padding:15px; border-radius:5px; margin:20px 0;">
        <p style="margin:8px 0; color:#555;"><strong style="color:#333;">Company:</strong> {escape_html(company_name)}</p>
        <p style="margin:8px 0; color:#555;"><strong style="color:#333;">Form Type:</strong> {escape_html(form_type)}</p>
        <p style="margin:8px 0; color:#555;"><strong style="color:#333;">CIK Number:</strong> {escape_html(cik_number)}</p>
        <p style="margin:8px 0; color:#555;"><strong style="color:#333;">Proxy SEC URL:</strong> {escape_html(proxy_sec_url)}</p>
      </div>
    </div>
    {inline_html}
    <div style="text-align:center; margin:30px 0;">
      <a href="{escape_html(DOCX_URL)}"
         style="display:inline-block; background-color:#4a90e2; color:#ffffff; padding:15px 30px; text-decoration:none; border-radius:5px; font-size:16px; font-weight:bold; box-shadow:0 2px 4px rgba(0,0,0,0.2);"
         target="_blank">&#11015; Download Summary Document</a>
    </div>
    <div style="margin-top:30px; padding:15px; background-color:#e8f4f8; border-radius:5px; border-left:4px solid #4a90e2;">
      <p style="margin:0; color:#555; font-size:14px;"><strong>Note:</strong> This document contains the merger background analysis and Q&amp;A summary generated from the proxy statement.</p>
    </div>
  </div>
</body>
</html>"""

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(html_email)

print(f"\nHTML saved to: {OUTPUT_FILE}")
