"""
S-8 Registration Statement Summarizer — Multi-level summaries via Claude API
Usage: python s8_summary.py
"""

import anthropic
import re
import json
import sys
import os
import io
from pathlib import Path
from ._naming import filing_uid, sanitize_date_part, sanitize_filename_part
from ._deal_context import inject_deal_context

# ──── PASTE YOUR S-8 URL HERE ────
FILING_URL = ""
DEAL_CONTEXT = None
# ──── OUTPUT FOLDER ────
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "Output Summaries"
# ─────────────────────────────────


try:
    import requests
    from bs4 import BeautifulSoup
    from dotenv import load_dotenv
    from docx import Document as DocxDocument
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                          "requests", "beautifulsoup4", "python-dotenv", "python-docx", "-q"])
    import requests
    from bs4 import BeautifulSoup
    from dotenv import load_dotenv
    from docx import Document as DocxDocument
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

try:
    from ._config import get_anthropic_api_key
except ImportError:
    from _config import get_anthropic_api_key

ANTHROPIC_API_KEY = get_anthropic_api_key()
if not ANTHROPIC_API_KEY and __name__ == "__main__":
    print("❌ ANTHROPIC_API_KEY not found. Set it in .env or Django settings (ANTHROPIC_API_KEY).")
    sys.exit(1)


SUMMARY_PROMPT = """You are an expert analyst summarizing SEC Form S-8 registration statements for a merger arbitrage desk.

Form S-8 registers securities to be offered to employees under benefit plans (stock option plans, employee stock purchase plans, 401(k) plans, restricted stock units, etc.). In an M&A context, S-8 filings by the acquirer may signal post-merger equity compensation plans, assumption of target equity awards, or inducement grants for retained employees.

Given the S-8 text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "ticker": "<ticker symbol as stated in the filing, or null if not stated>",
  "company": "<registrant company name>",
  "filing_date": "<MM/DD/YY>",

  "L1_headline": "+ <TICKER> – registers <shares> for employee plan. | <date>",

  "L2_brief": "<2-3 sentence summary covering: how many shares registered, under which plan, and any M&A context>",

  "L3_detailed": {
    "shares_registered": "<number of shares being registered>",
    "plan_name": "<name of the employee benefit plan>",
    "plan_type": "<Stock Option Plan | ESPP | RSU Plan | 401(k) | Inducement Plan | Omnibus Plan | Other>",
    "securities_type": "<common stock, preferred, options, etc.>",
    "offering_price_basis": "<how offering price is determined — market price, fixed, formula>",
    "dilution_impact": "<percentage of outstanding shares this registration represents>",
    "deal_context": "<connection to transaction as stated in the filing, if any. Otherwise 'N/A'>",
    "plan_details": "<key terms — vesting schedule, eligibility, administration>",
    "risks_flagged": ["<dilution concerns, large registration relative to float, post-merger equity integration>"]
  }
}

Rules:
- CRITICAL — FACTS ONLY: Every statement in your summary must be directly traceable to the filing text. Report ONLY what the document says. Do NOT add analysis, assess significance, interpret motives, predict outcomes, evaluate probability, or editorialize. Do NOT state what is "not disclosed" or "not mentioned" — simply omit fields where the filing is silent. If the filing does not say it, do not write it.
  GOOD: "CADE requested revenue data for 2021-2025 across four markets."
  BAD: "The broad scope of information requested indicates potentially detailed competitive analysis ahead."
  GOOD: "The offer expires June 10, 2026."
  BAD: "This tight timeline may create pressure on shareholders to tender quickly."
- PRECISION: Use the filing's exact terminology for legal, regulatory, and financial terms. Do NOT paraphrase in ways that broaden or narrow the stated meaning. GOOD: "All 14 Pennsylvania PUC hearings have concluded." BAD: "Regulatory proceedings concluded in Pennsylvania."
- L1 format MUST be: + <TICKER> – registers <share count> for <plan type>. | <date>
- Extract exact share counts and plan names
- Calculate dilution as a percentage of outstanding shares if information is available
- In M&A contexts, note whether the S-8 is for assumed target awards or new acquirer plans
- Flag unusually large registrations relative to shares outstanding
- Note if this is a post-merger filing (often indicates deal closing or integration)

S-8 TEXT:
"""

EXTRACTION_GUIDANCE = """This is an S-8 employee benefit plan registration statement.
Extract:
- Number of shares being registered
- Name and type of benefit plan (stock option plan, ESPP, RSU plan, etc.)
- Securities type being registered
- Offering price or price basis
- Plan terms: vesting schedule, eligibility
- If M&A-related: connection to deal (assumed awards, inducement grants, post-merger plans)
- Shares outstanding for dilution calculation
- Any incorporation by reference details"""


def fetch_filing_text(source: str) -> str:
    """Fetch and extract text from an S-8 filing (URL, local file, or PDF)."""
    from .fetch_utils import fetch_text_with_extraction
    return fetch_text_with_extraction(source, extraction_guidance=EXTRACTION_GUIDANCE)


def summarize(text: str, model: str = "claude-opus-4-8") -> dict:
    """Call Claude API to produce multi-level summary."""
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Set it in .env or Django settings (ANTHROPIC_API_KEY).")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": inject_deal_context(SUMMARY_PROMPT, DEAL_CONTEXT) + "\n\n" + text
        }]
    )

    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


def print_summary(s: dict):
    """Pretty-print the multi-level summary."""
    print("\n" + "=" * 70)
    print("  S-8 SUMMARY (Employee Benefit Plan Registration)")
    print("=" * 70)

    print(
        f"\n   Company: {s.get('company', 'N/A')} ({s.get('ticker', 'N/A')})")
    print(f"   Date:    {s.get('filing_date', 'N/A')}")

    print(f"\n📌 L1 | HEADLINE")
    print(f"   {s['L1_headline']}")

    print(f"\n📋 L2 | BRIEF")
    print(f"   {s['L2_brief']}")

    d = s["L3_detailed"]
    print(f"\n📊 L3 | DETAILED")
    print(f"   Shares Registered: {d.get('shares_registered', 'N/A')}")
    print(f"   Plan Name:         {d.get('plan_name', 'N/A')}")
    print(f"   Plan Type:         {d.get('plan_type', 'N/A')}")
    print(f"   Securities:        {d.get('securities_type', 'N/A')}")
    print(f"   Price Basis:       {d.get('offering_price_basis', 'N/A')}")
    print(f"   Dilution Impact:   {d.get('dilution_impact', 'N/A')}")
    print(f"   Deal Context:      {d.get('deal_context', 'N/A')}")
    print(f"   Plan Details:      {d.get('plan_details', 'N/A')}")
    if d.get("risks_flagged"):
        print(f"   Risks:")
        for r in d["risks_flagged"]:
            print(f"     • {r}")

    print("=" * 70)


def export_docx(s: dict, s3_key_suffix: str):
    """Build summary as Word doc, upload to S3 (summary_docx/), return (s3_path, s3_url)."""
    from .fetch_utils import is_empty_value, has_content, add_field
    from .s3_utils import upload_docx_bytes

    ticker = s.get("ticker", "UNKNOWN")
    date = s.get("filing_date", "")
    doc = DocxDocument()

    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    title = doc.add_heading(f"S-8 Summary: {ticker}", level=0)
    title.runs[0].font.size = Pt(20)

    meta = doc.add_paragraph()
    add_field(meta, "Company: ", s.get("company"), newline=False)
    meta.add_run("    ")
    add_field(meta, "Filing Date: ", date, newline=False)

    doc.add_heading("L1 — Headline", level=1)
    p = doc.add_paragraph()
    run = p.add_run(s["L1_headline"])
    run.bold = True
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(0, 51, 102)

    doc.add_heading("L2 — Brief", level=1)
    doc.add_paragraph(s["L2_brief"])

    doc.add_heading("L3 — Detailed", level=1)
    d = s["L3_detailed"]

    doc.add_heading("Registration Details", level=2)
    reg_p = doc.add_paragraph()
    add_field(reg_p, "Shares Registered: ", d.get("shares_registered"))
    add_field(reg_p, "Plan Name: ", d.get("plan_name"))
    add_field(reg_p, "Plan Type: ", d.get("plan_type"))
    add_field(reg_p, "Securities Type: ", d.get("securities_type"))
    add_field(reg_p, "Offering Price Basis: ", d.get(
        "offering_price_basis"), newline=False)

    if not is_empty_value(d.get("dilution_impact")):
        doc.add_heading("Dilution Impact", level=2)
        doc.add_paragraph(d.get("dilution_impact"))

    if not is_empty_value(d.get("deal_context")):
        doc.add_heading("Deal Context", level=2)
        doc.add_paragraph(d.get("deal_context"))

    if not is_empty_value(d.get("plan_details")):
        doc.add_heading("Plan Details", level=2)
        doc.add_paragraph(d.get("plan_details"))

    items = d.get("risks_flagged")
    if has_content(items):
        doc.add_heading("Risks Flagged", level=2)
        for r in items:
            if not is_empty_value(r):
                doc.add_paragraph(r, style="List Bullet")

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    path, url = upload_docx_bytes(buf.read(), s3_key_suffix)
    return path, url


def main():
    source = FILING_URL

    print(f"Fetching S-8 from: {source}")

    text = fetch_filing_text(source)
    print(f"Extracted {len(text.split())} words of text")

    print("Generating summary via Claude Opus 4.5...")
    result = summarize(text)

    print_summary(result)

    uid = filing_uid(FILING_URL)
    from .s3_utils import upload_json

    s3_json_path, s3_json_url = upload_json(result, f"s8_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    safe_ticker = sanitize_filename_part(result.get("ticker"))
    safe_date = sanitize_date_part(result.get("filing_date"))
    docx_suffix = f"S8_Summary_{safe_ticker}_{safe_date}_{uid}.docx"
    s3_docx_path, s3_docx_url = export_docx(result, docx_suffix)
    print(f"DOCX uploaded to S3: {s3_docx_url}")

    result["s3_docx_path"] = s3_docx_path
    result["s3_docx_url"] = s3_docx_url
    result["s3_json_path"] = s3_json_path
    result["s3_json_url"] = s3_json_url
    return result


if __name__ == "__main__":
    main()
