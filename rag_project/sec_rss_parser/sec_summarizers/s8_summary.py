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
from ._naming import filing_uid
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
  "ticker": "<ticker symbol>",
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
    "deal_context": "<if related to M&A: explains connection — assumed awards, inducement grants, post-close retention. Otherwise 'N/A'>",
    "plan_details": "<key terms — vesting schedule, eligibility, administration>",
    "risks_flagged": ["<dilution concerns, large registration relative to float, post-merger equity integration>"]
  }
}

Rules:
- TONE: State only facts from the filing. Do NOT speculate on motives, interpret what actions "signal" or "suggest", assess confidence levels, or draw conclusions beyond what is explicitly stated. GOOD: "Company suspended earnings calls due to pending transaction." BAD: "Company suspended earnings calls, signaling high confidence in deal completion."
- L1 format MUST be: + <TICKER> – registers <share count> for <plan type>. | <date>
- Extract exact share counts and plan names
- Calculate dilution as a percentage of outstanding shares if information is available
- In M&A contexts, note whether the S-8 is for assumed target awards or new acquirer plans
- Flag unusually large registrations relative to shares outstanding
- Note if this is a post-merger filing (often indicates deal closing or integration)

S-8 TEXT:
"""


def fetch_filing_text(source: str) -> str:
    """Fetch and extract text from an S-8 filing (URL, local file, or PDF)."""
    from .fetch_utils import fetch_text
    return fetch_text(source)


def summarize(text: str, model: str = "claude-sonnet-4-6") -> dict:
    """Call Claude API to produce multi-level summary."""
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Set it in .env or Django settings (ANTHROPIC_API_KEY).")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    msg = client.messages.create(
        model=model,
        max_tokens=1500,
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
    from .s3_utils import upload_docx_bytes

    ticker = s.get("ticker", "UNKNOWN")
    doc = DocxDocument()

    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    title = doc.add_heading(f"S-8 Summary: {ticker}", level=0)
    title.runs[0].font.size = Pt(20)

    meta = doc.add_paragraph()
    meta.add_run(f"Company: ").bold = True
    meta.add_run(s.get("company", "N/A"))
    date = s.get("filing_date", "")
    meta.add_run(f"    Filing Date: ").bold = True
    meta.add_run(date)

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
    reg_p.add_run("Shares Registered: ").bold = True
    reg_p.add_run(d.get("shares_registered", "N/A") + "\n")
    reg_p.add_run("Plan Name: ").bold = True
    reg_p.add_run(d.get("plan_name", "N/A") + "\n")
    reg_p.add_run("Plan Type: ").bold = True
    reg_p.add_run(d.get("plan_type", "N/A") + "\n")
    reg_p.add_run("Securities Type: ").bold = True
    reg_p.add_run(d.get("securities_type", "N/A") + "\n")
    reg_p.add_run("Offering Price Basis: ").bold = True
    reg_p.add_run(d.get("offering_price_basis", "N/A"))

    doc.add_heading("Dilution Impact", level=2)
    doc.add_paragraph(d.get("dilution_impact", "N/A"))

    doc.add_heading("Deal Context", level=2)
    doc.add_paragraph(d.get("deal_context", "N/A"))

    doc.add_heading("Plan Details", level=2)
    doc.add_paragraph(d.get("plan_details", "N/A"))

    if d.get("risks_flagged"):
        doc.add_heading("Risks Flagged", level=2)
        for r in d["risks_flagged"]:
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

    ticker = result.get("ticker", "UNKNOWN")
    date = result.get("filing_date", "")
    safe_ticker = re.sub(r'[^\w\-\.]', '_', ticker)
    safe_date = date.replace("/", "-")
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
