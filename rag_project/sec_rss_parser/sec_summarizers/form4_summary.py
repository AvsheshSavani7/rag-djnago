"""
Form 4 Filing Summarizer — Multi-level summaries via Claude API
Usage: python form4_summary.py
"""

import anthropic
import re
import json
import sys
import os
import io
from pathlib import Path
from ._naming import filing_uid

# ──── PASTE YOUR FORM 4 URL HERE ────
FILING_URL = ""
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


SUMMARY_PROMPT = """You are an expert analyst summarizing SEC Form 4 filings (insider ownership changes) for a merger arbitrage desk.

Form 4 reports changes in beneficial ownership by company insiders (officers, directors, and 10%+ shareholders). These filings reveal insider buying, selling, option exercises, and other transactions that can signal confidence or concern about a company's prospects — especially relevant during pending mergers.

Given the Form 4 text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "ticker": "<issuer ticker symbol>",
  "issuer": "<issuer company name>",
  "filing_date": "<MM/DD/YY>",
  "insider_name": "<name of reporting person>",
  "insider_title": "<officer title, director, or 10% owner>",
  "relationship": "<Officer | Director | 10% Owner | Officer & Director | Other>",

  "L1_headline": "+ <TICKER> – <insider name> <bought/sold/exercised> <shares/options>. | <date>",

  "L2_brief": "<2-3 sentence summary covering: who traded, what they did, how many shares, at what price, and what it might signal>",

  "L3_detailed": {
    "transactions": [
      {
        "type": "<Purchase | Sale | Option Exercise | Gift | Conversion | Other>",
        "date": "<transaction date>",
        "shares": "<number of shares>",
        "price_per_share": "<price per share, or N/A for gifts>",
        "total_value": "<total dollar value of transaction>",
        "acquired_or_disposed": "<Acquired (A) | Disposed (D)>"
      }
    ],
    "post_transaction_holdings": "<total shares held after transaction>",
    "ownership_type": "<Direct | Indirect (through trust, LLC, etc.)>",
    "deal_signal": "<for pending M&A: what this insider activity may signal about deal confidence. Otherwise 'N/A'>",
    "pattern_notes": "<any notable pattern — cluster buying, first purchase, selling into strength, etc.>",
    "risks_flagged": ["<10b5-1 plan noted, blackout period concerns, large disposal relative to holdings>"]
  }
}

Rules:
- TONE: State only facts from the filing. Do NOT speculate on motives, interpret what actions "signal" or "suggest", assess confidence levels, or draw conclusions beyond what is explicitly stated. GOOD: "Company suspended earnings calls due to pending transaction." BAD: "Company suspended earnings calls, signaling high confidence in deal completion."
- L1 format MUST be: + <TICKER> – <insider> <action>. | <date>
- Extract EVERY transaction listed in the filing with exact shares, prices, and dates
- Calculate total transaction value where possible
- Note whether trades were under a 10b5-1 plan (pre-planned) vs discretionary
- For companies involved in pending mergers, assess whether insider activity signals deal confidence
- Flag unusually large transactions relative to the insider's total holdings

FORM 4 TEXT:
"""


def fetch_filing_text(source: str) -> str:
    """Fetch and extract text from a Form 4 filing (URL, local file, or PDF)."""
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
            "content": SUMMARY_PROMPT + "\n\n" + text
        }]
    )

    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


def print_summary(s: dict):
    """Pretty-print the multi-level summary."""
    print("\n" + "=" * 70)
    print("  FORM 4 SUMMARY (Insider Ownership Change)")
    print("=" * 70)

    print(
        f"\n   Issuer:   {s.get('issuer', 'N/A')} ({s.get('ticker', 'N/A')})")
    print(
        f"   Insider:  {s.get('insider_name', 'N/A')} — {s.get('insider_title', 'N/A')}")
    print(f"   Role:     {s.get('relationship', 'N/A')}")
    print(f"   Date:     {s.get('filing_date', 'N/A')}")

    print(f"\n📌 L1 | HEADLINE")
    print(f"   {s['L1_headline']}")

    print(f"\n📋 L2 | BRIEF")
    print(f"   {s['L2_brief']}")

    d = s["L3_detailed"]
    print(f"\n📊 L3 | DETAILED")
    print(f"   Transactions:")
    for t in d.get("transactions", []):
        print(f"     • {t.get('type', 'N/A')}: {t.get('shares', 'N/A')} shares @ "
              f"${t.get('price_per_share', 'N/A')} on {t.get('date', 'N/A')} "
              f"({t.get('acquired_or_disposed', 'N/A')}) — {t.get('total_value', 'N/A')}")
    print(f"   Post-Txn Holdings: {d.get('post_transaction_holdings', 'N/A')}")
    print(f"   Ownership Type:    {d.get('ownership_type', 'N/A')}")
    print(f"   Deal Signal:       {d.get('deal_signal', 'N/A')}")
    print(f"   Pattern Notes:     {d.get('pattern_notes', 'N/A')}")
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

    title = doc.add_heading(f"Form 4 Summary: {ticker}", level=0)
    title.runs[0].font.size = Pt(20)

    meta = doc.add_paragraph()
    meta.add_run(f"Issuer: ").bold = True
    meta.add_run(s.get("issuer", "N/A"))
    meta.add_run(f"    Insider: ").bold = True
    meta.add_run(
        f"{s.get('insider_name', 'N/A')} ({s.get('insider_title', 'N/A')})")

    date = s.get("filing_date", "")
    meta2 = doc.add_paragraph()
    meta2.add_run(f"Filing Date: ").bold = True
    meta2.add_run(date)
    meta2.add_run(f"    Relationship: ").bold = True
    meta2.add_run(s.get("relationship", "N/A"))

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

    doc.add_heading("Transactions", level=2)
    for t in d.get("transactions", []):
        doc.add_paragraph(
            f"{t.get('type', 'N/A')}: {t.get('shares', 'N/A')} shares @ "
            f"${t.get('price_per_share', 'N/A')} on {t.get('date', 'N/A')} "
            f"({t.get('acquired_or_disposed', 'N/A')}) — {t.get('total_value', 'N/A')}",
            style="List Bullet"
        )

    details = doc.add_paragraph()
    details.add_run("Post-Transaction Holdings: ").bold = True
    details.add_run(d.get("post_transaction_holdings", "N/A") + "\n")
    details.add_run("Ownership Type: ").bold = True
    details.add_run(d.get("ownership_type", "N/A"))

    doc.add_heading("Deal Signal", level=2)
    doc.add_paragraph(d.get("deal_signal", "N/A"))

    doc.add_heading("Pattern Notes", level=2)
    doc.add_paragraph(d.get("pattern_notes", "N/A"))

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

    print(f"Fetching Form 4 from: {source}")

    text = fetch_filing_text(source)
    print(f"Extracted {len(text.split())} words of text")

    print("Generating summary via Claude Opus 4.5...")
    result = summarize(text)

    print_summary(result)

    uid = filing_uid(FILING_URL)
    from .s3_utils import upload_json

    s3_json_path, s3_json_url = upload_json(
        result, f"form4_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    ticker = result.get("ticker", "UNKNOWN")
    date = result.get("filing_date", "")
    safe_ticker = re.sub(r'[^\w\-\.]', '_', ticker)
    safe_date = date.replace("/", "-")
    docx_suffix = f"Form4_Summary_{safe_ticker}_{safe_date}_{uid}.docx"
    s3_docx_path, s3_docx_url = export_docx(result, docx_suffix)
    print(f"DOCX uploaded to S3: {s3_docx_url}")

    result["s3_docx_path"] = s3_docx_path
    result["s3_docx_url"] = s3_docx_url
    result["s3_json_path"] = s3_json_path
    result["s3_json_url"] = s3_json_url
    return result


if __name__ == "__main__":
    main()
