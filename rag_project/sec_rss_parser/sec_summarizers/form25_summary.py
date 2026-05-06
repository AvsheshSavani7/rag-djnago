"""
Form 25 (Notification of Delisting/Deregistration) Summarizer — Multi-level summaries via Claude API
Usage: python form25_summary.py

Form 25 is filed to notify the SEC that an issuer's securities are being removed from
listing on a national securities exchange. In M&A, this is filed after a merger closes
and the target's shares are delisted — it's the final confirmation that a deal is done.
It can also be filed for voluntary delisting or going-dark transactions.
"""

import anthropic
import re
import json
import sys
import os
import io
from pathlib import Path
from ._naming import filing_uid

# ──── PASTE YOUR FORM 25 URL HERE ────
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


SUMMARY_PROMPT = """You are an expert merger arbitrage analyst summarizing SEC Form 25 filings (Notification of Removal from Listing and/or Registration) for a merger arbitrage trading desk.

Form 25 is filed when an issuer's securities are being delisted from a national securities exchange. In the context of merger arbitrage, this filing typically signals one of:
1. **Post-merger delisting** — the target's shares are removed after a merger closes (most common for arb)
2. **Voluntary delisting** — a company chooses to delist (going dark, moving to another exchange)
3. **Involuntary delisting** — the exchange removes the company for non-compliance

This is a short, structured filing. Extract every detail available.

Given the Form 25 text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "ticker": "<ticker being delisted>",
  "company": "<issuer name>",
  "filing_date": "<MM/DD/YY>",
  "filed_by": "<who filed — the exchange or the issuer>",

  "L1_headline": "+ <TICKER> – delisted from <exchange>. | <date>",

  "L2_brief": "<2-3 sentence summary covering: what is being delisted, from which exchange, the reason (merger completion, voluntary, involuntary), and effective date>",

  "L3_detailed": {
    "securities_delisted": "<description of securities — common stock, preferred, warrants, etc.>",
    "exchange": "<NYSE | NASDAQ | NYSE American | CBOE | Other>",
    "delisting_reason": "<Post-Merger | Voluntary | Involuntary/Non-Compliance | Transfer to Another Exchange | Other>",
    "effective_date": "<date delisting becomes effective — typically 10 days after filing>",
    "rule_12d2_2_date": "<date Form 25 becomes effective under Rule 12d2-2, if stated>",
    "merger_context": {
      "acquirer": "<acquiring company, if this is a post-merger delisting>",
      "deal_closed_date": "<date the merger closed, if stated or inferable>",
      "final_consideration": "<what shareholders received — price per share, exchange ratio>",
      "short_form_merger": "<was this a short-form merger (no shareholder vote required)? Yes/No/Unknown>"
    },
    "deregistration": "<will the company also deregister (suspend SEC reporting obligations)? Yes/No/Unknown>",
    "last_trading_date": "<last date shares traded on the exchange, if stated>",
    "cusip": "<CUSIP number if mentioned>",
    "arb_implications": "<what this means for arb positions — deal confirmed closed, final payment timeline, any remaining stub/CVR considerations>"
  }
}

Rules:
- TONE: State only facts from the filing. Do NOT speculate on motives, interpret what actions "signal" or "suggest", assess confidence levels, or draw conclusions beyond what is explicitly stated. GOOD: "Company suspended earnings calls due to pending transaction." BAD: "Company suspended earnings calls, signaling high confidence in deal completion."
- L1 format MUST be: + <TICKER> – delisted from <exchange>. | <date>
- Identify whether this is a post-merger delisting (most important for arb) or other reason
- Extract the effective date — this is when shares officially stop trading
- Note the 10-day rule: Form 25 becomes effective 10 days after filing under Rule 12d2-2
- If merger-related: identify the acquirer and what shareholders received
- Flag whether deregistration (Form 15) will follow — this ends SEC reporting obligations
- Note the CUSIP if available
- For arb: confirm the deal is done and note any remaining considerations (CVRs, escrow, earnouts)

FORM 25 TEXT:
"""


def fetch_filing_text(source: str) -> str:
    """Fetch and extract text from a Form 25 filing (URL, local file, or PDF)."""
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
        max_tokens=2500,
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
    print("  FORM 25 SUMMARY (Delisting / Deregistration)")
    print("=" * 70)

    print(
        f"\n   Company:  {s.get('company', 'N/A')} ({s.get('ticker', 'N/A')})")
    print(f"   Filed By: {s.get('filed_by', 'N/A')}")
    print(f"   Date:     {s.get('filing_date', 'N/A')}")

    print(f"\n📌 L1 | HEADLINE")
    print(f"   {s['L1_headline']}")

    print(f"\n📋 L2 | BRIEF")
    print(f"   {s['L2_brief']}")

    d = s["L3_detailed"]
    print(f"\n📊 L3 | DETAILED")
    print(f"   Securities:     {d.get('securities_delisted', 'N/A')}")
    print(f"   Exchange:       {d.get('exchange', 'N/A')}")
    print(f"   Reason:         {d.get('delisting_reason', 'N/A')}")
    print(f"   Effective Date: {d.get('effective_date', 'N/A')}")
    if d.get("rule_12d2_2_date"):
        print(f"   Rule 12d2-2:    {d['rule_12d2_2_date']}")
    print(f"   Last Trading:   {d.get('last_trading_date', 'N/A')}")
    if d.get("cusip"):
        print(f"   CUSIP:          {d['cusip']}")
    print(f"   Deregistration: {d.get('deregistration', 'N/A')}")

    mc = d.get("merger_context", {})
    if mc.get("acquirer"):
        print(f"\n   ── MERGER CONTEXT ──")
        print(f"   Acquirer:       {mc.get('acquirer', 'N/A')}")
        print(f"   Deal Closed:    {mc.get('deal_closed_date', 'N/A')}")
        print(f"   Consideration:  {mc.get('final_consideration', 'N/A')}")
        print(f"   Short-Form:     {mc.get('short_form_merger', 'N/A')}")

    print(f"\n   ── ARB IMPLICATIONS ──")
    print(f"   {d.get('arb_implications', 'N/A')}")

    print("=" * 70)


def export_docx(s: dict, s3_key_suffix: str):
    """Build summary as Word doc, upload to S3 (summary_docx/), return (s3_path, s3_url)."""
    from .s3_utils import upload_docx_bytes

    ticker = s.get("ticker", "UNKNOWN")
    doc = DocxDocument()

    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    title = doc.add_heading(f"Form 25 Summary: {ticker}", level=0)
    title.runs[0].font.size = Pt(20)

    meta = doc.add_paragraph()
    meta.add_run("Company: ").bold = True
    meta.add_run(s.get("company", "N/A"))
    meta.add_run("    Filed By: ").bold = True
    meta.add_run(s.get("filed_by", "N/A"))

    date = s.get("filing_date", "")
    meta2 = doc.add_paragraph()
    meta2.add_run("Filing Date: ").bold = True
    meta2.add_run(date)

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

    doc.add_heading("Delisting Details", level=2)
    det_p = doc.add_paragraph()
    det_p.add_run("Securities Delisted: ").bold = True
    det_p.add_run(d.get("securities_delisted", "N/A") + "\n")
    det_p.add_run("Exchange: ").bold = True
    det_p.add_run(d.get("exchange", "N/A") + "\n")
    det_p.add_run("Reason: ").bold = True
    reason = d.get("delisting_reason", "N/A")
    reason_run = det_p.add_run(reason + "\n")
    if "post-merger" in reason.lower():
        reason_run.font.color.rgb = RGBColor(0, 128, 0)
        reason_run.bold = True
    det_p.add_run("Effective Date: ").bold = True
    det_p.add_run(d.get("effective_date", "N/A") + "\n")
    if d.get("rule_12d2_2_date"):
        det_p.add_run("Rule 12d2-2 Date: ").bold = True
        det_p.add_run(d["rule_12d2_2_date"] + "\n")
    det_p.add_run("Last Trading Date: ").bold = True
    det_p.add_run(d.get("last_trading_date", "N/A") + "\n")
    if d.get("cusip"):
        det_p.add_run("CUSIP: ").bold = True
        det_p.add_run(d["cusip"] + "\n")
    det_p.add_run("Deregistration: ").bold = True
    det_p.add_run(d.get("deregistration", "N/A"))

    mc = d.get("merger_context", {})
    if mc.get("acquirer"):
        doc.add_heading("Merger Context", level=2)
        mc_p = doc.add_paragraph()
        mc_p.add_run("Acquirer: ").bold = True
        mc_p.add_run(mc.get("acquirer", "N/A") + "\n")
        mc_p.add_run("Deal Closed: ").bold = True
        mc_p.add_run(mc.get("deal_closed_date", "N/A") + "\n")
        mc_p.add_run("Final Consideration: ").bold = True
        mc_p.add_run(mc.get("final_consideration", "N/A") + "\n")
        mc_p.add_run("Short-Form Merger: ").bold = True
        mc_p.add_run(mc.get("short_form_merger", "N/A"))

    doc.add_heading("Arb Implications", level=2)
    doc.add_paragraph(d.get("arb_implications", "N/A"))

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    path, url = upload_docx_bytes(buf.read(), s3_key_suffix)
    return path, url


def main():
    source = FILING_URL

    print(f"Fetching Form 25 from: {source}")

    text = fetch_filing_text(source)
    print(f"Extracted {len(text.split())} words of text")

    print("Generating summary via Claude Opus 4.5...")
    result = summarize(text)

    print_summary(result)

    uid = filing_uid(FILING_URL)
    from .s3_utils import upload_json

    s3_json_path, s3_json_url = upload_json(
        result, f"form25_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    ticker = result.get("ticker", "UNKNOWN")
    date = result.get("filing_date", "")
    safe_ticker = re.sub(r'[^\w\-\.]', '_', ticker)
    safe_date = date.replace("/", "-")
    docx_suffix = f"Form25_Summary_{safe_ticker}_{safe_date}_{uid}.docx"
    s3_docx_path, s3_docx_url = export_docx(result, docx_suffix)
    print(f"DOCX uploaded to S3: {s3_docx_url}")

    result["s3_docx_path"] = s3_docx_path
    result["s3_docx_url"] = s3_docx_url
    result["s3_json_path"] = s3_json_path
    result["s3_json_url"] = s3_json_url
    return result


if __name__ == "__main__":
    main()
