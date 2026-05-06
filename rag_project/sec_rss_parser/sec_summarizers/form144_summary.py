"""
Form 144 Filing Summarizer — Multi-level summaries via Claude API
Usage: python form144_summary.py
"""

import anthropic
import re
import json
import sys
import os
import io
from pathlib import Path
from ._naming import filing_uid

# ──── PASTE YOUR FORM 144 URL HERE ────
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


SUMMARY_PROMPT = """You are an expert analyst summarizing SEC Form 144 filings (notice of proposed sale of securities) for a merger arbitrage desk.

Form 144 is filed by affiliates or insiders who intend to sell restricted or control securities under Rule 144. These filings signal upcoming insider selling and can indicate confidence levels in a company's stock price — particularly important during pending mergers when insiders selling may signal concerns about deal completion.

Given the Form 144 text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "ticker": "<ticker symbol>",
  "issuer": "<issuer company name>",
  "filing_date": "<MM/DD/YY>",
  "seller_name": "<name of person filing>",
  "seller_relationship": "<Officer | Director | 10% Owner | Affiliate | Other>",

  "L1_headline": "+ <TICKER> – <seller> files to sell <amount>. | <date>",

  "L2_brief": "<2-3 sentence summary covering: who plans to sell, how many shares, approximate value, and what it may signal>",

  "L3_detailed": {
    "proposed_sale": {
      "shares_to_sell": "<number of shares proposed for sale>",
      "estimated_value": "<approximate dollar value based on recent price>",
      "securities_type": "<common stock, preferred, options, warrants, etc.>",
      "acquisition_date": "<when the securities were originally acquired>",
      "acquisition_method": "<how acquired — compensation, open market, option exercise, etc.>"
    },
    "broker_info": "<name of broker through whom sale will be made, if disclosed>",
    "seller_total_holdings": "<total shares held by seller, if disclosed>",
    "percentage_of_holdings": "<what % of total holdings this sale represents, if calculable>",
    "deal_signal": "<for pending M&A: what this proposed sale may signal. Otherwise 'N/A'>",
    "risks_flagged": ["<large sale relative to holdings, timing concerns, pending deal implications>"]
  }
}

Rules:
- TONE: State only facts from the filing. Do NOT speculate on motives, interpret what actions "signal" or "suggest", assess confidence levels, or draw conclusions beyond what is explicitly stated. GOOD: "Company suspended earnings calls due to pending transaction." BAD: "Company suspended earnings calls, signaling high confidence in deal completion."
- L1 format MUST be: + <TICKER> – <seller> files to sell <shares>. | <date>
- Extract exact share counts, dates, and acquisition details
- Calculate approximate value using recent trading price if mentioned
- Note the relationship between the seller and the issuer
- For companies in pending mergers, assess whether this sale filing is routine or potentially signals deal risk
- Flag if the proposed sale is a large percentage of total holdings

FORM 144 TEXT:
"""


def fetch_filing_text(source: str) -> str:
    """Fetch and extract text from a Form 144 filing (URL, local file, or PDF)."""
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
    print("  FORM 144 SUMMARY (Notice of Proposed Sale)")
    print("=" * 70)

    print(f"\n   Issuer:  {s.get('issuer', 'N/A')} ({s.get('ticker', 'N/A')})")
    print(
        f"   Seller:  {s.get('seller_name', 'N/A')} — {s.get('seller_relationship', 'N/A')}")
    print(f"   Date:    {s.get('filing_date', 'N/A')}")

    print(f"\n📌 L1 | HEADLINE")
    print(f"   {s['L1_headline']}")

    print(f"\n📋 L2 | BRIEF")
    print(f"   {s['L2_brief']}")

    d = s["L3_detailed"]
    ps = d.get("proposed_sale", {})
    print(f"\n📊 L3 | DETAILED")
    print(f"   Proposed Sale:")
    print(f"     Shares:       {ps.get('shares_to_sell', 'N/A')}")
    print(f"     Est. Value:   {ps.get('estimated_value', 'N/A')}")
    print(f"     Security:     {ps.get('securities_type', 'N/A')}")
    print(
        f"     Acquired:     {ps.get('acquisition_date', 'N/A')} via {ps.get('acquisition_method', 'N/A')}")
    print(f"   Broker:         {d.get('broker_info', 'N/A')}")
    print(f"   Total Holdings: {d.get('seller_total_holdings', 'N/A')}")
    print(f"   % of Holdings:  {d.get('percentage_of_holdings', 'N/A')}")
    print(f"   Deal Signal:    {d.get('deal_signal', 'N/A')}")
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

    title = doc.add_heading(f"Form 144 Summary: {ticker}", level=0)
    title.runs[0].font.size = Pt(20)

    meta = doc.add_paragraph()
    meta.add_run(f"Issuer: ").bold = True
    meta.add_run(s.get("issuer", "N/A"))
    meta.add_run(f"    Seller: ").bold = True
    meta.add_run(
        f"{s.get('seller_name', 'N/A')} ({s.get('seller_relationship', 'N/A')})")

    date = s.get("filing_date", "")
    meta2 = doc.add_paragraph()
    meta2.add_run(f"Filing Date: ").bold = True
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
    ps = d.get("proposed_sale", {})

    doc.add_heading("Proposed Sale", level=2)
    sale_p = doc.add_paragraph()
    sale_p.add_run("Shares to Sell: ").bold = True
    sale_p.add_run(ps.get("shares_to_sell", "N/A") + "\n")
    sale_p.add_run("Estimated Value: ").bold = True
    sale_p.add_run(ps.get("estimated_value", "N/A") + "\n")
    sale_p.add_run("Securities Type: ").bold = True
    sale_p.add_run(ps.get("securities_type", "N/A") + "\n")
    sale_p.add_run("Acquisition Date: ").bold = True
    sale_p.add_run(ps.get("acquisition_date", "N/A") + "\n")
    sale_p.add_run("Acquisition Method: ").bold = True
    sale_p.add_run(ps.get("acquisition_method", "N/A"))

    details = doc.add_paragraph()
    details.add_run("Broker: ").bold = True
    details.add_run(d.get("broker_info", "N/A") + "\n")
    details.add_run("Total Holdings: ").bold = True
    details.add_run(d.get("seller_total_holdings", "N/A") + "\n")
    details.add_run("% of Holdings: ").bold = True
    details.add_run(d.get("percentage_of_holdings", "N/A"))

    doc.add_heading("Deal Signal", level=2)
    doc.add_paragraph(d.get("deal_signal", "N/A"))

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

    print(f"Fetching Form 144 from: {source}")

    text = fetch_filing_text(source)
    print(f"Extracted {len(text.split())} words of text")

    print("Generating summary via Claude Opus 4.5...")
    result = summarize(text)

    print_summary(result)

    uid = filing_uid(FILING_URL)
    from .s3_utils import upload_json

    s3_json_path, s3_json_url = upload_json(
        result, f"form144_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    ticker = result.get("ticker", "UNKNOWN")
    date = result.get("filing_date", "")
    safe_ticker = re.sub(r'[^\w\-\.]', '_', ticker)
    safe_date = date.replace("/", "-")
    docx_suffix = f"Form144_Summary_{safe_ticker}_{safe_date}_{uid}.docx"
    s3_docx_path, s3_docx_url = export_docx(result, docx_suffix)
    print(f"DOCX uploaded to S3: {s3_docx_url}")

    result["s3_docx_path"] = s3_docx_path
    result["s3_docx_url"] = s3_docx_url
    result["s3_json_path"] = s3_json_path
    result["s3_json_url"] = s3_json_url
    return result


if __name__ == "__main__":
    main()
