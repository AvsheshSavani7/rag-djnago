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
from ._naming import filing_uid, sanitize_date_part, sanitize_filename_part
from ._deal_context import inject_deal_context

# ──── PASTE YOUR FORM 144 URL HERE ────
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


SUMMARY_PROMPT = """You are an expert analyst summarizing SEC Form 144 filings (notice of proposed sale of securities) for a merger arbitrage desk.

Form 144 is filed by affiliates or insiders who intend to sell restricted or control securities under Rule 144. These filings signal upcoming insider selling and can indicate confidence levels in a company's stock price — particularly important during pending mergers when insiders selling may signal concerns about deal completion.

Given the Form 144 text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "ticker": "<ticker symbol as stated in the filing, or null if not stated>",
  "issuer": "<issuer company name>",
  "filing_date": "<MM/DD/YY>",
  "seller_name": "<name of person filing>",
  "seller_relationship": "<Officer | Director | 10% Owner | Affiliate | Other>",

  "L1_headline": "+ <TICKER> – <seller> files to sell <amount>. | <date>",

  "L2_brief": "<2-3 sentence summary covering: who plans to sell, how many shares, approximate value, and and relationship to any pending transaction>",

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
    "deal_signal": "<for pending M&A: connection to any pending transaction as stated in the filing. Otherwise 'N/A'>",,
    "risks_flagged": ["<large sale relative to holdings, timing concerns, pending deal implications>"]
  }
}

Rules:
- CRITICAL — FACTS ONLY: Every statement in your summary must be directly traceable to the filing text. Report ONLY what the document says. Do NOT add analysis, assess significance, interpret motives, predict outcomes, evaluate probability, or editorialize. Do NOT state what is "not disclosed" or "not mentioned" — simply omit fields where the filing is silent. If the filing does not say it, do not write it.
  GOOD: "CADE requested revenue data for 2021-2025 across four markets."
  BAD: "The broad scope of information requested indicates potentially detailed competitive analysis ahead."
  GOOD: "The offer expires June 10, 2026."
  BAD: "This tight timeline may create pressure on shareholders to tender quickly."
- PRECISION: Use the filing's exact terminology for legal, regulatory, and financial terms. Do NOT paraphrase in ways that broaden or narrow the stated meaning. GOOD: "All 14 Pennsylvania PUC hearings have concluded." BAD: "Regulatory proceedings concluded in Pennsylvania."
- L1 format MUST be: + <TICKER> – <seller> files to sell <shares>. | <date>
- Extract exact share counts, dates, and acquisition details
- Calculate approximate value using recent trading price if mentioned
- Note the relationship between the seller and the issuer
- For companies in pending mergers, note any connection to pending transactions as stated in the filing
- Flag if the proposed sale is a large percentage of total holdings

FORM 144 TEXT:
"""

EXTRACTION_GUIDANCE = """This is a Form 144 notice of proposed sale of restricted securities.
Extract:
- Seller name and relationship/title at issuer
- Number of shares proposed for sale
- Securities type and class
- Approximate date of sale
- Acquisition date and method of acquisition
- Broker/dealer information
- Seller's total holdings before and after proposed sale
- If M&A-related: connection to pending deal"""


def fetch_filing_text(source: str) -> str:
    """Fetch and extract text from a Form 144 filing (URL, local file, or PDF)."""
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
    from .fetch_utils import is_empty_value, has_content, add_field
    from .s3_utils import upload_docx_bytes

    ticker = s.get("ticker", "UNKNOWN")
    date = s.get("filing_date", "")
    doc = DocxDocument()

    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    title = doc.add_heading(f"Form 144 Summary: {ticker}", level=0)
    title.runs[0].font.size = Pt(20)

    meta = doc.add_paragraph()
    add_field(meta, "Issuer: ", s.get("issuer"), newline=False)
    seller_name = s.get("seller_name")
    seller_rel = s.get("seller_relationship")
    if not is_empty_value(seller_name):
        meta.add_run("    ")
        meta.add_run("Seller: ").bold = True
        seller_text = str(seller_name)
        if not is_empty_value(seller_rel):
            seller_text += f" ({seller_rel})"
        meta.add_run(seller_text)

    meta2 = doc.add_paragraph()
    add_field(meta2, "Filing Date: ", date, newline=False)

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
    add_field(sale_p, "Shares to Sell: ", ps.get("shares_to_sell"))
    add_field(sale_p, "Estimated Value: ", ps.get("estimated_value"))
    add_field(sale_p, "Securities Type: ", ps.get("securities_type"))
    add_field(sale_p, "Acquisition Date: ", ps.get("acquisition_date"))
    add_field(sale_p, "Acquisition Method: ", ps.get(
        "acquisition_method"), newline=False)

    details = doc.add_paragraph()
    add_field(details, "Broker: ", d.get("broker_info"))
    add_field(details, "Total Holdings: ", d.get("seller_total_holdings"))
    add_field(details, "% of Holdings: ", d.get(
        "percentage_of_holdings"), newline=False)

    if not is_empty_value(d.get("deal_signal")):
        doc.add_heading("Deal Signal", level=2)
        doc.add_paragraph(d.get("deal_signal"))

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

    print(f"Fetching Form 144 from: {source}")

    text = fetch_filing_text(source)
    print(f"Extracted {len(text.split())} words of text")

    print("Generating summary via Claude Opus 4.5...")
    result = summarize(text)
    from ._ticker_context import apply_known_tickers
    result = apply_known_tickers(result, DEAL_CONTEXT)

    print_summary(result)

    uid = filing_uid(FILING_URL)
    from .s3_utils import upload_json

    s3_json_path, s3_json_url = upload_json(
        result, f"form144_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    safe_ticker = sanitize_filename_part(result.get("ticker"))
    safe_date = sanitize_date_part(result.get("filing_date"))
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
