"""
424(b)(5) Prospectus Supplement Summarizer — Multi-level summaries via Claude API
Usage: python 424b5_summary.py
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

# ──── PASTE YOUR 424(B)(5) URL HERE ────
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


SUMMARY_PROMPT = """You are an expert analyst summarizing SEC Form 424(B)(5) prospectus supplement filings for a merger arbitrage desk.

424(B)(5) filings are prospectus supplements that contain the final terms of a securities offering — pricing, share count, underwriter details, and use of proceeds. These are critical for understanding dilution, capital raises, and in M&A contexts, the securities being issued to fund or complete a deal.

Given the 424(B)(5) text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "ticker": "<ticker symbol as stated in the filing, or null if not stated>",
  "company": "<company name>",
  "filing_date": "<MM/DD/YY>",
  "offering_type": "<Common Stock | Preferred Stock | Debt/Notes | Convertible Notes | Units | Warrants | Mixed>",

  "L1_headline": "+ <TICKER> – <offering type> offering at $<price>. | <date>",

  "L2_brief": "<2-3 sentence summary covering: what is being offered, at what price, total proceeds, and why (use of proceeds)>",

  "L3_detailed": {
    "offering_terms": {
      "securities_offered": "<description of securities — e.g., '5,000,000 shares of common stock'>",
      "offering_price": "<price per share/unit>",
      "gross_proceeds": "<total gross proceeds>",
      "net_proceeds": "<net proceeds after underwriting discounts>",
      "underwriting_discount": "<discount/commission per share and total>",
      "overallotment_option": "<greenshoe option details, if any>"
    },
    "use_of_proceeds": "<stated use of proceeds — general corporate, debt repayment, acquisition financing, etc.>",
    "dilution": "<dilution impact — shares outstanding before/after, percentage dilution if stated>",
    "underwriters": ["<lead underwriters/bookrunners>"],
    "settlement_date": "<expected settlement/closing date>",
    "deal_relevance": "<if related to M&A financing: connection to pending transaction as stated in the filing. Otherwise 'N/A'>",
    "key_risk_factors": ["<offering-specific risks — dilution, market conditions, use of proceeds risk>"]
  }
}

Rules:
- CRITICAL — FACTS ONLY: Every statement in your summary must be directly traceable to the filing text. Report ONLY what the document says. Do NOT add analysis, assess significance, interpret motives, predict outcomes, evaluate probability, or editorialize. Do NOT state what is "not disclosed" or "not mentioned" — simply omit fields where the filing is silent. If the filing does not say it, do not write it.
  GOOD: "CADE requested revenue data for 2021-2025 across four markets."
  BAD: "The broad scope of information requested indicates potentially detailed competitive analysis ahead."
  GOOD: "The offer expires June 10, 2026."
  BAD: "This tight timeline may create pressure on shareholders to tender quickly."
- PRECISION: Use the filing's exact terminology for legal, regulatory, and financial terms. Do NOT paraphrase in ways that broaden or narrow the stated meaning. GOOD: "All 14 Pennsylvania PUC hearings have concluded." BAD: "Regulatory proceedings concluded in Pennsylvania."
- L1 format MUST be: + <TICKER> – <offering description>. | <date>
- Extract EXACT offering price, share count, gross/net proceeds, and underwriting terms
- Note the overallotment (greenshoe) option if present — this affects total potential dilution
- Identify the lead underwriters/bookrunners
- For M&A-related offerings: explain how proceeds relate to the deal (acquisition financing, etc.)
- Calculate dilution percentage if pre/post share counts are available
- Flag any lock-up agreements mentioned

424(B)(5) TEXT:
"""

EXTRACTION_GUIDANCE = """This is a 424(B)(5) prospectus supplement (securities offering).
Extract:
- Securities offered: type, number of shares/units
- Offering price per share/unit
- Gross and net proceeds
- Underwriting discount/commission
- Overallotment (greenshoe) option details
- Use of proceeds
- Dilution: shares outstanding before and after offering
- Lead underwriters/bookrunners
- Settlement date and lock-up agreements
- If M&A-related: connection to pending deal
"""


def fetch_filing_text(source: str) -> str:
    """Fetch and extract text from a 424(B)(5) filing (URL, local file, or PDF)."""
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
        max_tokens=8000,
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
    print("  424(B)(5) PROSPECTUS SUPPLEMENT SUMMARY")
    print("=" * 70)

    print(
        f"\n   Company:  {s.get('company', 'N/A')} ({s.get('ticker', 'N/A')})")
    print(f"   Type:     {s.get('offering_type', 'N/A')}")
    print(f"   Date:     {s.get('filing_date', 'N/A')}")

    print(f"\n📌 L1 | HEADLINE")
    print(f"   {s['L1_headline']}")

    print(f"\n📋 L2 | BRIEF")
    print(f"   {s['L2_brief']}")

    d = s["L3_detailed"]
    ot = d.get("offering_terms", {})
    print(f"\n📊 L3 | DETAILED")
    print(f"   Offering Terms:")
    print(f"     Securities:   {ot.get('securities_offered', 'N/A')}")
    print(f"     Price:        {ot.get('offering_price', 'N/A')}")
    print(f"     Gross:        {ot.get('gross_proceeds', 'N/A')}")
    print(f"     Net:          {ot.get('net_proceeds', 'N/A')}")
    print(f"     UW Discount:  {ot.get('underwriting_discount', 'N/A')}")
    print(f"     Greenshoe:    {ot.get('overallotment_option', 'N/A')}")
    print(f"   Use of Proceeds: {d.get('use_of_proceeds', 'N/A')}")
    print(f"   Dilution:        {d.get('dilution', 'N/A')}")
    print(f"   Settlement:      {d.get('settlement_date', 'N/A')}")
    print(f"   Deal Relevance:  {d.get('deal_relevance', 'N/A')}")
    if d.get("underwriters"):
        print(f"   Underwriters:")
        for u in d["underwriters"]:
            print(f"     • {u}")
    if d.get("key_risk_factors"):
        print(f"   Key Risks:")
        for r in d["key_risk_factors"]:
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

    title = doc.add_heading(f"424(B)(5) Summary: {ticker}", level=0)
    title.runs[0].font.size = Pt(20)

    meta = doc.add_paragraph()
    add_field(meta, "Company: ", s.get("company"), newline=False)
    add_field(meta, "    Offering Type: ", s.get(
        "offering_type"), newline=False)

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
    ot = d.get("offering_terms", {})

    doc.add_heading("Offering Terms", level=2)
    terms_p = doc.add_paragraph()
    add_field(terms_p, "Securities Offered: ", ot.get("securities_offered"))
    add_field(terms_p, "Offering Price: ", ot.get("offering_price"))
    add_field(terms_p, "Gross Proceeds: ", ot.get("gross_proceeds"))
    add_field(terms_p, "Net Proceeds: ", ot.get("net_proceeds"))
    add_field(terms_p, "Underwriting Discount: ",
              ot.get("underwriting_discount"))
    add_field(terms_p, "Overallotment Option: ", ot.get(
        "overallotment_option"), newline=False)

    if not is_empty_value(d.get("use_of_proceeds")):
        doc.add_heading("Use of Proceeds", level=2)
        doc.add_paragraph(d.get("use_of_proceeds"))

    if not is_empty_value(d.get("dilution")):
        doc.add_heading("Dilution", level=2)
        doc.add_paragraph(d.get("dilution"))

    if not is_empty_value(d.get("settlement_date")):
        doc.add_heading("Settlement Date", level=2)
        doc.add_paragraph(d.get("settlement_date"))

    underwriters = d.get("underwriters")
    if has_content(underwriters):
        doc.add_heading("Underwriters", level=2)
        for u in underwriters:
            if not is_empty_value(u):
                doc.add_paragraph(u, style="List Bullet")

    if not is_empty_value(d.get("deal_relevance")):
        doc.add_heading("Deal Relevance", level=2)
        doc.add_paragraph(d.get("deal_relevance"))

    risks = d.get("key_risk_factors")
    if has_content(risks):
        doc.add_heading("Key Risk Factors", level=2)
        for r in risks:
            if not is_empty_value(r):
                doc.add_paragraph(r, style="List Bullet")

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    path, url = upload_docx_bytes(buf.read(), s3_key_suffix)
    return path, url


def main():
    source = FILING_URL

    print(f"Fetching 424(B)(5) from: {source}")

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
        result, f"424b5_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    safe_ticker = sanitize_filename_part(result.get("ticker"))
    safe_date = sanitize_date_part(result.get("filing_date"))
    docx_suffix = f"424B5_Summary_{safe_ticker}_{safe_date}_{uid}.docx"
    s3_docx_path, s3_docx_url = export_docx(result, docx_suffix)
    print(f"DOCX uploaded to S3: {s3_docx_url}")

    result["s3_docx_path"] = s3_docx_path
    result["s3_docx_url"] = s3_docx_url
    result["s3_json_path"] = s3_json_path
    result["s3_json_url"] = s3_json_url
    return result


if __name__ == "__main__":
    main()
