"""
6-K Filing Summarizer — Multi-level summaries via Claude API
Usage: python 6k_summary.py
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

# ──── PASTE YOUR 6-K URL HERE ────
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


SUMMARY_PROMPT = """You are an expert analyst summarizing SEC Form 6-K filings for a merger arbitrage desk.

Form 6-K is the report used by foreign private issuers (non-US companies listed on US exchanges) to disclose material information. It is the international equivalent of the 8-K. These filings often contain earnings results, deal announcements, shareholder meeting results, regulatory updates, or material agreements for companies like those listed as ADRs.

Given the 6-K text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "ticker": "<US ticker or ADR symbol as stated in the filing, or null if not stated>",
  "company": "<company name>",
  "home_country": "<country of incorporation>",
  "filing_date": "<MM/DD/YY>",
  "report_type": "<Earnings | Deal Announcement | Shareholder Meeting | Regulatory Update | Material Agreement | Other>",

  "L1_headline": "+ <TICKER> – <key event in ≤8 words>. | <date>",

  "L2_brief": "<2-3 sentence summary covering: what happened, key numbers, and stated impact if any>",

  "L3_detailed": {
    "event": "<what happened>",
    "key_figures": ["<financial figures, vote percentages, deal values, share prices>"],
    "market_impact": "<stated effects on US-listed shares or ADRs, if any>",
    "deal_relevance": "<if M&A related: deal-related facts stated in the filing. If not M&A: 'N/A'>",
    "regulatory_notes": "<any regulatory body mentions — local regulators, EU Commission, competition authorities>",
    "cross_border_considerations": "<currency, jurisdiction, or structural notes relevant to US investors>",
    "risks_flagged": ["<any risks, litigation, regulatory issues, FX exposure>"]
  }
}

Rules:
- CRITICAL — FACTS ONLY: Every statement in your summary must be directly traceable to the filing text. Report ONLY what the document says. Do NOT add analysis, assess significance, interpret motives, predict outcomes, evaluate probability, or editorialize. Do NOT state what is "not disclosed" or "not mentioned" — simply omit fields where the filing is silent. If the filing does not say it, do not write it.
  GOOD: "CADE requested revenue data for 2021-2025 across four markets."
  BAD: "The broad scope of information requested indicates potentially detailed competitive analysis ahead."
  GOOD: "The offer expires June 10, 2026."
  BAD: "This tight timeline may create pressure on shareholders to tender quickly."
- PRECISION: Use the filing's exact terminology for legal, regulatory, and financial terms. Do NOT paraphrase in ways that broaden or narrow the stated meaning. GOOD: "All 14 Pennsylvania PUC hearings have concluded." BAD: "Regulatory proceedings concluded in Pennsylvania."
- L1 format MUST be: + <TICKER> – <event>. | <date>
- Use the US-listed ticker or ADR symbol
- Note the home country and any cross-border regulatory considerations stated in the filing
- Extract exact figures, dates, and percentages
- Flag any currency-related details (reporting currency vs USD)
- For merger-related 6-Ks, extract stated deal terms, regulatory filings, and timeline updates

6-K TEXT:
"""
EXTRACTION_GUIDANCE = """This is a 6-K report by a foreign private issuer.
Extract:
- Type of event being reported (earnings, deal update, regulatory filing, shareholder meeting, etc.)
- Key financial figures (revenue, earnings, deal values) with currencies
- Cross-border regulatory mentions and jurisdiction details
- Deal-related information if M&A (terms, conditions, timeline)
- Shareholder meeting results if applicable (votes, resolutions)
- Material agreements or contracts
- Currency and exchange rate details
"""


def fetch_filing_text(source: str) -> str:
    """Fetch and extract text from a 6-K filing (URL, local file, or PDF)."""
    from .fetch_utils import fetch_text_with_extraction
    return fetch_text_with_extraction(source, EXTRACTION_GUIDANCE)


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
    print("  6-K SUMMARY (Foreign Private Issuer)")
    print("=" * 70)

    print(
        f"\n   Company:  {s.get('company', 'N/A')} ({s.get('ticker', 'N/A')})")
    print(f"   Country:  {s.get('home_country', 'N/A')}")
    print(f"   Type:     {s.get('report_type', 'N/A')}")
    print(f"   Date:     {s.get('filing_date', 'N/A')}")

    print(f"\n📌 L1 | HEADLINE")
    print(f"   {s['L1_headline']}")

    print(f"\n📋 L2 | BRIEF")
    print(f"   {s['L2_brief']}")

    d = s["L3_detailed"]
    print(f"\n📊 L3 | DETAILED")
    print(f"   Event:          {d['event']}")
    print(f"   Key Figures:")
    for f in d.get("key_figures", []):
        print(f"     • {f}")
    print(f"   Market Impact:  {d.get('market_impact', 'N/A')}")
    print(f"   Deal Relevance: {d.get('deal_relevance', 'N/A')}")
    print(f"   Regulatory:     {d.get('regulatory_notes', 'N/A')}")
    print(f"   Cross-Border:   {d.get('cross_border_considerations', 'N/A')}")
    if d.get("risks_flagged"):
        print(f"   Risks:")
        for r in d["risks_flagged"]:
            print(f"     • {r}")

    print("=" * 70)


def export_docx(s: dict, s3_key_suffix: str):
    """Build summary as Word doc, upload to S3 (summary_docx/), return S3 path only."""
    from .fetch_utils import is_empty_value, has_content, add_field
    from .s3_utils import upload_docx_bytes

    ticker = s.get("ticker", "UNKNOWN")
    doc = DocxDocument()

    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    title = doc.add_heading(f"6-K Summary: {ticker}", level=0)
    title.runs[0].font.size = Pt(20)

    meta = doc.add_paragraph()
    add_field(meta, "Company: ", s.get("company"), newline=False)
    add_field(meta, "    Country: ", s.get("home_country"), newline=False)

    meta2 = doc.add_paragraph()
    add_field(meta2, "Filing Date: ", s.get("filing_date"), newline=False)
    add_field(meta2, "    Report Type: ", s.get("report_type"), newline=False)

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

    if not is_empty_value(d.get("event")):
        doc.add_heading("Event", level=2)
        doc.add_paragraph(d.get("event"))

    key_figures = d.get("key_figures")
    if has_content(key_figures):
        doc.add_heading("Key Figures", level=2)
        for fig in key_figures:
            if not is_empty_value(fig):
                doc.add_paragraph(fig, style="List Bullet")

    if not is_empty_value(d.get("market_impact")):
        doc.add_heading("Market Impact", level=2)
        doc.add_paragraph(d.get("market_impact"))

    if not is_empty_value(d.get("deal_relevance")):
        doc.add_heading("Deal Relevance", level=2)
        doc.add_paragraph(d.get("deal_relevance"))

    if not is_empty_value(d.get("regulatory_notes")):
        doc.add_heading("Regulatory Notes", level=2)
        doc.add_paragraph(d.get("regulatory_notes"))

    if not is_empty_value(d.get("cross_border_considerations")):
        doc.add_heading("Cross-Border Considerations", level=2)
        doc.add_paragraph(d.get("cross_border_considerations"))

    risks = d.get("risks_flagged")
    if has_content(risks):
        doc.add_heading("Risks Flagged", level=2)
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

    print(f"Fetching 6-K from: {source}")

    text = fetch_filing_text(source)
    print(f"Extracted {len(text.split())} words of text")

    print("Generating summary via Claude Opus 4.5...")
    result = summarize(text)

    print_summary(result)

    uid = filing_uid(FILING_URL)
    from .s3_utils import upload_json

    s3_json_path, s3_json_url = upload_json(result, f"6k_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    safe_ticker = sanitize_filename_part(result.get("ticker"))
    safe_date = sanitize_date_part(result.get("filing_date"))
    docx_suffix = f"6K_Summary_{safe_ticker}_{safe_date}_{uid}.docx"
    s3_docx_path, s3_docx_url = export_docx(result, docx_suffix)
    print(f"DOCX uploaded to S3: {s3_docx_url}")

    result["s3_docx_path"] = s3_docx_path
    result["s3_docx_url"] = s3_docx_url
    result["s3_json_path"] = s3_json_path
    result["s3_json_url"] = s3_json_url
    return result


if __name__ == "__main__":
    main()
