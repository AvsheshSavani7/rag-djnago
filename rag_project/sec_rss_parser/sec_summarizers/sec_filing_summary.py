"""
Generic SEC Filing Summarizer (Catch-All) — Multi-level summaries via Claude API
Usage: python sec_filing_summary.py

Use this script for any SEC filing type not covered by a dedicated summarizer
(e.g., 10-K, 10-Q, S-1, S-4, DEF 14A, DEFM14A, SC TO, SC 14D-9, 8-A, etc.)
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

# ──── PASTE YOUR SEC FILING URL HERE ────
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


SUMMARY_PROMPT = """You are an expert analyst summarizing SEC filings for a merger arbitrage desk.

You may receive ANY type of SEC filing — 10-K, 10-Q, S-1, S-4, DEF 14A, DEFM14A, SC TO-T, SC 14D-9, 8-A, EFFECT, ARS, or any other form. Your job is to identify the filing type, extract the most important information, and frame it through a merger arbitrage lens where relevant.

Given the SEC filing text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "ticker": "<ticker symbol, if identifiable>",
  "company": "<company name>",
  "filing_type": "<detected filing type — e.g., 10-K, DEF 14A, S-4, SC TO-T, etc.>",
  "filing_date": "<MM/DD/YY>",

  "L1_headline": "+ <TICKER> – <key takeaway in ≤8 words>. | <date>  (omit the ticker prefix if ticker is null)",

"L2_brief": "<2-3 sentence summary covering: what this filing is, the most important information it contains, and stated purpose>",


  "L3_detailed": {
    "filing_purpose": "<why this filing was made — what event or requirement triggered it>",
    "key_information": ["<the 3-5 most important facts, figures, or disclosures from the filing>"],
    "financial_highlights": ["<any key financial figures — revenue, earnings, deal values, share prices>"],
   "deal_relevance": "<if M&A related: deal-related facts stated in the filing, if any>",
    "regulatory_mentions": "<any regulatory bodies, approvals, investigations, or compliance matters mentioned>",
    "timeline_or_dates": ["<important dates mentioned — close dates, meeting dates, deadlines, effective dates>"],
    "conditions_or_requirements": ["<any conditions precedent, requirements, or contingencies>"],
    "risks_flagged": ["<any risks, litigation, regulatory issues, material uncertainties>"]
  }
}

Rules:
- CRITICAL — FACTS ONLY: Every statement in your summary must be directly traceable to the filing text. Report ONLY what the document says. Do NOT add analysis, assess significance, interpret motives, predict outcomes, evaluate probability, or editorialize. Do NOT state what is "not disclosed" or "not mentioned" — simply omit fields where the filing is silent. If the filing does not say it, do not write it.
  GOOD: "CADE requested revenue data for 2021-2025 across four markets."
  BAD: "The broad scope of information requested indicates potentially detailed competitive analysis ahead."
  GOOD: "The offer expires June 10, 2026."
  BAD: "This tight timeline may create pressure on shareholders to tender quickly."
- PRECISION: Use the filing's exact terminology for legal, regulatory, and financial terms. Do NOT paraphrase in ways that broaden or narrow the stated meaning. GOOD: "All 14 Pennsylvania PUC hearings have concluded." BAD: "Regulatory proceedings concluded in Pennsylvania."
- L1 format MUST be: + <TICKER> – <takeaway>. | <date>
- FIRST identify the filing type from the document content — this determines how to read it
- Adapt your focus based on filing type:
  * Proxy (DEF 14A, DEFM14A): focus on vote matters, board recommendations, deal terms
  * Registration (S-1, S-4): focus on offering/deal terms, risk factors, financial statements
  * Tender offer (SC TO, SC 14D-9): focus on offer price, conditions, board recommendation
  * Periodic reports (10-K, 10-Q): focus on financial performance, risk factors, M&A disclosures
  * Other: extract the most material information and frame through M&A/arb lens
- Extract exact dollar amounts, percentages, share counts, and dates
- Extract any stated references to pending or completed M&A transactions
- Flag any material risks, litigation, or regulatory developments

SEC FILING TEXT:
"""

EXTRACTION_GUIDANCE = """This is a generic SEC filing (catch-all for types without a dedicated summarizer).
Extract:
- Filing type identification and purpose
- All deal terms (prices, exchange ratios, premiums, conditions)
- Regulatory approvals mentioned (required, obtained, pending) — every jurisdiction
- Timeline information (closing dates, deadlines, meeting dates)
- Risk factors and litigation mentions
- Board recommendations and fairness opinions
- Key financial figures (revenue, earnings, deal value)
- Any material conditions precedent
- Vote results if present
- Background of any transaction"""


def fetch_filing_text(source: str) -> str:
    """Fetch and extract text from any SEC filing (URL, local file, or PDF)."""
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
        # 1500 was too low for 10-K/10-Q; truncation caused "Unterminated string" JSON error
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": inject_deal_context(SUMMARY_PROMPT, DEAL_CONTEXT) + "\n\n" + text
        }]
    )

    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # If response was truncated (stop_reason != "end_turn"), try to close the JSON
    if msg.stop_reason != "end_turn":
        # Attempt to salvage truncated JSON by closing open structures
        if raw.count('{') > raw.count('}'):
            raw += '"' + '}' * (raw.count('{') - raw.count('}'))
        if raw.count('[') > raw.count(']'):
            raw += ']' * (raw.count('[') - raw.count(']'))

    # JSONDecodeError "Unterminated string" = Claude response truncated (max_tokens) or unescaped " in a string
    return json.loads(raw)


def print_summary(s: dict):
    """Pretty-print the multi-level summary."""
    print("\n" + "=" * 70)
    print(f"  SEC FILING SUMMARY — {s.get('filing_type', 'UNKNOWN TYPE')}")
    print("=" * 70)

    print(
        f"\n   Company:  {s.get('company', 'N/A')} ({s.get('ticker', 'N/A')})")
    print(f"   Type:     {s.get('filing_type', 'N/A')}")
    print(f"   Date:     {s.get('filing_date', 'N/A')}")

    print(f"\n📌 L1 | HEADLINE")
    print(f"   {s['L1_headline']}")

    print(f"\n📋 L2 | BRIEF")
    print(f"   {s['L2_brief']}")

    d = s["L3_detailed"]
    print(f"\n📊 L3 | DETAILED")
    print(f"   Purpose:        {d.get('filing_purpose', 'N/A')}")
    if d.get("key_information"):
        print(f"   Key Information:")
        for k in d["key_information"]:
            print(f"     • {k}")
    if d.get("financial_highlights"):
        print(f"   Financial Highlights:")
        for f in d["financial_highlights"]:
            print(f"     • {f}")
    print(f"   Deal Relevance: {d.get('deal_relevance', 'N/A')}")
    print(f"   Regulatory:     {d.get('regulatory_mentions', 'N/A')}")
    if d.get("timeline_or_dates"):
        print(f"   Key Dates:")
        for t in d["timeline_or_dates"]:
            print(f"     • {t}")
    if d.get("conditions_or_requirements"):
        print(f"   Conditions:")
        for c in d["conditions_or_requirements"]:
            print(f"     • {c}")
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
    filing_type = s.get("filing_type", "SEC")
    date = s.get("filing_date", "")
    doc = DocxDocument()

    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    title = doc.add_heading(f"{filing_type} Summary: {ticker}", level=0)
    title.runs[0].font.size = Pt(20)

    meta = doc.add_paragraph()
    add_field(meta, "Company: ", s.get("company"), newline=False)
    meta.add_run("    ")
    add_field(meta, "Filing Type: ", filing_type, newline=False)

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

    if not is_empty_value(d.get("filing_purpose")):
        doc.add_heading("Filing Purpose", level=2)
        doc.add_paragraph(d.get("filing_purpose"))

    items = d.get("key_information")
    if has_content(items):
        doc.add_heading("Key Information", level=2)
        for k in items:
            if not is_empty_value(k):
                doc.add_paragraph(k, style="List Bullet")

    items = d.get("financial_highlights")
    if has_content(items):
        doc.add_heading("Financial Highlights", level=2)
        for f in items:
            if not is_empty_value(f):
                doc.add_paragraph(f, style="List Bullet")

    if not is_empty_value(d.get("deal_relevance")):
        doc.add_heading("Deal Relevance", level=2)
        doc.add_paragraph(d.get("deal_relevance"))

    if not is_empty_value(d.get("regulatory_mentions")):
        doc.add_heading("Regulatory Mentions", level=2)
        doc.add_paragraph(d.get("regulatory_mentions"))

    items = d.get("timeline_or_dates")
    if has_content(items):
        doc.add_heading("Key Dates & Timeline", level=2)
        for t in items:
            if not is_empty_value(t):
                doc.add_paragraph(t, style="List Bullet")

    items = d.get("conditions_or_requirements")
    if has_content(items):
        doc.add_heading("Conditions & Requirements", level=2)
        for c in items:
            if not is_empty_value(c):
                doc.add_paragraph(c, style="List Bullet")

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

    print(f"Fetching SEC filing from: {source}")

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
        result, f"sec_filing_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    safe_type = sanitize_filename_part(result.get("filing_type"), "SEC")
    safe_ticker = sanitize_filename_part(result.get("ticker"))
    safe_date = sanitize_date_part(result.get("filing_date"))
    docx_suffix = f"{safe_type}_Summary_{safe_ticker}_{safe_date}_{uid}.docx"
    s3_docx_path, s3_docx_url = export_docx(result, docx_suffix)
    print(f"DOCX uploaded to S3: {s3_docx_url}")

    result["s3_docx_path"] = s3_docx_path
    result["s3_docx_url"] = s3_docx_url
    result["s3_json_path"] = s3_json_path
    result["s3_json_url"] = s3_json_url
    return result


if __name__ == "__main__":
    main()
