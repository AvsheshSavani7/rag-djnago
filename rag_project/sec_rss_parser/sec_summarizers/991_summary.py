"""
Exhibit 99.1 Filing Summarizer — Multi-level summaries via Claude API
Usage: python 991_summary.py
"""

import anthropic
import re
import json
import sys
import os
import io
from pathlib import Path
from ._naming import filing_uid

# ──── PASTE YOUR EXHIBIT 99.1 URL HERE ────
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


SUMMARY_PROMPT = """You are an expert analyst summarizing SEC Exhibit 99.1 filings (press releases filed as exhibits to 8-Ks) for a merger arbitrage desk.

Exhibit 99.1 filings typically contain press releases attached to 8-K filings. These often announce earnings results, material agreements, deal updates, leadership changes, or other corporate events.

Given the Exhibit 99.1 text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "ticker": "<ticker symbol>",
  "company": "<company name>",
  "filing_date": "<MM/DD/YY>",
  "exhibit_type": "<Earnings Release | Deal Announcement | Deal Update | Leadership Change | Guidance Update | Asset Sale | Restructuring | Other>",

  "L1_headline": "+ <TICKER> – <key event in ≤8 words>. | <date>",

   "L2_brief": "<2-3 sentence summary covering: what the press release announces, key numbers, and key figures>",

  "L3_detailed": {
    "event": "<what was announced>",
    "key_figures": ["<revenue, EPS, deal value, per-share price, guidance numbers>"],
    "market_impact": "<market-related facts stated in the filing>",
    "forward_guidance": "<any forward-looking statements, updated guidance, or timeline changes>",
   "deal_relevance": "<if M&A related: deal-related facts stated in the filing. If not M&A: 'N/A'>",
    "risks_flagged": ["<any risks, litigation, regulatory issues, or cautionary statements>"]
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
- Distinguish between earnings releases, deal announcements, and other press release types
- Extract exact dollar amounts, percentages, per-share figures, and dates
- For earnings: capture revenue, EPS (GAAP and non-GAAP), and any guidance changes
- For deal-related exhibits: focus on deal terms, conditions, and probability impact
- Note any forward-looking statement caveats or risk factors mentioned

EXHIBIT 99.1 TEXT:
"""

EXTRACTION_GUIDANCE = """This is an Exhibit 99.1 filing (press release or document filed as exhibit).
Extract the following:
- Type of announcement (earnings, deal announcement, deal update, leadership change, guidance, etc.)
- Key financial figures: revenue, EPS, net income, guidance numbers
- Deal terms if M&A-related (price, premium, conditions, timeline)
- Forward guidance or outlook statements
- Any risk factors or cautionary language
- Quotes from executives about strategy or deal rationale"""


def fetch_filing_text(source: str) -> str:
    """Fetch and extract text from an Exhibit 99.1 filing (URL, local file, or PDF)."""
    from .fetch_utils import fetch_text_with_extraction
    return fetch_text_with_extraction(source, EXTRACTION_GUIDANCE)


def summarize(text: str, model: str = "claude-opus-4-6") -> dict:
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
    print("  EXHIBIT 99.1 SUMMARY")
    print("=" * 70)

    print(
        f"\n   Company: {s.get('company', 'N/A')} ({s.get('ticker', 'N/A')})")
    print(f"   Type:    {s.get('exhibit_type', 'N/A')}")
    print(f"   Date:    {s.get('filing_date', 'N/A')}")

    # L1 — Headline
    print(f"\n📌 L1 | HEADLINE")
    print(f"   {s['L1_headline']}")

    # L2 — Brief
    print(f"\n📋 L2 | BRIEF")
    print(f"   {s['L2_brief']}")

    # L3 — Detailed
    d = s["L3_detailed"]
    print(f"\n📊 L3 | DETAILED")
    print(f"   Event:          {d['event']}")
    print(f"   Key Figures:")
    for f in d.get("key_figures", []):
        print(f"     • {f}")
    print(f"   Market Impact:  {d.get('market_impact', 'N/A')}")
    print(f"   Fwd Guidance:   {d.get('forward_guidance', 'N/A')}")
    print(f"   Deal Relevance: {d.get('deal_relevance', 'N/A')}")
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

    title = doc.add_heading(f"Exhibit 99.1 Summary: {ticker}", level=0)
    title.runs[0].font.size = Pt(20)

    meta = doc.add_paragraph()
    meta.add_run(f"Company: ").bold = True
    meta.add_run(s.get("company", "N/A"))
    date = s.get("filing_date", "")
    meta.add_run(f"    Filing Date: ").bold = True
    meta.add_run(date)
    meta.add_run(f"    Type: ").bold = True
    meta.add_run(s.get("exhibit_type", "N/A"))

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

    doc.add_heading("Event", level=2)
    doc.add_paragraph(d["event"])

    doc.add_heading("Key Figures", level=2)
    for fig in d.get("key_figures", []):
        doc.add_paragraph(fig, style="List Bullet")

    doc.add_heading("Market Impact", level=2)
    doc.add_paragraph(d.get("market_impact", "N/A"))

    doc.add_heading("Forward Guidance", level=2)
    doc.add_paragraph(d.get("forward_guidance", "N/A"))

    doc.add_heading("Deal Relevance", level=2)
    doc.add_paragraph(d.get("deal_relevance", "N/A"))

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

    print(f"Fetching Exhibit 99.1 from: {source}")

    text = fetch_filing_text(source)
    print(f"Extracted {len(text.split())} words of text")

    print("Generating summary via Claude Opus 4.5...")
    result = summarize(text)

    print_summary(result)

    uid = filing_uid(FILING_URL)
    from .s3_utils import upload_json

    s3_json_path, s3_json_url = upload_json(result, f"991_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    ticker = result.get("ticker", "UNKNOWN")
    date = result.get("filing_date", "")
    safe_ticker = re.sub(r'[^\w\-\.]', '_', ticker)
    safe_date = date.replace("/", "-")
    docx_suffix = f"991_Summary_{safe_ticker}_{safe_date}_{uid}.docx"
    s3_docx_path, s3_docx_url = export_docx(result, docx_suffix)
    print(f"DOCX uploaded to S3: {s3_docx_url}")

    result["s3_docx_path"] = s3_docx_path
    result["s3_docx_url"] = s3_docx_url
    result["s3_json_path"] = s3_json_path
    result["s3_json_url"] = s3_json_url
    return result


if __name__ == "__main__":
    main()
