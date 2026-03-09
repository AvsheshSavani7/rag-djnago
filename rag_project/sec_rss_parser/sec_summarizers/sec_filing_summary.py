"""
Generic SEC Filing Summarizer (Catch-All) — Multi-level summaries via Claude API
Usage: python sec_filing_summary.py

Use this script for any SEC filing type not covered by a dedicated summarizer
(e.g., 10-K, 10-Q, S-1, S-4, DEF 14A, DEFM14A, SC TO, SC 14D-9, 8-A, etc.)
"""

from pathlib import Path
from ._naming import filing_uid

# ──── PASTE YOUR SEC FILING URL HERE ────
FILING_URL = ""
# ──── OUTPUT FOLDER ────
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "Output Summaries"
# ─────────────────────────────────

import io
import os
import sys
import json
import re
import anthropic

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

  "L1_headline": "+ <TICKER> – <key takeaway in ≤8 words>. | <date>",

  "L2_brief": "<2-3 sentence summary covering: what this filing is, the most important information it contains, and any M&A relevance>",

  "L3_detailed": {
    "filing_purpose": "<why this filing was made — what event or requirement triggered it>",
    "key_information": ["<the 3-5 most important facts, figures, or disclosures from the filing>"],
    "financial_highlights": ["<any key financial figures — revenue, earnings, deal values, share prices>"],
    "deal_relevance": "<if M&A related: specific impact on deal timeline, probability, terms, or structure. If not directly M&A: any indirect relevance to pending deals or corporate strategy>",
    "regulatory_mentions": "<any regulatory bodies, approvals, investigations, or compliance matters mentioned>",
    "timeline_or_dates": ["<important dates mentioned — close dates, meeting dates, deadlines, effective dates>"],
    "conditions_or_requirements": ["<any conditions precedent, requirements, or contingencies>"],
    "risks_flagged": ["<any risks, litigation, regulatory issues, material uncertainties>"]
  }
}

Rules:
- TONE: State only facts from the filing. Do NOT speculate on motives, interpret what actions "signal" or "suggest", assess confidence levels, or draw conclusions beyond what is explicitly stated. GOOD: "Company suspended earnings calls due to pending transaction." BAD: "Company suspended earnings calls, signaling high confidence in deal completion."
- L1 format MUST be: + <TICKER> – <takeaway>. | <date>
- FIRST identify the filing type from the document content — this determines how to read it
- Adapt your focus based on filing type:
  * Proxy (DEF 14A, DEFM14A): focus on vote matters, board recommendations, deal terms
  * Registration (S-1, S-4): focus on offering/deal terms, risk factors, financial statements
  * Tender offer (SC TO, SC 14D-9): focus on offer price, conditions, board recommendation
  * Periodic reports (10-K, 10-Q): focus on financial performance, risk factors, M&A disclosures
  * Other: extract the most material information and frame through M&A/arb lens
- Extract exact dollar amounts, percentages, share counts, and dates
- Always assess: "Does this filing affect any pending or potential M&A transaction?"
- Flag any material risks, litigation, or regulatory developments

SEC FILING TEXT:
"""


def fetch_filing_text(source: str) -> str:
    """Fetch and extract text from any SEC filing (URL, local file, or PDF)."""
    from .fetch_utils import fetch_text
    return fetch_text(source)


def summarize(text: str, model: str = "claude-opus-4-6") -> dict:
    """Call Claude API to produce multi-level summary."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set. Set it in .env or Django settings (ANTHROPIC_API_KEY).")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    msg = client.messages.create(
        model=model,
        max_tokens=4096,  # 1500 was too low for 10-K/10-Q; truncation caused "Unterminated string" JSON error
        messages=[{
            "role": "user",
            "content": SUMMARY_PROMPT + "\n\n" + text
        }]
    )

    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # JSONDecodeError "Unterminated string" = Claude response truncated (max_tokens) or unescaped " in a string
    return json.loads(raw)


def print_summary(s: dict):
    """Pretty-print the multi-level summary."""
    print("\n" + "=" * 70)
    print(f"  SEC FILING SUMMARY — {s.get('filing_type', 'UNKNOWN TYPE')}")
    print("=" * 70)

    print(f"\n   Company:  {s.get('company', 'N/A')} ({s.get('ticker', 'N/A')})")
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
    meta.add_run(f"Company: ").bold = True
    meta.add_run(s.get("company", "N/A"))
    meta.add_run(f"    Filing Type: ").bold = True
    meta.add_run(filing_type)

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

    doc.add_heading("Filing Purpose", level=2)
    doc.add_paragraph(d.get("filing_purpose", "N/A"))

    if d.get("key_information"):
        doc.add_heading("Key Information", level=2)
        for k in d["key_information"]:
            doc.add_paragraph(k, style="List Bullet")

    if d.get("financial_highlights"):
        doc.add_heading("Financial Highlights", level=2)
        for f in d["financial_highlights"]:
            doc.add_paragraph(f, style="List Bullet")

    doc.add_heading("Deal Relevance", level=2)
    doc.add_paragraph(d.get("deal_relevance", "N/A"))

    doc.add_heading("Regulatory Mentions", level=2)
    doc.add_paragraph(d.get("regulatory_mentions", "N/A"))

    if d.get("timeline_or_dates"):
        doc.add_heading("Key Dates & Timeline", level=2)
        for t in d["timeline_or_dates"]:
            doc.add_paragraph(t, style="List Bullet")

    if d.get("conditions_or_requirements"):
        doc.add_heading("Conditions & Requirements", level=2)
        for c in d["conditions_or_requirements"]:
            doc.add_paragraph(c, style="List Bullet")

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

    print(f"Fetching SEC filing from: {source}")

    text = fetch_filing_text(source)
    print(f"Extracted {len(text.split())} words of text")

    print("Generating summary via Claude Opus 4.5...")
    result = summarize(text)

    print_summary(result)

    uid = filing_uid(FILING_URL)
    from .s3_utils import upload_json

    s3_json_path, s3_json_url = upload_json(result, f"sec_filing_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    filing_type = result.get("filing_type", "SEC")
    safe_type = re.sub(r'[^\w\-\.]', '_', filing_type)
    ticker = result.get("ticker", "UNKNOWN")
    date = result.get("filing_date", "")
    safe_ticker = re.sub(r'[^\w\-\.]', '_', ticker)
    safe_date = date.replace("/", "-")
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
