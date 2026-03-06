"""
6-K Filing Summarizer — Multi-level summaries via Claude API
Usage: python 6k_summary.py
"""

from pathlib import Path

from ._naming import filing_uid

# ──── PASTE YOUR 6-K URL HERE ────
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


SUMMARY_PROMPT = """You are an expert analyst summarizing SEC Form 6-K filings for a merger arbitrage desk.

Form 6-K is the report used by foreign private issuers (non-US companies listed on US exchanges) to disclose material information. It is the international equivalent of the 8-K. These filings often contain earnings results, deal announcements, shareholder meeting results, regulatory updates, or material agreements for companies like those listed as ADRs.

Given the 6-K text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "ticker": "<US ticker or ADR symbol>",
  "company": "<company name>",
  "home_country": "<country of incorporation>",
  "filing_date": "<MM/DD/YY>",
  "report_type": "<Earnings | Deal Announcement | Shareholder Meeting | Regulatory Update | Material Agreement | Other>",

  "L1_headline": "+ <TICKER> – <key event in ≤8 words>. | <date>",

  "L2_brief": "<2-3 sentence summary covering: what happened, key numbers, and what it means for investors>",

  "L3_detailed": {
    "event": "<what happened>",
    "key_figures": ["<financial figures, vote percentages, deal values, share prices>"],
    "market_impact": "<significance for US-listed shares / ADRs>",
    "deal_relevance": "<if M&A related: impact on deal timeline/probability/terms. If not M&A: 'N/A'>",
    "regulatory_notes": "<any regulatory body mentions — local regulators, EU Commission, competition authorities>",
    "cross_border_considerations": "<currency, jurisdiction, or structural notes relevant to US investors>",
    "risks_flagged": ["<any risks, litigation, regulatory issues, FX exposure>"]
  }
}

Rules:
- TONE: State only facts from the filing. Do NOT speculate on motives, interpret what actions "signal" or "suggest", assess confidence levels, or draw conclusions beyond what is explicitly stated. GOOD: "Company suspended earnings calls due to pending transaction." BAD: "Company suspended earnings calls, signaling high confidence in deal completion."
- L1 format MUST be: + <TICKER> – <event>. | <date>
- Use the US-listed ticker or ADR symbol
- Note the home country and any cross-border regulatory considerations
- Extract exact figures, dates, and percentages
- Flag any currency-related details (reporting currency vs USD)
- For merger-related 6-Ks, focus on deal probability impact and cross-border regulatory hurdles

6-K TEXT:
"""


def fetch_filing_text(source: str) -> str:
    """Fetch and extract text from a 6-K filing (URL, local file, or PDF)."""
    from .fetch_utils import fetch_text
    return fetch_text(source)


def summarize(text: str, model: str = "claude-opus-4-6") -> dict:
    """Call Claude API to produce multi-level summary."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set. Set it in .env or Django settings (ANTHROPIC_API_KEY).")
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
    print("  6-K SUMMARY (Foreign Private Issuer)")
    print("=" * 70)

    print(f"\n   Company:  {s.get('company', 'N/A')} ({s.get('ticker', 'N/A')})")
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
    from .s3_utils import upload_docx_bytes

    ticker = s.get("ticker", "UNKNOWN")
    doc = DocxDocument()

    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    title = doc.add_heading(f"6-K Summary: {ticker}", level=0)
    title.runs[0].font.size = Pt(20)

    meta = doc.add_paragraph()
    meta.add_run(f"Company: ").bold = True
    meta.add_run(s.get("company", "N/A"))
    meta.add_run(f"    Country: ").bold = True
    meta.add_run(s.get("home_country", "N/A"))

    date = s.get("filing_date", "")
    meta2 = doc.add_paragraph()
    meta2.add_run(f"Filing Date: ").bold = True
    meta2.add_run(date)
    meta2.add_run(f"    Report Type: ").bold = True
    meta2.add_run(s.get("report_type", "N/A"))

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

    doc.add_heading("Deal Relevance", level=2)
    doc.add_paragraph(d.get("deal_relevance", "N/A"))

    doc.add_heading("Regulatory Notes", level=2)
    doc.add_paragraph(d.get("regulatory_notes", "N/A"))

    doc.add_heading("Cross-Border Considerations", level=2)
    doc.add_paragraph(d.get("cross_border_considerations", "N/A"))

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

    ticker = result.get("ticker", "UNKNOWN")
    date = result.get("filing_date", "")
    safe_ticker = re.sub(r'[^\w\-\.]', '_', ticker)
    safe_date = date.replace("/", "-")
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
