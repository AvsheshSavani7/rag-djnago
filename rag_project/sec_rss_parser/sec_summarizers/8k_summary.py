"""
8-K Filing Summarizer — Multi-level summaries via Claude API
Usage: python summarize_8k.py
"""

from pathlib import Path
from ._naming import filing_uid

# ──── PASTE YOUR 8-K URL HERE ────
FILING_URL = "https://www.sec.gov/Archives/edgar/data/1434868/000110465925097988/tm2527818-3_424b5.htm"
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


SUMMARY_PROMPT = """You are an expert analyst summarizing SEC 8-K filings for a merger arbitrage desk.

Given the 8-K text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "ticker": "<ticker symbol>",
  "filing_date": "<MM/DD/YY>",
  "items_reported": ["<Item numbers, e.g. Item 5.07, Item 8.01>"],
  
  "L1_headline": "<ticker> – <key event in ≤8 words>. | <date>",
  
  "L2_brief": "<2-3 sentence summary covering: what happened, key numbers, what it means for the deal>",
  
  "L3_detailed": {
    "event": "<what happened>",
    "key_figures": ["<vote %, dollar amounts, dates, conditions>"],
    "deal_implications": "<impact on deal timeline/probability>",
    "remaining_conditions": ["<what still needs to happen>"],
    "risks_flagged": ["<any risks, litigation, regulatory issues>"]
  }
}

Rules:
- TONE: State only facts from the filing. Do NOT speculate on motives, interpret what actions "signal" or "suggest", assess confidence levels, or draw conclusions beyond what is explicitly stated. GOOD: "Company suspended earnings calls due to pending transaction." BAD: "Company suspended earnings calls, signaling high confidence in deal completion."
- L1 format MUST be: + <TICKER> – <event>. | <date>
- For merger-related 8-Ks, focus on deal probability impact
- Extract exact vote percentages, dollar figures, dates
- Flag any conditions precedent still outstanding
- Note any litigation or regulatory mentions

8-K TEXT:
"""


def fetch_8k_text(source: str) -> str:
    """Fetch and extract text from an 8-K filing (URL, local file, or PDF)."""
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
    # Strip markdown fences if present
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    
    return json.loads(raw)


def print_summary(s: dict):
    """Pretty-print the multi-level summary."""
    print("\n" + "=" * 70)
    print("  8-K SUMMARY")
    print("=" * 70)
    
    # L1 — Headline
    print(f"\n📌 L1 | HEADLINE")
    print(f"   {s['L1_headline']}")
    
    # L2 — Brief
    print(f"\n📋 L2 | BRIEF")
    print(f"   {s['L2_brief']}")
    
    # L3 — Detailed
    d = s["L3_detailed"]
    print(f"\n📊 L3 | DETAILED")
    print(f"   Event:       {d['event']}")
    print(f"   Key Figures:")
    for f in d["key_figures"]:
        print(f"     • {f}")
    print(f"   Deal Impact: {d['deal_implications']}")
    if d.get("remaining_conditions"):
        print(f"   Remaining Conditions:")
        for c in d["remaining_conditions"]:
            print(f"     • {c}")
    if d.get("risks_flagged"):
        print(f"   Risks:")
        for r in d["risks_flagged"]:
            print(f"     • {r}")
    
    print(f"\n   Items: {', '.join(s.get('items_reported', []))}")
    print("=" * 70)


def export_docx(s: dict, s3_key_suffix: str):
    """Build summary as Word doc, upload to S3 (summary_docx/), return (s3_path, s3_url)."""
    from .s3_utils import upload_docx_bytes

    ticker = s.get("ticker", "UNKNOWN")
    doc = DocxDocument()

    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    title = doc.add_heading(f"8-K Summary: {ticker}", level=0)
    title.runs[0].font.size = Pt(20)

    date = s.get("filing_date", "")
    meta = doc.add_paragraph()
    meta.add_run(f"Filing Date: ").bold = True
    meta.add_run(date)
    meta.add_run(f"    Items: ").bold = True
    meta.add_run(", ".join(s.get("items_reported", [])))

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

    doc.add_heading("Deal Implications", level=2)
    doc.add_paragraph(d.get("deal_implications", "N/A"))

    if d.get("remaining_conditions"):
        doc.add_heading("Remaining Conditions", level=2)
        for c in d["remaining_conditions"]:
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
    
    print(f"Fetching 8-K from: {source}")
    
    text = fetch_8k_text(source)
    print(f"Extracted {len(text.split())} words of text")
    
    print("Generating summary via Claude Opus 4.5...")
    result = summarize(text)
    
    print_summary(result)

    uid = filing_uid(FILING_URL)
    from .s3_utils import upload_json

    s3_json_path, s3_json_url = upload_json(result, f"8k_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    ticker = result.get("ticker", "UNKNOWN")
    date = result.get("filing_date", "")
    safe_ticker = re.sub(r'[^\w\-\.]', '_', ticker)
    safe_date = date.replace("/", "-")
    docx_suffix = f"8K_Summary_{safe_ticker}_{safe_date}_{uid}.docx"
    s3_docx_path, s3_docx_url = export_docx(result, docx_suffix)
    print(f"DOCX uploaded to S3: {s3_docx_url}")

    result["s3_docx_path"] = s3_docx_path
    result["s3_docx_url"] = s3_docx_url
    result["s3_json_path"] = s3_json_path
    result["s3_json_url"] = s3_json_url
    return result


if __name__ == "__main__":
    main()