"""
8-K Filing Summarizer — Multi-level summaries via Claude API
Usage: python summarize_8k.py
"""

import anthropic
import re
import json
import sys
import os
import io
from pathlib import Path
from ._naming import filing_uid
from ._deal_context import inject_deal_context

# ──── PASTE YOUR 8-K URL HERE ────
FILING_URL = "https://www.sec.gov/Archives/edgar/data/1434868/000110465925097988/tm2527818-3_424b5.htm"
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


SUMMARY_PROMPT = """You are an expert analyst summarizing SEC 8-K filings for a merger arbitrage desk.

Given the 8-K text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "ticker": "<ticker symbol as stated in the filing, or null if not stated>",
  "filing_date": "<MM/DD/YY>",
  "items_reported": ["<Item numbers, e.g. Item 5.07, Item 8.01>"],

  "L1_headline": "<ticker> – <key event in ≤8 words>. | <date>",

   "L2_brief": "<2-3 sentence summary covering: what happened, key numbers, and current deal status if applicable>",

  "L3_detailed": {
    "event": "<what happened>",
    "key_figures": ["<vote %, dollar amounts, dates, conditions>"],
    "deal_implications": "<deal-related facts stated in the filing (timeline updates, condition status, regulatory filings)>",
    "remaining_conditions": ["<conditions to closing listed in the filing that remain outstanding>"],
    "risks_flagged": ["<any risks, litigation, regulatory issues>"]
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
- For merger-related 8-Ks, extract stated deal terms, conditions, and timeline updates
- Extract exact vote percentages, dollar figures, dates
- Flag any conditions precedent still outstanding
- Note any litigation or regulatory mentions

8-K TEXT:
"""

EXTRACTION_GUIDANCE = """This is an 8-K current report filing.
Extract the following:
- Item numbers reported (e.g., Item 1.01, Item 5.07, Item 8.01)
- The full text of each reported Item
- Vote results with exact percentages if present
- Deal-related disclosures: merger agreement terms, closing conditions, regulatory updates, timeline changes
- Any dollar amounts, share counts, or financial figures
- Litigation or regulatory mentions
- Forward-looking statements about pending transactions
- Any exhibits referenced and their descriptions"""


def fetch_8k_text(source: str) -> str:
    """Fetch and extract text from an 8-K filing (URL, local file, or PDF)."""
    from .fetch_utils import fetch_text_with_extraction
    return fetch_text_with_extraction(source, extraction_guidance=EXTRACTION_GUIDANCE)


def summarize(text: str, model: str = "claude-opus-4-6") -> dict:
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
    print("  8-K SUMMARY")
    print("=" * 70)

    print(f"\n📌 L1 | HEADLINE")
    print(f"   {s['L1_headline']}")

    print(f"\n📋 L2 | BRIEF")
    print(f"   {s['L2_brief']}")

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
    from .fetch_utils import is_empty_value, has_content
    from .s3_utils import upload_docx_bytes

    ticker = s.get("ticker", "UNKNOWN")
    date = s.get("filing_date", "")

    doc = DocxDocument()

    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    title = doc.add_heading(f"8-K Summary: {ticker}", level=0)
    title.runs[0].font.size = Pt(20)

    meta = doc.add_paragraph()
    meta.add_run("Filing Date: ").bold = True
    meta.add_run(date)
    items = s.get("items_reported", [])
    if has_content(items):
        meta.add_run("    Items: ").bold = True
        meta.add_run(", ".join(items))

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
        doc.add_paragraph(d["event"])

    key_figures = d.get("key_figures", [])
    if has_content(key_figures):
        doc.add_heading("Key Figures", level=2)
        for fig in key_figures:
            if not is_empty_value(fig):
                doc.add_paragraph(fig, style="List Bullet")

    if not is_empty_value(d.get("deal_implications")):
        doc.add_heading("Deal Implications", level=2)
        doc.add_paragraph(d["deal_implications"])

    remaining = d.get("remaining_conditions", [])
    if has_content(remaining):
        doc.add_heading("Remaining Conditions", level=2)
        for c in remaining:
            if not is_empty_value(c):
                doc.add_paragraph(c, style="List Bullet")

    risks = d.get("risks_flagged", [])
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

    print(f"Fetching 8-K from: {source}")
    if isinstance(source, list) and len(source) > 1:
        print(f"   ({len(source)} URLs will be combined for extraction)")

    text = fetch_8k_text(source)
    print(f"Extracted {len(text.split())} words of text")

    print("Generating summary via Claude Opus...")
    result = summarize(text)

    print_summary(result)

    uid = filing_uid(FILING_URL)
    from .s3_utils import upload_json

    s3_json_path, s3_json_url = upload_json(result, f"8k_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    ticker = result.get("ticker", "UNKNOWN")
    date = result.get("filing_date", "")
    safe_ticker = re.sub(r"[^\w\-\.]", "_", str(ticker))
    safe_date = str(date).replace("/", "-")
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
