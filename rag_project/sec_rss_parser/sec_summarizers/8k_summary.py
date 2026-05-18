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

# ──── PASTE YOUR 8-K URL HERE ────
FILING_URL = "https://www.sec.gov/Archives/edgar/data/1434868/000110465925097988/tm2527818-3_424b5.htm"
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
  "ticker": "<ticker symbol>",
  "company": "<company name>",
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


COMBINED_PROMPT = """You are an expert analyst summarizing SEC filings for a merger arbitrage desk.

You are given an 8-K filing followed by its Exhibit 99.1 press release. Analyze BOTH documents together and produce ONE unified summary — do not duplicate information between sections.

Given the combined filing text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "ticker": "<ticker symbol>",
  "company": "<company name>",
  "filing_date": "<MM/DD/YY>",
  "items_reported": ["<Item numbers from the 8-K, e.g. Item 7.01, Item 9.01>"],
  "exhibit_type": "<Earnings Release | Deal Announcement | Deal Update | Leadership Change | Guidance Update | Asset Sale | Restructuring | Other>",

  "L1_headline": "+ <TICKER> – <key event in ≤8 words>. | <date>",

  "L2_brief": "<2-3 sentence summary covering: what the 8-K discloses, what the press release announces, key numbers, and deal/market significance>",

  "L3_detailed": {
    "event": "<what was announced across the 8-K and press release>",
    "key_figures": ["<vote %, dollar amounts, revenue, EPS, deal values, per-share prices, dates>"],
    "deal_implications": "<impact on deal timeline/probability if M&A related, otherwise N/A>",
    "market_impact": "<why this matters to investors — deal probability, valuation, earnings trajectory>",
    "forward_guidance": "<any forward-looking statements, updated guidance, or timeline changes>",
    "remaining_conditions": ["<what still needs to happen for any deal or stated goal>"],
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
- Produce ONE unified summary — synthesize both documents, do not repeat the same facts twice
- Extract exact vote percentages, dollar figures, per-share amounts, and dates
- For merger-related filings: focus on deal probability, timeline, and conditions
- For earnings releases: capture revenue, EPS (GAAP and non-GAAP), and guidance changes
- Flag any conditions precedent still outstanding

8-K AND EXHIBIT 99.1 TEXT:
"""


def fetch_8k_text(source: str) -> str:
    """Fetch and extract text from an 8-K filing (URL, local file, or PDF)."""
    from .fetch_utils import fetch_text_with_extraction
    return fetch_text_with_extraction(source, extraction_guidance=EXTRACTION_GUIDANCE)


def find_exhibit_991_url(source_url: str):
    """Find an Exhibit 99.1 URL from the EDGAR filing index for the given 8-K URL.

    Uses the structured filing index page (accession-index.html) which has an explicit
    Type column (EX-99.1), so it works even when exhibit filenames have opaque names.
    Returns the URL string or None.
    """
    if not source_url.startswith("http") or "edgar/data" not in source_url:
        return None

    parts = source_url.rsplit("/", 1)
    if len(parts) < 2:
        return None

    folder_url = parts[0]
    folder_name = folder_url.rsplit("/", 1)[-1]

    headers = {"User-Agent": "ResearchBot/1.0 (research@example.com)"}

    index_url = None
    if len(folder_name) == 18 and folder_name.isdigit():
        accession = f"{folder_name[:10]}-{folder_name[10:12]}-{folder_name[12:]}"
        index_url = f"{folder_url}/{accession}-index.html"

    if index_url:
        try:
            resp = requests.get(index_url, headers=headers, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            for row in soup.find_all("tr"):
                cells = row.find_all(["td", "th"])
                cell_texts = [c.get_text(strip=True).upper() for c in cells]
                if "EX-99.1" in cell_texts:
                    a = row.find("a", href=True)
                    if a:
                        href = a["href"]
                        if href.startswith("/"):
                            return "https://www.sec.gov" + href
                        if href.startswith("http"):
                            return href
                        return folder_url + "/" + href
        except Exception:
            pass

    try:
        resp = requests.get(folder_url + "/", headers=headers, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            link_text = a.get_text(strip=True).lower()
            if (re.search(r'ex[\-_]?99[\-_\.]?1', href.lower()) or
                    "exhibit 99.1" in link_text or "ex-99.1" in link_text):
                if href.startswith("/"):
                    return "https://www.sec.gov" + href
                if href.startswith("http"):
                    return href
                return folder_url + "/" + href
    except Exception:
        pass

    return None


def summarize(text: str, model: str = "claude-opus-4-6", prompt: str = None) -> dict:
    """Call Claude API to produce multi-level summary."""
    if prompt is None:
        prompt = SUMMARY_PROMPT

    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Set it in .env or Django settings (ANTHROPIC_API_KEY).")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    msg = client.messages.create(
        model=model,
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": prompt + "\n\n" + text
        }]
    )

    raw = msg.content[0].text.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


def print_summary(s: dict):
    """Pretty-print the multi-level summary."""
    is_combined = "exhibit_type" in s
    header = "  8-K + EXHIBIT 99.1 COMBINED SUMMARY" if is_combined else "  8-K SUMMARY"

    print("\n" + "=" * 70)
    print(header)
    print("=" * 70)

    if s.get("company"):
        print(
            f"\n   Company:  {s.get('company', 'N/A')} ({s.get('ticker', 'N/A')})")
    if is_combined:
        print(f"   Type:     {s.get('exhibit_type', 'N/A')}")
    print(f"   Date:     {s.get('filing_date', 'N/A')}")

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
    print(f"   Deal Impact: {d.get('deal_implications', 'N/A')}")
    if d.get("market_impact"):
        print(f"   Market Impact: {d['market_impact']}")
    if d.get("forward_guidance"):
        print(f"   Fwd Guidance:  {d['forward_guidance']}")
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
    is_combined = "exhibit_type" in s
    doc = DocxDocument()

    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    title_text = f"8-K + Exhibit 99.1 Summary: {ticker}" if is_combined else f"8-K Summary: {ticker}"
    title = doc.add_heading(title_text, level=0)
    title.runs[0].font.size = Pt(20)

    date = s.get("filing_date", "")
    meta = doc.add_paragraph()
    if s.get("company"):
        meta.add_run("Company: ").bold = True
        meta.add_run(s.get("company", "N/A"))
        meta.add_run("    ")
    meta.add_run("Filing Date: ").bold = True
    meta.add_run(date)
    meta.add_run("    Items: ").bold = True
    meta.add_run(", ".join(s.get("items_reported", [])))
    if is_combined and s.get("exhibit_type"):
        meta2 = doc.add_paragraph()
        meta2.add_run("Exhibit Type: ").bold = True
        meta2.add_run(s.get("exhibit_type", "N/A"))

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

    if d.get("market_impact"):
        doc.add_heading("Market Impact", level=2)
        doc.add_paragraph(d["market_impact"])

    if d.get("forward_guidance"):
        doc.add_heading("Forward Guidance", level=2)
        doc.add_paragraph(d["forward_guidance"])

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
    primary_url = source[0] if isinstance(source, list) else source

    print(f"Fetching 8-K from: {primary_url}")
    if isinstance(source, list) and len(source) > 1:
        print(f"   ({len(source)} URLs will be combined for extraction)")
    text_8k = fetch_8k_text(source)
    print(f"Extracted {len(text_8k.split())} words from 8-K")

    exhibit_url = find_exhibit_991_url(primary_url)
    if exhibit_url:
        print(f"Found Exhibit 99.1: {exhibit_url}")
        text_991 = fetch_8k_text(exhibit_url)
        print(f"Extracted {len(text_991.split())} words from Exhibit 99.1")
        combined_text = (
            "=== 8-K FILING ===\n\n" + text_8k +
            "\n\n=== EXHIBIT 99.1 (PRESS RELEASE) ===\n\n" + text_991
        )
        print("Generating combined 8-K + 99.1 summary via Claude...")
        result = summarize(combined_text, prompt=COMBINED_PROMPT)
        result["summary_type"] = "combined"
        filename_prefix = "8K_99.1_Combined_Summary"
    else:
        print("No Exhibit 99.1 found — generating standard 8-K summary...")
        result = summarize(text_8k)
        result["summary_type"] = "single"
        filename_prefix = "8K_Summary"

    print_summary(result)

    uid = filing_uid(FILING_URL)
    from .s3_utils import upload_json

    s3_json_path, s3_json_url = upload_json(result, f"8k_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    ticker = result.get("ticker", "UNKNOWN")
    date = result.get("filing_date", "")
    safe_ticker = re.sub(r'[^\w\-\.]', '_', ticker)
    safe_date = date.replace("/", "-")
    docx_suffix = f"{filename_prefix}_{safe_ticker}_{safe_date}_{uid}.docx"
    s3_docx_path, s3_docx_url = export_docx(result, docx_suffix)
    print(f"DOCX uploaded to S3: {s3_docx_url}")

    result["s3_docx_path"] = s3_docx_path
    result["s3_docx_url"] = s3_docx_url
    result["s3_json_path"] = s3_json_path
    result["s3_json_url"] = s3_json_url
    return result


if __name__ == "__main__":
    main()
