"""
Rule 425 Filing Summarizer — Multi-level summaries via Claude API
Usage: python 425_summary.py
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

# ──── PASTE YOUR 425 URL HERE ────
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


SUMMARY_PROMPT = """You are an expert analyst summarizing SEC Rule 425 filings (business combination communications) for a merger arbitrage desk.

Rule 425 filings are communications related to proposed business combinations (mergers, acquisitions, exchange offers). Companies file these to publicly distribute materials about a pending deal — investor presentations, employee communications, customer letters, media transcripts, Q&A documents, and shareholder solicitations. These are gold mines for merger arb because they often contain the latest deal messaging, management sentiment, timeline updates, and integration plans.

Given the Rule 425 text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "ticker": "<subject company ticker, or null if not stated in the filing>",
  "filer": "<company that filed the 425>",
  "subject_company": "<company that is the subject of the transaction>",
  "filing_date": "<MM/DD/YY>",
  "communication_type": "<Investor Presentation | Press Release | Employee Communication | Customer Letter | Conference Call Transcript | FAQ Document | Shareholder Letter | Media Statement | Other>",

  "L1_headline": "+ <TICKER> – <key deal update in ≤8 words>. | <date>",

   "L2_brief": "<2-3 sentence summary covering: what this communication says about the deal, any timeline/terms updates, and stated deal status>",

  "L3_detailed": {
    "deal_parties": "<acquirer and target, with tickers>",
    "deal_status_update": "<current status of the deal as described in this filing>",
    "timeline_update": "<any new information about expected close date, regulatory timeline, or milestones>",
    "terms_update": "<any changes or reaffirmation of deal terms — price, exchange ratio, conditions>",
        "management_tone": "<direct quotes or stated characterizations from management about deal progress>",
    "key_messages": ["<main talking points from the communication>"],
    "regulatory_update": "<any updates on regulatory approvals — antitrust, CFIUS, sector-specific regulators>",
    "shareholder_vote_info": "<any info about proxy, record date, vote date, recommendation>",
    "integration_details": "<any mentions of integration planning, synergies, organizational changes>",
    "risks_flagged": ["<deal risks mentioned, opposition, litigation, regulatory concerns, financing issues>"]
  }
}

Rules:
- CRITICAL — FACTS ONLY: Every statement in your summary must be directly traceable to the filing text. Report ONLY what the document says. Do NOT add analysis, assess significance, interpret motives, predict outcomes, evaluate probability, or editorialize. Do NOT state what is "not disclosed" or "not mentioned" — simply omit fields where the filing is silent. If the filing does not say it, do not write it.
  GOOD: "CADE requested revenue data for 2021-2025 across four markets."
  BAD: "The broad scope of information requested indicates potentially detailed competitive analysis ahead."
  GOOD: "The offer expires June 10, 2026."
  BAD: "This tight timeline may create pressure on shareholders to tender quickly."
- PRECISION: Use the filing's exact terminology for legal, regulatory, and financial terms. Do NOT paraphrase in ways that broaden or narrow the stated meaning. GOOD: "All 14 Pennsylvania PUC hearings have concluded." BAD: "Regulatory proceedings concluded in Pennsylvania."
- L1 format MUST be: + <TICKER> – <deal update>. | <date>
- Identify the type of communication (presentation, letter, transcript, etc.)
- Focus on what's NEW in this communication vs. what was already known
- Extract any updated timeline, regulatory status, or deal term changes
- Extract direct quotes from management about deal status and progress
- Flag any language suggesting deal complications, opposition, or changed circumstances
- Note any shareholder vote details (record date, meeting date, board recommendation)
- Capture synergy estimates or integration timeline if mentioned

RULE 425 TEXT:
"""

EXTRACTION_GUIDANCE = """This is a Rule 425 business combination communication.
Extract the following:
- Type of communication (investor presentation, shareholder letter, transcript, FAQ, etc.)
- Deal parties and tickers
- Any timeline updates (expected close date, regulatory milestones)
- Any terms updates or reaffirmation of deal terms
- Management statements about deal progress and rationale
- Regulatory approval updates and status
- Shareholder vote information (date, threshold, recommendation)
- Integration planning details and synergy estimates
- Any deal risks, opposition, or litigation mentioned
- Key messages to shareholders"""


def fetch_filing_text(source: str) -> str:
    """Fetch and extract text from a Rule 425 filing (URL, local file, or PDF)."""
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
    print("  RULE 425 SUMMARY (Business Combination Communication)")
    print("=" * 70)

    print(f"\n   Filer:    {s.get('filer', 'N/A')}")
    print(
        f"   Subject:  {s.get('subject_company', 'N/A')} ({s.get('ticker', 'N/A')})")
    print(f"   Type:     {s.get('communication_type', 'N/A')}")
    print(f"   Date:     {s.get('filing_date', 'N/A')}")

    print(f"\n📌 L1 | HEADLINE")
    print(f"   {s['L1_headline']}")

    print(f"\n📋 L2 | BRIEF")
    print(f"   {s['L2_brief']}")

    d = s["L3_detailed"]
    print(f"\n📊 L3 | DETAILED")
    print(f"   Deal Parties:     {d.get('deal_parties', 'N/A')}")
    print(f"   Status Update:    {d.get('deal_status_update', 'N/A')}")
    print(f"   Timeline Update:  {d.get('timeline_update', 'N/A')}")
    print(f"   Terms Update:     {d.get('terms_update', 'N/A')}")
    print(f"   Mgmt Tone:        {d.get('management_tone', 'N/A')}")
    if d.get("key_messages"):
        print(f"   Key Messages:")
        for m in d["key_messages"]:
            print(f"     • {m}")
    print(f"   Regulatory:       {d.get('regulatory_update', 'N/A')}")
    print(f"   Vote Info:        {d.get('shareholder_vote_info', 'N/A')}")
    print(f"   Integration:      {d.get('integration_details', 'N/A')}")
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

    title = doc.add_heading(f"Rule 425 Summary: {ticker}", level=0)
    title.runs[0].font.size = Pt(20)

    meta = doc.add_paragraph()
    meta.add_run("Filer: ").bold = True
    meta.add_run(s.get("filer", "N/A"))
    meta.add_run("    Subject: ").bold = True
    meta.add_run(s.get("subject_company", "N/A"))

    meta2 = doc.add_paragraph()
    meta2.add_run("Filing Date: ").bold = True
    meta2.add_run(date)
    meta2.add_run("    Communication Type: ").bold = True
    meta2.add_run(s.get("communication_type", "N/A"))

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

    deal_parties = d.get("deal_parties")
    deal_status = d.get("deal_status_update")
    timeline = d.get("timeline_update")
    terms = d.get("terms_update")
    if (not is_empty_value(deal_parties) or not is_empty_value(deal_status)
            or not is_empty_value(timeline) or not is_empty_value(terms)):
        doc.add_heading("Deal Status", level=2)
        status_p = doc.add_paragraph()
        add_field(status_p, "Parties: ", deal_parties)
        add_field(status_p, "Current Status: ", deal_status)
        add_field(status_p, "Timeline: ", timeline)
        add_field(status_p, "Terms: ", terms)

    if not is_empty_value(d.get("management_tone")):
        doc.add_heading("Management Tone", level=2)
        doc.add_paragraph(d["management_tone"])

    key_messages = d.get("key_messages", [])
    if has_content(key_messages):
        doc.add_heading("Key Messages", level=2)
        for m in key_messages:
            if not is_empty_value(m):
                doc.add_paragraph(m, style="List Bullet")

    if not is_empty_value(d.get("regulatory_update")):
        doc.add_heading("Regulatory Update", level=2)
        doc.add_paragraph(d["regulatory_update"])

    if not is_empty_value(d.get("shareholder_vote_info")):
        doc.add_heading("Shareholder Vote Info", level=2)
        doc.add_paragraph(d["shareholder_vote_info"])

    if not is_empty_value(d.get("integration_details")):
        doc.add_heading("Integration Details", level=2)
        doc.add_paragraph(d["integration_details"])

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

    print(f"Fetching Rule 425 from: {source}")

    text = fetch_filing_text(source)
    print(f"Extracted {len(text.split())} words of text")

    print("Generating summary via Claude Opus 4.5...")
    result = summarize(text)
    from ._ticker_context import apply_known_tickers
    result = apply_known_tickers(result, DEAL_CONTEXT)

    print_summary(result)

    uid = filing_uid(FILING_URL)
    from .s3_utils import upload_json

    s3_json_path, s3_json_url = upload_json(result, f"425_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    safe_ticker = sanitize_filename_part(result.get("ticker"))
    safe_date = sanitize_date_part(result.get("filing_date"))
    docx_suffix = f"425_Summary_{safe_ticker}_{safe_date}_{uid}.docx"
    s3_docx_path, s3_docx_url = export_docx(result, docx_suffix)
    print(f"DOCX uploaded to S3: {s3_docx_url}")

    result["s3_docx_path"] = s3_docx_path
    result["s3_docx_url"] = s3_docx_url
    result["s3_json_path"] = s3_json_path
    result["s3_json_url"] = s3_json_url
    return result


if __name__ == "__main__":
    main()
