"""
SC 13D/13G Filing Summarizer — Multi-level summaries via Claude API
Usage: python sc13d_summary.py
"""

import anthropic
import re
import json
import sys
import os
import io
from pathlib import Path
from ._naming import filing_uid

# ──── PASTE YOUR SC 13D/13G URL HERE ────
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


SUMMARY_PROMPT = """You are an expert analyst summarizing SEC Schedule 13D and 13G filings (beneficial ownership reports) for a merger arbitrage desk.

Schedule 13D is filed when an investor acquires more than 5% of a company's shares and may have activist intentions. Schedule 13G is the shorter, passive version for investors with no intent to influence control. Amendments (13D/A, 13G/A) report changes in position. These filings are critical for merger arb because they reveal activist stakes, potential acquirers building positions, and large holders who can influence deal outcomes.

Given the SC 13D or 13G text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "ticker": "<subject company ticker>",
  "subject_company": "<name of the company whose shares are owned>",
  "filing_date": "<MM/DD/YY>",
  "filing_type": "<SC 13D | SC 13D/A | SC 13G | SC 13G/A>",
  "filer_name": "<name of reporting person/entity>",
  "filer_type": "<Activist Fund | Hedge Fund | Mutual Fund | PE Firm | Individual | Corporation | Other>",

  "L1_headline": "+ <TICKER> – <filer> discloses <X>% stake. | <date>",

  "L2_brief": "<2-3 sentence summary covering: who filed, what percentage they own, whether this is a new position or change, and what their stated intentions are>",

  "L3_detailed": {
    "ownership_details": {
      "shares_held": "<number of shares beneficially owned>",
      "percentage_owned": "<percentage of outstanding shares>",
      "sole_voting_power": "<shares with sole voting power>",
      "shared_voting_power": "<shares with shared voting power>",
      "sole_dispositive_power": "<shares with sole dispositive power>",
      "shared_dispositive_power": "<shares with shared dispositive power>"
    },
    "position_change": "<new position, increased, decreased, or unchanged — with prior % if amendment>",
    "source_of_funds": "<personal funds, working capital, margin, etc.>",
    "purpose_of_transaction": "<stated purpose — investment, influence board, seek merger, oppose deal, passive, etc.>",
    "activist_intentions": "<any plans to: seek board seats, propose transactions, influence management, or change business strategy>",
    "deal_implications": "<for pending M&A: how this stake affects deal probability, voting dynamics, or potential competing bids>",
    "related_agreements": "<any standstill, voting, or lock-up agreements mentioned>",
    "risks_flagged": ["<activist risk, potential competing bid, deal opposition, regulatory implications>"]
  }
}

Rules:
- TONE: State only facts from the filing. Do NOT speculate on motives, interpret what actions "signal" or "suggest", assess confidence levels, or draw conclusions beyond what is explicitly stated. GOOD: "Company suspended earnings calls due to pending transaction." BAD: "Company suspended earnings calls, signaling high confidence in deal completion."
- L1 format MUST be: + <TICKER> – <filer> discloses <X>% stake. | <date>
- Distinguish between 13D (potentially activist) and 13G (passive) — this signals intent
- Extract EXACT share counts, percentages, and voting/dispositive power breakdown
- The "Purpose of Transaction" section (Item 4 in 13D) is the most important — quote key language
- For amendments: compare current vs prior position and highlight the change
- Flag any language suggesting the filer may seek to influence the company or a pending deal
- Note any agreements with other shareholders or the company (standstills, voting agreements)

SC 13D/13G TEXT:
"""


def fetch_filing_text(source: str) -> str:
    """Fetch and extract text from an SC 13D/13G filing (URL, local file, or PDF)."""
    from .fetch_utils import fetch_text
    return fetch_text(source)


def summarize(text: str, model: str = "claude-sonnet-4-6") -> dict:
    """Call Claude API to produce multi-level summary."""
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Set it in .env or Django settings (ANTHROPIC_API_KEY).")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    msg = client.messages.create(
        model=model,
        max_tokens=2500,
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
    print("  SC 13D/13G SUMMARY (Beneficial Ownership)")
    print("=" * 70)

    print(
        f"\n   Company:  {s.get('subject_company', 'N/A')} ({s.get('ticker', 'N/A')})")
    print(
        f"   Filer:    {s.get('filer_name', 'N/A')} ({s.get('filer_type', 'N/A')})")
    print(f"   Type:     {s.get('filing_type', 'N/A')}")
    print(f"   Date:     {s.get('filing_date', 'N/A')}")

    print(f"\n📌 L1 | HEADLINE")
    print(f"   {s['L1_headline']}")

    print(f"\n📋 L2 | BRIEF")
    print(f"   {s['L2_brief']}")

    d = s["L3_detailed"]
    od = d.get("ownership_details", {})
    print(f"\n📊 L3 | DETAILED")
    print(f"   Ownership:")
    print(f"     Shares:       {od.get('shares_held', 'N/A')}")
    print(f"     % Owned:      {od.get('percentage_owned', 'N/A')}")
    print(f"     Sole Vote:    {od.get('sole_voting_power', 'N/A')}")
    print(f"     Shared Vote:  {od.get('shared_voting_power', 'N/A')}")
    print(f"   Position Change:  {d.get('position_change', 'N/A')}")
    print(f"   Source of Funds:  {d.get('source_of_funds', 'N/A')}")
    print(f"   Purpose:          {d.get('purpose_of_transaction', 'N/A')}")
    print(f"   Activist Intent:  {d.get('activist_intentions', 'N/A')}")
    print(f"   Deal Impact:      {d.get('deal_implications', 'N/A')}")
    print(f"   Agreements:       {d.get('related_agreements', 'N/A')}")
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

    title = doc.add_heading(f"SC 13D/13G Summary: {ticker}", level=0)
    title.runs[0].font.size = Pt(20)

    meta = doc.add_paragraph()
    meta.add_run(f"Subject Company: ").bold = True
    meta.add_run(s.get("subject_company", "N/A"))
    meta.add_run(f"    Filing Type: ").bold = True
    meta.add_run(s.get("filing_type", "N/A"))

    meta2 = doc.add_paragraph()
    meta2.add_run(f"Filer: ").bold = True
    meta2.add_run(
        f"{s.get('filer_name', 'N/A')} ({s.get('filer_type', 'N/A')})")
    date = s.get("filing_date", "")
    meta2.add_run(f"    Date: ").bold = True
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
    od = d.get("ownership_details", {})

    doc.add_heading("Ownership Details", level=2)
    own_p = doc.add_paragraph()
    own_p.add_run("Shares Held: ").bold = True
    own_p.add_run(od.get("shares_held", "N/A") + "\n")
    own_p.add_run("Percentage Owned: ").bold = True
    own_p.add_run(od.get("percentage_owned", "N/A") + "\n")
    own_p.add_run("Sole Voting Power: ").bold = True
    own_p.add_run(od.get("sole_voting_power", "N/A") + "\n")
    own_p.add_run("Shared Voting Power: ").bold = True
    own_p.add_run(od.get("shared_voting_power", "N/A") + "\n")
    own_p.add_run("Sole Dispositive Power: ").bold = True
    own_p.add_run(od.get("sole_dispositive_power", "N/A") + "\n")
    own_p.add_run("Shared Dispositive Power: ").bold = True
    own_p.add_run(od.get("shared_dispositive_power", "N/A"))

    doc.add_heading("Position Change", level=2)
    doc.add_paragraph(d.get("position_change", "N/A"))

    doc.add_heading("Source of Funds", level=2)
    doc.add_paragraph(d.get("source_of_funds", "N/A"))

    doc.add_heading("Purpose of Transaction", level=2)
    doc.add_paragraph(d.get("purpose_of_transaction", "N/A"))

    doc.add_heading("Activist Intentions", level=2)
    doc.add_paragraph(d.get("activist_intentions", "N/A"))

    doc.add_heading("Deal Implications", level=2)
    doc.add_paragraph(d.get("deal_implications", "N/A"))

    doc.add_heading("Related Agreements", level=2)
    doc.add_paragraph(d.get("related_agreements", "N/A"))

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

    print(f"Fetching SC 13D/13G from: {source}")

    text = fetch_filing_text(source)
    print(f"Extracted {len(text.split())} words of text")

    print("Generating summary via Claude Opus 4.5...")
    result = summarize(text)

    print_summary(result)

    uid = filing_uid(FILING_URL)
    from .s3_utils import upload_json

    s3_json_path, s3_json_url = upload_json(
        result, f"sc13d_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    ticker = result.get("ticker", "UNKNOWN")
    date = result.get("filing_date", "")
    safe_ticker = re.sub(r'[^\w\-\.]', '_', ticker)
    safe_date = date.replace("/", "-")
    docx_suffix = f"SC13D_Summary_{safe_ticker}_{safe_date}_{uid}.docx"
    s3_docx_path, s3_docx_url = export_docx(result, docx_suffix)
    print(f"DOCX uploaded to S3: {s3_docx_url}")

    result["s3_docx_path"] = s3_docx_path
    result["s3_docx_url"] = s3_docx_url
    result["s3_json_path"] = s3_json_path
    result["s3_json_url"] = s3_json_url
    return result


if __name__ == "__main__":
    main()
