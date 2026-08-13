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
from ._naming import filing_uid, sanitize_date_part, sanitize_filename_part
from ._deal_context import inject_deal_context

# ──── PASTE YOUR SC 13D/13G URL HERE ────
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


SUMMARY_PROMPT = """You are an expert analyst summarizing SEC Schedule 13D and 13G filings (beneficial ownership reports) for a merger arbitrage desk.

Schedule 13D is filed when an investor acquires more than 5% of a company's shares and may have activist intentions. Schedule 13G is the shorter, passive version for investors with no intent to influence control. Amendments (13D/A, 13G/A) report changes in position. These filings are critical for merger arb because they reveal activist stakes, potential acquirers building positions, and large holders who can influence deal outcomes.

Given the SC 13D or 13G text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "ticker": "<subject company ticker, or null if not stated in the filing>",
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
    "activist_intentions": "<plans or proposals explicitly stated in Item 4 of the filing>",
    "deal_implications": "<for pending M&A: stated intentions regarding any pending transaction or corporate action>",
    "related_agreements": "<any standstill, voting, or lock-up agreements mentioned>",
    "risks_flagged": ["<activist risk, potential competing bid, deal opposition, regulatory implications>"]
  }
}

Rules:
- CRITICAL — FACTS ONLY: Every statement in your summary must be directly traceable to the filing text. Report ONLY what the document says. Do NOT add analysis, assess significance, interpret motives, predict outcomes, evaluate probability, or editorialize. Do NOT state what is "not disclosed" or "not mentioned" — simply omit fields where the filing is silent. If the filing does not say it, do not write it.
  GOOD: "CADE requested revenue data for 2021-2025 across four markets."
  BAD: "The broad scope of information requested indicates potentially detailed competitive analysis ahead."
  GOOD: "The offer expires June 10, 2026."
  BAD: "This tight timeline may create pressure on shareholders to tender quickly."
- PRECISION: Use the filing's exact terminology for legal, regulatory, and financial terms. Do NOT paraphrase in ways that broaden or narrow the stated meaning. GOOD: "All 14 Pennsylvania PUC hearings have concluded." BAD: "Regulatory proceedings concluded in Pennsylvania."
- L1 format MUST be: + <TICKER> – <filer> discloses <X>% stake. | <date>
- Distinguish between 13D (potentially activist) and 13G (passive) — this signals intent
- Extract EXACT share counts, percentages, and voting/dispositive power breakdown
- The "Purpose of Transaction" section (Item 4 in 13D) is the most important — quote key language
- For amendments: compare current vs prior position and highlight the change
- Flag any language suggesting the filer may seek to influence the company or a pending deal
- Note any agreements with other shareholders or the company (standstills, voting agreements)

SC 13D/13G TEXT:
"""

EXTRACTION_GUIDANCE = """This is an SC 13D or SC 13G beneficial ownership filing (5%+ stake disclosure).
Extract:
- Reporting person/entity name and type (individual, fund, corporation)
- Total shares beneficially owned and percentage of class
- Voting power breakdown (sole vs shared)
- Dispositive power breakdown (sole vs shared)
- Item 4: Purpose of Transaction — extract in FULL (this is the most critical section for arb)
- Source and amount of funds used for acquisition
- If amendment: prior share count/percentage and what changed
- Any agreements (standstill, voting, lock-up, joint filing agreements)
- Plans regarding merger, board seats, strategic changes, or activism
"""


def fetch_filing_text(source: str) -> str:
    """Fetch and extract text from an SC 13D/13G filing (URL, local file, or PDF)."""
    from .fetch_utils import fetch_text_with_extraction
    return fetch_text_with_extraction(source, extraction_guidance=EXTRACTION_GUIDANCE)


def summarize(text: str, model: str = "claude-opus-4-8", deal_context: dict | None = None) -> dict:
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
            "content": inject_deal_context(SUMMARY_PROMPT, deal_context) + "\n\n" + text
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
    from .fetch_utils import is_empty_value, has_content, add_field
    from .s3_utils import upload_docx_bytes

    ticker = s.get("ticker", "UNKNOWN")
    date = s.get("filing_date", "")
    doc = DocxDocument()

    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    title = doc.add_heading(f"SC 13D/13G Summary: {ticker}", level=0)
    title.runs[0].font.size = Pt(20)

    meta = doc.add_paragraph()
    add_field(meta, "Subject Company: ", s.get(
        "subject_company"), newline=False)
    add_field(meta, "    Filing Type: ", s.get("filing_type"), newline=False)

    meta2 = doc.add_paragraph()
    filer_name = s.get("filer_name")
    filer_type = s.get("filer_type")
    if not is_empty_value(filer_name):
        meta2.add_run("Filer: ").bold = True
        label = str(filer_name)
        if not is_empty_value(filer_type):
            label += f" ({filer_type})"
        meta2.add_run(label)
    add_field(meta2, "    Date: ", date, newline=False)

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
    add_field(own_p, "Shares Held: ", od.get("shares_held"))
    add_field(own_p, "Percentage Owned: ", od.get("percentage_owned"))
    add_field(own_p, "Sole Voting Power: ", od.get("sole_voting_power"))
    add_field(own_p, "Shared Voting Power: ", od.get("shared_voting_power"))
    add_field(own_p, "Sole Dispositive Power: ",
              od.get("sole_dispositive_power"))
    add_field(own_p, "Shared Dispositive Power: ", od.get(
        "shared_dispositive_power"), newline=False)

    if not is_empty_value(d.get("position_change")):
        doc.add_heading("Position Change", level=2)
        doc.add_paragraph(d.get("position_change"))

    if not is_empty_value(d.get("source_of_funds")):
        doc.add_heading("Source of Funds", level=2)
        doc.add_paragraph(d.get("source_of_funds"))

    if not is_empty_value(d.get("purpose_of_transaction")):
        doc.add_heading("Purpose of Transaction", level=2)
        doc.add_paragraph(d.get("purpose_of_transaction"))

    if not is_empty_value(d.get("activist_intentions")):
        doc.add_heading("Activist Intentions", level=2)
        doc.add_paragraph(d.get("activist_intentions"))

    if not is_empty_value(d.get("deal_implications")):
        doc.add_heading("Deal Implications", level=2)
        doc.add_paragraph(d.get("deal_implications"))

    if not is_empty_value(d.get("related_agreements")):
        doc.add_heading("Related Agreements", level=2)
        doc.add_paragraph(d.get("related_agreements"))

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


def main(filing_url=None, deal_context: dict | None = None):
    ctx = deal_context if deal_context is not None else DEAL_CONTEXT
    source = filing_url if filing_url is not None else FILING_URL

    print(f"Fetching SC 13D/13G from: {source}")

    text = fetch_filing_text(source)
    print(f"Extracted {len(text.split())} words of text")

    print("Generating summary via Claude Opus 4.5...")
    result = summarize(text, deal_context=ctx)
    from ._ticker_context import apply_known_tickers
    result = apply_known_tickers(result, ctx)

    print_summary(result)

    uid = filing_uid(source)
    from .s3_utils import upload_json

    s3_json_path, s3_json_url = upload_json(
        result, f"sc13d_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    safe_ticker = sanitize_filename_part(result.get("ticker"))
    safe_date = sanitize_date_part(result.get("filing_date"))
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
