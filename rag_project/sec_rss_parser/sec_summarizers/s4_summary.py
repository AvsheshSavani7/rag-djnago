"""
S-4 Registration Statement Summarizer — Multi-level summaries via Claude API
Usage: python s4_summary.py

S-4 filings register securities issued in business combinations (mergers, acquisitions,
exchange offers). They contain the merger agreement, pro forma financials, risk factors,
and often serve as the combined proxy/prospectus for the deal.
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

# ──── PASTE YOUR S-4 URL HERE ────
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


SUMMARY_PROMPT = """You are an expert merger arbitrage analyst summarizing SEC Form S-4 registration statements for a merger arbitrage trading desk.

S-4 filings register securities to be issued in a business combination. They often serve as the combined proxy statement/prospectus and contain the full merger agreement, pro forma financials, risk factors, fairness opinions, and background of the transaction. S-4/A amendments update prior filings.

These are long filings — focus on extracting the information most critical to a merger arb position.

Given the S-4 text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "filing_type": "<S-4 | S-4/A>",
  "acquirer": "<acquiring company>",
  "acquirer_ticker": "<acquirer ticker>",
  "target": "<target company>",
  "target_ticker": "<target ticker as stated in the filing, or null if not stated>",
  "filing_date": "<MM/DD/YY>",

  "L1_headline": "+ <TARGET TICKER> – <key event in ≤8 words>. | <date>",

  "L2_brief": "<2-3 sentence summary covering: the deal structure, consideration, and current status>",

  "L3_detailed": {
    "deal_structure": "<merger, stock-for-stock, cash-and-stock, reverse merger, etc.>",
    "consideration": {
      "type": "<Cash | Stock | Cash & Stock>",
      "per_share_value": "<per-share value to target shareholders>",
      "exchange_ratio": "<exchange ratio if stock deal, or null>",
      "cash_component": "<cash per share if mixed, or null>",
      "total_deal_value": "<aggregate deal value>",
      "premium": "<premium with reference price/date>"
    },
    "conditions_precedent": ["<shareholder approvals, regulatory approvals, financing, MAE, other closing conditions>"],
    "regulatory_approvals": {
      "required": ["<HSR, DOJ, FTC, CFIUS, EU, sector-specific>"],
      "status": "<current status>"
    },
    "deal_protections": {
      "breakup_fee_target": "<target termination fee>",
      "breakup_fee_acquirer": "<acquirer/reverse termination fee>",
      "go_shop": "<go-shop details or 'None'>",
      "matching_rights": "<matching rights details>",
      "no_shop": "<no-shop/no-solicitation details>"
    },
    "expected_timeline": "<expected closing date and key milestones>",
    "shareholder_vote": "<which shareholders must approve, vote threshold, record date, meeting date if set>",
    "pro_forma_highlights": ["<key pro forma financial metrics — combined revenue, EPS accretion/dilution, synergies>"],
    "risk_factors": ["<top 3-5 deal-specific risk factors from the filing>"],
    "fairness_opinion": "<advisor name and conclusion>",
    "background_summary": "<brief summary of negotiation history and how the deal came about>"
  }
}

Rules:
- CRITICAL — FACTS ONLY: Every statement in your summary must be directly traceable to the filing text. Report ONLY what the document says. Do NOT add analysis, assess significance, interpret motives, predict outcomes, evaluate probability, or editorialize. Do NOT state what is "not disclosed" or "not mentioned" — simply omit fields where the filing is silent. If the filing does not say it, do not write it.
  GOOD: "CADE requested revenue data for 2021-2025 across four markets."
  BAD: "The broad scope of information requested indicates potentially detailed competitive analysis ahead."
  GOOD: "The offer expires June 10, 2026."
  BAD: "This tight timeline may create pressure on shareholders to tender quickly."
- PRECISION: Use the filing's exact terminology for legal, regulatory, and financial terms. Do NOT paraphrase in ways that broaden or narrow the stated meaning. GOOD: "All 14 Pennsylvania PUC hearings have concluded." BAD: "Regulatory proceedings concluded in Pennsylvania."
- L1 format MUST be: + <TARGET TICKER> – <event>. | <date>
- Extract EXACT per-share consideration, exchange ratio, total deal value, and premium
- List ALL closing conditions individually — these determine arb risk
- Capture deal protection terms precisely (termination fees as dollar amounts AND percentages)
- Note whether this is an initial S-4 or an amendment — if amendment, highlight what changed
- Extract pro forma financial highlights and synergy estimates
- risk factors stated in the filing related to the transaction
- If the filing includes the merger agreement, extract key terms from it

S-4 TEXT:
"""

EXTRACTION_GUIDANCE = """This is an S-4 registration statement (merger proxy/prospectus).
Extract the following sections in full:
- Deal terms: per-share consideration, exchange ratio, cash component, total deal value, premium calculation
- Deal structure description (merger, stock-for-stock, reverse merger, etc.)
- ALL conditions precedent / closing conditions listed individually
- Regulatory approvals required and their current status — every jurisdiction (HSR, EU, CFIUS, sector-specific)
- Deal protections: termination fees (target AND acquirer amounts and triggers), go-shop period, matching rights, no-shop/no-solicitation
- Expected timeline and key milestones (closing date, shareholder meeting, record date)
- Shareholder vote details: which shareholders, threshold, record date, meeting date
- Fairness opinion: advisor name, conclusion, and fee — do NOT extract the full financial analyses, methodologies, DCF tables, or comparable company details
- Background of the Transaction (negotiation history) — key events, dates, and decisions only; skip routine procedural details
- Pro forma financial highlights and synergy estimates
- Top deal-specific risk factors — extract only the top 5-10 most material risks, skip boilerplate
- Material U.S. Federal Income Tax Consequences — extract ONLY the conclusion on tax-free reorganization qualification; skip detailed REIT tax analysis and general tax law discussion
- If amendment (S-4/A): what specifically changed from prior filing"""


def fetch_filing_text(source: str) -> str:
    """Fetch and extract text from an S-4 filing (URL, local file, or PDF)."""
    from .fetch_utils import fetch_text_with_extraction
    return fetch_text_with_extraction(source, extraction_guidance=EXTRACTION_GUIDANCE)


def summarize(text: str, model: str = "claude-opus-4-8") -> dict:
    """Call Claude API to produce multi-level summary."""
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Set it in .env or Django settings (ANTHROPIC_API_KEY).")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    msg = client.messages.create(
        model=model,
        max_tokens=8000,
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
    print(f"  S-4 SUMMARY (M&A Registration Statement)")
    print("=" * 70)

    print(
        f"\n   Acquirer: {s.get('acquirer', 'N/A')} ({s.get('acquirer_ticker') or '—'})")
    print(
        f"   Target:   {s.get('target', 'N/A')} ({s.get('target_ticker') or '—'})")
    print(f"   Type:     {s.get('filing_type', 'N/A')}")
    print(f"   Date:     {s.get('filing_date', 'N/A')}")

    print(f"\n📌 L1 | HEADLINE")
    print(f"   {s['L1_headline']}")

    print(f"\n📋 L2 | BRIEF")
    print(f"   {s['L2_brief']}")

    d = s["L3_detailed"]
    c = d.get("consideration", {})
    print(f"\n📊 L3 | DETAILED")
    print(f"\n   ── DEAL TERMS ──")
    print(f"   Structure:    {d.get('deal_structure', 'N/A')}")
    print(f"   Type:         {c.get('type', 'N/A')}")
    print(f"   Per Share:    {c.get('per_share_value', 'N/A')}")
    if c.get("exchange_ratio"):
        print(f"   Exch Ratio:   {c['exchange_ratio']}")
    if c.get("cash_component"):
        print(f"   Cash Comp:    {c['cash_component']}")
    print(f"   Deal Value:   {c.get('total_deal_value', 'N/A')}")
    print(f"   Premium:      {c.get('premium', 'N/A')}")

    if d.get("conditions_precedent"):
        print(f"\n   ── CONDITIONS ──")
        for cond in d["conditions_precedent"]:
            print(f"     • {cond}")

    reg = d.get("regulatory_approvals", {})
    print(f"\n   ── REGULATORY ──")
    if reg.get("required"):
        for r in reg["required"]:
            print(f"     • {r}")
    print(f"   Status:       {reg.get('status', 'N/A')}")

    dp = d.get("deal_protections", {})
    print(f"\n   ── DEAL PROTECTIONS ──")
    print(f"   Target Fee:   {dp.get('breakup_fee_target', 'N/A')}")
    print(f"   Acquirer Fee: {dp.get('breakup_fee_acquirer', 'N/A')}")
    print(f"   Go-Shop:      {dp.get('go_shop', 'N/A')}")
    print(f"   Match Rights: {dp.get('matching_rights', 'N/A')}")
    print(f"   No-Shop:      {dp.get('no_shop', 'N/A')}")

    print(f"\n   Timeline:     {d.get('expected_timeline', 'N/A')}")
    print(f"   Vote:         {d.get('shareholder_vote', 'N/A')}")
    print(f"   Fairness:     {d.get('fairness_opinion', 'N/A')}")

    if d.get("pro_forma_highlights"):
        print(f"\n   ── PRO FORMA ──")
        for p in d["pro_forma_highlights"]:
            print(f"     • {p}")

    if d.get("background_summary"):
        print(f"\n   ── BACKGROUND ──")
        print(f"   {d['background_summary']}")

    if d.get("risk_factors"):
        print(f"\n   ── KEY RISKS ──")
        for r in d["risk_factors"]:
            print(f"     • {r}")

    print("=" * 70)


def export_docx(s: dict, s3_key_suffix: str):
    """Build summary as Word doc, upload to S3 (summary_docx/), return (s3_path, s3_url)."""
    from .fetch_utils import is_empty_value, has_content, add_field
    from .s3_utils import upload_docx_bytes

    date = s.get("filing_date", "")
    doc = DocxDocument()

    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    title = doc.add_heading(f"S-4 Summary: {s.get('target', 'N/A')}", level=0)
    title.runs[0].font.size = Pt(20)

    meta = doc.add_paragraph()
    meta.add_run("Acquirer: ").bold = True
    meta.add_run(
        f"{s.get('acquirer', 'N/A')} ({s.get('acquirer_ticker') or '—'})")
    meta.add_run("    Target: ").bold = True
    meta.add_run(f"{s.get('target', 'N/A')} ({s.get('target_ticker') or '—'})")

    meta2 = doc.add_paragraph()
    meta2.add_run("Filing Date: ").bold = True
    meta2.add_run(date)
    meta2.add_run("    Filing Type: ").bold = True
    meta2.add_run(s.get("filing_type", "S-4"))

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
    c = d.get("consideration", {})

    if has_content(c) or not is_empty_value(d.get("deal_structure")):
        doc.add_heading("Deal Terms", level=2)
        terms_p = doc.add_paragraph()
        add_field(terms_p, "Structure: ", d.get("deal_structure"))
        add_field(terms_p, "Consideration Type: ", c.get("type"))
        add_field(terms_p, "Per Share Value: ", c.get("per_share_value"))
        add_field(terms_p, "Exchange Ratio: ", c.get("exchange_ratio"))
        add_field(terms_p, "Cash Component: ", c.get("cash_component"))
        add_field(terms_p, "Total Deal Value: ", c.get("total_deal_value"))
        add_field(terms_p, "Premium: ", c.get("premium"))

    conditions = d.get("conditions_precedent", [])
    if has_content(conditions):
        doc.add_heading("Conditions Precedent", level=2)
        for cond in conditions:
            if not is_empty_value(cond):
                doc.add_paragraph(cond, style="List Bullet")

    reg = d.get("regulatory_approvals", {})
    if has_content(reg):
        doc.add_heading("Regulatory Approvals", level=2)
        required = reg.get("required", [])
        if has_content(required):
            for r in required:
                if not is_empty_value(r):
                    doc.add_paragraph(r, style="List Bullet")
        reg_p = doc.add_paragraph()
        add_field(reg_p, "Status: ", reg.get("status"))

    dp = d.get("deal_protections", {})
    if has_content(dp):
        doc.add_heading("Deal Protections", level=2)
        dp_p = doc.add_paragraph()
        add_field(dp_p, "Target Termination Fee: ",
                  dp.get("breakup_fee_target"))
        add_field(dp_p, "Acquirer Termination Fee: ",
                  dp.get("breakup_fee_acquirer"))
        add_field(dp_p, "Go-Shop: ", dp.get("go_shop"))
        add_field(dp_p, "Matching Rights: ", dp.get("matching_rights"))
        add_field(dp_p, "No-Shop: ", dp.get("no_shop"))

    timeline = d.get("expected_timeline")
    vote = d.get("shareholder_vote")
    if not is_empty_value(timeline) or not is_empty_value(vote):
        doc.add_heading("Timeline & Vote", level=2)
        tv_p = doc.add_paragraph()
        add_field(tv_p, "Expected Timeline: ", timeline)
        add_field(tv_p, "Shareholder Vote: ", vote)

    if not is_empty_value(d.get("fairness_opinion")):
        doc.add_heading("Fairness Opinion", level=2)
        doc.add_paragraph(d["fairness_opinion"])

    pro_forma = d.get("pro_forma_highlights", [])
    if has_content(pro_forma):
        doc.add_heading("Pro Forma Highlights", level=2)
        for pf in pro_forma:
            if not is_empty_value(pf):
                doc.add_paragraph(pf, style="List Bullet")

    if not is_empty_value(d.get("background_summary")):
        doc.add_heading("Background of Transaction", level=2)
        doc.add_paragraph(d["background_summary"])

    risks = d.get("risk_factors", [])
    if has_content(risks):
        doc.add_heading("Key Risk Factors", level=2)
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

    print(f"Fetching S-4 from: {source}")

    text = fetch_filing_text(source)
    print(f"Extracted {len(text.split())} words of text")

    print("Generating summary via Claude Opus 4.5...")
    result = summarize(text)
    from ._ticker_context import apply_known_tickers
    result = apply_known_tickers(result, DEAL_CONTEXT)

    print_summary(result)

    uid = filing_uid(FILING_URL)
    from .s3_utils import upload_json

    s3_json_path, s3_json_url = upload_json(result, f"s4_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    safe_ticker = sanitize_filename_part(result.get("target_ticker"))
    safe_date = sanitize_date_part(result.get("filing_date"))
    docx_suffix = f"S4_Summary_{safe_ticker}_{safe_date}_{uid}.docx"
    s3_docx_path, s3_docx_url = export_docx(result, docx_suffix)
    print(f"DOCX uploaded to S3: {s3_docx_url}")

    result["s3_docx_path"] = s3_docx_path
    result["s3_docx_url"] = s3_docx_url
    result["s3_json_path"] = s3_json_path
    result["s3_json_url"] = s3_json_url
    return result


if __name__ == "__main__":
    main()
