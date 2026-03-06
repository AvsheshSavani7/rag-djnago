"""
S-4 Registration Statement Summarizer — Multi-level summaries via Claude API
Usage: python s4_summary.py

S-4 filings register securities issued in business combinations (mergers, acquisitions,
exchange offers). They contain the merger agreement, pro forma financials, risk factors,
and often serve as the combined proxy/prospectus for the deal.
"""

from pathlib import Path
from ._naming import filing_uid

# ──── PASTE YOUR S-4 URL HERE ────
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


SUMMARY_PROMPT = """You are an expert merger arbitrage analyst summarizing SEC Form S-4 registration statements for a merger arbitrage trading desk.

S-4 filings register securities to be issued in a business combination. They often serve as the combined proxy statement/prospectus and contain the full merger agreement, pro forma financials, risk factors, fairness opinions, and background of the transaction. S-4/A amendments update prior filings.

These are long filings — focus on extracting the information most critical to a merger arb position.

Given the S-4 text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "filing_type": "<S-4 | S-4/A>",
  "acquirer": "<acquiring company>",
  "acquirer_ticker": "<acquirer ticker>",
  "target": "<target company>",
  "target_ticker": "<target ticker>",
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
- TONE: State only facts from the filing. Do NOT speculate on motives, interpret what actions "signal" or "suggest", assess confidence levels, or draw conclusions beyond what is explicitly stated. GOOD: "Company suspended earnings calls due to pending transaction." BAD: "Company suspended earnings calls, signaling high confidence in deal completion."
- L1 format MUST be: + <TARGET TICKER> – <event>. | <date>
- Extract EXACT per-share consideration, exchange ratio, total deal value, and premium
- List ALL closing conditions individually — these determine arb risk
- Capture deal protection terms precisely (termination fees as dollar amounts AND percentages)
- Note whether this is an initial S-4 or an amendment — if amendment, highlight what changed
- Extract pro forma financial highlights and synergy estimates
- Summarize the top risk factors most relevant to deal completion
- If the filing includes the merger agreement, extract key terms from it

S-4 TEXT:
"""


def fetch_filing_text(source: str) -> str:
    """Fetch and extract text from an S-4 filing (URL, local file, or PDF)."""
    from .fetch_utils import fetch_text
    return fetch_text(source, word_limit=20000)


def summarize(text: str, model: str = "claude-opus-4-6") -> dict:
    """Call Claude API to produce multi-level summary."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set. Set it in .env or Django settings (ANTHROPIC_API_KEY).")
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
    print(f"  S-4 SUMMARY (M&A Registration Statement)")
    print("=" * 70)

    print(f"\n   Acquirer: {s.get('acquirer', 'N/A')} ({s.get('acquirer_ticker') or '—'})")
    print(f"   Target:   {s.get('target', 'N/A')} ({s.get('target_ticker') or '—'})")
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
    from .s3_utils import upload_docx_bytes

    doc = DocxDocument()

    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    title = doc.add_heading(f"S-4 Summary: {s.get('target', 'N/A')}", level=0)
    title.runs[0].font.size = Pt(20)

    meta = doc.add_paragraph()
    meta.add_run("Acquirer: ").bold = True
    meta.add_run(f"{s.get('acquirer', 'N/A')} ({s.get('acquirer_ticker') or '—'})")
    meta.add_run("    Target: ").bold = True
    meta.add_run(f"{s.get('target', 'N/A')} ({s.get('target_ticker') or '—'})")

    meta2 = doc.add_paragraph()
    date = s.get("filing_date", "")
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

    doc.add_heading("Deal Terms", level=2)
    terms_p = doc.add_paragraph()
    terms_p.add_run("Structure: ").bold = True
    terms_p.add_run(d.get("deal_structure", "N/A") + "\n")
    terms_p.add_run("Consideration Type: ").bold = True
    terms_p.add_run(c.get("type", "N/A") + "\n")
    terms_p.add_run("Per Share Value: ").bold = True
    terms_p.add_run(c.get("per_share_value", "N/A") + "\n")
    if c.get("exchange_ratio"):
        terms_p.add_run("Exchange Ratio: ").bold = True
        terms_p.add_run(c["exchange_ratio"] + "\n")
    if c.get("cash_component"):
        terms_p.add_run("Cash Component: ").bold = True
        terms_p.add_run(c["cash_component"] + "\n")
    terms_p.add_run("Total Deal Value: ").bold = True
    terms_p.add_run(c.get("total_deal_value", "N/A") + "\n")
    terms_p.add_run("Premium: ").bold = True
    terms_p.add_run(c.get("premium", "N/A"))

    if d.get("conditions_precedent"):
        doc.add_heading("Conditions Precedent", level=2)
        for cond in d["conditions_precedent"]:
            doc.add_paragraph(cond, style="List Bullet")

    reg = d.get("regulatory_approvals", {})
    doc.add_heading("Regulatory Approvals", level=2)
    if reg.get("required"):
        for r in reg["required"]:
            doc.add_paragraph(r, style="List Bullet")
    reg_p = doc.add_paragraph()
    reg_p.add_run("Status: ").bold = True
    reg_p.add_run(reg.get("status", "N/A"))

    dp = d.get("deal_protections", {})
    doc.add_heading("Deal Protections", level=2)
    dp_p = doc.add_paragraph()
    dp_p.add_run("Target Termination Fee: ").bold = True
    dp_p.add_run(dp.get("breakup_fee_target", "N/A") + "\n")
    dp_p.add_run("Acquirer Termination Fee: ").bold = True
    dp_p.add_run(dp.get("breakup_fee_acquirer", "N/A") + "\n")
    dp_p.add_run("Go-Shop: ").bold = True
    dp_p.add_run(dp.get("go_shop", "N/A") + "\n")
    dp_p.add_run("Matching Rights: ").bold = True
    dp_p.add_run(dp.get("matching_rights", "N/A") + "\n")
    dp_p.add_run("No-Shop: ").bold = True
    dp_p.add_run(dp.get("no_shop", "N/A"))

    doc.add_heading("Timeline & Vote", level=2)
    tv_p = doc.add_paragraph()
    tv_p.add_run("Expected Timeline: ").bold = True
    tv_p.add_run(d.get("expected_timeline", "N/A") + "\n")
    tv_p.add_run("Shareholder Vote: ").bold = True
    tv_p.add_run(d.get("shareholder_vote", "N/A"))

    doc.add_heading("Fairness Opinion", level=2)
    doc.add_paragraph(d.get("fairness_opinion", "N/A"))

    if d.get("pro_forma_highlights"):
        doc.add_heading("Pro Forma Highlights", level=2)
        for pf in d["pro_forma_highlights"]:
            doc.add_paragraph(pf, style="List Bullet")

    if d.get("background_summary"):
        doc.add_heading("Background of Transaction", level=2)
        doc.add_paragraph(d["background_summary"])

    if d.get("risk_factors"):
        doc.add_heading("Key Risk Factors", level=2)
        for r in d["risk_factors"]:
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

    print_summary(result)

    uid = filing_uid(FILING_URL)
    from .s3_utils import upload_json

    s3_json_path, s3_json_url = upload_json(result, f"s4_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    target_ticker = result.get("target_ticker", "UNKNOWN")
    date = result.get("filing_date", "")
    safe_ticker = re.sub(r'[^\w\-\.]', '_', target_ticker)
    safe_date = date.replace("/", "-")
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
