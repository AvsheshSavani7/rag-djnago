"""
F-4 Registration Statement Summarizer — Multi-level summaries via Claude API
Usage: python f4_summary.py

F-4 is the foreign private issuer equivalent of the S-4. It registers securities
issued in business combinations involving at least one foreign private issuer.
Used for cross-border M&A where the acquirer or combined entity is a foreign company
listed (or to be listed) on a US exchange.
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

# ──── PASTE YOUR F-4 URL HERE ────
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


SUMMARY_PROMPT = """You are an expert merger arbitrage analyst summarizing SEC Form F-4 registration statements for a merger arbitrage trading desk.

F-4 is the foreign private issuer equivalent of the S-4. It registers securities to be issued in a business combination where at least one party is a foreign private issuer. These filings are used in cross-border M&A and contain the merger agreement, pro forma financials, risk factors, and deal terms — plus additional cross-border considerations like currency, jurisdiction, and international regulatory approvals.

Given the F-4 text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "filing_type": "<F-4 | F-4/A>",
  "acquirer": "<acquiring company>",
  "acquirer_ticker": "<acquirer US ticker or ADR>",
  "acquirer_home_country": "<country of incorporation>",
  "target": "<target company>",
  "target_ticker": "<target ticker as stated in the filing, or null if not stated>",
  "target_home_country": "<target country of incorporation>",
  "filing_date": "<MM/DD/YY>",

  "L1_headline": "+ <TARGET TICKER> – <key event in ≤8 words>. | <date>",

  "L2_brief": "<2-3 sentence summary covering: the cross-border deal structure, consideration, and current status>",

  "L3_detailed": {
    "deal_structure": "<merger, scheme of arrangement, exchange offer, reverse merger, etc.>",
    "consideration": {
      "type": "<Cash | Stock | Cash & Stock | ADR exchange>",
      "per_share_value": "<per-share value to target shareholders>",
      "exchange_ratio": "<exchange ratio if stock/ADR deal>",
      "cash_component": "<cash per share if mixed>",
      "total_deal_value": "<aggregate deal value in USD>",
      "premium": "<premium with reference price/date>",
      "currency_details": "<currencies involved, FX conversion mechanics>"
    },
    "conditions_precedent": ["<shareholder approvals (both sides), regulatory approvals, court approvals for schemes, financing, MAE>"],
    "regulatory_approvals": {
      "required": ["<US: HSR/DOJ/FTC/CFIUS. EU: European Commission. Plus home-country regulators, sector-specific>"],
      "status": "<current status of each>"
    },
    "cross_border_considerations": {
      "structure_rationale": "<stated reasons for the transaction structure, if disclosed>",
      "tax_treatment": "<tax implications for US shareholders, withholding, treaty benefits>",
      "listing_plans": "<where combined company will be listed — US exchange, home exchange, dual listing>",
      "currency_exposure": "<FX risk to shareholders — hedging, conversion at closing>"
    },
    "deal_protections": {
      "breakup_fee_target": "<target termination fee>",
      "breakup_fee_acquirer": "<acquirer/reverse termination fee>",
      "matching_rights": "<matching rights details>",
      "other_protections": "<scheme court approval, no-shop, etc.>"
    },
    "expected_timeline": "<expected closing date and key milestones>",
    "shareholder_vote": "<which shareholders must approve, vote thresholds (note: schemes often require 75%), court approvals>",
    "pro_forma_highlights": ["<key combined financials, synergies, accretion/dilution>"],
    "risk_factors": ["<top 3-5 deal-specific risks, including cross-border risks>"]
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
- Identify the countries involved and any cross-border structural considerations
- Note if this is a scheme of arrangement (common in UK/Australian deals) vs. a traditional merger
- Extract exact consideration in BOTH local currency and USD equivalents where available
- Flag currency conversion mechanics — fixed vs. floating exchange ratios
- List ALL regulatory bodies across ALL jurisdictions that must approve
- Note any different vote thresholds (schemes often need 75% vs. 50%+1 for US mergers)
- Highlight tax implications for US-based shareholders
- Note listing plans for the combined entity

F-4 TEXT:
"""
EXTRACTION_GUIDANCE = """This is an F-4 registration statement (cross-border merger registration).
Extract the following sections in full:
- Deal structure: merger, scheme of arrangement, exchange offer
- Consideration: per-share value, exchange ratio, cash component, total value, premium, currencies
- Currency conversion mechanics (fixed vs floating)
- ALL conditions precedent including court approvals for schemes
- Regulatory approvals across ALL jurisdictions (US, EU, home-country, sector-specific) with status
- Cross-border considerations: tax treatment, listing plans, currency exposure, structure rationale
- Deal protections: termination fees (both sides), matching rights, go-shop, no-shop
- Shareholder vote thresholds (note if scheme requires 75%)
- Pro forma financials and synergy estimates
- Fairness opinion: advisor name and conclusion
- Background of the Transaction
- Key risk factors including cross-border risks
- If amendment (F-4/A): what changed from prior filing"""


def fetch_filing_text(source: str) -> str:
    """Fetch and extract text from an F-4 filing (URL, local file, or PDF)."""
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
    print(f"  F-4 SUMMARY (Cross-Border M&A Registration)")
    print("=" * 70)

    print(
        f"\n   Acquirer: {s.get('acquirer', 'N/A')} ({s.get('acquirer_ticker') or '—'}) — {s.get('acquirer_home_country', 'N/A')}")
    print(
        f"   Target:   {s.get('target', 'N/A')} ({s.get('target_ticker') or '—'}) — {s.get('target_home_country', 'N/A')}")
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
    if c.get("currency_details"):
        print(f"   Currency:     {c['currency_details']}")

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

    cb = d.get("cross_border_considerations", {})
    print(f"\n   ── CROSS-BORDER ──")
    print(f"   Structure Rationale: {cb.get('structure_rationale', 'N/A')}")
    print(f"   Tax Treatment:      {cb.get('tax_treatment', 'N/A')}")
    print(f"   Listing Plans:      {cb.get('listing_plans', 'N/A')}")
    print(f"   Currency Exposure:  {cb.get('currency_exposure', 'N/A')}")

    dp = d.get("deal_protections", {})
    print(f"\n   ── DEAL PROTECTIONS ──")
    print(f"   Target Fee:   {dp.get('breakup_fee_target', 'N/A')}")
    print(f"   Acquirer Fee: {dp.get('breakup_fee_acquirer', 'N/A')}")
    print(f"   Match Rights: {dp.get('matching_rights', 'N/A')}")
    print(f"   Other:        {dp.get('other_protections', 'N/A')}")

    print(f"\n   Timeline:     {d.get('expected_timeline', 'N/A')}")
    print(f"   Vote:         {d.get('shareholder_vote', 'N/A')}")

    if d.get("pro_forma_highlights"):
        print(f"\n   ── PRO FORMA ──")
        for p in d["pro_forma_highlights"]:
            print(f"     • {p}")

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

    title = doc.add_heading(f"F-4 Summary: {s.get('target', 'N/A')}", level=0)
    title.runs[0].font.size = Pt(20)

    meta = doc.add_paragraph()
    meta.add_run("Acquirer: ").bold = True
    meta.add_run(
        f"{s.get('acquirer', 'N/A')} ({s.get('acquirer_ticker') or '—'}) — {s.get('acquirer_home_country', 'N/A')}")

    meta1b = doc.add_paragraph()
    meta1b.add_run("Target: ").bold = True
    meta1b.add_run(
        f"{s.get('target', 'N/A')} ({s.get('target_ticker') or '—'}) — {s.get('target_home_country', 'N/A')}")

    meta2 = doc.add_paragraph()
    meta2.add_run("Filing Date: ").bold = True
    meta2.add_run(date)
    meta2.add_run("    Filing Type: ").bold = True
    meta2.add_run(s.get("filing_type", "F-4"))

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
        add_field(terms_p, "Currency Details: ", c.get("currency_details"))

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

    cb = d.get("cross_border_considerations", {})
    if has_content(cb):
        doc.add_heading("Cross-Border Considerations", level=2)
        cb_p = doc.add_paragraph()
        add_field(cb_p, "Structure Rationale: ", cb.get("structure_rationale"))
        add_field(cb_p, "Tax Treatment: ", cb.get("tax_treatment"))
        add_field(cb_p, "Listing Plans: ", cb.get("listing_plans"))
        add_field(cb_p, "Currency Exposure: ", cb.get("currency_exposure"))

    dp = d.get("deal_protections", {})
    if has_content(dp):
        doc.add_heading("Deal Protections", level=2)
        dp_p = doc.add_paragraph()
        add_field(dp_p, "Target Termination Fee: ",
                  dp.get("breakup_fee_target"))
        add_field(dp_p, "Acquirer Termination Fee: ",
                  dp.get("breakup_fee_acquirer"))
        add_field(dp_p, "Matching Rights: ", dp.get("matching_rights"))
        add_field(dp_p, "Other Protections: ", dp.get("other_protections"))

    timeline = d.get("expected_timeline")
    vote = d.get("shareholder_vote")
    if not is_empty_value(timeline) or not is_empty_value(vote):
        doc.add_heading("Timeline & Vote", level=2)
        tv_p = doc.add_paragraph()
        add_field(tv_p, "Expected Timeline: ", timeline)
        add_field(tv_p, "Shareholder Vote: ", vote)

    pro_forma = d.get("pro_forma_highlights", [])
    if has_content(pro_forma):
        doc.add_heading("Pro Forma Highlights", level=2)
        for pf in pro_forma:
            if not is_empty_value(pf):
                doc.add_paragraph(pf, style="List Bullet")

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

    print(f"Fetching F-4 from: {source}")

    text = fetch_filing_text(source)
    print(f"Extracted {len(text.split())} words of text")

    print("Generating summary via Claude Opus 4.5...")
    result = summarize(text)

    print_summary(result)

    uid = filing_uid(FILING_URL)
    from .s3_utils import upload_json

    s3_json_path, s3_json_url = upload_json(result, f"f4_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    safe_ticker = sanitize_filename_part(result.get("target_ticker"))
    safe_date = sanitize_date_part(result.get("filing_date"))
    docx_suffix = f"F4_Summary_{safe_ticker}_{safe_date}_{uid}.docx"
    s3_docx_path, s3_docx_url = export_docx(result, docx_suffix)
    print(f"DOCX uploaded to S3: {s3_docx_url}")

    result["s3_docx_path"] = s3_docx_path
    result["s3_docx_url"] = s3_docx_url
    result["s3_json_path"] = s3_json_path
    result["s3_json_url"] = s3_json_url
    return result


if __name__ == "__main__":
    main()
