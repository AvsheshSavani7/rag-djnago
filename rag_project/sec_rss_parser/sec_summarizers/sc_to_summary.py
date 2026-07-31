"""
SC TO-T / SC 14D-9 Tender Offer Summarizer — Multi-level summaries via Claude API
Usage: python sc_to_summary.py

Covers:
  SC TO-T  — Tender offer statement by third party (the bidder's filing)
  SC TO-T/A — Amendment to tender offer statement
  SC 14D-9  — Solicitation/recommendation statement (the target board's response)
  SC 14D-9/A — Amendment to solicitation/recommendation
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

# ──── PASTE YOUR SC TO-T or SC 14D-9 URL HERE ────
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


SUMMARY_PROMPT = """You are an expert merger arbitrage analyst summarizing SEC tender offer filings (SC TO-T and SC 14D-9) for a merger arbitrage trading desk.

These are among the most critical filings for merger arb:
- **SC TO-T** is filed by the bidder (or third party) making the tender offer. It lays out the offer price, conditions, expiration, financing, and bidder's intentions.
- **SC 14D-9** is filed by the target company's board in response. It contains the board's recommendation (accept/reject/neutral), the background of the transaction, fairness opinion, and any competing offers or alternatives considered.

Amendments (/A) update prior filings — track what changed.

Given the tender offer filing text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "filing_type": "<SC TO-T | SC TO-T/A | SC 14D-9 | SC 14D-9/A>",
  "bidder": "<name of entity making the tender offer>",
  "bidder_ticker": "<bidder ticker as stated in the filing, or null if not stated>",
  "target": "<name of target company>",
  "target_ticker": "<target ticker as stated in the filing, or null if not stated>",
  "filing_date": "<MM/DD/YY>",

  "L1_headline": "+ <TARGET TICKER> – <key event in ≤8 words>. | <date>",

  "L2_brief": "<2-3 sentence summary covering: who is bidding for whom, offer price and premium, board recommendation, and current status>",

  "L3_detailed": {
    "offer_terms": {
      "offer_price": "<price per share — cash, stock, or mix>",
      "offer_type": "<Cash | Stock | Cash & Stock | Exchange Offer>",
      "exchange_ratio": "<if stock deal: exchange ratio, or null>",
      "premium": "<premium to unaffected price or last close, with reference date>",
      "total_deal_value": "<aggregate deal value>",
      "shares_sought": "<number/percentage of shares sought — any/all vs. minimum>",
      "minimum_condition": "<minimum tender threshold — e.g., 'majority of outstanding shares'>",
      "top_up_option": "<does bidder have a top-up option to reach short-form merger threshold? Details if yes>"
    },
    "timing": {
      "offer_commenced": "<date tender offer commenced>",
      "expiration_date": "<current expiration date and time>",
      "withdrawal_deadline": "<last date to withdraw tendered shares>",
      "expected_closing": "<expected closing/settlement date>",
      "extensions": "<any extensions already made or extension rights>"
    },
    "board_recommendation": "<RECOMMEND | AGAINST | NO RECOMMENDATION | UNABLE TO DETERMINE — with detail on reasoning>",
    "fairness_opinion": "<which advisor delivered the opinion, and their conclusion>",
    "background_of_transaction": "<summary of how the deal came about — timeline of negotiations, competing bidders, prior approaches>",
    "conditions_to_offer": ["<each material condition — minimum tender, regulatory approvals, financing, MAE, no injunctions, etc.>"],
    "financing": {
      "source": "<how the offer is being financed — cash on hand, committed debt financing, equity>",
      "committed_financing": "<details of commitment letters — lenders, amounts>",
      "financing_condition": "<is there a financing condition/financing out? YES or NO>"
    },
    "regulatory_approvals": {
      "required": ["<each regulatory approval needed — HSR/DOJ, FTC, CFIUS, EU, sector-specific>"],
      "status": "<current status of regulatory review — filed, under review, second request, approved, etc.>",
      "expected_timeline": "<expected regulatory timeline>"
    },
    "deal_protections": {
      "breakup_fee": "<termination/breakup fee amount and triggers>",
      "reverse_breakup_fee": "<reverse termination fee, if any>",
      "go_shop": "<go-shop period details — duration, end date, matching rights>",
      "matching_rights": "<does bidder have matching rights on competing offers? Details>",
      "force_the_vote": "<any force-the-vote provisions>"
    },
    "competing_offers": "<any competing or superior proposals mentioned>",
    "litigation": "<any lawsuits filed related to the tender offer>",
    "dissenting_shareholders": "<any known opposition from major shareholders>",
    "amendment_changes": "<if this is an amendment (/A): what specifically changed from the prior filing>",
    "risks_flagged": ["<deal risks — regulatory, financing, conditions, litigation, competing bids, MAC/MAE>"]
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
- Be extremely thorough — extract all stated facts
- Extract EXACT offer price, premium calculation (with reference date/price), minimum condition, and expiration date
- The board recommendation and fairness opinion are critical — quote key language
- List EVERY material condition to the offer individually
- Distinguish between financing condition and committed financing as stated
- For SC 14D-9: extract the full "Background of the Transaction" negotiation history
- For amendments: clearly state what changed from the prior filing
- List all conditions to the offer as stated
- Note the top-up option and short-form merger mechanics if present

TENDER OFFER FILING TEXT:
"""

EXTRACTION_GUIDANCE = """This is a TENDER OFFER filing (SC TO-T, SC TO-T/A, SC 14D-9, or SC 14D-9/A).
Extract the following sections in full:
- Offer terms: offer price, exchange ratio, premium calculation (with reference date/price), shares sought, minimum condition, top-up option
- Expiration date and time, withdrawal deadline, any extensions or extension rights
- Board recommendation and reasoning (for SC 14D-9)
- Fairness opinion: advisor name, conclusion, and key language
- Background of the Transaction (full negotiation history timeline)
- ALL conditions to the offer (listed individually)
- Financing: source of funds, commitment letters, lenders, amounts, financing condition (YES/NO)
- Regulatory approvals: EVERY jurisdiction mentioned (HSR, EU, CFIUS, country-specific antitrust, FDI reviews) with filing dates and current status
- Deal protections: breakup fee amounts and triggers, reverse breakup fee, go-shop period, matching rights, force-the-vote provisions
- Competing offers or superior proposals
- Litigation related to the offer
- Top-up option and short-form merger mechanics (DGCL Section 251(h))
- If amendment (/A): what specifically changed from prior filing"""


def fetch_filing_text(source: str) -> str:
    """Fetch and extract text from a tender offer filing (URL, local file, or PDF)."""
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
    print(f"  {s.get('filing_type', 'TENDER OFFER')} SUMMARY")
    print("=" * 70)

    print(
        f"\n   Bidder:  {s.get('bidder', 'N/A')} ({s.get('bidder_ticker') or '—'})")
    print(
        f"   Target:  {s.get('target', 'N/A')} ({s.get('target_ticker') or '—'})")
    print(f"   Type:    {s.get('filing_type', 'N/A')}")
    print(f"   Date:    {s.get('filing_date', 'N/A')}")

    print(f"\n📌 L1 | HEADLINE")
    print(f"   {s['L1_headline']}")

    print(f"\n📋 L2 | BRIEF")
    print(f"   {s['L2_brief']}")

    d = s["L3_detailed"]

    # Offer Terms
    ot = d.get("offer_terms", {})
    print(f"\n📊 L3 | DETAILED")
    print(f"\n   ── OFFER TERMS ──")
    print(f"   Price:        {ot.get('offer_price', 'N/A')}")
    print(f"   Type:         {ot.get('offer_type', 'N/A')}")
    if ot.get("exchange_ratio"):
        print(f"   Exch Ratio:   {ot['exchange_ratio']}")
    print(f"   Premium:      {ot.get('premium', 'N/A')}")
    print(f"   Deal Value:   {ot.get('total_deal_value', 'N/A')}")
    print(f"   Shares Sought:{ot.get('shares_sought', 'N/A')}")
    print(f"   Min Condition: {ot.get('minimum_condition', 'N/A')}")
    if ot.get("top_up_option"):
        print(f"   Top-Up Option: {ot['top_up_option']}")

    # Timing
    tm = d.get("timing", {})
    print(f"\n   ── TIMING ──")
    print(f"   Commenced:    {tm.get('offer_commenced', 'N/A')}")
    print(f"   Expires:      {tm.get('expiration_date', 'N/A')}")
    print(f"   Withdrawal:   {tm.get('withdrawal_deadline', 'N/A')}")
    print(f"   Exp. Close:   {tm.get('expected_closing', 'N/A')}")
    if tm.get("extensions"):
        print(f"   Extensions:   {tm['extensions']}")

    # Board & Fairness
    print(f"\n   ── BOARD RECOMMENDATION ──")
    print(f"   Recommendation: {d.get('board_recommendation', 'N/A')}")
    print(f"   Fairness Opinion: {d.get('fairness_opinion', 'N/A')}")

    # Background
    if d.get("background_of_transaction"):
        print(f"\n   ── BACKGROUND ──")
        print(f"   {d['background_of_transaction']}")

    # Conditions
    if d.get("conditions_to_offer"):
        print(f"\n   ── CONDITIONS TO OFFER ──")
        for c in d["conditions_to_offer"]:
            print(f"     • {c}")

    # Financing
    fin = d.get("financing", {})
    print(f"\n   ── FINANCING ──")
    print(f"   Source:       {fin.get('source', 'N/A')}")
    print(f"   Committed:    {fin.get('committed_financing', 'N/A')}")
    print(f"   Fin. Condition: {fin.get('financing_condition', 'N/A')}")

    # Regulatory
    reg = d.get("regulatory_approvals", {})
    print(f"\n   ── REGULATORY ──")
    if reg.get("required"):
        for r in reg["required"]:
            print(f"     • {r}")
    print(f"   Status:       {reg.get('status', 'N/A')}")
    print(f"   Timeline:     {reg.get('expected_timeline', 'N/A')}")

    # Deal Protections
    dp = d.get("deal_protections", {})
    print(f"\n   ── DEAL PROTECTIONS ──")
    print(f"   Breakup Fee:  {dp.get('breakup_fee', 'N/A')}")
    print(f"   Reverse Fee:  {dp.get('reverse_breakup_fee', 'N/A')}")
    print(f"   Go-Shop:      {dp.get('go_shop', 'N/A')}")
    print(f"   Match Rights: {dp.get('matching_rights', 'N/A')}")
    if dp.get("force_the_vote"):
        print(f"   Force Vote:   {dp['force_the_vote']}")

    # Other
    if d.get("competing_offers"):
        print(f"\n   Competing Offers: {d['competing_offers']}")
    if d.get("litigation"):
        print(f"   Litigation:       {d['litigation']}")
    if d.get("dissenting_shareholders"):
        print(f"   Dissenters:       {d['dissenting_shareholders']}")
    if d.get("amendment_changes"):
        print(f"\n   ── AMENDMENT CHANGES ──")
        print(f"   {d['amendment_changes']}")

    if d.get("risks_flagged"):
        print(f"\n   ── RISKS ──")
        for r in d["risks_flagged"]:
            print(f"     • {r}")

    print("=" * 70)


def export_docx(s: dict, s3_key_suffix: str):
    """Build summary as Word doc, upload to S3 (summary_docx/), return (s3_path, s3_url)."""
    from .fetch_utils import is_empty_value, has_content, add_field
    from .s3_utils import upload_docx_bytes

    filing_type = s.get("filing_type", "SC_TO")
    date = s.get("filing_date", "")
    doc = DocxDocument()

    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    title = doc.add_heading(
        f"{filing_type} Summary: {s.get('target', 'N/A')}", level=0)
    title.runs[0].font.size = Pt(20)

    # Deal parties
    meta = doc.add_paragraph()
    meta.add_run("Bidder: ").bold = True
    meta.add_run(f"{s.get('bidder', 'N/A')} ({s.get('bidder_ticker') or '—'})")
    meta.add_run("    Target: ").bold = True
    meta.add_run(f"{s.get('target', 'N/A')} ({s.get('target_ticker') or '—'})")

    meta2 = doc.add_paragraph()
    meta2.add_run("Filing Date: ").bold = True
    meta2.add_run(date)
    meta2.add_run("    Filing Type: ").bold = True
    meta2.add_run(filing_type)

    # L1
    doc.add_heading("L1 — Headline", level=1)
    p = doc.add_paragraph()
    run = p.add_run(s["L1_headline"])
    run.bold = True
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(0, 51, 102)

    # L2
    doc.add_heading("L2 — Brief", level=1)
    doc.add_paragraph(s["L2_brief"])

    # L3
    doc.add_heading("L3 — Detailed", level=1)
    d = s["L3_detailed"]

    # Offer Terms
    ot = d.get("offer_terms", {})
    if has_content(ot):
        doc.add_heading("Offer Terms", level=2)
        terms_p = doc.add_paragraph()
        add_field(terms_p, "Offer Price: ", ot.get("offer_price"))
        add_field(terms_p, "Offer Type: ", ot.get("offer_type"))
        add_field(terms_p, "Exchange Ratio: ", ot.get("exchange_ratio"))
        add_field(terms_p, "Premium: ", ot.get("premium"))
        add_field(terms_p, "Total Deal Value: ", ot.get("total_deal_value"))
        add_field(terms_p, "Shares Sought: ", ot.get("shares_sought"))
        add_field(terms_p, "Minimum Condition: ", ot.get("minimum_condition"))
        add_field(terms_p, "Top-Up Option: ", ot.get("top_up_option"))

    # Timing
    tm = d.get("timing", {})
    if has_content(tm):
        doc.add_heading("Timing", level=2)
        time_p = doc.add_paragraph()
        add_field(time_p, "Offer Commenced: ", tm.get("offer_commenced"))
        add_field(time_p, "Expiration Date: ", tm.get("expiration_date"))
        add_field(time_p, "Withdrawal Deadline: ",
                  tm.get("withdrawal_deadline"))
        add_field(time_p, "Expected Closing: ", tm.get("expected_closing"))
        add_field(time_p, "Extensions: ", tm.get("extensions"))

    # Board Recommendation (keep special color formatting)
    board_rec = d.get("board_recommendation")
    if not is_empty_value(board_rec):
        doc.add_heading("Board Recommendation", level=2)
        rec_p = doc.add_paragraph()
        rec_run = rec_p.add_run(board_rec)
        rec_run.bold = True
        rec_run.font.size = Pt(12)
        rec_upper = board_rec.upper()
        if "RECOMMEND" in rec_upper and "AGAINST" not in rec_upper:
            rec_run.font.color.rgb = RGBColor(0, 128, 0)
        elif "AGAINST" in rec_upper:
            rec_run.font.color.rgb = RGBColor(192, 0, 0)

    if not is_empty_value(d.get("fairness_opinion")):
        doc.add_heading("Fairness Opinion", level=2)
        doc.add_paragraph(d["fairness_opinion"])

    # Background
    if not is_empty_value(d.get("background_of_transaction")):
        doc.add_heading("Background of Transaction", level=2)
        doc.add_paragraph(d["background_of_transaction"])

    # Conditions
    conditions = d.get("conditions_to_offer", [])
    if has_content(conditions):
        doc.add_heading("Conditions to Offer", level=2)
        for c in conditions:
            if not is_empty_value(c):
                doc.add_paragraph(c, style="List Bullet")

    # Financing (keep special color formatting for financing condition)
    fin = d.get("financing", {})
    if has_content(fin):
        doc.add_heading("Financing", level=2)
        fin_p = doc.add_paragraph()
        add_field(fin_p, "Source: ", fin.get("source"))
        add_field(fin_p, "Committed Financing: ",
                  fin.get("committed_financing"))
        fin_cond = fin.get("financing_condition")
        if not is_empty_value(fin_cond):
            fin_p.add_run("Financing Condition: ").bold = True
            fin_run = fin_p.add_run(fin_cond)
            if "YES" in fin_cond.upper():
                fin_run.font.color.rgb = RGBColor(192, 0, 0)
                fin_run.bold = True

    # Regulatory
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
        add_field(reg_p, "Expected Timeline: ", reg.get("expected_timeline"))

    # Deal Protections
    dp = d.get("deal_protections", {})
    if has_content(dp):
        doc.add_heading("Deal Protections", level=2)
        dp_p = doc.add_paragraph()
        add_field(dp_p, "Breakup Fee: ", dp.get("breakup_fee"))
        add_field(dp_p, "Reverse Breakup Fee: ", dp.get("reverse_breakup_fee"))
        add_field(dp_p, "Go-Shop: ", dp.get("go_shop"))
        add_field(dp_p, "Matching Rights: ", dp.get("matching_rights"))
        add_field(dp_p, "Force the Vote: ", dp.get("force_the_vote"))

    # Other sections
    if not is_empty_value(d.get("competing_offers")):
        doc.add_heading("Competing Offers", level=2)
        doc.add_paragraph(d["competing_offers"])

    if not is_empty_value(d.get("litigation")):
        doc.add_heading("Litigation", level=2)
        doc.add_paragraph(d["litigation"])

    if not is_empty_value(d.get("dissenting_shareholders")):
        doc.add_heading("Dissenting Shareholders", level=2)
        doc.add_paragraph(d["dissenting_shareholders"])

    if not is_empty_value(d.get("amendment_changes")):
        doc.add_heading("Amendment Changes", level=2)
        doc.add_paragraph(d["amendment_changes"])

    risks = d.get("risks_flagged", [])
    if not is_empty_value(risks):
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

    print(f"Fetching tender offer filing from: {source}")

    text = fetch_filing_text(source)
    print(f"Extracted {len(text.split())} words of text")

    print("Generating summary via Claude Opus 4.5...")
    result = summarize(text)

    print_summary(result)

    uid = filing_uid(FILING_URL)
    from .s3_utils import upload_json

    s3_json_path, s3_json_url = upload_json(
        result, f"sc_to_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    safe_type = sanitize_filename_part(result.get("filing_type"), "SC_TO")
    safe_ticker = sanitize_filename_part(result.get("target_ticker"))
    safe_date = sanitize_date_part(result.get("filing_date"))
    docx_suffix = f"{safe_type}_Summary_{safe_ticker}_{safe_date}_{uid}.docx"
    s3_docx_path, s3_docx_url = export_docx(result, docx_suffix)
    print(f"DOCX uploaded to S3: {s3_docx_url}")

    result["s3_docx_path"] = s3_docx_path
    result["s3_docx_url"] = s3_docx_url
    result["s3_json_path"] = s3_json_path
    result["s3_json_url"] = s3_json_url
    return result


if __name__ == "__main__":
    main()
