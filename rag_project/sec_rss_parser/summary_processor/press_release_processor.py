"""
Press Release Processor
Extracts structured deal financial data from press release summaries (L1/L2/L3 format)
using Claude Haiku, then saves to MongoDB (fo_press_release_extraction collection).

Input:  L1/L2/L3 summary text from EX-99.1
Output: fo_press_release_extraction MongoDB record
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)


def get_press_release_data(deal_id: str = None, accession_number: str = None):
    """
    Return saved press release extraction for a deal or accession number, or None.
    Queries MongoDB fo_press_release_extraction collection.
    """
    from sec_rss_parser.models import FOPressReleaseExtraction

    if deal_id:
        record = FOPressReleaseExtraction.objects(deal_id=deal_id).first()
        if record:
            return {
                "deal_id": record.deal_id,
                "accession_number": record.accession_number,
                "extracted": record.extracted,
                "source_text": record.source_text,
                "extracted_at": record.extracted_at.isoformat() + "Z" if record.extracted_at else None,
                "filing_date": record.filing_date,
                "press_release_id": record.press_release_id,
                "press_release_docx": record.press_release_docx,
                "company_name": record.company_name,
                "cik_number": record.cik_number,
            }

    if accession_number:
        record = FOPressReleaseExtraction.objects(
            accession_number=accession_number).first()
        if record:
            return {
                "deal_id": record.deal_id,
                "accession_number": record.accession_number,
                "extracted": record.extracted,
                "source_text": record.source_text,
                "extracted_at": record.extracted_at.isoformat() + "Z" if record.extracted_at else None,
                "filing_date": record.filing_date,
                "press_release_id": record.press_release_id,
                "press_release_docx": record.press_release_docx,
                "company_name": record.company_name,
                "cik_number": record.cik_number,
            }

    return None


def extract_from_press_release(
    summary_text: str,
    deal_id: str = None,
    accession_number: str = None,
    company_name: str = None,
    cik_number: str = None,
    press_release_id: str = None,
    press_release_docx: str = None,
    filing_date: str = None,
    send_email: bool = True,
) -> dict:
    """
    Send the L1/L2/L3 press release summary to Claude Haiku for structured extraction.
    Saves result to MongoDB fo_press_release_extraction collection and returns it.

    Args:
        summary_text: The L1/L2/L3 summary text from EX-99.1
        deal_id: Deal ID (optional)
        accession_number: SEC accession number (optional)
        company_name: Company name (optional)
        cik_number: CIK number (optional)
        press_release_id: Reference to sec_filing_summary._id (optional)
        press_release_docx: S3 URL of the press release summary DOCX (optional)
        filing_date: Filing date string (optional)
        send_email: Whether to send extraction email (default True)

    Returns:
        dict with deal_id, extracted, source_text, extracted_at, filing_date, etc.
    """
    import anthropic
    from sec_rss_parser.models import FOPressReleaseExtraction
    from sec_rss_parser.email_templates import generate_press_release_extraction_email_html
    from sec_rss_parser.utils_8k import send_webhook_notification

    N8N_WEBHOOK_URL = os.environ.get("N8N_WEKHOOK_INTERNAL_WITH_JOSH",
                                     "https://n8n.arbintel.cloud/webhook/b3007d21-6845-47b5-aece-7b26583758bc")

    logger.info(
        "press_release_processor: extract_from_press_release deal_id=%s accession=%s",
        deal_id, accession_number
    )

    client = anthropic.Anthropic()

    prompt = f"""Extract structured deal financial data from this press release summary.
Return ONLY valid JSON with these fields (use null for anything not stated):

{{
  "target": "Target company name",
  "acquirer": "Acquirer name(s)",
  "offer_price_cash": 0.00,
  "cvr_value": 0.00,
  "stock_exchange_ratio": null,
  "total_consideration": "PER-SHARE total (cash + stock value + CVR). NOT the aggregate deal value. E.g. if offer is $21.50 cash, total_consideration = 21.50",
  "deal_value_bn": "Aggregate deal/equity value in BILLIONS (e.g. 2.5 for a $2.5B deal)",
  "deal_type": "cash / stock / cash+stock / cash+CVR",
  "premium_pct": 0.0,
  "undisturbed_date": "YYYY-MM-DD or null",
  "undisturbed_reference": "description of what the undisturbed price is measured against",
  "expected_close": "Use standard finance shorthand: 1H26, 2H26, Q1 26, Q2 26, mid-2026, etc.",
  "expected_close_date": "YYYY-MM-DD midpoint of the stated range (e.g. 1H26 = 2026-04-01, Q2 26 = 2026-05-15, mid-2026 = 2026-07-01)",
  "announce_date": "YYYY-MM-DD date the deal was announced",
  "go_shop_days": null,
  "diluted_shares_mm": null,
  "cash_on_hand_bn": null,
  "debt_bn": null,
  "financing": "description of financing",
  "regulatory_bodies": ["list of regulatory bodies mentioned"],
  "shareholder_approval_required": true,
  "dividend_info": "any dividend or distribution info mentioned",
  "special_conditions": "any CVR milestones, earnouts, or special terms",
  "minority_investors": ["any minority equity investors"],
  "raw_summary": "the one-line L1 headline"
}}

Press release summary:
{summary_text}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    extracted = json.loads(text)

    extracted_filing_date = filing_date or extracted.get("announce_date")

    now = datetime.utcnow()

    existing = None
    if deal_id:
        existing = FOPressReleaseExtraction.objects(deal_id=deal_id).first()
    if not existing and accession_number:
        existing = FOPressReleaseExtraction.objects(
            accession_number=accession_number).first()

    if existing:
        existing.extracted = extracted
        existing.source_text = summary_text
        existing.extracted_at = now
        existing.filing_date = extracted_filing_date
        existing.press_release_id = press_release_id or existing.press_release_id
        existing.press_release_docx = press_release_docx or existing.press_release_docx
        existing.company_name = company_name or existing.company_name
        existing.cik_number = cik_number or existing.cik_number
        if deal_id:
            existing.deal_id = deal_id
        if accession_number:
            existing.accession_number = accession_number
        existing.save()
        logger.info(
            "press_release_processor: updated existing record _id=%s", existing._id)
    else:
        record = FOPressReleaseExtraction(
            deal_id=deal_id,
            accession_number=accession_number,
            extracted=extracted,
            source_text=summary_text,
            extracted_at=now,
            filing_date=extracted_filing_date,
            press_release_id=press_release_id,
            press_release_docx=press_release_docx,
            company_name=company_name,
            cik_number=cik_number,
        )
        record.save()
        logger.info(
            "press_release_processor: created new record _id=%s", record._id)

    result = {
        "deal_id": deal_id,
        "accession_number": accession_number,
        "extracted": extracted,
        "source_text": summary_text,
        "extracted_at": now.isoformat() + "Z",
        "filing_date": extracted_filing_date,
        "press_release_id": press_release_id,
        "press_release_docx": press_release_docx,
        "company_name": company_name,
        "cik_number": cik_number,
    }

    if send_email:
        try:
            subject, html_email = generate_press_release_extraction_email_html(
                company_name=company_name or extracted.get(
                    "target") or "Unknown Company",
                deal_id=deal_id or "N/A",
                accession_number=accession_number or "N/A",
                extracted=extracted,
                filing_date=extracted_filing_date,
                cik_number=cik_number,
                press_release_docx=press_release_docx,
            )
            payload = {
                "subject": subject,
                "html": html_email,
                "company_name": company_name or extracted.get("target") or "Unknown Company",
                "deal_id": deal_id,
                "accession_number": accession_number,
                "email_type": "press_release_extraction",
            }
            send_webhook_notification(
                N8N_WEBHOOK_URL, payload, "Press Release Extraction email")
            logger.info(
                "press_release_processor: sent extraction email for deal_id=%s", deal_id)
        except Exception as e:
            logger.exception(
                "press_release_processor: failed to send email error=%s", str(e))

    return result
