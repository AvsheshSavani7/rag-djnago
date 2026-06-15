"""
DMA Summary Processor
Extracts structured deal data from DMA (Definitive Merger Agreement / EX-2.1) summaries
using Claude Haiku, then cross-references against press release extraction
to flag inconsistencies.

Input:  DMA summary text from EX-2.1 processing (deal_dma_summary)
Output: fo_dma_extraction MongoDB record
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)


def get_dma_extract(deal_id: str = None, accession_number: str = None):
    """
    Return saved DMA extraction for a deal or accession number, or None.
    Queries MongoDB fo_dma_extraction collection.
    """
    from sec_rss_parser.models import FODmaExtraction

    if deal_id:
        record = FODmaExtraction.objects(deal_id=deal_id).first()
        if record:
            return {
                "deal_id": record.deal_id,
                "accession_number": record.accession_number,
                "extracted": record.extracted,
                "inconsistencies": record.inconsistencies,
                "source_text": record.source_text,
                "extracted_at": record.extracted_at.isoformat() + "Z" if record.extracted_at else None,
                "filing_date": record.filing_date,
                "dma_summary_id": record.dma_summary_id,
                "dma_summary_docx": record.dma_summary_docx,
                "company_name": record.company_name,
                "cik_number": record.cik_number,
            }

    if accession_number:
        record = FODmaExtraction.objects(
            accession_number=accession_number).first()
        if record:
            return {
                "deal_id": record.deal_id,
                "accession_number": record.accession_number,
                "extracted": record.extracted,
                "inconsistencies": record.inconsistencies,
                "source_text": record.source_text,
                "extracted_at": record.extracted_at.isoformat() + "Z" if record.extracted_at else None,
                "filing_date": record.filing_date,
                "dma_summary_id": record.dma_summary_id,
                "dma_summary_docx": record.dma_summary_docx,
                "company_name": record.company_name,
                "cik_number": record.cik_number,
            }

    return None


def _load_press_release(deal_id: str = None, accession_number: str = None):
    """Load existing press release extraction from MongoDB for comparison."""
    from sec_rss_parser.summary_processor.press_release_processor import get_press_release_data

    pr = get_press_release_data(
        deal_id=deal_id, accession_number=accession_number)
    if pr and "extracted" in pr:
        return pr["extracted"]
    return None


def _check_inconsistencies(dma: dict, pr: dict) -> list:
    """Compare DMA extraction against press release — only financial mismatches."""
    flags = []

    def _flag(field: str, label: str, dma_val, pr_val):
        flags.append({
            "field": field,
            "label": label,
            "dma_value": str(dma_val),
            "pr_value": str(pr_val),
            "note": f"DMA: {dma_val}  /  PR: {pr_val}",
        })

    def _safe_float(val):
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    dma_offer = _safe_float(dma.get("offer_price_cash"))
    pr_offer = _safe_float(pr.get("offer_price_cash"))
    if dma_offer is not None and pr_offer is not None:
        if abs(dma_offer - pr_offer) > 0.01:
            _flag("offer_price_cash", "Offer Price",
                  f"${dma_offer}", f"${pr_offer}")

    dma_cvr = _safe_float(dma.get("cvr_value"))
    pr_cvr = _safe_float(pr.get("cvr_value"))
    if dma_cvr is not None and pr_cvr is not None:
        if abs(dma_cvr - pr_cvr) > 0.01:
            _flag("cvr_value", "CVR Value",
                  f"${dma_cvr}", f"${pr_cvr}")

    if dma.get("deal_type") and pr.get("deal_type"):
        if str(dma["deal_type"]).lower() != str(pr["deal_type"]).lower():
            _flag("deal_type", "Deal Type", dma["deal_type"], pr["deal_type"])

    dma_ratio = _safe_float(dma.get("stock_exchange_ratio"))
    pr_ratio = _safe_float(pr.get("stock_exchange_ratio"))
    if dma_ratio is not None and pr_ratio is not None:
        if abs(dma_ratio - pr_ratio) > 0.001:
            _flag("stock_exchange_ratio", "Exchange Ratio",
                  dma_ratio, pr_ratio)

    return flags


def extract_from_dma_summary(
    summary_text: str,
    deal_id: str = None,
    accession_number: str = None,
    company_name: str = None,
    cik_number: str = None,
    dma_summary_id: str = None,
    dma_summary_docx: str = None,
    filing_date: str = None,
    send_email: bool = True,
) -> dict:
    """
    Send the DMA summary to Claude Haiku for structured extraction.
    Cross-references against existing press release data from MongoDB.
    Saves to MongoDB fo_dma_extraction collection and returns it.

    Args:
        summary_text: The DMA summary text from EX-2.1 processing
        deal_id: Deal ID (optional)
        accession_number: SEC accession number (optional)
        company_name: Company name (optional)
        cik_number: CIK number (optional)
        dma_summary_id: Reference to deal_dma_summary._id (optional)
        dma_summary_docx: S3 URL of the DMA summary DOCX (optional)
        filing_date: Filing date string (optional)
        send_email: Whether to send extraction email (default True)

    Returns:
        dict with deal_id, extracted, inconsistencies, source_text, extracted_at, etc.
    """
    import anthropic
    from sec_rss_parser.models import FODmaExtraction
    from sec_rss_parser.email_templates import generate_dma_extraction_email_html
    from sec_rss_parser.utils_8k import send_webhook_notification
    from sec_rss_parser.email_service.email_dispatch_service import send_report_email

    N8N_WEBHOOK_URL = os.environ.get("N8N_WEKHOOK_INTERNAL_WITH_JOSH",
                                     "https://n8n.arbintel.cloud/webhook/b3007d21-6845-47b5-aece-7b26583758bc")

    logger.info(
        "dma_summary_processor: extract_from_dma_summary deal_id=%s accession=%s",
        deal_id, accession_number
    )

    client = anthropic.Anthropic()

    prompt = f"""Extract structured deal data from this Definitive Merger Agreement (DMA) summary.
Return ONLY valid JSON with these fields (use null for anything not stated):

{{
  "target": "Target company name",
  "acquirer": "Acquirer name(s)",
  "offer_price_cash": 0.00,
  "cvr_value": 0.00,
  "stock_exchange_ratio": null,
  "total_consideration": "PER-SHARE total (cash + stock value + CVR). NOT the aggregate deal value.",
  "deal_type": "cash / stock / cash+stock / cash+CVR",
  "outside_date": "YYYY-MM-DD — the initial termination/drop-dead date",
  "outside_date_extension": "YYYY-MM-DD — extended outside date if applicable, or null",
  "outside_date_extension_condition": "condition that triggers the extension, or null",
  "expected_close": "Use standard finance shorthand: 1H26, 2H26, Q1 26, etc.",
  "go_shop_days": null,
  "go_shop_end_date": "YYYY-MM-DD or null",
  "target_break_fee_mm": null,
  "acquirer_reverse_break_fee_mm": null,
  "voting_threshold": "e.g. majority of outstanding shares",
  "regulatory_approvals_required": ["specific agencies: HSR/FTC, EC, CFIUS, etc."],
  "regulatory_filing_deadlines": "any specific deadlines for regulatory filings",
  "dividend_allowed": "what dividends/distributions are permitted during pendency",
  "financing_condition": "is closing conditioned on financing? describe",
  "conditions_to_closing": ["list of key closing conditions"],
  "specific_termination_triggers": ["list of specific termination rights"],
  "interim_operating_covenants": "summary of key restrictions on target operations",
  "announce_date": "YYYY-MM-DD or null"
}}

DMA summary:
{summary_text}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    extracted = json.loads(text)

    pr_data = _load_press_release(
        deal_id=deal_id, accession_number=accession_number)
    inconsistencies = _check_inconsistencies(
        extracted, pr_data) if pr_data else []

    if inconsistencies:
        logger.info(
            "dma_summary_processor: found %d inconsistencies for deal_id=%s",
            len(inconsistencies), deal_id
        )

    extracted_filing_date = filing_date or extracted.get("announce_date")

    now = datetime.utcnow()

    existing = None
    if deal_id:
        existing = FODmaExtraction.objects(deal_id=deal_id).first()
    if not existing and accession_number:
        existing = FODmaExtraction.objects(
            accession_number=accession_number).first()

    if existing:
        existing.extracted = extracted
        existing.inconsistencies = inconsistencies
        existing.source_text = summary_text
        existing.extracted_at = now
        existing.filing_date = extracted_filing_date
        existing.dma_summary_id = dma_summary_id or existing.dma_summary_id
        existing.dma_summary_docx = dma_summary_docx or existing.dma_summary_docx
        existing.company_name = company_name or existing.company_name
        existing.cik_number = cik_number or existing.cik_number
        if deal_id:
            existing.deal_id = deal_id
        if accession_number:
            existing.accession_number = accession_number
        existing.save()
        logger.info(
            "dma_summary_processor: updated existing record _id=%s", existing._id)
    else:
        record = FODmaExtraction(
            deal_id=deal_id,
            accession_number=accession_number,
            extracted=extracted,
            inconsistencies=inconsistencies,
            source_text=summary_text,
            extracted_at=now,
            filing_date=extracted_filing_date,
            dma_summary_id=dma_summary_id,
            dma_summary_docx=dma_summary_docx,
            company_name=company_name,
            cik_number=cik_number,
        )
        record.save()
        logger.info(
            "dma_summary_processor: created new record _id=%s", record._id)

    result = {
        "deal_id": deal_id,
        "accession_number": accession_number,
        "extracted": extracted,
        "inconsistencies": inconsistencies,
        "source_text": summary_text,
        "extracted_at": now.isoformat() + "Z",
        "filing_date": extracted_filing_date,
        "dma_summary_id": dma_summary_id,
        "dma_summary_docx": dma_summary_docx,
        "company_name": company_name,
        "cik_number": cik_number,
    }

    if send_email:
        try:
            subject, html_email = generate_dma_extraction_email_html(
                company_name=company_name or extracted.get(
                    "target") or "Unknown Company",
                deal_id=deal_id or "N/A",
                accession_number=accession_number or "N/A",
                extracted=extracted,
                inconsistencies=inconsistencies,
                filing_date=extracted_filing_date,
                cik_number=cik_number,
                dma_summary_docx=dma_summary_docx,
            )
            payload = {
                "subject": subject,
                "html": html_email,
                "company_name": company_name or extracted.get("target") or "Unknown Company",
                "deal_id": deal_id,
                "accession_number": accession_number,
                "email_type": "dma_extraction",
                "inconsistencies_count": len(inconsistencies),
            }
            send_webhook_notification(
                N8N_WEBHOOK_URL, payload, "DMA Extraction email"
            )  # TODO: comment out after org-aware send is stable
            logger.info(
                "dma_summary_processor: sent extraction email for deal_id=%s", deal_id)

            result_dispatch = send_report_email(
                report_type="sec_dma_press_release_extraction",
                payload=payload,
                org_id="6a031d87e4f1d72367bd2f92",
            )
            logger.info(
                "dma_summary_processor: org-aware email done deal_id=%s orgs_sent=%s/%s",
                deal_id, result_dispatch["orgs_sent"], result_dispatch["orgs_processed"]
            )
        except Exception as e:
            logger.exception(
                "dma_summary_processor: failed to send email error=%s", str(e))

    return result
