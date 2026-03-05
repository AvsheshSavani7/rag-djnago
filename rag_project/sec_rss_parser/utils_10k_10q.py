"""
Utility functions for processing 10-K and 10-Q filings.

This module provides helper functions to:
1. Fetch additional 10-K/10-Q filings from SEC API using company CIK
2. Extract announce date from deal DB or via LLM
3. Save filings to SECFilingSummary.ten_k_ten_q
4. Send email notifications with filing details

Similar to utils_8k.py but specifically for 10-K/10-Q processing.
"""

import logging
from datetime import datetime, timedelta
from mongoengine.errors import NotUniqueError

from .sec_Last_Year import print_filings as fetch_sec_filings
from .models import SECFilingSummary
from .utils_8k import (
    normalize_cik,
    parse_filing_date,
    send_webhook_notification,
    log_and_print,
)
from .email_templates import generate_sec_filings_email_html
from document_processor.models import ProcessingJob

logger = logging.getLogger(__name__)

# Deal status constants
DEAL_STATUS_OPEN_OR_UNKNOWN = ["Open", "Unknown"]

# N8N webhook URL for 10-K/10-Q emails
N8N_WEBHOOK_URL_10K_10Q = "https://n8n-xwx1.onrender.com/webhook/80830c6d-ff5b-45e3-9ef3-a061db1fbf0c"


def _extract_announce_date_with_llm(company_details_str):
    """
    Extract announce date using LLM + web search.
    
    Args:
        company_details_str: String with company details (name, CIK, target, acquirer, etc.)
    
    Returns:
        datetime object or None
    """
    try:
        # Import here to avoid circular dependency
        from .services import _extract_announce_date_with_llm as llm_extract
        return llm_extract(company_details_str)
    except Exception as e:
        logger.error(f"Error extracting announce date with LLM: {e}")
        return None


def get_announce_date_for_10k_10q(cik_number, matched_deal=None, item_data=None):
    """
    Get announce date for 10-K/10-Q processing.
    
    Priority:
    1. From matched_deal.announce_date
    2. From DB lookup by CIK (target or acquirer)
    3. From LLM + web search using company details
    
    Args:
        cik_number: Company CIK number
        matched_deal: Optional matched deal object
        item_data: Optional item data dict with company_name, etc.
    
    Returns:
        datetime object or None
    """
    announce_date = None
    deal = None
    
    # Try matched_deal first
    if matched_deal and getattr(matched_deal, 'announce_date', None):
        announce_date = matched_deal.announce_date
        log_and_print(f"📅 Using announce date from matched_deal: {announce_date}")
    
    # Try DB lookup by CIK
    if not announce_date:
        cik_normalized = normalize_cik(cik_number)
        
        # Try as target CIK
        deal = ProcessingJob.objects(
            cik=cik_normalized,
            deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN,
        ).first()
        
        # Try as acquirer CIK
        if not deal:
            deal = ProcessingJob.objects(
                acquirer_cik=cik_normalized,
                deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN,
            ).first()
        
        if deal and getattr(deal, 'announce_date', None):
            announce_date = deal.announce_date
            log_and_print(f"📅 Using announce date from DB deal: {announce_date}")
    
    # Ensure announce_date is datetime object
    if announce_date and not isinstance(announce_date, datetime):
        announce_date = parse_filing_date(
            announce_date.strftime('%Y-%m-%d') if hasattr(announce_date, 'strftime') else str(announce_date)
        )
    
    # Try LLM + web search if still no announce date
    if not announce_date and item_data:
        deal_for_llm = matched_deal or deal
        parts = [
            f"Company (filing registrant): {item_data.get('company_name', '') or 'N/A'}",
            f"CIK: {cik_number}",
        ]
        if deal_for_llm:
            parts.append(f"Target: {getattr(deal_for_llm, 'target_name', '') or 'N/A'}")
            parts.append(f"Acquirer: {getattr(deal_for_llm, 'acquire_name', '') or 'N/A'}")
            sec_url = getattr(deal_for_llm, 'sec_url', None)
            if sec_url:
                parts.append(f"SEC URL: {sec_url}")
        
        company_details_str = "\n".join(parts)
        log_and_print("🔍 No deal announce date in DB; trying LLM + web search...")
        
        announce_date = _extract_announce_date_with_llm(company_details_str)
        if announce_date:
            log_and_print(f"✅ LLM extracted announce date: {announce_date.strftime('%Y-%m-%d')}")
    
    return announce_date, deal


def fetch_and_save_additional_10k_10q_filings(
    cik_number,
    company_name,
    form_type,
    announce_date=None,
    deal_id=None,
    matched_deal=None,
    item_data=None,
):
    """
    Fetch additional 10-K/10-Q filings from SEC API and save to SECFilingSummary.
    
    This function:
    1. Fetches all 10-K/10-Q filings from announce_date (or 1 year before today)
    2. Saves each filing to SECFilingSummary.ten_k_ten_q (if not already present)
    3. Sends email with all filings
    
    Args:
        cik_number: Company CIK number
        company_name: Company name
        form_type: Form type (10-K or 10-Q)
        announce_date: Optional announce date (datetime object)
        deal_id: Optional deal ID
        matched_deal: Optional matched deal object
        item_data: Optional item data dict
    
    Returns:
        dict with status and count of filings processed
    """
    try:
        # Get announce date if not provided
        if not announce_date:
            announce_date, deal = get_announce_date_for_10k_10q(
                cik_number, matched_deal, item_data
            )
            if deal and not deal_id:
                deal_id = str(deal.id)
        
        # Determine start_date for fetching filings
        start_date = None
        if announce_date:
            start_date = announce_date.strftime('%Y-%m-%d')
            log_and_print(f"📅 Using announce date as start_date: {start_date}")
        else:
            log_and_print("⏭️ No announce date (DB or LLM); using start_date=None (1 year before today)")
        
        # Fetch filings from SEC API
        log_and_print(f"🔍 Fetching 10-K/10-Q filings for CIK {cik_number} from {start_date or '1 year ago'}...")
        filings = fetch_sec_filings(
            str(cik_number),
            start_date=start_date,
            form_types=["10-K", "10-Q"],
        )
        
        if not filings:
            log_and_print(f"⚠️ No 10-K/10-Q filings found for CIK {cik_number}")
            return {
                'success': True,
                'filings_count': 0,
                'saved_count': 0,
                'message': 'No filings found'
            }
        
        log_and_print(f"📥 Found {len(filings)} 10-K/10-Q filings for CIK {cik_number}")
        
        # Save each filing to SECFilingSummary.ten_k_ten_q
        saved_count = 0
        for filing in filings:
            acc = filing.get("accession_number")
            if not acc:
                continue
            
            # Check if already exists
            existing = SECFilingSummary.objects(
                accession_number=acc,
                form_type=filing.get("form")
            ).first()
            
            if existing:
                log_and_print(f"⏭️ Skipping existing filing: {acc}")
                continue
            
            # Create ten_k_ten_q payload
            ten_payload = {
                "processed": False,
                "processed_at": None,
                "s3_json_url": None,
                "s3_docx_url": None,
                "s3_comparison_json_url": None,
                "s3_redline_docx_url": None,
                "s3_client_report_docx_url": None,
                "s3_exec_summary_docx_url": None,
                "label": None,
            }
            
            # Parse filing date
            filing_date = None
            if filing.get("filing_date"):
                filing_date = parse_filing_date(filing.get("filing_date"))
            
            try:
                # Create new SECFilingSummary record
                SECFilingSummary(
                    form_type=filing.get("form"),
                    accession_number=acc,
                    cik_number=str(cik_number),
                    sec_document_url=filing.get("url", ""),
                    filing_date=filing_date,
                    deal_id=deal_id,
                    ten_k_ten_q=ten_payload,
                    proxy=None,
                    eight_k=None,
                    other_filings=None,
                ).save()
                
                saved_count += 1
                log_and_print(f"💾 Saved 10-K/10-Q record to sec_filing_summary: {acc}")
                
            except NotUniqueError:
                log_and_print(f"⚠️ Duplicate filing (race condition): {acc}", "warning")
            except Exception as save_e:
                log_and_print(f"❌ Error saving filing {acc}: {save_e}", "error")
        
        # Run 10-K/10-Q summary pipeline (orchestrator) with fetched URLs and deal_id
        urls_from_filings = [f.get("url") for f in filings if f.get("url")]
        if urls_from_filings and deal_id:
            try:
                from .tenK_tenQ_pipeline.orchestrator import run_pipeline
                log_and_print(f"🔄 Running 10-K/10-Q summary pipeline for {len(urls_from_filings)} filing(s), deal_id={deal_id}...")
                run_pipeline(urls=urls_from_filings, deal_id=deal_id)
                log_and_print("✅ 10-K/10-Q summary pipeline completed.")
            except Exception as pipeline_e:
                log_and_print(f"❌ 10-K/10-Q summary pipeline failed: {pipeline_e}", "error")
        elif not deal_id:
            log_and_print("⏭️ Skipping summary pipeline: no deal_id.", "warning")
        
        # Send email with all filings
        try:
            sec_subject, sec_html = generate_sec_filings_email_html(
                company_name, filings, form_type=form_type
            )
            
            sec_payload = {
                'subject': sec_subject,
                'html': sec_html,
                'company_name': company_name,
                'email_type': 'sec_filings_last_year',
            }
            
            send_webhook_notification(
                N8N_WEBHOOK_URL_10K_10Q, sec_payload, "email"
            )
            
            log_and_print(
                f"📤 Sent SEC form filings email: {len(filings)} {form_type} filings for {company_name}"
                + (f" (from announce date: {start_date})" if announce_date else " (from 1 year before today)")
            )
        except Exception as email_e:
            log_and_print(f"❌ Error sending 10-K/10-Q email: {email_e}", "error")
        
        return {
            'success': True,
            'filings_count': len(filings),
            'saved_count': saved_count,
            'message': f'Processed {len(filings)} filings, saved {saved_count} new records'
        }
        
    except Exception as e:
        log_and_print(f"❌ Error fetching/processing 10-K/10-Q filings: {e}", "error")
        return {
            'success': False,
            'error': str(e),
            'filings_count': 0,
            'saved_count': 0
        }
