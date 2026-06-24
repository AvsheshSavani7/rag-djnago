"""
Proxy processing helper functions for sec_rss_parser.
V2: Works directly with SECFilingSummary (no ProxyDocument, no sync layer).

Flow:
1. process_sec_document_for_filing_summary() - creates/updates SECFilingSummary.proxy
2. process_proxy_async() - runs agentic processor, updates SECFilingSummary.proxy
3. process_sections_with_pinecone_v2() - uploads to Pinecone, updates SECFilingSummary.proxy
4. generate_proxy_summary_v2() - generates summary doc, updates SECFilingSummary.proxy
"""

from sec_rss_parser.proxy_summary_service_v2 import (
    ProxySummaryServiceV2,
    is_sc14d_chronological_summary_only_form,
)
from sec_rss_parser.sec_processor_and_pinecone_v2 import SectionProcessorV2
from sec_rss_parser.agentic_sec_processor_v2 import AgenticSECProcessor
from sec_rss_parser.models import SECFilingSummary, SECFiling
from sec_rss_parser.utils_8k import (
    get_ticker_for_deal_and_cik,
    get_deal_tickers,
    normalize_cik,
    normalize_sec_url,
)
from sec_rss_parser.email_templates import _build_proxy_background_summary_email_subject
from sec_rss_parser.email_service.email_dispatch_service import send_report_email
import os
import sys
import logging
import threading
import time
import re
import requests
from datetime import datetime

# Django setup
import django
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()


logger = logging.getLogger(__name__)

N8N_WEBHOOK_SEND_TO_ALL = os.environ.get(
    "N8N_WEBHOOK_SEND_TO_ALL", "https://n8n.arbintel.cloud/webhook/3ff1b0ea-7114-4dda-940e-95ce81e08017")


def initial_proxy_payload(company_name=None):
    """Default proxy subdocument for a new or full pipeline rerun."""
    return {
        "proxy_parsing_status": "pending",
        "empty_percentage": 100.0,
        "processing_state": {
            "pdf_created": False,
            "toc_found": False,
            "toc_extracted": False,
            "sections_extracted": False,
            "empty_percentage": 100.0,
            "iteration_count": 0,
        },
        "s3_urls": {
            "pdf_url": None,
            "toc_pdf_url": None,
            "toc_json_url": None,
            "sections_json_url": None,
        },
        "pinecone_processing_status": "pending",
        "pinecone_processed_at": None,
        "pinecone_error_message": None,
        "summary_generation_status": "pending",
        "summary_docx_url": None,
        "summary_generated_at": None,
        "agent_response": None,
        "error_message": None,
        "completed_at": None,
        "total_sections": 0,
        "empty_sections": 0,
        "iteration_count": 0,
        "company_name": company_name,
    }


def reset_proxy_node_only(filing_summary, step="all"):
    """
    Reset only SECFilingSummary.proxy for pipeline rerun.
    Does not modify L1/L2/L3, s3_docx_url, s3_json_url, or other top-level fields.
    """
    existing = filing_summary.proxy or {}
    company_name = existing.get("company_name")

    if step == "all":
        filing_summary.proxy = initial_proxy_payload(company_name=company_name)
    elif step == "pinecone":
        proxy_data = dict(existing)
        proxy_data["pinecone_processing_status"] = "pending"
        proxy_data["pinecone_processed_at"] = None
        proxy_data["pinecone_error_message"] = None
        proxy_data["summary_generation_status"] = "pending"
        proxy_data["summary_docx_url"] = None
        proxy_data["summary_generated_at"] = None
        filing_summary.proxy = proxy_data
    elif step == "summary":
        proxy_data = dict(existing)
        proxy_data["summary_generation_status"] = "pending"
        proxy_data["summary_docx_url"] = None
        proxy_data["summary_generated_at"] = None
        filing_summary.proxy = proxy_data
    else:
        raise ValueError(f"Unknown reset step: {step}")

    filing_summary.save()
    return filing_summary


def rerun_proxy_pipeline(filing_summary_id, sync=False, step="all", skip_email=True):
    """
    Re-run proxy pipeline for an existing SECFilingSummary without touching L1/L2/L3.

    step:
      - all: reset proxy node, agentic scrape -> Pinecone -> background summary
      - pinecone: re-embed sections (requires proxy.s3_urls.sections_json_url)
      - summary: re-generate background summary DOCX only (requires Pinecone chunks)

    skip_email: when True (default for reruns), do not send summary notification email.
    """
    filing_summary = SECFilingSummary.objects(_id=filing_summary_id).first()
    if not filing_summary:
        raise ValueError(f"SECFilingSummary not found: {filing_summary_id}")
    if not filing_summary.sec_document_url:
        raise ValueError(
            f"sec_document_url missing on SECFilingSummary {filing_summary_id}")

    fid = str(filing_summary.id)
    proxy_sec_url = filing_summary.sec_document_url

    if step == "all":
        reset_proxy_node_only(filing_summary, step="all")
        if sync:
            process_proxy_async(
                fid, proxy_sec_url, chain_sync=True, skip_email=skip_email)
        else:
            from core.logging_context import get_pipeline, get_run_id, get_accession, get_doc_type
            _ctx = (get_pipeline(), get_run_id(),
                    get_accession(), get_doc_type())
            thread = threading.Thread(
                target=process_proxy_async,
                args=(fid, proxy_sec_url, *_ctx),
                kwargs={"chain_sync": False, "skip_email": skip_email},
            )
            thread.daemon = True
            thread.start()
        return {
            "filing_summary_id": fid,
            "step": step,
            "mode": "sync" if sync else "async",
            "sec_document_url": proxy_sec_url,
            "skip_email": skip_email,
        }

    if step == "pinecone":
        sections_json_url = (filing_summary.proxy or {}).get(
            "s3_urls", {}
        ).get("sections_json_url")
        if not sections_json_url:
            raise ValueError(
                "sections_json_url missing on proxy.s3_urls; run with step=all first"
            )
        reset_proxy_node_only(filing_summary, step="pinecone")
        if sync:
            process_sections_with_pinecone_v2(
                fid, sections_json_url, start_summary_thread=False, skip_email=skip_email)
            generate_proxy_summary_v2(fid, skip_email=skip_email)
        else:
            from core.logging_context import get_pipeline, get_run_id, get_accession, get_doc_type
            _ctx = (get_pipeline(), get_run_id(),
                    get_accession(), get_doc_type())
            thread = threading.Thread(
                target=process_sections_with_pinecone_v2,
                args=(fid, sections_json_url, *_ctx),
                kwargs={"skip_email": skip_email},
            )
            thread.daemon = True
            thread.start()
        return {
            "filing_summary_id": fid,
            "step": step,
            "mode": "sync" if sync else "async",
            "sections_json_url": sections_json_url,
            "skip_email": skip_email,
        }

    if step == "summary":
        reset_proxy_node_only(filing_summary, step="summary")
        if sync:
            generate_proxy_summary_v2(fid, skip_email=skip_email)
        else:
            from core.logging_context import get_pipeline, get_run_id, get_accession, get_doc_type
            _ctx = (get_pipeline(), get_run_id(),
                    get_accession(), get_doc_type())
            thread = threading.Thread(
                target=generate_proxy_summary_v2,
                args=(fid, *_ctx),
                kwargs={"skip_email": skip_email},
            )
            thread.daemon = True
            thread.start()
        return {
            "filing_summary_id": fid,
            "step": step,
            "mode": "sync" if sync else "async",
            "skip_email": skip_email,
        }

    raise ValueError(f"Unknown step: {step}")


def _parse_filing_date(value):
    """Parse filing_date string to datetime for SECFilingSummary.filing_date."""
    if not value or not isinstance(value, str):
        return None
    value = str(value).strip()
    if not value:
        return None
    # YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", value)
    if m:
        try:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return datetime(y, mo, d, 0, 0, 0, 0)
        except (ValueError, TypeError):
            pass
    # YYYY/MM/DD
    m = re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})$", value)
    if m:
        try:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return datetime(y, mo, d, 0, 0, 0, 0)
        except (ValueError, TypeError):
            pass
    return None


def process_sec_document_for_filing_summary(
    cik_number,
    company_name,
    sec_filling_id,
    filing_date,
    form_type,
    proxy_sec_url,
    deal_id=None,
    accession_number=None,
):
    """
    Process SEC proxy document by creating/updating SECFilingSummary directly.
    No ProxyDocument is used. SECFilingSummary.proxy is the canonical record.

    Args:
        cik_number: CIK number of the company
        company_name: Name of the company
        sec_filling_id: SEC filing ID (MongoDB ObjectId as string)
        filing_date: Filing date (string)
        form_type: Form type (e.g., DEF 14A, DEFM14A)
        proxy_sec_url: URL of the SEC proxy document
        deal_id: Optional deal ID if CIK matches with deal table
        accession_number: Optional accession number

    Returns:
        dict with status and sec_filing_summary_id, or None if error
    """
    # Inherit run_id from caller; set proxy pipeline context
    from core.logging_context import set_pipeline_context, get_run_id
    set_pipeline_context(
        pipeline="proxy",
        run_id=get_run_id(),
        accession=accession_number or "-",
        doc_type=(form_type or "PROXY").replace(" ", "_").upper(),
    )

    try:
        proxy_sec_url = normalize_sec_url(proxy_sec_url)
        if not proxy_sec_url:
            logger.error("Invalid or empty proxy_sec_url after normalization")
            return None

        filing_dt = _parse_filing_date(filing_date)

        proxy_payload = initial_proxy_payload(
            company_name=company_name or None)

        # Check if filing summary already exists (by sec_document_url + form_type)
        existing = SECFilingSummary.objects(
            sec_document_url=proxy_sec_url,
            form_type=form_type,
        ).first()

        acc = accession_number or sec_filling_id

        if existing:
            # Update existing record (preserve company_name from existing proxy if not provided)
            old_company_name = (existing.proxy or {}).get("company_name")
            existing.accession_number = acc
            existing.cik_number = cik_number
            existing.filing_date = filing_dt
            existing.deal_id = deal_id
            existing.proxy = proxy_payload
            existing.proxy["company_name"] = company_name or old_company_name
            existing.save()
            filing_summary_id = str(existing.id)
            logger.info(
                f"Updated existing SECFilingSummary: {filing_summary_id}")
        else:
            # Create new record
            doc = SECFilingSummary(
                accession_number=acc,
                cik_number=cik_number,
                sec_document_url=proxy_sec_url,
                filing_date=filing_dt,
                deal_id=deal_id,
                form_type=form_type,
                proxy=proxy_payload,
                ten_k_ten_q=None,
                eight_k=None,
                other_filings=None,
            )
            doc.save()
            filing_summary_id = str(doc.id)
            logger.info(f"Created new SECFilingSummary: {filing_summary_id}")

        # Update SEC filing collection with following status
        sec_filing_updated = False
        try:
            sec_filing = SECFiling.objects(_id=sec_filling_id).first()
            if sec_filing:
                sec_filing.following = True
                sec_filing.following_status = "In Progress"
                sec_filing.save()
                sec_filing_updated = True
                logger.info(
                    f"Updated SEC filing {sec_filling_id} with following=True and following_status='In Progress'")
            else:
                logger.warning(
                    f"SEC filing with _id {sec_filling_id} not found in sec_filings collection")
        except Exception as e:
            logger.error(
                f"Error updating SEC filing {sec_filling_id}: {str(e)}")

        # Capture logging context now (ContextVar is not inherited by threads).
        from core.logging_context import get_pipeline, get_run_id, get_accession, get_doc_type
        _ctx = (get_pipeline(), get_run_id(), get_accession(), get_doc_type())

        # Start processing in a separate thread
        processing_thread = threading.Thread(
            target=process_proxy_async,
            args=(filing_summary_id, proxy_sec_url, *_ctx)
        )
        processing_thread.daemon = True
        processing_thread.start()

        logger.info(
            f"Started processing thread for SECFilingSummary {filing_summary_id}")

        return {
            'sec_filing_summary_id': filing_summary_id,
            'status': 'In Progress',
            'message': 'Proxy document processing started',
            'company_name': company_name,
            'cik_number': cik_number,
            'proxy_sec_url': proxy_sec_url,
            'sec_filing_updated': sec_filing_updated,
            'sec_filing_id': sec_filling_id
        }

    except Exception as e:
        logger.error(f"Error starting proxy document processing: {str(e)}")
        return None


def process_proxy_async(
    filing_summary_id,
    proxy_sec_url,
    _log_pipeline="proxy",
    _log_run_id="-",
    _log_accession="-",
    _log_doc_type="PROXY",
    chain_sync=False,
    skip_email=False,
):
    """
    Process proxy document asynchronously.
    Updates SECFilingSummary.proxy node directly.
    """
    from core.logging_context import set_pipeline_context
    set_pipeline_context(
        pipeline=_log_pipeline,
        run_id=_log_run_id,
        accession=_log_accession,
        doc_type=_log_doc_type,
    )
    try:
        # Get the filing summary
        filing_summary = SECFilingSummary.objects(
            _id=filing_summary_id).first()
        if not filing_summary:
            logger.error(f"Filing summary not found: {filing_summary_id}")
            return

        # Update status to processing
        proxy_data = filing_summary.proxy or {}
        proxy_data["proxy_parsing_status"] = "processing"
        filing_summary.proxy = proxy_data
        filing_summary.save()

        logger.info(f"Starting agentic SEC processing for {filing_summary_id}")

        # Initialize the processor
        processor = AgenticSECProcessor(proxy_sec_url)

        # Process the document
        results = processor.process_document()

        # Update filing summary with results
        empty_percentage = results.get('empty_percentage', 100.0)
        proxy_data["empty_percentage"] = empty_percentage
        proxy_data["agent_response"] = results.get('agent_response', '')
        proxy_data["processing_state"] = processor.processing_state
        proxy_data["s3_urls"] = results.get('s3_urls', {})
        proxy_data["total_sections"] = processor.processing_state.get(
            'total_sections', 0)
        proxy_data["empty_sections"] = processor.processing_state.get(
            'empty_sections', 0)
        proxy_data["iteration_count"] = processor.processing_state.get(
            'iteration_count', 0)

        # Check if empty percentage is too high (>40%)
        if empty_percentage > 40.0:
            # Mark as failed due to high empty percentage
            proxy_data["proxy_parsing_status"] = "failed"
            proxy_data[
                "error_message"] = f'Processing failed: Empty percentage too high ({empty_percentage:.1f}% > 40%)'
            proxy_data["completed_at"] = datetime.utcnow()
            filing_summary.proxy = proxy_data
            filing_summary.save()

            # Update SEC filing status to Failed
            try:
                sec_filing = SECFiling.objects(
                    _id=filing_summary.accession_number).first()
                if sec_filing:
                    sec_filing.following_status = "Failed"
                    sec_filing.save()
                    logger.info(
                        f"Updated SEC filing with following_status='Failed' due to high empty percentage")
            except Exception as e:
                logger.error(
                    f"Error updating SEC filing status to Failed: {str(e)}")

            logger.warning(
                f"Processing failed for {filing_summary_id}: Empty percentage {empty_percentage:.1f}% exceeds 40% threshold")

        else:
            # Empty percentage is acceptable, proceed with completion
            proxy_data["proxy_parsing_status"] = "completed"
            proxy_data["completed_at"] = datetime.utcnow()
            filing_summary.proxy = proxy_data
            filing_summary.save()

            # Update SEC filing status to Completed
            try:
                sec_filing = SECFiling.objects(
                    _id=filing_summary.accession_number).first()
                if sec_filing:
                    sec_filing.following_status = "Completed"
                    sec_filing.save()
                    logger.info(
                        f"Updated SEC filing with following_status='Completed'")
            except Exception as e:
                logger.error(
                    f"Error updating SEC filing status to Completed: {str(e)}")

            logger.info(
                f"Successfully completed processing. Empty percentage: {empty_percentage:.1f}%")

            # Start Pinecone processing if sections JSON URL is available
            s3_urls = results.get('s3_urls', {})
            sections_json_url = s3_urls.get('sections_json_url')

            if sections_json_url:
                logger.info(
                    f"Starting Pinecone processing for sections: {sections_json_url}")

                from core.logging_context import get_pipeline, get_run_id, get_accession, get_doc_type
                _pctx = (get_pipeline(), get_run_id(),
                         get_accession(), get_doc_type())

                if chain_sync:
                    process_sections_with_pinecone_v2(
                        filing_summary_id,
                        sections_json_url,
                        *_pctx,
                        start_summary_thread=False,
                        skip_email=skip_email,
                    )
                    generate_proxy_summary_v2(
                        filing_summary_id, *_pctx, skip_email=skip_email)
                    logger.info(
                        f"Completed sync Pinecone + summary for {filing_summary_id}")
                else:
                    pinecone_thread = threading.Thread(
                        target=process_sections_with_pinecone_v2,
                        args=(filing_summary_id, sections_json_url, *_pctx),
                        kwargs={"skip_email": skip_email},
                    )
                    pinecone_thread.daemon = True
                    pinecone_thread.start()
                    logger.info(
                        f"Started Pinecone processing thread for {filing_summary_id}")
            else:
                logger.warning(
                    f"No sections JSON URL found in S3 URLs: {s3_urls}")

    except Exception as e:
        # Update filing summary status to failed
        try:
            filing_summary = SECFilingSummary.objects(
                _id=filing_summary_id).first()
            if filing_summary:
                proxy_data = filing_summary.proxy or {}
                proxy_data["proxy_parsing_status"] = "failed"
                proxy_data["error_message"] = str(e)
                proxy_data["completed_at"] = datetime.utcnow()
                filing_summary.proxy = proxy_data
                filing_summary.save()

            # Update SEC filing status to Failed
            try:
                sec_filing = SECFiling.objects(
                    _id=filing_summary.accession_number).first()
                if sec_filing:
                    sec_filing.following_status = "Failed"
                    sec_filing.save()
                    logger.info(
                        f"Updated SEC filing with following_status='Failed'")
            except Exception as sec_error:
                logger.error(
                    f"Error updating SEC filing status to Failed: {str(sec_error)}")
        except:
            pass

        logger.error(
            f"Error processing proxy document {filing_summary_id}: {str(e)}")


def process_sections_with_pinecone_v2(
    filing_summary_id,
    sections_json_url,
    _log_pipeline="proxy",
    _log_run_id="-",
    _log_accession="-",
    _log_doc_type="PROXY",
    start_summary_thread=True,
    skip_email=False,
):
    """
    Process sections with Pinecone after SEC processing is complete.
    Updates SECFilingSummary.proxy node directly.
    """
    from core.logging_context import set_pipeline_context
    set_pipeline_context(
        pipeline=_log_pipeline,
        run_id=_log_run_id,
        accession=_log_accession,
        doc_type=_log_doc_type,
    )
    try:
        # Get the filing summary
        filing_summary = SECFilingSummary.objects(
            _id=filing_summary_id).first()
        if not filing_summary:
            logger.error(f"Filing summary not found: {filing_summary_id}")
            return
        proxy_data = filing_summary.proxy or {}

        logger.info(f"Starting Pinecone processing for {filing_summary_id}")

        # Initialize SectionProcessorV2 with filing_summary_id and deal_id
        processor = SectionProcessorV2(
            sec_filing_summary_id=str(filing_summary.id),
            deal_id=filing_summary.deal_id
        )

        # Process sections from S3 URL
        processor.process_from_s3_url(sections_json_url)

        # Update filing summary with Pinecone processing completion
        proxy_data["pinecone_processing_status"] = "completed"
        proxy_data["pinecone_processed_at"] = datetime.utcnow()
        filing_summary.proxy = proxy_data
        filing_summary.save()

        logger.info(
            f"Successfully completed Pinecone processing for {filing_summary_id}")

        if start_summary_thread:
            try:
                time.sleep(2)
                logger.info(
                    f"Starting summary generation for {filing_summary_id}")

                from core.logging_context import get_pipeline, get_run_id, get_accession, get_doc_type
                _sctx = (get_pipeline(), get_run_id(),
                         get_accession(), get_doc_type())
                summary_thread = threading.Thread(
                    target=generate_proxy_summary_v2,
                    args=(filing_summary_id, *_sctx),
                    kwargs={"skip_email": skip_email},
                )
                summary_thread.daemon = True
                summary_thread.start()

                logger.info(
                    f"Started summary generation thread for {filing_summary_id}")
            except Exception as summary_error:
                logger.error(
                    f"Error starting summary generation: {str(summary_error)}")

    except Exception as e:
        # Update filing summary status to failed
        try:
            filing_summary = SECFilingSummary.objects(
                _id=filing_summary_id).first()
            if filing_summary:
                proxy_data = filing_summary.proxy or {}
                proxy_data["pinecone_processing_status"] = "failed"
                proxy_data["pinecone_error_message"] = str(e)
                filing_summary.proxy = proxy_data
                filing_summary.save()
        except:
            pass

        logger.error(
            f"Error in Pinecone processing for {filing_summary_id}: {str(e)}")


def generate_proxy_summary_v2(
    filing_summary_id,
    _log_pipeline="proxy",
    _log_run_id="-",
    _log_accession="-",
    _log_doc_type="PROXY",
    skip_email=False,
):
    """
    Generate summary document for proxy after Pinecone processing completes.
    Updates SECFilingSummary.proxy node directly.
    """
    from core.logging_context import set_pipeline_context
    set_pipeline_context(
        pipeline=_log_pipeline,
        run_id=_log_run_id,
        accession=_log_accession,
        doc_type=_log_doc_type,
    )
    try:
        # Get the filing summary
        filing_summary = SECFilingSummary.objects(
            _id=filing_summary_id).first()
        if not filing_summary:
            logger.error(f"Filing summary not found: {filing_summary_id}")
            return
        proxy_data = filing_summary.proxy or {}

        # Update status to processing
        proxy_data["summary_generation_status"] = "processing"
        filing_summary.proxy = proxy_data
        filing_summary.save()

        logger.info(
            f"Starting summary document generation for {filing_summary_id}")

        # Initialize summary service
        summary_service = ProxySummaryServiceV2()

        # Check if questions file exists
        questions_file = None
        possible_paths = [
            os.path.join(os.path.dirname(
                os.path.abspath(__file__)), '..', 'proxy_processor', 'quetions.json'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
                __file__))), 'rag_project', 'proxy_processor', 'quetions.json'),
        ]
        for questions_path in possible_paths:
            if os.path.exists(questions_path):
                questions_file = questions_path
                break

        # Generate summary document
        result = summary_service.generate_summary_document(
            sec_filing_summary_id=str(filing_summary.id),
            questions_file=questions_file
        )

        if result.get('success'):
            # Update filing summary with summary URL
            proxy_data["summary_docx_url"] = result.get('docx_url')
            proxy_data["summary_generation_status"] = "completed"
            proxy_data["summary_generated_at"] = datetime.utcnow()
            filing_summary.proxy = proxy_data
            filing_summary.save()

            logger.info(
                f"Successfully generated summary document for {filing_summary_id}: {result.get('docx_url')}")

            if skip_email:
                logger.info(
                    "Skipping summary email notification (skip_email=True)")
            else:
                try:
                    logger.info(
                        f"📧 Sending summary document email notification")
                    send_summary_email_notification_v2(filing_summary)
                    logger.info(
                        f"✅ Summary email notification sent successfully")
                except Exception as email_error:
                    logger.error(
                        f"❌ Error sending summary email notification: {str(email_error)}")

        else:
            # Mark summary generation as failed
            proxy_data["summary_generation_status"] = "failed"
            proxy_data["error_message"] = result.get('error', 'Unknown error')
            filing_summary.proxy = proxy_data
            filing_summary.save()

            logger.error(
                f"Summary generation failed for {filing_summary_id}: {result.get('error')}")

    except Exception as e:
        # Update filing summary status to failed
        try:
            filing_summary = SECFilingSummary.objects(
                _id=filing_summary_id).first()
            if filing_summary:
                proxy_data = filing_summary.proxy or {}
                proxy_data["summary_generation_status"] = "failed"
                proxy_data["error_message"] = str(e)
                filing_summary.proxy = proxy_data
                filing_summary.save()
        except:
            pass

        logger.error(
            f"Error generating summary for {filing_summary_id}: {str(e)}")


def escape_html(text):
    """Escape HTML special characters"""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&#x27;")
    return text


def _render_proxy_qa_html(qa_items: list, chronological_summary: list = None) -> str:
    """Build inline HTML for the 5 proxy Q&A items and optional Chronological Summary."""
    if not qa_items and not chronological_summary:
        return ""

    rows = ""
    for item in (qa_items or []):
        question = escape_html(item.get("question", ""))
        answer = escape_html(item.get("answer", ""))
        rows += f"""
      <div style="margin-bottom:20px;">
        <p style="margin:0 0 6px 0; font-size:14px; font-weight:bold; color:#4a90e2;">{question}</p>
        <p style="margin:0 0 0 14px; font-size:13px; line-height:1.6; color:#333;">
          <span style="font-weight:bold; margin-right:6px; color:#333;">+</span>{answer}
        </p>
      </div>"""

    chrono_html = ""
    if chronological_summary:
        chrono_rows = "".join(
            f'<p style="margin:0 0 0 14px; font-size:13px; line-height:1.6; color:#333;">'
            f'{escape_html(line)}</p>'
            for line in chronological_summary
        )
        chrono_html = f"""
      <div style="margin-top:24px; padding-top:18px; border-top:1px solid #e8e8e8;">
        <p style="margin:0 0 12px 0; font-size:14px; font-weight:bold; color:#4a90e2;">
          Chronological Summary
        </p>
        {chrono_rows}
      </div>"""

    return f"""
    <div style="margin:30px 0; border-top:2px solid #e0e0e0; padding-top:20px;">
      <p style="margin:0 0 16px 0; font-size:15px; font-weight:bold; color:#333; text-decoration:underline;">
        Proxy Summary
      </p>
      {rows}
      {chrono_html}
    </div>"""


def generate_summary_email_html(
    company_name: str,
    form_type: str,
    summary_doc_url: str,
    cik_number: str,
    proxy_sec_url: str,
    ticker: str = None,
    filing_date=None,
    qa_items: list = None,
    chronological_summary: list = None,
    chronological_summary_only: bool = False,
    target_ticker: str = None,
    target_name: str = None,
    acquirer_ticker: str = None,
    acquirer_name: str = None,
    matched_cik_label: str = None,
) -> tuple:
    """
    Generate HTML email for proxy summary document notification.

    Args:
        company_name: Name of the company
        form_type: Form type (e.g., DEFM14A)
        summary_doc_url: URL of the generated summary document
        cik_number: CIK number
        proxy_sec_url: URL of the SEC proxy document
        ticker: Deprecated; kept for call-site compatibility (not used in subject)
        filing_date: Deprecated; kept for call-site compatibility (not used in subject)
        qa_items: Optional list of Q&A dicts with "question" and "answer" keys
        chronological_summary: Optional list of chronological summary lines
        chronological_summary_only: True for SC 14D family — Q&A omitted; label email accordingly
        target_ticker: Deal target ticker for subject prefix (preferred over target_name)
        target_name: Deal target name for subject prefix when target_ticker is missing
        matched_cik_label: "(target)" or "(acquirer)" for Parent/Target Form in subject
    Returns:
        tuple: (subject, html_email)
    """
    subject = _build_proxy_background_summary_email_subject(
        target_ticker=target_ticker,
        target_name=target_name,
        acquirer_ticker=acquirer_ticker,
        acquirer_name=acquirer_name,
        matched_cik_label=matched_cik_label,
        cik_number=cik_number,
        form_type=form_type,
    )

    notice_html = ""

    inline_qa_html = _render_proxy_qa_html(qa_items, chronological_summary)

    banner_title = (
        "New Proxy Summary Document (Background summary)"
        if chronological_summary_only
        else "New Proxy Summary Document"
    )

    html_email = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{escape_html(subject)}</title>
</head>
<body style="margin:0; padding:0; font-family:Arial,sans-serif; background-color:#f4f4f4;">
  <div style="max-width:700px; margin:20px auto; background-color:#ffffff; padding:30px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <h2 style="color:#333; text-align:center; margin-top:0; padding-bottom:20px; border-bottom:3px solid #4a90e2;">
      {escape_html(banner_title)}
    </h2>

    <div style="margin-bottom:30px;">
      <p style="color:#333; font-size:16px; line-height:1.6;">
        The proxy summary document has been successfully generated for:
      </p>

      <div style="background-color:#f9f9f9; padding:15px; border-radius:5px; margin:20px 0;">
        <p style="margin:8px 0; color:#555;">
          <strong style="color:#333;">Company:</strong> {escape_html(company_name)}
        </p>
        <p style="margin:8px 0; color:#555;">
          <strong style="color:#333;">Form Type:</strong> {escape_html(form_type)}
        </p>
        <p style="margin:8px 0; color:#555;">
          <strong style="color:#333;">CIK:</strong> {escape_html(cik_number)}
        </p>
      </div>
    </div>

    {notice_html}

    {inline_qa_html}

    <div style="text-align:center; margin:30px 0;">
      <a href="{escape_html(summary_doc_url)}" 
         style="display:inline-block; background-color:#4a90e2; color:#ffffff; padding:15px 30px; text-decoration:none; border-radius:5px; font-size:16px; font-weight:bold;">
        📄 View Summary Document
      </a>
    </div>

    <div style="margin-top:20px; padding-top:20px; border-top:1px solid #e0e0e0;">
      <p style="color:#666; font-size:14px; margin:5px 0;">
        <strong>Original SEC Document:</strong>
      </p>
      <p style="margin:5px 0;">
        <a href="{escape_html(proxy_sec_url)}" 
           style="color:#4a90e2; text-decoration:none; word-break:break-all;">
          {escape_html(proxy_sec_url)}
        </a>
      </p>
    </div>

    <div style="margin-top:30px; padding-top:20px; border-top:2px solid #e0e0e0; text-align:center;">
      <p style="color:#999; font-size:12px; margin:5px 0;">
        This is an automated notification from the MNA Finder system.
      </p>
    </div>
  </div>
</body>
</html>
"""

    return subject, html_email


def send_summary_email_notification_v2(filing_summary):
    """
    Send email notification with summary document URL.
    Uses SECFilingSummary instead of ProxyDocument.
    Matches the old flow from proxy_processor/views.py.
    """
    try:
        import requests

        proxy_data = filing_summary.proxy or {}
        summary_url = proxy_data.get("summary_docx_url")

        if not summary_url:
            logger.warning(
                f"No summary document URL available for filing summary {filing_summary.id}")
            return

        company_name = proxy_data.get(
            "company_name") or filing_summary.cik_number or "Unknown"
        logger.info(
            f"Preparing to send summary email for: {company_name} (CIK {filing_summary.cik_number})")

        deal_id = getattr(filing_summary, "deal_id", None)
        cik_number = getattr(filing_summary, "cik_number", None)
        deal_tickers = get_deal_tickers(deal_id, cik_number)
        matched_cik_label = None
        if deal_id and cik_number:
            try:
                from bson import ObjectId
                from document_processor.models import ProcessingJob

                deal = ProcessingJob.objects(id=ObjectId(deal_id)).only(
                    "cik", "acquirer_cik"
                ).first()
                cik_n = normalize_cik(cik_number)
                if deal and cik_n:
                    if normalize_cik(deal.acquirer_cik) == cik_n:
                        matched_cik_label = "(acquirer)"
                    elif normalize_cik(deal.cik) == cik_n:
                        matched_cik_label = "(target)"
            except Exception as deal_e:
                logger.warning(
                    f"Deal lookup for proxy summary email subject: {deal_e}")

        ticker = get_ticker_for_deal_and_cik(deal_id, cik_number)
        filing_date = getattr(filing_summary, "filing_date", None)

        chronological_summary_only = is_sc14d_chronological_summary_only_form(
            getattr(filing_summary, "form_type", None)
        )

        # Parse the DOCX to extract Q&A and Chronological Summary for inline display
        qa_items = None
        chronological_summary = None
        try:
            from proxy_processor.proxy_docx_parser import parse_proxy_summary_docx
            parsed = parse_proxy_summary_docx(summary_url)
            qa_items = parsed.get("qa_items")
            chronological_summary = parsed.get("chronological_summary")
        except Exception as parse_err:
            logger.warning(
                f"Could not parse proxy DOCX for inline content: {parse_err}")

        if chronological_summary_only:
            qa_items = None

        # Generate email HTML (same as old flow)
        subject, html_email = generate_summary_email_html(
            company_name=company_name,
            form_type=filing_summary.form_type,
            summary_doc_url=summary_url,
            cik_number=filing_summary.cik_number or "",
            proxy_sec_url=filing_summary.sec_document_url,
            ticker=ticker,
            filing_date=filing_date,
            qa_items=qa_items,
            chronological_summary=chronological_summary,
            chronological_summary_only=chronological_summary_only,
            target_ticker=deal_tickers.get("target_ticker"),
            target_name=deal_tickers.get("target_name"),
            acquirer_ticker=deal_tickers.get("acquirer_ticker"),
            acquirer_name=deal_tickers.get("acquirer_name"),
            matched_cik_label=matched_cik_label,
        )
        logger.info(f"Generated email subject: {subject}")

        # Send email via n8n webhook (same URL as old flow)
        webhook_url = N8N_WEBHOOK_SEND_TO_ALL
        logger.info(f"📤 Sending summary email via n8n webhook: {webhook_url}")

        # Prepare payload for n8n webhook
        payload = {
            'subject': subject,
            'html': html_email,
            'company_name': company_name,
            'form_type': filing_summary.form_type,
            'summary_doc_url': summary_url,
            'sec_filing_summary_id': str(filing_summary.id)
        }

        # Send POST request to n8n webhook
        try:
            # response = requests.post(
            #     webhook_url,
            #     json=payload,
            #     headers={'Content-Type': 'application/json'},
            #     timeout=30
            # )
            # response.raise_for_status()

            # logger.info(
            #     f"✅ Summary email sent successfully via n8n webhook! Status: {response.status_code}")
            # logger.info(f"📧 Response: {response.text[:200]}")

            # Org-aware send via email dispatch service
            logger.info(
                "📤 Sending org-aware email (sec_background_summary_proxy)")
            result_dispatch = send_report_email(
                report_type="sec_background_summary_proxy",
                payload=payload
            )
            logger.info(
                "✅ Org-aware email done — orgs_sent=%s/%s",
                result_dispatch["orgs_sent"], result_dispatch["orgs_processed"]
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error sending summary email via n8n webhook: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(
                    f"❌ Response status: {e.response.status_code}, Response body: {e.response.text[:200]}")
            raise

    except Exception as e:
        logger.error(f"Error sending summary email: {str(e)}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    result = process_sec_document_for_filing_summary(
        cik_number="0001408100",
        company_name="Test Company",
        sec_filling_id="test_filling_id",
        filing_date="2025-01-15",
        form_type="DEFM14A",
        proxy_sec_url="https://www.sec.gov/Archives/edgar/data/1408100/000110465925000001/tm2500001-1_defm14a.htm",
        deal_id="test_deal_id",
        accession_number="0001104659-25-000001",
    )

    if result:
        print(f"✅ Processing started: {result}")
    else:
        print("❌ Failed to start processing")
