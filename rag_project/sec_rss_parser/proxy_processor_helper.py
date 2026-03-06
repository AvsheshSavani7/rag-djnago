"""
Proxy processing helper functions for sec_rss_parser.
V2: Works directly with SECFilingSummary (no ProxyDocument, no sync layer).

Flow:
1. process_sec_document_for_filing_summary() - creates/updates SECFilingSummary.proxy
2. process_proxy_async() - runs agentic processor, updates SECFilingSummary.proxy
3. process_sections_with_pinecone_v2() - uploads to Pinecone, updates SECFilingSummary.proxy
4. generate_proxy_summary_v2() - generates summary doc, updates SECFilingSummary.proxy
"""
from sec_rss_parser.proxy_summary_service_v2 import ProxySummaryServiceV2
from sec_rss_parser.sec_processor_and_pinecone_v2 import SectionProcessorV2
from sec_rss_parser.agentic_sec_processor_v2 import AgenticSECProcessor
from sec_rss_parser.models import SECFilingSummary, SECFiling
import os
import sys
import logging
import threading
import re
import requests
from datetime import datetime

# Django setup
import django
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()


logger = logging.getLogger(__name__)


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
    try:
        filing_dt = _parse_filing_date(filing_date)

        # Initial proxy payload (all statuses pending)
        proxy_payload = {
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
            "company_name": company_name or None,
        }

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

        # Start processing in a separate thread
        processing_thread = threading.Thread(
            target=process_proxy_async,
            args=(filing_summary_id, proxy_sec_url)
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


def process_proxy_async(filing_summary_id, proxy_sec_url):
    """
    Process proxy document asynchronously.
    Updates SECFilingSummary.proxy node directly.
    """
    try:
        # Get the filing summary
        filing_summary = SECFilingSummary.objects(_id=filing_summary_id).first()
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

                # Start Pinecone processing in a separate thread
                pinecone_thread = threading.Thread(
                    target=process_sections_with_pinecone_v2,
                    args=(filing_summary_id, sections_json_url)
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
            filing_summary = SECFilingSummary.objects(_id=filing_summary_id).first()
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


def process_sections_with_pinecone_v2(filing_summary_id, sections_json_url):
    """
    Process sections with Pinecone after SEC processing is complete.
    Updates SECFilingSummary.proxy node directly.
    """
    try:
        # Get the filing summary
        filing_summary = SECFilingSummary.objects(_id=filing_summary_id).first()
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

        # Start summary generation after Pinecone processing completes
        try:
            logger.info(
                f"Starting summary generation for {filing_summary_id}")

            # Start summary generation in a separate thread
            summary_thread = threading.Thread(
                target=generate_proxy_summary_v2,
                args=(filing_summary_id,)
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
            filing_summary = SECFilingSummary.objects(_id=filing_summary_id).first()
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


def generate_proxy_summary_v2(filing_summary_id):
    """
    Generate summary document for proxy after Pinecone processing completes.
    Updates SECFilingSummary.proxy node directly.
    """
    try:
        # Get the filing summary
        filing_summary = SECFilingSummary.objects(_id=filing_summary_id).first()
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

            # Send email notification with summary document URL
            try:
                logger.info(
                    f"📧 Sending summary document email notification")
                send_summary_email_notification_v2(filing_summary)
                logger.info(f"✅ Summary email notification sent successfully")
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
            filing_summary = SECFilingSummary.objects(_id=filing_summary_id).first()
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


def generate_summary_email_html(company_name: str, form_type: str, summary_doc_url: str, cik_number: str, proxy_sec_url: str) -> tuple:
    """
    Generate HTML email for proxy summary document notification.

    Args:
        company_name: Name of the company
        form_type: Form type (e.g., DEFM14A)
        summary_doc_url: URL of the generated summary document
        cik_number: CIK number
        proxy_sec_url: URL of the SEC proxy document
    Returns:
        tuple: (subject, html_email)
    """
    subject = f"New Proxy Summary Document – {form_type} – {company_name}"

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
      New Proxy Summary Document
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

        company_name = proxy_data.get("company_name") or filing_summary.cik_number or "Unknown"
        logger.info(
            f"Preparing to send summary email for: {company_name} (CIK {filing_summary.cik_number})")

        # Generate email HTML (same as old flow)
        subject, html_email = generate_summary_email_html(
            company_name=company_name,
            form_type=filing_summary.form_type,
            summary_doc_url=summary_url,
            cik_number=filing_summary.cik_number or "",
            proxy_sec_url=filing_summary.sec_document_url
        )
        logger.info(f"Generated email subject: {subject}")

        # Send email via n8n webhook (same URL as old flow)
        webhook_url = "https://n8n-xwx1.onrender.com/webhook/80830c6d-ff5b-45e3-9ef3-a061db1fbf0c"
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
            response = requests.post(
                webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            response.raise_for_status()

            logger.info(
                f"✅ Summary email sent successfully via n8n webhook! Status: {response.status_code}")
            logger.info(f"📧 Response: {response.text[:200]}")

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
