from coreschema import Null
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xml.etree.ElementTree as ET
from datetime import datetime
import time
import logging
import asyncio
import re
import os
import threading
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from django.core.mail import send_mail, EmailMultiAlternatives
from django.conf import settings
from bson import ObjectId
from mongoengine.errors import NotUniqueError
from .models import SECFiling, SECFeedStatus, LastCronJob, AccessionLookedUp
from .document_analyzer import SECDocumentAnalyzer
from .websocket_service import SECWebSocketService
from document_processor.models import ProcessingJob
from document_processor.services import DocumentProcessingService, SummaryGenerationService
from proxy_processor.views import process_sec_document_helper
from node_proxy.utils import call_node_api
from node_proxy.views import AnnouncementWithUrlView

logger = logging.getLogger(__name__)


def escape_html(text):
    """Escape HTML special characters"""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&#039;")
    return text


def generate_filing_email_html(filing_data, doc_files):
    """Generate HTML email for SEC filing notification"""
    form_type = filing_data.get('form_type', 'N/A')
    company_name = filing_data.get('company_name', 'Unknown Company')
    accession_no = filing_data.get('accession_number', 'N/A')
    filing_date = filing_data.get('filing_date', 'N/A')
    company_details = filing_data.get('company_details', None)
    if isinstance(filing_date, datetime):
        filing_date = filing_date.strftime('%Y-%m-%d')
    accepted_date = filing_data.get('acceptance_datetime_utc', 'N/A')
    if isinstance(accepted_date, datetime):
        accepted_date = accepted_date.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(accepted_date, str):
        try:
            dt = datetime.fromisoformat(accepted_date.replace('Z', '+00:00'))
            accepted_date = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass
    period = filing_data.get('period', 'N/A')
    cik = filing_data.get('cik_number', 'N/A')
    filing_url = filing_data.get('link', '')

    # Count documents
    documents_count = len(doc_files) if doc_files else 0

    # Generate document files table
    doc_files_html = ""
    if doc_files and len(doc_files) > 0:
        doc_files_html = """
    <table style="width:100%; border-collapse:collapse; margin-top:10px;">
      <thead>
        <tr style="background-color:#f5f5f5;">
          <th style="padding:8px; border:1px solid #ddd; text-align:left;">Seq</th>
          <th style="padding:8px; border:1px solid #ddd; text-align:left;">Description</th>
          <th style="padding:8px; border:1px solid #ddd; text-align:left;">Document</th>
          <th style="padding:8px; border:1px solid #ddd; text-align:left;">Type</th>
          <th style="padding:8px; border:1px solid #ddd; text-align:left;">Size</th>
        </tr>
      </thead>
      <tbody>
"""
        for idx, file in enumerate(doc_files):
            bg = "#ffffff" if idx % 2 == 0 else "#f9f9f9"
            seq = escape_html(file.get('sequence', ''))
            description = escape_html(file.get('description', ''))
            doc_name = escape_html(file.get('file', ''))
            doc_type = escape_html(file.get('type', ''))
            size = escape_html(file.get('size', ''))

            # Build full URL
            doc_url = file.get('url', '')
            if doc_url:
                if not (doc_url.startswith('http://') or doc_url.startswith('https://')):
                    doc_url = f"https://www.sec.gov{doc_url}"
                doc_name_html = f'<a href="{escape_html(doc_url)}" style="color:#4a90e2; text-decoration:none;" target="_blank">{doc_name}</a>'
            else:
                doc_name_html = doc_name

            doc_files_html += f"""
      <tr style="background-color:{bg};">
        <td style="padding:8px; border:1px solid #ddd;">{seq}</td>
        <td style="padding:8px; border:1px solid #ddd;">{description}</td>
        <td style="padding:8px; border:1px solid #ddd;">{doc_name_html}</td>
        <td style="padding:8px; border:1px solid #ddd;">{doc_type}</td>
        <td style="padding:8px; border:1px solid #ddd;">{size}</td>
      </tr>
"""
        doc_files_html += """
      </tbody>
    </table>
"""
    else:
        doc_files_html = "<p><em>No Document Format Files found.</em></p>"

    title_text = f"{form_type} – {company_name}" if form_type != 'N/A' and company_name != 'Unknown Company' else f"Filing #{accession_no}"
    subject = f"SEC Filing – {form_type} – {company_name}"

    html_email = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{escape_html(subject)}</title>
</head>
<body style="margin:0; padding:0; font-family:Arial,sans-serif; background-color:#f4f4f4;">
  <div style="max-width:900px; margin:20px auto; background-color:#ffffff; padding:30px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <h2 style="color:#333; text-align:center; margin-top:0; padding-bottom:20px; border-bottom:3px solid #4a90e2;">
      {escape_html(title_text)}
    </h2>

    <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
      <tr>
        <td style="padding:8px; font-weight:bold; width:170px; color:#555;">Form Type:</td>
        <td style="padding:8px; color:#333;">{escape_html(form_type)}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">Accession No.:</td>
        <td style="padding:8px; color:#333;">{escape_html(accession_no)}</td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Filing Date:</td>
        <td style="padding:8px; color:#333;">{escape_html(filing_date)}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">Accepted:</td>
        <td style="padding:8px; color:#333;">{escape_html(accepted_date)}</td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Period of Report:</td>
        <td style="padding:8px; color:#333;">{escape_html(period)}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">Documents Count:</td>
        <td style="padding:8px; color:#333;">{escape_html(str(documents_count))}</td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Company:</td>
        <td style="padding:8px; color:#333;">{escape_html(company_name)}</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">CIK:</td>
        <td style="padding:8px; color:#333;">{escape_html(cik)}</td>
      </tr>
"""

    # Add company details for 8-K forms
    if form_type == '8-K' and company_details:
        html_email += """
      <tr>
        <td colspan="2" style="padding:12px 8px 8px 8px; font-weight:bold; color:#4a90e2; font-size:14px; border-top:2px solid #e0e0e0;">
          Company Details (M&A Information)
        </td>
      </tr>
"""

        # Target Company Info
        target_name = company_details.get('target_name', '')
        if target_name:
            html_email += f"""
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">Target Company:</td>
        <td style="padding:8px; color:#333;">{escape_html(target_name)}</td>
      </tr>
"""

        target_cik = company_details.get('target_cik', '')
        if target_cik:
            html_email += f"""
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Target CIK:</td>
        <td style="padding:8px; color:#333;">{escape_html(target_cik)}</td>
      </tr>
"""

        # Acquirer Company Info
        acquirer_name = company_details.get('acquirer_name', '')
        if acquirer_name:
            html_email += f"""
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">Acquirer Company:</td>
        <td style="padding:8px; color:#333;">{escape_html(acquirer_name)}</td>
      </tr>
"""

        acquirer_cik = company_details.get('acquirer_cik', '')
        if acquirer_cik:
            html_email += f"""
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Acquirer CIK:</td>
        <td style="padding:8px; color:#333;">{escape_html(acquirer_cik)}</td>
      </tr>
"""

        # US Listed Status
        is_us_listed = company_details.get('is_target_us_listed')
        if is_us_listed is not None:
            listed_text = "✓ Yes" if is_us_listed else "✗ No"
            listed_color = "#28a745" if is_us_listed else "#dc3545"
            html_email += f"""
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; font-weight:bold; color:#555;">Target US Listed:</td>
        <td style="padding:8px; color:{listed_color}; font-weight:bold;">{escape_html(listed_text)}</td>
      </tr>
"""

        # Market Cap > $100M
        is_cap_gt_100m = company_details.get(
            'is_target_market_cap_greater_than_100m')
        if is_cap_gt_100m is not None:
            cap_text = "✓ Yes (> $100M)" if is_cap_gt_100m else "✗ No (< $100M)"
            cap_color = "#28a745" if is_cap_gt_100m else "#dc3545"
            html_email += f"""
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Market Cap > $100M:</td>
        <td style="padding:8px; color:{cap_color}; font-weight:bold;">{escape_html(cap_text)}</td>
      </tr>
"""

    html_email += """
      
"""

    if filing_url:
        html_email += f"""
      <tr>
        <td style="padding:8px; font-weight:bold; color:#555;">Filing URL:</td>
        <td style="padding:8px;">
          <a href="{escape_html(filing_url)}" style="color:#4a90e2; text-decoration:none;" target="_blank">
            View Filing Detail Page
          </a>
        </td>
      </tr>
"""

    html_email += f"""
    </table>

    <h3 style="color:#333; margin-top:20px; margin-bottom:10px;">Document Format Files</h3>
    {doc_files_html}

    <div style="margin-top:30px; padding-top:20px; border-top:1px solid #e0e0e0; text-align:center; color:#999; font-size:12px;">
      <p>This is an automated email generated from SEC EDGAR filing detail pages.</p>
    </div>
  </div>
</body>
</html>
"""

    return subject, html_email


def generate_8k_summary_email_html(company_name: str, form_type: str, summary_doc_url: str, cik_number: str, sec_url: str, accession_number: str) -> tuple:
    """
    Generate HTML email for 8-K summary document notification.

    Args:
        company_name: Name of the company
        form_type: Form type (e.g., 8-K)
        summary_doc_url: URL of the generated summary document
        cik_number: CIK number
        sec_url: URL of the SEC filing
        accession_number: SEC accession number
    Returns:
        tuple: (subject, html_email)
    """
    subject = f"New 8-K Summary Document – {form_type} – {company_name}"

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
      New 8-K Summary Document
    </h2>

    <div style="margin-bottom:30px;">
      <p style="color:#333; font-size:16px; line-height:1.6;">
        The 8-K summary document has been successfully generated for:
      </p>
      
      <div style="background-color:#f9f9f9; padding:15px; border-radius:5px; margin:20px 0;">
        <p style="margin:8px 0; color:#555;">
          <strong style="color:#333;">Company:</strong> {escape_html(company_name)}
        </p>
        <p style="margin:8px 0; color:#555;">
          <strong style="color:#333;">Form Type:</strong> {escape_html(form_type)}
        </p>
        <p style="margin:8px 0; color:#555;">
          <strong style="color:#333;">CIK Number:</strong> {escape_html(cik_number)}
        </p>
        <p style="margin:8px 0; color:#555;">
          <strong style="color:#333;">Accession Number:</strong> {escape_html(accession_number)}
        </p>
        <p style="margin:8px 0; color:#555;">
          <strong style="color:#333;">SEC URL:</strong> <a href="{escape_html(sec_url)}" style="color:#4a90e2; text-decoration:none;" target="_blank">{escape_html(sec_url)}</a>
        </p>
      </div>
    </div>

    <div style="text-align:center; margin:30px 0;">
      <a href="{escape_html(summary_doc_url)}" 
         style="display:inline-block; background-color:#4a90e2; color:#ffffff; padding:15px 30px; text-decoration:none; border-radius:5px; font-size:16px; font-weight:bold; box-shadow:0 2px 4px rgba(0,0,0,0.2);">
        Download Summary Document
      </a>
    </div>

    <div style="margin-top:30px; padding:15px; background-color:#e8f4f8; border-radius:5px; border-left:4px solid #4a90e2;">
      <p style="margin:0; color:#555; font-size:14px;">
        <strong>Note:</strong> This document contains the summary generated from the 8-K filing.
      </p>
    </div>

    <div style="margin-top:30px; padding-top:20px; border-top:1px solid #e0e0e0; text-align:center; color:#999; font-size:12px;">
      <p>This is an automated email notification for 8-K summary document generation.</p>
      <p style="margin-top:5px;">
        <a href="{escape_html(summary_doc_url)}" style="color:#4a90e2; text-decoration:none; word-break:break-all;">
          {escape_html(summary_doc_url)}
        </a>
      </p>
    </div>
  </div>
</body>
</html>
"""

    return subject, html_email


def send_8k_summary_email(deal_id, company_name, form_type, cik_number, sec_url, accession_number):
    """
    Send email notification with 8-K summary document URL.

    Args:
        deal_id: ProcessingJob ID
        company_name: Company name
        form_type: Form type (e.g., 8-K)
        cik_number: CIK number
        sec_url: SEC filing URL
        accession_number: SEC accession number
    """
    try:
        # Get the job to retrieve summary_docx_url
        try:
            object_id = ObjectId(deal_id)
            job = ProcessingJob.objects.get(id=object_id)
        except Exception as e:
            logger.error(f"Error retrieving job {deal_id}: {e}")
            return

        if not job.summary_docx_url:
            logger.warning(
                f"No summary document URL available for job {deal_id}")
            return

        logger.info(
            f"Preparing to send 8-K summary email for: {company_name}")

        # Generate email HTML
        subject, html_email = generate_8k_summary_email_html(
            company_name=company_name,
            form_type=form_type,
            summary_doc_url=job.summary_docx_url,
            cik_number=cik_number,
            sec_url=sec_url,
            accession_number=accession_number
        )
        logger.info(f"Generated email subject: {subject}")

        # Send email via n8n webhook
        webhook_url = "https://n8n-xwx1.onrender.com/webhook/b3007d21-6845-47b5-aece-7b26583758bc"
        logger.info(
            f"📤 Sending 8-K summary email via n8n webhook: {webhook_url}")

        # Prepare payload for n8n webhook
        payload = {
            'subject': subject,
            'html': html_email,
            'company_name': company_name,
            'form_type': form_type,
            'summary_doc_url': job.summary_docx_url,
            'deal_id': deal_id
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
                f"✅ 8-K summary email sent successfully via n8n webhook! Status: {response.status_code}")
            logger.info(f"📧 Response: {response.text[:200]}")

        except requests.exceptions.RequestException as e:
            logger.error(
                f"❌ Error sending 8-K summary email via n8n webhook: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(
                    f"❌ Response status: {e.response.status_code}, Response body: {e.response.text[:200]}")
            raise

    except Exception as e:
        logger.error(
            f"❌ Error sending 8-K summary email notification: {e}", exc_info=True)
        # Don't fail the entire process if email sending fails


def generate_8k_summary_async(deal_id, company_name, form_type, cik_number, sec_url, accession_number, max_attempts=60, delay_seconds=30):
    """
    Async function to generate summary for 8-K after processing completes.
    Monitors for schema_results and then generates summary.

    Args:
        deal_id: ProcessingJob ID
        company_name: Company name
        form_type: Form type
        cik_number: CIK number
        sec_url: SEC filing URL
        accession_number: SEC accession number
        max_attempts: Maximum number of attempts to check for schema_results
        delay_seconds: Delay between attempts in seconds
    """
    try:
        logger.info(
            f"🔍 Starting summary generation monitoring for deal_id: {deal_id}")

        # Monitor for schema_results
        object_id = ObjectId(deal_id)
        attempts = 0

        while attempts < max_attempts:
            try:
                job = ProcessingJob.objects(id=object_id).first()

                if not job:
                    logger.error(f"❌ Job {deal_id} not found")
                    return

                # Check if schema_results are available
                if job.schema_results and job.embedding_status == 'COMPLETED':
                    logger.info(
                        f"✅ Schema results available for deal_id: {deal_id}, generating summary")

                    # Update summary status to processing
                    job.summary_status = 'PROCESSING'
                    job.save()

                    # Generate summary
                    summary_service = SummaryGenerationService()
                    result = summary_service.generate_summary_engine(
                        deal_id=deal_id,
                        temperature=1,
                        provider='openai',
                        model='gpt-5'
                    )

                    if result:
                        # Update job with summary URL
                        job.summary_docx_url = result
                        job.summary_using = "gpt-5"
                        job.summary_status = 'COMPLETED'
                        job.save()

                        logger.info(
                            f"✅ Summary generated successfully for deal_id: {deal_id}")

                        # Send email notification
                        try:
                            logger.info(
                                f"📧 Sending 8-K summary email for: {company_name}")
                            send_8k_summary_email(
                                deal_id=deal_id,
                                company_name=company_name,
                                form_type=form_type,
                                cik_number=cik_number,
                                sec_url=sec_url,
                                accession_number=accession_number
                            )
                            logger.info(
                                f"✅ 8-K summary email sent successfully")
                        except Exception as email_error:
                            logger.error(
                                f"❌ Error sending 8-K summary email: {str(email_error)}")

                        return
                    else:
                        logger.error(
                            f"❌ Summary generation returned None for deal_id: {deal_id}")
                        job.summary_status = 'FAILED'
                        job.error_message = 'Summary generation returned None'
                        job.save()
                        return

            except Exception as e:
                logger.error(f"❌ Error checking job status: {e}")
                # Continue to next attempt unless it's a critical error
                if 'not found' in str(e).lower() or 'does not exist' in str(e).lower():
                    logger.error(f"❌ Job {deal_id} not found")
                    return

            # Wait before next attempt
            attempts += 1
            if attempts < max_attempts:
                logger.info(
                    f"⏳ Waiting for schema_results... (attempt {attempts}/{max_attempts})")
                time.sleep(delay_seconds)

        logger.warning(
            f"⚠️ Timeout waiting for schema_results for deal_id: {deal_id}")

    except Exception as e:
        logger.error(
            f"❌ Error in generate_8k_summary_async: {e}", exc_info=True)


def process_8k_document_async(ex21_url, cik_number, company_name, sec_filing_id, filing_date, item_data, company_details):
    """
    Async function to process 8-K document using Node API.

    Args:
        ex21_url: URL of EX-2.1 HTM file
        cik_number: CIK number
        company_name: Company name
        sec_filing_id: SEC filing ID
        filing_date: Filing date
        item_data: Item data from SEC filing
    """
    try:
        logger.info(f"🚀 Starting 8-K document processing for: {company_name}")

        # Prepare data for Node API (company_details keys: target_name, target_cik, acquirer_name, acquirer_cik, ...)
        target_cik = company_details.get('target_cik', '')
        target_name = company_details.get('target_name', '')
        acquirer_cik = company_details.get('acquirer_cik', '')
        acquirer_name = company_details.get('acquirer_name', '')
        acquired_name = company_details.get('acquired_name', '')
        announce_data = filing_date.strftime(
            '%Y-%m-%d') if isinstance(filing_date, datetime) else str(filing_date)

        data = {
            "target_cik": target_cik,
            "target_name": target_name,
            "announce_data": announce_data,
            "acquired_name": acquired_name,
            "acquirer_cik": acquirer_cik,
            "url": ex21_url,
            "sec_filing_id": sec_filing_id,
            "company_details": company_details,
        }

        # Update data with extracted values
        if not data.get('target_cik'):
            data['target_cik'] = target_cik
        if not data.get('target_name'):
            data['target_name'] = target_name
        if not data.get('announce_data'):
            data['announce_data'] = announce_data

        # Validate required fields
        if not all([data.get('target_cik'), data.get('announce_data'), data.get('target_name')]):
            missing_fields = []
            if not data.get('target_cik'):
                missing_fields.append('target_cik')
            if not data.get('announce_data'):
                missing_fields.append('announce_data')
            if not data.get('target_name'):
                missing_fields.append('target_name')

            logger.error(
                f"❌ Missing required fields: {', '.join(missing_fields)}")
            # Send failure event
            doc_processor = DocumentProcessingService()
            doc_processor._send_sec_filing_event(
                sec_filing_id, "Fail", f"Could not obtain all required fields: {', '.join(missing_fields)}")
            return

        # Send processing started event
        doc_processor = DocumentProcessingService()
        doc_processor._send_sec_filing_event(sec_filing_id, "In Progress")

        # Call Node API
        logger.info(f"📞 Calling Node API with data: {data}")
        response = call_node_api(
            endpoint="deal/process-with-url",
            method="POST",
            data={
                "url": ex21_url,
                "target_cik": data.get('target_cik', ''),
                "announce_data": data.get('announce_data'),
                "target_name": data.get('target_name', ''),
                "acquired_name": data.get('acquirer_name', ''),
                "sec_filing_id": sec_filing_id,
                "acquirer_cik": data.get('acquirer_cik', ''),
                "is_from_ui": False
            }
        )

        logger.info(f"📥 Node API response: {response}")

        # Check if we got a successful response with jsonUrl
        if response.get('status') and response.get('data', {}).get('jsonUrl'):
            deal_id = response['data'].get('deal_id')
            json_url = response['data']['jsonUrl']

            if deal_id:
                logger.info(f"✅ Processing started, deal_id: {deal_id}")

                # Process the document using the JSON URL
                process_result = doc_processor.process_document(
                    file_url=json_url,
                    deal_id=deal_id,
                    sec_filing_id=sec_filing_id,
                    embed_data=True
                )

                logger.info(f"✅ Document processing started: {process_result}")

                # Start monitoring for summary generation in a separate thread
                summary_thread = threading.Thread(
                    target=generate_8k_summary_async,
                    args=(
                        deal_id,
                        company_name,
                        item_data.get('form_type', '8-K'),
                        cik_number,
                        item_data.get('link', ''),
                        item_data.get('accession_number', '')
                    )
                )
                summary_thread.daemon = True
                summary_thread.start()

                logger.info(
                    f"✅ Started summary generation monitoring thread for deal_id: {deal_id}")
            else:
                logger.error("❌ No deal_id in Node API response")
                doc_processor._send_sec_filing_event(
                    sec_filing_id, "Fail", "No deal_id returned from Node API")
        else:
            logger.error(
                f"❌ Node API response did not contain expected data: {response}")
            doc_processor._send_sec_filing_event(
                sec_filing_id, "Fail", "Invalid response from Node API")

    except Exception as e:
        logger.error(
            f"❌ Error in process_8k_document_async: {e}", exc_info=True)
        try:
            doc_processor = DocumentProcessingService()
            doc_processor._send_sec_filing_event(
                sec_filing_id, "Fail", str(e))
        except:
            pass


def process_8k_document_helper(cik_number, company_name, sec_filing_id, filing_date, form_type, ex21_url, item_data, company_details):
    """
    Helper function to process 8-K document programmatically.
    Similar to process_sec_document_helper but for 8-K filings.

    Args:
        cik_number: CIK number of the company
        company_name: Name of the company
        sec_filing_id: SEC filing ID (MongoDB ObjectId as string)
        filing_date: Filing date
        form_type: Form type (should be '8-K')
        ex21_url: URL of the EX-2.1 HTM file
        item_data: Full item data from SEC filing
        company_details: Company details from GPT analysis
    Returns:
        dict with status, or None if error
    """
    try:
        logger.info(
            f"🚀 Starting 8-K document processing for: {company_name} - {sec_filing_id}")

        # Start processing in a separate thread
        processing_thread = threading.Thread(
            target=process_8k_document_async,
            args=(
                ex21_url,
                cik_number,
                company_name,
                sec_filing_id,
                filing_date,
                item_data,
                company_details
            )
        )
        processing_thread.daemon = True
        processing_thread.start()

        logger.info(
            f"✅ Started 8-K document processing thread for: {company_name}")

        return {
            'status': 'In Progress',
            'message': '8-K document processing started',
            'company_name': company_name,
            'cik_number': cik_number,
            'sec_filing_id': sec_filing_id,
            'ex21_url': ex21_url
        }

    except Exception as e:
        logger.error(
            f"❌ Error starting 8-K document processing: {str(e)}", exc_info=True)
        return None


class SECRSSParser:
    def __init__(self, form_type=None):
        self.form_type = form_type
        self.headers = {
            "User-Agent":
            "MNA-Finder/1.0 (https://teqnodux.com; contact: ashish.kachadiya@teqnodux.com)",
            'Accept': 'application/atom+xml, application/xml, text/xml, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'https://www.sec.gov/',
        }

        self.form_types = ["8-k", "DEFM14A",
                           "DEFM14C", "PREM14A", "PREM14C", "S-4", "S-4/A", "F-4", "F-4/A", "SC 14D9", "SC 14D9/A", "10-Q", "10-K"]

        self.proxy_watcher = [
            {
                "target_name": "CIVITAS RESOURCES, INC",
                "acquire_name": "SM Energy Company",
                "target_cik": "0001509589",
                "acquirer_cik": "0000893538"
            },
            {
                "target_name": "Forge Global, Inc.",
                "acquire_name": "SCHWAB CHARLES CORP",
                "target_cik": "0001827821",
                "acquirer_cik": "000316709"
            },
            {
                "target_name": "Semrush Holdings, Inc.",
                "acquire_name": "Adobe Inc.",
                "target_cik": "0001831840",
                "acquirer_cik": "0000796343"
            },
            {
                "target_name": "HOLOGIC INC",
                "acquire_name": "Blackstone Inc. and TPG",
                "target_cik": "0000859737",
                "acquirer_cik": "0001393818"
            },
            {
                "target_name": "Brighthouse Financial, Inc.",
                "acquire_name": None,
                "target_cik": "0001685040",
                "acquirer_cik": None
            },
            {
                "target_name": "Exact Sciences Corporation",
                "acquire_name": "Abbott Laboratories",
                "target_cik": "0001124140",
                "acquirer_cik": "0000001800"
            },
            {
                "target_name": "Axalta Coating Systems Ltd.",
                "acquire_name": "AKZO NOBEL NV",
                "target_cik": "0001616862",
                "acquirer_cik": "0000003124"
            },
            {
                "target_name": "Sealed Air Corporation",
                "acquire_name": "Clayton, Dubilier & Rice (CD&R) affiliated funds",
                "target_cik": "0001012100",
                "acquirer_cik": ""
            },
            {
                "target_name": "TreeHouse Foods, Inc.",
                "acquire_name": "Investindustrial",
                "target_cik": "0001320695",
                "acquirer_cik": ""
            },
            {
                "target_name": "Alexander & Baldwin, Inc.",
                "acquire_name": "Blackstone Inc.",
                "target_cik": "0001545654",
                "acquirer_cik": "0001393818"
            },
            {
                "target_name": "Confluent, Inc.",
                "acquire_name": "International Business Machines Corporation",
                "target_cik": "0001699838",
                "acquirer_cik": "0000051143"
            },
            {
                "target_name": "Warner Bros. Discovery, Inc.",
                "acquire_name": "Netflix, Inc.",
                "target_cik": "0001210476",
                "acquirer_cik": "0001065280"
            },
            {
                "target_name": "Blue Foundry Bancorp",
                "acquire_name": "Fulton Financial Corporation",
                "target_cik": "0001846017",
                "acquirer_cik": "0000700564"
            },
            {
                "target_name": "Eventbrite, Inc.",
                "acquire_name": "Bending Spoons",
                "target_cik": "0001475115",
                "acquirer_cik": ""
            },
        ]

        # Initialize feed URL placeholder (set per form type)
        self.feed_url = None
        self.set_feed_url(self.form_type)

        # Create session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def set_feed_url(self, form_type):
        """Update the feed URL for a specific form type."""
        if form_type:
            self.feed_url = (
                "https://www.sec.gov/cgi-bin/browse-edgar?"
                f"action=getcurrent&CIK=&type={form_type}&company=&dateb=&owner=include&start=0&count=40&output=atom"
            )
        else:
            self.feed_url = None

    def fetch_rss_feed(self):
        max_retries = 3
        if not self.feed_url:
            logger.error("Feed URL is not set; cannot fetch RSS/Atom feed")
            return None
        for attempt in range(max_retries):
            try:
                # Add a longer delay to be respectful to SEC servers
                time.sleep(2 + attempt)  # Progressive delay: 2s, 3s, 4s

                response = self.session.get(
                    self.feed_url, headers=self.headers, timeout=30)
                response.raise_for_status()
                print("response.text[:1200]", response.text[:1200])
                return response.text
            except Exception as e:
                logger.error(
                    f"Error fetching RSS feed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(5)  # Wait 5 seconds before retry
        return None

    def parse_rss_content(self, rss_content):
        """Parse the SEC Atom feed (new flow only)."""
        try:
            root = ET.fromstring(rss_content)
            return self.parse_atom_content(root)
        except Exception as e:
            logger.error(f"Error parsing feed XML: {e}")
            print(f"Exception in parse_rss_content: {e}")
            return []

    def parse_atom_content(self, root):
        """Parse Atom feed format"""
        try:
            # Find all entry elements in Atom feed
            items = []
            entry_elements = root.findall(
                './/{http://www.w3.org/2005/Atom}entry')
            if not entry_elements:
                entry_elements = root.findall('.//entry')

            print(f"Found {len(entry_elements)} entry elements in Atom feed")

            for i, entry in enumerate(entry_elements):
                item_data = self.parse_atom_entry(entry, i+1)
                if item_data:
                    items.append(item_data)
                else:
                    print(f"Failed to parse entry {i+1}")

            print(f"Total items parsed from Atom feed: {len(items)}")
            return items
        except Exception as e:
            logger.error(f"Error parsing Atom feed: {e}")
            print(f"Exception in parse_atom_content: {e}")
            return []

    def parse_atom_entry(self, entry_elem, entry_number):
        """Parse Atom feed entry element"""
        try:
            # Extract basic Atom entry fields
            title_elem = entry_elem.find(
                './/{http://www.w3.org/2005/Atom}title')
            if title_elem is None:
                title_elem = entry_elem.find('.//title')
            title = title_elem.text if title_elem is not None else ''

            # Extract link
            link_elem = entry_elem.find(
                './/{http://www.w3.org/2005/Atom}link[@rel="alternate"]')
            if link_elem is None:
                link_elem = entry_elem.find(
                    './/{http://www.w3.org/2005/Atom}link')
            if link_elem is None:
                link_elem = entry_elem.find('.//link')
            link = link_elem.get('href') if link_elem is not None else ''

            # Extract ID (used as guid)
            id_elem = entry_elem.find('.//{http://www.w3.org/2005/Atom}id')
            if id_elem is None:
                id_elem = entry_elem.find('.//id')
            guid = id_elem.text if id_elem is not None else ''

            # Extract summary/description
            summary_elem = entry_elem.find(
                './/{http://www.w3.org/2005/Atom}summary')
            if summary_elem is None:
                summary_elem = entry_elem.find('.//summary')
            description = summary_elem.text if summary_elem is not None else ''

            # Extract updated date
            updated_elem = entry_elem.find(
                './/{http://www.w3.org/2005/Atom}updated')
            if updated_elem is None:
                updated_elem = entry_elem.find('.//updated')
            pubDate = updated_elem.text if updated_elem is not None else None

            # Extract form type from category
            form_type = None
            category_elems = entry_elem.findall(
                './/{http://www.w3.org/2005/Atom}category')
            if not category_elems:
                category_elems = entry_elem.findall('.//category')

            for cat in category_elems:
                term = cat.get('term', '')
                if term and term.strip():
                    form_type = term.strip()
                    break

            # Extract accession number from ID
            accession_number = None
            if guid:
                # ID format: urn:tag:sec.gov,2008:accession-number=0001493152-25-027089
                match = re.search(r'accession-number=([\d-]+)', guid)
                if match:
                    accession_number = match.group(1)

            # Return basic entry data - HTML parsing will be done later
            return {
                'title': title,
                'link': link,
                'guid': guid,
                'description': description,
                'pubDate': pubDate,
                'form_type': form_type,
                'accession_number': accession_number,

            }
        except Exception as e:
            logger.error(f"Error parsing Atom entry: {e}")
            print(f"Exception parsing Atom entry: {e}")
            return None

    def fetch_and_parse_html(self, html_url, form_type_from_feed=None):
        """Fetch HTML from filing link and parse all relevant information

        Args:
            html_url: URL to the filing HTML page
            form_type_from_feed: Form type already extracted from Atom feed (from <category term="...">)
        """
        try:
            time.sleep(2)  # Be respectful to SEC servers
            response = self.session.get(
                html_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            html_content = response.text

            soup = BeautifulSoup(html_content, 'html.parser')

            # Find company_info div (needed for both form_type extraction and company info extraction)
            company_info = soup.find('div', class_='companyInfo')

            # Use form_type from Atom feed if available, otherwise extract from HTML
            form_type = form_type_from_feed

            # Only extract from HTML if not already provided from Atom feed
            if not form_type:
                # Extract form type - try from companyInfo first (more reliable)
                if company_info:
                    ident_info = company_info.find('p', class_='identInfo')
                    if ident_info:
                        # Extract form type from "Type: <strong>8-K</strong>" pattern
                        # Find all text nodes and strong tags to locate the one after "Type:"
                        ident_text = ident_info.get_text()
                        # Try regex first
                        type_match = re.search(
                            r'Type[:\s]+([A-Z0-9\s-]+)', ident_text)
                        if type_match:
                            form_type = type_match.group(1).strip()
                            print(
                                f"Extracted form_type from companyInfo: {form_type}")
                        else:
                            # Fallback: find strong tag that appears after "Type:" text
                            for elem in ident_info.find_all('strong'):
                                # Get all previous siblings/text to check if "Type:" appears before this strong tag
                                prev_siblings = list(elem.previous_siblings)
                                prev_text = ' '.join([
                                    str(s) for s in prev_siblings if isinstance(s, str) or (hasattr(s, 'get_text') and s.get_text())
                                ])
                                # Check if "Type:" appears in the text before this strong tag
                                if 'Type' in prev_text or 'Type:' in prev_text:
                                    form_type = elem.get_text().strip()
                                    break

                # Fallback to formName if not found in companyInfo
                if not form_type:
                    form_name_elem = soup.find('div', {'id': 'formName'})
                    if form_name_elem:
                        form_text = form_name_elem.get_text()
                        # Extract form type like "DEF 14A" or "8-K"
                        match = re.search(r'Form\s+([A-Z0-9\s-]+)', form_text)
                        if match:
                            form_type = match.group(1).strip()
            else:
                # Use form_type from Atom feed (already extracted from <category term="...">)
                print(f"Using form_type from Atom feed: {form_type}")

            # Normalize form_type: remove suffixes like " - O", " - A", etc.
            # if form_type:
            #     if ' - ' in form_type:
            #         form_type = form_type.split(' - ')[0].strip()
            #     # Also handle cases like "DEF 14A-O" -> "DEF 14A"
            #     form_type = re.sub(r'-[A-Z]$', '', form_type).strip()
                # Note: Keep "8-K" in code, will normalize to "8" only when saving to DB

            # Extract accession number
            sec_num_elem = soup.find('div', {'id': 'secNum'})
            accession_number = None
            if sec_num_elem:
                acc_text = sec_num_elem.get_text()
                match = re.search(r'(\d{10}-\d{2}-\d{6})', acc_text)
                if match:
                    accession_number = match.group(1)

            # Extract filing date
            filing_date = None
            info_heads = soup.find_all('div', class_='infoHead')
            for info_head in info_heads:
                if 'Filing Date' in info_head.get_text():
                    info_elem = info_head.find_next_sibling(
                        'div', class_='info')
                    if info_elem:
                        try:
                            filing_date = datetime.strptime(
                                info_elem.get_text().strip(), '%Y-%m-%d')
                        except:
                            pass
                    break

            # Extract accepted date
            acceptance_datetime_utc = None
            for info_head in info_heads:
                if 'Accepted' in info_head.get_text():
                    info_elem = info_head.find_next_sibling(
                        'div', class_='info')
                    if info_elem:
                        accepted_text = info_elem.get_text().strip()
                        try:
                            # Parse format: 2025-12-08 06:00:31
                            dt = datetime.strptime(
                                accepted_text, '%Y-%m-%d %H:%M:%S')
                            from zoneinfo import ZoneInfo
                            dt_et = dt.replace(
                                tzinfo=ZoneInfo("America/New_York"))
                            acceptance_datetime_utc = dt_et.astimezone(
                                ZoneInfo("UTC")).isoformat()
                        except:
                            pass
                    break

            # Extract period of report
            period = None
            for info_head in info_heads:
                if 'Period of Report' in info_head.get_text():
                    info_elem = info_head.find_next_sibling(
                        'div', class_='info')
                    if info_elem:
                        period = info_elem.get_text().strip()
                    break

            # Extract company information
            company_name = None
            cik_number = None
            file_number = None
            ein = None
            state_of_incorp = None
            fiscal_year_end = None
            assigned_sic = None

            # company_info already found above when extracting form_type
            if company_info:
                # Extract company name and CIK
                company_name_elem = company_info.find(
                    'span', class_='companyName')
                if company_name_elem:
                    company_text = company_name_elem.get_text()
                    # Extract company name (before CIK)
                    match = re.match(r'^([^(]+)', company_text)
                    if match:
                        company_name = match.group(1).strip()

                    # Extract CIK
                    cik_match = re.search(r'CIK[:\s]+(\d+)', company_text)
                    if cik_match:
                        cik_number = cik_match.group(
                            1).zfill(10)  # Pad to 10 digits

                # Extract other company info
                ident_info = company_info.find('p', class_='identInfo')
                if ident_info:
                    ident_text = ident_info.get_text()

                    # Extract EIN
                    ein_match = re.search(r'EIN[.\s:]+(\d+)', ident_text)
                    if ein_match:
                        ein = ein_match.group(1)

                    # Extract State of Incorporation
                    state_match = re.search(
                        r'State of Incorp[.:\s]+([A-Z]{2})', ident_text)
                    if state_match:
                        state_of_incorp = state_match.group(1)

                    # Extract Fiscal Year End
                    fye_match = re.search(
                        r'Fiscal Year End[:\s]+(\d{4})', ident_text)
                    if fye_match:
                        fiscal_year_end = fye_match.group(1)

                    # Extract File Number
                    file_num_match = re.search(
                        r'File No[.:\s]+(\d{3}-\d+)', ident_text)
                    if file_num_match:
                        file_number = file_num_match.group(1)

                    # Extract SIC
                    sic_match = re.search(r'SIC[:\s]+(\d{4})', ident_text)
                    if sic_match:
                        try:
                            assigned_sic = int(sic_match.group(1))
                        except:
                            pass

            # Extract document files from table
            xbrl_files = []
            table = soup.find('table', class_='tableFile')
            if table:
                rows = table.find_all('tr')[1:]  # Skip header row
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 4:
                        seq = cells[0].get_text().strip()
                        description = cells[1].get_text().strip()
                        doc_link = cells[2].find('a')
                        doc_type = cells[3].get_text().strip()
                        size_text = cells[4].get_text().strip() if len(
                            cells) > 4 else '0'

                        if doc_link:
                            doc_url = doc_link.get('href', '')
                            if not doc_url.startswith('http'):
                                doc_url = urljoin(
                                    'https://www.sec.gov', doc_url)

                            # Extract file size
                            size = 0
                            if size_text:
                                size_match = re.search(
                                    r'(\d+)', size_text.replace(',', ''))
                                if size_match:
                                    size = int(size_match.group(1))

                            file_data = {
                                'sequence': int(seq) if seq.isdigit() else 0,
                                'file': doc_link.get_text().strip(),
                                'type': doc_type,
                                'size': size,
                                'description': description,
                                'url': doc_url,
                                'doc_type': doc_type
                            }

                            # Filter files based on form type
                            # For 8-K: include all HTM files (EX-2.1 when present, plus other .htm e.g. main 8-K)
                            # For DEF 14A/PRE 14A: only include DEF 14A/PRE 14A HTM files (exclude GRAPHIC)
                            # For other forms: include all HTM files
                            should_include = False
                            if form_type in ["DEF 14A", "PRE 14A"]:
                                if (doc_type in ["DEF 14A", "PRE 14A"]) and doc_url.endswith('.htm'):
                                    should_include = True
                            else:
                                # For other forms, include HTM files
                                if doc_url.endswith('.htm'):
                                    should_include = True

                            if should_include:
                                xbrl_files.append(file_data)

            # Check if EX-2.1 exists for 8-K filings
            has_ex21 = False
            if form_type == "8-K":
                has_ex21 = any(
                    'EX-2.1' in file.get('type',
                                         '') or 'EX-2.1' in file.get('description', '')
                    for file in xbrl_files
                )

            # Ensure we have at least form_type and accession_number
            if not form_type:
                logger.warning(f"Could not extract form_type from {html_url}")
                print(f"Warning: Could not extract form_type from {html_url}")
                # Try to get form_type from the title or other sources
                title_elem = soup.find('title')
                if title_elem:
                    title_text = title_elem.get_text()
                    # Try to extract from title like "DEF 14A - COMPANY NAME"
                    title_match = re.search(
                        r'([A-Z0-9\s]+)\s+-\s+', title_text)
                    if title_match:
                        form_type = title_match.group(1).strip()
                        # Normalize
                        if ' - ' in form_type:
                            form_type = form_type.split(' - ')[0].strip()
                        form_type = re.sub(r'-[A-Z]$', '', form_type).strip()
                        logger.info(
                            f"Extracted form_type from title: {form_type}")

            if not accession_number:
                logger.warning(
                    f"Could not extract accession_number from {html_url}")
                print(
                    f"Warning: Could not extract accession_number from {html_url}")
                # Try to extract from URL
                url_match = re.search(r'/(\d{10}-\d{2}-\d{6})', html_url)
                if url_match:
                    accession_number = url_match.group(1)
                    logger.info(
                        f"Extracted accession_number from URL: {accession_number}")

            if not form_type or not accession_number:
                logger.error(
                    f"Missing required fields - form_type: {form_type}, accession_number: {accession_number} for {html_url}")
                print(
                    f"Error: Missing required fields for {html_url} - form_type: {form_type}, accession_number: {accession_number}")
                return None

            print(
                f"form_type: {form_type}, accession_number: {accession_number}")

            return {
                'form_type': form_type,
                'accession_number': accession_number,
                'filing_date': filing_date,
                'acceptance_datetime_utc': acceptance_datetime_utc,
                'period': period,
                'company_name': company_name,
                'cik_number': cik_number,
                'file_number': file_number,
                'ein': ein,
                'state_of_incorp': state_of_incorp,
                'fiscal_year_end': fiscal_year_end,
                'assigned_sic': assigned_sic,
                'xbrl_files': xbrl_files,
                'has_ex21': has_ex21
            }
        except Exception as e:
            logger.error(
                f"Error fetching/parsing HTML from {html_url}: {e}", exc_info=True)
            print(f"Exception in fetch_and_parse_html for {html_url}: {e}")
            import traceback
            print(traceback.format_exc())
            return None


class SECFeedProcessor:
    def __init__(self, form_type=None):
        self.parser = SECRSSParser(form_type=form_type)
        self.document_analyzer = SECDocumentAnalyzer()
        self.form_type = form_type

    def check_cik_in_deals(self, cik_number: str) -> bool:
        """Check if CIK exists in the Deals collection"""
        try:
            if not cik_number:
                return False

            # Check if any deal exists with this CIK
            deal_exists = ProcessingJob.objects(
                cik=cik_number).first() is not None

            logger.info(
                f"CIK {cik_number} {'found' if deal_exists else 'not found'} in Deals collection")
            return deal_exists

        except Exception as e:
            logger.error(
                f"Error checking CIK {cik_number} in Deals collection: {e}")
            return False

    def _process_single_form_type(self, form_type):
        """Process a single form type - extracted for parallel processing"""
        try:
            # Create a new parser instance for thread safety
            parser = SECRSSParser(form_type=form_type)
            parser.set_feed_url(form_type)
            rss_content = parser.fetch_rss_feed()
            if not rss_content:
                print(f"Failed to fetch feed for form type: {form_type}")
                return {
                    'form_type': form_type,
                    'processed_items': [],
                    'new_items_count': 0,
                    'success': False,
                    'error': 'Failed to fetch RSS feed'
                }

            items = parser.parse_rss_content(rss_content)
            print(
                f"Parsed {len(items)} items from feed for form type: {form_type}")

            # Filter items by checking accession numbers in accession_lookedup and database
            unique_items = []
            new_accession_numbers = []  # Track new accession numbers to add to AccessionLookedUp

            for item_data in items:
                accession_number = item_data.get('accession_number')
                if not accession_number and item_data.get('guid'):
                    match = re.search(
                        r'accession-number=([\d-]+)', item_data.get('guid', ''))
                    if match:
                        accession_number = match.group(1)

                if accession_number:
                    # First check if accession number exists in AccessionLookedUp
                    looked_up = AccessionLookedUp.objects(
                        accession_number=accession_number).first()
                    if looked_up:
                        print(
                            f"Skipping already looked up filing: {accession_number}")
                        continue

                    # Then check if it exists in SECFiling
                    existing = SECFiling.objects(
                        accession_number=accession_number).first()
                    if existing:
                        print(
                            f"Skipping existing filing: {accession_number}")
                        # Add to AccessionLookedUp to avoid checking again in future
                        try:
                            AccessionLookedUp(
                                accession_number=accession_number).save()
                        except Exception as e:
                            # Ignore duplicate key errors (race condition)
                            if 'duplicate' not in str(e).lower() and 'E11000' not in str(e):
                                logger.warning(
                                    f"Failed to save accession number to AccessionLookedUp: {e}")
                        continue

                    # New accession number - add to list for processing
                    unique_items.append(item_data)
                    new_accession_numbers.append(accession_number)
                else:
                    # No accession number - process it
                    unique_items.append(item_data)

            # Deduplicate by accession_number so we never save or email twice for same record in one run
            seen_accessions = set()
            unique_items_deduped = []
            for item in unique_items:
                acc = item.get('accession_number')
                if not acc and item.get('guid'):
                    match = re.search(
                        r'accession-number=([\d-]+)', item.get('guid', ''))
                    if match:
                        acc = match.group(1)
                if acc and acc in seen_accessions:
                    print(
                        f"Skipping duplicate accession in same feed: {acc}")
                    continue
                if acc:
                    seen_accessions.add(acc)
                unique_items_deduped.append(item)
            unique_items = unique_items_deduped

            # Bulk insert new accession numbers to AccessionLookedUp
            if new_accession_numbers:
                try:
                    for acc_num in new_accession_numbers:
                        try:
                            AccessionLookedUp(accession_number=acc_num).save()
                        except Exception as e:
                            # Ignore duplicate key errors (race condition)
                            if 'duplicate' not in str(e).lower() and 'E11000' not in str(e):
                                logger.warning(
                                    f"Failed to save accession number {acc_num} to AccessionLookedUp: {e}")
                except Exception as e:
                    logger.error(
                        f"Error bulk saving accession numbers to AccessionLookedUp: {e}")

            print(
                f"Found {len(unique_items)} unique new items to process for {form_type}")

            # Process each unique item
            processed_items = []
            for item_data in unique_items:

                form_type_from_feed = item_data.get('form_type')

                html_url = item_data.get('link')
                if html_url:
                    # Pass form_type from Atom feed to HTML parser
                    print(
                        f"Fetching HTML for: {html_url} (form_type from feed: {form_type_from_feed})")
                    html_data = parser.fetch_and_parse_html(
                        html_url, form_type_from_feed=form_type_from_feed)
                    if html_data:
                        item_data.update(html_data)

                        if item_data.get('form_type') == '8-K' and not item_data.get('has_ex21'):
                            print(
                                f"Skipping 8-K filing without EX-2.1: {item_data.get('accession_number')}")
                            continue
                    else:
                        print(f"Failed to parse HTML for: {html_url}")
                        continue

                # 8-K/A: allow through for save + email (no document processing)
                # No final skip for 8-K/A so they get saved and can trigger email like 10-Q/10-K

                processed_items.append(item_data)

            print(
                f"Processing {len(processed_items)} items after HTML parsing and filtering for {form_type}")
            new_items_count = 0

            for item_data in processed_items:
                if self.save_filing(item_data):
                    new_items_count += 1

            # Emit processing statistics per form type
            if new_items_count > 0:
                stats = {
                    'total_processed': len(processed_items),
                    'new_filings': new_items_count,
                    'processing_time': datetime.utcnow().isoformat(),
                    'feed_url': parser.feed_url,
                    'form_type': form_type
                }
                SECWebSocketService.emit_sec_processing_stats(stats)

            return {
                'form_type': form_type,
                'processed_items': processed_items,
                'new_items_count': new_items_count,
                'success': True
            }
        except Exception as e:
            logger.error(f"Error processing form type {form_type}: {e}")
            return {
                'form_type': form_type,
                'processed_items': [],
                'new_items_count': 0,
                'success': False,
                'error': str(e)
            }

    def process_feed(self):
        try:
            # Decide which form types to process (single provided or all configured)
            form_types_to_process = [
                self.form_type] if self.form_type else self.parser.form_types

            all_processed_items = []
            total_new_items = 0

            # Use ThreadPoolExecutor with 5 workers to process form types in parallel
            with ThreadPoolExecutor(max_workers=8) as executor:
                # Submit all form types for processing
                future_to_form_type = {
                    executor.submit(self._process_single_form_type, ft): ft
                    for ft in form_types_to_process
                }

                # Collect results as they complete
                for future in as_completed(future_to_form_type):
                    form_type = future_to_form_type[future]
                    try:
                        result = future.result()
                        if result['success']:
                            all_processed_items.extend(
                                result['processed_items'])
                            total_new_items += result['new_items_count']
                            print(
                                f"Completed processing form type {form_type}: {result['new_items_count']} new items")
                        else:
                            print(
                                f"Failed to process form type {form_type}: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        logger.error(
                            f"Exception occurred while processing form type {form_type}: {e}")
                        print(f"Error processing form type {form_type}: {e}")

            return {
                'success': True,
                'message': f'Processed {len(all_processed_items)} items across {len(form_types_to_process)} form types, {total_new_items} new',
                'total_items': len(all_processed_items),
                'new_items': total_new_items
            }
        except Exception as e:
            logger.error(f"Error processing SEC feed: {e}")
            return {'success': False, 'error': str(e)}

    def save_filing(self, item_data):
        try:
            # Check if filing already exists
            accession_number = item_data.get('accession_number')
            if not accession_number:
                logger.warning("Cannot save filing without accession_number")
                return False

            existing = SECFiling.objects(
                accession_number=accession_number).first()
            if existing:
                return False

            # Analyze document with GPT before saving
            # For 8-K with EX-2.1, analyze using GPT
            if item_data.get('form_type') == '8-K' and item_data.get('has_ex21'):
                # Find EX-2.1 file
                ex21_files = [
                    file for file in item_data.get('xbrl_files', [])
                    if ('EX-2.1' in file.get('type', '') or 'EX-2.1' in file.get('description', ''))
                    and file.get('url', '').endswith('.htm')
                ]

                if ex21_files:
                    # Update item_data to match expected format
                    item_data['has_htm_files'] = True
                    # Ensure xbrl_files have correct type field
                    for file in ex21_files:
                        if 'EX-2.1' not in file.get('type', ''):
                            file['type'] = 'EX-2.1'

                    logger.info(
                        f"🔍 Analyzing 8-K document for: {item_data.get('company_name')}")
                    item_data = self.document_analyzer.analyze_filing(
                        item_data)

                    # Log the analysis result
                    if item_data.get('is_new_deal') is True:
                        logger.info(
                            f"✅ NEW DEAL detected: {item_data.get('company_name')}")
                    elif item_data.get('is_new_deal') is False:
                        logger.info(
                            f"📝 AMENDMENT detected: {item_data.get('company_name')}")
                    else:
                        logger.info(
                            f"❓ Analysis inconclusive: {item_data.get('company_name')}")
                else:
                    logger.info(
                        f"⚠️ 8-K filing has EX-2.1 flag but no HTM file found: {item_data.get('company_name')}")
                    item_data['is_new_deal'] = None
                    item_data['following'] = False
            elif item_data.get('form_type') and item_data.get('form_type').startswith(("DEFM14A",
                                                                                       "DEFM14C", "PREM14A", "PREM14C", "S-4", "F-4")):
                # Normalize form_type for comparison (handle cases like "DEF 14A - O")
                normalized_form_type = item_data.get(
                    'form_type').split(' - ')[0].strip()
                normalized_form_type = re.sub(
                    r'-[A-Z]$', '', normalized_form_type).strip()

                # Update form_type to normalized version
                item_data['form_type'] = normalized_form_type

                # Analyze DEF 14A/PRE 14A documents for document kind detection
                logger.info(
                    f"🔍 Analyzing {normalized_form_type} document for: {item_data.get('company_name')}")
                item_data = self.document_analyzer.analyze_def14a_filing(
                    item_data)

                # Log the analysis result
                if item_data.get('document_kind'):
                    logger.info(
                        f"📋 Document kind detected: {item_data.get('document_kind')} for {item_data.get('company_name')}")
                else:
                    logger.info(
                        f"❓ Document kind analysis inconclusive: {item_data.get('company_name')}")
            else:
                # For other filings, set default values
                item_data['is_new_deal'] = None
                item_data['following'] = False

            # Convert string datetime back to datetime object for MongoDB
            if item_data.get('acceptance_datetime_utc') and isinstance(item_data['acceptance_datetime_utc'], str):
                try:
                    from datetime import datetime
                    item_data['acceptance_datetime_utc'] = datetime.fromisoformat(
                        item_data['acceptance_datetime_utc'].replace('Z', '+00:00'))
                except Exception as e:
                    print(
                        f"Error converting acceptance_datetime_utc back to datetime: {e}")
                    item_data['acceptance_datetime_utc'] = None

            # Convert filing_date string to datetime object if needed
            if item_data.get('filing_date'):
                if isinstance(item_data['filing_date'], str):
                    try:
                        from datetime import datetime
                        # Try different date formats
                        for date_format in ['%Y-%m-%d', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S']:
                            try:
                                item_data['filing_date'] = datetime.strptime(
                                    item_data['filing_date'], date_format)
                                break
                            except ValueError:
                                continue
                        else:
                            # If all formats fail, set to None
                            print(
                                f"Could not parse filing_date: {item_data['filing_date']}")
                            item_data['filing_date'] = None
                    except Exception as e:
                        print(
                            f"Error converting filing_date back to datetime: {e}")
                        item_data['filing_date'] = None
                # If it's already a datetime object, keep it as is

            # Ensure has_htm_files is set correctly based on has_ex21 or existing has_htm_files
            if item_data.get('has_ex21'):
                item_data['has_htm_files'] = True
            elif 'has_htm_files' not in item_data:
                item_data['has_htm_files'] = False

            # Normalize guid to a valid URL; fall back to link or blank if invalid
            guid = item_data.get('guid', '') or ''
            link = item_data.get('link', '') or ''
            if guid.startswith('urn:tag:sec.gov'):
                # Use link when urn scheme is not acceptable for URLField
                item_data['guid'] = link or ''

            # Truncate description to fit model max_length=50
            desc = item_data.get('description')
            if desc:
                item_data['description'] = desc[:50]

            # Drop fields that are not part of the SECFiling model schema
            allowed_fields = {
                'title', 'link', 'guid', 'description', 'pubDate',
                'enclosure_url', 'enclosure_length', 'enclosure_type',
                'company_name', 'form_type', 'filing_date', 'cik_number',
                'accession_number', 'file_number', 'acceptance_datetime_utc',
                'period', 'fiscal_year_end', 'assigned_sic', 'xbrl_files',
                'has_htm_files', 'processed', 'is_new_deal', 'document_kind',
                'following', 'following_status', 'created_at', 'updated_at',
                'company_details'
            }
            item_data = {k: v for k, v in item_data.items()
                         if k in allowed_fields}

            # Only save if document_kind is "Definitive Merger Agreement"
            document_kind = item_data.get('document_kind')
            # if item_data.get('form_type') == '8-K' and document_kind != "Definitive Merger Agreement":
            #     logger.info(
            #         f"⏭️  Skipping record - document_kind is '{document_kind}' (not 'Definitive Merger Agreement') for: {item_data.get('company_name')}")
            #     return False

            # Re-check for existing just before save (handles race with concurrent runs)
            existing = SECFiling.objects(
                accession_number=accession_number).first()
            if existing:
                logger.info(
                    f"⏭️ Skipping duplicate save (existing): {accession_number}")
                return False

            # Create and save the filing
            filing = SECFiling(**item_data)
            try:
                filing.save()
            except NotUniqueError:
                logger.warning(
                    f"⏭️ Duplicate accession_number (race): {accession_number} - skipping save and email")
                return False
            except Exception as e:
                if 'E11000' in str(e) or 'duplicate' in str(e).lower():
                    logger.warning(
                        f"⏭️ Duplicate accession_number (race): {accession_number} - skipping save and email")
                    return False
                raise

            logger.info(
                f"💾 Saved filing: {item_data.get('company_name')} - {item_data.get('accession_number')}")

            # Prepare filing data for WebSocket emission
            def safe_isoformat(value):
                """Safely convert datetime to ISO format string"""
                if value and hasattr(value, 'isoformat'):
                    return value.isoformat()
                elif isinstance(value, str):
                    return value
                else:
                    return None

            filing_data = {
                '_id': str(filing._id),
                'company_name': filing.company_name,
                'form_type': filing.form_type,
                'accession_number': filing.accession_number,
                'title': filing.title,
                'link': filing.link,
                'description': filing.description,
                'cik_number': filing.cik_number,
                'filing_date': safe_isoformat(filing.filing_date),
                'acceptance_datetime_utc': safe_isoformat(filing.acceptance_datetime_utc),
                'has_htm_files': filing.has_htm_files,
                'is_new_deal': filing.is_new_deal,
                'following': filing.following,
                'following_status': filing.following_status,
                'xbrl_files': filing.xbrl_files,
                'created_at': safe_isoformat(filing.created_at),
                'updated_at': safe_isoformat(filing.updated_at),
                'document_kind': filing.document_kind,
                'company_details': filing.company_details if filing.company_details else None
            }

            # Emit WebSocket event for new SEC filing
            SECWebSocketService.emit_new_sec_filing(filing_data)

            # If GPT analysis was performed, emit analysis result
            if item_data.get('is_new_deal') is not None:
                if item_data.get('is_new_deal') is True:
                    analysis_result = 'new_deal'
                elif item_data.get('is_new_deal') is False:
                    analysis_result = 'amendment'
                else:
                    analysis_result = 'inconclusive'

                SECWebSocketService.emit_sec_analysis_complete(
                    filing_data, analysis_result)

            # Email notification logic:
            # - 8-K with EX-2.1 + Definitive Merger Agreement: Send email directly
            # - 8-K (any), 10-Q, 10-K: Send email only if filing CIK matches a deal's target cik
            # - Other form types: Send email only if filing CIK matches a deal's cik or acquirer_cik
            form_type = item_data.get('form_type', '')
            cik_number = item_data.get('cik_number', '')

            logger.info(
                f"🔍 Checking email notification - form_type: {form_type}, cik_number: {cik_number}")
            print(
                f"🔍 Checking email notification - form_type: {form_type}, cik_number: {cik_number}")

            should_send_email = False
            matched_deal = None

            if form_type == '8-K' and filing.has_htm_files and item_data.get('document_kind') == 'Definitive Merger Agreement':
                # For 8-K filings, send email directly
                should_send_email = True
                logger.info(f"✅ Form type is 8-K - will send email")
                print(f"✅ Form type is 8-K - will send email")

            elif (form_type == "8-K" or form_type == '8-K/A' or form_type == '10-Q' or form_type == '10-K') and cik_number:
                # 8-K, 8-K/A (no doc processing), 10-Q, 10-K: send email only if filing CIK matches a deal's target cik
                cik_normalized = str(cik_number).zfill(
                    10) if cik_number else ''
                logger.info(f"📋 Normalized CIK: {cik_normalized}")
                print(f"📋 Normalized CIK: {cik_normalized}")

                # Check if CIK matches any deal's target cik in MongoDB
                try:
                    matched_deal = ProcessingJob.objects(
                        cik=cik_normalized).first()

                    if matched_deal:
                        should_send_email = True
                    else:
                        logger.info(
                            f"ℹ️ No deal match found for CIK {cik_normalized}")
                        print(
                            f"ℹ️ No deal match found for CIK {cik_normalized}")
                except Exception as e:
                    logger.error(
                        f"Error checking deals collection for CIK {cik_normalized}: {e}")
                    print(
                        f"Error checking deals collection for CIK {cik_normalized}: {e}")

            elif form_type != '8-K' and cik_number:
                # For non-8-K filings, check MongoDB deals (ProcessingJob): cik = target CIK, acquirer_cik = acquirer CIK
                logger.info(
                    f"✅ Form type is not 8-K ({form_type}) and CIK exists ({cik_number}), checking deals collection...")
                print(
                    f"✅ Form type is not 8-K ({form_type}) and CIK exists ({cik_number}), checking deals collection...")

                # Normalize CIK (pad to 10 digits for comparison)
                cik_normalized = str(cik_number).zfill(
                    10) if cik_number else ''
                logger.info(f"📋 Normalized CIK: {cik_normalized}")
                print(f"📋 Normalized CIK: {cik_normalized}")

                # Check if CIK matches any deal in MongoDB (target cik or acquirer_cik)
                try:
                    matched_deal = ProcessingJob.objects(
                        cik=cik_normalized).first()
                    if not matched_deal:
                        matched_deal = ProcessingJob.objects(
                            acquirer_cik=cik_normalized).first()

                    if matched_deal:
                        should_send_email = True
                        logger.info(
                            f"✅ MATCH FOUND! Deal: {getattr(matched_deal, 'target_name', 'Unknown')} (ID: {matched_deal.id})")
                        print(
                            f"✅ MATCH FOUND! Deal: {getattr(matched_deal, 'target_name', 'Unknown')} (ID: {matched_deal.id})")
                    else:
                        logger.info(
                            f"ℹ️ No deal match found for CIK {cik_normalized}")
                        print(
                            f"ℹ️ No deal match found for CIK {cik_normalized}")
                except Exception as e:
                    logger.error(
                        f"Error checking deals collection for CIK {cik_normalized}: {e}")
                    print(
                        f"Error checking deals collection for CIK {cik_normalized}: {e}")
            else:
                if not cik_number:
                    logger.info(f"ℹ️ Skipping email check - no CIK number")
                    print(f"ℹ️ Skipping email check - no CIK number")

            # Send email if conditions are met
            if should_send_email:
                try:
                    if matched_deal:
                        logger.info(
                            f"📧 Preparing to send email for matched deal: {getattr(matched_deal, 'target_name', 'Unknown')}")
                        print(
                            f"📧 Preparing to send email for matched deal: {getattr(matched_deal, 'target_name', 'Unknown')}")
                    else:
                        logger.info(
                            f"📧 Preparing to send email for 8-K filing: {item_data.get('company_name', 'Unknown')}")
                        print(
                            f"📧 Preparing to send email for 8-K filing: {item_data.get('company_name', 'Unknown')}")

                    # Generate email HTML
                    subject, html_email = generate_filing_email_html(
                        item_data, item_data.get('xbrl_files', []))
                    logger.info(f"📝 Generated email subject: {subject}")
                    print(f"📝 Generated email subject: {subject}")

                    # Send email via n8n webhook
                    webhook_url = "https://n8n-xwx1.onrender.com/webhook/3ff1b0ea-7114-4dda-940e-95ce81e08017"
                    logger.info(
                        f"📤 Sending email via n8n webhook: {webhook_url}")
                    print(
                        f"📤 Sending email via n8n webhook: {webhook_url}")

                    # Prepare payload for n8n webhook
                    payload = {
                        'subject': subject,
                        'html': html_email,
                        'company_name': item_data.get('company_name', 'Unknown Company'),
                        'accession_number': item_data.get('accession_number', 'N/A'),
                        'form_type': item_data.get('form_type', 'N/A'),
                        'filing_url': item_data.get('link', '')
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
                            f"✅ Email sent successfully via n8n webhook! Status: {response.status_code}")
                        print(
                            f"✅ Email sent successfully via n8n webhook! Status: {response.status_code}")
                        logger.info(f"📧 Response: {response.text[:200]}")
                        print(f"📧 Response: {response.text[:200]}")
                    except requests.exceptions.RequestException as e:
                        logger.error(
                            f"❌ Error sending email via n8n webhook: {e}")
                        print(
                            f"❌ Error sending email via n8n webhook: {e}")
                        if hasattr(e, 'response') and e.response is not None:
                            logger.error(
                                f"❌ Response status: {e.response.status_code}, Response body: {e.response.text[:200]}")
                            print(
                                f"❌ Response status: {e.response.status_code}, Response body: {e.response.text[:200]}")
                        raise

                    # After email is sent, process 8-K document if it's an 8-K filing with EX-2.1
                    # Check if EX-2.1 exists in xbrl_files (more reliable than checking has_ex21 which might not be preserved)
                    has_ex21_in_files = False
                    xbrl_files = item_data.get('xbrl_files', [])
                    for file in xbrl_files:
                        doc_type = file.get('type', '')
                        doc_url = file.get('url', '')
                        if ('EX-2.1' in doc_type or 'EX-2.1' in file.get('description', '')) and doc_url.endswith('.htm'):
                            has_ex21_in_files = True
                            break

                    # Also check item_data and filing object
                    has_ex21_from_data = item_data.get('has_ex21', False)
                    company_details = item_data.get('company_details', None)
                    has_ex21_from_filing = (
                        hasattr(filing, 'has_htm_files') and filing.has_htm_files) if filing else False
                    has_ex21 = has_ex21_in_files or has_ex21_from_data or has_ex21_from_filing

                    logger.info(
                        f"🔍 Checking 8-K processing condition - form_type: {form_type}, has_ex21_in_files: {has_ex21_in_files}, has_ex21_from_data: {has_ex21_from_data}, has_ex21_from_filing: {has_ex21_from_filing}, final has_ex21: {has_ex21}")
                    print(
                        f"🔍 Checking 8-K processing condition - form_type: {form_type}, has_ex21_in_files: {has_ex21_in_files}, has_ex21_from_data: {has_ex21_from_data}, has_ex21_from_filing: {has_ex21_from_filing}, final has_ex21: {has_ex21}")

                    if (form_type == '8-K' and has_ex21 and item_data.get('document_kind') == 'Definitive Merger Agreement' and company_details
                        and company_details.get('is_target_market_cap_greater_than_100m')
                            and company_details.get('is_target_us_listed')):

                        try:
                            logger.info(
                                f"🚀 Starting 8-K document processing after email notification for: {item_data.get('company_name')}")
                            print(
                                f"🚀 Starting 8-K document processing after email notification for: {item_data.get('company_name')}")

                            # Find EX-2.1 HTM file URL
                            ex21_url = None
                            xbrl_files = item_data.get('xbrl_files', [])

                            for file in xbrl_files:
                                doc_type = file.get('type', '')
                                doc_url = file.get('url', '')
                                if ('EX-2.1' in doc_type or 'EX-2.1' in file.get('description', '')) and doc_url.endswith('.htm'):
                                    # Build full URL if needed
                                    if not doc_url.startswith('http://') and not doc_url.startswith('https://'):
                                        ex21_url = f"https://www.sec.gov{doc_url}"
                                    else:
                                        ex21_url = doc_url
                                    break

                            if ex21_url:
                                # Get sec_filling_id (MongoDB ObjectId as string)
                                sec_filling_id = str(
                                    filing._id) if filing else None

                                if sec_filling_id:
                                    # Get filing date as string
                                    filing_date_str = None
                                    filing_date = item_data.get('filing_date')
                                    if filing_date:
                                        if isinstance(filing_date, datetime):
                                            filing_date_str = filing_date
                                        else:
                                            try:
                                                filing_date_str = datetime.strptime(
                                                    str(filing_date), '%Y-%m-%d')
                                            except:
                                                filing_date_str = None

                                    # Call helper function to process 8-K document
                                    result = process_8k_document_helper(
                                        cik_number=cik_number,
                                        company_name=item_data.get(
                                            'company_name', ''),
                                        sec_filing_id=sec_filling_id,
                                        filing_date=filing_date_str,
                                        form_type=form_type,
                                        ex21_url=ex21_url,
                                        item_data=item_data,
                                        company_details=company_details
                                    )

                                    if result:
                                        logger.info(
                                            f"✅ Successfully started 8-K document processing: {result.get('message', 'Processing started')}")
                                        print(
                                            f"✅ Successfully started 8-K document processing: {result.get('message', 'Processing started')}")
                                    else:
                                        logger.error(
                                            f"❌ Failed to start 8-K document processing for: {item_data.get('company_name')}")
                                        print(
                                            f"❌ Failed to start 8-K document processing for: {item_data.get('company_name')}")
                                else:
                                    logger.warning(
                                        f"⚠️ Cannot process 8-K document: sec_filling_id not found")
                                    print(
                                        f"⚠️ Cannot process 8-K document: sec_filling_id not found")
                            else:
                                logger.warning(
                                    f"⚠️ No EX-2.1 HTM file found in xbrl_files for 8-K filing: {item_data.get('company_name')}")
                                print(
                                    f"⚠️ No EX-2.1 HTM file found in xbrl_files for 8-K filing: {item_data.get('company_name')}")
                        except Exception as e:
                            logger.error(
                                f"❌ Error processing 8-K document after email: {str(e)}", exc_info=True)
                            print(
                                f"❌ Error processing 8-K document after email: {str(e)}")
                            import traceback
                            print(traceback.format_exc())

                except Exception as e:
                    logger.error(
                        f"❌ Error sending email notification: {e}", exc_info=True)
                    print(f"❌ Error sending email notification: {e}")
                    import traceback
                    print(traceback.format_exc())

            # After saving filing, check if CIK matches deals collection or deals collection
            # If match found and form_type is any from given array, process the document
            cik_number = item_data.get('cik_number', '')
            form_type = item_data.get('form_type', '')

            if cik_number and form_type in ["DEFM14A",
                                            "DEFM14C", "PREM14A", "PREM14C", "S-4", "F-4"]:
                logger.info(
                    f"🔍 Checking for CIK match in deals collection for: {item_data.get('company_name')}")
                print(
                    f"🔍 Checking for CIK match in deals collection for: {item_data.get('company_name')}")
                # Normalize CIK for comparison (pad to 10 digits)
                cik_normalized = str(cik_number).zfill(
                    10) if cik_number else ''

                matched_deal = None
                deal_id = None

                # Check if CIK matches deals collection (cik = target CIK, acquirer_cik = acquirer CIK)
                if cik_normalized:
                    try:
                        matched_deal = ProcessingJob.objects(
                            cik=cik_normalized
                        ).first()

                        if not matched_deal:
                            matched_deal = ProcessingJob.objects(
                                acquirer_cik=cik_normalized
                            ).first()

                        if matched_deal:
                            deal_id = str(matched_deal.id)
                            logger.info(
                                f"✅ CIK {cik_normalized} matched with deal collection: {deal_id}")
                    except Exception as e:
                        logger.error(
                            f"Error checking deals collection for CIK {cik_normalized}: {str(e)}")

                # If match found, process the SEC document
                if matched_deal:
                    # Find the HTM file URL from xbrl_files for given form type
                    proxy_sec_url = None
                    xbrl_files = item_data.get('xbrl_files', [])

                    for file in xbrl_files:
                        doc_type = file.get('type', '')
                        doc_url = file.get('url', '')
                        if (doc_type in ["DEFM14A",
                                         "DEFM14C", "PREM14A", "PREM14C", "S-4", "F-4"]) and doc_url.endswith('.htm'):
                            # Build full URL if needed
                            if not doc_url.startswith('http://') and not doc_url.startswith('https://'):
                                proxy_sec_url = f"https://www.sec.gov{doc_url}"
                            else:
                                proxy_sec_url = doc_url
                            break

                    if proxy_sec_url:
                        try:
                            # Get filing date as string
                            filing_date_str = None
                            filing_date = item_data.get('filing_date')
                            if filing_date:
                                if isinstance(filing_date, datetime):
                                    filing_date_str = filing_date.strftime(
                                        '%Y-%m-%d')
                                else:
                                    filing_date_str = str(filing_date)

                            # Get sec_filling_id (MongoDB ObjectId as string)
                            sec_filling_id = str(
                                filing._id) if filing else None

                            if sec_filling_id:
                                logger.info(
                                    f"🚀 Starting proxy document processing for CIK {cik_normalized} (matched deal: {deal_id})")

                                # Call the helper function to process the document
                                result = process_sec_document_helper(
                                    cik_number=cik_number,
                                    company_name=item_data.get(
                                        'company_name', ''),
                                    sec_filling_id=sec_filling_id,
                                    filing_date=filing_date_str or '',
                                    form_type=form_type,
                                    proxy_sec_url=proxy_sec_url,
                                    deal_id=deal_id
                                )

                                if result:
                                    logger.info(
                                        f"✅ Successfully started proxy document processing: {result.get('proxy_document_id')}")
                                else:
                                    logger.error(
                                        f"❌ Failed to start proxy document processing for CIK {cik_normalized}")
                            else:
                                logger.warning(
                                    f"⚠️ Cannot process document: sec_filling_id not found")
                        except Exception as e:
                            logger.error(
                                f"❌ Error processing SEC document for CIK {cik_normalized}: {str(e)}", exc_info=True)
                    else:
                        logger.warning(
                            f"⚠️ No HTM file found in xbrl_files for DEF 14A/PRE 14A filing: {item_data.get('company_name')}")

            return True

        except Exception as e:
            logger.error(f"Error saving filing: {e}")
            return False
