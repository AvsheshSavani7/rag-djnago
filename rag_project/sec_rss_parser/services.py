from coreschema import Null
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import time
import logging
import asyncio
import re
import os
import json
import tempfile
import threading
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
from django.core.mail import send_mail, EmailMultiAlternatives
from django.conf import settings
from bson import ObjectId
from mongoengine.errors import NotUniqueError
from .s3_upload_utils import build_parsed_jsons_s3_key, upload_json_file_to_s3
from .docx_parser import parse_dma_summary_docx
from .models import (
    SECFiling,
    SECFeedStatus,
    LastCronJob,
    AccessionLookedUp,
    EightKSummary,
    Ex99_1Summary,
    TenKTenQSummary,
    DealDmaSummary,
)
from .document_analyzer import SECDocumentAnalyzer
from .websocket_service import SECWebSocketService
from .email_templates import (
    generate_filing_email_html,
    generate_ex99_1_merger_email_html,
    generate_8k_summary_email_html,
    generate_sec_filings_email_html,
    generate_8k_99_1_summary_email_html,
    generate_parsing_error_email_html,
    generate_parsing_success_email_html,
)
from .sec_Last_Year import print_filings as fetch_sec_filings

from .Eight_k_summary import summarize_8k_filing
from document_processor.models import ProcessingJob, DealSchemaResults
from document_processor.services import DocumentProcessingService, SummaryGenerationService
from proxy_processor.views import process_sec_document_helper
from node_proxy.utils import call_node_api
from node_proxy.views import AnnouncementWithUrlView

try:
    import openai
except ImportError:
    openai = None

logger = logging.getLogger(__name__)

# Constants
N8N_WEBHOOK_URL_8K_SUMMARY = "https://n8n-xwx1.onrender.com/webhook/b3007d21-6845-47b5-aece-7b26583758bc"  # to avs/kd/josh
N8N_WEBHOOK_URL_8K_SUMMARY_L123 = os.environ.get(
    "N8N_WEBHOOK_SEND_TO_ALL", "https://n8n-xwx1.onrender.com/webhook/3ff1b0ea-7114-4dda-940e-95ce81e08017")  # to all
N8N_WEBHOOK_URL_FILING = "https://n8n-xwx1.onrender.com/webhook/3ff1b0ea-7114-4dda-940e-95ce81e08017"  # to all
N8N_WEBHOOK_URL_FOR_TESTING = "https://n8n-xwx1.onrender.com/webhook/80830c6d-ff5b-45e3-9ef3-a061db1fbf0c"  # only avshesh
N8N_WEBHOOK_URL_PARSING_ERROR = "https://n8n-xwx1.onrender.com/webhook/80830c6d-ff5b-45e3-9ef3-a061db1fbf0c"
N8N_WEBHOOK_URL_PARSING_SUCCESS = "https://n8n-xwx1.onrender.com/webhook/80830c6d-ff5b-45e3-9ef3-a061db1fbf0c"

SEC_BASE_URL = "https://www.sec.gov"
ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
CIK_LENGTH = 10
MAX_DESCRIPTION_LENGTH = 50

FORM_TYPES = ["8-k", "DEFM14A", "DEFM14C", "PREM14A", "PREM14C", "S-4",
              "S-4/A", "F-4", "F-4/A", "SC 14D9", "SC 14D9/A", "10-Q", "10-K"]
PROXY_FORM_TYPES = ["DEFM14A", "DEFM14C", "PREM14A", "PREM14C", "S-4", "F-4"]
PERIODIC_FORM_TYPES = ["8-K", "8-K/A", "10-Q", "10-K"]

ALLOW_EMAIL_TO_CLIENT_FORM = ["DEFM14A", "DEFM14C", "PREM14A",
                              "PREM14C", "S-4", "F-4", "S-4/A", "F-4/A", "10-K", "10-Q", "10-K/A", "425"]

# Only consider deals with status Open or Unknown (or null/not set) when matching by CIK
DEAL_STATUS_OPEN_OR_UNKNOWN = ["Open", "Unknown"]


ALLOWED_FILING_FIELDS = {
    'title', 'link', 'guid', 'description', 'pubDate',
    'enclosure_url', 'enclosure_length', 'enclosure_type',
    'company_name', 'form_type', 'filing_date', 'cik_number',
    'accession_number', 'file_number', 'acceptance_datetime_utc',
    'period', 'fiscal_year_end', 'assigned_sic', 'xbrl_files',
    'has_htm_files', 'processed', 'is_new_deal', 'document_kind',
    'following', 'following_status', 'created_at', 'updated_at',
    'company_details'
}


# Helper Functions
def normalize_cik(cik_number):
    """Normalize CIK by padding to 10 digits"""
    return str(cik_number).zfill(CIK_LENGTH) if cik_number else ''


def log_and_print(message, level='info'):
    """Log and print a message"""
    log_func = getattr(logger, level, logger.info)
    log_func(message)
    print(message)


def safe_isoformat(value):
    """Safely convert datetime to ISO format string"""
    if value and hasattr(value, 'isoformat'):
        return value.isoformat()
    elif isinstance(value, str):
        return value
    return None


def parse_filing_date(filing_date_str):
    """Parse filing date from string to datetime object"""
    if not filing_date_str:
        return None

    if isinstance(filing_date_str, datetime):
        return filing_date_str

    for date_format in ['%Y-%m-%d', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S']:
        try:
            return datetime.strptime(filing_date_str, date_format)
        except ValueError:
            continue

    log_and_print(f"Could not parse filing_date: {filing_date_str}", 'warning')
    return None


def _extract_announce_date_with_llm(company_details_str):
    """
    Call LLM with web search to extract M&A deal announcement date from company details.
    company_details_str: plain text with company names, CIKs, target/acquirer if available.
    Returns datetime.date or None. On any failure or if not found, returns None.
    """
    if not openai or not os.environ.get("OPENAI_API_KEY"):
        return None
    prompt = f"""You are a financial research assistant. Use web search to find the **deal announcement date** (the date the merger/acquisition was first publicly announced) for this company/deal.

Company/Deal details:
{company_details_str}

Return **only** valid JSON in this exact format:
{{ "announce_date": "YYYY-MM-DD" }}

If you cannot find a reliable announcement date, return: {{ "announce_date": null }}

Use only the date of the initial public announcement (e.g. press release, 8-K filing date of the deal announcement), not closing or other dates.
"""
    try:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.responses.create(
            model="gpt-5",
            tools=[{"type": "web_search"}],
            input=prompt,
            reasoning={"effort": "low"},
        )
        result_text = None
        for item in response.output:
            if getattr(item, "type", None) == "message" and hasattr(item, "content"):
                for content_item in item.content:
                    if getattr(content_item, "type", None) == "output_text":
                        result_text = getattr(content_item, "text", None)
                        break
            if result_text:
                break
        if not result_text:
            return None
        result_text = result_text.strip()
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            result_text = "\n".join(json_lines).strip()
        start, end = result_text.find("{"), result_text.rfind("}")
        if start != -1 and end != -1:
            result_text = result_text[start: end + 1]
        data = json.loads(result_text)
        raw = data.get("announce_date")
        if not raw:
            return None
        return parse_filing_date(raw)
    except Exception as e:
        log_and_print(f"LLM announce date extraction failed: {e}", "warning")
        return None


def build_full_sec_url(url):
    """Build full SEC URL if relative path provided"""
    if not url:
        return None
    if url.startswith('http://') or url.startswith('https://'):
        return url
    return f"{SEC_BASE_URL}{url}"


def find_file_by_type(xbrl_files, file_types, extension=(".htm", ".html", ".xml")):
    """Find file in xbrl_files by type and extension. extension can be a str or iterable of str."""
    exts = (extension,) if isinstance(extension, str) else tuple(extension)
    for file in xbrl_files:
        doc_type = file.get("type", "")
        description = file.get("description", "")
        doc_url = file.get("url", "")

        for file_type in file_types if isinstance(file_types, list) else [file_types]:
            if (file_type in doc_type or file_type in description) and any(
                doc_url.endswith(e) for e in exts
            ):
                return file
    return None


def extract_accession_from_guid(guid):
    """Extract accession number from GUID"""
    if not guid:
        return None
    match = re.search(r'accession-number=([\d-]+)', guid)
    return match.group(1) if match else None


def send_webhook_notification(webhook_url, payload, notification_type="notification"):
    """Send notification via webhook"""
    try:
        log_and_print(
            f"📤 Sending {notification_type} via webhook: {webhook_url}")

        response = requests.post(
            webhook_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        response.raise_for_status()

        log_and_print(
            f"✅ {notification_type} sent successfully! Status: {response.status_code}")
        log_and_print(f"📧 Response: {response.text[:200]}")
        return True

    except requests.exceptions.RequestException as e:
        log_and_print(
            f"❌ Error sending {notification_type} via webhook: {e}", 'error')
        if hasattr(e, 'response') and e.response is not None:
            log_and_print(
                f"❌ Response status: {e.response.status_code}, Response body: {e.response.text[:200]}", 'error')
        raise

# Below fucntion is still use in new fetch form by cik flow


def send_summary_email_via_webhook(summary_doc_url, company_name, form_type, cik_number, sec_url, accession_number, summary_kind: str, l1_headline: str = None, l2_brief: str = None, l3_detailed: str = None, ticker: str = None, filing_date=None, matched_cik_label: str = None, form_affects_deal: bool = None):
    """Generate 8-K/EX-99.1 summary email HTML and send via N8N testing webhook (includes .docx URL and L1 headline so user can see content without opening doc). Subject uses ticker if provided, else company_name. matched_cik_label is '(target)' or '(acquirer)' to show beside company name; form_affects_deal is set only for acquirer filings (LLM)."""
    try:
        subject, html_email = generate_8k_99_1_summary_email_html(
            company_name=company_name,
            form_type=form_type,
            summary_doc_url=summary_doc_url,
            cik_number=cik_number,
            sec_url=sec_url or "",
            accession_number=accession_number or "",
            summary_kind=summary_kind,
            l1_headline=l1_headline,
            l2_brief=l2_brief,
            l3_detailed=l3_detailed,
            ticker=ticker,
            filing_date=filing_date,
            matched_cik_label=matched_cik_label,
            form_affects_deal=form_affects_deal,
        )
        payload = {
            "subject": subject,
            "html": html_email,
            "company_name": company_name,
            "form_type": form_type,
            "summary_doc_url": summary_doc_url,
            "accession_number": accession_number,
            "cik_number": cik_number,
            "sec_url": sec_url,
        }
        if matched_cik_label is not None:
            payload["matched_cik_label"] = matched_cik_label
        if form_affects_deal is not None:
            payload["form_affects_deal"] = form_affects_deal

        webhook_url = (
            N8N_WEBHOOK_URL_8K_SUMMARY_L123
            if summary_kind in ALLOW_EMAIL_TO_CLIENT_FORM
            else N8N_WEBHOOK_URL_8K_SUMMARY
        )
        send_webhook_notification(
            webhook_url, payload, f"{summary_kind} summary email"
        )
    except Exception as e:
        log_and_print(
            f"❌ Failed to send {summary_kind} summary email via webhook: {e}", "error")
        raise


# Main Functions
def send_8k_summary_email(deal_id, company_name, form_type, cik_number, sec_url, accession_number, summary_kind: str):
    """Send email notification with 8-K summary document URL."""
    try:
        # Get DMA summary record to retrieve summary_docx_url
        try:
            object_id = ObjectId(deal_id)
            dma_summary = DealDmaSummary.objects(deal_id=object_id).first()
        except Exception as e:
            log_and_print(
                f"Error retrieving deal_dma_summary for {deal_id}: {e}", 'error')
            return

        if not dma_summary or not dma_summary.summary_docx_url:
            log_and_print(
                f"No summary document URL available for deal_dma_summary {deal_id}", 'warning')
            return

        log_and_print(
            f"Preparing to send 8-K summary email for: {company_name}")

        # Parse the DOCX to extract concise sections for inline display
        concise_sections = None
        try:
            parsed = parse_dma_summary_docx(dma_summary.summary_docx_url)
            concise_sections = parsed.get("concise_sections")
        except Exception as parse_err:
            log_and_print(
                f"Warning: could not parse DOCX for inline summary: {parse_err}", 'warning')

        # Generate email HTML
        subject, html_email = generate_8k_summary_email_html(
            company_name=company_name,
            form_type=form_type,
            summary_doc_url=dma_summary.summary_docx_url,
            cik_number=cik_number,
            sec_url=sec_url,
            accession_number=accession_number,
            summary_kind=summary_kind,
            concise_sections=concise_sections,
        )
        log_and_print(f"Generated email subject: {subject}")

        # Prepare payload for n8n webhook
        payload = {
            'subject': subject,
            'html': html_email,
            'company_name': company_name,
            'form_type': form_type,
            'summary_doc_url': dma_summary.summary_docx_url,
            'deal_id': deal_id
        }

        # Send via webhook
        send_webhook_notification(
            N8N_WEBHOOK_URL_8K_SUMMARY, payload, "8-K summary email")

    except Exception as e:
        log_and_print(
            f"❌ Error sending 8-K summary email notification: {e}", 'error')


def generate_8k_summary_async(deal_id, company_name, form_type, cik_number, sec_url, accession_number, max_attempts=60, delay_seconds=30):
    """Async function to generate summary for 8-K after processing completes."""
    object_id = None
    try:
        log_and_print(
            f"🔍 Starting summary generation monitoring for deal_id: {deal_id}")

        object_id = ObjectId(deal_id)
        attempts = 0

        while attempts < max_attempts:
            try:
                job = ProcessingJob.objects(id=object_id).first()

                if not job:
                    log_and_print(f"❌ Job {deal_id} not found", 'error')
                    DealDmaSummary.save_or_update(
                        deal_id=object_id,
                        summary_status='FAILED',
                        summary_docx_url=None,
                        summary_using="gpt-5.2-2025-12-11",
                    )
                    return

                # Check if schema_results are available (from dedicated collection)
                schema_record = None
                try:
                    schema_record = DealSchemaResults.objects(
                        deal_id=object_id).first()
                except Exception:
                    schema_record = None

                if schema_record and schema_record.schema_results and job.embedding_status == 'COMPLETED':
                    log_and_print(
                        f"✅ Schema results available for deal_id: {deal_id}, generating summary")

                    # Update summary status to processing
                    job.summary_status = 'PROCESSING'
                    job.save()

                    # Upsert PROCESSING state into dedicated collection.
                    DealDmaSummary.save_or_update(
                        deal_id=object_id,
                        summary_status='PROCESSING',
                        summary_docx_url=None,
                        summary_using="gpt-5.2-2025-12-11",
                    )

                    # Generate summary
                    summary_service = SummaryGenerationService()
                    result = summary_service.generate_summary_engine(
                        deal_id=deal_id,
                        temperature=0,
                        provider='openai',
                        model='gpt-5.2-2025-12-11'
                    )
                    log_and_print(f"Result: {result}")

                    if result:
                        # Update job with summary URL
                        job.summary_docx_url = result
                        job.summary_using = "gpt-5.2-2025-12-11"
                        job.summary_status = 'COMPLETED'
                        job.save()

                        # Upsert COMPLETED state.
                        DealDmaSummary.save_or_update(
                            deal_id=object_id,
                            summary_status='COMPLETED',
                            summary_docx_url=result,
                            summary_using=job.summary_using,
                        )

                        log_and_print(
                            f"✅ Summary generated successfully for deal_id: {deal_id}")

                        # Send email notification
                        try:
                            log_and_print(
                                f"📧 Sending 8-K summary email for: {company_name}")
                            send_8k_summary_email(
                                deal_id=deal_id,
                                company_name=company_name,
                                form_type=form_type,
                                cik_number=cik_number,
                                sec_url=sec_url,
                                accession_number=accession_number,
                                summary_kind="EX-2.1"
                            )
                            log_and_print(
                                f"✅ 8-K summary email sent successfully")
                        except Exception as email_error:
                            log_and_print(
                                f"❌ Error sending 8-K summary email: {str(email_error)}", 'error')

                        # Extract structured data from DMA summary
                        try:
                            log_and_print(
                                f"📊 Extracting structured data from DMA summary for deal_id: {deal_id}")
                            _extract_dma_data(
                                deal_id=deal_id,
                                company_name=company_name,
                                cik_number=cik_number,
                                accession_number=accession_number,
                                dma_summary_docx=result,
                            )
                            log_and_print(
                                f"✅ DMA extraction completed for deal_id: {deal_id}")
                        except Exception as dma_error:
                            log_and_print(
                                f"❌ Error extracting DMA data: {str(dma_error)}", 'error')

                        return
                    else:
                        log_and_print(
                            f"❌ Summary generation returned None for deal_id: {deal_id}", 'error')
                        job.summary_status = 'FAILED'
                        job.error_message = 'Summary generation returned None'
                        job.save()

                        # Upsert FAILED state.
                        DealDmaSummary.save_or_update(
                            deal_id=object_id,
                            summary_status='FAILED',
                            summary_docx_url=None,
                            summary_using=job.summary_using or "gpt-5.2-2025-12-11",
                        )
                        return

            except Exception as e:
                log_and_print(f"❌ Error checking job status: {e}", 'error')
                if 'not found' in str(e).lower() or 'does not exist' in str(e).lower():
                    log_and_print(f"❌ Job {deal_id} not found", 'error')
                    DealDmaSummary.save_or_update(
                        deal_id=object_id,
                        summary_status='FAILED',
                        summary_docx_url=None,
                        summary_using="gpt-5.2-2025-12-11",
                    )
                    return

            # Wait before next attempt
            attempts += 1
            if attempts < max_attempts:
                log_and_print(
                    f"⏳ Waiting for schema_results... (attempt {attempts}/{max_attempts})")
                time.sleep(delay_seconds)

        log_and_print(
            f"⚠️ Timeout waiting for schema_results for deal_id: {deal_id}", 'warning')

    except Exception as e:
        # Ensure we record the FAILED state even if the status-check loop errors.
        if object_id is not None:
            try:
                DealDmaSummary.save_or_update(
                    deal_id=object_id,
                    summary_status='FAILED',
                    summary_docx_url=None,
                    summary_using="gpt-5.2-2025-12-11",
                )
            except Exception:
                # Don't mask the original error with DB issues.
                pass
        log_and_print(f"❌ Error in generate_8k_summary_async: {e}", 'error')


def _extract_dma_data(deal_id, company_name, cik_number, accession_number, dma_summary_docx):
    """
    Extract structured deal data from DMA (EX-2.1) summary using Claude Haiku.
    Saves to fo_dma_extraction collection and sends email.

    This function also updates the corresponding press release extraction record
    (fo_press_release_extraction) with the deal_id, since it was saved earlier
    without deal_id (deal wasn't created yet at that point).

    Args:
        deal_id: Deal ID
        company_name: Company name
        cik_number: CIK number
        accession_number: SEC accession number
        dma_summary_docx: S3 URL of the DMA summary DOCX
    """
    from sec_rss_parser.models import FOPressReleaseExtraction

    try:
        log_and_print(
            f"📊 Starting DMA extraction for deal_id: {deal_id}")

        # Update press release extraction with deal_id (it was saved without deal_id earlier)
        if accession_number and deal_id:
            try:
                pr_record = FOPressReleaseExtraction.objects(
                    accession_number=accession_number
                ).first()
                if pr_record and not pr_record.deal_id:
                    pr_record.deal_id = deal_id
                    pr_record.save()
                    log_and_print(
                        f"✅ Updated press release extraction with deal_id: {deal_id}")
            except Exception as pr_e:
                log_and_print(
                    f"⚠️ Failed to update press release extraction with deal_id: {pr_e}",
                    'warning'
                )

        dma_summary_record = DealDmaSummary.objects(
            deal_id=ObjectId(deal_id)
        ).first()

        dma_summary_id = str(
            dma_summary_record.id) if dma_summary_record else None

        summary_text = None

        if dma_summary_docx:
            try:
                import docx
                response = requests.get(dma_summary_docx, timeout=60)
                if response.status_code == 200:
                    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
                        tmp.write(response.content)
                        tmp_path = tmp.name

                    doc = docx.Document(tmp_path)
                    paragraphs = [
                        p.text for p in doc.paragraphs if p.text.strip()]
                    summary_text = "\n\n".join(paragraphs)

                    os.unlink(tmp_path)

                    log_and_print(
                        f"📄 Extracted {len(paragraphs)} paragraphs from DMA summary DOCX")
                else:
                    log_and_print(
                        f"⚠️ Failed to download DMA summary DOCX: HTTP {response.status_code}",
                        'warning'
                    )
            except Exception as docx_e:
                log_and_print(
                    f"⚠️ Error reading DMA summary DOCX: {docx_e}", 'warning')

        if not summary_text:
            try:
                job = ProcessingJob.objects(id=ObjectId(deal_id)).first()
                if job and hasattr(job, 'summary_text') and job.summary_text:
                    summary_text = job.summary_text
                    log_and_print("📄 Using summary_text from ProcessingJob")
            except Exception:
                pass

        if not summary_text:
            log_and_print(
                f"⚠️ No DMA summary text available for deal_id: {deal_id}",
                'warning'
            )
            return

        from sec_rss_parser.summary_processor.dma_summary_processor import extract_from_dma_summary

        # DMA extraction will use accession_number to load press release for comparison
        result = extract_from_dma_summary(
            summary_text=summary_text,
            deal_id=deal_id,
            accession_number=accession_number,
            company_name=company_name,
            cik_number=cik_number,
            dma_summary_id=dma_summary_id,
            dma_summary_docx=dma_summary_docx,
            filing_date=None,
            send_email=True,
        )

        if result:
            inconsistencies_count = len(result.get('inconsistencies', []))
            log_and_print(
                f"✅ DMA extraction completed for deal_id: {deal_id} "
                f"(target: {result.get('extracted', {}).get('target', 'N/A')}, "
                f"inconsistencies: {inconsistencies_count})"
            )
        else:
            log_and_print(
                f"⚠️ DMA extraction returned no result for deal_id: {deal_id}",
                'warning'
            )

    except Exception as e:
        log_and_print(
            f"❌ Error in _extract_dma_data for deal_id {deal_id}: {e}", 'error')


def _fetch_recent_deals_excerpts(limit=10):
    """
    Fetch the N most recent deals and return a list of deal excerpts (deal_id + key fields)
    for GPT matching. Returns list of dicts with deal_id, target_name, acquire_name, target_cik,
    acquirer_cik, target_ticker, acquirer_ticker, announce_date.
    """
    try:
        jobs = ProcessingJob.objects.order_by('-createdAt').limit(limit)
        excerpts = []
        for job in jobs:
            aid = str(job.id)
            ann = job.announce_date
            ann_str = ann.strftime(
                '%Y-%m-%d') if ann and hasattr(ann, 'strftime') else (str(ann) if ann else '')
            excerpts.append({
                "deal_id": aid,
                "target_name": (job.target_name or '').strip(),
                "acquire_name": (job.acquire_name or '').strip(),
                "target_cik": (job.cik or '').strip() if job.cik else '',
                "acquirer_cik": (job.acquirer_cik or '').strip() if job.acquirer_cik else '',
                "target_ticker": (job.target_ticker or '').strip() if job.target_ticker else '',
                "acquirer_ticker": (job.acquirer_ticker or '').strip() if job.acquirer_ticker else '',
                "announce_date": ann_str,
                "url": (job.sec_url or '').strip() if job.sec_url else '',
            })
        return excerpts
    except Exception as e:
        log_and_print(
            f"⚠️ Failed to fetch recent deals for matching: {e}", 'warning')
        return []


def _match_deal_with_gpt(current_deal_str, excerpts_str):
    """
    Ask GPT whether the current deal details match any of the given deal excerpts.
    Returns matched deal_id (str) if one match is found, else None.
    """
    if not openai or not os.environ.get("OPENAI_API_KEY"):
        return None
    if not excerpts_str or not current_deal_str:
        return None
    prompt = f"""You are an expert M&A deal matcher.

Your task is to determine whether the "Current deal details" refer to the SAME M&A deal as one of the "Existing deals".

Match deals using the following priority:

1. STRONG MATCH (any of these is enough):
   - Exact match of target_cik OR acquirer_cik
   - Exact match of both target_name AND acquire_name

2. MEDIUM MATCH:
   - Similar target_name (ignore suffixes like Inc., Ltd., plc, Corp.)
   - Similar acquire_name
   - AND announcement dates are within ±3 days

3. WEAK MATCH (only if no strong/medium match):
   - Matching tickers (target_ticker or acquirer_ticker)
   - OR strong partial name overlap (e.g., "AZEK" vs "The AZEK Company Inc.")

Important rules:
- Treat parent company and merger subsidiary as the SAME acquirer.
- Ignore minor formatting differences in names.
- If multiple matches exist, choose the BEST match (highest confidence).
- If no confident match exists, return no match.


Existing deals (each has deal_id; match to this if it's the same deal):
{excerpts_str}

Current deal details (from a new 8-K/EX-2.1 filing):
{current_deal_str}

Return ONLY valid JSON in this exact format:
- If the current deal matches ONE existing deal: {{ "matched": true, "deal_id": "<that deal_id>" }}
- If it does not match any existing deal: {{ "matched": false, "deal_id": null }}

Use only the deal_id values from the Existing deals list. Return nothing else."""

    try:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.responses.create(
            model="gpt-5.2",
            tools=[{"type": "web_search"}],
            input=prompt,
            reasoning={"effort": "high"},
        )
        text = None
        for item in response.output:
            if getattr(item, "type", None) == "message" and hasattr(item, "content"):
                for content_item in item.content:
                    if getattr(content_item, "type", None) == "output_text":
                        text = getattr(content_item, "text", None)
                        break
            if text:
                break
        text = (text or "").strip()
        print(f"GPT Response: {text}")
        if not text:
            return None
        # Strip markdown code block if present
        if "```" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]
        else:
            start, end = text.find("{"), text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]
        parsed = json.loads(text)
        if parsed.get("matched") and parsed.get("deal_id"):
            return str(parsed["deal_id"]).strip()
        return None
    except Exception as e:
        log_and_print(f"⚠️ GPT deal match failed: {e}", 'warning')
        return None


def _update_existing_deal_with_details(deal_id, data):
    """
    Update an existing ProcessingJob (deal) with current 8-K details.
    data should have target_cik, target_name, acquirer_cik, acquirer_name, target_ticker,
    acquirer_ticker, announce_data (date str), and optionally url (sec_url).
    """
    try:
        job = ProcessingJob.objects.get(id=deal_id)
    except Exception as e:
        log_and_print(
            f"⚠️ Could not load deal {deal_id} for update: {e}", 'warning')
        return False

    try:
        if data.get('target_cik'):
            job.cik = str(data['target_cik']).strip()
        if data.get('target_name'):
            job.target_name = str(data['target_name']).strip()
        if data.get('acquirer_cik'):
            job.acquirer_cik = str(data['acquirer_cik']).strip()
        if data.get('acquirer_name'):
            job.acquire_name = str(data['acquirer_name']).strip()
        if data.get('target_ticker'):
            job.target_ticker = str(data['target_ticker']).strip()
        if data.get('acquirer_ticker'):
            job.acquirer_ticker = str(data['acquirer_ticker']).strip()
        if data.get('announce_data'):
            job.announce_date = parse_filing_date(data['announce_data'])
        if data.get('sec_filing_id'):
            job.sec_filing_id = str(data['sec_filing_id']).strip()
        if data.get('url'):
            job.sec_url = str(data['url']).strip()
        job.deal_status = "Open"
        job.updatedAt = datetime.utcnow()
        job.save()
        log_and_print(
            f"✅ Updated existing deal {deal_id} with current 8-K details")
        return True
    except Exception as e:
        log_and_print(f"⚠️ Failed to update deal {deal_id}: {e}", 'warning')
        return False


_EXTRACTION_2_1_MODULE = None


def _get_extraction_worker():
    """
    Load worker() from document_processor/extraction_2-1.py.
    The filename has a hyphen so we use importlib.
    """
    global _EXTRACTION_2_1_MODULE
    if _EXTRACTION_2_1_MODULE is not None:
        return _EXTRACTION_2_1_MODULE.worker

    extraction_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "document_processor",
        "extraction_2-1.py",
    )
    spec = importlib.util.spec_from_file_location(
        "extraction_2_1",
        extraction_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _EXTRACTION_2_1_MODULE = module
    return module.worker


def _create_processing_job_deal(data: dict, sec_filing_id: str, sec_url: str, parsed_json_url: str):
    """
    Create a ProcessingJob (deal) directly in MongoDB.
    """
    job = ProcessingJob(
        cik=(data.get("target_cik") or "").strip() or None,
        target_name=(data.get("target_name") or "").strip() or None,
        acquirer_cik=(data.get("acquirer_cik") or "").strip() or None,
        acquire_name=(data.get("acquirer_name") or "").strip() or None,
        target_ticker=(data.get("target_ticker") or "").strip() or None,
        acquirer_ticker=(data.get("acquirer_ticker") or "").strip() or None,
        announce_date=parse_filing_date(data.get("announce_data") or ""),
        embedding_status="PENDING",
        summary_status="PENDING",
        sec_filing_id=sec_filing_id,
        parsed_json_url=parsed_json_url,
        sec_url=sec_url,
        deal_status="Open",
    )
    job.save()
    return str(job.id)


def _reset_job_for_reprocessing(deal_id: str, sec_filing_id: str, sec_url: str, parsed_json_url: str):
    """
    Reset fields that would otherwise cause summary generation to trigger immediately.
    """
    try:
        job = ProcessingJob.objects.get(id=ObjectId(deal_id))
    except Exception:
        return

    try:
        job.parsed_json_url = parsed_json_url
        job.sec_filing_id = sec_filing_id
        job.sec_url = sec_url
        job.embedding_status = "PENDING"
        job.summary_status = "PENDING"
        job.schema_results = None
        job.schema_processing_completed = False
        job.flattened_json_url = None
        job.summary_docx_url = None
        job.error_message = None
        job.updatedAt = datetime.utcnow()
        job.save()
    except Exception:
        # Best-effort: even if reset fails, process_document can still run.
        return


def process_8k_document_async(ex21_url, cik_number, company_name, sec_filing_id, filing_date, item_data, company_details):
    """Async function to process 8-K document using Node API."""
    try:
        log_and_print(
            f"🚀 Starting 8-K document processing for: {company_name}")

        # Prepare data for Node API (company_details includes target_ticker, acquirer_ticker from document_analyzer)
        cd = company_details or {}
        target_cik = cd.get('target_cik', '')
        target_name = cd.get('target_name', '')
        acquirer_cik = cd.get('acquirer_cik', '')
        acquirer_name = cd.get('acquirer_name', '')
        target_ticker = (cd.get('target_ticker') or '').strip(
        ) if cd.get('target_ticker') else ''
        acquirer_ticker = (cd.get('acquirer_ticker') or '').strip(
        ) if cd.get('acquirer_ticker') else ''
        announce_data = filing_date.strftime(
            '%Y-%m-%d') if isinstance(filing_date, datetime) else str(filing_date)

        data = {
            "target_cik": target_cik,
            "target_name": target_name,
            "announce_data": announce_data,
            "acquirer_name": acquirer_name,
            "acquirer_cik": acquirer_cik,
            "url": ex21_url,
            "sec_filing_id": sec_filing_id,
            "company_details": company_details,
            "target_ticker": target_ticker,
            "acquirer_ticker": acquirer_ticker,
        }

        # Validate required fields
        required_fields = ['target_cik', 'announce_data', 'target_name']
        missing_fields = [
            field for field in required_fields if not data.get(field)]

        if missing_fields:
            log_and_print(
                f"❌ Missing required fields: {', '.join(missing_fields)}", 'error')
            doc_processor = DocumentProcessingService()
            doc_processor._send_sec_filing_event(
                sec_filing_id, "Fail", f"Could not obtain all required fields: {', '.join(missing_fields)}")
            return

        # Fetch recent deals, build excerpts, and ask GPT if current deal matches any existing deal
        matched_deal_id = None
        recent_excerpts = _fetch_recent_deals_excerpts(limit=10)
        if recent_excerpts:
            excerpts_str = "\n\n".join(
                f"deal_id: {e['deal_id']}\n  target_name: {e['target_name']}\n  acquire_name: {e['acquire_name']}\n  target_cik: {e['target_cik']}\n  acquirer_cik: {e['acquirer_cik']}\n  target_ticker: {e['target_ticker']}\n  acquirer_ticker: {e['acquirer_ticker']}\n  announce_date: {e['announce_date']}\n  url: {e['url']}"
                for e in recent_excerpts
            )
            current_deal_str = (
                f"target_name: {data.get('target_name', '')}\n"
                f"acquire_name: {data.get('acquirer_name', '')}\n"
                f"target_cik: {data.get('target_cik', '')}\n"
                f"acquirer_cik: {data.get('acquirer_cik', '')}\n"
                f"target_ticker: {data.get('target_ticker', '')}\n"
                f"acquirer_ticker: {data.get('acquirer_ticker', '')}\n"
                f"announce_data: {data.get('announce_data', '')}\n"
                f"url: {data.get('url', '')}"
            )
            matched_deal_id = _match_deal_with_gpt(
                current_deal_str, excerpts_str)
            if matched_deal_id:
                log_and_print(
                    f"🔗 Current 8-K matched existing deal_id: {matched_deal_id}")
                _update_existing_deal_with_details(matched_deal_id, data)
            else:
                log_and_print(
                    "📋 No matching existing deal; will create deal via extraction/upload (fallback: Node API)")

        # Send processing started event
        doc_processor = DocumentProcessingService()
        doc_processor._send_sec_filing_event(sec_filing_id, "In Progress")

        # Build Node API payload (include deal_id when we matched an existing deal)
        # Note: we only call Node API if extraction/upload fails.
        node_payload = {
            "url": ex21_url,
            "target_cik": data.get('target_cik', ''),
            "announce_data": data.get('announce_data'),
            "target_name": data.get('target_name', ''),
            "acquired_name": data.get('acquirer_name', ''),
            "sec_filing_id": sec_filing_id,
            "acquirer_cik": data.get('acquirer_cik', ''),
            "target_ticker": data.get('target_ticker', ''),
            "acquirer_ticker": data.get('acquirer_ticker', ''),
            "is_from_ui": False
        }
        if matched_deal_id:
            node_payload["deal_id"] = matched_deal_id

        sec_filing_accession_number = item_data.get(
            "accession_number", "") or ""
        extracted_json_url = None
        deal_id = matched_deal_id
        form_type = item_data.get("form_type", "8-K(2.1)")
        extraction_result = None
        extraction_warnings = []
        parsing_error_sent = False
        parsing_success_sent = False

        def _send_parsing_error_email(error_message: str, log_records: list):
            nonlocal parsing_error_sent
            if parsing_error_sent:
                return
            parsing_error_sent = True

            try:
                subject, html_email = generate_parsing_error_email_html(
                    company_name=company_name,
                    form_type=form_type,
                    sec_filing_id=sec_filing_id,
                    sec_url=ex21_url,
                    accession_number=sec_filing_accession_number,
                    error_message=error_message,
                    log_records=log_records,
                )

                payload = {
                    "subject": subject,
                    "html": html_email,
                    "company_name": company_name,
                    "accession_number": sec_filing_accession_number,
                    "form_type": form_type,
                    "email_type": "parsing_error",
                    "sec_filing_id": sec_filing_id,
                    "sec_url": ex21_url,
                }

                send_webhook_notification(
                    N8N_WEBHOOK_URL_PARSING_ERROR,
                    payload,
                    "parsing_error",
                )
            except Exception as email_e:
                # Never block the main processing flow due to webhook/email issues.
                log_and_print(
                    f"❌ Failed to send parsing error email via webhook: {email_e}",
                    "error",
                )

        def _send_parsing_success_email(log_records: list, parsed_json_url: str):
            nonlocal parsing_success_sent
            if parsing_success_sent:
                return
            parsing_success_sent = True

            try:
                subject, html_email = generate_parsing_success_email_html(
                    company_name=company_name,
                    form_type=form_type,
                    sec_filing_id=sec_filing_id,
                    sec_url=ex21_url,
                    accession_number=sec_filing_accession_number,
                    parsed_json_url=parsed_json_url,
                    deal_id=deal_id,
                    log_records=log_records,
                )

                payload = {
                    "subject": subject,
                    "html": html_email,
                    "company_name": company_name,
                    "accession_number": sec_filing_accession_number,
                    "form_type": form_type,
                    "email_type": "parsing_success",
                    "sec_filing_id": sec_filing_id,
                    "sec_url": ex21_url,
                    "deal_id": deal_id,
                    "parsed_json_url": parsed_json_url,
                }

                send_webhook_notification(
                    N8N_WEBHOOK_URL_PARSING_SUCCESS,
                    payload,
                    "parsing_success",
                )
            except Exception as email_e:
                # Never block the main processing flow due to webhook/email issues.
                log_and_print(
                    f"❌ Failed to send parsing success email via webhook: {email_e}",
                    "error",
                )

        # 1) Try local extraction + S3 upload first
        try:
            worker = _get_extraction_worker()
            extraction_result = worker(ex21_url)

            if extraction_result and extraction_result.get("status") == "success":
                output = extraction_result.get("output")
                output_file = extraction_result.get("output_file")
                extraction_warnings = extraction_result.get(
                    "warnings") or []

                # Valid means: non-empty JSON output + output file exists.
                if (
                    isinstance(output, list)
                    and len(output) > 0
                    and output_file
                    and os.path.exists(output_file)
                ):
                    s3_key = build_parsed_jsons_s3_key(
                        ex21_url,
                        sec_filing_accession_number,
                    )
                    extracted_json_url = upload_json_file_to_s3(
                        output_file,
                        s3_key,
                    )
                else:
                    error_message = (
                        "Extraction returned empty/invalid JSON for "
                        f"{ex21_url}"
                    )
                    log_and_print(
                        f"❌ {error_message}",
                        "error",
                    )
                    _send_parsing_error_email(
                        error_message=error_message,
                        log_records=extraction_warnings,
                    )
            else:
                extraction_warnings = extraction_result.get(
                    "warnings") or [] if extraction_result else []
                extraction_reason = extraction_result.get(
                    "reason") if extraction_result else None
                error_message = (
                    extraction_reason
                    or f"Extraction failed for {ex21_url}: {extraction_result}"
                )

                log_and_print(
                    f"❌ {error_message}",
                    "error",
                )
                _send_parsing_error_email(
                    error_message=error_message,
                    log_records=extraction_warnings,
                )
        except Exception as e:
            extraction_warnings = []
            log_and_print(
                f"❌ Extraction worker threw an error: {e}",
                "error",
            )

        # 2) If extraction/upload succeeded, create/reset deal + process_document
        if extracted_json_url:
            try:
                if not deal_id:
                    deal_id = _create_processing_job_deal(
                        data=data,
                        sec_filing_id=sec_filing_id,
                        sec_url=ex21_url,
                        parsed_json_url=extracted_json_url,
                    )
                else:
                    _reset_job_for_reprocessing(
                        deal_id=deal_id,
                        sec_filing_id=sec_filing_id,
                        sec_url=ex21_url,
                        parsed_json_url=extracted_json_url,
                    )

                log_and_print(f"✅ Processing started, deal_id: {deal_id}")

                # Send a "success parsing" email before kicking off doc processing.
                _send_parsing_success_email(
                    log_records=extraction_warnings,
                    parsed_json_url=extracted_json_url,
                )

                process_result = doc_processor.process_document(
                    file_url=extracted_json_url,
                    deal_id=deal_id,
                    sec_filing_id=sec_filing_id,
                    embed_data=True,
                )

                log_and_print(
                    f"✅ Document processing started: {process_result}")

                summary_thread = threading.Thread(
                    target=generate_8k_summary_async,
                    args=(
                        deal_id,
                        company_name,
                        item_data.get("form_type", "8-K"),
                        cik_number,
                        item_data.get("link", ""),
                        item_data.get("accession_number", ""),
                    ),
                )
                summary_thread.daemon = True
                summary_thread.start()

                log_and_print(
                    f"✅ Started summary generation monitoring thread for deal_id: {deal_id}"
                )
                return
            except Exception as e:
                log_and_print(
                    f"❌ process_document failed after extraction/upload: {e}",
                    "error",
                )
                # Fall through to Node API fallback below.

        # 3) Extraction/upload failed -> call Node API
        # log_and_print(
        #     "📞 Extraction/upload failed; falling back to Node API",
        #     "warning",
        # )

        # log_and_print(f"📞 Calling Node API with data: {node_payload}")
        # response = call_node_api(
        #     endpoint="deal/process-with-url",
        #     method="POST",
        #     data=node_payload,
        # )

        # log_and_print(f"📥 Node API response: {response}")

        # if response.get("status") and response.get("data", {}).get("jsonUrl"):
        #     deal_id = response["data"].get("deal_id")
        #     json_url = response["data"]["jsonUrl"]

        #     if deal_id and json_url:
        #         log_and_print(f"✅ Processing started, deal_id: {deal_id}")

        #         process_result = doc_processor.process_document(
        #             file_url=json_url,
        #             deal_id=deal_id,
        #             sec_filing_id=sec_filing_id,
        #             embed_data=True,
        #         )

        #         log_and_print(
        #             f"✅ Document processing started: {process_result}")

        #         summary_thread = threading.Thread(
        #             target=generate_8k_summary_async,
        #             args=(
        #                 deal_id,
        #                 company_name,
        #                 item_data.get("form_type", "8-K"),
        #                 cik_number,
        #                 item_data.get("link", ""),
        #                 item_data.get("accession_number", ""),
        #             ),
        #         )
        #         summary_thread.daemon = True
        #         summary_thread.start()

        #         log_and_print(
        #             f"✅ Started summary generation monitoring thread for deal_id: {deal_id}"
        #         )
        #     else:
        #         log_and_print(
        #             "❌ No deal_id/jsonUrl in Node API response", 'error')
        #         doc_processor._send_sec_filing_event(
        #             sec_filing_id,
        #             "Fail",
        #             "No deal_id/jsonUrl returned from Node API",
        #         )
        # else:
        #     log_and_print(
        #         f"❌ Node API response did not contain expected data: {response}",
        #         "error",
        #     )
        #     doc_processor._send_sec_filing_event(
        #         sec_filing_id,
        #         "Fail",
        #         "Invalid response from Node API",
        #     )

    except Exception as e:
        log_and_print(f"❌ Error in process_8k_document_async: {e}", 'error')
        try:
            doc_processor = DocumentProcessingService()
            doc_processor._send_sec_filing_event(sec_filing_id, "Fail", str(e))
        except:
            pass


def process_8k_document_helper(cik_number, company_name, sec_filing_id, filing_date, form_type, ex21_url, item_data, company_details):
    """Helper function to process 8-K document programmatically."""
    try:
        log_and_print(
            f"🚀 Starting 8-K document processing for: {company_name} - {sec_filing_id}")

        # Start processing in a separate thread
        processing_thread = threading.Thread(
            target=process_8k_document_async,
            args=(ex21_url, cik_number, company_name, sec_filing_id,
                  filing_date, item_data, company_details)
        )
        processing_thread.daemon = True
        processing_thread.start()

        log_and_print(
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
        log_and_print(
            f"❌ Error starting 8-K document processing: {str(e)}", 'error')
        return None


class SECRSSParser:
    def __init__(self, form_type=None):
        self.form_type = form_type
        self.headers = {
            "User-Agent": "MNA-Finder/1.0 (https://teqnodux.com; contact: ashish.kachadiya@teqnodux.com)",
            'Accept': 'application/atom+xml, application/xml, text/xml, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'https://www.sec.gov/',
        }

        self.form_types = FORM_TYPES
        self.feed_url = None
        self.set_feed_url(self.form_type)

        # Create session with retry strategy
        self.session = self._create_session()

    def _create_session(self):
        """Create requests session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

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
        """Fetch RSS/Atom feed from SEC"""
        max_retries = 3
        if not self.feed_url:
            log_and_print(
                "Feed URL is not set; cannot fetch RSS/Atom feed", 'error')
            return None

        for attempt in range(max_retries):
            try:
                time.sleep(2 + attempt)  # Progressive delay
                response = self.session.get(
                    self.feed_url, headers=self.headers, timeout=30)
                response.raise_for_status()
                return response.text
            except Exception as e:
                log_and_print(
                    f"Error fetching RSS feed (attempt {attempt + 1}/{max_retries}): {e}", 'error')
                if attempt == max_retries - 1:
                    return None
                time.sleep(5)
        return None

    def parse_rss_content(self, rss_content):
        """Parse the SEC Atom feed"""
        try:
            root = ET.fromstring(rss_content)
            return self.parse_atom_content(root)
        except Exception as e:
            log_and_print(f"Error parsing feed XML: {e}", 'error')
            print(f"Exception in parse_rss_content: {e}")
            return []

    def parse_atom_content(self, root):
        """Parse Atom feed format"""
        try:
            items = []
            entry_elements = root.findall(
                f'.//{{{ATOM_NAMESPACE}}}entry') or root.findall('.//entry')
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
            log_and_print(f"Error parsing Atom feed: {e}", 'error')
            print(f"Exception in parse_atom_content: {e}")
            return []

    def _find_element(self, parent, tag_name):
        """Find element with or without namespace"""
        elem = parent.find(f'.//{{{ATOM_NAMESPACE}}}{tag_name}')
        return elem if elem is not None else parent.find(f'.//{tag_name}')

    def parse_atom_entry(self, entry_elem, entry_number):
        """Parse Atom feed entry element"""
        try:
            # Extract basic fields
            title_elem = self._find_element(entry_elem, 'title')
            title = title_elem.text if title_elem is not None else ''

            # Extract link
            link_elem = entry_elem.find(
                f'.//{{{ATOM_NAMESPACE}}}link[@rel="alternate"]')
            if link_elem is None:
                link_elem = self._find_element(entry_elem, 'link')
            link = link_elem.get('href') if link_elem is not None else ''

            # Extract ID
            id_elem = self._find_element(entry_elem, 'id')
            guid = id_elem.text if id_elem is not None else ''

            # Extract summary
            summary_elem = self._find_element(entry_elem, 'summary')
            description = summary_elem.text if summary_elem is not None else ''

            # Extract updated date
            updated_elem = self._find_element(entry_elem, 'updated')
            pubDate = updated_elem.text if updated_elem is not None else None

            # Extract form type from category
            form_type = None
            category_elems = entry_elem.findall(
                f'.//{{{ATOM_NAMESPACE}}}category') or entry_elem.findall('.//category')
            for cat in category_elems:
                term = cat.get('term', '')
                if term and term.strip():
                    form_type = term.strip()
                    break

            # Extract accession number from ID
            accession_number = extract_accession_from_guid(guid)

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
            log_and_print(f"Error parsing Atom entry: {e}", 'error')
            print(f"Exception parsing Atom entry: {e}")
            return None

    def _extract_form_type_from_html(self, soup, company_info):
        """Extract form type from HTML"""
        form_type = None

        if company_info:
            ident_info = company_info.find('p', class_='identInfo')
            if ident_info:
                ident_text = ident_info.get_text()
                type_match = re.search(r'Type[:\s]+([A-Z0-9\s-]+)', ident_text)
                if type_match:
                    form_type = type_match.group(1).strip()
                    print(f"Extracted form_type from companyInfo: {form_type}")
                else:
                    # Fallback: find strong tag after "Type:"
                    for elem in ident_info.find_all('strong'):
                        prev_siblings = list(elem.previous_siblings)
                        prev_text = ' '.join([str(s) for s in prev_siblings if isinstance(
                            s, str) or (hasattr(s, 'get_text') and s.get_text())])
                        if 'Type' in prev_text or 'Type:' in prev_text:
                            form_type = elem.get_text().strip()
                            break

        # Fallback to formName
        if not form_type:
            form_name_elem = soup.find('div', {'id': 'formName'})
            if form_name_elem:
                form_text = form_name_elem.get_text()
                match = re.search(r'Form\s+([A-Z0-9\s-]+)', form_text)
                if match:
                    form_type = match.group(1).strip()

        return form_type

    def _extract_company_info(self, company_info):
        """Extract company information from HTML"""
        result = {
            'company_name': None,
            'cik_number': None,
            'ein': None,
            'state_of_incorp': None,
            'fiscal_year_end': None,
            'file_number': None,
            'assigned_sic': None
        }

        if not company_info:
            return result

        # Extract company name and CIK
        company_name_elem = company_info.find('span', class_='companyName')
        if company_name_elem:
            company_text = company_name_elem.get_text()
            match = re.match(r'^([^(]+)', company_text)
            if match:
                result['company_name'] = match.group(1).strip()

            cik_match = re.search(r'CIK[:\s]+(\d+)', company_text)
            if cik_match:
                result['cik_number'] = cik_match.group(1).zfill(CIK_LENGTH)

        # Extract other info
        ident_info = company_info.find('p', class_='identInfo')
        if ident_info:
            ident_text = ident_info.get_text()

            patterns = {
                'ein': r'EIN[.\s:]+(\d+)',
                'state_of_incorp': r'State of Incorp[.:\s]+([A-Z]{2})',
                'fiscal_year_end': r'Fiscal Year End[:\s]+(\d{4})',
                'file_number': r'File No[.:\s]+(\d{3}-\d+)',
                'assigned_sic': r'SIC[:\s]+(\d{4})'
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, ident_text)
                if match:
                    value = match.group(1)
                    result[key] = int(
                        value) if key == 'assigned_sic' else value

        return result

    def _extract_filing_dates(self, soup):
        """Extract filing and acceptance dates"""
        filing_date = None
        acceptance_datetime_utc = None
        period = None

        info_heads = soup.find_all('div', class_='infoHead')

        for info_head in info_heads:
            head_text = info_head.get_text()
            info_elem = info_head.find_next_sibling('div', class_='info')

            if not info_elem:
                continue

            if 'Filing Date' in head_text:
                try:
                    filing_date = datetime.strptime(
                        info_elem.get_text().strip(), '%Y-%m-%d')
                except:
                    pass
            elif 'Accepted' in head_text:
                accepted_text = info_elem.get_text().strip()
                try:
                    from zoneinfo import ZoneInfo
                    dt = datetime.strptime(accepted_text, '%Y-%m-%d %H:%M:%S')
                    dt_et = dt.replace(tzinfo=ZoneInfo("America/New_York"))
                    acceptance_datetime_utc = dt_et.astimezone(
                        ZoneInfo("UTC")).isoformat()
                except:
                    pass
            elif 'Period of Report' in head_text:
                period = info_elem.get_text().strip()

        return filing_date, acceptance_datetime_utc, period

    def _extract_xbrl_files(self, soup, form_type):
        """Extract XBRL files from document table"""
        xbrl_files = []
        table = soup.find('table', class_='tableFile')

        if not table:
            return xbrl_files

        rows = table.find_all('tr')[1:]  # Skip header
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 4:
                continue

            seq = cells[0].get_text().strip()
            description = cells[1].get_text().strip()
            doc_link = cells[2].find('a')
            doc_type = cells[3].get_text().strip()
            size_text = cells[4].get_text().strip() if len(cells) > 4 else '0'

            if not doc_link:
                continue

            doc_url = doc_link.get('href', '')
            if not doc_url.startswith('http'):
                doc_url = urljoin(SEC_BASE_URL, doc_url)

            # Extract file size
            size = 0
            if size_text:
                size_match = re.search(r'(\d+)', size_text.replace(',', ''))
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
            should_include = False
            if form_type in ["DEF 14A", "PRE 14A"]:
                if (doc_type in ["DEF 14A", "PRE 14A"]) and doc_url.endswith('.htm'):
                    should_include = True
            else:
                if doc_url.endswith('.htm'):
                    should_include = True

            if should_include:
                xbrl_files.append(file_data)

        return xbrl_files

    def fetch_and_parse_html(self, html_url, form_type_from_feed=None):
        """Fetch HTML from filing link and parse all relevant information"""
        try:
            time.sleep(2)  # Be respectful
            response = self.session.get(
                html_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            html_content = response.text

            soup = BeautifulSoup(html_content, 'html.parser')
            company_info = soup.find('div', class_='companyInfo')

            # Extract form type
            form_type = form_type_from_feed or self._extract_form_type_from_html(
                soup, company_info)
            if form_type_from_feed:
                print(f"Using form_type from Atom feed: {form_type}")

            # Extract accession number
            accession_number = None
            sec_num_elem = soup.find('div', {'id': 'secNum'})
            if sec_num_elem:
                acc_text = sec_num_elem.get_text()
                match = re.search(r'(\d{10}-\d{2}-\d{6})', acc_text)
                if match:
                    accession_number = match.group(1)

            # Extract dates
            filing_date, acceptance_datetime_utc, period = self._extract_filing_dates(
                soup)

            # Extract company info
            company_data = self._extract_company_info(company_info)

            # Extract XBRL files
            xbrl_files = self._extract_xbrl_files(soup, form_type)

            # Check for specific exhibits
            has_ex21 = any('EX-2.1' in file.get('type', '') or 'EX-2.1' in file.get(
                'description', '') for file in xbrl_files) if form_type == "8-K" else False
            has_ex99_1 = any('EX-99.1' in file.get('type', '') or 'EX-99.1' in file.get(
                'description', '') for file in xbrl_files) if form_type == "8-K" else False

            has_8k_document = any('8-K' in file.get('type', '') or '8-K' in file.get(
                'description', '') for file in xbrl_files) if form_type == "8-K" else False

            # Fallback extraction for missing required fields
            if not form_type:
                log_and_print(
                    f"Could not extract form_type from {html_url}", 'warning')
                title_elem = soup.find('title')
                if title_elem:
                    title_match = re.search(
                        r'([A-Z0-9\s]+)\s+-\s+', title_elem.get_text())
                    if title_match:
                        form_type = title_match.group(1).strip()
                        if ' - ' in form_type:
                            form_type = form_type.split(' - ')[0].strip()
                        form_type = re.sub(r'-[A-Z]$', '', form_type).strip()
                        log_and_print(
                            f"Extracted form_type from title: {form_type}")

            if not accession_number:
                log_and_print(
                    f"Could not extract accession_number from {html_url}", 'warning')
                url_match = re.search(r'/(\d{10}-\d{2}-\d{6})', html_url)
                if url_match:
                    accession_number = url_match.group(1)
                    log_and_print(
                        f"Extracted accession_number from URL: {accession_number}")

            if not form_type or not accession_number:
                log_and_print(
                    f"Missing required fields - form_type: {form_type}, accession_number: {accession_number} for {html_url}", 'error')
                return None

            print(
                f"form_type: {form_type}, accession_number: {accession_number}")

            return {
                'form_type': form_type,
                'accession_number': accession_number,
                'filing_date': filing_date,
                'acceptance_datetime_utc': acceptance_datetime_utc,
                'period': period,
                **company_data,
                'xbrl_files': xbrl_files,
                'has_ex21': has_ex21,
                'has_ex99_1': has_ex99_1,
                'has_8k_document': has_8k_document
            }
        except Exception as e:
            log_and_print(
                f"Error fetching/parsing HTML from {html_url}: {e}", 'error')
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

            deal_exists = ProcessingJob.objects(
                cik=cik_number,
                deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN,
            ).first() is not None
            log_and_print(
                f"CIK {cik_number} {'found' if deal_exists else 'not found'} in Deals collection")
            return deal_exists

        except Exception as e:
            log_and_print(
                f"Error checking CIK {cik_number} in Deals collection: {e}", 'error')
            return False

    def _cik_matches_deal_target_or_acquirer(self, cik_number: str) -> bool:
        """Return True if CIK matches any deal's target (cik) or acquirer (acquirer_cik)."""
        if not cik_number:
            return False
        cik_normalized = normalize_cik(cik_number)
        try:
            by_target = ProcessingJob.objects(
                cik=cik_normalized,
                deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN,
            ).first()
            if by_target:
                return True
            by_acquirer = ProcessingJob.objects(
                acquirer_cik=cik_normalized,
                deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN,
            ).first()
            return by_acquirer is not None
        except Exception as e:
            log_and_print(
                f"Error checking CIK in deals (target/acquirer): {e}", 'error')
            return False

    def _filter_unique_items(self, items):
        """Filter items to only include new accession numbers"""
        unique_items = []
        new_accession_numbers = []
        seen_accessions = set()

        for item_data in items:
            accession_number = item_data.get(
                'accession_number') or extract_accession_from_guid(item_data.get('guid'))

            if not accession_number:
                unique_items.append(item_data)
                continue

            # Check AccessionLookedUp cache
            if AccessionLookedUp.objects(accession_number=accession_number).first():
                print(f"Skipping already looked up filing: {accession_number}")
                continue

            # Check SECFiling database
            if SECFiling.objects(accession_number=accession_number).first():
                print(f"Skipping existing filing: {accession_number}")
                # Cache for future runs
                try:
                    AccessionLookedUp(accession_number=accession_number).save()
                except Exception as e:
                    if 'duplicate' not in str(e).lower() and 'E11000' not in str(e):
                        log_and_print(
                            f"Failed to save accession number to AccessionLookedUp: {e}", 'warning')
                continue

            # Check for duplicates in current batch
            if accession_number in seen_accessions:
                print(
                    f"Skipping duplicate accession in same feed: {accession_number}")
                continue

            seen_accessions.add(accession_number)
            unique_items.append(item_data)
            new_accession_numbers.append(accession_number)

        # Bulk insert new accession numbers
        if new_accession_numbers:
            for acc_num in new_accession_numbers:
                try:
                    AccessionLookedUp(accession_number=acc_num).save()
                except Exception as e:
                    if 'duplicate' not in str(e).lower() and 'E11000' not in str(e):
                        log_and_print(
                            f"Failed to save accession number {acc_num} to AccessionLookedUp: {e}", 'warning')

        return unique_items

    def _process_single_form_type(self, form_type):
        """Process a single form type - extracted for parallel processing"""
        try:
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

            # Filter to unique items
            unique_items = self._filter_unique_items(items)
            print(
                f"Found {len(unique_items)} unique new items to process for {form_type}")

            # Process each unique item
            processed_items = []
            for item_data in unique_items:
                form_type_from_feed = item_data.get('form_type')
                html_url = item_data.get('link')

                if html_url:
                    print(
                        f"Fetching HTML for: {html_url} (form_type from feed: {form_type_from_feed})")
                    html_data = parser.fetch_and_parse_html(
                        html_url, form_type_from_feed=form_type_from_feed)

                    if html_data:
                        item_data.update(html_data)

                        # Skip 8-K without EX-2.1 or EX-99.1
                        if item_data.get('form_type') == '8-K' and (not item_data.get('has_ex21') and not item_data.get('has_ex99_1') and not item_data.get('has_8k_document')):
                            print(
                                f"Skipping 8-K filing without both EX-2.1 and EX-99.1: {item_data.get('accession_number')}")
                            continue
                    else:
                        print(f"Failed to parse HTML for: {html_url}")
                        continue

                processed_items.append(item_data)

            print(
                f"Processing {len(processed_items)} items after HTML parsing and filtering for {form_type}")

            new_items_count = sum(
                1 for item in processed_items if self.save_filing(item))

            # Emit processing statistics
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
            log_and_print(
                f"Error processing form type {form_type}: {e}", 'error')
            return {
                'form_type': form_type,
                'processed_items': [],
                'new_items_count': 0,
                'success': False,
                'error': str(e)
            }

    def process_feed(self):
        """Process SEC feed for all form types in parallel"""
        try:
            form_types_to_process = [
                self.form_type] if self.form_type else self.parser.form_types
            all_processed_items = []
            total_new_items = 0

            # Process form types in parallel
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_form_type = {
                    executor.submit(self._process_single_form_type, ft): ft
                    for ft in form_types_to_process
                }

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
                        log_and_print(
                            f"Exception occurred while processing form type {form_type}: {e}", 'error')

            return {
                'success': True,
                'message': f'Processed {len(all_processed_items)} items across {len(form_types_to_process)} form types, {total_new_items} new',
                'total_items': len(all_processed_items),
                'new_items': total_new_items
            }
        except Exception as e:
            log_and_print(f"Error processing SEC feed: {e}", 'error')
            return {'success': False, 'error': str(e)}

    def _analyze_8k_filing(self, item_data):
        """Analyze 8-K filing (EX-2.1 or EX-99.1)"""
        has_ex21 = item_data.get('has_ex21')
        has_ex99_1 = item_data.get('has_ex99_1')

        if has_ex21:
            # Find EX-2.1 files (matching original logic that finds ALL matching files)
            ex21_files = [
                file for file in item_data.get('xbrl_files', [])
                if ('EX-2.1' in file.get('type', '') or 'EX-2.1' in file.get('description', ''))
                and file.get('url', '').endswith('.htm')
            ]

            if ex21_files:
                item_data['has_htm_files'] = True
                # Ensure xbrl_files have correct type field
                for file in ex21_files:
                    if 'EX-2.1' not in file.get('type', ''):
                        file['type'] = 'EX-2.1'

                log_and_print(
                    f"🔍 Analyzing 8-K document for: {item_data.get('company_name')}")
                item_data = self.document_analyzer.analyze_filing(item_data)

                # Log result
                if item_data.get('is_new_deal') is True:
                    log_and_print(
                        f"✅ NEW DEAL detected: {item_data.get('company_name')}")
                elif item_data.get('is_new_deal') is False:
                    log_and_print(
                        f"📝 AMENDMENT detected: {item_data.get('company_name')}")
                else:
                    log_and_print(
                        f"❓ Analysis inconclusive: {item_data.get('company_name')}")
            else:
                log_and_print(
                    f"⚠️ 8-K filing has EX-2.1 flag but no HTM file found: {item_data.get('company_name')}")
                item_data['is_new_deal'] = None
                item_data['following'] = False

        elif has_ex99_1:
            # Find EX-99.1 files (matching original logic that finds ALL matching files)
            ex99_1_files = [
                file for file in item_data.get('xbrl_files', [])
                if ('EX-99.1' in file.get('type', '') or 'EX-99.1' in file.get('description', ''))
                and file.get('url', '').endswith('.htm')
            ]

            if ex99_1_files:
                item_data['has_htm_files'] = True
                # Ensure xbrl_files have correct type field
                for file in ex99_1_files:
                    if 'EX-99.1' not in file.get('type', ''):
                        file['type'] = 'EX-99.1'

                log_and_print(
                    f"🔍 Analyzing 8-K EX-99.1 document for: {item_data.get('company_name')}")
                item_data = self.document_analyzer.analyze_ex99_1_filing(
                    item_data)

                # Log result
                if item_data.get('is_merger_related') is True:
                    log_and_print(
                        f"✅ EX-99.1 merger-related (confidence: {item_data.get('ex99_1_confidence', 0)}%): {item_data.get('company_name')}")
                elif item_data.get('is_merger_related') is False:
                    log_and_print(
                        f"📝 EX-99.1 not merger-related (confidence: {item_data.get('ex99_1_confidence', 0)}%): {item_data.get('company_name')}")
                    return False
                else:
                    log_and_print(
                        f"❓ EX-99.1 analysis inconclusive: {item_data.get('company_name')}")
            else:
                log_and_print(
                    f"⚠️ 8-K filing has EX-99.1 flag but no HTM file found: {item_data.get('company_name')}")
                return False

        return item_data

    def _analyze_proxy_filing(self, item_data):
        """Analyze proxy filing (DEF 14A, PRE 14A, S-4, F-4, etc.)"""
        # Normalize form_type
        normalized_form_type = item_data.get(
            'form_type').split(' - ')[0].strip()
        normalized_form_type = re.sub(
            r'-[A-Z]$', '', normalized_form_type).strip()
        item_data['form_type'] = normalized_form_type

        log_and_print(
            f"🔍 Analyzing {normalized_form_type} document for: {item_data.get('company_name')}")
        item_data = self.document_analyzer.analyze_def14a_filing(item_data)

        if item_data.get('document_kind'):
            log_and_print(
                f"📋 Document kind detected: {item_data.get('document_kind')} for {item_data.get('company_name')}")
        else:
            log_and_print(
                f"❓ Document kind analysis inconclusive: {item_data.get('company_name')}")

        return item_data

    def _prepare_filing_data(self, item_data):
        """Prepare item_data for database save"""
        # Convert acceptance_datetime_utc
        if item_data.get('acceptance_datetime_utc') and isinstance(item_data['acceptance_datetime_utc'], str):
            try:
                item_data['acceptance_datetime_utc'] = datetime.fromisoformat(
                    item_data['acceptance_datetime_utc'].replace('Z', '+00:00'))
            except Exception as e:
                print(f"Error converting acceptance_datetime_utc: {e}")
                item_data['acceptance_datetime_utc'] = None

        # Convert filing_date
        if item_data.get('filing_date'):
            item_data['filing_date'] = parse_filing_date(
                item_data['filing_date'])

        # Set has_htm_files
        if item_data.get('has_ex21') or item_data.get('has_ex99_1'):
            item_data['has_htm_files'] = True
        elif 'has_htm_files' not in item_data:
            item_data['has_htm_files'] = False

        # Normalize guid
        guid = item_data.get('guid', '') or ''
        link = item_data.get('link', '') or ''
        if guid.startswith('urn:tag:sec.gov'):
            item_data['guid'] = link or ''

        # Truncate description
        desc = item_data.get('description')
        if desc:
            item_data['description'] = desc[:MAX_DESCRIPTION_LENGTH]

        return item_data

    def _should_send_email(self, item_data):
        """Determine if email should be sent and what type"""
        form_type = item_data.get('form_type', '')
        cik_number = item_data.get('cik_number', '')

        log_and_print(
            f"🔍 Checking email notification - form_type: {form_type}, cik_number: {cik_number}")

        # 8-K EX-2.1 Definitive Merger Agreement
        if (form_type == '8-K' and item_data.get('has_htm_files') and
                item_data.get('document_kind') == 'Definitive Merger Agreement'):
            log_and_print(
                f"✅ Form type is 8-K (EX-2.1 Definitive Merger) - will send email")
            return True, 'ex21_merger', None

        # 8-K EX-99.1 merger-related
        if (form_type == '8-K' and item_data.get('has_ex99_1') and
                item_data.get('is_merger_related')):
            log_and_print(
                f"✅ Form type is 8-K (EX-99.1 merger-related) - will send different email")
            return True, 'ex99_1_merger', None

        # Periodic filings - check if CIK matches deal
        if form_type in PERIODIC_FORM_TYPES and cik_number:
            cik_normalized = normalize_cik(cik_number)
            log_and_print(f"📋 Normalized CIK: {cik_normalized}")

            try:
                matched_deal = ProcessingJob.objects(
                    cik=cik_normalized,
                    deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN,
                ).first()
                if matched_deal:
                    return True, 'standard', matched_deal
                else:
                    log_and_print(
                        f"ℹ️ No deal match found for CIK {cik_normalized}")
            except Exception as e:
                log_and_print(
                    f"Error checking deals collection for CIK {cik_normalized}: {e}", 'error')

        # Non-8-K filings - check deals collection
        elif form_type != '8-K' and cik_number:
            log_and_print(
                f"✅ Form type is not 8-K ({form_type}) and CIK exists ({cik_number}), checking deals collection...")
            cik_normalized = normalize_cik(cik_number)
            log_and_print(f"📋 Normalized CIK: {cik_normalized}")

            try:
                matched_deal = ProcessingJob.objects(
                    cik=cik_normalized,
                    deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN,
                ).first()
                if not matched_deal:
                    matched_deal = ProcessingJob.objects(
                        acquirer_cik=cik_normalized,
                        deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN,
                    ).first()

                if matched_deal:
                    log_and_print(
                        f"✅ MATCH FOUND! Deal: {getattr(matched_deal, 'target_name', 'Unknown')} (ID: {matched_deal.id})")
                    return True, 'standard', matched_deal
                else:
                    log_and_print(
                        f"ℹ️ No deal match found for CIK {cik_normalized}")
            except Exception as e:
                log_and_print(
                    f"Error checking deals collection for CIK {cik_normalized}: {e}", 'error')
        else:
            if not cik_number:
                log_and_print(f"ℹ️ Skipping email check - no CIK number")

        return False, None, None

    def _send_filing_email(self, item_data, email_type, matched_deal, filing):
        """Send email notification for filing"""
        try:
            form_type = item_data.get('form_type', 'N/A')
            if matched_deal:
                log_and_print(
                    f"📧 Preparing to send email for matched deal: {getattr(matched_deal, 'target_name', 'Unknown')}")
            else:
                log_and_print(
                    f"📧 Preparing to send email for 8-K filing: {item_data.get('company_name', 'Unknown')}")

            # Generate email HTML
            if email_type == 'ex99_1_merger':
                item_data_for_email = {
                    **item_data,
                    'ex99_1_confidence': item_data.get('ex99_1_confidence'),
                    'ex99_1_reasoning': item_data.get('ex99_1_reasoning'),
                }
                subject, html_email = generate_ex99_1_merger_email_html(
                    item_data_for_email, item_data.get('xbrl_files', []))
                log_and_print(
                    f"📝 Generated EX-99.1 merger email subject: {subject}")
            else:
                subject, html_email = generate_filing_email_html(
                    item_data, item_data.get('xbrl_files', []))
                log_and_print(f"📝 Generated email subject: {subject}")

            # Prepare payload
            payload = {
                'subject': subject,
                'html': html_email,
                'company_name': item_data.get('company_name', 'Unknown Company'),
                'accession_number': item_data.get('accession_number', 'N/A'),
                'form_type': item_data.get('form_type', 'N/A'),
                'filing_url': item_data.get('link', ''),
                'email_type': email_type or 'standard',
            }

            # Send via webhook
            if email_type == 'ex99_1_merger':
                try:
                    send_webhook_notification(
                        N8N_WEBHOOK_URL_8K_SUMMARY, payload, "email")
                except Exception as e:
                    log_and_print(
                        f"❌ Error sending 8-K summary email: {e}", 'error')
            else:
                try:
                    company_details = item_data.get('company_details') or {}
                    use_filing_webhook = (
                        company_details.get('is_target_us_listed') and
                        company_details.get(
                            'is_target_market_cap_greater_than_100m')
                    )
                    webhook_url = (
                        N8N_WEBHOOK_URL_FILING if use_filing_webhook
                        else N8N_WEBHOOK_URL_8K_SUMMARY
                    )
                    send_webhook_notification(webhook_url, payload, "email")

                    # After 2.1 email: fetch all SEC form filings (1 year before announce date), build HTML table, send to testing webhook

                    if email_type == 'ex21_merger' and form_type == '8-K':
                        try:
                            cik_number = item_data.get('cik_number')
                            filing_date = item_data.get('filing_date')
                            if cik_number and filing_date:
                                if not isinstance(filing_date, datetime):
                                    filing_date = parse_filing_date(
                                        filing_date)
                                if filing_date:
                                    start_date = (
                                        filing_date - timedelta(days=365)).strftime('%Y-%m-%d')
                                    filings = fetch_sec_filings(
                                        str(cik_number), start_date=start_date)
                                    company_name = item_data.get(
                                        'company_name', 'Unknown Company')
                                    sec_subject, sec_html = generate_sec_filings_email_html(
                                        company_name, filings, form_type="8-K(EX-2.1)")
                                    sec_payload = {
                                        'subject': sec_subject,
                                        'html': sec_html,
                                        'company_name': company_name,
                                        'email_type': 'sec_filings_last_year',
                                    }
                                    send_webhook_notification(
                                        N8N_WEBHOOK_URL_8K_SUMMARY, sec_payload, "email"
                                    )
                                    log_and_print(
                                        f"📤 Sent SEC form filings (last year) email: {len(filings)} filings for {company_name}"
                                    )
                        except Exception as sec_e:
                            log_and_print(
                                f"❌ Error fetching/sending SEC form filings email: {sec_e}", 'error'
                            )

                    elif form_type in ('10-K', '10-Q') and email_type == 'standard':
                        # 10-K/10-Q: use deal's announce date (match by CIK in DB), filter by form type
                        try:
                            cik_number = item_data.get('cik_number')
                            if not cik_number:
                                log_and_print(
                                    "⏭️ Skipping SEC form filings email for 10-K/10-Q: no CIK", 'warning'
                                )
                            else:
                                # Get announce date from deal matching CIK (matched_deal or DB lookup)
                                announce_date = None
                                deal = None
                                if matched_deal and getattr(matched_deal, 'announce_date', None):
                                    announce_date = matched_deal.announce_date
                                if not announce_date:
                                    cik_normalized = normalize_cik(cik_number)
                                    deal = ProcessingJob.objects(
                                        cik=cik_normalized,
                                        deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN,
                                    ).first()
                                    if not deal:
                                        deal = ProcessingJob.objects(
                                            acquirer_cik=cik_normalized,
                                            deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN,
                                        ).first()
                                    if deal and getattr(deal, 'announce_date', None):
                                        announce_date = deal.announce_date
                                if announce_date and not isinstance(announce_date, datetime):
                                    announce_date = parse_filing_date(
                                        announce_date.strftime(
                                            '%Y-%m-%d') if hasattr(announce_date, 'strftime') else str(announce_date)
                                    )

                                # If still no announce date: LLM + web search with company details
                                if not announce_date:
                                    deal_for_llm = matched_deal or deal
                                    parts = [
                                        f"Company (filing registrant): {item_data.get('company_name', '') or 'N/A'}",
                                        f"CIK: {cik_number}",
                                    ]
                                    if deal_for_llm:
                                        parts.append(
                                            f"Target: {getattr(deal_for_llm, 'target_name', '') or 'N/A'}"
                                        )
                                        parts.append(
                                            f"Acquirer: {getattr(deal_for_llm, 'acquire_name', '') or 'N/A'}"
                                        )
                                        sec_url = getattr(
                                            deal_for_llm, 'sec_url', None)
                                        if sec_url:
                                            parts.append(f"SEC URL: {sec_url}")
                                    company_details_str = "\n".join(parts)
                                    log_and_print(
                                        "🔍 No deal announce date in DB; trying LLM + web search..."
                                    )
                                    announce_date = _extract_announce_date_with_llm(
                                        company_details_str
                                    )
                                    if announce_date:
                                        log_and_print(
                                            f"✅ LLM extracted announce date: {announce_date.strftime('%Y-%m-%d')}"
                                        )

                                company_name = item_data.get(
                                    'company_name', 'Unknown Company')
                                start_date = None
                                if announce_date:
                                    start_date = announce_date.strftime(
                                        '%Y-%m-%d')
                                    log_and_print(
                                        f"📅 Using announce date as start_date: {start_date}"
                                    )
                                else:
                                    log_and_print(
                                        "⏭️ No announce date (DB or LLM); using start_date=None (1 year before today)"
                                    )
                                filings = fetch_sec_filings(
                                    str(cik_number),
                                    start_date=start_date,
                                    form_types=["10-K", "10-Q"],
                                )
                                # Persist each filing to 10k_10Q_Summary if not already present
                                deal_for_summary = deal or matched_deal
                                deal_id_str = str(
                                    deal_for_summary.id) if deal_for_summary else None
                                for f in filings:
                                    acc = f.get("accession_number")
                                    if not acc:
                                        continue
                                    if TenKTenQSummary.objects(accession_number=acc).first() is None:
                                        try:
                                            TenKTenQSummary(
                                                sec_document_url=f.get(
                                                    "url", ""),
                                                deal_id=deal_id_str,
                                                s3_json_url=None,
                                                s3_docx_url=None,
                                                cik_number=str(cik_number),
                                                accession_number=acc,
                                                filing_date=f.get(
                                                    "filing_date"),
                                                form_type=f.get("form"),
                                            ).save()
                                            log_and_print(
                                                f"📥 Saved 10-K/10-Q record: {acc}"
                                            )
                                        except (NotUniqueError, Exception) as save_e:
                                            log_and_print(
                                                f"⚠️ Could not save 10k_10Q_Summary {acc}: {save_e}",
                                                "warning",
                                            )

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
                                    N8N_WEBHOOK_URL_8K_SUMMARY, sec_payload, "email"
                                )
                                log_and_print(
                                    f"📤 Sent SEC form filings email: {len(filings)} {form_type} filings for {company_name}"
                                    + (f" (from announce date: {start_date})" if announce_date else " (from 1 year before today)")
                                )
                        except Exception as sec_e:
                            log_and_print(
                                f"❌ Error fetching/sending SEC form filings email (10-K/10-Q): {sec_e}", 'error'
                            )

                except Exception as e:
                    log_and_print(
                        f"❌ Error sending filing email: {e}", 'error')
            return True
        except Exception as e:
            log_and_print(f"❌ Error sending email notification: {e}", 'error')
            import traceback
            print(traceback.format_exc())
            return False

    def _process_8k_after_email(self, item_data, filing):
        """Process 8-K document after email is sent"""
        form_type = item_data.get('form_type', '')
        company_details = item_data.get('company_details')

        # Check conditions
        has_ex21_in_files = find_file_by_type(
            item_data.get('xbrl_files', []), 'EX-2.1') is not None
        has_ex21 = has_ex21_in_files or item_data.get('has_ex21', False)

        log_and_print(
            f"🔍 Checking 8-K processing condition - form_type: {form_type}, has_ex21: {has_ex21}")

        if (form_type == '8-K' and has_ex21 and
            item_data.get('document_kind') == 'Definitive Merger Agreement' and
            company_details and
            company_details.get('is_target_market_cap_greater_than_100m') and
                company_details.get('is_target_us_listed')):

            try:
                log_and_print(
                    f"🚀 Starting 8-K document processing after email notification for: {item_data.get('company_name')}")

                # Find EX-2.1 HTM file URL
                ex21_file = find_file_by_type(
                    item_data.get('xbrl_files', []), 'EX-2.1')

                if ex21_file:
                    ex21_url = build_full_sec_url(ex21_file.get('url'))
                    sec_filling_id = str(filing._id) if filing else None

                    if ex21_url and sec_filling_id:
                        filing_date_obj = parse_filing_date(
                            item_data.get('filing_date'))

                        # Call helper function
                        result = process_8k_document_helper(
                            cik_number=item_data.get('cik_number'),
                            company_name=item_data.get('company_name', ''),
                            sec_filing_id=sec_filling_id,
                            filing_date=filing_date_obj,
                            form_type=form_type,
                            ex21_url=ex21_url,
                            item_data=item_data,
                            company_details=company_details
                        )

                        if result:
                            log_and_print(
                                f"✅ Successfully started 8-K document processing: {result.get('message', 'Processing started')}")
                        else:
                            log_and_print(
                                f"❌ Failed to start 8-K document processing for: {item_data.get('company_name')}", 'error')
                    else:
                        log_and_print(
                            f"⚠️ Cannot process 8-K document: missing ex21_url or sec_filling_id", 'warning')
                else:
                    log_and_print(
                        f"⚠️ No EX-2.1 HTM file found in xbrl_files for 8-K filing: {item_data.get('company_name')}", 'warning')
            except Exception as e:
                log_and_print(
                    f"❌ Error processing 8-K document after email: {str(e)}", 'error')
                import traceback
                print(traceback.format_exc())

    def _process_proxy_if_matched(self, item_data, filing):
        """Process proxy document if CIK matches deal"""
        cik_number = item_data.get('cik_number', '')
        form_type = item_data.get('form_type', '')

        if not (cik_number and form_type in PROXY_FORM_TYPES):
            return

        log_and_print(
            f"🔍 Checking for CIK match in deals collection for: {item_data.get('company_name')}")
        cik_normalized = normalize_cik(cik_number)

        # Check deals collection (only Open or Unknown status)
        try:
            matched_deal = ProcessingJob.objects(
                cik=cik_normalized,
                deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN,
            ).first()
            if not matched_deal:
                matched_deal = ProcessingJob.objects(
                    acquirer_cik=cik_normalized,
                    deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN,
                ).first()

            if matched_deal:
                deal_id = str(matched_deal.id)
                log_and_print(
                    f"✅ CIK {cik_normalized} matched with deal collection: {deal_id}")

                # Find HTM file
                proxy_file = find_file_by_type(
                    item_data.get('xbrl_files', []), PROXY_FORM_TYPES)

                if proxy_file:
                    proxy_sec_url = build_full_sec_url(proxy_file.get('url'))

                    if proxy_sec_url:
                        filing_date_str = item_data.get('filing_date')
                        if isinstance(filing_date_str, datetime):
                            filing_date_str = filing_date_str.strftime(
                                '%Y-%m-%d')
                        else:
                            filing_date_str = str(
                                filing_date_str) if filing_date_str else ''

                        sec_filling_id = str(filing._id) if filing else None

                        if sec_filling_id:
                            log_and_print(
                                f"🚀 Starting proxy document processing for CIK {cik_normalized} (matched deal: {deal_id})")

                            result = process_sec_document_helper(
                                cik_number=cik_number,
                                company_name=item_data.get('company_name', ''),
                                sec_filling_id=sec_filling_id,
                                filing_date=filing_date_str,
                                form_type=form_type,
                                proxy_sec_url=proxy_sec_url,
                                deal_id=deal_id
                            )

                            if result:
                                log_and_print(
                                    f"✅ Successfully started proxy document processing: {result.get('proxy_document_id')}")
                            else:
                                log_and_print(
                                    f"❌ Failed to start proxy document processing for CIK {cik_normalized}", 'error')
                        else:
                            log_and_print(
                                f"⚠️ Cannot process document: sec_filling_id not found", 'warning')
                else:
                    log_and_print(
                        f"⚠️ No HTM file found in xbrl_files for proxy filing: {item_data.get('company_name')}", 'warning')
        except Exception as e:
            log_and_print(
                f"❌ Error processing SEC document for CIK {cik_normalized}: {str(e)}", 'error')

    def save_filing(self, item_data):
        """Save filing to database"""
        print(f"Saving filing: {item_data}")
        try:
            accession_number = item_data.get('accession_number')
            if not accession_number:
                log_and_print(
                    "Cannot save filing without accession_number", 'warning')
                return False

            # Check if already exists
            if SECFiling.objects(accession_number=accession_number).first():
                return False

            # Analyze document based on form type
            form_type = item_data.get('form_type', '')

            if form_type == "8-K" and (item_data.get('has_8k_document') or item_data.get('has_ex99_1')):
                if not self._cik_matches_deal_target_or_acquirer(item_data.get('cik_number')):
                    log_and_print(
                        f"⏭️ Skipping 8-K/EX-99.1 summary - CIK {item_data.get('cik_number') or 'N/A'} not in deals (target or acquirer)")
                else:
                    cik_normalized = normalize_cik(
                        item_data.get('cik_number') or '')
                    matched_deal = ProcessingJob.objects(
                        cik=cik_normalized,
                        deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN,
                    ).first()
                    if not matched_deal:
                        matched_deal = ProcessingJob.objects(
                            acquirer_cik=cik_normalized,
                            deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN,
                        ).first()
                    deal_id_str = str(
                        matched_deal.id) if matched_deal else None

                    xbrl_files = item_data.get('xbrl_files', [])
                    output_dir = tempfile.mkdtemp()

                    if summarize_8k_filing and item_data.get('has_8k_document'):
                        file_8k = find_file_by_type(xbrl_files, '8-K')
                        if file_8k and file_8k.get('url'):
                            url_8k = (
                                build_full_sec_url(file_8k.get('url'))
                                or file_8k.get('url')
                            )
                            # Use direct document URL (strip ix?doc=/ wrapper)
                            if url_8k and 'ix?doc=/' in url_8k:
                                url_8k = url_8k.replace('ix?doc=/', '', 1)
                            try:
                                result_8k = summarize_8k_filing(
                                    url_8k,
                                    output_dir,
                                    upload_to_s3=True,
                                    s3_folder="8k",
                                    verbose=False,
                                )
                                if result_8k.get('s3_url'):
                                    log_and_print(
                                        f"✅ 8-K summary uploaded to S3: {result_8k['s3_url']}")
                                    try:
                                        doc_8k = EightKSummary(
                                            accession_number=accession_number,
                                            company_name=item_data.get(
                                                'company_name'),
                                            cik_number=item_data.get(
                                                'cik_number'),
                                            sec_document_url=url_8k,
                                            s3_docx_url=result_8k.get(
                                                's3_url'),
                                            s3_json_url=result_8k.get(
                                                's3_json_url'),
                                            ticker=result_8k.get('ticker'),
                                            filing_date=result_8k.get(
                                                'filing_date'),
                                            items_reported=result_8k.get(
                                                'items_reported') or [],
                                            deal_id=deal_id_str,
                                            one_line_summary=result_8k.get(
                                                'L1_headline'),
                                        )
                                        doc_8k.save()
                                        log_and_print(
                                            f"💾 8-K summary saved to DB (8k_summary)")
                                        try:
                                            send_summary_email_via_webhook(
                                                summary_doc_url=doc_8k.s3_docx_url,
                                                company_name=item_data.get(
                                                    'company_name') or '',
                                                form_type='8-K',
                                                cik_number=item_data.get(
                                                    'cik_number') or '',
                                                sec_url=item_data.get(
                                                    'link') or url_8k,
                                                accession_number=accession_number,
                                                summary_kind='8-K',
                                                l1_headline=result_8k.get(
                                                    'L1_headline'),
                                            )
                                            log_and_print(
                                                f"📧 8-K summary email sent via webhook (docx link included)")
                                        except Exception as email_e:
                                            log_and_print(
                                                f"❌ Failed to send 8-K summary email: {email_e}", 'error')
                                    except Exception as db_e:
                                        log_and_print(
                                            f"❌ Failed to save 8-K summary to DB: {db_e}", 'error')
                            except Exception as e:
                                log_and_print(
                                    f"❌ 8-K summary failed: {e}", 'error')

                    if summarize_8k_filing and item_data.get('has_ex99_1'):
                        file_ex99 = find_file_by_type(xbrl_files, 'EX-99.1')
                        if file_ex99 and file_ex99.get('url'):
                            url_ex99 = build_full_sec_url(
                                file_ex99.get('url')) or file_ex99.get('url')
                            try:
                                result_99 = summarize_8k_filing(
                                    url_ex99,
                                    output_dir,
                                    upload_to_s3=True,
                                    s3_folder="99_1",
                                    verbose=False,
                                )
                                if result_99.get('s3_url'):
                                    log_and_print(
                                        f"✅ EX-99.1 summary uploaded to S3: {result_99['s3_url']}")
                                    try:
                                        doc_99 = Ex99_1Summary(
                                            accession_number=accession_number,
                                            company_name=item_data.get(
                                                'company_name'),
                                            cik_number=item_data.get(
                                                'cik_number'),
                                            sec_document_url=url_ex99,
                                            s3_docx_url=result_99.get(
                                                's3_url'),
                                            s3_json_url=result_99.get(
                                                's3_json_url'),
                                            ticker=result_99.get('ticker'),
                                            filing_date=result_99.get(
                                                'filing_date'),
                                            items_reported=result_99.get(
                                                'items_reported') or [],
                                            deal_id=deal_id_str,
                                            one_line_summary=result_99.get(
                                                'L1_headline'),
                                        )
                                        doc_99.save()
                                        log_and_print(
                                            f"💾 EX-99.1 summary saved to DB (99_1_summary)")
                                        try:
                                            send_summary_email_via_webhook(
                                                summary_doc_url=doc_99.s3_docx_url,
                                                company_name=item_data.get(
                                                    'company_name') or '',
                                                form_type='8-K (EX-99.1)',
                                                cik_number=item_data.get(
                                                    'cik_number') or '',
                                                sec_url=item_data.get(
                                                    'link') or url_ex99,
                                                accession_number=accession_number,
                                                summary_kind='EX-99.1',
                                                l1_headline=result_99.get(
                                                    'L1_headline'),
                                            )
                                            log_and_print(
                                                f"📧 EX-99.1 summary email sent via webhook (docx link included)")
                                        except Exception as email_e:
                                            log_and_print(
                                                f"❌ Failed to send EX-99.1 summary email: {email_e}", 'error')
                                    except Exception as db_e:
                                        log_and_print(
                                            f"❌ Failed to save EX-99.1 summary to DB: {db_e}", 'error')
                            except Exception as e:
                                log_and_print(
                                    f"❌ EX-99.1 summary failed: {e}", 'error')

            if form_type == '8-K' and (item_data.get('has_ex21') or item_data.get('has_ex99_1')):
                result = self._analyze_8k_filing(item_data)
                if result is False:
                    return False
                item_data = result if result else item_data
            elif form_type and form_type.startswith(("DEFM14A", "DEFM14C", "PREM14A", "PREM14C", "S-4", "F-4")):
                item_data = self._analyze_proxy_filing(item_data)
            else:
                item_data['is_new_deal'] = None
                item_data['following'] = False

            # Prepare data for database
            item_data = self._prepare_filing_data(item_data)

            # Capture EX-99.1 fields before filtering (for email and logic)
            has_ex99_1 = item_data.get('has_ex99_1')
            is_merger_related = item_data.get('is_merger_related')
            ex99_1_confidence = item_data.get('ex99_1_confidence')
            ex99_1_reasoning = item_data.get('ex99_1_reasoning')
            ex99_1_is_target_us_listed = item_data.get('is_target_us_listed')
            ex99_1_is_target_market_cap_gt_100m = item_data.get(
                'is_target_market_cap_greater_than_100m')

            # Filter allowed fields
            item_data = {k: v for k, v in item_data.items()
                         if k in ALLOWED_FILING_FIELDS}

            # Re-check for duplicates (race condition)
            if SECFiling.objects(accession_number=accession_number).first():
                log_and_print(
                    f"⏭️ Skipping duplicate save (existing): {accession_number}")
                return False

            # Create and save filing
            filing = SECFiling(**item_data)
            try:
                print(f"Saving filing: {item_data}")
                filing.save()
            except (NotUniqueError, Exception) as e:
                if 'E11000' in str(e) or 'duplicate' in str(e).lower():
                    log_and_print(
                        f"⏭️ Duplicate accession_number (race): {accession_number} - skipping", 'warning')
                    return False
                raise

            log_and_print(
                f"💾 Saved filing: {item_data.get('company_name')} - {item_data.get('accession_number')}")

            # Prepare filing data for WebSocket
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

            # Emit WebSocket events
            SECWebSocketService.emit_new_sec_filing(filing_data)

            if item_data.get('is_new_deal') is not None:
                analysis_result = 'new_deal' if item_data.get(
                    'is_new_deal') else 'amendment'
                SECWebSocketService.emit_sec_analysis_complete(
                    filing_data, analysis_result)

            # Email notification logic
            # Restore EX-99.1 fields for email check and email content
            item_data['has_ex99_1'] = has_ex99_1
            item_data['is_merger_related'] = is_merger_related
            item_data['ex99_1_confidence'] = ex99_1_confidence
            item_data['ex99_1_reasoning'] = ex99_1_reasoning
            item_data['is_target_us_listed'] = ex99_1_is_target_us_listed
            item_data['is_target_market_cap_greater_than_100m'] = ex99_1_is_target_market_cap_gt_100m

            should_send_email, email_type, matched_deal = self._should_send_email(
                item_data)

            if should_send_email:
                if self._send_filing_email(item_data, email_type, matched_deal, filing):
                    # Process 8-K document after email
                    self._process_8k_after_email(item_data, filing)

            # Process proxy documents if CIK matches
            self._process_proxy_if_matched(item_data, filing)

            return True

        except Exception as e:
            log_and_print(f"Error saving filing: {e}", 'error')
            return False
