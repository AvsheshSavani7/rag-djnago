"""
8-K-only feed processor script.

This module processes only 8-K filings from SEC RSS feed with the following workflow:
1. Fetch 8-K feed from SEC
2. Filter by accession number (check AccessionLookedUp and SECFiling)
3. Fetch filing details for new accessions
4. Check for EX-2.1 or EX99.1 in XBRL files
5. For EX-2.1: check if US-related and market cap > $100M
6. Generate and send emails
7. For qualified EX-2.1: process via 8-K document helper (Node API, same as services.py)
"""

import copy
import logging
import tempfile
from datetime import datetime, timedelta
from mongoengine.errors import NotUniqueError

from .utils_8k import (
    SECRSSParser,
    extract_accession_from_guid,
    build_full_sec_url,
    find_file_by_type,
    normalize_cik,
    parse_filing_date,
    send_webhook_notification,
    log_and_print,
    safe_isoformat,
)
from .models import SECFiling, AccessionLookedUp, SECFilingSummary
from .document_analyzer_new import SECDocumentAnalyzer
from .email_templates import (
    generate_filing_email_html,
    generate_8k_document_email_html,
    generate_ex99_1_merger_email_html,
    generate_sec_filings_email_html,
)
from .Eight_k_summary import summarize_8k_filing
from .sec_Last_Year import print_filings as fetch_sec_filings
from document_processor.models import ProcessingJob
from .services import process_8k_document_helper
from .websocket_service import SECWebSocketService

logger = logging.getLogger(__name__)

# Deal status constants
DEAL_STATUS_OPEN_OR_UNKNOWN = ["Open", "Unknown"]

# https://n8n-xwx1.onrender.com/webhook/b3007d21-6845-47b5-aece-7b26583758bc #me ,josh,kaushal
# https://n8n-xwx1.onrender.com/webhook/3ff1b0ea-7114-4dda-940e-95ce81e08017 #all
# https://n8n-xwx1.onrender.com/webhook/80830c6d-ff5b-45e3-9ef3-a061db1fbf0c #me only

# Constants (from services.py)
# N8N_WEBHOOK_URL_8K_SUMMARY = "https://n8n-xwx1.onrender.com/webhook/b3007d21-6845-47b5-aece-7b26583758bc" #me ,josh,kaushal
# N8N_WEBHOOK_URL_FILING = "https://n8n-xwx1.onrender.com/webhook/3ff1b0ea-7114-4dda-940e-95ce81e08017" #all
N8N_WEBHOOK_URL_8K_SUMMARY = "https://n8n-xwx1.onrender.com/webhook/b3007d21-6845-47b5-aece-7b26583758bc"
N8N_WEBHOOK_URL_FILING = "https://n8n-xwx1.onrender.com/webhook/3ff1b0ea-7114-4dda-940e-95ce81e08017"
MAX_DESCRIPTION_LENGTH = 50

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


class EightKFeedProcessor:
    """Processor for 8-K filings only"""

    def __init__(self):
        self.parser = SECRSSParser(form_type='8-K')
        self.document_analyzer = SECDocumentAnalyzer()
        self.processed_count = 0
        self.ex21_processed_count = 0
        self.ex99_processed_count = 0
        self.summary_8k_count = 0
        self.summary_ex99_count = 0
        self.skipped_count = 0
        self.error_count = 0

    def run(self, rss_content=None, rss_file=None):
        """
        Main entry point to process 8-K feed.

        For testing you can pass manual RSS content:
          - rss_content: raw XML string (e.g. from open('rss.xml').read())
          - rss_file: path to a local XML file (e.g. 'sec_rss_parser/rss.xml')
        If either is provided, the live SEC feed is not fetched.
        """
        try:
            log_and_print("=" * 80)
            log_and_print("🚀 Starting 8-K-only feed processor")
            log_and_print("=" * 80)

            # Use manual RSS for testing if provided
            if rss_content is None and rss_file:
                try:
                    with open(rss_file, 'r', encoding='utf-8', errors='replace') as f:
                        rss_content = f.read()
                    log_and_print(f"📂 Using RSS from file: {rss_file}")
                except Exception as e:
                    log_and_print(
                        f"❌ Failed to read RSS file {rss_file}: {e}", 'error')
                    return {
                        'success': False,
                        'error': f'Failed to read RSS file: {e}',
                        'form_type': '8-K'
                    }
            if rss_content is None:
                self.parser.set_feed_url('8-K')
                rss_content = self.parser.fetch_rss_feed()
            else:
                log_and_print("📂 Using provided RSS content (skip live fetch)")

            if not rss_content:
                log_and_print("❌ Failed to fetch 8-K RSS feed", 'error')
                return {
                    'success': False,
                    'error': 'Failed to fetch RSS feed',
                    'form_type': '8-K'
                }

            # Parse feed
            items = self.parser.parse_rss_content(rss_content)
            log_and_print(f"📋 Parsed {len(items)} items from 8-K feed")

            if not items:
                log_and_print("⚠️ No items found in 8-K feed")
                return {
                    'success': True,
                    'message': 'No items in feed',
                    'form_type': '8-K',
                    'total_items': 0
                }

            # Filter by accession (check AccessionLookedUp)
            unique_items = self._filter_unique_items(items)
            log_and_print(
                f"✅ Found {len(unique_items)} new items to process (after accession filtering)")

            # Process each unique item
            for item_data in unique_items:
                self._process_single_item(item_data)

            # Log summary
            log_and_print("=" * 80)
            log_and_print("📊 8-K Processing Summary:")
            log_and_print(f"   Total items in feed: {len(items)}")
            log_and_print(f"   New items processed: {self.processed_count}")
            log_and_print(f"   EX-2.1 processed: {self.ex21_processed_count}")
            log_and_print(f"   EX-99.1 processed: {self.ex99_processed_count}")
            log_and_print(
                f"   8-K summaries generated: {self.summary_8k_count}")
            log_and_print(
                f"   EX-99.1 summaries generated: {self.summary_ex99_count}")
            log_and_print(f"   Skipped: {self.skipped_count}")
            log_and_print(f"   Errors: {self.error_count}")
            log_and_print("=" * 80)

            return {
                'success': True,
                'message': f'Processed {self.processed_count} new 8-K items',
                'form_type': '8-K',
                'total_items': len(items),
                'new_items': self.processed_count,
                'ex21_processed': self.ex21_processed_count,
                'ex99_processed': self.ex99_processed_count,
                'summary_8k_generated': self.summary_8k_count,
                'summary_ex99_generated': self.summary_ex99_count,
                'skipped': self.skipped_count,
                'errors': self.error_count
            }

        except Exception as e:
            log_and_print(f"❌ Error in 8-K feed processor: {e}", 'error')
            import traceback
            log_and_print(traceback.format_exc(), 'error')
            return {
                'success': False,
                'error': str(e),
                'form_type': '8-K'
            }

    def _filter_unique_items(self, items):
        """Filter items to only include new accession numbers"""
        unique_items = []
        new_accession_numbers = []
        seen_accessions = set()

        for item_data in items:
            accession_number = item_data.get(
                'accession_number') or extract_accession_from_guid(item_data.get('guid'))

            if not accession_number:
                log_and_print(
                    f"⚠️ Item without accession number: {item_data.get('title', 'N/A')}", 'warning')
                unique_items.append(item_data)
                continue

            # Check AccessionLookedUp cache
            if AccessionLookedUp.objects(accession_number=accession_number).first():
                log_and_print(
                    f"⏭️ Skipping already looked up filing: {accession_number}")
                continue

            # Check SECFiling database
            if SECFiling.objects(accession_number=accession_number).first():
                log_and_print(
                    f"⏭️ Skipping existing filing: {accession_number}")
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
                log_and_print(
                    f"⏭️ Skipping duplicate accession in same feed: {accession_number}")
                continue

            seen_accessions.add(accession_number)
            unique_items.append(item_data)
            new_accession_numbers.append(accession_number)

        # Bulk insert new accession numbers
        if new_accession_numbers:
            log_and_print(
                f"💾 Adding {len(new_accession_numbers)} new accessions to AccessionLookedUp")
            for acc_num in new_accession_numbers:
                try:
                    AccessionLookedUp(accession_number=acc_num).save()
                except Exception as e:
                    if 'duplicate' not in str(e).lower() and 'E11000' not in str(e):
                        log_and_print(
                            f"Failed to save accession number {acc_num} to AccessionLookedUp: {e}", 'warning')

        return unique_items

    def _check_cik_matches_deal(self, cik_number):
        """Check if CIK matches any deal (target or acquirer)"""
        if not cik_number:
            return False, None

        try:
            cik_normalized = normalize_cik(cik_number)

            # Check as target CIK
            matched_deal = ProcessingJob.objects(
                cik=cik_normalized,
                deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN,
            ).first()

            # Check as acquirer CIK
            if not matched_deal:
                matched_deal = ProcessingJob.objects(
                    acquirer_cik=cik_normalized,
                    deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN,
                ).first()

            if matched_deal:
                log_and_print(
                    f"✅ CIK {cik_normalized} matches deal: {matched_deal.id}")
                return True, str(matched_deal.id)
            else:
                log_and_print(f"⏭️ CIK {cik_normalized} not in deals")
                return False, None

        except Exception as e:
            log_and_print(f"❌ Error checking CIK in deals: {e}", 'error')
            return False, None

    def _filing_date_for_summary(self, item_filing_date, result_filing_date):
        """Return datetime for SECFilingSummary.filing_date from item or result."""
        if item_filing_date:
            if isinstance(item_filing_date, datetime):
                return item_filing_date
            parsed = parse_filing_date(item_filing_date)
            if parsed:
                return parsed
        if result_filing_date:
            parsed = parse_filing_date(result_filing_date)
            if parsed:
                return parsed
        return None

    def _process_8k_document(self, item_data, filing_entry):
        """
        8-K document flow:
        - If cik_matches_deal: generate summary, save to sec_filing_summary, send summary email (no GPT).
        - If not cik_matches_deal: run GPT; send main 8-K email only when all of
          is_merger_related, is_target_us_listed, is_target_market_cap_greater_than_100m are truthy.
        """
        try:
            url_8k = filing_entry.get('url')
            if not url_8k:
                log_and_print(
                    "⚠️ No 8-K document URL in filing entry", 'warning')
                return

            if item_data.get('cik_matches_deal'):
                log_and_print(
                    "✅ CIK matches deal → generating 8-K summary (no GPT)")
                self._generate_8k_summary_and_send(item_data, url_8k)
                return

            log_and_print(
                f"🔍 8-K: GPT analysis: {item_data.get('company_name')}")
            data = copy.deepcopy(item_data)
            data['xbrl_files'] = [
                {'type': '8-K', 'description': filing_entry.get('description', '8-K'), 'url': url_8k}]
            result = self.document_analyzer.analyze_ex99_1_filing(data)

            item_data['is_merger_related'] = result.get('is_merger_related')
            item_data['confidence'] = result.get('confidence', 0)
            item_data['reasoning'] = result.get('reasoning', '')
            item_data['is_target_us_listed'] = result.get(
                'is_target_us_listed')
            item_data['is_target_market_cap_greater_than_100m'] = result.get(
                'is_target_market_cap_greater_than_100m')
            log_and_print(
                f"   8-K Merger-related: {item_data.get('is_merger_related', False)}")
            log_and_print(
                f"   Market cap > $100M: {item_data.get('is_target_market_cap_greater_than_100m')}")
            log_and_print(
                f"   Confidence: {item_data.get('confidence', 0)}%")

            # Send 8-K email only if all three are truthy; otherwise skip
            is_merger = item_data.get('is_merger_related')
            is_listed = item_data.get('is_target_us_listed')
            cap_gt_100m = item_data.get(
                'is_target_market_cap_greater_than_100m')
            if is_merger and is_listed and cap_gt_100m:
                doc_files = [{
                    'type': filing_entry.get('document_type', '8-K'),
                    'url': filing_entry.get('url'),
                    'description': filing_entry.get('description', ''),
                    'file': filing_entry.get('file', ''),
                    'size': filing_entry.get('size', 0),
                    'sequence': filing_entry.get('sequence', 0),
                }]
                self._send_8k_gpt_email(item_data, doc_files=doc_files)
            else:
                log_and_print(
                    "⏭️ 8-K main email skipped (need is_merger_related, is_target_us_listed, and market cap > $100M)")
        except Exception as e:
            log_and_print(f"❌ Error in _process_8k_document: {e}", 'error')

    def _send_8k_gpt_email(self, item_data, doc_files=None):
        """Send email for main 8-K document (subject: 8-K – Company Name). Distinct from EX-99.1 email."""
        try:
            log_and_print("📧 Sending 8-K email with GPT data")
            if doc_files is None:
                doc_files = item_data.get('xbrl_files', [])
            subject, html_email = generate_8k_document_email_html(
                item_data, doc_files)
            payload = {
                'subject': subject,
                'html': html_email,
                'company_name': item_data.get('company_name', 'Unknown Company'),
                'accession_number': item_data.get('accession_number', 'N/A'),
                'form_type': '8-K',
                'filing_url': item_data.get('link', ''),
                'email_type': '8k_gpt',
            }
            send_webhook_notification(
                N8N_WEBHOOK_URL_8K_SUMMARY, payload, "8-K GPT email")
            log_and_print("✅ 8-K GPT email sent")
        except Exception as e:
            log_and_print(f"❌ Error sending 8-K GPT email: {e}", 'error')

    def _generate_8k_summary_and_send(self, item_data, url_8k):
        """Generate 8-K summary, save to sec_filing_summary, send summary email."""
        try:
            accession_number = item_data.get('accession_number')
            deal_id = item_data.get('deal_id')
            output_dir = tempfile.mkdtemp()
            log_and_print(f"📝 Generating 8-K summary: {url_8k}")
            result_8k = summarize_8k_filing(
                url_8k, output_dir, upload_to_s3=True, s3_folder="8k", verbose=False)
            if not result_8k.get('s3_url'):
                return
            log_and_print(
                f"✅ 8-K summary uploaded to S3: {result_8k['s3_url']}")
            try:
                filing_dt = self._filing_date_for_summary(
                    item_data.get('filing_date'), result_8k.get('filing_date'))
                # One document per accession_number, form_type always 8-K; EX-99.1 goes in eight_k.filings[]
                existing = SECFilingSummary.objects(
                    accession_number=accession_number, form_type='8-K'
                ).first()
                filings = []
                if existing and existing.eight_k:
                    filings = list(existing.eight_k.get('filings') or [])
                eight_k_payload = {
                    'one_line_summary': result_8k.get('L1_headline'),
                    'items_reported': result_8k.get('items_reported') or [],
                    's3_docx_url': result_8k.get('s3_url'),
                    's3_json_url': result_8k.get('s3_json_url'),
                    'filings': filings,
                }
                if existing:
                    existing.sec_document_url = url_8k
                    existing.filing_date = filing_dt
                    existing.deal_id = deal_id
                    existing.eight_k = eight_k_payload
                    existing.save()
                    doc_8k = existing
                else:
                    doc_8k = SECFilingSummary(
                        form_type='8-K',
                        accession_number=accession_number,
                        cik_number=item_data.get('cik_number'),
                        sec_document_url=url_8k,
                        filing_date=filing_dt,
                        deal_id=deal_id,
                        eight_k=eight_k_payload,
                    )
                    doc_8k.save()
                log_and_print("💾 8-K summary saved to DB (sec_filing_summary)")
                self.summary_8k_count += 1
                self._send_8k_summary_email(item_data, result_8k, url_8k)
            except Exception as db_e:
                log_and_print(
                    f"❌ Failed to save 8-K summary to DB: {db_e}", 'error')
        except Exception as e:
            log_and_print(f"❌ 8-K summary generation failed: {e}", 'error')

    def _process_single_item(self, item_data):
        """Process a single 8-K item"""
        try:
            html_url = item_data.get('link')
            accession_number = item_data.get(
                'accession_number') or extract_accession_from_guid(item_data.get('guid'))

            log_and_print("-" * 80)
            log_and_print(
                f"🔍 Processing: {item_data.get('title', 'N/A')[:80]}")
            log_and_print(f"   Accession: {accession_number}")
            log_and_print(f"   item_data 1: {item_data}")

            if not html_url:
                log_and_print("⚠️ No link URL found, skipping", 'warning')
                self.skipped_count += 1
                return

            # Fetch HTML and parse filing details
            log_and_print(f"🌐 Fetching filing details from: {html_url}")
            html_data = self.parser.fetch_and_parse_html(
                html_url, form_type_from_feed='8-K')

            log_and_print(f"   html_data 1: {html_data}")

            if not html_data:
                log_and_print(
                    f"❌ Failed to parse HTML for: {html_url}", 'error')
                self.error_count += 1
                return

            # Merge HTML data into item_data
            item_data.update(html_data)

            log_and_print(f"   item_data 2: {item_data}")

            # CIK check early (for summary: only generate when CIK matches a deal)
            cik_number = item_data.get('cik_number')
            cik_matches_deal, deal_id = self._check_cik_matches_deal(
                cik_number)
            item_data['deal_id'] = deal_id
            item_data['cik_matches_deal'] = cik_matches_deal

            log_and_print(f"   item_data 3: {item_data}")

            filing_array = item_data.get('filing_array', [])

            log_and_print(f"   filing_array 1: {filing_array}")
            if not filing_array:
                log_and_print(
                    "⚠️ No 8-K / EX-2.1 / EX-99.1 .htm documents in filing, skipping", 'warning')
                self.skipped_count += 1
                return

            log_and_print(
                f"   Form type: {item_data.get('form_type', 'N/A')}")
            log_and_print(
                f"   Filing array: {len(filing_array)} document(s) – {[f.get('document_type') for f in filing_array]}")

            item_data['has_htm_files'] = True

            # Save SECFiling once per item (has at least one .htm document)
            self._save_filing(item_data)

            # Iterate over filing_array and process by document_type
            for filing in filing_array:
                doc_type = filing.get('document_type')
                if doc_type == '8-K':
                    self._process_8k_document(item_data, filing)
                elif doc_type == 'EX-99.1':
                    self._process_ex99_filing(item_data, filing)
                elif doc_type == 'EX-2.1':
                    self._process_ex21_filing(item_data, filing)

        except Exception as e:
            log_and_print(
                f"❌ Error processing item: {e}", 'error')
            import traceback
            log_and_print(traceback.format_exc(), 'error')
            self.error_count += 1

    def _process_ex21_filing(self, item_data, filing_entry):
        """Process 8-K filing with EX-2.1 (one document from filing_array)."""
        try:
            log_and_print("📄 Processing EX-2.1 filing")
            url_ex21 = filing_entry.get('url')
            if not url_ex21:
                log_and_print(
                    "⚠️ No EX-2.1 document URL in filing entry", 'warning')
                return

            # Set xbrl_files to this filing for analyzer
            data = copy.deepcopy(item_data)
            data['xbrl_files'] = [
                {'type': 'EX-2.1', 'description': filing_entry.get('description', 'EX-2.1'), 'url': url_ex21}]
            # Analyze filing to get document_kind and company_details
            log_and_print(
                f"🔍 Analyzing EX-2.1 document for: {item_data.get('company_name')}")
            result = self.document_analyzer.analyze_filing(data)
            item_data.update(result)

            # Check document_kind
            document_kind = item_data.get('document_kind')
            log_and_print(f"   Document kind: {document_kind}")

            # Get company_details
            company_details = item_data.get('company_details') or {}
            is_us_listed = company_details.get('is_target_us_listed')
            market_cap_gt_100m = company_details.get(
                'is_target_market_cap_greater_than_100m')

            log_and_print(f"   US listed: {is_us_listed}")
            log_and_print(f"   Market cap > $100M: {market_cap_gt_100m}")

            # Update existing SECFiling (saved at start of _process_single_item) with document_kind and company_details
            filing = SECFiling.objects(
                accession_number=item_data.get('accession_number')).first()
            if filing:
                filing.document_kind = item_data.get('document_kind')
                filing.company_details = item_data.get('company_details')
                filing.save()
            else:
                filing = self._save_filing(item_data)
            if not filing:
                log_and_print("⚠️ Failed to save/update filing", 'warning')
                self.error_count += 1
                return

            # Send email
            self._send_ex21_email(
                item_data, company_details, filing_entry=filing_entry)

            # Process EX-2.1 via 8-K document helper (Node API) if US-related and market cap > $100M
            if is_us_listed and market_cap_gt_100m:
                # Send historical 8-K filings email (last 1 year)
                log_and_print(
                    "✅ Qualified for 8-K EX-2.1 document processing (US-listed + market cap > $100M)")
                self._send_historical_8k_email(item_data)
                self._process_ex21_via_8k_helper(item_data, filing)
                self.ex21_processed_count += 1
            else:
                log_and_print(
                    "⏭️ Not qualified for 8-K EX-2.1 document processing (US-listed or market cap criteria not met)")

            self.processed_count += 1

        except Exception as e:
            log_and_print(f"❌ Error processing EX-2.1 filing: {e}", 'error')
            import traceback
            log_and_print(traceback.format_exc(), 'error')
            self.error_count += 1

    def _process_ex99_filing(self, item_data, filing_entry):
        """
        EX-99.1 document flow:
        - If cik_matches_deal: generate summary, save to sec_filing_summary, send summary email (no GPT).
        - If not cik_matches_deal: run GPT; send main EX-99.1 email only when all of
          is_merger_related, is_target_us_listed, is_target_market_cap_greater_than_100m are truthy.
        """
        try:
            log_and_print("📄 Processing EX-99.1 filing")
            url_ex99 = filing_entry.get('url')
            if not url_ex99:
                log_and_print(
                    "⚠️ No EX-99.1 document URL in filing entry", 'warning')
                return

            if item_data.get('cik_matches_deal'):
                log_and_print(
                    "✅ CIK matches deal → generating EX-99.1 summary (no GPT)")
                self._generate_ex99_summary(item_data, url_ex99)
                self.processed_count += 1
                self.ex99_processed_count += 1
                return

            data = copy.deepcopy(item_data)
            data['xbrl_files'] = [
                {'type': 'EX-99.1', 'description': filing_entry.get('description', 'EX-99.1'), 'url': url_ex99}]
            log_and_print(
                f"🔍 EX-99.1: GPT analysis: {item_data.get('company_name')}")
            result = self.document_analyzer.analyze_ex99_1_filing(data)
            item_data.update(result)

            is_merger_related = item_data.get('is_merger_related')
            is_listed = item_data.get('is_target_us_listed')
            market_cap_gt_100m = item_data.get(
                'is_target_market_cap_greater_than_100m')
            log_and_print(f"   Merger-related: {is_merger_related}")
            log_and_print(f"   Market cap > $100M: {market_cap_gt_100m}")
            log_and_print(
                f"   Confidence: {item_data.get('confidence', 0)}%")

            # Send EX-99.1 email only if all three are truthy; otherwise skip
            if is_merger_related and is_listed and market_cap_gt_100m:
                self._send_ex99_email(item_data, filing_entry=filing_entry)
            else:
                log_and_print(
                    "⏭️ EX-99.1 main email skipped (need is_merger_related, is_target_us_listed, and market cap > $100M)")

            self.processed_count += 1
            self.ex99_processed_count += 1

        except Exception as e:
            log_and_print(
                f"❌ Error processing EX-99.1 filing: {e}", 'error')
            import traceback
            log_and_print(traceback.format_exc(), 'error')
            self.error_count += 1

    def _save_filing(self, item_data):
        """Save filing to SECFiling database"""
        try:
            accession_number = item_data.get('accession_number')
            if not accession_number:
                log_and_print(
                    "Cannot save filing without accession_number", 'warning')
                return None

            # Check if already exists (race condition)
            existing = SECFiling.objects(
                accession_number=accession_number).first()
            if existing:
                log_and_print(
                    f"⏭️ Filing already exists: {accession_number}")
                return existing

            # Prepare filing data
            item_data = self._prepare_filing_data(item_data)

            # Filter allowed fields
            filing_data = {k: v for k, v in item_data.items()
                           if k in ALLOWED_FILING_FIELDS}

            # Create and save filing
            filing = SECFiling(**filing_data)
            try:
                filing.save()
                log_and_print(
                    f"💾 Saved filing: {item_data.get('company_name')} - {accession_number}")

                # Emit WebSocket event
                filing_ws_data = {
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
                SECWebSocketService.emit_new_sec_filing(filing_ws_data)

                return filing

            except (NotUniqueError, Exception) as e:
                if 'E11000' in str(e) or 'duplicate' in str(e).lower():
                    log_and_print(
                        f"⏭️ Duplicate accession_number (race): {accession_number}", 'warning')
                    # Try to fetch existing
                    return SECFiling.objects(accession_number=accession_number).first()
                raise

        except Exception as e:
            log_and_print(f"❌ Error saving filing: {e}", 'error')
            return None

    def _prepare_filing_data(self, item_data):
        """Prepare item_data for database save"""
        # Convert acceptance_datetime_utc
        if item_data.get('acceptance_datetime_utc') and isinstance(item_data['acceptance_datetime_utc'], str):
            try:
                item_data['acceptance_datetime_utc'] = datetime.fromisoformat(
                    item_data['acceptance_datetime_utc'].replace('Z', '+00:00'))
            except Exception as e:
                log_and_print(
                    f"Error converting acceptance_datetime_utc: {e}", 'warning')
                item_data['acceptance_datetime_utc'] = None

        # Convert filing_date
        if item_data.get('filing_date'):
            item_data['filing_date'] = parse_filing_date(
                item_data['filing_date'])

        # Set has_htm_files (filing_array entries are all .htm; legacy has_ex21/has_ex99_1)
        if item_data.get('filing_array') or item_data.get('has_ex21') or item_data.get('has_ex99_1'):
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

        # Set following defaults
        if 'following' not in item_data:
            item_data['following'] = False
        if 'following_status' not in item_data:
            item_data['following_status'] = None

        return item_data

    def _send_ex21_email(self, item_data, company_details, filing_entry=None):
        """Send email for EX-2.1 filing. If filing_entry is provided, use it for doc table (file/size from SEC)."""
        try:
            log_and_print("📧 Generating and sending EX-2.1 email")

            # Build doc_files for template (match SEC table when filing_entry available)
            if filing_entry:
                doc_files = [{
                    'type': filing_entry.get('document_type', 'EX-2.1'),
                    'url': filing_entry.get('url'),
                    'description': filing_entry.get('description', ''),
                    'file': filing_entry.get('file', ''),
                    'size': filing_entry.get('size', 0),
                    'sequence': filing_entry.get('sequence', 0),
                }]
            else:
                doc_files = item_data.get('xbrl_files') or []

            # Generate email HTML
            subject, html_email = generate_filing_email_html(
                item_data, doc_files)

            # Prepare payload
            payload = {
                'subject': subject,
                'html': html_email,
                'company_name': item_data.get('company_name', 'Unknown Company'),
                'accession_number': item_data.get('accession_number', 'N/A'),
                'form_type': item_data.get('form_type', 'N/A'),
                'filing_url': item_data.get('link', ''),
                'email_type': 'ex21_merger',
            }

            # Choose webhook based on company_details
            use_filing_webhook = (
                company_details.get('is_target_us_listed') and
                company_details.get('is_target_market_cap_greater_than_100m')
            )
            webhook_url = (
                N8N_WEBHOOK_URL_FILING if use_filing_webhook
                else N8N_WEBHOOK_URL_8K_SUMMARY
            )

            send_webhook_notification(webhook_url, payload, "EX-2.1 email")
            log_and_print("✅ EX-2.1 email sent successfully")

        except Exception as e:
            log_and_print(
                f"❌ Error sending EX-2.1 email: {e}", 'error')

    def _send_historical_8k_email(self, item_data):
        """
        Send email with all 8-K filings from the last 1 year.

        This provides historical context for the company's recent 8-K filing activity.
        Similar to services.py logic for EX-2.1 filings.
        """
        try:
            cik_number = item_data.get('cik_number')
            filing_date = item_data.get('filing_date')
            company_name = item_data.get('company_name', 'Unknown Company')

            if not cik_number or not filing_date:
                log_and_print(
                    "⏭️ Skipping historical 8-K email: missing CIK or filing date",
                    'warning'
                )
                return

            # Parse filing date if it's a string
            if not isinstance(filing_date, datetime):
                filing_date = parse_filing_date(filing_date)

            if not filing_date:
                log_and_print(
                    "⏭️ Skipping historical 8-K email: invalid filing date",
                    'warning'
                )
                return

            # Fetch 8-K filings from 1 year before filing date
            start_date = (filing_date - timedelta(days=365)
                          ).strftime('%Y-%m-%d')

            log_and_print(
                f"🔍 Fetching historical 8-K filings for CIK {cik_number} from {start_date}..."
            )

            filings = fetch_sec_filings(
                str(cik_number),
                start_date=start_date,
                form_types=None
            )

            if not filings:
                log_and_print(
                    f"⚠️ No historical 8-K filings found for CIK {cik_number}",
                    'warning'
                )
                return

            log_and_print(f"📥 Found {len(filings)} historical 8-K filings")

            # Generate email HTML
            sec_subject, sec_html = generate_sec_filings_email_html(
                company_name,
                filings,
                form_type="8-K(EX-2.1)"
            )

            # Prepare payload
            sec_payload = {
                'subject': sec_subject,
                'html': sec_html,
                'company_name': company_name,
                'email_type': 'sec_filings_last_year',
            }

            # Send email
            send_webhook_notification(
                N8N_WEBHOOK_URL_8K_SUMMARY,
                sec_payload,
                "Historical 8-K filings email"
            )

            log_and_print(
                f"📤 Sent historical 8-K filings email: {len(filings)} filings for {company_name}"
            )

        except Exception as e:
            log_and_print(
                f"❌ Error sending historical 8-K email: {e}",
                'error'
            )

    def _send_ex99_email(self, item_data, filing_entry=None):
        """Send email for EX-99.1 filing. If filing_entry is provided, use it for doc table (file/size from SEC table)."""
        try:
            log_and_print("📧 Generating and sending EX-99.1 email")

            # Use filing_entry so document name and size match SEC table; fallback to xbrl_files
            if filing_entry:
                doc_files = [{
                    'type': filing_entry.get('document_type', 'EX-99.1'),
                    'url': filing_entry.get('url'),
                    'description': filing_entry.get('description', ''),
                    'file': filing_entry.get('file', ''),
                    'size': filing_entry.get('size', 0),
                    'sequence': filing_entry.get('sequence', 0),
                }]
                log_and_print(
                    f"   doc_files for EX-99.1 email: file={doc_files[0].get('file')!r} size={doc_files[0].get('size')}")
            else:
                doc_files = item_data.get('xbrl_files', [])
            subject, html_email = generate_ex99_1_merger_email_html(
                item_data, doc_files)

            # Prepare payload
            payload = {
                'subject': subject,
                'html': html_email,
                'company_name': item_data.get('company_name', 'Unknown Company'),
                'accession_number': item_data.get('accession_number', 'N/A'),
                'form_type': item_data.get('form_type', 'N/A'),
                'filing_url': item_data.get('link', ''),
                'email_type': 'ex99_1_merger',
            }

            # Always use 8K summary webhook for EX-99.1
            send_webhook_notification(
                N8N_WEBHOOK_URL_8K_SUMMARY, payload, "EX-99.1 email")
            log_and_print("✅ EX-99.1 email sent successfully")

        except Exception as e:
            log_and_print(
                f"❌ Error sending EX-99.1 email: {e}", 'error')

    def _process_ex21_via_8k_helper(self, item_data, filing):
        """Process EX-2.1 filing via 8-K document helper (Node API deal/process-with-url), same as services.py."""
        try:
            log_and_print(
                "🚀 Starting 8-K EX-2.1 document processing")

            ex21_file = find_file_by_type(
                item_data.get('xbrl_files', []), 'EX-2.1')
            if not ex21_file:
                log_and_print(
                    "❌ No EX-2.1 file found in xbrl_files", 'error')
                return

            ex21_url = build_full_sec_url(ex21_file.get('url'))
            if not ex21_url:
                log_and_print("❌ Could not build EX-2.1 URL", 'error')
                return

            cik_number = item_data.get('cik_number', '')
            company_name = item_data.get('company_name', '')
            sec_filing_id = str(filing._id)
            company_details = item_data.get('company_details')
            filing_date = item_data.get('filing_date')
            filing_date_obj = parse_filing_date(
                filing_date) if filing_date else None

            log_and_print(f"   EX-2.1 URL: {ex21_url}")
            log_and_print(f"   SEC Filing ID: {sec_filing_id}")

            result = process_8k_document_helper(
                cik_number=cik_number,
                company_name=company_name,
                sec_filing_id=sec_filing_id,
                filing_date=filing_date_obj,
                form_type='8-K',
                ex21_url=ex21_url,
                item_data=item_data,
                company_details=company_details,
            )

            if result:
                log_and_print(
                    f"✅ 8-K EX-2.1 document processing started: {result.get('message', 'Processing started')}")
                log_and_print(f"   Status: {result.get('status')}")
            else:
                log_and_print(
                    "❌ Failed to start 8-K EX-2.1 document processing", 'error')

        except Exception as e:
            log_and_print(
                f"❌ Error processing EX-2.1 via 8-K helper: {e}", 'error')
            import traceback
            log_and_print(traceback.format_exc(), 'error')

    def _generate_ex99_summary(self, item_data, url_ex99):
        """Generate summary for EX-99.1 document and save to sec_filing_summary."""
        try:
            log_and_print("📝 Generating summary for EX-99.1 document")

            accession_number = item_data.get('accession_number')
            deal_id = item_data.get('deal_id')
            output_dir = tempfile.mkdtemp()

            if url_ex99:
                try:
                    log_and_print(
                        f"   Summarizing EX-99.1 document: {url_ex99}")
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

                        # One document per accession_number (form_type 8-K); EX-99.1 stored in eight_k.filings[]
                        try:
                            filing_dt = self._filing_date_for_summary(
                                item_data.get('filing_date'), result_99.get('filing_date'))
                            ex99_filing_entry = {
                                'filing_date': filing_dt,
                                'filing_url': url_ex99,
                                's3_docx_url': result_99.get('s3_url'),
                                's3_json_url': result_99.get('s3_json_url'),
                                'exhibit_type': 'EX_99.1',
                            }
                            existing = SECFilingSummary.objects(
                                accession_number=accession_number, form_type='8-K'
                            ).first()
                            if existing and existing.eight_k:
                                filings = list(
                                    existing.eight_k.get('filings') or [])
                                filings.append(ex99_filing_entry)
                                existing.eight_k = dict(existing.eight_k)
                                existing.eight_k['filings'] = filings
                                existing.save()
                            else:
                                # No 8-K doc yet: create one with EX-99.1 in filings only
                                url_8k = None
                                for fe in item_data.get('filing_array') or []:
                                    if fe.get('document_type') == '8-K' and fe.get('url'):
                                        url_8k = fe.get('url')
                                        break
                                doc_8k = SECFilingSummary(
                                    form_type='8-K',
                                    accession_number=accession_number,
                                    cik_number=item_data.get('cik_number'),
                                    sec_document_url=url_8k or url_ex99,
                                    filing_date=filing_dt,
                                    deal_id=deal_id,
                                    eight_k={
                                        'one_line_summary': None,
                                        'items_reported': [],
                                        's3_docx_url': None,
                                        's3_json_url': None,
                                        'filings': [ex99_filing_entry],
                                    },
                                )
                                doc_8k.save()
                            log_and_print(
                                "💾 EX-99.1 summary saved to DB (sec_filing_summary, inside eight_k.filings)")
                            self.summary_ex99_count += 1

                            # Send summary email
                            self._send_ex99_summary_email(
                                item_data, result_99, url_ex99)

                        except Exception as db_e:
                            log_and_print(
                                f"❌ Failed to save EX-99.1 summary to DB: {db_e}", 'error')
                except Exception as e:
                    log_and_print(
                        f"❌ EX-99.1 summary generation failed: {e}", 'error')
        except Exception as e:
            log_and_print(f"❌ Error in _generate_ex99_summary: {e}", 'error')

    def _send_8k_summary_email(self, item_data, summary_result, doc_url):
        """Send email with 8-K summary document link"""
        try:
            log_and_print("📧 Sending 8-K summary email")

            from .email_templates import generate_8k_99_1_summary_email_html

            subject, html_email = generate_8k_99_1_summary_email_html(
                company_name=item_data.get('company_name') or '',
                form_type='8-K',
                summary_doc_url=summary_result.get('s3_url'),
                cik_number=item_data.get('cik_number') or '',
                sec_url=item_data.get('link') or doc_url,
                accession_number=item_data.get('accession_number') or '',
                summary_kind='8-K',
                l1_headline=summary_result.get('L1_headline'),
            )

            payload = {
                'subject': subject,
                'html': html_email,
                'company_name': item_data.get('company_name', 'Unknown Company'),
                'form_type': '8-K',
                'summary_doc_url': summary_result.get('s3_url'),
                'accession_number': item_data.get('accession_number'),
                'cik_number': item_data.get('cik_number'),
                'sec_url': doc_url,
            }

            send_webhook_notification(
                N8N_WEBHOOK_URL_8K_SUMMARY, payload, "8-K summary email")
            log_and_print("✅ 8-K summary email sent successfully")

        except Exception as e:
            log_and_print(f"❌ Error sending 8-K summary email: {e}", 'error')

    def _send_ex99_summary_email(self, item_data, summary_result, doc_url):
        """Send email with EX-99.1 summary document link"""
        try:
            log_and_print("📧 Sending EX-99.1 summary email")

            from .email_templates import generate_8k_99_1_summary_email_html

            subject, html_email = generate_8k_99_1_summary_email_html(
                company_name=item_data.get('company_name') or '',
                form_type='8-K (EX-99.1)',
                summary_doc_url=summary_result.get('s3_url'),
                cik_number=item_data.get('cik_number') or '',
                sec_url=item_data.get('link') or doc_url,
                accession_number=item_data.get('accession_number') or '',
                summary_kind='EX-99.1',
                l1_headline=summary_result.get('L1_headline'),
            )

            payload = {
                'subject': subject,
                'html': html_email,
                'company_name': item_data.get('company_name', 'Unknown Company'),
                'form_type': '8-K (EX-99.1)',
                'summary_doc_url': summary_result.get('s3_url'),
                'accession_number': item_data.get('accession_number'),
                'cik_number': item_data.get('cik_number'),
                'sec_url': doc_url,
            }

            send_webhook_notification(
                N8N_WEBHOOK_URL_8K_SUMMARY, payload, "EX-99.1 summary email")
            log_and_print("✅ EX-99.1 summary email sent successfully")

        except Exception as e:
            log_and_print(
                f"❌ Error sending EX-99.1 summary email: {e}", 'error')


def run_8k_processor(rss_content=None, rss_file=None):
    """
    Entry point to run the 8-K processor.

    For testing with manual RSS:
      run_8k_processor(rss_file='sec_rss_parser/rss.xml')
      run_8k_processor(rss_content=open('sec_rss_parser/rss.xml').read())
    """
    processor = EightKFeedProcessor()
    return processor.run(rss_content=rss_content, rss_file=rss_file)
