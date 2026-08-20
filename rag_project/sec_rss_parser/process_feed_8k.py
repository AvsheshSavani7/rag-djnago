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
import re
import os
from datetime import datetime, timedelta
from mongoengine.errors import NotUniqueError
from mongoengine.queryset.visitor import Q


from .utils_8k import (
    SECRSSParser,
    extract_accession_from_guid,
    build_full_sec_url,
    find_file_by_type,
    get_deal_tickers,
    get_ticker_for_deal_and_cik,
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
    generate_filing_email_with_deal_html,
    generate_8k_document_email_html,
    generate_ex99_1_merger_email_html,
    generate_sec_filings_email_html,
    generate_item_5_02_one_year_filings_email_html,
)
from .sec_summarizers.filing_router import route_and_summarize
from .sec_Last_Year import print_filings as fetch_sec_filings
from document_processor.models import ProcessingJob
from .services import process_8k_document_helper
from .websocket_service import SECWebSocketService
from .accession_lock import (
    acquire_accession_lock,
    mark_accession_processed,
    release_accession_lock,
)
from sec_rss_parser.email_service.email_dispatch_service import send_report_email

logger = logging.getLogger(__name__)

# Deal status constants
DEAL_STATUS_OPEN_OR_UNKNOWN = ["Open", "Unknown"]


# Constants (from services.py)

N8N_WEBHOOK_URL_8K_SUMMARY = os.environ.get(
    "N8N_WEKHOOK_INTERNAL_WITH_JOSH", "https://n8n.arbintel.cloud/webhook/b3007d21-6845-47b5-aece-7b26583758bc")
N8N_WEBHOOK_URL_8K_SUMMARY_L123 = os.environ.get(
    "N8N_WEBHOOK_SEND_TO_ALL", "https://n8n.arbintel.cloud/webhook/3ff1b0ea-7114-4dda-940e-95ce81e08017")
N8N_WEBHOOK_URL_FILING = os.environ.get(
    "N8N_WEBHOOK_SEND_TO_ALL", "https://n8n.arbintel.cloud/webhook/3ff1b0ea-7114-4dda-940e-95ce81e08017")
MAX_DESCRIPTION_LENGTH = 50

LOG_PREFIX = "form by form_type: 8-K"

CIK_PAD_LENGTH = 10


def _extract_cik_from_url(url):
    """Extract filer CIK from SEC filing URL like /Archives/edgar/data/{CIK}/..."""
    if not url:
        return None
    m = re.search(r"/Archives/edgar/data/(\d+)", url)
    if m:
        return m.group(1).zfill(CIK_PAD_LENGTH)
    m = re.search(r"[?&]CIK=(\d+)", url, re.IGNORECASE)
    if m:
        return m.group(1).zfill(CIK_PAD_LENGTH)
    return None


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
            logger.info(f"{LOG_PREFIX} :run: step=start")
            log_and_print("=" * 80)
            log_and_print(
                f"{LOG_PREFIX} :run: 🚀 Starting 8-K-only feed processor")
            log_and_print("=" * 80)

            # Use manual RSS for testing if provided
            if rss_content is None and rss_file:
                try:
                    with open(rss_file, 'r', encoding='utf-8', errors='replace') as f:
                        rss_content = f.read()
                    log_and_print(
                        f"{LOG_PREFIX} :run: 📂 Using RSS from file: {rss_file}")
                    logger.info(
                        f"{LOG_PREFIX} :run: step=rss_source source=file path=%s", rss_file)
                except Exception as e:
                    log_and_print(
                        f"{LOG_PREFIX} :run: ❌ Failed to read RSS file {rss_file}: {e}", 'error')
                    return {
                        'success': False,
                        'error': f'Failed to read RSS file: {e}',
                        'form_type': '8-K'
                    }
            if rss_content is None:
                self.parser.set_feed_url('8-K')
                rss_content = self.parser.fetch_rss_feed()
                if rss_content:
                    logger.info(
                        f"{LOG_PREFIX} :run: step=fetch_feed source=live")
            else:
                log_and_print(
                    f"{LOG_PREFIX} :run: 📂 Using provided RSS content (skip live fetch)")
                logger.info("8k_processor_run step=rss_source source=provided")

            if not rss_content:
                log_and_print(
                    f"{LOG_PREFIX} :run: ❌ Failed to fetch 8-K RSS feed", 'error')
                return {
                    'success': False,
                    'error': 'Failed to fetch RSS feed',
                    'form_type': '8-K'
                }

            # Parse feed
            items = self.parser.parse_rss_content(rss_content)
            logger.info(
                f"{LOG_PREFIX} :run: step=parse_feed total_items=%s", len(items))
            log_and_print(f"📋 Parsed {len(items)} items from 8-K feed")

            if not items:
                log_and_print(
                    f"{LOG_PREFIX} :run: ⚠️ No items found in 8-K feed")
                return {
                    'success': True,
                    'message': 'No items in feed',
                    'form_type': '8-K',
                    'total_items': 0
                }

            # Filter by accession (check AccessionLookedUp)
            unique_items = self._filter_unique_items(items)
            logger.info(f"{LOG_PREFIX} :run: step=filter_accessions total=%s new_to_process=%s", len(
                items), len(unique_items))
            log_and_print(
                f"{LOG_PREFIX} :run: ✅ Found {len(unique_items)} new items to process (after accession filtering)")

            # Process each unique item
            for item_data in unique_items:
                self._process_single_item(item_data)

            # Log summary
            logger.info(
                f"{LOG_PREFIX} :run: step=summary total_items=%s new_items=%s ex21=%s ex99=%s summary_8k=%s summary_ex99=%s skipped=%s errors=%s",
                len(items), self.processed_count, self.ex21_processed_count, self.ex99_processed_count,
                self.summary_8k_count, self.summary_ex99_count, self.skipped_count, self.error_count
            )
            log_and_print("=" * 80)
            log_and_print(f"{LOG_PREFIX} :run: 📊 8-K Processing Summary:")
            log_and_print(
                f"{LOG_PREFIX} :run:    Total items in feed: {len(items)}")
            log_and_print(
                f"{LOG_PREFIX} :run:    New items processed: {self.processed_count}")
            log_and_print(
                f"{LOG_PREFIX} :run:    EX-2.1 processed: {self.ex21_processed_count}")
            log_and_print(
                f"{LOG_PREFIX} :run:    EX-99.1 processed: {self.ex99_processed_count}")
            log_and_print(
                f"{LOG_PREFIX} :run:   8-K summaries generated: {self.summary_8k_count}")
            log_and_print(
                f"{LOG_PREFIX} :run:   EX-99.1 summaries generated: {self.summary_ex99_count}")
            log_and_print(
                f"{LOG_PREFIX} :run:   Skipped: {self.skipped_count}")
            # log_and_print(f"{LOG_PREFIX} :run:   Errors: {self.error_count}")
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
            logger.exception(f"{LOG_PREFIX} :run: step=error error=%s", str(e))
            log_and_print(
                f"{LOG_PREFIX} :run: ❌ Error in 8-K feed processor: {e}", 'error')
            import traceback
            log_and_print(traceback.format_exc(), 'error')
            return {
                'success': False,
                'error': str(e),
                'form_type': '8-K'
            }

    def _item_cik_for_filter(self, item_data):
        """CIK from item fields or filing link (RSS entries often lack cik_number)."""
        cik = item_data.get('cik_number')
        if cik:
            return normalize_cik(cik)
        return _extract_cik_from_url(item_data.get('link') or '')

    def _pick_item_for_accession(self, accession_number, candidates):
        """
        Prefer the candidate whose CIK matches an open/unknown deal we follow.
        If none match (or only one candidate), use the first feed row.
        Still returns exactly one item so accession remains unique downstream.
        """
        if len(candidates) == 1:
            return candidates[0], "single"

        for item_data in candidates:
            cik = self._item_cik_for_filter(item_data)
            matches, deal_id = self._check_cik_matches_deal(cik)
            if matches:
                logger.info(
                    f"{LOG_PREFIX} :_filter_unique_items: accession=%s reason=deal_cik_preferred "
                    f"cik=%s deal_id=%s candidates=%s",
                    accession_number, cik, deal_id, len(candidates),
                )
                log_and_print(
                    f"{LOG_PREFIX} :_filter_unique_items: ✅ Prefer deal CIK {cik} "
                    f"for accession {accession_number} (deal={deal_id}, "
                    f"skipped {len(candidates) - 1} other CIK row(s))"
                )
                return item_data, "deal_cik_preferred"

        logger.info(
            f"{LOG_PREFIX} :_filter_unique_items: accession=%s reason=first_of_duplicates "
            f"candidates=%s",
            accession_number, len(candidates),
        )
        log_and_print(
            f"{LOG_PREFIX} :_filter_unique_items: ⏭️ No deal CIK for accession "
            f"{accession_number}; using first of {len(candidates)} rows"
        )
        return candidates[0], "first_of_duplicates"

    def _filter_unique_items(self, items):
        """Filter items to only include new accession numbers.

        When the same accession appears with different CIKs in one feed, prefer
        the row whose CIK matches an open/unknown deal; otherwise keep the first.
        Still emits at most one row per accession (final store stays unique).

        Do not add to AccessionLookedUp here; add only after successful processing
        in _process_single_item so read timeouts / failures can be retried next run.
        """
        unique_items = []
        by_accession = {}
        order = []

        for item_data in items:
            accession_number = item_data.get(
                'accession_number') or extract_accession_from_guid(item_data.get('guid'))

            if not accession_number:
                logger.warning(f"{LOG_PREFIX} :_filter_unique_items: accession=missing title=%s",
                               (item_data.get('title') or 'N/A')[:80])
                log_and_print(
                    f"{LOG_PREFIX} :_filter_unique_items: ⚠️ Item without accession number: {item_data.get('title', 'N/A')}", 'warning')
                unique_items.append(item_data)
                continue

            if accession_number not in by_accession:
                by_accession[accession_number] = []
                order.append(accession_number)
            by_accession[accession_number].append(item_data)

        for accession_number in order:
            candidates = by_accession[accession_number]

            # Check AccessionLookedUp cache
            if AccessionLookedUp.objects(accession_number=accession_number).first():
                logger.info(
                    f"{LOG_PREFIX} :_filter_unique_items: accession=%s reason=already_looked_up", accession_number)
                log_and_print(
                    f"⏭️ Skipping already looked up filing: {accession_number}")
                continue

            # Check SECFiling database
            if SECFiling.objects(accession_number=accession_number).first():
                logger.info(
                    f"{LOG_PREFIX} :_filter_unique_items: accession=%s reason=existing_filing", accession_number)
                log_and_print(
                    f"{LOG_PREFIX} :_filter_unique_items: ⏭️ Skipping existing filing: {accession_number}")
                # Cache for future runs
                try:
                    AccessionLookedUp(accession_number=accession_number).save()
                except Exception as e:
                    if 'duplicate' not in str(e).lower() and 'E11000' not in str(e):
                        log_and_print(
                            f"{LOG_PREFIX} :_filter_unique_items: Failed to save accession number to AccessionLookedUp: {e}", 'warning')
                continue

            chosen, pick_reason = self._pick_item_for_accession(
                accession_number, candidates)
            unique_items.append(chosen)
            logger.info(
                f"{LOG_PREFIX} :_filter_unique_items: accession=%s reason=queued "
                f"pick=%s cik=%s candidates=%s",
                accession_number,
                pick_reason,
                self._item_cik_for_filter(chosen),
                len(candidates),
            )

        return unique_items

    def _check_cik_matches_deal(self, cik_number):
        """Check if CIK matches any deal (target or acquirer)"""
        if not cik_number:
            return False, None

        try:
            cik_normalized = normalize_cik(cik_number)

            # Match: deal_status in Open/Unknown, or deal_status is null/missing
            deal_status_filter = (
                Q(deal_status__in=DEAL_STATUS_OPEN_OR_UNKNOWN)
                | Q(deal_status=None)
                | Q(deal_status__exists=False)
            )

            # Check as target CIK
            matched_deal = ProcessingJob.objects(
                Q(cik=cik_normalized) & deal_status_filter
            ).first()

            # Check as acquirer CIK
            if not matched_deal:
                matched_deal = ProcessingJob.objects(
                    Q(acquirer_cik=cik_normalized) & deal_status_filter
                ).first()

            if matched_deal:
                logger.info(f"{LOG_PREFIX} :_check_cik_matches_deal: cik=%s match=true deal_id=%s",
                            cik_normalized, str(matched_deal.id))
                log_and_print(
                    f"{LOG_PREFIX} :_check_cik_matches_deal: ✅ CIK {cik_normalized} matches deal: {matched_deal.id}")
                return True, str(matched_deal.id)
            else:
                logger.info(
                    f"{LOG_PREFIX} :_check_cik_matches_deal: cik=%s match=false", cik_normalized)
                log_and_print(
                    f"{LOG_PREFIX} :_check_cik_matches_deal: ⏭️ CIK {cik_normalized} not in deals")
                return False, None

        except Exception as e:
            logger.exception(
                f"{LOG_PREFIX} :_check_cik_matches_deal: cik=%s error=%s", cik_number, str(e))
            log_and_print(
                f"{LOG_PREFIX} :_check_cik_matches_deal: ❌ Error checking CIK in deals: {e}", 'error')
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
        accession_number = item_data.get('accession_number', 'N/A')
        try:
            url_8k = filing_entry.get('url')
            logger.info("process_8k_document accession=%s step=start url=%s",
                        accession_number, url_8k or '')
            if not url_8k:
                logger.warning(
                    f"{LOG_PREFIX} :_process_8k_document: step=start accession=%s step=skip reason=no_url", accession_number)
                log_and_print(
                    f"{LOG_PREFIX} :_process_8k_document: ⚠️ No 8-K document URL in filing entry", 'warning')
                return

            if item_data.get('cik_matches_deal'):
                logger.info(
                    f"{LOG_PREFIX} :_process_8k_document: accession=%s step=cik_matches_deal generating_summary", accession_number)
                log_and_print(
                    f"{LOG_PREFIX} :_process_8k_document: ✅ CIK matches deal → generating 8-K summary (no GPT)")
                self._generate_8k_summary_and_send(item_data, url_8k)
                return

            logger.info(f"{LOG_PREFIX} :_process_8k_document: accession=%s step=gpt_analysis company=%s",
                        accession_number, item_data.get('company_name', ''))
            log_and_print(
                f"{LOG_PREFIX} :_process_8k_document: 🔍 8-K: GPT analysis: {item_data.get('company_name')}")
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
            item_data['target_market_cap_usd'] = result.get(
                'target_market_cap_usd')
            logger.info(
                f"{LOG_PREFIX} :_process_8k_document: accession=%s step=gpt_result is_merger=%s is_us_listed=%s cap_gt_100m=%s confidence=%s",
                accession_number,
                item_data.get('is_merger_related'),
                item_data.get('is_target_us_listed'),
                item_data.get('is_target_market_cap_greater_than_100m'),
                item_data.get('confidence', 0),
            )
            log_and_print(
                f"{LOG_PREFIX} :_process_8k_document:   8-K Merger-related: {item_data.get('is_merger_related', False)}")
            log_and_print(
                f"{LOG_PREFIX} :_process_8k_document:   Market cap > $100M: {item_data.get('is_target_market_cap_greater_than_100m')}")
            log_and_print(
                f"{LOG_PREFIX} :_process_8k_document:   Confidence: {item_data.get('confidence', 0)}%")

            # Send 8-K email only if all three are truthy; otherwise skip
            is_merger = item_data.get('is_merger_related')
            is_listed = item_data.get('is_target_us_listed')
            cap_gt_100m = item_data.get(
                'is_target_market_cap_greater_than_100m')
            if is_merger and is_listed and cap_gt_100m:
                logger.info(
                    f"{LOG_PREFIX} :_process_8k_document: accession=%s step=send_8k_gpt_email", accession_number)
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
                logger.info(
                    f"{LOG_PREFIX} :_process_8k_document: accession=%s step=email_skipped reason=criteria_not_met", accession_number)
                log_and_print(
                    f"{LOG_PREFIX} :_process_8k_document: ⏭️ 8-K main email skipped (need is_merger_related, is_target_us_listed, and market cap > $100M)")
        except Exception as e:
            logger.exception(
                f"{LOG_PREFIX} :_process_8k_document: accession=%s step=error error=%s", accession_number, str(e))
            log_and_print(
                f"{LOG_PREFIX} :_process_8k_document: ❌ Error in _process_8k_document: {e}", 'error')

    def _send_8k_gpt_email(self, item_data, doc_files=None):
        """Send email for main 8-K document (subject: 8-K – Company Name). Distinct from EX-99.1 email."""
        accession_number = item_data.get('accession_number', 'N/A')
        try:
            logger.info(f"{LOG_PREFIX} :_send_8k_gpt_email: accession=%s company=%s",
                        accession_number, item_data.get('company_name', ''))
            log_and_print(
                f"{LOG_PREFIX} :_send_8k_gpt_email: 📧 Sending 8-K email with GPT data")
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
            # send_webhook_notification(
            #     N8N_WEBHOOK_URL_8K_SUMMARY, payload, "8-K GPT email"
            # )  # TODO: comment out after org-aware send is stable
            logger.info(f"{LOG_PREFIX} :_send_8k_gpt_email: accession=%s step=sent",
                        accession_number)
            log_and_print(
                f"{LOG_PREFIX} :_send_8k_gpt_email: ✅ 8-K GPT email sent")

            send_report_email(
                report_type="sec_new_deal_probably_announced",
                payload=payload
            )
            logger.info(f"{LOG_PREFIX} :_send_8k_gpt_email: accession=%s step=org_aware_sent",
                        accession_number)
            log_and_print(
                f"{LOG_PREFIX} :_send_8k_gpt_email: ✅ Org-aware email sent (sec_new_deal_probably_announced)")
        except Exception as e:
            logger.exception(
                f"{LOG_PREFIX} :_send_8k_gpt_email: accession=%s error=%s", accession_number, str(e))
            log_and_print(
                f"{LOG_PREFIX} :_send_8k_gpt_email: ❌ Error sending 8-K GPT email: {e}", 'error')

    def _generate_8k_summary_and_send(self, item_data, url_8k):
        """Generate 8-K summary via filing router, save to sec_filing_summary (parent-level), send summary email."""
        accession_number = item_data.get('accession_number', 'N/A')
        try:
            deal_id = item_data.get('deal_id')
            cik_number = item_data.get('cik_number')
            logger.info(f"{LOG_PREFIX} :_generate_8k_summary_and_send: accession=%s step=start url=%s deal_id=%s",
                        accession_number, url_8k, deal_id)
            log_and_print(
                f"{LOG_PREFIX} :_generate_8k_summary_and_send: 📝 Generating 8-K summary: {url_8k}")
            deal_tickers = get_deal_tickers(deal_id, cik_number)
            deal_context = {
                "primary_ticker":  deal_tickers.get("ticker"),
                "target_ticker":   deal_tickers.get("target_ticker"),
                "target_name":     deal_tickers.get("target_name"),
                "acquirer_ticker": deal_tickers.get("acquirer_ticker"),
                "acquirer_name":   deal_tickers.get("acquirer_name"),
            } if any(deal_tickers.values()) else None
            result_8k = route_and_summarize(url_8k, deal_context=deal_context)
            s3_url = result_8k.get('s3_docx_url') or result_8k.get('s3_url')
            if not s3_url:
                logger.warning(
                    f"{LOG_PREFIX} :_generate_8k_summary_and_send: accession=%s step=skip reason=no_s3_url", accession_number)
                return
            logger.info(f"{LOG_PREFIX} :_generate_8k_summary_and_send: accession=%s step=s3_uploaded s3_url=%s",
                        accession_number, (s3_url or '')[:80])
            log_and_print(
                f"{LOG_PREFIX} :_generate_8k_summary_and_send: ✅ 8-K summary uploaded to S3: {s3_url}")
            try:
                filing_dt = self._filing_date_for_summary(
                    item_data.get('filing_date'), result_8k.get('filing_date'))
                existing = SECFilingSummary.objects(
                    accession_number=accession_number, form_type='8-K'
                ).first()
                if existing:
                    existing.sec_document_url = url_8k
                    existing.filing_date = filing_dt
                    existing.deal_id = deal_id
                    existing.items_reported = result_8k.get(
                        'items_reported') or []
                    existing.L1_headline = result_8k.get('L1_headline')
                    existing.L2_brief = result_8k.get('L2_brief')
                    existing.L3_detailed = result_8k.get('L3_detailed') or {}
                    existing.s3_docx_url = result_8k.get(
                        's3_docx_url') or result_8k.get('s3_url')
                    existing.s3_json_url = result_8k.get('s3_json_url')
                    existing.save()
                else:
                    doc_8k = SECFilingSummary(
                        form_type='8-K',
                        accession_number=accession_number,
                        cik_number=item_data.get('cik_number'),
                        sec_document_url=url_8k,
                        filing_date=filing_dt,
                        deal_id=deal_id,
                        items_reported=result_8k.get('items_reported') or [],
                        L1_headline=result_8k.get('L1_headline'),
                        L2_brief=result_8k.get('L2_brief'),
                        L3_detailed=result_8k.get('L3_detailed') or {},
                        s3_docx_url=result_8k.get(
                            's3_docx_url') or result_8k.get('s3_url'),
                        s3_json_url=result_8k.get('s3_json_url'),
                    )
                    doc_8k.save()
                logger.info(
                    f"{LOG_PREFIX} :_generate_8k_summary_and_send: accession=%s step=db_saved", accession_number)
                log_and_print(
                    f"{LOG_PREFIX} :_generate_8k_summary_and_send: 💾 8-K summary saved to DB (sec_filing_summary)")
                self.summary_8k_count += 1

                # If 8-K has Item 5.02 (or 5.02): fetch one-year filings and send separate email (same template as fetch_sec_feed_by_deal_cik)
                # items_reported = result_8k.get('items_reported') or []
                # _has_item_502 = any(
                #     'Item 5.02' in str(i) or str(i).strip() == '5.02'
                #     for i in items_reported
                # )
                # cik_number = item_data.get('cik_number')
                # company_name = item_data.get(
                #     'company_name') or 'Unknown Company'
                # if _has_item_502 and cik_number:
                #     try:
                #         start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                #         filings = fetch_sec_filings(str(cik_number), start_date=start_date)
                #         ticker_item502 = get_ticker_for_deal_and_cik(deal_id, cik_number)
                #         sec_subject, sec_html = generate_item_5_02_one_year_filings_email_html(
                #             company_name,
                #             filings,
                #             trigger_accession_number=accession_number,
                #             trigger_filing_date=filing_dt,
                #             cik_number=cik_number,
                #             ticker=ticker_item502,
                #         )
                #         payload = {
                #             'subject': sec_subject,
                #             'html': sec_html,
                #             'company_name': company_name,
                #             'email_type': 'item_5_02_one_year_filings',
                #         }
                #         send_webhook_notification(
                #             N8N_WEBHOOK_URL_8K_SUMMARY, payload, 'Item 5.02 one-year filings email')
                #         log_and_print(
                #             f"{LOG_PREFIX} :_generate_8k_summary_and_send: 📤 Sent Item 5.02 one-year filings email: {len(filings)} filings for {company_name}")
                #     except Exception as item502_e:
                #         log_and_print(
                #             f"{LOG_PREFIX} :_generate_8k_summary_and_send: ❌ Item 5.02 one-year filings email failed: {item502_e}",
                #             'error')

                self._send_8k_summary_email(item_data, result_8k, url_8k)
            except Exception as db_e:
                logger.exception(
                    f"{LOG_PREFIX} :_generate_8k_summary_and_send: accession=%s step=db_error error=%s", accession_number, str(db_e))
                log_and_print(
                    f"{LOG_PREFIX} :_generate_8k_summary_and_send: ❌ Failed to save 8-K summary to DB: {db_e}", 'error')
        except Exception as e:
            logger.exception(
                f"{LOG_PREFIX} :_generate_8k_summary_and_send: accession=%s error=%s", accession_number, str(e))
            log_and_print(
                f"{LOG_PREFIX} :_generate_8k_summary_and_send: ❌ 8-K summary generation failed: {e}", 'error')

    def _process_single_item(self, item_data):
        """Process a single 8-K item"""
        accession_number = item_data.get(
            'accession_number') or extract_accession_from_guid(item_data.get('guid'))

        # Set pipeline context — propagates automatically to all threads spawned here
        from core.pipeline_logger import start_pipeline, SEC_8K
        start_pipeline(SEC_8K, accession=accession_number, doc_type="8K")

        lock_owner = None
        if accession_number:
            lock_owner = acquire_accession_lock(
                accession_number, source="process_feed_8k"
            )
            if not lock_owner:
                logger.info(
                    f"{LOG_PREFIX} :_process_single_item: accession=%s step=skip reason=lock_or_looked_up",
                    accession_number,
                )
                log_and_print(
                    f"{LOG_PREFIX} :_process_single_item: ⏭️ Skipping {accession_number} (in-progress by another worker or already finalized)"
                )
                self.skipped_count += 1
                return
        try:
            html_url = item_data.get('link')
            logger.info(f"{LOG_PREFIX} :_process_single_item: accession=%s step=start title=%s link=%s",
                        accession_number, (item_data.get('title') or 'N/A')[:60], html_url or '')

            log_and_print(f"{LOG_PREFIX} :_process_single_item: -" * 80)
            log_and_print(
                f"{LOG_PREFIX} :_process_single_item: 🔍 Processing: {item_data.get('title', 'N/A')[:80]}")
            log_and_print(
                f"{LOG_PREFIX} :_process_single_item:   Accession: {accession_number}")
            log_and_print(
                f"{LOG_PREFIX} :_process_single_item:   item_data 1: {item_data}")

            if not html_url:
                logger.warning(
                    f"{LOG_PREFIX} :_process_single_item: accession=%s step=skip reason=no_link", accession_number)
                log_and_print(
                    f"{LOG_PREFIX} :_process_single_item: ⚠️ No link URL found, skipping", 'warning')
                self.skipped_count += 1
                return

            # Fetch HTML and parse filing details
            logger.info(
                f"{LOG_PREFIX} :_process_single_item: accession=%s step=fetch_html url=%s", accession_number, html_url)
            log_and_print(
                f"{LOG_PREFIX} :_process_single_item: 🌐 Fetching filing details from: {html_url}")
            html_data = self.parser.fetch_and_parse_html(
                html_url, form_type_from_feed='8-K')

            log_and_print(
                f"{LOG_PREFIX} :_process_single_item:   html_data 1: {html_data}")

            if not html_data:
                logger.error(
                    f"{LOG_PREFIX} :_process_single_item: accession=%s step=parse_failed url=%s", accession_number, html_url)
                log_and_print(
                    f"{LOG_PREFIX} :_process_single_item: ❌ Failed to parse HTML for: {html_url}", 'error')
                self.error_count += 1
                # Index proxy failed — do not LookedUp; next processor tick retries.
                return

            # Merge HTML data into item_data; always use filer CIK from URL.
            item_data.update(html_data)
            item_data['cik_number'] = _extract_cik_from_url(
                html_url) or item_data.get('cik_number')
            filing_array = item_data.get('filing_array', [])
            logger.info(f"{LOG_PREFIX} :_process_single_item: accession=%s step=html_merged cik=%s filing_array_len=%s doc_types=%s",
                        accession_number, item_data.get('cik_number'), len(filing_array or []), [f.get('document_type') for f in (filing_array or [])])

            log_and_print(
                f"{LOG_PREFIX} :_process_single_item:   item_data 2: {item_data}")

            # CIK check early (for summary: only generate when CIK matches a deal)
            cik_number = item_data.get('cik_number')
            cik_matches_deal, deal_id = self._check_cik_matches_deal(
                cik_number)
            item_data['deal_id'] = deal_id
            item_data['cik_matches_deal'] = cik_matches_deal

            # Resolve deal party name for emails (target vs acquirer)
            if deal_id and cik_number:
                try:
                    from bson import ObjectId
                    deal = ProcessingJob.objects(id=ObjectId(deal_id)).only(
                        "cik", "acquirer_cik", "target_name", "acquire_name"
                    ).first()
                    cik_n = normalize_cik(cik_number)
                    if deal and cik_n:
                        if normalize_cik(deal.acquirer_cik) == cik_n:
                            item_data['matched_cik_label'] = "(acquirer)"
                            item_data['email_company_name'] = deal.acquire_name or item_data.get(
                                'company_name')
                        elif normalize_cik(deal.cik) == cik_n:
                            item_data['matched_cik_label'] = "(target)"
                            item_data['email_company_name'] = deal.target_name or item_data.get(
                                'company_name')
                except Exception as deal_e:
                    logger.warning(
                        f"{LOG_PREFIX} :_process_single_item: Deal name lookup failed: {deal_e}")

            logger.info(f"{LOG_PREFIX} :_process_single_item: accession=%s step=cik_check cik_matches_deal=%s deal_id=%s",
                        accession_number, cik_matches_deal, deal_id)

            log_and_print(
                f"{LOG_PREFIX} :_process_single_item:   item_data 3: {item_data}")

            filing_array = item_data.get('filing_array', [])

            log_and_print(
                f"{LOG_PREFIX} :_process_single_item:   filing_array 1: {filing_array}")
            if not filing_array:
                logger.warning(
                    f"{LOG_PREFIX} :_process_single_item: accession=%s step=skip reason=no_filing_array", accession_number)
                log_and_print(
                    f"{LOG_PREFIX} :_process_single_item: ⚠️ No 8-K / EX-2.1 / EX-99.1 .htm documents in filing, skipping", 'warning')
                self.skipped_count += 1
                return

            logger.info(f"{LOG_PREFIX} :_process_single_item: accession=%s step=save_filing form_type=%s doc_types=%s",
                        accession_number, item_data.get('form_type'), [f.get('document_type') for f in filing_array])
            log_and_print(
                f"{LOG_PREFIX} :_process_single_item:   Form type: {item_data.get('form_type', 'N/A')}")
            log_and_print(
                f"{LOG_PREFIX} :_process_single_item:   Filing array: {len(filing_array)} document(s) – {[f.get('document_type') for f in filing_array]}")

            item_data['has_htm_files'] = True

            # Save SECFiling once per item (has at least one .htm document)
            self._save_filing(item_data)

            # Case 1: EX-2.1 present → process only via _process_ex21_filing (pass other filings for conditional 8-K/EX-99.1 summary).
            # Case 2: No EX-2.1 → process 8-K and EX-99.1 in the loop as before.
            ex21_entries = [f for f in filing_array if f.get(
                'document_type') == 'EX-2.1']
            if ex21_entries:
                other_filings = [f for f in filing_array if f.get(
                    'document_type') != 'EX-2.1']
                ex21_filing = ex21_entries[0]
                logger.info(
                    f"{LOG_PREFIX} :_process_single_item: accession=%s step=ex21_present other_count=%s",
                    accession_number, len(other_filings))
                self._process_ex21_filing(
                    item_data, ex21_filing, other_filings=other_filings)
            else:
                for filing in filing_array:
                    doc_type = filing.get('document_type')
                    logger.info(
                        f"{LOG_PREFIX} :_process_single_item: accession=%s step=process_doc doc_type=%s", accession_number, doc_type)
                    if doc_type == '8-K':
                        logger.info(
                            f"{LOG_PREFIX} :_process_single_item: accession=%s step=process_doc doc_type=8-K", accession_number)
                        self._process_8k_document(item_data, filing)
                    elif doc_type == 'EX-99.1':
                        logger.info(
                            f"{LOG_PREFIX} :_process_single_item: accession=%s step=process_doc doc_type=EX-99.1", accession_number)
                        if not item_data.get('cik_matches_deal'):
                            self._process_ex99_filing(item_data, filing)
                        else:
                            log_and_print(
                                f"{LOG_PREFIX} :_process_single_item: ⏭️ Skipping EX-99.1 processing (cik_matches_deal=True, summary handled elsewhere)")
            logger.info(
                f"{LOG_PREFIX} :_process_single_item: accession=%s step=done", accession_number)
            # Only add to lookup after successful processing so read timeouts/failures can retry next run
            if accession_number:
                mark_accession_processed(accession_number)
                logger.info(
                    f"{LOG_PREFIX} :_process_single_item: accession=%s step=lookup_saved", accession_number
                )

        except Exception as e:
            logger.exception(
                f"{LOG_PREFIX} :_process_single_item: accession=%s step=error error=%s", accession_number, str(e))
            log_and_print(
                f"{LOG_PREFIX} :_process_single_item: ❌ Error processing item: {e}", 'error')
            import traceback
            log_and_print(
                f"{LOG_PREFIX} :_process_single_item: {traceback.format_exc()}", 'error')
            self.error_count += 1
        finally:
            if accession_number and lock_owner:
                release_accession_lock(accession_number, lock_owner)

    def _process_ex21_filing(self, item_data, filing_entry, other_filings=None):
        """Process 8-K filing with EX-2.1 (one document from filing_array).
        When other_filings is provided and document_kind is Definitive Merger Agreement + is_us_listed + market_cap_gt_100m,
        also runs 8-K and EX-99.1 summary generation and send email (same as cik_matches_deal path).
        """
        accession_number = item_data.get('accession_number', 'N/A')
        try:
            # Refine context: same run_id but switch pipeline to ex21 + doc_type
            from core.logging_context import set_pipeline_context, get_run_id
            set_pipeline_context(
                pipeline="ex21",
                run_id=get_run_id(),
                accession=accession_number,
                doc_type="EX21",
            )

            logger.info(
                f"{LOG_PREFIX} :_process_ex21_filing: accession=%s step=start", accession_number)
            log_and_print(
                f"{LOG_PREFIX} :_process_ex21_filing: 📄 Processing EX-2.1 filing")
            url_ex21 = filing_entry.get('url')
            if not url_ex21:
                logger.warning(
                    f"{LOG_PREFIX} :_process_ex21_filing: accession=%s step=skip reason=no_url", accession_number)
                log_and_print(
                    f"{LOG_PREFIX} :_process_ex21_filing: ⚠️ No EX-2.1 document URL in filing entry", 'warning')
                return

            # Set xbrl_files to this filing for analyzer
            data = copy.deepcopy(item_data)
            data['xbrl_files'] = [
                {'type': 'EX-2.1', 'description': filing_entry.get('description', 'EX-2.1'), 'url': url_ex21}]
            # Analyze filing to get document_kind and company_details
            logger.info(f"{LOG_PREFIX} :_process_ex21_filing: accession=%s step=analyze company=%s",
                        accession_number, item_data.get('company_name', ''))
            log_and_print(
                f"{LOG_PREFIX} :_process_ex21_filing: 🔍 Analyzing EX-2.1 document for: {item_data.get('company_name')}")
            result = self.document_analyzer.analyze_filing(data)
            item_data.update(result)

            # Check document_kind
            document_kind = item_data.get('document_kind')
            # Get company_details
            company_details = item_data.get('company_details') or {}
            is_us_listed = company_details.get('is_target_us_listed')
            market_cap_gt_100m = company_details.get(
                'is_target_market_cap_greater_than_100m')
            logger.info(
                f"{LOG_PREFIX} :_process_ex21_filing: accession=%s step=result document_kind=%s is_us_listed=%s market_cap_gt_100m=%s",
                accession_number, document_kind, is_us_listed, market_cap_gt_100m,
            )
            log_and_print(
                f"{LOG_PREFIX} :_process_ex21_filing:   Document kind: {document_kind}")
            log_and_print(
                f"{LOG_PREFIX} :_process_ex21_filing:   US listed: {is_us_listed}")
            log_and_print(
                f"{LOG_PREFIX} :_process_ex21_filing:   Market cap > $100M: {market_cap_gt_100m}")

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
                log_and_print(
                    f"{LOG_PREFIX} :_process_ex21_filing: ⚠️ Failed to save/update filing", 'warning')
                self.error_count += 1
                return

            # Process EX-2.1 via 8-K document helper (Node API) if US-related and market cap > $100M
            if document_kind == "Definitive Merger Agreement" and is_us_listed and market_cap_gt_100m:
                # Fetch deal details when CIK matches an existing deal
                deal_details = None
                # True when deal already has a real sec_url (already processed)
                skip_processing = False
                if item_data.get('cik_matches_deal') and item_data.get('deal_id'):
                    try:
                        from bson import ObjectId
                        matched_deal = ProcessingJob.objects(
                            id=ObjectId(item_data['deal_id'])
                        ).only(
                            "target_name", "acquire_name", "cik", "acquirer_cik",
                            "deal_status", "announce_date", "target_ticker", "acquirer_ticker",
                            "sec_url",
                        ).first()
                        if matched_deal:
                            deal_details = {
                                'target_name': matched_deal.target_name or '',
                                'acquire_name': matched_deal.acquire_name or '',
                                'cik': matched_deal.cik or '',
                                'acquirer_cik': matched_deal.acquirer_cik or '',
                                'deal_status': matched_deal.deal_status or '',
                                'announce_date': matched_deal.announce_date,
                                'target_ticker': getattr(matched_deal, 'target_ticker', '') or '',
                                'acquirer_ticker': getattr(matched_deal, 'acquirer_ticker', '') or '',
                                'matched_cik_label': item_data.get('matched_cik_label', ''),
                            }
                            existing_sec_url = getattr(
                                matched_deal, 'sec_url', None) or ''
                            skip_processing = bool(existing_sec_url)
                            logger.info(
                                f"{LOG_PREFIX} :_process_ex21_filing: accession=%s step=deal_details_fetched deal_id=%s target=%s acquirer=%s sec_url=%s skip_processing=%s",
                                accession_number, item_data['deal_id'],
                                deal_details.get('target_name'), deal_details.get(
                                    'acquire_name'),
                                existing_sec_url[:60] if existing_sec_url else None, skip_processing)
                    except Exception as deal_e:
                        logger.warning(
                            f"{LOG_PREFIX} :_process_ex21_filing: accession=%s step=deal_details_failed error=%s",
                            accession_number, str(deal_e))

                # Send email (deal-aware template when deal_details available)
                self._send_ex21_email(
                    item_data, company_details, filing_entry=filing_entry, deal_details=deal_details)
                logger.info(
                    f"{LOG_PREFIX} :_process_ex21_filing: accession=%s step=qualified sending_historical_and_helper", accession_number)
                log_and_print(
                    f"{LOG_PREFIX} :_process_ex21_filing: ✅ Qualified for 8-K EX-2.1 document processing (US-listed + market cap > $100M)")

                if skip_processing:
                    # Deal already has a real sec_url — it was previously processed from a
                    # 2.1 filing (e.g. target company). Skip parsing/embedding for this
                    # filing (e.g. parent company filing the same 2.1). Email already sent above.
                    logger.info(
                        f"{LOG_PREFIX} :_process_ex21_filing: accession=%s step=skip_processing reason=deal_already_has_sec_url",
                        accession_number)
                    log_and_print(
                        f"{LOG_PREFIX} :_process_ex21_filing: ⏭️ Skipping parse/embed — deal already processed (sec_url exists)")
                else:
                    # self._send_historical_8k_email(item_data)
                    logger.info(
                        f"{LOG_PREFIX} :_process_ex21_filing: accession=%s step=process_ex21_via_8k_helper", accession_number)
                    self._process_ex21_via_8k_helper(item_data, filing)
                # Definitive Merger Agreement + other_filings: run 8-K and EX-99.1 summary generation and send email (same as cik_matches_deal path)
                if other_filings:
                    log_and_print(
                        f"{LOG_PREFIX} :_process_ex21_filing: 📝 Definitive Merger Agreement + qualified → generating 8-K/EX-99.1 summaries and sending emails")
                    for f in other_filings:
                        doc_type = f.get('document_type')
                        log_and_print(f"doc_type: {doc_type}")
                        if doc_type == '8-K':
                            url_8k = f.get('url')
                            log_and_print(f"url_8k: {url_8k}")
                            if url_8k:
                                self._generate_8k_summary_and_send(
                                    item_data, url_8k)
                        elif doc_type == 'EX-99.1':
                            url_ex99 = f.get('url')
                            log_and_print(f"url_ex99: {url_ex99}")
                            if url_ex99:
                                ex99_result = self._generate_ex99_summary(
                                    item_data, url_ex99)
                                # Extract press release data when both EX-2.1 and EX-99.1 are present
                                if ex99_result:
                                    self._extract_press_release_data(
                                        item_data, ex99_result, url_ex99)
                self.ex21_processed_count += 1
            else:
                logger.info(f"{LOG_PREFIX} :_process_ex21_filing: accession=%s step=not_qualified is_us_listed=%s market_cap_gt_100m=%s",
                            accession_number, is_us_listed, market_cap_gt_100m)
                log_and_print(
                    f"{LOG_PREFIX} :_process_ex21_filing: ⏭️ Not qualified for 8-K EX-2.1 document processing (US-listed or market cap criteria not met)")

            self.processed_count += 1
            logger.info(
                f"{LOG_PREFIX} :_process_ex21_filing: accession=%s step=done", accession_number)

        except Exception as e:
            logger.exception(
                f"{LOG_PREFIX} :_process_ex21_filing: accession=%s error=%s", accession_number, str(e))
            log_and_print(
                f"{LOG_PREFIX} :_process_ex21_filing: ❌ Error processing EX-2.1 filing: {e}", 'error')
            import traceback
            log_and_print(
                f"{LOG_PREFIX} :_process_ex21_filing: {traceback.format_exc()}", 'error')
            self.error_count += 1

    def _process_ex99_filing(self, item_data, filing_entry):
        """
        EX-99.1 document flow:
        - If cik_matches_deal: generate summary, save to sec_filing_summary, send summary email (no GPT).
        - If not cik_matches_deal: run GPT; send main EX-99.1 email only when all of
          is_merger_related, is_target_us_listed, is_target_market_cap_greater_than_100m are truthy.
        """
        accession_number = item_data.get('accession_number', 'N/A')
        try:
            logger.info(
                f"{LOG_PREFIX} :_process_ex99_filing: accession=%s step=start", accession_number)
            log_and_print(
                f"{LOG_PREFIX} :_process_ex99_filing: 📄 Processing EX-99.1 filing")
            url_ex99 = filing_entry.get('url')
            if not url_ex99:
                logger.warning(
                    f"{LOG_PREFIX} :_process_ex99_filing: accession=%s step=skip reason=no_url", accession_number)
                log_and_print(
                    f"{LOG_PREFIX} :_process_ex99_filing: ⚠️ No EX-99.1 document URL in filing entry", 'warning')
                return

            if item_data.get('cik_matches_deal'):
                logger.info(
                    f"{LOG_PREFIX} :_process_ex99_filing: accession=%s step=cik_matches_deal generating_summary", accession_number)
                log_and_print(
                    f"{LOG_PREFIX} :_process_ex99_filing:   ✅ CIK matches deal → generating EX-99.1 summary (no GPT)")
                self._generate_ex99_summary(item_data, url_ex99)
                logger.info(
                    f"{LOG_PREFIX} :_process_ex99_filing: accession=%s step=done", accession_number)
                self.processed_count += 1
                self.ex99_processed_count += 1
                return

            logger.info(f"{LOG_PREFIX} :_process_ex99_filing: accession=%s step=gpt_analysis company=%s",
                        accession_number, item_data.get('company_name', ''))
            data = copy.deepcopy(item_data)
            data['xbrl_files'] = [
                {'type': 'EX-99.1', 'description': filing_entry.get('description', 'EX-99.1'), 'url': url_ex99}]
            log_and_print(
                f"{LOG_PREFIX} :_process_ex99_filing: 🔍 EX-99.1: GPT analysis: {item_data.get('company_name')}")
            result = self.document_analyzer.analyze_ex99_1_filing(data)
            logger.info(
                f"{LOG_PREFIX} :_process_ex99_filing: accession=%s step=gpt_result result=%s", accession_number, result)

            item_data.update(result)

            is_merger_related = item_data.get('is_merger_related')
            is_listed = item_data.get('is_target_us_listed')
            market_cap_gt_100m = item_data.get(
                'is_target_market_cap_greater_than_100m')
            logger.info(
                f"{LOG_PREFIX} :_process_ex99_filing: accession=%s step=gpt_result is_merger=%s is_listed=%s cap_gt_100m=%s confidence=%s",
                accession_number, is_merger_related, is_listed, market_cap_gt_100m, item_data.get(
                    'confidence', 0),
            )
            log_and_print(
                f"{LOG_PREFIX} :_process_ex99_filing:   Merger-related: {is_merger_related}")
            log_and_print(
                f"{LOG_PREFIX} :_process_ex99_filing:   Market cap > $100M: {market_cap_gt_100m}")
            log_and_print(
                f"{LOG_PREFIX} :_process_ex99_filing:   Confidence: {item_data.get('confidence', 0)}%")

            # Send EX-99.1 email only if all three are truthy; otherwise skip
            if is_merger_related and is_listed and market_cap_gt_100m:
                logger.info(
                    f"{LOG_PREFIX} :_process_ex99_filing: accession=%s step=send_ex99_email", accession_number)
                self._send_ex99_email(item_data, filing_entry=filing_entry)
                logger.info(
                    f"{LOG_PREFIX} :_process_ex99_filing: accession=%s step=done", accession_number)
            else:
                logger.info(
                    f"{LOG_PREFIX} :_process_ex99_filing: accession=%s step=email_skipped reason=criteria_not_met", accession_number)
                log_and_print(
                    f"{LOG_PREFIX} :_process_ex99_filing: ⏭️ EX-99.1 main email skipped (need is_merger_related, is_target_us_listed, and market cap > $100M)")

            self.processed_count += 1
            self.ex99_processed_count += 1
            logger.info(
                f"{LOG_PREFIX} :_process_ex99_filing: accession=%s step=done", accession_number)

        except Exception as e:
            logger.exception(
                f"{LOG_PREFIX} :_process_ex99_filing: accession=%s error=%s", accession_number, str(e))
            log_and_print(
                f"{LOG_PREFIX} :_process_ex99_filing: ❌ Error processing EX-99.1 filing: {e}", 'error')
            import traceback
            log_and_print(
                f"{LOG_PREFIX} :_process_ex99_filing: {traceback.format_exc()}", 'error')
            self.error_count += 1

    def _save_filing(self, item_data):
        """Save filing to SECFiling database"""
        accession_number = item_data.get('accession_number')
        try:
            if not accession_number:
                logger.warning(
                    f"{LOG_PREFIX} :_save_filing: accession=missing")
                log_and_print(
                    f"{LOG_PREFIX} :_save_filing: Cannot save filing without accession_number", 'warning')
                return None

            # Check if already exists (race condition)
            existing = SECFiling.objects(
                accession_number=accession_number).first()
            if existing:
                logger.info(f"{LOG_PREFIX} :_save_filing: accession=%s step=existing filing_id=%s",
                            accession_number, str(existing._id))
                log_and_print(
                    f"{LOG_PREFIX} :_save_filing: ⏭️ Filing already exists: {accession_number}")
                return existing

            logger.info(f"{LOG_PREFIX} :_save_filing:    accession=%s step=prepare company=%s",
                        accession_number, item_data.get('company_name', ''))
            # Prepare filing data
            item_data = self._prepare_filing_data(item_data)

            # Filter allowed fields
            filing_data = {k: v for k, v in item_data.items()
                           if k in ALLOWED_FILING_FIELDS}

            # Create and save filing
            filing = SECFiling(**filing_data)
            try:
                filing.save()
                logger.info(f"{LOG_PREFIX} :_save_filing: accession=%s step=saved filing_id=%s company=%s",
                            accession_number, str(filing._id), filing.company_name or '')
                log_and_print(
                    f"{LOG_PREFIX} :_save_filing: 💾 Saved filing: {item_data.get('company_name')} - {accession_number}")

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
                    logger.info(
                        f"{LOG_PREFIX} :_save_filing: accession=%s step=duplicate_race using_existing", accession_number)
                    log_and_print(
                        f"{LOG_PREFIX} :_save_filing: ⏭️ Duplicate accession_number (race): {accession_number}", 'warning')
                    # Try to fetch existing
                    return SECFiling.objects(accession_number=accession_number).first()
                raise

        except Exception as e:
            logger.exception(f"{LOG_PREFIX} :_save_filing: accession=%s error=%s",
                             accession_number, str(e))
            log_and_print(
                f"{LOG_PREFIX} :_save_filing: ❌ Error saving filing: {e}", 'error')
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
                    f"{LOG_PREFIX} :_prepare_filing_data: Error converting acceptance_datetime_utc: {e}", 'warning')
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

        # description is required on SECFiling (max 50). JSON feed items often
        # omit Atom <summary>; default to form_type so save does not ValidationError.
        desc = (item_data.get('description') or '').strip()
        if not desc:
            desc = (item_data.get('form_type') or '8-K').strip() or '8-K'
        item_data['description'] = desc[:MAX_DESCRIPTION_LENGTH]

        # Set following defaults
        if 'following' not in item_data:
            item_data['following'] = False
        if 'following_status' not in item_data:
            item_data['following_status'] = None

        return item_data

    def _send_ex21_email(self, item_data, company_details, filing_entry=None, deal_details=None):
        """Send email for EX-2.1 filing. If filing_entry is provided, use it for doc table (file/size from SEC).
        If deal_details is provided, uses the deal-aware email template with existing deal info."""
        accession_number = item_data.get('accession_number', 'N/A')
        try:
            logger.info(f"{LOG_PREFIX} :_send_ex21_email: accession=%s step=start company=%s deal_match=%s",
                        accession_number, item_data.get('company_name', ''), bool(deal_details))
            log_and_print(
                f"{LOG_PREFIX} :_send_ex21_email: 📧 Generating and sending EX-2.1 email")

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

            # Use deal-aware template when CIK matches an existing deal
            if deal_details:
                subject, html_email = generate_filing_email_with_deal_html(
                    item_data, doc_files, deal_details)
            else:
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
            logger.info(f"{LOG_PREFIX} :_send_ex21_email: accession=%s step=send webhook=%s",
                        accession_number, 'filing' if use_filing_webhook else '8k_summary')

            # TODO: comment out after org-aware send is stable
            # send_webhook_notification(webhook_url, payload, "EX-2.1 email")
            logger.info(f"{LOG_PREFIX} :_send_ex21_email: accession=%s step=sent",
                        accession_number)
            log_and_print(
                f"{LOG_PREFIX} :_send_ex21_email: ✅ EX-2.1 email sent successfully")

            _report_type = "sec_new_deal_announcement" if use_filing_webhook else "sec_new_deal_announced_without_threshold"
            send_report_email(
                report_type=_report_type,
                payload=payload
            )
            logger.info(f"{LOG_PREFIX} :_send_ex21_email: accession=%s step=org_aware_sent report_type=%s",
                        accession_number, _report_type)
            log_and_print(
                f"{LOG_PREFIX} :_send_ex21_email: ✅ Org-aware email sent ({_report_type})")

        except Exception as e:
            logger.exception(
                f"{LOG_PREFIX} :_send_ex21_email: accession=%s error=%s", accession_number, str(e))
            log_and_print(
                f"{LOG_PREFIX} :_send_ex21_email: ❌ Error sending EX-2.1 email: {e}", 'error')

    def _send_historical_8k_email(self, item_data):
        """
        Send email with all 8-K filings from the last 1 year.

        This provides historical context for the company's recent 8-K filing activity.
        Similar to services.py logic for EX-2.1 filings.
        """
        accession_number = item_data.get('accession_number', 'N/A')
        try:
            cik_number = item_data.get('cik_number')
            filing_date = item_data.get('filing_date')
            company_name = item_data.get('company_name', 'Unknown Company')
            logger.info(f"{LOG_PREFIX} :_send_historical_8k_email: accession=%s step=start cik=%s company=%s",
                        accession_number, cik_number, company_name)

            # if not cik_number or not filing_date:
            #     logger.warning(
            #         f"{LOG_PREFIX} :_send_historical_8k_email: accession=%s step=skip reason=missing_cik_or_date", accession_number)
            #     log_and_print(
            #         f"{LOG_PREFIX} :_send_historical_8k_email: ⏭️ Skipping historical 8-K email: missing CIK or filing date",
            #         'warning'
            #     )
            #     return

            # Parse filing date if it's a string
            # if not isinstance(filing_date, datetime):
            #     filing_date = parse_filing_date(filing_date)

            # if not filing_date:
            #     logger.warning(
            #         f"{LOG_PREFIX} :_send_historical_8k_email: accession=%s step=skip reason=invalid_filing_date", accession_number)
            #     log_and_print(
            #         f"{LOG_PREFIX} :_send_historical_8k_email: ⏭️ Skipping historical 8-K email: invalid filing date",
            #         'warning'
            #     )
            #     return

            # Fetch 8-K filings from 1 year before filing date
            # start_date = (filing_date - timedelta(days=365)
            #               ).strftime('%Y-%m-%d')
            # logger.info(f"{LOG_PREFIX} :_send_historical_8k_email: accession=%s step=fetch cik=%s start_date=%s",
            #             accession_number, cik_number, start_date)

            # log_and_print(
            #     f"{LOG_PREFIX} :_send_historical_8k_email:  🔍 Fetching historical 8-K filings for CIK {cik_number} from {start_date}..."
            # )

            # filings = fetch_sec_filings(
            #     str(cik_number),
            #     start_date=start_date,
            #     form_types=None
            # )

            # if not filings:
            #     logger.info(
            #         f"{LOG_PREFIX} :_send_historical_8k_email: accession=%s step=no_filings cik=%s", accession_number, cik_number)
            #     log_and_print(
            #         f"{LOG_PREFIX} :_send_historical_8k_email: ⚠️ No historical 8-K filings found for CIK {cik_number}",
            #         'warning'
            #     )
            #     return

            # logger.info(f"{LOG_PREFIX} :_send_historical_8k_email: accession=%s step=fetched filings_count=%s",
            #             accession_number, len(filings))
            # log_and_print(
            #     f"{LOG_PREFIX} :_send_historical_8k_email: 📥 Found {len(filings)} historical 8-K filings")

            # Generate email HTML
            # sec_subject, sec_html = generate_sec_filings_email_html(
            #     company_name,
            #     filings,
            #     form_type="8-K(EX-2.1)"
            # )

            # # Prepare payload
            # sec_payload = {
            #     'subject': sec_subject,
            #     'html': sec_html,
            #     'company_name': company_name,
            #     'email_type': 'sec_filings_last_year',
            # }

            # Send email
            # send_webhook_notification(
            #     N8N_WEBHOOK_URL_8K_SUMMARY,
            #     sec_payload,
            #     "Historical 8-K filings email"
            # )
            # logger.info(f"{LOG_PREFIX} :_send_historical_8k_email: accession=%s step=sent filings_count=%s company=%s",
            #             accession_number, len(filings), company_name)
            # log_and_print(
            #     f"{LOG_PREFIX} :_send_historical_8k_email: 📤 Sent historical 8-K filings email: {len(filings)} filings for {company_name}"
            # )

        except Exception as e:
            # logger.exception(
            #     f"{LOG_PREFIX} :_send_historical_8k_email: accession=%s error=%s", accession_number, str(e))
            log_and_print(
                f"{LOG_PREFIX} :_send_historical_8k_email: ❌ Error sending historical 8-K email: {e}",
                'error'
            )

    def _send_ex99_email(self, item_data, filing_entry=None):
        """Send email for EX-99.1 filing. If filing_entry is provided, use it for doc table (file/size from SEC table)."""
        accession_number = item_data.get('accession_number', 'N/A')
        try:
            logger.info(f"{LOG_PREFIX} :_send_ex99_email: accession=%s step=start company=%s",
                        accession_number, item_data.get('company_name', ''))
            log_and_print(
                f"{LOG_PREFIX} :_send_ex99_email: 📧 Generating and sending EX-99.1 email")

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
                    f"{LOG_PREFIX} :_send_ex99_email:   doc_files for EX-99.1 email: file={doc_files[0].get('file')!r} size={doc_files[0].get('size')}")
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
            # send_webhook_notification(
            #     N8N_WEBHOOK_URL_8K_SUMMARY, payload, "EX-99.1 email"
            # )  # TODO: comment out after org-aware send is stable
            logger.info(f"{LOG_PREFIX} :_send_ex99_email: accession=%s step=sent",
                        accession_number)
            log_and_print(
                f"{LOG_PREFIX} :_send_ex99_email: ✅ EX-99.1 email sent successfully")

            send_report_email(
                report_type="sec_new_deal_probably_announced",
                payload=payload
            )
            logger.info(f"{LOG_PREFIX} :_send_ex99_email: accession=%s step=org_aware_sent",
                        accession_number)
            log_and_print(
                f"{LOG_PREFIX} :_send_ex99_email: ✅ Org-aware email sent (sec_new_deal_probably_announced)")

        except Exception as e:
            logger.exception(
                f"{LOG_PREFIX} :_send_ex99_email: accession=%s error=%s", accession_number, str(e))
            log_and_print(
                f"{LOG_PREFIX} :_send_ex99_email: ❌ Error sending EX-99.1 email: {e}", 'error')

    def _process_ex21_via_8k_helper(self, item_data, filing):
        """Process EX-2.1 filing via 8-K document helper (Node API deal/process-with-url), same as services.py."""
        accession_number = item_data.get('accession_number', 'N/A')
        try:
            logger.info(f"{LOG_PREFIX} :_process_ex21_via_8k_helper: accession=%s step=start sec_filing_id=%s",
                        accession_number, str(filing._id))
            log_and_print(
                f"{LOG_PREFIX} :_process_ex21_via_8k_helper: 🚀 Starting 8-K EX-2.1 document processing")

            ex21_file = find_file_by_type(
                item_data.get('xbrl_files', []), 'EX-2.1')
            if not ex21_file:
                logger.error(
                    f"{LOG_PREFIX} :_process_ex21_via_8k_helper: accession=%s step=skip reason=no_ex21_in_xbrl", accession_number)
                log_and_print(
                    f"{LOG_PREFIX} :_process_ex21_via_8k_helper: ❌ No EX-2.1 file found in xbrl_files", 'error')
                return

            ex21_url = build_full_sec_url(ex21_file.get('url'))
            if not ex21_url:
                logger.error(
                    f"{LOG_PREFIX} :_process_ex21_via_8k_helper: accession=%s step=skip reason=no_url", accession_number)
                log_and_print(
                    f"{LOG_PREFIX} :_process_ex21_via_8k_helper: ❌ Could not build EX-2.1 URL", 'error')
                return

            cik_number = item_data.get('cik_number', '')
            company_name = item_data.get('company_name', '')
            sec_filing_id = str(filing._id)
            company_details = item_data.get('company_details')
            filing_date = item_data.get('filing_date')
            filing_date_obj = parse_filing_date(
                filing_date) if filing_date else None
            logger.info(f"{LOG_PREFIX} :_process_ex21_via_8k_helper: accession=%s step=call_helper ex21_url=%s sec_filing_id=%s",
                        accession_number, ex21_url[:80] if ex21_url else '', sec_filing_id)

            log_and_print(
                f"{LOG_PREFIX} :_process_ex21_via_8k_helper:   EX-2.1 URL: {ex21_url}")
            log_and_print(
                f"{LOG_PREFIX} :_process_ex21_via_8k_helper:   SEC Filing ID: {sec_filing_id}")

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
                logger.info(f"{LOG_PREFIX} :_process_ex21_via_8k_helper: accession=%s step=result status=%s message=%s",
                            accession_number, result.get('status'), result.get('message', '')[:80])
                log_and_print(
                    f"{LOG_PREFIX} :_process_ex21_via_8k_helper: ✅ 8-K EX-2.1 document processing started: {result.get('message', 'Processing started')}")
                log_and_print(f"   Status: {result.get('status')}")
            else:
                logger.error(
                    f"{LOG_PREFIX} :_process_ex21_via_8k_helper: accession=%s step=failed result=empty", accession_number)
                log_and_print(
                    f"{LOG_PREFIX} :_process_ex21_via_8k_helper: ❌ Failed to start 8-K EX-2.1 document processing", 'error')

        except Exception as e:
            logger.exception(
                f"{LOG_PREFIX} :_process_ex21_via_8k_helper: accession=%s error=%s", accession_number, str(e))
            log_and_print(
                f"{LOG_PREFIX} :_process_ex21_via_8k_helper: ❌ Error processing EX-2.1 via 8-K helper: {e}", 'error')
            import traceback
            log_and_print(
                f"{LOG_PREFIX} :_process_ex21_via_8k_helper: {traceback.format_exc()}", 'error')

    def _generate_ex99_summary(self, item_data, url_ex99):
        """
        Generate summary for EX-99.1 via filing router; save to sec_filing_summary 99_1 node only.
        Returns the summary result dict (with L1_headline, L2_brief, L3_detailed, s3_docx_url, etc.) or None.
        """
        accession_number = item_data.get('accession_number', 'N/A')
        try:
            logger.info(f"{LOG_PREFIX} :_generate_ex99_summary: accession=%s step=start url=%s",
                        accession_number, url_ex99 or '')
            log_and_print(
                f"{LOG_PREFIX} :_generate_ex99_summary: 📝 Generating summary for EX-99.1 document")

            deal_id = item_data.get('deal_id')
            cik_number = item_data.get('cik_number')
            deal_tickers = get_deal_tickers(deal_id, cik_number)
            deal_context = {
                "primary_ticker":  deal_tickers.get("ticker"),
                "target_ticker":   deal_tickers.get("target_ticker"),
                "target_name":     deal_tickers.get("target_name"),
                "acquirer_ticker": deal_tickers.get("acquirer_ticker"),
                "acquirer_name":   deal_tickers.get("acquirer_name"),
            } if any(deal_tickers.values()) else None

            if url_ex99:
                try:
                    log_and_print(
                        f"{LOG_PREFIX} :_generate_ex99_summary:    Summarizing EX-99.1 document: {url_ex99}")
                    result_99 = route_and_summarize(
                        url_ex99, deal_context=deal_context)
                    s3_url = result_99.get(
                        's3_docx_url') or result_99.get('s3_url')

                    if s3_url:
                        logger.info(f"{LOG_PREFIX} :_generate_ex99_summary: accession=%s step=s3_uploaded s3_url=%s",
                                    accession_number, (s3_url or '')[:80])
                        log_and_print(
                            f"{LOG_PREFIX} :_generate_ex99_summary: ✅ EX-99.1 summary uploaded to S3: {s3_url}")

                        try:
                            filing_dt = self._filing_date_for_summary(
                                item_data.get('filing_date'), result_99.get('filing_date'))
                            # 99_1 node: only these 6 fields; parent-level summary fields stay null
                            ex99_1_payload = {
                                'items_reported': result_99.get('items_reported') or [],
                                'L1_headline': result_99.get('L1_headline'),
                                'L2_brief': result_99.get('L2_brief'),
                                'L3_detailed': result_99.get('L3_detailed') or {},
                                's3_docx_url': result_99.get('s3_docx_url') or result_99.get('s3_url'),
                                's3_json_url': result_99.get('s3_json_url'),
                            }
                            url_8k = None
                            for fe in item_data.get('filing_array') or []:
                                if fe.get('document_type') == '8-K' and fe.get('url'):
                                    url_8k = fe.get('url')
                                    break
                            existing = SECFilingSummary.objects(
                                accession_number=accession_number, form_type='8-K'
                            ).first()
                            if existing:
                                existing.sec_document_url = url_8k or url_ex99
                                existing.filing_date = filing_dt
                                existing.deal_id = deal_id
                                existing.ex99_1 = ex99_1_payload
                                existing.save()
                            else:
                                SECFilingSummary(
                                    form_type='8-K',
                                    accession_number=accession_number,
                                    cik_number=item_data.get('cik_number'),
                                    sec_document_url=url_8k or url_ex99,
                                    filing_date=filing_dt,
                                    deal_id=deal_id,
                                    items_reported=[],
                                    L1_headline=None,
                                    L2_brief=None,
                                    L3_detailed=None,
                                    s3_docx_url=None,
                                    s3_json_url=None,
                                    ex99_1=ex99_1_payload,
                                ).save()
                            logger.info(
                                f"{LOG_PREFIX} :_generate_ex99_summary: accession=%s step=db_saved", accession_number)
                            log_and_print(
                                f"{LOG_PREFIX} :_generate_ex99_summary: 💾 EX-99.1 summary saved to DB (sec_filing_summary, 99_1 node)")
                            self.summary_ex99_count += 1
                            self._send_ex99_summary_email(
                                item_data, result_99, url_ex99)
                            return result_99
                        except Exception as db_e:
                            logger.exception(
                                f"{LOG_PREFIX} :_generate_ex99_summary: accession=%s step=db_error error=%s", accession_number, str(db_e))
                            log_and_print(
                                f"{LOG_PREFIX} :_generate_ex99_summary: ❌ Failed to save EX-99.1 summary to DB: {db_e}", 'error')
                except Exception as e:
                    logger.exception(
                        f"{LOG_PREFIX} :_generate_ex99_summary: accession=%s step=summary_error error=%s", accession_number, str(e))
                    log_and_print(
                        f"{LOG_PREFIX} :_generate_ex99_summary: ❌ EX-99.1 summary generation failed: {e}", 'error')
        except Exception as e:
            logger.exception(
                f"{LOG_PREFIX} :_generate_ex99_summary: accession=%s error=%s", accession_number, str(e))
            log_and_print(
                f"{LOG_PREFIX} :_generate_ex99_summary: ❌ Error in _generate_ex99_summary: {e}", 'error')
        return None

    def _send_8k_summary_email(self, item_data, summary_result, doc_url):
        """Send email with 8-K summary document link"""
        accession_number = item_data.get('accession_number', 'N/A')
        summary_doc_url = summary_result.get(
            's3_docx_url') or summary_result.get('s3_url')
        try:
            logger.info(f"{LOG_PREFIX} :_send_8k_summary_email: accession=%s step=start summary_url=%s",
                        accession_number, (summary_doc_url or '')[:80])
            log_and_print(
                f"{LOG_PREFIX} :_send_8k_summary_email:   📧 Sending 8-K summary email")

            from sec_rss_parser.email_templates import generate_8k_99_1_summary_email_html
            from sec_rss_parser.sec_summarizers._ticker_context import resolve_l1_for_email

            deal_tickers = get_deal_tickers(
                item_data.get('deal_id'), item_data.get('cik_number'))
            # Fallback: when no deal_id, use company_details populated by EX-2.1 GPT analysis
            if not any(deal_tickers.values()):
                cd = item_data.get('company_details') or {}
                deal_tickers = {
                    'ticker': cd.get('target_ticker'),
                    'target_ticker': cd.get('target_ticker'),
                    'target_name': cd.get('target_name'),
                    'acquirer_ticker': cd.get('acquirer_ticker'),
                    'acquirer_name': cd.get('acquirer_name'),
                }
            email_company_name = item_data.get(
                'email_company_name') or item_data.get('company_name') or ''
            matched_cik_label = item_data.get('matched_cik_label')
            l1_headline = resolve_l1_for_email(
                summary_result.get('L1_headline'),
                primary_ticker=deal_tickers.get('ticker'),
                matched_cik_label=matched_cik_label,
                target_ticker=deal_tickers.get('target_ticker'),
                acquirer_ticker=deal_tickers.get('acquirer_ticker'),
            )

            summary_kind = '8-K + EX-99.1' if summary_result.get(
                'summary_type') == 'combined' else '8-K'
            subject, html_email = generate_8k_99_1_summary_email_html(
                company_name=email_company_name,
                form_type='8-K',
                summary_doc_url=summary_doc_url,
                cik_number=item_data.get('cik_number') or '',
                sec_url=item_data.get('link') or doc_url,
                accession_number=item_data.get('accession_number') or '',
                summary_kind=summary_kind,
                l1_headline=l1_headline,
                l2_brief=summary_result.get('L2_brief'),
                l3_detailed=summary_result.get('L3_detailed'),
                matched_cik_label=matched_cik_label,
                target_ticker=deal_tickers.get('target_ticker'),
                target_name=deal_tickers.get('target_name'),
                acquirer_ticker=deal_tickers.get('acquirer_ticker'),
                acquirer_name=deal_tickers.get('acquirer_name'),

            )

            payload = {
                'subject': subject,
                'html': html_email,
                'company_name': email_company_name,
                'form_type': '8-K',
                'summary_doc_url': summary_doc_url,
                'accession_number': item_data.get('accession_number'),
                'cik_number': item_data.get('cik_number'),
                'sec_url': doc_url,
            }

            # send_webhook_notification(
            #     N8N_WEBHOOK_URL_8K_SUMMARY_L123, payload, "8-K summary email"
            # )  # TODO: comment out after org-aware send is stable
            logger.info(
                f"{LOG_PREFIX} :_send_8k_summary_email: accession=%s step=sent", accession_number)
            log_and_print(
                f"{LOG_PREFIX} :_send_8k_summary_email: ✅ 8-K summary email sent successfully")

            send_report_email(
                report_type="sec_form_type_proxy_10k_q_425",
                payload=payload,
                deal_id=item_data.get('deal_id')
            )
            logger.info(
                f"{LOG_PREFIX} :_send_8k_summary_email: accession=%s step=org_aware_sent", accession_number)
            log_and_print(
                f"{LOG_PREFIX} :_send_8k_summary_email: ✅ Org-aware email sent (sec_form_type_proxy_10k_q_425)")

        except Exception as e:
            logger.exception(
                f"{LOG_PREFIX} :_send_8k_summary_email: accession=%s error=%s", accession_number, str(e))
            log_and_print(
                f"{LOG_PREFIX} :_send_8k_summary_email: ❌ Error sending 8-K summary email: {e}", 'error')

    def _send_ex99_summary_email(self, item_data, summary_result, doc_url):
        """Send email with EX-99.1 summary document link"""
        accession_number = item_data.get('accession_number', 'N/A')
        summary_doc_url = summary_result.get(
            's3_docx_url') or summary_result.get('s3_url')
        try:
            logger.info(f"{LOG_PREFIX} :_send_ex99_summary_email: accession=%s step=start summary_url=%s",
                        accession_number, (summary_doc_url or '')[:80])
            log_and_print(
                f"{LOG_PREFIX} :_send_ex99_summary_email:   📧 Sending EX-99.1 summary email")

            from sec_rss_parser.email_templates import generate_8k_99_1_summary_email_html
            from sec_rss_parser.sec_summarizers._ticker_context import resolve_l1_for_email

            deal_tickers = get_deal_tickers(
                item_data.get('deal_id'), item_data.get('cik_number'))
            # Fallback: when no deal_id, use company_details populated by EX-2.1 GPT analysis
            if not any(deal_tickers.values()):
                cd = item_data.get('company_details') or {}
                deal_tickers = {
                    'ticker': cd.get('target_ticker'),
                    'target_ticker': cd.get('target_ticker'),
                    'target_name': cd.get('target_name'),
                    'acquirer_ticker': cd.get('acquirer_ticker'),
                    'acquirer_name': cd.get('acquirer_name'),
                }
            email_company_name = item_data.get(
                'email_company_name') or item_data.get('company_name') or ''
            matched_cik_label = item_data.get('matched_cik_label')
            l1_headline = resolve_l1_for_email(
                summary_result.get('L1_headline'),
                primary_ticker=deal_tickers.get('ticker'),
                matched_cik_label=matched_cik_label,
                target_ticker=deal_tickers.get('target_ticker'),
                acquirer_ticker=deal_tickers.get('acquirer_ticker'),
            )

            subject, html_email = generate_8k_99_1_summary_email_html(
                company_name=email_company_name,
                form_type='8-K (EX-99.1)',
                summary_doc_url=summary_doc_url,
                cik_number=item_data.get('cik_number') or '',
                sec_url=item_data.get('link') or doc_url,
                accession_number=item_data.get('accession_number') or '',
                summary_kind='EX-99.1',
                l1_headline=l1_headline,
                l2_brief=summary_result.get('L2_brief'),
                matched_cik_label=matched_cik_label,
                target_ticker=deal_tickers.get('target_ticker'),
                target_name=deal_tickers.get('target_name'),
                acquirer_ticker=deal_tickers.get('acquirer_ticker'),
                acquirer_name=deal_tickers.get('acquirer_name'),
            )

            payload = {
                'subject': subject,
                'html': html_email,
                'company_name': email_company_name,
                'form_type': '8-K (EX-99.1)',
                'summary_doc_url': summary_doc_url,
                'accession_number': item_data.get('accession_number'),
                'cik_number': item_data.get('cik_number'),
                'sec_url': doc_url,
            }

            # send_webhook_notification(
            #     N8N_WEBHOOK_URL_8K_SUMMARY, payload, "EX-99.1 summary email"
            # )  # TODO: comment out after org-aware send is stable
            logger.info(
                f"{LOG_PREFIX} :_send_ex99_summary_email: accession=%s step=sent", accession_number)
            log_and_print(
                f"{LOG_PREFIX} :_send_ex99_summary_email: ✅ EX-99.1 summary email sent successfully")

            send_report_email(
                report_type="sec_all_other_forms",
                payload=payload,
                deal_id=item_data.get('deal_id')
            )
            logger.info(
                f"{LOG_PREFIX} :_send_ex99_summary_email: accession=%s step=org_aware_sent", accession_number)
            log_and_print(
                f"{LOG_PREFIX} :_send_ex99_summary_email: ✅ Org-aware email sent (sec_standard_summary)")

        except Exception as e:
            logger.exception(
                f"{LOG_PREFIX} :_send_ex99_summary_email: accession=%s error=%s", accession_number, str(e))
            log_and_print(
                f"{LOG_PREFIX} :_send_ex99_summary_email: ❌ Error sending EX-99.1 summary email: {e}", 'error')

    def _extract_press_release_data(self, item_data, summary_result, url_ex99):
        """
        Extract structured deal financial data from EX-99.1 (Press Release) summary.
        Uses Claude Haiku via press_release_processor, saves to fo_press_release_extraction.
        """
        accession_number = item_data.get('accession_number', 'N/A')
        try:
            logger.info(
                f"{LOG_PREFIX} :_extract_press_release_data: accession=%s step=start",
                accession_number
            )
            log_and_print(
                f"{LOG_PREFIX} :_extract_press_release_data: 📊 Extracting structured data from Press Release"
            )

            l1 = summary_result.get('L1_headline') or ''
            l2 = summary_result.get('L2_brief') or ''
            l3 = summary_result.get('L3_detailed') or {}

            if isinstance(l3, dict):
                l3_text_parts = []
                for key, val in l3.items():
                    if val:
                        l3_text_parts.append(f"{key}: {val}")
                l3_text = "\n".join(l3_text_parts)
            else:
                l3_text = str(l3) if l3 else ''

            summary_text = f"""L1 HEADLINE:
{l1}

L2 BRIEF:
{l2}

L3 DETAILED:
{l3_text}
"""

            if not l1.strip() and not l2.strip():
                logger.warning(
                    f"{LOG_PREFIX} :_extract_press_release_data: accession=%s step=skip reason=no_summary_content",
                    accession_number
                )
                log_and_print(
                    f"{LOG_PREFIX} :_extract_press_release_data: ⚠️ No summary content to extract from",
                    'warning'
                )
                return

            from sec_rss_parser.summary_processor.press_release_processor import extract_from_press_release

            deal_id = item_data.get('deal_id')
            company_name = item_data.get('company_name')
            cik_number = item_data.get('cik_number')
            filing_date = item_data.get('filing_date')
            if isinstance(filing_date, datetime):
                filing_date = filing_date.strftime('%Y-%m-%d')

            s3_docx_url = summary_result.get(
                's3_docx_url') or summary_result.get('s3_url')

            existing_summary = SECFilingSummary.objects(
                accession_number=accession_number, form_type='8-K'
            ).first()
            press_release_id = str(
                existing_summary._id) if existing_summary else None

            result = extract_from_press_release(
                summary_text=summary_text,
                deal_id=deal_id,
                accession_number=accession_number,
                company_name=company_name,
                cik_number=cik_number,
                press_release_id=press_release_id,
                press_release_docx=s3_docx_url,
                filing_date=filing_date,
                send_email=True,
            )

            if result:
                logger.info(
                    f"{LOG_PREFIX} :_extract_press_release_data: accession=%s step=extracted target=%s",
                    accession_number, result.get(
                        'extracted', {}).get('target', 'N/A')
                )
                log_and_print(
                    f"{LOG_PREFIX} :_extract_press_release_data: ✅ Press Release extraction completed and saved"
                )
            else:
                logger.warning(
                    f"{LOG_PREFIX} :_extract_press_release_data: accession=%s step=no_result",
                    accession_number
                )

        except Exception as e:
            logger.exception(
                f"{LOG_PREFIX} :_extract_press_release_data: accession=%s error=%s",
                accession_number, str(e)
            )
            log_and_print(
                f"{LOG_PREFIX} :_extract_press_release_data: ❌ Press Release extraction failed: {e}",
                'error'
            )


def run_8k_processor(rss_content=None, rss_file=None):
    """
    Entry point to run the 8-K processor.

    For testing with manual RSS:
      run_8k_processor(rss_file='sec_rss_parser/rss.xml')
      run_8k_processor(rss_content=open('sec_rss_parser/rss.xml').read())
    """
    logger.info(f"{LOG_PREFIX} :run_8k_processor: entry rss_file=%s rss_content_len=%s",
                rss_file, len(rss_content) if rss_content else 0)
    processor = EightKFeedProcessor()
    return processor.run(rss_content=rss_content, rss_file=rss_file)
