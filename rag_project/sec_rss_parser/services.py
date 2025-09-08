import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xml.etree.ElementTree as ET
from datetime import datetime
import time
import logging
import asyncio
from .models import SECFiling, SECFeedStatus, LastCronJob
from .document_analyzer import SECDocumentAnalyzer
from .websocket_service import SECWebSocketService
from document_processor.models import ProcessingJob

logger = logging.getLogger(__name__)


class SECRSSParser:
    def __init__(self):
        self.feed_url = "https://www.sec.gov/Archives/edgar/usgaap.rss.xml"
        self.headers = {
            "User-Agent":
            "MNA-Finder/1.0 (https://teqnodux.com; contact: ashish.kachadiya@teqnodux.com)",
            'Accept': 'application/rss+xml, application/xml, text/xml, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'https://www.sec.gov/',
        }

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

    def fetch_rss_feed(self):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Add a longer delay to be respectful to SEC servers
                time.sleep(2 + attempt)  # Progressive delay: 2s, 3s, 4s

                response = self.session.get(
                    self.feed_url, headers=self.headers, timeout=30)
                response.raise_for_status()
                # print("response.text[:1000]", response.text[:1200])
                return response.text
            except Exception as e:
                logger.error(
                    f"Error fetching RSS feed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(5)  # Wait 5 seconds before retry
        return None

    def parse_rss_content(self, rss_content):
        try:
            root = ET.fromstring(rss_content)

            # print("root", root)
            last_build_date = root.findall('.//lastBuildDate')
            if last_build_date:
                last_build_date = last_build_date[0].text
            else:
                last_build_date = None
            print("last_build_date", last_build_date)

            # Check if this RSS content has already been processed
            if last_build_date:
                job_name = "sec_rss_feed"
                try:
                    # Try to get existing cron job record
                    existing_job = LastCronJob.objects(
                        job_name=job_name).first()

                    if existing_job and existing_job.last_build_date == last_build_date:
                        print(
                            f"RSS content already processed. Last build date: {last_build_date}")
                        return []

                    # Process items if last_build_date is different or doesn't exist
                    print(
                        f"Processing RSS content with new last_build_date: {last_build_date}")

                except Exception as db_error:
                    logger.error(f"Error checking LastCronJob: {db_error}")
                    print(
                        f"Error checking database, proceeding with processing: {db_error}")

            # Find all item elements
            items = []
            item_elements = root.findall('.//item')
            print(f"Found {len(item_elements)} item elements")

            for i, item in enumerate(item_elements):
                # print(f"\n--- Processing item {i+1} ---")
                item_data = self.parse_item(item, i+1)
                if item_data:
                    items.append(item_data)
                    # print(f"Successfully parsed item {i+1}")
                else:
                    print(f"Failed to parse item {i+1}")

            # Update the last_build_date in database after successful processing
            if last_build_date and items:
                try:
                    job_name = "sec_rss_feed"
                    existing_job = LastCronJob.objects(
                        job_name=job_name).first()

                    if existing_job:
                        existing_job.last_build_date = last_build_date
                        existing_job.save()
                        print(
                            f"Updated existing LastCronJob with new last_build_date: {last_build_date}")
                    else:
                        new_job = LastCronJob(
                            job_name=job_name,
                            last_build_date=last_build_date
                        )
                        new_job.save()
                        print(
                            f"Created new LastCronJob with last_build_date: {last_build_date}")

                except Exception as db_error:
                    logger.error(f"Error updating LastCronJob: {db_error}")
                    print(f"Error updating database: {db_error}")

            print(f"Total items parsed: {len(items)}")
            return items
        except Exception as e:
            logger.error(f"Error parsing RSS XML: {e}")
            print(f"Exception in parse_rss_content: {e}")
            return []

    def parse_item(self, item_elem, item_number):
        try:
            # print("item_elem", item_elem, item_number)
            title = self.get_text(item_elem, 'title')
            link = self.get_text(item_elem, 'link')
            guid = self.get_text(item_elem, 'guid')
            description = self.get_text(item_elem, 'description')
            pubDate = self.get_text(item_elem, 'pubDate')

            # print(f"Parsing item: {title} - {description}")

            # Try different ways to find the edgar element
            edgar_elem = item_elem.find(
                './/{https://www.sec.gov/Archives/edgar}edgar:xbrlFiling')
            if edgar_elem is None:
                # Try with the actual namespace used in the XML
                edgar_elem = item_elem.find(
                    './/{https://www.sec.gov/Archives/edgar}xbrlFiling')
            if edgar_elem is None:
                # Try with just the tag name
                edgar_elem = item_elem.find('.//xbrlFiling')

            if edgar_elem is None:
                # print(f"No edgar element found for: {title}")
                # Try to find any element with 'xbrlFiling' in the name
                all_elems = item_elem.findall('.//*')
                xbrl_elems = [
                    elem for elem in all_elems if 'xbrlFiling' in elem.tag]
                # print(
                #     f"Found {len(xbrl_elems)} elements with 'xbrlFiling' in tag: {[elem.tag for elem in xbrl_elems]}")
                if xbrl_elems:
                    edgar_elem = xbrl_elems[0]
                    # print(f"Using first xbrlFiling element: {edgar_elem.tag}")
                else:
                    return None

            # print(f"Found edgar element for: {title}")

            company_name = self.get_text(
                edgar_elem, '{https://www.sec.gov/Archives/edgar}companyName')
            if not company_name:
                company_name = self.get_text(edgar_elem, 'companyName')

            form_type = self.get_text(
                edgar_elem, '{https://www.sec.gov/Archives/edgar}formType')
            if not form_type:
                form_type = self.get_text(edgar_elem, 'formType')
            filing_date = self.get_text(
                edgar_elem, '{https://www.sec.gov/Archives/edgar}filingDate')
            if not filing_date:
                filing_date = self.get_text(edgar_elem, 'filingDate')
                # print("filing_date1", filing_date)

            file_number = self.get_text(
                edgar_elem, '{https://www.sec.gov/Archives/edgar}fileNumber')
            # print("file_number1", file_number)
            if not file_number:
                file_number = self.get_text(edgar_elem, 'fileNumber')

            period = self.get_text(
                edgar_elem, '{https://www.sec.gov/Archives/edgar}period')
            if not period:
                period = self.get_text(edgar_elem, 'period')

            fiscal_year_end = self.get_text(
                edgar_elem, '{https://www.sec.gov/Archives/edgar}fiscalYearEnd')
            if not fiscal_year_end:
                fiscal_year_end = self.get_text(edgar_elem, 'fiscalYearEnd')

            cik_number = self.get_text(
                edgar_elem, '{https://www.sec.gov/Archives/edgar}cikNumber')
            if not cik_number:
                cik_number = self.get_text(edgar_elem, 'cikNumber')

            accession_number = self.get_text(
                edgar_elem, '{https://www.sec.gov/Archives/edgar}accessionNumber')
            if not accession_number:
                accession_number = self.get_text(edgar_elem, 'accessionNumber')

            acceptance_datetime_utc = self.get_text(
                edgar_elem, '{https://www.sec.gov/Archives/edgar}acceptanceDatetime')
            if not acceptance_datetime_utc:
                acceptance_datetime_utc = self.get_text(
                    edgar_elem, 'acceptanceDatetime')

            assigned_sic = self.get_text(
                edgar_elem, '{https://www.sec.gov/Archives/edgar}assignedSic')
            if not assigned_sic:
                assigned_sic = self.get_text(edgar_elem, 'assignedSic')

            # Convert acceptance datetime to UTC ISO format
            if acceptance_datetime_utc:
                try:
                    from datetime import datetime
                    from zoneinfo import ZoneInfo

                    # Parse the raw datetime string (format: YYYYMMDDHHMMSS)
                    dt_et = datetime.strptime(acceptance_datetime_utc, "%Y%m%d%H%M%S").replace(
                        tzinfo=ZoneInfo("America/New_York"))
                    dt_utc_iso = dt_et.astimezone(ZoneInfo("UTC")).isoformat()
                    acceptance_datetime_utc = dt_utc_iso
                    # print(
                    #     f"Converted acceptance datetime: {acceptance_datetime_utc}")
                except Exception as e:
                    print(f"Error converting acceptance datetime: {e}")
                    acceptance_datetime_utc = None

            # print(f"Extracted: {company_name} - {form_type} - {cik_number}")

            xbrl_files = self.parse_xbrl_files(edgar_elem, form_type)

            # print("xbrl_files1", xbrl_files)

            # Check if this is an 8-K filing with EX-2.1 file that ends with .htm
            has_ex21_htm = (
                form_type == "8-K" and
                any(
                    file.get('type') == 'EX-2.1' and
                    file.get('url', '').endswith('.htm')
                    for file in xbrl_files
                )
            )

            # Check if this is a DEF 14A or PRE 14A filing with xbrlFiles
            has_def14a_files = (
                form_type in ["DEF 14A", "PRE 14A"] and
                len(xbrl_files) > 0
            )

            # print("has_ex21_htm", has_ex21_htm)
            # print("has_def14a_files", has_def14a_files)

            if has_ex21_htm:
                # print(
                #     f"✅ Found 8-K filing with EX-2.1 HTM file: {company_name}")
                return {
                    'title': title,
                    'link': link,
                    'guid': guid,
                    'description': description,
                    'pubDate': pubDate or None,
                    'company_name': company_name,
                    'form_type': form_type,
                    'filing_date': filing_date or None,
                    'cik_number': cik_number,
                    'file_number': file_number,
                    'accession_number': accession_number,
                    'acceptance_datetime_utc': acceptance_datetime_utc,
                    'period': period or None,
                    'fiscal_year_end': fiscal_year_end or None,
                    'assigned_sic': assigned_sic or None,
                    'xbrl_files': xbrl_files,
                    'has_htm_files': has_ex21_htm  # Only set to True for EX-2.1 HTM files
                }
            elif has_def14a_files:
                # print(
                #     f"✅ Found DEF 14A/PRE 14A filing with xbrlFiles: {company_name}")
                return {
                    'title': title,
                    'link': link,
                    'guid': guid,
                    'description': description,
                    'pubDate': pubDate or None,
                    'company_name': company_name,
                    'form_type': form_type,
                    'filing_date': filing_date or None,
                    'cik_number': cik_number,
                    'file_number': file_number,
                    'accession_number': accession_number,
                    'acceptance_datetime_utc': acceptance_datetime_utc,
                    'period': period or None,
                    'fiscal_year_end': fiscal_year_end or None,
                    'assigned_sic': assigned_sic or None,
                    'xbrl_files': xbrl_files,
                    'has_htm_files': False  # Set to False for DEF 14A/PRE 14A
                }
            else:
                if form_type == "8-K":
                    print(
                        f"❌ Skipping 8-K filing: {company_name} (no EX-2.1 HTM file)")
                elif form_type in ["DEF 14A", "PRE 14A"]:
                    print(
                        f"❌ Skipping DEF 14A/PRE 14A filing: {company_name} (no xbrlFiles)")
                else:
                    print(
                        f"⏭️ Skipping non-target filing: {company_name} - {form_type}")
                return None
        except Exception as e:
            logger.error(f"Error parsing RSS item: {e}")
            print(f"Exception parsing item: {e}")
            return None

    def parse_xbrl_files(self, edgar_elem, form_type=None):
        xbrl_files = []
        # print("edgar_elem1", edgar_elem)

        # Try different ways to find xbrlFiles element
        xbrl_files_elem = edgar_elem.find(
            './/{https://www.sec.gov/Archives/edgar}xbrlFiles')
        if xbrl_files_elem is None:
            xbrl_files_elem = edgar_elem.find('.//xbrlFiles')

        if xbrl_files_elem is not None:
            # Try different ways to find xbrlFile elements
            file_elems = xbrl_files_elem.findall(
                './/{https://www.sec.gov/Archives/edgar}xbrlFile')
            # print("file_elems1", file_elems)
            if not file_elems:
                file_elems = xbrl_files_elem.findall('.//xbrlFile')

            for file_elem in file_elems:
                # Try to get attributes with edgar namespace first, then fallback to regular attributes
                sequence = file_elem.get(
                    '{https://www.sec.gov/Archives/edgar}sequence') or file_elem.get('sequence', '0')
                file_name = file_elem.get(
                    '{https://www.sec.gov/Archives/edgar}file') or file_elem.get('file', '')
                file_type = file_elem.get(
                    '{https://www.sec.gov/Archives/edgar}type') or file_elem.get('type', '')
                file_size = file_elem.get(
                    '{https://www.sec.gov/Archives/edgar}size') or file_elem.get('size', '0')
                file_description = file_elem.get(
                    '{https://www.sec.gov/Archives/edgar}description') or file_elem.get('description', '')
                file_url = file_elem.get(
                    '{https://www.sec.gov/Archives/edgar}url') or file_elem.get('url', '')

                file_data = {
                    'sequence': int(sequence),
                    'file': file_name,
                    'type': file_type,
                    'size': int(file_size),
                    'description': file_description,
                    'url': file_url
                }

                # For 8-K filings, only save EX-2.1 HTM files
                if form_type == "8-K" and file_type == "EX-2.1" and file_url.endswith(".htm"):
                    xbrl_files.append(file_data)
                    # print(
                    #     f"Found XBRL file: {file_data['file']} - {file_data['type']} - {file_data['url']}")

                # For DEF 14A and PRE 14A filings, save all xbrlFiles when edgar:type matches
                elif form_type in ["DEF 14A", "PRE 14A"] and (file_type == "DEF 14A" or file_type == "PRE 14A") and file_url.endswith(".htm"):
                    # Save all xbrlFiles for DEF 14A/PRE 14A (edgar:type is already matched in the XML)
                    xbrl_files.append(file_data)
                    # print(
                    #     f"Found DEF 14A/PRE 14A XBRL file: {file_data['file']} - {file_data['type']} - {file_data['url']}")

        # print(f"Total XBRL files found: {len(xbrl_files)}")
        return xbrl_files

    def get_text(self, elem, tag):
        child = elem.find(tag)
        return child.text if child is not None else ''


class SECFeedProcessor:
    def __init__(self):
        self.parser = SECRSSParser()
        self.document_analyzer = SECDocumentAnalyzer()

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

    def process_feed(self):
        try:
            rss_content = self.parser.fetch_rss_feed()
            print("rss_content1", rss_content[:1000])

            if not rss_content:
                return {'success': False, 'error': 'Failed to fetch RSS feed'}

            items = self.parser.parse_rss_content(rss_content)
            # print("items", items)
            new_items_count = 0

            for item_data in items:
                if self.save_filing(item_data):
                    new_items_count += 1

            # Emit processing statistics via WebSocket
            if new_items_count > 0:
                stats = {
                    'total_processed': len(items),
                    'new_filings': new_items_count,
                    'processing_time': datetime.utcnow().isoformat(),
                    'feed_url': self.parser.feed_url
                }
                SECWebSocketService.emit_sec_processing_stats(stats)

            return {
                'success': True,
                'message': f'Processed {len(items)} items, {new_items_count} new',
                'total_items': len(items),
                'new_items': new_items_count
            }
        except Exception as e:
            logger.error(f"Error processing SEC feed: {e}")
            return {'success': False, 'error': str(e)}

    def save_filing(self, item_data):
        try:
            # Check if filing already exists
            existing = SECFiling.objects(
                accession_number=item_data['accession_number']).first()
            if existing:
                return False

            # Analyze document with GPT before saving
            if (item_data.get('form_type') == '8-K' and
                item_data.get('has_htm_files') and
                any(file.get('type') == 'EX-2.1' and file.get('url', '').endswith('.htm')
                    for file in item_data.get('xbrl_files', []))):

                logger.info(
                    f"🔍 Analyzing 8-K document for: {item_data.get('company_name')}")
                item_data = self.document_analyzer.analyze_filing(item_data)

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
            elif item_data.get('form_type') in ['DEF 14A', 'PRE 14A']:
                # Analyze DEF 14A/PRE 14A documents for document kind detection
                logger.info(
                    f"🔍 Analyzing DEF 14A/PRE 14A document for: {item_data.get('company_name')}")
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
            if item_data.get('filing_date') and isinstance(item_data['filing_date'], str):
                try:
                    from datetime import datetime
                    # Parse MM/DD/YYYY format
                    item_data['filing_date'] = datetime.strptime(
                        item_data['filing_date'], '%m/%d/%Y')
                except Exception as e:
                    print(
                        f"Error converting filing_date back to datetime: {e}")
                    item_data['filing_date'] = None

            # Create and save the filing
            filing = SECFiling(**item_data)
            filing.save()

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
                'document_kind': filing.document_kind
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

            return True

        except Exception as e:
            logger.error(f"Error saving filing: {e}")
            return False
