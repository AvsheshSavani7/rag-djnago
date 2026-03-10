"""
Utilities for 8-K feed processor.

This module contains all helper functions and the SECRSSParser class
extracted from services.py to make the 8-K processor standalone and easier to debug.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xml.etree.ElementTree as ET
from datetime import datetime
import time
import logging
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

# Constants
SEC_BASE_URL = "https://www.sec.gov"
ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
CIK_LENGTH = 10

# Form types
FORM_TYPES = ["8-k", "DEFM14A", "DEFM14C", "PREM14A", "PREM14C", "S-4",
              "S-4/A", "F-4", "F-4/A", "SC 14D9", "SC 14D9/A", "10-Q", "10-K"]


# Helper Functions
def normalize_cik(cik_number):
    """Normalize CIK by padding to 10 digits"""
    return str(cik_number).zfill(CIK_LENGTH) if cik_number else ''


def get_ticker_for_deal_and_cik(deal_id, cik_number):
    """
    Return target_ticker from ProcessingJob if cik_number matches the deal's cik (target)
    or the deal's acquirer_cik. Otherwise return None.
    """
    if not deal_id or cik_number is None:
        return None
    try:
        from bson import ObjectId
        from document_processor.models import ProcessingJob
        job = ProcessingJob.objects.get(id=ObjectId(deal_id))
        cik_norm = normalize_cik(cik_number)
        job_cik = normalize_cik(getattr(job, 'cik', None) or '')
        job_acquirer_cik = normalize_cik(getattr(job, 'acquirer_cik', None) or '')
        if cik_norm and (cik_norm == job_acquirer_cik or cik_norm == job_cik):
            return getattr(job, 'target_ticker', None) or None
        return None
    except Exception:
        return None


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


def build_full_sec_url(url):
    """Build full SEC URL if relative path provided"""
    if not url:
        return None
    if url.startswith('http://') or url.startswith('https://'):
        return url
    return f"{SEC_BASE_URL}{url}"


def find_file_by_type(xbrl_files, file_types, extension='.htm'):
    """Find file in xbrl_files by type and extension"""
    for file in xbrl_files:
        doc_type = file.get('type', '')
        description = file.get('description', '')
        doc_url = file.get('url', '')

        for file_type in file_types if isinstance(file_types, list) else [file_types]:
            if (file_type in doc_type or file_type in description) and doc_url.endswith(extension):
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


class SECRSSParser:
    """Parser for SEC RSS/Atom feeds"""

    def __init__(self, form_type=None):
        self.form_type = "8-K"
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
            time.sleep(2)  # Be respectful to SEC servers
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

            # Extract XBRL files (already filtered to .htm in _extract_xbrl_files)
            xbrl_files = self._extract_xbrl_files(soup, form_type)

            # Build filing_array: only 8-K, EX-2.1, EX-99.1 with .htm (xbrl_files already .htm)
            filing_array = []
            if form_type == "8-K":
                for file in xbrl_files:
                    doc_type = file.get('type', '')
                    desc = file.get('description', '')
                    doc_url = file.get('url', '')
                    if not doc_url or not doc_url.endswith('.htm'):
                        continue
                    document_type = None
                    if 'EX-2.1' in doc_type or 'EX-2.1' in desc:
                        document_type = 'EX-2.1'
                    elif 'EX-99.1' in doc_type or 'EX-99.1' in desc:
                        document_type = 'EX-99.1'
                    elif '8-K' in doc_type or '8-K' in desc:
                        document_type = '8-K'
                    if document_type:
                        url_full = build_full_sec_url(doc_url) or doc_url
                        if url_full and 'ix?doc=/' in url_full:
                            url_full = url_full.replace('ix?doc=/', '', 1)
                        # Include document name (link text from SEC table) and size for email table
                        doc_name = file.get('file', '')
                        if not doc_name and url_full:
                            doc_name = url_full.rstrip('/').split('/')[-1]
                        filing_array.append({
                            'document_type': document_type,
                            'url': url_full,
                            'description': desc or doc_type,
                            'file': doc_name,
                            'size': file.get('size', 0),
                            'sequence': file.get('sequence', 0),
                        })
            # Keep xbrl_files in same shape for SECFiling and email (include file, size, sequence)
            xbrl_files_for_return = [
                {
                    'type': e['document_type'],
                    'url': e['url'],
                    'description': e['description'],
                    'file': e.get('file', ''),
                    'size': e.get('size', 0),
                    'sequence': e.get('sequence', 0),
                }
                for e in filing_array
            ]
            if filing_array:
                log_and_print(
                    f"   filing_array from table: seq/file/size: {[(f.get('sequence'), f.get('file'), f.get('size')) for f in filing_array]}")

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
                'filing_array': filing_array,
                'xbrl_files': xbrl_files_for_return,
            }
        except Exception as e:
            log_and_print(
                f"Error fetching/parsing HTML from {html_url}: {e}", 'error')
            print(f"Exception in fetch_and_parse_html for {html_url}: {e}")
            import traceback
            print(traceback.format_exc())
            return None
