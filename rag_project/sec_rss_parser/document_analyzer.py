import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import re
from bs4 import BeautifulSoup
import logging
import openai
import os
from typing import Optional, Dict, Any
import json

logger = logging.getLogger(__name__)


class SECDocumentAnalyzer:
    """Service to download and analyze SEC documents using GPT"""

    def __init__(self):
        self.headers = {
            "User-Agent": "MNA-Finder/1.0 (https://teqnodux.com; contact: ashish.kachadiya@teqnodux.com)",
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
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

        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"))

    def download_htm_file(self, url: str) -> Optional[str]:
        """Download HTM file from SEC URL"""
        try:
            # Add delay to be respectful to SEC servers
            time.sleep(2)

            response = self.session.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            logger.info(f"Successfully downloaded HTM file from: {url}")
            return response.text

        except Exception as e:
            logger.error(f"Error downloading HTM file from {url}: {e}")
            return None

    def extract_document_pages(self, html_content: str, max_pages: int = 4) -> str:
        """Extract first few pages of content from HTML document"""
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text content
            text = soup.get_text()

            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip()
                      for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            # Estimate pages (assuming ~3000 characters per page)
            chars_per_page = 3000
            max_chars = chars_per_page * max_pages

            if len(text) > max_chars:
                text = text[:max_chars] + "..."

            logger.info(f"Extracted {len(text)} characters from document")
            return text

        except Exception as e:
            logger.error(f"Error extracting document pages: {e}")
            return ""

    def analyze_document_with_gpt(self, document_text: str, company_name: str) -> Dict[str, Any]:
        """Analyze document with GPT to determine if it's a new deal or amendment"""
        try:
            if not self.openai_client.api_key:
                logger.error("OpenAI API key not configured")
                return {
                    'is_new_deal': None,
                    'document_kind': None,
                    'confidence': 0,
                    'reasoning': 'OpenAI API key not configured',
                    'error': 'API key missing'
                }

            prompt = f"""
You are an expert in analyzing SEC filings and merger & acquisition documents. 

Please analyze the following document excerpt from a Form 8-K filing by {company_name} and determine:

1. Is this a NEW DEAL/MERGER/ACQUISITION AGREEMENT or an AMENDMENT to an existing deal/agreement?
2. Classify the **document kind** precisely.

Document excerpt:
{document_text}

Please respond with a JSON object containing:
- "classification": either "new_deal" or "amendment"
- "document_kind": one of the categories below
- "confidence": a number from 0-100
- "reasoning": brief explanation
- "key_indicators": list of key phrases or sections that led to your conclusion

---

### Classification Rules

**Deal Classification**
- "new_deal" → if it is a new agreement (merger, acquisition, sale, or reorganization).
- "amendment" → if it modifies, amends, or restates a prior agreement.

**Document Kind**
Choose the most precise one:

- "mna_definitive" → ONLY for **Agreement and Plan of Merger** (true third-party merger agreements with purchase price/consideration, reps & warranties, covenants, indemnities, disclosure schedules, etc.).  
  ⚠️ Do NOT use this for Stock Purchase Agreements or Asset Purchase Agreements or Transaction Agreement.

- "purchase_agreement" → for Stock Purchase Agreements, Asset Purchase Agreements, Equity Purchase Agreements, Membership Interest Purchase Agreements, etc.

- "transaction_agreement" → labeled explicitly as "Transaction Agreement" (broader than merger or purchase agreement).

- "joint_venture_agreement" → agreements establishing a joint venture or strategic alliance.

- "support_or_voting_agreement" → shareholder support, tender support, or voting agreements.

- "tender_offer_agreement" → tender offer or acquisition offer agreements.

- "mna_initial" → preliminary non-binding agreements (LOI, MOU, Term Sheet, Expression of Interest, press release).

- "amendment" → amendment or modification to an existing agreement.

- "reincorporation_merger" → parent-subsidiary merger, reincorporation, change of domicile, short-form merger.

- "internal_reorganization" → intra-group reorganization, simplification agreement.

- "other_corporate_agreement" → fallback if none of the above fit.

---

Respond only with valid JSON.
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert SEC filing analyst. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content.strip()

            # Parse JSON response
            print(f"GPT Response: {result_text}")

            result = json.loads(result_text)

            # Validate and normalize response
            classification = result.get('classification', '').lower()
            is_new_deal = classification == 'new_deal'

            analysis_result = {
                'is_new_deal': is_new_deal,
                'document_kind': result.get('document_kind', ''),
                'confidence': result.get('confidence', 0),
                'reasoning': result.get('reasoning', ''),
                'key_indicators': result.get('key_indicators', []),
                'raw_response': result_text
            }

            logger.info(
                f"GPT Analysis Result: {classification} (confidence: {result.get('confidence', 0)}%)")
            return analysis_result

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing GPT JSON response: {e}")
            return {
                'is_new_deal': None,
                'document_kind': None,
                'confidence': 0,
                'reasoning': 'Failed to parse GPT response',
                'error': str(e)
            }
        except Exception as e:
            logger.error(f"Error analyzing document with GPT: {e}")
            return {
                'is_new_deal': None,
                'document_kind': None,
                'confidence': 0,
                'reasoning': 'GPT analysis failed',
                'error': str(e)
            }

    def analyze_filing(self, filing_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method to analyze a filing
        Returns updated filing_data with is_new_deal and following fields
        """
        try:
            # Check if this is an 8-K with EX-2.1 HTM files
            if filing_data.get('form_type') != '8-K':
                logger.info("Skipping analysis - not an 8-K filing")
                return filing_data

            xbrl_files = filing_data.get('xbrl_files', [])
            ex21_files = [
                file for file in xbrl_files
                if file.get('type') == 'EX-2.1' and file.get('url', '').endswith('.htm')
            ]

            if not ex21_files:
                logger.info("Skipping analysis - no EX-2.1 HTM files found")
                return filing_data

            # Analyze the first EX-2.1 file
            ex21_file = ex21_files[0]
            htm_url = ex21_file.get('url')

            logger.info(f"Analyzing EX-2.1 file: {htm_url}")

            # Download document
            html_content = self.download_htm_file(htm_url)
            if not html_content:
                logger.error("Failed to download HTM file")
                filing_data['is_new_deal'] = None
                filing_data['following'] = False
                return filing_data

            # Extract pages
            document_text = self.extract_document_pages(
                html_content, max_pages=4)
            if not document_text:
                logger.error("Failed to extract document text")
                filing_data['is_new_deal'] = None
                filing_data['following'] = False
                return filing_data

            # Analyze with GPT
            analysis = self.analyze_document_with_gpt(
                document_text,
                filing_data.get('company_name', 'Unknown Company')
            )

            # Update filing data based on analysis
            if analysis.get('is_new_deal') is True:
                # New deal
                filing_data['is_new_deal'] = True
                filing_data['following'] = False
                filing_data['document_kind'] = analysis.get(
                    'document_kind', '')
                logger.info(
                    f"✅ Classified as NEW DEAL: {filing_data.get('company_name')}")
            elif analysis.get('is_new_deal') is False:
                # Amendment
                filing_data['is_new_deal'] = False
                filing_data['document_kind'] = analysis.get(
                    'document_kind', '')
                # You can modify this logic if needed
                filing_data['following'] = False
                logger.info(
                    f"📝 Classified as AMENDMENT: {filing_data.get('company_name')}")
            else:
                # Analysis failed
                filing_data['is_new_deal'] = None
                filing_data['following'] = False
                filing_data['document_kind'] = None

                logger.warning(
                    f"❓ Analysis inconclusive: {filing_data.get('company_name')}")

            # Note: GPT analysis metadata is logged but not stored in the document
            # to avoid schema conflicts. Consider adding these fields to the model if needed.

            return filing_data

        except Exception as e:
            logger.error(f"Error in analyze_filing: {e}")
            filing_data['is_new_deal'] = None
            filing_data['following'] = False
            return filing_data
