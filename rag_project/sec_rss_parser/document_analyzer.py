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

            Please analyze the following document excerpt from a Form 8-K/2.1 filing by {company_name} and determine:

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

            **Document Kind (choose EXACTLY one; use these exact strings only)**
            - "Definitive Merger Agreement"
            - "Business Combination Agreement"
            - "Stock Purchase / Share Exchange Agreement"
            - "Asset Purchase Agreement"
            - "Plan of Reorganization (Bankruptcy)"
            - "Plan of Liquidation / Dissolution"
            - "Succession / Arrangement Plans"
            - "Amendment"
            - "Other Corporate Agreement" 

            ### Title Keyword Hints (for recognition only; NEVER copy these into output unless they appear verbatim in the excerpt)

            - Definitive Merger Agreement → "Agreement and Plan of Merger" or "Arrangement Agreement and plan of Merger"; 
            - Business Combination Agreement → "Business Combination Agreement"; 
            - Stock Purchase / Share Exchange Agreement → "Stock Purchase Agreement"; "Share Exchange Agreement"
            - Asset Purchase Agreement → "Asset Purchase Agreement"; "Bill of Sale"
            - Plan of Reorganization (Bankruptcy) → "Plan of Reorganization"; "Joint Prepackaged Plan"
            - Plan of Liquidation / Dissolution → "Plan of Liquidation"; "Plan of Dissolution"
            - Succession / Arrangement Plans → "Succession Agreement"; "Arrangement Plan"; "Corporate Arrangement"
            - Amendment → "Amendment"; "Modification"; "Restatement"
            - Other Corporate Agreement → fallback if none of the above fit.


            - Preserve **original text exactly as it appears** for anything quoted in "key_indicators" and referenced in "reasoning".
            - **Do not normalize, autocorrect, or map** phrases.
            - Example: if the excerpt says **"Plan of recoganization"** (typo), output **"Plan of recoganization"** exactly in "key_indicators".
            - Keep original casing, punctuation, hyphenation, whitespace, and typos.
            - Use short verbatim snippets (≤ 12 words) that directly justify the decision.

            Important:
            - Use the hints only to choose the closest `document_kind` from the fixed list above.
            - Output must be valid JSON.

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

    @staticmethod
    def _normalize_cik(cik: str) -> str:
        """Ensure CIK is 10 digits with leading zeros if needed."""
        if not cik or not isinstance(cik, str):
            return cik or ""
        digits = re.sub(r"\D", "", cik)
        if not digits:
            return cik.strip()
        return digits.zfill(10)

    def analyze_company_details_with_gpt(self, document_text: str, company_name: str) -> Dict[str, Any]:
        """Analyze document with GPT to extract target/acquirer company details."""
        try:
            if not self.openai_client.api_key:
                logger.error("OpenAI API key not configured")
                return {
                    'target_name': '',
                    'target_cik': '',
                    'acquirer_name': '',
                    'acquirer_cik': '',
                    'is_acquirer_us_listed': None,
                    'is_acquirer_cap_greater_than_100m': None,
                    'error': 'API key missing'
                }

            full_prompt = f"""You are an expert SEC filing analyst with web search access. Use web search to verify company listing status and market cap.

Analyze the following document excerpt from a Form 8-K/2.1 filing by {company_name}.

Document excerpt:
{document_text}

Extract the following information. First extract company names the document. Then use web search to verify listing status and market cap for the target company and CIKs for both target and acquirer.

Respond with a JSON object containing:

1. "target_name" (string): The full legal name of the target company being acquired (from document).
2. "target_cik" (string): The SEC Central Index Key (CIK) of the target. Output digits only; it will be normalized to 10 digits with leading zeros elsewhere.
3. "acquirer_name" (string): The full legal name of the acquirer company (the one doing the acquisition, from document).
4. "acquirer_cik" (string): The SEC CIK of the acquirer. Output digits only; it will be normalized to 10 digits with leading zeros elsewhere.
5. "is_target_us_listed" (boolean): USE WEB SEARCH to verify if the target is currently listed on a US stock exchange (NYSE, NASDAQ, etc.). Set to true if listed, false if not listed or delisted, null if cannot determine.
6. "is_target_market_cap_greater_than_100m" (boolean): USE WEB SEARCH to find the current market capitalization of the target company. Set to true if market cap is greater than $100 million USD, false if less than $100M, null if cannot determine.

IMPORTANT:
- Extract target_name and acquirer_name from the document excerpt above
- For acquirer_cik, target_cik, is_target_us_listed and is_target_market_cap_greater_than_100m, you MUST perform web searches to get current, accurate information
- Search for "[target company name] stock exchange listing" and "[target company name] market cap"
- If information cannot be found in document, use empty string "" for strings and null for booleans

Respond only with valid JSON.
"""

            response = self.openai_client.responses.create(
                model="gpt-5",
                tools=[{"type": "web_search"}],
                input=full_prompt,
                reasoning={"effort": "low"}
            )

            print(f"GPT Response: {response}")

            # Extract text from Responses API output
            result_text = None
            for item in response.output:
                if item.type == 'message' and hasattr(item, 'content'):
                    for content_item in item.content:
                        if content_item.type == 'output_text':
                            result_text = content_item.text
                            break
                if result_text:
                    break

            if not result_text:
                raise ValueError("No text output found in response")

            # Extract JSON from response (handle markdown code blocks or plain JSON)
            result_text = result_text.strip()

            # Remove markdown code blocks if present
            if result_text.startswith('```'):
                # Find the JSON content between ```json and ``` or ``` and ```
                lines = result_text.split('\n')
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.strip().startswith('```'):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        json_lines.append(line)
                result_text = '\n'.join(json_lines).strip()

            # Try to find JSON object if there's extra text
            if not result_text.startswith('{'):
                # Look for first { and last }
                start = result_text.find('{')
                end = result_text.rfind('}')
                if start != -1 and end != -1:
                    result_text = result_text[start:end+1]

            result = json.loads(result_text)

            target_cik = result.get('target_cik', '')
            acquirer_cik = result.get('acquirer_cik', '')
            if isinstance(target_cik, (int, float)):
                target_cik = str(int(target_cik))
            else:
                target_cik = str(target_cik or '').strip()
            if isinstance(acquirer_cik, (int, float)):
                acquirer_cik = str(int(acquirer_cik))
            else:
                acquirer_cik = str(acquirer_cik or '').strip()

            company_details = {
                'target_name': (result.get('target_name') or '').strip(),
                'target_cik': self._normalize_cik(target_cik),
                'acquirer_name': (result.get('acquirer_name') or '').strip(),
                'acquirer_cik': self._normalize_cik(acquirer_cik),
                'is_target_us_listed': result.get('is_target_us_listed'),
                'is_target_market_cap_greater_than_100m': result.get('is_target_market_cap_greater_than_100m'),
            }

            logger.info(
                f"GPT Company details: target={company_details['target_name']}, acquirer={company_details['acquirer_name']}")
            return company_details

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing GPT company details JSON: {e}")
            return {
                'target_name': '',
                'target_cik': '',
                'acquirer_name': '',
                'acquirer_cik': '',
                'is_target_us_listed': None,
                'is_target_market_cap_greater_than_100m': None,
                'error': str(e)
            }
        except Exception as e:
            logger.error(f"Error analyzing company details with GPT: {e}")
            return {
                'target_name': '',
                'target_cik': '',
                'acquirer_name': '',
                'acquirer_cik': '',
                'is_target_us_listed': None,
                'is_target_market_cap_greater_than_100m': None,
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
                html_content, max_pages=5)
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

            if analysis.get('document_kind') == 'Definitive Merger Agreement':

                # Extract company details (target/acquirer) with GPT
                company_details = self.analyze_company_details_with_gpt(
                    document_text,
                    filing_data.get('company_name', 'Unknown Company')
                )
                filing_data['company_details'] = company_details

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

    def analyze_ex99_1_document_with_gpt(self, document_text: str, company_name: str) -> Dict[str, Any]:
        """Analyze EX-99.1 document with GPT (with web search) to determine if it's merger-related.
        Returns is_merger_related, confidence, reasoning, is_target_us_listed, is_target_market_cap_greater_than_100m."""
        try:
            if not self.openai_client.api_key:
                logger.error("OpenAI API key not configured")
                return {
                    'is_merger_related': None,
                    'confidence': 0,
                    'reasoning': 'OpenAI API key not configured',
                    'is_target_us_listed': None,
                    'is_target_market_cap_greater_than_100m': None,
                    'error': 'API key missing'
                }

            prompt = f"""
You are an expert in analyzing SEC filings and M&A disclosures with web search access.

You are reviewing an EX-99.1 exhibit attached to Form 8-K.

Company (filer): {company_name}

Document excerpt:
{document_text}

Objective:
1. Determine whether THIS press release announces that the company has JUST entered into (or just signed) a NEW merger, acquisition, or business combination agreement involving corporate ownership or control.
2. If it is merger-related, identify the TARGET company (the company being acquired) from the document.
3. USE WEB SEARCH to verify the target's US listing status and market cap.

Core Question:
Is this document announcing that the company has just signed a binding agreement that results in:

Acquisition of another company’s equity or voting control
Being acquired by another company
A statutory merger
A business combination
A change of control transaction
A reverse merger
A stock purchase or share exchange resulting in control

IMPORTANT:
The transaction may not yet be closed.
It may be subject to regulatory or shareholder approval.
These still qualify as NEW if just signed.

DO NOT classify as NEW if the document announces:
Asset sales
Divestitures
Sale of business units or portfolios
Infrastructure sales
Sale of towers, properties, assets, or subsidiaries
Strategic partnerships without equity acquisition
Financing transactions
Debt repayment plans
Previously announced deals
Closing of prior deals

Decision Rule:
Return true ONLY if a newly signed agreement results in a merger, acquisition of equity control, or business combination between corporate entities.

For is_target_us_listed and is_target_market_cap_greater_than_100m:
- First identify the target company name from the document.
- USE WEB SEARCH to verify if the target is currently listed on a US stock exchange (NYSE, NASDAQ, etc.). Set is_target_us_listed to true if listed, false if not listed or delisted, null if cannot determine.
- USE WEB SEARCH to find the current market capitalization of the target company. Set is_target_market_cap_greater_than_100m to true if market cap is greater than $100 million USD, false if less than $100M, null if cannot determine.
- Search for "[target company name] stock exchange listing" and "[target company name] market cap"
- If the document is NOT merger-related, set both to null.

Respond ONLY with valid JSON:

```json
{{
  "is_merger_related": boolean,
  "confidence": number (0-100),
  "reasoning": "concise explanation",
  "is_target_us_listed": boolean or null,
  "is_target_market_cap_greater_than_100m": boolean or null
}}
```

Respond ONLY with valid JSON.
"""

            response = self.openai_client.responses.create(
                model="gpt-5",
                tools=[{"type": "web_search"}],
                input=prompt,
                reasoning={"effort": "low"}
            )

            result_text = None
            for item in response.output:
                if item.type == 'message' and hasattr(item, 'content'):
                    for content_item in item.content:
                        if content_item.type == 'output_text':
                            result_text = content_item.text
                            break
                if result_text:
                    break

            if not result_text:
                raise ValueError("No text output found in response")

            result_text = result_text.strip()

            # Remove markdown code blocks if present
            if result_text.startswith('```'):
                lines = result_text.split('\n')
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.strip().startswith('```'):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        json_lines.append(line)
                result_text = '\n'.join(json_lines).strip()

            if not result_text.startswith('{'):
                start = result_text.find('{')
                end = result_text.rfind('}')
                if start != -1 and end != -1:
                    result_text = result_text[start:end+1]

            print(f"GPT Response for EX-99.1: {result_text}")

            result = json.loads(result_text)

            analysis_result = {
                'is_merger_related': result.get('is_merger_related'),
                'confidence': result.get('confidence', 0),
                'reasoning': result.get('reasoning', ''),
                'is_target_us_listed': result.get('is_target_us_listed'),
                'is_target_market_cap_greater_than_100m': result.get('is_target_market_cap_greater_than_100m'),
                'raw_response': result_text
            }

            logger.info(
                f"GPT EX-99.1 Analysis Result: is_merger_related={result.get('is_merger_related')} "
                f"(confidence: {result.get('confidence', 0)}%) "
                f"is_target_us_listed={result.get('is_target_us_listed')} "
                f"is_target_market_cap_greater_than_100m={result.get('is_target_market_cap_greater_than_100m')}")
            return analysis_result

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing GPT JSON response for EX-99.1: {e}")
            return {
                'is_merger_related': None,
                'confidence': 0,
                'reasoning': 'Failed to parse GPT response',
                'is_target_us_listed': None,
                'is_target_market_cap_greater_than_100m': None,
                'error': str(e)
            }
        except Exception as e:
            logger.error(f"Error analyzing EX-99.1 document with GPT: {e}")
            return {
                'is_merger_related': None,
                'confidence': 0,
                'reasoning': 'GPT analysis failed',
                'is_target_us_listed': None,
                'is_target_market_cap_greater_than_100m': None,
                'error': str(e)
            }

    def analyze_ex99_1_filing(self, filing_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an 8-K filing that has EX-99.1 (press release/material exhibit).
        Uses LLM to determine if the document is related to a new merger.
        Returns updated filing_data with is_merger_related, ex99_1_confidence, ex99_1_reasoning.
        """
        try:
            if filing_data.get('form_type') != '8-K':
                logger.info("Skipping EX-99.1 analysis - not an 8-K filing")
                return filing_data

            xbrl_files = filing_data.get('xbrl_files', [])
            ex99_1_files = [
                file for file in xbrl_files
                if ('EX-99.1' in file.get('type', '') or 'EX-99.1' in file.get('description', ''))
                and file.get('url', '').endswith('.htm')
            ]

            if not ex99_1_files:
                logger.info(
                    "Skipping EX-99.1 analysis - no EX-99.1 HTM files found")
                filing_data['is_merger_related'] = None
                filing_data['ex99_1_confidence'] = 0
                return filing_data

            ex99_1_file = ex99_1_files[0]
            htm_url = ex99_1_file.get('url')

            logger.info(f"Analyzing EX-99.1 file: {htm_url}")

            html_content = self.download_htm_file(htm_url)
            if not html_content:
                logger.error("Failed to download EX-99.1 HTM file")
                filing_data['is_merger_related'] = None
                filing_data['ex99_1_confidence'] = 0
                return filing_data

            document_text = self.extract_document_pages(
                html_content, max_pages=5)
            if not document_text:
                logger.error("Failed to extract EX-99.1 document text")
                filing_data['is_merger_related'] = None
                filing_data['ex99_1_confidence'] = 0
                return filing_data

            analysis = self.analyze_ex99_1_document_with_gpt(
                document_text,
                filing_data.get('company_name', 'Unknown Company')
            )

            print(f"EX-99.1 Analysis: {analysis}")

            filing_data['is_merger_related'] = analysis.get(
                'is_merger_related')
            filing_data['ex99_1_confidence'] = analysis.get('confidence', 0)
            filing_data['ex99_1_reasoning'] = analysis.get('reasoning', '')
            filing_data['is_target_us_listed'] = analysis.get('is_target_us_listed')
            filing_data['is_target_market_cap_greater_than_100m'] = analysis.get('is_target_market_cap_greater_than_100m')

            return filing_data

        except Exception as e:
            logger.error(f"Error in analyze_ex99_1_filing: {e}")
            filing_data['is_merger_related'] = None
            filing_data['ex99_1_confidence'] = 0

            return filing_data

    def analyze_def14a_document_with_gpt(self, document_text: str, company_name: str) -> Dict[str, Any]:
        """Analyze DEF 14A document with GPT to determine document kind based on checkbox text"""
        try:
            if not self.openai_client.api_key:
                logger.error("OpenAI API key not configured")
                return {
                    'document_kind': None,
                    'confidence': 0,
                    'reasoning': 'OpenAI API key not configured',
                    'error': 'API key missing'
                }

            prompt = f"""
You are an expert in analyzing SEC proxy statements (DEF 14A and PRE 14A).

Please analyze the following document excerpt from a DEF 14A/PRE 14A filing by {company_name} and determine the document kind based on the checkbox selection.

Document excerpt:
{document_text}

Look for the checkbox section that typically appears like this:

Check the appropriate box:

☐ Preliminary Proxy Statement.
☐ Confidential, for Use of the Commission Only (as permitted by Rule 14a-6(e)(2))
☒ Definitive Proxy Statement.
☐ Definitive Additional Materials.
☐ Soliciting Material under §240.14a-12.

Please respond with a JSON object containing:
- "document_kind": one of the categories below based on which checkbox is selected
- "confidence": a number from 0-100
- "reasoning": brief explanation of which checkbox was found and selected
- "checkbox_found": boolean indicating if the checkbox section was found

---

### Document Kind Categories

Based on the checkbox selection:

- "preliminary_proxy" → if "Preliminary Proxy Statement" is checked
- "definitive_proxy" → if "Definitive Proxy Statement" is checked  
- "definitive_additional_materials" → if "Definitive Additional Materials" is checked
- "soliciting_material" → if "Soliciting Material under §240.14a-12" is checked
- "confidential_commission_only" → if "Confidential, for Use of the Commission Only" is checked
- "unknown" → if checkbox section is not found or unclear

---

Respond only with valid JSON.
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert SEC proxy statement analyst. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content.strip()

            # Parse JSON response
            print(f"GPT Response for DEF 14A: {result_text}")

            result = json.loads(result_text)

            analysis_result = {
                'document_kind': result.get('document_kind', 'unknown'),
                'confidence': result.get('confidence', 0),
                'reasoning': result.get('reasoning', ''),
                'checkbox_found': result.get('checkbox_found', False),
                'raw_response': result_text
            }

            logger.info(
                f"GPT DEF 14A Analysis Result: {result.get('document_kind', 'unknown')} (confidence: {result.get('confidence', 0)}%)")
            return analysis_result

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing GPT JSON response for DEF 14A: {e}")
            return {
                'document_kind': 'unknown',
                'confidence': 0,
                'reasoning': 'Failed to parse GPT response',
                'error': str(e)
            }
        except Exception as e:
            logger.error(f"Error analyzing DEF 14A document with GPT: {e}")
            return {
                'document_kind': 'unknown',
                'confidence': 0,
                'reasoning': 'GPT analysis failed',
                'error': str(e)
            }

    def analyze_def14a_filing(self, filing_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method to analyze a DEF 14A/PRE 14A filing for document kind detection
        Returns updated filing_data with document_kind field
        """
        try:
            # Check if this is a DEF 14A or PRE 14A
            if filing_data.get('form_type') not in ['DEF 14A', 'PRE 14A']:
                logger.info("Skipping analysis - not a DEF 14A/PRE 14A filing")
                return filing_data

            xbrl_files = filing_data.get('xbrl_files', [])

            # For DEF 14A, we need to find the main document file (usually the first HTML file)
            # Look for files that might contain the main document content
            main_document_files = [
                file for file in xbrl_files
                if file.get('url', '').endswith('.htm') or file.get('url', '').endswith('.html')
            ]

            if not main_document_files:
                logger.info(
                    "Skipping analysis - no HTML files found in DEF 14A/PRE 14A")
                filing_data['document_kind'] = 'unknown'
                return filing_data

            # Use the first HTML file as the main document
            main_file = main_document_files[0]
            htm_url = main_file.get('url')

            logger.info(f"Analyzing DEF 14A/PRE 14A file: {htm_url}")

            # Download document
            html_content = self.download_htm_file(htm_url)
            if not html_content:
                logger.error("Failed to download DEF 14A/PRE 14A HTM file")
                filing_data['document_kind'] = 'unknown'
                return filing_data

            # Extract pages (first few pages should contain the checkbox)
            document_text = self.extract_document_pages(
                html_content, max_pages=2)  # Only need first 2 pages for checkbox
            if not document_text:
                logger.error("Failed to extract DEF 14A/PRE 14A document text")
                filing_data['document_kind'] = 'unknown'
                return filing_data

            # Analyze with GPT
            analysis = self.analyze_def14a_document_with_gpt(
                document_text,
                filing_data.get('company_name', 'Unknown Company')
            )

            # Update filing data based on analysis
            filing_data['document_kind'] = analysis.get(
                'document_kind', 'unknown')
            filing_data['is_new_deal'] = None  # Not applicable for DEF 14A
            filing_data['following'] = False   # Not applicable for DEF 14A

            logger.info(
                f"📋 DEF 14A/PRE 14A Document kind: {filing_data.get('document_kind')} for {filing_data.get('company_name')}")

            return filing_data

        except Exception as e:
            logger.error(f"Error in analyze_def14a_filing: {e}")
            filing_data['document_kind'] = 'unknown'
            filing_data['is_new_deal'] = None
            filing_data['following'] = False
            return filing_data
