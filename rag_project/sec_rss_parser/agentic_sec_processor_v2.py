import pdb
from llama_index.core.settings import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from urllib.parse import urljoin
from playwright.async_api import async_playwright
from difflib import SequenceMatcher
import html2text
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import openai
import PyPDF2
from bs4 import BeautifulSoup
import requests
import re
import os
import json
import logging
import sys
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Set up logging


def setup_logging():
    """Setup logging configuration."""
    # Clear any existing handlers to avoid conflicts
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create file handler with explicit flushing
    file_handler = logging.FileHandler(
        'agentic_sec_processor.log', mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Configure root logger
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)

    # Force immediate flush
    file_handler.flush()

    return logging.getLogger(__name__)


# Initialize logging
logger = setup_logging()


# LlamaIndex imports


logger.info("Starting agentic SEC processor module")
logger.info("Testing logger functionality")

# Load environment variables and set OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY_SEC_FILING")

# Configure LlamaIndex settings
Settings.llm = OpenAI(model="gpt-4.1-mini", temperature=0.1)


class S3Service:
    """Service to handle S3 operations for agentic SEC processor"""

    def __init__(self):
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        self.bucket = os.environ.get("AWS_S3_BUCKET")
        logger.info(f"S3Service initialized with bucket: {self.bucket}")

    def upload_pdf(self, file_path: str, s3_key: str) -> str:
        """Upload PDF file to S3 and return the full URL"""
        try:
            logger.info(f"Uploading PDF to S3: {file_path} -> {s3_key}")
            self.s3.upload_file(file_path, self.bucket, s3_key)
            s3_url = f"https://{self.bucket}.s3.amazonaws.com/{s3_key}"
            logger.info(f"PDF uploaded successfully: {s3_url}")
            return s3_url
        except Exception as e:
            logger.error(f"Error uploading PDF to S3: {e}")
            raise

    def upload_json(self, data: dict, s3_key: str) -> str:
        """Upload JSON data to S3 and return the full URL"""
        try:
            logger.info(f"Uploading JSON to S3: {s3_key}")
            json_bytes = json.dumps(
                data, indent=2, ensure_ascii=False).encode("utf-8")
            self.s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=json_bytes,
                ContentType="application/json"
            )
            s3_url = f"https://{self.bucket}.s3.amazonaws.com/{s3_key}"
            logger.info(f"JSON uploaded successfully: {s3_url}")
            return s3_url
        except Exception as e:
            logger.error(f"Error uploading JSON to S3: {e}")
            raise

    def upload_file(self, file_path: str, s3_key: str, content_type: str = None) -> str:
        """Upload any file to S3 and return the full URL"""
        try:
            logger.info(f"Uploading file to S3: {file_path} -> {s3_key}")
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type

            self.s3.upload_file(file_path, self.bucket,
                                s3_key, ExtraArgs=extra_args)
            s3_url = f"https://{self.bucket}.s3.amazonaws.com/{s3_key}"
            logger.info(f"File uploaded successfully: {s3_url}")
            return s3_url
        except Exception as e:
            logger.error(f"Error uploading file to S3: {e}")
            raise


class AgenticSECProcessor:
    def __init__(self, sec_url: str, max_workers: int = 8):
        """
        Initialize the agentic SEC processor.

        Args:
            sec_url: URL of the SEC document
            max_workers: Maximum number of parallel workers for table cleaning (default: 8)
        """
        self.sec_url = sec_url
        self.max_workers = max_workers
        self.pdf_path = None
        self.toc_path = None
        self.sections_path = None
        self.document_name = self._extract_document_name()
        self.notice_data = None  # Store notice content for later integration

        # S3 URLs for storing files
        self.s3_urls = {
            'pdf_url': None,
            'toc_pdf_url': None,
            'toc_json_url': None,
            'sections_json_url': None
        }

        self.processing_state = {
            'pdf_created': False,
            'toc_found': False,
            'toc_extracted': False,
            'sections_extracted': False,
            'empty_percentage': 100.0,
            'iteration_count': 0
        }

        # Create output directories
        os.makedirs('pdf_documents', exist_ok=True)
        os.makedirs('new_table_of_content', exist_ok=True)
        os.makedirs('extracted_sections', exist_ok=True)

        # Initialize S3 service
        self.s3_service = S3Service()

        # Initialize agent with tools
        self.agent = self._create_agent()

    def _extract_document_name(self) -> str:
        """Extract document name from SEC URL."""
        url_parts = self.sec_url.split('/')
        filename = url_parts[-1].split('#')[0]
        if '.' in filename:
            return filename.split('.')[0]
        return "sec_document"

    def _create_agent(self) -> ReActAgent:
        """Create the ReAct agent with all necessary tools."""

        # Define all tools
        tools = [
            FunctionTool.from_defaults(
                fn=self.convert_html_to_pdf,
                name="convert_html_to_pdf",
                description="Convert SEC HTML document to PDF format"
            ),
            FunctionTool.from_defaults(
                fn=self.find_toc_page,
                name="find_toc_page",
                description="Find the page number that contains the Table of Contents"
            ),
            FunctionTool.from_defaults(
                fn=self.extract_toc_pages,
                name="extract_toc_pages",
                description="Extract TOC pages and create a new PDF with just those pages"
            ),
            FunctionTool.from_defaults(
                fn=self.upload_pdf_to_openai,
                name="upload_pdf_to_openai",
                description="Upload PDF to OpenAI and return the file ID"
            ),
            FunctionTool.from_defaults(
                fn=self.extract_toc_structure,
                name="extract_toc_structure",
                description="Extract TOC structure using the uploaded PDF file"
            ),
            FunctionTool.from_defaults(
                fn=self.flatten_subsections,
                name="flatten_subsections",
                description="Flatten nested subsections to max 2 levels deep"
            ),
            FunctionTool.from_defaults(
                fn=self.remove_sequential_duplicates,
                name="remove_sequential_duplicates",
                description="Remove sequential duplicate entries within the same level."
            ),
            FunctionTool.from_defaults(
                fn=self.save_toc,
                name="save_toc",
                description="Save TOC to JSON file"
            ),
            FunctionTool.from_defaults(
                fn=self.extract_sections_content,
                name="extract_sections_content",
                description="Extract content for each section using the TOC"
            ),
            FunctionTool.from_defaults(
                fn=self.check_empty_content,
                name="check_empty_content",
                description="Check empty content count in extracted sections"
            ),
            FunctionTool.from_defaults(
                fn=self.verify_toc_titles,
                name="verify_toc_titles",
                description="Verify TOC titles against document headings"
            ),
            FunctionTool.from_defaults(
                fn=self.fix_empty_sections,
                name="fix_empty_sections",
                description="Fix empty sections by validating titles one by one"
            ),
            FunctionTool.from_defaults(
                fn=self.split_content_from_previous,
                name="split_content_from_previous",
                description="Split content from previous sections for empty sections"
            ),
            FunctionTool.from_defaults(
                fn=self.get_processing_state,
                name="get_processing_state",
                description="Get current processing state and statistics"
            ),


            FunctionTool.from_defaults(
                fn=self.extract_notice_content,
                name="extract_notice_content",
                description="Extract all content from pages before the TOC page and create a 'Preamble' section. This captures introductory content, notices, and other preliminary information without using GPT analysis."
            ),
            FunctionTool.from_defaults(
                fn=self.clean_chunk_tables,
                name="clean_chunk_tables",
                description="Clean and format tables in text content using GPT. Detects tables and reformats them into clean Markdown format for better readability."
            ),
            # FunctionTool.from_defaults(
            #     fn=self.clean_all_tables_final,
            #     name="clean_all_tables_final",
            #     description="Clean and format all tables in the extracted sections using parallel processing. This runs only once at the end after all other processing is complete. Use this as the final step before completion."
            # )
        ]

        # Create memory buffer
        memory = ChatMemoryBuffer.from_defaults(token_limit=128_000)

        # Create agent with system prompt
        system_prompt = f"""You are an expert SEC document processing agent. Your goal is to process the SEC document at {self.sec_url} and extract its table of contents and section content with minimal empty sections.

Follow this workflow:
1. Convert HTML to PDF
2. Find TOC page
3. Extract preamble content from pages before TOC (use extract_notice_content tool)
4. Extract TOC pages
5. Upload TOC PDF to OpenAI
6. Extract TOC structure
7. Flatten subsections
8. Remove sequential duplicates
9. Save TOC
10. Extract sections content
11. Check empty content percentage
12. Clean all tables final

Based on empty content percentage, apply correction strategies:
- If empty % < 10%: Try splitting content from previous sections
- If empty % > 30%: Verify TOC titles and re-extract
- As a final fallback: Fix individual empty sections


IMPORTANT: After all processing is complete and you have achieved the desired empty content percentage, use the clean_all_tables_final tool as the very last step. This tool will clean and format all tables in the final sections using parallel processing, running only once at the end.

Always check the processing state before proceeding to understand what has been completed."""

        agent = ReActAgent.from_tools(
            tools=tools,
            llm=Settings.llm,
            memory=memory,
            system_prompt=system_prompt,
            verbose=True,
            max_iterations=50
        )

        return agent

    # Tool implementations (same logic as AutomatedSECProcessor but as individual functions)

    async def convert_html_to_pdf(self) -> str:
        """Convert SEC HTML document to PDF."""
        logger.info("Converting HTML to PDF...")

        try:
            headers = {
                "User-Agent": "MNA-Finder/1.0 (https://teqnodux.com; contact: ashish.kachadiya@teqnodux.com)",
                "Accept": "application/json",
            }

            # Add timeout to the request
            response = requests.get(self.sec_url, headers=headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            for script in soup(["script", "style"]):
                script.decompose()

            html_content = str(soup)

            # Create pdf_documents directory if it doesn't exist
            import os
            os.makedirs("pdf_documents", exist_ok=True)

            self.pdf_path = f"pdf_documents/{self.document_name}.pdf"

            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=['--no-sandbox',
                          '--disable-dev-shm-usage', '--disable-gpu']
                )
                page = await browser.new_page()

                # Set page timeout
                page.set_default_timeout(60000)  # 60 seconds

                # Use load instead of networkidle to avoid hanging
                await page.set_content(html_content, wait_until='load')

                # Reduced timeout
                await page.wait_for_timeout(1000)  # 1 second instead of 3

                # Generate PDF with timeout
                await page.pdf(
                    path=self.pdf_path,
                    format='A4',
                    margin={'top': '0.75in', 'right': '0.75in',
                            'bottom': '0.75in', 'left': '0.75in'},
                    print_background=True,
                    prefer_css_page_size=False
                )

                await browser.close()

            self.processing_state['pdf_created'] = True
            logger.info(f"PDF saved to: {self.pdf_path}")

            # Upload PDF to S3
            s3_key = f"proxy-pdf/{self.document_name}.pdf"
            self.s3_urls['pdf_url'] = self.s3_service.upload_pdf(
                self.pdf_path, s3_key)

            return f"PDF successfully created at {self.pdf_path} and uploaded to S3: {self.s3_urls['pdf_url']}"

        except requests.exceptions.Timeout:
            logger.error("Request timeout while fetching SEC document")
            return "Error: Request timeout while fetching SEC document"
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error while fetching SEC document: {e}")
            return f"Error: Request failed - {str(e)}"
        except Exception as e:
            logger.error(f"Error converting HTML to PDF: {e}")
            return f"Error: {str(e)}"

    def find_toc_page(self) -> str:
        """Find the page number that contains the Table of Contents."""
        logger.info("Finding TOC page...")

        if not self.processing_state['pdf_created']:
            return "Error: PDF must be created first"

        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num in range(min(15, len(pdf_reader.pages))):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()

                    response = openai.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {"role": "system", "content": """You are a helpful assistant that identifies if a given page is an actual Table of Contents.
Reply ONLY with true or false.

Reply true only if the text looks like a Table of Contents, meaning it lists multiple section or subsection titles along with their corresponding page numbers, typically in a structured list or outline format.
If the text only contains the phrase "Table of Contents" but does not contain a structured list of titles and page numbers, reply with false.
If in doubt, reply false.
Do not reply with anything except true or false."""},
                            {"role": "user", "content": f"Is this page a Table of Contents? Reply only true or false.\n\n{text}"}
                        ]
                    )

                    is_toc = response.choices[0].message.content.strip(
                    ).lower() == 'true'

                    if is_toc:
                        self.toc_page_num = page_num
                        self.processing_state['toc_found'] = True
                        logger.info(f"Found TOC on page {page_num + 1}")
                        return f"TOC found on page {page_num + 1}"

            return "Error: No TOC found in first 15 pages"

        except Exception as e:
            logger.error(f"Error finding TOC page: {e}")
            return f"Error: {str(e)}"

    def extract_toc_pages(self) -> str:
        """Extract TOC pages and create a new PDF with just those pages."""
        logger.info("Extracting TOC pages...")

        if not self.processing_state['toc_found']:
            return "Error: TOC page must be found first"

        try:
            pdf_writer = PyPDF2.PdfWriter()

            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for i in range(5):
                    page_num = self.toc_page_num + i
                    if page_num < len(pdf_reader.pages):
                        pdf_writer.add_page(pdf_reader.pages[page_num])

            self.toc_pdf_path = f"pdf_documents/{self.document_name}_toc_pages.pdf"
            with open(self.toc_pdf_path, 'wb') as output_file:
                pdf_writer.write(output_file)

            self.processing_state['toc_extracted'] = True
            logger.info(f"Saved TOC pages to {self.toc_pdf_path}")

            # Upload TOC PDF to S3
            s3_key = f"proxy-pdf-toc/{self.document_name}_toc_pages.pdf"
            self.s3_urls['toc_pdf_url'] = self.s3_service.upload_pdf(
                self.toc_pdf_path, s3_key)

            return f"TOC pages extracted to {self.toc_pdf_path} and uploaded to S3: {self.s3_urls['toc_pdf_url']}"

        except Exception as e:
            logger.error(f"Error extracting TOC pages: {e}")
            return f"Error: {str(e)}"

    def upload_pdf_to_openai(self, pdf_path: str = None) -> str:
        """Upload PDF to OpenAI and return the file ID."""
        if pdf_path is None:
            pdf_path = self.toc_pdf_path

        try:
            logger.info(f"Uploading PDF file: {pdf_path}")

            with open(pdf_path, "rb") as f:
                response = openai.files.create(
                    file=f,
                    purpose="assistants"
                )

            logger.info(f"Successfully uploaded file. File ID: {response.id}")
            return response.id

        except Exception as e:
            logger.error(f"Error uploading PDF: {str(e)}")
            return f"Error: {str(e)}"

    def extract_toc_structure(self, file_id: str) -> str:
        """Extract TOC structure using the uploaded PDF file."""
        logger.info("Extracting TOC structure...")

        try:
            system_prompt = """You are a helpful assistant that extracts table of contents from documents.
            Your task is to extract the table of contents and output ONLY the JSON array, nothing else.
            
            Rules:
            1. Each entry should have a title and page number
            2. Maintain the hierarchy (main sections and subsections), hierarchy can be only 2 levels deep
            3. Only include entries that have page numbers
            4. Sort entries by page number
            5. Remove any duplicate entries
            6. For Section & Subsections rely on indentation
            7. Dont forgot any characters in the title. For example, "1. Terms of the Offer" dont remove 1 from title.
            8. Dont add "Table of Contents" in the title. If it is there, remove it.

            Output format must be exactly this JSON structure with no additional text:
            [
                {
                    "title": "MAIN SECTION",
                    "page-no": "X",
                    "subsection": [
                        {
                            "title": "Subsection",
                            "page-no": "Y"
                        }
                    ]
                }
            ]"""

            user_message = "Extract the table of contents from the uploaded PDF and output ONLY the JSON array, no other text."

            response = openai.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [{
                        "type": "file",
                        "file": {
                            "file_id": file_id
                        }
                    }, {
                        "type": "text",
                        "text": user_message}]}
                ]
            )

            toc_text = response.choices[0].message.content.strip()

            try:
                self.toc_data = json.loads(toc_text)
                return f"TOC structure extracted successfully with {len(self.toc_data)} main sections"
            except json.JSONDecodeError:
                start = toc_text.find('[')
                end = toc_text.rfind(']') + 1
                if start != -1 and end != 0:
                    toc_json = toc_text[start:end]
                    try:
                        self.toc_data = json.loads(toc_json)
                        return f"TOC structure extracted successfully with {len(self.toc_data)} main sections"
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON array: {e}")
                        return f"Error parsing JSON: {str(e)}"
                else:
                    logger.error(
                        "Could not find JSON array in OpenAI response")
                    return "Error: Could not find JSON array in response"

        except Exception as e:
            logger.error(f"Error in OpenAI API call: {str(e)}")
            return f"Error: {str(e)}"

    def flatten_subsections(self) -> str:
        """Flatten nested subsections to max 2 levels deep."""
        logger.info("Flattening nested subsections...")

        if not hasattr(self, 'toc_data'):
            return "Error: TOC data not available"

        def process(node):
            result = node.copy()
            if "subsection" in result:
                new_subsections = []
                for sub in result["subsection"]:
                    if "subsection" in sub:
                        processed_sub = process(sub.copy())
                        new_subsections.append({
                            "title": processed_sub["title"],
                            "page-no": processed_sub.get("page-no", "")
                        })
                        for nested_sub in processed_sub["subsection"]:
                            processed_nested = process(nested_sub)
                            new_subsections.append(processed_nested)
                    else:
                        new_subsections.append(process(sub))
                result["subsection"] = new_subsections
            return result

        self.toc_data = [process(item) for item in self.toc_data]
        return "Subsections flattened successfully"

    def remove_sequential_duplicates(self) -> str:
        """Remove sequential duplicate entries within the same level."""
        logger.info("Removing sequential duplicates...")

        if not hasattr(self, 'toc_data'):
            return "Error: TOC data not available"

        def remove_duplicates_in_list(items):
            if not items:
                return []
            result = [items[0]]
            for i in range(1, len(items)):
                current = items[i]
                previous = items[i-1]
                if not (current["title"] == previous["title"] and
                        current.get("page-no") == previous.get("page-no")):
                    result.append(current)
            return result

        for section in self.toc_data:
            if "subsection" in section and section["subsection"]:
                section["subsection"] = remove_duplicates_in_list(
                    section["subsection"])

        self.toc_data = remove_duplicates_in_list(self.toc_data)
        return "Sequential duplicates removed successfully"

    def save_toc(self) -> str:
        """Save TOC to JSON file."""
        logger.info("Saving TOC to JSON file...")

        if not hasattr(self, 'toc_data'):
            return "Error: TOC data not available"

        try:
            self.toc_path = f'new_table_of_content/table_of_contents_new_{self.document_name}.json'
            with open(self.toc_path, 'w', encoding='utf-8') as f:
                json.dump(self.toc_data, f, indent=2)
            logger.info(f"TOC saved to: {self.toc_path}")

            # Upload TOC JSON to S3
            s3_key = f"proxy-parse-json/table_of_contents_new_{self.document_name}.json"
            self.s3_urls['toc_json_url'] = self.s3_service.upload_json(
                self.toc_data, s3_key)

            return f"TOC saved successfully to {self.toc_path} and uploaded to S3: {self.s3_urls['toc_json_url']}"
        except Exception as e:
            return f"Error saving TOC: {str(e)}"

    def extract_sections_content(self) -> str:
        """Extract content for each section using the TOC."""
        logger.info("Extracting sections content...")

        if not self.toc_path:
            return "Error: TOC file not available"

        try:
            try:
                from rag_project.proxy_processor.extract_sections_html_class import SECDocumentProcessor
            except ImportError:
                try:
                    from proxy_processor.extract_sections_html_class import SECDocumentProcessor
                except ImportError:
                    # Fallback for direct-script execution where package imports are not resolvable
                    import importlib.util
                    from pathlib import Path
                    module_path = Path(__file__).resolve().parents[1] / \
                        "proxy_processor" / "extract_sections_html_class.py"
                    spec = importlib.util.spec_from_file_location(
                        "extract_sections_html_class", str(module_path))
                    if spec is None or spec.loader is None:
                        raise ImportError(
                            f"Could not load module spec from {module_path}")
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    SECDocumentProcessor = module.SECDocumentProcessor
            processor = SECDocumentProcessor(self.sec_url, self.toc_path)
            sections = processor.process_document()

            # Add notice content at the beginning if it exists
            if hasattr(self, 'notice_data') and self.notice_data:
                logger.info(
                    f"Adding {len(self.notice_data)} notice sections to the beginning of sections")
                # Reverse to maintain order when inserting at beginning
                for notice in reversed(self.notice_data):
                    notice_section = {
                        "title": notice["heading"],
                        "page-no": str(notice["page_numbers"][0]) if notice["page_numbers"] else "1",
                        "content": notice["content"]
                    }
                    sections.insert(0, notice_section)

            self.sections_path = f"extracted_sections/sections_with_content_html_{self.document_name}.json"
            with open(self.sections_path, 'w', encoding='utf-8') as f:
                json.dump(sections, f, indent=2, ensure_ascii=False)

            self.processing_state['sections_extracted'] = True

            # Upload sections JSON to S3
            s3_key = f"proxy-parse-json/sections_with_content_html_{self.document_name}.json"
            self.s3_urls['sections_json_url'] = self.s3_service.upload_json(
                sections, s3_key)

            notice_info = f" (including {len(self.notice_data)} notice sections)" if hasattr(
                self, 'notice_data') and self.notice_data else ""
            logger.info(
                f"Sections with content saved to: {self.sections_path}{notice_info}")
            return f"Sections content extracted successfully to {self.sections_path}{notice_info} and uploaded to S3: {self.s3_urls['sections_json_url']}"

        except Exception as e:
            logger.error(f"Error extracting sections content: {e}")
            return f"Error: {str(e)}"

    def _clean_single_chunk(self, chunk_data: Tuple[str, str, str]) -> Tuple[str, str, str, str]:
        """
        Worker function to clean a single chunk of text.

        Args:
            chunk_data: Tuple of (unique_id, title, content)

        Returns:
            Tuple of (unique_id, title, cleaned_content, original_content)
        """
        unique_id, title, content = chunk_data

        try:
            logger.info(f"Worker processing {unique_id}: {title}")
            cleaned_content = self.clean_chunk_tables(content)
            logger.info(f"Worker completed {unique_id}: {title}")
            return (unique_id, title, cleaned_content, content)
        except Exception as e:
            logger.error(
                f"Error in worker processing {unique_id} '{title}': {str(e)}")
            # Return original content on error
            return (unique_id, title, content, content)

    def clean_chunk_tables(self, chunk_text: str) -> str:
        """Clean and format tables in chunk text using GPT"""
        try:
            logger.info("Cleaning tables in chunk text")

            prompt = f"""You are given a chunk of text from a financial or legal filing.  
Your task:  

1. Detect if the chunk contains one or more tables (ASCII-style, space-padded, or misaligned tabular data).  
2. For each table found, reformat it into a **clean, valid Markdown table** with these rules:  
   - Remove empty filler columns (like `|       |`).  
   - Normalize headers so there is a single header row (e.g., years, metrics, categories).  
   - Align all rows under the same set of headers.  
   - Preserve every number, label, and unit exactly as written (don't change values).  
   - Keep column order and row order the same as in the original table.  
   - If the source table has sub-headers or multi-level headings, flatten them into a single Markdown header row.  
   - If multiple tables exist, reformat all of them separately.  
3. Keep all surrounding narrative text exactly as-is, without rewriting or summarizing.  
4. Always return valid JSON with this structure: 

{{
  "is_table_found": "true" or "false",
  "content": "the full cleaned text with updated Markdown table formatting (only if is_table_found is true, otherwise omit this field)"
}}

⚠️ Rules:  
- Do not omit any text outside the tables.  
- Do not summarize or paraphrase the text; preserve it verbatim.  
- If multiple tables exist, reformat all of them.  
- If no tables exist, return `"is_table_found": "false"` and omit the content field.  

Chunk text to process:
{chunk_text}"""

            response = openai.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a specialized assistant that detects and formats tables in financial documents. Always return valid JSON with the exact structure requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=30000,
                response_format={"type": "json_object"},
                stream=True
            )

            logger.info(f"clean_chunk_tables prompt: {prompt}")

            # Collect streaming response
            result = ""
            chunk_count = 0
            for chunk in response:
                if chunk.choices[0].delta.content is not None:

                    result += chunk.choices[0].delta.content
                    chunk_count += 1
                    # Log progress every 10 chunks for large responses
                    if chunk_count % 10 == 0:
                        logger.debug(
                            f"Received {chunk_count} chunks, current length: {len(result)}")

            logger.info(
                f"Table cleaning completed - received {chunk_count} chunks, total length: {len(result)}")
            logger.info(f"cleaned result: {result}")

            # Parse JSON response
            try:
                parsed_result = json.loads(result)
                is_table_found = parsed_result.get("is_table_found", "false")

                if is_table_found == "true":
                    logger.info("Table found and formatted")
                    return parsed_result.get("content", chunk_text)
                else:
                    logger.info("No table found, using original text")
                    return chunk_text  # Use original text directly

            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse JSON response from table cleaner, returning original text")
                return chunk_text

        except Exception as e:
            logger.error(f"Error cleaning tables: {str(e)}")
            return chunk_text

    def clean_all_tables_final(self) -> str:
        """Clean and format all tables in the extracted sections using parallel processing. This runs only once at the end."""
        logger.info("Starting final table cleaning for all sections...")

        if not self.sections_path or not os.path.exists(self.sections_path):
            return "Error: Sections file not available"

        try:
            import time
            start_time = time.time()

            # Load the sections file
            with open(self.sections_path, 'r', encoding='utf-8') as f:
                sections = json.load(f)

            logger.info(f"Loaded {len(sections)} sections for table cleaning")

            # Collect all chunks that need processing with unique identifiers
            chunks_to_process = []
            # Maps chunk_idx to (section_idx, type, sub_idx, unique_id)
            chunk_mapping = {}

            chunk_idx = 0
            for section_idx, section in enumerate(sections):
                if section.get('content', '').strip():
                    # Create unique identifier combining section index and title
                    unique_id = f"section_{section_idx}_{section.get('title', 'Unknown').replace(' ', '_')}"
                    chunks_to_process.append(
                        (unique_id, section.get('title', 'Unknown'), section['content']))
                    chunk_mapping[chunk_idx] = (
                        section_idx, 'main', None, unique_id)
                    chunk_idx += 1

                # Also collect subsections
                if 'subsection' in section:
                    for sub_idx, subsection in enumerate(section['subsection']):
                        if subsection.get('content', '').strip():
                            # Create unique identifier combining section index, sub index and title
                            unique_id = f"subsection_{section_idx}_{sub_idx}_{subsection.get('title', 'Unknown').replace(' ', '_')}"
                            chunks_to_process.append((unique_id, subsection.get(
                                'title', 'Unknown'), subsection['content']))
                            chunk_mapping[chunk_idx] = (
                                section_idx, 'subsection', sub_idx, unique_id)
                            chunk_idx += 1

            if not chunks_to_process:
                logger.info("No content found to clean")
                return "No content found to clean"

            logger.info(
                f"Processing {len(chunks_to_process)} chunks with parallel workers...")

            # Process chunks in parallel
            max_workers = min(self.max_workers, len(chunks_to_process))
            processed_chunks = {}

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_chunk = {
                    executor.submit(self._clean_single_chunk, chunk_data): i
                    for i, chunk_data in enumerate(chunks_to_process)
                }

                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    try:
                        unique_id, title, cleaned_content, original_content = future.result()
                        processed_chunks[chunk_idx] = (
                            unique_id, title, cleaned_content, original_content)
                        logger.info(
                            f"Completed processing chunk {chunk_idx + 1}/{len(chunks_to_process)}: {unique_id}")
                    except Exception as e:
                        logger.error(
                            f"Error processing chunk {chunk_idx}: {str(e)}")
                        # Use original content on error
                        original_chunk = chunks_to_process[chunk_idx]
                        processed_chunks[chunk_idx] = (
                            original_chunk[0], original_chunk[1], original_chunk[2], original_chunk[2])

            # Apply cleaned content back to sections using unique ID mapping
            for chunk_idx, (unique_id, title, cleaned_content, original_content) in processed_chunks.items():
                if chunk_idx in chunk_mapping:
                    section_idx, section_type, sub_idx, mapped_unique_id = chunk_mapping[
                        chunk_idx]

                    # Verify the unique IDs match (safety check)
                    if unique_id == mapped_unique_id:
                        if section_type == 'main':
                            sections[section_idx]['content'] = cleaned_content
                            logger.info(
                                f"Applied cleaned content to section {section_idx}: {title}")
                        elif section_type == 'subsection':
                            sections[section_idx]['subsection'][sub_idx]['content'] = cleaned_content
                            logger.info(
                                f"Applied cleaned content to subsection {section_idx}.{sub_idx}: {title}")
                    else:
                        logger.error(
                            f"Unique ID mismatch for chunk {chunk_idx}: {unique_id} != {mapped_unique_id}")
                        # Fallback: use original content
                        if section_type == 'main':
                            sections[section_idx]['content'] = original_content
                        elif section_type == 'subsection':
                            sections[section_idx]['subsection'][sub_idx]['content'] = original_content

            # Save the cleaned sections back to file
            with open(self.sections_path, 'w', encoding='utf-8') as f:
                json.dump(sections, f, indent=2, ensure_ascii=False)

            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(
                f"Final table cleaning completed successfully in {processing_time:.2f} seconds using {max_workers} workers")

            return f"Final table cleaning completed successfully in {processing_time:.2f} seconds. Processed {len(chunks_to_process)} chunks using {max_workers} workers."

        except Exception as e:
            logger.error(f"Error in final table cleaning: {str(e)}")
            return f"Error in final table cleaning: {str(e)}"

    def check_empty_content(self) -> str:
        """Check empty content count in extracted sections."""
        logger.info("Checking empty content count...")

        if not self.sections_path:
            return "Error: Sections file not available"

        try:
            with open(self.sections_path, 'r', encoding='utf-8') as f:
                sections = json.load(f)

            total_count = 0
            empty_count = 0

            def count_sections(items):
                nonlocal total_count, empty_count
                for item in items:
                    total_count += 1
                    if not item.get('content', '').strip():
                        empty_count += 1
                    if 'subsection' in item:
                        count_sections(item['subsection'])

            count_sections(sections)

            empty_percentage = (empty_count / total_count) * \
                100 if total_count > 0 else 0
            self.processing_state['empty_percentage'] = empty_percentage

            logger.info(
                f"Empty content statistics: {empty_count}/{total_count} ({empty_percentage:.1f}%)")
            return f"Empty content: {empty_count}/{total_count} ({empty_percentage:.1f}%)"

        except Exception as e:
            return f"Error checking empty content: {str(e)}"

    def verify_toc_titles(self, full_pdf_file_id: str, model: str = "gpt-4.1-mini") -> str:
        """Verify TOC titles against document headings."""
        logger.info("Verifying TOC titles against document headings...")

        if not self.toc_path:
            return "Error: TOC file not available"

        try:
            with open(self.toc_path, 'r', encoding='utf-8') as f:
                toc = json.load(f)

            toc_json_str = json.dumps(toc)
            system_prompt = f"""You are a document title matching assistant.

Given a Table of Contents (TOC) title and the actual text of a document, check if the TOC title appears **as an exact heading** in the body of the document (not just in the TOC list itself). Only consider it a match if the TOC title exactly matches a document heading, allowing for differences in letter case and leading/trailing whitespace, but not if the TOC title is only a part, prefix, or substring of a longer heading. If there is no exact match, return the actual heading from the document that most closely corresponds to the TOC title.

Return your result as follows:

[{{
  "toc_title": "<TOC title provided>",
  "found_in_document": true or false,
  "correct_document_title": "<actual title as it appears in the document body if not exact match, otherwise leave blank>"
}}]

Only output the JSON.

below is my toc extracted from attached document
{toc_json_str}
"""

            user_message = "Check if the TOC titles match the document headings and return the results as JSON."

            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [{
                        "type": "file",
                        "file": {
                            "file_id": full_pdf_file_id
                        }
                    }, {
                        "type": "text",
                        "text": user_message}]}
                ]
            )

            response_text = response.choices[0].message.content.strip()
            try:
                title_matches = json.loads(response_text)
            except json.JSONDecodeError:
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                if start != -1 and end != 0:
                    json_text = response_text[start:end]
                    try:
                        title_matches = json.loads(json_text)
                    except json.JSONDecodeError:
                        return "Error parsing title verification response"

            title_corrections = {}
            for match in title_matches:
                if not match.get("found_in_document", True) and match.get("correct_document_title"):
                    title_corrections[match["toc_title"]
                                      ] = match["correct_document_title"]

            def update_titles(items):
                for item in items:
                    if item["title"] in title_corrections:
                        logger.info(
                            f"Updating title: '{item['title']}' -> '{title_corrections[item['title']]}'")
                        item["title"] = title_corrections[item["title"]]
                    if "subsection" in item:
                        update_titles(item["subsection"])
                return items

            updated_toc = update_titles(toc)

            with open(self.toc_path, 'w', encoding='utf-8') as f:
                json.dump(updated_toc, f, indent=2)

            logger.info("Title verification and correction complete")
            return f"Title verification complete. {len(title_corrections)} titles corrected."

        except Exception as e:
            logger.error(f"Error in title verification: {str(e)}")
            return f"Error: {str(e)}"

    def fix_empty_sections(self, full_pdf_file_id: str, max_iterations: int = 5) -> str:
        """Fix empty sections by validating titles one by one."""
        logger.info("Fixing empty sections by validating titles...")

        if not self.toc_path or not self.sections_path:
            return "Error: TOC and sections files not available"

        try:
            with open(self.toc_path, 'r', encoding='utf-8') as f:
                toc = json.load(f)

            processed_titles = set()
            iteration_count = 0

            while iteration_count < max_iterations:
                iteration_count += 1

                with open(self.sections_path, 'r', encoding='utf-8') as f:
                    sections = json.load(f)

                empty_sections = []

                def find_empty_sections(items):
                    for item in items:
                        if not item.get('content', '').strip() and item["title"] not in processed_titles:
                            empty_sections.append({
                                "title": item["title"],
                                "page_num": item.get('page-no', '')
                            })
                        if 'subsection' in item:
                            find_empty_sections(item['subsection'])

                find_empty_sections(sections)

                if not empty_sections:
                    break

                section = empty_sections[0]
                title = section["title"]
                page_num = section["page_num"]
                processed_titles.add(title)

                if not page_num:
                    continue

                system_prompt = """You are a document title validation assistant.

Given a proposed title and a page number in a document, check if the exact title appears on that page. The comparison should ignore differences in letter case and leading/trailing whitespace, but must match exactly otherwise (no partial or substring matches).

If the title on the page exactly matches the given title, return the given title.
If the title on the page differs, return the actual title found on that page.
example:
proposed title: "Board Recommendation"
actual title: "Recommendations of the Transaction Committee and the Board"
return: "Recommendations of the Transaction Committee and the Board"

Respond with only the final title string."""

                user_message = f"Given title: \"{title}\"\n\nPage number: {page_num}"

                response = openai.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [{
                            "type": "file",
                            "file": {
                                "file_id": full_pdf_file_id
                            }
                        }, {
                            "type": "text",
                            "text": user_message}]}
                    ]
                )

                corrected_title = response.choices[0].message.content.strip()

                if corrected_title != title:
                    def update_title_in_toc(items, old_title, new_title):
                        for item in items:
                            if item["title"] == old_title:
                                item["title"] = new_title
                                return True
                            if "subsection" in item:
                                if update_title_in_toc(item["subsection"], old_title, new_title):
                                    return True
                        return False

                    if update_title_in_toc(toc, title, corrected_title):
                        with open(self.toc_path, 'w', encoding='utf-8') as f:
                            json.dump(toc, f, indent=2)

                        # Re-extract sections
                        self.extract_sections_content()

            self.processing_state['iteration_count'] = iteration_count
            return f"Fixed empty sections in {iteration_count} iterations"

        except Exception as e:
            return f"Error fixing empty sections: {str(e)}"

    def split_content_from_previous(self) -> str:
        """Split content from previous sections for empty sections."""
        logger.info("Splitting content from previous sections...")

        if not self.sections_path:
            return "Error: Sections file not available"

        try:
            with open(self.sections_path, 'r', encoding='utf-8') as f:
                sections = json.load(f)

            empty_sections = []

            def find_empty_and_prev_sections(items, prev_item=None):
                for idx, item in enumerate(items):
                    if not item.get('content', '').strip() and prev_item is not None:
                        empty_sections.append({
                            "empty_section": item,
                            "prev_section": prev_item
                        })

                    if 'subsection' in item and item['subsection']:
                        prev_sub = None
                        for sub_item in item['subsection']:
                            if prev_sub is not None:
                                find_empty_and_prev_sections(
                                    [sub_item], prev_sub)
                            prev_sub = sub_item

                    prev_item = item

            if len(sections) > 1:
                prev_section = None
                for section in sections:
                    if prev_section is not None:
                        find_empty_and_prev_sections([section], prev_section)
                    prev_section = section

            changes_made = 0
            for entry in empty_sections:
                empty_section = entry["empty_section"]
                prev_section = entry["prev_section"]
                empty_title = empty_section["title"]

                if not prev_section.get('content', '').strip():
                    continue

                prev_content = prev_section["content"]
                patterns = [
                    f"\n\n{re.escape(empty_title)}\n",
                    f"\n{re.escape(empty_title)}\n",
                    f"{re.escape(empty_title)}\n"
                ]

                for pattern in patterns:
                    match = re.search(pattern, prev_content)
                    if match:
                        split_pos = match.start()
                        prev_content_part = prev_content[:split_pos].rstrip()
                        empty_content_part = prev_content[split_pos:].lstrip()

                        prev_section["content"] = prev_content_part
                        empty_section["content"] = empty_content_part

                        logger.info(
                            f"Split content for section '{empty_title}' from previous section")
                        changes_made += 1
                        break

            if changes_made > 0:
                with open(self.sections_path, 'w', encoding='utf-8') as f:
                    json.dump(sections, f, indent=2, ensure_ascii=False)

                return f"Split content for {changes_made} sections"
            else:
                return "No content could be split from previous sections"

        except Exception as e:
            return f"Error splitting content: {str(e)}"

    def update_section_title_by_path(self, sections, old_title, new_title, path=None, prev_title=None, next_title=None):
        """
        Update a section title based on its hierarchical path and surrounding context.

        Args:
            sections: The sections data structure
            old_title: The original title to replace
            new_title: The new title to use
            path: Optional list of parent section titles leading to the target section
            prev_title: Optional title of the previous section (for context)
            next_title: Optional title of the next section (for context)

        Returns:
            Tuple of (updated_successfully, updated_sections)
        """
        logger.info(f"Updating section title '{old_title}' to '{new_title}'")
        if path:
            logger.info(f"Path context: {' > '.join(path)}")
        if prev_title:
            logger.info(f"Previous section: {prev_title}")
        if next_title:
            logger.info(f"Next section: {next_title}")

        # Track if we found and updated the section
        updated = False

        def find_and_update_section(items, current_path=None, prev=None):
            """
            Recursively find and update the correct section based on path and context.

            Args:
                items: List of section items to search
                current_path: Current path in the hierarchy
                prev: Previous section title

            Returns:
                Tuple of (updated_successfully, next_title)
            """
            nonlocal updated

            if current_path is None:
                current_path = []

            # Process each item in the current level
            for i, item in enumerate(items):
                # Skip if we've already found and updated the target section
                if updated:
                    continue

                current_title = item.get("title", "")

                # Determine next title if available
                next_item_title = items[i +
                                        1].get("title", "") if i+1 < len(items) else None

                # Check if this is our target section
                if current_title.lower() == old_title.lower():
                    # Path matching (if path is provided)
                    path_matches = True
                    if path:
                        # Check if the current path matches the expected path
                        if len(current_path) != len(path):
                            path_matches = False
                        else:
                            for j, path_item in enumerate(path):
                                if j >= len(current_path) or path_item.lower() != current_path[j].lower():
                                    path_matches = False
                                    break

                    # Context matching (if context is provided)
                    context_matches = True
                    if prev_title and prev:
                        if prev.lower() != prev_title.lower():
                            context_matches = False

                    if next_title and next_item_title:
                        if next_item_title.lower() != next_title.lower():
                            context_matches = False

                    # If both path and context match (or weren't provided), update the title
                    if path_matches and context_matches:
                        logger.info(
                            f"Found matching section at path: {' > '.join(current_path + [current_title])}")
                        item["title"] = new_title
                        updated = True
                        return True, next_item_title

                # Recursively check subsections
                if "subsection" in item and item["subsection"]:
                    # Add current title to path for subsections
                    new_path = current_path + [current_title]
                    sub_updated, _ = find_and_update_section(
                        item["subsection"], new_path, current_title)
                    if sub_updated:
                        return True, next_item_title

                # Track previous title for next iteration
                prev = current_title

            return False, None

        # Start the recursive search
        find_and_update_section(sections)

        return updated, sections

    def extract_notice_content(self) -> str:
        """
        Extract all content from pages before the TOC page and create a 'Preamble' section.
        This captures introductory content, notices, and other preliminary information.

        Returns:
            Status message about the extraction process
        """
        logger.info("Extracting preamble content from pages before TOC...")

        if not self.processing_state['pdf_created']:
            return "Error: PDF must be created first"

        if not self.processing_state['toc_found']:
            return "Error: TOC page must be found first"

        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                # Extract text from all pages before TOC
                pages_text = []
                for page_num in range(min(self.toc_page_num, len(pdf_reader.pages))):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    pages_text.append({
                        'page_num': page_num + 1,
                        'text': text
                    })

                if not pages_text:
                    return "No pages found before TOC"

                # Combine all text from pages before TOC
                combined_text = "\n\n".join(
                    [page['text'] for page in pages_text])

                # Create preamble section with all content from pages before TOC
                page_numbers = [page['page_num'] for page in pages_text]
                notice_data = [{
                    "heading": "Preamble",
                    "content": combined_text.strip(),
                    "page_numbers": page_numbers
                }]

                logger.info(
                    f"Created preamble section with content from pages {page_numbers}")

                # Store notice data to be added to main sections file later
                self.notice_data = notice_data

                # If sections file already exists, add notice content immediately
                if self.sections_path and os.path.exists(self.sections_path):
                    try:
                        with open(self.sections_path, 'r', encoding='utf-8') as f:
                            sections = json.load(f)

                        # Add notice sections at the beginning
                        for notice in notice_data:
                            notice_section = {
                                "title": notice["heading"],
                                "page-no": str(notice["page_numbers"][0]) if notice["page_numbers"] else "1",
                                "content": notice["content"]
                            }
                            # Insert at the beginning of sections
                            sections.insert(0, notice_section)

                        # Save updated sections
                        with open(self.sections_path, 'w', encoding='utf-8') as f:
                            json.dump(sections, f, indent=2,
                                      ensure_ascii=False)

                        logger.info(
                            f"Added preamble section to main sections file")
                        return f"Successfully created preamble section and added to main sections file"
                    except Exception as e:
                        logger.warning(
                            f"Could not add preamble content to main sections file: {str(e)}")
                        return f"Created preamble section but could not add to main file: {str(e)}"
                else:
                    logger.info(
                        f"Preamble section stored for later integration")
                    return f"Successfully created preamble section. Will be added to main sections file when it's created."

        except Exception as e:
            logger.error(f"Error extracting notice content: {str(e)}")
            return f"Error: {str(e)}"

    def get_processing_state(self) -> str:
        """Get current processing state and statistics."""
        state_info = []
        for key, value in self.processing_state.items():
            state_info.append(f"{key}: {value}")

        return f"Processing state: {', '.join(state_info)}"

    def process_document(self) -> Dict[str, Any]:
        """
        Main method to process the document using the agent.

        Returns:
            Dictionary with processing results
        """
        logger.info("Starting agentic SEC document processing...")

        try:
            # Define the goal for the agent
            goal = f"""Process the SEC document at {self.sec_url} following this workflow:

1. Convert the HTML document to PDF
2. Find the Table of Contents page
3. Extract preamble content from pages before TOC (use extract_notice_content tool)
4. Extract TOC pages to a separate PDF
5. Upload the TOC PDF to OpenAI
6. Extract the TOC structure as JSON
7. Flatten any nested subsections
8. Remove sequential duplicates
9. Save the TOC to a JSON file
10. Extract content for each section
11. Check the empty content percentage
12. Clean all tables final

Based on the empty content percentage, apply correction strategies:
- If empty % < 10%: Try splitting content from previous sections
- If empty % > 30%: Verify TOC titles against document headings and re-extract
- As a final fallback: Fix individual empty sections

IMPORTANT: After all processing is complete and you have achieved the desired empty content percentage, use the clean_all_tables_final tool as the very last step. This tool will clean and format all tables in the final sections using parallel processing, running only once at the end.


The goal is to achieve 0% empty sections. Use the available tools to complete each step and make corrections as needed.


Start by checking the current processing state, then proceed with the workflow."""

            # Run the agent
            response = self.agent.chat(goal)

            logger.info("Agentic processing completed")

            return {
                's3_urls': self.s3_urls,
                'empty_percentage': self.processing_state['empty_percentage'],
                'agent_response': str(response)
            }

        except Exception as e:
            logger.error(f"Error in agentic processing: {e}")
            raise


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("STARTING AGENTIC SEC PROCESSOR")
    logger.info("=" * 60)

    # Check if OPENAI_API_KEY_SEC_FILING is set
    if not os.getenv('OPENAI_API_KEY_SEC_FILING'):
        logger.error(
            "OPENAI_API_KEY_SEC_FILING environment variable is not set")
        print("Please set your OPENAI_API_KEY_SEC_FILING environment variable")
        # Flush logs before exit
        for handler in logging.root.handlers:
            handler.flush()
        sys.exit(1)

    # Example usage
    sec_url = "https://www.sec.gov/Archives/edgar/data/2016561/000149315226013001/forms-4.htm"
    logger.info(f"Processing SEC document: {sec_url}")

    try:
        logger.info("Initializing AgenticSECProcessor...")
        # You can adjust max_workers based on your system capabilities
        # More workers = faster processing, but more API calls and memory usage
        processor = AgenticSECProcessor(sec_url, max_workers=8)

        logger.info("Starting document processing...")
        results = processor.process_document()

        logger.info("=" * 60)
        logger.info("AGENTIC SEC PROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(
            f"Empty Content Percentage: {results['empty_percentage']:.1f}%")
        logger.info("S3 URLs:")
        for key, url in results['s3_urls'].items():
            if url:
                logger.info(f"  {key}: {url}")
        logger.info("=" * 60)

        print("\n" + "="*50)
        print("AGENTIC SEC PROCESSING COMPLETED")
        print("="*50)
        print(f"Empty Content Percentage: {results['empty_percentage']:.1f}%")
        print("S3 URLs:")
        for key, url in results['s3_urls'].items():
            if url:
                print(f"  {key}: {url}")
        print(f"Agent Response: {results['agent_response']}")
        print("="*50)

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"Error: {e}")
    finally:
        # Ensure all log messages are flushed to file
        logger.info("Flushing logs and exiting...")
        for handler in logging.root.handlers:
            handler.flush()
        logger.info("Process completed.")
