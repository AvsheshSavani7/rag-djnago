"""
Service to generate proxy summary document after processing completes.
Uses proxy_id to filter Pinecone chunks.
"""

from datetime import datetime
from typing import Dict, Any, List, Tuple
import tempfile
import logging
import json
import sys
import os
import anthropic
import pinecone
import openai
from dotenv import load_dotenv
from docx import Document
from proxy_processor.merger_background_8_cleaned_format import ProxyBackgroundAnalyzer, DOCXFormatter
from sec_rss_parser.agentic_sec_processor_v2 import S3Service
from proxy_processor.arb_summary_doc_new_02_Dec_25 import QueryProcessor
import re
# Helper functions (copied from arb_summary_doc_new_02_Dec_25.py to avoid circular imports)


def _is_table(text: str) -> bool:
    """Check if text appears to be a table."""
    lines = text.split('\n')
    pipe_lines = [line for line in lines if '|' in line]
    return len(pipe_lines) >= 3 and len(pipe_lines) / max(len(lines), 1) > 0.5


def _parse_extraction_for_document_a(extraction_text: str) -> List[Tuple[str, str]]:
    """Extract only sections needed for Document A (client deliverables)."""
    required_sections = {
        '2. Starting Point': [],
        '3. Process Structure': [],
        '4. Bidder Universe': [],
        '5. Complete Bid Timeline': [],
        '6. Sales Process Metrics': [],
        '7. Final Round Analysis': [],
        '8. Board Selection Rationale': [],
        '9. Risk Analysis - Regulatory': [],
        '10. Risk Analysis - Financing': [],
        '13. Key Dates Summary': []
    }

    # Map section number → canonical key so LLM variants like
    # "4. Bidder Universe (Complete Census)" or
    # "9. Risk Analysis - Regulatory/Antitrust" still match.
    section_num_map = {}
    for key in required_sections:
        m = re.match(r'^(\d+)\.', key)
        if m:
            section_num_map[m.group(1)] = key

    lines = extraction_text.split('\n')
    current_section = None
    current_content = []

    for line in lines:
        if line.startswith('## '):
            if current_section and current_section in required_sections:
                required_sections[current_section] = '\n'.join(current_content)

            section_name = line.replace('##', '').strip()
            if section_name in required_sections:
                current_section = section_name
                current_content = []
            else:
                # Fall back to number-prefix matching to handle LLM header variations
                m = re.match(r'^(\d+)\.', section_name)
                if m and m.group(1) in section_num_map:
                    current_section = section_num_map[m.group(1)]
                    current_content = []
                else:
                    current_section = None
                current_section = None
        elif current_section:
            current_content.append(line)

    if current_section and current_section in required_sections:
        required_sections[current_section] = '\n'.join(current_content)

    return [(k, v) for k, v in required_sections.items() if v]


# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Import merger background analyzer

load_dotenv()

logger = logging.getLogger(__name__)


class ProxySummaryServiceV2:
    """Service to generate proxy summary documents using sec_filing_summary_id (V2)"""

    def __init__(self):
        self.s3_service = S3Service()
        self.analyzer = ProxyBackgroundAnalyzer()
        # Initialize OpenAI client (for embeddings)
        self.openai_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )

    def get_background_chunks_by_filing_id(self, sec_filing_summary_id: str) -> str:
        """Get background section chunks using sec_filing_summary_id filter"""
        try:
            logger.info(
                f"Searching for background chunks (Filing Summary ID: {sec_filing_summary_id})")

            # Initialize Pinecone
            pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
            index_name = os.environ.get(
                "PINECONE_INDEX_NAME_PROXY", "contract-chunks")
            index = pc.Index(index_name)

            # Search for background section chunks using title filters
            filters = [
                {"title": {"$eq": "Background of the Mergers"}},
                {"title": {"$eq": "BACKGROUND OF THE MERGERS"}},
                {"title": {"$eq": "Background of the Transaction"}},
                {"title": {"$eq": "BACKGROUND OF THE TRANSACTION"}},
                {"title": {"$eq": "Background of the Merger"}},
                {"title": {"$eq": "BACKGROUND OF THE MERGER"}},
                {"title": {"$eq": "Background of the Transactions"}},
                {"title": {"$eq": "BACKGROUND OF THE TRANSACTIONS"}},
            ]

            # Use a dummy vector for filter-only search
            dummy_vector = [0.0] * 3072

            search_response = index.query(
                vector=dummy_vector,
                top_k=50,  # Get more chunks for background
                include_metadata=True,
                filter={
                    "sec_filing_summary_id": sec_filing_summary_id,
                    "$or": filters
                }
            )

            # Format results
            background_chunks = []
            for match in search_response.matches:
                result = {
                    'score': match.score,
                    'text': match.metadata.get('original_text', ''),
                    'id': match.id
                }
                background_chunks.append(result)

            logger.info(f"Found {len(background_chunks)} background chunks")

            if not background_chunks:
                logger.warning("No background chunks found by title BGM")

                filters = [
                    {"title": {"$eq": "The Merger"}},
                    {"title": {"$eq": "THE MERGER"}}
                ]

                # Use a dummy vector for filter-only search
                dummy_vector = [0.0] * 3072

                search_response = index.query(
                    vector=dummy_vector,
                    top_k=10,  # Get more chunks for background
                    include_metadata=True,
                    filter={
                        "sec_filing_summary_id": sec_filing_summary_id,
                        "$or": filters
                    }
                )

                for match in search_response.matches:
                    result = {
                        'score': match.score,
                        'text': match.metadata.get('original_text', ''),
                        'title': match.metadata.get('title', ''),
                        'id': match.id
                    }
                    background_chunks.append(result)

                logger.info(
                    f"Found {len(background_chunks)} background chunks by proxy_id & query")

                logger.info(f"Background chunks1: {background_chunks}")

            if not background_chunks:
                logger.warning(
                    "No background chunks found by title The merger")

                query = "Background of the Mergers , Background of the Transaction,Background of the Merger,timeline of events leading to the merger, chronology of negotiations, deal process timeline, discussions between company and buyer, board deliberations, The Merger, Effects of the Merger,banker engagement, strategic alternatives review, key meetings, proposals, term sheets, letters of intent, fairness opinion process."
                query_embedding = self.openai_client.embeddings.create(
                    input=query,
                    model="text-embedding-3-large"
                )
                query_embedding = query_embedding.data[0].embedding

                search_response = index.query(
                    vector=query_embedding,
                    top_k=15,  # Get more chunks for background
                    include_metadata=True,
                    filter={
                        "sec_filing_summary_id": sec_filing_summary_id,
                    }
                )

                for match in search_response.matches:
                    print(f"match.id: {match.id}")
                    result = {
                        'score': match.score,
                        'text': match.metadata.get('original_text', ''),
                        'id': match.id
                    }
                    background_chunks.append(result)

                logger.info(
                    f"Found {len(background_chunks)} background chunks by proxy_id & query")

                logger.info(f"Background chunks1: {background_chunks}")

            # Format chunks to document
            document_parts = []
            for i, result in enumerate(background_chunks, 1):
                document_parts.append(f"[Chunk {i}]")
                document_parts.append(result['text'])
                document_parts.append("")

            document_text = "\n".join(document_parts)
            logger.info(f"Document length: {len(document_text):,} characters")

            return document_text

        except Exception as e:
            logger.error(f"Error getting background chunks: {str(e)}")
            return ""

    def generate_summary_document(self, sec_filing_summary_id: str, questions_file: str = None) -> Dict[str, Any]:
        """
        Generate summary document for a proxy filing.

        Args:
            sec_filing_summary_id: The SECFilingSummary document ID
            questions_file: Optional path to questions JSON file

        Returns:
            Dict with success status and docx_url if successful
        """
        try:
            logger.info(
                f"Starting summary generation for sec_filing_summary_id: {sec_filing_summary_id}")

            # Step 1: Get background chunks
            document_text = self.get_background_chunks_by_filing_id(
                sec_filing_summary_id)

            if not document_text:
                logger.warning(
                    "No background chunks found, skipping summary generation")
                return {
                    "success": False,
                    "error": "No background chunks found in Pinecone"
                }

            # Step 2: Generate merger background analysis
            logger.info("Generating merger background analysis")
            merger_background_results = self._generate_merger_background_analysis(
                document_text)

            # Step 3: Process questions if provided
            qa_content = ""
            if questions_file and os.path.exists(questions_file):
                logger.info(f"Processing questions from {questions_file}")
                qa_content = self._process_questions(
                    sec_filing_summary_id, questions_file)

            # Step 4: Create DOCX document
            logger.info("Creating DOCX document")
            docx_path = self._create_docx_document(
                sec_filing_summary_id=sec_filing_summary_id,
                qa_content=qa_content,
                merger_background_results=merger_background_results
            )

            if not docx_path:
                return {
                    "success": False,
                    "error": "Failed to create DOCX document"
                }

            # Step 5: Upload to S3
            logger.info("Uploading DOCX to S3")
            s3_key = f"proxy-summaries/{sec_filing_summary_id}_summary.docx"
            docx_url = self.s3_service.upload_file(
                file_path=docx_path,
                s3_key=s3_key,
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

            # Cleanup temp file
            try:
                if os.path.exists(docx_path):
                    os.unlink(docx_path)
            except Exception as e:
                logger.warning(f"Could not delete temp file: {e}")

            logger.info(f"Summary document generated and uploaded: {docx_url}")

            return {
                "success": True,
                "docx_url": docx_url,
                "s3_key": s3_key
            }

        except Exception as e:
            logger.error(
                f"Error generating summary document: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    def _generate_merger_background_analysis(self, document_text: str) -> Dict[str, Any]:
        """Generate comprehensive background analysis"""
        try:
            if not document_text:
                return {
                    "success": False,
                    "error": "No document text provided"
                }

            # Stage 1: Deep extraction
            extraction_result = self.analyzer.stage_1_extraction(document_text)

            if not extraction_result.get('success'):
                return {
                    "success": False,
                    "error": extraction_result.get('error', 'Extraction failed'),
                    "extraction_result": extraction_result
                }

            # Stage 2a: Strict summary
            strict_summary_result = self.analyzer.stage_2a_strict_summary(
                extraction_result['extraction_text']
            )

            return {
                "success": True,
                "extraction_result": extraction_result,
                "strict_summary_result": strict_summary_result
            }

        except Exception as e:
            logger.error(f"Error in merger background analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def _process_questions(self, sec_filing_summary_id: str, questions_file: str) -> str:
        """Process questions and generate Q&A content"""
        try:
            # Load questions
            with open(questions_file, "r", encoding="utf-8") as f:
                questions_data = json.load(f)

            # Initialize processor
            processor = self._create_query_processor()

            if processor is None:
                logger.warning(
                    "QueryProcessor could not be initialized, skipping Q&A processing")
                return ""

            all_summaries = []

            for question_key, question in questions_data.items():
                try:
                    # Search for similar chunks using sec_filing_summary_id
                    results = self._search_chunks_by_filing_id(
                        processor, question, sec_filing_summary_id)

                    if not results:
                        logger.warning(
                            f"No results found for question {question_key}")
                        continue

                    # Get Claude response
                    claude_answer = processor.get_claude_response(
                        question, results)

                    # Generate arbitrage summary
                    arb_summary = processor.generate_arb_summary(
                        claude_answer, question)
                    all_summaries.append(arb_summary)

                except Exception as e:
                    logger.error(
                        f"Error processing question {question_key}: {str(e)}", exc_info=True)
                    continue

            return "\n\n".join(all_summaries)

        except Exception as e:
            logger.error(
                f"Error processing questions: {str(e)}", exc_info=True)
            return ""

    def _create_query_processor(self):
        """Create QueryProcessor instance for Q&A processing"""
        try:
            # QueryProcessor is imported at the top level
            processor = QueryProcessor()
            logger.info("Successfully initialized QueryProcessor")
            return processor
        except Exception as e:
            logger.error(
                f"Could not initialize QueryProcessor: {str(e)}", exc_info=True)
            return None

    def _search_chunks_by_filing_id(self, processor, query: str, sec_filing_summary_id: str, top_k: int = 8) -> List[Dict]:
        """Search for chunks using sec_filing_summary_id"""
        try:
            if processor is None:
                logger.error("Processor is None, cannot search chunks")
                return []

            if not hasattr(processor, 'create_query_embedding'):
                logger.error(
                    "Processor does not have create_query_embedding method")
                return []

            # Create query embedding
            query_embedding = processor.create_query_embedding(query)
            all_results = []

            # Search for preamble chunks
            preamble_results = self._search_preamble_chunks_by_filing_id(
                processor, query_embedding, sec_filing_summary_id, top_k
            )
            all_results.extend(preamble_results)

            # Search for general chunks
            general_results = self._search_general_chunks_by_filing_id(
                processor, query_embedding, sec_filing_summary_id, top_k
            )
            all_results.extend(general_results)

            # Remove duplicates
            seen_ids = set()
            deduplicated_results = []
            for result in all_results:
                if result['id'] not in seen_ids:
                    seen_ids.add(result['id'])
                    deduplicated_results.append(result)

            return deduplicated_results

        except Exception as e:
            logger.error(f"Error searching chunks: {str(e)}", exc_info=True)
            return []

    def _search_preamble_chunks_by_filing_id(self, processor, query_embedding: List[float], sec_filing_summary_id: str, top_k: int = 8) -> List[Dict]:
        """Search for preamble chunks using sec_filing_summary_id"""
        try:
            if not hasattr(processor, 'index'):
                logger.error("Processor does not have index attribute")
                return []

            search_response = processor.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter={
                    "sec_filing_summary_id": sec_filing_summary_id,
                    "title": {"$eq": "Preamble"}
                }
            )

            results = []
            for match in search_response.matches:
                result = {
                    'score': match.score,
                    'text': match.metadata.get('original_text', ''),
                    'id': match.id
                }
                results.append(result)

            return results
        except Exception as e:
            logger.error(f"Error searching preamble chunks: {str(e)}")
            return []

    def _search_general_chunks_by_filing_id(self, processor, query_embedding: List[float], sec_filing_summary_id: str, top_k: int = 8) -> List[Dict]:
        """Search for general chunks using sec_filing_summary_id"""
        try:
            if not hasattr(processor, 'index'):
                logger.error("Processor does not have index attribute")
                return []

            search_response = processor.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter={
                    "sec_filing_summary_id": sec_filing_summary_id
                }
            )

            results = []
            for match in search_response.matches:
                result = {
                    'score': match.score,
                    'text': match.metadata.get('original_text', ''),
                    'id': match.id
                }
                results.append(result)

            return results
        except Exception as e:
            logger.error(f"Error searching general chunks: {str(e)}")
            return []

    def _create_docx_document(self, sec_filing_summary_id: str, qa_content: str, merger_background_results: Dict[str, Any]) -> str:
        """Create DOCX document combining Q&A and merger background"""
        try:
            doc = Document()
            formatter = DOCXFormatter(doc)

            # Add title
            formatter.add_title('Proxy Summary')
            doc.add_paragraph()

            # Process Q&A content if available
            if qa_content:
                self._process_content_section(doc, qa_content)

            # Add merger background analysis if available
            if merger_background_results and merger_background_results.get('success'):
                doc.add_page_break()
                doc.add_heading(
                    'Merger Background Analysis - Client Deliverables', 1)

                extraction_result = merger_background_results.get(
                    'extraction_result', {})
                strict_summary_result = merger_background_results.get(
                    'strict_summary_result', {})

                # Section 1: Chronological Summary
                doc.add_heading('Chronological Summary', 1)

                if strict_summary_result.get('success'):
                    formatter.add_simple_numbered_summary(
                        strict_summary_result['summary_text'])

                    # Validation info
                    if 'validation' in strict_summary_result:
                        formatter.add_horizontal_line()
                        from docx.shared import Pt, RGBColor
                        val_para = doc.add_paragraph()
                        val_para.paragraph_format.space_before = Pt(6)

                        val_color = RGBColor(
                            0, 128, 0) if strict_summary_result['validation']['passed'] else RGBColor(255, 0, 0)

                        val_run = val_para.add_run(
                            f"✓ Format Validation: {strict_summary_result['validation']['score']}/{strict_summary_result['validation']['total']} checks passed"
                        )
                        val_run.font.size = Pt(9)
                        val_run.font.color.rgb = val_color

                        if 'max_sentence_length' in strict_summary_result['validation']:
                            val_para.add_run(
                                f" | Max sentence: {strict_summary_result['validation']['max_sentence_length']} words"
                            )
                else:
                    doc.add_paragraph(
                        f"❌ Error: {strict_summary_result.get('error', 'Unknown error')}")

                # Extract and add other client deliverable sections
                if extraction_result.get('success'):
                    doc.add_page_break()
                    doc.add_heading(
                        'Extraction of other client deliverable sections', 1)
                    extraction_text = extraction_result['extraction_text']
                    sections = _parse_extraction_for_document_a(
                        extraction_text)

                    for section_title, section_content in sections:
                        doc.add_page_break()
                        doc.add_heading(section_title, 1)

                        # Check if it's a table
                        if _is_table(section_content):
                            formatter.add_professional_table(section_content)
                        else:
                            # Regular text
                            paragraphs = section_content.split('\n\n')
                            for para in paragraphs:
                                if para.strip():
                                    clean_para = formatter._clean_markdown(
                                        para.strip())
                                    doc.add_paragraph(clean_para)

            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix='.docx')
            temp_file.close()
            doc.save(temp_file.name)

            return temp_file.name

        except Exception as e:
            logger.error(f"Error creating DOCX document: {str(e)}")
            return None

    def _process_content_section(self, doc: Document, content: str):
        """Helper function to process content and add to document"""
        import re
        from docx.shared import Pt, RGBColor

        lines = content.split('\n')
        in_bullet_section = False
        current_header = None

        for line in lines:
            # Check if line is a header (starts with **)
            if line.strip().startswith("**") and line.strip().endswith("**"):
                if current_header:
                    doc.add_paragraph()
                    doc.add_paragraph()

                header_text = line.strip().replace("**", "")
                header_para = doc.add_paragraph()
                header_run = header_para.add_run(header_text)
                header_run.bold = True
                header_run.font.size = Pt(11)
                header_run.font.color.rgb = RGBColor(0x4f, 0x81, 0xbd)
                header_para.paragraph_format.space_after = Pt(8)
                current_header = header_text
                in_bullet_section = True

            # Check if line is a bullet point
            elif line.strip().startswith(("•", "-", "*")):
                bullet_text = line.strip()[1:].strip()
                if bullet_text:
                    para = doc.add_paragraph()
                    run = para.add_run("+     ")
                    run.font.size = Pt(11)

                    parts = re.split(r'(\*\*.*?\*\*)', bullet_text)
                    for part in parts:
                        if part.startswith("**") and part.endswith("**"):
                            text = part[2:-2]
                            run = para.add_run(text)
                            run.font.size = Pt(11)
                        elif part:
                            run = para.add_run(part)
                            run.font.size = Pt(11)

                    para.paragraph_format.left_indent = Pt(25)
                    para.paragraph_format.first_line_indent = Pt(-20)
                    para.paragraph_format.space_after = Pt(6)

            # Regular text
            elif line.strip() and "="*40 not in line:
                if in_bullet_section and len(line.strip()) > 10:
                    para = doc.add_paragraph()
                    run = para.add_run("+     ")
                    run.font.size = Pt(11)

                    parts = re.split(r'(\*\*.*?\*\*)', line.strip())
                    for part in parts:
                        if part.startswith("**") and part.endswith("**"):
                            text = part[2:-2]
                            run = para.add_run(text)
                            run.font.size = Pt(11)
                        elif part:
                            run = para.add_run(part)
                            run.font.size = Pt(11)

                    para.paragraph_format.left_indent = Pt(25)
                    para.paragraph_format.first_line_indent = Pt(-20)
                    para.paragraph_format.space_after = Pt(6)
                else:
                    para = doc.add_paragraph(line.strip())
                    para.paragraph_format.space_after = Pt(6)
