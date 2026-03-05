import os
import json
import logging
import pinecone
import openai
from dotenv import load_dotenv
from typing import Dict, Any, List
import time
import tiktoken
import requests
import tempfile


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flattened_to_pinecone.log',
                            mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SectionProcessorV2:
    """
    Process proxy sections and upload to Pinecone.
    V2: Uses sec_filing_summary_id instead of proxy_id.
    """
    def __init__(self, sec_filing_summary_id: str = None, deal_id: str = None):
        # Load environment variables
        load_dotenv()

        # Store filing summary information
        self.sec_filing_summary_id = sec_filing_summary_id
        self.deal_id = deal_id

        # Initialize OpenAI
        self.openai_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )

        # Initialize Pinecone
        pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

        # Connect to index
        self.index_name = os.environ.get(
            "PINECONE_INDEX_NAME_PROXY", "contract-chunks")
        self.index = pc.Index(self.index_name)

        # Maximum metadata size (40KB)
        self.MAX_METADATA_SIZE = 40960

        # Store flattened sections
        self.flattened_sections = []

        # Maximum tokens per chunk (leaving some buffer)
        self.MAX_TOKENS = 7000
        self.EMBEDDING_MAX_TOKENS = 8191

        # Overlap tokens between chunks
        self.OVERLAP_TOKENS = 200

    def download_json_from_s3(self, s3_url: str) -> str:
        """
        Download JSON file from S3 URL and return the local file path.

        Args:
            s3_url: The S3 URL of the JSON file

        Returns:
            str: Local file path of the downloaded JSON file
        """
        try:
            logger.info(f"Downloading JSON from S3 URL: {s3_url}")

            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(
                mode='w+', suffix='.json', delete=False)
            temp_file_path = temp_file.name
            temp_file.close()

            # Download the file
            response = requests.get(s3_url, timeout=30)
            response.raise_for_status()

            # Parse and validate JSON
            json_data = response.json()

            # Write to temporary file
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully downloaded JSON to: {temp_file_path}")
            return temp_file_path

        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading JSON from S3: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from S3: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading JSON: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text"""
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(text))
        return num_tokens

    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks based on token count with overlap"""
        # First try splitting on double newlines
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            logger.info(f"Paragraph tokens: {para_tokens}")

            # If a single paragraph exceeds max tokens, split it into smaller pieces
            if para_tokens > self.MAX_TOKENS:
                words = para.split()
                temp_chunk = []
                temp_tokens = 0

                logger.info(f"Splitting text into words: {words}")

                for word in words:

                    word_tokens = self.count_tokens(word + ' ')
                    if temp_tokens + word_tokens > self.MAX_TOKENS:
                        chunks.append(' '.join(temp_chunk))
                        # Keep some overlap
                        # Keep last 50 words for overlap
                        overlap_words = temp_chunk[-50:]
                        temp_chunk = overlap_words + [word]
                        temp_tokens = self.count_tokens(' '.join(temp_chunk))
                    else:
                        temp_chunk.append(word)
                        temp_tokens += word_tokens

                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))

            # For normal paragraphs
            elif current_tokens + para_tokens > self.MAX_TOKENS:
                logger.info(
                    f"Splitting paragraph into chunks: {current_chunk}")
                logger.info(
                    f"Current tokens: {current_tokens + para_tokens}")
                logger.info(f"chunks: {chunks}")
                chunks.append('\n\n'.join(current_chunk))

                # Initialize overlap paragraphs with the last paragraph
                overlap_paragraphs = [current_chunk[-1]]
                overlap_tokens = self.count_tokens(overlap_paragraphs[0])

                # Calculate tokens needed for the new paragraph
                new_para_tokens = self.count_tokens(para)

                # If overlap is less than minimum (50 tokens), try to add more paragraphs
                # But don't exceed maximum overlap (250 tokens) or MAX_TOKENS when combined with new paragraph
                if overlap_tokens < 50 and len(current_chunk) > 1:
                    for prev_para in reversed(current_chunk[:-1]):
                        prev_para_tokens = self.count_tokens(prev_para)
                        # Check if adding this paragraph would exceed either limit
                        if (overlap_tokens + prev_para_tokens <= 150 and
                                overlap_tokens + prev_para_tokens + new_para_tokens <= self.MAX_TOKENS):
                            overlap_paragraphs.insert(0, prev_para)
                            overlap_tokens += prev_para_tokens
                            if overlap_tokens >= 50:
                                break
                        else:
                            break

                # Set the new current chunk with proper overlap
                current_chunk = overlap_paragraphs + [para]
                current_tokens = self.count_tokens('\n\n'.join(current_chunk))

                logger.info(f"Overlap tokens: {overlap_tokens}")
                logger.info(f"New paragraph tokens: {new_para_tokens}")
                logger.info(f"Total tokens in new chunk: {current_tokens}")
                logger.info(f"Current chunk2: {current_chunk}")
                logger.info(f"Current tokens2: {current_tokens}")
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        # Add the last chunk if there's anything left
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        # Only for debugging
        total = 0
        for idx, chunk in enumerate(chunks):
            tokens = self.count_tokens(chunk)
            total += tokens
            logger.info(f"Chunk {idx+1} - Tokens: {tokens}")

        logger.info(f"Total (sum of individual chunks): {total}")
        return chunks

    def trim_metadata(self, metadata):
        # Try serializing first
        meta_bytes = json.dumps(metadata).encode("utf-8")
        if len(meta_bytes) <= self.MAX_METADATA_SIZE:
            return metadata

        logger.warning(
            f"Metadata size ({len(meta_bytes)} bytes) exceeds maximum allowed size ({self.MAX_METADATA_SIZE} bytes). Metadata: {metadata}")
        # Sort keys by importance
        priority_keys = [
            "title",
            "sec_filing_summary_id",
            "page_no",
            "parent_section",
            "original_text",
        ]

        logger.info(f"Metadata: {metadata}")

        trimmed = {}
        for key in priority_keys:
            if key not in metadata:
                continue
            value = metadata[key]

            # Temporarily add full value
            trimmed[key] = value
            size = len(json.dumps(trimmed).encode("utf-8"))

            # Trim the value if adding it exceeds the limit
            if size > self.MAX_METADATA_SIZE:
                if isinstance(value, str):
                    # Binary search to trim the string exactly to fit
                    left, right = 0, len(value)
                    while left < right:
                        mid = (left + right) // 2
                        trimmed[key] = value[:mid]
                        size = len(json.dumps(trimmed).encode("utf-8"))
                        if size <= self.MAX_METADATA_SIZE:
                            left = mid + 1
                        else:
                            right = mid - 1
                    trimmed[key] = value[:right]
                else:
                    # Remove non-string or too-large fields
                    trimmed.pop(key)

        return trimmed

    def create_embedding(self, text: str) -> List[float]:
        """Create an embedding for text using OpenAI API"""
        try:
            # Check token count before creating embedding
            token_count = self.count_tokens(text)
            if token_count > self.EMBEDDING_MAX_TOKENS:
                raise ValueError(
                    f"Text too long: {token_count} tokens (max {self.EMBEDDING_MAX_TOKENS})")

            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error for title: {text[:300]}")
            logger.error(f"Error creating embedding: {str(e)}")
            raise

    def flatten_section(self, section: Dict[Any, Any], parent_title: str = None) -> None:
        """Flatten a section and its subsections recursively and split into chunks"""
        # Get the content and title
        content = section.get('content', '')
        title = section.get('title', '')
        page_no = section.get('page-no', '')

        logger.info(f"Parent title: {parent_title}")
        logger.info(f"Flattening section: {title}")

        if content:
            # Split content into chunks
            content_chunks = self.split_text_into_chunks(content)

            # Create a flattened entry for each chunk
            for i, chunk in enumerate(content_chunks):
                flattened = {
                    'title': title,
                    'page_no': page_no,
                    'parent_section': parent_title if parent_title else '',
                    'content': chunk,
                    'chunk_index': i,
                    'total_chunks': len(content_chunks),
                    'vector_id': f"{title.lower().replace(' ', '_')}_{i}"
                }
                self.flattened_sections.append(flattened)
                logger.info(
                    f"Flattened section: {title} - Chunk {i+1}/{len(content_chunks)}")

        # Process subsections recursively
        for subsection in section.get('subsection', []):
            self.flatten_section(subsection, parent_title=title)

    def process_flattened_sections(self) -> None:
        """Process all flattened sections and upload to Pinecone"""
        total_sections = len(self.flattened_sections)
        logger.info(f"Starting to process {total_sections} flattened sections")

        for i, section in enumerate(self.flattened_sections, 1):
            try:
                # Create embedding for the chunk
                embedding = self.create_embedding(section['content'])

                # Prepare metadata with sec_filing_summary_id
                metadata = {
                    'title': section['title'],
                    'page_no': section['page_no'],
                    'deal_id': self.deal_id or '',
                    'sec_filing_summary_id': self.sec_filing_summary_id or '',
                    'parent_section': section['parent_section'],
                    'original_text': section['content'],
                    'metadata': True
                }

                # Add deal_id to metadata if available
                if self.deal_id:
                    metadata['deal_id'] = self.deal_id

                # Trim metadata if needed
                metadata = self.trim_metadata(metadata)

                # Upload to Pinecone with filing-summary-based ID
                filing_prefix = self.sec_filing_summary_id.lower().replace(
                    ' ', '_') if self.sec_filing_summary_id else 'filing'
                vector_id = f"{filing_prefix}_{i}"

                self.index.upsert(
                    vectors=[{
                        'id': vector_id,
                        'values': embedding,
                        'metadata': metadata
                    }]
                )

                logger.info(
                    f"Processed section {i}/{total_sections}: {section['title']} (Chunk {section['chunk_index'] + 1}/{section['total_chunks']})")

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                logger.error(
                    f"Error processing section {section['title']}: {str(e)}")

    def process_file(self, file_path: str) -> None:
        """Process the entire JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sections = json.load(f)

            # First, flatten all sections and split into chunks
            logger.info("Starting to flatten sections and create chunks")
            for section in sections:
                self.flatten_section(section)

            # Save flattened sections with chunks to JSON
            filing_name = self.sec_filing_summary_id.lower().replace(
                ' ', '_') if self.sec_filing_summary_id else 'filing'
            output_filename = f'flattened_sections_{filing_name}.json'
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(self.flattened_sections, f, indent=2)
            logger.info(
                f"Saved {len(self.flattened_sections)} flattened sections (including chunks) to {output_filename}")

            # Process flattened sections and upload to Pinecone
            self.process_flattened_sections()

            logger.info("Completed processing all sections")

        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise

    def process_from_s3_url(self, s3_url: str) -> None:
        """
        Process sections from S3 URL by downloading the JSON file first.

        Args:
            s3_url: The S3 URL of the sections JSON file
        """
        temp_file_path = None
        try:
            logger.info(f"Starting to process sections from S3 URL: {s3_url}")

            # Download JSON from S3
            temp_file_path = self.download_json_from_s3(s3_url)

            # Process the downloaded file
            self.process_file(temp_file_path)

            logger.info("Successfully completed processing from S3 URL")

        except Exception as e:
            logger.error(f"Error processing from S3 URL: {str(e)}")
            raise
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.info(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(
                        f"Could not clean up temporary file {temp_file_path}: {e}")


if __name__ == "__main__":
    processor = SectionProcessorV2()
    processor.process_file(
        'parse-json/sections_with_content_html_STEEL_NEWCO.json')
