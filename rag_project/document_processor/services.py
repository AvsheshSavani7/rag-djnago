import os
import json
import boto3
import uuid
import requests
from datetime import datetime
from django.conf import settings
from dotenv import load_dotenv
import openai
import pinecone
from tqdm.auto import tqdm
import time
import logging
from bson import ObjectId
import concurrent.futures
import traceback

from .models import ProcessingJob

load_dotenv()

logger = logging.getLogger(__name__)


class DocumentProcessingService:
    """
    Service to process document files and create embeddings
    Provides the same functionality as ProcessFileView but can be called directly
    """

    def __init__(self):
        # Set up executor for background tasks
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

    def process_document(self, file_url, deal_id, embed_data=True):
        """
        Process a document file from a URL

        Args:
            file_url (str): URL of the file to process
            deal_id (str): ID of the deal/job
            embed_data (bool): Whether to also generate embeddings

        Returns:
            dict: Result of the operation with status and details
        """
        try:
            # Validate parameters
            if not file_url or not isinstance(file_url, str):
                error_msg = f"Invalid file_url parameter: {file_url}"
                logger.error(error_msg)
                return {"error": error_msg, "status": "failed"}

            if not deal_id or not isinstance(deal_id, str):
                error_msg = f"Invalid deal_id parameter: {deal_id}"
                logger.error(error_msg)
                return {"error": error_msg, "status": "failed"}

            # Ensure embed_data is a boolean
            embed_data = bool(embed_data)

            logger.info(
                f"Processing document with file_url={file_url}, deal_id={deal_id}, embed_data={embed_data}")

            # Find the existing job by _id (convert string to ObjectId)
            try:
                object_id = ObjectId(deal_id)
                job = ProcessingJob.objects.get(_id=object_id)
                logger.info(f"Found job in database: {job}")
            except ProcessingJob.DoesNotExist:
                error_msg = f"No processing job found for deal_id {deal_id}"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "status": "failed"
                }
            except Exception as e:
                error_msg = f"Invalid deal_id format: {str(e)}"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "status": "failed"
                }

            logger.info(f"Processing file_url: {file_url}")
            # Initialize the processor
            processor = FlattenProcessor(file_url=file_url)

            # Process the file
            try:
                result = processor.process()
                logger.info(f"Processing result: {result}")
            except Exception as proc_e:
                error_msg = f"Error processing file: {str(proc_e)}"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "status": "failed"
                }

            # Update the job with results
            job.flattened_json_url = result.get('flattened_json_url')
            job.save()

            # Start embedding process in background if requested
            if embed_data:
                # Start embedding task in the executor
                self.executor.submit(
                    self._process_embeddings,
                    str(job._id),
                    job.flattened_json_url
                )
                logger.info(
                    f"Submitted embedding task for job {job._id} to executor")

                # Return response with embedding status
                return {
                    'deal_id': str(job._id),
                    'flattened_json_url': job.flattened_json_url,
                    'embedding_status': 'PROCESSING',
                    'message': 'File processed successfully. Embeddings are being generated in the background.',
                    'status': 'success'
                }

            # Return response without embedding
            return {
                'deal_id': str(job._id),
                'flattened_json_url': job.flattened_json_url,
                'status': 'success'
            }

        except Exception as e:
            # Log the error
            logger.error(f"Error processing file: {str(e)}")

            # Update the job with error if it exists
            if 'job' in locals():
                job.error_message = str(e)
                job.save()

            # Return error response
            return {
                'error': str(e),
                'deal_id': deal_id if 'job' not in locals() else str(job._id),
                'status': 'failed'
            }

    def _process_embeddings(self, job_id, flattened_json_url):
        """
        Background task to process embeddings

        Args:
            job_id (str): ID of the job
            flattened_json_url (str): URL to the flattened JSON file
        """
        logger.info(
            f"Starting embedding process for job {job_id} with URL: {flattened_json_url}")
        try:
            # Get the job
            object_id = ObjectId(job_id)
            job = ProcessingJob.objects.get(_id=object_id)
            logger.info(f"Found job in database: {job}")

            # Update job status to processing
            job.update_embedding_status('PROCESSING')
            logger.info("Updated job status to PROCESSING")

            # Download flattened JSON
            s3_service = S3Service()
            logger.info(
                f"Downloading flattened JSON from URL: {flattened_json_url}")
            chunks = s3_service.download_from_url(flattened_json_url)
            logger.info(f"Downloaded {len(chunks)} chunks")

            # Process embeddings
            logger.info("Initializing embedding service")
            embedding_service = EmbeddingService()
            logger.info(
                f"Starting embedding creation for {len(chunks)} chunks")
            result = embedding_service.process_chunks(chunks, str(job_id))
            logger.info(f"Embedding completed with result: {result}")

            # Update job status to completed
            job.update_embedding_status('COMPLETED')
            logger.info(f"Updated job status to COMPLETED")

        except Exception as e:
            logger.error(f"Error processing embeddings: {str(e)}")

            # Update job status to failed
            try:
                object_id = ObjectId(job_id)
                job = ProcessingJob.objects.get(_id=object_id)
                job.update_embedding_status('FAILED', str(e))
                logger.error(f"Updated job status to FAILED: {str(e)}")
            except Exception as inner_e:
                logger.error(f"Error updating job status: {str(inner_e)}")


class S3Service:
    """Service to handle S3 operations"""

    def __init__(self):
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
        self.bucket = os.environ.get("AWS_S3_BUCKET")
        print("env", os.environ.get("AWS_ACCESS_KEY_ID"))

    def upload_json(self, data, key):
        """Upload JSON data to S3 and return the full URL"""
        json_bytes = json.dumps(data, indent=2).encode("utf-8")
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json_bytes,
            ContentType='application/json'
        )
        # Generate and return the full S3 URL
        s3_url = f"https://{self.bucket}.s3.amazonaws.com/{key}"
        return s3_url

    def download_from_url(self, url):
        """Download JSON from a URL"""
        try:
            logger.info(f"Downloading from URL: {url}")
            response = requests.get(url)
            logger.info(f"Response status code: {response.status_code}")
            logger.info(
                f"Response content type: {response.headers.get('Content-Type', 'unknown')}")

            # Try to log the first part of the response content to see what we're getting
            content_preview = response.text[:200] if response.text else "empty response"
            logger.info(f"Response content preview: {content_preview}")

            response.raise_for_status()  # Raise exception for bad responses

            # Try parsing as JSON and handle errors explicitly
            try:
                json_data = response.json()
                logger.info(
                    f"Successfully parsed JSON data, type: {type(json_data)}")
                return json_data
            except ValueError as json_err:
                logger.error(
                    f"Failed to parse JSON from response: {str(json_err)}")
                raise Exception(f"Invalid JSON in response: {str(json_err)}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"HTTP request failed: {str(req_err)}")
            raise Exception(f"Failed to download from URL: {str(req_err)}")

    def download_json(self, key):
        """Download and parse JSON from S3"""
        response = self.s3.get_object(Bucket=self.bucket, Key=key)
        return json.load(response["Body"])

    def list_files(self, prefix):
        """List files in S3 with the given prefix"""
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".json")]


class EmbeddingService:
    """Service to create embeddings using OpenAI and store them in Pinecone"""

    def __init__(self):
        print("Initializing EmbeddingService")
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        print(
            f"OpenAI API key set: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No'}")

        # Initialize Pinecone
        pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        print(
            f"Pinecone API key set: {'Yes' if os.environ.get('PINECONE_API_KEY') else 'No'}")

        # Get or create index
        self.index_name = os.environ.get(
            "PINECONE_INDEX_NAME", "contract-chunks")
        print(f"Using Pinecone index: {self.index_name}")

        # Check if index exists
        try:
            indexes = [index.name for index in pc.list_indexes()]
            print(f"Available Pinecone indexes: {indexes}")

            if self.index_name not in indexes:
                # Create index if it doesn't exist
                print(f"Creating new Pinecone index: {self.index_name}")
                pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # Dimensions for text-embedding-3-small
                    metric="cosine"
                )
                print(f"Created new Pinecone index: {self.index_name}")

            # Connect to the index
            self.index = pc.Index(self.index_name)
            print(f"Connected to Pinecone index: {self.index_name}")

            # Initialize the metadata enhancement service
            self.metadata_service = MetadataEnhancementService()
            print("Metadata enhancement service initialized")
        except Exception as e:
            print(f"Error with Pinecone initialization: {str(e)}")
            raise

    def create_embedding(self, text):
        """Create an embedding for a single text using OpenAI"""
        try:
            print(f"Creating embedding for text (length: {len(text)})")
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            print("Embedding created successfully")
            return response.data[0].embedding
        except Exception as e:
            print(f"Error creating embedding: {str(e)}")
            raise

    def process_chunks(self, chunks, deal_id):
        """Process a list of text chunks and store embeddings in Pinecone one by one"""
        total_chunks = len(chunks)
        processed_chunks = 0

        print(f"Processing {total_chunks} chunks for deal ID: {deal_id}")

        try:
            for i, chunk in enumerate(tqdm(chunks)):
                print(f"Processing chunk {i+1}/{total_chunks}")
                # Create a unique ID for each chunk
                chunk_id = f"{deal_id}_{i}"

                # Extract text from chunk
                text = chunk.get("combined_text", "")
                if not text:
                    print(f"Skipping chunk {i+1} - no text content")
                    continue

                # Enhance chunk metadata with category information
                try:
                    print(f"Enhancing metadata for chunk {i+1}")
                    enhanced_chunk = self.metadata_service.enhance_chunk_metadata(
                        chunk)
                    category_name = enhanced_chunk.get("category_name", "")
                    print(
                        f"Category determined for chunk {i+1}: {category_name}")
                except Exception as e:
                    print(
                        f"Error enhancing metadata for chunk {i+1}: {str(e)}")
                    # Continue with original chunk if enhancement fails
                    enhanced_chunk = chunk
                    enhanced_chunk["category_name"] = ""

                # Create embedding
                try:
                    embedding = self.create_embedding(text)
                    print(
                        f"Created embedding for chunk {i+1} - vector size: {len(embedding)}")
                except Exception as e:
                    print(
                        f"Error creating embedding for chunk {i+1}: {str(e)}")
                    raise Exception(
                        f"Failed to create embedding for chunk {i+1}: {str(e)}")

                # Start with required metadata fields
                metadata = {
                    "deal_id": str(deal_id),
                    "label": enhanced_chunk.get("label", "") or "",
                    "definition_terms": "" if enhanced_chunk.get("definition_terms") is None else str(enhanced_chunk.get("definition_terms", "")),
                    "original_text": enhanced_chunk.get("original_text", "") or "",
                    "combined_text": enhanced_chunk.get("combined_text", "") or "",
                    "category_name": enhanced_chunk.get("category_name", "") or "",
                    "chunk_index": i
                }

                # Add all other fields from enhanced_chunk
                # This will include any fields added by extract_structured_metadata
                for key, value in enhanced_chunk.items():
                    # Skip fields we already added and those that start with underscore (private fields)
                    if key not in metadata and not key.startswith('_'):
                        # Convert lists and dicts to strings to ensure compatibility with Pinecone
                        if isinstance(value, (list, dict)):
                            metadata[key] = json.dumps(value)
                        else:
                            metadata[key] = str(
                                value) if value is not None else ""

                print(
                    f"Metadata fields for chunk {i+1}: {list(metadata.keys())}")

                # Upsert single vector to Pinecone
                try:
                    self.index.upsert(
                        vectors=[{
                            "id": chunk_id,
                            "values": embedding,
                            "metadata": metadata
                        }]
                    )
                    print(
                        f"Uploaded chunk {i+1} to Pinecone with ID: {chunk_id}")
                except Exception as e:
                    print(f"Error uploading chunk {i+1} to Pinecone: {str(e)}")
                    print(f"Problematic metadata: {metadata}")
                    raise Exception(
                        f"Failed to upload chunk {i+1} to Pinecone: {str(e)}")

                processed_chunks += 1

                # Add a small delay to avoid rate limits
                time.sleep(0.2)

            print(
                f"Successfully processed {processed_chunks} out of {total_chunks} chunks")
            return {
                "status": "success",
                "chunks_processed": processed_chunks,
                "index_name": self.index_name
            }

        except Exception as e:
            print(f"Error processing chunks: {str(e)}")
            raise Exception(f"Failed to process chunks: {str(e)}")

    def search(self, query_text, deal_id=None, top_k=10):
        """
        Search for similar text chunks in Pinecone

        Args:
            query_text (str): The text to search for
            deal_id (str, optional): Filter results to a specific deal ID
            top_k (int, optional): Number of results to return. Defaults to 10.

        Returns:
            dict: Search results with matching chunks
        """
        try:
            print(
                f"Creating embedding for search query: {query_text[:100]}...")
            query_embedding = self.create_embedding(query_text)

            # Prepare filter if deal_id is provided
            filter_dict = {}
            if deal_id:
                filter_dict = {"deal_id": str(deal_id)}
                print(f"Filtering search to deal ID: {deal_id}")

            # Search in Pinecone
            print(f"Searching Pinecone index: {self.index_name}")
            search_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )

            # Format results
            results = []
            for match in search_response.matches:
                results.append({
                    "id": match.id,
                    "score": match.score,
                    "deal_id": match.metadata.get("deal_id"),
                    "label": match.metadata.get("label"),
                    "combined_text": match.metadata.get("combined_text"),
                    "definition_terms": match.metadata.get("definition_terms")
                })

            print(f"Found {len(results)} matching results")
            return {
                "query": query_text,
                "results": results,
                "total": len(results)
            }

        except Exception as e:
            print(f"Error searching for text: {str(e)}")
            raise Exception(f"Failed to search for text: {str(e)}")


class MetadataEnhancementService:
    """Service to enhance document metadata using GPT"""

    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        # Schema URL
        self.schema_url = "https://rag-mna.s3.eu-north-1.amazonaws.com/master-schema/enhanced_metadata_by_category.json"
        # Cache the categories
        self._schema_categories = None
        # Cache the full schema
        self._full_schema = None
        logger.info("MetadataEnhancementService initialized")

    def get_schema_categories(self):
        """
        Fetches and caches the schema categories from the URL

        Returns:
            list: List of category names
        """
        # Return cached categories if available
        if self._schema_categories:
            return self._schema_categories

        try:
            # Fetch the schema
            logger.info(f"Fetching schema from URL: {self.schema_url}")
            response = requests.get(self.schema_url)
            response.raise_for_status()

            # Parse the schema
            schema = response.json()
            # Cache the full schema
            self._full_schema = schema

            # Extract category names (keys)
            categories = list(schema.keys())
            logger.info(f"Extracted {len(categories)} categories from schema")

            # Cache for future use
            self._schema_categories = categories
            return categories

        except Exception as e:
            logger.error(f"Error fetching schema categories: {str(e)}")
            # Return empty list in case of error
            return []

    def get_full_schema(self):
        """
        Fetches and caches the full schema from URL

        Returns:
            dict: The full schema with all categories and their fields
        """
        # If schema is already cached, return it
        if self._full_schema:
            return self._full_schema

        # Otherwise, call get_schema_categories to fetch and cache both
        self.get_schema_categories()
        return self._full_schema

    # 1st First GPT call to determine the category
    def determine_category(self, text):
        """
        Determine the appropriate category for the given text using GPT

        Args:
            text (str): The text to categorize

        Returns:
            str: The determined category name, or empty string if not found
        """
        try:
            # Get all categories
            categories = self.get_schema_categories()
            if not categories:
                logger.warning("No categories available for classification")
                return ""

            # Prepare the prompt for GPT
            category_list = "\n".join(
                [f"{i+1}. {category}" for i, category in enumerate(categories)])

            prompt = f"""You are tasked with categorizing legal contract text into the most appropriate category.
Given the following text from a contract:

"{text}"

Please determine which of the following categories best fits this text:
{category_list}

Return ONLY the exact name of the single best matching category. If none match well, return an empty string."""

            logger.info("Calling GPT to determine category")
            # Call GPT
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a legal contract classifier assistant. Your task is to assign text to the most appropriate category."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            # Extract the determined category
            category = response.choices[0].message.content.strip()
            logger.info(f"GPT determined category: {category}")

            # Validate if the returned category is in our list
            if category in categories:
                return category
            elif category == "":
                return ""
            else:
                # Try to find a close match
                logger.warning(
                    f"Category '{category}' not found in schema, trying to find a close match")
                for schema_category in categories:
                    if category.lower() in schema_category.lower():
                        logger.info(f"Found close match: {schema_category}")
                        return schema_category

                logger.warning(f"No matching category found for: {category}")
                return ""

        except Exception as e:
            logger.error(f"Error determining category: {str(e)}")
            logger.error(traceback.format_exc())
            return ""

    # 2nd Second GPT call to extract structured metadata

    def extract_structured_metadata(self, text, category_name):
        """
        Extract structured metadata from text based on schema for the specified category

        Args:
            text (str): The text to extract metadata from
            category_name (str): The category name to get schema fields

        Returns:
            dict: Dictionary containing extracted metadata fields
        """
        try:
            # Get the full schema
            schema = self.get_full_schema()

            # Check if category exists in schema
            if not category_name or category_name not in schema:
                logger.warning(
                    f"Category '{category_name}' not found in schema")
                return {}

            # Get the fields for this category
            category_fields = schema[category_name]

            # Prepare a nice representation of the fields for the prompt
            field_descriptions = []
            for field in category_fields:
                field_descriptions.append(
                    f"Field: {field.get('name')}\nDescription: {field.get('description')}\nType: {field.get('data_type')}\nExample: {field.get('example_value')}")

            fields_text = "\n\n".join(field_descriptions)

            # Prepare the prompt for GPT
            prompt = f"""You are tasked with extracting structured metadata from a legal contract text.
Given the following text from a '{category_name}' section of a contract:

```
{text}
```

Please extract the following information based on the given fields. If a value cannot be determined, write "NA" or "Unknown":

{fields_text}

Also, please provide:
1. A brief summary of this clause (1-2 sentences) with key: "clause_summary"
2. A list of relevant tags for this clause with key: "clause_tags_llm" in array format

Return your answer as a valid JSON object with each field name as the key and the extracted value as its value.
"""

            logger.info(
                f"Calling GPT to extract structured metadata for category: {category_name}")
            # Call GPT
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a legal metadata extraction assistant. Your task is to extract structured information from contract text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )

            # Extract the structured metadata
            metadata_text = response.choices[0].message.content.strip()
            logger.info(f"GPT extracted metadata: {metadata_text[:200]}...")

            # Try to parse as JSON
            try:
                # Find json content if it's not directly formatted as json
                if not metadata_text.startswith('{'):
                    import re
                    json_match = re.search(
                        r'```json\s*([\s\S]*?)\s*```', metadata_text)
                    if json_match:
                        metadata_text = json_match.group(1)
                    else:
                        # Try to find any JSON-like structure
                        json_match = re.search(r'({[\s\S]*})', metadata_text)
                        if json_match:
                            metadata_text = json_match.group(1)

                structured_metadata = json.loads(metadata_text)
                logger.info(
                    f"Successfully parsed structured metadata with {len(structured_metadata)} fields")
                return structured_metadata
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse metadata as JSON: {str(e)}")
                logger.error(f"Problematic content: {metadata_text}")

                # Attempt to create a basic structured response
                return {
                    "clause_summary": "Failed to extract structured metadata",
                    "clause_tags_llm": [],
                    "error": "Failed to parse JSON from GPT response"
                }

        except Exception as e:
            logger.error(f"Error extracting structured metadata: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def enhance_chunk_metadata(self, chunk):
        """
        Enhance a chunk's metadata with additional information

        Args:
            chunk (dict): The chunk to enhance

        Returns:
            dict: The enhanced chunk with additional metadata
        """
        try:
            # Get the text to classify
            text = chunk.get("combined_text", "")
            if not text:
                logger.warning(
                    "Empty text in chunk, skipping category determination")
                chunk["category_name"] = ""
                return chunk

            # Determine the category
            category = self.determine_category(text)

            # Add the category to the chunk
            chunk["category_name"] = category
            chunk["category_tags"] = []
            chunk["category_summary"] = ""
            logger.info(f"Enhanced chunk with category: {category}")

            # If we have a valid category, extract structured metadata
            if category:
                logger.info(
                    f"Extracting structured metadata for category: {category}")
                structured_metadata = self.extract_structured_metadata(
                    text, category)

                # Add structured metadata to chunk
                if structured_metadata:
                    logger.info(
                        f"Adding {len(structured_metadata)} structured metadata fields to chunk")
                    for key, value in structured_metadata.items():
                        chunk[key] = value

            return chunk

        except Exception as e:
            logger.error(f"Error enhancing chunk metadata: {str(e)}")
            # Return the original chunk if enhancement fails
            chunk["category_name"] = ""
            return chunk


class FlattenProcessor:
    """Process and flatten nested JSON contract data"""

    def __init__(self, file_url, output_bucket=None):
        self.s3_service = S3Service()
        self.file_url = file_url
        self.output_bucket = output_bucket or os.environ.get("AWS_S3_BUCKET")
        self.total_definitions = 0
        self.total_clauses = 0

    def clean_unicode_quotes(self, text):
        if not text:
            return ""
        return text.replace('\u201c', '"').replace('\u201d', '"') \
                   .replace('\u2018', "'").replace('\u2019', "'") \
                   .replace('\u2013', "-").replace('\u2014', "-")

    def is_definition_section(self, text):
        return bool(text and ":" in text and "means" in text)

    def walk_structure(self, article, path=None):
        if path is None:
            path = []

        # Debug article data
        try:
            logger.info(f"Processing article type: {type(article)}")
            if isinstance(article, dict):
                logger.info(f"Article keys: {article.keys()}")
            else:
                logger.info(
                    f"Article is not a dictionary: {str(article)[:100]}")
                return []  # Return empty list for non-dict articles
        except Exception as e:
            logger.error(
                f"Error examining article in walk_structure: {str(e)}")
            return []  # Return empty list for problematic articles

        outputs = []
        article_label = f"ARTICLE {article.get('article', '')} {article.get('title', '')}".strip(
        )
        article_text = self.clean_unicode_quotes(article.get("text", ""))
        path = path + [article_label]

        if not article.get("sections"):
            if article_text:
                outputs.append({
                    "label": " > ".join(path),
                    "original_text": article_text.strip(),
                    "combined_text": article_text.strip(),
                    "definition_terms": article_text.strip() if self.is_definition_section(article_text) else None
                })
            return outputs

        for section in article["sections"]:
            section_label = f"Section {section.get('section', '')} {section.get('title', '')}".strip(
            )
            section_text = self.clean_unicode_quotes(section.get("text", ""))
            path_section = path + [section_label]

            if section.get("clauses"):
                for clause in section["clauses"]:
                    clause_label = f"Clause {clause.get('clause', '')} {clause.get('title', '')}".strip(
                    )
                    clause_text = self.clean_unicode_quotes(
                        clause.get("text", ""))
                    path_clause = path_section + [clause_label]

                    subclauses = clause.get(
                        "subclause") or clause.get("subclauses") or []
                    for sub in subclauses:
                        if isinstance(sub, dict):
                            sub_label = f"Subclause {sub.get('subclause', '')}".strip(
                            )
                            path_sub = path_clause + [sub_label]
                            sub_text = self.clean_unicode_quotes(
                                sub.get("text", ""))
                            outputs.append({
                                "label": " > ".join(path_sub),
                                "original_text": sub_text.strip(),
                                "combined_text": sub_text.strip(),
                                "definition_terms": sub_text.strip() if self.is_definition_section(sub_text) else None
                            })
                        elif isinstance(sub, str):
                            sub_text = self.clean_unicode_quotes(sub)
                            outputs.append({
                                "label": " > ".join(path_clause + ["Subclause"]),
                                "original_text": sub_text.strip(),
                                "combined_text": sub_text.strip(),
                                "definition_terms": sub_text.strip() if self.is_definition_section(sub_text) else None
                            })

                    if not subclauses and clause_text:
                        outputs.append({
                            "label": " > ".join(path_clause),
                            "original_text": clause_text.strip(),
                            "combined_text": clause_text.strip(),
                            "definition_terms": clause_text.strip() if self.is_definition_section(clause_text) else None
                        })

            elif section_text:
                outputs.append({
                    "label": " > ".join(path_section),
                    "original_text": section_text.strip(),
                    "combined_text": section_text.strip(),
                    "definition_terms": section_text.strip() if self.is_definition_section(section_text) else None
                })

        return outputs

    def process(self):
        """
        Process a JSON file from a URL and save the flattened result to S3
        """
        try:
            # Download the file from URL
            logger.info(f"Attempting to download from URL: {self.file_url}")
            file_data = self.s3_service.download_from_url(self.file_url)
            logger.info(f"Downloaded data type: {type(file_data)}")
            logger.info(f"File data content sample: {str(file_data)[:200]}")

            # Extract filename from URL for the output key
            filename = self.file_url.split('/')[-1]
            output_key = f"flatten_json/{filename}"

            # Process the data
            flattened_results = []

            # Handle both list and dictionary formats
            if isinstance(file_data, list):
                logger.info("Processing file_data as a list")
                for article in file_data:
                    flattened_results.extend(self.walk_structure(article))
            elif isinstance(file_data, dict):
                logger.info("Processing file_data as a dictionary")
                # If it's a dictionary, we'll try a few common structures

                # Option 1: The dictionary itself is the article
                if any(key in file_data for key in ['article', 'title', 'text', 'sections']):
                    logger.info("Dictionary appears to be a single article")
                    flattened_results.extend(self.walk_structure(file_data))

                # Option 2: Dictionary has a list of articles under a key
                elif 'data' in file_data and isinstance(file_data['data'], list):
                    logger.info("Found articles under 'data' key")
                    for article in file_data['data']:
                        flattened_results.extend(self.walk_structure(article))

                # Option 3: Dictionary has a list of articles under 'articles' key
                elif 'articles' in file_data and isinstance(file_data['articles'], list):
                    logger.info("Found articles under 'articles' key")
                    for article in file_data['articles']:
                        flattened_results.extend(self.walk_structure(article))

                # Option 4: The dictionary has other keys we can try to process
                else:
                    logger.info("Treating dictionary keys as separate entries")
                    for key, value in file_data.items():
                        if isinstance(value, dict):
                            # Try to process this as an article
                            flattened_results.extend(
                                self.walk_structure(value))
                        elif isinstance(value, list):
                            # If the value is a list, process each item
                            for item in value:
                                if isinstance(item, dict):
                                    flattened_results.extend(
                                        self.walk_structure(item))
            else:
                logger.error(
                    f"Unexpected file_data format. Expected list or dict, got {type(file_data)}")
                raise Exception(
                    f"Unexpected data format from {self.file_url}. Expected JSON array or object.")

            # Log the results
            if not flattened_results:
                logger.warning(
                    f"No structured content was extracted from {self.file_url}")
            else:
                logger.info(
                    f"Successfully extracted {len(flattened_results)} content elements")

            # Calculate statistics
            self.total_clauses = len(flattened_results)
            self.total_definitions = sum(
                1 for item in flattened_results if item.get("definition_terms"))

            # Save to S3 with the same filename
            s3_url = self.s3_service.upload_json(flattened_results, output_key)

            return {
                "output_key": output_key,
                "flattened_json_url": s3_url
            }

        except Exception as e:
            logger.error(f"Error in FlattenProcessor.process: {str(e)}")
            raise Exception(f"Error processing file: {str(e)}")


class ChatWithAIService:
    """Service to handle AI chat interactions using OpenAI and Pinecone vectors"""

    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        # Initialize embedding service for vector retrieval
        self.embedding_service = EmbeddingService()
        logger.info("ChatWithAIService initialized")

    def chat(self, query, deal_id=None, message_history=None, top_k=5, temperature=0.7):
        """
        Process a chat query with RAG (Retrieval Augmented Generation)

        Args:
            query (str): The user's question
            deal_id (str, optional): Filter results to a specific deal ID
            message_history (list, optional): Previous chat messages
            top_k (int, optional): Number of relevant chunks to retrieve. Defaults to 5.
            temperature (float, optional): Temperature for OpenAI generation. Defaults to 0.7.

        Returns:
            dict: Chat response with answer and context
        """
        try:
            # Initialize message history if not provided
            if message_history is None:
                message_history = []

            logger.info(f"Processing chat query: {query[:100]}...")

            # Search for relevant chunks in Pinecone
            search_results = None
            context_chunks = []

            if query:
                # Retrieve relevant vectors from Pinecone
                search_results = self.embedding_service.search(
                    query_text=query,
                    deal_id=deal_id,
                    top_k=top_k
                )

                # Extract the relevant chunks for context
                if search_results and search_results.get("results"):
                    for result in search_results["results"]:
                        context_chunks.append({
                            "content": result.get("combined_text", ""),
                            "label": result.get("label", ""),
                            "category": result.get("category_name", ""),
                            "score": result.get("score", 0)
                        })
                        # context_chunks.append(result)
                        print(f"Context chunk: {result}")

            print(f"Context chunks: {context_chunks}")
            # Prepare the system message with context
            system_message = self._prepare_system_message(context_chunks)

            # Prepare the chat messages
            messages = [
                {"role": "system", "content": system_message}
            ]

            # Add message history
            for msg in message_history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })

            # Add the current user query
            messages.append({"role": "user", "content": query})

            # Call OpenAI API
            logger.info(f"Calling OpenAI API with {len(messages)} messages")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=temperature,
                max_tokens=1000
            )

            # Extract the assistant's response
            answer = response.choices[0].message.content
            print(f"Answer: {answer}")
            logger.info(f"Generated response of length: {len(answer)}")

            # Return response with context
            return {
                "answer": answer,
                "context": context_chunks,
                "deal_id": deal_id,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }

        except Exception as e:
            logger.error(f"Error in chat service: {str(e)}")
            logger.error(traceback.format_exc())

            raise Exception(f"Failed to process chat: {str(e)}")

    def _prepare_system_message(self, context_chunks):
        """
        Prepare the system message with context for the AI

        Args:
            context_chunks (list): List of context chunks from vector search

        Returns:
            str: Formatted system message
        """
        system_template = """You are an AI assistant that helps users understand legal documents and contracts.
        
The following are relevant sections from the document to help you answer the user's question:

{context}

Based on the above information, please provide a detailed and accurate response to the user's question.
If the provided context doesn't contain enough information to answer confidently, acknowledge this limitation.
Always cite specific sections when referring to the document content.
"""

        # Format the context sections
        context_text = ""
        for i, chunk in enumerate(context_chunks):
            category_info = f" [Category: {chunk.get('category', '')}]" if chunk.get(
                'category') else ""
            context_text += f"[{i+1}] {chunk.get('label', 'Section')}{category_info}: {chunk.get('content', '')}\n\n"

        return system_template.format(context=context_text)


class SummaryGenerationService:
    """Service to generate summaries of legal documents using OpenAI and Pinecone vectors"""

    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        # Initialize embedding service for vector retrieval
        self.embedding_service = EmbeddingService()
        logger.info("SummaryGenerationService initialized")

    def generate_summary(self, deal_id, temperature=0.7):
        """
        Generate a document summary with static tags and questions

        Args:
            deal_id (str): The deal ID to summarize
            temperature (float, optional): Temperature for OpenAI generation. Defaults to 0.7.

        Returns:
            dict: Summary response with summary text and context
        """
        try:
            # Static tags and questions to query the document
            static_queries = [
                "What is the purpose of this agreement?",
                "Who are the main parties involved in this agreement?",
                "What are the key obligations of each party?",
                "What are the important dates and deadlines in this agreement?",
                "What are the termination conditions?",
                "What are the key financial terms?"
            ]

            logger.info(f"Generating summary for deal ID: {deal_id}")

            # Initialize context for all queries
            all_context_chunks = []

            # Search for relevant chunks for each query
            for query in static_queries:
                # Create embedding for the query and search
                search_results = self.embedding_service.search(
                    query_text=query,
                    deal_id=deal_id,
                    top_k=3  # Get top 3 chunks per query
                )

                # Extract the relevant chunks for context
                if search_results and search_results.get("results"):
                    for result in search_results["results"]:
                        # Check if this chunk is already in the context to avoid duplicates
                        is_duplicate = False
                        for existing_chunk in all_context_chunks:
                            if existing_chunk.get("content") == result.get("combined_text", ""):
                                is_duplicate = True
                                break

                        if not is_duplicate:
                            all_context_chunks.append({
                                "content": result.get("combined_text", ""),
                                "label": result.get("label", ""),
                                "category": result.get("category_name", ""),
                                "score": result.get("score", 0),
                                "query": query
                            })

            # Prepare the system message with context
            system_message = self._prepare_summary_prompt(all_context_chunks)

            # Prepare the chat messages
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": "Generate a comprehensive summary of this document."}
            ]

            # Call OpenAI API
            logger.info(f"Calling OpenAI API for summary generation")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=temperature,
                max_tokens=1500
            )

            # Extract the assistant's response
            summary = response.choices[0].message.content
            logger.info(f"Generated summary of length: {len(summary)}")

            # Return response with context
            return {
                "summary": summary,
                "context": all_context_chunks,
                "deal_id": deal_id,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }

        except Exception as e:
            logger.error(f"Error in summary generation service: {str(e)}")
            logger.error(traceback.format_exc())
            raise Exception(f"Failed to generate summary: {str(e)}")

    def _prepare_summary_prompt(self, context_chunks):
        """
        Prepare the system message with context for the AI to generate a summary

        Args:
            context_chunks (list): List of context chunks from vector search

        Returns:
            str: Formatted system message
        """
        system_template = """You are an AI assistant that helps users understand legal documents and contracts by providing comprehensive summaries.
        
The following are relevant sections from the document that you should use to create your summary:

{context}

Based on the above information, please generate a comprehensive summary of the document that includes:
1. Purpose of the agreement
2. Main parties involved
3. Key obligations of each party
4. Important dates and deadlines
5. Termination conditions
6. Key financial terms

Your summary should be well-structured with clear headings for each section. If the provided context doesn't contain enough information for any section, acknowledge this limitation.
Always cite specific sections when referring to the document content.
"""

        # Format the context sections
        context_text = ""
        for i, chunk in enumerate(context_chunks):
            category_info = f" [Category: {chunk.get('category', '')}]" if chunk.get(
                'category') else ""
            context_text += f"[{i+1}] {chunk.get('label', 'Section')}{category_info} (Related to: {chunk.get('query', 'General')}): {chunk.get('content', '')}\n\n"

        return system_template.format(context=context_text)
