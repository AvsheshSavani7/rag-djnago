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
import json
import re
from .utils import get_impherior_prompt, get_experior_prompt
import pytz
from .models import ProcessingJob
from mongoengine.errors import DoesNotExist
from .summary_utils.clause_config_util import ClauseConfigUtil

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
                f"Processing document with file_url={file_url}, deal_id={deal_id}, embed_data={embed_data}"
            )

            # Find the existing job by id (convert string to ObjectId)
            try:
                object_id = ObjectId(deal_id)
                job = ProcessingJob.objects.get(id=object_id)
                logger.info(f"Found job in database: {job}")
            except DoesNotExist:
                error_msg = f"No processing job found for deal_id {deal_id}"
                logger.error(error_msg)
                return {"error": error_msg, "status": "failed"}
            except Exception as e:
                error_msg = f"Invalid deal_id format: {str(e)}"
                logger.error(error_msg)
                return {"error": error_msg, "status": "failed"}

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
                return {"error": error_msg, "status": "failed"}

            # Update the job with results
            job.flattened_json_url = result.get("flattened_json_url")

            if job.schema_results is not None and not isinstance(
                job.schema_results, dict
            ):
                logger.warning(
                    f"⚠️ Invalid schema_results type: {type(job.schema_results)}. Resetting to empty dict."
                )
            else:
                job.save()

            # Start embedding process in background if requested
            if embed_data:
                # Start embedding task in the executor
                self.executor.submit(
                    self._process_embeddings, str(
                        job.id), job.flattened_json_url
                )
                logger.info(
                    f"Submitted embedding task for job {job.id} to executor")

                # Return response with embedding status
                return {
                    "deal_id": str(job.id),
                    "flattened_json_url": job.flattened_json_url,
                    "embedding_status": "PROCESSING",
                    "message": "File processed successfully. Embeddings are being generated in the background.",
                    "status": "success",
                }

            # Return response without embedding
            return {
                "deal_id": str(job.id),
                "flattened_json_url": job.flattened_json_url,
                "status": "success",
            }

        except Exception as e:
            # Log the error
            logger.error(f"Error processing file: {str(e)}")

            # Update the job with error if it exists
            if "job" in locals():
                job.error_message = str(e)
                job.save()

            # Return error response
            return {
                "error": str(e),
                "deal_id": deal_id if "job" not in locals() else str(job.id),
                "status": "failed",
            }

    def _process_embeddings(self, job_id, flattened_json_url):
        """
        Background task to process embeddings

        Args:
            job_id (str): ID of the job
            flattened_json_url (str): URL to the flattened JSON file
        """
        logger.info(
            f"Starting embedding process for job {job_id} with URL: {flattened_json_url}"
        )
        try:
            # Get the job
            object_id = ObjectId(job_id)
            job = ProcessingJob.objects.get(id=object_id)
            logger.info(f"Found job in database: {job}")

            # Update job status to processing
            job.update_embedding_status("PROCESSING")
            logger.info("Updated job status to PROCESSING")

            # Download flattened JSON
            s3_service = S3Service()
            logger.info(
                f"Downloading flattened JSON from URL: {flattened_json_url}")
            chunks = s3_service.download_from_url(flattened_json_url)
            logger.info(f"Downloaded {len(chunks)} chunks")

            # Process embeddings
            # logger.info("Initializing embedding service")
            # embedding_service = EmbeddingService()
            # logger.info(
            #     f"Starting embedding creation for {len(chunks)} chunks")
            # result = embedding_service.process_chunks(chunks, str(job_id))
            # logger.info(f"Embedding completed with result: {result}")

            # Create an instance of the class
            schema_search = SchemaCategorySearch()

            category_results = schema_search.search_all_schema_categories(
                deal_id=str(job_id)
            )
            # Update job status to completed
            # job.save_json_to_db(category_results)
            job.upsert_json_to_db(category_results)
            job.update_embedding_status("COMPLETED")
            logger.info(f"Updated job status to COMPLETED")

        except Exception as e:
            logger.error(f"Error processing embeddings: {str(e)}")

            # Update job status to failed
            try:
                object_id = ObjectId(job_id)
                job = ProcessingJob.objects.get(id=object_id)
                job.update_embedding_status("FAILED", str(e))
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
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        self.bucket = os.environ.get("AWS_S3_BUCKET")
        print("env", os.environ.get("AWS_ACCESS_KEY_ID"))

    def upload_json(self, data, key):
        """Upload JSON data to S3 and return the full URL"""
        json_bytes = json.dumps(data, indent=2).encode("utf-8")
        self.s3.put_object(
            Bucket=self.bucket, Key=key, Body=json_bytes, ContentType="application/json"
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
                f"Response content type: {response.headers.get('Content-Type', 'unknown')}"
            )

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
        return [
            obj["Key"]
            for obj in response.get("Contents", [])
            if obj["Key"].endswith(".json")
        ]


class EmbeddingService:
    """Service to create embeddings using OpenAI and store them in Pinecone"""

    def __init__(self):
        print("Initializing EmbeddingService")
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"))
        self.MAX_METADATA_SIZE = 40960
        print(
            f"OpenAI API key set: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No'}"
        )

        # Initialize Pinecone
        pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        print(
            f"Pinecone API key set: {'Yes' if os.environ.get('PINECONE_API_KEY') else 'No'}"
        )

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
                    dimension=3072,  # Dimensions for text-embedding-3-small
                    metric="cosine",
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
                input=text, model="text-embedding-3-large"
            )
            print("Embedding created successfully")
            return response.data[0].embedding
        except Exception as e:
            print(f"Error creating embedding: {str(e)}")
            raise

    def trim_metadata(self, metadata):
        # Try serializing first
        meta_bytes = json.dumps(metadata).encode("utf-8")
        if len(meta_bytes) <= self.MAX_METADATA_SIZE:
            return metadata  # ✅ Already valid

        # Sort keys by importance (edit this order as needed)
        priority_keys = [
            "categories",
            "chunk_index",
            "clause_summary",
            "combined_text",
            "deal_id",
            "deal_name",
            "label",
            "original_text",
        ]

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
                # try:
                #     print(f"Enhancing metadata for chunk {i+1}")
                #     enhanced_chunk = self.metadata_service.enhance_chunk_metadata(
                #         chunk)
                # category_name = enhanced_chunk.get("categories", "")
                # print(
                #     f"Category determined for chunk {i+1}: {category_name}")
                # except Exception as e:
                #     print(
                #         f"Error enhancing metadata for chunk {i+1}: {str(e)}")
                # Continue with original chunk if enhancement fails
                # enhanced_chunk = chunk
                # enhanced_chunk["categories"] = []

                # Below is the chunk without metadata enhancement (categories, clause_summary)
                enhanced_chunk = chunk

                # Create embedding
                try:
                    embedding = self.create_embedding(text)
                    print(
                        f"Created embedding for chunk {i+1} - vector size: {len(embedding)}"
                    )
                except Exception as e:
                    print(
                        f"Error creating embedding for chunk {i+1}: {str(e)}")
                    raise Exception(
                        f"Failed to create embedding for chunk {i+1}: {str(e)}"
                    )

                # Start with required metadata fields
                metadata = {
                    "deal_id": str(deal_id),
                    "deal_name": enhanced_chunk.get("deal_name", "") or "",
                    "label": enhanced_chunk.get("label", "") or "",
                    "definition_terms": (
                        ""
                        if enhanced_chunk.get("definition_terms") is None
                        else str(enhanced_chunk.get("definition_terms", ""))
                    ),
                    "original_text": enhanced_chunk.get("original_text", "") or "",
                    "combined_text": enhanced_chunk.get("combined_text", "") or "",
                    # "categories": enhanced_chunk.get("categories", "") or "",
                    "chunk_index": i,
                    # "clause_summary": enhanced_chunk.get("clause_summary", "") or "",
                }

                # Upsert single vector to Pinecone
                metadata = self.trim_metadata(metadata)
                try:
                    self.index.upsert(
                        vectors=[
                            {"id": chunk_id, "values": embedding,
                                "metadata": metadata}
                        ]
                    )
                    print(
                        f"Uploaded chunk {i+1} to Pinecone with ID: {chunk_id}")
                except Exception as e:
                    print(f"Error uploading chunk {i+1} to Pinecone: {str(e)}")
                    print(f"Problematic metadata: {metadata}")
                    raise Exception(
                        f"Failed to upload chunk {i+1} to Pinecone: {str(e)}"
                    )

                processed_chunks += 1

                # Add a small delay to avoid rate limits
                time.sleep(0.2)

            print(
                f"Successfully processed {processed_chunks} out of {total_chunks} chunks"
            )
            return {
                "status": "success",
                "chunks_processed": processed_chunks,
                "index_name": self.index_name,
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
                filter_dict = {
                    "deal_id": str(deal_id),
                    #  "test_tag": {"$in": ["a"]}
                }

                print(f"Filtering search to deal ID: {deal_id}")

            # Search in Pinecone
            print(f"Searching Pinecone index: {self.index_name}")
            search_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict,
            )
            print(f"Search response: {search_response}")
            with open("search_response_matches.json", "w") as f:
                json.dump(search_response.matches, f, default=str, indent=4)

            # Format results
            results = []
            for match in search_response.matches:
                results.append(
                    {
                        "id": match.id,
                        "score": match.score,
                        "deal_id": match.metadata.get("deal_id"),
                        "label": match.metadata.get("label"),
                        "combined_text": match.metadata.get("combined_text"),
                        "definition_terms": match.metadata.get("definition_terms"),
                    }
                )

            print(f"Found {len(results)} matching results")
            return {"query": query_text, "results": results, "total": len(results)}

        except Exception as e:
            print(f"Error searching for text: {str(e)}")
            raise Exception(f"Failed to search for text: {str(e)}")


class MetadataEnhancementService:
    """Service to enhance document metadata using GPT"""

    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"))
        # Schema URL
        self.schema_url = "https://mna-docs.s3.eu-north-1.amazonaws.com/clauses_category_template/Clauses_Category_Template.json"
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
            categories = [item["category_name"] for item in schema]
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
                [f"{i+1}. {category}" for i, category in enumerate(categories)]
            )

            prompt = f"""You are a legal AI assistant helping classify a merger-related clause in a transaction agreement.

Your task is to:
1. Carefully read the clause below.
2. Identify **all directly relevant categories** from the list that reflect:
   - The clause's main legal function.
   - Any embedded legal obligations or mechanisms, even if secondary (e.g., guarantees, assignments, waivers).
3. Do NOT omit a category simply because it is not the dominant topic. If a clause includes enforceable language tied to a listed category (like 'guarantee', 'waiver', 'termination'), include that category as well.
4. Provide a concise summary of the clause (1–2 sentences) in plain English.

Clause:
'''{text}'''

Available Categories:
{category_list}

Return a JSON object in this exact format:
{{
  \"categories\": [list of relevant categories],
  \"clause_summary\": \"concise plain-language summary of the clause\"
}}
"""

            logger.info("Calling GPT to determine category")
            # Call GPT
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a legal contract classifier assistant. Your task is to assign text to the most appropriate category.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=750,
            )

            # Extract the determined category
            category = response.choices[0].message.content.strip()
            logger.info(f"GPT determined category: {category}")

            # Clean markdown code block like ```json ... ```  # Clean markdown code block like ```json ... ```
            markdown_match = re.search(
                r"```(?:json)?\s*([\s\S]+?)\s*```", category)
            if markdown_match:
                category = markdown_match.group(1).strip()
            # Validate if the returned category is in our list
            try:
                parsed = json.loads(category)
                if isinstance(parsed, dict):
                    return parsed.get("categories", []), parsed.get(
                        "clause_summary", ""
                    )
            except json.JSONDecodeError:
                logger.warning(f"GPT response could not be parsed: {category}")
            return [], ""

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
                    f"Field: {field.get('name')}\nDescription: {field.get('description')}\nType: {field.get('data_type')}\nExample: {field.get('example_value')}"
                )

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
                f"Calling GPT to extract structured metadata for category: {category_name}"
            )
            # Call GPT
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a legal metadata extraction assistant. Your task is to extract structured information from contract text.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )

            # Extract the structured metadata
            metadata_text = response.choices[0].message.content.strip()
            logger.info(f"GPT extracted metadata: {metadata_text[:200]}...")

            # Try to parse as JSON
            try:
                # Find json content if it's not directly formatted as json
                if not metadata_text.startswith("{"):
                    import re

                    json_match = re.search(
                        r"```json\s*([\s\S]*?)\s*```", metadata_text)
                    if json_match:
                        metadata_text = json_match.group(1)
                    else:
                        # Try to find any JSON-like structure
                        json_match = re.search(r"({[\s\S]*})", metadata_text)
                        if json_match:
                            metadata_text = json_match.group(1)

                structured_metadata = json.loads(metadata_text)
                logger.info(
                    f"Successfully parsed structured metadata with {structured_metadata} fields"
                )
                return structured_metadata
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse metadata as JSON: {str(e)}")
                logger.error(f"Problematic content: {metadata_text}")

                # Attempt to create a basic structured response
                return {
                    "clause_summary": "Failed to extract structured metadata",
                    "clause_tags_llm": [],
                    "error": "Failed to parse JSON from GPT response",
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
            # category = self.determine_category(text)

            # Add the category to the chunk
            # chunk["categories"] = category[0]
            # chunk["category_tags"] = []
            # chunk["clause_summary"] = category[1]

            # logger.info(f"Enhanced chunk with category: {category}")

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
        self.deal_name = self.extract_deal_name(file_url)

    def clean_unicode_quotes(self, text):
        if not text:
            return ""
        return (
            text.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u2013", "-")
            .replace("\u2014", "-")
        )

    def is_definition_section(self, text):
        return bool(text and ":" in text and "means" in text)

    @staticmethod
    def extract_deal_name(url):
        """
        Extract and format deal name from S3 URL

        Args:
            url (str): The S3 URL containing the deal name

        Returns:
            str: The extracted and formatted deal name
        """
        try:
            # Extract the filename from the URL
            filename = url.split("/")[-1]

            # Remove the file extension (.json)
            if filename.endswith(".json"):
                filename = filename[:-5]  # Remove .json

            # Remove date part if present (format: YYYY-MM-DD)
            deal_name = re.sub(r"_\d{4}-\d{2}-\d{2}$", "", filename)

            # Format the deal name: replace underscores with spaces and capitalize words
            formatted_name = " ".join(
                word.capitalize() for word in deal_name.split("_")
            )

            return formatted_name
        except Exception as e:
            print(f"Error extracting deal name: {str(e)}")
            return ""

    def walk_structure(self, article, path=None):
        if path is None:
            path = []

        outputs = []
        if article.get("article") == "Definitions":

            outputs = []

            for item in article.get("definitions"):
                # if "Material Adverse Effect" in item.get('term', ""):
                outputs.append(
                    {
                        "label": f"Definition > {item['term']}",
                        "original_text": f"{item['term']} {item['definition']}",
                        "combined_text": f"{item['term']} {item['definition']}",
                        "deal_name": self.deal_name,
                    }
                )
            return outputs

        else:
            print(f"Article: else start")
            article_label = f"ARTICLE {article.get('article', '')} {article.get('title', '')}".strip(
            )
            path = path + [article_label]

            article_text = self.clean_unicode_quotes(
                article.get("text", "")).strip()

            if not article.get("sections"):
                if article_text:
                    outputs.append(
                        {
                            "label": " > ".join(path),
                            "original_text": article_text,
                            "combined_text": article_text,
                            "deal_name": self.deal_name,
                        }
                    )
                return outputs

            for section in article["sections"]:

                if (
                    "definition" in section.get("title", "").lower()
                    and "definitions" in section
                    and section.get("definitions") is not None
                ):
                    section_label = f"Section {section.get('section', '')} {section.get('title', '')}".strip(
                    )
                    path_section = path + [section_label]

                    for item in section.get("definitions"):
                        # if "Material Adverse Effect" in item.get('term', ""):
                        outputs.append(
                            {
                                "label": f"{' > '.join(path_section)} > {item['term']}",
                                "original_text": f"{item['term']} {item['definition']}",
                                "combined_text": f"{item['term']} {item['definition']}",
                                "deal_name": self.deal_name,
                            }
                        )

                else:
                    section_label = f"Section {section.get('section', '')} {section.get('title', '')}".strip(
                    )
                    # section_text = self.clean_unicode_quotes(section.get("text", "")).strip()
                    section_text = (
                        " | ".join(
                            [
                                f'"{item["term"]}" : {item["definition"]}'
                                for item in section.get("definitions", [])
                            ]
                        )
                        if "definition" in section.get("title", "").lower()
                        else self.clean_unicode_quotes(section.get("text", ""))
                    )
                    path_section = path + [section_label]

                    # Combine article-level text and section-level text
                    combined_text = (
                        f"{article_text}\n\n{section_text}"
                        if article_text
                        else section_text
                    )
                    outputs.append(
                        {
                            "label": " > ".join(path_section),
                            "original_text": section_text,
                            "combined_text": combined_text,
                            "deal_name": self.deal_name,
                        }
                    )

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
            filename = self.file_url.split("/")[-1]
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
                if any(
                    key in file_data for key in ["article", "title", "text", "sections"]
                ):
                    logger.info("Dictionary appears to be a single article")
                    flattened_results.extend(self.walk_structure(file_data))

                # Option 2: Dictionary has a list of articles under a key
                elif "data" in file_data and isinstance(file_data["data"], list):
                    logger.info("Found articles under 'data' key")
                    for article in file_data["data"]:
                        flattened_results.extend(self.walk_structure(article))

                # Option 3: Dictionary has a list of articles under 'articles' key
                elif "articles" in file_data and isinstance(
                    file_data["articles"], list
                ):
                    logger.info("Found articles under 'articles' key")
                    for article in file_data["articles"]:
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
                    f"Unexpected file_data format. Expected list or dict, got {type(file_data)}"
                )
                raise Exception(
                    f"Unexpected data format from {self.file_url}. Expected JSON array or object."
                )

            # Log the results
            if not flattened_results:
                logger.warning(
                    f"No structured content was extracted from {self.file_url}"
                )
            else:
                logger.info(
                    f"Successfully extracted {len(flattened_results)} content elements"
                )

            # Calculate statistics
            self.total_clauses = len(flattened_results)
            self.total_definitions = sum(
                1 for item in flattened_results if item.get("definition_terms")
            )

            # Save to S3 with the same filename
            s3_url = self.s3_service.upload_json(flattened_results, output_key)

            return {"output_key": output_key, "flattened_json_url": s3_url}

        except Exception as e:
            logger.error(f"Error in FlattenProcessor.process: {str(e)}")
            raise Exception(f"Error processing file: {str(e)}")


class ChatWithAIService:
    """Service to handle AI chat interactions using OpenAI and Pinecone vectors"""

    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"))
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
                    query_text=query, deal_id=deal_id, top_k=top_k
                )

                # Extract the relevant chunks for context
                if search_results and search_results.get("results"):
                    for result in search_results["results"]:
                        context_chunks.append(
                            {
                                "content": result.get("combined_text", ""),
                                "label": result.get("label", ""),
                                "category": result.get("category_name", ""),
                                "score": result.get("score", 0),
                            }
                        )
                        # context_chunks.append(result)
                        print(f"Context chunk: {result}")

            print(f"Context chunks: {context_chunks}")
            # Prepare the system message with context
            system_message = self._prepare_system_message(context_chunks)

            # Prepare the chat messages
            messages = [{"role": "system", "content": system_message}]

            # Add message history
            for msg in message_history:
                messages.append(
                    {"role": msg.get("role", "user"),
                     "content": msg.get("content", "")}
                )

            # Add the current user query
            messages.append({"role": "user", "content": query})

            # Call OpenAI API
            logger.info(f"Calling OpenAI API with {len(messages)} messages")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=temperature,
                max_tokens=1000,
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
                    "total_tokens": response.usage.total_tokens,
                },
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
            category_info = (
                f" [Category: {chunk.get('category', '')}]"
                if chunk.get("category")
                else ""
            )
            context_text += f"[{i+1}] {chunk.get('label', 'Section')}{category_info}: {chunk.get('content', '')}\n\n"

        return system_template.format(context=context_text)


class SummaryGenerationService:
    """Service to generate summaries of legal documents using OpenAI and Pinecone vectors"""

    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"))
        # Initialize embedding service for vector retrieval
        self.embedding_service = EmbeddingService()
        logger.info("SummaryGenerationService initialized")

    def generate_summary_v2(self, deal_id, temperature=0.7):
        """
        Generate document summaries based on schema results for the given deal_id
        """
        try:
            logger.info(f"Generating summary for deal ID: {deal_id}")
            # Find the job by deal_id
            try:
                object_id = ObjectId(deal_id)
                job = ProcessingJob.objects.get(id=object_id)
                # print the job
                print(f"Job: {job}")
                logger.info(f"Found job in database: {job}")

                schema_results = job.schema_results

                if not schema_results:
                    logger.info(
                        "No schema results found, returning empty list")
                    return []

                # Parse JSON string to dictionary
                if isinstance(schema_results, str):
                    schema_results = json.loads(schema_results)

                clause_util = ClauseConfigUtil()
                final_summary_sections = clause_util.get_organized_sections_for_summary(
                    schema_results)
                return final_summary_sections

            except DoesNotExist:
                logger.error(f"No processing job found for deal_id {deal_id}")
                return []
        except Exception as e:
            logger.error(f"Error in summary generation service: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def generate_summary(self, deal_id, temperature=0.7):
        """
        Generate document summaries based on schema results for the given deal_id

        Args:
            deal_id (str): The deal ID to summarize
            temperature (float, optional): Temperature for OpenAI generation. Defaults to 0.7.

        Returns:
            list: Array of summaries generated from schema results
        """
        try:
            logger.info(f"Generating summary for deal ID: {deal_id}")

            # Find the job by deal_id
            try:
                object_id = ObjectId(deal_id)
                job = ProcessingJob.objects.get(_id=object_id)
                logger.info(f"Found job in database: {job}")
            except ProcessingJob.DoesNotExist:
                logger.error(f"No processing job found for deal_id {deal_id}")
                return []
            except Exception as e:
                logger.error(f"Error finding job: {str(e)}")
                return []

            # Check if schema_results exist
            schema_results = job.schema_results
            if not schema_results:
                logger.warning(
                    f"No schema results found for deal_id {deal_id}")
                return []

            logger.info(f"Retrieved schema results for deal_id {deal_id}")

            # Array to store generated summaries
            summaries = []
            sections = [
                "Complex Consideration",
                "Go-Shop Terms",
                "Unusual Closing Conditions",
                "Confidentiality Agreement Sign Date",
                "Outside Date + Extensions + Reasons",
                "Regulatory Best Efforts",
                "Termination and Reverse Termination Fees + Triggers",
                "Standard or Unusual",
                "Merger_Agreement_Details",
                "Complete_Effects_on_Capital_Stock",
                "R_W_Parent",
                "Antitrust_Commitment",
                "Breach_Monitoring_and_Ongoing_Operations",
                "Timeline",
                "Acquirer",
                "Guarantor",
                "Guarantee",
                "Best_Efforts",
                "Closing",
                "Company_Material_Adverse_Change",
                "Ordinary_Course",
                "No_Solicitation",
                "Dividends",
                "Board_Approval",
                "Proxy_Statement",
                "Shareholder_Approval",
                "Voting_Agreement",
                "Confidentiality_Agreement",
                "Clean_Room_Agreement",
                "Financing",
                "Regulatory_Approvals",
                "Regulatory_Obligations_Timing",
                "Out_Date",
                "Other",
                "Termination_Rights_and_Causes",
                "Specific_Performance",
                "Law_and_Jurisdiction",
            ]

            # Create a ThreadPoolExecutor for parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=18) as executor:
                # Dictionary to store futures and their corresponding section names
                future_to_section = {}

                # Submit each section to the executor
                for section_name in sections:
                    if section_name != "Complex Consideration":
                        continue
                    logger.info(
                        f"Submitting section for processing: {section_name}")
                    future = executor.submit(
                        self._process_section, section_name, schema_results, temperature
                    )
                    future_to_section[future] = section_name

                # Create an ordered results dictionary to maintain section order
                ordered_results = []

                # Collect results in the original section order
                for section_name in sections:
                    # Find the future corresponding to this section
                    for future, name in future_to_section.items():
                        if name == section_name:
                            try:
                                result = future.result()
                                if result:
                                    ordered_results.append(result)
                                    logger.info(
                                        f"Added result for section: {section_name}"
                                    )
                                else:
                                    logger.info(
                                        f"No result generated for section: {section_name}"
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Error processing section {section_name}: {str(e)}"
                                )
                                logger.error(traceback.format_exc())
                            break

                # Use the ordered results instead of summaries collected out of order
                summaries = ordered_results

            # Get current time in UTC
            utc_now = datetime.now(pytz.UTC)

            # Convert to IST
            ist = pytz.timezone("Asia/Kolkata")

            ist_now = utc_now.astimezone(ist)

            # Create timestamp in IST
            timestamp = ist_now.strftime("%d-%m-%y_%I-%M_%p")
            filename = f"summary_results_{timestamp}.json"

            try:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(summaries, f, indent=2, ensure_ascii=False)
                logger.info(f"Summaries saved to file: {filename}")
            except Exception as save_error:
                logger.error(
                    f"Error saving summaries to file: {str(save_error)}")

            return summaries

        except Exception as e:
            logger.error(f"Error in summary generation service: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def _process_section(self, section_name, schema_results, temperature):
        """
        Process a single section for summary generation

        Args:
            section_name (str): The name of the section to process
            schema_results (dict): The schema results
            temperature (float): Temperature for OpenAI generation

        Returns:
            dict: Section summary or None if not applicable
        """
        try:
            logger.info(f"Processing section: {section_name}")
            section_key = section_name.replace(" ", "_")

            # Special case handling for different section types
            if section_name == "Acquirer":
                acquirer_fields = schema_results.get("Acquirer", [])
                result = next(
                    (
                        item["answer"]
                        for item in acquirer_fields
                        if item.get("field_name") == "acquirer_name"
                    ),
                    "Not found",
                )
                if result != "Not found":
                    return {section_name: result}

            elif section_name == "Guarantor":
                guarantor_fields = schema_results.get("Guarantor", [])
                result = next(
                    (
                        item["answer"]
                        for item in guarantor_fields
                        if item.get("field_name") == "guarantor_name"
                    ),
                    "Not found",
                )
                if result != "Not found":
                    return {section_name: result}

            elif section_name == "Specific_Performance":
                specific_performance_fields = schema_results.get(
                    "Specific_Performance", []
                )
                result = next(
                    (
                        item["answer"]
                        for item in specific_performance_fields
                        if item.get("field_name") == "specific_performance_available"
                    ),
                    "Not found",
                )
                if result != "Not found":
                    section_name = section_name.replace("_", " ")
                    return {
                        section_name: (
                            "Available"
                            if result == "true" or result is True
                            else "Not Available"
                        )
                    }

            elif section_name == "Law_and_Jurisdiction":
                law_and_jurisdiction_fields = schema_results.get(
                    "Law_and_Jurisdiction", []
                )
                result = next(
                    (
                        item["answer"]
                        for item in law_and_jurisdiction_fields
                        if item.get("field_name") == "governing_law"
                    ),
                    "Not found",
                )
                if result != "Not found":
                    section_name = section_name.replace("_", " ")
                    return {section_name: result}

            elif section_name == "Confidentiality_Agreement":
                confidentiality_agreement_fields = schema_results.get(
                    "Confidentiality_Agreement", []
                )
                result = next(
                    (
                        item["answer"]
                        for item in confidentiality_agreement_fields
                        if item.get("field_name") == "confidentiality_agreement_date"
                    ),
                    "Not found",
                )
                if result != "Not found":
                    section_name = section_name.replace("_", " ")
                    return {section_name: result}

            elif section_name == "Clean_Room_Agreement":
                section_data = schema_results.get(section_key)
                if section_data:
                    filtered_section_data = [
                        item
                        for item in schema_results.get(section_key, [])
                        if str(item.get("answer")).lower() != "not found"
                        and item.get("field_name")
                        in [
                            "clean_room_agreement_startdate",
                            "clean_room_agreement_enddate",
                        ]
                    ]
                    return self._generate_section_summary(
                        section_name, filtered_section_data, temperature, schema_results
                    )
                else:
                    logger.warning(
                        f"Section key '{section_key}' not found in schema_results."
                    )

            elif section_name == "Complex Consideration":
                logger.info(f"Schema results: {schema_results}")

                cvr_present = False
                if schema_results.get("Complete_Effects_on_Capital_Stock"):
                    for item in schema_results.get(
                        "Complete_Effects_on_Capital_Stock", []
                    ):
                        if (
                            item.get("field_name") == "is_cvr_present"
                            and str(item.get("answer")).lower() == "true"
                        ):
                            cvr_present = True
                            break
                proration_present = False
                if schema_results.get("Complete_Effects_on_Capital_Stock"):
                    for item in schema_results.get(
                        "Complete_Effects_on_Capital_Stock", []
                    ):
                        if (
                            item.get("field_name") == "is_proration_present"
                            and str(item.get("answer")).lower() == "true"
                        ):
                            proration_present = True
                            break

                contingent_payment_present = False
                if schema_results.get("Complete_Effects_on_Capital_Stock"):
                    for item in schema_results.get(
                        "Complete_Effects_on_Capital_Stock", []
                    ):
                        if (
                            item.get(
                                "field_name") == "is_contingent_payment_present"
                            and str(item.get("answer")).lower() == "true"
                        ):
                            contingent_payment_present = True
                            break

                ticking_fee_present = False
                if schema_results.get("Covenants"):
                    for item in schema_results.get("Covenants", []):
                        if (
                            item.get("field_name") == "is_ticking_fee_present"
                            and str(item.get("answer")).lower() == "true"
                        ):
                            ticking_fee_present = True
                            break

                election_present = False
                if schema_results.get("Merger_Agreement_Details"):
                    for item in schema_results.get("Merger_Agreement_Details", []):
                        if (
                            item.get("field_name") == "is_election_present"
                            and str(item.get("answer")).lower() == "true"
                        ):
                            election_present = True
                            break

                sections_to_combine = {
                    "Complete_Effects_on_Capital_Stock": [
                        "forms_of_consideration_used",
                        "conversion_ratios",
                        "proration_calculation_method" if proration_present else None,
                        (
                            "consideration_election_mechanics"
                            if election_present
                            else None
                        ),
                        "cvr_terms_summary" if cvr_present else None,
                        (
                            "contingent_payment_conditions"
                            if contingent_payment_present
                            else None
                        ),
                        "fractional_share_handling",
                        "contingent_payment_amount_or_range",
                        (
                            "contingent_payment_type"
                            if contingent_payment_present
                            else None
                        ),
                    ],
                    "Merger_Agreement_Details": [
                        "consideration_structure_type",
                        "maximum_cash_cap",
                        "maximum_stock_cap",
                        "proration_formula_summary" if proration_present else None,
                        "earnout_cap_or_ceiling",
                        "cvr_trigger_events" if cvr_present else None,
                        "election_deadline" if election_present else None,
                        "adjustment_for_acquirer_dividends",
                    ],
                    "Covenants": (
                        [
                            "ticking_fee_terms",
                            "ticking_fee_start_date",
                            "ticking_fee_rate_or_amount",
                            "ticking_fee_cap_or_maximum",
                            "ticking_fee_trigger_conditions",
                        ]
                        if ticking_fee_present
                        else []
                    ),
                }
                logger.info(f"Sections to combine: {sections_to_combine}")

                combined_data = []
                for section_key, field_names in sections_to_combine.items():
                    new_field_names = []
                    for field in field_names:
                        if field is not None:
                            new_field_names.append(field)
                        else:
                            print(f"Found None value, type: {type(field)}")
                    field_names = new_field_names
                    logger.info(f"Field names {section_key}: {field_names}")

                    section_data = schema_results.get(section_key, [])
                    if section_data:
                        filtered_data = [
                            item
                            for item in section_data
                            if str(item.get("answer")).lower() != "not found"
                            and item.get("field_name") in field_names
                        ]
                        combined_data.extend(filtered_data)

                special_fields = [
                    "proration_calculation_method",
                    "consideration_election_mechanics",
                    "cvr_terms_summary",
                    "contingent_payment_conditions",
                ]
                has_valid_special_field = False
                for item in combined_data:
                    if (
                        item.get("field_name") in special_fields
                        and str(item.get("answer")).lower() != "not found"
                    ):
                        has_valid_special_field = True
                        break

                if combined_data:

                    if has_valid_special_field:
                        return self._generate_section_summary(
                            section_name, combined_data, temperature, schema_results
                        )
                    else:
                        return {section_name: "No Complex Consideration found."}
                else:
                    logger.warning(
                        f"No relevant data found for Complex Consideration section."
                    )

            elif section_name == "Go-Shop Terms":
                sections_to_combine = {
                    "No_Solicitation": [
                        "go_shop_period_included",
                        "go_shop_duration_and_conditions",
                        "termination_right_for_superior_proposal",
                    ],
                    "Timeline": ["marketing_period_end_date"],
                    "Best_Efforts": ["match_right_period"],
                    "Covenants": ["go_shop_fee_discount"],
                }

                combined_data = []
                for section_key, field_names in sections_to_combine.items():
                    section_data = schema_results.get(section_key, [])
                    if section_data:
                        filtered_data = [
                            item
                            for item in section_data
                            if str(item.get("answer")).lower() != "not found"
                            and item.get("field_name") in field_names
                        ]
                        combined_data.extend(filtered_data)

                special_fields = [
                    "go_shop_period_included",
                    "go_shop_duration_and_conditions",
                    "go_shop_fee_discount",
                ]
                has_valid_special_field = False
                for item in combined_data:
                    if (
                        item.get("field_name") in special_fields
                        and str(item.get("answer")).lower() != "not found"
                    ):
                        has_valid_special_field = True
                        break

                go_shop_period_included = False
                for item in combined_data:
                    if item.get("field_name") == "go_shop_period_included":
                        answer = str(item.get("answer"))
                        go_shop_period_included = (
                            True if answer.lower() == "true" else False
                        )
                        break

                if combined_data and go_shop_period_included:
                    if has_valid_special_field:
                        return self._generate_section_summary(
                            section_name, combined_data, temperature, schema_results
                        )
                    else:
                        return {section_name: "No Go-Shop Terms found."}
                else:
                    logger.warning(
                        f"No relevant data found for Go-Shop Terms section.")

            elif section_name == "Unusual Closing Conditions":
                sections_to_combine = {
                    "Conditions_to_Closing": [
                        "unusual_closing_conditions_present",
                        "unusual_condition_summary",
                        "buyer_no_target_mae_condition",
                        "buyer_officer_certificate_condition",
                        "buyer_target_compliance_with_covenants_condition",
                        "target_officer_certificate_condition",
                        "target_parent_representations_and_warranties_true_condition",
                        "target_no_parent_mae_condition",
                        "other_mutual_conditions_condition",
                        "absence_of_material_adverse_effect_condition",
                        "financing_condition",
                        "unusual_or_deal_specific_closing_conditions",
                        "material_customer_or_supplier_condition",
                        "employee_retention_condition",
                        "no_governmental_inquiry_condition",
                        "conditions_with_subjective_language",
                    ],
                    "Closing": [
                        "pre_closing_obligations_or_conditions",
                        "conditions_tied_to_stock_price_or_rating",
                    ],
                    "Financing": ["financing_required_for_closing"],
                }

                combined_data = []
                for section_key, field_names in sections_to_combine.items():
                    section_data = schema_results.get(section_key, [])
                    if section_data:
                        filtered_data = [
                            item
                            for item in section_data
                            if str(item.get("answer")).lower() != "not found"
                            and item.get("field_name") in field_names
                        ]
                        combined_data.extend(filtered_data)

                special_fields = [
                    "unusual_or_deal_specific_closing_conditions"]
                has_valid_special_field = False
                for item in combined_data:
                    if (
                        item.get("field_name") in special_fields
                        and str(item.get("answer")).lower() != "not found"
                    ):
                        has_valid_special_field = True
                        break

                unusual_closing_conditions_present = False
                for item in combined_data:
                    if item.get("field_name") == "unusual_closing_conditions_present":
                        answer = str(item.get("answer"))
                        unusual_closing_conditions_present = (
                            True if answer.lower() == "true" else False
                        )
                        break

                if combined_data and unusual_closing_conditions_present:
                    if has_valid_special_field:
                        return self._generate_section_summary(
                            section_name, combined_data, temperature, schema_results
                        )
                    else:
                        return {section_name: "No Unusual Closing Conditions found."}
                else:
                    logger.warning(
                        f"No relevant data found for Unusual Closing Conditions section."
                    )

            elif section_name == "Confidentiality Agreement Sign Date":
                sections_to_combine = {
                    "Confidentiality_Agreement": ["confidentiality_agreement_date"]
                }

                combined_data = []
                for section_key, field_names in sections_to_combine.items():
                    section_data = schema_results.get(section_key, [])
                    if section_data:
                        filtered_data = [
                            item
                            for item in section_data
                            if str(item.get("answer")).lower() != "not found"
                            and item.get("field_name") in field_names
                        ]
                        combined_data.extend(filtered_data)

                if combined_data:
                    return self._generate_section_summary(
                        section_name, combined_data, temperature, schema_results
                    )
                else:
                    logger.warning(
                        f"No relevant data found for Confidentiality Agreement Sign Date section."
                    )

            elif section_name == "Outside Date + Extensions + Reasons":
                sections_to_combine = {
                    "Timeline": ["outside_date"],
                    "Termination_Rights_and_Causes": [
                        "outside_date_termination_right",
                        "outside_date_extension_terms",
                        "outside_date_extension_duration_days",
                    ],
                    "Out_Date": [
                        "maximum_extended_outside_date",
                        "conditions_to_extend_outside_date",
                        "extension_conditions_specified",
                    ],
                    "Antitrust_Commitment": ["extension_conditions"],
                }

                combined_data = []
                for section_key, field_names in sections_to_combine.items():
                    section_data = schema_results.get(section_key, [])
                    if section_data:
                        filtered_data = [
                            item
                            for item in section_data
                            if str(item.get("answer")).lower() != "not found"
                            and item.get("field_name") in field_names
                        ]
                        combined_data.extend(filtered_data)

                if combined_data:
                    return self._generate_section_summary(
                        section_name, combined_data, temperature, schema_results
                    )
                else:
                    logger.warning(
                        f"No relevant data found for Outside Date section.")

            elif section_name == "Regulatory Best Efforts":
                sections_to_combine = {
                    "Best_Efforts": ["regulatory_best_efforts_standard"],
                    "Regulatory_Obligations_Best_Efforts": [
                        "hell_or_high_water_standard_explicitly_applies"
                    ],
                    "Covenants": [
                        "regulatory_remedy_commitments",
                        "regulatory_divestiture_caps",
                        "excluded_business_lines_from_remedies",
                    ],
                    "Regulatory_Approvals": [
                        "scope_of_divestiture_or_remedy_obligation"
                    ],
                }

                combined_data = []
                for section_key, field_names in sections_to_combine.items():
                    section_data = schema_results.get(section_key, [])
                    if section_data:
                        filtered_data = [
                            item
                            for item in section_data
                            if str(item.get("answer")).lower() != "not found"
                            and item.get("field_name") in field_names
                        ]
                        combined_data.extend(filtered_data)

                if combined_data:
                    return self._generate_section_summary(
                        section_name, combined_data, temperature, schema_results
                    )
                else:
                    logger.warning(
                        f"No relevant data found for Regulatory Best Efforts section."
                    )

            elif section_name == "Termination and Reverse Termination Fees + Triggers":
                sections_to_combine = {
                    "Termination_Rights_and_Causes": [
                        "reverse_termination_fee",
                        "reverse_termination_fee_triggers",
                    ],
                    "Termination_Fees__Parent_to_Target_": [
                        "reverse_termination_fee_amount"
                    ],
                    "Termination_Fees__Target_to_Parent": [
                        "termination_fee_amount_target_to_parent"
                    ],
                    "Termination_Fees__Other_": [
                        "termination_fee_reason_category",
                        "reverse_fee_reason_category",
                        "fee_payor_entity",
                        "termination_fee_payment_due_days",
                    ],
                }

                combined_data = []
                for section_key, field_names in sections_to_combine.items():
                    section_data = schema_results.get(section_key, [])
                    if section_data:
                        filtered_data = [
                            item
                            for item in section_data
                            if str(item.get("answer")).lower() != "not found"
                            and item.get("field_name") in field_names
                        ]
                        combined_data.extend(filtered_data)

                if combined_data:
                    return self._generate_section_summary(
                        section_name, combined_data, temperature, schema_results
                    )
                else:
                    logger.warning(
                        f"No relevant data found for Termination Fees section."
                    )

            elif section_name == "Standard or Unusual":
                sections_to_combine = {
                    "Company_Material_Adverse_Change": [
                        "cmac_definition_text",
                        "biotech_mae_disproportionate_effects",
                        "is_mae_biotech_style",
                        "mae_summary_classification",
                        "mae_subjective_terms_flagged",
                    ],
                    "Absolute_Carve-Outs": [
                        "carved_out_events_or_conditions",
                        "carve_outs_explicitly_unqualified",
                    ],
                }

                combined_data = []
                for section_key, field_names in sections_to_combine.items():
                    section_data = schema_results.get(section_key, [])
                    if section_data:
                        filtered_data = [
                            item
                            for item in section_data
                            if str(item.get("answer")).lower() != "not found"
                            and item.get("field_name") in field_names
                        ]
                        combined_data.extend(filtered_data)

                if combined_data:
                    return self._generate_section_summary(
                        section_name, combined_data, temperature, schema_results
                    )
                else:
                    logger.warning(
                        f"No relevant data found for Standard or Unusual section."
                    )

            else:
                # Default case for other sections
                section_data = schema_results.get(section_key)
                if section_data:
                    section_data = [
                        item
                        for item in schema_results.get(section_key, [])
                        if str(item.get("answer")).lower() != "not found"
                    ]
                    return self._generate_section_summary(
                        section_name, section_data, temperature, schema_results
                    )
                else:
                    logger.warning(
                        f"Section key '{section_key}' not found in schema_results."
                    )

            return None

        except Exception as e:
            logger.error(f"Error processing section {section_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def _generate_section_summary(
        self, section_name, section_data, temperature=0.1, schema_results=None
    ):
        """
        Generate a summary for an entire section

        Args:
            section_name (str): The name of the section
            section_data (list): The list of fields and values for the section
            temperature (float, optional): Temperature for OpenAI generation

        Returns:
            dict: The generated summary
        """
        try:
            # Skip if no section data
            if not section_data:
                logger.info(f"Skipping section {section_name} with no data")
                return None

            logger.info(
                f"Generating summary for section: {section_name} with {len(section_data)} fields"
            )

            # Format section data for prompt
            formatted_data = self._format_section_data(section_data)

            # Get the appropriate prompt based on section name
            prompt = self._get_prompt_for_section(
                section_name, formatted_data, schema_results
            )
            logger.info(f"Prompt: {prompt}")

            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise legal document analyzer that creates concise summaries.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=700,
            )

            # Extract the summary
            summary_text = response.choices[0].message.content.strip()
            logger.info(
                f"Generated summary for section {section_name}: {summary_text[:100]}..."
            )

            section_name = (
                section_name.replace("_", " ")
                if section_name != "Regulatory_Obligations_Timing"
                else "Regulatory Obligations"
            )
            # Return the summary object
            return {f"{section_name}": summary_text}

        except Exception as e:
            logger.error(f"Error generating section summary: {str(e)}")
            return None

    def _format_section_data(self, section_data):

        json_string = json.dumps(section_data, indent=2)

        return json_string

    def _get_format_detail_instructions(self, format_type):
        # Format instructions
        if format_type == "paragraph":
            format_instruction = "Return a concise paragraph summary — no bullet points, headings, or extra commentary."
            format_display_text = "paragraph form"
        else:  # default to bullet
            format_instruction = "Return only bullet points — do not include an introduction, conclusion, or any extra commentary."
            format_display_text = "bullet points"

        return format_instruction, format_display_text

    def _get_summary_config(self):

        # Default settings
        default_config = {
            "format_type": "bullet points",  # or "paragraph"
            "level_of_details": "5",  # or "concise", "detailed"
        }

        # Section-specific settings - override default settings as needed
        config = {
            "Complex Consideration": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Go-Shop Terms": {"format_type": "bullet points", "level_of_details": "3"},
            "Unusual Closing Conditions": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Confidentiality Agreement Sign Date": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Outside Date + Extensions + Reasons": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Regulatory Best Efforts": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Termination and Reverse Termination Fees + Triggers": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Standard or Unusual": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Guarantee": {"format_type": "bullet points", "level_of_details": "3"},
            "Best_Efforts": {"format_type": "bullet points", "level_of_details": "3"},
            "Closing": {"format_type": "bullet points", "level_of_details": "3"},
            "Company_Material_Adverse_Change": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Ordinary_Course": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "No_Solicitation": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Dividends": {"format_type": "bullet points", "level_of_details": "3"},
            "Board_Approval": {"format_type": "bullet points", "level_of_details": "3"},
            "Proxy_Statement": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Shareholder_Approval": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Voting_Agreement": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Confidentiality_Agreement": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Clean_Room_Agreement": {
                "format_type": "paragraph",
                "level_of_details": "3",
            },
            "Financing": {"format_type": "bullet points", "level_of_details": "3"},
            "Regulatory_Approvals": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Regulatory_Obligations_Timing": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Out_Date": {"format_type": "bullet points", "level_of_details": "3"},
            "Termination_Rights_and_Causes": {
                "format_type": "paragraph",
                "level_of_details": "3",
            },
            "Merger_Agreement_Details": {
                "format_type": "paragraph",
                "level_of_details": "3",
            },
            "Complete_Effects_on_Capital_Stock": {
                "format_type": "paragraph",
                "level_of_details": "3",
            },
            "R_W_Parent": {"format_type": "paragraph", "level_of_details": "3"},
            "Timeline": {"format_type": "bullet points", "level_of_details": "3"},
            "Breach_Monitoring_and_Ongoing_Operations": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Antitrust_Commitment": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Go-Shop_Terms": {"format_type": "bullet points", "level_of_details": "3"},
            "Complex_Consideration": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Unusual_Closing_Conditions": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Confidentiality_Agreement_Sign_Date": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Outside_Date_Extensions_Reasons": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Regulatory_Best_Efforts": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Termination_and_Reverse_Termination_Fees_Triggers": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
            "Standard_or_Unusual": {
                "format_type": "bullet points",
                "level_of_details": "3",
            },
        }

        return config, default_config

    def _get_level_of_details_config(
        self, level_of_details="5", format_type="bullet points"
    ):

        return f"""Please generate the summary based on the following preferences:
- Level of Detail: {level_of_details} out of 10 (1 = ~50 words, 10 = ~500+ words).
- Format Type: {format_type} (Paragraph or Bullet points).

Adjust both the length and depth of the content accordingly. If 'Bullet points' is selected, keep bullets concise but adjust the total number of points based on the detail level. If 'Paragraph' is selected, increase or decrease paragraph length proportionally.

Be precise:
- Level 1-3 → High-level overview (50-150 words).
- Level 4-7 → Moderate detail (150-300 words).
- Level 8-10 → Highly detailed (300-500+ words)."""

    # Summary prompts section
    def _get_prompt_for_section(
        self, section_name, formatted_data, schema_results=None
    ):

        # Map section names to prompt generator functions
        prompt_generators = {
            "Absolute_Carve-Outs": self._get_absolute_carveouts_prompt,
            "Guarantee": self._get_guarantee_prompt,
            "Company_Material_Adverse_Change": self._get_cmac_prompt,
            "Ordinary_Course": self._get_ordinary_course_prompt,
            "No_Solicitation": self._get_no_solicitation_prompt,
            "Dividends": self._get_dividends_prompt,
            "Board_Approval": self._get_board_approval_prompt,
            "Proxy_Statement": self._get_proxy_statement_prompt,
            "Voting_Agreement": self._get_voting_agreement_prompt,
            "Financing": self._get_financing_prompt,
            "Regulatory_Approvals": self._get_regulatory_approvals_prompt,
            "Regulatory_Obligations_Timing": self._get_regulatory_obligations_prompt,
            "Out_Date": self._get_out_date_prompt,
            "Confidentiality_Agreement": self._get_confidentiality_agreement_prompt,
            "Clean_Room_Agreement": self._get_clean_room_agreement_prompt,
            "Law_and_Jurisdiction": self._get_law_and_jurisdiction_prompt,
            "Termination_Rights_and_Causes": self._get_termination_fees_prompt,
            "Merger_Agreement_Details": self._get_merger_agreement_details_prompt,
            "Complete_Effects_on_Capital_Stock": self._get_complete_effects_on_capital_stock_prompt,
            "R_W_Parent": self._get_R_W_prompt,
            "Timeline": self._get_timeline_prompt,
            "Breach_Monitoring_and_Ongoing_Operations": self._get_breach_monitoring_and_ongoing_operations_prompt,
            "Antitrust_Commitment": self._get_antitrust_commitment_prompt,
            "Go-Shop Terms": self._get_go_shop_prompt,
            "Complex Consideration": self._get_complex_consideration_prompt,
            "Unusual Closing Conditions": self._get_unusual_closing_conditions_prompt,
            "Confidentiality Agreement Sign Date": self._get_confidentiality_agreement_sign_date_prompt,
            "Outside Date + Extensions + Reasons": self._get_outside_date_extensions_reasons_prompt,
            "Regulatory Best Efforts": self._get_regulatory_best_efforts_prompt,
            "Termination and Reverse Termination Fees + Triggers": self._get_termination_and_reverse_termination_fees_prompt,
            "Standard or Unusual": self._get_standard_or_unusual_prompt,
        }

        # Get summary configuration for this section
        config_dict, default_config = self._get_summary_config()
        logger.info(f"Config dict: {config_dict}")
        section_config = config_dict.get(section_name, default_config)

        logger.info(f"Section config1:{section_name}: {section_config}")

        # Extract format and detail level preferences
        format_type = section_config.get("format_type", "bullet points")
        level_of_details = section_config.get("level_of_details", "5")

        # Get the appropriate prompt generator or use default
        prompt_generator = prompt_generators.get(
            section_name, self._get_default_prompt)

        # Generate and return the prompt
        return prompt_generator(
            section_name, formatted_data, schema_results, format_type, level_of_details
        )

    def _get_absolute_carveouts_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        return f"""You are a legal AI assistant specializing in analyzing M&A agreements, with a focus on Material Adverse Effect (MAE) clauses.

Based on the section below, provide a concise, {format_type} summary of the **Absolute Carve-Outs** identified in the MAE provision.

Section Name: {section_name}

Section Data (JSON format):
{formatted_data}

Your response should:
- Clearly explain what "Absolute Carve-Outs" are in the context of MAE clauses.
- Identify specific events or conditions that are excluded from triggering a MAE.
- Highlight how these exclusions shift risk allocation between buyer and seller.
- Use clear, professional language that is accessible to business stakeholders.
- Provide {format_type} that capture the legal and commercial significance.
- If no relevant information is present (e.g., "No relevant document sections found"), state that no Absolute Carve-Outs were identified.

{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ Return only {format_type} — do not include an introduction, conclusion, or meta commentary.
"""

    def _get_guarantee_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        return f"""You are a legal AI assistant specializing in analyzing M&A agreements.

Your task is to generate a concise, professional-grade {format_type} summary of the section below.

Section: {section_name}

Section Content:
{formatted_data}

Instructions:
- Return **only {format_type}** — no introduction, no conclusion, and no extra commentary.
- Use **clear, neutral, professional tone** appropriate for senior business or legal audiences.
- Extract {format_type} summarizing the **material legal and commercial implications** of the section.
- Prioritize key elements such as: obligations, triggers, limitations, survival periods, remedies, and caps.
- Use **precise wording from the agreement** where relevant, but simplify language to ensure readability for non-legal professionals.
- If critical details (e.g., caps, survival periods, indemnification scope) are explicitly absent, state that fact clearly and neutrally.
- If the data says "No relevant document sections found" or is missing, return: "- No summary could be generated due to insufficient information."


{self._get_level_of_details_config(level_of_details, format_type)}
"""

    def _get_cmac_prompt(
        self,
        section_name,
        formatted_data,
        schema_results,
        format_type=None,
        level_of_details=None,
    ):

        section_data = schema_results.get("Absolute_Carve-Outs")

        section_name = section_name.replace("_", " ")

        if section_data:
            section_data = [
                item
                for item in schema_results.get("Absolute_Carve-Outs", [])
                if item.get("answer").lower() != "not found"
            ]

        # Format section data for prompt
        absolute_carveouts = self._format_section_data(section_data)

        return f"""You are a legal AI assistant specializing in summarizing M&A agreements for professional legal and business audiences.

Given the extracted data for the section named below, generate a concise, professional {format_type} summary.

Section: {section_name}

Data:
{formatted_data}

Your output must:
- Return precisely {format_type}.
- Use business-accessible plain language while preserving legal accuracy.
- Refer to exclusions, carve-backs, thresholds, temporal scope, and cross-references where applicable.
- Clearly note if elements like quantitative thresholds or survival periods are not stated.
- Be limited to {format_type} only — no headings, explanations, or commentary outside the {format_type}.
- If no relevant information is present, write: "No summary could be generated due to insufficient information."
- If there are absolute carve-outs present, include a section called 'Absolute Carve-outs' with the following:
  - List each carve-out and exception found, formatted as {format_type}.

{self._get_level_of_details_config(level_of_details, format_type)}

Your tone should match that of a high-quality M&A summary for internal legal diligence.
{f"Absolute Carve-outs: {absolute_carveouts}" if absolute_carveouts else ""}

"""

    def _get_ordinary_course_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):
        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A documents.

Using the section below, generate a concise, professional {format_type} summary for the specified topic within the merger agreement.

**Section:** {section_name}

**Section Data:**
{formatted_data}

Instructions:
- Return precisely {format_type}.
- Use clear, concise business language with appropriate legal phrasing where available.
- Emphasize substantive deal terms such as definitions, restrictions, standards, consent rights, consequences of breach, survival, caps, or exceptions.
- **If key elements are missing** (e.g., no survival clause, no carveouts, undefined terms), **state that explicitly but neutrally**.
- **Do not repeat full clauses** — paraphrase where possible while retaining key legal nuance.
- If the section content is empty or labeled "No relevant document sections found," clearly state: "No summary could be generated due to insufficient information."

{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ Return **only {format_type}** — no introduction, commentary, or headings.
"""

    def _get_no_solicitation_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A documents.

Based on the section provided below, generate a clear, professional {format_type} summary.

Section: {section_name}

Section Text:
{formatted_data}

Guidelines:
- Focus on summarizing the key legal and commercial implications of this section.
- Use accessible business language, while incorporating precise legal terminology where applicable.
- Mention explicitly if any common deal terms (e.g., triggers, thresholds, survival, termination rights) are absent.
- If no substantive text is present or if the section states "No relevant document sections found", write: "No summary could be generated due to insufficient information."


{self._get_level_of_details_config(level_of_details, format_type)}

Return only {format_type} — no titles, introductions, or commentary.
"""

    def _get_dividends_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A agreements.

Your task is to generate a concise, {format_type} summary of the section identified below.

Section: {section_name}

Section Text:
{formatted_data}

Guidelines:
- Summarize the key legal and commercial implications of the section.
- Use professional, business-accessible language with precise legal terminology where applicable.
- If the section explicitly excludes key elements (e.g., caps, triggers, thresholds, survival), mention that clearly and factually.
- If the text is missing or indicates no relevant content (e.g., "No relevant document sections found"), return: "No summary could be generated due to insufficient information."

{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ Output only the {format_type} — no headings, introductions, or conclusions.
"""

    def _get_board_approval_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A agreements.

Based on the extracted content below, generate a concise {format_type} summary of the specified section within the Material Adverse Effect (MAE) provision.

Section: {section_name}

Extracted Data:
{formatted_data}

Your response must:
- Present a clear, professional {format_type} summary.
- Use business-accessible language while preserving legal precision.
- Quote or paraphrase key terms from the source text where applicable.
- Explicitly note if standard elements (e.g., caps, triggers, survival periods) are absent, using objective and neutral language.
- Include {format_type} summarizing the legal and commercial implications.

{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ Output only {format_type} — no headings, introductions, or explanations.
⚠️ If no relevant data is present or content is "Not found", write: "No summary could be generated due to insufficient information."
"""

    def _get_proxy_statement_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in summarizing M&A documents for legal and business professionals.

Based on the section below, generate a concise, professional {format_type} summary.

Section Title: {section_name}

Section Content:
{formatted_data}

Instructions:
- Write {format_type}.
- Use plain language suitable for business readers, but retain legal accuracy where applicable.
- Highlight legal and commercial implications: e.g., triggers, thresholds, carveouts, timelines, survival periods.
- If key details are explicitly missing (e.g., no caps or carveouts), state that clearly but professionally.
- If the section content is missing or says "No relevant document sections found", respond: "No summary could be generated due to insufficient information."

{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ Return only {format_type}. Do not include any introductions, conclusions, or explanatory commentary.
"""

    def _get_shareholder_approval_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A agreements.

Based on the content below, generate a **concise, {format_type} summary** of the specified section from the MAE (Material Adverse Effect) provision.

Section: {section_name}

Section Text:
{formatted_data}

Your response **must** follow these rules:
- Return **only {format_type}** — no introduction, summary heading, or conclusion.
- Use **clear, professional language** that is understandable to business professionals without legal training.
- Where possible, incorporate **precise legal terms** quoted or paraphrased from the text.
- Focus on the **legal and commercial implications** (e.g., definitions, carveouts, thresholds, survival terms).
- If key elements (e.g., carve-backs, temporal scope, quantitative caps) are **explicitly missing**, state that clearly and professionally.
- If the section text is empty or says "No relevant document sections found", return: "- No summary could be generated due to insufficient information."


{self._get_level_of_details_config(level_of_details, format_type)}

Respond strictly with {format_type} and **do not** add commentary or interpretation outside the source material.
"""

    def _get_voting_agreement_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A agreements.

Based on the provided section, generate a concise, professional {format_type} summary for the following MAE-related provision.

Section: {section_name}

Section Data:
{formatted_data}

Your output must:
- Be limited strictly to {format_type} (no introduction or closing commentary).
- Use clear, business-accessible language while maintaining legal precision.
- Extract and summarize key legal and commercial elements (e.g., thresholds, triggers, carveouts, survival periods).
- If any such elements are not explicitly addressed, clearly state that they are "not specified" or "not stated," using professional phrasing.
- Return **{format_type}** only, reflecting the most relevant details.
- If the section data is empty or states "No relevant document sections found", respond with: "No summary could be generated due to insufficient information."

{self._get_level_of_details_config(level_of_details, format_type)}

"""

    def _get_financing_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A documents.

Based on the section below, provide a concise, professional {format_type} summary for the section identified in the MAE provision.

Section: {section_name}

Section Data:
{formatted_data}

Instructions:
- Summarize the section in **{format_type}**.
- Use plain, business-accessible language with precise legal phrasing where applicable.
- If key elements (e.g., caps, triggers, survival, termination rights) are missing, state this clearly and professionally.
- Focus on legal and commercial implications derived from the text.
- If the section content is missing or says "No relevant document sections found", state: "No summary could be generated due to insufficient information."

{self._get_level_of_details_config(level_of_details, format_type)}

- ⚠️ Return **only** {format_type}. Do **not** include any introductory or concluding text.

"""

    def _get_regulatory_approvals_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A documents.

Based on the section below, provide a concise, {format_type} summary of the following section identified in the MAE provision.

Section: {section_name}

Section Data:
{formatted_data}

Your response should:
- Provide a clear, professional {format_type} summary.
- Use plain language that is accessible to non-legal business professionals.
- Incorporate precise legal language from the data where available.
- If key elements (e.g., caps, triggers, survival periods) are explicitly missing, state that clearly and professionally.
- Focusing on the legal and commercial implications.
- If the data states "No relevant document sections found" or is missing, indicate that no summary could be generated due to insufficient information.


{self._get_level_of_details_config(level_of_details, format_type)}


⚠️ Return only {format_type} — do not include an introduction, conclusion, or any extra commentary.
"""

    def _get_regulatory_obligations_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):
        section_data1 = schema_results.get("Regulatory_Obligations_Timing", [])
        section_data2 = schema_results.get(
            "Regulatory_Obligations_Best_Efforts", [])

        section_name = "Regulatory Obligations"

        # Filter out 'Not found' answers

        section_data1 = [
            item
            for item in section_data1
            if str(item.get("answer", "")).lower() != "not found"
        ]
        section_data2 = [
            item
            for item in section_data2
            if str(item.get("answer", "")).lower() != "not found"
        ]

        # Format section data or apply fallback
        timing = self._format_section_data(section_data1)
        if not timing:
            timing = "- Specific regulatory filing deadlines and compliance commitments are not specified."

        best_efforts = self._format_section_data(section_data2)
        if not best_efforts:
            best_efforts = "- Regulatory efforts standards, cooperation commitments, and remedies obligations are not specified."

        return f"""You are a legal AI assistant specializing in summarizing M&A agreements for professional legal and business audiences.

Generate a polished, professional {format_type} summary for the section below, suitable for inclusion in an executive-level M&A diligence report.

Section: {section_name}

Your output must:
- Include subheadings 'Timing' and 'Efforts'.
- Use clean, concise {format_type} under each subheading.
- Use professional, plain language appropriate for senior legal and business professionals.
- If numeric values are present, express large amounts in millions using the "$" symbol and the word "million" (e.g., "$140,000,000" → "$140 million").
- Instead of listing "False" or "True," rephrase for clarity (e.g., "Hell or high water standard: does not explicitly apply.").
- If no data is available for multiple items within a section, consolidate and state them clearly in one grouped sentence (e.g., "Other details, including response timing to agency requests, second request compliance timing commitments, standstill period cooperation, and extended review engagement obligations, are not specified.").
- Avoid repetitive use of "Not specified" after each item; group such missing information into a single sentence as shown in the example.
- Do not include any commentary, explanations, or headings beyond what is requested.


{self._get_level_of_details_config(level_of_details, format_type)}

Example Output:

Timing
- Initial regulatory filing deadlines: within 25 business days following the date of this agreement.
- Pull and refile commitments: Not required.
- Other details, including response timing to agency requests, second request compliance timing commitments, standstill period cooperation, and extended review engagement obligations, are not specified.
-below is the Timing section data:
{timing}

Efforts
- Buyer regulatory efforts standard: reasonable best efforts.
- Buyer divestiture or remedy obligation: Buyer is obligated to divest or remedy.
- Buyer remedy scope summary: Buyer and its subsidiaries are not obliged to take remedy actions affecting company businesses or products with net sales over $140 million in fiscal year 2024.
- Hell or high water standard: does not explicitly apply.
- Mutual cooperation commitment: present, with both parties committed to cooperate.
- Regulatory approvals summary: general governmental and antitrust authorities specified.
- Other details, including seller regulatory efforts standard, regulatory delay risk allocation, and termination rights due to regulatory failure, are not specified.
-below is the Efforts section data:
{best_efforts}
"""

    def _get_out_date_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A documents.

Based on the section below, generate a polished, professional {format_type} summary suitable for both legal professionals and senior business executives.

Section: {section_name}

Section Data:
{formatted_data}

Your response **must**:
- Provide a clear, professional {format_type} summary.
- Use plain language for accessibility, but retain precise legal terminology where appropriate.
- Include specific dates, amounts, and triggers if mentioned. If not explicitly stated, respond with 'Not specified.'
- Clarify ambiguous terms such as 'automatic extension' by explaining the triggering conditions, or explicitly state that triggers are not detailed.
- State the commercial relevance of any termination fee, including whether the amount is specified.
- Focusing only on the most material and commercially significant legal and business implications.
- If no relevant data is found, respond with: 'Not specified.'
- Return only {format_type} — do not include introductions, conclusions, or any extra commentary.

{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ Ensure the output is ready for direct inclusion in a professional M&A deal summary without further editing.
"""

    def _get_confidentiality_agreement_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A documents.

Based on the section below, generate a **concise, professional {format_type} summary** for the section identified in the MAE provision.

Section: {section_name}

Section Data:
{formatted_data}

Your response must adhere to the following guidelines:
- Use precise legal language directly from the source when available; avoid paraphrasing critical legal terms.
- Ensure the summary is accessible to non-legal business professionals while maintaining legal accuracy.
- Highlight key commercial and legal implications, including any material terms such as caps, triggers, survival periods, and enforcement rights.
- If any critical elements (e.g., caps, triggers, survival) are **explicitly missing**, clearly state this in a professional manner.
- If the data contains "No relevant document sections found" or is missing, clearly state: "No summary could be generated due to insufficient information."

{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ **Important:** Return only {format_type}. Do not include any introduction, conclusion, or additional commentary.
"""

    def _get_clean_room_agreement_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in M&A agreements for professional legal and business audiences.

Generate a concise, polished {format_type} summary focused on material legal and commercial implications.

Section: {section_name}

Section Data:
{formatted_data}

Your response must:
- Provide 1 to 5 {format_type} focused only on material information, avoiding procedural or low-value details.
- Use authoritative legal language (e.g., governs, restricts, mandates, requires) and avoid permissive language (e.g., allows, may).
- If key elements (effective dates, expiration dates, survival obligations, caps, triggers) are missing, state this directly and clarify termination triggers if available.
- Avoid vague phrases like 'potential uncertainty'; instead, clearly state what is missing and the legal implications.
- If no relevant information is found, return exactly: "No summary could be generated due to insufficient information."

{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ Return only {format_type}. Do not include introductions, headings, or any additional commentary.
"""

    def _get_law_and_jurisdiction_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A documents.

Based on the section below, provide a **polished, professional {format_type} summary** for the section identified in the MAE provision.

Section: {section_name}

Section Data:
{formatted_data}

Your response must:
- Provide a **clear, professional {format_type} summary**.
- Use **plain language** accessible to non-legal business professionals, while maintaining **precise legal terminology**.
- If courts are referenced, **use the official court name** (e.g., "U.S. District Court for the Southern District of New York" instead of informal descriptions).
- Highlight key legal and commercial implications. If critical elements (e.g., caps, triggers, survival clauses, dispute timelines) are **missing**, clearly state "Not specified."
- Limit the output to **1 to 5 {format_type}**, each covering a distinct concept without redundancy.

If the section states "No relevant document sections found" or contains **insufficient information**, state:
- *"No summary could be generated due to insufficient information."*

{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ **Important: Return only {format_type}. Do not include introductions, conclusions, or any additional commentary.**
"""

    def _get_termination_fees_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):
        section_data1 = schema_results.get(
            "Termination_Fees__Parent_to_Target_", [])
        section_data2 = schema_results.get(
            "Termination_Fees__Target_to_Parent_", [])
        section_data3 = schema_results.get("Termination_Fees__Other_", [])

        section_name = "Termination Fees"

        # Filter out 'Not found' answers
        section_data1 = [
            item
            for item in section_data1
            if str(item.get("answer", "")).lower() != "not found"
        ]
        section_data2 = [
            item
            for item in section_data2
            if str(item.get("answer", "")).lower() != "not found"
        ]
        section_data3 = [
            item
            for item in section_data3
            if str(item.get("answer", "")).lower() != "not found"
        ]

        # Format section data or apply fallback
        parent_to_target = self._format_section_data(section_data1)
        if not parent_to_target:
            parent_to_target = "- Specific regulatory filing deadlines and compliance commitments are not specified."

        target_to_parent = self._format_section_data(section_data2)
        if not target_to_parent:
            target_to_parent = "- Regulatory efforts standards, cooperation commitments, and remedies obligations are not specified."

        other = self._format_section_data(section_data3)
        if not other:
            other = "- Other termination fees are not specified."

        return f"""You are a legal AI assistant specializing in generating executive-level summaries of M&A agreements for professional legal, financial, and business audiences.

Your task is to produce a highly polished, professional summary for the section below, suitable for inclusion in a senior executive M&A diligence report.

Section: {section_name}

Section Data:
{formatted_data}

Instructions:
- Organize content under the headings: 'K to Acquirer', 'Acquirer to K', and 'Other'.
- Present key points as plain text (no bullet points) under each heading.
- Use formal and precise language appropriate for senior legal counsel, C-level executives, and investment professionals.
- When numeric values appear, express large amounts in millions using the "$" symbol and the word "million" (e.g., "$140,000,000" → "$140 million").
- Express percentages clearly using the "%" symbol (e.g., "50 percent" → "50%").
- Format dates uniformly as 'Month Day, Year' (e.g., 'March 23, 2025').
- Replace boolean values (e.g., "True" or "False") with clear, explanatory statements (e.g., "Hell or high water standard: does not apply.").
- Consolidate missing or unspecified information into a single professional statement (e.g., "Other details, including response timing to agency requests, second request compliance timing commitments, standstill period cooperation, and extended review engagement obligations, are not specified.").
- Avoid adding explanations, commentary, or headings beyond what is explicitly requested.
- Ensure the tone remains objective, professional, and free of unnecessary adjectives or filler content.

{self._get_level_of_details_config(level_of_details, format_type)}

Example Output:


K to Acquirer:
{target_to_parent}

Acquirer to K:
{parent_to_target}

Other:
{other}
"""

    def _get_merger_agreement_details_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a highly skilled legal AI assistant specializing in M&A contract analysis.

Your task is to generate a **concise, executive-level {format_type} summary** of the section identified below from the MAE provision.

Section: {section_name}

Section Data:
{formatted_data}

**Instructions:**
- Present exactly **1 to 5 clear, professional {format_type}**.
- Use **business-friendly language** while accurately applying legal terminology where appropriate.
- Prioritize content that reflects **legal risk, commercial impact, and decision-making relevance**.
- If key elements such as **caps, triggers, survival periods, material qualifiers, or carve-outs** are explicitly missing, **state this clearly and professionally** (e.g., "No survival periods or material qualifiers specified.").
- Focus on actionable insights and **material implications for stakeholders**.
- If the data includes "No relevant document sections found" or lacks sufficient content, clearly state: **"No summary generated due to insufficient information."**

⚠️ **Important:**
- Return **only {format_type}**.
- Do **not** include any introductions, conclusions, explanations, or commentary.


{self._get_level_of_details_config(level_of_details, format_type)}

"""

    def _get_complete_effects_on_capital_stock_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""
You are a senior M&A legal assistant specializing in preparing professional-grade deal summaries for executive-level readers.

Based on the section below, generate a **clear, precise, and commercially focused {format_type} summary** suitable for inclusion in a formal M&A transaction report.

**Section:** {section_name}

**Section Data:**
{formatted_data}

### Instructions:
- Output only **{format_type}**—do not include introductions, conclusions, or extra commentary.
- Use **strong, active language** appropriate for a professional M&A summary.
- Maintain a balance between **plain language for non-legal professionals** and **accurate legal terminology**.
- Clearly capture and highlight:
    - **Consideration details** (cash amounts, exchange ratios, stock treatment).
    - **Treatment of options, RSUs, and PSUs**, specifying whether they are canceled, converted, or settled.
    - **Voting rights impacts and termination**.
    - **Treatment of dissenting shares (reference DGCL Section 262 if applicable)**.
    - **Treatment of fractional shares**.
    - **Any conditions, thresholds, or regulatory/tax considerations** (mention if explicitly stated or missing).
- Use phrases like **"will be," "shall,"** or **"must"** to enforce clarity and avoid passive language.
- Provide **exact numerical values** if available; otherwise, state "as defined in the agreement."
- Limit the output to a maximum of **5 concise and impactful {format_type}**.
- If the data is missing or includes "No relevant document sections found," state: *"No summary could be generated due to insufficient information."*


{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ Return only the {format_type}. Do not include any section headings, explanations, or formatting beyond the {format_type}.
"""

    def _get_R_W_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):
        parent_section = schema_results.get("R_W_Parent", [])
        target_section = schema_results.get("R_W_Target", [])

        section_name = "R & W"
        parent = self._format_section_data(
            [
                item
                for item in parent_section
                if str(item.get("answer", "")).lower() != "not found"
            ]
        )
        target = self._format_section_data(
            [
                item
                for item in target_section
                if str(item.get("answer", "")).lower() != "not found"
            ]
        )

        if not parent:
            parent = (
                "- Parent's representations and warranties details are not specified."
            )
        if not target:
            target = (
                "- Target's representations and warranties details are not specified."
            )

        return f"""You are a highly skilled legal AI assistant specializing in drafting executive-level summaries of M&A agreements for professional legal, financial, and corporate leadership audiences.

Your task is to generate a polished, authoritative summary for the section below, suitable for direct inclusion in a senior executive M&A diligence report without further editing.

Section: {section_name}

Instructions:
- Organize content strictly under the headings: 'Parent' and 'Target'.
- Present key points in formal, continuous prose under each heading (avoid bullet points, numbered lists, or casual language).
- Use precise legal and financial terminology appropriate for senior legal counsel, board members, C-level executives, and investment professionals.
- When numeric values appear:
  - Convert large amounts into millions using the "$" symbol and "million" (e.g., "$140,000,000" → "$140 million").
  - Express percentages using the "%" symbol (e.g., "50 percent" → "50%").
- Format all dates uniformly as 'Month Day, Year' (e.g., 'March 23, 2025').
- Replace boolean values (e.g., "True", "False") with formal explanatory statements (e.g., "Hell or high water standard: does not apply.").
- If information is missing, consolidated, or unspecified, use the following standardized statement:
  "Other relevant details, including agency response timing, second request compliance commitments, standstill cooperation periods, and extended regulatory engagement terms, are not specified."
- Avoid any commentary, analysis, or content outside of what is explicitly requested.
- Ensure the tone is objective, authoritative, and free from subjective adjectives or filler content.


{self._get_level_of_details_config(level_of_details, format_type)}

Example Output Format:

Parent:
{parent}

Target:
{target}
"""

    def _get_timeline_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant with expertise in M&A contract analysis.

Your task is to generate a precise, business-friendly summary of the section below, which pertains to the Material Adverse Effect (MAE) provisions.

Section: {section_name}

Section Data:
{formatted_data}

Instructions:
- Return a concise, executive-style summary in {format_type} only (no intro or conclusion).
- Use clear, professional language suitable for legal, financial, and business readers.
- Extract and reflect precise legal terms or triggers where available (e.g., thresholds, carve-outs, timeframes, qualifiers).
- Present the extracted events, obligations, and conditions in chronological order based on their occurrence or specified timelines.
- If specific elements (such as survival periods, caps, or exclusions) are not included, state clearly: "Not specified in the agreement."
- Focus on commercial and legal implications.
- If the source data states "No relevant document sections found" or lacks substantive detail, return: "No summary available due to insufficient information."

{self._get_level_of_details_config(level_of_details, format_type)}
"""

    def _get_breach_monitoring_and_ongoing_operations_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A agreements.

Based on the section provided below, generate a highly professional, concise {format_type} summary tailored for senior legal, financial, and business executives.

Section: {section_name}

Section Data:
{formatted_data}

Your response **must**:
- Return only {format_type} without any introduction, conclusion, or commentary.
- Use clear, professional language suitable for executive-level reporting.
- Favor plain English explanations but retain precise legal terms from the data when appropriate.
- Explicitly state if key legal elements such as **caps**, **triggers**, **survival periods**, or **materiality qualifiers** are missing or not specified.
- Focus the {format_type} on the most significant **legal and commercial implications**.
- Limit the response to a maximum of **5 highly relevant {format_type}**.
- If no relevant data is found (e.g., "No relevant document sections found"), return a single {format_type}:
  - "No summary could be generated due to insufficient information."

{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ Ensure the output is strictly {format_type} only—do not include any preamble, explanations, or summaries beyond the {format_type}.
"""

    def _get_antitrust_commitment_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in M&A transactions.

Based on the section provided below, generate a highly professional, {format_type} summary of the following section identified in the MAE provision.

Section: {section_name}

Section Data:
{formatted_data}

Your response must:
- Deliver a precise and business-relevant summary using {format_type}.
- Use clear, plain language accessible to non-legal business professionals while maintaining appropriate legal accuracy and terminology.
- Prioritize key deal terms, including specific caps, thresholds, obligations, triggers, survival periods, and exceptions. If such elements are explicitly missing, clearly and professionally state their absence.
- Highlight commercial and legal implications of the section, including potential regulatory, financial, and transaction risks or commitments.
- Where possible, quantify commitments or limitations and mention any materiality thresholds or qualifiers directly from the data.
- If the data states "No relevant document sections found" or contains no meaningful content, explicitly state that no summary can be generated due to insufficient information.
- Keep the summary concise, objective, and suitable for inclusion in a professional-grade M&A transaction summary report.

Ensure the output reads as a final deliverable, requiring no further editing.


{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ Return only {format_type} — do not include an introduction, conclusion, or any extra commentary.

"""

    def _get_go_shop_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A documents.

Based on the section below, provide a concise, {format_type} summary of thefollowing section identified in the MAE provision.

Section: {section_name}

Section Data:
{formatted_data}

Your response should:

- If the data clearly indicates that the agreement does not provide a Go-Shop period or allows only consideration of unsolicited proposals, return this exact single-line response:
    No Go-Shop Terms found.
- Otherwise, provide a clear, professional summary in {format_type}.
- Use plain language that is accessible to non-legal business professionals.
- Use precise legal language from the data where available. If key elements (like caps, triggers, survival) are explicitly missing, state that clearly but professionally.
- Provide {format_type} capturing the legal and commercial implications.
- If the data says "No relevant document sections found" or is missing, state that no summary could be generated due to insufficient information.

{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ Important:

    - Return only the single-line statement if no Go-Shop terms exist.
    - Do not include introductions, conclusions, or commentary.

"""

    def _get_complex_consideration_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A documents.

Based on the section below, provide a concise, {format_type} summary of thefollowing section identified in the MAE provision.

Section: {section_name}

Section Data:
{formatted_data}

Your response should:
- Provide a clear, professional summary in {format_type}.
- Use plain language that is accessible to non-legal business professionals.
- Use precise legal language from the data where available. If key elements (like caps, triggers, survival) are explicitly missing, state that clearly but professionally.
- Provide {format_type} capturing the legal and commercial implications.
- If the data says "No relevant document sections found" or is missing, state that no summary could be generated due to insufficient information.

{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ Return only {format_type} — do not include an introduction, conclusion, or any extra commentary.

"""

    def _get_unusual_closing_conditions_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A documents.

Based on the section below, provide a concise, {format_type} summary of thefollowing section identified in the MAE provision.

Section: {section_name}

Section Data:
{formatted_data}

Your response should:
- Provide a clear, professional summary in {format_type}.
- Use plain language that is accessible to non-legal business professionals.
- Use precise legal language from the data where available. If key elements (like caps, triggers, survival) are explicitly missing, state that clearly but professionally.
- Provide {format_type} capturing the legal and commercial implications.
- If the data says "No relevant document sections found" or is missing, state that no summary could be generated due to insufficient information.

{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ Return only {format_type} — do not include an introduction, conclusion, or any extra commentary.

"""

    def _get_confidentiality_agreement_sign_date_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A documents.

Based on the section below, provide a concise, {format_type} summary of thefollowing section identified in the MAE provision.

Section: {section_name}

Section Data:
{formatted_data}

Your response should:
- Provide a clear, professional summary in {format_type}.
- Use plain language that is accessible to non-legal business professionals.
- Use precise legal language from the data where available. If key elements (like caps, triggers, survival) are explicitly missing, state that clearly but professionally.
- Provide {format_type} capturing the legal and commercial implications.
- If the data says "No relevant document sections found" or is missing, state that no summary could be generated due to insufficient information.

{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ Return only {format_type} — do not include an introduction, conclusion, or any extra commentary.

"""

    def _get_outside_date_extensions_reasons_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A documents.

Based on the section below, provide a concise, {format_type} summary of thefollowing section identified in the MAE provision.

Section: {section_name}

Section Data:
{formatted_data}

Your response should:
- Provide a clear, professional summary in {format_type}.
- Use plain language that is accessible to non-legal business professionals.
- Use precise legal language from the data where available. If key elements (like caps, triggers, survival) are explicitly missing, state that clearly but professionally.
- Provide {format_type} capturing the legal and commercial implications.
- If the data says "No relevant document sections found" or is missing, state that no summary could be generated due to insufficient information.

{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ Return only {format_type} — do not include an introduction, conclusion, or any extra commentary.

"""

    def _get_regulatory_best_efforts_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A documents.

Based on the section below, provide a concise, {format_type} summary of thefollowing section identified in the MAE provision.

Section: {section_name}

Section Data:
{formatted_data}

Your response should:
- Provide a clear, professional summary in {format_type}.
- Use plain language that is accessible to non-legal business professionals.
- Use precise legal language from the data where available. If key elements (like caps, triggers, survival) are explicitly missing, state that clearly but professionally.
- Provide {format_type} capturing the legal and commercial implications.
- If the data says "No relevant document sections found" or is missing, state that no summary could be generated due to insufficient information.

{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ Return only {format_type} — do not include an introduction, conclusion, or any extra commentary.

"""

    def _get_termination_and_reverse_termination_fees_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A documents.

Based on the section below, provide a concise, {format_type} summary of thefollowing section identified in the MAE provision.

Section: {section_name}

Section Data:
{formatted_data}

Your response should:
- Provide a clear, professional summary in {format_type}.
- Use plain language that is accessible to non-legal business professionals.
- Use precise legal language from the data where available. If key elements (like caps, triggers, survival) are explicitly missing, state that clearly but professionally.
- Provide {format_type} capturing the legal and commercial implications.
- If the data says "No relevant document sections found" or is missing, state that no summary could be generated due to insufficient information.

{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ Return only {format_type} — do not include an introduction, conclusion, or any extra commentary.

"""

    def _get_standard_or_unusual_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A documents.

Based on the section below, provide a concise, {format_type} summary of thefollowing section identified in the MAE provision.

Section: {section_name}

Section Data:
{formatted_data}

Your response should:
- Provide a clear, professional summary in {format_type}.
- Use plain language that is accessible to non-legal business professionals.
- Use precise legal language from the data where available. If key elements (like caps, triggers, survival) are explicitly missing, state that clearly but professionally.
- Provide {format_type} capturing the legal and commercial implications.
- If the data says "No relevant document sections found" or is missing, state that no summary could be generated due to insufficient information.

{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ Return only {format_type} — do not include an introduction, conclusion, or any extra commentary.

"""

    def _get_default_prompt(
        self,
        section_name,
        formatted_data,
        schema_results=None,
        format_type=None,
        level_of_details=None,
    ):

        section_name = section_name.replace("_", " ")

        return f"""You are a legal AI assistant specializing in analyzing M&A documents.

Based on the section below, provide a concise, {format_type} summary of thefollowing section identified in the MAE provision.

Section: {section_name}

Section Data:
{formatted_data}

Your response should:
- Provide a clear, professional summary in {format_type}.
- Use plain language that is accessible to non-legal business professionals.
- Use precise legal language from the data where available. If key elements (like caps, triggers, survival) are explicitly missing, state that clearly but professionally.
- Provide {format_type} capturing the legal and commercial implications.
- If the data says "No relevant document sections found" or is missing, state that no summary could be generated due to insufficient information.

{self._get_level_of_details_config(level_of_details, format_type)}

⚠️ Return only {format_type} — do not include an introduction, conclusion, or any extra commentary.

"""


class SchemaCategorySearch:
    """Service to search vector store for chunks matching specific schema categories and generate field values"""

    def __init__(self):
        # Initialize embedding service for vector search
        self.embedding_service = EmbeddingService()
        # S3 service to download schema JSON
        self.s3_service = S3Service()
        # OpenAI client for GPT queries
        self.openai_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"))
        # Schema URL
        self.schema_url = "https://mna-docs.s3.eu-north-1.amazonaws.com/clauses_category_template/schema_by_summary_sections.json"
        # Cache for schema
        self._schema = None
        logger.info("SchemaCategorySearch initialized")

    def get_schema(self):
        """
        Fetch and cache the schema JSON

        Returns:
            dict: The schema JSON object
        """
        if self._schema:
            return self._schema

        try:
            logger.info(f"Fetching schema from URL: {self.schema_url}")
            schema = self.s3_service.download_from_url(self.schema_url)
            self._schema = schema
            logger.info(
                f"Successfully downloaded schema with {len(schema)} sections")
            return schema
        except Exception as e:
            logger.error(f"Error fetching schema: {str(e)}")
            return {}

    #  This is a function to extract field answer using GPT.
    def extract_field_value_with_gpt(
        self, field, chunks, section_name, subsection_name=None
    ):
        """
        Extract field value using GPT based on chunks

        Args:
            field (dict): The field information from schema
            chunks (list): List of document chunks to analyze

        Returns:
            str: The extracted field value
        """
        try:
            # Extract relevant info for prompt
            field_name = field.get("field_name", "")
            instructions = field.get("instructions", "")

            # Get the field type
            field_type = field.get("recommended_prompt_type", "")

            # Skip if no chunks found
            if not chunks:
                logger.warning(f"No chunks found for field: {field_name}")
                return "No relevant document sections found"

            if field_type == "Inference-optimized":
                prompt = get_impherior_prompt(
                    section_name, field_name, instructions, chunks, subsection_name
                )
            elif field_type == "Precision-optimized":
                prompt = get_experior_prompt(
                    section_name, field_name, instructions, chunks, subsection_name
                )
            else:
                # Default to experior prompt if type is not specified
                prompt = get_experior_prompt(
                    section_name, field_name, instructions, chunks, subsection_name
                )

            # Call GPT
            logger.info(
                f"Calling GPT to extract value for field: {field_name}")
            logger.info(f"Find answer Promt {prompt}")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
            )

            # Extract and return the value
            value = response.choices[0].message.content.strip()
            logger.info(
                f"Extracted value for field '{field_name}': {value}...")
            try:
                # First check if the response is wrapped in markdown code block
                markdown_match = re.search(
                    r"```(?:json)?\s*([\s\S]+?)\s*```", value)
                if markdown_match:
                    # Extract the JSON content from the markdown code block
                    json_content = markdown_match.group(1).strip()
                    parsed_response = json.loads(json_content)
                else:
                    # Try parsing directly if not in markdown format
                    parsed_response = json.loads(value)

                # Extract just the answer field
                answer = parsed_response.get("answer", "")
                # Extract all fields
                return {
                    "answer": parsed_response.get("answer", ""),
                    "confidence": parsed_response.get("confidence", 1.0),
                    "reason": parsed_response.get("reason", ""),
                    "clause_text": parsed_response.get("clause_text", ""),
                    "reference_section": parsed_response.get("reference_section", ""),
                }

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GPT response as JSON: {value}")
                logger.error(f"JSON parse error: {str(e)}")

                # If we can't parse the JSON, try to extract answer with regex
                answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', value)
                if answer_match:
                    return answer_match.group(1)

                # Default to empty string if all else fails
        except Exception as e:
            logger.error(f"Error extracting field value with GPT: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error: {str(e)}"

    #

    def process_schema_field(self, section_name, field, deal_id, subsection_name=None):
        """
        Process a single schema field - search chunks and extract value

        Args:
            section_name (str): The section name
            field (dict): The field information
            deal_id (str): The deal ID

        Returns:
            dict: Object with field information and extracted value
        """
        try:
            field_name = field.get("field_name", "")
            logger.info(
                f"Processing field: {field_name} in section: {section_name}")

            # Get top categories for this field
            # Get all categories for this field without sorting or limiting
            categories = []
            if "category_mapping" in field and isinstance(
                field["category_mapping"], list
            ):
                # Extract all category names directly
                categories = [
                    mapping["category"]
                    for mapping in field["category_mapping"]
                    if "category" in mapping
                ]

                if not categories:
                    logger.warning(
                        f"No categories found for field: {field_name}")
                    return {
                        "section": section_name,
                        "field_name": field_name,
                        "value": "No categories defined for this field",
                        "categories_used": [],
                    }

            filter_dict = {}
            if deal_id:

                filter_dict = {"deal_id": str(deal_id)}
                print(f"Filter dictionary: {filter_dict}")
                logger.info(f"Filter dictionary: {filter_dict}")

            # Access the Pinecone index directly
            index = self.embedding_service.index

            embedding = EmbeddingService()

            query_embedding = embedding.create_embedding(field["instructions"])

            # Use a dummy vector for metadata-only search
            dummy_vector = [0.0] * 3072  # Dimension for text-embedding-3-small

            # Search in Pinecone using metadata filtering
            search_response = index.query(
                vector=query_embedding,
                top_k=10,  # Get enough results for all categories
                include_metadata=True,
                filter=filter_dict,
            )
            logger.info(f"Search response: {len(search_response.matches)}")

            # Extract metadata from matches
            all_chunks = [match.metadata for match in search_response.matches]

            # Remove duplicates if any
            unique_chunks = []
            seen_texts = set()
            for chunk in all_chunks:
                text = chunk.get("combined_text", "")
                if text not in seen_texts:
                    seen_texts.add(text)
                    unique_chunks.append(chunk)

            logger.info(
                f"Found All {len(all_chunks)} All chunks matching categories")
            logger.info(
                f"Found Unique {len(unique_chunks)} unique chunks matching categories"
            )

            # Combine all text into a single string
            combined_text = ""
            for idx, chunk in enumerate(unique_chunks):
                chunk_text = chunk.get("combined_text", "")
                if chunk_text:
                    if combined_text:
                        combined_text += "\n\n"
                    combined_text += (
                        f"[Label : {chunk.get("label", "")}]\n\n{chunk_text.strip()}"
                    )

            # Extract value with GPT
            value = self.extract_field_value_with_gpt(
                field, combined_text, section_name, subsection_name
            )

            # Return result object
            return {
                "section": section_name,
                "field_name": field_name,
                "answer": value["answer"],
                "confidence": value["confidence"],
                "reason": value["reason"],
                "categories_used": categories,
            }

        except Exception as e:
            logger.error(f"Error processing schema field: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "section": section_name,
                "field_name": field.get("field_name", "unknown"),
                "value": f"Error: {str(e)}",
                "categories_used": [],
            }

    def process_schema_field1(self, section_name, field, deal_id, subsection_name=None):

        try:
            field_name = field.get("field_name", "")
            question_query = field.get("question_query", "")
            logger.info(
                f"Processing field: {field_name} in section: {section_name}")

            # Get categories for this field (same as before)
            categories = []
            # if "category_mapping" in field and isinstance(field["category_mapping"], list):
            #     categories = [mapping["category"]
            #                   for mapping in field["category_mapping"] if "category" in mapping]

            #     if not categories:
            #         logger.warning(
            #             f"No categories found for field: {field_name}")
            #         return {
            #             "section": section_name,
            #             "field_name": field_name,
            #             "value": "No categories defined for this field",
            #             "categories_used": []
            #         }

            # Prepare filter dictionary for deal_id
            filter_dict = {}
            if deal_id:
                filter_dict = {"deal_id": str(deal_id)}
                print(f"Filter dictionary: {filter_dict}")
                logger.info(f"Filter dictionary: {filter_dict}")

            # Access the Pinecone index directly
            index = self.embedding_service.index
            embedding = EmbeddingService()

            # FIRST QUERY: using field instructions (original query)
            query_embedding_1 = embedding.create_embedding(
                field["instructions"])

            # Search in Pinecone using metadata filtering with original query
            search_response_1 = index.query(
                vector=query_embedding_1,
                top_k=5,  # Reduced from 10 to 5 as requested
                include_metadata=True,
                filter=filter_dict,
            )
            logger.info(
                f"Search response 1 (original query): {len(search_response_1.matches)}"
            )

            # Extract metadata from first query matches
            chunks_1 = [match.metadata for match in search_response_1.matches]

            # SECOND QUERY: using section name as query
            new_section_name = question_query.replace("_", " ")
            query_embedding_2 = embedding.create_embedding(new_section_name)
            logger.info(f"Query embedding 2: {new_section_name}")

            # Search in Pinecone using metadata filtering with section name
            search_response_2 = index.query(
                vector=query_embedding_2,
                top_k=5,  # Use top_k=5 for section name query too
                include_metadata=True,
                filter=filter_dict,
            )
            logger.info(
                f"Search response 2 (section name): {len(search_response_2.matches)}"
            )

            # Extract metadata from second query matches
            chunks_2 = [match.metadata for match in search_response_2.matches]

            # Combine results from both queries
            all_chunks = chunks_1 + chunks_2
            logger.info(
                f"Combined chunks before deduplication: {len(all_chunks)}")

            # Remove duplicates
            unique_chunks = []
            seen_texts = set()
            for chunk in all_chunks:
                text = chunk.get("combined_text", "")
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    unique_chunks.append(chunk)

            logger.info(
                f"Found All {len(all_chunks)} chunks matching categories")
            logger.info(
                f"Found Unique {len(unique_chunks)} unique chunks after deduplication"
            )

            # Combine all text into a single string (same as before)
            combined_text = ""
            for idx, chunk in enumerate(unique_chunks):
                chunk_text = chunk.get("combined_text", "")
                if chunk_text:
                    if combined_text:
                        combined_text += "\n\n"
                    combined_text += (
                        f"[Label : {chunk.get('label', '')}]\n\n{chunk_text.strip()}"
                    )

            # Extract value with GPT (same as before)
            value = self.extract_field_value_with_gpt(
                field, combined_text, section_name, subsection_name
            )

            # Return result object (same as before)
            return {
                "section": section_name,
                "field_name": field_name,
                "answer": value["answer"],
                "confidence": value["confidence"],
                "reason": value["reason"],
                "clause_text": value["clause_text"],
                "reference_section": value["reference_section"],
                # "categories_used": categories
            }

        except Exception as e:
            logger.error(f"Error processing schema field: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "section": section_name,
                "field_name": field.get("field_name", "unknown"),
                "value": f"Error: {str(e)}",
                "categories_used": [],
            }

    def remove_duplicate_chunks(chunks):

        seen_ids = set()
        unique_chunks = []

        for chunk in chunks:
            chunk_id = chunk.get("id")
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_chunks.append(chunk)

        return unique_chunks

    def search_all_schema_categories(self, deal_id):
        """
        Process all fields in the schema for a specific deal

        Args:
            deal_id (str): The deal ID to process

        Returns:
            dict: Dictionary with sections and their extracted field values
        """
        schema = self.get_schema()
        if not schema:
            logger.warning("No schema available")
            return {"error": "Schema not available"}

        try:
            results = {}
            # Set up a ThreadPoolExecutor with a reasonable number of workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                # Dictionary to track all future objects by section and field
                futures = {}

                # First pass: Submit all tasks to the executor
                for section_name, section_value in schema.items():
                    # Check if this section should be processed
                    if section_name in [
                        "termination",
                        "ordinary_course",
                        "board_approval",
                        "party_details",
                        "conditions_to_closing",
                        "closing_mechanics",
                        "specific_performance",
                        "confidentiality_and_clean_room",
                        "complex_consideration_and_dividends",
                        "law_and_jurisdiction",
                        "financing",
                        "proxy_statement",
                        "timeline",
                        "material_adverse_effect",
                        "non_solicitation",
                        "best_efforts",
                    ]:
                        # Change this to your desired section
                        # if section_name == "best_efforts":

                        logger.info(
                            f"Submitting tasks for section: {section_name}")

                        # Initialize the section's results
                        results[section_name] = {}
                        futures[section_name] = {}

                        # Check if section_value is an array or object
                        if section_name == "termination_clauses":
                            # Initialize as dictionary
                            results[section_name] = {}
                            futures[section_name] = []

                            # First process each clause to determine if it exists in the document
                            for i, field in enumerate(section_value):

                                # if i > 0:
                                #     break
                                field_name = field.get("field_name", "")

                                if (
                                    field_name
                                    == "Failure to Receive Required Approvals"
                                ):

                                    logger.info(
                                        f"Submitting task for clause identification: {field_name} ({i+1}/{len(section_value)})"
                                    )

                                    # Submit the task to check if this clause exists
                                    future = executor.submit(
                                        self.process_schema_field1,
                                        section_name,
                                        field,
                                        deal_id,
                                    )
                                    futures[section_name].append(
                                        # Store clause, future, and the full field object
                                        (field_name, future, field)
                                    )
                                else:
                                    logger.info(
                                        f"Skipping clause identification: {field_name} ({i+1}/{len(section_value)})"
                                    )

                        elif isinstance(section_value, list):
                            # Handle as a simple array of fields (old format)
                            # Use array for old format
                            results[section_name] = []
                            futures[section_name] = []

                            for i, field in enumerate(section_value):
                                field_name = field.get("field_name", "")
                                # if field_name == "hsr_clearance_required":
                                logger.info(
                                    f"Submitting task for field: {field_name} ({i+1}/{len(section_value)})"
                                )

                                # Submit the task to the executor and store the future
                                future = executor.submit(
                                    self.process_schema_field1,
                                    section_name,
                                    field,
                                    deal_id,
                                )
                                futures[section_name].append(
                                    (field_name, future))
                                # else:
                                #     logger.info(
                                # f"Skipping field: {field_name}")

                        elif isinstance(section_value, dict):
                            # Handle as a nested object with subsections (new format)
                            for subsection_name, fields in section_value.items():
                                logger.info(
                                    f"Processing subsection: {subsection_name}")
                                # if subsection_name == "reverse_termination_fee":
                                # continue

                                # Initialize subsection results and futures
                                results[section_name][subsection_name] = []
                                futures[section_name][subsection_name] = []

                                for i, field in enumerate(fields):
                                    field_name = field.get("field_name", "")
                                    # if field_name == "divestiture_clause_summary":
                                    # continue
                                    logger.info(
                                        f"Submitting task for field: {field_name} ({i+1}/{len(fields)})"
                                    )

                                    # Submit the task to the executor and store the future
                                    future = executor.submit(
                                        self.process_schema_field1,
                                        section_name,
                                        field,
                                        deal_id,
                                        subsection_name,
                                    )
                                    futures[section_name][subsection_name].append(
                                        (field_name, future)
                                    )
                                    # else:
                                    #     logger.info(
                                    #         f"Skipping field: {field_name}")
                                # else:
                                #     logger.info(
                                #         f"Skipping section: {section_name}")
                        # else:
                        #     logger.info(
                        #         f"Skipping section: {section_name}")
                    else:
                        logger.info(f"Skipping section: {section_name}")

                # Second pass: Collect results as they complete
                for section_name, section_futures in futures.items():

                    if section_name == "termination_clauses":
                        for field_name, future, field_obj in section_futures:
                            try:
                                # Get the result for the clause check
                                clause_result = future.result()
                                logger.info(
                                    f"Completed clause check for {field_name}: {clause_result}"
                                )

                                # Check if the answer indicates this clause exists
                                if (
                                    clause_result
                                    and "answer" in clause_result
                                    and clause_result["answer"]
                                    and str(clause_result["answer"]).lower()
                                    not in ["not found", "false", "no"]
                                ):

                                    # Initialize results for this clause
                                    results[section_name][field_name] = []

                                    # Create a list to track field processing futures
                                    field_futures = []

                                    # Now process each field within the clause
                                    if "fields" in field_obj:
                                        logger.info(
                                            f"Processing {len(field_obj['fields'])} fields for clause: {field_name}"
                                        )

                                        for field_item in field_obj["fields"]:
                                            # if field_item.get("field_name", "") == "clause_summary_bullets" or field_item.get("field_name", "") == "clause_summary_text":

                                            # Submit this field for processing
                                            field_future = executor.submit(
                                                self.process_schema_field1,
                                                section_name,
                                                field_item,
                                                deal_id,
                                                field_name,
                                            )

                                            field_futures.append(
                                                (field_item, field_future)
                                            )
                                            # else:
                                            #     logger.info(
                                            #         f"Skipping field: {field_item.get('field_name')}")

                                        # Process the results for each field
                                        for field_item, field_future in field_futures:
                                            try:
                                                field_result = field_future.result()
                                                logger.info(
                                                    f"Field result for {field_item.get('field_name')}: {field_result}"
                                                )

                                                # Add the field result to the clause results
                                                results[section_name][
                                                    field_name
                                                ].append(field_result)

                                            except Exception as field_e:
                                                logger.error(
                                                    f"Error processing field {field_item.get('field_name')}: {str(field_e)}"
                                                )
                                                # Include a placeholder entry for the failed field

                                                results[section_name][
                                                    field_name
                                                ].append(
                                                    {
                                                        "section": section_name,
                                                        "subsection": field_name,
                                                        "field_name": field_item.get(
                                                            "field_name", ""
                                                        ),
                                                        "answer": f"Error: {str(field_e)}",
                                                        "confidence": 0.0,
                                                        "reason": "Processing error",
                                                        "categories_used": [],
                                                        "clause_text": "",
                                                        "reference_section": "",
                                                    }
                                                )

                            except Exception as e:
                                logger.error(
                                    f"Error processing clause {field_name}: {str(e)}"
                                )
                                # Skip this clause entirely if there was an error determining its existence

                    elif isinstance(section_futures, list):
                        # Old format handling (array of fields)
                        for field_name, future in section_futures:
                            try:
                                # Get the result from the future
                                field_result = future.result()
                                logger.info(
                                    f"Completed field result: {field_result}")
                                results[section_name].append(field_result)
                            except Exception as e:
                                logger.error(
                                    f"Error processing field {field_name}: {str(e)}"
                                )
                                # Add error result
                                results[section_name].append(
                                    {
                                        "section": section_name,
                                        "field_name": field_name,
                                        "answer": f"Error: {str(e)}",
                                        "confidence": 0.0,
                                        "reason": "Processing error",
                                        "categories_used": [],
                                        "clause_text": "",
                                        "reference_section": "",
                                    }
                                )
                    elif isinstance(section_futures, dict):
                        # New format handling (nested structure)
                        for (
                            subsection_name,
                            subsection_futures,
                        ) in section_futures.items():
                            for field_name, future in subsection_futures:
                                try:
                                    # Get the result from the future
                                    field_result = future.result()
                                    logger.info(
                                        f"Completed field result: {field_result}"
                                    )
                                    results[section_name][subsection_name].append(
                                        field_result
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Error processing field {field_name}: {str(e)}"
                                    )
                                    # Add error result
                                    results[section_name][subsection_name].append(
                                        {
                                            "section": section_name,
                                            "subsection": subsection_name,
                                            "field_name": field_name,
                                            "answer": f"Error: {str(e)}",
                                            "confidence": 0.0,
                                            "reason": "Processing error",
                                            "categories_used": [],
                                            "clause_text": "",
                                            "reference_section": "",
                                        }
                                    )

            # Get current time in UTC and convert to IST
            utc_now = datetime.now(pytz.UTC)
            ist = pytz.timezone("Asia/Kolkata")
            ist_now = utc_now.astimezone(ist)

            # Create timestamp in IST
            timestamp = ist_now.strftime("%d-%m-%y_%I-%M_%p")

            # Create a filename with timestamp
            filename = f"schema_results_{timestamp}.json"

            # Save results to JSON file
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info(f"Results saved to file: {filename}")
            except Exception as save_error:
                logger.error(
                    f"Error saving results to file: {str(save_error)}")

            return results

        except Exception as e:
            logger.error(f"Error processing schema: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
