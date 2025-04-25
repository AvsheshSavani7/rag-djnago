import os
import json
import openai
import pinecone
import time
import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

# Load environment variables
load_dotenv()


class SimpleMetadataService:
    """Simplified metadata service for testing"""

    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        # Schema URL
        self.schema_url = "https://rag-mna.s3.eu-north-1.amazonaws.com/master-schema/enhanced_metadata_by_category.json"
        # Cache the schema
        self._schema = None

    def get_schema(self):
        """Fetch and cache the schema"""
        if self._schema:
            return self._schema

        response = requests.get(self.schema_url)
        response.raise_for_status()
        self._schema = response.json()
        return self._schema

    def determine_category(self, text):
        """Determine category for the given text"""
        schema = self.get_schema()
        categories = list(schema.keys())

        # Prepare prompt for categorization
        category_list = "\n".join(
            [f"{i+1}. {category}" for i, category in enumerate(categories)])

        prompt = f"""You are tasked with categorizing legal contract text into the most appropriate category.
Given the following text from a contract:

"{text}"

Please determine which of the following categories best fits this text:
{category_list}

Return ONLY the exact name of the single best matching category. If none match well, return an empty string."""

        # Call GPT for categorization
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a legal contract classifier assistant. Your task is to assign text to the most appropriate category."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        category = response.choices[0].message.content.strip()
        print(f"Determined category: {category}")

        # Validate category
        if category in categories:
            return category
        elif category == "":
            return ""
        else:
            # Try to find close match
            for schema_category in categories:
                if category.lower() in schema_category.lower():
                    print(f"Found close match: {schema_category}")
                    return schema_category
            return ""

    def extract_structured_metadata(self, text, category_name):
        """Extract structured metadata based on schema"""
        schema = self.get_schema()

        if not category_name or category_name not in schema:
            print(f"Category '{category_name}' not found in schema")
            return {}

        # Get fields for this category
        category_fields = schema[category_name]

        # Format fields for prompt
        field_descriptions = []
        for field in category_fields:
            field_descriptions.append(
                f"Field: {field.get('name')}\nDescription: {field.get('description')}\nType: {field.get('data_type')}\nExample: {field.get('example_value')}")

        fields_text = "\n\n".join(field_descriptions)

        # Prepare prompt for extraction
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

        # Call GPT for extraction
        print(f"Extracting structured metadata for category: {category_name}")
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a legal metadata extraction assistant. Your task is to extract structured information from contract text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )

        metadata_text = response.choices[0].message.content.strip()
        print(f"Extraction response: {metadata_text}")

        # Parse response
        try:
            # Find JSON content
            if not metadata_text.startswith('{'):
                import re
                json_match = re.search(
                    r'```json\s*([\s\S]*?)\s*```', metadata_text)
                if json_match:
                    metadata_text = json_match.group(1)
                else:
                    json_match = re.search(r'({[\s\S]*})', metadata_text)
                    if json_match:
                        metadata_text = json_match.group(1)

            structured_metadata = json.loads(metadata_text)
            print(
                f"Successfully parsed structured metadata with {len(structured_metadata)} fields")
            return structured_metadata
        except Exception as e:
            print(f"Error parsing metadata as JSON: {str(e)}")
            return {
                "clause_summary": "Failed to extract structured metadata",
                "clause_tags_llm": [],
                "error": "Failed to parse JSON response"
            }

    def process_text(self, text, label):
        """Process text and extract all metadata"""
        print(f"Processing text: {text[:50]}...")

        # Determine category
        category = self.determine_category(text)

        # Create base metadata
        metadata = {
            "label": label,
            "combined_text": text,
            "category_name": category
        }

        # Extract structured metadata if category found
        if category:
            structured_metadata = self.extract_structured_metadata(
                text, category)
            metadata.update(structured_metadata)

        return metadata


class SimpleEmbeddingService:
    """Simplified embedding service for testing"""

    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )

        # Initialize Pinecone
        self.pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index_name = os.environ.get(
            "PINECONE_INDEX_NAME", "contract-chunks")

        # Check if index exists
        indexes = [index.name for index in self.pc.list_indexes()]
        print(f"Available Pinecone indexes: {indexes}")

        if self.index_name not in indexes:
            print(f"Creating index '{self.index_name}'")
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # for text-embedding-3-small
                metric="cosine"
            )

        # Connect to index
        self.index = self.pc.Index(self.index_name)
        print(f"Connected to Pinecone index: {self.index_name}")

        # Initialize metadata service
        self.metadata_service = SimpleMetadataService()

    def create_embedding(self, text):
        """Create embedding for text"""
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def process_chunks(self, chunks, deal_id):
        """Process chunks and store in Pinecone"""
        print(f"Processing {len(chunks)} chunks for deal ID: {deal_id}")
        processed = 0

        for i, chunk in enumerate(tqdm(chunks)):
            print(f"\nProcessing chunk {i+1}/{len(chunks)}")
            chunk_id = f"{deal_id}_{i}"

            # Get text
            text = chunk.get("combined_text", "")
            if not text:
                print(f"Skipping chunk {i+1} - no text content")
                continue

            # Get label
            label = chunk.get("label", "")

            # Process metadata
            print("Enhancing metadata...")
            metadata = self.metadata_service.process_text(text, label)

            # Add deal_id and chunk_index
            metadata["deal_id"] = deal_id
            metadata["chunk_index"] = i

            # Convert complex types to strings
            for key, value in list(metadata.items()):
                if isinstance(value, (list, dict)):
                    metadata[key] = json.dumps(value)
                elif value is None:
                    metadata[key] = ""
                else:
                    metadata[key] = str(value)

            # Create embedding
            print("Creating embedding...")
            embedding = self.create_embedding(text)

            # Store in Pinecone
            print(f"Uploading to Pinecone with ID: {chunk_id}")
            self.index.upsert(
                vectors=[{
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": metadata
                }]
            )

            processed += 1
            time.sleep(0.2)  # Avoid rate limits

        return {
            "status": "success",
            "processed": processed,
            "total": len(chunks)
        }

    def fetch_vectors(self, deal_id):
        """Fetch vectors for a deal"""
        zero_vector = [0.0] * 1536
        return self.index.query(
            vector=zero_vector,
            top_k=10,
            include_metadata=True,
            filter={"deal_id": deal_id}
        )


def run_test():
    """Run the test"""
    # Sample deal_id
    deal_id = "68089f5e478abf06ec1a13d7"

    # Sample chunks
    chunks = [
        {
            "label": "ARTICLE 1 > Section 1.1 > Closing Conditions",
            "combined_text": "The obligations of each party to consummate the transactions contemplated hereby shall be subject to the fulfillment, at or prior to the Closing, of each of the following conditions: (a) no Governmental Authority shall have enacted, issued, promulgated, enforced or entered any Law or Governmental Order which is in effect and has the effect of making the transactions contemplated by this Agreement illegal, otherwise restraining or prohibiting consummation of such transactions or causing any of the transactions contemplated hereunder to be rescinded following completion thereof; (b) the Required Regulatory Approvals shall have been received and shall be in full force and effect; and (c) Shareholders holding at least 75% of the outstanding shares entitled to vote shall have approved the merger agreement."
        },
        {
            "label": "ARTICLE 5 > Section 5.3 > Financial Terms",
            "combined_text": "Purchase Price. The aggregate purchase price for the Shares shall be an amount in cash equal to $100,000,000 (the 'Purchase Price'), subject to adjustment as set forth in Section 5.4. The Purchase Price shall be paid in immediately available funds by wire transfer to an account designated by Seller, such designation to be made at least three Business Days prior to the Closing Date."
        }
    ]

    # Initialize service
    service = SimpleEmbeddingService()

    # Process chunks
    print(f"Starting test with {len(chunks)} sample chunks...")
    result = service.process_chunks(chunks, deal_id)
    print(f"Processing completed with result: {result}")

    # Fetch results
    print("\nFetching processed vectors to verify metadata...")
    response = service.fetch_vectors(deal_id)

    # Display results
    print(f"Found {len(response.matches)} vectors")
    for i, match in enumerate(response.matches):
        print(f"\nVector {i+1}:")
        print(f"  ID: {match.id}")
        print(f"  Category: {match.metadata.get('category_name', 'None')}")

        # Show structured metadata
        has_structured = False
        for key in match.metadata:
            if key not in ["deal_id", "label", "combined_text", "category_name", "chunk_index"]:
                has_structured = True
                value = match.metadata[key]
                # Try to parse JSON
                try:
                    if value.startswith('{') or value.startswith('['):
                        value = json.loads(value)
                        value = json.dumps(value, indent=2)
                except:
                    pass

                # Truncate long values
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")

        if not has_structured:
            print("  No structured metadata found")


if __name__ == "__main__":
    run_test()
