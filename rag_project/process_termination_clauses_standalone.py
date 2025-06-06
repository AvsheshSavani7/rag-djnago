import os
import json
import logging
import datetime
from zoneinfo import ZoneInfo
import openai
import pinecone
from dotenv import load_dotenv
import time
from typing import Dict, List, Any
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytz

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("termination_clauses_processing.log")
    ]
)
logger = logging.getLogger(__name__)


class TerminationClauseProcessor:
    """Process termination clauses using direct OpenAI and Pinecone access"""

    def __init__(self, max_workers: int = 4):
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"))

        # Initialize Pinecone connection
        self.pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index_name = os.environ.get(
            "PINECONE_INDEX_NAME", "contract-chunks")
        self.index = self.pc.Index(self.index_name)

        # Results storage
        self.results = {}

        # Add thread pool
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(
            f"Initialized ThreadPoolExecutor with {max_workers} workers")

    def load_termination_clauses(self) -> List[Dict[str, Any]]:
        """Load termination clauses from JSON file"""
        try:
            with open("termination_clauses.json", "r") as f:
                data = json.load(f)
            logger.info(
                f"Loaded {len(data['termination_clauses'])} termination clauses")
            return data["termination_clauses"]
        except Exception as e:
            logger.error(f"Error loading termination clauses: {str(e)}")
            raise

    def create_embedding(self, text: str) -> List[float]:
        """Create an embedding for a text using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise

    def search_pinecone(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search Pinecone for relevant chunks"""
        try:
            search_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )

            chunks = []
            for match in search_response.matches:
                chunks.append(match.metadata)

            return chunks
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            return []

    def get_relevant_chunks(self, field: Dict[str, Any]) -> str:
        """Get relevant document chunks using two queries"""
        try:
            # FIRST QUERY: using field instructions (original query)
            query_embedding_1 = self.create_embedding(
                field.get("instructions", ""))

            # Search in Pinecone using first query
            search_response_1 = self.index.query(
                vector=query_embedding_1,
                top_k=5,  # Reduced from 10 to 3 as in service.py
                include_metadata=True
            )
            logger.info(
                f"Search response 1 (original query): {len(search_response_1.matches)}")

            # Extract metadata from first query matches
            chunks_1 = [match.metadata for match in search_response_1.matches]

            # SECOND QUERY: using section name as query
            question_query = field.get("question_query", "termination_clauses")
            new_section_name = question_query.replace("_", " ")
            query_embedding_2 = self.create_embedding(new_section_name)
            logger.info(f"Query embedding 2: {new_section_name}")

            # Search in Pinecone using section name
            search_response_2 = self.index.query(
                vector=query_embedding_2,
                top_k=10,  # Use top_k=9 as in service.py
                include_metadata=True
            )
            logger.info(
                f"Search response 2 (section name): {len(search_response_2.matches)}")

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
                f"Found Unique {len(unique_chunks)} unique chunks after deduplication")

            # Combine all text into a single string
            combined_text = ""
            for idx, chunk in enumerate(unique_chunks):
                chunk_text = chunk.get("combined_text", "")
                if chunk_text:
                    if combined_text:
                        combined_text += "\n\n"
                    combined_text += f"[Label: {chunk.get('label', '')}]\n\n{chunk_text.strip()}"

            return combined_text

        except Exception as e:
            logger.error(f"Error getting relevant chunks: {str(e)}")
            return ""

    def extract_field_value_with_gpt(self, clause: Dict[str, Any], context_text: str) -> Dict[str, Any]:
        """Extract field value using GPT based on context"""
        field_name = clause.get("field_name", "")
        instructions = clause.get("instructions", "")

        fields = clause.get("fields", [])

        # Skip if no context text found
        if not context_text:
            logger.warning(f"No context text found for clause: {field_name}")
            return {
                "field_name": field_name,
                "answer": "No relevant document sections found",
                "confidence": 0.0,
                "reason": "No context available",
                "clause_text": "",
                "reference_section": "",
                "summary": "",
                "fields": []
            }

        # Create fields part of the prompt
        fields_text = ""
        if fields:
            fields_text = f"\n\nIf the clause exists (answer is 'true'), please answer these additional questions about the {field_name} termination clause:\n"
            for i, field in enumerate(fields, 1):
                field_name_clean = field.get("field_name", "").lower()
                # Create dynamic field instruction based on common patterns
                if field_name_clean == "trigger":
                    fields_text += f"\n{i}. trigger: What specific condition or event triggers termination under the {field_name} clause?"
                elif field_name_clean == "who":
                    fields_text += f"\n{i}. who: Which party or parties have the right to terminate under the {field_name} clause?"
                elif field_name_clean == "fee_required":
                    fields_text += f"\n{i}. fee_required: Is a termination fee required when terminating under the {field_name} clause? Answer 'Yes', 'No', or 'Conditional'."
                elif field_name_clean == "fee_amount":
                    fields_text += f"\n{i}. fee_amount: What is the specific amount of any termination fee required under the {field_name} clause? If none, state 'N/A'."
                elif field_name_clean == "cure_period":
                    fields_text += f"\n{i}. cure_period: How many days are provided to cure before termination can occur under the {field_name} clause? If none, state 'None'."
                elif field_name_clean == "notice_required":
                    fields_text += f"\n{i}. notice_required: Is written notice required before termination under the {field_name} clause? Answer 'Yes' or 'No'."
                elif field_name_clean == "structured_summary":
                    fields_text += f"\n{i}. structured_summary: Provide a brief summary of how the {field_name} termination clause operates."
                elif field_name_clean == "clause_summary_bullets":
                    fields_text += f"\n{i}. clause_summary_bullets: List 2-3 key points about the {field_name} termination clause."
                else:
                    # Use original instruction if no pattern matches, but include field name
                    fields_text += f"\n{i}. {field.get('field_name', '')}: For the {field_name} termination clause, {field.get('instructions', '')}"

        # Create prompt for GPT
        prompt = f"""You are a legal AI expert analyzing contract clauses.

                    TASK:
                    Analyze the contract sections below and answer questions about a termination clause.

                    MAIN QUESTION ABOUT "{field_name}":
                    {instructions}

                    {fields_text}

                    CONTRACT SECTIONS:
                    {context_text}

                    Return your answer as a JSON object with the following structure:
                    {{
                    "answer": "your direct answer to the main question (true/false)",
                    "confidence": a number between 0 and 1 indicating your confidence in the answer,
                    "reason": "brief reasoning for your answer",
                    "clause_text": "the exact text from the contract that supports your answer",
                    "reference_section": "label or identifier of the relevant section",
                    "summary": "a brief summary of the relevant clause",
                    "fields": [
                        {{
                        "field_name": "name of the field",
                        "answer": "answer to the field question",
                        "confidence": confidence score for this answer,
                        "reason": "reasoning for this answer",
                        "clause_text": "supporting text for this answer",
                        "reference_section": "relevant section for this answer"
                        }},
                        // ... one object for each field if the clause exists
                    ]
                    }}

                    If you cannot find the clause (answer is "false" or "not found"), return an empty array for fields.
                    If you find the clause but cannot answer a specific field question, include the field in the array but set its answer to "Not found" with 0 confidence."""

        logger.info(f"Field Name: {field_name}")
        logger.info(f"Final Prompt: {prompt}")

        # Call GPT
        try:
            logger.info(f"Calling GPT for clause: {field_name}")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=3000
            )

            # Extract and parse the response
            result_text = response.choices[0].message.content.strip()
            logger.info(
                f"GPT response for {field_name}: {result_text[:100]}...")

            try:
                # Check if the response is in JSON format or inside a code block
                import re
                json_match = re.search(
                    r"```(?:json)?\s*([\s\S]+?)\s*```", result_text)
                if json_match:
                    # Extract JSON from code block
                    json_str = json_match.group(1).strip()
                    result = json.loads(json_str)
                else:
                    # Try parsing directly
                    result = json.loads(result_text)

                # Ensure all required fields are present
                return {
                    "field_name": field_name,
                    "answer": result.get("answer", ""),
                    "confidence": result.get("confidence", 0.0),
                    "reason": result.get("reason", ""),
                    "clause_text": result.get("clause_text", ""),
                    "reference_section": result.get("reference_section", ""),
                    "summary": result.get("summary", ""),
                    "fields": result.get("fields", [])
                }
            except json.JSONDecodeError as je:
                logger.error(
                    f"Failed to parse GPT response as JSON: {str(je)}")
                return {
                    "field_name": field_name,
                    "answer": "Error parsing response",
                    "confidence": 0.0,
                    "reason": f"JSON parsing error: {str(je)}",
                    "clause_text": "",
                    "reference_section": "",
                    "summary": "",
                    "fields": []
                }

        except Exception as e:
            logger.error(
                f"Error calling GPT for clause {field_name}: {str(e)}")
            return {
                "field_name": field_name,
                "answer": f"Error: {str(e)}",
                "confidence": 0.0,
                "reason": "API error",
                "clause_text": "",
                "reference_section": "",
                "summary": "",
                "fields": []
            }

    def process_clause(self, clause: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single termination clause"""
        field_name = clause.get("field_name", "")
        logger.info(f"Processing termination clause: {field_name}")

        try:
            # Get relevant context for the clause using both queries
            context_text = self.get_relevant_chunks(clause)

            # Process the clause and all its fields in one GPT call
            result = self.extract_field_value_with_gpt(clause, context_text)

            logger.info(f"Final Result: {result}")

            # Check if the answer indicates this clause exists
            clause_exists = (
                result and
                "answer" in result and
                str(result.get("answer", "")).lower() not in [
                    "not found", "false", "no"]
            )

            return {
                "clause": field_name,
                "exists": clause_exists,
                "answer": result.get("answer", ""),
                "confidence": result.get("confidence", 0.0),
                "reason": result.get("reason", ""),
                "summary": result.get("summary", ""),
                "clause_text": result.get("clause_text", ""),
                "reference_section": result.get("reference_section", ""),
                "fields": result.get("fields", [])
                # "fields": result.get("fields", []) if clause_exists else []
            }

        except Exception as e:
            logger.error(f"Error processing clause {field_name}: {str(e)}")
            return {
                "clause": field_name,
                "exists": False,
                "error": str(e),
                "fields": []
            }

    def process_all_clauses(self) -> Dict[str, Any]:
        """Process all termination clauses using thread pool"""
        try:
            # Load the termination clauses
            clauses = self.load_termination_clauses()

            # Initialize results
            self.results = {
                "processed_at": datetime.datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
                "clauses": {},
                "processing_stats": {
                    "total_clauses": len(clauses),
                    "successful_clauses": 0,
                    "failed_clauses": 0,
                    "processing_time": 0
                }
            }

            start_time = time.time()

            # Create futures for each clause
            futures = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                for clause in clauses:
                    future = executor.submit(self.process_clause, clause)
                    futures.append((clause.get("field_name", ""), future))

                # Process results as they complete
                for field_name, future in futures:
                    try:
                        clause_result = future.result()
                        self.results["clauses"][field_name] = clause_result

                        # Update stats
                        if clause_result.get("exists", False):
                            self.results["processing_stats"]["successful_clauses"] += 1
                        else:
                            self.results["processing_stats"]["failed_clauses"] += 1

                        logger.info(
                            f"Completed processing clause: {field_name}")
                    except Exception as e:
                        logger.error(
                            f"Error processing clause {field_name}: {str(e)}")
                        self.results["clauses"][field_name] = {
                            "clause": field_name,
                            "exists": False,
                            "error": str(e),
                            "fields": []
                        }
                        self.results["processing_stats"]["failed_clauses"] += 1

            # Calculate total processing time
            end_time = time.time()
            processing_time = end_time - start_time
            self.results["processing_stats"]["processing_time"] = processing_time

            logger.info(
                f"Processing completed in {processing_time:.2f} seconds")
            logger.info(
                f"Successful clauses: {self.results['processing_stats']['successful_clauses']}")
            logger.info(
                f"Failed clauses: {self.results['processing_stats']['failed_clauses']}")

            # Save results to JSON file
            self._save_results()

            return self.results

        except Exception as e:
            logger.error(f"Error in process_all_clauses: {str(e)}")
            return {
                "processed_at": datetime.datetime.now().isoformat(),
                "error": str(e),
                "clauses": {},
                "processing_stats": {
                    "total_clauses": 0,
                    "successful_clauses": 0,
                    "failed_clauses": 0,
                    "processing_time": 0
                }
            }

    def _save_results(self):
        """Save results to JSON file with processing stats"""
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        utc_now = datetime.datetime.now(pytz.UTC)

        # Convert to IST
        ist = pytz.timezone('Asia/Kolkata')
        ist_now = utc_now.astimezone(ist)

        # Format timestamp with Indian time
        timestamp = ist_now.strftime("%d-%m-%y_%H-%M-%S")
        filename = f"termination_clauses_results_{timestamp}.json"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to file: {filename}")

            # Also save a summary file
            summary_filename = f"termination_clauses_summary_{timestamp}.txt"
            with open(summary_filename, "w", encoding="utf-8") as f:
                f.write("Processing Summary\n")
                f.write("=================\n\n")
                f.write(
                    f"Total clauses processed: {self.results['processing_stats']['total_clauses']}\n")
                f.write(
                    f"Successful clauses: {self.results['processing_stats']['successful_clauses']}\n")
                f.write(
                    f"Failed clauses: {self.results['processing_stats']['failed_clauses']}\n")
                f.write(
                    f"Total processing time: {self.results['processing_stats']['processing_time']:.2f} seconds\n")
                f.write("\nProcessed Clauses:\n")
                for clause_name, clause_data in self.results["clauses"].items():
                    status = "✓" if clause_data.get("exists", False) else "✗"
                    f.write(f"{status} {clause_name}\n")

            logger.info(f"Summary saved to file: {summary_filename}")

        except Exception as e:
            logger.error(f"Error saving results to file: {str(e)}")


def main():
    """Main function to run the script"""
    logger.info("Starting termination clause processing")

    # Create processor with specified number of workers
    # Adjust number of workers as needed
    processor = TerminationClauseProcessor(max_workers=8)
    results = processor.process_all_clauses()

    logger.info(
        f"Processing complete. Found {len(results.get('clauses', {}))} clauses.")
    logger.info(
        f"Processing time: {results.get('processing_stats', {}).get('processing_time', 0):.2f} seconds")


if __name__ == "__main__":
    main()
