import os
import json
import boto3
import requests
from openai import OpenAI
from dotenv import load_dotenv
import tempfile
import logging
from datetime import datetime
import re
# Configure logging
log_filename = f'pdf_processor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
logger.info("OpenAI client initialized")


def download_pdf_from_s3(url):
    """Download PDF from S3 URL and return the file path"""
    logger.info(f"Attempting to download PDF from: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        logger.debug(
            f"PDF download successful. Content length: {len(response.content)} bytes")
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(response.content)
        temp_file.close()
        logger.info(f"PDF saved to temporary file: {temp_file.name}")
        return temp_file.name
    else:
        error_msg = f"Failed to download PDF: {response.status_code}"
        logger.error(error_msg)
        raise Exception(error_msg)


def upload_to_openai(file_path):
    """Upload PDF to OpenAI and return the file ID"""
    logger.info(f"Attempting to upload PDF to OpenAI: {file_path}")
    try:
        with open(file_path, 'rb') as file:
            response = client.files.create(
                file=file,
                purpose='assistants'
            )
            logger.info(
                f"PDF successfully uploaded to OpenAI. File ID: {response.id}")
            return response.id
    except Exception as e:
        error_msg = f"Failed to upload PDF to OpenAI: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def get_schema_from_s3(url):
    """Download and parse JSON schema from S3"""
    logger.info(f"Attempting to download schema from: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        schema = response.json()
        logger.info(
            f"Schema downloaded successfully. Number of sections: {len(schema)}")
        logger.debug(
            f"Schema structure: {json.dumps(list(schema.keys()), indent=2)}")
        return schema
    else:
        error_msg = f"Failed to download schema: {response.status_code}"
        logger.error(error_msg)
        raise Exception(error_msg)


def process_section_with_gpt(section_data, file_id, section_name):
    """Process a section with GPT and return the enhanced data"""

    logger.info(f"Processing section: {section_name}")

    # Create a prompt for GPT
    prompt = f"""You are a highly skilled legal AI assistant specializing in analyzing Mergers & Acquisitions (M&A) agreements.  
Use the provided schema fields, including their **definitions, purposes, and instructions**, to extract precise answers directly from the attached M&A legal document.

Section Field Data:  
{json.dumps(section_data, indent=2)}

**Extraction Guidelines:**  
- Carefully read the 'definition', 'purpose', 'rag_note', and 'instructions' for each field.  
- Follow the 'instructions' field strictly when determining how to extract the answer.  
- If the value is explicitly stated, provide the exact text as it appears in the document.  
- If the value is implied but can be confidently inferred based on standard M&A language, provide the inferred value.  
- If the value is not found, return `"answer": "Not found"`.

**Formatting Rules (Strict):**  
- Return all field results as a **single JSON array** using the following structure:  

[
  {{
    "field_name": "<field_name_value>",
    "answer": "<extracted_or_inferred_value>",
    "confidence": "<confidence_score between 0 and 1>"
  }},
  ...
]

- **Important:**  
  - Return only **one JSON array** containing all objects.  
  - Do **NOT** return multiple separate JSON objects.  
  - Do **NOT** include markdown, triple backticks, or any explanatory text.  
  - Ensure the JSON is syntactically valid and directly parsable.

**Correct Example Response:**

[
  {{
    "field_name": "extension_exercise_mechanism",
    "answer": "The term may be extended by Parent upon written notice to the Company no later than 10 days prior to the expiration of the current term.",
    "confidence": "1"
  }},
  {{
    "field_name": "termination_rights_post_outside_date",
    "answer": "Either party may terminate without penalty after the Outside Date.",
    "confidence": "1"
  }}
]

If any value is not found:

[
  {{
    "field_name": "maximum_extended_outside_date",
    "answer": "Not found",
    "confidence": "0"
  }}
]

Respond strictly in this JSON array format. Do not include any other text or formatting.
"""

    logger.debug(f"Generated prompt for section {section_name}")

    try:
        # Call GPT
        logger.info(f"Sending request to GPT for section: {section_name}")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {
                                "file_id": file_id
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        logger.debug(f"Received response from GPT for section {section_name}")
        logger.debug(f"Response: {response}")
        # Parse the response and return the enhanced JSON
        try:
            response_text = response.choices[0].message.content

            markdown_match = re.search(
                r"```(?:json)?\s*([\s\S]+?)\s*```", response_text)
            if markdown_match:
                # Extract the JSON content from the markdown code block
                json_content = markdown_match.group(1).strip()
                parsed_response = json.loads(json_content)
            else:
                # Try parsing directly if not in markdown format
                parsed_response = json.loads(
                    response.choices[0].message.content)

            logger.info(f"Successfully processed section: {section_name}")
            return parsed_response
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse GPT response for section {section_name}: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "raw_response": response.choices[0].message.content}
    except Exception as e:
        error_msg = f"Error processing section {section_name}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


def main():
    logger.info("Starting PDF processing workflow")

    # URLs
    pdf_url = "https://test-data-parsed.s3.eu-north-1.amazonaws.com/deal_pdfs/azek_co_inc_2025-03-24.pdf"
    schema_url = "https://mna-docs.s3.eu-north-1.amazonaws.com/clauses_category_template/schema_by_summary_sections.json"

    try:
        # Download and process PDF
        logger.info("Step 1: Downloading PDF")
        pdf_path = download_pdf_from_s3(pdf_url)

        logger.info("Step 2: Uploading PDF to OpenAI")
        # file_id = upload_to_openai(pdf_path)
        file_id = "file-2Npy7gVBRWgqA8euekgQSn"
        logger.info(f"PDF uploaded to OpenAI with file_id: {file_id}")

        # Get schema
        logger.info("Step 3: Downloading schema")
        schema = get_schema_from_s3(schema_url)

        # Process each section
        logger.info("Step 4: Processing sections")
        processed_sections = {}
        for section_name, section_data in schema.items():
            logger.info(f"Processing section: {section_name}")
            if section_name == "Clean_Room_Agreement":
                processed_section = process_section_with_gpt(
                    section_data, file_id, section_name)
                processed_sections[section_name] = processed_section

                # Save each section's response to a separate JSON file
                output_file = f"processed_{section_name}.json"
                logger.info(f"Saving section response to: {output_file}")
                with open(output_file, 'w') as f:
                    json.dump(processed_section, f, indent=2)
                    logger.info(f"Successfully saved section to {output_file}")

            else:
                logger.info(f"Skipping section: {section_name}")

        # Save complete response
        logger.info("Step 5: Saving complete response")
        with open('complete_processed_response.json', 'w') as f:
            json.dump(processed_sections, f, indent=2)
        logger.info("Successfully saved complete processed response")

        # Clean up temporary PDF file
        logger.info("Step 6: Cleaning up temporary files")
        os.unlink(pdf_path)
        logger.info("Temporary PDF file removed")

        logger.info("PDF processing workflow completed successfully")

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        logger.error(error_msg)
        raise


if __name__ == "__main__":
    main()
