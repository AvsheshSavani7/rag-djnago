from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .utils import call_node_api
from rest_framework.parsers import MultiPartParser
import pandas as pd
import os
import logging
import tempfile
import requests
import json
from document_processor.services import DocumentProcessingService
from django.conf import settings
import math
from openai import OpenAI
from urllib.parse import urlparse
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Create your views here.


class ProcessView(APIView):
    parser_classes = [MultiPartParser]

    def sanitize_row_dict(row):
        return {
            k: (None if isinstance(v, float) and (
                math.isnan(v) or math.isinf(v)) else v)
            for k, v in row.items()
        }

    def post(self, request, format=None):
        try:

            file = request.FILES.get("file")
            if not file:
                return Response({"error": "No file uploaded."}, status=400)

            if not file.name.endswith((".xlsx", ".xls")):
                return Response({"error": "Invalid file format."}, status=400)

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                for chunk in file.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name

            # Read Excel using pandas
            sheet_name = request.data.get("sheetName", 0)
            # Use dtype parameter to ensure CIK Number is read as string
            print(f"sheet_name: {sheet_name}")
            df = pd.read_excel(tmp_path, sheet_name=sheet_name,
                               dtype={"CIK Number": str})

            print(f"df: {df}")
            os.unlink(tmp_path)

            # Initialize document processing service
            doc_processor = DocumentProcessingService()

            status_updates = []

            for _, row in df.iterrows():

                name = str(row.get("Target Name", "")).strip()
                cik = str(row.get("CIK Number", "")).strip()

                result = {
                    "name": name,
                    "cik": cik,
                    "status": False,
                    "reason": "",
                    "processed_files": []
                }

                if not name:
                    result["reason"] = "Missing company name"
                    status_updates.append(result)
                    continue

                if not cik or cik == "Not Found":
                    result["reason"] = "Invalid or missing CIK"
                    status_updates.append(result)
                    continue

                try:
                    # If Node.js API is not available, create a test job directly

                    # Normal flow - call Node API
                    # Convert pandas Series to dict before sending to API
                    # row_dict = row.to_dict()
                    row_dict = ProcessView.sanitize_row_dict(row)
                    print(f"row_dict: {row_dict}")
                    # Call Node API to get deal data
                    response = call_node_api(
                        endpoint="deal/process-single-deals-rag",  # Remove leading slash
                        method="POST",
                        # Send as list of dict with the correct parameter name 'rows'
                        data={"rows": row_dict}
                    )
                    print("response", response)
                    # Check if response is a list of deals
                    if isinstance(response.get("data"), list):
                        logger.info(
                            f"Received {len(response.get('data'))} deals from Node API")

                        for item in response.get("data"):
                            file_url = item.get("file_url")
                            deal_id = item.get("deal_id")

                            if file_url and deal_id:
                                logger.info(
                                    f"Processing document: {file_url} for deal ID {deal_id}")

                                try:
                                    # Validate file_url format - simple check
                                    if not isinstance(file_url, str) or not file_url.startswith("http"):
                                        raise ValueError(
                                            f"Invalid file_url format: {file_url}")

                                    # Process document directly using the service
                                    process_result = doc_processor.process_document(
                                        file_url=file_url,
                                        # file_url="https://rag-mna.s3.eu-north-1.amazonaws.com/parsed_jsons/spirit_airlines__inc__2022-07-28_original.json",
                                        deal_id=deal_id
                                    )

                                    # Process the result
                                    if isinstance(process_result, dict):
                                        result["processed_files"].append({
                                            "deal_id": deal_id,
                                            "file_url": file_url,
                                            "status": process_result.get("status", "error"),
                                            "message": process_result.get("message", "Unknown result")
                                        })

                                        # If at least one file was processed successfully
                                        if process_result.get("status") == "success":
                                            result["status"] = True
                                    else:
                                        # Handle unexpected result type
                                        logger.error(
                                            f"Unexpected process_result type: {type(process_result)}")
                                        result["processed_files"].append({
                                            "deal_id": deal_id,
                                            "file_url": file_url,
                                            "status": "error",
                                            "message": f"Unexpected result type: {type(process_result)}"
                                        })

                                except Exception as e:
                                    logger.error(
                                        f"Error processing document {file_url}: {str(e)}")
                                    result["processed_files"].append({
                                        "deal_id": deal_id,
                                        "file_url": file_url,
                                        "status": "error",
                                        "message": f"Processing error: {str(e)}"
                                    })
                            else:
                                logger.warning(
                                    f"Missing file_url or deal_id in item: {item}")
                    else:
                        # Handle non-list response
                        result["status"] = response.get("success", False)
                        result["message"] = response.get(
                            "message", "No deals found")

                except Exception as e:
                    logger.error(f"Error processing {name}: {e}")
                    result["reason"] = str(e)

                status_updates.append(result)

            return Response({
                "message": "✅ File processed",
                "total": len(status_updates),
                "results": status_updates
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"❌ Error in ProcessView: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _check_node_api(self):
        """Check if Node.js API is accessible"""
        try:
            # Try multiple endpoints in case health endpoint doesn't exist
            endpoints = ['health', '', 'status', 'items']

            for endpoint in endpoints:
                try:
                    # Construct URL with or without the endpoint
                    if endpoint:
                        url = f"{settings.NODE_API_BASE_URL}/{endpoint}"
                    else:
                        url = f"{settings.NODE_API_BASE_URL.rstrip('/')}"

                    logger.info(
                        f"Attempting to connect to Node.js API at: {url}")
                    response = requests.get(url, timeout=3)

                    # Any successful response means the API is up
                    if 200 <= response.status_code < 400:
                        logger.info(
                            f"Successfully connected to Node.js API at {url}")
                        return {"connected": True, "message": f"Node.js API is available at {url}"}
                except Exception as e:
                    logger.warning(f"Failed to connect to {url}: {e}")
                    continue

            # If we get here, all attempts failed
            logger.error(
                f"Could not connect to Node.js API at {settings.NODE_API_BASE_URL} on any endpoint")
            return {
                "connected": False,
                "message": f"Could not connect to Node.js API at {settings.NODE_API_BASE_URL}. Is the server running?"
            }
        except Exception as e:
            logger.error(f"Unexpected error checking Node.js API: {e}")
            return {"connected": False, "message": str(e)}


class ItemsListView(APIView):
    """
    API view to handle listing items from Node.js API
    """

    def get(self, request, format=None):
        try:
            # Get query parameters from request
            params = request.query_params.dict()

            # Call Node.js API to get items
            response_data = call_node_api('items', method='GET', params=params)
            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error in ItemsListView: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AnnouncementView(APIView):
    """
    API endpoint to handle announcement data and forward it to Node API
    """

    def post(self, request, format=None):
        try:
            # Log the incoming request data
            logger.info(f"Received request data: {request.data}")

            # Extract data from request
            announce_data = request.data.get('announce_data')
            target_name = request.data.get('target_name')
            acquirer_name = request.data.get('acquired_name')
            target_cik = request.data.get('target_cik')

            # Log the extracted values
            logger.info(
                f"Extracted values: announce_data={announce_data}, target_cik={target_cik}")

            # Validate required fields
            if not all([announce_data, target_cik]):
                missing_fields = []
                if not announce_data:
                    missing_fields.append('announce_data')

                if not target_cik:
                    missing_fields.append('target_cik')

                return Response({
                    "error": f"Missing required fields: {', '.join(missing_fields)}"
                }, status=status.HTTP_400_BAD_REQUEST)

            # Prepare data for Node API
            data = {
                "announce_data": announce_data,
                "target_cik": target_cik
            }

            # Log the data being sent to Node API
            logger.info(f"Sending data to Node API: {data}")

            # Call Node API
            response = call_node_api(
                endpoint="deal/get-deal-sec-urls",
                method="POST",
                data=data
            )

            # Log the response from Node API
            logger.info(f"Received response from Node API: {response}")

            return Response(response, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error in AnnouncementView: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AnnouncementWithUrlView(APIView):
    """
    API endpoint to handle announcement data with URL and OpenAI integration
    """

    def extract_missing_fields_with_openai(self, url, existing_data):
        """
        Use OpenAI to extract missing fields from the document URL
        """
        try:
            # Prepare the prompt for OpenAI
            missing_fields = []
            if not existing_data.get('target_cik'):
                missing_fields.append('target_cik')
            if not existing_data.get('announce_data'):
                missing_fields.append('announce_data')
            if not existing_data.get('target_name'):
                missing_fields.append('target_name')
            if not existing_data.get('acquirer_name'):
                missing_fields.append('acquirer_name')
            if not existing_data.get('acquirer_cik'):
                missing_fields.append('acquirer_cik')

            if not missing_fields:
                return existing_data

            logger.info(f"Extracting missing fields: {missing_fields}")

            # Get document content directly from URL
            try:
                headers = {
                    "User-Agent": "Avshesh/1.0 (avshesh.savani@teqnodux.com)",
                    "Accept-Language": "en-US,en;q=0.9"
                }
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()  # Raise exception for bad status codes
                # limit to first 100,000 characters
                html_content = response.text[:100000]

                # Parse HTML to clean text
                soup = BeautifulSoup(html_content, "html.parser")
                document_content = soup.get_text(separator="\n")

            except Exception as e:
                logger.error(f"Error fetching document content: {e}")
                document_content = ""

            fields_prompt = ", ".join(missing_fields)

            # Create field-specific instructions based on what's missing
            field_instructions = []
            if 'announce_data' in missing_fields:
                field_instructions.append(
                    "For `announce_data`: Look for the date the deal was publicly announced, usually in the first paragraph or preamble (e.g., 'dated as of...'). Format: YYYY-MM-DD")

            if 'target_cik' in missing_fields:
                field_instructions.append(
                    "For `target_cik`: The SEC CIK of the Target (the company being acquired, usually defined as the 'Company'). "
                    "If not present in the document text, return an empty string."
                    "Rule: add zeros to the left of the CIK to make it 10 digits long."
                )

            if 'target_name' in missing_fields:
                field_instructions.append(
                    "For `target_name`: The company being acquired (“Target/Company”). "
                    "Rule: If one party is an SEC registrant/public company (has a CIK) and the other is a private “Holdings”/buyer entity, the SEC registrant/public company is the Target.")
            if 'acquirer_name' in missing_fields:
                field_instructions.append(
                    "For `acquirer_name`: The company buying/acquiring the Target (often “Parent/Buyer”)."
                    "Rule: If there is a private “Holdings” entity and a public registrant, the Holdings entity is the Acquirer.")
            if 'acquirer_cik' in missing_fields:
                field_instructions.append(
                    "For `acquirer_cik`: The SEC CIK of the Acquirer/Buyer only if the acquirer is an SEC registrant; otherwise return an empty string. Do NOT copy the target’s CIK."
                    "Rule: add zeros to the left of the CIK to make it 10 digits long."
                )
            instructions_text = "\n".join(field_instructions)

            print(f"instructions_text: {instructions_text}")

            prompt = f"""Extract the following information from the merger agreement document: {fields_prompt}
            
            {instructions_text}

            Document url: {url}
            
            Here is the beginning of the document content:
            {document_content}

            Return ONLY a JSON object with the extracted fields. Example:
            {{
                "target_cik": "{{target_cik}}",
                "announce_data": "{{announce_data}}",
                "target_name": "{{target_name}}",
                "acquirer_name": "{{acquirer_name}}",
                "acquirer_cik": "{{acquirer_cik}}"
            }}

            Only include the fields that were requested. Do not include any other text or explanation.
            """

            # Call OpenAI API directly
            try:
                completion = openai_client.chat.completions.create(
                    model="gpt-4.1",  # or your preferred model
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that extracts specific information from merger agreement documents. You only return JSON objects with the requested fields."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )

                print(f"prompt: {prompt}")
                print(f"completion: {completion}")

                logger.info(f"completion: {completion}")
                # Extract the JSON response, handling code blocks if present
                content = completion.choices[0].message.content.strip()

                print(f"content: {content}")
                logger.info(f"content: {content}")

                # Remove code block markers if present
                if content.startswith('```') and content.endswith('```'):
                    # Remove first line (```json or ```) and last line (```)
                    content_lines = content.split('\n')
                    content = '\n'.join(content_lines[1:-1])

                # Parse the JSON
                extracted_data = json.loads(content)

                print(f"extracted_data: {extracted_data}")

                return extracted_data

            except Exception as e:
                logger.error(f"Error calling OpenAI API: {e}")
                logger.error(
                    f"Raw content received: {completion.choices[0].message.content}")
                raise Exception(
                    f"Could not extract information using OpenAI: {str(e)}")

        except Exception as e:
            logger.error(f"Error extracting fields with OpenAI: {e}")
            raise

    def post(self, request, format=None):
        try:
            # Log the incoming request data
            logger.info(f"Received request data: {request.data}")

            # Extract data from request
            url = request.data.get('url')
            sec_filing_id = request.data.get('sec_filing_id')
            print(f"url:1 {url}")
            if not url:
                return Response({
                    "error": "URL is required"
                }, status=status.HTTP_400_BAD_REQUEST)

            # Extract CIK from SEC URL if available
            extracted_cik = None
            if url and '/data/' in url:
                try:
                    # Extract CIK from URL path after /data/
                    url_parts = url.split('/data/')
                    if len(url_parts) > 1:
                        cik_part = url_parts[1].split('/')[0]
                        # Pad with leading zeros to make it 10 digits
                        extracted_cik = cik_part.zfill(10)
                        logger.info(f"Extracted CIK from URL: {extracted_cik}")
                except Exception as e:
                    logger.warning(f"Could not extract CIK from URL: {e}")

            # Extract other fields
            data = {
                "target_cik": request.data.get('target_cik') or extracted_cik,
                "announce_data": request.data.get('announce_data'),
                "target_name": request.data.get('target_name'),
                "acquirer_name": request.data.get('acquired_name'),
                "url": url,
                "sec_filing_id": request.data.get('sec_filing_id') if request.data.get('sec_filing_id') else None,
                "acquirer_cik": ""
            }

            # Check if we have all required fields
            has_all_fields = all([
                data.get('target_cik'),
                data.get('announce_data'),
                data.get('target_name'),
                data.get('acquirer_name'),
                data.get('acquirer_cik')

            ])

            doc_processor = DocumentProcessingService()
            doc_processor._send_sec_filing_event(sec_filing_id, "In Progress")

            # If not all fields are present, try to extract them using OpenAI
            if not has_all_fields:
                logger.info(
                    "Not all fields present, attempting to extract using OpenAI")
                try:
                    extracted_data = self.extract_missing_fields_with_openai(
                        url, data)
                    print(f"extracted_data: {extracted_data}")

                    # Merge extracted data with existing data (only add missing fields)
                    for key, value in extracted_data.items():
                        # Only add if field is missing and has a value
                        if not data.get(key) and value:
                            data[key] = value
                            logger.info(f"Added missing field {key}: {value}")

                    print(f"merged data: {data}")
                except Exception as e:
                    logger.error(f"Error extracting missing fields: {e}")
                    return Response({
                        "error": f"Could not extract missing fields: {str(e)}"
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Validate that we now have all required fields
            missing_fields = []
            if not data.get('target_cik'):
                missing_fields.append('target_cik')
            if not data.get('announce_data'):
                missing_fields.append('announce_data')
            if not data.get('target_name'):
                missing_fields.append('target_name')
            if not data.get('acquirer_name'):
                missing_fields.append('acquirer_name')

            if missing_fields:
                doc_processor._send_sec_filing_event(
                    sec_filing_id, "Fail", f"Could not obtain all required fields: {', '.join(missing_fields)}")
                return Response({
                    "error": f"Could not obtain all required fields: {', '.join(missing_fields)}"
                }, status=status.HTTP_400_BAD_REQUEST)

            print(f"data1: {data}")

            # Call Node API with complete data
            response = call_node_api(
                endpoint="deal/process-with-url",  # Using the correct endpoint for URL processing
                method="POST",
                data={
                    "url": url,
                    "target_cik": data.get('target_cik'),
                    "announce_data": data.get('announce_data'),
                    "target_name": data.get('target_name'),
                    "acquired_name": data.get('acquirer_name'),
                    "sec_filing_id": data.get('sec_filing_id') if data.get('sec_filing_id') else None,
                    "acquirer_cik": data.get('acquirer_cik')
                }
            )

            # Check if we got a successful response with jsonUrl
            if response.get('status') and response.get('data', {}).get('jsonUrl'):
                # Initialize document processing service
                doc_processor = DocumentProcessingService()

                print(f"response: {response}")

                # Process the document using the JSON URL
                process_result = doc_processor.process_document(
                    file_url=response['data']['jsonUrl'],
                    # file_url="https://rag-mna.s3.eu-north-1.amazonaws.com/parsed_jsons/spirit_airlines__inc__2022-07-28_original.json",
                    # file_url="https://rag-embedding.s3.eu-north-1.amazonaws.com/parsed_jsons/spirit_airlines__inc__2022-07-28_original.json",
                    deal_id=response['data']['deal_id'],
                    sec_filing_id=data.get('sec_filing_id')
                )

                # Add processing result to response
                response['processing_result'] = process_result

            return Response(response, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error in AnnouncementWithUrlView: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
