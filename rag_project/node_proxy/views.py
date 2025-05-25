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
from document_processor.services import DocumentProcessingService
from django.conf import settings
import math


logger = logging.getLogger(__name__)

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
            df = pd.read_excel(tmp_path, sheet_name=sheet_name,
                               dtype={"CIK Number": str})
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
