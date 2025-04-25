from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
import logging
import traceback
from bson import ObjectId
import threading
import concurrent.futures
import json

from .models import ProcessingJob
from .serializers import (
    ProcessingJobSerializer,
    FileProcessRequestSerializer
)
from .services import FlattenProcessor, EmbeddingService, S3Service, ChatWithAIService, SummaryGenerationService

logger = logging.getLogger(__name__)

# Create a ThreadPoolExecutor for background tasks
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)


def process_embeddings(job_id, flattened_json_url):
    """Background task to process embeddings"""
    print(
        f"Starting embedding process for job {job_id} with URL: {flattened_json_url}")
    try:
        # Get the job
        object_id = ObjectId(job_id)
        job = ProcessingJob.objects.get(_id=object_id)
        print(f"Found job in database: {job}")

        # Update job status to processing
        job.update_embedding_status('PROCESSING')
        print("Updated job status to PROCESSING")

        # Download flattened JSON
        s3_service = S3Service()
        print(f"Downloading flattened JSON from URL: {flattened_json_url}")
        chunks = s3_service.download_from_url(flattened_json_url)
        print(f"Downloaded {len(chunks)} chunks")

        # Process embeddings
        print("Initializing embedding service")
        embedding_service = EmbeddingService()
        print(f"Starting embedding creation for {len(chunks)} chunks")
        result = embedding_service.process_chunks(chunks, str(job_id))
        print(f"Embedding completed with result: {result}")

        # Update job status to completed
        job.update_embedding_status('COMPLETED')
        print(f"Updated job status to COMPLETED")
        logger.info(f"Embedding completed for job {job_id}: {result}")

    except Exception as e:
        print(f"Error processing embeddings: {str(e)}")
        logger.error(f"Error processing embeddings for job {job_id}: {str(e)}")
        logger.error(traceback.format_exc())

        # Update job status to failed
        try:
            object_id = ObjectId(job_id)
            job = ProcessingJob.objects.get(_id=object_id)
            job.update_embedding_status('FAILED', str(e))
            print(f"Updated job status to FAILED: {str(e)}")
        except Exception as inner_e:
            print(f"Error updating job status: {str(inner_e)}")
            logger.error(f"Error updating job status: {str(inner_e)}")


class ProcessFileView(APIView):
    """
    API endpoint to process a file from a URL and save to S3
    """
    authentication_classes = []
    permission_classes = []

    def post(self, request, format=None):

        file_url = request.data.get('file_url')
        deal_id = request.data.get('deal_id')
        embed_data = request.data.get('embed_data', True)  # Default to True

        print("deal_id, file_url", deal_id, file_url)
        if not deal_id:
            return Response({"error": "deal_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Find the existing job by _id (convert string to ObjectId)
        try:
            object_id = ObjectId(deal_id)
            job = ProcessingJob.objects.get(_id=object_id)
            print("job", job)
        except ProcessingJob.DoesNotExist:
            return Response({"error": f"No processing job found for deal_id {deal_id}"},
                            status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": f"Invalid deal_id format: {str(e)}"},
                            status=status.HTTP_400_BAD_REQUEST)

        try:
            # Initialize the processor
            processor = FlattenProcessor(file_url=file_url)

            # Process the file
            result = processor.process()
            print("result", result)

            # Update the job with results
            job.flattened_json_url = result.get(
                'flattened_json_url')  # Update the flattened URL
            job.save()

            # Start embedding process in background if requested
            if embed_data:
                # Start embedding task in the executor
                executor.submit(process_embeddings, str(
                    job._id), job.flattened_json_url)
                print(
                    f"Submitted embedding task for job {job._id} to executor")

                # Return response with embedding status
                return Response({
                    'deal_id': str(job._id),
                    'flattened_json_url': job.flattened_json_url,
                    'embedding_status': 'PROCESSING',
                    'message': 'File processed successfully. Embeddings are being generated in the background.'
                }, status=status.HTTP_200_OK)

            # Return response without embedding
            return Response({
                'deal_id': str(job._id),
                'flattened_json_url': job.flattened_json_url
            }, status=status.HTTP_200_OK)

        except Exception as e:
            # Log the error
            logger.error(f"Error processing file: {str(e)}")
            logger.error(traceback.format_exc())

            # Update the job with error
            job.error_message = str(e)
            job.save()

            # Return error response
            return Response({
                'error': str(e),
                'deal_id': str(job._id)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ProcessingJobDetailView(APIView):
    """
    API endpoint to get details of a processing job
    """
    authentication_classes = []
    permission_classes = []

    def get(self, request, id, format=None):
        try:
            object_id = ObjectId(id)
            job = get_object_or_404(ProcessingJob, _id=object_id)
            serializer = ProcessingJobSerializer(job)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": f"Invalid ID format: {str(e)}"},
                            status=status.HTTP_400_BAD_REQUEST)


class ProcessEmbeddingsView(APIView):
    """
    API endpoint to process embeddings for an existing job
    """
    authentication_classes = []
    permission_classes = []

    def post(self, request, format=None):
        deal_id = request.data.get('deal_id')

        if not deal_id:
            return Response({"error": "deal_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Find the existing job
        try:
            object_id = ObjectId(deal_id)
            job = ProcessingJob.objects.get(_id=object_id)
        except ProcessingJob.DoesNotExist:
            return Response({"error": f"No processing job found for deal_id {deal_id}"},
                            status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": f"Invalid deal_id format: {str(e)}"},
                            status=status.HTTP_400_BAD_REQUEST)

        # Check if flattened file URL exists
        if not job.flattened_json_url:
            return Response({"error": "Job does not have a flattened JSON URL. Process the file first."},
                            status=status.HTTP_400_BAD_REQUEST)

        # Start embedding task in the executor
        executor.submit(process_embeddings, str(
            job._id), job.flattened_json_url)
        print(f"Submitted embedding task for job {job._id} to executor")

        # Update status
        job.update_embedding_status('PROCESSING')

        # Return response
        return Response({
            'deal_id': str(job._id),
            'embedding_status': 'PROCESSING',
            'message': 'Embeddings are being generated in the background.'
        }, status=status.HTTP_200_OK)


class ListAllDealsView(APIView):
    """
    API endpoint to get a list of all deals
    """
    authentication_classes = []
    permission_classes = []

    def get(self, request, format=None):
        try:
            # Get all deals from the database
            jobs = ProcessingJob.objects.all().order_by('-createdAt')

            # Serialize the deals
            serializer = ProcessingJobSerializer(jobs, many=True)

            # Return the serialized data
            return Response({
                'deals': serializer.data,
                'total': len(serializer.data)
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error fetching deals: {str(e)}")
            logger.error(traceback.format_exc())

            # Return error response
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PineconeVectorListView(APIView):
    """
    API endpoint to list all Pinecone vectors for a specific deal
    """
    authentication_classes = []
    permission_classes = []

    def get(self, request, deal_id=None, format=None):
        if not deal_id:
            return Response({"error": "deal_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Initialize the embedding service
            embedding_service = EmbeddingService()

            # Verify that the deal exists
            try:
                object_id = ObjectId(deal_id)
                job = ProcessingJob.objects.get(_id=object_id)
            except ProcessingJob.DoesNotExist:
                return Response({"error": f"No processing job found for deal_id {deal_id}"},
                                status=status.HTTP_404_NOT_FOUND)
            except Exception as e:
                return Response({"error": f"Invalid deal_id format: {str(e)}"},
                                status=status.HTTP_400_BAD_REQUEST)

            # Use the index.query method with a filter for deal_id and a high top_k value
            # Use a dummy query (zero vector) with high top_k to get all vectors
            # Initialize a zero vector of the right dimension
            zero_vector = [0.0] * 1536  # Dimension for text-embedding-3-small

            # Query Pinecone with a filter for the deal_id
            filter_dict = {"deal_id": str(deal_id)}

            # Set a high top_k to return many results (maximum allowed by Pinecone)
            top_k = 10000

            print(f"Fetching vectors for deal ID: {deal_id} from Pinecone")
            # Query the index
            query_response = embedding_service.index.query(
                vector=zero_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )

            # Format the results
            results = []
            for match in query_response.matches:
                # Start with basic vector info
                vector_data = {
                    "id": match.id,
                    # "score": match.score,
                    "chunk_index": match.metadata.get("chunk_index"),
                }

                # Include all metadata fields from the vector
                # This will include all structured fields added by extract_structured_metadata
                if hasattr(match, 'metadata') and match.metadata:
                    for key, value in match.metadata.items():
                        # Try to parse JSON strings back to objects for certain fields
                        if key in ["clause_tags_llm"] and isinstance(value, str):
                            try:
                                vector_data[key] = json.loads(value)
                            except json.JSONDecodeError:
                                vector_data[key] = value
                        else:
                            vector_data[key] = value

                results.append(vector_data)

            print(f"Found {len(results)} vectors for deal ID: {deal_id}")

            # Sort the results by chunk_index
            results.sort(key=lambda x: x.get("chunk_index", 0))

            # Then return the sorted results
            return Response({
                "deal_id": str(deal_id),
                "vectors": results,
                "total": len(results)
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error fetching Pinecone vectors: {str(e)}")
            logger.error(traceback.format_exc())

            # Return error response
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class UpdatePineconeVectorView(APIView):
    """
    API endpoint to update metadata for a specific vector in Pinecone
    """
    authentication_classes = []
    permission_classes = []

    def patch(self, request, vector_id=None, format=None):
        if not vector_id:
            return Response({"error": "vector_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Get the metadata to update
        metadata = request.data.get('metadata')
        if not metadata or not isinstance(metadata, dict):
            return Response({"error": "metadata object is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Initialize the embedding service
            embedding_service = EmbeddingService()

            print(f"Updating metadata for vector ID: {vector_id}")

            # Use Pinecone's update method to update just the metadata
            embedding_service.index.update(
                id=vector_id,
                set_metadata=metadata
            )

            print(f"Successfully updated metadata for vector ID: {vector_id}")

            # Return success response
            return Response({
                "vector_id": vector_id,
                "message": "Metadata updated successfully"
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error updating Pinecone vector metadata: {str(e)}")
            logger.error(traceback.format_exc())

            # Return error response
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatWithAIView(APIView):
    """
    API endpoint for chat interactions with AI using document context
    """
    authentication_classes = []
    permission_classes = []

    def post(self, request, format=None):
        # Get request parameters
        query = request.data.get('query')
        deal_id = request.data.get('deal_id')
        message_history = request.data.get('message_history', [])
        top_k = int(request.data.get('top_k', 5))
        temperature = float(request.data.get('temperature', 0.7))

        # Validate parameters
        if not query:
            return Response({"error": "query is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Initialize the chat service
        try:
            chat_service = ChatWithAIService()

            # Process the chat query
            result = chat_service.chat(
                query=query,
                deal_id=deal_id,
                message_history=message_history,
                top_k=top_k,
                temperature=temperature
            )

            # Return the result
            return Response(result, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error in ChatWithAIView: {str(e)}")
            logger.error(traceback.format_exc())

            # Return error response
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SummaryGenerationView(APIView):
    """
    API endpoint for generating document summaries using AI
    """
    authentication_classes = []
    permission_classes = []

    def post(self, request, format=None):
        # Get request parameters
        deal_id = request.data.get('deal_id')
        temperature = float(request.data.get('temperature', 0.7))

        # Validate parameters
        if not deal_id:
            return Response({"error": "deal_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Initialize the summary service
        try:
            summary_service = SummaryGenerationService()

            # Generate the summary
            result = summary_service.generate_summary(
                deal_id=deal_id,
                temperature=temperature
            )

            # Return the result
            return Response(result, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error in SummaryGenerationView: {str(e)}")
            logger.error(traceback.format_exc())

            # Return error response
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
