from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import logging
import traceback
from bson import ObjectId
import threading
import concurrent.futures
import json

from .models import ProcessingJob, CompanyProducts, CompetitiveAnalysis, HighValueFollowers, SearchQuery, Tweet, RedditPost
from .serializers import (
    ProcessingJobSerializer,
    FileProcessRequestSerializer,
    HighValueFollowersSerializer,
    TweetSerializer,
    RedditPostSerializer
)
from .services import FlattenProcessor, EmbeddingService, S3Service, ChatWithAIService, SummaryGenerationService
from mongoengine.errors import DoesNotExist, ValidationError

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
        job = ProcessingJob.objects.get(id=object_id)
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
            job = ProcessingJob.objects.get(id=object_id)
            job.update_embedding_status('FAILED', str(e))
            print(f"Updated job status to FAILED: {str(e)}")
        except Exception as inner_e:
            print(f"Error updating job status: {str(inner_e)}")
            logger.error(f"Error updating job status: {str(inner_e)}")


class ProcessFileView(APIView):
    """
    API endpoint to process a file from a URL and save to S3
    """

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
            job = ProcessingJob.objects.get(id=object_id)
            print("job", job)
        except DoesNotExist:
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

            if job.schema_results is not None and not isinstance(job.schema_results, dict):
                logger.warning(
                    f"⚠️ Invalid schema_results type: {type(job.schema_results)}. Resetting to empty dict.")
            else:
                job.save()

            # Start embedding process in background if requested
            if embed_data:
                # Start embedding task in the executor
                executor.submit(process_embeddings, str(
                    job.id), job.flattened_json_url)
                print(
                    f"Submitted embedding task for job {job.id} to executor")

                # Return response with embedding status
                return Response({
                    'deal_id': str(job.id),
                    'flattened_json_url': job.flattened_json_url,
                    'embedding_status': 'PROCESSING',
                    'message': 'File processed successfully. Embeddings are being generated in the background.'
                }, status=status.HTTP_200_OK)

            # Return response without embedding
            return Response({
                'deal_id': str(job.id),
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
                'deal_id': str(job.id)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ProcessingJobDetailView(APIView):
    """
    API endpoint to get details of a processing job
    """

    def get(self, request, id, format=None):
        try:
            object_id = ObjectId(id)
            job = ProcessingJob.objects.get(id=object_id)

            # Convert schema_results from a JSON string to a dictionary if applicable, ensuring valid DictField representation
            if isinstance(job.schema_results, str):
                try:
                    job.schema_results = json.loads(job.schema_results)
                except json.JSONDecodeError:
                    job.schema_results = {}

            serializer = ProcessingJobSerializer(job)
            job_data = serializer.data

            # Add product information to the job
            deal_id = job_data['id']

            # Fetch company products for this deal
            try:
                # Get products for both target and acquire companies
                company_products = CompanyProducts.objects(deal_id=deal_id)
                target_products = []
                acquire_products = []

                for cp in company_products:
                    if cp.company_type == 'target':
                        target_products = self._extract_product_names(
                            cp.products)
                    elif cp.company_type == 'acquire':
                        acquire_products = self._extract_product_names(
                            cp.products)

                # Get competitive analysis if available
                competitive_analysis = None
                try:
                    comp_analysis = CompetitiveAnalysis.objects(
                        deal_id=deal_id).first()
                    if comp_analysis:
                        competitive_analysis = {
                            'competitive_pairs': comp_analysis.competitive_pairs,
                            'analysis_timestamp': comp_analysis.analysis_timestamp.isoformat() if comp_analysis.analysis_timestamp else None,
                            'processing_status': comp_analysis.processing_status
                        }
                except Exception as e:
                    logger.warning(
                        f"Could not fetch competitive analysis for deal {deal_id}: {e}")

                # Add product information to the job
                job_data['products'] = {
                    'target_company_products': target_products,
                    'acquire_company_products': acquire_products,
                    'competitive_analysis': competitive_analysis
                }

            except Exception as e:
                logger.warning(
                    f"Could not fetch products for deal {deal_id}: {e}")
                job_data['products'] = {
                    'target_company_products': [],
                    'acquire_company_products': [],
                    'competitive_analysis': None
                }

            return Response(job_data, status=status.HTTP_200_OK)

        except (DoesNotExist, ValidationError):
            return Response({"error": "Job not found or invalid ID."}, status=status.HTTP_404_NOT_FOUND)

        except Exception as e:
            return Response({"error": f"Unexpected error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _extract_product_names(self, products):
        """Helper method to extract product names from structured product data"""
        product_names = []
        try:
            if isinstance(products, list):
                for category in products:
                    if isinstance(category, dict) and 'products' in category:
                        for product in category['products']:
                            if isinstance(product, dict) and 'name' in product:
                                product_names.append(product['name'])
                            elif isinstance(product, str):
                                product_names.append(product)
                    elif isinstance(category, str):
                        product_names.append(category)
            elif isinstance(products, str):
                product_names.append(products)
        except Exception as e:
            logger.warning(f"Error extracting product names: {e}")

        return product_names


class ProcessEmbeddingsView(APIView):
    """
    API endpoint to process embeddings for an existing job
    """

    def post(self, request, format=None):
        deal_id = request.data.get('deal_id')

        if not deal_id:
            return Response({"error": "deal_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Find the existing job
        try:
            object_id = ObjectId(deal_id)
            job = ProcessingJob.objects.get(id=object_id)
        except DoesNotExist:
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
            job.id), job.flattened_json_url)
        print(f"Submitted embedding task for job {job.id} to executor")

        # Update status
        job.update_embedding_status('PROCESSING')

        # Return response
        return Response({
            'deal_id': str(job.id),
            'embedding_status': 'PROCESSING',
            'message': 'Embeddings are being generated in the background.'
        }, status=status.HTTP_200_OK)


class ListAllDealsView(APIView):
    """
    API endpoint to get a list of all deals
    """

    def get(self, request, format=None):
        try:
            # Get all deals from the database
            jobs = ProcessingJob.objects.all().order_by('-createdAt')

            # Convert schema_results from a JSON string to a dictionary if applicable, ensuring valid DictField representation
            for job in jobs:
                if isinstance(job.schema_results, str):
                    try:
                        job.schema_results = json.loads(job.schema_results)
                    except json.JSONDecodeError:
                        job.schema_results = {}

            # Serialize the deals
            serializer = ProcessingJobSerializer(jobs, many=True)
            deals_data = serializer.data

            # Add product information to each deal
            for deal in deals_data:
                deal_id = deal['id']

                # Fetch company products for this deal
                try:
                    # Get products for both target and acquire companies
                    company_products = CompanyProducts.objects(deal_id=deal_id)
                    target_products = []
                    acquire_products = []

                    for cp in company_products:
                        if cp.company_type == 'target':
                            target_products = self._extract_product_names(
                                cp.products)
                        elif cp.company_type == 'acquire':
                            acquire_products = self._extract_product_names(
                                cp.products)

                    # Get competitive analysis if available
                    competitive_analysis = None
                    try:
                        comp_analysis = CompetitiveAnalysis.objects(
                            deal_id=deal_id).first()
                        if comp_analysis:
                            competitive_analysis = {
                                'competitive_pairs': comp_analysis.competitive_pairs,
                                'analysis_timestamp': comp_analysis.analysis_timestamp.isoformat() if comp_analysis.analysis_timestamp else None,
                                'processing_status': comp_analysis.processing_status
                            }
                    except Exception as e:
                        logger.warning(
                            f"Could not fetch competitive analysis for deal {deal_id}: {e}")

                    # Add product information to the deal
                    deal['products'] = {
                        'target_company_products': target_products,
                        'acquire_company_products': acquire_products,
                        'competitive_analysis': competitive_analysis
                    }

                except Exception as e:
                    logger.warning(
                        f"Could not fetch products for deal {deal_id}: {e}")
                    deal['products'] = {
                        'target_company_products': [],
                        'acquire_company_products': [],
                        'competitive_analysis': None
                    }

            # Return the serialized data with products
            return Response({
                'deals': deals_data,
                'total': len(deals_data)
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error fetching deals: {str(e)}")
            logger.error(traceback.format_exc())

            # Return error response
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _extract_product_names(self, products):
        """Helper method to extract product names from structured product data"""
        product_names = []
        try:
            if isinstance(products, list):
                for category in products:
                    if isinstance(category, dict) and 'products' in category:
                        for product in category['products']:
                            if isinstance(product, dict) and 'name' in product:
                                product_names.append(product['name'])
                            elif isinstance(product, str):
                                product_names.append(product)
                    elif isinstance(category, str):
                        product_names.append(category)
            elif isinstance(products, str):
                product_names.append(products)
        except Exception as e:
            logger.warning(f"Error extracting product names: {e}")

        return product_names


class PineconeVectorListView(APIView):
    """
    API endpoint to list all Pinecone vectors for a specific deal
    """

    def get(self, request, deal_id=None, format=None):
        if not deal_id:
            return Response({"error": "deal_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Initialize the embedding service
            embedding_service = EmbeddingService()

            # Verify that the deal exists
            try:
                object_id = ObjectId(deal_id)
                job = ProcessingJob.objects.get(id=object_id)
            except DoesNotExist:
                return Response({"error": f"No processing job found for deal_id {deal_id}"},
                                status=status.HTTP_404_NOT_FOUND)
            except Exception as e:
                return Response({"error": f"Invalid deal_id format: {str(e)}"},
                                status=status.HTTP_400_BAD_REQUEST)

            # Use the index.query method with a filter for deal_id and a high top_k value
            # Use a dummy query (zero vector) with high top_k to get all vectors
            # Initialize a zero vector of the right dimension
            zero_vector = [0.0] * 3072  # Dimension for text-embedding-3-small

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
            # result = summary_service.generate_summary(
            #     deal_id=deal_id,
            #     temperature=temperature
            # )
            result = summary_service.generate_summary_v2(
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


class SummaryEngineView(APIView):
    """
    API endpoint for generating document summaries using AI
    """
    # Create a ThreadPoolExecutor for background tasks
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _generate_summary_background(self, deal_id, temperature):
        """Background task to generate summary"""
        try:
            summary_service = SummaryGenerationService()
            result = summary_service.generate_summary_engine(
                deal_id=deal_id,
                temperature=temperature
            )

            # Update the job with the summary URL
            try:
                object_id = ObjectId(deal_id)
                job = ProcessingJob.objects.get(id=object_id)
                job.summary_docx_url = result
                job.summary_status = 'COMPLETED'
                job.save()
            except Exception as e:
                logger.error(f"Error updating job with summary URL: {str(e)}")

        except Exception as e:
            logger.error(f"Error in background summary generation: {str(e)}")
            logger.error(traceback.format_exc())

            # Update job status to failed
            try:
                object_id = ObjectId(deal_id)
                job = ProcessingJob.objects.get(id=object_id)
                job.summary_status = 'FAILED'
                job.error_message = str(e)
                job.save()
            except Exception as inner_e:
                logger.error(f"Error updating job status: {str(inner_e)}")

    def post(self, request, format=None):
        # Get request parameters
        deal_id = request.data.get('deal_id')
        temperature = float(request.data.get('temperature', 0.7))

        # Validate parameters
        if not deal_id:
            return Response({"error": "deal_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Get the job and update status
            object_id = ObjectId(deal_id)
            job = ProcessingJob.objects.get(id=object_id)
            job.summary_status = 'PROCESSING'
            job.save()

            # Start summary generation in background
            self.executor.submit(
                self._generate_summary_background, deal_id, temperature)

            # Return immediate response
            return Response({
                'id': str(deal_id),
                'summary_status': 'PROCESSING',
                'message': 'Summary generation started in background.'
            }, status=status.HTTP_200_OK)

        except DoesNotExist:
            return Response({"error": f"No processing job found for deal_id {deal_id}"},
                            status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error in SummaryEngineView: {str(e)}")
            logger.error(traceback.format_exc())
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class JobStatusView(APIView):
    """
    API endpoint to check the status of a processing job
    """

    def get(self, request, job_id, format=None):
        try:
            # Convert string ID to ObjectId
            object_id = ObjectId(job_id)
            job = ProcessingJob.objects.get(id=object_id)

            # Convert schema_results from a JSON string to a dictionary if needed
            if isinstance(job.schema_results, str):
                try:
                    job.schema_results = json.loads(job.schema_results)
                except json.JSONDecodeError:
                    job.schema_results = {}

            # Serialize the job data
            serializer = ProcessingJobSerializer(job)

            return Response(serializer.data, status=status.HTTP_200_OK)

        except DoesNotExist:
            return Response(
                {"error": f"No processing job found for job_id {job_id}"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Error checking job status: {str(e)}")
            logger.error(traceback.format_exc())
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class HighValueFollowersView(APIView):
    """
    API endpoint to get high value followers for a specific deal with pagination
    """

    def get(self, request, deal_id, format=None):
        try:
            # Get pagination parameters from query string
            page = int(request.query_params.get('page', 1))
            limit = int(request.query_params.get('limit', 10))
            company_handle = request.query_params.get('company_handle', '')

            # Validate pagination parameters
            if page < 1:
                page = 1
            if limit < 1 or limit > 1000:  # Set reasonable limits
                limit = 10

            # Calculate skip value for pagination
            skip = (page - 1) * limit

            # Build query filter
            query_filter = {'deal_id': deal_id}
            if company_handle:
                query_filter['company_handle'] = company_handle

            # Get total count of high value followers for this deal
            total_count = HighValueFollowers.objects(**query_filter).count()

            # Fetch high value followers with pagination, ordered by overall score
            followers = HighValueFollowers.objects(**query_filter).order_by(
                '-overall_score', '-created_at').skip(skip).limit(limit)

            # Serialize the followers
            serializer = HighValueFollowersSerializer(followers, many=True)

            # Calculate pagination metadata
            total_pages = (total_count + limit -
                           1) // limit  # Ceiling division

            # Get unique company handles for this deal
            unique_company_handles = HighValueFollowers.objects(
                deal_id=deal_id).distinct('company_handle')

            # Return response with pagination metadata
            return Response({
                'followers': serializer.data,
                'pagination': {
                    'current_page': page,
                    'total_pages': total_pages,
                    'total_count': total_count,
                    'limit': limit,
                    'has_next': page < total_pages,
                    'has_previous': page > 1
                },
                'filters': {
                    'deal_id': deal_id,
                    'company_handle': company_handle if company_handle else 'all',
                    'available_company_handles': [handle for handle in unique_company_handles if handle]
                }
            }, status=status.HTTP_200_OK)

        except ValueError as e:
            return Response({
                'error': f"Invalid pagination parameters: {str(e)}"
            }, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(
                f"Error fetching high value followers for deal {deal_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class TweetsView(APIView):
    """
    API endpoint to get tweets for a specific deal with pagination and approach filtering
    """

    def get(self, request, deal_id, format=None):
        try:
            # Get pagination parameters from query string
            page = int(request.query_params.get('page', 1))
            limit = int(request.query_params.get('limit', 10))

            # Get approach filter from query string
            approach = request.query_params.get('approach', '')

            # Validate pagination parameters
            if page < 1:
                page = 1
            if limit < 1 or limit > 1000:  # Set reasonable limits
                limit = 10

            # Calculate skip value for pagination
            skip = (page - 1) * limit

            # Build query for SearchQuery to get search queries for this deal
            search_query_filter = {'deal_id': deal_id}
            if approach and approach.upper() in ['RF1', 'RF2', 'RF3', 'GUNSHOT']:
                search_query_filter['approach'] = approach.upper()

            # Get search queries for this deal and approach
            search_queries = SearchQuery.objects(**search_query_filter)
            search_query_ids = [str(sq.id) for sq in search_queries]

            if not search_query_ids:
                # Return empty response if no search queries found
                return Response({
                    'tweets': [],
                    'pagination': {
                        'current_page': page,
                        'total_pages': 0,
                        'total_count': 0,
                        'limit': limit,
                        'has_next': False,
                        'has_previous': False
                    }
                }, status=status.HTTP_200_OK)

            # Get total count of tweets for this deal and approach
            total_count = Tweet.objects(
                search_query_id__in=search_query_ids).count()

            # Fetch tweets with pagination, ordered by tweet_created_at (newest first)
            tweets = Tweet.objects(search_query_id__in=search_query_ids).order_by(
                '-tweet_created_at').skip(skip).limit(limit)

            # Create a mapping of search query IDs to search query details
            search_query_map = {}
            for sq in search_queries:
                search_query_map[str(sq.id)] = {
                    'search_query': sq.search_query,
                    'approach': sq.approach,
                    'combination': sq.combination,
                    'total_tweets': sq.total_tweets
                }

            # Enrich tweet data with search query information
            enriched_tweets = []
            for tweet in tweets:
                tweet_data = {
                    'id': str(tweet.id),
                    'search_query_id': str(tweet.search_query_id.id),
                    'tweet': tweet.tweet,
                    'created_at': tweet.created_at,
                    'approach': tweet.approach,
                    'search_query_info': search_query_map.get(str(tweet.search_query_id.id), {})
                }
                enriched_tweets.append(tweet_data)

            # Serialize the enriched tweets
            serializer = TweetSerializer(enriched_tweets, many=True)

            # Calculate pagination metadata
            total_pages = (total_count + limit -
                           1) // limit  # Ceiling division

            # Return response with pagination metadata
            return Response({
                'tweets': serializer.data,
                'pagination': {
                    'current_page': page,
                    'total_pages': total_pages,
                    'total_count': total_count,
                    'limit': limit,
                    'has_next': page < total_pages,
                    'has_previous': page > 1
                },
                'filters': {
                    'deal_id': deal_id,
                    'approach': approach if approach else 'all'
                }
            }, status=status.HTTP_200_OK)

        except ValueError as e:
            return Response({
                'error': f"Invalid pagination parameters: {str(e)}"
            }, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(
                f"Error fetching tweets for deal {deal_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class RedditPostsView(APIView):
    """
    API endpoint to get Reddit posts for a specific deal with pagination and filtering
    """

    def get(self, request, deal_id, format=None):
        try:
            # Get pagination parameters from query string
            page = int(request.query_params.get('page', 1))
            limit = int(request.query_params.get('limit', 10))

            # Get filter parameters from query string
            competition = request.query_params.get('competition', '')
            approach = request.query_params.get('approach', '')

            # Validate pagination parameters
            if page < 1:
                page = 1
            if limit < 1 or limit > 100:  # Set reasonable limits
                limit = 10

            # Calculate skip value for pagination
            skip = (page - 1) * limit

            # Build query filter
            query_filter = {'deal_id': deal_id}

            # Add competition filter if provided
            if competition:
                query_filter['competition'] = competition

            # Add approach filter if provided
            if approach:
                query_filter['approach'] = approach

            # Get total count of Reddit posts for this deal
            total_count = RedditPost.objects(**query_filter).count()

            # Fetch Reddit posts with pagination, ordered by creation date (newest first)
            reddit_posts = RedditPost.objects(**query_filter).order_by(
                '-created_at').skip(skip).limit(limit)

            # Serialize the Reddit posts
            serializer = RedditPostSerializer(reddit_posts, many=True)

            # Calculate pagination metadata
            total_pages = (total_count + limit -
                           1) // limit  # Ceiling division

            # Get unique competitions and approaches for filter options
            unique_competitions = RedditPost.objects(
                deal_id=deal_id).distinct('competition')
            unique_approaches = RedditPost.objects(
                deal_id=deal_id).distinct('approach')

            # Return response with pagination metadata
            return Response({
                'reddit_posts': serializer.data,
                'pagination': {
                    'current_page': page,
                    'total_pages': total_pages,
                    'total_count': total_count,
                    'limit': limit,
                    'has_next': page < total_pages,
                    'has_previous': page > 1
                },
                'filters': {
                    'deal_id': deal_id,
                    'competition': competition if competition else 'all',
                    'approach': approach if approach else 'all',
                    'available_competitions': [c for c in unique_competitions if c],
                    'available_approaches': [a for a in unique_approaches if a]
                }
            }, status=status.HTTP_200_OK)

        except ValueError as e:
            return Response({
                'error': f"Invalid pagination parameters: {str(e)}"
            }, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(
                f"Error fetching Reddit posts for deal {deal_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
