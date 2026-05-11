from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import logging
import traceback
from bson import ObjectId
import threading
import concurrent.futures
import json
from io import BytesIO
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

from .models import (
    ProcessingJob,
    CompanyProducts,
    CompetitiveAnalysis,
    HighValueFollowers,
    SearchQuery,
    Tweet,
    RedditPost,
    DealSchemaResults,
)
# Remove Celery imports for simple approach
# from .tasks import run_daily_reddit_scraper, run_reddit_scraper_for_deal, run_reddit_scraper_for_deals
from .serializers import (
    ProcessingJobSerializer,
    FileProcessRequestSerializer,
    HighValueFollowersSerializer,
    TweetSerializer,
    RedditPostSerializer
)
from .services import FlattenProcessor, EmbeddingService, S3Service, ChatWithAIService, SummaryGenerationService
from mongoengine import Q
from mongoengine.errors import DoesNotExist, ValidationError
from sec_rss_parser.models import DealDmaSummary

logger = logging.getLogger(__name__)

# Create a ThreadPoolExecutor for background tasks
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)


def _load_schema_results_for_job(job):
    """
    Hydrate `job.schema_results` from the dedicated `deal_schema_results` collection.
    This keeps API responses consistent after schema writes moved.
    """
    try:
        schema_record = DealSchemaResults.objects(deal_id=job.id).first()
    except Exception:
        return

    if not schema_record or schema_record.schema_results is None:
        job.schema_results = {}
        return

    schema_results = schema_record.schema_results
    if isinstance(schema_results, str):
        try:
            job.schema_results = json.loads(schema_results)
        except json.JSONDecodeError:
            job.schema_results = {}
    else:
        job.schema_results = schema_results or {}


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

            # Persist flattened_json_url update regardless of how legacy `schema_results`
            # is currently stored on ProcessingJob (schema is served from DealSchemaResults).
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

            # Hydrate schema_results from dedicated collection.
            _load_schema_results_for_job(job)

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
            # Get pagination parameters
            offset = int(request.query_params.get('offset', 0))
            limit = int(request.query_params.get('limit', 10))

            # Optional search filter (matches target_name, acquire_name, cik, acquirer_cik)
            search = request.query_params.get('search', '').strip()

            # Optional embedding status filter (All means no filter)
            embedding_status = request.query_params.get(
                'embedding_status', 'All')
            jobs_query = ProcessingJob.objects
            if embedding_status and embedding_status.upper() != 'ALL':
                jobs_query = jobs_query.filter(
                    embedding_status=embedding_status.upper())

            if search:
                search_q = (
                    Q(target_name__icontains=search) |
                    Q(acquire_name__icontains=search) |
                    Q(cik__icontains=search) |
                    Q(acquirer_cik__icontains=search)
                )
                if ObjectId.is_valid(search):
                    search_q = search_q | Q(id=ObjectId(search))
                jobs_query = jobs_query.filter(search_q)

            # Get total count before pagination
            total_count = jobs_query.count()

            # Get deals with pagination
            jobs = jobs_query.order_by(
                '-createdAt').skip(offset).limit(limit)

            for job in jobs:
                _load_schema_results_for_job(job)

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
                'total': total_count,
                'offset': offset,
                'limit': limit
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


class ListAllDealsNoPaginationView(ListAllDealsView):
    """
    API endpoint to get all deals without pagination
    """

    def get(self, request, format=None):
        try:
            # Get all deals
            jobs = ProcessingJob.objects.all().order_by('-createdAt')

            for job in jobs:
                _load_schema_results_for_job(job)

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

            return Response({
                'deals': deals_data,
                'total': len(deals_data)
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error fetching all deals: {str(e)}")
            logger.error(traceback.format_exc())

            # Return error response
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


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

    def _generate_summary_background(self, deal_id, temperature, provider, model):
        """Background task to generate summary"""
        try:
            object_id = ObjectId(deal_id)

            # Persist PROCESSING status in dedicated collection.
            DealDmaSummary.save_or_update(
                deal_id=object_id,
                summary_status='PROCESSING',
                summary_docx_url=None,
                summary_using=f"{provider}-{model}",
            )

            summary_service = SummaryGenerationService()
            result = summary_service.generate_summary_engine(
                deal_id=deal_id,
                temperature=temperature,
                provider=provider,
                model=model
            )

            # Update the job with the summary URL and provider info
            try:
                job = ProcessingJob.objects.get(id=object_id)
                job.summary_docx_url = result
                job.summary_using = f"{provider}-{model}"
                job.summary_status = 'COMPLETED'
                job.save()

                # Persist COMPLETED status in dedicated collection.
                DealDmaSummary.save_or_update(
                    deal_id=object_id,
                    summary_status='COMPLETED',
                    summary_docx_url=result,
                    summary_using=job.summary_using,
                )
            except Exception as e:
                logger.error(f"Error updating job with summary URL: {str(e)}")
                # If job update fails, record FAILED state as well.
                try:
                    job = ProcessingJob.objects.get(id=object_id)
                    job.summary_status = 'FAILED'
                    job.error_message = str(e)
                    job.save()
                except Exception:
                    pass

                DealDmaSummary.save_or_update(
                    deal_id=object_id,
                    summary_status='FAILED',
                    summary_docx_url=None,
                    summary_using=f"{provider}-{model}",
                )

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
                DealDmaSummary.save_or_update(
                    deal_id=object_id,
                    summary_status='FAILED',
                    summary_docx_url=None,
                    summary_using=f"{provider}-{model}",
                )
            except Exception as inner_e:
                logger.error(f"Error updating job status: {str(inner_e)}")

                # Ensure FAILED record is still written to DMA summary.
                try:
                    object_id = ObjectId(deal_id)
                except Exception:
                    object_id = None

                if object_id is not None:
                    DealDmaSummary.save_or_update(
                        deal_id=object_id,
                        summary_status='FAILED',
                        summary_docx_url=None,
                        summary_using=f"{provider}-{model}",
                    )
            # Note: deal_dma_summary FAILED upsert is handled in both branches above.

    def post(self, request, format=None):
        # Get request parameters
        deal_id = request.data.get('deal_id')
        temperature = float(request.data.get('temperature', 1))
        provider = request.data.get('provider', 'openai')
        model = request.data.get('model', 'gpt-5')

        # Validate parameters
        if not deal_id:
            return Response({"error": "deal_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Validate provider
        valid_providers = ['openai', 'google', 'anthropic']
        if provider.lower() not in valid_providers:
            return Response({
                "error": f"Invalid provider. Must be one of: {', '.join(valid_providers)}"
            }, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Get the job and update status
            object_id = ObjectId(deal_id)
            job = ProcessingJob.objects.get(id=object_id)
            job.summary_status = 'PROCESSING'
            job.save()

            # Start summary generation in background
            self.executor.submit(
                self._generate_summary_background, deal_id, temperature, provider, model)

            # Return immediate response
            return Response({
                'id': str(deal_id),
                'summary_status': 'PROCESSING',
                'provider': provider,
                'model': model,
                'summary_using': f"{provider}-{model}",
                'temperature': temperature,
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

            # Hydrate schema_results from dedicated collection.
            _load_schema_results_for_job(job)

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
            relevance_score_from = request.query_params.get(
                'relevance_score_from', '')
            relevance_score_to = request.query_params.get(
                'relevance_score_to', '')

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

            # Add relevance_score range filter if provided
            if relevance_score_from or relevance_score_to:
                try:
                    # Set default values if not provided
                    from_value = float(
                        relevance_score_from) if relevance_score_from else 0.0
                    to_value = float(
                        relevance_score_to) if relevance_score_to else 100.0

                    # Create range condition: posts with relevance_score between from and to OR posts without relevance_score
                    relevance_range_query = (
                        Q(post__relevance_score__gte=from_value) &
                        Q(post__relevance_score__lte=to_value) |
                        Q(post__relevance_score__exists=False)
                    )

                    # Convert existing filter to Q object and combine
                    if isinstance(query_filter, dict):
                        base_query = Q(**query_filter)
                    else:
                        base_query = query_filter
                    query_filter = base_query & relevance_range_query
                except ValueError:
                    # If conversion fails, ignore the filter
                    pass

            # Get total count of Reddit posts for this deal
            if isinstance(query_filter, dict):
                total_count = RedditPost.objects(**query_filter).count()
                # Fetch Reddit posts with pagination, ordered by creation date (newest first)
                reddit_posts = RedditPost.objects(**query_filter).order_by(
                    '-created_at').skip(skip).limit(limit)
            else:
                # Handle Q object query
                total_count = RedditPost.objects(query_filter).count()
                # Fetch Reddit posts with pagination, ordered by creation date (newest first)
                reddit_posts = RedditPost.objects(query_filter).order_by(
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
                    'relevance_score_from': relevance_score_from if relevance_score_from else 'all',
                    'relevance_score_to': relevance_score_to if relevance_score_to else 'all',
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


class RedditScraperTaskView(APIView):
    """
    Simple API endpoint to trigger Reddit scraper (no Celery needed)
    """

    def post(self, request, format=None):
        try:
            action = request.data.get('action')

            if action == 'run_daily':
                # Run daily Reddit scraper for all deals (synchronous)
                from .reddit_utils.deal_reddit_scraper import run_deal_reddit_analysis
                from .models import ProcessingJob

                deals = ProcessingJob.objects.all()
                total_deals = deals.count()
                processed_deals = 0
                failed_deals = 0
                results = []

                for deal in deals:
                    try:
                        result = run_deal_reddit_analysis(str(deal.id))
                        if result:
                            processed_deals += 1
                            results.append({
                                'deal_id': str(deal.id),
                                'status': 'success',
                                'deal_name': f"{deal.acquire_name} acquiring {deal.target_name}"
                            })
                        else:
                            failed_deals += 1
                            results.append({
                                'deal_id': str(deal.id),
                                'status': 'failed',
                                'error': 'No result returned'
                            })
                    except Exception as e:
                        failed_deals += 1
                        results.append({
                            'deal_id': str(deal.id),
                            'status': 'failed',
                            'error': str(e)
                        })

                return Response({
                    'message': 'Daily Reddit scraper completed',
                    'total_deals': total_deals,
                    'processed_deals': processed_deals,
                    'failed_deals': failed_deals,
                    'results': results
                }, status=status.HTTP_200_OK)

            elif action == 'run_deal':
                # Run scraper for a specific deal
                deal_id = request.data.get('deal_id')
                if not deal_id:
                    return Response({
                        'error': 'deal_id is required for run_deal action'
                    }, status=status.HTTP_400_BAD_REQUEST)

                from .reddit_utils.deal_reddit_scraper import run_deal_reddit_analysis

                try:
                    result = run_deal_reddit_analysis(deal_id)
                    if result:
                        return Response({
                            'message': f'Reddit scraper completed for deal {deal_id}',
                            'deal_id': deal_id,
                            'status': 'success',
                            'result': result
                        }, status=status.HTTP_200_OK)
                    else:
                        return Response({
                            'message': f'Reddit scraper failed for deal {deal_id}',
                            'deal_id': deal_id,
                            'status': 'failed'
                        }, status=status.HTTP_200_OK)
                except Exception as e:
                    return Response({
                        'error': f'Error processing deal {deal_id}: {str(e)}',
                        'deal_id': deal_id,
                        'status': 'failed'
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            else:
                return Response({
                    'error': 'Invalid action. Use: run_daily or run_deal'
                }, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            logger.error(f"Error triggering Reddit scraper: {str(e)}")
            logger.error(traceback.format_exc())
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ExportDealsExcelView(APIView):
    """
    GET  - returns the list of exportable field names (for the frontend checkbox popup)
    POST - accepts selected fields + optional search/filter, returns an Excel file
    """

    EXPORTABLE_FIELDS = {
        'id': 'ID',
        'cik': 'CIK',
        'acquirer_cik': 'Acquirer CIK',
        'acquire_name': 'Acquirer Name',
        'target_name': 'Target Name',
        'announce_date': 'Announce Date',
        'embedding_status': 'Embedding Status',
        'summary_status': 'Summary Status',
        'summary_using': 'Summary Using',
        'sec_filing_id': 'SEC Filing ID',
        'file_url': 'File URL',
        'pdf_url': 'PDF URL',
        'parsed_json_url': 'Parsed JSON URL',
        'flattened_json_url': 'Flattened JSON URL',
        'summary_docx_url': 'Summary DOCX URL',
        'sec_url': 'SEC URL',
        'target_ticker': 'Target Ticker',
        'acquirer_ticker': 'Acquirer Ticker',
        'deal_status': 'Deal Status',
        'schema_processing_completed': 'Schema Processing Completed',
        'schema_processing_timestamp': 'Schema Processing Timestamp',
        'RF1_approach_done': 'RF1 Approach Done',
        'RF2_approach_done': 'RF2 Approach Done',
        'RF3_approach_done': 'RF3 Approach Done',
        'GUNSHOT_approach_done': 'GUNSHOT Approach Done',
        'parent_aliases': 'Parent Aliases',
        'target_aliases': 'Target Aliases',
        'createdAt': 'Created At',
        'updatedAt': 'Updated At',
    }

    def get(self, request, format=None):
        """Return the list of exportable fields for the frontend popup."""
        fields = [
            {'key': key, 'label': label}
            for key, label in self.EXPORTABLE_FIELDS.items()
        ]
        return Response({'fields': fields}, status=status.HTTP_200_OK)

    def post(self, request, format=None):
        """
        Generate and return an Excel file for the selected fields.

        Payload:
        {
            "fields": ["target_name", "acquire_name", "cik", ...],
            "search": "optional search text",
            "embedding_status": "ALL"
        }
        """
        try:
            selected_fields = request.data.get('fields', [])
            search = request.data.get('search', '').strip()
            embedding_status = request.data.get('embedding_status', 'All')

            if not selected_fields:
                return Response(
                    {'error': 'At least one field must be selected.'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            valid_fields = [
                f for f in selected_fields if f in self.EXPORTABLE_FIELDS]
            if not valid_fields:
                return Response(
                    {'error': 'None of the provided fields are valid.'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            jobs_query = ProcessingJob.objects
            if embedding_status and embedding_status.upper() != 'ALL':
                jobs_query = jobs_query.filter(
                    embedding_status=embedding_status.upper())

            if search:
                jobs_query = jobs_query.filter(
                    Q(target_name__icontains=search) |
                    Q(acquire_name__icontains=search) |
                    Q(cik__icontains=search) |
                    Q(acquirer_cik__icontains=search)
                )

            jobs = jobs_query.order_by('-createdAt')

            wb = Workbook()
            ws = wb.active
            ws.title = 'Deals'

            header_font = Font(bold=True, color='FFFFFF', size=11)
            header_fill = PatternFill(
                start_color='4472C4', end_color='4472C4', fill_type='solid')
            header_alignment = Alignment(
                horizontal='center', vertical='center', wrap_text=True)

            headers = [self.EXPORTABLE_FIELDS[f] for f in valid_fields]
            for col_idx, header in enumerate(headers, 1):
                cell = ws.cell(row=1, column=col_idx, value=header)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = header_alignment

            for row_idx, job in enumerate(jobs, 2):
                for col_idx, field in enumerate(valid_fields, 1):
                    value = getattr(job, field, None)
                    if isinstance(value, list):
                        value = ', '.join(str(v) for v in value)
                    elif hasattr(value, 'isoformat'):
                        value = value.strftime('%Y-%m-%d %H:%M:%S')
                    elif isinstance(value, dict):
                        value = json.dumps(value, default=str)
                    ws.cell(row=row_idx, column=col_idx, value=str(
                        value) if value is not None else '')

            for col_idx in range(1, len(valid_fields) + 1):
                max_len = max(
                    len(str(ws.cell(row=r, column=col_idx).value or ''))
                    for r in range(1, ws.max_row + 1)
                )
                ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = min(
                    max_len + 4, 50)

            ws.auto_filter.ref = ws.dimensions

            buffer = BytesIO()
            wb.save(buffer)
            buffer.seek(0)

            response = HttpResponse(
                buffer.getvalue(),
                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            response['Content-Disposition'] = 'attachment; filename="deals_export.xlsx"'
            return response

        except Exception as e:
            logger.error(f"Error exporting deals: {str(e)}")
            logger.error(traceback.format_exc())
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class RegeneratePipelineView(APIView):
    """
    Admin endpoint to re-run any combination of deal pipelines.

    POST /api/files/regenerate/
    {
        "deal_id": "69cbae4f640784d45bf14678",
        "steps": ["embedding", "termination", "covenant"]
    }
    → Returns 202 with { run_id, plan, ... }

    GET  /api/files/regenerate/<run_id>/
    → Returns current progress of that run

    GET  /api/files/regenerate/deal/<deal_id>/
    → Returns the latest run for a deal (convenience)

    Valid steps:
        DMA cascade (selecting an earlier step implies all later ones):
            parsing → embedding → schema → summary
        Independent (run as selected):
            termination, covenant, mae, entity_resolution
    """

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    def post(self, request, format=None):
        from .regeneration_pipeline import (
            ALL_VALID_STEPS,
            resolve_execution_plan,
            validate_steps,
            run_regeneration_pipeline,
            generate_run_id,
            _create_run,
        )

        deal_id = request.data.get("deal_id")
        steps = request.data.get("steps", [])

        if not deal_id:
            return Response(
                {"error": "deal_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not steps or not isinstance(steps, list):
            return Response(
                {"error": "steps is required and must be a non-empty array",
                 "valid_steps": ALL_VALID_STEPS},
                status=status.HTTP_400_BAD_REQUEST,
            )

        invalid = validate_steps(steps)
        if invalid:
            return Response(
                {"error": f"Invalid steps: {invalid}",
                 "valid_steps": ALL_VALID_STEPS},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            object_id = ObjectId(deal_id)
            ProcessingJob.objects.get(id=object_id)
        except DoesNotExist:
            return Response(
                {"error": f"No deal found for deal_id {deal_id}"},
                status=status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
            return Response(
                {"error": f"Invalid deal_id: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        plan = resolve_execution_plan(steps)
        run_id = generate_run_id()
        _create_run(run_id, deal_id, plan, steps)

        self.executor.submit(run_regeneration_pipeline, deal_id, steps, run_id)

        return Response(
            {
                "deal_id": deal_id,
                "run_id": run_id,
                "status": "processing",
                "plan": plan,
                "message": "Regeneration pipeline started in background.",
            },
            status=status.HTTP_202_ACCEPTED,
        )


class RegeneratePipelineStatusView(APIView):
    """
    Polling endpoints for regeneration pipeline progress.

    GET /api/files/regenerate/<run_id>/       — status by run_id
    GET /api/files/regenerate/deal/<deal_id>/ — latest run for a deal

    Response includes `retry_after` (seconds) so the frontend knows
    how long to wait before the next poll.
    """

    POLL_INTERVAL_PROCESSING = 10  # seconds between polls while running
    POLL_INTERVAL_DONE = 0         # no more polling needed

    def get(self, request, run_id=None, deal_id=None, format=None):
        from .regeneration_pipeline import get_run_status, get_latest_run_for_deal

        if run_id:
            run = get_run_status(run_id)
        elif deal_id:
            run = get_latest_run_for_deal(deal_id)
        else:
            return Response(
                {"error": "run_id or deal_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not run:
            return Response(
                {"error": "No regeneration run found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        is_running = run.get("status") == "processing"
        data = {
            **run,
            "retry_after": self.POLL_INTERVAL_PROCESSING if is_running else self.POLL_INTERVAL_DONE,
        }

        return Response(data, status=status.HTTP_200_OK)
