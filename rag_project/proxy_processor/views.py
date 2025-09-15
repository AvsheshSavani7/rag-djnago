import logging
import threading
from datetime import datetime
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from .models import ProxyDocument, ProxyProcessingLog
from .serializers import (
    ProxyDocumentSerializer,
    ProxyProcessingRequestSerializer
)
from .agentic_sec_processor import AgenticSECProcessor
from .sec_processor_and_pinecone import SectionProcessor
from sec_rss_parser.websocket_service import SECWebSocketService
from sec_rss_parser.models import SECFiling

logger = logging.getLogger(__name__)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def process_sec_document(request):
    """
    Start processing a SEC proxy document.

    POST /api/proxy-processor/proxry-processor/
    Body: {
        "cik_number": "1234567890",
        "company_name": "Example Corp",
        "sec_filling_id": "0001234567-24-000001",
        "filing_date": "2024-01-15",
        "form_type": "DEF 14A",
        "proxy_sec_url": "https://www.sec.gov/...",
        "deal_id": "optional_deal_id"
    }
    """
    try:
        serializer = ProxyProcessingRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # Extract validated data
        validated_data = serializer.validated_data
        cik_number = validated_data['cik_number']
        company_name = validated_data['company_name']
        sec_filling_id = validated_data['sec_filling_id']
        filing_date = validated_data['filing_date']
        form_type = validated_data['form_type']
        proxy_sec_url = validated_data['proxy_sec_url']
        deal_id = validated_data.get('deal_id', None)

        # Check if proxy_sec_url already exists - if so, update it; otherwise create new
        existing_doc = ProxyDocument.objects(
            proxy_sec_url=proxy_sec_url).first()
        if existing_doc:
            # Update existing document
            existing_doc.cik_number = cik_number
            existing_doc.company_name = company_name
            existing_doc.sec_filling_id = sec_filling_id
            existing_doc.filing_date = filing_date
            existing_doc.form_type = form_type
            existing_doc.deal_id = deal_id
            existing_doc.proxy_parsing_status = 'pending'
            existing_doc.processing_state = {
                'pdf_created': False,
                'toc_found': False,
                'toc_extracted': False,
                'sections_extracted': False,
                'empty_percentage': 100.0,
                'iteration_count': 0
            }
            existing_doc.save()
            proxy_doc = existing_doc
        else:
            # Create a new proxy document
            proxy_doc = ProxyDocument(
                cik_number=cik_number,
                company_name=company_name,
                sec_filling_id=sec_filling_id,
                filing_date=filing_date,
                form_type=form_type,
                proxy_sec_url=proxy_sec_url,
                deal_id=deal_id,
                proxy_parsing_status='pending',
                processing_state={
                    'pdf_created': False,
                    'toc_found': False,
                    'toc_extracted': False,
                    'sections_extracted': False,
                    'empty_percentage': 100.0,
                    'iteration_count': 0
                }
            )
            proxy_doc.save()

        # Update SEC filing collection with following and following_status
        sec_filing_updated = False
        try:
            # Find the SEC filing by _id (sec_filling_id is the MongoDB ObjectId)
            sec_filing = SECFiling.objects(_id=sec_filling_id).first()
            if sec_filing:
                sec_filing.following = True
                sec_filing.following_status = "In Progress"
                sec_filing.save()
                sec_filing_updated = True
                logger.info(
                    f"Updated SEC filing {sec_filling_id} with following=True and following_status='In Progress'")
            else:
                logger.warning(
                    f"SEC filing with _id {sec_filling_id} not found in sec_filings collection")
        except Exception as e:
            logger.error(
                f"Error updating SEC filing {sec_filling_id}: {str(e)}")

        # Start processing in a separate thread
        processing_thread = threading.Thread(
            target=process_proxy_document_async,
            args=(proxy_doc.id, proxy_sec_url)
        )
        processing_thread.daemon = True
        processing_thread.start()

        # Determine if this was an update or new document
        is_update = existing_doc is not None
        action = 'Updated and restarted processing' if is_update else 'Started processing'

        # Log the start of processing
        log_entry = ProxyProcessingLog(
            proxy_document_id=str(proxy_doc.id),
            level='INFO',
            message=f'{action} proxy document for {company_name}: {proxy_sec_url}',
            module='proxy_processor.views'
        )
        log_entry.save()

        return Response({
            'proxy_document_id': str(proxy_doc.id),
            'status': 'In Progress',
            'message': f'Proxy document {action.lower()}',
            'company_name': company_name,
            'cik_number': cik_number,
            'proxy_sec_url': proxy_sec_url,
            'is_update': is_update,
            'sec_filing_updated': sec_filing_updated,
            'sec_filing_id': sec_filling_id
        }, status=status.HTTP_202_ACCEPTED)

    except Exception as e:
        logger.error(f"Error starting proxy document processing: {str(e)}")
        return Response({
            'error': 'Failed to start processing',
            'message': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def process_proxy_document_async(proxy_doc_id, proxy_sec_url):
    """
    Process proxy document asynchronously.
    """
    try:
        # Update document status to processing
        proxy_doc = ProxyDocument.objects.get(id=proxy_doc_id)
        proxy_doc.proxy_parsing_status = 'processing'
        proxy_doc.save()

        # Log processing start
        log_entry = ProxyProcessingLog(
            proxy_document_id=str(proxy_doc_id),
            level='INFO',
            message='Starting agentic SEC processing',
            module='proxy_processor.views'
        )
        log_entry.save()

        # Initialize the processor
        processor = AgenticSECProcessor(proxy_sec_url)

        # Process the document
        results = processor.process_document()

        # Update document with results
        empty_percentage = results.get('empty_percentage', 100.0)
        proxy_doc.empty_percentage = empty_percentage
        proxy_doc.agent_response = results.get('agent_response', '')
        proxy_doc.processing_state = processor.processing_state
        proxy_doc.s3_urls = results.get('s3_urls', {})

        # Check if empty percentage is too high (>40%)
        if empty_percentage > 40.0:
            # Mark as failed due to high empty percentage
            proxy_doc.proxy_parsing_status = 'failed'
            proxy_doc.error_message = f'Processing failed: Empty percentage too high ({empty_percentage:.1f}% > 40%)'
            proxy_doc.completed_at = datetime.utcnow()
            proxy_doc.save()

            # Update SEC filing status to Failed
            try:
                sec_filing = SECFiling.objects(
                    _id=proxy_doc.sec_filling_id).first()
                if sec_filing:
                    sec_filing.following_status = "Failed"
                    sec_filing.save()
                    logger.info(
                        f"Updated SEC filing {proxy_doc.sec_filling_id} with following_status='Failed' due to high empty percentage")
            except Exception as e:
                logger.error(
                    f"Error updating SEC filing status to Failed for {proxy_doc.sec_filling_id}: {str(e)}")

            # Log failure
            log_entry = ProxyProcessingLog(
                proxy_document_id=str(proxy_doc_id),
                level='ERROR',
                message=f'Processing failed due to high empty percentage: {empty_percentage:.1f}% (threshold: 40%)',
                module='proxy_processor.views'
            )
            log_entry.save()

            logger.warning(
                f"Processing failed for {proxy_doc.company_name}: Empty percentage {empty_percentage:.1f}% exceeds 40% threshold")

            # Emit WebSocket event for processing failure
            try:
                # Prepare filing data for WebSocket emission
                filing_data = {
                    '_id': str(proxy_doc.sec_filling_id),
                    'proxy_document_id': str(proxy_doc.id),
                    'company_name': proxy_doc.company_name,
                    'form_type': proxy_doc.form_type,
                    'cik_number': proxy_doc.cik_number,
                    'sec_filling_id': proxy_doc.sec_filling_id,
                    'filing_date': proxy_doc.filing_date,
                    'proxy_sec_url': proxy_doc.proxy_sec_url,
                    'deal_id': proxy_doc.deal_id,
                    'following': True,  # Still following, but failed
                    'following_status': 'Failed',  # Set to Failed
                    'proxy_parsing_status': proxy_doc.proxy_parsing_status,
                    'empty_percentage': empty_percentage,
                    'error_message': proxy_doc.error_message,
                    'completed_at': proxy_doc.completed_at.isoformat() if proxy_doc.completed_at else None,
                    'created_at': proxy_doc.created_at.isoformat() if proxy_doc.created_at else None,
                    'updated_at': proxy_doc.updated_at.isoformat() if proxy_doc.updated_at else None
                }

                logger.info(
                    f"🔍 DEBUG: About to emit WebSocket event for processing failure with data: {filing_data}")
                logger.info(f"🔍 DEBUG: Analysis result: processing_failed")

                # Emit WebSocket event for processing failure
                SECWebSocketService.emit_sec_analysis_complete(
                    filing_data, 'processing_failed')

                logger.info(
                    f"✅ Emitted WebSocket event for processing failure: {proxy_doc.company_name}")

            except Exception as ws_error:
                logger.error(
                    f"Error emitting WebSocket event for processing failure: {ws_error}")
                # Don't fail the entire process if WebSocket emission fails

        else:
            # Empty percentage is acceptable, proceed with completion
            proxy_doc.proxy_parsing_status = 'completed'
            proxy_doc.completed_at = datetime.utcnow()
            proxy_doc.save()

            # Update SEC filing status to Completed
            try:
                sec_filing = SECFiling.objects(
                    _id=proxy_doc.sec_filling_id).first()
                if sec_filing:
                    sec_filing.following_status = "Completed"
                    sec_filing.save()
                    logger.info(
                        f"Updated SEC filing {proxy_doc.sec_filling_id} with following_status='Completed'")
            except Exception as e:
                logger.error(
                    f"Error updating SEC filing status to Completed for {proxy_doc.sec_filling_id}: {str(e)}")

            # Log completion
            log_entry = ProxyProcessingLog(
                proxy_document_id=str(proxy_doc_id),
                level='INFO',
                message=f'Successfully completed processing. Empty percentage: {empty_percentage:.1f}%',
                module='proxy_processor.views'
            )
            log_entry.save()

            # Start Pinecone processing if sections JSON URL is available
            s3_urls = results.get('s3_urls', {})
            sections_json_url = s3_urls.get('sections_json_url')

            if sections_json_url:
                logger.info(
                    f"Starting Pinecone processing for sections: {sections_json_url}")

                # Start Pinecone processing in a separate thread
                pinecone_thread = threading.Thread(
                    target=process_sections_with_pinecone,
                    args=(proxy_doc_id, sections_json_url)
                )
                pinecone_thread.daemon = True
                pinecone_thread.start()

                # Log the start of Pinecone processing
                log_entry = ProxyProcessingLog(
                    proxy_document_id=str(proxy_doc_id),
                    level='INFO',
                    message=f'Started Pinecone processing thread for sections',
                    module='proxy_processor.views'
                )
                log_entry.save()
            else:
                logger.warning(
                    f"No sections JSON URL found in S3 URLs: {s3_urls}")
                log_entry = ProxyProcessingLog(
                    proxy_document_id=str(proxy_doc_id),
                    level='WARNING',
                    message='No sections JSON URL found for Pinecone processing',
                    module='proxy_processor.views'
                )
                log_entry.save()

    except Exception as e:
        # Update document status to failed
        try:
            proxy_doc = ProxyDocument.objects.get(id=proxy_doc_id)
            proxy_doc.proxy_parsing_status = 'failed'
            proxy_doc.error_message = str(e)
            proxy_doc.completed_at = datetime.utcnow()
            proxy_doc.save()

            # Update SEC filing status to Failed
            try:
                sec_filing = SECFiling.objects(
                    _id=proxy_doc.sec_filling_id).first()
                if sec_filing:
                    sec_filing.following_status = "Failed"
                    sec_filing.save()
                    logger.info(
                        f"Updated SEC filing {proxy_doc.sec_filling_id} with following_status='Failed'")
            except Exception as sec_error:
                logger.error(
                    f"Error updating SEC filing status to Failed for {proxy_doc.sec_filling_id}: {str(sec_error)}")
        except:
            pass

        # Log error
        log_entry = ProxyProcessingLog(
            proxy_document_id=str(proxy_doc_id),
            level='ERROR',
            message=f'Processing failed: {str(e)}',
            module='proxy_processor.views'
        )
        log_entry.save()

        logger.error(
            f"Error processing proxy document {proxy_doc_id}: {str(e)}")


def process_sections_with_pinecone(proxy_doc_id, sections_json_url):
    """
    Process sections with Pinecone after SEC processing is complete.
    """
    try:
        # Get the proxy document
        proxy_doc = ProxyDocument.objects.get(id=proxy_doc_id)

        # Log start of Pinecone processing
        log_entry = ProxyProcessingLog(
            proxy_document_id=str(proxy_doc_id),
            level='INFO',
            message='Starting Pinecone processing for sections',
            module='proxy_processor.views'
        )
        log_entry.save()

        # Initialize SectionProcessor with proxy information
        processor = SectionProcessor(
            proxy_id=str(proxy_doc.id)
        )

        # Process sections from S3 URL
        processor.process_from_s3_url(sections_json_url)

        # Update document with Pinecone processing completion
        proxy_doc.pinecone_processing_status = 'completed'
        proxy_doc.pinecone_processed_at = datetime.utcnow()
        proxy_doc.save()

        # Log completion
        log_entry = ProxyProcessingLog(
            proxy_document_id=str(proxy_doc_id),
            level='INFO',
            message='Successfully completed Pinecone processing',
            module='proxy_processor.views'
        )
        log_entry.save()

        logger.info(
            f"Successfully completed Pinecone processing for proxy document {proxy_doc_id}")

        # Emit WebSocket event after Pinecone processing completion
        try:
            # Prepare filing data for WebSocket emission
            filing_data = {
                '_id': str(proxy_doc.sec_filling_id),
                'proxy_document_id': str(proxy_doc.id),
                'company_name': proxy_doc.company_name,
                'form_type': proxy_doc.form_type,
                'cik_number': proxy_doc.cik_number,
                'sec_filling_id': proxy_doc.sec_filling_id,
                'filing_date': proxy_doc.filing_date,
                'proxy_sec_url': proxy_doc.proxy_sec_url,
                'deal_id': proxy_doc.deal_id,
                'following': True,  # Set to False as requested
                'following_status': 'Completed',  # Set to Completed as requested
                'pinecone_processing_status': proxy_doc.pinecone_processing_status,
                'pinecone_processed_at': proxy_doc.pinecone_processed_at.isoformat() if proxy_doc.pinecone_processed_at else None,
                'created_at': proxy_doc.created_at.isoformat() if proxy_doc.created_at else None,
                'updated_at': proxy_doc.updated_at.isoformat() if proxy_doc.updated_at else None
            }

            # Emit WebSocket event for SEC analysis update
            SECWebSocketService.emit_sec_analysis_complete(
                filing_data, 'pinecone_completed')

            logger.info(
                f"✅ Emitted WebSocket event for Pinecone completion: {proxy_doc.company_name}")

        except Exception as ws_error:
            logger.error(
                f"Error emitting WebSocket event for Pinecone completion: {ws_error}")
            # Don't fail the entire process if WebSocket emission fails

    except Exception as e:
        # Update document status to failed
        try:
            proxy_doc = ProxyDocument.objects.get(id=proxy_doc_id)
            proxy_doc.pinecone_processing_status = 'failed'
            proxy_doc.pinecone_error_message = str(e)
            proxy_doc.save()
        except:
            pass

        # Log error
        log_entry = ProxyProcessingLog(
            proxy_document_id=str(proxy_doc_id),
            level='ERROR',
            message=f'Pinecone processing failed: {str(e)}',
            module='proxy_processor.views'
        )
        log_entry.save()

        logger.error(
            f"Error in Pinecone processing for proxy document {proxy_doc_id}: {str(e)}")


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_processing_jobs(request):
    """
    List all proxy document processing jobs.

    GET /api/proxy-processor/jobs/
    """
    try:
        proxy_docs = ProxyDocument.objects.all().order_by('-created_at')
        serializer = ProxyDocumentSerializer(proxy_docs, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error listing proxy processing jobs: {str(e)}")
        return Response({
            'error': 'Failed to list processing jobs',
            'message': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
