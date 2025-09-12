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
            'status': 'processing',
            'message': f'Proxy document {action.lower()}',
            'company_name': company_name,
            'cik_number': cik_number,
            'proxy_sec_url': proxy_sec_url,
            'is_update': is_update
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
        proxy_doc.proxy_parsing_status = 'completed'
        proxy_doc.completed_at = datetime.utcnow()
        proxy_doc.empty_percentage = results.get('empty_percentage', 100.0)
        proxy_doc.agent_response = results.get('agent_response', '')
        proxy_doc.processing_state = processor.processing_state
        proxy_doc.s3_urls = results.get('s3_urls', {})

        # Generate AWS URL for proxy parsing results
        # Format: proxy-parse-jsons/{cik_number}/{sec_filling_id}/

        proxy_doc.save()

        # Log completion
        log_entry = ProxyProcessingLog(
            proxy_document_id=str(proxy_doc_id),
            level='INFO',
            message=f'Successfully completed processing. Empty percentage: {proxy_doc.empty_percentage:.1f}%',
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
