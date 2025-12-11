from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import logging
import threading
from datetime import datetime
from .services import SECFeedProcessor
from .models import SECFiling, SECFeedStatus
from .serializers import (
    SECFilingSerializer,
    SECFilingListSerializer,
    SECFilingDetailSerializer,
    SECFeedStatusSerializer
)
from document_processor.models import ProcessingJob

logger = logging.getLogger(__name__)


class ProcessSECFeedView(APIView):
    """API endpoint to manually trigger SEC RSS feed processing"""
    permission_classes = [AllowAny]

    def get(self, request, format=None):
        """Handle GET requests (for cron jobs)"""
        return self.process_feed_request()

    def post(self, request, format=None):
        """Handle POST requests"""
        return self.process_feed_request()

    def process_feed_request(self):
        """Common method to process SEC feed in background"""
        try:
            # Get form_type from request (query params for GET, body for POST)
            form_type = None
            if hasattr(self.request, 'query_params'):
                form_type = self.request.query_params.get('form_type')
            if not form_type and hasattr(self.request, 'data'):
                form_type = self.request.data.get('form_type')

            # Start processing in background thread
            def process_in_background():
                try:
                    processor = SECFeedProcessor(form_type=form_type)
                    result = processor.process_feed()
                    logger.info(
                        f"Background SEC processing completed: {result}")
                except Exception as e:
                    logger.error(f"Error in background SEC processing: {e}")

            thread = threading.Thread(target=process_in_background)
            thread.daemon = True
            thread.start()

            return Response({
                'success': True,
                'message': 'SEC feed processing started in background',
                'status': 'processing',
                'form_type': form_type
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error starting SEC processing: {e}")
            return Response(
                {'error': 'Failed to start processing'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SECFilingListView(APIView):
    """API endpoint to list SEC filings with filters"""
    permission_classes = [AllowAny]

    def get(self, request, format=None):
        try:
            # Get query parameters
            form_type = request.query_params.get('form_type')
            cik_number = request.query_params.get('cik_number')
            has_htm_files = request.query_params.get('has_htm_files')
            offset = int(request.query_params.get('offset', 0))
            limit = int(request.query_params.get('limit', 10))

            # Build query
            query = {}
            if form_type:
                query['form_type'] = form_type
            if cik_number:
                query['cik_number'] = cik_number
            if has_htm_files is not None:
                query['has_htm_files'] = has_htm_files.lower() == 'true'

            # Get total count before pagination
            total_count = SECFiling.objects(**query).count()

            # Get filings with pagination
            filings = SECFiling.objects(
                **query).order_by('-created_at').skip(offset).limit(limit)

            # Serialize
            serializer = SECFilingListSerializer(filings, many=True)

            # Add deal_found field for DEF 14A and PRE 14A filings
            filings_data = serializer.data
            for filing_data in filings_data:
                if filing_data.get('form_type') in ['DEF 14A', 'PRE 14A']:
                    cik_number = filing_data.get('cik_number')
                    if cik_number:
                        try:
                            deal = ProcessingJob.objects(
                                cik=cik_number).first()
                            if deal:
                                filing_data['deal_found'] = True
                                filing_data['deal_id'] = str(deal.id)
                            else:
                                filing_data['deal_found'] = False
                                filing_data['deal_id'] = None
                        except Exception as e:
                            logger.error(
                                f"Error checking CIK {cik_number} in Deals collection: {e}")
                            filing_data['deal_found'] = False
                            filing_data['deal_id'] = None
                    else:
                        filing_data['deal_found'] = False
                        filing_data['deal_id'] = None
                else:
                    # Not applicable for other form types
                    filing_data['deal_found'] = None
                    filing_data['deal_id'] = None

            return Response({
                'success': True,
                'filings': filings_data,
                'count': total_count
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error in SECFilingListView: {e}")
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SECFilingDetailView(APIView):
    """API endpoint to get detailed information about a specific SEC filing"""
    permission_classes = [AllowAny]

    def get(self, request, filing_id, format=None):
        try:
            filing = SECFiling.objects(_id=filing_id).first()

            if not filing:
                return Response(
                    {'error': 'Filing not found'},
                    status=status.HTTP_404_NOT_FOUND
                )

            serializer = SECFilingDetailSerializer(filing)
            filing_data = serializer.data

            # Add deal_found field for DEF 14A and PRE 14A filings
            if filing_data.get('form_type') in ['DEF 14A', 'PRE 14A']:
                cik_number = filing_data.get('cik_number')
                if cik_number:
                    try:
                        deal = ProcessingJob.objects(cik=cik_number).first()
                        if deal:
                            filing_data['deal_found'] = True
                            filing_data['deal_id'] = str(deal.id)
                        else:
                            filing_data['deal_found'] = False
                            filing_data['deal_id'] = None
                    except Exception as e:
                        logger.error(
                            f"Error checking CIK {cik_number} in Deals collection: {e}")
                        filing_data['deal_found'] = False
                        filing_data['deal_id'] = None
                else:
                    filing_data['deal_found'] = False
                    filing_data['deal_id'] = None
            else:
                # Not applicable for other form types
                filing_data['deal_found'] = None
                filing_data['deal_id'] = None

            return Response({
                'success': True,
                'filing': filing_data
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error in SECFilingDetailView: {e}")
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SEC8KFilingListView(APIView):
    """API endpoint to list 8-K filings with EX-2.1 HTM files"""
    permission_classes = [AllowAny]

    def get(self, request, format=None):
        try:
            limit = int(request.query_params.get('limit', 100))

            filings = SECFiling.objects(
                form_type='8-K',
                has_htm_files=True
            ).order_by('-created_at').limit(limit)

            serializer = SECFilingListSerializer(filings, many=True)

            # Add deal_found field for DEF 14A and PRE 14A filings
            filings_data = serializer.data
            for filing_data in filings_data:
                if filing_data.get('form_type') in ['DEF 14A', 'PRE 14A']:
                    cik_number = filing_data.get('cik_number')
                    if cik_number:
                        try:
                            deal = ProcessingJob.objects(
                                cik=cik_number).first()
                            if deal:
                                filing_data['deal_found'] = True
                                filing_data['deal_id'] = str(deal.id)
                            else:
                                filing_data['deal_found'] = False
                                filing_data['deal_id'] = None
                        except Exception as e:
                            logger.error(
                                f"Error checking CIK {cik_number} in Deals collection: {e}")
                            filing_data['deal_found'] = False
                            filing_data['deal_id'] = None
                    else:
                        filing_data['deal_found'] = False
                        filing_data['deal_id'] = None
                else:
                    # Not applicable for other form types
                    filing_data['deal_found'] = None
                    filing_data['deal_id'] = None

            return Response({
                'success': True,
                'filings': filings_data,
                'count': len(filings_data)
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error in SEC8KFilingListView: {e}")
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SECFeedStatusView(APIView):
    """API endpoint to get SEC feed processing status"""
    permission_classes = [AllowAny]

    def get(self, request, format=None):
        try:
            feed_status = SECFeedStatus.objects().first()

            if not feed_status:
                return Response({
                    'success': True,
                    'status': 'No feed status found'
                }, status=status.HTTP_200_OK)

            serializer = SECFeedStatusSerializer(feed_status)

            return Response({
                'success': True,
                'status': serializer.data
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error in SECFeedStatusView: {e}")
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SECFilingStatsView(APIView):
    """API endpoint to get statistics about SEC filings"""
    permission_classes = [AllowAny]

    def get(self, request, format=None):
        try:
            # Get basic stats
            total_filings = SECFiling.objects.count()
            eight_k_filings = SECFiling.objects(form_type='8-K').count()
            eight_k_with_htm = SECFiling.objects(
                form_type='8-K', has_htm_files=True).count()

            # Get recent filings by form type
            recent_form_types = SECFiling.objects.aggregate([
                {'$group': {'_id': '$form_type', 'count': {'$sum': 1}}},
                {'$sort': {'count': -1}},
                {'$limit': 10}
            ])

            return Response({
                'success': True,
                'stats': {
                    'total_filings': total_filings,
                    'eight_k_filings': eight_k_filings,
                    'eight_k_with_htm': eight_k_with_htm,
                    'recent_form_types': list(recent_form_types)
                }
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error in SECFilingStatsView: {e}")
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
