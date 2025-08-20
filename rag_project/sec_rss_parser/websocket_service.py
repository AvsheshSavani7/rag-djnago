import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
from rss_feeds.websocket_service import sio

logger = logging.getLogger(__name__)


class SECWebSocketService:
    """Service for handling SEC filing WebSocket notifications"""

    @staticmethod
    def emit_new_sec_filing(filing_data: Dict[str, Any]):
        """
        Emit new SEC filing to all connected clients
        This is a synchronous wrapper for the async emit function

        Args:
            filing_data: Dictionary containing SEC filing data
        """
        try:
            # Run the async emit in a new event loop or existing one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, schedule the coroutine
                    asyncio.create_task(
                        SECWebSocketService._emit_new_sec_filing_async(filing_data))
                else:
                    # If no loop is running, run it
                    loop.run_until_complete(
                        SECWebSocketService._emit_new_sec_filing_async(filing_data))
            except RuntimeError:
                # Create new event loop if none exists
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    SECWebSocketService._emit_new_sec_filing_async(filing_data))
                loop.close()

        except Exception as e:
            logger.error(f"Error emitting SEC filing WebSocket event: {e}")

    @staticmethod
    async def _emit_new_sec_filing_async(filing_data: Dict[str, Any]):
        """
        Async method to emit new SEC filing to all connected clients

        Args:
            filing_data: Dictionary containing SEC filing data
        """
        try:
            # Prepare notification payload
            notification = {
                'type': 'new_sec_filing',
                'filing': {
                    'id': filing_data.get('_id') or filing_data.get('id'),
                    'company_name': filing_data.get('company_name'),
                    'form_type': filing_data.get('form_type'),
                    'accession_number': filing_data.get('accession_number'),
                    'title': filing_data.get('title'),
                    'link': filing_data.get('link'),
                    'description': filing_data.get('description'),
                    'cik_number': filing_data.get('cik_number'),
                    'filing_date': filing_data.get('filing_date'),
                    'acceptance_datetime_utc': filing_data.get('acceptance_datetime_utc'),
                    'has_htm_files': filing_data.get('has_htm_files', False),
                    'is_new_deal': filing_data.get('is_new_deal'),
                    'following': filing_data.get('following', False),
                    'xbrl_files': filing_data.get('xbrl_files', []),
                    'created_at': filing_data.get('created_at'),
                    'updated_at': filing_data.get('updated_at')
                },
                'timestamp': datetime.utcnow().isoformat()
            }

            # Emit to all connected clients
            await sio.emit('sec_filing_update', notification)

            # Also emit to specific room for SEC filings
            await sio.emit('sec_filing_update', notification, room='sec_filings')

            logger.info(
                f"✅ Emitted SEC filing WebSocket event: {filing_data.get('company_name')} - {filing_data.get('form_type')}")

        except Exception as e:
            logger.error(f"Error in async SEC filing emit: {e}")

    @staticmethod
    def emit_sec_analysis_complete(filing_data: Dict[str, Any], analysis_result: str):
        """
        Emit GPT analysis completion event

        Args:
            filing_data: Dictionary containing SEC filing data
            analysis_result: Result of GPT analysis ('new_deal', 'amendment', or 'inconclusive')
        """
        try:
            # Run the async emit
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(SECWebSocketService._emit_analysis_complete_async(
                        filing_data, analysis_result))
                else:
                    loop.run_until_complete(SECWebSocketService._emit_analysis_complete_async(
                        filing_data, analysis_result))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(SECWebSocketService._emit_analysis_complete_async(
                    filing_data, analysis_result))
                loop.close()

        except Exception as e:
            logger.error(f"Error emitting SEC analysis WebSocket event: {e}")

    @staticmethod
    async def _emit_analysis_complete_async(filing_data: Dict[str, Any], analysis_result: str):
        """
        Async method to emit GPT analysis completion

        Args:
            filing_data: Dictionary containing SEC filing data
            analysis_result: Result of GPT analysis
        """
        try:
            # Prepare notification payload
            notification = {
                'type': 'sec_analysis_complete',
                'filing': {
                    'id': filing_data.get('_id') or filing_data.get('id'),
                    'company_name': filing_data.get('company_name'),
                    'form_type': filing_data.get('form_type'),
                    'accession_number': filing_data.get('accession_number'),
                    'is_new_deal': filing_data.get('is_new_deal'),
                    'following': filing_data.get('following', False),
                    'following_status': filing_data.get('following_status', 'Not Started')
                },
                'analysis_result': analysis_result,
                'timestamp': datetime.utcnow().isoformat()
            }

            # Emit to all connected clients
            await sio.emit('sec_analysis_update', notification)

            # Also emit to specific room for SEC analysis
            await sio.emit('sec_analysis_update', notification, room='sec_analysis')

            logger.info(
                f"✅ Emitted SEC analysis WebSocket event: {filing_data.get('company_name')} → {analysis_result}")

        except Exception as e:
            logger.error(f"Error in async SEC analysis emit: {e}")

    @staticmethod
    def emit_sec_processing_stats(stats: Dict[str, Any]):
        """
        Emit SEC processing statistics

        Args:
            stats: Dictionary containing processing statistics
        """
        try:
            # Run the async emit
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(
                        SECWebSocketService._emit_stats_async(stats))
                else:
                    loop.run_until_complete(
                        SECWebSocketService._emit_stats_async(stats))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    SECWebSocketService._emit_stats_async(stats))
                loop.close()

        except Exception as e:
            logger.error(f"Error emitting SEC stats WebSocket event: {e}")

    @staticmethod
    async def _emit_stats_async(stats: Dict[str, Any]):
        """
        Async method to emit processing statistics

        Args:
            stats: Dictionary containing processing statistics
        """
        try:
            # Prepare notification payload
            notification = {
                'type': 'sec_processing_stats',
                'stats': stats,
                'timestamp': datetime.utcnow().isoformat()
            }

            # Emit to all connected clients
            await sio.emit('sec_stats_update', notification)

            logger.info(f"✅ Emitted SEC stats WebSocket event")

        except Exception as e:
            logger.error(f"Error in async SEC stats emit: {e}")
