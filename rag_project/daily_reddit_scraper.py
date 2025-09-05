#!/usr/bin/env python3
"""
Simple daily Reddit scraper script for Render.io cron jobs.
This runs daily and scrapes Reddit for all deals in the database.
"""

from document_processor.reddit_utils.deal_reddit_scraper import run_deal_reddit_analysis
from document_processor.models import ProcessingJob
import os
import sys
import django
from datetime import datetime
import logging

# Django setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run Reddit scraper for all deals"""

    start_time = datetime.now()
    logger.info("🚀 Starting daily Reddit scraper")
    logger.info("=" * 60)
    logger.info(f"⏰ Start time: {start_time}")

    try:
        # Get all deals from database
        deals = ProcessingJob.objects.all()
        total_deals = deals.count()

        logger.info(f"📊 Found {total_deals} deals to process")

        if total_deals == 0:
            logger.info("✅ No deals found in database")
            return {
                'status': 'success',
                'message': 'No deals found',
                'total_deals': 0,
                'processed_deals': 0,
                'failed_deals': 0
            }

        processed_deals = 0
        failed_deals = 0
        results = []

        # Process each deal
        for i, deal in enumerate(deals, 1):
            try:
                logger.info(f"🔄 Processing deal {i}/{total_deals}: {deal.id}")
                logger.info(
                    f"   Deal: {deal.acquire_name} acquiring {deal.target_name}")

                # Run Reddit scraper for this deal
                result = run_deal_reddit_analysis(str(deal.id))

                if result:
                    processed_deals += 1
                    logger.info(f"✅ Successfully processed deal {deal.id}")
                    results.append({
                        'deal_id': str(deal.id),
                        'status': 'success',
                        'deal_name': f"{deal.acquire_name} acquiring {deal.target_name}"
                    })
                else:
                    failed_deals += 1
                    logger.error(
                        f"❌ Failed to process deal {deal.id} - No result returned")
                    results.append({
                        'deal_id': str(deal.id),
                        'status': 'failed',
                        'error': 'No result returned',
                        'deal_name': f"{deal.acquire_name} acquiring {deal.target_name}"
                    })

            except Exception as e:
                failed_deals += 1
                error_msg = f"Error processing deal {deal.id}: {str(e)}"
                logger.error(error_msg)
                results.append({
                    'deal_id': str(deal.id),
                    'status': 'failed',
                    'error': error_msg,
                    'deal_name': f"{deal.acquire_name} acquiring {deal.target_name}"
                })

        # Calculate total time
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Final summary
        logger.info("=" * 60)
        logger.info("🎉 Daily Reddit scraper completed!")
        logger.info(f"📊 Total deals: {total_deals}")
        logger.info(f"✅ Successfully processed: {processed_deals}")
        logger.info(f"❌ Failed: {failed_deals}")
        logger.info(f"⏱️ Total execution time: {total_time:.2f} seconds")
        logger.info(f"⏰ Completed at: {end_time}")
        logger.info("=" * 60)

        return {
            'status': 'completed',
            'total_deals': total_deals,
            'processed_deals': processed_deals,
            'failed_deals': failed_deals,
            'execution_time_seconds': total_time,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'results': results
        }

    except Exception as e:
        error_msg = f"Critical error in daily Reddit scraper: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())

        return {
            'status': 'failed',
            'error': error_msg,
            'total_deals': 0,
            'processed_deals': 0,
            'failed_deals': 0
        }


if __name__ == "__main__":
    result = main()
    print(f"\n📋 Final Result: {result}")

    # Exit with appropriate code
    if result['status'] == 'completed':
        sys.exit(0)
    else:
        sys.exit(1)
