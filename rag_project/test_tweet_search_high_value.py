#!/usr/bin/env python3
"""
Test script for the new Tweet Search High Value Followers processor
"""

from document_processor.twitter_utils.high_value_followers_tweet_search import HighValueFollowersTweetSearchProcessor
import django
import os
import sys
import logging
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()


def test_tweet_search_processor():
    """Test the tweet search processor with a sample deal"""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    # Test configuration - using smaller numbers for testing
    config_overrides = {
        'max_followers_per_company': 100,  # Test with only 10 followers per company
        'max_workers': 5,  # Use 5 parallel workers
        'batch_size': 5,  # Small batch size for testing
        'use_parallel_processing': True,
        'delay_between_requests': 1,  # Shorter delays for testing
        'delay_every_n_requests': 1,
        'delay_for_rate_limit': 1
    }

    try:
        # Initialize processor
        processor = HighValueFollowersTweetSearchProcessor(
            config_overrides=config_overrides)

        # Example deal ID - replace with a real deal ID from your database
        deal_id = "68ac4a254a6006a0946ec3bb"  # Replace with actual deal ID

        logger.info(f"Testing Tweet Search processor with deal ID: {deal_id}")
        logger.info("Configuration:")
        for key, value in config_overrides.items():
            logger.info(f"  {key}: {value}")

        # Run processing
        result_file = processor.process_deal(deal_id)

        if result_file:
            logger.info("✅ Test completed successfully!")
            logger.info(f"Results saved to: {result_file}")

            # Print summary of what was processed
            logger.info("\n" + "="*50)
            logger.info("TEST SUMMARY")
            logger.info("="*50)
            logger.info(f"Deal ID: {deal_id}")
            logger.info(f"Result file: {result_file}")
            logger.info("="*50)

        else:
            logger.error("❌ Test failed - no results generated")

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    test_tweet_search_processor()
