#!/usr/bin/env python3
"""
Test script for the new Tweet Search High Value Followers TESTING processor
This script tests the testing version that only saves to JSON files (no MongoDB operations)

python test_tweet_search_high_value_test.py
"""

from document_processor.twitter_utils.high_value_followers_tweet_search_test import HighValueFollowersTweetSearchProcessorTest
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


def test_tweet_search_processor_test():
    """Test the tweet search testing processor with a sample deal"""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    # Test configuration - using smaller numbers for testing
    config_overrides = {}

    try:
        # Initialize processor
        processor = HighValueFollowersTweetSearchProcessorTest(
            config_overrides=config_overrides)

        # Example deal ID - replace with a real deal ID from your database
        deal_id = "68ac4a254a6006a0946ec3bb"  # Replace with actual deal ID

        logger.info(
            f"Testing Tweet Search TESTING processor with deal ID: {deal_id}")
        logger.info("Configuration:")
        for key, value in config_overrides.items():
            logger.info(f"  {key}: {value}")

        logger.info("\n" + "="*60)
        logger.info("TESTING FEATURES:")
        logger.info("="*60)
        logger.info("✓ No MongoDB operations - only JSON file output")
        logger.info("✓ Removed description filter requirement")
        logger.info("✓ Enhanced cursor pagination for all tweets")
        logger.info("✓ JSONL debug logging for all API calls")
        logger.info("✓ User-wise JSON output organization")
        logger.info("✓ DUAL APPROACH: Real-time JSONL + Batch JSON")
        logger.info("="*60)

        # Run processing
        result_file = processor.process_deal(deal_id)

        if result_file:
            logger.info("✅ Test completed successfully!")
            logger.info(f"Results summary saved to: {result_file}")

            # Print summary of what was processed
            logger.info("\n" + "="*60)
            logger.info("TEST SUMMARY")
            logger.info("="*60)
            logger.info(f"Deal ID: {deal_id}")
            logger.info(f"Summary file: {result_file}")
            logger.info(f"Output directory: {processor.output_dir}")
            logger.info(f"Debug JSONL file: {processor.jsonl_file}")
            logger.info(
                f"Tweet responses JSONL file: {processor.tweet_responses_file}")

            # Verify dual approach is working
            logger.info("\n🔍 Verifying Dual Approach (JSONL + JSON):")
            verification = processor.verify_dual_approach()
            logger.info(f"  Status: {verification['status']}")

            if verification['jsonl_files']['tweet_responses'].get('exists'):
                size_kb = verification['jsonl_files']['tweet_responses']['size_bytes'] / 1024
                logger.info(f"  📝 Tweet responses JSONL: {size_kb:.1f} KB")

            if verification['jsonl_files']['api_debug'].get('exists'):
                size_kb = verification['jsonl_files']['api_debug']['size_bytes'] / 1024
                logger.info(f"  🐛 API debug JSONL: {size_kb:.1f} KB")

            logger.info(
                f"  📊 JSON files generated: {len(verification['json_files'])}")

            # List output files
            logger.info("\nGenerated files:")
            if os.path.exists(processor.output_dir):
                for file in os.listdir(processor.output_dir):
                    if file.startswith(f"tweet_search_summary_{deal_id}"):
                        logger.info(f"  📊 Summary: {file}")
                    elif file.startswith(f"tweet_details_"):
                        logger.info(f"  📝 Tweet Details: {file}")
                    elif file.startswith(f"api_calls_debug_"):
                        logger.info(f"  🐛 Debug Log: {file}")

            # Check filter directory
            filter_dir = os.path.join(
                os.path.dirname(processor.output_dir), 'filter')
            if os.path.exists(filter_dir):
                logger.info(f"\nFilter files in: {filter_dir}")
                for file in os.listdir(filter_dir):
                    if file.startswith("filtered_"):
                        logger.info(f"  🔍 Filtered: {file}")

            logger.info("="*60)

        else:
            logger.error("❌ Test failed - no results generated")

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    # Test basic functionality
    test_tweet_search_processor_test()

    print("\n🎉 All tests completed! Check the output files for results.")
