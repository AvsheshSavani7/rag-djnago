#!/usr/bin/env python3
"""
Test High Value Followers Processor
Test script for the High Value Followers Processor with different configurations.

Usage:
# python test_high_value_followers.py
# python test_high_value_followers.py --deal-id 68ac4a254a6006a0946ec3bb
# python test_high_value_followers.py --deal-id 68ac4a254a6006a0946ec3bb --max-followers 10 --min-score 7

 python test_high_value_followers.py \
  --deal-id 68ac4a254a6006a0946ec3bb

"""

from document_processor.twitter_utils.high_value_followers_processor import HighValueFollowersProcessor
import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the processor


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def test_processor_configuration():
    """Test the processor configuration and dependencies"""
    logger = logging.getLogger(__name__)
    logger.info("=== Testing Processor Configuration ===")

    try:
        # Test OpenAI API key
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            logger.error("❌ OPENAI_API_KEY environment variable not set")
            return False
        else:
            logger.info("✅ OPENAI_API_KEY found")

        # Test Twitter API key (optional)
        twitter_api_key = os.getenv('TWITTER_API_KEY')
        if twitter_api_key:
            logger.info("✅ TWITTER_API_KEY found")
        else:
            logger.warning("⚠️  TWITTER_API_KEY not set (optional)")

        # Test processor initialization
        processor = HighValueFollowersProcessor()
        logger.info("✅ HighValueFollowersProcessor initialized successfully")

        return True

    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


def test_deal_data_fetching(deal_id: str):
    """Test fetching deal data from database"""
    logger = logging.getLogger(__name__)
    logger.info(f"=== Testing Deal Data Fetching for Deal ID: {deal_id} ===")

    try:
        processor = HighValueFollowersProcessor()

        # Test fetching deal data
        deal_data = processor.fetch_deal_data(deal_id)
        if not deal_data:
            logger.error(f"❌ Deal with ID {deal_id} not found")
            return False

        logger.info("✅ Deal data fetched successfully")
        logger.info(f"   Target Company: {deal_data.get('target_name')}")
        logger.info(f"   Acquire Company: {deal_data.get('acquire_name')}")
        logger.info(f"   CIK: {deal_data.get('cik')}")

        # Test Twitter handles extraction
        twitter_handles = processor.extract_twitter_handles(deal_data)
        if not twitter_handles:
            logger.warning("⚠️  No Twitter handles found in deal data")
            return False

        logger.info("✅ Twitter handles extracted successfully")
        for company, handle in twitter_handles.items():
            logger.info(f"   {company}: @{handle}")

        return True

    except Exception as e:
        logger.error(f"❌ Deal data fetching test failed: {e}")
        return False


def test_follower_retrieval(deal_id: str, company_handle: str):
    """Test retrieving followers from database"""
    logger = logging.getLogger(__name__)
    logger.info(f"=== Testing Follower Retrieval for @{company_handle} ===")

    try:
        processor = HighValueFollowersProcessor()

        # Test follower retrieval
        followers = processor.follower_utils.get_followers_for_company(
            deal_id, company_handle, processor.config['approach'])

        if not followers:
            logger.warning(f"⚠️  No followers found for @{company_handle}")
            return False

        logger.info(
            f"✅ Retrieved {len(followers)} followers for @{company_handle}")

        # Test filtering
        filtered_followers = processor.filter_followers(followers)
        logger.info(f"✅ Filtered to {len(filtered_followers)} followers")

        if filtered_followers:
            # Show sample follower data
            sample_follower = filtered_followers[0]
            logger.info("Sample follower data:")
            logger.info(f"   Name: {sample_follower.get('name')}")
            logger.info(
                f"   Screen Name: @{sample_follower.get('screen_name')}")
            logger.info(
                f"   Followers Count: {sample_follower.get('followers_count')}")
            logger.info(
                f"   Statuses Count: {sample_follower.get('statuses_count')}")
            logger.info(
                f"   Description: {sample_follower.get('description', '')[:100]}...")

        return True

    except Exception as e:
        logger.error(f"❌ Follower retrieval test failed: {e}")
        return False


def test_gpt_analysis(deal_id: str, company_handle: str, max_test_followers: int = 2):
    """Test GPT analysis with a few followers"""
    logger = logging.getLogger(__name__)
    logger.info(
        f"=== Testing GPT Analysis for @{company_handle} (max {max_test_followers} followers) ===")

    try:
        processor = HighValueFollowersProcessor()

        # Get followers
        followers = processor.follower_utils.get_followers_for_company(
            deal_id, company_handle, processor.config['approach'])
        if not followers:
            logger.warning(f"⚠️  No followers found for @{company_handle}")
            return False

        # Filter followers
        filtered_followers = processor.filter_followers(followers)
        if not filtered_followers:
            logger.warning(
                f"⚠️  No filtered followers found for @{company_handle}")
            return False

        # Test with limited followers
        test_followers = filtered_followers[:max_test_followers]
        logger.info(
            f"Testing GPT analysis with {len(test_followers)} followers")

        for i, follower in enumerate(test_followers, 1):
            logger.info(
                f"Analyzing follower {i}/{len(test_followers)}: @{follower.get('screen_name')}")

            # Test GPT analysis
            analysis = processor.analyze_follower_with_gpt(follower)
            if analysis:
                logger.info(f"✅ GPT analysis successful:")
                logger.info(
                    f"   Overall Score: {analysis.get('overall_score')}")
                logger.info(
                    f"   Key Indicators: {analysis.get('key_indicators', [])}")
            else:
                logger.warning(
                    f"⚠️  GPT analysis failed for @{follower.get('screen_name')}")

        return True

    except Exception as e:
        logger.error(f"❌ GPT analysis test failed: {e}")
        return False


def run_full_test(deal_id: str, max_followers: int = None, min_score: int = 0, config_overrides: Dict[str, Any] = None):
    """Run a full test of the processor with limited followers"""
    logger = logging.getLogger(__name__)
    logger.info("=== Running Full Processor Test ===")

    try:
        # Use provided config overrides or create defaults
        if config_overrides is None:
            config_overrides = {
                'min_followers_count': 100,
                'min_statuses_count': 100,
                'min_overall_score': min_score
            }

            if max_followers is not None:
                config_overrides['max_followers_per_company'] = max_followers

        processor = HighValueFollowersProcessor(
            config_overrides=config_overrides)

        # Run the full process
        result_file = processor.process_deal(deal_id)

        if result_file:
            logger.info(f"✅ Full test completed successfully!")
            logger.info(f"Results saved to: {result_file}")
            return True
        else:
            logger.error("❌ Full test failed")
            return False

    except Exception as e:
        logger.error(f"❌ Full test failed: {e}")
        return False


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(
        description='Test High Value Followers Processor')
    parser.add_argument('--deal-id', type=str, help='Deal ID to test with')
    parser.add_argument('--max-followers', type=int, default=None,
                        help='Maximum followers to process per company (default: None = process all)')
    parser.add_argument('--min-score', type=int, default=0,
                        help='Minimum GPT score to save (default: 0)')
    parser.add_argument('--min-followers', type=int, default=250,
                        help='Minimum followers count for filtering (default: 250)')
    parser.add_argument('--min-statuses', type=int, default=250,
                        help='Minimum statuses count for filtering (default: 250)')
    parser.add_argument('--gpt-model', type=str, default='gpt-4o-mini',
                        help='GPT model to use (default: gpt-4o-mini)')
    parser.add_argument('--workers', type=int, default=70,
                        help='Number of parallel workers (default: 1 = sequential)')
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel processing with 4 workers')
    parser.add_argument('--test-mode', choices=['config', 'deal-data', 'followers', 'gpt', 'full'],
                        default='full', help='Test mode to run')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    logger.info("🧪 High Value Followers Processor Test Suite")
    logger.info("=" * 50)

    # Test configuration
    if not test_processor_configuration():
        logger.error("❌ Configuration test failed. Exiting.")
        sys.exit(1)

    # If no deal ID provided, show available options
    if not args.deal_id:
        logger.error("❌ Please provide a deal ID using --deal-id")
        logger.info("Example deal IDs you can test with:")
        logger.info("  - 682f00def21b9fca8e1d04fe")
        logger.info("  - 68184d52478abf06ec1a28ec")
        logger.info("  - (or any other deal ID from your database)")
        sys.exit(1)

    deal_id = args.deal_id
    logger.info(f"Testing with Deal ID: {deal_id}")

    # Run tests based on mode
    if args.test_mode == 'config':
        logger.info("✅ Configuration test completed")

    elif args.test_mode == 'deal-data':
        if test_deal_data_fetching(deal_id):
            logger.info("✅ Deal data test completed")
        else:
            logger.error("❌ Deal data test failed")
            sys.exit(1)

    elif args.test_mode == 'followers':
        # Test with first available company
        processor = HighValueFollowersProcessor()
        deal_data = processor.fetch_deal_data(deal_id)
        if deal_data:
            twitter_handles = processor.extract_twitter_handles(deal_data)
            if twitter_handles:
                first_handle = list(twitter_handles.values())[0]
                if test_follower_retrieval(deal_id, first_handle):
                    logger.info("✅ Followers test completed")
                else:
                    logger.error("❌ Followers test failed")
                    sys.exit(1)

    elif args.test_mode == 'gpt':
        # Test with first available company
        processor = HighValueFollowersProcessor()
        deal_data = processor.fetch_deal_data(deal_id)
        if deal_data:
            twitter_handles = processor.extract_twitter_handles(deal_data)
            if twitter_handles:
                first_handle = list(twitter_handles.values())[0]
                if test_gpt_analysis(deal_id, first_handle, max_test_followers=2):
                    logger.info("✅ GPT analysis test completed")
                else:
                    logger.error("❌ GPT analysis test failed")
                    sys.exit(1)

    elif args.test_mode == 'full':
        # Create config overrides from command line arguments
        config_overrides = {
            'min_followers_count': args.min_followers,
            'min_statuses_count': args.min_statuses,
            'min_overall_score': args.min_score,
            'gpt_model': args.gpt_model,
            'max_workers': args.workers,
            'use_parallel_processing': args.parallel or args.workers > 1
        }

        if args.max_followers is not None:
            config_overrides['max_followers_per_company'] = args.max_followers
            logger.info(
                f"Running full test with max {args.max_followers} followers per company, min score {args.min_score}")
        else:
            logger.info(
                f"Running full test with ALL followers per company, min score {args.min_score}")

        if run_full_test(deal_id, args.max_followers, args.min_score, config_overrides):
            logger.info("✅ Full test completed")
        else:
            logger.error("❌ Full test failed")
            sys.exit(1)

    logger.info("🎉 All tests completed successfully!")


if __name__ == "__main__":
    main()
