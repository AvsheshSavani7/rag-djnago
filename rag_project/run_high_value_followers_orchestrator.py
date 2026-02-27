#!/usr/bin/env python3
"""
High Value Followers Orchestrator Runner
Script to run the high value followers orchestrator from the rag_project level

Usage:
# python run_high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb
# python run_high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb --disable-step-1
# python run_high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb --gpt-workers 100 --tweet-workers 20
# python run_high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb --disable-step-1 --disable-step-2 --disable-step-3

python high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb \
  --disable-step-1 --disable-step-2 --disable-step-3
"""

from document_processor.twitter_utils.high_value_followers_orchestrator import HighValueFollowersOrchestrator
import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import the orchestrator


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def print_usage():
    """Print usage information"""
    logger = logging.getLogger(__name__)
    logger.error(
        "Usage: python run_high_value_followers_orchestrator.py <deal_id> [options]")
    logger.error("")
    logger.error("Options:")
    logger.error(
        "  --disable-step-1          Disable Gun Shot Approach (follower fetching)")
    logger.error("  --disable-step-2          Disable GPT Analysis")
    logger.error("  --disable-step-3          Disable Tweet Search")
    logger.error("  --disable-step-4          Disable Tweet Analysis")
    logger.error(
        "  --gpt-workers <number>    GPT analysis workers (default: 70)")
    logger.error(
        "  --tweet-workers <number>  Tweet search workers (default: 10)")
    logger.error(
        "  --tweet-analysis-workers <n> Tweet analysis workers (default: 5)")
    logger.error(
        "  --gpt-max-followers <n>   Max followers for GPT (default: 10)")
    logger.error(
        "  --tweet-max-followers <n> Max followers for tweets (default: 200)")
    logger.error(
        "  --tweet-analysis-max-tweets <n> Max tweets per user for analysis (default: 100)")
    logger.error(
        "  --min-score <n>           Minimum GPT score to save (default: 0)")
    logger.error(
        "  --batch-size <n>          Batch size for MongoDB saves (default: 1000)")
    logger.error("")
    logger.error("Examples:")
    logger.error("  # Run all steps")
    logger.error(
        "  python run_high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb")
    logger.error("")
    logger.error("  # Skip follower fetching (if already done)")
    logger.error(
        "  python run_high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb --disable-step-1")
    logger.error("")
    logger.error("  # GPT analysis only")
    logger.error(
        "  python run_high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb --disable-step-1 --disable-step-3")
    logger.error("")
    logger.error("  # Tweet search only")
    logger.error(
        "  python run_high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb --disable-step-1 --disable-step-2")
    logger.error("")
    logger.error("  # Tweet analysis only")
    logger.error(
        "  python run_high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb --disable-step-1 --disable-step-2 --disable-step-3")
    logger.error("")
    logger.error("  # High-throughput configuration")
    logger.error("  python run_high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb --gpt-workers 100 --tweet-workers 20 --tweet-analysis-workers 10 --gpt-max-followers 50")
    logger.error("")
    logger.error("  # Stricter filtering")
    logger.error(
        "  python run_high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb --min-score 5 --gpt-max-followers 20")


def parse_arguments():
    """Parse command line arguments"""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    deal_id = sys.argv[1]
    config_overrides = {}

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--disable-step-1':
            config_overrides['enable_step_1_gun_shot'] = False
            i += 1
        elif sys.argv[i] == '--disable-step-2':
            config_overrides['enable_step_2_gpt_analysis'] = False
            i += 1
        elif sys.argv[i] == '--disable-step-3':
            config_overrides['enable_step_3_tweet_search'] = False
            i += 1
        elif sys.argv[i] == '--disable-step-4':
            config_overrides['enable_step_4_tweet_analysis'] = False
            i += 1
        elif sys.argv[i] == '--gpt-workers' and i + 1 < len(sys.argv):
            workers = int(sys.argv[i + 1])
            if workers < 1:
                logger.error("❌ --gpt-workers must be at least 1")
                sys.exit(1)
            config_overrides['gpt_max_workers'] = workers
            i += 2
        elif sys.argv[i] == '--tweet-workers' and i + 1 < len(sys.argv):
            workers = int(sys.argv[i + 1])
            if workers < 1:
                logger.error("❌ --tweet-workers must be at least 1")
                sys.exit(1)
            config_overrides['tweet_search_max_workers'] = workers
            i += 2
        elif sys.argv[i] == '--tweet-analysis-workers' and i + 1 < len(sys.argv):
            workers = int(sys.argv[i + 1])
            if workers < 1:
                logger.error("❌ --tweet-analysis-workers must be at least 1")
                sys.exit(1)
            config_overrides['tweet_analysis_max_workers'] = workers
            i += 2
        elif sys.argv[i] == '--gpt-max-followers' and i + 1 < len(sys.argv):
            max_followers = int(sys.argv[i + 1])
            if max_followers <= 0:
                logger.error("❌ --gpt-max-followers must be greater than 0")
                sys.exit(1)
            config_overrides['gpt_max_followers_per_company'] = max_followers
            i += 2
        elif sys.argv[i] == '--tweet-max-followers' and i + 1 < len(sys.argv):
            max_followers = int(sys.argv[i + 1])
            if max_followers <= 0:
                logger.error("❌ --tweet-max-followers must be greater than 0")
                sys.exit(1)
            config_overrides['tweet_search_max_followers_per_company'] = max_followers
            i += 2
        elif sys.argv[i] == '--tweet-analysis-max-tweets' and i + 1 < len(sys.argv):
            max_tweets = int(sys.argv[i + 1])
            if max_tweets <= 0:
                logger.error(
                    "❌ --tweet-analysis-max-tweets must be greater than 0")
                sys.exit(1)
            config_overrides['tweet_analysis_max_tweets_per_user'] = max_tweets
            i += 2
        elif sys.argv[i] == '--min-score' and i + 1 < len(sys.argv):
            min_score = int(sys.argv[i + 1])
            if min_score < 0 or min_score > 10:
                logger.error("❌ --min-score must be between 0 and 10")
                sys.exit(1)
            config_overrides['gpt_min_overall_score'] = min_score
            i += 2
        elif sys.argv[i] == '--batch-size' and i + 1 < len(sys.argv):
            batch_size = int(sys.argv[i + 1])
            if batch_size < 1:
                logger.error("❌ --batch-size must be at least 1")
                sys.exit(1)
            config_overrides['batch_size'] = batch_size
            i += 2
        else:
            logger.error(f"❌ Unknown option: {sys.argv[i]}")
            print_usage()
            sys.exit(1)

    return deal_id, config_overrides


def run_orchestrator(deal_id: str, config_overrides: dict):
    """Run the orchestrator with given configuration"""
    logger = logging.getLogger(__name__)

    try:
        # Initialize orchestrator with config overrides
        logger.info("🚀 Initializing High Value Followers Orchestrator...")
        orchestrator = HighValueFollowersOrchestrator(
            config_overrides=config_overrides)

        # Display configuration
        logger.info("📋 Configuration:")
        logger.info(f"  Deal ID: {deal_id}")
        logger.info(
            f"  Step 1 (Gun Shot): {'✓ Enabled' if orchestrator.config['enable_step_1_gun_shot'] else '✗ Disabled'}")
        logger.info(
            f"  Step 2 (GPT Analysis): {'✓ Enabled' if orchestrator.config['enable_step_2_gpt_analysis'] else '✗ Disabled'}")
        logger.info(
            f"  Step 3 (Tweet Search): {'✓ Enabled' if orchestrator.config['enable_step_3_tweet_search'] else '✗ Disabled'}")
        logger.info(
            f"  Step 4 (Tweet Analysis): {'✓ Enabled' if orchestrator.config['enable_step_4_tweet_analysis'] else '✗ Disabled'}")

        if orchestrator.config['enable_step_2_gpt_analysis']:
            logger.info(
                f"  GPT Workers: {orchestrator.config['gpt_max_workers']}")
            logger.info(
                f"  GPT Max Followers: {orchestrator.config['gpt_max_followers_per_company']}")
            logger.info(
                f"  GPT Min Score: {orchestrator.config['gpt_min_overall_score']}")

        if orchestrator.config['enable_step_3_tweet_search']:
            logger.info(
                f"  Tweet Workers: {orchestrator.config['tweet_search_max_workers']}")
            logger.info(
                f"  Tweet Max Followers: {orchestrator.config['tweet_search_max_followers_per_company']}")

        if orchestrator.config['enable_step_4_tweet_analysis']:
            logger.info(
                f"  Tweet Analysis Workers: {orchestrator.config['tweet_analysis_max_workers']}")
            logger.info(
                f"  Tweet Analysis Max Tweets: {orchestrator.config['tweet_analysis_max_tweets_per_user']}")

        logger.info(f"  Batch Size: {orchestrator.config['batch_size']}")
        logger.info("")

        # Run processing
        logger.info("🔄 Starting processing...")
        result_file = orchestrator.process_deal(deal_id)

        if result_file:
            logger.info("✅ Orchestrator processing completed successfully!")
            logger.info(f"📁 Results saved to: {result_file}")

            # Show file location relative to current directory
            relative_path = os.path.relpath(result_file, current_dir)
            logger.info(f"📂 Relative path: {relative_path}")

            return True
        else:
            logger.error("❌ Processing failed or no results found")
            return False

    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def main():
    """Main function"""
    # Setup logging
    logger = setup_logging()

    logger.info("🎯 High Value Followers Orchestrator Runner")
    logger.info("=" * 50)

    # Parse arguments
    deal_id, config_overrides = parse_arguments()

    # Run orchestrator
    success = run_orchestrator(deal_id, config_overrides)

    if success:
        logger.info("🎉 All done! Check the results file for details.")
        sys.exit(0)
    else:
        logger.error("💥 Processing failed. Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
