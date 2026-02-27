#!/usr/bin/env python3
"""
Runner script for High Value Followers Tweet Analyzer
"""

from document_processor.twitter_utils.high_value_followers_tweet_analyzer import HighValueFollowersTweetAnalyzer
import sys
import logging


def main():
    """Main function to run the tweet analyzer"""
    if len(sys.argv) < 2:
        print(
            "Usage: python run_high_value_followers_tweet_analyzer.py <deal_id> [options]")
        print("Options:")
        print("  --workers <number>           Number of parallel workers (default: 5)")
        print("  --max-tweets <number>        Max tweets per user (default: 100)")
        print("  --max-tweets-all             Fetch all available tweets per user")
        print("  --gpt-model <model>          GPT model to use (default: gpt-4o-mini)")
        print("  --skip-gpt-analysis          Skip GPT analysis (testing mode)")
        print("  --skip-mongodb-save          Skip MongoDB saving (testing mode)")
        print("Examples:")
        print(
            "  python run_high_value_followers_tweet_analyzer.py 68184d52478abf06ec1a28ec")
        print("  python run_high_value_followers_tweet_analyzer.py 68184d52478abf06ec1a28ec --workers 10")
        sys.exit(1)

    deal_id = sys.argv[1]
    config_overrides = {}

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--workers' and i + 1 < len(sys.argv):
            workers = int(sys.argv[i + 1])
            if workers < 1:
                print("❌ --workers must be at least 1")
                sys.exit(1)
            config_overrides['max_workers'] = workers
            config_overrides['use_parallel_processing'] = workers > 1
            i += 2
        elif sys.argv[i] == '--max-tweets' and i + 1 < len(sys.argv):
            max_tweets = int(sys.argv[i + 1])
            if max_tweets < 1:
                print("❌ --max-tweets must be at least 1")
                sys.exit(1)
            config_overrides['max_tweets_per_user'] = max_tweets
            i += 2
        elif sys.argv[i] == '--max-tweets-all':
            config_overrides['max_tweets_per_user'] = None
            i += 1
        elif sys.argv[i] == '--gpt-model' and i + 1 < len(sys.argv):
            config_overrides['gpt_model'] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--skip-gpt-analysis':
            config_overrides['skip_gpt_analysis'] = True
            i += 1
        elif sys.argv[i] == '--skip-mongodb-save':
            config_overrides['skip_mongodb_save'] = True
            i += 1
        else:
            i += 1

    try:
        # Initialize analyzer with config overrides
        analyzer = HighValueFollowersTweetAnalyzer(
            config_overrides=config_overrides)

        # Run processing
        result_file = analyzer.process_deal(deal_id)

        if result_file:
            print("✅ High Value Followers Tweet Analysis completed successfully!")
            print(f"Results saved to: {result_file}")
        else:
            print("❌ Processing failed or no results found")
            sys.exit(1)

    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
