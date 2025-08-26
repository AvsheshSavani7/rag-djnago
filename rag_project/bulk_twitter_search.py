#!/usr/bin/env python3
"""
Bulk Twitter Search Script for All Deals
This script runs Twitter search for all deals in the database and saves results.

below command to run bulk deal search
# python bulk_twitter_search.py --max-deals 1 

"""

from document_processor.twitter_utils.riffle_approach_1 import DealTwitterAnalyzer
from document_processor.models import ProcessingJob
import django
import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import time

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()


class BulkTwitterSearch:
    """Bulk Twitter search processor for all deals"""

    def __init__(self, output_dir: str = None):
        """
        Initialize bulk search processor

        Args:
            output_dir: Directory to save results (defaults to twitter_utils/twitter_search_results)
        """
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(__file__),
                'document_processor',
                'twitter_utils',
                'twitter_search_results'
            )

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize Twitter analyzer
        self.analyzer = DealTwitterAnalyzer()

        # Setup logging
        self.setup_logging()

        # Statistics
        self.stats = {
            'total_deals': 0,
            'processed_deals': 0,
            'successful_searches': 0,
            'failed_searches': 0,
            'total_tweets_found': 0,
            'start_time': None,
            'end_time': None
        }

    def setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory
        logs_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        # Create log filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'bulk_twitter_search_{timestamp}.log'
        log_filepath = os.path.join(logs_dir, log_filename)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filepath),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Bulk Twitter search started. Log file: {log_filepath}")

    def get_all_deals(self) -> List[ProcessingJob]:
        """
        Get all deals from the database

        Returns:
            List of ProcessingJob objects
        """
        try:
            deals = ProcessingJob.objects.all().order_by('-createdAt')
            self.logger.info(f"Found {len(deals)} deals in database")
            return deals
        except Exception as e:
            self.logger.error(f"Error fetching deals from database: {e}")
            return []

    def process_single_deal(self, deal: ProcessingJob, force_reprocess: bool = False) -> Optional[Dict[str, Any]]:
        """
        Process a single deal for Twitter search

        Args:
            deal: ProcessingJob object
            force_reprocess: If True, force reprocessing even if RF1_approach_done is True

        Returns:
            Dictionary with search results or None if failed
        """
        deal_id = str(deal.id)
        target_name = deal.target_name or "Unknown"

        self.logger.info(f"Processing deal {deal_id}: {target_name}")

        # Check if RF1 approach is already done (unless forcing reprocess)
        if deal.RF1_approach_done and not force_reprocess:
            self.logger.info(
                f"✅ Deal {deal_id} already processed with RF1 approach. Skipping.")
            return {
                'deal_id': deal_id,
                'target_name': target_name,
                'acquire_name': deal.acquire_name,
                'announce_date': deal.announce_date.strftime('%Y-%m-%d') if deal.announce_date else None,
                'result_file': None,
                'total_tweets': 0,
                'combinations_searched': 0,
                'status': 'skipped',
                'message': 'RF1 approach already completed',
                'tweet_data': []
            }

        try:
            # Run the Twitter search for this deal (don't save individual files in bulk mode)
            result_file = self.analyzer.analyze_deal(
                deal_id, force_reprocess=force_reprocess, save_file=False)

            if result_file == "SKIPPED":
                self.logger.info(
                    f"✅ Deal {deal_id} already processed with RF1 approach. Skipping.")
                return {
                    'deal_id': deal_id,
                    'target_name': target_name,
                    'acquire_name': deal.acquire_name,
                    'announce_date': deal.announce_date.strftime('%Y-%m-%d') if deal.announce_date else None,
                    'result_file': None,
                    'total_tweets': 0,
                    'combinations_searched': 0,
                    'status': 'skipped',
                    'message': 'RF1 approach already completed',
                    'tweet_data': []
                }
            elif result_file:
                # Read the result file to get tweet count and data
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)

                total_tweets = sum(r['tweet_count']
                                   for r in result_data.get('results', []))

                # Extract all tweet data from the results
                all_tweets = []
                for combination_result in result_data.get('results', []):
                    tweets = combination_result.get('tweets', [])
                    for tweet in tweets:
                        tweet_with_metadata = {
                            'deal_id': deal_id,
                            'target_name': target_name,
                            'acquire_name': deal.acquire_name,
                            'combination': combination_result.get('combination', {}),
                            'search_query': combination_result.get('search_query', ''),
                            'tweet': tweet
                        }
                        all_tweets.append(tweet_with_metadata)

                self.logger.info(
                    f"✅ Deal {deal_id} completed. Found {total_tweets} tweets")

                return {
                    'deal_id': deal_id,
                    'target_name': target_name,
                    'acquire_name': deal.acquire_name,
                    'announce_date': deal.announce_date.strftime('%Y-%m-%d') if deal.announce_date else None,
                    'result_file': result_file,
                    'total_tweets': total_tweets,
                    'combinations_searched': len(result_data.get('results', [])),
                    'status': 'success',
                    'tweet_data': all_tweets
                }
            elif result_file is None:
                # No file returned but analysis completed (bulk mode)
                # We need to get the results from the database
                from document_processor.models import SearchQuery, Tweet

                # Get all search queries for this deal with RF1 approach
                search_queries = SearchQuery.objects(
                    deal_id=deal_id, approach="RF1")

                total_tweets = 0
                all_tweets = []

                for sq in search_queries:
                    # Get tweets for this search query
                    tweets = Tweet.objects(search_query_id=sq.id)
                    total_tweets += len(tweets)

                    for tweet in tweets:
                        tweet_with_metadata = {
                            'deal_id': deal_id,
                            'target_name': target_name,
                            'acquire_name': deal.acquire_name,
                            'combination': sq.combination,
                            'search_query': sq.search_query,
                            'tweet': tweet.tweet
                        }
                        all_tweets.append(tweet_with_metadata)

                self.logger.info(
                    f"✅ Deal {deal_id} completed. Found {total_tweets} tweets (from database)")

                return {
                    'deal_id': deal_id,
                    'target_name': target_name,
                    'acquire_name': deal.acquire_name,
                    'announce_date': deal.announce_date.strftime('%Y-%m-%d') if deal.announce_date else None,
                    'result_file': None,
                    'total_tweets': total_tweets,
                    'combinations_searched': len(search_queries),
                    'status': 'success',
                    'tweet_data': all_tweets
                }
            else:
                self.logger.warning(f"❌ Deal {deal_id} failed to process")
                return {
                    'deal_id': deal_id,
                    'target_name': target_name,
                    'acquire_name': deal.acquire_name,
                    'announce_date': deal.announce_date.strftime('%Y-%m-%d') if deal.announce_date else None,
                    'result_file': None,
                    'total_tweets': 0,
                    'combinations_searched': 0,
                    'status': 'failed',
                    'tweet_data': []
                }

        except Exception as e:
            self.logger.error(f"❌ Error processing deal {deal_id}: {e}")
            return {
                'deal_id': deal_id,
                'target_name': target_name,
                'acquire_name': deal.acquire_name,
                'announce_date': deal.announce_date.strftime('%Y-%m-%d') if deal.announce_date else None,
                'result_file': None,
                'total_tweets': 0,
                'combinations_searched': 0,
                'status': 'error',
                'error_message': str(e),
                'tweet_data': []
            }

    def save_summary_report(self, results: List[Dict[str, Any]]):
        """
        Save a summary report of all processed deals

        Args:
            results: List of deal processing results
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_filename = f'bulk_twitter_search_summary_{timestamp}.json'
        summary_filepath = os.path.join(self.output_dir, summary_filename)

        # Calculate statistics
        successful_deals = [r for r in results if r['status'] == 'success']
        failed_deals = [r for r in results if r['status']
                        in ['failed', 'error']]
        skipped_deals = [r for r in results if r['status'] == 'skipped']
        total_tweets = sum(r['total_tweets'] for r in results)

        summary_data = {
            'search_timestamp': datetime.now().isoformat(),
            'total_deals_processed': len(results),
            'successful_deals': len(successful_deals),
            'failed_deals': len(failed_deals),
            'skipped_deals': len(skipped_deals),
            'total_tweets_found': total_tweets,
            'execution_time_seconds': (self.stats['end_time'] - self.stats['start_time']).total_seconds() if self.stats['end_time'] else None,
            'results': results
        }

        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Summary report saved to: {summary_filepath}")
        return summary_filepath

    def save_target_company_summary(self, results: List[Dict[str, Any]]):
        """
        Save a summary organized by target company name

        Args:
            results: List of deal processing results
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        target_summary_filename = f'target_company_twitter_summary_{timestamp}.json'
        target_summary_filepath = os.path.join(
            self.output_dir, target_summary_filename)

        # Group by target company
        target_companies = {}
        for result in results:
            target_name = result['target_name']
            if target_name not in target_companies:
                target_companies[target_name] = {
                    'target_name': target_name,
                    'deal_count': 0,
                    'total_tweets': 0,
                    'deals': []
                }

            target_companies[target_name]['deal_count'] += 1
            target_companies[target_name]['total_tweets'] += result['total_tweets']
            target_companies[target_name]['deals'].append({
                'deal_id': result['deal_id'],
                'acquire_name': result['acquire_name'],
                'announce_date': result['announce_date'],
                'tweet_count': result['total_tweets'],
                'status': result['status']
            })

        # Convert to list and sort by total tweets
        target_summary = list(target_companies.values())
        target_summary.sort(key=lambda x: x['total_tweets'], reverse=True)

        summary_data = {
            'search_timestamp': datetime.now().isoformat(),
            'total_target_companies': len(target_summary),
            'total_tweets_across_all_targets': sum(t['total_tweets'] for t in target_summary),
            'target_companies': target_summary
        }

        with open(target_summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        self.logger.info(
            f"Target company summary saved to: {target_summary_filepath}")
        return target_summary_filepath

    def save_consolidated_tweets_file(self, results: List[Dict[str, Any]]):
        """
        Save all tweets from all deals in a single consolidated file

        Args:
            results: List of deal processing results
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        consolidated_filename = f'consolidated_tweets_rf1_{timestamp}.json'
        consolidated_filepath = os.path.join(
            self.output_dir, consolidated_filename)

        # Collect all tweets from all deals
        all_tweets = []
        for result in results:
            if result['status'] == 'success' and result['tweet_data']:
                all_tweets.extend(result['tweet_data'])

        # Create consolidated data structure
        consolidated_data = {
            'search_timestamp': datetime.now().isoformat(),
            'approach': 'RF1',
            'total_deals_processed': len(results),
            'total_tweets_consolidated': len(all_tweets),
            'deals_summary': [
                {
                    'deal_id': result['deal_id'],
                    'target_name': result['target_name'],
                    'acquire_name': result['acquire_name'],
                    'announce_date': result['announce_date'],
                    'status': result['status'],
                    'tweet_count': result['total_tweets'],
                    'combinations_searched': result['combinations_searched']
                }
                for result in results
            ],
            'all_tweets': all_tweets
        }

        # Save to file
        with open(consolidated_filepath, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, indent=2, ensure_ascii=False)

        self.logger.info(
            f"Consolidated tweets file saved to: {consolidated_filepath}")
        self.logger.info(f"Total tweets consolidated: {len(all_tweets)}")
        return consolidated_filepath

    def run_bulk_search(self, max_deals: Optional[int] = None, delay_seconds: int = 2, force_reprocess: bool = False):
        """
        Run Twitter search for all deals in the database

        Args:
            max_deals: Maximum number of deals to process (None for all)
            delay_seconds: Delay between processing deals to avoid rate limiting
            force_reprocess: If True, force reprocessing even if RF1_approach_done is True
        """
        self.stats['start_time'] = datetime.now()
        self.logger.info("🚀 Starting bulk Twitter search for all deals")

        # Get all deals
        deals = self.get_all_deals()
        if not deals:
            self.logger.error("No deals found in database")
            return

        # Limit deals if specified
        if max_deals:
            deals = deals[:max_deals]
            self.logger.info(f"Processing first {max_deals} deals")

        self.stats['total_deals'] = len(deals)
        self.logger.info(f"Processing {len(deals)} deals")

        # Process each deal
        results = []
        for i, deal in enumerate(deals, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing deal {i}/{len(deals)}")

            result = self.process_single_deal(
                deal, force_reprocess=force_reprocess)
            if result:
                results.append(result)

                if result['status'] == 'success':
                    self.stats['successful_searches'] += 1
                    self.stats['total_tweets_found'] += result['total_tweets']
                elif result['status'] == 'skipped':
                    # Don't count skipped as failed
                    pass
                else:
                    self.stats['failed_searches'] += 1

            self.stats['processed_deals'] += 1

            # Progress update
            progress = (i / len(deals)) * 100
            self.logger.info(f"Progress: {progress:.1f}% ({i}/{len(deals)})")

            # Delay between deals to avoid rate limiting
            if i < len(deals):
                self.logger.info(
                    f"Waiting {delay_seconds} seconds before next deal...")
                time.sleep(delay_seconds)

        self.stats['end_time'] = datetime.now()

        # Save summary reports
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 Generating summary reports...")

        summary_file = self.save_summary_report(results)
        target_summary_file = self.save_target_company_summary(results)
        consolidated_file = self.save_consolidated_tweets_file(results)

        # Final statistics
        execution_time = (self.stats['end_time'] -
                          self.stats['start_time']).total_seconds()
        self.logger.info("\n" + "="*60)
        self.logger.info("🎉 BULK TWITTER SEARCH COMPLETED!")
        self.logger.info(
            f"Total deals processed: {self.stats['processed_deals']}")
        self.logger.info(
            f"Successful searches: {self.stats['successful_searches']}")
        self.logger.info(f"Failed searches: {self.stats['failed_searches']}")
        self.logger.info(
            f"Total tweets found: {self.stats['total_tweets_found']}")
        self.logger.info(f"Execution time: {execution_time:.2f} seconds")
        self.logger.info(f"Summary report: {summary_file}")
        self.logger.info(f"Target company summary: {target_summary_file}")
        self.logger.info(f"Consolidated tweets file: {consolidated_file}")
        self.logger.info("="*60)


def main():
    """Main function to run bulk Twitter search"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Bulk Twitter search for all deals')
    parser.add_argument('--max-deals', type=int,
                        help='Maximum number of deals to process')
    parser.add_argument('--delay', type=int, default=2,
                        help='Delay between deals in seconds (default: 2)')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for results')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing even if RF1_approach_done is True')

    args = parser.parse_args()

    try:
        # Initialize bulk search processor
        bulk_processor = BulkTwitterSearch(output_dir=args.output_dir)

        # Run bulk search
        bulk_processor.run_bulk_search(
            max_deals=args.max_deals,
            delay_seconds=args.delay,
            force_reprocess=args.force
        )

    except KeyboardInterrupt:
        print("\n⚠️  Bulk search interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during bulk search: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
