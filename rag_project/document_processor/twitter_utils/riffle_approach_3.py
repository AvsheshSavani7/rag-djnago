#!/usr/bin/env python3
"""
Twitter Search Script for Deal Analysis - Approach 3 (Competitive Products)
This script fetches Twitter data for competitive product combinations from M&A deals.

Usage:
# python run_twitter_search.py 682f00def21b9fca8e1d04fe --approach=3
# For daily cron job: python run_twitter_search.py 682f00def21b9fca8e1d04fe --approach=3 --daily
"""

from .twitter_cleanup_utils import TwitterCleanupUtils
from document_processor.models import ProcessingJob, SearchQuery, Tweet, CompetitiveAnalysis
import django
import os
import sys
import json
import requests
import itertools
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import time
import logging
import re

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()


class CompetitiveProductsTwitterSearchService:
    """Service for searching Twitter using competitive product queries"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Twitter search service

        Args:
            api_key: Twitter API key from twitterapi.io
        """
        self.api_key = api_key or os.getenv('TWITTER_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Twitter API key is required. Set TWITTER_API_KEY environment variable.")

        self.base_url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
        self.headers = {
            'X-API-Key': self.api_key,
            'Content-Type': 'application/json'
        }

        # Setup logger
        self.logger = logging.getLogger(__name__)

    def search_tweets(self, query: str, max_results: int = 5000) -> List[Dict[str, Any]]:
        """
        Search for tweets using the advanced search API

        Args:
            query: Search query string (includes date filters)
            max_results: Maximum number of results to fetch

        Returns:
            List of tweet data
        """
        all_tweets = []
        cursor = ""

        while len(all_tweets) < max_results:
            self.logger.info(f"Searching for tweets with query: {query}")
            params = {
                'query': query,
                'queryType': 'Latest',
                'cursor': cursor
            }

            try:
                response = requests.get(
                    self.base_url, headers=self.headers, params=params)
                response.raise_for_status()

                data = response.json()
                tweets = data.get('tweets', [])
                self.logger.info(
                    f"Found {len(tweets)} tweets for query: {query}")

                if not tweets:
                    break

                all_tweets.extend(tweets)

                # Check if there are more pages
                if not data.get('has_next_page', False):
                    break

                cursor = data.get('next_cursor', '')
                if not cursor:
                    break

                # Rate limiting - wait between requests
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching tweets: {e}")
                break

        return all_tweets[:max_results]


class CompetitiveProductsDealAnalyzer:
    """Main class for analyzing Twitter data using competitive product queries"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the analyzer

        Args:
            api_key: Twitter API key
        """
        self.twitter_service = CompetitiveProductsTwitterSearchService(api_key)
        self.output_dir = os.path.join(
            os.path.dirname(__file__),
            'twitter_search_results'
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger(__name__)

        # Initialize cleanup utilities
        self.cleanup_utils = TwitterCleanupUtils()

    def clean_product_name(self, company_name: str, product_name: str) -> str:
        """
        Clean product name by removing company name prefix

        Args:
            company_name: Name of the company
            product_name: Name of the product

        Returns:
            Cleaned product name without company prefix
        """
        # Get the first word of the company name
        company_first_word = company_name.split()[0]

        # Use regex to remove it only if it's at the start of product_name
        cleaned = re.sub(rf"^{re.escape(company_first_word)}\s+",
                         "", product_name, flags=re.IGNORECASE)

        return cleaned

    def fetch_deal_data(self, deal_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch deal data from database using deal_id

        Args:
            deal_id: MongoDB ObjectId string

        Returns:
            Deal data dictionary or None if not found
        """
        try:
            deal = ProcessingJob.objects.get(id=deal_id)

            # Convert to dictionary format
            deal_data = {
                'id': str(deal.id),
                'cik': deal.cik,
                'acquire_name': deal.acquire_name,
                'target_name': deal.target_name,
                'announce_date': deal.announce_date.strftime('%Y-%m-%d') if deal.announce_date else None,
                'schema_results': deal.schema_results
            }

            return deal_data

        except ProcessingJob.DoesNotExist:
            self.logger.error(f"Deal with ID {deal_id} not found")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching deal data: {e}")
            return None

    def fetch_competitive_products(self, deal_id: str) -> List[Dict[str, Any]]:
        """
        Fetch competitive products from database

        Args:
            deal_id: Deal ID

        Returns:
            List of competitive product pairs
        """
        competitive_pairs = []

        try:
            # Get competitive analysis for this deal
            competitive_analysis = CompetitiveAnalysis.objects(
                deal_id=deal_id).first()

            if competitive_analysis and competitive_analysis.competitive_pairs:
                competitive_pairs = competitive_analysis.competitive_pairs
                self.logger.info(
                    f"Found {len(competitive_pairs)} competitive product pairs for deal {deal_id}")

        except Exception as e:
            self.logger.error(f"Error fetching competitive products: {e}")

        return competitive_pairs

    def build_competitive_product_query(self, target_product: str, acquire_product: str, target_company: str = "", acquire_company: str = "", announce_date: str = None, years_back: int = 5, daily_mode: bool = False) -> str:
        """
        Build Twitter search query for competitive product combination

        Args:
            target_product: Target company product name
            acquire_product: Acquire company product name
            target_company: Target company name (for cleaning product name)
            acquire_company: Acquire company name (for cleaning product name)
            announce_date: Announcement date in YYYY-MM-DD format
            years_back: Number of years to search back from announcement date
            daily_mode: If True, search only last 24 hours using within_time:24h filter

        Returns:
            Complete Twitter search query string with date filters
        """
        # Clean product names by removing company prefixes
        cleaned_target_product = self.clean_product_name(
            target_company, target_product) if target_company else target_product
        cleaned_acquire_product = self.clean_product_name(
            acquire_company, acquire_product) if acquire_company else acquire_product

        # Build the base query with both competitive products and language filter
        query = f'"{cleaned_target_product}" "{cleaned_acquire_product}" lang:en'

        # Determine date range
        if daily_mode:
            # For daily mode, use within_time:24h filter
            query_with_dates = f'{query} within_time:24h'
        else:
            # For initial mode, search last 5 years from today (not announcement date)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years_back * 365)

            # Format dates for Twitter search (YYYY-MM-DD format with UTC time)
            since_date = start_date.strftime('%Y-%m-%d')
            until_date = end_date.strftime('%Y-%m-%d')

            # Add date filters to the query string with UTC time
            query_with_dates = f'{query} since:{since_date}_00:00:00_UTC until:{until_date}_23:59:59_UTC'

        return query_with_dates

    def search_tweets_for_competitive_pair(self, target_product: str, acquire_product: str, target_company: str, acquire_company: str, deal_id: str, announce_date: str = None, daily_mode: bool = False) -> Dict[str, Any]:
        """
        Search tweets for a specific competitive product pair and save to database

        Args:
            target_product: Target company product name
            acquire_product: Acquire company product name
            target_company: Target company name
            acquire_company: Acquire company name
            deal_id: Deal ID for database reference
            announce_date: Announcement date
            daily_mode: If True, search only last 24 hours

        Returns:
            Dictionary containing search results and metadata
        """
        query = self.build_competitive_product_query(
            target_product, acquire_product, target_company, acquire_company, announce_date, daily_mode=daily_mode)

        self.logger.info(
            f"Searching tweets for competitive products: {target_product} vs {acquire_product}")
        self.logger.debug(f"Query: {query}")

        tweets = self.twitter_service.search_tweets(query, max_results=5000)

        # Save search query to database
        combination_data = {
            'target_product': target_product,
            'acquire_product': acquire_product,
            'daily_mode': daily_mode
        }

        search_query_obj = SearchQuery(
            search_query=query,
            deal_id=deal_id,
            approach="RF3",
            combination=combination_data,
            total_tweets=len(tweets)
        )
        search_query_obj.save()

        self.logger.info(
            f"Saved search query to database with ID: {search_query_obj.id}")

        # Save individual tweets to database
        saved_tweet_ids = []
        for tweet_data in tweets:
            tweet_obj = Tweet(
                search_query_id=search_query_obj,
                tweet=tweet_data
            )
            tweet_obj.save()
            saved_tweet_ids.append(str(tweet_obj.id))

        self.logger.info(f"Saved {len(saved_tweet_ids)} tweets to database")

        # Clean product names for reference
        cleaned_target_product = self.clean_product_name(
            target_company, target_product) if target_company else target_product
        cleaned_acquire_product = self.clean_product_name(
            acquire_company, acquire_product) if acquire_company else acquire_product

        result = {
            'target_product': target_product,
            'acquire_product': acquire_product,
            'cleaned_target_product': cleaned_target_product,
            'cleaned_acquire_product': cleaned_acquire_product,
            'query': query,
            'search_query_id': str(search_query_obj.id),
            'daily_mode': daily_mode,
            'search_timestamp': datetime.now().isoformat(),
            'tweet_count': len(tweets),
            'tweets': tweets,
            'saved_tweet_ids': saved_tweet_ids
        }

        return result

    def save_results(self, deal_id: str, all_results: List[Dict[str, Any]], daily_mode: bool = False) -> str:
        """
        Save search results to JSON file

        Args:
            deal_id: Deal ID
            all_results: List of search results for all competitive pairs
            daily_mode: If True, this is a daily update

        Returns:
            Path to saved file
        """
        mode_suffix = "_daily" if daily_mode else "_initial"
        filename = f"twitter_search_rf3_{deal_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{mode_suffix}.json"
        filepath = os.path.join(self.output_dir, filename)

        output_data = {
            'deal_id': deal_id,
            'search_timestamp': datetime.now().isoformat(),
            'approach': 'RF3',
            'daily_mode': daily_mode,
            'total_competitive_pairs': len(all_results),
            'results': all_results
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results saved to: {filepath}")
        return filepath

    def analyze_deal(self, deal_id: str, force_reprocess: bool = False, daily_mode: bool = False) -> Optional[str]:
        """
        Main method to analyze a deal using competitive product queries

        Args:
            deal_id: Deal ID to analyze
            force_reprocess: If True, force reprocessing even if RF3_approach_done is True
            daily_mode: If True, search only last 24 hours (for daily cron job)

        Returns:
            Path to results file or None if failed
        """
        mode_text = "daily RF3" if daily_mode else "RF3"
        self.logger.info(
            f"Starting {mode_text} Twitter analysis for deal ID: {deal_id}")

        # Step 1: Fetch deal data
        deal_data = self.fetch_deal_data(deal_id)
        if not deal_data:
            return None

        # Get the actual ProcessingJob object to check and update flags
        try:
            processing_job = ProcessingJob.objects.get(id=deal_id)
        except ProcessingJob.DoesNotExist:
            self.logger.error(f"ProcessingJob with ID {deal_id} not found")
            return None

        # Check if RF3 approach is already done (only for initial mode)
        if not daily_mode:
            if hasattr(processing_job, 'RF3_approach_done') and processing_job.RF3_approach_done and not force_reprocess:
                self.logger.info(
                    f"RF3 approach already completed for deal {deal_id}. Skipping processing.")
                return "SKIPPED"

            # Clean up existing data before processing to avoid duplicates (only for initial mode)
            if not hasattr(processing_job, 'RF3_approach_done') or not processing_job.RF3_approach_done or force_reprocess:
                self.logger.info(
                    f"Cleaning up existing RF3 search data for deal {deal_id}")
                self.cleanup_utils.cleanup_existing_search_data(deal_id, "RF3")

        # Step 2: Fetch competitive products
        competitive_pairs = self.fetch_competitive_products(deal_id)

        if not competitive_pairs:
            self.logger.warning(
                f"No competitive products found for deal {deal_id}")
            return None

        self.logger.info(
            f"Found {len(competitive_pairs)} competitive product pairs")

        # Step 3: Search tweets for each competitive pair
        all_results = []
        for i, pair in enumerate(competitive_pairs, 1):
            # Extract product names from the competitive pair
            target_product = pair.get('target_product', '')
            acquire_product = pair.get('acquire_product', '')

            if not target_product or not acquire_product:
                self.logger.warning(
                    f"Skipping pair {i}: missing product names")
                continue

            # Get company names for cleaning product names
            target_company = deal_data.get('target_name', '')
            acquire_company = deal_data.get('acquire_name', '')

            self.logger.info(
                f"Processing competitive pair {i}/{len(competitive_pairs)}: {target_product} vs {acquire_product}")

            try:
                result = self.search_tweets_for_competitive_pair(
                    target_product, acquire_product, target_company, acquire_company, deal_id,
                    deal_data.get('announce_date'), daily_mode
                )
                all_results.append(result)
                self.logger.info(f"Found {result['tweet_count']} tweets")

                # Rate limiting between pairs
                if i < len(competitive_pairs):
                    time.sleep(2)

            except Exception as e:
                self.logger.error(
                    f"Error searching for competitive pair {target_product} vs {acquire_product}: {e}")
                # Continue with next pair
                continue

        # Step 4: Save results
        if all_results:
            filepath = self.save_results(deal_id, all_results, daily_mode)

            # Log summary
            total_tweets = sum(r['tweet_count'] for r in all_results)
            self.logger.info(f"=== {mode_text.upper()} Analysis Complete ===")
            self.logger.info(f"Deal ID: {deal_id}")
            self.logger.info(
                f"Total competitive pairs processed: {len(all_results)}")
            self.logger.info(f"Total tweets found: {total_tweets}")
            self.logger.info(f"Results saved to: {filepath}")

            # Step 5: Mark RF3 approach as completed (only for initial mode)
            if not daily_mode:
                try:
                    processing_job.RF3_approach_done = True
                    processing_job.updatedAt = datetime.utcnow()
                    processing_job.save()
                    self.logger.info(
                        f"Marked RF3 approach as completed for deal {deal_id}")
                except Exception as e:
                    self.logger.error(
                        f"Error updating RF3_approach_done flag: {e}")

            return filepath
        else:
            self.logger.warning("No results to save")
            return None


def main():
    """Main function to run the script"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)

    if len(sys.argv) < 2 or len(sys.argv) > 4:
        logger.error(
            "Usage: python riffle_approach_3.py <deal_id> [--force] [--daily]")
        logger.error(
            "Example: python riffle_approach_3.py 68184d52478abf06ec1a28ec")
        logger.error(
            "Example (force reprocess): python riffle_approach_3.py 68184d52478abf06ec1a28ec --force")
        logger.error(
            "Example (daily mode): python riffle_approach_3.py 68184d52478abf06ec1a28ec --daily")
        sys.exit(1)

    deal_id = sys.argv[1]
    force_reprocess = "--force" in sys.argv
    daily_mode = "--daily" in sys.argv

    try:
        # Initialize analyzer (will use TWITTER_API_KEY environment variable)
        analyzer = CompetitiveProductsDealAnalyzer()

        # Run analysis
        result_file = analyzer.analyze_deal(
            deal_id, force_reprocess=force_reprocess, daily_mode=daily_mode)

        if result_file == "SKIPPED":
            logger.info(
                "Analysis skipped - RF3 approach already completed for this deal")
            logger.info("Use --force flag to reprocess if needed")
        elif result_file:
            mode_text = "Daily RF3" if daily_mode else "RF3"
            logger.info(f"{mode_text} Analysis completed successfully!")
            logger.info(f"Results saved to: {result_file}")
        else:
            logger.error("Analysis failed or no results found")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
