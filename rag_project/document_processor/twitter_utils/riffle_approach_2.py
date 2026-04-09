#!/usr/bin/env python3
"""
Twitter Search Script for Deal Analysis - Approach 2
This script fetches Twitter data using risk-based queries for M&A deals.

Usage:
# python run_twitter_search.py 682f00def21b9fca8e1d04fe --approach=2
"""

from .twitter_cleanup_utils import TwitterCleanupUtils
from document_processor.models import ProcessingJob, SearchQuery, Tweet, CompanyProducts
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

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()


class RiskBasedTwitterSearchService:
    """Service for searching Twitter using risk-based queries"""

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

        # Load risk words
        self.risk_words = self.load_risk_words()

    def load_risk_words(self) -> Dict[str, Any]:
        """Load risk words from JSON file"""
        risk_words_path = os.path.join(
            os.path.dirname(__file__), 'riskwords.json')
        try:
            with open(risk_words_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading risk words: {e}")
            return {}

    def search_tweets(self, query: str, max_results: int = 5000) -> List[Dict[str, Any]]:
        """
        Search for tweets using the advanced search API (last 24 hours only)

        Args:
            query: Search query string
            max_results: Maximum number of results to fetch

        Returns:
            List of tweet data from the last 24 hours
        """
        all_tweets = []
        seen_tweet_ids = set()
        cursor = ""
        max_iterations = 300
        iteration_count = 0

        # Add 24-hour time filter to the query string
        query_with_time = f"{query} within_time:24h"

        self.logger.info(f"Searching tweets with query: {query_with_time}")

        while len(all_tweets) < max_results and iteration_count < max_iterations:
            iteration_count += 1
            params = {
                'query': query_with_time,
                'queryType': 'Latest',
                'cursor': cursor
            }

            try:
                response = requests.get(
                    self.base_url, headers=self.headers, params=params)
                response.raise_for_status()

                data = response.json()
                tweets = data.get('tweets', [])

                if not tweets:
                    break

                new_count = 0
                for tweet in tweets:
                    tweet_id = tweet.get('id', '')
                    if tweet_id and tweet_id not in seen_tweet_ids:
                        seen_tweet_ids.add(tweet_id)
                        all_tweets.append(tweet)
                        new_count += 1

                self.logger.info(
                    f"Page {iteration_count}: {len(tweets)} fetched, {new_count} new, {len(all_tweets)} total unique")

                if new_count == 0:
                    self.logger.warning(
                        "No new tweets in this page, stopping pagination")
                    break
                self.logger.info(
                    f"has_next_page: {data.get('has_next_page', False)}")
                self.logger.info(f"cursor: {cursor}")
                if not data.get('has_next_page', False):
                    break

                next_cursor = data.get('next_cursor', '')
                self.logger.info(f"next_cursor: {next_cursor}")
                if not next_cursor or next_cursor == cursor:
                    if next_cursor == cursor:
                        self.logger.warning(
                            f"Cursor unchanged, stopping pagination at {len(all_tweets)} tweets")
                    break

                cursor = next_cursor
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching tweets: {e}")
                break

        if iteration_count >= max_iterations:
            self.logger.warning(
                f"Max iterations ({max_iterations}) reached at {len(all_tweets)} tweets")

        return all_tweets[:max_results]


class RiskBasedDealAnalyzer:
    """Main class for analyzing Twitter data using risk-based queries"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the analyzer

        Args:
            api_key: Twitter API key
        """
        self.twitter_service = RiskBasedTwitterSearchService(api_key)
        self.output_dir = os.path.join(
            os.path.dirname(__file__),
            'twitter_search_results'
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger(__name__)

        # Initialize cleanup utilities
        self.cleanup_utils = TwitterCleanupUtils()

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

    def fetch_company_products(self, deal_id: str) -> Dict[str, List[str]]:
        """
        Fetch company products from database

        Args:
            deal_id: Deal ID

        Returns:
            Dictionary with company names as keys and product lists as values
        """
        products_data = {}

        try:
            # Get all company products for this deal
            company_products = CompanyProducts.objects(deal_id=deal_id)

            for cp in company_products:
                company_name = cp.company
                products = cp.products

                # Extract product names from structured data
                product_names = []
                if isinstance(products, list):
                    for category in products:
                        if isinstance(category, dict) and 'products' in category:
                            for product in category['products']:
                                if isinstance(product, dict) and 'name' in product:
                                    product_names.append(product['name'])
                                elif isinstance(product, str):
                                    product_names.append(product)
                        elif isinstance(category, str):
                            product_names.append(category)

                products_data[company_name] = product_names

        except Exception as e:
            self.logger.error(f"Error fetching company products: {e}")

        return products_data

    def build_query_1(self, company_names: List[str], risk_category: str, risk_words: List[str]) -> str:
        """
        Build Query 1: Company names + risk words from Q-1

        Args:
            company_names: List of company names
            risk_category: Risk category name
            risk_words: List of risk words

        Returns:
            Twitter search query string
        """
        # Build company names part
        company_query = ' OR '.join(
            [f'"{company}"' for company in company_names])

        # Build risk words part
        risk_query = ' OR '.join([f'"{word}"' for word in risk_words])

        # Combine with parentheses
        query = f'({company_query}) ({risk_query})'

        return query

    def build_query_2(self, company_names: List[str], risk_category: str, risk_words: List[str]) -> str:
        """
        Build Query 2: Merger/Acquisition + Company names + concern words from Q-2

        Args:
            company_names: List of company names
            risk_category: Risk category name
            risk_words: List of risk words

        Returns:
            Twitter search query string
        """
        # Fixed merger/acquisition terms
        merger_terms = 'merger OR acquisition'

        # Build company names part
        company_query = ' OR '.join(
            [f'"{company}"' for company in company_names])

        # Build concern words part
        concern_query = ' OR '.join([f'"{word}"' for word in risk_words])

        # Combine with brackets and parentheses
        query = f'[{merger_terms} ({company_query}) ({concern_query})]'

        return query

    def build_query_3(self, product_names: List[str], risk_category: str, risk_words: List[str]) -> str:
        """
        Build Query 3: Product names + risk words from Q-3

        Args:
            product_names: List of product names
            risk_category: Risk category name
            risk_words: List of risk words

        Returns:
            Twitter search query string
        """
        # Build product names part
        product_query = ' OR '.join(
            [f'"{product}"' for product in product_names])

        # Build risk words part
        risk_query = ' OR '.join([f'"{word}"' for word in risk_words])

        # Combine with parentheses
        query = f'({product_query}) ({risk_query})'

        return query

    def search_tweets_for_query(self, query: str, query_type: str, category: str, deal_id: str) -> Dict[str, Any]:
        """
        Search tweets for a specific query and save to database

        Args:
            query: Twitter search query
            query_type: Type of query (Q1, Q2, Q3)
            category: Risk category name
            deal_id: Deal ID for database reference

        Returns:
            Dictionary containing search results and metadata
        """
        self.logger.info(f"Searching tweets for {query_type} - {category}")
        self.logger.debug(f"Query: {query}")

        tweets = self.twitter_service.search_tweets(query, max_results=5000)

        # Save search query to database
        combination_data = {
            'query_type': query_type,
            'category': category,
            'query': query
        }

        search_query_obj = SearchQuery(
            search_query=query,
            deal_id=deal_id,
            approach="RF2",
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

        result = {
            'query_type': query_type,
            'category': category,
            'query': query,
            'search_query_id': str(search_query_obj.id),
            'search_timestamp': datetime.now().isoformat(),
            'tweet_count': len(tweets),
            'tweets': tweets,
            'saved_tweet_ids': saved_tweet_ids
        }

        return result

    def save_results(self, deal_id: str, all_results: List[Dict[str, Any]]) -> str:
        """
        Save search results to JSON file

        Args:
            deal_id: Deal ID
            all_results: List of search results for all queries

        Returns:
            Path to saved file
        """
        filename = f"twitter_search_rf2_{deal_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)

        output_data = {
            'deal_id': deal_id,
            'search_timestamp': datetime.now().isoformat(),
            'approach': 'RF2',
            'total_queries': len(all_results),
            'results': all_results
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results saved to: {filepath}")
        return filepath

    def analyze_deal(self, deal_id: str, force_reprocess: bool = True) -> Optional[str]:
        """
        Main method to analyze a deal using risk-based queries

        Args:
            deal_id: Deal ID to analyze
            force_reprocess: If True, force reprocessing even if RF2_approach_done is True

        Returns:
            Path to results file or None if failed
        """
        self.logger.info(
            f"Starting RF2 Twitter analysis for deal ID: {deal_id}")

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

        # Check if RF2 approach is already done
        if hasattr(processing_job, 'RF2_approach_done') and processing_job.RF2_approach_done and not force_reprocess:
            self.logger.info(
                f"RF2 approach already completed for deal {deal_id}. Skipping processing.")
            return "SKIPPED"

        # Clean up existing data before processing to avoid duplicates
        if not hasattr(processing_job, 'RF2_approach_done') or not processing_job.RF2_approach_done or force_reprocess:
            self.logger.info(
                f"Cleaning up existing RF2 search data for deal {deal_id}")
            self.cleanup_utils.cleanup_existing_search_data(deal_id, "RF2")

        # Step 2: Extract company names
        company_names = []
        if deal_data.get('acquire_name'):
            company_names.append(deal_data['acquire_name'])
        if deal_data.get('target_name'):
            company_names.append(deal_data['target_name'])

        self.logger.info(
            f"Found {len(company_names)} companies: {company_names}")

        if len(company_names) < 1:
            self.logger.warning("No companies found for analysis")
            return None

        # Step 3: Fetch company products
        company_products = self.fetch_company_products(deal_id)
        self.logger.info(
            f"Found products for {len(company_products)} companies")

        # Step 4: Execute Query 1 (Q-1 categories)
        all_results = []

        # Query 1: Company names + Q-1 risk words
        if 'Q-1' in self.twitter_service.risk_words:
            for category, risk_words in self.twitter_service.risk_words['Q-1'].items():
                query = self.build_query_1(company_names, category, risk_words)

                try:
                    result = self.search_tweets_for_query(
                        query, "Q1", category, deal_id
                    )
                    all_results.append(result)
                    self.logger.info(
                        f"Q1-{category}: Found {result['tweet_count']} tweets")

                    # Rate limiting between queries
                    time.sleep(2)

                except Exception as e:
                    self.logger.error(f"Error in Q1-{category}: {e}")
                    continue

        # Query 2: Merger/Acquisition + Company names + Q-2 concern words
        if 'Q-2' in self.twitter_service.risk_words:
            for category, risk_words in self.twitter_service.risk_words['Q-2'].items():
                query = self.build_query_2(company_names, category, risk_words)

                try:
                    result = self.search_tweets_for_query(
                        query, "Q2", category, deal_id
                    )
                    all_results.append(result)
                    self.logger.info(
                        f"Q2-{category}: Found {result['tweet_count']} tweets")

                    # Rate limiting between queries
                    time.sleep(2)

                except Exception as e:
                    self.logger.error(f"Error in Q2-{category}: {e}")
                    continue

        # Query 3: Product names + Q-3 risk words (for each company)
        if 'Q-3' in self.twitter_service.risk_words:
            for company_name, product_names in company_products.items():
                if product_names:  # Only if company has products
                    for category, risk_words in self.twitter_service.risk_words['Q-3'].items():
                        query = self.build_query_3(
                            product_names, category, risk_words)

                        try:
                            result = self.search_tweets_for_query(
                                query, "Q3", f"{category}_{company_name}", deal_id
                            )
                            all_results.append(result)
                            self.logger.info(
                                f"Q3-{category}-{company_name}: Found {result['tweet_count']} tweets")

                            # Rate limiting between queries
                            time.sleep(2)

                        except Exception as e:
                            self.logger.error(
                                f"Error in Q3-{category}-{company_name}: {e}")
                            continue

        # Step 5: Save results
        if all_results:
            filepath = self.save_results(deal_id, all_results)

            # Log summary
            total_tweets = sum(r['tweet_count'] for r in all_results)
            self.logger.info("=== RF2 Analysis Complete ===")
            self.logger.info(f"Deal ID: {deal_id}")
            self.logger.info(f"Total queries processed: {len(all_results)}")
            self.logger.info(f"Total tweets found: {total_tweets}")
            self.logger.info(f"Results saved to: {filepath}")

            # Step 6: Mark RF2 approach as completed
            try:
                processing_job.RF2_approach_done = True
                processing_job.updatedAt = datetime.utcnow()
                processing_job.save()
                self.logger.info(
                    f"Marked RF2 approach as completed for deal {deal_id}")
            except Exception as e:
                self.logger.error(
                    f"Error updating RF2_approach_done flag: {e}")

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

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        logger.error("Usage: python riffle_approach_2.py <deal_id> [--force]")
        logger.error(
            "Example: python riffle_approach_2.py 68184d52478abf06ec1a28ec")
        logger.error(
            "Example (force reprocess): python riffle_approach_2.py 68184d52478abf06ec1a28ec --force")
        sys.exit(1)

    deal_id = sys.argv[1]
    force_reprocess = len(sys.argv) == 3 and sys.argv[2] == "--force"

    try:
        # Initialize analyzer (will use TWITTER_API_KEY environment variable)
        analyzer = RiskBasedDealAnalyzer()

        # Run analysis
        result_file = analyzer.analyze_deal(
            deal_id, force_reprocess=force_reprocess)

        if result_file == "SKIPPED":
            logger.info(
                "Analysis skipped - RF2 approach already completed for this deal")
            logger.info("Use --force flag to reprocess if needed")
        elif result_file:
            logger.info("RF2 Analysis completed successfully!")
            logger.info(f"Results saved to: {result_file}")
        else:
            logger.error("Analysis failed or no results found")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
