#!/usr/bin/env python3
"""
High Value Followers Tweet Search Processor
Identifies high-value followers by searching for tweets mentioning company or products.

Usage:
# python high_value_followers_tweet_search.py 682f00def21b9fca8e1d04fe
"""

from document_processor.twitter_utils.twitter_cleanup_utils import TwitterCleanupUtils
from document_processor.twitter_utils.follower_utils import FollowerUtils
from document_processor.models import ProcessingJob, Followers, HighValueFollowers, CompanyProducts
import django
import os
import sys
import json
import time
import logging
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pdb

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()

# Load environment variables
load_dotenv()


class HighValueFollowersTweetSearchProcessor:
    """Main class for processing high-value followers using tweet search"""

    def __init__(self, twitter_api_key: Optional[str] = None, config_overrides: Dict[str, Any] = None):
        """
        Initialize the processor

        Args:
            twitter_api_key: Twitter API key for tweet search
            config_overrides: Optional configuration overrides
        """
        # Setup configuration
        self.config = {
            # Follower filtering criteria
            'min_followers_count': 250,
            'min_statuses_count': 250,

            # Processing limits
            'max_followers_per_company': None,  # None = process all followers

            # Tweet search settings
            'search_date_range_years': 5,  # Search tweets from last N years
            'max_tweets_per_user': 1,  # Only need 1 tweet to qualify as high-value
            'search_timeout': 120,  # Timeout for each search request

            # Rate limiting
            'delay_between_requests': 0,  # seconds
            'delay_every_n_requests': 0.2,
            'delay_for_rate_limit': 0,  # seconds

            # Concurrency settings
            'max_workers': 10,  # Number of parallel tweet searches
            'use_parallel_processing': True,

            # Processing approach
            'approach': 'GUNSHOT',

            # Output settings
            'save_results_summary': True,
            'save_tweet_details': True,

            # Batch processing for MongoDB
            'batch_size': 100,
            'use_batch_saving': True
        }

        # Apply any config overrides
        if config_overrides:
            self.config.update(config_overrides)

        # Setup Twitter API key
        self.twitter_api_key = twitter_api_key or os.getenv('TWITTER_API_KEY')
        if not self.twitter_api_key:
            raise ValueError(
                "Twitter API key is required. Set TWITTER_API_KEY environment variable.")

        self.base_url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
        self.headers = {
            'X-API-Key': self.twitter_api_key,
            'Content-Type': 'application/json'
        }

        # Setup utilities
        self.follower_utils = FollowerUtils()
        self.cleanup_utils = TwitterCleanupUtils()

        # Setup logger
        self.logger = logging.getLogger(__name__)

        # Setup output directory
        self.output_dir = os.path.join(
            os.path.dirname(__file__),
            'high_value_followers_tweet_results'
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Thread-safe counters
        self.processed_count = 0
        self.followers_with_tweets = 0
        self.total_tweets_found = 0
        self.lock = threading.Lock()

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

            deal_data = {
                'id': str(deal.id),
                'cik': deal.cik,
                'acquire_name': deal.acquire_name,
                'target_name': deal.target_name,
                'announce_date': deal.announce_date.strftime('%Y-%m-%d') if deal.announce_date else None,
                'schema_results': deal.schema_results,
                'twitter_details': deal.twitter_details
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

    def extract_twitter_handles(self, deal_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract Twitter handles for target and acquire companies

        Args:
            deal_data: Deal data dictionary

        Returns:
            Dictionary with company names as keys and Twitter handles as values
        """
        twitter_handles = {}
        twitter_details = deal_data.get('twitter_details', {})

        if not twitter_details:
            self.logger.warning("No Twitter details found in deal data")
            return twitter_handles

        company_handles = twitter_details.get('company_handles', {})

        # Extract target company handle
        target_name = deal_data.get('target_name')
        if target_name and target_name in company_handles:
            target_handle = company_handles[target_name].get(
                'main_twitter_handle')
            if target_handle:
                target_handle = target_handle.lstrip('@')
                twitter_handles[target_name] = target_handle
                self.logger.info(
                    f"Found target Twitter handle: @{target_handle}")

        # Extract acquire company handle
        acquire_name = deal_data.get('acquire_name')
        if acquire_name and acquire_name in company_handles:
            acquire_handle = company_handles[acquire_name].get(
                'main_twitter_handle')
            if acquire_handle:
                acquire_handle = acquire_handle.lstrip('@')
                twitter_handles[acquire_name] = acquire_handle
                self.logger.info(
                    f"Found acquire Twitter handle: @{acquire_handle}")

        return twitter_handles

    def filter_followers(self, followers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter followers based on criteria (no description requirement)

        Args:
            followers: List of follower objects

        Returns:
            List of filtered followers
        """
        min_followers_count = self.config['min_followers_count']
        min_statuses_count = self.config['min_statuses_count']

        original_count = len(followers)
        filtered_followers = []

        for follower in followers:
            followers_count = follower.get('followers_count', 0)
            statuses_count = follower.get('statuses_count', 0)
            protected = follower.get('protected', False)
            description = follower.get('description', '')
            has_description = description and description.strip()

            if (followers_count >= min_followers_count and
                statuses_count >= min_statuses_count and
                    not protected and not has_description):
                filtered_followers.append(follower)

        self.logger.info(
            f"Filtered {original_count} followers -> {len(filtered_followers)} followers")
        self.logger.info(
            f"Criteria: followers_count >= {min_followers_count} AND statuses_count >= {min_statuses_count} AND not protected")

        # Save filtered followers to filter/filtered_[companyName].json for tracking
        # Attempt to get company name from self.current_company if available
        company_name = getattr(self, 'current_company', None)
        if not company_name:
            company_name = "unknown_company"
        # Sanitize company name for filename
        safe_company_name = "".join(c if c.isalnum() or c in (
            ' ', '_', '-') else '_' for c in company_name).replace(' ', '_')
        filter_dir = os.path.join(os.path.dirname(__file__), 'filter')
        os.makedirs(filter_dir, exist_ok=True)

        # Add timestamp to make filename unique
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[
            :-3]  # Include milliseconds
        filter_path = os.path.join(
            filter_dir, f"filtered_{safe_company_name}_{timestamp}.json")

        # Create JSON structure with metadata
        json_data = {
            'company_name': company_name,
            'filter_timestamp': datetime.now().isoformat(),
            'filter_criteria': {
                'min_followers_count': min_followers_count,
                'min_statuses_count': min_statuses_count
            },
            'original_count': original_count,
            'filtered_count': len(filtered_followers),
            'filtered_followers': filtered_followers
        }

        with open(filter_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Saved filtered followers to {filter_path}")

        return filtered_followers

    def build_search_query(self, username: str, company_name: str, products: List[str], company_handle: str = None) -> str:
        """
        Build Twitter search query for user mentioning company or products

        Args:
            username: Twitter username to search from
            company_name: Company name to search for
            products: List of product names to search for
            company_handle: Company Twitter handle to search for

        Returns:
            Twitter search query string
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = datetime(
            end_date.year - self.config['search_date_range_years'], 1, 1)

        # Format dates for Twitter API
        since_date = start_date.strftime("%Y-%m-%d_00:00:00_UTC")
        until_date = end_date.strftime("%Y-%m-%d_23:59:59_UTC")

        # Build query components
        query_parts = [f"from:{username}"]

        # Add company name
        if company_name:
            query_parts.append(f'"{company_name}"')

        # Add company Twitter handle with @ symbol
        if company_handle:
            # Remove @ symbol if present to avoid duplication
            clean_handle = company_handle.lstrip('@')
            query_parts.append(f'@{clean_handle}')

        # Add top 3 products
        for product in products:
            if product and product.strip():
                query_parts.append(f'"{product.strip()}"')

        self.logger.info(f"products list: {products}")
        # Create OR condition for company, handle, and products
        search_terms = " OR ".join(query_parts[1:]) if len(
            query_parts) > 1 else ""

        if search_terms:
            final_query = f"from:{username} ({search_terms}) since:{since_date} until:{until_date} lang:en"
        else:
            final_query = f"from:{username} since:{since_date} until:{until_date} lang:en"

        self.logger.info(f"Search query: {final_query}")

        return final_query

    def search_tweets_single(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Search for tweets using the advanced search API (returns first tweet found)

        Args:
            query: Search query string

        Returns:
            First tweet found or None if no tweets
        """
        params = {
            'query': query,
            'queryType': 'Latest',
            'cursor': ""
        }

        try:
            response = requests.get(
                self.base_url, headers=self.headers, params=params,
                timeout=self.config['search_timeout'])
            response.raise_for_status()

            data = response.json()
            tweets = data.get('tweets', [])

            if tweets:
                # Return first tweet found
                return tweets[0]

            return None

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching tweets: {e}")
            return None

    def search_follower_tweets(self, follower: Dict[str, Any], company_name: str, products: List[str], company_handle: str = None) -> Optional[Dict[str, Any]]:
        """
        Search for tweets from a specific follower mentioning company or products

        Args:
            follower: Follower data
            company_name: Company name
            products: List of product names
            company_handle: Company Twitter handle

        Returns:
            Dictionary containing search results or None if no tweets
        """
        username = follower.get('screen_name') or follower.get('userName')
        if not username:
            return None

        query = self.build_search_query(
            username, company_name, products, company_handle)
        tweet = self.search_tweets_single(query)

        if tweet:
            return {
                'follower': follower,
                'company_name': company_name,
                'products_searched': products,
                'search_query': query,
                'tweet_found': tweet,
                'search_timestamp': datetime.now().isoformat()
            }

        return None

    def process_single_follower(self, args: tuple) -> Optional[Dict[str, Any]]:
        """
        Process a single follower for tweet search (for thread pool)

        Args:
            args: Tuple containing (follower, company_name, products, company_handle, index, total)

        Returns:
            Dictionary containing search results or None if no tweets
        """
        follower, company_name, products, company_handle, index, total = args
        username = follower.get('screen_name') or follower.get('userName')

        self.logger.info(f"args tuple: {args}")

        if not username:
            return None

        try:
            self.logger.info(
                f"Searching tweets for {index}/{total}: @{username}")

            result = self.search_follower_tweets(
                follower, company_name, products, company_handle)

            # Update counters thread-safely
            with self.lock:
                self.processed_count += 1
                if result:
                    self.followers_with_tweets += 1
                    self.total_tweets_found += 1
                    self.logger.info(f"✓ Found tweet for @{username}")
                else:
                    self.logger.info(f"✗ No tweets found for @{username}")

            return result

        except Exception as e:
            self.logger.error(f"Error processing follower @{username}: {e}")
            with self.lock:
                self.processed_count += 1
            return None

    def save_high_value_follower_tweet(self, result: Dict[str, Any], deal_id: str, company_handle: str) -> Optional[str]:
        """
        Save a high-value follower (with tweet) to MongoDB

        Args:
            result: Tweet search result with follower data
            deal_id: Deal ID
            company_handle: Company Twitter handle

        Returns:
            ID of saved record or None if failed
        """
        try:
            follower = result['follower']
            tweet = result['tweet_found']

            # Create HighValueFollowers object
            high_value_follower = HighValueFollowers(
                deal_id=deal_id,
                company_name=result['company_name'],
                company_handle=company_handle,

                # Follower information
                follower_id=str(follower.get('id', '')),
                name=follower.get('name', ''),
                screen_name=follower.get('screen_name', ''),
                description=follower.get('description', ''),
                location=follower.get('location', ''),
                followers_count=follower.get('followers_count', 0),
                statuses_count=follower.get('statuses_count', 0),
                protected=follower.get('protected', False),
                verified=follower.get('verified', False),
                created_at_twitter=follower.get('created_at'),

                # Tweet-based analysis results
                overall_score=10,  # All followers with tweets get max score
                reason=f"Found tweet mentioning {result['company_name']} or products",
                key_indicators=[
                    f"Tweet ID: {tweet.get('id', 'unknown')}",
                    f"Tweet text: {tweet.get('text', '')[:100]}...",
                    f"Products searched: {', '.join(result['products_searched'])}"
                ],
                analysis_timestamp=datetime.now(),
                gpt_model_used="tweet_search",  # No GPT used

                # Processing metadata
                processing_status='completed',
                approach=self.config['approach']
            )

            high_value_follower.save()

            self.logger.info(
                f"Saved high-value follower: @{follower.get('screen_name')} (Tweet-based)")
            return str(high_value_follower.id)

        except Exception as e:
            self.logger.error(f"Error saving high-value follower: {e}")
            return None

    def save_high_value_followers_batch(self, results: List[Dict[str, Any]], deal_id: str, company_handle: str) -> List[str]:
        """
        Save multiple high-value followers to MongoDB in a batch

        Args:
            results: List of tweet search results
            deal_id: Deal ID
            company_handle: Company Twitter handle

        Returns:
            List of saved record IDs
        """
        try:
            high_value_followers = []

            for result in results:
                follower = result['follower']
                tweet = result['tweet_found']

                high_value_follower = HighValueFollowers(
                    deal_id=deal_id,
                    company_name=result['company_name'],
                    company_handle=company_handle,

                    # Follower information
                    follower_id=str(follower.get('id', '')),
                    name=follower.get('name', ''),
                    screen_name=follower.get('screen_name', ''),
                    description=follower.get('description', ''),
                    location=follower.get('location', ''),
                    followers_count=follower.get('followers_count', 0),
                    statuses_count=follower.get('statuses_count', 0),
                    protected=follower.get('protected', False),
                    verified=follower.get('verified', False),
                    created_at_twitter=follower.get('created_at'),

                    # Tweet-based analysis results
                    overall_score=10,  # All followers with tweets get max score
                    reason=f"Found tweet mentioning {result['company_name']} or products",
                    key_indicators=[
                        f"Tweet ID: {tweet.get('id', 'unknown')}",
                        f"Tweet text: {tweet.get('text', '')[:100]}...",
                        f"Products searched: {', '.join(result['products_searched'])}"
                    ],
                    analysis_timestamp=datetime.now(),
                    gpt_model_used="tweet_search",  # No GPT used

                    # Processing metadata
                    processing_status='completed',
                    approach=self.config['approach']
                )
                high_value_followers.append(high_value_follower)

            # Bulk insert all followers
            HighValueFollowers.objects.insert(high_value_followers)

            saved_ids = [str(follower.id) for follower in high_value_followers]

            self.logger.info(
                f"Batch saved {len(saved_ids)} high-value followers")
            return saved_ids

        except Exception as e:
            self.logger.error(f"Error batch saving high-value followers: {e}")
            return []

    def save_tweet_details_json(self, results: List[Dict[str, Any]], deal_id: str, company_name: str) -> str:
        """
        Save tweet details to JSON file for analysis

        Args:
            results: List of tweet search results
            deal_id: Deal ID
            company_name: Company name

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_company_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_'
                                    for c in company_name).replace(' ', '_')

        filename = f"tweet_details_{safe_company_name}_{deal_id}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Prepare data for JSON
        json_data = {
            'deal_id': deal_id,
            'company_name': company_name,
            'search_timestamp': datetime.now().isoformat(),
            'total_followers_with_tweets': len(results),
            'tweet_details': []
        }

        for result in results:
            follower = result['follower']
            tweet = result['tweet_found']

            tweet_detail = {
                'follower_info': {
                    'screen_name': follower.get('screen_name'),
                    'name': follower.get('name'),
                    'followers_count': follower.get('followers_count'),
                    'statuses_count': follower.get('statuses_count')
                },
                'search_query': result['search_query'],
                'products_searched': result['products_searched'],
                'tweet_found': {
                    'id': tweet.get('id'),
                    'text': tweet.get('text'),
                    'created_at': tweet.get('created_at'),
                    'retweet_count': tweet.get('retweet_count'),
                    'favorite_count': tweet.get('favorite_count')
                }
            }
            json_data['tweet_details'].append(tweet_detail)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Tweet details saved to: {filepath}")
        return filepath

    def process_company_followers(self, deal_id: str, company_name: str, company_handle: str, products: List[str]) -> Dict[str, Any]:
        """
        Process followers for a specific company using tweet search

        Args:
            deal_id: Deal ID
            company_name: Company name
            company_handle: Company Twitter handle
            products: List of products to search for

        Returns:
            Dictionary with processing results
        """
        self.logger.info(
            f"Processing followers for {company_name} (@{company_handle})")

        # Step 1: Get followers from database
        followers = self.follower_utils.get_followers_for_company(
            deal_id, company_handle, self.config['approach'])

        self.logger.info(
            f"Found {len(followers)} all followers")

        if not followers:
            self.logger.warning(f"No followers found for @{company_handle}")
            return {
                'company_name': company_name,
                'company_handle': company_handle,
                'total_followers': 0,
                'filtered_followers': 0,
                'followers_with_tweets': 0,
                'processing_status': 'failed'
            }

        self.logger.info(
            f"Retrieved {len(followers)} followers for @{company_handle}")

        # Step 2: Filter followers
        filtered_followers = self.filter_followers(followers)

        self.logger.info(
            f"Filtered {len(followers)} followers -> {len(filtered_followers)} followers")
        # pdb.set_trace()

        # Step 3: Apply follower limit if configured
        max_followers = self.config['max_followers_per_company']
        if max_followers is not None and max_followers > 0:
            original_count = len(filtered_followers)
            filtered_followers = filtered_followers[:max_followers]
            self.logger.info(
                f"Limited to {max_followers} followers for processing (from {original_count} filtered followers)")

        # Step 4: Search tweets for followers
        self.processed_count = 0
        self.followers_with_tweets = 0
        self.total_tweets_found = 0

        tweet_results = []

        if self.config['use_parallel_processing'] and self.config['max_workers'] > 1:
            self.logger.info(
                f"Using parallel processing with {self.config['max_workers']} workers")

            # Prepare arguments for parallel processing
            args_list = [(follower, company_name, products, company_handle, i+1, len(filtered_followers))
                         for i, follower in enumerate(filtered_followers)]

            with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
                future_to_args = {executor.submit(self.process_single_follower, args): args
                                  for args in args_list}

                for future in as_completed(future_to_args):
                    try:
                        result = future.result()
                        if result:
                            tweet_results.append(result)

                        # Rate limiting
                        if self.processed_count % self.config['delay_every_n_requests'] == 0:
                            time.sleep(self.config['delay_for_rate_limit'])
                        else:
                            time.sleep(self.config['delay_between_requests'])

                    except Exception as e:
                        self.logger.error(
                            f"Exception in parallel processing: {e}")
        else:
            # Sequential processing
            self.logger.info("Using sequential processing")
            for i, follower in enumerate(filtered_followers, 1):
                result = self.process_single_follower(
                    (follower, company_name, products, company_handle, i, len(filtered_followers)))
                if result:
                    tweet_results.append(result)

                # Rate limiting
                if i % self.config['delay_every_n_requests'] == 0:
                    time.sleep(self.config['delay_for_rate_limit'])
                else:
                    time.sleep(self.config['delay_between_requests'])

        # Step 5: Save high-value followers
        saved_ids = []
        if tweet_results:
            if self.config.get('use_batch_saving', False):
                # Batch save
                batch_size = self.config.get('batch_size', 10)
                for i in range(0, len(tweet_results), batch_size):
                    batch = tweet_results[i:i + batch_size]
                    batch_saved_ids = self.save_high_value_followers_batch(
                        batch, deal_id, company_handle)
                    saved_ids.extend(batch_saved_ids)
            else:
                # Individual save
                for result in tweet_results:
                    saved_id = self.save_high_value_follower_tweet(
                        result, deal_id, company_handle)
                    if saved_id:
                        saved_ids.append(saved_id)

            # Save tweet details JSON
            if self.config.get('save_tweet_details', True):
                self.save_tweet_details_json(
                    tweet_results, deal_id, company_name)

        return {
            'company_name': company_name,
            'company_handle': company_handle,
            'total_followers': len(followers),
            'filtered_followers': len(filtered_followers),
            'followers_with_tweets': len(tweet_results),
            'saved_high_value_followers': len(saved_ids),
            'processing_status': 'completed'
        }

    def process_deal(self, deal_id: str) -> Optional[str]:
        """
        Main method to process high-value followers for a deal using tweet search

        Args:
            deal_id: Deal ID to process

        Returns:
            Path to results file or None if failed
        """
        self.logger.info(
            f"Starting Tweet Search processing for deal ID: {deal_id}")

        # Step 1: Fetch deal data
        deal_data = self.fetch_deal_data(deal_id)
        if not deal_data:
            return None

        # Step 2: Extract Twitter handles
        twitter_handles = self.extract_twitter_handles(deal_data)

        if len(twitter_handles) < 2:
            self.logger.warning(
                f"Not enough Twitter handles found for deal {deal_id}. Found: {twitter_handles}")
            return None

        # Step 3: Fetch company products from database
        products_data = self.fetch_company_products(deal_id)
        self.logger.info(
            f"Found products for companies: {list(products_data.keys())}")

        # Step 4: Process followers for each company
        all_results = []

        target_name = deal_data.get('target_name')
        acquire_name = deal_data.get('acquire_name')

        target_handle = twitter_handles.get(target_name)
        acquire_handle = twitter_handles.get(acquire_name)

        if not target_handle or not acquire_handle:
            self.logger.error(
                "Missing Twitter handles for target or acquire company")
            return None

        # Process target company followers
        try:
            target_products = products_data.get(target_name, [])
            target_result = self.process_company_followers(
                deal_id, target_name, target_handle, target_products
            )
            all_results.append(target_result)
            self.logger.info(
                f"Completed processing for target company @{target_handle}")

        except Exception as e:
            self.logger.error(
                f"Error processing target company @{target_handle}: {e}")

        # Process acquire company followers
        try:
            acquire_products = products_data.get(acquire_name, [])
            acquire_result = self.process_company_followers(
                deal_id, acquire_name, acquire_handle, acquire_products
            )
            all_results.append(acquire_result)
            self.logger.info(
                f"Completed processing for acquire company @{acquire_handle}")

        except Exception as e:
            self.logger.error(
                f"Error processing acquire company @{acquire_handle}: {e}")

        # Step 5: Save results summary
        if all_results:
            summary_filepath = self.save_results_summary(deal_id, all_results)

            # Log summary
            total_high_value = sum(r.get('followers_with_tweets', 0)
                                   for r in all_results)
            self.logger.info("=== Tweet Search Processing Complete ===")
            self.logger.info(f"Deal ID: {deal_id}")
            self.logger.info(f"Total companies processed: {len(all_results)}")
            self.logger.info(
                f"Total high-value followers found: {total_high_value}")
            self.logger.info(f"Summary saved to: {summary_filepath}")

            return summary_filepath
        else:
            self.logger.warning("No results to save")
            return None

    def save_results_summary(self, deal_id: str, all_results: List[Dict[str, Any]]) -> str:
        """
        Save processing results summary to JSON file

        Args:
            deal_id: Deal ID
            all_results: List of processing results for all companies

        Returns:
            Path to saved file
        """
        filename = f"tweet_search_summary_{deal_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Calculate totals
        total_followers = sum(r.get('total_followers', 0) for r in all_results)
        total_filtered = sum(r.get('filtered_followers', 0)
                             for r in all_results)
        total_with_tweets = sum(r.get('followers_with_tweets', 0)
                                for r in all_results)

        output_data = {
            'deal_id': deal_id,
            'processing_timestamp': datetime.now().isoformat(),
            'approach': 'TWEET_SEARCH',
            'total_companies': len(all_results),
            'summary': {
                'total_followers': total_followers,
                'total_filtered_followers': total_filtered,
                'total_followers_with_tweets': total_with_tweets
            },
            'company_results': all_results,
            'config_used': self.config
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results summary saved to: {filepath}")
        return filepath


def main():
    """Main function to run the script"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)

    if len(sys.argv) < 2:
        logger.error(
            "Usage: python high_value_followers_tweet_search.py <deal_id> [options]")
        logger.error("Options:")
        logger.error(
            "  --workers <number>           Number of parallel workers (default: 10)")
        logger.error(
            "  --batch-size <number>        Batch size for MongoDB saves (default: 100)")
        logger.error(
            "  --max-followers <number>     Maximum followers per company (default: 200)")
        logger.error("Examples:")
        logger.error(
            "  python high_value_followers_tweet_search.py 68184d52478abf06ec1a28ec")
        logger.error(
            "  python high_value_followers_tweet_search.py 68184d52478abf06ec1a28ec --workers 20")
        sys.exit(1)

    deal_id = sys.argv[1]

    # Parse optional arguments
    config_overrides = {}

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--workers' and i + 1 < len(sys.argv):
            workers = int(sys.argv[i + 1])
            if workers < 1:
                logger.error("❌ --workers must be at least 1")
                sys.exit(1)
            config_overrides['max_workers'] = workers
            config_overrides['use_parallel_processing'] = workers > 1
            i += 2
        elif sys.argv[i] == '--batch-size' and i + 1 < len(sys.argv):
            batch_size = int(sys.argv[i + 1])
            if batch_size < 1:
                logger.error("❌ --batch-size must be at least 1")
                sys.exit(1)
            config_overrides['batch_size'] = batch_size
            config_overrides['use_batch_saving'] = True
            i += 2
        elif sys.argv[i] == '--max-followers' and i + 1 < len(sys.argv):
            max_followers = int(sys.argv[i + 1])
            if max_followers <= 0:
                logger.error("❌ --max-followers must be greater than 0")
                sys.exit(1)
            config_overrides['max_followers_per_company'] = max_followers
            i += 2
        else:
            i += 1

    try:
        # Initialize processor with config overrides
        processor = HighValueFollowersTweetSearchProcessor(
            config_overrides=config_overrides)

        # Run processing
        result_file = processor.process_deal(deal_id)

        if result_file:
            logger.info("Tweet Search processing completed successfully!")
            logger.info(f"Results saved to: {result_file}")
        else:
            logger.error("Processing failed or no results found")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
