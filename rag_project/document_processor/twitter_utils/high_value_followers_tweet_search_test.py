#!/usr/bin/env python3
"""
High Value Followers Tweet Search Processor - TESTING VERSION
Identifies high-value followers by searching for tweets mentioning company or products.
This is a testing version that only saves data to JSON files, no MongoDB operations.

Usage:
# python high_value_followers_tweet_search_test.py 682f00def21b9fca8e1d04fe
"""

from document_processor.twitter_utils.twitter_cleanup_utils import TwitterCleanupUtils
from document_processor.twitter_utils.follower_utils import FollowerUtils
from document_processor.models import ProcessingJob, Followers, CompanyProducts
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

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()

# Load environment variables
load_dotenv()


class HighValueFollowersTweetSearchProcessorTest:
    """Main class for processing high-value followers using tweet search - TESTING VERSION"""

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
            'search_timeout': 180,  # Timeout for each search request

            # Rate limiting
            'delay_between_requests': 0,  # seconds
            'delay_every_n_requests': 0,  # Apply delay every N requests
            'delay_for_rate_limit': 0.001,  # seconds

            # Concurrency settings
            'max_workers': 15,  # Number of parallel tweet searches
            'use_parallel_processing': True,

            # Processing approach
            'approach': 'GUNSHOT',

            # GPT integration settings
            'use_gpt_for_query_refinement': True,  # Enable GPT-powered query refinement
            # Only use GPT if confidence < threshold (0.0-1.0)
            'gpt_refinement_threshold': 0.7,
            'gpt_model': 'gpt-4.1',  # GPT model to use for query refinement
            'gpt_temperature': 0.1,  # Temperature for GPT generation
            'gpt_max_tokens': 2000,  # Max tokens for GPT response

            # Output settings
            'save_results_summary': True,
            'save_tweet_details': True,
            'save_aggregated_results': True,  # Enable both approaches - JSONL + JSON

            # Batch processing
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
            'high_value_followers_tweet_results_test'
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Thread-safe counters
        self.processed_count = 0
        self.followers_with_tweets = 0
        self.total_tweets_found = 0
        self.lock = threading.Lock()

        # JSONL file for debugging API calls
        self.jsonl_file = None
        self.jsonl_lock = threading.Lock()

        # JSONL file for tweet responses (real-time saving)
        self.tweet_responses_file = None
        self.tweet_responses_lock = threading.Lock()

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

    def build_refined_search_query_with_gpt(self, company_name: str, products: List[str], company_handle: str = None) -> str:
        """
        Build a refined Twitter search query by first creating a base query, then asking GPT to refine it if needed.
        This query is built without username and will be used as a template.

        Args:
            company_name: Company name to search for
            products: List of product names to search for
            company_handle: Company Twitter handle to search for

        Returns:
            Refined Twitter search query string (without username)
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = datetime(
            end_date.year - self.config['search_date_range_years'], 1, 1)

        # Format dates for Twitter API
        since_date = start_date.strftime("%Y-%m-%d_00:00:00_UTC")
        until_date = end_date.strftime("%Y-%m-%d_23:59:59_UTC")

        # First, build our base query
        query_parts = []

        # Add company name in quotes
        if company_name:
            query_parts.append(f'"{company_name}"')

        # Add company Twitter handle with @ symbol
        if company_handle:
            clean_handle = company_handle.lstrip('@')
            query_parts.append(f'"{clean_handle}"')
            query_parts.append(f'"@{clean_handle}"')

        # Add products in quotes
        for product in products:
            if product and product.strip():
                query_parts.append(f'"{product.strip()}"')

        # Create OR condition for company, handle, and products
        search_terms = " OR ".join(query_parts) if query_parts else ""

        if search_terms:
            base_query = f"({search_terms}) since:{since_date} until:{until_date} lang:en"
        else:
            base_query = f"since:{since_date} until:{until_date} lang:en"

        self.logger.info(f"Base query template: {base_query}")

        # Check if GPT integration is enabled
        if not self.config.get('use_gpt_for_query_refinement', True):
            self.logger.info(
                "GPT integration disabled, using base query structure")
            return base_query

        try:
            # Import OpenAI client
            import openai

            # Check if OpenAI API key is available
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                self.logger.warning(
                    "OPENAI_API_KEY not found, using base query structure")
                return base_query

            # Initialize OpenAI client
            openai_client = openai.OpenAI(api_key=openai_api_key)

            # Create GPT prompt asking to refine our existing query
            gpt_prompt = f"""
            I have a Twitter search query for TwitterAPI.io that I want you to refine and optimize.

            Current query: {base_query}

            Context:
            - Company Name: {company_name}
            - Company Twitter Handle: {company_handle if company_handle else 'N/A'}
            - Products: {', '.join(products) if products else 'N/A'}

            Instructions:
            1. Simplify or generalize overly long product names into shorter terms people actually use in tweets.
            2. Remove uncommon elements such as nested parentheses or extra punctuation that may break search.
            3. Add relevant variations, abbreviations, or hashtags if they are commonly used (#AI, #cloud, #5G, etc.).
            4. Avoid redundant or duplicate terms.
            5. Dont remove "". It is important for the search.
            6. Ensure the query stays within Twitter search length limits.
            7. Preserve the original date range, `from:` filter, `until:` filter, and `lang:` filter.
            8. Return **only the refined query string** (no explanations, no formatting, no quotes).

            If the query is already optimal, return it unchanged.
            """

            # Call GPT API for query refinement
            response = openai_client.chat.completions.create(
                model=self.config.get('gpt_model', 'gpt-4'),
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Twitter search query expert. Refine and optimize existing Twitter search queries for TwitterAPI.io to find relevant tweets more effectively."
                    },
                    {
                        "role": "user",
                        "content": gpt_prompt
                    }
                ],
                temperature=self.config.get('gpt_temperature', 0.3),
                max_tokens=self.config.get('gpt_max_tokens', 200)
            )

            # Extract the refined query from GPT response
            refined_query = response.choices[0].message.content.strip()

            self.logger.info(f"GPT response: {refined_query}")

            # Validate that the query contains essential elements
            if not any(keyword in refined_query.lower() for keyword in ['since:', 'until:', 'lang:']):
                self.logger.warning(
                    "GPT response missing date/language filters, using base query")
                return base_query

            # Check if GPT actually made improvements
            if refined_query.strip() == base_query.strip():
                self.logger.info(
                    "GPT determined the base query was already optimal")
                return base_query

            self.logger.info(
                f"GPT refined the query: {base_query} -> {refined_query}")
            return refined_query

        except Exception as e:
            self.logger.error(f"Error refining query with GPT: {e}")
            self.logger.info("Using base query structure due to GPT error")
            return base_query

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

    def build_search_query_with_template(self, username: str, query_template: str) -> str:
        """
        Build search query by combining username with pre-built query template

        Args:
            username: Twitter username to search from
            query_template: Pre-built query template without username

        Returns:
            Complete Twitter search query string
        """
        if not username:
            return query_template

        # Add username to the beginning of the template
        if query_template.startswith("("):
            # If template starts with parentheses, insert username before it
            final_query = f"from:{username} {query_template}"
        else:
            # Otherwise, just prepend username
            final_query = f"from:{username} {query_template}"

        self.logger.info(f"Final search query with username: {final_query}")
        return final_query

    def filter_followers(self, followers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter followers based on criteria (NO description requirement for testing)

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
            # Removed description filter for testing

            if (followers_count >= min_followers_count and
                statuses_count >= min_statuses_count and
                    not protected):
                filtered_followers.append(follower)

        self.logger.info(
            f"Filtered {original_count} followers -> {len(filtered_followers)} followers")
        self.logger.info(
            f"Criteria: followers_count >= {min_followers_count} AND statuses_count >= {min_statuses_count} AND not protected")

        # Save filtered followers to filter/filtered_[companyName].json for tracking
        company_name = getattr(self, 'current_company', 'unknown_company')
        safe_company_name = "".join(c if c.isalnum() or c in (
            ' ', '_', '-') else '_' for c in company_name).replace(' ', '_')
        filter_dir = os.path.join(os.path.dirname(__file__), 'filter')
        os.makedirs(filter_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        filter_path = os.path.join(
            filter_dir, f"filtered_{safe_company_name}_{timestamp}.json")

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

    def search_tweets_with_cursor(self, query: str, username: str) -> List[Dict[str, Any]]:
        """
        Search for tweets using the advanced search API with cursor pagination

        Args:
            query: Search query string
            username: Twitter username to search from
        Returns:
            List of all tweets found
        """
        all_tweets = []
        cursor = ""
        page_count = 0
        max_pages = 100  # Limit to prevent infinite loops

        while cursor is not None and page_count < max_pages:
            page_count += 1
            self.logger.info(
                f"Fetching page {page_count} for query: {query[:100]}...")

            params = {
                'query': query,
                'queryType': 'Latest',
                'cursor': cursor
            }

            try:
                response = requests.get(
                    self.base_url, headers=self.headers, params=params,
                    timeout=self.config['search_timeout'])
                response.raise_for_status()

                data = response.json()
                self.logger.info(f"response_data1: {username} {data}")
                tweets = data.get('tweets', [])

                # Log API response for debugging
                self.log_api_call(
                    query, params, response.status_code, data, page_count)

                if tweets:
                    all_tweets.extend(tweets)
                    self.logger.info(
                        f"Found {len(tweets)} tweets on page {page_count}")

                # Check if there are more pages
                if data.get('has_next_page', False):
                    cursor = data.get('next_cursor', None)
                    self.logger.info(f"Next cursor: {cursor}")
                else:
                    cursor = None
                    self.logger.info("No more pages available")

                # Rate limiting
                time.sleep(self.config['delay_between_requests'])

            except requests.exceptions.RequestException as e:
                self.logger.error(
                    f"Error fetching tweets on page {page_count}: {e}")
                # Log the error in JSONL for debugging
                self.log_api_call(query, params, None, {
                                  'error': str(e)}, page_count)
                break

        self.logger.info(
            f"Total tweets found across {page_count} pages: {len(all_tweets)}")
        self.logger.info(f"All tweets: {all_tweets[:100]}")
        if len(all_tweets) > 0:
            filename = f"{self.current_company}/{username}_tweets.json"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(all_tweets, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved tweets to {filepath}")

        return all_tweets

    def log_api_call(self, query: str, params: Dict[str, Any], status_code: Optional[int],
                     response_data: Dict[str, Any], page_count: int):
        """
        Log API call details to JSONL file for debugging

        Args:
            query: Search query
            params: API parameters
            status_code: HTTP status code
            response_data: API response data
            page_count: Current page number
        """
        if not self.jsonl_file:
            return

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'params': params,
            'status_code': status_code,
            'page_count': page_count,
            'response_data': response_data
        }

        with self.jsonl_lock:
            with open(self.jsonl_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    def save_tweet_response_realtime(self, result: Dict[str, Any], deal_id: str, company_name: str):
        """
        Save individual tweet response in real-time to JSONL file

        Args:
            result: Tweet search result for a single follower
            deal_id: Deal ID
            company_name: Company name
        """
        if not self.tweet_responses_file:
            return

        # Prepare response data for JSONL
        response_entry = {
            'timestamp': datetime.now().isoformat(),
            'deal_id': deal_id,
            'company_name': company_name,
            'follower_info': {
                'screen_name': result['follower'].get('screen_name'),
                'name': result['follower'].get('name'),
                'followers_count': result['follower'].get('followers_count'),
                'statuses_count': result['follower'].get('statuses_count')
            },
            'search_query': result['search_query'],
            'tweet_count': result['tweet_count'],
            'tweets_found': result['tweets_found'],
            'processing_status': 'completed'
        }

        # Save to JSONL file immediately
        with self.tweet_responses_lock:
            with open(self.tweet_responses_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(response_entry, ensure_ascii=False) + '\n')

        self.logger.info(
            f"Saved tweet response for @{result['follower'].get('screen_name')} to JSONL")

    def save_aggregated_results_json(self, results: List[Dict[str, Any]], deal_id: str, company_name: str) -> str:
        """
        Save aggregated tweet results to JSON file (for backward compatibility)

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
            'note': 'This is aggregated data. Individual responses are saved in real-time to JSONL file.',
            'tweet_details': []
        }

        for result in results:
            follower = result['follower']
            tweets = result['tweets_found']

            tweet_detail = {
                'follower_info': {
                    'screen_name': follower.get('screen_name'),
                    'name': follower.get('name'),
                    'followers_count': follower.get('followers_count'),
                    'statuses_count': follower.get('statuses_count')
                },
                'search_query': result['search_query'],
                'tweet_count': result['tweet_count'],
                'tweets_found': []
            }

            # Add tweet details
            for tweet in tweets:
                tweet_info = tweet
                tweet_detail['tweets_found'].append(tweet_info)

            json_data['tweet_details'].append(tweet_detail)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Aggregated tweet details saved to: {filepath}")
        return filepath

    def search_follower_tweets_with_template(self, follower: Dict[str, Any], refined_query_template: str) -> Optional[Dict[str, Any]]:
        """
        Search for tweets from a specific follower using a pre-built refined query template

        Args:
            follower: Follower data
            refined_query_template: Pre-built query template without username

        Returns:
            Dictionary containing search results or None if no tweets
        """
        username = follower.get('screen_name') or follower.get('userName')
        if not username:
            return None

        # Build final query by combining username with template
        query = self.build_search_query_with_template(
            username, refined_query_template)
        tweets = self.search_tweets_with_cursor(query, username)

        if tweets:
            return {
                'follower': follower,
                'company_name': getattr(self, 'current_company', 'unknown'),
                'products_searched': [],  # Products are now part of the template
                'search_query': query,
                'tweets_found': tweets,
                'tweet_count': len(tweets),
                'search_timestamp': datetime.now().isoformat()
            }

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
        tweets = self.search_tweets_with_cursor(query)

        if tweets:
            return {
                'follower': follower,
                'company_name': company_name,
                'products_searched': products,
                'search_query': query,
                'tweets_found': tweets,
                'tweet_count': len(tweets),
                'search_timestamp': datetime.now().isoformat()
            }

        return None

    def process_single_follower(self, args: tuple) -> Optional[Dict[str, Any]]:
        """
        Process a single follower for tweet search (for thread pool)

        Args:
            args: Tuple containing (follower, company_name, products, company_handle, refined_query_template, index, total)

        Returns:
            Dictionary containing search results or None if no tweets
        """
        follower, company_name, products, company_handle, refined_query_template, index, total = args
        username = follower.get('screen_name') or follower.get('userName')

        if not username:
            return None

        try:
            self.logger.info(
                f"Searching tweets for {index}/{total}: @{username}")

            result = self.search_follower_tweets_with_template(
                follower, refined_query_template)

            # Update counters thread-safely
            with self.lock:
                self.processed_count += 1
                if result:
                    self.followers_with_tweets += 1
                    self.total_tweets_found += result.get('tweet_count', 0)
                    self.logger.info(
                        f"✓ Found {result.get('tweet_count', 0)} tweets for @{username}")
                else:
                    self.logger.info(f"✗ No tweets found for @{username}")

            return result

        except Exception as e:
            self.logger.error(f"Error processing follower @{username}: {e}")
            with self.lock:
                self.processed_count += 1
            return None

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
            tweets = result['tweets_found']

            tweet_detail = {
                'follower_info': {
                    'screen_name': follower.get('screen_name'),
                    'name': follower.get('name'),
                    'followers_count': follower.get('followers_count'),
                    'statuses_count': follower.get('statuses_count')
                },
                'search_query': result['search_query'],

                'tweet_count': result['tweet_count'],
                'tweets_found': []
            }

            # Add tweet details
            for tweet in tweets:
                tweet_info = tweet
                tweet_detail['tweets_found'].append(tweet_info)

            json_data['tweet_details'].append(tweet_detail)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Tweet details saved to: {filepath}")
        return filepath

    def process_company_followers(self, deal_id: str, company_name: str, company_handle: str, products: List[str]) -> Dict[str, Any]:
        """
        Process followers for a single company using tweet search.
        This method is designed to be called by process_deal for parallel processing.

        Args:
            deal_id: Deal ID
            company_name: Company name
            company_handle: Company Twitter handle
            products: List of product names for this company

        Returns:
            Dictionary containing processing results for this company.
        """
        self.logger.info(
            f"Processing followers for company @{company_handle}: {company_name}")

        # Set current company for filtering
        self.current_company = company_name

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

        # Step 3: Apply follower limit if configured
        max_followers = self.config['max_followers_per_company']
        if max_followers is not None and max_followers > 0:
            original_count = len(filtered_followers)
            filtered_followers = filtered_followers[:max_followers]
            self.logger.info(
                f"Limited to {max_followers} followers for processing (from {original_count} filtered followers)")

        # Step 4: Build refined search query template once
        self.logger.info("Building refined search query template using GPT...")
        refined_query_template = self.build_refined_search_query_with_gpt(
            company_name, products, company_handle)

        self.logger.info(
            f"FinalRefined search query template(GPT): {refined_query_template}")

        # Step 5: Search tweets for followers
        self.processed_count = 0
        self.followers_with_tweets = 0
        self.total_tweets_found = 0

        tweet_results = []

        if self.config['use_parallel_processing'] and self.config['max_workers'] > 1:
            self.logger.info(
                f"Using parallel processing with {self.config['max_workers']} workers")

            # Prepare arguments for parallel processing - now includes refined query template
            args_list = [(follower, company_name, products, company_handle, refined_query_template, i+1, len(filtered_followers))
                         for i, follower in enumerate(filtered_followers)]

            with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
                future_to_args = {executor.submit(self.process_single_follower, args): args
                                  for args in args_list}

                for future in as_completed(future_to_args):
                    try:
                        result = future.result()
                        if result:
                            tweet_results.append(result)
                            self.save_tweet_response_realtime(
                                result, deal_id, company_name)

                        # Rate limiting
                        if (self.config['delay_every_n_requests'] > 0 and
                                self.processed_count % self.config['delay_every_n_requests'] == 0):
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
                    (follower, company_name, products, company_handle, refined_query_template, i, len(filtered_followers)))
                if result:
                    tweet_results.append(result)
                    self.save_tweet_response_realtime(
                        result, deal_id, company_name)

                # Rate limiting
                if (self.config['delay_every_n_requests'] > 0 and
                        i % self.config['delay_every_n_requests'] == 0):
                    time.sleep(self.config['delay_for_rate_limit'])
                else:
                    time.sleep(self.config['delay_between_requests'])

        # Step 5: Save both JSONL (real-time) and JSON (batch) files
        if tweet_results:
            # JSONL is already saved in real-time during processing
            self.logger.info(
                f"Real-time JSONL responses saved to: {self.tweet_responses_file}")

            # Also save aggregated JSON for backward compatibility and analysis
            if self.config.get('save_tweet_details', True):
                json_file = self.save_tweet_details_json(
                    tweet_results, deal_id, company_name)
                self.logger.info(f"Aggregated JSON saved to: {json_file}")

            # Save aggregated results in new format if enabled
            if self.config.get('save_aggregated_results', True):
                aggregated_file = self.save_aggregated_results_json(
                    tweet_results, deal_id, company_name)
                self.logger.info(
                    f"New aggregated format saved to: {aggregated_file}")

        return {
            'company_name': company_name,
            'company_handle': company_handle,
            'total_followers': len(followers),
            'filtered_followers': len(filtered_followers),
            'followers_with_tweets': len(tweet_results),
            'total_tweets_found': self.total_tweets_found,
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

        # Setup JSONL file for debugging API calls
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.jsonl_file = os.path.join(
            self.output_dir, f"api_calls_debug_{deal_id}_{timestamp}.jsonl")

        # Create empty JSONL file
        with open(self.jsonl_file, 'w', encoding='utf-8') as f:
            pass
        self.logger.info(f"Debug JSONL file created: {self.jsonl_file}")

        # Setup JSONL file for tweet responses (real-time saving)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.tweet_responses_file = os.path.join(
            self.output_dir, f"tweet_responses_{deal_id}_{timestamp}.jsonl")

        # Create empty JSONL file
        with open(self.tweet_responses_file, 'w', encoding='utf-8') as f:
            pass
        self.logger.info(
            f"Tweet responses JSONL file created: {self.tweet_responses_file}")

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
            total_tweets = sum(r.get('total_tweets_found', 0)
                               for r in all_results)
            self.logger.info("=== Tweet Search Processing Complete ===")
            self.logger.info(f"Deal ID: {deal_id}")
            self.logger.info(f"Total companies processed: {len(all_results)}")
            self.logger.info(
                f"Total high-value followers found: {total_high_value}")
            self.logger.info(f"Total tweets found: {total_tweets}")
            self.logger.info(f"Summary saved to: {summary_filepath}")
            self.logger.info(f"Debug JSONL saved to: {self.jsonl_file}")

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
        total_tweets = sum(r.get('total_tweets_found', 0)
                           for r in all_results)

        # Get JSONL summary for more accurate data
        jsonl_summary = self.get_tweet_responses_summary()

        # Verify both approaches are working
        dual_verification = self.verify_dual_approach()

        output_data = {
            'deal_id': deal_id,
            'processing_timestamp': datetime.now().isoformat(),
            'approach': 'TWEET_SEARCH_TEST',
            'total_companies': len(all_results),
            'summary': {
                'total_followers': total_followers,
                'total_filtered_followers': total_filtered,
                'total_followers_with_tweets': total_with_tweets,
                'total_tweets_found': total_tweets
            },
            'company_results': all_results,
            'config_used': self.config,
            'debug_files': {
                'jsonl_debug_file': self.jsonl_file,
                'tweet_responses_jsonl_file': self.tweet_responses_file
            },
            'jsonl_summary': jsonl_summary,
            'note': 'BOTH approaches are enabled: 1) Real-time JSONL streaming for immediate persistence, 2) Traditional JSON files for analysis and backward compatibility.',
            'dual_approach_enabled': True,
            'files_generated': {
                'real_time_jsonl': self.tweet_responses_file,
                'debug_jsonl': self.jsonl_file,
                'aggregated_json': 'Multiple JSON files per company',
                'summary_json': filepath
            },
            'dual_approach_verification': dual_verification
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results summary saved to: {filepath}")
        return filepath

    def read_tweet_responses_jsonl(self) -> List[Dict[str, Any]]:
        """
        Read all tweet responses from the JSONL file

        Returns:
            List of tweet response dictionaries
        """
        if not self.tweet_responses_file or not os.path.exists(self.tweet_responses_file):
            return []

        responses = []
        try:
            with open(self.tweet_responses_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            response = json.loads(line.strip())
                            responses.append(response)
                        except json.JSONDecodeError as e:
                            self.logger.warning(
                                f"Error parsing line {line_num} in JSONL: {e}")
                            continue
        except Exception as e:
            self.logger.error(f"Error reading tweet responses JSONL: {e}")

        return responses

    def get_tweet_responses_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all tweet responses from the JSONL file

        Returns:
            Dictionary with summary statistics
        """
        responses = self.read_tweet_responses_jsonl()

        if not responses:
            return {'total_responses': 0, 'companies': {}, 'total_tweets': 0}

        # Group by company
        companies = {}
        total_tweets = 0

        for response in responses:
            company = response.get('company_name', 'unknown')
            tweet_count = response.get('tweet_count', 0)

            if company not in companies:
                companies[company] = {
                    'followers_processed': 0,
                    'total_tweets_found': 0,
                    'followers_with_tweets': 0
                }

            companies[company]['followers_processed'] += 1
            companies[company]['total_tweets_found'] += tweet_count
            if tweet_count > 0:
                companies[company]['followers_with_tweets'] += 1

            total_tweets += tweet_count

        return {
            'total_responses': len(responses),
            'companies': companies,
            'total_tweets': total_tweets,
            'jsonl_file': self.tweet_responses_file
        }

    def verify_dual_approach(self) -> Dict[str, Any]:
        """
        Verify that both JSONL and JSON approaches are working

        Returns:
            Dictionary with verification results
        """
        verification = {
            'dual_approach_enabled': True,
            'jsonl_files': {},
            'json_files': {},
            'status': 'unknown'
        }

        # Check JSONL files
        if self.tweet_responses_file and os.path.exists(self.tweet_responses_file):
            jsonl_stats = os.stat(self.tweet_responses_file)
            verification['jsonl_files']['tweet_responses'] = {
                'path': self.tweet_responses_file,
                'size_bytes': jsonl_stats.st_size,
                'exists': True
            }
        else:
            verification['jsonl_files']['tweet_responses'] = {'exists': False}

        if self.jsonl_file and os.path.exists(self.jsonl_file):
            jsonl_stats = os.stat(self.jsonl_file)
            verification['jsonl_files']['api_debug'] = {
                'path': self.jsonl_file,
                'size_bytes': jsonl_stats.st_size,
                'exists': True
            }
        else:
            verification['jsonl_files']['api_debug'] = {'exists': False}

        # Check JSON files in output directory
        if os.path.exists(self.output_dir):
            json_files = []
            for file in os.listdir(self.output_dir):
                if file.endswith('.json'):
                    file_path = os.path.join(self.output_dir, file)
                    file_stats = os.stat(file_path)
                    json_files.append({
                        'name': file,
                        'size_bytes': file_stats.st_size,
                        'path': file_path
                    })
            verification['json_files'] = json_files

        # Determine overall status
        jsonl_working = any(f.get('exists', False)
                            for f in verification['jsonl_files'].values())
        json_working = len(verification['json_files']) > 0

        if jsonl_working and json_working:
            verification['status'] = 'both_working'
        elif jsonl_working:
            verification['status'] = 'jsonl_only'
        elif json_working:
            verification['status'] = 'json_only'
        else:
            verification['status'] = 'neither_working'

        return verification


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
            "Usage: python high_value_followers_tweet_search_test.py <deal_id> [options]")
        logger.error("Options:")
        logger.error(
            "  --workers <number>           Number of parallel workers (default: 10)")
        logger.error(
            "  --max-followers <number>     Maximum followers per company (default: all)")
        logger.error("Examples:")
        logger.error(
            "  python high_value_followers_tweet_search_test.py 68184d52478abf06ec1a28ec")
        logger.error(
            "  python high_value_followers_tweet_search_test.py 68184d52478abf06ec1a28ec --workers 20")
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
        processor = HighValueFollowersTweetSearchProcessorTest(
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
