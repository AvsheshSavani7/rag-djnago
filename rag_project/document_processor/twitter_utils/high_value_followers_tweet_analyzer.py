#!/usr/bin/env python3
"""
High Value Followers Tweet Analyzer
Reads high-value followers and searches for their tweets about company/products,
then verifies them with GPT for antitrust/business importance.

Usage:
# python high_value_followers_tweet_analyzer.py 682f00def21b9fca8e1d04fe
"""

from document_processor.models import ProcessingJob, HighValueFollowers, CompanyProducts, SearchQuery, Tweet
import django
import os
import sys
import json
import time
import logging
import requests
import openai
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


class HighValueFollowersTweetAnalyzer:
    """Main class for analyzing tweets from high-value followers"""

    def __init__(self, twitter_api_key: Optional[str] = None, openai_api_key: Optional[str] = None, config_overrides: Dict[str, Any] = None):
        """Initialize the analyzer"""
        # Setup configuration
        self.config = {
            'search_date_range_years': 5,
            'max_tweets_per_user': None,  # Default to 100, None = fetch all tweets
            'search_timeout': 120,
            'gpt_model': 'gpt-4o-mini',
            'gpt_max_tokens': 150,
            'gpt_temperature': 0.1,
            'delay_between_requests': 0,
            'max_workers': 20,
            'use_parallel_processing': True,
            'approach': 'GUNSHOT',
            'save_results_summary': True,
            'save_tweet_details': True,
            'batch_size': 50,
            'use_batch_saving': True,

            # Temporary testing mode
            'skip_gpt_analysis': True,  # Skip GPT analysis for testing
            'skip_mongodb_save': False,  # Skip MongoDB saving for testing
            'save_to_json_only': False,  # Only save to JSON files
        }

        if config_overrides:
            self.config.update(config_overrides)

        # Setup API keys
        self.twitter_api_key = twitter_api_key or os.getenv('TWITTER_API_KEY')
        self.openai_api_key = openai_api_key or os.getenv(
            'OPENAI_API_KEY_SEC_FILING')

        if not self.twitter_api_key:
            raise ValueError(
                "Twitter API key is required. Set TWITTER_API_KEY environment variable.")
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY_SEC_FILING environment variable.")

        # Setup clients
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        self.base_url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
        self.headers = {
            'X-API-Key': self.twitter_api_key,
            'Content-Type': 'application/json'
        }

        # Setup logger and output directory
        self.logger = logging.getLogger(__name__)
        self.output_dir = os.path.join(os.path.dirname(
            __file__), 'high_value_followers_tweet_analysis_results')
        os.makedirs(self.output_dir, exist_ok=True)

        # Log testing mode status
        if self.config.get('skip_gpt_analysis', False):
            self.logger.info(
                "🧪 TESTING MODE: GPT analysis disabled - all tweets will be considered important")
        if self.config.get('skip_mongodb_save', False):
            self.logger.info(
                "🧪 TESTING MODE: MongoDB saving disabled - results will be saved to JSON only")

        # Thread-safe counters
        self.processed_count = 0
        self.followers_with_tweets = 0
        self.total_tweets_found = 0
        self.important_tweets_found = 0
        self.lock = threading.Lock()

    def fetch_deal_data(self, deal_id: str) -> Optional[Dict[str, Any]]:
        """Fetch deal data from database"""
        try:
            deal = ProcessingJob.objects.get(id=deal_id)
            return {
                'id': str(deal.id),
                'cik': deal.cik,
                'acquire_name': deal.acquire_name,
                'target_name': deal.target_name,
                'announce_date': deal.announce_date.strftime('%Y-%m-%d') if deal.announce_date else None,
                'schema_results': deal.schema_results,
                'twitter_details': deal.twitter_details
            }
        except ProcessingJob.DoesNotExist:
            self.logger.error(f"Deal with ID {deal_id} not found")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching deal data: {e}")
            return None

    def fetch_company_products(self, deal_id: str) -> Dict[str, List[str]]:
        """Fetch company products from database"""
        products_data = {}
        try:
            company_products = CompanyProducts.objects(deal_id=deal_id)
            for cp in company_products:
                company_name = cp.company
                products = cp.products
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
        """Extract Twitter handles for target and acquire companies"""
        twitter_handles = {}
        twitter_details = deal_data.get('twitter_details', {})
        if not twitter_details:
            self.logger.warning("No Twitter details found in deal data")
            return twitter_handles

        company_handles = twitter_details.get('company_handles', {})

        target_name = deal_data.get('target_name')
        if target_name and target_name in company_handles:
            target_handle = company_handles[target_name].get(
                'main_twitter_handle')
            if target_handle:
                target_handle = target_handle.lstrip('@')
                twitter_handles[target_name] = target_handle
                self.logger.info(
                    f"Found target Twitter handle: @{target_handle}")

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

    def get_high_value_followers(self, deal_id: str) -> List[Dict[str, Any]]:
        """Get high-value followers from database"""
        try:
            self.logger.info(
                f"Querying HighValueFollowers for deal_id: {deal_id}")
            followers = HighValueFollowers.objects.filter(deal_id=deal_id)
            self.logger.info(
                f"Found {followers.count()} high-value followers in database")
            followers_data = []
            for follower in followers:
                follower_dict = {
                    'id': str(follower.id),
                    'deal_id': follower.deal_id,
                    'company_name': follower.company_name,
                    'company_handle': follower.company_handle,
                    'follower_id': follower.follower_id,
                    'name': follower.name,
                    'screen_name': follower.screen_name,
                    'description': follower.description,
                    'location': follower.location,
                    'followers_count': follower.followers_count,
                    'statuses_count': follower.statuses_count,
                    'protected': follower.protected,
                    'verified': follower.verified,
                    'overall_score': follower.overall_score,
                    'reason': follower.reason,
                    'key_indicators': follower.key_indicators,
                    'gpt_model_used': follower.gpt_model_used,
                    'approach': follower.approach
                }
                followers_data.append(follower_dict)
            self.logger.info(
                f"Processed {len(followers_data)} high-value followers")
            return followers_data
        except Exception as e:
            self.logger.error(f"Error fetching high-value followers: {e}")
            return []

    def build_search_query(self, username: str, company_name: str, products: List[str], company_handle: str = None) -> str:
        """Build Twitter search query for user mentioning company or products"""
        end_date = datetime.now()
        start_date = datetime(
            end_date.year - self.config['search_date_range_years'], 1, 1)
        since_date = start_date.strftime("%Y-%m-%d_00:00:00_UTC")
        until_date = end_date.strftime("%Y-%m-%d_23:59:59_UTC")

        query_parts = [f"from:{username}"]
        if company_name:
            query_parts.append(f'"{company_name}"')
        if company_handle:
            clean_handle = company_handle.lstrip('@')
            query_parts.append(f'@{clean_handle}')
        for product in products:
            if product and product.strip():
                query_parts.append(f'"{product.strip()}"')

        search_terms = " OR ".join(query_parts[1:]) if len(
            query_parts) > 1 else ""
        if search_terms:
            final_query = f"from:{username} ({search_terms}) since:{since_date} until:{until_date} lang:en"
        else:
            final_query = f"from:{username} since:{since_date} until:{until_date} lang:en"

        self.logger.info(f"Search query: {final_query}")
        return final_query

    def search_tweets_with_cursor(self, query: str, max_tweets: int = 100) -> List[Dict[str, Any]]:
        """Search for tweets using the advanced search API with cursor pagination"""
        # Handle None value for max_tweets - fetch all tweets if None

        all_tweets = []
        cursor = ""
        page_count = 0
        max_pages = 10

        while page_count < max_pages:
            # Check if we've reached the max tweets limit (only if max_tweets is not infinity)
            if max_tweets != float('inf') and len(all_tweets) >= max_tweets:
                all_tweets = all_tweets[:max_tweets]  # Truncate to exact limit
                self.logger.info(f"Reached max tweets limit: {max_tweets}")
                break

            params = {
                'query': query,
                'queryType': 'Latest',
                'cursor': cursor
            }

            self.logger.info(f"Search params: {params}")

            try:
                response = requests.get(
                    self.base_url, headers=self.headers, params=params,
                    timeout=self.config['search_timeout'])
                response.raise_for_status()

                data = response.json()
                tweets = data.get('tweets', [])

                if not tweets:
                    break

                all_tweets.extend(tweets)
                cursor = data.get('next_cursor', '')
                if not cursor:
                    break

                page_count += 1
                time.sleep(self.config['delay_between_requests'])

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching tweets: {e}")
                break

        # Final truncation if needed
        if max_tweets != float('inf'):
            all_tweets = all_tweets[:max_tweets]

        self.logger.info(
            f"Found {len(all_tweets)} tweets in {page_count} pages")
        return all_tweets

    def analyze_tweet_with_gpt(self, tweet_text: str, company_name: str) -> Optional[Dict[str, Any]]:
        """Analyze tweet with GPT for antitrust/business importance"""
        try:
            prompt = f"""
Analyze this tweet about {company_name} for antitrust and business importance.

Tweet: "{tweet_text}"

Determine if this tweet is:
1. Related to antitrust concerns, regulatory issues, or competition
2. Important from a business perspective (market analysis, strategic insights, etc.)
3. From a user whose voice/opinion is significant in the industry

Respond in this exact JSON format:
{{
    "important": "Yes" or "No",
    "reason": "Brief explanation in 50 words or less"
}}

Only respond with the JSON object, nothing else.
"""

            response = self.openai_client.chat.completions.create(
                model=self.config['gpt_model'],
                messages=[
                    {"role": "system", "content": "You are an expert in antitrust law and business analysis. Provide concise, accurate assessments."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config['gpt_max_tokens'],
                temperature=self.config['gpt_temperature']
            )

            content = response.choices[0].message.content.strip()
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                self.logger.error(
                    f"Failed to parse GPT response as JSON: {content}")
                return None

        except Exception as e:
            self.logger.error(f"Error analyzing tweet with GPT: {e}")
            return None

    def save_important_tweet(self, tweet_data: Dict[str, Any], search_query_id: str, gpt_analysis: Dict[str, Any]) -> Optional[str]:
        """Save important tweet to database"""
        try:
            tweet = Tweet(
                search_query_id=search_query_id,
                tweet=tweet_data,
                approach=self.config['approach']
            )
            tweet.save()
            self.logger.info(
                f"Saved important tweet: {tweet_data.get('id', 'unknown')}")
            return str(tweet.id)
        except Exception as e:
            self.logger.error(f"Error saving tweet: {e}")
            return None

    def create_search_query_record(self, query: str, deal_id: str, follower_info: Dict[str, Any], company_name: str) -> str:
        """Create a search query record in database"""
        try:
            combination = {
                'follower_screen_name': follower_info.get('screen_name'),
                'follower_id': follower_info.get('follower_id'),
                'company_name': company_name,
                'follower_score': follower_info.get('overall_score'),
                'follower_approach': follower_info.get('approach')

            }

            self.logger.info(f"Search query: {query}")

            search_query = SearchQuery(
                search_query=query,
                deal_id=deal_id,
                approach=self.config['approach'],
                combination=combination,
                total_tweets=0
            )
            search_query.save()
            return str(search_query.id)
        except Exception as e:
            self.logger.error(f"Error creating search query record: {e}")
            return None

    def process_single_follower(self, args: tuple) -> Optional[Dict[str, Any]]:
        """Process a single high-value follower for tweet analysis"""
        follower, company_name, products, company_handle, index, total = args
        username = follower.get('screen_name')

        if not username:
            return None

        try:
            self.logger.info(
                f"Analyzing tweets for {index}/{total}: @{username}")

            query = self.build_search_query(
                username, company_name, products, company_handle)

            # Get max_tweets_per_user with safety check
            max_tweets = self.config.get('max_tweets_per_user', 100)
            if max_tweets is None:
                self.logger.info(
                    f"max_tweets_per_user is None for @{username}, will fetch all available tweets")

            tweets = self.search_tweets_with_cursor(
                query, max_tweets)

            if not tweets:
                with self.lock:
                    self.processed_count += 1
                self.logger.info(f"✗ No tweets found for @{username}")
                return None

            # Skip search query record creation if MongoDB saving is disabled
            search_query_id = None
            if not self.config.get('skip_mongodb_save', False):
                search_query_id = self.create_search_query_record(
                    query, follower['deal_id'], follower, company_name)
                if not search_query_id:
                    with self.lock:
                        self.processed_count += 1
                    return None

            important_tweets = []
            for tweet in tweets:
                tweet_text = tweet.get('text', '')
                if not tweet_text:
                    continue

                # Skip GPT analysis if disabled
                if self.config.get('skip_gpt_analysis', False):
                    # For testing, consider all tweets as "important"
                    gpt_analysis = {
                        "important": "Yes", "reason": "Testing mode - all tweets considered important"}
                else:
                    gpt_analysis = self.analyze_tweet_with_gpt(
                        tweet_text, company_name)

                if gpt_analysis and gpt_analysis.get('important') == 'Yes':
                    # Skip MongoDB saving if disabled
                    saved_tweet_id = None
                    if not self.config.get('skip_mongodb_save', False):
                        saved_tweet_id = self.save_important_tweet(
                            tweet, search_query_id, gpt_analysis)

                    important_tweets.append({
                        'tweet': tweet,
                        'gpt_analysis': gpt_analysis,
                        'saved_tweet_id': saved_tweet_id
                    })

                time.sleep(self.config['delay_between_requests'])

            with self.lock:
                self.processed_count += 1
                if tweets:
                    self.followers_with_tweets += 1
                    self.total_tweets_found += len(tweets)
                    self.important_tweets_found += len(important_tweets)

            if important_tweets:
                self.logger.info(
                    f"✓ Found {len(important_tweets)} important tweets for @{username}")
            else:
                self.logger.info(
                    f"✗ No important tweets found for @{username}")

            return {
                'follower': follower,
                'company_name': company_name,
                'search_query': query,
                'search_query_id': search_query_id,
                'total_tweets_found': len(tweets),
                'important_tweets': important_tweets,
                'all_tweets': tweets if self.config.get('skip_gpt_analysis', False) else None,
                'analysis_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error processing follower @{username}: {e}")
            with self.lock:
                self.processed_count += 1
            return None

    def process_company_followers(self, deal_id: str, company_name: str, company_handle: str, products: List[str]) -> Dict[str, Any]:
        """Process high-value followers for a specific company"""
        self.logger.info(
            f"Processing high-value followers for {company_name} (@{company_handle})")

        followers = self.get_high_value_followers(deal_id)
        self.logger.info(f"Total followers found for deal: {len(followers)}")

        company_followers = [
            f for f in followers if f['company_name'] == company_name]

        self.logger.info(
            f"Found {len(company_followers)} high-value followers for {company_name}")

        if not company_followers:
            self.logger.warning(
                f"No high-value followers found for {company_name}")
            return {
                'company_name': company_name,
                'company_handle': company_handle,
                'total_followers': 0,
                'followers_with_tweets': 0,
                'total_tweets_found': 0,
                'important_tweets_found': 0,
                'analysis_results': [],
                'processing_status': 'failed'
            }

        self.processed_count = 0
        self.followers_with_tweets = 0
        self.total_tweets_found = 0
        self.important_tweets_found = 0

        analysis_results = []

        if self.config['use_parallel_processing'] and self.config['max_workers'] > 1:
            self.logger.info(
                f"Using parallel processing with {self.config['max_workers']} workers")
            args_list = [(follower, company_name, products, company_handle, i+1, len(company_followers))
                         for i, follower in enumerate(company_followers)]

            with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
                future_to_args = {executor.submit(self.process_single_follower, args): args
                                  for args in args_list}

                for future in as_completed(future_to_args):
                    try:
                        result = future.result()
                        if result:
                            analysis_results.append(result)
                    except Exception as e:
                        self.logger.error(
                            f"Exception in parallel processing: {e}")
        else:
            self.logger.info("Using sequential processing")
            for i, follower in enumerate(company_followers, 1):
                result = self.process_single_follower(
                    (follower, company_name, products, company_handle, i, len(company_followers)))
                if result:
                    analysis_results.append(result)

        return {
            'company_name': company_name,
            'company_handle': company_handle,
            'total_followers': len(company_followers),
            'followers_with_tweets': self.followers_with_tweets,
            'total_tweets_found': self.total_tweets_found,
            'important_tweets_found': self.important_tweets_found,
            'analysis_results': analysis_results,
            'processing_status': 'completed'
        }

    def save_results_summary(self, deal_id: str, all_results: List[Dict[str, Any]]) -> str:
        """Save processing results summary to JSON file"""
        filename = f"high_value_tweet_analysis_summary_{deal_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)

        total_followers = sum(r.get('total_followers', 0) for r in all_results)
        total_with_tweets = sum(r.get('followers_with_tweets', 0)
                                for r in all_results)
        total_tweets = sum(r.get('total_tweets_found', 0) for r in all_results)
        total_important = sum(r.get('important_tweets_found', 0)
                              for r in all_results)

        output_data = {
            'deal_id': deal_id,
            'processing_timestamp': datetime.now().isoformat(),
            'approach': 'GUNSHOT',
            'total_companies': len(all_results),
            'summary': {
                'total_high_value_followers': total_followers,
                'followers_with_tweets': total_with_tweets,
                'total_tweets_found': total_tweets,
                'important_tweets_found': total_important
            },
            'company_results': all_results,
            'config_used': self.config
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results summary saved to: {filepath}")
        return filepath

    def save_detailed_results(self, deal_id: str, all_results: List[Dict[str, Any]]) -> str:
        """Save detailed analysis results to JSON file for debugging"""
        filename = f"high_value_tweet_analysis_detailed_{deal_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)

        detailed_data = {
            'deal_id': deal_id,
            'processing_timestamp': datetime.now().isoformat(),
            'approach': 'GUNSHOT',
            'detailed_results': []
        }

        for result in all_results:
            company_data = {
                'company_name': result['company_name'],
                'company_handle': result['company_handle'],
                'summary': {
                    'total_followers': result['total_followers'],
                    'followers_with_tweets': result['followers_with_tweets'],
                    'total_tweets_found': result['total_tweets_found'],
                    'important_tweets_found': result['important_tweets_found']
                },
                'follower_analyses': []
            }

            for analysis in result.get('analysis_results', []):
                follower_analysis = {
                    'follower_screen_name': analysis['follower']['screen_name'],
                    'follower_score': analysis['follower']['overall_score'],
                    'search_query': analysis['search_query'],
                    'total_tweets_found': analysis['total_tweets_found'],
                    'important_tweets': []
                }

                # In testing mode, include all tweets, not just important ones
                if self.config.get('skip_gpt_analysis', False):
                    # Include all tweets found for this follower
                    for tweet in analysis.get('all_tweets', []):
                        tweet_info = {
                            'tweet_id': tweet.get('id'),
                            'tweet_text': tweet.get('text', '')[:200] + '...' if len(tweet.get('text', '')) > 200 else tweet.get('text', ''),
                            'tweet_created_at': tweet.get('created_at'),
                            'gpt_analysis': {"important": "Yes", "reason": "Testing mode - all tweets included"},
                            'saved_tweet_id': None
                        }
                        follower_analysis['important_tweets'].append(
                            tweet_info)
                else:
                    # Normal mode - only include important tweets
                    for tweet_data in analysis.get('important_tweets', []):
                        tweet_info = {
                            'tweet_id': tweet_data['tweet'].get('id'),
                            'tweet_text': tweet_data['tweet'].get('text', '')[:200] + '...' if len(tweet_data['tweet'].get('text', '')) > 200 else tweet_data['tweet'].get('text', ''),
                            'tweet_created_at': tweet_data['tweet'].get('created_at'),
                            'gpt_analysis': tweet_data['gpt_analysis'],
                            'saved_tweet_id': tweet_data['saved_tweet_id']
                        }
                        follower_analysis['important_tweets'].append(
                            tweet_info)

                company_data['follower_analyses'].append(follower_analysis)

            detailed_data['detailed_results'].append(company_data)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Detailed results saved to: {filepath}")
        return filepath

    def process_deal(self, deal_id: str) -> Optional[str]:
        """Main method to process high-value followers' tweets for a deal"""
        self.logger.info(
            f"Starting High Value Followers Tweet Analysis for deal ID: {deal_id}")

        deal_data = self.fetch_deal_data(deal_id)
        if not deal_data:
            return None

        twitter_handles = self.extract_twitter_handles(deal_data)
        if len(twitter_handles) < 2:
            self.logger.warning(
                f"Not enough Twitter handles found for deal {deal_id}. Found: {twitter_handles}")
            return None

        products_data = self.fetch_company_products(deal_id)
        self.logger.info(
            f"Found products for companies: {list(products_data.keys())}")

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
                deal_id, target_name, target_handle, target_products)
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
                deal_id, acquire_name, acquire_handle, acquire_products)
            all_results.append(acquire_result)
            self.logger.info(
                f"Completed processing for acquire company @{acquire_handle}")
        except Exception as e:
            self.logger.error(
                f"Error processing acquire company @{acquire_handle}: {e}")

        # Save results
        if all_results:
            self.logger.info(
                f"Processing completed. Found {len(all_results)} company results")
            for result in all_results:
                self.logger.info(
                    f"  {result.get('company_name', 'Unknown')}: {result.get('total_followers', 0)} followers, {result.get('total_tweets_found', 0)} tweets")

            summary_filepath = self.save_results_summary(deal_id, all_results)
            detailed_filepath = self.save_detailed_results(
                deal_id, all_results)

            total_important = sum(r.get('important_tweets_found', 0)
                                  for r in all_results)
            self.logger.info(
                "=== High Value Followers Tweet Analysis Complete ===")
            self.logger.info(f"Deal ID: {deal_id}")
            self.logger.info(f"Total companies processed: {len(all_results)}")
            self.logger.info(
                f"Total important tweets found: {total_important}")
            self.logger.info(f"Summary saved to: {summary_filepath}")
            self.logger.info(f"Detailed results saved to: {detailed_filepath}")

            return summary_filepath
        else:
            self.logger.warning("No results to save - all_results is empty")
            # Still create a minimal summary file to show what happened
            minimal_results = [{
                'company_name': 'No companies processed',
                'company_handle': 'none',
                'total_followers': 0,
                'followers_with_tweets': 0,
                'total_tweets_found': 0,
                'important_tweets_found': 0,
                'analysis_results': [],
                'processing_status': 'no_results'
            }]
            summary_filepath = self.save_results_summary(
                deal_id, minimal_results)
            self.logger.info(f"Minimal summary saved to: {summary_filepath}")
            return summary_filepath


def main():
    """Main function to run the script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logger = logging.getLogger(__name__)

    if len(sys.argv) < 2:
        logger.error(
            "Usage: python high_value_followers_tweet_analyzer.py <deal_id> [options]")
        logger.error("Options:")
        logger.error(
            "  --workers <number>           Number of parallel workers (default: 5)")
        logger.error(
            "  --max-tweets <number>        Max tweets per user (default: 100)")
        logger.error(
            "  --max-tweets-all             Fetch all available tweets per user")
        logger.error(
            "  --gpt-model <model>          GPT model to use (default: gpt-4o-mini)")
        logger.error(
            "  --skip-gpt-analysis        Skip GPT analysis (testing mode)")
        logger.error(
            "  --skip-mongodb-save         Skip MongoDB saving (testing mode)")
        logger.error("Examples:")
        logger.error(
            "  python high_value_followers_tweet_analyzer.py 68184d52478abf06ec1a28ec")
        logger.error(
            "  python high_value_followers_tweet_analyzer.py 68184d52478abf06ec1a28ec --workers 10")
        logger.error(
            "  python high_value_followers_tweet_analyzer.py 68184d52478abf06ec1a28ec --skip-gpt-analysis --skip-mongodb-save")
        logger.error(
            "  python high_value_followers_tweet_analyzer.py 68184d52478abf06ec1a28ec --max-tweets-all --skip-gpt-analysis")
        sys.exit(1)

    deal_id = sys.argv[1]
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
        elif sys.argv[i] == '--max-tweets' and i + 1 < len(sys.argv):
            max_tweets = int(sys.argv[i + 1])
            if max_tweets < 1:
                logger.error("❌ --max-tweets must be at least 1")
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
        analyzer = HighValueFollowersTweetAnalyzer(
            config_overrides=config_overrides)
        result_file = analyzer.process_deal(deal_id)

        if result_file:
            logger.info(
                "High Value Followers Tweet Analysis completed successfully!")
            logger.info(f"Results saved to: {result_file}")

            # Show testing mode status
            if analyzer.config.get('skip_gpt_analysis', False):
                logger.info(
                    "🧪 Note: GPT analysis was skipped (testing mode)")
            if analyzer.config.get('skip_mongodb_save', False):
                logger.info(
                    "🧪 Note: MongoDB saving was skipped (testing mode)")
        else:
            logger.error("Processing failed or no results found")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
