#!/usr/bin/env python3
"""
High Value Followers Orchestrator
Combines all three approaches for comprehensive follower analysis:
1. Gun Shot Approach - Fetch all followers
2. GPT Analysis - Analyze followers with descriptions
3. Tweet Search - Analyze followers without descriptions

# python high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb --disable-step-1

Usage:
# python high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb
"""

from document_processor.twitter_utils.gun_shot_approach import GunShotFollowersAnalyzer
from document_processor.twitter_utils.high_value_followers_processor import HighValueFollowersProcessor
from document_processor.twitter_utils.high_value_followers_tweet_search import HighValueFollowersTweetSearchProcessor
from document_processor.twitter_utils.high_value_followers_tweet_analyzer import HighValueFollowersTweetAnalyzer
from document_processor.models import ProcessingJob, Followers, HighValueFollowers
from document_processor.twitter_utils.follower_utils import FollowerUtils
import django
import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()

# Load environment variables
load_dotenv()


class HighValueFollowersOrchestrator:
    """Main orchestrator class for comprehensive follower analysis"""

    def __init__(self, config_overrides: Dict[str, Any] = None):
        """
        Initialize the orchestrator

        Args:
            config_overrides: Optional configuration overrides
        """
        # Setup configuration
        self.config = {
            # Step configuration
            'enable_step_1_gun_shot': False,      # Fetch all followers
            'enable_step_2_gpt_analysis': False,  # Analyze followers with descriptions
            'enable_step_3_tweet_search': False,  # Analyze followers without descriptions
            # Analyze tweets from high-value followers
            'enable_step_4_tweet_analysis': True,

            # Gun Shot Approach settings
            'gun_shot_max_followers_per_company': None,  # None = fetch all

            # GPT Analysis settings
            'gpt_max_followers_per_company': None,  # Limit for GPT analysis
            'gpt_min_overall_score': 6,
            'gpt_max_workers': 100,
            'gpt_model': 'gpt-4.1-mini',

            # Tweet Search settings
            'tweet_search_max_followers_per_company': None,  # Limit for tweet search
            'tweet_search_max_workers': 20,
            'tweet_search_date_range_years': 5,

            # Tweet Analysis settings (Step 4)
            # Max tweets per high-value follower
            'tweet_analysis_max_tweets_per_user': None,  # Default to 100, None = fetch all
            'tweet_analysis_max_workers': 20,  # Number of parallel tweet analysis workers
            'tweet_analysis_gpt_model': 'gpt-4o-mini',

            # Common filtering criteria
            'min_followers_count': 250,
            'min_statuses_count': 250,

            # Processing approach
            'approach': 'GUNSHOT',

            # Output settings
            'save_results_summary': True,
            'save_individual_records': True,

            # Batch processing
            'batch_size': 1000,
            'use_batch_saving': True
        }

        # Apply any config overrides
        if config_overrides:
            self.config.update(config_overrides)

        # Setup utilities
        self.follower_utils = FollowerUtils()

        # Setup logger
        self.logger = logging.getLogger(__name__)

        # Setup output directory
        self.output_dir = os.path.join(
            os.path.dirname(__file__),
            'high_value_followers_orchestrator_results'
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize processors (lazy loading)
        self.gun_shot_analyzer = None
        self.gpt_processor = None
        self.tweet_search_processor = None
        self.tweet_analyzer = None

    def get_gun_shot_analyzer(self):
        """Lazy load gun shot analyzer"""
        if self.gun_shot_analyzer is None:
            self.gun_shot_analyzer = GunShotFollowersAnalyzer()
        return self.gun_shot_analyzer

    def get_gpt_processor(self):
        """Lazy load GPT processor"""
        if self.gpt_processor is None:
            gpt_config = {
                'max_followers_per_company': self.config['gpt_max_followers_per_company'],
                'min_overall_score': self.config['gpt_min_overall_score'],
                'max_workers': self.config['gpt_max_workers'],
                'gpt_model': self.config['gpt_model'],
                'min_followers_count': self.config['min_followers_count'],
                'min_statuses_count': self.config['min_statuses_count'],
                'approach': self.config['approach'],
                'batch_size': self.config['batch_size'],
                'use_batch_saving': self.config['use_batch_saving']
            }
            self.gpt_processor = HighValueFollowersProcessor(
                config_overrides=gpt_config)
        return self.gpt_processor

    def get_tweet_search_processor(self):
        """Lazy load tweet search processor"""
        if self.tweet_search_processor is None:
            tweet_config = {
                'max_followers_per_company': self.config['tweet_search_max_followers_per_company'],
                'max_workers': self.config['tweet_search_max_workers'],
                'search_date_range_years': self.config['tweet_search_date_range_years'],
                'min_followers_count': self.config['min_followers_count'],
                'min_statuses_count': self.config['min_statuses_count'],
                'approach': self.config['approach'],
                'batch_size': self.config['batch_size'],
                'use_batch_saving': self.config['use_batch_saving']
            }
            self.tweet_search_processor = HighValueFollowersTweetSearchProcessor(
                config_overrides=tweet_config)
        return self.tweet_search_processor

    def get_tweet_analyzer(self):
        """Lazy load tweet analyzer"""
        if self.tweet_analyzer is None:
            tweet_analysis_config = {
                'max_tweets_per_user': self.config['tweet_analysis_max_tweets_per_user'],
                'max_workers': self.config['tweet_analysis_max_workers'],
                'gpt_model': self.config['tweet_analysis_gpt_model'],
                'search_date_range_years': 5,  # Search last 5 years
                'approach': self.config['approach']
            }
            self.tweet_analyzer = HighValueFollowersTweetAnalyzer(
                config_overrides=tweet_analysis_config)
        return self.tweet_analyzer

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

    def step_1_gun_shot_approach(self, deal_id: str, deal_data: Dict[str, Any], twitter_handles: Dict[str, str]) -> Dict[str, Any]:
        """
        Step 1: Gun Shot Approach - Fetch all followers

        Args:
            deal_id: Deal ID
            deal_data: Deal data
            twitter_handles: Twitter handles dictionary

        Returns:
            Dictionary with step results
        """
        if not self.config['enable_step_1_gun_shot']:
            self.logger.info("Step 1 (Gun Shot Approach) is disabled")
            return {'status': 'skipped', 'message': 'Step disabled in config'}

        self.logger.info(
            "=== Step 1: Gun Shot Approach - Fetching Followers ===")

        try:
            analyzer = self.get_gun_shot_analyzer()

            # Get target and acquire handles
            target_name = deal_data.get('target_name')
            acquire_name = deal_data.get('acquire_name')
            target_handle = twitter_handles.get(target_name)
            acquire_handle = twitter_handles.get(acquire_name)

            results = {
                'target_company': {'status': 'failed', 'followers_fetched': 0},
                'acquire_company': {'status': 'failed', 'followers_fetched': 0}
            }

            # Fetch followers for target company
            if target_handle:
                self.logger.info(
                    f"Fetching followers for target company: @{target_handle}")
                try:
                    target_result = analyzer.fetch_followers_for_company(
                        target_handle, target_name, deal_id
                    )
                    results['target_company'] = {
                        'status': 'success',
                        'followers_fetched': target_result.get('total_followers', 0),
                        'handle': target_handle
                    }
                    self.logger.info(
                        f"✓ Target company: {target_result.get('total_followers', 0)} followers fetched")
                except Exception as e:
                    self.logger.error(
                        f"✗ Error fetching target company followers: {e}")
                    results['target_company']['error'] = str(e)

            # Fetch followers for acquire company
            if acquire_handle:
                self.logger.info(
                    f"Fetching followers for acquire company: @{acquire_handle}")
                try:
                    acquire_result = analyzer.fetch_followers_for_company(
                        acquire_handle, acquire_name, deal_id
                    )
                    results['acquire_company'] = {
                        'status': 'success',
                        'followers_fetched': acquire_result.get('total_followers', 0),
                        'handle': acquire_handle
                    }
                    self.logger.info(
                        f"✓ Acquire company: {acquire_result.get('total_followers', 0)} followers fetched")
                except Exception as e:
                    self.logger.error(
                        f"✗ Error fetching acquire company followers: {e}")
                    results['acquire_company']['error'] = str(e)

            return results

        except Exception as e:
            self.logger.error(f"Error in Step 1: {e}")
            return {'status': 'failed', 'error': str(e)}

    def step_2_gpt_analysis(self, deal_id: str, deal_data: Dict[str, Any], twitter_handles: Dict[str, str]) -> Dict[str, Any]:
        """
        Step 2: GPT Analysis - Analyze followers with descriptions

        Args:
            deal_id: Deal ID
            deal_data: Deal data
            twitter_handles: Twitter handles dictionary

        Returns:
            Dictionary with step results
        """
        if not self.config['enable_step_2_gpt_analysis']:
            self.logger.info("Step 2 (GPT Analysis) is disabled")
            return {'status': 'skipped', 'message': 'Step disabled in config'}

        self.logger.info(
            "=== Step 2: GPT Analysis - Analyzing Followers with Descriptions ===")

        try:
            processor = self.get_gpt_processor()

            # Get target and acquire handles
            target_name = deal_data.get('target_name')
            acquire_name = deal_data.get('acquire_name')
            target_handle = twitter_handles.get(target_name)
            acquire_handle = twitter_handles.get(acquire_name)

            results = {
                'target_company': {'status': 'failed', 'high_value_followers': 0},
                'acquire_company': {'status': 'failed', 'high_value_followers': 0}
            }

            # Process target company followers
            if target_handle:
                self.logger.info(
                    f"Processing target company followers: @{target_handle}")
                try:
                    target_result = processor.process_company_followers(
                        deal_id, target_name, target_handle
                    )
                    results['target_company'] = {
                        'status': 'success',
                        'high_value_followers': target_result.get('high_value_followers', 0),
                        'total_analyzed': target_result.get('analyzed_followers', 0),
                        'handle': target_handle
                    }
                    self.logger.info(
                        f"✓ Target company: {target_result.get('high_value_followers', 0)} high-value followers found")
                except Exception as e:
                    self.logger.error(
                        f"✗ Error processing target company: {e}")
                    results['target_company']['error'] = str(e)

            # Process acquire company followers
            if acquire_handle:
                self.logger.info(
                    f"Processing acquire company followers: @{acquire_handle}")
                try:
                    acquire_result = processor.process_company_followers(
                        deal_id, acquire_name, acquire_handle
                    )
                    results['acquire_company'] = {
                        'status': 'success',
                        'high_value_followers': acquire_result.get('high_value_followers', 0),
                        'total_analyzed': acquire_result.get('analyzed_followers', 0),
                        'handle': acquire_handle
                    }
                    self.logger.info(
                        f"✓ Acquire company: {acquire_result.get('high_value_followers', 0)} high-value followers found")
                except Exception as e:
                    self.logger.error(
                        f"✗ Error processing acquire company: {e}")
                    results['acquire_company']['error'] = str(e)

            return results

        except Exception as e:
            self.logger.error(f"Error in Step 2: {e}")
            return {'status': 'failed', 'error': str(e)}

    def step_3_tweet_search(self, deal_id: str, deal_data: Dict[str, Any], twitter_handles: Dict[str, str]) -> Dict[str, Any]:
        """
        Step 3: Tweet Search - Analyze followers without descriptions

        Args:
            deal_id: Deal ID
            deal_data: Deal data
            twitter_handles: Twitter handles dictionary

        Returns:
            Dictionary with step results
        """
        if not self.config['enable_step_3_tweet_search']:
            self.logger.info("Step 3 (Tweet Search) is disabled")
            return {'status': 'skipped', 'message': 'Step disabled in config'}

        self.logger.info(
            "=== Step 3: Tweet Search - Analyzing Followers without Descriptions ===")

        try:
            processor = self.get_tweet_search_processor()

            # Get target and acquire handles
            target_name = deal_data.get('target_name')
            acquire_name = deal_data.get('acquire_name')
            target_handle = twitter_handles.get(target_name)
            acquire_handle = twitter_handles.get(acquire_name)

            # Fetch company products from database
            self.logger.info("Fetching company products from database...")
            products_data = processor.fetch_company_products(deal_id)
            self.logger.info(
                f"Found products for companies: {list(products_data.keys())}")

            results = {
                'target_company': {'status': 'failed', 'followers_with_tweets': 0},
                'acquire_company': {'status': 'failed', 'followers_with_tweets': 0}
            }

            # Process target company followers
            if target_handle:
                self.logger.info(
                    f"Processing target company followers: @{target_handle}")
                try:
                    # Get products for target company from database
                    target_products = products_data.get(target_name, [])
                    self.logger.info(
                        f"Target company products: {target_products}")

                    target_result = processor.process_company_followers(
                        deal_id, target_name, target_handle, target_products
                    )
                    results['target_company'] = {
                        'status': 'success',
                        'followers_with_tweets': target_result.get('followers_with_tweets', 0),
                        'total_analyzed': target_result.get('filtered_followers', 0),
                        'handle': target_handle,
                        'products_count': len(target_products)
                    }
                    self.logger.info(
                        f"✓ Target company: {target_result.get('followers_with_tweets', 0)} followers with tweets found")
                except Exception as e:
                    self.logger.error(
                        f"✗ Error processing target company: {e}")
                    results['target_company']['error'] = str(e)

            # Process acquire company followers
            if acquire_handle:
                self.logger.info(
                    f"Processing acquire company followers: @{acquire_handle}")
                try:
                    # Get products for acquire company from database
                    acquire_products = products_data.get(acquire_name, [])
                    self.logger.info(
                        f"Acquire company products: {acquire_products}")

                    acquire_result = processor.process_company_followers(
                        deal_id, acquire_name, acquire_handle, acquire_products
                    )
                    results['acquire_company'] = {
                        'status': 'success',
                        'followers_with_tweets': acquire_result.get('followers_with_tweets', 0),
                        'total_analyzed': acquire_result.get('filtered_followers', 0),
                        'handle': acquire_handle,
                        'products_count': len(acquire_products)
                    }
                    self.logger.info(
                        f"✓ Acquire company: {acquire_result.get('followers_with_tweets', 0)} followers with tweets found")
                except Exception as e:
                    self.logger.error(
                        f"✗ Error processing acquire company: {e}")
                    results['acquire_company']['error'] = str(e)

            return results

        except Exception as e:
            self.logger.error(f"Error in Step 3: {e}")
            return {'status': 'failed', 'error': str(e)}

    def step_4_tweet_analysis(self, deal_id: str, deal_data: Dict[str, Any], twitter_handles: Dict[str, str]) -> Dict[str, Any]:
        """
        Step 4: Tweet Analysis - Analyze tweets from high-value followers

        Args:
            deal_id: Deal ID
            deal_data: Deal data
            twitter_handles: Twitter handles dictionary

        Returns:
            Dictionary with step results
        """
        if not self.config['enable_step_4_tweet_analysis']:
            self.logger.info("Step 4 (Tweet Analysis) is disabled")
            return {'status': 'skipped', 'message': 'Step disabled in config'}

        self.logger.info(
            "=== Step 4: Tweet Analysis - Analyzing Tweets from High-Value Followers ===")

        try:
            analyzer = self.get_tweet_analyzer()

            # Call the main process_deal method which handles both companies and creates JSON files
            result_file = analyzer.process_deal(deal_id)

            if result_file:
                self.logger.info(
                    f"✓ Step 4 completed successfully. Results saved to: {result_file}")
                return {
                    'status': 'success',
                    'result_file': result_file,
                    'message': 'Tweet analysis completed and JSON files created'
                }
            else:
                self.logger.warning(
                    "✗ Step 4 completed but no result file was created")
                return {
                    'status': 'completed_no_results',
                    'message': 'Tweet analysis completed but no results found'
                }

        except Exception as e:
            self.logger.error(f"Error in Step 4: {e}")
            return {'status': 'failed', 'error': str(e)}

    def get_final_summary(self, deal_id: str, step_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate final summary of all steps

        Args:
            deal_id: Deal ID
            step_results: Results from all steps

        Returns:
            Final summary dictionary
        """
        # Count total high-value followers from database
        total_high_value_followers = HighValueFollowers.objects.filter(
            deal_id=deal_id).count()

        # Count by approach
        gpt_followers = HighValueFollowers.objects.filter(
            deal_id=deal_id,
            gpt_model_used__ne="tweet_search"
        ).count()

        tweet_search_followers = HighValueFollowers.objects.filter(
            deal_id=deal_id,
            gpt_model_used="tweet_search"
        ).count()

        summary = {
            'deal_id': deal_id,
            'processing_timestamp': datetime.now().isoformat(),
            'total_high_value_followers': total_high_value_followers,
            'breakdown_by_approach': {
                'gpt_analysis': gpt_followers,
                'tweet_search': tweet_search_followers
            },
            'step_results': step_results,
            'config_used': self.config
        }

        return summary

    def save_final_summary(self, summary: Dict[str, Any]) -> str:
        """
        Save final summary to JSON file

        Args:
            summary: Final summary dictionary

        Returns:
            Path to saved file
        """
        deal_id = summary['deal_id']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"orchestrator_summary_{deal_id}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Final summary saved to: {filepath}")
        return filepath

    def process_deal(self, deal_id: str) -> Optional[str]:
        """
        Main method to process high-value followers using all approaches

        Args:
            deal_id: Deal ID to process

        Returns:
            Path to results file or None if failed
        """
        self.logger.info(f"=== Starting High Value Followers Orchestrator ===")
        self.logger.info(f"Deal ID: {deal_id}")
        self.logger.info(f"Configuration:")
        self.logger.info(
            f"  Step 1 (Gun Shot): {'✓ Enabled' if self.config['enable_step_1_gun_shot'] else '✗ Disabled'}")
        self.logger.info(
            f"  Step 2 (GPT Analysis): {'✓ Enabled' if self.config['enable_step_2_gpt_analysis'] else '✗ Disabled'}")
        self.logger.info(
            f"  Step 3 (Tweet Search): {'✓ Enabled' if self.config['enable_step_3_tweet_search'] else '✗ Disabled'}")
        self.logger.info(
            f"  Step 4 (Tweet Analysis): {'✓ Enabled' if self.config['enable_step_4_tweet_analysis'] else '✗ Disabled'}")

        # Step 0: Fetch deal data and extract Twitter handles
        self.logger.info("=== Step 0: Fetching Deal Data ===")
        deal_data = self.fetch_deal_data(deal_id)
        if not deal_data:
            self.logger.error("Failed to fetch deal data")
            return None

        twitter_handles = self.extract_twitter_handles(deal_data)
        if len(twitter_handles) < 2:
            self.logger.error(
                f"Not enough Twitter handles found: {twitter_handles}")
            return None

        self.logger.info(f"Found Twitter handles: {twitter_handles}")

        # Initialize step results
        step_results = {}

        # Step 1: Gun Shot Approach
        step_results['step_1_gun_shot'] = self.step_1_gun_shot_approach(
            deal_id, deal_data, twitter_handles)

        # Step 2: GPT Analysis
        step_results['step_2_gpt_analysis'] = self.step_2_gpt_analysis(
            deal_id, deal_data, twitter_handles)

        # Step 3: Tweet Search
        step_results['step_3_tweet_search'] = self.step_3_tweet_search(
            deal_id, deal_data, twitter_handles)

        # Step 4: Tweet Analysis
        step_results['step_4_tweet_analysis'] = self.step_4_tweet_analysis(
            deal_id, deal_data, twitter_handles)

        # Generate final summary
        final_summary = self.get_final_summary(deal_id, step_results)

        # Save final summary
        summary_filepath = self.save_final_summary(final_summary)

        # Log final results
        self.logger.info("=== Orchestrator Processing Complete ===")
        self.logger.info(
            f"Total high-value followers found: {final_summary['total_high_value_followers']}")
        self.logger.info(
            f"  - GPT Analysis: {final_summary['breakdown_by_approach']['gpt_analysis']}")
        self.logger.info(
            f"  - Tweet Search: {final_summary['breakdown_by_approach']['tweet_search']}")
        self.logger.info(f"Results saved to: {summary_filepath}")

        return summary_filepath

    def process_deal_step_by_step(self, deal_id: str) -> Dict[str, Any]:
        """
        Process deal step by step with detailed results for each step

        Args:
            deal_id: Deal ID to process

        Returns:
            Dictionary with detailed results for each step
        """
        self.logger.info(f"=== Starting Step-by-Step Processing ===")
        self.logger.info(f"Deal ID: {deal_id}")

        # Step 0: Fetch deal data and extract Twitter handles
        self.logger.info("=== Step 0: Fetching Deal Data ===")
        deal_data = self.fetch_deal_data(deal_id)
        if not deal_data:
            return {'error': 'Failed to fetch deal data'}

        twitter_handles = self.extract_twitter_handles(deal_data)
        if len(twitter_handles) < 2:
            return {'error': f'Not enough Twitter handles found: {twitter_handles}'}

        self.logger.info(f"Found Twitter handles: {twitter_handles}")

        # Initialize step results
        step_results = {}

        # Step 1: Gun Shot Approach
        step_results['step_1_gun_shot'] = self.step_1_gun_shot_approach(
            deal_id, deal_data, twitter_handles)

        # Step 2: GPT Analysis
        step_results['step_2_gpt_analysis'] = self.step_2_gpt_analysis(
            deal_id, deal_data, twitter_handles)

        # Step 3: Tweet Search
        step_results['step_3_tweet_search'] = self.step_3_tweet_search(
            deal_id, deal_data, twitter_handles)

        # Step 4: Tweet Analysis
        step_results['step_4_tweet_analysis'] = self.step_4_tweet_analysis(
            deal_id, deal_data, twitter_handles)

        # Generate final summary
        final_summary = self.get_final_summary(deal_id, step_results)

        return {
            'deal_data': deal_data,
            'twitter_handles': twitter_handles,
            'step_results': step_results,
            'final_summary': final_summary
        }


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
            "Usage: python high_value_followers_orchestrator.py <deal_id> [options]")
        logger.error("Options:")
        logger.error("  --disable-step-1          Disable Gun Shot Approach")
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
        logger.error("Examples:")
        logger.error(
            "  python high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb")
        logger.error(
            "  python high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb --disable-step-2")
        logger.error(
            "  python high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb --gpt-workers 100 --tweet-workers 20")
        sys.exit(1)

    deal_id = sys.argv[1]

    # Parse optional arguments
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
        else:
            i += 1

    try:
        # Initialize orchestrator with config overrides
        orchestrator = HighValueFollowersOrchestrator(
            config_overrides=config_overrides)

        # Run processing
        result_file = orchestrator.process_deal(deal_id)

        if result_file:
            logger.info("✅ Orchestrator processing completed successfully!")
            logger.info(f"Results saved to: {result_file}")
        else:
            logger.error("❌ Processing failed or no results found")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
