#!/usr/bin/env python3
"""
High Value Followers Processor
Combines follower fetching, filtering, and GPT analysis for deal-based follower processing.

Usage:
# python high_value_followers_processor.py 68ac4a254a6006a0946ec3bb
"""

from document_processor.twitter_utils.twitter_cleanup_utils import TwitterCleanupUtils
from document_processor.twitter_utils.follower_utils import FollowerUtils
from document_processor.models import ProcessingJob, Followers, HighValueFollowers
import django
import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI
from dotenv import load_dotenv
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()

# Import after Django setup

# Load environment variables
load_dotenv()


class HighValueFollowersProcessor:
    """Main class for processing high-value followers from deal data"""

    def __init__(self, openai_api_key: Optional[str] = None, twitter_api_key: Optional[str] = None, config_overrides: Dict[str, Any] = None):
        """
        Initialize the processor

        Args:
            openai_api_key: OpenAI API key
            twitter_api_key: Twitter API key (optional, for fetching if needed)
            config_overrides: Optional configuration overrides
        """
        # Setup configuration
        self.config = {
            # Follower filtering criteria
            'min_followers_count': 250,
            'min_statuses_count': 250,

            # Processing limits
            'max_followers_per_company': 10,  # None = process all followers

            # GPT analysis settings
            'min_overall_score': 0,
            'gpt_model': 'gpt-4.1-mini',
            'gpt_max_tokens': 1000,
            'gpt_temperature': 0.1,

            # Rate limiting
            'delay_between_requests': 0,  # seconds
            'delay_every_n_requests': 0,
            'delay_for_rate_limit': 0,  # seconds

            # Concurrency settings
            # Number of parallel GPT calls (1 = sequential)
            'max_workers': 70,
            'use_parallel_processing': True,  # Enable/disable parallel processing

            # Processing approach
            'approach': 'GUNSHOT',

            # Output settings
            'save_results_summary': True,
            'save_individual_records': True,

            # Batch processing for MongoDB
            'batch_size': 1000,  # Save followers in batches of this size
            'use_batch_saving': True  # Enable batch saving for better performance
        }

        # Apply any config overrides
        if config_overrides:
            self.config.update(config_overrides)

        # Setup OpenAI client
        self.openai_api_key = openai_api_key or os.getenv(
            'OPENAI_API_KEY_SEC_FILING')
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY_SEC_FILING environment variable.")

        self.client = OpenAI(api_key=self.openai_api_key)

        # Setup Twitter API key (optional)
        self.twitter_api_key = twitter_api_key or os.getenv('TWITTER_API_KEY')

        # Setup utilities
        self.follower_utils = FollowerUtils()
        self.cleanup_utils = TwitterCleanupUtils()

        # Setup logger
        self.logger = logging.getLogger(__name__)

        # Setup output directory
        self.output_dir = os.path.join(
            os.path.dirname(__file__),
            'high_value_followers_results'
        )
        os.makedirs(self.output_dir, exist_ok=True)

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
                # Remove @ symbol if present
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
                # Remove @ symbol if present
                acquire_handle = acquire_handle.lstrip('@')
                twitter_handles[acquire_name] = acquire_handle
                self.logger.info(
                    f"Found acquire Twitter handle: @{acquire_handle}")

        return twitter_handles

    def filter_followers(self, followers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter followers based on criteria from config (adapted from filter_followers.py)

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

            # Check if description exists and is not empty (after stripping whitespace)
            has_description = description and description.strip()

            if (followers_count >= min_followers_count and
                    statuses_count >= min_statuses_count and
                        not protected and has_description
                    ):
                filtered_followers.append(follower)

        self.logger.info(
            f"Filtered {original_count} followers -> {len(filtered_followers)} followers")
        self.logger.info(
            f"Criteria: followers_count >= {min_followers_count} AND statuses_count >= {min_statuses_count} AND has_description")
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

    def create_analysis_prompt(self, follower: Dict[str, Any], company_name: str) -> str:
        """Create a detailed prompt for GPT analysis (adapted from gpt_follower_analyzer.py)"""
        description = follower.get('description', 'No description available')

        prompt = f"""
This account follows {company_name} on Twitter. Score their intelligence value from 0-10 based on their likelihood to accidentally reveal competitive insights, market dynamics, or antitrust-relevant information. 10 = Must monitor every tweet. 0 = Ignore completely.

USER PROFILE BIO:
{description}

Your task:
1. Provide ONE overall relevance score (0–10) for potential business intelligence purposes.
2. Base your reasoning ONLY on the bio (no external assumptions).
3. Explain briefly how the bio indicates or lacks signals across the parameters.
4. List key indicators (phrases, titles, keywords) from the bio that influenced your assessment.

Format your response strictly as JSON:
{{
    "overall_score": <0-10>,
    "reason": "short explanation of why you gave this score",# in 50 words or less
    "key_indicators": ["<indicator1>", "<indicator2>", "..."]
}}
"""
        self.logger.info(f"Prompt : {prompt}")

        return prompt

    def analyze_follower_with_gpt(self, follower: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze a single follower using GPT"""
        try:
            prompt = self.create_analysis_prompt(
                follower, self.current_company)

            response = self.client.chat.completions.create(
                model=self.config['gpt_model'],
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert analyst specializing in business intelligence, antitrust law, and corporate affairs. Provide accurate, objective assessments based on publicly available profile information."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config['gpt_max_tokens'],
                temperature=self.config['gpt_temperature'],
                response_format={"type": "json_object"}
            )
            self.logger.info(f"GPT response: {response}")

            analysis = json.loads(response.choices[0].message.content)

            # Add follower metadata to analysis
            analysis['follower_id'] = follower.get('id')
            analysis['name'] = follower.get('name')
            analysis['screen_name'] = follower.get('screen_name')
            analysis['description'] = follower.get('description')
            analysis['location'] = follower.get('location')
            analysis['followers_count'] = follower.get('followers_count')
            analysis['statuses_count'] = follower.get('statuses_count')
            analysis['protected'] = follower.get('protected', False)
            analysis['verified'] = follower.get('verified', False)
            analysis['created_at_twitter'] = follower.get('created_at')
            analysis['analyzed_at'] = datetime.now().isoformat()
            analysis['token_usage'] = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }

            self.logger.info(
                f"Successfully analyzed @{follower.get('screen_name')} - Overall Score: {analysis.get('overall_score', 'N/A')}")
            return analysis

        except Exception as e:
            self.logger.error(
                f"Error analyzing follower @{follower.get('screen_name', 'unknown')}: {str(e)}")
            return None

    def analyze_follower_parallel(self, follower_data: tuple) -> Optional[Dict[str, Any]]:
        """Analyze a single follower for parallel processing"""
        follower, index, total = follower_data
        try:
            self.logger.info(
                f"Worker analyzing follower {index}/{total}: @{follower.get('screen_name', 'unknown')}")

            analysis = self.analyze_follower_with_gpt(follower)
            return analysis

        except Exception as e:
            self.logger.error(
                f"Worker error analyzing follower {index}: {str(e)}")
            return None

    def save_high_value_follower(self, analysis: Dict[str, Any], deal_id: str, company_name: str, company_handle: str) -> Optional[str]:
        """
        Save a single high-value follower to MongoDB

        Args:
            analysis: GPT analysis result with follower data
            deal_id: Deal ID
            company_name: Company name
            company_handle: Company Twitter handle

        Returns:
            ID of saved record or None if failed
        """
        try:
            # Create HighValueFollowers object
            high_value_follower = HighValueFollowers(
                deal_id=deal_id,
                company_name=company_name,
                company_handle=company_handle,

                # Follower information
                follower_id=str(analysis.get('follower_id', '')),
                name=analysis.get('name', ''),
                screen_name=analysis.get('screen_name', ''),
                description=analysis.get('description', ''),
                location=analysis.get('location', ''),
                followers_count=analysis.get('followers_count', 0),
                statuses_count=analysis.get('statuses_count', 0),
                protected=analysis.get('protected', False),
                verified=analysis.get('verified', False),
                created_at_twitter=analysis.get('created_at_twitter'),

                # GPT Analysis results
                overall_score=analysis.get('overall_score', 0),
                reason=analysis.get('reason', ''),
                key_indicators=analysis.get('key_indicators', []),
                analysis_timestamp=datetime.now(),
                gpt_model_used=self.config['gpt_model'],

                # Processing metadata
                processing_status='completed',
                approach=self.config['approach']
            )

            high_value_follower.save()

            self.logger.info(
                f"Saved high-value follower: @{analysis.get('screen_name')} (Score: {analysis.get('overall_score')})")
            return str(high_value_follower.id)

        except Exception as e:
            self.logger.error(f"Error saving high-value follower: {e}")
            return None

    def save_high_value_followers_batch(self, analyses: List[Dict[str, Any]], deal_id: str, company_name: str, company_handle: str) -> List[str]:
        """
        Save multiple high-value followers to MongoDB in a batch for better performance

        Args:
            analyses: List of GPT analysis results with follower data
            deal_id: Deal ID
            company_name: Company name
            company_handle: Company Twitter handle

        Returns:
            List of saved record IDs
        """
        try:
            high_value_followers = []

            for analysis in analyses:
                # Create HighValueFollowers object
                high_value_follower = HighValueFollowers(
                    deal_id=deal_id,
                    company_name=company_name,
                    company_handle=company_handle,

                    # Follower information
                    follower_id=str(analysis.get('follower_id', '')),
                    name=analysis.get('name', ''),
                    screen_name=analysis.get('screen_name', ''),
                    description=analysis.get('description', ''),
                    location=analysis.get('location', ''),
                    followers_count=analysis.get('followers_count', 0),
                    statuses_count=analysis.get('statuses_count', 0),
                    protected=analysis.get('protected', False),
                    verified=analysis.get('verified', False),
                    created_at_twitter=analysis.get('created_at_twitter'),

                    # GPT Analysis results
                    overall_score=analysis.get('overall_score', 0),
                    reason=analysis.get('reason', ''),
                    key_indicators=analysis.get('key_indicators', []),
                    analysis_timestamp=datetime.now(),
                    gpt_model_used=self.config['gpt_model'],

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

    def process_company_followers(self, deal_id: str, company_name: str, company_handle: str) -> Dict[str, Any]:
        """
        Process followers for a specific company using config settings

        Args:
            deal_id: Deal ID
            company_name: Company name
            company_handle: Company Twitter handle

        Returns:
            Dictionary with processing results
        """
        # Set current company for filename generation
        self.current_company = company_name

        self.logger.info(
            f"Processing followers for {company_name} (@{company_handle})")

        # Step 1: Get followers from database
        followers = self.follower_utils.get_followers_for_company(
            deal_id, company_handle, self.config['approach'])

        if not followers:
            self.logger.warning(f"No followers found for @{company_handle}")
            return {
                'company_name': company_name,
                'company_handle': company_handle,
                'total_followers': 0,
                'filtered_followers': 0,
                'analyzed_followers': 0,
                'high_value_followers': 0,
                'processing_status': 'failed'
            }

        self.logger.info(
            f"Retrieved {len(followers)} followers for @{company_handle}")

        # Step 2: Filter followers
        filtered_followers = self.filter_followers(followers)

        # Step 3: Apply follower limit if configured
        max_followers = self.config['max_followers_per_company']
        if max_followers is not None and max_followers > 0:
            original_count = len(filtered_followers)
            filtered_followers = filtered_followers[:max_followers]
            self.logger.info(
                f"Limited to {max_followers} followers for processing (from {original_count} filtered followers)")
        else:
            self.logger.info(
                f"Processing all {len(filtered_followers)} filtered followers")

        # Step 4: Analyze followers with GPT
        analyzed_count = 0
        high_value_count = 0
        saved_ids = []
        total_tokens = 0
        output_token = 0
        input_token = 0

        # Check if parallel processing is enabled
        use_parallel = self.config['use_parallel_processing']
        max_workers = self.config['max_workers']

        if use_parallel and max_workers > 1:
            self.logger.info(
                f"Using parallel processing with {max_workers} workers")
            analyses = self._analyze_followers_parallel(
                filtered_followers, max_workers)
        else:
            self.logger.info("Using sequential processing")
            analyses = self._analyze_followers_sequential(filtered_followers)

        # Process all analyses
        high_value_analyses = []

        for analysis in analyses:
            if analysis:
                analyzed_count += 1
                # Add total tokens from this analysis
                token_usage_data = analysis.get('token_usage', {})
                if isinstance(token_usage_data, dict):
                    total_tokens += token_usage_data.get('total_tokens', 0)
                    output_token += token_usage_data.get(
                        'completion_tokens', 0)
                    input_token += token_usage_data.get('prompt_tokens', 0)

                # Check if score meets minimum threshold
                min_overall_score = self.config['min_overall_score']
                if analysis.get('overall_score', 0) >= min_overall_score:
                    high_value_analyses.append(analysis)
                    self.logger.info(
                        f"✓ High-value follower found (Score: {analysis.get('overall_score')})")
                else:
                    self.logger.info(
                        f"✗ Filtered out (Score: {analysis.get('overall_score')} < {min_overall_score})")

        # Save high-value followers (batch or individual)
        if high_value_analyses:
            if self.config.get('use_batch_saving', False):
                # Batch save for better performance
                batch_size = self.config.get('batch_size', 10)
                for i in range(0, len(high_value_analyses), batch_size):
                    batch = high_value_analyses[i:i + batch_size]
                    batch_saved_ids = self.save_high_value_followers_batch(
                        batch, deal_id, company_name, company_handle)
                    saved_ids.extend(batch_saved_ids)
                    high_value_count += len(batch_saved_ids)
                    self.logger.info(
                        f"Batch saved {len(batch_saved_ids)} followers")
            else:
                # Individual save (original method)
                for analysis in high_value_analyses:
                    saved_id = self.save_high_value_follower(
                        analysis, deal_id, company_name, company_handle)
                    if saved_id:
                        high_value_count += 1
                        saved_ids.append(saved_id)
                    else:
                        self.logger.warning(
                            f"✗ Failed to save high-value follower")

        return {
            'company_name': company_name,
            'company_handle': company_handle,
            'total_followers': len(followers),
            'filtered_followers': len(filtered_followers),
            'analyzed_followers': analyzed_count,
            'high_value_followers': high_value_count,
            'total_tokens_used': total_tokens,
            'output_token': output_token,
            'input_token': input_token,
            'processing_status': 'completed'
        }

    def _analyze_followers_sequential(self, filtered_followers: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """Analyze followers sequentially with rate limiting"""
        analyses = []

        for i, follower in enumerate(filtered_followers, 1):
            try:
                self.logger.info(
                    f"Analyzing follower {i}/{len(filtered_followers)}: @{follower.get('screen_name', 'unknown')}")

                # Analyze with GPT
                analysis = self.analyze_follower_with_gpt(follower)
                analyses.append(analysis)

                # Rate limiting - pause between requests
                delay_every_n = self.config['delay_every_n_requests']
                delay_between = self.config['delay_between_requests']
                delay_rate_limit = self.config['delay_for_rate_limit']

                # Only apply rate limiting if delay_every_n is greater than 0
                if delay_every_n > 0 and i % delay_every_n == 0:
                    self.logger.info(
                        f"Processed {i} followers, pausing for rate limiting...")
                    time.sleep(delay_rate_limit)
                elif delay_between > 0:
                    time.sleep(delay_between)

            except KeyboardInterrupt:
                self.logger.info("Processing interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Error processing follower {i}: {str(e)}")
                analyses.append(None)
                continue

        return analyses

    def _analyze_followers_parallel(self, filtered_followers: List[Dict[str, Any]], max_workers: int) -> List[Optional[Dict[str, Any]]]:
        """Analyze followers in parallel using ThreadPoolExecutor"""
        analyses = [None] * \
            len(filtered_followers)  # Pre-allocate result array

        # Prepare data for parallel processing
        follower_data = [(follower, i+1, len(filtered_followers))
                         for i, follower in enumerate(filtered_followers)]

        self.logger.info(f"followers_Data : {follower_data[0]}")

        self.logger.info(
            f"Starting parallel analysis with {max_workers} workers for {len(filtered_followers)} followers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and track their positions
            future_to_position = {}
            for i, data in enumerate(follower_data):
                future = executor.submit(self.analyze_follower_parallel, data)
                future_to_position[future] = i

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_position):
                position = future_to_position[future]
                try:
                    analysis = future.result()
                    analyses[position] = analysis
                except Exception as e:
                    self.logger.error(
                        f"Worker error at position {position}: {str(e)}")
                    analyses[position] = None

        self.logger.info(
            f"Parallel analysis completed. Processed {len([a for a in analyses if a is not None])} followers")
        return analyses

    def save_results_summary(self, deal_id: str, all_results: List[Dict[str, Any]]) -> str:
        """
        Save processing results summary to JSON file

        Args:
            deal_id: Deal ID
            all_results: List of processing results for all companies

        Returns:
            Path to saved file
        """
        filename = f"high_value_followers_summary_{deal_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Calculate totals
        total_followers = sum(r.get('total_followers', 0) for r in all_results)
        total_filtered = sum(r.get('filtered_followers', 0)
                             for r in all_results)
        total_analyzed = sum(r.get('analyzed_followers', 0)
                             for r in all_results)
        total_high_value = sum(r.get('high_value_followers', 0)
                               for r in all_results)

        output_data = {
            'deal_id': deal_id,
            'processing_timestamp': datetime.now().isoformat(),
            'approach': 'HIGH_VALUE_FOLLOWERS',
            'total_companies': len(all_results),
            'summary': {
                'total_followers': total_followers,
                'total_filtered_followers': total_filtered,
                'total_analyzed_followers': total_analyzed,
                'total_high_value_followers': total_high_value
            },
            'company_results': all_results
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results summary saved to: {filepath}")
        return filepath

    def save_high_value_followers_json(self, deal_id: str, all_results: List[Dict[str, Any]]) -> str:
        """
        Save high-value followers data to JSON file for local analysis

        Args:
            deal_id: Deal ID
            all_results: List of processing results for all companies

        Returns:
            Path to saved file
        """
        filename = f"high_value_followers_{deal_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Collect all high-value followers from database
        from document_processor.models import HighValueFollowers

        high_value_followers = HighValueFollowers.objects.filter(
            deal_id=deal_id)

        # Convert to list of dictionaries
        followers_data = []
        for follower in high_value_followers:
            follower_dict = {
                'deal_id': follower.deal_id,
                'company_name': follower.company_name,
                'company_handle': follower.company_handle,

                # Follower information
                'follower_id': follower.follower_id,
                'name': follower.name,
                'screen_name': follower.screen_name,
                'description': follower.description,
                'location': follower.location,
                'followers_count': follower.followers_count,
                'statuses_count': follower.statuses_count,
                'protected': follower.protected,
                'verified': follower.verified,
                'created_at_twitter': follower.created_at_twitter.isoformat() if follower.created_at_twitter else None,

                # GPT Analysis results
                'overall_score': follower.overall_score,
                'key_indicators': follower.key_indicators,
                'analysis_timestamp': follower.analysis_timestamp.isoformat(),
                'gpt_model_used': follower.gpt_model_used,

                # Processing metadata
                'processing_status': follower.processing_status,
                'approach': follower.approach,

                # Timestamps
                'created_at': follower.created_at.isoformat(),
                'updated_at': follower.updated_at.isoformat()
            }
            followers_data.append(follower_dict)

        # Calculate totals
        total_followers = sum(r.get('total_followers', 0) for r in all_results)
        total_filtered = sum(r.get('filtered_followers', 0)
                             for r in all_results)
        total_analyzed = sum(r.get('analyzed_followers', 0)
                             for r in all_results)
        total_high_value = sum(r.get('high_value_followers', 0)
                               for r in all_results)

        # Create comprehensive JSON structure
        json_data = {
            'deal_id': deal_id,
            'export_timestamp': datetime.now().isoformat(),
            'approach': 'HIGH_VALUE_FOLLOWERS',
            'gpt_model_used': self.config['gpt_model'],
            'filter_criteria': {
                'min_followers_count': self.config['min_followers_count'],
                'min_statuses_count': self.config['min_statuses_count'],
                'min_overall_score': self.config['min_overall_score']
            },
            'processing_summary': {
                'total_companies': len(all_results),
                'total_followers': total_followers,
                'total_filtered_followers': total_filtered,
                'total_analyzed_followers': total_analyzed,
                'total_high_value_followers': total_high_value
            },
            'company_results': all_results,
            'high_value_followers': followers_data
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"High-value followers JSON saved to: {filepath}")
        self.logger.info(
            f"Exported {len(followers_data)} high-value followers")
        return filepath

    def process_deal(self, deal_id: str) -> Optional[str]:
        """
        Main method to process high-value followers for a deal using config settings

        Args:
            deal_id: Deal ID to process

        Returns:
            Path to results file or None if failed
        """
        self.logger.info(
            f"Starting High Value Followers processing for deal ID: {deal_id}")

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

        self.logger.info(
            f"Found {len(twitter_handles)} Twitter handles: {twitter_handles}")

        # Step 3: Get target and acquire handles
        target_name = deal_data.get('target_name')
        acquire_name = deal_data.get('acquire_name')

        target_handle = twitter_handles.get(target_name)
        acquire_handle = twitter_handles.get(acquire_name)

        if not target_handle or not acquire_handle:
            self.logger.error(
                f"Missing Twitter handles for target or acquire company")
            return None

        # Step 4: Process followers for each company
        all_results = []

        # Process target company followers
        try:
            target_result = self.process_company_followers(
                deal_id, target_name, target_handle
            )
            all_results.append(target_result)
            self.logger.info(
                f"Completed processing for target company @{target_handle}")

        except Exception as e:
            self.logger.error(
                f"Error processing target company @{target_handle}: {e}")

        # Process acquire company followers
        try:
            acquire_result = self.process_company_followers(
                deal_id, acquire_name, acquire_handle
            )
            all_results.append(acquire_result)
            self.logger.info(
                f"Completed processing for acquire company @{acquire_handle}")

        except Exception as e:
            self.logger.error(
                f"Error processing acquire company @{acquire_handle}: {e}")

        # Step 5: Save results summary and high-value followers JSON
        if all_results:
            summary_filepath = self.save_results_summary(deal_id, all_results)
            json_filepath = self.save_high_value_followers_json(
                deal_id, all_results)

            # Log summary
            total_high_value = sum(r.get('high_value_followers', 0)
                                   for r in all_results)
            self.logger.info(
                "=== High Value Followers Processing Complete ===")
            self.logger.info(f"Deal ID: {deal_id}")
            self.logger.info(f"Total companies processed: {len(all_results)}")
            self.logger.info(
                f"Total high-value followers found: {total_high_value}")
            self.logger.info(f"Summary saved to: {summary_filepath}")
            self.logger.info(
                f"High-value followers JSON saved to: {json_filepath}")

            return json_filepath
        else:
            self.logger.warning("No results to save")
            return None


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
            "Usage: python high_value_followers_processor.py <deal_id> [options]")
        logger.error("Options:")
        logger.error(
            "  --workers <number>           Number of parallel workers (default: 50)")
        logger.error(
            "  --batch-size <number>        Batch size for MongoDB saves (default: 10)")
        logger.error(
            "  --max-followers <number>     Maximum followers per company (default: 10)")
        logger.error(
            "  --min-score <number>         Minimum GPT score to save (default: 0)")
        logger.error("Examples:")
        logger.error(
            "  python high_value_followers_processor.py 68184d52478abf06ec1a28ec")
        logger.error(
            "  python high_value_followers_processor.py 68184d52478abf06ec1a28ec --workers 100")
        logger.error(
            "  python high_value_followers_processor.py 68184d52478abf06ec1a28ec --batch-size 20")
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
        elif sys.argv[i] == '--min-score' and i + 1 < len(sys.argv):
            min_score = int(sys.argv[i + 1])
            config_overrides['min_overall_score'] = min_score
            i += 2
        else:
            i += 1

    try:
        # Initialize processor with config overrides
        processor = HighValueFollowersProcessor(
            config_overrides=config_overrides)

        # Run processing
        result_file = processor.process_deal(deal_id)

        if result_file:
            logger.info(
                "High Value Followers processing completed successfully!")
            logger.info(f"Results saved to: {result_file}")
        else:
            logger.error("Processing failed or no results found")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
