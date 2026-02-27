#!/usr/bin/env python3
"""
Twitter Followers Script for Deal Analysis - Gun Shot Approach
This script fetches Twitter followers using official Twitter handles from the deal.

Usage:
# python gun_shot_approach.py 682f00def21b9fca8e1d04fe
"""

import django
import os
import sys
import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import time
import logging
from document_processor.twitter_utils.twitter_cleanup_utils import TwitterCleanupUtils
from document_processor.models import ProcessingJob, SearchQuery, Tweet, Followers, FollowersMetadata
from document_processor.twitter_utils.high_value_followers_processor import HighValueFollowersProcessor

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()

# Import after Django setup


class GunShotFollowersService:
    """Service for fetching Twitter followers using official Twitter handles"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Twitter followers service

        Args:
            api_key: Twitter API key from twitterapi.io
        """
        self.api_key = api_key or os.getenv('TWITTER_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Twitter API key is required. Set TWITTER_API_KEY environment variable.")

        self.base_url = "https://api.twitterapi.io/twitter/user/followers"
        self.headers = {
            'X-API-Key': self.api_key,
            'Content-Type': 'application/json'
        }

        # Setup logger
        self.logger = logging.getLogger(__name__)

        # Setup output directory for immediate JSON saves
        self.api_calls_dir = os.path.join(
            os.path.dirname(__file__),
            'twitter_search_results',
            'api_calls'
        )
        os.makedirs(self.api_calls_dir, exist_ok=True)

    def save_api_response(self, response_data: Dict[str, Any], username: str, page_number: int, deal_id: str, cursor: str = "") -> str:
        """
        Save individual API response to JSON file immediately

        Args:
            response_data: The API response data
            username: Twitter username
            page_number: Current page number
            deal_id: Deal ID for organization
            cursor: Current cursor value

        Returns:
            Path to the saved JSON file
        """
        try:
            # Create deal-specific directory
            deal_dir = os.path.join(self.api_calls_dir, deal_id)
            os.makedirs(deal_dir, exist_ok=True)

            # Create username-specific subdirectory
            user_dir = os.path.join(deal_dir, username)
            os.makedirs(user_dir, exist_ok=True)

            # Generate filename with timestamp and page info
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[
                :-3]  # Include milliseconds
            filename = f"page_{page_number:03d}_{timestamp}.json"
            filepath = os.path.join(user_dir, filename)

            # Prepare data structure with metadata
            api_call_data = {
                'deal_id': deal_id,
                'username': username,
                'page_number': page_number,
                'cursor': cursor,
                'api_call_timestamp': datetime.now().isoformat(),
                'response_data': response_data,
                'metadata': {
                    'file_created': datetime.now().isoformat(),
                    'file_version': '1.0',
                    'data_source': 'twitterapi.io',
                    'approach': 'GUNSHOT',
                    'page_followers_count': len(response_data.get('followers', [])),
                    'has_next_page': response_data.get('has_next_page', False),
                    'next_cursor': response_data.get('next_cursor', '')
                }
            }

            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(api_call_data, f, indent=2, ensure_ascii=False)

            self.logger.info(
                f"Saved API response for @{username} page {page_number} to: {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Error saving API response to JSON file: {e}")
            return ""

    def fetch_followers(self, username: str, deal_id: str, max_pages: int = 2, page_size: int = 200, delay: float = 1.0, max_retries: int = 3) -> List[Dict[str, Any]]:
        """
        Fetch followers for a given username with pagination and retry logic
        Each API call is immediately saved to JSON for backup

        Args:
            username: Twitter username (without @)
            deal_id: Deal ID for organizing saved files
            max_pages: Maximum number of pages to fetch
            page_size: Number of followers per page (max 200)
            delay: Delay between requests in seconds
            max_retries: Maximum number of retries per page

        Returns:
            List of follower data
        """
        all_followers = []
        cursor = ""  # First page cursor is empty
        page_count = 0
        stop_pagination = False
        saved_files = []  # Track all saved files

        self.logger.info(f"Starting followers fetch for user: @{username}")
        self.logger.info(
            f"Page size: {page_size}, Max pages: {max_pages}, Max retries: {max_retries}")

        while page_count < max_pages and not stop_pagination:
            page_success = False
            retry_count = 0

            # Retry loop for each page
            while not page_success and retry_count <= max_retries:
                try:
                    # Prepare request parameters
                    params = {
                        "userName": username,
                        "pageSize": min(page_size, 200)  # API max is 200
                    }

                    # Add cursor if we have one (not for first page)
                    if cursor:
                        params["cursor"] = cursor

                    if retry_count > 0:
                        self.logger.info(
                            f"Retry {retry_count}/{max_retries}: Fetching page {page_count + 1} for @{username}")
                    else:
                        self.logger.info(
                            f"Fetching page {page_count + 1} for @{username}")

                    # Make API request with timeout
                    response = requests.get(
                        self.base_url,
                        headers=self.headers,
                        params=params,
                        timeout=60
                    )

                    self.logger.info(f"Response: {response.text[:1000]}")

                    # Check if request was successful
                    if response.status_code == 429:  # Rate limited
                        rate_limit_delay = 60
                        self.logger.warning(
                            f"Rate limited! Waiting {rate_limit_delay} seconds...")
                        time.sleep(rate_limit_delay)
                        continue  # Retry the same page

                    elif response.status_code != 200:
                        self.logger.error(
                            f"Error: API returned status code {response.status_code}")
                        self.logger.error(f"Response: {response.text}")

                        # For 5xx errors, retry with exponential backoff
                        if response.status_code >= 500:
                            retry_delay = min(30, (2 ** retry_count) * delay)
                            self.logger.warning(
                                f"Server error, retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_count += 1
                            continue
                        else:
                            # For 4xx errors (client errors), don't retry
                            self.logger.error(f"Client error, not retrying")
                            stop_pagination = True
                            break

                    # Parse JSON response
                    try:
                        data = response.json()
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON decode error: {e}")
                        retry_delay = min(30, (2 ** retry_count) * delay)
                        self.logger.warning(
                            f"JSON decode failed, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_count += 1
                        continue

                    # IMMEDIATELY save this API response to JSON file
                    saved_file = self.save_api_response(
                        data, username, page_count + 1, deal_id, cursor)
                    if saved_file:
                        saved_files.append(saved_file)

                    # Check if we have followers in the response
                    if 'followers' not in data or not data['followers']:
                        self.logger.info(
                            "No more followers found. Stopping pagination.")
                        stop_pagination = True
                        page_success = True
                        break

                    followers = data['followers']
                    all_followers.extend(followers)

                    self.logger.info(
                        f"Page {page_count + 1}: Found {len(followers)} followers")
                    self.logger.info(
                        f"Total followers so far: {len(all_followers)}")

                    # Check if there are more pages using next_cursor
                    has_next_page = data.get('has_next_page', False)
                    next_cursor = data.get('next_cursor', '')

                    if not has_next_page or not next_cursor:
                        self.logger.info(
                            "No more pages available. Stopping pagination.")
                        page_success = True
                        stop_pagination = True
                        break

                    cursor = next_cursor
                    page_count += 1
                    page_success = True

                    # Add delay between requests to be respectful to the API
                    if page_count < max_pages:
                        self.logger.info(
                            f"Waiting {delay} seconds before next request...")
                        time.sleep(delay)

                except requests.exceptions.Timeout as e:
                    self.logger.error(f"Request timeout: {e}")
                    retry_delay = min(30, (2 ** retry_count) * delay)
                    self.logger.warning(
                        f"Timeout occurred, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_count += 1

                except requests.exceptions.ConnectionError as e:
                    self.logger.error(f"Connection error: {e}")
                    retry_delay = min(30, (2 ** retry_count) * delay)
                    self.logger.warning(
                        f"Connection error, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_count += 1

                except requests.exceptions.RequestException as e:
                    self.logger.error(f"Request error: {e}")
                    retry_delay = min(30, (2 ** retry_count) * delay)
                    self.logger.warning(
                        f"Request failed, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_count += 1

                except Exception as e:
                    self.logger.error(f"Unexpected error: {e}")
                    retry_delay = min(30, (2 ** retry_count) * delay)
                    self.logger.warning(
                        f"Unexpected error, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_count += 1

            # If we've exhausted all retries for this page, log and continue to next page
            if not page_success and not stop_pagination:
                self.logger.error(
                    f"Failed to fetch page {page_count + 1} after {max_retries} retries. Moving to next page.")
                # Try to continue with next page if possible
                if cursor:
                    page_count += 1
                else:
                    # If we can't get a cursor, we have to stop
                    self.logger.error(
                        "Cannot continue without cursor. Stopping pagination.")
                    break

        # Final summary
        if stop_pagination:
            self.logger.info(
                f"Exited cleanly after {page_count} pages. Total followers: {len(all_followers)}"
            )

        self.logger.info(
            f"Followers fetch completed for @{username}. Total followers: {len(all_followers)}")
        self.logger.info(f"Saved {len(saved_files)} API response files")

        # Return both followers and saved files info
        return all_followers, saved_files


class GunShotFollowersAnalyzer:
    """Main class for analyzing Twitter followers using official Twitter handles"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the analyzer

        Args:
            api_key: Twitter API key
        """
        self.followers_service = GunShotFollowersService(api_key)
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

    def save_followers_chunked(self, followers: List[Dict[str, Any]], username: str, company_name: str, deal_id: str, chunk_size: int = 500) -> Dict[str, Any]:
        """
        Save followers in chunks to avoid MongoDB 16MB document limit

        Args:
            followers: List of follower objects
            username: Twitter username (without @)
            company_name: Name of the company
            deal_id: Deal ID for database reference
            chunk_size: Number of followers per chunk (default: 500)

        Returns:
            Dictionary containing metadata about the saved followers
        """
        if not followers:
            self.logger.warning(f"No followers to save for @{username}")
            return {
                'company_name': company_name,
                'username': username,
                'total_followers': 0,
                'total_chunks': 0,
                'processing_status': 'failed'
            }

        # Split followers into chunks
        chunks = [followers[i:i + chunk_size]
                  for i in range(0, len(followers), chunk_size)]
        total_chunks = len(chunks)

        self.logger.info(
            f"Saving {len(followers)} followers for @{username} in {total_chunks} chunks")

        # Save each chunk as a separate document
        chunk_ids = []
        for chunk_index, chunk in enumerate(chunks):
            followers_obj = Followers(
                deal_id=deal_id,
                company_handle=username,
                company_name=company_name,
                followers=chunk,
                chunk_index=chunk_index,
                chunk_size=chunk_size,
                approach="GUNSHOT",
                total_followers=len(followers),  # Total across all chunks
                processing_status="completed"
            )
            followers_obj.save()
            chunk_ids.append(str(followers_obj.id))

            self.logger.info(
                f"Saved chunk {chunk_index + 1}/{total_chunks} with {len(chunk)} followers")

        # Save metadata
        metadata_obj = FollowersMetadata(
            deal_id=deal_id,
            company_handle=username,
            company_name=company_name,
            total_followers=len(followers),
            total_chunks=total_chunks,
            approach="GUNSHOT",
            processing_status="completed"
        )
        metadata_obj.save()

        self.logger.info(
            f"Saved metadata for @{username}: {len(followers)} followers in {total_chunks} chunks")

        # Save to local JSON file
        json_filepath = self.save_followers_to_json(
            followers, username, company_name, deal_id)

        return {
            'company_name': company_name,
            'username': username,
            'metadata_id': str(metadata_obj.id),
            'total_followers': len(followers),
            'total_chunks': total_chunks,
            'chunk_ids': chunk_ids,
            'json_filepath': json_filepath,
            'processing_status': 'completed'
        }

    def save_followers_to_json(self, followers: List[Dict[str, Any]], username: str, company_name: str, deal_id: str) -> str:
        """
        Save followers data to local JSON file

        Args:
            followers: List of follower objects
            username: Twitter username (without @)
            company_name: Name of the company
            deal_id: Deal ID for reference

        Returns:
            Path to the saved JSON file
        """
        try:
            # Create followers directory if it doesn't exist
            followers_dir = os.path.join(
                os.path.dirname(__file__),
                'twitter_search_results',
                'followers'
            )
            os.makedirs(followers_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"followers_{deal_id}_{username}_{timestamp}.json"
            filepath = os.path.join(followers_dir, filename)

            # Prepare data structure
            followers_data = {
                'deal_id': deal_id,
                'company_name': company_name,
                'company_handle': username,
                'search_timestamp': datetime.now().isoformat(),
                'approach': 'GUNSHOT',
                'total_followers': len(followers),
                'followers': followers,
                'metadata': {
                    'file_created': datetime.now().isoformat(),
                    'file_version': '1.0',
                    'data_source': 'twitterapi.io',
                    'processing_notes': 'Followers fetched using Gun Shot approach'
                }
            }

            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(followers_data, f, indent=2, ensure_ascii=False)

            self.logger.info(
                f"Saved {len(followers)} followers to JSON file: {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Error saving followers to JSON file: {e}")
            return ""

    def fetch_followers_for_company(self, username: str, company_name: str, deal_id: str) -> Dict[str, Any]:
        """
        Fetch followers for a specific company and save to database

        Args:
            username: Twitter username (without @)
            company_name: Name of the company
            deal_id: Deal ID for database reference

        Returns:
            Dictionary containing followers data and metadata
        """
        self.logger.info(
            f"Fetching followers for @{username} ({company_name})")

        # Fetch followers
        followers, saved_files = self.followers_service.fetch_followers(
            username, deal_id, max_pages=1200, page_size=200, delay=0.3)

        # Save followers using chunked storage
        result = self.save_followers_chunked(
            followers, username, company_name, deal_id)

        # Add search timestamp and saved files info
        result['search_timestamp'] = datetime.now().isoformat()
        result['api_call_files'] = saved_files
        result['total_api_call_files'] = len(saved_files)

        return result

    def save_results(self, deal_id: str, all_results: List[Dict[str, Any]]) -> str:
        """
        Save followers results to JSON file

        Args:
            deal_id: Deal ID
            all_results: List of followers results for all companies

        Returns:
            Path to saved file
        """
        filename = f"twitter_followers_gunshot_{deal_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Collect JSON file paths
        json_files = [r.get('json_filepath', '')
                      for r in all_results if r.get('json_filepath')]

        # Collect API call file paths
        api_call_files = []
        total_api_calls = 0
        for result in all_results:
            api_files = result.get('api_call_files', [])
            api_call_files.extend(api_files)
            total_api_calls += result.get('total_api_call_files', 0)

        output_data = {
            'deal_id': deal_id,
            'search_timestamp': datetime.now().isoformat(),
            'approach': 'GUNSHOT_FOLLOWERS',
            'total_companies': len(all_results),
            'json_files_created': len(json_files),
            'json_file_paths': json_files,
            'api_call_files_created': total_api_calls,
            'api_call_file_paths': api_call_files,
            'api_calls_directory': os.path.join(self.followers_service.api_calls_dir, deal_id),
            'results': all_results
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results saved to: {filepath}")
        self.logger.info(
            f"API call files saved to: {os.path.join(self.followers_service.api_calls_dir, deal_id)}")
        return filepath

    def analyze_deal(self, deal_id: str, force_reprocess: bool = True) -> Optional[str]:
        """
        Main method to analyze a deal using official Twitter handles to fetch followers

        Args:
            deal_id: Deal ID to analyze
            force_reprocess: If True, force reprocessing even if GUNSHOT_approach_done is True

        Returns:
            Path to results file or None if failed
        """
        self.logger.info(
            f"Starting GUNSHOT Followers analysis for deal ID: {deal_id}")

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

        # Check if GUNSHOT approach is already done
        if hasattr(processing_job, 'GUNSHOT_approach_done') and processing_job.GUNSHOT_approach_done and not force_reprocess:
            self.logger.info(
                f"GUNSHOT approach already completed for deal {deal_id}. Skipping processing.")
            return "SKIPPED"

        # Clean up existing data before processing to avoid duplicates
        if not hasattr(processing_job, 'GUNSHOT_approach_done') or not processing_job.GUNSHOT_approach_done or force_reprocess:
            self.logger.info(
                f"Cleaning up existing GUNSHOT search data for deal {deal_id}")
            self.cleanup_utils.cleanup_existing_search_data(deal_id, "GUNSHOT")

            # Also clean up existing followers data
            existing_followers = Followers.objects.filter(
                deal_id=deal_id, approach="GUNSHOT")
            existing_followers.delete()
            self.logger.info(
                f"Cleaned up existing followers data for deal {deal_id}")

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

        # Step 4: Fetch followers for each company separately
        all_results = []

        # Fetch followers for target company
        try:
            target_result = self.fetch_followers_for_company(
                target_handle, target_name, deal_id
            )
            all_results.append(target_result)
            self.logger.info(
                f"Completed followers fetch for target company @{target_handle}")

        except Exception as e:
            self.logger.error(
                f"Error fetching followers for target company @{target_handle}: {e}")

        # Fetch followers for acquire company
        try:
            acquire_result = self.fetch_followers_for_company(
                acquire_handle, acquire_name, deal_id
            )
            all_results.append(acquire_result)
            self.logger.info(
                f"Completed followers fetch for acquire company @{acquire_handle}")

        except Exception as e:
            self.logger.error(
                f"Error fetching followers for acquire company @{acquire_handle}: {e}")

        # Step 5: Save results
        if all_results:
            filepath = self.save_results(deal_id, all_results)

            # Log summary
            total_followers = sum(r.get('total_followers', 0)
                                  for r in all_results)
            total_api_calls = sum(r.get('total_api_call_files', 0)
                                  for r in all_results)
            self.logger.info("=== GUNSHOT Followers Analysis Complete ===")
            self.logger.info(f"Deal ID: {deal_id}")
            self.logger.info(f"Total companies processed: {len(all_results)}")
            self.logger.info(f"Total followers found: {total_followers}")
            self.logger.info(f"Total API calls saved: {total_api_calls}")
            self.logger.info(f"Results saved to: {filepath}")
            self.logger.info(
                f"API call files saved to: {os.path.join(self.followers_service.api_calls_dir, deal_id)}")

            # Step 6: Mark GUNSHOT approach as completed
            try:
                processing_job.GUNSHOT_approach_done = True
                processing_job.updatedAt = datetime.utcnow()
                processing_job.save()
                self.logger.info(
                    f"Marked GUNSHOT approach as completed for deal {deal_id}")
            except Exception as e:
                self.logger.error(
                    f"Error updating GUNSHOT_approach_done flag: {e}")

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
        logger.error("Usage: python gun_shot_approach.py <deal_id> [--force]")
        logger.error(
            "Example: python gun_shot_approach.py 68184d52478abf06ec1a28ec")
        logger.error(
            "Example (force reprocess): python gun_shot_approach.py 68184d52478abf06ec1a28ec --force")
        sys.exit(1)

    deal_id = sys.argv[1]
    force_reprocess = len(sys.argv) == 3 and sys.argv[2] == "--force"

    try:
        # Initialize analyzer (will use TWITTER_API_KEY environment variable)
        analyzer = GunShotFollowersAnalyzer()

        # Run analysis
        result_file = analyzer.analyze_deal(
            deal_id, force_reprocess=force_reprocess)

        if result_file == "SKIPPED":
            logger.info(
                "Analysis skipped - GUNSHOT approach already completed for this deal")
            logger.info("Use --force flag to reprocess if needed")
        elif result_file:
            logger.info("GUNSHOT Followers Analysis completed successfully!")
            logger.info(f"Results saved to: {result_file}")
        else:
            logger.error("Analysis failed or no results found")
            sys.exit(1)

        # Step 6: Process high value followers with description filter
        high_value_followers_processor = HighValueFollowersProcessor()
        high_value_followers_result_file = high_value_followers_processor.process_deal(
            deal_id)
        logger.info(
            f"High Value Followers results saved to: {high_value_followers_result_file}")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
