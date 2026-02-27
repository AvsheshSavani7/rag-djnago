#!/usr/bin/env python3
"""
Twitter Search Script for Deal Analysis
This script fetches Twitter data for company combinations related to M&A deals.

below command to run single deal search
# python run_twitter_search.py 682f00def21b9fca8e1d04fe;
"""


from .gpt_product_analyzer import GPTProductAnalyzer
from .twitter_cleanup_utils import TwitterCleanupUtils
from document_processor.models import ProcessingJob, SearchQuery, Tweet, CompanyProducts, CompetitiveAnalysis
from document_processor.models import ProcessingJob, SearchQuery, Tweet
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

# Import Django models after setup


class TwitterSearchService:
    """Service for searching Twitter using twitterapi.io"""

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
            query: Search query string
            max_results: Maximum number of results to fetch

        Returns:
            List of tweet data
        """
        all_tweets = []
        cursor = ""

        while len(all_tweets) < max_results:
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
                self.logger.info(f"data: {data}")

                if not tweets:
                    break

                all_tweets.extend(tweets)

                # Check if there are more pages
                if not data.get('has_next_page', False):
                    break

                cursor = data.get('next_cursor', '')
                self.logger.info(f"next_cursor: {cursor}")
                if not cursor:
                    break

                # Rate limiting - wait between requests
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching tweets: {e}")
                break

        return all_tweets[:max_results]


class DealTwitterAnalyzer:
    """Main class for analyzing Twitter data for M&A deals"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the analyzer

        Args:
            api_key: Twitter API key
        """
        self.twitter_service = TwitterSearchService(api_key)
        self.output_dir = os.path.join(
            os.path.dirname(__file__),
            'twitter_search_results'
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger(__name__)

        # Initialize cleanup utilities and GPT analyzer
        self.cleanup_utils = TwitterCleanupUtils()
        try:
            self.gpt_analyzer = GPTProductAnalyzer()
        except ValueError as e:
            self.logger.warning(f"GPT analyzer not available: {e}")
            self.gpt_analyzer = None

    def cleanup_existing_search_data(self, deal_id: str) -> None:
        """
        Remove existing search queries and related tweets for a deal with RF1 approach

        Args:
            deal_id: Deal ID to clean up data for
        """
        self.cleanup_utils.cleanup_existing_search_data(deal_id, "RF1")

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

            # Convert to dictionary format similar to the example
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

    def extract_company_names(self, deal_data: Dict[str, Any]) -> List[str]:
        """
        Extract all company names from deal data

        Args:
            deal_data: Deal data dictionary

        Returns:
            List of unique company names
        """
        companies = []

        # Add acquirer and target names
        if deal_data.get('acquire_name'):
            companies.append(deal_data['acquire_name'])

        if deal_data.get('target_name'):
            companies.append(deal_data['target_name'])

        # Extract subsidiary names from schema_results
        schema_results = deal_data.get('schema_results', {})
        if schema_results is None:
            schema_results = {}

        party_details = schema_results.get('party_details', {})
        if party_details is None:
            party_details = {}

        # Target subsidiaries
        subsidiaries = party_details.get('subsidiaries', {})
        if subsidiaries is None:
            subsidiaries = {}
        target_subs = subsidiaries.get('target_subsidiaries', {})
        if isinstance(target_subs, dict):
            answer = target_subs.get('answer', '')
            if answer and answer != 'Not found' and answer != 'No target subsidiaries explicitly involved in the merger structure are mentioned.':
                # Try to parse JSON if it's a JSON string
                try:
                    if answer.startswith('['):
                        subs_list = json.loads(answer)
                        for sub in subs_list:
                            if isinstance(sub, dict) and 'name' in sub:
                                companies.append(sub['name'])
                    else:
                        # If it's a plain text description, we might need to parse it differently
                        # For now, we'll skip complex parsing
                        pass
                except json.JSONDecodeError:
                    pass

        # Acquirer subsidiaries
        acquirer_subs = subsidiaries.get('acquirer_subsidiaries', {})
        if isinstance(acquirer_subs, dict):
            answer = acquirer_subs.get('answer', '')
            if answer and answer != 'Not found':
                try:
                    if answer.startswith('['):
                        subs_list = json.loads(answer)
                        for sub in subs_list:
                            if isinstance(sub, dict) and 'name' in sub:
                                companies.append(sub['name'])
                except json.JSONDecodeError:
                    pass

        # Remove duplicates and clean up names
        unique_companies = []
        for company in companies:
            if company and company.strip():
                clean_name = company.strip()
                if clean_name not in unique_companies:
                    unique_companies.append(clean_name)

        return unique_companies

    def create_company_combinations(self, companies: List[str]) -> List[Tuple[str, str]]:
        """
        Create all possible combinations of companies (2 at a time)

        Args:
            companies: List of company names`

        Returns:
            List of company pairs
        """
        if len(companies) < 2:
            return []

        # Create all unique pairs
        combinations = list(itertools.combinations(companies, 2))
        return combinations

    def build_twitter_query(self, company1: str, company2: str, announce_date: str, years_back: int = 5) -> str:
        """
        Build Twitter search query for company combination

        Args:
            company1: First company name
            company2: Second company name
            announce_date: Announcement date in YYYY-MM-DD format
            years_back: Number of years to search back from announcement date

        Returns:
            Twitter search query string
        """
        if not announce_date:
            # If no announce date, search last 5 years from today
            end_date = datetime.now()
        else:
            end_date = datetime.strptime(announce_date, '%Y-%m-%d')

        start_date = end_date - timedelta(days=years_back * 365)

        # Format dates for Twitter API (YYYY-MM-DD_HH:MM:SS_UTC)
        since_date = start_date.strftime('%Y-%m-%d_00:00:00_UTC')
        until_date = end_date.strftime('%Y-%m-%d_23:59:59_UTC')

        # Build query - search for both companies mentioned together

        # Simple query with just company names
        query = f'"{company1}" "{company2}" since:{since_date} until:{until_date}'

        return query

    def search_tweets_for_combination(self, company1: str, company2: str, announce_date: str, deal_id: str) -> Dict[str, Any]:
        """
        Search tweets for a specific company combination and save to database

        Args:
            company1: First company name
            company2: Second company name
            announce_date: Announcement date
            deal_id: Deal ID for database reference

        Returns:
            Dictionary containing search results and metadata
        """
        query = self.build_twitter_query(
            company1, company2, announce_date)

        self.logger.info(f"Searching tweets for: {company1} + {company2}")
        self.logger.debug(f"Query: {query}")

        tweets = self.twitter_service.search_tweets(query, max_results=5000)

        # Save search query to database
        combination_data = {
            'company1': company1,
            'company2': company2
        }

        search_query_obj = SearchQuery(
            search_query=query,
            deal_id=deal_id,
            approach="RF1",
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
            'combination': combination_data,
            'search_query': query,
            'search_query_id': str(search_query_obj.id),
            'announce_date': announce_date,
            'search_timestamp': datetime.now().isoformat(),
            'tweet_count': len(tweets),
            'tweets': tweets,
            'saved_tweet_ids': saved_tweet_ids
        }

        return result

    def save_results(self, deal_id: str, combination_results: List[Dict[str, Any]]) -> str:
        """
        Save search results to JSON file

        Args:
            deal_id: Deal ID
            combination_results: List of search results for each combination

        Returns:
            Path to saved file
        """
        filename = f"twitter_search_{deal_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)

        output_data = {
            'deal_id': deal_id,
            'search_timestamp': datetime.now().isoformat(),
            'total_combinations': len(combination_results),
            'results': combination_results
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results saved to: {filepath}")
        return filepath

    def analyze_deal(self, deal_id: str, force_reprocess: bool = True, save_file: bool = True) -> Optional[str]:
        """
        Main method to analyze a deal and search for related tweets

        Args:
            deal_id: Deal ID to analyze
            force_reprocess: If True, force reprocessing even if RF1_approach_done is True
            save_file: If True, save results to file (default: True)

        Returns:
            Path to results file or None if failed
        """
        self.logger.info(f"Starting Twitter analysis for deal ID: {deal_id}")

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

        # Step 1.5: Check if RF1 approach is already done
        if processing_job.RF1_approach_done and not force_reprocess:
            self.logger.info(
                f"RF1 approach already completed for deal {deal_id}. Skipping processing.")
            return "SKIPPED"

        # Clean up existing data before processing to avoid duplicates
        # This happens when:
        # 1. RF1_approach_done is False (normal reprocessing case)
        # 2. force_reprocess is True (forced reprocessing)
        if not processing_job.RF1_approach_done or force_reprocess:
            self.logger.info(
                f"Cleaning up existing RF1 search data for deal {deal_id}")
            self.cleanup_existing_search_data(deal_id)

        self.logger.info(
            f"Deal: {deal_data.get('acquire_name', 'N/A')} acquiring {deal_data.get('target_name', 'N/A')}")
        self.logger.info(
            f"Announce date: {deal_data.get('announce_date', 'N/A')}")

        # Step 2: Extract company names
        companies = self.extract_company_names(deal_data)
        self.logger.info(f"Found {len(companies)} companies: {companies}")

        if len(companies) < 2:
            self.logger.warning(
                "Not enough companies found for meaningful combinations")
            return None

        # Step 2.5: Find Twitter handles for unique companies
        try:
            from .twitter_handle_finder import TwitterHandleFinder
            twitter_finder = TwitterHandleFinder()
            twitter_details = twitter_finder.process_deal_twitter_handles(
                deal_id, companies)
            self.logger.info(
                f"Twitter handle search completed for {len(companies)} companies")
            self.logger.info(
                f"Companies with handles: {twitter_details.get('summary', {}).get('companies_with_handles', 0)}")
        except Exception as e:
            self.logger.error(f"Error finding Twitter handles: {e}")
            # Continue with Twitter search even if handle finding fails

        # Step 3: Create combinations
        combinations = self.create_company_combinations(companies)
        self.logger.info(f"combinations: {combinations}")
        self.logger.info(f"Created {len(combinations)} company combinations")

        # Step 4: Search tweets for each combination
        all_results = []
        for i, (company1, company2) in enumerate(combinations, 1):
            self.logger.info(
                f"Processing combination {i}/{len(combinations)}: {company1} + {company2}")

            try:
                result = self.search_tweets_for_combination(
                    company1, company2, deal_data.get('announce_date'), deal_id
                )
                all_results.append(result)
                self.logger.info(f"Found {result['tweet_count']} tweets")

                # Rate limiting between combinations
                if i < len(combinations):
                    time.sleep(2)

            except Exception as e:
                self.logger.error(
                    f"Error searching for combination {company1} + {company2}: {e}")
                # Continue with next combination
                continue

        # Step 5: Save results
        if all_results:
            filepath = None
            if save_file:
                filepath = self.save_results(deal_id, all_results)

            # Log summary
            total_tweets = sum(r['tweet_count'] for r in all_results)
            self.logger.info("=== Analysis Complete ===")
            self.logger.info(f"Deal ID: {deal_id}")
            self.logger.info(
                f"Total combinations processed: {len(all_results)}")
            self.logger.info(f"Total tweets found: {total_tweets}")
            if filepath:
                self.logger.info(f"Results saved to: {filepath}")
            else:
                self.logger.info(
                    "Results not saved to file (bulk processing mode)")

            # Step 6: Run product analysis if GPT analyzer is available
            if self.gpt_analyzer:
                self.logger.info(
                    "Starting product analysis for competitive intelligence...")
                try:
                    competitive_analysis = self.gpt_analyzer.process_deal_product_analysis(
                        deal_id)
                    if competitive_analysis:
                        self.logger.info(
                            f"Product analysis completed: {len(competitive_analysis.competitive_pairs)} competitive pairs identified")
                    else:
                        self.logger.warning(
                            "Product analysis failed or returned no results")
                except Exception as e:
                    self.logger.error(f"Error during product analysis: {e}")
            else:
                self.logger.info(
                    "GPT analyzer not available, skipping product analysis")

            # Step 7: Mark RF1 approach as completed
            try:
                processing_job.RF1_approach_done = True
                processing_job.updatedAt = datetime.utcnow()
                processing_job.save()
                self.logger.info(
                    f"Marked RF1 approach as completed for deal {deal_id}")
            except Exception as e:
                self.logger.error(
                    f"Error updating RF1_approach_done flag: {e}")

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
        logger.error("Usage: python riffle_approach_1.py <deal_id> [--force]")
        logger.error(
            "Example: python riffle_approach_1.py 68184d52478abf06ec1a28ec")
        logger.error(
            "Example (force reprocess): python riffle_approach_1.py 68184d52478abf06ec1a28ec --force")
        sys.exit(1)

    deal_id = sys.argv[1]
    force_reprocess = len(sys.argv) == 3 and sys.argv[2] == "--force"

    try:
        # Initialize analyzer (will use TWITTER_API_KEY environment variable)
        analyzer = DealTwitterAnalyzer()

        # Run analysis
        result_file = analyzer.analyze_deal(
            deal_id, force_reprocess=force_reprocess)

        if result_file == "SKIPPED":
            logger.info(
                "Analysis skipped - RF1 approach already completed for this deal")
            logger.info("Use --force flag to reprocess if needed")
        elif result_file:
            logger.info("Analysis completed successfully!")
            logger.info(f"Results saved to: {result_file}")
        else:
            logger.error("Analysis failed or no results found")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


# Now when search tweet for deal combination is complated then add give flag in deal table:

# RF1_approch_done(boolean)
# so next time if we search for same deal then it first check this flag if true then no need to do anything.

# if the flag do flase then
