#!/usr/bin/env python3
"""
Simple Tweet Search Script
Search for tweets using a search query and save results to JSON

Usage:
python tweet_search_simple.py
"""

import os
import sys
import json
import time
import logging
import requests
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SimpleTweetSearcher:
    """Simple tweet searcher"""

    def __init__(self):
        """Initialize the tweet searcher"""
        # Setup Twitter API key
        self.twitter_api_key = os.getenv('TWITTER_API_KEY')
        if not self.twitter_api_key:
            raise ValueError(
                "Twitter API key is required. Set TWITTER_API_KEY environment variable.")

        self.base_url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
        self.headers = {
            'X-API-Key': self.twitter_api_key,
            'Content-Type': 'application/json'
        }

        # Setup logger
        self.logger = logging.getLogger(__name__)

        # Setup output directory
        self.output_dir = os.path.join(
            os.path.dirname(__file__), 'tweet_search_results')
        os.makedirs(self.output_dir, exist_ok=True)

    def search_tweets(self, search_query: str, max_pages: int = 50) -> List[Dict[str, Any]]:
        """
        Search for tweets using the provided query with cursor pagination

        Args:
            search_query: Search query to use
            max_pages: Maximum number of pages to fetch

        Returns:
            List of all tweets found
        """
        all_tweets = []
        cursor = ""
        page_count = 0

        self.logger.info(f"Starting tweet search with query: {search_query}")
        self.logger.info(f"Max pages: {max_pages}")

        while cursor is not None and page_count < max_pages:
            page_count += 1
            self.logger.info(f"Fetching page {page_count}...")

            params = {
                'query': search_query,
                'queryType': 'Latest',
                'cursor': cursor
            }

            try:
                response = requests.get(
                    self.base_url, headers=self.headers, params=params, timeout=120)
                response.raise_for_status()

                data = response.json()
                tweets = data.get('tweets', [])

                self.logger.info(
                    f"Page {page_count}: Found {len(tweets)} tweets")

                if tweets:
                    all_tweets.extend(tweets)
                    self.logger.info(
                        f"Total tweets collected so far: {len(all_tweets)}")

                # Check if there are more pages
                if data.get('has_next_page', False):
                    cursor = data.get('next_cursor', None)
                    self.logger.info(f"Next cursor: {cursor}")
                else:
                    cursor = None
                    self.logger.info("No more pages available")

                # Rate limiting
                time.sleep(0.5)

            except requests.exceptions.RequestException as e:
                self.logger.error(
                    f"Error fetching tweets on page {page_count}: {e}")
                break
            except Exception as e:
                self.logger.error(
                    f"Unexpected error on page {page_count}: {e}")
                break

        self.logger.info(
            f"Search completed. Total tweets found: {len(all_tweets)} across {page_count} pages")
        return all_tweets

    def save_results_to_json(self, search_query: str, tweets: List[Dict[str, Any]]) -> str:
        """
        Save search results to JSON file

        Args:
            search_query: Search query used
            tweets: List of tweets found

        Returns:
            Path to saved JSON file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create safe filename
        safe_query = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_'
                             for c in search_query[:50]).replace(' ', '_')

        filename = f"tweet_search_{safe_query}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Prepare data for JSON
        json_data = {
            'search_info': {
                'search_query': search_query,
                'search_timestamp': datetime.now().isoformat(),
                'total_tweets_found': len(tweets)
            },
            'tweets': tweets,
            'metadata': {
                'script_version': '1.0',
                'api_endpoint': self.base_url
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results saved to: {filepath}")
        return filepath

    def search_and_save(self, search_query: str) -> str:
        """
        Main method to search tweets and save results

        Args:
            search_query: Search query to use

        Returns:
            Path to saved JSON file
        """
        self.logger.info(f"Starting tweet search")
        self.logger.info(f"Search query: {search_query}")

        # Search for tweets
        tweets = self.search_tweets(search_query)

        if not tweets:
            self.logger.warning("No tweets found")
            # Still save empty results for debugging
            return self.save_results_to_json(search_query, [])

        # Save results
        result_file = self.save_results_to_json(search_query, tweets)

        # Print summary
        self.logger.info("=" * 60)
        self.logger.info("SEARCH COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Search query: {search_query}")
        self.logger.info(f"Total tweets found: {len(tweets)}")
        self.logger.info(f"Results saved to: {result_file}")
        self.logger.info("=" * 60)

        return result_file


def main():
    """Main function to run the script"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)

    # Single search query variable - modify this as needed
    # SEARCH_QUERY = 'from:sethfiermonti ("Hewlett Packard Enterprise Company" OR "HPE" OR "@HPE" OR "HPE GreenLake Central" OR "HPE Ezmeral Data Fabric" OR "HPE Ezmeral Container Platform" OR "HPE OneView" OR "HPE InfoSight" OR "HPE SimpliVity RapidDR" OR "HPE Aruba Central" OR "HPE StoreOnce Catalyst" OR "HPE Data Protector" OR "HPE IMC" OR "HPE Recovery Manager Central" OR "HPE ProLiant Servers" OR "HPE Apollo Systems" OR "HPE Synergy" OR "HPE Superdome Flex" OR "HPE Nimble Storage Arrays" OR "HPE 3PAR StoreServ" OR "HPE Alletra" OR "HPE SimpliVity" OR "HPE MSA Storage" OR "HPE StoreOnce Backup Systems" OR "HPE Aruba Switches" OR "HPE Edgeline Converged Edge Systems" OR "HPE BladeSystem" OR "HPE Integrity Servers" OR "HPE XP Storage" OR "HPE Pointnext Technology Services" OR "HPE Advisory and Professional Services" OR "HPE Operational Support Services" OR "HPE Education Services" OR "HPE GreenLake" OR "HPE Managed Services" OR "HPE Financial Services" OR "HPE Cloud Consulting Services" OR "HPE Support Services" OR "HPE Aruba Wireless Access Points" OR "HPE Networking Transceivers and Cables" OR "HPE Tape Storage" OR "HPE StoreEver" OR "HPE Networking Accessories")since:2020-01-01_00:00:00_UTC until:2025-09-03_23:59:59_UTC lang:en'
    SEARCH_QUERY = 'from:sethfiermonti ("Hewlett Packard Enterprise" OR "HPE" OR "@HPE" OR "HPE GreenLake" OR "GreenLake" OR "#HPEGreenLake" OR "HPE Ezmeral" OR "Ezmeral" OR "HPE OneView" OR "OneView" OR "HPE InfoSight" OR "InfoSight" OR "HPE SimpliVity" OR "SimpliVity" OR "HPE Aruba" OR "Aruba Central" OR "Aruba Switches" OR "HPE StoreOnce" OR "StoreOnce" OR "HPE Data Protector" OR "Data Protector" OR "HPE Nimble" OR "Nimble Storage" OR "HPE IMC" OR "Intelligent Management Center" OR "HPE ProLiant" OR "ProLiant" OR "HPE Apollo" OR "Apollo Systems" OR "HPE Synergy" OR "Synergy" OR "HPE Superdome" OR "Superdome Flex" OR "HPE 3PAR" OR "3PAR StoreServ" OR "HPE Alletra" OR "Alletra" OR "HPE MSA" OR "MSA Storage" OR "HPE Edgeline" OR "Edgeline" OR "HPE BladeSystem" OR "BladeSystem" OR "HPE Integrity" OR "Integrity Servers" OR "HPE XP Storage" OR "XP Storage" OR "HPE Pointnext" OR "Pointnext" OR "HPE Advisory Services" OR "HPE Professional Services" OR "HPE Operational Support" OR "HPE Education Services" OR "HPE Managed Services" OR "HPE Financial Services" OR "HPE Cloud Consulting" OR "HPE Support Services" OR "HPE Wireless Access Points" OR "HPE Networking" OR "HPE Tape Storage" OR "HPE StoreEver" OR "StoreEver" OR "#HPE" OR "#Aruba" OR "#GreenLake" OR "#Ezmeral" OR "#ProLiant" OR "#Nimble" OR "#InfoSight" OR "#SimpliVity" OR "#StoreOnce") since:2020-01-01_00:00:00_UTC until:2025-09-03_23:59:59_UTC lang:en'

    logger.info("=" * 60)
    logger.info("SIMPLE TWEET SEARCH")
    logger.info("=" * 60)
    logger.info(f"Search Query: {SEARCH_QUERY}")
    logger.info("=" * 60)

    try:
        # Initialize searcher
        searcher = SimpleTweetSearcher()

        # Run search
        result_file = searcher.search_and_save(SEARCH_QUERY)

        logger.info("✅ Tweet search completed successfully!")
        logger.info(f"📁 Results saved to: {result_file}")

        # Show file size
        if os.path.exists(result_file):
            size_kb = os.path.getsize(result_file) / 1024
            logger.info(f"📊 File size: {size_kb:.1f} KB")

    except Exception as e:
        logger.error(f"❌ Error during tweet search: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
