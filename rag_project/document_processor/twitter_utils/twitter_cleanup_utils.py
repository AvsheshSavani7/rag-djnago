#!/usr/bin/env python3
"""
Twitter Search Cleanup Utilities
Common functionality for cleaning up existing search data
"""

import logging
from typing import Optional
from document_processor.models import SearchQuery, Tweet


class TwitterCleanupUtils:
    """Utility class for cleaning up Twitter search data"""

    def __init__(self):
        """Initialize cleanup utilities"""
        self.logger = logging.getLogger(__name__)

    def cleanup_existing_search_data(self, deal_id: str, approach: str = "RF1") -> None:
        """
        Remove existing search queries and related tweets for a deal with specific approach

        Args:
            deal_id: Deal ID to clean up data for
            approach: Search approach (default: "RF1")
        """
        try:
            # Find all search queries for this deal with specified approach
            existing_queries = SearchQuery.objects(
                deal_id=deal_id, approach=approach)

            total_queries = len(existing_queries)
            total_tweets_deleted = 0

            for query in existing_queries:
                # Delete all tweets related to this search query
                tweets_to_delete = Tweet.objects(search_query_id=query)
                tweet_count = len(tweets_to_delete)
                tweets_to_delete.delete()
                total_tweets_deleted += tweet_count

                # Delete the search query
                query.delete()

            self.logger.info(
                f"Cleaned up {total_queries} search queries and {total_tweets_deleted} tweets for deal {deal_id} with approach {approach}")

        except Exception as e:
            self.logger.error(
                f"Error cleaning up existing search data for deal {deal_id}: {e}")

    def get_existing_search_stats(self, deal_id: str, approach: str = "RF1") -> dict:
        """
        Get statistics about existing search data for a deal

        Args:
            deal_id: Deal ID to check
            approach: Search approach (default: "RF1")

        Returns:
            Dictionary with search statistics
        """
        try:
            existing_queries = SearchQuery.objects(
                deal_id=deal_id, approach=approach)
            total_tweets = 0

            for query in existing_queries:
                tweets = Tweet.objects(search_query_id=query)
                total_tweets += len(tweets)

            return {
                'total_queries': len(existing_queries),
                'total_tweets': total_tweets,
                'deal_id': deal_id,
                'approach': approach
            }

        except Exception as e:
            self.logger.error(
                f"Error getting search stats for deal {deal_id}: {e}")
            return {
                'total_queries': 0,
                'total_tweets': 0,
                'deal_id': deal_id,
                'approach': approach,
                'error': str(e)
            }
