#!/usr/bin/env python3
"""
Delete Tweets by Search Query ID
This script deletes all tweets associated with specific search query IDs.

Usage:
# python delete_tweets_by_search_query.py
"""

from document_processor.models import SearchQuery, Tweet
import django
import os
import sys
import logging
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()

# Import after Django setup


def delete_tweets_by_search_query_ids(search_query_ids):
    """
    Delete all tweets associated with specific search query IDs

    Args:
        search_query_ids: List of search query IDs to delete tweets for
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    total_tweets_deleted = 0
    total_queries_deleted = 0

    for search_query_id in search_query_ids:
        try:
            logger.info(f"Processing search query ID: {search_query_id}")

            # Check if search query exists
            try:
                search_query = SearchQuery.objects.get(id=search_query_id)
                logger.info(
                    f"Found search query: {search_query.search_query[:50]}...")
                logger.info(f"Deal ID: {search_query.deal_id}")
                logger.info(f"Approach: {search_query.approach}")
                logger.info(f"Total tweets: {search_query.total_tweets}")
            except SearchQuery.DoesNotExist:
                logger.error(
                    f"Search query with ID {search_query_id} not found")
                continue

            # Delete all tweets associated with this search query
            tweets_to_delete = Tweet.objects.filter(
                search_query_id=search_query_id)
            tweet_count = len(tweets_to_delete)

            if tweet_count > 0:
                tweets_to_delete.delete()
                total_tweets_deleted += tweet_count
                logger.info(
                    f"Deleted {tweet_count} tweets for search query {search_query_id}")
            else:
                logger.info(
                    f"No tweets found for search query {search_query_id}")

            # Delete the search query itself
            search_query.delete()
            total_queries_deleted += 1
            logger.info(f"Deleted search query {search_query_id}")

        except Exception as e:
            logger.error(
                f"Error processing search query {search_query_id}: {e}")
            continue

    logger.info("=" * 50)
    logger.info("DELETION SUMMARY:")
    logger.info(f"Total search queries deleted: {total_queries_deleted}")
    logger.info(f"Total tweets deleted: {total_tweets_deleted}")
    logger.info("=" * 50)

    return {
        'queries_deleted': total_queries_deleted,
        'tweets_deleted': total_tweets_deleted
    }


def main():
    """Main function to run the script"""

    # Search query IDs to delete
    search_query_ids = [
        "68aea198bce4a36766cd6802",
        "68aea131bce4a36766cd6031"
    ]

    print("Twitter Tweet Deletion Script")
    print("=" * 40)
    print(f"Search Query IDs to delete: {search_query_ids}")
    print()

    # Ask for confirmation
    confirm = input(
        "Are you sure you want to delete all tweets for these search queries? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Operation cancelled.")
        return

    # Perform deletion
    try:
        results = delete_tweets_by_search_query_ids(search_query_ids)

        print("\n" + "=" * 50)
        print("DELETION COMPLETED SUCCESSFULLY!")
        print(f"Search queries deleted: {results['queries_deleted']}")
        print(f"Tweets deleted: {results['tweets_deleted']}")
        print("=" * 50)

    except Exception as e:
        print(f"Error during deletion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
