#!/usr/bin/env python3
"""
Test script for Gun Shot Approach (Followers)
This script tests the gun shot approach functionality for fetching followers
"""

from document_processor.twitter_utils.gun_shot_approach import GunShotFollowersAnalyzer
from document_processor.models import ProcessingJob, Followers, FollowersMetadata
from document_processor.twitter_utils.follower_utils import FollowerUtils
import os
import sys
import django
import logging

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()

# Import after Django setup


def test_gun_shot_approach():
    """Test the gun shot approach with a specific deal ID"""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    # Static deal ID for testing - CHANGE THIS ID FOR DIFFERENT DEALS
    # Juniper Networks, Inc./Hewlett Packard Enterprise Company
    DEAL_ID = "68ac4a254a6006a0946ec3bb"
    # DEAL_ID = "68ada8914a6006a0946ec7fe"  # Altair Engineering Inc

    try:
        # Get the specific deal by ID
        deal = ProcessingJob.objects.get(id=DEAL_ID)
        deal_id = str(deal.id)
        logger.info(f"Testing with deal ID: {deal_id}")
        logger.info(f"Deal: {deal.acquire_name} acquiring {deal.target_name}")

        # Check if Twitter details exist
        twitter_details = deal.twitter_details
        if not twitter_details:
            logger.error("No Twitter details found in deal")
            return False

        logger.info(f"Twitter details found: {twitter_details}")

        # Initialize analyzer
        analyzer = GunShotFollowersAnalyzer()

        # Test fetching deal data
        deal_data = analyzer.fetch_deal_data(deal_id)
        if not deal_data:
            logger.error("Failed to fetch deal data")
            return False

        logger.info("Successfully fetched deal data")

        # Test extracting Twitter handles
        twitter_handles = analyzer.extract_twitter_handles(deal_data)
        logger.info(f"Extracted Twitter handles: {twitter_handles}")

        if len(twitter_handles) < 2:
            logger.warning("Not enough Twitter handles found for testing")
            return False

        # Test fetching followers for each company
        target_name = deal_data.get('target_name')
        acquire_name = deal_data.get('acquire_name')

        target_handle = twitter_handles.get(target_name)
        acquire_handle = twitter_handles.get(acquire_name)

        if target_handle and acquire_handle:
            logger.info(
                f"Testing followers fetch for target company: @{target_handle}")
            logger.info(
                f"Testing followers fetch for acquire company: @{acquire_handle}")

            # Test fetching followers for target company (just a small sample)
            try:
                target_result = analyzer.fetch_followers_for_company(
                    target_handle, target_name, deal_id
                )
                logger.info(
                    f"Target company followers test completed: {target_result['total_followers']} followers")

                # Check if followers were saved to database
                follower_utils = FollowerUtils()
                metadata = follower_utils.get_followers_metadata(
                    deal_id, target_handle)

                if metadata:
                    logger.info(f"Followers metadata found: {metadata[0]}")
                    logger.info(
                        f"Total followers: {metadata[0]['total_followers']}")
                    logger.info(f"Total chunks: {metadata[0]['total_chunks']}")

                    # Test retrieving followers
                    followers = follower_utils.get_followers_for_company(
                        deal_id, target_handle)
                    logger.info(
                        f"Retrieved {len(followers)} followers from chunks")

                    # Test JSON file functionality
                    json_files = follower_utils.find_follower_json_files(
                        deal_id, target_handle)
                    if json_files:
                        logger.info(
                            f"Found {len(json_files)} JSON files for target company")
                        for json_file in json_files:
                            logger.info(f"JSON file: {json_file}")
                            # Test loading from JSON
                            json_data = follower_utils.load_followers_from_json(
                                json_file)
                            if json_data:
                                logger.info(
                                    f"Loaded {json_data.get('total_followers', 0)} followers from JSON")
                    else:
                        logger.warning(
                            "No JSON files found for target company")
                else:
                    logger.warning("Followers metadata not found in database")

            except Exception as e:
                logger.error(f"Error testing target company followers: {e}")

            # Test fetching followers for acquire company (just a small sample)
            try:
                acquire_result = analyzer.fetch_followers_for_company(
                    acquire_handle, acquire_name, deal_id
                )
                logger.info(
                    f"Acquire company followers test completed: {acquire_result['total_followers']} followers")

                # Check if followers were saved to database
                metadata = follower_utils.get_followers_metadata(
                    deal_id, acquire_handle)

                if metadata:
                    logger.info(f"Followers metadata found: {metadata[0]}")
                    logger.info(
                        f"Total followers: {metadata[0]['total_followers']}")
                    logger.info(f"Total chunks: {metadata[0]['total_chunks']}")

                    # Test retrieving followers
                    followers = follower_utils.get_followers_for_company(
                        deal_id, acquire_handle)
                    logger.info(
                        f"Retrieved {len(followers)} followers from chunks")

                    # Test JSON file functionality
                    json_files = follower_utils.find_follower_json_files(
                        deal_id, acquire_handle)
                    if json_files:
                        logger.info(
                            f"Found {len(json_files)} JSON files for acquire company")
                        for json_file in json_files:
                            logger.info(f"JSON file: {json_file}")
                            # Test loading from JSON
                            json_data = follower_utils.load_followers_from_json(
                                json_file)
                            if json_data:
                                logger.info(
                                    f"Loaded {json_data.get('total_followers', 0)} followers from JSON")
                    else:
                        logger.warning(
                            "No JSON files found for acquire company")
                else:
                    logger.warning("Followers metadata not found in database")

            except Exception as e:
                logger.error(f"Error testing acquire company followers: {e}")

            logger.info(
                "Gun Shot Approach (Followers) test completed successfully!")
            return True
        else:
            logger.error(
                "Missing Twitter handles for target or acquire company")
            return False

    except ProcessingJob.DoesNotExist:
        logger.error(
            f"Deal with ID {DEAL_ID} not found. Please check the deal ID.")
        return False
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        return False


if __name__ == "__main__":
    success = test_gun_shot_approach()
    if success:
        print("✅ Gun Shot Approach (Followers) test passed!")
    else:
        print("❌ Gun Shot Approach (Followers) test failed!")
        sys.exit(1)
