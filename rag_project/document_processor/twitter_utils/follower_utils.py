#!/usr/bin/env python3
"""
Follower Utilities
Helper functions for working with chunked follower data
"""

import logging
import os
import json
from typing import List, Dict, Any, Optional
from document_processor.models import Followers, FollowersMetadata


class FollowerUtils:
    """Utility class for working with chunked follower data"""

    def __init__(self):
        """Initialize follower utilities"""
        self.logger = logging.getLogger(__name__)

    def get_followers_for_company(self, deal_id: str, company_handle: str, approach: str = "GUNSHOT") -> List[Dict[str, Any]]:
        """
        Retrieve all followers for a specific company from chunked storage

        Args:
            deal_id: Deal ID
            company_handle: Twitter handle (without @)
            approach: Search approach (default: "GUNSHOT")

        Returns:
            List of all follower objects
        """
        try:
            # Get all chunks for this company
            chunks = Followers.objects.filter(
                deal_id=deal_id,
                company_handle=company_handle,
                approach=approach
            ).order_by('chunk_index')

            if not chunks:
                self.logger.warning(
                    f"No follower chunks found for @{company_handle} in deal {deal_id}")
                return []

            # Combine all chunks
            all_followers = []
            for chunk in chunks:
                all_followers.extend(chunk.followers)

            self.logger.info(
                f"Retrieved {len(all_followers)} followers for @{company_handle} from {len(chunks)} chunks")
            return all_followers

        except Exception as e:
            self.logger.error(
                f"Error retrieving followers for @{company_handle}: {e}")
            return []

    def get_followers_metadata(self, deal_id: str, company_handle: str = None, approach: str = "GUNSHOT") -> List[Dict[str, Any]]:
        """
        Get metadata about follower collections

        Args:
            deal_id: Deal ID
            company_handle: Twitter handle (optional, if None returns all companies)
            approach: Search approach (default: "GUNSHOT")

        Returns:
            List of metadata objects
        """
        try:
            query = {
                'deal_id': deal_id,
                'approach': approach
            }

            if company_handle:
                query['company_handle'] = company_handle

            metadata = FollowersMetadata.objects.filter(**query)

            return [
                {
                    'company_name': m.company_name,
                    'company_handle': m.company_handle,
                    'total_followers': m.total_followers,
                    'total_chunks': m.total_chunks,
                    'processing_status': m.processing_status,
                    'created_at': m.created_at.isoformat(),
                    'metadata_id': str(m.id)
                }
                for m in metadata
            ]

        except Exception as e:
            self.logger.error(f"Error retrieving follower metadata: {e}")
            return []

    def delete_followers_for_company(self, deal_id: str, company_handle: str, approach: str = "GUNSHOT") -> bool:
        """
        Delete all follower data for a specific company

        Args:
            deal_id: Deal ID
            company_handle: Twitter handle (without @)
            approach: Search approach (default: "GUNSHOT")

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete follower chunks
            chunks = Followers.objects.filter(
                deal_id=deal_id,
                company_handle=company_handle,
                approach=approach
            )
            chunk_count = len(chunks)
            chunks.delete()

            # Delete metadata
            metadata = FollowersMetadata.objects.filter(
                deal_id=deal_id,
                company_handle=company_handle,
                approach=approach
            )
            metadata_count = len(metadata)
            metadata.delete()

            self.logger.info(
                f"Deleted {chunk_count} chunks and {metadata_count} metadata records for @{company_handle}")
            return True

        except Exception as e:
            self.logger.error(
                f"Error deleting followers for @{company_handle}: {e}")
            return False
