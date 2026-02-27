#!/usr/bin/env python3
"""
High Value Followers Export Script (RAG Project Level)
Exports high-value followers from MongoDB to JSON format with filtering and analysis options.

This script can be run from the rag_project directory level.

Usage:
# python export_high_value_followers.py
# python export_high_value_followers.py --deal-id 68ac4a254a6006a0946ec3bb
# python export_high_value_followers.py --min-score 5 --company "Target Company"
# python export_high_value_followers.py --all-deals --output-dir ./exports
"""

from document_processor.models import HighValueFollowers, ProcessingJob
import django
import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()

# Import after Django setup


class HighValueFollowersExporter:
    """Export high-value followers from MongoDB to JSON format"""

    def __init__(self, output_dir: str = None):
        """
        Initialize the exporter

        Args:
            output_dir: Directory to save JSON files (default: high_value_followers_exports)
        """
        self.output_dir = output_dir or os.path.join(
            current_dir,
            'high_value_followers_exports'
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger(__name__)

    def get_all_deals(self) -> List[str]:
        """Get all unique deal IDs that have high-value followers"""
        try:
            deal_ids = HighValueFollowers.objects.distinct('deal_id')
            self.logger.info(
                f"Found {len(deal_ids)} deals with high-value followers")
            return deal_ids
        except Exception as e:
            self.logger.error(f"Error fetching deal IDs: {e}")
            return []

    def get_deal_info(self, deal_id: str) -> Optional[Dict[str, Any]]:
        """Get deal information from ProcessingJob"""
        try:
            # Use MongoEngine syntax
            deal = ProcessingJob.objects.get(id=deal_id)

            return {
                'id': str(deal.id),
                'cik': deal.cik,
                'acquire_name': deal.acquire_name,
                'target_name': deal.target_name,
                'announce_date': deal.announce_date.strftime('%Y-%m-%d') if deal.announce_date else None,
                'created_at': deal.createdAt.isoformat() if deal.createdAt else None
            }
        except ProcessingJob.DoesNotExist:
            self.logger.warning(f"Deal {deal_id} not found in ProcessingJob")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching deal info for {deal_id}: {e}")
            return None

    def fetch_followers(self,
                        deal_id: str = None,
                        company_name: str = None,
                        company_handle: str = None,
                        min_score: int = None,
                        max_score: int = None,
                        approach: str = None,
                        limit: int = None) -> List[Dict[str, Any]]:
        """
        Fetch high-value followers with optional filters

        Args:
            deal_id: Filter by specific deal ID
            company_name: Filter by company name
            company_handle: Filter by company Twitter handle
            min_score: Minimum overall score
            max_score: Maximum overall score
            approach: Filter by processing approach
            limit: Maximum number of records to return

        Returns:
            List of follower dictionaries
        """
        try:
            # Build query
            query = {}

            if deal_id:
                query['deal_id'] = deal_id
            if company_name:
                query['company_name'] = company_name
            if company_handle:
                query['company_handle'] = company_handle
            if min_score is not None:
                query['overall_score__gte'] = min_score
            if max_score is not None:
                query['overall_score__lte'] = max_score
            if approach:
                query['approach'] = approach

            # Execute query without default sorting to avoid memory issues
            followers = HighValueFollowers.objects.filter(
                **query).order_by('id')

            if limit:
                followers = followers[:limit]

            # Convert to list of dictionaries
            followers_data = []
            for follower in followers:
                follower_dict = {
                    'id': str(follower.id),
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
                    'reason': follower.reason,
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

            self.logger.info(
                f"Fetched {len(followers_data)} high-value followers")
            return followers_data

        except Exception as e:
            self.logger.error(f"Error fetching followers: {e}")
            return []

    def fetch_followers_paginated(self,
                                  deal_id: str = None,
                                  company_name: str = None,
                                  company_handle: str = None,
                                  min_score: int = None,
                                  max_score: int = None,
                                  approach: str = None,
                                  batch_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Fetch high-value followers with pagination to handle large datasets

        Args:
            deal_id: Filter by specific deal ID
            company_name: Filter by company name
            company_handle: Filter by company Twitter handle
            min_score: Minimum overall score
            max_score: Maximum overall score
            approach: Filter by processing approach
            batch_size: Number of records to fetch per batch

        Returns:
            List of follower dictionaries
        """
        try:
            # Build query
            query = {}

            if deal_id:
                query['deal_id'] = deal_id
            if company_name:
                query['company_name'] = company_name
            if company_handle:
                query['company_handle'] = company_handle
            if min_score is not None:
                query['overall_score__gte'] = min_score
            if max_score is not None:
                query['overall_score__lte'] = max_score
            if approach:
                query['approach'] = approach

            all_followers_data = []
            skip = 0

            while True:
                # Fetch batch with pagination
                followers_batch = HighValueFollowers.objects.filter(
                    **query).order_by('id').skip(skip).limit(batch_size)

                # Convert batch to list
                batch_data = []
                for follower in followers_batch:
                    follower_dict = {
                        'id': str(follower.id),
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
                        'reason': follower.reason,
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
                    batch_data.append(follower_dict)

                # Add batch to results
                all_followers_data.extend(batch_data)

                # Check if we got fewer records than batch_size (end of data)
                if len(batch_data) < batch_size:
                    break

                # Move to next batch
                skip += batch_size
                self.logger.info(
                    f"Fetched batch of {len(batch_data)} followers (total: {len(all_followers_data)})")

            self.logger.info(
                f"Fetched {len(all_followers_data)} high-value followers in total")
            return all_followers_data

        except Exception as e:
            self.logger.error(f"Error fetching followers with pagination: {e}")
            return []

    def generate_statistics(self, followers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive statistics from followers data"""
        if not followers:
            return {}

        stats = {
            'total_followers': len(followers),
            'unique_deals': len(set(f['deal_id'] for f in followers)),
            'unique_companies': len(set(f['company_name'] for f in followers)),
            'score_distribution': defaultdict(int),
            'company_distribution': defaultdict(int),
            'deal_distribution': defaultdict(int),
            'approach_distribution': defaultdict(int),
            'gpt_model_distribution': defaultdict(int),
            'avg_score': 0,
            'min_score': float('inf'),
            'max_score': 0,
            'verified_count': 0,
            'protected_count': 0,
            'total_followers_count': 0,
            'total_statuses_count': 0
        }

        total_score = 0
        for follower in followers:
            score = follower.get('overall_score', 0)
            stats['score_distribution'][score] += 1
            stats['company_distribution'][follower.get(
                'company_name', 'Unknown')] += 1
            stats['deal_distribution'][follower.get('deal_id', 'Unknown')] += 1
            stats['approach_distribution'][follower.get(
                'approach', 'Unknown')] += 1
            stats['gpt_model_distribution'][follower.get(
                'gpt_model_used', 'Unknown')] += 1

            total_score += score
            stats['min_score'] = min(stats['min_score'], score)
            stats['max_score'] = max(stats['max_score'], score)

            if follower.get('verified', False):
                stats['verified_count'] += 1
            if follower.get('protected', False):
                stats['protected_count'] += 1

            stats['total_followers_count'] += follower.get(
                'followers_count', 0)
            stats['total_statuses_count'] += follower.get('statuses_count', 0)

        if followers:
            stats['avg_score'] = total_score / len(followers)

        # Convert defaultdict to regular dict for JSON serialization
        stats['score_distribution'] = dict(stats['score_distribution'])
        stats['company_distribution'] = dict(stats['company_distribution'])
        stats['deal_distribution'] = dict(stats['deal_distribution'])
        stats['approach_distribution'] = dict(stats['approach_distribution'])
        stats['gpt_model_distribution'] = dict(stats['gpt_model_distribution'])

        return stats

    def export_followers(self,
                         followers: List[Dict[str, Any]],
                         filename: str = None,
                         include_stats: bool = True,
                         include_deal_info: bool = True) -> str:
        """
        Export followers to JSON file

        Args:
            followers: List of follower dictionaries
            filename: Custom filename (optional)
            include_stats: Include statistics in export
            include_deal_info: Include deal information in export

        Returns:
            Path to saved JSON file
        """
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"high_value_followers_export_{timestamp}.json"

            filepath = os.path.join(self.output_dir, filename)

            # Prepare export data
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_followers': len(followers),
                'followers': followers
            }

            # Add statistics if requested
            if include_stats:
                export_data['statistics'] = self.generate_statistics(followers)

            # Add deal information if requested
            if include_deal_info:
                deal_ids = list(set(f['deal_id'] for f in followers))
                deal_info = {}
                for deal_id in deal_ids:
                    deal_data = self.get_deal_info(deal_id)
                    if deal_data:
                        deal_info[deal_id] = deal_data
                export_data['deal_information'] = deal_info

            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            self.logger.info(
                f"Exported {len(followers)} followers to: {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Error exporting followers: {e}")
            return None

    def export_by_deal(self, deal_id: str, include_stats: bool = True, batch_size: int = 1000) -> str:
        """Export all followers for a specific deal"""
        self.logger.info(f"Exporting followers for deal: {deal_id}")

        # Get deal info
        deal_info = self.get_deal_info(deal_id)
        if not deal_info:
            self.logger.error(f"Deal {deal_id} not found")
            return None

        # Fetch followers with pagination to handle large datasets
        followers = self.fetch_followers_paginated(
            deal_id=deal_id, batch_size=batch_size)

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"high_value_followers_deal_{deal_id}_{timestamp}.json"

        return self.export_followers(followers, filename, include_stats)

    def export_all_deals(self, include_stats: bool = True) -> List[str]:
        """Export followers for all deals"""
        self.logger.info("Exporting followers for all deals")

        deal_ids = self.get_all_deals()
        exported_files = []

        for deal_id in deal_ids:
            try:
                filepath = self.export_by_deal(deal_id, include_stats)
                if filepath:
                    exported_files.append(filepath)
            except Exception as e:
                self.logger.error(f"Error exporting deal {deal_id}: {e}")
                continue

        self.logger.info(f"Exported {len(exported_files)} deal files")
        return exported_files

    def export_consolidated(self,
                            deal_id: str = None,
                            min_score: int = None,
                            max_score: int = None,
                            company_name: str = None,
                            batch_size: int = 1000) -> str:
        """Export consolidated data with filters"""
        self.logger.info("Exporting consolidated followers data")

        # Fetch followers with filters and pagination
        followers = self.fetch_followers_paginated(
            deal_id=deal_id,
            min_score=min_score,
            max_score=max_score,
            company_name=company_name,
            batch_size=batch_size
        )

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filters = []
        if deal_id:
            filters.append(f"deal_{deal_id}")
        if min_score is not None:
            filters.append(f"min_score_{min_score}")
        if max_score is not None:
            filters.append(f"max_score_{max_score}")
        if company_name:
            safe_company = "".join(
                c if c.isalnum() else '_' for c in company_name)
            filters.append(f"company_{safe_company}")

        filter_suffix = "_".join(filters) if filters else "all"
        filename = f"high_value_followers_consolidated_{filter_suffix}_{timestamp}.json"

        return self.export_followers(followers, filename, include_stats=True, include_deal_info=True)

    def list_deals(self) -> None:
        """List all deals with high-value followers and their counts"""
        try:
            deal_ids = self.get_all_deals()

            if not deal_ids:
                self.logger.info("No deals found with high-value followers")
                return

            self.logger.info("Deals with high-value followers:")
            self.logger.info("-" * 80)

            for deal_id in deal_ids:
                # Get deal info
                deal_info = self.get_deal_info(deal_id)
                if deal_info:
                    target_name = deal_info.get('target_name', 'Unknown')
                    acquire_name = deal_info.get('acquire_name', 'Unknown')
                    announce_date = deal_info.get('announce_date', 'Unknown')
                else:
                    target_name = acquire_name = announce_date = 'Unknown'

                # Get follower count for this deal
                follower_count = HighValueFollowers.objects.filter(
                    deal_id=deal_id).count()

                self.logger.info(f"Deal ID: {deal_id}")
                self.logger.info(f"  Target: {target_name}")
                self.logger.info(f"  Acquire: {acquire_name}")
                self.logger.info(f"  Announce Date: {announce_date}")
                self.logger.info(f"  High-Value Followers: {follower_count}")
                self.logger.info("-" * 80)

        except Exception as e:
            self.logger.error(f"Error listing deals: {e}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get overall summary statistics for all high-value followers"""
        try:
            total_followers = HighValueFollowers.objects.count()
            unique_deals = len(HighValueFollowers.objects.distinct('deal_id'))
            unique_companies = len(
                HighValueFollowers.objects.distinct('company_name'))

            # Get score statistics using MongoEngine aggregation
            score_stats = HighValueFollowers.objects.aggregate([
                {'$group': {
                    '_id': None,
                    'avg_score': {'$avg': '$overall_score'},
                    'min_score': {'$min': '$overall_score'},
                    'max_score': {'$max': '$overall_score'}
                }}
            ])

            # Extract stats from aggregation result
            if score_stats:
                stats_result = list(score_stats)[0]
                avg_score = stats_result['avg_score'] or 0
                min_score = stats_result['min_score'] or 0
                max_score = stats_result['max_score'] or 0
            else:
                avg_score = min_score = max_score = 0

            # Get approach distribution using MongoEngine
            approaches = HighValueFollowers.objects.aggregate([
                {'$group': {'_id': '$approach', 'count': {'$sum': 1}}}
            ])
            approach_dist = {item['_id']: item['count'] for item in approaches}

            # Get company distribution using MongoEngine
            companies = HighValueFollowers.objects.aggregate([
                {'$group': {'_id': '$company_name', 'count': {'$sum': 1}}}
            ])
            company_dist = {item['_id']: item['count'] for item in companies}

            summary = {
                'total_high_value_followers': total_followers,
                'unique_deals': unique_deals,
                'unique_companies': unique_companies,
                'score_statistics': {
                    'average_score': round(avg_score, 2),
                    'minimum_score': min_score,
                    'maximum_score': max_score
                },
                'approach_distribution': approach_dist,
                'company_distribution': company_dist
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error getting summary stats: {e}")
            return {}


def main():
    """Main function to run the export script"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Export high-value followers to JSON')
    parser.add_argument('--deal-id', help='Export specific deal ID')
    parser.add_argument('--all-deals', action='store_true',
                        help='Export all deals')
    parser.add_argument('--consolidated', action='store_true',
                        help='Export consolidated data')
    parser.add_argument('--company', help='Filter by company name')
    parser.add_argument('--company-handle',
                        help='Filter by company Twitter handle')
    parser.add_argument('--min-score', type=int, help='Minimum overall score')
    parser.add_argument('--max-score', type=int, help='Maximum overall score')
    parser.add_argument('--approach', help='Filter by processing approach')
    parser.add_argument('--limit', type=int, help='Maximum number of records')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Batch size for pagination (default: 1000)')
    parser.add_argument('--output-dir', help='Output directory for JSON files')
    parser.add_argument('--no-stats', action='store_true',
                        help='Exclude statistics from export')
    parser.add_argument('--no-deal-info', action='store_true',
                        help='Exclude deal information from export')
    parser.add_argument('--list-deals', action='store_true',
                        help='List all deals with high-value followers')
    parser.add_argument('--summary', action='store_true',
                        help='Show summary statistics')

    args = parser.parse_args()

    try:
        # Initialize exporter
        exporter = HighValueFollowersExporter(output_dir=args.output_dir)

        if args.list_deals:
            # List all deals
            exporter.list_deals()

        elif args.summary:
            # Show summary statistics
            summary = exporter.get_summary_stats()
            if summary:
                logger.info("High-Value Followers Summary:")
                logger.info("-" * 50)
                logger.info(
                    f"Total High-Value Followers: {summary['total_high_value_followers']}")
                logger.info(f"Unique Deals: {summary['unique_deals']}")
                logger.info(f"Unique Companies: {summary['unique_companies']}")
                logger.info(
                    f"Average Score: {summary['score_statistics']['average_score']}")
                logger.info(
                    f"Score Range: {summary['score_statistics']['minimum_score']} - {summary['score_statistics']['maximum_score']}")
                logger.info("-" * 50)
                logger.info("Company Distribution:")
                for company, count in summary['company_distribution'].items():
                    logger.info(f"  {company}: {count} followers")
                logger.info("-" * 50)
                logger.info("Approach Distribution:")
                for approach, count in summary['approach_distribution'].items():
                    logger.info(f"  {approach}: {count} followers")
            else:
                logger.error("Failed to get summary statistics")

        elif args.all_deals:
            # Export all deals
            exported_files = exporter.export_all_deals(
                include_stats=not args.no_stats)
            logger.info(
                f"Successfully exported {len(exported_files)} deal files")

        elif args.consolidated:
            # Export consolidated data
            filepath = exporter.export_consolidated(
                deal_id=args.deal_id,
                min_score=args.min_score,
                max_score=args.max_score,
                company_name=args.company,
                batch_size=args.batch_size
            )
            if filepath:
                logger.info(
                    f"Successfully exported consolidated data to: {filepath}")
            else:
                logger.error("Failed to export consolidated data")

        elif args.deal_id:
            # Export specific deal
            filepath = exporter.export_by_deal(
                args.deal_id, include_stats=not args.no_stats, batch_size=args.batch_size)
            if filepath:
                logger.info(f"Successfully exported deal data to: {filepath}")
            else:
                logger.error(f"Failed to export deal {args.deal_id}")

        else:
            # Export with filters using pagination
            followers = exporter.fetch_followers_paginated(
                deal_id=args.deal_id,
                company_name=args.company,
                company_handle=args.company_handle,
                min_score=args.min_score,
                max_score=args.max_score,
                approach=args.approach,
                batch_size=args.batch_size
            )

            if followers:
                filepath = exporter.export_followers(
                    followers,
                    include_stats=not args.no_stats,
                    include_deal_info=not args.no_deal_info
                )
                if filepath:
                    logger.info(f"Successfully exported data to: {filepath}")
                else:
                    logger.error("Failed to export data")
            else:
                logger.warning("No followers found matching the criteria")

    except Exception as e:
        logger.error(f"Error during export: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
