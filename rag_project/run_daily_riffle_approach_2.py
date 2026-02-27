#!/usr/bin/env python3
"""
Daily Runner for Riffle Approach 2 (RF2) Twitter Analysis
This script runs daily to process all deals where RF1_approach_done is True.

Usage:
    python run_daily_riffle_approach_2.py [--dry-run] [--limit N] [--force]

Options:
    --dry-run: Show which deals would be processed without actually processing them
    --limit N: Limit the number of deals to process (useful for testing)
    --force: Force reprocessing in the analyzer (overrides existing RF2 data)
"""

from document_processor.twitter_utils.riffle_approach_2 import RiskBasedDealAnalyzer
from document_processor.models import ProcessingJob
import os
import sys
import django
import logging
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()


class DailyRiffleApproach2Runner:
    """Daily runner for processing deals with Riffle Approach 2"""

    def __init__(self, dry_run: bool = False, limit: Optional[int] = None, force: bool = False):
        """
        Initialize the daily runner

        Args:
            dry_run: If True, only show what would be processed without actually processing
            limit: Maximum number of deals to process (None for no limit)
            force: Force reprocessing in the analyzer (overrides existing RF2 data)
        """
        self.dry_run = dry_run
        self.limit = limit
        self.force = force

        # Setup logger
        self.logger = logging.getLogger(__name__)

        # Initialize the analyzer (will be created only if not dry run)
        self.analyzer = None
        if not self.dry_run:
            try:
                self.analyzer = RiskBasedDealAnalyzer()
                self.logger.info("Initialized RiskBasedDealAnalyzer")
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize RiskBasedDealAnalyzer: {e}")
                raise

    def get_eligible_deals(self) -> List[ProcessingJob]:
        """
        Get all deals that are eligible for RF2 processing

        Returns:
            List of ProcessingJob objects that need RF2 processing
        """
        # Get all deals where RF1 is done
        query = ProcessingJob.objects(RF1_approach_done=True)

        # Apply limit if specified
        if self.limit:
            query = query[:self.limit]

        deals = list(query)

        self.logger.info(
            f"Found {len(deals)} eligible deals for RF2 processing")

        return deals

    def process_deal(self, deal: ProcessingJob) -> Dict[str, Any]:
        """
        Process a single deal with RF2 approach

        Args:
            deal: ProcessingJob object to process

        Returns:
            Dictionary with processing results
        """
        deal_id = str(deal.id)
        deal_name = f"{deal.acquire_name}/{deal.target_name}" if deal.acquire_name and deal.target_name else deal_id

        self.logger.info(f"Processing deal: {deal_name} (ID: {deal_id})")

        if self.dry_run:
            return {
                'deal_id': deal_id,
                'deal_name': deal_name,
                'status': 'dry_run',
                'message': 'Would process this deal'
            }

        try:
            # Run the RF2 analysis
            result_file = self.analyzer.analyze_deal(
                deal_id, force_reprocess=self.force)

            if result_file == "SKIPPED":
                return {
                    'deal_id': deal_id,
                    'deal_name': deal_name,
                    'status': 'skipped',
                    'message': 'RF2 already completed (skipped)'
                }
            elif result_file:
                return {
                    'deal_id': deal_id,
                    'deal_name': deal_name,
                    'status': 'success',
                    'result_file': result_file,
                    'message': 'RF2 processing completed successfully'
                }
            else:
                return {
                    'deal_id': deal_id,
                    'deal_name': deal_name,
                    'status': 'failed',
                    'message': 'RF2 processing failed or no results'
                }

        except Exception as e:
            self.logger.error(f"Error processing deal {deal_id}: {e}")
            return {
                'deal_id': deal_id,
                'deal_name': deal_name,
                'status': 'error',
                'error': str(e),
                'message': f'Error during processing: {e}'
            }

    def run_daily_processing(self) -> Dict[str, Any]:
        """
        Run the daily processing for all eligible deals

        Returns:
            Dictionary with processing summary
        """
        start_time = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info("Starting Daily RF2 Processing")
        self.logger.info(f"Start time: {start_time}")
        self.logger.info(f"Dry run: {self.dry_run}")
        self.logger.info(f"Force reprocessing: {self.force}")
        if self.limit:
            self.logger.info(f"Limit: {self.limit} deals")
        self.logger.info("=" * 60)

        # Get eligible deals
        eligible_deals = self.get_eligible_deals()

        if not eligible_deals:
            self.logger.info("No eligible deals found for RF2 processing")
            return {
                'status': 'completed',
                'total_deals': 0,
                'processed_deals': 0,
                'skipped_deals': 0,
                'failed_deals': 0,
                'error_deals': 0,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_minutes': 0
            }

        # Process each deal
        results = []
        processed_count = 0
        skipped_count = 0
        failed_count = 0
        error_count = 0

        for i, deal in enumerate(eligible_deals, 1):
            self.logger.info(f"Processing deal {i}/{len(eligible_deals)}")

            result = self.process_deal(deal)
            results.append(result)

            # Count results by status
            if result['status'] == 'success':
                processed_count += 1
            elif result['status'] == 'skipped':
                skipped_count += 1
            elif result['status'] == 'failed':
                failed_count += 1
            elif result['status'] == 'error':
                error_count += 1
            elif result['status'] == 'dry_run':
                processed_count += 1  # Count dry run as processed for summary

            # Log result
            self.logger.info(
                f"  Result: {result['status']} - {result['message']}")

            # Add delay between deals to avoid rate limiting
            if not self.dry_run and i < len(eligible_deals):
                self.logger.debug("Waiting 5 seconds before next deal...")
                import time
                time.sleep(5)

        # Calculate summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60  # minutes

        summary = {
            'status': 'completed',
            'total_deals': len(eligible_deals),
            'processed_deals': processed_count,
            'skipped_deals': skipped_count,
            'failed_deals': failed_count,
            'error_deals': error_count,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_minutes': round(duration, 2),
            'results': results
        }

        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("Daily RF2 Processing Summary")
        self.logger.info(f"Total deals found: {summary['total_deals']}")
        self.logger.info(
            f"Successfully processed: {summary['processed_deals']}")
        self.logger.info(f"Skipped: {summary['skipped_deals']}")
        self.logger.info(f"Failed: {summary['failed_deals']}")
        self.logger.info(f"Errors: {summary['error_deals']}")
        self.logger.info(f"Duration: {summary['duration_minutes']} minutes")
        self.logger.info(f"End time: {end_time}")
        self.logger.info("=" * 60)

        return summary

    def save_summary_report(self, summary: Dict[str, Any]) -> str:
        """
        Save processing summary to a JSON file

        Args:
            summary: Processing summary dictionary

        Returns:
            Path to saved report file
        """
        import json

        # Create reports directory
        reports_dir = os.path.join(os.path.dirname(__file__), 'daily_reports')
        os.makedirs(reports_dir, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"daily_rf2_report_{timestamp}.json"
        filepath = os.path.join(reports_dir, filename)

        # Save report
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Summary report saved to: {filepath}")
        return filepath


def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('daily_rf2_runner.log')
        ]
    )


def main():
    """Main function to run the daily RF2 processor"""
    parser = argparse.ArgumentParser(
        description='Daily runner for Riffle Approach 2 Twitter analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be processed
  python run_daily_riffle_approach_2.py --dry-run
  
  # Process up to 5 deals (for testing)
  python run_daily_riffle_approach_2.py --limit 5
  
  # Force reprocess all deals (overrides existing RF2 data)
  python run_daily_riffle_approach_2.py --force
  
  # Normal daily run
  python run_daily_riffle_approach_2.py
        """
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show which deals would be processed without actually processing them'
    )

    parser.add_argument(
        '--limit',
        type=int,
        help='Limit the number of deals to process (useful for testing)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing in the analyzer (overrides existing RF2 data)'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set the logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Initialize runner
        runner = DailyRiffleApproach2Runner(
            dry_run=args.dry_run,
            limit=args.limit,
            force=args.force
        )

        # Run daily processing
        summary = runner.run_daily_processing()

        # Save summary report
        if not args.dry_run:
            report_file = runner.save_summary_report(summary)
            logger.info(f"Processing complete. Report saved to: {report_file}")
        else:
            logger.info(
                "Dry run complete. No actual processing was performed.")

        # Exit with appropriate code
        if summary['error_deals'] > 0 or summary['failed_deals'] > 0:
            logger.warning("Processing completed with some errors/failures")
            sys.exit(1)
        else:
            logger.info("Processing completed successfully")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error during daily processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
