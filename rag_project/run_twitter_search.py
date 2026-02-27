#!/usr/bin/env python3
"""
Wrapper script to run the Twitter search from the correct directory

# Initial run (5-year search)
python run_twitter_search.py <deal_id> --approach=3

# Daily mode (24-hour search for cron jobs)
python run_twitter_search.py <deal_id> --approach=3 --daily

# Force reprocess
python run_twitter_search.py <deal_id> --approach=3 --force
"""

import os
import sys
import django
import argparse

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()


def main():
    parser = argparse.ArgumentParser(description='Run Twitter search analysis')
    parser.add_argument('deal_id', help='Deal ID to analyze')
    parser.add_argument('--approach', choices=['1', '2', '3'], default='1',
                        help='Twitter search approach (1, 2, or 3)')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing even if already completed')
    parser.add_argument('--daily', action='store_true',
                        help='Run in daily mode (for approach 3 only)')

    args = parser.parse_args()

    # Import the appropriate approach
    if args.approach == '1':
        from document_processor.twitter_utils.riffle_approach_1 import main as approach_main
    elif args.approach == '2':
        from document_processor.twitter_utils.riffle_approach_2 import main as approach_main
    elif args.approach == '3':
        from document_processor.twitter_utils.riffle_approach_3 import main as approach_main

    # Set up sys.argv for the approach script
    sys.argv = [sys.argv[0], args.deal_id]
    if args.force:
        sys.argv.append('--force')
    if args.daily:
        sys.argv.append('--daily')

    # Run the selected approach
    approach_main()


if __name__ == "__main__":
    main()
