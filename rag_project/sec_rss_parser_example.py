#!/usr/bin/env python3
"""
Example script demonstrating how to use the SEC RSS Parser app.

This script shows different ways to interact with the SEC RSS parser:
1. Manual processing of the RSS feed
2. Querying stored filings
3. Getting statistics
"""

from sec_rss_parser.models import SECFiling, SECFeedStatus
from sec_rss_parser.services import SECFeedProcessor
import os
import sys
import django
from datetime import datetime

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()


def main():
    print("=== SEC RSS Parser Example ===\n")

    # Initialize the processor
    processor = SECFeedProcessor()

    # 1. Process the RSS feed
    print("1. Processing SEC RSS feed...")
    result = processor.process_feed()

    if result['success']:
        print(f"✅ {result['message']}")
        print(f"   Total items: {result['total_items']}")
        print(f"   New items: {result['new_items']}")
    else:
        print(f"❌ Error: {result['error']}")

    print()

    # 2. Get basic statistics
    print("2. Getting filing statistics...")
    total_filings = SECFiling.objects.count()
    eight_k_filings = SECFiling.objects(form_type='8-K').count()
    eight_k_with_htm = SECFiling.objects(
        form_type='8-K', has_htm_files=True).count()

    print(f"   Total filings: {total_filings}")
    print(f"   8-K filings: {eight_k_filings}")
    print(f"   8-K filings with HTM files: {eight_k_with_htm}")

    print()

    # 3. Get recent 8-K filings with HTM files
    print("3. Recent 8-K filings with HTM files:")
    recent_8k = SECFiling.objects(
        form_type='8-K',
        has_htm_files=True
    ).order_by('-created_at').limit(5)

    for filing in recent_8k:
        print(f"   📄 {filing.company_name} - {filing.form_type}")
        print(f"      CIK: {filing.cik_number}")
        print(f"      Accession: {filing.accession_number}")
        print(f"      Created: {filing.created_at}")
        print()

    # 4. Get filings by form type
    print("4. Recent filings by form type:")
    form_types = ['8-K', '10-Q', '10-K']

    for form_type in form_types:
        count = SECFiling.objects(form_type=form_type).count()
        print(f"   {form_type}: {count} filings")

    print()

    # 5. Show feed status
    print("5. Feed processing status:")
    feed_status = SECFeedStatus.objects().first()

    if feed_status:
        print(f"   Last fetch: {feed_status.last_fetch_time}")
        print(f"   Total processed: {feed_status.total_items_processed}")
        print(f"   New items found: {feed_status.new_items_found}")
        if feed_status.error_message:
            print(f"   Error: {feed_status.error_message}")
    else:
        print("   No feed status found")

    print("\n=== Example completed ===")


if __name__ == "__main__":
    main()
