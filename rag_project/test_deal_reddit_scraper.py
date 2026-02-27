#!/usr/bin/env python3
"""
Test script for deal Reddit scraper
Usage: python test_deal_reddit_scraper.py <deal_id>
"""

from document_processor.reddit_utils.deal_reddit_scraper import run_deal_reddit_analysis
import sys
import os

# Add the document_processor path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__),
                'document_processor', 'reddit_utils'))

# Import the function


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_deal_reddit_scraper.py <deal_id>")
        print("Example: python test_deal_reddit_scraper.py 68ada8914a6006a0946ec7fe")
        sys.exit(1)

    deal_id = sys.argv[1]

    try:
        result = run_deal_reddit_analysis(deal_id)
        if result:
            print(f"\n✅ Test completed successfully!")
            print(f"Result: {result}")
        else:
            print(f"\n❌ Test failed for deal {deal_id}")
    except Exception as e:
        print(f"❌ Error during test: {e}")


if __name__ == "__main__":
    main()
