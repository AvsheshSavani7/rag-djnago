#!/usr/bin/env python3
"""
Example script showing how to use the SEC document analyzer programmatically
"""

from test_analyze_sec_document import analyze_sec_document
import sys
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    """Example usage of the SEC document analyzer"""

    # Example SEC document URL (this is a placeholder - replace with real URL)
    example_url = "https://www.sec.gov/Archives/edgar/data/910073/000091007325000121/exhibit21-amendedandrest.htm"
    company_name = "Example Company Inc."

    print("🚀 Example: Analyzing SEC Document Programmatically")
    print("=" * 60)

    # Analyze the document
    result = analyze_sec_document(
        url=example_url,
        company_name=company_name,
        max_pages=4
    )

    # Check if analysis was successful
    if result.get('success'):
        print("\n✅ Analysis completed successfully!")

        # Access specific results
        classification = result.get('classification')
        document_kind = result.get('document_kind')
        confidence = result.get('confidence')

        print(f"📊 Results Summary:")
        print(f"   Classification: {classification}")
        print(f"   Document Kind: {document_kind}")
        print(f"   Confidence: {confidence}%")

        # Save results to file
        output_file = "analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"💾 Results saved to: {output_file}")

    else:
        print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
