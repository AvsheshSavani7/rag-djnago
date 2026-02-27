#!/usr/bin/env python3
"""
Standalone script to analyze SEC documents by providing a URL with .htm extension.
This script downloads the document and analyzes it using GPT to determine:
1. If it's a new deal or amendment
2. Document kind classification
"""

from sec_rss_parser.document_analyzer import SECDocumentAnalyzer
import sys
import os
import json
import argparse
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the current directory to Python path to import from sec_rss_parser
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def analyze_sec_document(url: str, company_name: str = "Unknown Company", max_pages: int = 4) -> Dict[str, Any]:
    """
    Analyze a SEC document from a given URL

    Args:
        url: SEC document URL with .htm extension
        company_name: Company name for context
        max_pages: Maximum number of pages to analyze (default: 4)

    Returns:
        Dictionary containing analysis results
    """
    print(f"🔍 Analyzing SEC document from URL: {url}")
    print(f"📊 Company: {company_name}")
    print(f"📄 Max pages to analyze: {max_pages}")
    print("-" * 80)

    # Check if OpenAI API key is configured before initializing analyzer
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OpenAI API key not configured!")
        print(
            "Please set the OPENAI_API_KEY environment variable or create a .env file with:")
        print("OPENAI_API_KEY=your_api_key_here")
        return {
            'error': 'OpenAI API key not configured',
            'success': False
        }

    # Initialize the document analyzer
    analyzer = SECDocumentAnalyzer()

    try:
        # Download the HTM file
        print("📥 Downloading document...")
        html_content = analyzer.download_htm_file(url)

        if not html_content:
            print("❌ Failed to download the document")
            return {
                'error': 'Failed to download document',
                'success': False
            }

        print(
            f"✅ Successfully downloaded document ({len(html_content)} characters)")

        # Extract document text
        print("📝 Extracting document text...")
        document_text = analyzer.extract_document_pages(
            html_content, max_pages=max_pages)

        if not document_text:
            print("❌ Failed to extract document text")
            return {
                'error': 'Failed to extract document text',
                'success': False
            }

        print(f"✅ Extracted {len(document_text)} characters of text")

        # Analyze with GPT
        print("🤖 Analyzing document with GPT...")
        analysis_result = analyzer.analyze_document_with_gpt(
            document_text, company_name)

        # Display results
        print("\n" + "=" * 80)
        print("📋 ANALYSIS RESULTS")
        print("=" * 80)

        if analysis_result.get('error'):
            print(f"❌ Analysis failed: {analysis_result['error']}")
            return {
                'error': analysis_result['error'],
                'success': False
            }

        # Classification
        classification = "NEW DEAL" if analysis_result.get(
            'is_new_deal') is True else "AMENDMENT" if analysis_result.get('is_new_deal') is False else "INCONCLUSIVE"
        print(f"🏷️  Classification: {classification}")

        # Document kind
        document_kind = analysis_result.get('document_kind', 'Unknown')
        print(f"📄 Document Kind: {document_kind}")

        # Confidence
        confidence = analysis_result.get('confidence', 0)
        print(f"🎯 Confidence: {confidence}%")

        # Reasoning
        reasoning = analysis_result.get('reasoning', 'No reasoning provided')
        print(f"💭 Reasoning: {reasoning}")

        # Key indicators
        key_indicators = analysis_result.get('key_indicators', [])
        if key_indicators:
            print(f"🔑 Key Indicators:")
            for indicator in key_indicators:
                print(f"   • {indicator}")

        # Raw response
        raw_response = analysis_result.get('raw_response', '')
        if raw_response:
            print(f"\n📄 Raw GPT Response:")
            try:
                # Pretty print JSON
                parsed_response = json.loads(raw_response)
                print(json.dumps(parsed_response, indent=2))
            except:
                print(raw_response)

        print("\n" + "=" * 80)

        return {
            'success': True,
            'url': url,
            'company_name': company_name,
            'classification': classification,
            'document_kind': document_kind,
            'confidence': confidence,
            'reasoning': reasoning,
            'key_indicators': key_indicators,
            'raw_response': raw_response,
            'analysis_result': analysis_result
        }

    except Exception as e:
        print(f"❌ Error analyzing document: {e}")
        return {
            'error': str(e),
            'success': False
        }


def main():
    """Main function to handle command line arguments and run analysis"""
    parser = argparse.ArgumentParser(
        description="Analyze SEC documents from URL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_sec_document.py "https://www.sec.gov/Archives/edgar/data/1234567/000123456724000001/abc123.htm"
  python analyze_sec_document.py "https://www.sec.gov/Archives/edgar/data/1234567/000123456724000001/abc123.htm" --company "Apple Inc."
  python analyze_sec_document.py "https://www.sec.gov/Archives/edgar/data/1234567/000123456724000001/abc123.htm" --pages 6
        """
    )

    parser.add_argument(
        'url',
        help='SEC document URL with .htm extension'
    )

    parser.add_argument(
        '--company', '-c',
        default='Unknown Company',
        help='Company name for context (default: "Unknown Company")'
    )

    parser.add_argument(
        '--pages', '-p',
        type=int,
        default=4,
        help='Maximum number of pages to analyze (default: 4)'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output file to save results as JSON (optional)'
    )

    args = parser.parse_args()

    # Validate URL
    if not args.url.endswith('.htm'):
        print("⚠️  Warning: URL does not end with .htm extension")

    # Run analysis
    result = analyze_sec_document(args.url, args.company, args.pages)

    # Save to file if requested
    if args.output and result.get('success'):
        try:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Results saved to: {args.output}")
        except Exception as e:
            print(f"❌ Error saving results to file: {e}")

    # Exit with appropriate code
    sys.exit(0 if result.get('success') else 1)


if __name__ == "__main__":
    main()
