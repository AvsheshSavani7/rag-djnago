#!/usr/bin/env python3
"""
API Calls Recovery Utility for Gun Shot Approach
This script helps analyze and recover data from the new API calls folder structure.

Usage:
# python api_calls_recovery.py <deal_id>
# python api_calls_recovery.py <deal_id> --recover
"""

import os
import json
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import glob


def analyze_api_calls_directory(deal_id: str) -> Dict[str, Any]:
    """
    Analyze the API calls directory for a specific deal

    Args:
        deal_id: Deal ID to analyze

    Returns:
        Dictionary containing analysis results
    """
    # Get the API calls directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    api_calls_dir = os.path.join(
        script_dir, 'twitter_search_results', 'api_calls', deal_id)

    if not os.path.exists(api_calls_dir):
        return {
            'deal_id': deal_id,
            'status': 'not_found',
            'message': f'API calls directory not found: {api_calls_dir}'
        }

    analysis = {
        'deal_id': deal_id,
        'status': 'found',
        'api_calls_directory': api_calls_dir,
        'companies': {},
        'total_api_calls': 0,
        'total_followers': 0,
        'analysis_timestamp': datetime.now().isoformat()
    }

    # Find all company subdirectories
    company_dirs = [d for d in os.listdir(api_calls_dir)
                    if os.path.isdir(os.path.join(api_calls_dir, d))]

    for company_dir in company_dirs:
        company_path = os.path.join(api_calls_dir, company_dir)
        json_files = glob.glob(os.path.join(company_path, '*.json'))

        company_data = {
            'username': company_dir,
            'api_call_files': len(json_files),
            'followers_count': 0,
            'pages_found': set(),
            'file_paths': json_files,
            'last_page': 0
        }

        # Analyze each JSON file
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                page_number = data.get('page_number', 0)
                company_data['pages_found'].add(page_number)
                company_data['last_page'] = max(
                    company_data['last_page'], page_number)

                # Count followers in this page
                followers = data.get('response_data', {}).get('followers', [])
                company_data['followers_count'] += len(followers)

            except Exception as e:
                print(f"Error reading {json_file}: {e}")

        # Convert set to list for JSON serialization
        company_data['pages_found'] = sorted(list(company_data['pages_found']))
        analysis['companies'][company_dir] = company_data
        analysis['total_api_calls'] += company_data['api_call_files']
        analysis['total_followers'] += company_data['followers_count']

    return analysis


def recover_followers_from_api_calls(deal_id: str) -> Dict[str, Any]:
    """
    Recover followers data from API calls directory

    Args:
        deal_id: Deal ID to recover data for

    Returns:
        Dictionary containing recovered followers data
    """
    analysis = analyze_api_calls_directory(deal_id)

    if analysis['status'] != 'found':
        return analysis

    recovered_data = {
        'deal_id': deal_id,
        'recovery_timestamp': datetime.now().isoformat(),
        'companies': {},
        'total_followers_recovered': 0
    }

    for company_name, company_data in analysis['companies'].items():
        all_followers = []

        # Sort files by page number to maintain order
        sorted_files = sorted(company_data['file_paths'],
                              key=lambda x: int(os.path.basename(x).split('_')[1]))

        for json_file in sorted_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                followers = data.get('response_data', {}).get('followers', [])
                all_followers.extend(followers)

            except Exception as e:
                print(f"Error reading {json_file}: {e}")

        recovered_data['companies'][company_name] = {
            'username': company_data['username'],
            'followers': all_followers,
            'total_followers': len(all_followers),
            'pages_recovered': len(company_data['pages_found']),
            'last_page': company_data['last_page']
        }

        recovered_data['total_followers_recovered'] += len(all_followers)

    return recovered_data


def save_recovered_data(recovered_data: Dict[str, Any], deal_id: str) -> str:
    """
    Save recovered data to JSON file

    Args:
        recovered_data: Recovered followers data
        deal_id: Deal ID

    Returns:
        Path to saved file
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(
        script_dir, 'twitter_search_results', 'recovered')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"recovered_followers_{deal_id}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(recovered_data, f, indent=2, ensure_ascii=False)

    return filepath


def print_analysis(analysis: Dict[str, Any]):
    """Print analysis results in a readable format"""
    print(f"\n=== API Calls Analysis for Deal {analysis['deal_id']} ===")
    print(f"Status: {analysis['status']}")

    if analysis['status'] == 'not_found':
        print(f"Message: {analysis['message']}")
        return

    print(f"API Calls Directory: {analysis['api_calls_directory']}")
    print(f"Total API Calls: {analysis['total_api_calls']}")
    print(f"Total Followers: {analysis['total_followers']}")
    print(f"Companies Found: {len(analysis['companies'])}")

    for company_name, company_data in analysis['companies'].items():
        print(f"\n  Company: {company_name}")
        print(f"    Username: @{company_data['username']}")
        print(f"    API Call Files: {company_data['api_call_files']}")
        print(f"    Followers Count: {company_data['followers_count']}")
        print(f"    Pages Found: {company_data['pages_found']}")
        print(f"    Last Page: {company_data['last_page']}")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python api_calls_recovery.py <deal_id> [--recover]")
        print("Example: python api_calls_recovery.py 68184d52478abf06ec1a28ec")
        print("Example: python api_calls_recovery.py 68184d52478abf06ec1a28ec --recover")
        sys.exit(1)

    deal_id = sys.argv[1]
    should_recover = len(sys.argv) > 2 and sys.argv[2] == '--recover'

    print(f"Analyzing API calls for deal ID: {deal_id}")

    # Analyze the directory
    analysis = analyze_api_calls_directory(deal_id)
    print_analysis(analysis)

    # If recovery is requested and data is found
    if should_recover and analysis['status'] == 'found':
        print(f"\nRecovering followers data...")
        recovered_data = recover_followers_from_api_calls(deal_id)

        if recovered_data['total_followers_recovered'] > 0:
            filepath = save_recovered_data(recovered_data, deal_id)
            print(
                f"Recovered {recovered_data['total_followers_recovered']} followers")
            print(f"Recovered data saved to: {filepath}")
        else:
            print("No followers data found to recover")

    print(f"\nAnalysis completed at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
