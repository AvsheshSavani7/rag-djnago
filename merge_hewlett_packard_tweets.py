#!/usr/bin/env python3
"""
Script to merge Hewlett Packard Enterprise Company tweet data user-wise.

This script:
1. Reads all tweet JSON files from the Hewlett Packard directory
2. Extracts usernames from filenames
3. Merges all tweets for each user into a single structure
4. Counts total tweets per user
5. Provides a summary of all tweet counts

Only for merge tweets from directory:
/Users/joshuatackel/Documents/RAG_BE/rag_project/document_processor/twitter_utils/high_value_followers_tweet_results_test/Hewlett Packard Enterprise Company

you can delete this file if you are not using it.
"""

import json
import os
import re
from collections import defaultdict
from datetime import datetime


def extract_username_from_filename(filename):
    """
    Extract username from filename like '1MikeyT_tweets.json'
    Returns the username part before '_tweets.json'
    """
    # Remove the '_tweets.json' suffix
    username = filename.replace('_tweets.json', '')
    return username


def read_tweet_file(file_path):
    """
    Read and parse a tweet JSON file.
    Returns the list of tweets or empty list if file is invalid.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
        print(f"Error reading {file_path}: {e}")
        return []


def process_hewlett_packard_tweets():
    """
    Main function to process all Hewlett Packard tweet files.
    """
    # Define the directory path
    base_dir = "/Users/joshuatackel/Documents/RAG_BE/rag_project/document_processor/twitter_utils/high_value_followers_tweet_results_test/Hewlett Packard Enterprise Company"

    # Dictionary to store user data
    user_data = defaultdict(lambda: {
        'username': '',
        'tweets': [],
        'total_tweets': 0,
        'user_info': None
    })

    # Statistics
    total_files_processed = 0
    total_tweets_all_users = 0
    files_with_errors = []

    print("Starting to process Hewlett Packard Enterprise Company tweet files...")
    print(f"Directory: {base_dir}")
    print("-" * 60)

    # Check if directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist!")
        return

    # Get all JSON files in the directory
    try:
        files = [f for f in os.listdir(base_dir) if f.endswith('_tweets.json')]
        print(f"Found {len(files)} tweet files to process")
        print("-" * 60)

        for filename in files:
            file_path = os.path.join(base_dir, filename)
            username = extract_username_from_filename(filename)

            print(f"Processing: {filename} -> Username: {username}")

            # Read tweets from file
            tweets = read_tweet_file(file_path)

            if tweets:
                # Store user info from first tweet (if available)
                if tweets and 'author' in tweets[0]:
                    user_data[username]['user_info'] = tweets[0]['author']

                # Add tweets to user data
                user_data[username]['username'] = username
                user_data[username]['tweets'].extend(tweets)
                user_data[username]['total_tweets'] = len(
                    user_data[username]['tweets'])

                total_tweets_all_users += len(tweets)
                print(
                    f"  -> Added {len(tweets)} tweets (Total for user: {user_data[username]['total_tweets']})")
            else:
                files_with_errors.append(filename)
                print(f"  -> No tweets found or error reading file")

            total_files_processed += 1

        print("-" * 60)
        print("PROCESSING COMPLETE")
        print("-" * 60)

        # Create final merged data structure
        merged_data = {
            'metadata': {
                'processed_at': datetime.now().isoformat(),
                'total_files_processed': total_files_processed,
                'total_users': len(user_data),
                'total_tweets_all_users': total_tweets_all_users,
                'files_with_errors': files_with_errors
            },
            'users': {}
        }

        # Add user data to merged structure
        for username, data in user_data.items():
            merged_data['users'][username] = {
                'username': data['username'],
                'total_tweets': data['total_tweets'],
                'user_info': data['user_info'],
                'tweets': data['tweets']
            }

        # Generate summary
        print("SUMMARY:")
        print(f"Total files processed: {total_files_processed}")
        print(f"Total users found: {len(user_data)}")
        print(f"Total tweets across all users: {total_tweets_all_users}")
        print(f"Files with errors: {len(files_with_errors)}")

        if files_with_errors:
            print(f"Files with errors: {', '.join(files_with_errors)}")

        print("\nTOP 10 USERS BY TWEET COUNT:")
        print("-" * 40)

        # Sort users by tweet count
        sorted_users = sorted(
            user_data.items(), key=lambda x: x[1]['total_tweets'], reverse=True)

        for i, (username, data) in enumerate(sorted_users[:10], 1):
            user_name = data['user_info']['name'] if data['user_info'] and 'name' in data['user_info'] else 'N/A'
            print(
                f"{i:2d}. {username:20s} ({user_name:30s}) - {data['total_tweets']:4d} tweets")

        # Save merged data to file
        output_filename = f"hewlett_packard_merged_tweets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = os.path.join(
            "/Users/joshuatackel/Documents/RAG_BE", output_filename)

        print(f"\nSaving merged data to: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)

        print(f"Successfully saved merged data to {output_filename}")

        # Also create a summary-only file
        summary_filename = f"hewlett_packard_tweet_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_path = os.path.join(
            "/Users/joshuatackel/Documents/RAG_BE", summary_filename)

        summary_data = {
            'metadata': merged_data['metadata'],
            'user_summary': {}
        }

        for username, data in user_data.items():
            summary_data['user_summary'][username] = {
                'username': data['username'],
                'total_tweets': data['total_tweets'],
                'user_name': data['user_info']['name'] if data['user_info'] and 'name' in data['user_info'] else 'N/A',
                'followers': data['user_info']['followers'] if data['user_info'] and 'followers' in data['user_info'] else 0,
                'following': data['user_info']['following'] if data['user_info'] and 'following' in data['user_info'] else 0
            }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        print(f"Successfully saved summary data to {summary_filename}")

    except Exception as e:
        print(f"Error processing directory: {e}")


if __name__ == "__main__":
    process_hewlett_packard_tweets()
