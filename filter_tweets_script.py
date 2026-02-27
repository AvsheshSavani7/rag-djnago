#!/usr/bin/env python3
"""
Script to filter tweets from a merged JSON file and create a new file with max 1000 tweets
collected from different users.

Only for testing purposes.Remove if not needed.
"""

import json
import sys
from datetime import datetime
from typing import Dict, List, Any


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load and parse the JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def collect_tweets_from_users(data: Dict[str, Any], max_tweets: int = 1000) -> List[Dict[str, Any]]:
    """
    Collect the first max_tweets from the dataset, maintaining the original order.
    """
    users = data.get('users', {})
    all_tweets = []

    # Collect all tweets with user information in the order they appear
    for username, user_data in users.items():
        tweets = user_data.get('tweets', [])
        user_info = user_data.get('user_info', {})

        for tweet in tweets:
            # Add user context to each tweet
            tweet_with_user = {
                'tweet': tweet,
                'user_info': user_info,
                'username': username
            }
            all_tweets.append(tweet_with_user)

    # Return the first max_tweets (or all if we have fewer)
    return all_tweets[:max_tweets]


def create_filtered_json(original_data: Dict[str, Any], selected_tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a new JSON structure with the filtered tweets."""

    # Create new structure
    filtered_data = {
        "metadata": {
            "original_processed_at": original_data.get('metadata', {}).get('processed_at', ''),
            "filtered_at": datetime.now().isoformat(),
            "original_total_users": original_data.get('metadata', {}).get('total_users', 0),
            "original_total_tweets": original_data.get('metadata', {}).get('total_tweets_all_users', 0),
            "filtered_total_tweets": len(selected_tweets),
            "filtered_total_users": len(set(tweet['username'] for tweet in selected_tweets)),
            "max_tweets_limit": 1000
        },
        "tweets": []
    }

    # Add tweets to the new structure
    for tweet_data in selected_tweets:
        filtered_data["tweets"].append({
            "username": tweet_data['username'],
            "user_info": tweet_data['user_info'],
            "tweet": tweet_data['tweet']
        })

    return filtered_data


def save_json_file(data: Dict[str, Any], output_path: str) -> None:
    """Save the filtered data to a new JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        print(f"Successfully saved filtered tweets to: {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
        sys.exit(1)


def main():
    """Main function to execute the tweet filtering process."""
    if len(sys.argv) != 2:
        print("Usage: python filter_tweets_script.py <input_json_file>")
        print("Example: python filter_tweets_script.py Juniper_Networks_merged_tweets_20250904_145830.json")
        sys.exit(1)

    input_file = sys.argv[1]

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"filtered_tweets_{timestamp}.json"

    print(f"Loading tweets from: {input_file}")

    # Load the original data
    original_data = load_json_file(input_file)

    print(f"Original data contains:")
    print(
        f"  - Total users: {original_data.get('metadata', {}).get('total_users', 0)}")
    print(
        f"  - Total tweets: {original_data.get('metadata', {}).get('total_tweets_all_users', 0)}")

    # Collect the first 1000 tweets
    print("Filtering to the first 1000 tweets...")
    selected_tweets = collect_tweets_from_users(original_data, max_tweets=1000)

    print(
        f"Selected {len(selected_tweets)} tweets from {len(set(tweet['username'] for tweet in selected_tweets))} users")

    # Create the filtered JSON structure
    filtered_data = create_filtered_json(original_data, selected_tweets)

    # Save to new file
    save_json_file(filtered_data, output_file)

    print(f"\nFiltering complete!")
    print(f"Output file: {output_file}")
    print(f"Filtered tweets: {len(selected_tweets)}")
    print(
        f"Users represented: {len(set(tweet['username'] for tweet in selected_tweets))}")


if __name__ == "__main__":
    main()
