#!/usr/bin/env python3
"""
Tweet Scorer Script
Reads tweet data from JSON files and scores them using LLM.
Creates a new JSON with text, score, username, and twitterUrl fields.

Usage:
python tweet_scorer.py input_file.json output_file.json


Not used in the project. Remove if not needed.
"""

import json
import os
import sys
import logging
import openai
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables
load_dotenv()

# Setup logging


def setup_logging(log_file='tweet_scorer.log'):
    """Setup logging to both console and file"""
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Initialize with default log file
logger = setup_logging()


class TweetScorer:
    """Class for scoring tweets using LLM"""

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the scorer"""
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)

        # Configuration
        self.config = {
            'gpt_model': 'gpt-5-mini',
            'gpt_max_tokens': 100,
            'gpt_temperature': 0.1,
            'max_workers': 5,  # Number of concurrent workers
        }

        # Thread lock for logging
        self.log_lock = threading.Lock()

    def score_tweet_with_gpt(self, tweet_text: str, target_company: str = "the company", acquire_company: str = "the company") -> Optional[str]:
        """Score tweet with GPT for importance/relevance"""
        try:
            prompt = f"""
You are evaluating a tweet for its relevance to antitrust, regulatory, competition, and deal-related insights.  
Target company: {target_company}  
Acquiring company: {acquire_company}  

Tweet: "{tweet_text}"  

Scoring criteria (1–10 scale):  
- Antitrust/Regulatory: Does it raise issues of competition, horizontal or vertical product overlap, or regulatory scrutiny?  
- Deal Importance: Does it reveal insights on mergers, acquisitions, or market strategy?  
- Business Significance: Is the user’s voice or opinion influential within the industry?  

Instructions:  
Respond with ONLY a single number from 1 to 10.  
- 1–3: Low importance/relevance  
- 4–6: Medium importance/relevance  
- 7–10: High importance/relevance  

Do not explain. Do not add text. Only output the score.
"""

            response = self.openai_client.chat.completions.create(
                model=self.config['gpt_model'],
                messages=[
                    {"role": "system", "content": "You are an expert in antitrust law and business analysis. Provide concise, accurate assessments."},
                    {"role": "user", "content": prompt}
                ],
            )

            content = response.choices[0].message.content.strip()

            logger.info(f"GPT response: {response}")

            # Validate that the response is a number between 1-10
            output_token = response.usage.completion_tokens
            input_token = response.usage.prompt_tokens

            try:
                score = int(content)
                if 1 <= score <= 10:
                    return str(score), output_token, input_token
                else:
                    logger.warning(
                        f"Score {score} is out of range 1-10, using 5 as default")
                    return "N/A", output_token, input_token
            except ValueError:
                logger.warning(
                    f"Invalid score response: {content}, using 5 as default")
                return "N/A", output_token, input_token

        except Exception as e:
            logger.error(f"Error scoring tweet with GPT: {e}")
            return "N/A", 0, 0  # Default score on error with zero tokens

    def extract_tweet_data(self, tweet_obj: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Extract relevant data from tweet object"""
        try:
            # Get tweet text
            tweet_data = tweet_obj.get('tweet', {})
            text = tweet_data.get('text', '')

            if not text:
                logger.warning("No tweet text found")
                return None

            # Get username
            username = tweet_obj.get('username', '')

            # Get Twitter URL
            twitter_url = tweet_data.get('twitterUrl', '')

            return {
                'text': text,
                'username': username,
                'twitterUrl': twitter_url
            }

        except Exception as e:
            logger.error(f"Error extracting tweet data: {e}")
            return None

    def process_single_tweet(self, tweet_obj: Dict[str, Any], target_company: str, acquire_company: str, tweet_index: int, total_tweets: int) -> Optional[Dict[str, str]]:
        """Process a single tweet - designed for thread pool execution"""
        try:
            with self.log_lock:
                logger.info(f"Processing tweet {tweet_index+1}/{total_tweets}")

            # Extract tweet data
            tweet_data = self.extract_tweet_data(tweet_obj)
            if not tweet_data:
                return None

            # Score the tweet
            score, output_token, input_token = self.score_tweet_with_gpt(
                tweet_data['text'], target_company, acquire_company)

            # Create scored tweet record
            scored_tweet = {
                'text': tweet_data['text'],
                'score': score,
                'username': tweet_data['username'],
                'twitterUrl': tweet_data['twitterUrl'],
                'output_token': output_token,
                'input_token': input_token
            }

            with self.log_lock:
                logger.info(
                    f"Scored tweet from @{tweet_data['username']}: {score if score != 'N/A' else 0}/10")

            return scored_tweet

        except Exception as e:
            with self.log_lock:
                logger.error(f"Error processing tweet {tweet_index+1}: {e}")
            return None

    def process_tweets(self, input_file: str, output_file: str, target_company: str = "the company", acquire_company: str = "the company") -> None:
        """Process tweets from input file and create scored output using ThreadPoolExecutor"""
        try:
            # Read input file
            logger.info(f"Reading tweets from {input_file}")
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            tweets = data.get('tweets', [])
            logger.info(f"Found {len(tweets)} tweets to process")

            # Limit tweets for processing
            tweet_limit = getattr(self, 'tweet_limit', 10)
            tweets_to_process = tweets[:tweet_limit]
            total_tweets = len(tweets_to_process)

            total_output_token = 0
            total_input_token = 0
            scored_tweets = []

            # Use ThreadPoolExecutor for concurrent processing
            logger.info(
                f"Starting processing with {self.config['max_workers']} workers")

            with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
                # Submit all tasks
                future_to_tweet = {
                    executor.submit(
                        self.process_single_tweet,
                        tweet_obj,
                        target_company,
                        acquire_company,
                        i,
                        total_tweets
                    ): (i, tweet_obj)
                    for i, tweet_obj in enumerate(tweets_to_process)
                }

                # Process completed tasks
                for future in as_completed(future_to_tweet):
                    tweet_index, tweet_obj = future_to_tweet[future]
                    try:
                        result = future.result()
                        if result:
                            # Extract tokens and add to totals
                            total_output_token += result.get('output_token', 0)
                            total_input_token += result.get('input_token', 0)

                            # Remove token fields from final record
                            final_record = {
                                'text': result['text'],
                                'score': result['score'],
                                'username': result['username'],
                                'twitterUrl': result['twitterUrl']
                            }
                            scored_tweets.append(final_record)

                    except Exception as e:
                        logger.error(
                            f"Error processing tweet {tweet_index+1}: {e}")

            # Sort scored tweets by original order
            scored_tweets.sort(key=lambda x: tweets_to_process.index(
                next(t for t in tweets_to_process if t.get(
                    'username') == x['username'])
            ) if any(t.get('username') == x['username'] for t in tweets_to_process) else 0)

            # Count score distribution
            score_counts = {}
            for tweet in scored_tweets:
                score = tweet['score']
                score_counts[score] = score_counts.get(score, 0) + 1

            # Log score distribution
            logger.info("Score distribution:")
            for score in sorted(score_counts.keys()):
                count = score_counts[score]
                logger.info(f"  Score {score}: {count} tweets")

            # Create output data
            output_data = {
                'metadata': {
                    'processed_at': datetime.now().isoformat(),
                    'total_tweets': len(scored_tweets),
                    'target_company': target_company,
                    'acquire_company': acquire_company,
                    'source_file': input_file,
                    'total_output_token': total_output_token,
                    'total_input_token': total_input_token,
                    'gpt_model': self.config['gpt_model'],
                    'max_workers': self.config['max_workers'],
                    'score_distribution': score_counts
                },
                'records': scored_tweets
            }

            # Write output file
            logger.info(f"Writing scored tweets to {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Successfully processed {len(scored_tweets)} tweets using {self.config['max_workers']} workers")

        except Exception as e:
            logger.error(f"Error processing tweets: {e}")
            raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Score tweets using LLM')
    parser.add_argument('--input_file', default='juniper.json',
                        help='Input JSON file containing tweets')
    parser.add_argument('--output_file', default='Juniper_scored.json',
                        help='Output JSON file for scored tweets')
    parser.add_argument('--target_company', default='Hewlett Packard Enterprise Company',
                        help='Target company name for context')
    parser.add_argument('--acquire_company', default='Juniper Networks, Inc.',
                        help='Acquiring company name for context')
    parser.add_argument('--workers', type=int, default=25,
                        help='Number of concurrent workers (default: 5)')
    parser.add_argument('--limit', type=int, default=500,
                        help='Limit number of tweets to process (default: 10)')
    parser.add_argument('--log_file', default='tweet_scorer.log',
                        help='Log file name (default: tweet_scorer.log)')

    args = parser.parse_args()

    target_company = args.target_company
    acquire_company = args.acquire_company
    input_file = args.input_file
    output_file = args.output_file
    max_workers = args.workers
    tweet_limit = args.limit
    log_file = args.log_file

    # Reinitialize logger with custom log file
    global logger
    logger = setup_logging(log_file)

    # Validate input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} does not exist")
        sys.exit(1)

    try:
        # Initialize scorer
        scorer = TweetScorer()

        # Update max workers from command line
        scorer.config['max_workers'] = max_workers

        # Update tweet limit
        scorer.tweet_limit = tweet_limit

        # Process tweets
        scorer.process_tweets(input_file, output_file,
                              target_company, acquire_company)

        logger.info("Tweet scoring completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
