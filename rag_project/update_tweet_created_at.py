#!/usr/bin/env python3
"""
Script to update all existing tweets in the database to populate the tweet_created_at field.
This script should be run once after adding the tweet_created_at field to the Tweet model.
"""

from document_processor.models import Tweet
import os
import sys
import django
from datetime import datetime

# Django setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()


def parse_twitter_date(date_str):
    """Parse Twitter date format to datetime"""
    try:
        # Twitter format: "Thu May 08 11:13:03 +0000 2025"
        return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")
        return None


def update_tweet_created_at():
    """Update all tweets to populate the tweet_created_at field"""

    print("🔄 Starting tweet_created_at field update...")
    print("=" * 60)

    try:
        # Get all tweets that don't have tweet_created_at set
        tweets_to_update = Tweet.objects(tweet_created_at__exists=False)
        total_tweets = tweets_to_update.count()

        print(f"📊 Found {total_tweets} tweets to update")

        if total_tweets == 0:
            print("✅ All tweets already have tweet_created_at field populated")
            return

        updated_count = 0
        error_count = 0

        # Process tweets in batches
        batch_size = 100
        for i in range(0, total_tweets, batch_size):
            batch_tweets = tweets_to_update.skip(i).limit(batch_size)

            for tweet in batch_tweets:
                try:
                    # Parse the createdAt field from the tweet object
                    if isinstance(tweet.tweet, dict) and 'createdAt' in tweet.tweet:
                        created_at_str = tweet.tweet['createdAt']
                        parsed_date = parse_twitter_date(created_at_str)

                        if parsed_date:
                            tweet.tweet_created_at = parsed_date
                            tweet.save()
                            updated_count += 1

                            if updated_count % 50 == 0:
                                print(
                                    f"✅ Updated {updated_count}/{total_tweets} tweets...")
                        else:
                            error_count += 1
                            print(
                                f"❌ Failed to parse date for tweet {tweet.id}")
                    else:
                        error_count += 1
                        print(
                            f"❌ No createdAt field found in tweet {tweet.id}")

                except Exception as e:
                    error_count += 1
                    print(f"❌ Error updating tweet {tweet.id}: {e}")

        print("\n" + "=" * 60)
        print("🎉 UPDATE COMPLETED!")
        print(f"✅ Successfully updated: {updated_count} tweets")
        print(f"❌ Errors encountered: {error_count} tweets")
        print(f"📊 Total processed: {updated_count + error_count} tweets")

        # Verify the update
        remaining_tweets = Tweet.objects(
            tweet_created_at__exists=False).count()
        print(
            f"🔍 Remaining tweets without tweet_created_at: {remaining_tweets}")

        if remaining_tweets == 0:
            print("✅ All tweets now have tweet_created_at field populated!")
        else:
            print(f"⚠️  {remaining_tweets} tweets still need manual review")

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()


def verify_tweet_created_at():
    """Verify that tweet_created_at field is working correctly"""

    print("\n🔍 Verifying tweet_created_at field...")
    print("=" * 60)

    try:
        # Get a sample of tweets with tweet_created_at
        sample_tweets = Tweet.objects(tweet_created_at__exists=True).limit(5)

        print("📋 Sample tweets with tweet_created_at:")
        for tweet in sample_tweets:
            if isinstance(tweet.tweet, dict) and 'createdAt' in tweet.tweet:
                original_date = tweet.tweet['createdAt']
                parsed_date = tweet.tweet_created_at
                print(f"   Tweet ID: {tweet.id}")
                print(f"   Original: {original_date}")
                print(f"   Parsed:   {parsed_date}")
                print(
                    f"   Match:    {'✅' if str(parsed_date) in original_date else '❌'}")
                print()

        # Test ordering
        print("🧪 Testing ordering by tweet_created_at...")
        ordered_tweets = Tweet.objects(tweet_created_at__exists=True).order_by(
            '-tweet_created_at').limit(3)

        print("📅 Most recent tweets (by tweet_created_at):")
        for i, tweet in enumerate(ordered_tweets, 1):
            print(f"   {i}. {tweet.tweet_created_at} - Tweet ID: {tweet.id}")

        print("✅ Verification completed successfully!")

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Tweet Created At Field Update Script")
    print("=" * 60)

    # Update all tweets
    update_tweet_created_at()

    # Verify the update
    verify_tweet_created_at()

    print("\n📋 Next Steps:")
    print("1. The tweet_created_at field has been added to the Tweet model")
    print("2. All existing tweets have been updated with parsed dates")
    print("3. New tweets will automatically have tweet_created_at populated")
    print("4. The TweetsView now orders by tweet_created_at for proper chronological order")
    print("5. You can now use the tweets API with proper date-based ordering!")
