# User Tweet Search Scripts

This directory contains scripts for searching tweets from specific users mentioning Juniper Networks or their products.

## Files

- `user_tweet_search.py` - **Main script** for searching tweets from all users with ThreadPoolExecutor (requires API key)
- `test_user_tweet_search_threaded.py` - **Test version** with ThreadPoolExecutor for concurrent processing
- `test_user_tweet_search_simple.py` - Simple test version that processes only 5 users
- `filtered_followers_JuniperNetworks.json` - Input file containing user data
- `product.json` - Input file containing company products

## Prerequisites

1. **Twitter API Key**: You need a Twitter API key from [twitterapi.io](https://twitterapi.io)
2. **Environment Variable**: Set your API key as an environment variable:
   ```bash
   export TWITTER_API_KEY="your_api_key_here"
   ```

## Usage Options

### Option 1: Test Threaded Version (Recommended First)

Run the threaded test version to verify concurrent processing works:

```bash
cd rag_project/document_processor/twitter_utils
python test_user_tweet_search_threaded.py
```

This will:
- Ask for number of concurrent threads (default: 3)
- Ask for number of users to process (default: 10)
- Process users concurrently using ThreadPoolExecutor
- Simulate search results (no API calls)
- Save results to `user_tweet_search_results/threaded_test_search_*.json`

### Option 2: Simple Test Version

Run the simple test version for quick verification:

```bash
cd rag_project/document_processor/twitter_utils
python test_user_tweet_search_simple.py
```

This will:
- Process only 5 users from the followers file
- Simulate search results (no API calls)
- Save results to `user_tweet_search_results/simulated_user_tweet_search_*.json`

### Option 3: Full Version with API Key and Threading

Once you have your API key set up, run the full threaded version:

```bash
cd rag_project/document_processor/twitter_utils
python user_tweet_search.py
```

This will:
- Ask for number of concurrent threads (default: 5)
- Ask for number of users to process (or ALL users)
- Process users concurrently using ThreadPoolExecutor
- Make real API calls to Twitter
- Save results to `user_tweet_search_results/user_tweet_search_*.json`

## Search Query Structure

The script builds Twitter search queries using the following format:

```
from:username ("Juniper Networks, Inc." OR "Junos OS" OR "Junos Space" OR "Contrail Networking") since:2019-01-09_00:00:00_UTC until:2024-01-09_23:59:59_UTC lang:en
```

This searches for:
- Tweets from specific users (`from:username`)
- Mentions of the company name OR any of the first 3 products
- Within the specified date range (2019-01-09 to 2024-01-09)
- In English language only

## Output Format

The script saves results in JSON format with the following structure:

```json
{
  "company_name": "Juniper Networks, Inc.",
  "products_searched": ["Junos OS", "Junos Space", "Contrail Networking"],
  "total_users_processed": 1824,
  "users_with_tweets": 45,
  "total_tweets_found": 1234,
  "search_timestamp": "2024-01-09T10:30:00.000000",
  "user_results": {
    "username1": {
      "username": "username1",
      "company_name": "Juniper Networks, Inc.",
      "products_searched": ["Junos OS", "Junos Space", "Contrail Networking"],
      "search_query": "from:username1 (\"Juniper Networks, Inc.\" OR \"Junos OS\" OR \"Junos Space\" OR \"Contrail Networking\") since:2019-01-09_00:00:00_UTC until:2024-01-09_23:59:59_UTC lang:en",
      "tweet_count": 5,
      "tweets": [
        {
          "id": "1234567890",
          "text": "Great experience with Juniper Networks Junos OS...",
          "created_at": "2023-12-15T10:30:00.000Z",
          "user": {
            "screen_name": "username1"
          }
        }
      ],
      "search_timestamp": "2024-01-09T10:30:00.000000"
    }
  }
}
```

## Threading and Performance

The scripts use ThreadPoolExecutor for concurrent processing:
- **Concurrent Processing**: Multiple users are processed simultaneously
- **Configurable Threads**: You can set the number of concurrent threads (default: 5 for main, 3 for test)
- **Thread-Safe Counters**: Progress tracking is thread-safe
- **Efficient Resource Usage**: While one thread waits for API response, others continue processing

## Rate Limiting

The script includes rate limiting to avoid hitting API limits:
- 2 seconds between API calls for the same user
- No delay between different users (handled by threading)
- Maximum 1000 tweets per user (100 for test version)

## Error Handling

The script handles various errors:
- Missing API key (runs in simulation mode)
- Missing input files
- API request failures
- Invalid user data
- JSON parsing errors

All errors are logged and the script continues processing other users.

## Getting a Twitter API Key

1. Go to [twitterapi.io](https://twitterapi.io)
2. Sign up for an account
3. Get your API key from the dashboard
4. Set it as an environment variable:
   ```bash
   export TWITTER_API_KEY="your_api_key_here"
   ```

## Testing Without API Key

If you don't have an API key, you can still test the functionality:

1. **Run the no-API version**: `python user_tweet_search_no_api.py`
2. **Run the simple test**: `python test_user_tweet_search_simple.py`

Both will simulate the search results and save them to JSON files for manual testing.

## Customization

You can modify the scripts to:
- Change the date range by modifying the `start_date` and `end_date` parameters
- Search for different companies by changing the `company_name` parameter
- Adjust the number of products to search (currently limited to first 3)
- Modify the maximum number of tweets per user
- Change the rate limiting delays
- Adjust the number of concurrent threads for better performance

## Notes

- The scripts do NOT save anything to MongoDB (as requested)
- Results are saved only to JSON files for manual testing
- The scripts use the Twitter Advanced Search API from twitterapi.io
- Search queries follow Twitter's advanced search syntax as documented in the [Twitter Advanced Search guide](https://github.com/igorbrigadir/twitter-advanced-search)

## Troubleshooting

### "Twitter API key is required" Error
- Set the environment variable: `export TWITTER_API_KEY="your_key_here"`
- Or use the no-API version: `python user_tweet_search_no_api.py`

### "Followers file not found" Error
- Make sure `filtered_followers_JuniperNetworks.json` exists in the same directory

### "Products file not found" Error
- Make sure `product.json` exists in the same directory

### API Rate Limiting
- The script includes built-in rate limiting
- If you hit rate limits, the script will wait and retry
- Consider reducing the number of users processed for testing
