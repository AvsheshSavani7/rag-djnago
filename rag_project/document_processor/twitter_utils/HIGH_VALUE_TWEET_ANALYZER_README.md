# High Value Followers Tweet Analyzer

## Overview

The High Value Followers Tweet Analyzer is a comprehensive tool that reads high-value followers from the database and searches for their tweets about company/products from the past 5 years. It then verifies each tweet with GPT to determine if it's related to antitrust concerns, business importance, or significant user voice.

## Features

### 1. High-Value Follower Processing
- **Database Integration**: Reads high-value followers from the `HighValueFollowers` collection
- **Company-Specific Processing**: Processes followers for both target and acquire companies separately
- **Score-Based Filtering**: Works with followers identified by previous analysis steps

### 2. Tweet Search with Cursor Pagination
- **Advanced Search API**: Uses Twitter Advanced Search API with cursor pagination
- **Comprehensive Queries**: Searches for company names, handles, and products
- **Date Range**: Searches tweets from the last 5 years
- **Language Filter**: Focuses on English tweets only

### 3. GPT Analysis for Tweet Importance
- **Antitrust Focus**: Analyzes tweets for antitrust concerns and regulatory issues
- **Business Intelligence**: Identifies tweets important from a business perspective
- **User Voice Assessment**: Evaluates if the user's voice is significant in the industry
- **Structured Response**: Returns Yes/No with detailed reasoning

### 4. Database Storage
- **SearchQuery Records**: Creates search query records for each follower analysis
- **Tweet Storage**: Saves important tweets to the `Tweet` collection
- **Comprehensive Metadata**: Stores analysis results and GPT reasoning

## Data Flow

### 1. High-Value Follower Retrieval
```python
# Get high-value followers from database
followers = analyzer.get_high_value_followers(deal_id)
```

### 2. Tweet Search
```python
# Build search query for each follower
query = analyzer.build_search_query(username, company_name, products, company_handle)

# Search tweets with cursor pagination
tweets = analyzer.search_tweets_with_cursor(query, max_tweets=100)
```

### 3. GPT Analysis
```python
# Analyze each tweet with GPT
gpt_analysis = analyzer.analyze_tweet_with_gpt(tweet_text, company_name)
# Returns: {"important": "Yes", "reason": "Brief explanation"}
```

### 4. Database Storage
```python
# Save important tweets to database
if gpt_analysis.get('important') == 'Yes':
    saved_tweet_id = analyzer.save_important_tweet(tweet, search_query_id, gpt_analysis)
```

## Usage

### Command Line Usage

```bash
# Basic usage
python high_value_followers_tweet_analyzer.py 68184d52478abf06ec1a28ec

# With custom configuration
python high_value_followers_tweet_analyzer.py 68184d52478abf06ec1a28ec \
  --workers 10 \
  --max-tweets 200 \
  --gpt-model gpt-4o-mini
```

### Runner Script Usage

```bash
# Using the runner script
python run_high_value_followers_tweet_analyzer.py 68184d52478abf06ec1a28ec
```

### Orchestrator Integration

```bash
# Run as Step 4 in the orchestrator
python high_value_followers_orchestrator.py 68184d52478abf06ec1a28ec

# Disable other steps, run only tweet analysis
python high_value_followers_orchestrator.py 68184d52478abf06ec1a28ec \
  --disable-step-1 --disable-step-2 --disable-step-3
```

## Configuration Options

### Tweet Search Settings
- `search_date_range_years`: Years back to search (default: 5)
- `max_tweets_per_user`: Maximum tweets per follower (default: 100)
- `search_timeout`: API request timeout (default: 10 seconds)

### GPT Analysis Settings
- `gpt_model`: GPT model to use (default: gpt-4o-mini)
- `gpt_max_tokens`: Maximum tokens for response (default: 150)
- `gpt_temperature`: Response creativity (default: 0.1)

### Concurrency Settings
- `max_workers`: Number of parallel workers (default: 5)
- `use_parallel_processing`: Enable/disable parallel processing (default: True)
- `delay_between_requests`: Delay between API calls (default: 0.5 seconds)

## GPT Analysis Prompt

The analyzer uses a specialized prompt to evaluate tweets:

```
Analyze this tweet about {company_name} for antitrust and business importance.

Tweet: "{tweet_text}"

Determine if this tweet is:
1. Related to antitrust concerns, regulatory issues, or competition
2. Important from a business perspective (market analysis, strategic insights, etc.)
3. From a user whose voice/opinion is significant in the industry

Respond in this exact JSON format:
{
    "important": "Yes" or "No",
    "reason": "Brief explanation in 50 words or less"
}
```

## Output Files

### Summary JSON
- **File**: `high_value_tweet_analysis_summary_{deal_id}_{timestamp}.json`
- **Content**: Processing statistics, company-wise results, configuration used

### Detailed Results JSON
- **File**: `high_value_tweet_analysis_detailed_{deal_id}_{timestamp}.json`
- **Content**: Complete analysis results with tweet details and GPT reasoning

### Database Records
- **SearchQuery**: Records for each follower's search query
- **Tweet**: Individual important tweets with metadata

## Example Output

### Console Output
```
2024-01-15 10:30:00 - Starting High Value Followers Tweet Analysis for deal ID: 68184d52478abf06ec1a28ec
2024-01-15 10:30:01 - Found target Twitter handle: @targetcompany
2024-01-15 10:30:01 - Found acquire Twitter handle: @acquirecompany
2024-01-15 10:30:02 - Found 127 high-value followers
2024-01-15 10:30:03 - Processing high-value followers for Target Company (@targetcompany)
2024-01-15 10:30:03 - Found 65 high-value followers for Target Company
2024-01-15 10:30:04 - Analyzing tweets for 1/65: @user1
2024-01-15 10:30:05 - ✓ Found 3 important tweets for @user1
...
2024-01-15 11:45:00 - === High Value Followers Tweet Analysis Complete ===
2024-01-15 11:45:00 - Deal ID: 68184d52478abf06ec1a28ec
2024-01-15 11:45:00 - Total companies processed: 2
2024-01-15 11:45:00 - Total important tweets found: 45
```

### Summary JSON Structure
```json
{
  "deal_id": "68184d52478abf06ec1a28ec",
  "processing_timestamp": "2024-01-15T11:45:00.123456",
  "approach": "HIGH_VALUE_TWEET_ANALYSIS",
  "total_companies": 2,
  "summary": {
    "total_high_value_followers": 127,
    "followers_with_tweets": 89,
    "total_tweets_found": 1247,
    "important_tweets_found": 45
  },
  "company_results": [
    {
      "company_name": "Target Company",
      "company_handle": "targetcompany",
      "total_followers": 65,
      "followers_with_tweets": 45,
      "total_tweets_found": 623,
      "important_tweets_found": 23,
      "processing_status": "completed"
    }
  ]
}
```

## Prerequisites

### Required Data
- Deal must exist in ProcessingJob collection
- Deal must have Twitter details with company handles
- High-value followers must exist in HighValueFollowers collection
- Company products must exist in CompanyProducts collection

### Dependencies
- Twitter API key (twitterapi.io)
- OpenAI API key
- MongoDB connection
- Django setup

## Error Handling

### Graceful Failures
- Continues processing if one follower fails
- Logs detailed error messages
- Saves partial results
- Rate limiting and retry logic

### Common Issues
- Missing API keys: Clear error messages
- No high-value followers: Logs warning and continues
- GPT API errors: Logs error and skips tweet
- Database errors: Logs error and continues

## Performance Considerations

### Rate Limiting
- Built-in delays between API calls
- Configurable rate limiting parameters
- Respects Twitter and OpenAI API limits

### Parallel Processing
- Configurable number of workers
- Thread-safe counters and logging
- Efficient resource usage

### Memory Management
- Processes followers one by one
- Saves to database immediately
- No large data structures in memory

## Integration with Orchestrator

The tweet analyzer is integrated as **Step 4** in the orchestrator:

1. **Step 1**: Gun Shot Approach - Fetch all followers
2. **Step 2**: GPT Analysis - Analyze followers with descriptions
3. **Step 3**: Tweet Search - Analyze followers without descriptions
4. **Step 4**: Tweet Analysis - Analyze tweets from high-value followers

### Orchestrator Configuration
```python
config_overrides = {
    'enable_step_4_tweet_analysis': True,
    'tweet_analysis_max_tweets_per_user': 100,
    'tweet_analysis_max_workers': 5,
    'tweet_analysis_gpt_model': 'gpt-4o-mini'
}
```

## Troubleshooting

### Common Issues
1. **No high-value followers found**: Ensure previous steps completed successfully
2. **Missing API keys**: Verify TWITTER_API_KEY and OPENAI_API_KEY_SEC_FILING environment variables
3. **Rate limiting**: Reduce worker count or increase delays
4. **Database errors**: Check MongoDB connection and permissions

### Debug Mode
Enable detailed logging by modifying the logging configuration in the script.

## Future Enhancements

### Potential Improvements
- Batch processing for better performance
- Additional filtering criteria
- Custom GPT prompts per use case
- Export to Excel/CSV
- Real-time processing with WebSockets
- Sentiment analysis integration
- Network analysis of tweet interactions
