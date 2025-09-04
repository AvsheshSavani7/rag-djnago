# Deal Reddit Scraper

This script scrapes Reddit discussions for competitive products associated with a specific deal ID.

## Overview

The `deal_reddit_scraper.py` script takes a `deal_id` as input and:

1. **Fetches Deal Data**: Retrieves deal information from the MongoDB database
2. **Gets Competitive Products**: Fetches competitive product pairs for the deal
3. **Scrapes Reddit**: Searches and scrapes Reddit discussions for each competitive product pair
4. **Saves Results**: Outputs comprehensive JSON files with all scraped data

## Prerequisites

### Required Python Packages
```bash
pip install serpapi praw python-dotenv django mongoengine
```

### Environment Variables
Make sure your `.env` file contains:
```
MONGODB_CONNECTION_STRING=your_mongodb_connection_string
MONGODB_NAME=your_database_name
```

### API Keys
The script uses the following APIs (configured in the script):
- **SerpAPI**: For Google search to find Reddit links
- **Reddit API**: For scraping Reddit posts and comments

## Usage

### Command Line
```bash
cd rag_project/document_processor/reddit_utils
python deal_reddit_scraper.py <deal_id>
```

### Example
```bash
python deal_reddit_scraper.py 507f1f77bcf86cd799439011
```

## Output

### Individual Competition Files
For each competitive product pair, the script creates:
```
deal_reddit_analysis/deal_{deal_id}_{product1}_vs_{product2}.json
```

### Final Consolidated File
```
deal_reddit_analysis/deal_{deal_id}_reddit_analysis.json
```

### Output Structure
```json
{
  "deal_id": "507f1f77bcf86cd799439011",
  "analysis_timestamp": "2025-01-XX...",
  "total_competitions": 5,
  "deduplication_stats": {
    "original_total_posts": 25,
    "deduplicated_total_posts": 20,
    "duplicates_removed": 5,
    "unique_post_ids": 20
  },
  "results": [
    {
      "competition": "Product A vs Product B",
      "deal_id": "507f1f77bcf86cd799439011",
      "search_query": "Product A vs Product B site:reddit.com",
      "total_posts_found": 4,
      "posts": [
        {
          "id": "post_id",
          "title": "Post Title",
          "author": "username",
          "score": 15,
          "url": "https://reddit.com/...",
          "selftext": "Post content...",
          "num_comments": 8,
          "created_utc": 1640995200,
          "comments": [
            {
              "id": "comment_id",
              "author": "commenter",
              "body": "Comment text...",
              "score": 3,
              "created_utc": 1640995300,
              "replies": []
            }
          ]
        }
      ],
      "timestamp": "2025-01-XX..."
    }
  ]
}
```

## Features

### 🔍 **Smart Search**
- Uses SerpAPI to find relevant Reddit discussions
- Searches for competitive product comparisons
- Filters for Reddit-specific content

### 📊 **Comprehensive Data**
- Scrapes post titles, content, scores, and metadata
- Extracts all comments and nested replies
- Includes author information and timestamps

### 🧹 **Deduplication**
- Removes duplicate posts across different competitions
- Tracks deduplication statistics
- Ensures unique content in final results

### 📁 **Organized Output**
- Individual files for each competition
- Consolidated final report
- Detailed logging and progress tracking

### ⚡ **Error Handling**
- Graceful handling of API failures
- Detailed error logging
- Continues processing even if some competitions fail

## Database Integration

The script integrates with the existing MongoDB database structure:

- **ProcessingJob**: Fetches deal information
- **CompetitiveAnalysis**: Retrieves competitive product pairs
- **CompanyProducts**: Gets product information for companies

## Logging

The script provides comprehensive logging:
- Console output with progress updates
- Detailed log file: `deal_reddit_scraper.log`
- Error tracking and debugging information

## Limitations

- **Rate Limits**: Respects Reddit API rate limits
- **Post Limit**: Scrapes maximum 5 posts per competition
- **Comment Depth**: Limited comment thread depth for performance
- **API Dependencies**: Requires valid SerpAPI and Reddit API credentials

## Troubleshooting

### Common Issues

1. **"Deal not found"**: Verify the deal_id exists in the database
2. **"No competitive products"**: Ensure competitive analysis has been run for the deal
3. **API errors**: Check API keys and rate limits
4. **Import errors**: Install required Python packages

### Debug Mode
Check the log file for detailed error information:
```bash
tail -f deal_reddit_scraper.log
```

## Example Workflow

1. **Run competitive analysis** on a deal to generate product pairs
2. **Execute the scraper** with the deal ID
3. **Review results** in the output directory
4. **Analyze data** for market insights and user sentiment

This script is particularly useful for:
- Market research and competitive analysis
- Understanding user sentiment about products
- Gathering real-world product comparisons
- Building datasets for further analysis
