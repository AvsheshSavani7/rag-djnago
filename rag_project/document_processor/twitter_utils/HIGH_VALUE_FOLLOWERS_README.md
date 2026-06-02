# High Value Followers Processor

## Overview

The High Value Followers Processor is a comprehensive tool that combines follower fetching, filtering, and GPT analysis to identify and store high-value Twitter followers for M&A deals. It processes followers from both target and acquire companies, filters them based on quality criteria, analyzes their profiles using GPT, and saves high-value followers to a dedicated MongoDB collection.

## Features

### 1. Deal-Based Processing
- **Deal ID Input**: Accepts a deal ID to fetch all relevant company data
- **Twitter Handle Extraction**: Automatically extracts official Twitter handles for target and acquire companies
- **Separate Processing**: Processes each company's followers independently

### 2. Follower Filtering
- **Quality Criteria**: Filters followers based on:
  - Minimum followers count (default: 10)
  - Minimum statuses count (default: 20)
  - Non-protected accounts
  - Non-empty descriptions
- **Configurable Thresholds**: All filtering criteria can be customized

### 3. GPT Analysis
- **Profile Analysis**: Uses GPT-3.5 Turbo to analyze follower bios
- **Relevance Scoring**: Provides 0-10 overall relevance score
- **Key Indicators**: Extracts key phrases and indicators from bios
- **Business Intelligence Focus**: Specialized for antitrust, M&A, and corporate affairs

### 4. Database Storage
- **HighValueFollowers Collection**: Dedicated MongoDB collection for high-value followers
- **Individual Records**: Each follower is stored as a separate document
- **Comprehensive Data**: Includes all follower details plus GPT analysis results
- **Real-time Saving**: Saves each follower immediately after GPT analysis

## Data Flow

### 1. Deal Data Retrieval
```python
# Fetch deal with Twitter details
deal_data = processor.fetch_deal_data(deal_id)
```

### 2. Twitter Handle Extraction
```python
# Extract official Twitter handles
twitter_handles = processor.extract_twitter_handles(deal_data)
# Returns: {'Target Company': 'target_handle', 'Acquire Company': 'acquire_handle'}
```

### 3. Follower Retrieval
```python
# Get followers from existing GUNSHOT approach data
followers = follower_utils.get_followers_for_company(deal_id, company_handle, "GUNSHOT")
```

### 4. Follower Filtering
```python
# Filter based on quality criteria
filtered_followers = processor.filter_followers(followers, min_followers_count, min_statuses_count)
```

### 5. GPT Analysis
```python
# Analyze each follower with GPT
analysis = processor.analyze_follower_with_gpt(follower)
# Returns: {'overall_score': 8, 'key_indicators': ['antitrust lawyer', 'M&A specialist'], ...}
```

### 6. Database Storage
```python
# Save high-value follower to MongoDB
saved_id = processor.save_high_value_follower(analysis, deal_id, company_name, company_handle)
```

## Usage

### Basic Usage
```bash
python high_value_followers_processor.py <deal_id>
```

### Advanced Usage
```bash
# Process all followers with custom minimum score
python high_value_followers_processor.py 68ac4a254a6006a0946ec3bb --min-score 7

# Use GPT-4o-mini model
python high_value_followers_processor.py 68ac4a254a6006a0946ec3bb --use-gpt4-mini

# Limit followers per company for testing
python high_value_followers_processor.py 68ac4a254a6006a0946ec3bb --max-followers 10 --min-score 0
```

### Parameters
- `deal_id`: MongoDB ObjectId of the deal to process
- `--max-followers <number>`: Maximum followers to process per company (optional, processes ALL followers by default)
- `--min-score <number>`: Minimum GPT score to save (default: 5)

## Configuration

### Environment Variables
- `OPENAI_API_KEY_SEC_FILING`: Required for GPT analysis
- `TWITTER_API_KEY`: Optional, for additional Twitter API access

### Default Settings
- Minimum followers count: 100
- Minimum statuses count: 100
- Minimum GPT score: 3
- Maximum followers per company: 10 (for testing)
- GPT model: gpt-4o-mini
- Rate limiting: 0.5s between requests, 0.5s every 1 request

## Database Schema

### HighValueFollowers Collection
```javascript
{
  // Deal and company information
  deal_id: String,
  company_name: String,
  company_handle: String,
  
  // Follower information (spread from original object)
  follower_id: String,
  name: String,
  screen_name: String,
  description: String,
  location: String,
  followers_count: Number,
  statuses_count: Number,
  protected: Boolean,
  verified: Boolean,
  created_at_twitter: Date,
  
  // GPT Analysis results
  overall_score: Number,  // 0-10
  key_indicators: [String],
  analysis_timestamp: Date,
  gpt_model_used: String,
  
  // Processing metadata
  processing_status: String,
  approach: String,
  
  // Timestamps
  created_at: Date,
  updated_at: Date
}
```

## Output Files

### Summary JSON
The processor creates a summary JSON file with:
- Processing statistics
- Company-wise results
- Total counts and metrics
- File location: `high_value_followers_results/high_value_followers_summary_{deal_id}_{timestamp}.json`

### High-Value Followers JSON
The processor creates a comprehensive JSON file with:
- All high-value followers data with GPT analysis results
- Complete follower details and metadata
- Processing configuration and criteria
- File location: `high_value_followers_results/high_value_followers_{deal_id}_{timestamp}.json`

### MongoDB Records
Each high-value follower is saved as an individual document in the `high_value_followers` collection.

## Prerequisites

### Required Data
- Deal must exist in ProcessingJob collection
- Deal must have Twitter details with company handles
- Followers must be fetched using GUNSHOT approach first

### Dependencies
- OpenAI API key
- MongoDB connection
- Django setup
- Required Python packages (openai, mongoengine, etc.)

## Error Handling

### Graceful Failures
- Continues processing if one company fails
- Logs detailed error messages
- Saves partial results
- Rate limiting and retry logic for API calls

### Common Issues
- Missing Twitter handles: Logs warning and skips company
- No followers found: Logs warning and continues
- GPT API errors: Logs error and skips follower
- Database errors: Logs error and continues with next follower

## Performance Considerations

### Rate Limiting
- Built-in delays between GPT API calls
- Configurable rate limiting parameters
- Respects API limits

### Memory Management
- Processes followers one by one
- Saves to database immediately after analysis
- No large data structures in memory

### Scalability
- Can handle large follower bases
- Configurable limits for testing
- Efficient database queries with indexes

## Example Output

### Console Output
```
2024-01-15 10:30:00 - Starting High Value Followers processing for deal ID: 68184d52478abf06ec1a28ec
2024-01-15 10:30:01 - Found target Twitter handle: @targetcompany
2024-01-15 10:30:01 - Found acquire Twitter handle: @acquirecompany
2024-01-15 10:30:02 - Retrieved 1500 followers for @targetcompany
2024-01-15 10:30:02 - Filtered 1500 followers -> 450 followers
2024-01-15 10:30:03 - Analyzing follower 1/450: @user1
2024-01-15 10:30:04 - Successfully analyzed @user1 - Overall Score: 8
2024-01-15 10:30:04 - ✓ Saved high-value follower (Score: 8)
...
2024-01-15 10:45:00 - === High Value Followers Processing Complete ===
2024-01-15 10:45:00 - Deal ID: 68184d52478abf06ec1a28ec
2024-01-15 10:45:00 - Total companies processed: 2
2024-01-15 10:45:00 - Total high-value followers found: 127
```

### Summary JSON
```json
{
  "deal_id": "68184d52478abf06ec1a28ec",
  "processing_timestamp": "2024-01-15T10:45:00.123456",
  "approach": "HIGH_VALUE_FOLLOWERS",
  "total_companies": 2,
  "summary": {
    "total_followers": 3000,
    "total_filtered_followers": 900,
    "total_analyzed_followers": 900,
    "total_high_value_followers": 127
  },
  "company_results": [
    {
      "company_name": "Target Company",
      "company_handle": "targetcompany",
      "total_followers": 1500,
      "filtered_followers": 450,
      "analyzed_followers": 450,
      "high_value_followers": 65,
      "processing_status": "completed"
    }
  ]
}
```

## Integration

### With Existing Workflow
1. Run GUNSHOT approach to fetch followers
2. Run High Value Followers Processor to analyze and filter
3. Query HighValueFollowers collection for business intelligence

### API Integration
The HighValueFollowers collection can be queried via:
- Django ORM
- MongoDB native queries
- REST API endpoints (if implemented)

## Troubleshooting

### Common Issues
1. **No followers found**: Ensure GUNSHOT approach completed successfully
2. **Missing Twitter handles**: Check deal.twitter_details structure
3. **GPT API errors**: Verify OPENAI_API_KEY_SEC_FILING and rate limits
4. **Database errors**: Check MongoDB connection and permissions

### Debug Mode
Enable debug logging by modifying the logging configuration in the script.

## Future Enhancements

### Potential Improvements
- Batch processing for better performance
- Additional filtering criteria
- Custom GPT prompts per use case
- Export to Excel/CSV
- Web interface for monitoring
- Real-time processing with WebSockets
