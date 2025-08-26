# Twitter Handle Finder

## Overview

The Twitter Handle Finder is a service that uses GPT to automatically find and store official Twitter handles for companies involved in M&A deals. It identifies main company handles, subsidiary handles, and additional corporate accounts.

## Features

### 1. Company Twitter Handle Discovery
- **Main Company Handles**: Official corporate Twitter accounts
- **Subsidiary Handles**: Twitter accounts for major subsidiaries and divisions
- **Additional Handles**: Regional, product-specific, and executive accounts
- **Verification**: Focuses on verified and official accounts

### 2. Automated Processing
- **GPT-Powered Search**: Uses GPT to find accurate Twitter handles
- **Structured Data**: Stores results in organized JSON format
- **Error Handling**: Graceful handling of missing or invalid data

### 3. Database Integration
- **Deal Table Storage**: Saves Twitter details directly to ProcessingJob model
- **Structured Format**: Organized by company type (target/acquire)
- **Summary Statistics**: Quick overview of found handles

## Data Structure

### Twitter Details Schema
```json
{
  "deal_id": "string",
  "search_timestamp": "2024-01-01T00:00:00Z",
  "unique_companies": ["Company A", "Company B", "Subsidiary C"],
  "company_handles": {
    "Company A": {
      "company_name": "Company A",
      "company_type": "company",
      "main_twitter_handle": "@companya",
      "subsidiaries": [
        {
          "subsidiary_name": "Subsidiary Name",
          "twitter_handle": "@subsidiary",
          "account_type": "subsidiary|division|regional|product|executive",
          "description": "Brief description"
        }
      ],
      "additional_handles": [
        {
          "handle": "@additional",
          "account_type": "subsidiary|division|regional|product|executive",
          "description": "Brief description"
        }
      ],
      "search_notes": "Search notes and verification status"
    },
    "Company B": {
      // Same structure as Company A
    }
  },
  "summary": {
    "total_companies": 3,
    "companies_with_handles": 2,
    "total_subsidiaries": 5,
    "total_additional_handles": 3
  }
}
```

## Usage

### 1. Command Line Usage

```bash
# Process entire deal
python twitter_handle_finder.py <deal_id>

# Search single company
python twitter_handle_finder.py <deal_id> --company "Company Name"
```

### 2. Programmatic Usage

```python
from document_processor.twitter_utils.twitter_handle_finder import TwitterHandleFinder

# Initialize finder
finder = TwitterHandleFinder()

# Find handles for single company
result = finder.find_company_twitter_handles("Company Name", "target")

# Process entire deal
deal_result = finder.process_deal_twitter_handles("deal_id")

# Retrieve stored data
stored_data = finder.get_twitter_handles_for_deal("deal_id")
```

### 3. Integration with Services

The Twitter handle finder is automatically integrated into the RF1 approach workflow. It runs after extracting unique companies and before creating company combinations for Twitter search.

## Database Schema

### ProcessingJob Model Updates
- Added `twitter_details` field (DynamicField, default=[])
- Stores complete Twitter handle information for the deal

### Serializer Updates
- Added `twitter_details` field to ProcessingJobSerializer
- Supports JSON serialization of Twitter handle data

## Account Types

The system categorizes Twitter accounts into different types:

1. **subsidiary**: Major subsidiary companies
2. **division**: Business divisions or units
3. **regional**: Geographic/regional accounts
4. **product**: Product-specific accounts
5. **executive**: Executive/leadership accounts

## Error Handling

- **Missing Companies**: Graceful handling when company names are missing
- **GPT Errors**: Fallback responses when GPT fails
- **JSON Parsing**: Error recovery for malformed responses
- **Database Errors**: Proper error logging and recovery

## Testing

Run the test script to verify functionality:

```bash
python test_twitter_handles.py
```

This will test:
- Single company search
- Full deal processing
- Data storage and retrieval
- Error handling

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for GPT API access

### Model Configuration
- Default model: `gpt-4.1`
- Temperature: 0.1 (for consistent results)
- Max tokens: 2000

## Benefits

- ✅ **Automated Discovery**: No manual research needed
- ✅ **Comprehensive Coverage**: Finds main and subsidiary handles
- ✅ **Structured Storage**: Organized data for easy access
- ✅ **Integration Ready**: Works with existing Twitter search approaches
- ✅ **Error Resilient**: Handles failures gracefully
- ✅ **Extensible**: Easy to add new account types or sources

## Future Enhancements

- **Real-time Verification**: Check if handles are still active
- **Follower Count**: Include follower statistics
- **Account Verification**: Verify blue checkmark status
- **Historical Data**: Track handle changes over time
- **API Integration**: Direct Twitter API integration for verification
