# API Calls Folder Structure - Gun Shot Approach

## Overview

The Gun Shot Approach has been enhanced to save each API call's JSON response immediately to a folder structure. This ensures that even if an error occurs during processing, you will have partial data available for recovery.

## Folder Structure

```
twitter_search_results/
├── api_calls/                    # New: Individual API call responses
│   └── {deal_id}/               # Organized by deal ID
│       ├── {username1}/         # Organized by Twitter username
│       │   ├── page_001_20241227_143022_123.json
│       │   ├── page_002_20241227_143025_456.json
│       │   └── ...
│       └── {username2}/
│           ├── page_001_20241227_143030_789.json
│           └── ...
├── followers/                    # Existing: Consolidated followers data
│   ├── followers_{deal_id}_{username}_{timestamp}.json
│   └── ...
├── recovered/                    # New: Recovered data from partial runs
│   └── recovered_followers_{deal_id}_{timestamp}.json
└── twitter_followers_gunshot_{deal_id}_{timestamp}.json  # Main results file
```

## API Call File Format

Each API call is saved as a JSON file with the following structure:

```json
{
  "deal_id": "68184d52478abf06ec1a28ec",
  "username": "company_handle",
  "page_number": 1,
  "cursor": "cursor_value",
  "api_call_timestamp": "2024-12-27T14:30:22.123456",
  "response_data": {
    "followers": [...],
    "has_next_page": true,
    "next_cursor": "next_cursor_value"
  },
  "metadata": {
    "file_created": "2024-12-27T14:30:22.123456",
    "file_version": "1.0",
    "data_source": "twitterapi.io",
    "approach": "GUNSHOT",
    "page_followers_count": 200,
    "has_next_page": true,
    "next_cursor": "next_cursor_value"
  }
}
```

## Benefits

1. **Fault Tolerance**: If the script crashes or encounters an error, you don't lose all the data
2. **Partial Recovery**: You can recover followers from completed pages even if the full process didn't finish
3. **Debugging**: Each API call is preserved for debugging and analysis
4. **Resume Capability**: You can potentially resume from where you left off
5. **Data Verification**: You can verify the data at each step

## Recovery Utility

A recovery utility script has been created to help analyze and recover data from partial runs.

### Usage

```bash
# Analyze API calls for a deal
python api_calls_recovery.py 68184d52478abf06ec1a28ec

# Analyze and recover followers data
python api_calls_recovery.py 68184d52478abf06ec1a28ec --recover
```

### Example Output

```
=== API Calls Analysis for Deal 68184d52478abf06ec1a28ec ===
Status: found
API Calls Directory: /path/to/api_calls/68184d52478abf06ec1a28ec
Total API Calls: 15
Total Followers: 3000
Companies Found: 2

  Company: CompanyA
    Username: @company_a
    API Call Files: 8
    Followers Count: 1600
    Pages Found: [1, 2, 3, 4, 5, 6, 7, 8]
    Last Page: 8

  Company: CompanyB
    Username: @company_b
    API Call Files: 7
    Followers Count: 1400
    Pages Found: [1, 2, 3, 4, 5, 6, 7]
    Last Page: 7
```

## File Naming Convention

API call files follow this naming pattern:
```
page_{page_number:03d}_{timestamp}.json
```

Where:
- `page_number`: Zero-padded page number (001, 002, etc.)
- `timestamp`: Format: YYYYMMDD_HHMMSS_microseconds

Example: `page_001_20241227_143022_123.json`

## Integration with Main Script

The main `gun_shot_approach.py` script now:

1. Saves each API response immediately after receiving it
2. Tracks all saved files in the results
3. Provides summary information about API calls saved
4. Includes API calls directory path in the main results file

### Updated Results File Structure

The main results file now includes:

```json
{
  "deal_id": "68184d52478abf06ec1a28ec",
  "search_timestamp": "2024-12-27T14:30:22.123456",
  "approach": "GUNSHOT_FOLLOWERS",
  "total_companies": 2,
  "json_files_created": 2,
  "json_file_paths": [...],
  "api_call_files_created": 15,
  "api_call_file_paths": [...],
  "api_calls_directory": "/path/to/api_calls/68184d52478abf06ec1a28ec",
  "results": [...]
}
```

## Error Handling

If an error occurs during processing:

1. **API Call Level**: Each successful API call is saved immediately
2. **Page Level**: If a page fails after some API calls succeed, those calls are preserved
3. **Company Level**: If one company fails, the other company's data is preserved
4. **Recovery**: Use the recovery utility to extract partial data

## Storage Considerations

- Each API call file is typically 50-200KB depending on follower count
- For large accounts with many pages, this can add up to significant storage
- Consider implementing cleanup strategies for old data if needed
- The files are organized by deal ID and username for easy management

## Best Practices

1. **Monitor Storage**: Keep an eye on disk space usage
2. **Regular Cleanup**: Consider archiving or deleting old API call files
3. **Backup Strategy**: Include API calls directory in your backup strategy
4. **Recovery Testing**: Test the recovery utility periodically
5. **Documentation**: Keep track of any custom modifications to the structure
