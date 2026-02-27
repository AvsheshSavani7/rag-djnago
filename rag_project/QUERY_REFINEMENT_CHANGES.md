# Query Refinement with GPT - Implementation Summary

## Overview
This document summarizes the changes made to implement GPT-powered query refinement in the `high_value_followers_tweet_search_test.py` file. The implementation allows building refined Twitter search queries using GPT without including usernames, then appending usernames when processing individual followers.

## Key Changes Made

### 1. New Methods Added

#### `build_refined_search_query_with_gpt()`
- **Purpose**: Builds a refined Twitter search query template using GPT
- **Inputs**: Company name, products list, company Twitter handle
- **Output**: Refined search query string (without username)
- **Features**:
  - Uses GPT-4 to optimize search queries for TwitterAPI.io
  - Includes company name, handle, and products in the query
  - Adds date range and language filters
  - Falls back to basic query structure if GPT fails

#### `build_search_query_with_template()`
- **Purpose**: Combines username with pre-built query template
- **Inputs**: Username and refined query template
- **Output**: Complete Twitter search query string
- **Features**:
  - Intelligently inserts username at the beginning of the template
  - Handles different template formats (with/without parentheses)

#### `search_follower_tweets_with_template()`
- **Purpose**: Searches tweets using the new template-based approach
- **Inputs**: Follower data and refined query template
- **Output**: Tweet search results
- **Features**:
  - Uses the new template-based query building
  - Maintains compatibility with existing result structure

### 2. Modified Methods

#### `process_company_followers()`
- **Changes**:
  - Now builds refined query template once at the beginning
  - Passes template to `process_single_follower` instead of individual parameters
  - Updated both parallel and sequential processing paths

#### `process_single_follower()`
- **Changes**:
  - Updated to accept refined query template as parameter
  - Now uses `search_follower_tweets_with_template()` instead of `search_follower_tweets()`
  - Maintains backward compatibility

### 3. Configuration Options Added

```python
# GPT integration settings
'use_gpt_for_query_refinement': True,  # Enable/disable GPT integration
'gpt_model': 'gpt-4',                  # GPT model to use
'gpt_temperature': 0.3,                # Temperature for generation
'gpt_max_tokens': 200,                 # Max tokens for response
```

## How It Works

### 1. Query Template Creation
```python
# In process_company_followers()
refined_query_template = self.build_refined_search_query_with_gpt(
    company_name, products, company_handle
)
```

### 2. Template Usage
```python
# In process_single_follower()
result = self.search_follower_tweets_with_template(
    follower, refined_query_template
)
```

### 3. Final Query Building
```python
# In search_follower_tweets_with_template()
query = self.build_search_query_with_template(username, refined_query_template)
# Result: "from:username (company OR @handle OR "product") since:date until:date lang:en"
```

## Benefits

1. **Efficiency**: Query template is built once per company instead of per follower
2. **Quality**: GPT-optimized queries for better search results
3. **Flexibility**: Can enable/disable GPT integration via configuration
4. **Fallback**: Graceful degradation to basic query structure if GPT fails
5. **Maintainability**: Cleaner separation of concerns

## Configuration

### Enable GPT Integration
```python
config_overrides = {
    'use_gpt_for_query_refinement': True,
    'gpt_model': 'gpt-4',
    'gpt_temperature': 0.3,
    'gpt_max_tokens': 200
}
```

### Disable GPT Integration
```python
config_overrides = {
    'use_gpt_for_query_refinement': False
}
```

## Requirements

- `OPENAI_API_KEY` environment variable must be set
- `openai` package must be installed (already in requirements.txt)
- Internet connection for GPT API calls

## Testing

A test script `test_query_refinement.py` has been created to verify the functionality:

```bash
cd rag_project
python test_query_refinement.py
```

## Error Handling

- **GPT API failures**: Falls back to basic query structure
- **Missing API key**: Logs warning and uses fallback
- **Invalid GPT response**: Validates response and falls back if needed
- **Configuration disabled**: Skips GPT integration entirely

## Future Enhancements

1. **Caching**: Cache GPT responses for similar company/product combinations
2. **Multiple models**: Support for different GPT models (GPT-3.5, Claude, etc.)
3. **Query analytics**: Track which queries perform better
4. **A/B testing**: Compare GPT vs. basic query performance
5. **Custom prompts**: Allow customization of GPT prompts per use case
