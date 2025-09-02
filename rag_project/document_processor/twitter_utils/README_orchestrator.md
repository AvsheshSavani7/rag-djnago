# High Value Followers Orchestrator

A comprehensive orchestrator that combines all three approaches for analyzing high-value followers from deal data.

## Overview

The orchestrator implements the complete "Shot Gun Approach" flow diagram with three main steps:

1. **Step 1: Gun Shot Approach** - Fetch all followers for target and acquire companies
2. **Step 2: GPT Analysis** - Analyze followers with descriptions using GPT
3. **Step 3: Tweet Search** - Analyze followers without descriptions by searching their tweets

## Flow Diagram Implementation

```
Shot Gun Approach
├── Step 1: Fetch all followers
│   ├── Target company followers
│   └── Acquire company followers
├── Step 2: Profile Evaluation (Has Description)
│   ├── GPT Analysis
│   ├── High Value → Save to Critical Voices DB
│   └── Low Value → Ignore/Deprioritize
└── Step 3: Profile Evaluation (No Description)
    ├── Tweet Search for company/products
    ├── Relevant tweets found → Save to Critical Voices DB
    └── No relevant tweets → Ignore
```

## Files

- `high_value_followers_orchestrator.py` - Main orchestrator class
- `example_orchestrator_usage.py` - Examples of how to use the orchestrator
- `README_orchestrator.md` - This documentation

## Usage

### Command Line Usage

```bash
# Run all steps
python high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb

# Disable specific steps
python high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb --disable-step-2
python high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb --disable-step-1 --disable-step-3

# Custom configuration
python high_value_followers_orchestrator.py 68ac4a254a6006a0946ec3bb \
  --gpt-workers 100 \
  --tweet-workers 20 \
  --gpt-max-followers 50 \
  --tweet-max-followers 200
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--disable-step-1` | Disable Gun Shot Approach | Enabled |
| `--disable-step-2` | Disable GPT Analysis | Enabled |
| `--disable-step-3` | Disable Tweet Search | Enabled |
| `--gpt-workers <n>` | GPT analysis workers | 70 |
| `--tweet-workers <n>` | Tweet search workers | 10 |
| `--gpt-max-followers <n>` | Max followers for GPT | 10 |
| `--tweet-max-followers <n>` | Max followers for tweets | 200 |

### Programmatic Usage

```python
from document_processor.twitter_utils.high_value_followers_orchestrator import HighValueFollowersOrchestrator

# Basic usage - all steps enabled
orchestrator = HighValueFollowersOrchestrator()
result_file = orchestrator.process_deal("68ac4a254a6006a0946ec3bb")

# Custom configuration
config_overrides = {
    'enable_step_1_gun_shot': True,
    'enable_step_2_gpt_analysis': False,  # Skip GPT analysis
    'enable_step_3_tweet_search': True,
    'gpt_max_workers': 100,
    'tweet_search_max_workers': 20
}

orchestrator = HighValueFollowersOrchestrator(config_overrides=config_overrides)
result_file = orchestrator.process_deal("68ac4a254a6006a0946ec3bb")

# Step-by-step processing with detailed results
results = orchestrator.process_deal_step_by_step("68ac4a254a6006a0946ec3bb")
```

## Configuration Options

### Step Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_step_1_gun_shot` | bool | True | Enable/disable follower fetching |
| `enable_step_2_gpt_analysis` | bool | True | Enable/disable GPT analysis |
| `enable_step_3_tweet_search` | bool | True | Enable/disable tweet search |

### GPT Analysis Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `gpt_max_followers_per_company` | int | 10 | Max followers to analyze with GPT |
| `gpt_min_overall_score` | int | 0 | Minimum score to save as high-value |
| `gpt_max_workers` | int | 70 | Number of parallel GPT workers |
| `gpt_model` | str | 'gpt-4.1' | GPT model to use |

### Tweet Search Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `tweet_search_max_followers_per_company` | int | 200 | Max followers to search tweets for |
| `tweet_search_max_workers` | int | 10 | Number of parallel tweet search workers |
| `tweet_search_date_range_years` | int | 5 | Years back to search tweets |

### Common Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `min_followers_count` | int | 250 | Minimum follower count filter |
| `min_statuses_count` | int | 250 | Minimum status count filter |
| `batch_size` | int | 1000 | Batch size for MongoDB saves |
| `use_batch_saving` | bool | True | Enable batch saving |

## Examples

### Example 1: Full Workflow

```python
# Run complete workflow with all steps
orchestrator = HighValueFollowersOrchestrator()
result_file = orchestrator.process_deal("68ac4a254a6006a0946ec3bb")
```

### Example 2: GPT Analysis Only

```python
# Skip follower fetching and tweet search, only do GPT analysis
config_overrides = {
    'enable_step_1_gun_shot': False,
    'enable_step_2_gpt_analysis': True,
    'enable_step_3_tweet_search': False,
    'gpt_max_followers_per_company': 20
}

orchestrator = HighValueFollowersOrchestrator(config_overrides=config_overrides)
result_file = orchestrator.process_deal("68ac4a254a6006a0946ec3bb")
```

### Example 3: Tweet Search Only

```python
# Skip follower fetching and GPT analysis, only do tweet search
config_overrides = {
    'enable_step_1_gun_shot': False,
    'enable_step_2_gpt_analysis': False,
    'enable_step_3_tweet_search': True,
    'tweet_search_max_followers_per_company': 100
}

orchestrator = HighValueFollowersOrchestrator(config_overrides=config_overrides)
result_file = orchestrator.process_deal("68ac4a254a6006a0946ec3bb")
```

### Example 4: High-Throughput Configuration

```python
# Optimize for speed with more workers and larger batches
config_overrides = {
    'gpt_max_workers': 100,
    'tweet_search_max_workers': 30,
    'batch_size': 2000,
    'gpt_max_followers_per_company': 50,
    'tweet_search_max_followers_per_company': 500
}

orchestrator = HighValueFollowersOrchestrator(config_overrides=config_overrides)
result_file = orchestrator.process_deal("68ac4a254a6006a0946ec3bb")
```

### Example 5: Integration with Other Functions

```python
def analyze_deal_followers(deal_id: str, analysis_type: str = 'full'):
    """Integrate orchestrator into other functions"""
    if analysis_type == 'full':
        config_overrides = {}
    elif analysis_type == 'gpt_only':
        config_overrides = {
            'enable_step_1_gun_shot': False,
            'enable_step_2_gpt_analysis': True,
            'enable_step_3_tweet_search': False
        }
    elif analysis_type == 'tweet_only':
        config_overrides = {
            'enable_step_1_gun_shot': False,
            'enable_step_2_gpt_analysis': False,
            'enable_step_3_tweet_search': True
        }
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    orchestrator = HighValueFollowersOrchestrator(config_overrides=config_overrides)
    return orchestrator.process_deal(deal_id)

# Usage
full_result = analyze_deal_followers("68ac4a254a6006a0946ec3bb", "full")
gpt_result = analyze_deal_followers("68ac4a254a6006a0946ec3bb", "gpt_only")
tweet_result = analyze_deal_followers("68ac4a254a6006a0946ec3bb", "tweet_only")
```

## Output

The orchestrator generates several output files:

1. **Orchestrator Summary** - `orchestrator_summary_{deal_id}_{timestamp}.json`
   - Overall processing results
   - Step-by-step breakdown
   - Configuration used

2. **GPT Analysis Results** - Generated by GPT processor
   - High-value followers from GPT analysis
   - Analysis details and scores

3. **Tweet Search Results** - Generated by tweet search processor
   - High-value followers from tweet search
   - Tweet details and search queries

4. **Filtered Followers** - Generated by each step
   - Filtered follower lists for tracking

## Database Integration

The orchestrator saves results to the `HighValueFollowers` collection with the following approach identifiers:

- **GPT Analysis**: `gpt_model_used` = "gpt-4.1" (or configured model)
- **Tweet Search**: `gpt_model_used` = "tweet_search"

This allows you to distinguish between followers found by different approaches.

## Error Handling

The orchestrator includes comprehensive error handling:

- Each step can fail independently without affecting other steps
- Detailed error logging for debugging
- Graceful degradation when steps are disabled
- Step results include success/failure status and error messages

## Performance Considerations

- **Parallel Processing**: Both GPT analysis and tweet search use parallel workers
- **Batch Saving**: MongoDB saves are batched for better performance
- **Lazy Loading**: Processors are only initialized when needed
- **Configurable Limits**: Control processing scope to manage costs and time

## Dependencies

- Django setup with MongoDB
- OpenAI API key for GPT analysis
- Twitter API key for tweet search
- All existing follower processing utilities

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure `OPENAI_API_KEY` and `TWITTER_API_KEY` are set
2. **Django Setup**: Make sure Django is properly configured
3. **Deal Not Found**: Verify the deal ID exists in the database
4. **Rate Limiting**: Adjust worker counts and delays if hitting API limits

### Debug Mode

Enable detailed logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Step-by-Step Debugging

Use the `process_deal_step_by_step()` method to get detailed results for each step:

```python
results = orchestrator.process_deal_step_by_step("68ac4a254a6006a0946ec3bb")
print(results['step_results']['step_1_gun_shot'])
print(results['step_results']['step_2_gpt_analysis'])
print(results['step_results']['step_3_tweet_search'])
```
