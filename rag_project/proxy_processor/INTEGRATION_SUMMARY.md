# SEC Processing to Pinecone Integration Summary

## Overview
This document summarizes the changes made to integrate SEC document processing with Pinecone vector database upload, using proxy IDs instead of deal IDs.

## Changes Made

### 1. Modified `sec_processor_and_pinecone.py`

#### Added Dependencies
- `requests`: For downloading JSON files from S3 URLs
- `tempfile`: For handling temporary file operations

#### Updated SectionProcessor Class
- **Constructor**: Now accepts `proxy_id` and `proxy_object_id` parameters
- **New Method**: `download_json_from_s3(s3_url)` - Downloads JSON files from S3 URLs
- **New Method**: `process_from_s3_url(s3_url)` - Processes sections directly from S3 URL
- **Updated Metadata**: Changed from `deal_id` to `proxy_id` and `proxy_object_id`
- **Updated Vector IDs**: Now uses proxy ID for generating unique vector identifiers

#### Key Features
- Downloads JSON files from S3 URLs automatically
- Uses proxy ID and proxy object ID in metadata instead of deal ID
- Generates vector IDs based on proxy ID
- Cleans up temporary files after processing
- Maintains all existing functionality for local file processing

### 2. Updated `views.py`

#### Added Import
- Imported `SectionProcessor` from `sec_processor_and_pinecone`

#### New Function
- **`process_sections_with_pinecone(proxy_doc_id, sections_json_url)`**: 
  - Processes sections with Pinecone after SEC processing completes
  - Runs in a separate thread to avoid blocking
  - Updates proxy document with Pinecone processing status
  - Logs all processing steps

#### Modified Main Processing Function
- **`process_proxy_document_async()`**: 
  - Now automatically triggers Pinecone processing after SEC processing completes
  - Extracts `sections_json_url` from S3 URLs
  - Starts Pinecone processing in a separate thread
  - Logs the start of Pinecone processing

### 3. Updated `models.py`

#### Added Fields to ProxyDocument Model
- **`pinecone_processing_status`**: Tracks Pinecone processing status (pending, processing, completed, failed)
- **`pinecone_processed_at`**: Timestamp when Pinecone processing completed
- **`pinecone_error_message`**: Error message if Pinecone processing fails

#### Updated Indexes
- Added `pinecone_processing_status` to the database indexes for efficient querying

## Workflow

### Complete Processing Flow
1. **SEC Document Processing**: 
   - User submits proxy document for processing
   - `AgenticSECProcessor` processes the SEC document
   - Generates PDF, extracts TOC, and creates sections JSON
   - Uploads all files to S3

2. **Automatic Pinecone Processing**:
   - After SEC processing completes, system extracts `sections_json_url` from S3 URLs
   - Starts `SectionProcessor` in a separate thread
   - Downloads sections JSON from S3
   - Processes sections into chunks and creates embeddings
   - Uploads vectors to Pinecone with proxy ID metadata

3. **Status Tracking**:
   - Both SEC processing and Pinecone processing statuses are tracked
   - Detailed logging for debugging and monitoring
   - Error handling for both processing stages

## Usage

### For New Documents
```python
# The system automatically handles the complete flow
# 1. Submit proxy document for processing
# 2. SEC processing runs automatically
# 3. Pinecone processing starts automatically after SEC processing completes
```

### For Manual Processing
```python
from proxy_processor.sec_processor_and_pinecone import SectionProcessor

# Initialize with proxy information
processor = SectionProcessor(
    proxy_id="your_proxy_id",
    proxy_object_id="your_proxy_object_id"
)

# Process from S3 URL
processor.process_from_s3_url("https://your-bucket.s3.amazonaws.com/sections.json")

# Or process from local file
processor.process_file("local_sections.json")
```

## Metadata Structure

### Before (Deal-based)
```json
{
    "title": "Section Title",
    "page_no": "1",
    "deal_id": "deal_123",
    "parent_section": "Parent Section",
    "original_text": "Section content...",
    "metadata": true
}
```

### After (Proxy-based)
```json
{
    "title": "Section Title",
    "page_no": "1",
    "proxy_id": "proxy_123",
    "proxy_object_id": "proxy_object_456",
    "parent_section": "Parent Section",
    "original_text": "Section content...",
    "metadata": true
}
```

## Vector ID Generation

### Before
```python
vector_id = f"deal_name_{i}"  # e.g., "steelnewco_1"
```

### After
```python
proxy_prefix = self.proxy_id.lower().replace(' ', '_') if self.proxy_id else 'proxy'
vector_id = f"{proxy_prefix}_{i}"  # e.g., "proxy_123_1"
```

## Testing

A test script `test_integration.py` has been created to verify the integration:
- Tests proxy ID functionality
- Verifies JSON processing
- Tests text chunking and metadata handling
- Validates the complete workflow

## Benefits

1. **Automatic Processing**: No manual intervention required after SEC processing
2. **Proxy-based Organization**: Uses proxy IDs for better organization and querying
3. **Scalable**: Processes documents in parallel threads
4. **Robust Error Handling**: Comprehensive error handling and logging
5. **Status Tracking**: Full visibility into processing status
6. **S3 Integration**: Seamless integration with S3 storage
7. **Backward Compatible**: Maintains all existing functionality

## Environment Variables Required

- `OPENAI_API_KEY`: For creating embeddings
- `PINECONE_API_KEY`: For Pinecone vector database access
- `PINECONE_INDEX_NAME`: Name of the Pinecone index to use

## Next Steps

1. Test the integration with real SEC documents
2. Monitor processing logs for any issues
3. Optimize performance if needed
4. Add additional metadata fields as required
5. Consider adding batch processing for multiple documents
