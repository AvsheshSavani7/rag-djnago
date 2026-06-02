# SEC Proxy Processor

This Django app provides a simplified API for processing SEC proxy documents using the agentic SEC processor functionality.

## Features

- **Asynchronous Processing**: SEC proxy documents are processed in the background using threading
- **MongoDB Storage**: All processing results and logs are stored in MongoDB
- **RESTful API**: Simple API with only 2 endpoints
- **Authentication**: JWT-based authentication required for all endpoints
- **Comprehensive Logging**: Detailed logging of all processing steps
- **Error Handling**: Robust error handling and status tracking

## API Endpoints

### 1. Process SEC Proxy Document
- **POST** `/api/proxy-processor/proxry-processor/`
- **Body**: 
```json
{
    "cik_number": "1234567890",
    "company_name": "Example Corp",
    "sec_filling_id": "0001234567-24-000001",
    "filing_date": "2024-01-15",
    "form_type": "DEF 14A",
    "proxy_sec_url": "https://www.sec.gov/...",
    "deal_id": "optional_deal_id"
}
```
- **Response**: Proxy document ID and processing status

### 2. List All Processing Jobs
- **GET** `/api/proxy-processor/jobs/`
- **Response**: List of all proxy documents with their status and results

## Models

### ProxyDocument
Stores the main proxy document information including:
- **Basic Info**: CIK number, company name, SEC filing ID, filing date, form type
- **URLs**: Proxy SEC URL and generated AWS parsing URL
- **Deal Info**: Deal ID (if CIK matches with deal table)
- **Processing Status**: pending, processing, completed, failed
- **Results**: PDF, TOC, sections paths and data
- **Statistics**: Processing statistics and timestamps

### ProxyProcessingLog
Stores detailed processing logs with:
- Log level and message
- Module information
- Timestamps

## Usage Example

```python
import requests

# Start processing a proxy document
response = requests.post('http://localhost:8000/api/proxy-processor/proxry-processor/', 
                        json={
                            "cik_number": "1234567890",
                            "company_name": "Example Corp",
                            "sec_filling_id": "0001234567-24-000001",
                            "filing_date": "2024-01-15",
                            "form_type": "DEF 14A",
                            "proxy_sec_url": "https://www.sec.gov/...",
                            "deal_id": "deal_123"
                        },
                        headers={'Authorization': 'Bearer <token>'})

proxy_doc_id = response.json()['proxy_document_id']

# List all proxy documents to see status and results
docs_response = requests.get('http://localhost:8000/api/proxy-processor/jobs/',
                            headers={'Authorization': 'Bearer <token>'})

# Find your document in the list
for doc in docs_response.json():
    if doc['id'] == proxy_doc_id:
        print(f"Status: {doc['proxy_parsing_status']}")
        if doc['proxy_parsing_status'] == 'completed':
        break
```

## Dependencies

The app requires the following Python packages:
- Django
- Django REST Framework
- MongoEngine
- LlamaIndex
- Playwright
- OpenAI
- BeautifulSoup4
- html2text

## Configuration

Make sure to set the following environment variables:
- `OPENAI_API_KEY_SEC_FILING`: OpenAI API key for GPT processing
- `MONGODB_CONNECTION_STRING`: MongoDB connection string
- `MONGODB_NAME`: MongoDB database name

## Testing

Run the tests with:
```bash
python manage.py test proxy_processor
```

## File Structure

```
proxy_processor/
├── __init__.py
├── admin.py
├── apps.py
├── models.py
├── serializers.py
├── views.py
├── urls.py
├── tests.py
├── README.md
├── agentic_sec_processor.py
├── extract_sections_html_class.py
└── playwright_sec_parser_only_doc_heading.py
```

## Simplified Workflow

1. **POST** to `/api/proxy-processor/proxry-processor/` with proxy document data
2. **GET** from `/api/proxy-processor/jobs/` to see all proxy documents and their results
3. Processing happens asynchronously in the background
4. Results are automatically saved as JSON files and stored in MongoDB
5. AWS URL is generated in format: `proxy-parse-jsons/{cik_number}/{sec_filling_id}/`

## Data Schema

### Input Schema
```json
{
    "cik_number": "1234567890",
    "company_name": "Example Corp",
    "sec_filling_id": "0001234567-24-000001",
    "filing_date": "2024-01-15",
    "form_type": "DEF 14A",
    "proxy_sec_url": "https://www.sec.gov/...",
    "deal_id": "optional_deal_id"
}
```

### Output Schema
```json
{
    "_id": "document_id",
    "cik_number": "1234567890",
    "deal_id": "deal_123",
    "company_name": "Example Corp",
    "sec_filling_id": "0001234567-24-000001",
    "filing_date": "2024-01-15",
    "form_type": "DEF 14A",
    "proxy_sec_url": "https://www.sec.gov/...",
    "proxy_parsing_status": "completed",
   
}
```