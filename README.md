# RAG Backend Project

A Retrieval Augmented Generation (RAG) backend service that processes legal documents, generates embeddings, and provides AI-powered insights.

## Features

- **Document Processing**: Convert and flatten document structures for analysis
- **Embedding Generation**: Create vector embeddings using OpenAI
- **Vector Storage**: Store and search embeddings using Pinecone
- **AI-Powered Chat**: Ask questions about documents with context-aware responses
- **Document Summarization**: Generate comprehensive summaries of legal documents

## API Endpoints

### Document Processing

- `POST /process/`: Process a document file from a URL
  - Parameters: `file_url`, `deal_id`, `embed_data` (optional)
  - Returns: Processing status and deal ID

### Embedding Operations

- `POST /embed/`: Generate embeddings for a processed document
  - Parameters: `deal_id`
  - Returns: Embedding generation status

### Deal Management

- `GET /deals/`: List all deals
  - Returns: List of all deals with their details
- `GET /deals/<str:id>/`: Get details for a specific deal
  - Returns: Detailed information about a deal

### Vector Operations

- `GET /vectors/<str:deal_id>/`: List all vectors for a specific deal
  - Returns: All vector chunks for the deal
- `PATCH /vectors/update/<str:vector_id>/`: Update metadata for a vector
  - Parameters: `metadata` (object)
  - Returns: Update status

### AI Interactions

- `POST /chat/`: Chat with AI about document content
  - Parameters: 
    - `query`: User's question
    - `deal_id` (optional): Specific deal to query
    - `message_history` (optional): Previous chat messages
    - `top_k` (optional): Number of relevant chunks to retrieve (default: 5)
    - `temperature` (optional): Temperature for OpenAI generation (default: 0.7)
  - Returns: AI response with answer, context, and usage statistics

- `POST /summary/`: Generate a document summary
  - Parameters:
    - `deal_id`: ID of the document to summarize
    - `temperature` (optional): Temperature for OpenAI generation (default: 0.7)
  - Returns: Comprehensive summary with context and usage statistics

## Environment Variables

The following environment variables need to be set:

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
AWS_S3_BUCKET=your_s3_bucket
```

## Summary Generation

The summary generation endpoint uses a set of predefined questions to retrieve relevant document sections:

1. What is the purpose of this agreement?
2. Who are the main parties involved in this agreement?
3. What are the key obligations of each party?
4. What are the important dates and deadlines in this agreement?
5. What are the termination conditions?
6. What are the key financial terms?

The system then consolidates these sections to generate a comprehensive document summary with clear headings for each topic.

## Setup and Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/rag-backend.git
   cd rag-backend
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Unix/MacOS:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

5. Create a `.env` file in the project root with the required environment variables:
   ```bash
   # OpenAI API
   OPENAI_API_KEY=your_openai_api_key
   
   # Pinecone Vector Database
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_pinecone_index_name
   
   # AWS S3 Configuration
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key
   AWS_REGION=your_aws_region
   AWS_S3_BUCKET=your_s3_bucket
   
   # MongoDB Configuration
   MONGODB_NAME=rag_db
   MONGODB_HOST=localhost
   MONGODB_PORT=27017
   MONGODB_USER=your_mongodb_user
   MONGODB_PASSWORD=your_mongodb_password
   # Alternatively, use a connection string
   # MONGODB_CONNECTION_STRING=mongodb://username:password@localhost:27017/rag_db
   ```

6. Set up MongoDB
   - Install MongoDB Community Edition: https://www.mongodb.com/try/download/community
   - Start the MongoDB service
   - Create a database named `rag_db`

7. Set up Pinecone
   - Create an account at https://www.pinecone.io/
   - Create a new index with dimensions=1536 (for OpenAI embeddings)
   - Set the metric to "cosine"

8. Run database migrations
   ```bash
   python manage.py migrate
   ```

9. Start the development server
   ```bash
   python manage.py runserver
   ```
   The server will be available at http://127.0.0.1:8000/

## Project Structure

```
rag_project/
├── document_processor/     # Core RAG functionality app
│   ├── services.py         # Business logic services
│   ├── views.py            # API views
│   ├── urls.py             # URL routing
│   └── models.py           # Data models
├── rag_project/            # Django project settings
│   ├── settings.py         # Project settings
│   ├── urls.py             # Main URL configuration
│   └── wsgi.py             # WSGI configuration
├── node_proxy/             # Node.js proxy app (if used)
├── .env                    # Environment variables (create this)
├── manage.py               # Django management script
└── requirements.txt        # Python dependencies
```

## Development Workflow

1. Start the development server
   ```bash
   python manage.py runserver
   ```

2. Process a document
   ```bash
   curl -X POST http://localhost:8000/process/ \
     -H "Content-Type: application/json" \
     -d '{"file_url": "https://example.com/document.json", "deal_id": "deal123"}'
   ```

3. Generate embeddings
   ```bash
   curl -X POST http://localhost:8000/embed/ \
     -H "Content-Type: application/json" \
     -d '{"deal_id": "deal123"}'
   ```

4. Chat with document
   ```bash
   curl -X POST http://localhost:8000/chat/ \
     -H "Content-Type: application/json" \
     -d '{"query": "What is this document about?", "deal_id": "deal123"}'
   ```

5. Generate a summary
   ```bash
   curl -X POST http://localhost:8000/summary/ \
     -H "Content-Type: application/json" \
     -d '{"deal_id": "deal123"}'
   ```

## Deployment

For production deployment:

1. Set `DEBUG=False` in settings.py
2. Configure a proper web server (Nginx, Apache) with Gunicorn or uWSGI
3. Set up proper CORS settings for production
4. Consider using environment variables for all sensitive configuration
5. Set up proper monitoring and logging

Example Gunicorn deployment:
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn rag_project.wsgi:application --bind 0.0.0.0:8000 --workers 3
```

## Technologies Used

- Django REST Framework
- OpenAI API
- Pinecone Vector Database
- AWS S3
- MongoDB (with PyMongo)




python manage.py runserver 0.0.0.0:8000
