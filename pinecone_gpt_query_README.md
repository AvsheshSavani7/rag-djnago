# Pinecone GPT Query Tool

This tool retrieves vector chunks from Pinecone based on a question, passes them to OpenAI's GPT, and saves the response in a JSON file.

## Setup

1. Make sure you have the required Python packages installed:
   ```
   pip install openai pinecone-client python-dotenv
   ```

2. Create a `.env` file with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_pinecone_index_name
   ```

## Usage

Run the script with the `--question` parameter:

```bash
python pinecone_gpt_query.py --question "Your question here"
```

### Optional Parameters

- `--deal_id`: Filter vector search to a specific deal ID
- `--top_k`: Number of vector chunks to retrieve (default: 5)
- `--output`: Output JSON file path (default: gpt_response.json)

Example:
```bash
python pinecone_gpt_query.py --question "What are the key terms of this agreement?" --deal_id "12345" --top_k 10 --output "custom_response.json"
```

## Output Files

The script generates the following output files:
- `retrieved_chunks.json`: Contains the raw vector chunks retrieved from Pinecone
- `gpt_response.json` (or custom name): Contains the GPT response along with chunks and usage statistics
- `pinecone_query.log`: Log file with detailed execution information

## Example Response

```json
{
    "answer": "The key terms of this agreement include...",
    "chunks": [
        {
            "id": "chunk-id",
            "score": 0.95,
            "metadata": {
                "combined_text": "Text content of the chunk...",
                "other_metadata": "..."
            }
        }
    ],
    "usage": {
        "prompt_tokens": 1256,
        "completion_tokens": 432,
        "total_tokens": 1688
    },
    "timestamp": "2023-10-25T15:30:45.123456"
}
``` 