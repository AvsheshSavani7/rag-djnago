# Deal Schema Export Scripts

This directory contains scripts to export deal data and schema results from the MongoDB database.

## Scripts Available

### 1. `export_schema_results_only.py` (Recommended)
**Purpose**: Export only schema results with target and acquire company names
**Output**: Minimal JSON with deal_id, acquire_name, target_name, and schema_results

### 2. `export_all_deals_schema.py` (Comprehensive)
**Purpose**: Export all deal data including schema results and metadata
**Output**: Complete JSON with all deal fields and schema results

## Prerequisites

1. **Environment Setup**: Make sure your `.env` file contains the MongoDB connection string:
   ```
   MONGODB_CONNECTION_STRING=your_mongodb_connection_string
   MONGODB_NAME=your_database_name
   ```

2. **Python Environment**: Ensure you're in the correct Python environment with all dependencies installed.

## Usage

### Option 1: Run from the rag_project directory
```bash
cd rag_project
python export_schema_results_only.py
```

### Option 2: Run with full path
```bash
python rag_project/export_schema_results_only.py
```

## Output

The scripts will create a JSON file with the following structure:

### For `export_schema_results_only.py`:
```json
{
  "export_timestamp": "2025-01-XX...",
  "total_deals": 10,
  "deals": [
    {
      "deal_id": "deal_id_here",
      "acquire_name": "Acquiring Company Name",
      "target_name": "Target Company Name", 
      "schema_results": {
        // The actual schema results from the deal
      }
    }
  ]
}
```

### For `export_all_deals_schema.py`:
```json
{
  "export_timestamp": "2025-01-XX...",
  "total_deals": 10,
  "deals": [
    {
      "deal_id": "deal_id_here",
      "cik": "CIK_number",
      "acquire_name": "Acquiring Company Name",
      "target_name": "Target Company Name",
      "announce_date": "2025-01-XX...",
      "schema_results": {
        // The actual schema results from the deal
      },
      // ... all other deal fields
    }
  ]
}
```

## File Naming

The output files are automatically named with timestamps:
- `schema_results_export_YYYYMMDD_HHMMSS.json`
- `all_deals_schema_export_YYYYMMDD_HHMMSS.json`

## Error Handling

- The scripts handle cases where `schema_results` might be stored as a string or object
- Invalid JSON in `schema_results` will be marked with an error message
- The scripts provide progress updates and summary statistics

## Troubleshooting

1. **Database Connection Issues**: Check your MongoDB connection string in the `.env` file
2. **Import Errors**: Make sure you're running from the correct directory
3. **Permission Issues**: Ensure you have write permissions in the output directory

## Example Output

```
🚀 Starting schema results export...
🔍 Fetching all deals from database...
📊 Found 5 deals in database
Processing deal 1/5: Microsoft/Activision Blizzard
Processing deal 2/5: Adobe/Figma
Processing deal 3/5: Salesforce/Slack
Processing deal 4/5: Meta/WhatsApp
Processing deal 5/5: Amazon/Whole Foods
💾 Saving export to: /path/to/schema_results_export_20250129_143022.json
✅ Export completed successfully!
📁 File saved as: schema_results_export_20250129_143022.json
📊 Total deals exported: 5

📈 Summary:
   • Deals with schema results: 4
   • Deals without schema results: 1
   • Total deals processed: 5

🎉 Export completed! File saved at: /path/to/schema_results_export_20250129_143022.json
```
