#!/usr/bin/env python3
"""
Script to export schema results from all deals with target and acquire company names.
This script fetches all ProcessingJob documents and extracts their schema_results
along with target and acquire company names only.
"""

from document_processor.models import ProcessingJob
import django
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')

django.setup()


def export_schema_results():
    """
    Fetch schema results from all deals with company names and save to JSON file.
    """
    print("🔍 Fetching all deals from database...")

    # Fetch all ProcessingJob documents
    deals = ProcessingJob.objects.all()

    print(f"📊 Found {deals.count()} deals in database")

    # Prepare the export data
    export_data = {
        "export_timestamp": datetime.utcnow().isoformat(),
        "total_deals": deals.count(),
        "deals": []
    }

    # Process each deal
    for i, deal in enumerate(deals, 1):
        print(
            f"Processing deal {i}/{deals.count()}: {deal.acquire_name}/{deal.target_name}")

        # Extract schema results
        schema_results = {}
        if hasattr(deal, 'schema_results') and deal.schema_results:
            if isinstance(deal.schema_results, str):
                try:
                    schema_results = json.loads(deal.schema_results)
                except json.JSONDecodeError:
                    print(
                        f"  ⚠️  Warning: Could not parse schema_results JSON for deal {deal.id}")
                    schema_results = {
                        "error": "Invalid JSON in schema_results"}
            else:
                schema_results = deal.schema_results

        # Create deal entry with only essential information
        deal_entry = {
            "deal_id": str(deal.id),
            "acquire_name": deal.acquire_name,
            "target_name": deal.target_name,
            "schema_results": schema_results
        }

        export_data["deals"].append(deal_entry)

    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"schema_results_export_{timestamp}.json"
    output_path = project_root / output_filename

    # Save to JSON file
    print(f"💾 Saving export to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"✅ Export completed successfully!")
    print(f"📁 File saved as: {output_filename}")
    print(f"📊 Total deals exported: {len(export_data['deals'])}")

    # Print summary statistics
    deals_with_schema = sum(
        1 for deal in export_data['deals'] if deal['schema_results'] and deal['schema_results'] != {})
    deals_without_schema = len(export_data['deals']) - deals_with_schema

    print(f"\n📈 Summary:")
    print(f"   • Deals with schema results: {deals_with_schema}")
    print(f"   • Deals without schema results: {deals_without_schema}")
    print(f"   • Total deals processed: {len(export_data['deals'])}")

    return output_path


def main():
    """Main function to run the export script."""
    try:
        print("🚀 Starting schema results export...")
        output_path = export_schema_results()
        print(f"\n🎉 Export completed! File saved at: {output_path}")

    except Exception as e:
        print(f"❌ Error during export: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
