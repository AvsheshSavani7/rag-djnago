#!/usr/bin/env python3
"""
Script to generate Excel sheet from schema results export file.
This script creates an Excel file with columns:
- Category (e.g., best_efforts)
- Sub Category (e.g., divestiture_commitments) 
- Field Name (e.g., divestiture_cap_buyer_notes)
- One column per deal (target_name vs acquire_name)
"""

import json
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
import argparse


def flatten_schema_results(schema_results, parent_key='', sep='.'):
    """
    Flatten nested schema results into a flat dictionary with dot notation keys.
    Returns a dictionary with keys like 'best_efforts.divestiture_commitments.divestiture_cap_buyer_notes'
    """
    items = []

    if isinstance(schema_results, dict):
        for key, value in schema_results.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):
                # Check if this is a field with 'answer' key (actual data)
                if 'answer' in value:
                    items.append((new_key, value.get('answer', '')))
                else:
                    # Recursively flatten nested dictionaries
                    items.extend(flatten_schema_results(value, new_key, sep))
            else:
                items.append((new_key, value))
    elif isinstance(schema_results, list):
        # Handle lists by joining with semicolon
        items.append((parent_key, '; '.join(str(item)
                     for item in schema_results)))
    else:
        items.append((parent_key, schema_results))

    return items


def parse_category_structure(key):
    """
    Parse a flattened key to extract category, sub_category, and field_name.
    The key format is typically: 'category.sub_category.field_name' or 'category.field_name'
    Returns: (category, sub_category, field_name)
    """
    # Split by dot to get the hierarchical structure
    parts = key.split('.')

    if len(parts) >= 3:
        # Format: category.sub_category.field_name
        category = parts[0]
        sub_category = parts[1]
        field_name = '.'.join(parts[2:])  # In case field_name contains dots
    elif len(parts) == 2:
        # Format: category.field_name
        category = parts[0]
        sub_category = parts[0]  # Same as category when no sub_category
        field_name = parts[1]
    else:
        # Single part - treat as category
        category = key
        sub_category = key
        field_name = key

    return category, sub_category, field_name


def format_value(value):
    """
    Format a value for display in Excel.
    """
    if value is None:
        return ""
    elif isinstance(value, bool):
        return "Yes" if value else "No"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, list):
        return "; ".join(str(item) for item in value)
    else:
        return str(value)


def merge_consecutive_cells(worksheet, column_letter, values):
    """
    Merge consecutive cells in a column that have the same value.

    Args:
        worksheet: The worksheet to modify
        column_letter: The column letter (e.g., 'A', 'B')
        values: List of values in the column (excluding header)
    """
    from openpyxl.styles import Alignment

    if not values:
        return

    start_row = 2  # Start from row 2 (after header)
    current_value = values[0]
    current_start = start_row

    for i, value in enumerate(values[1:], start=1):
        row = start_row + i

        if value != current_value:
            # Merge the previous group if it has more than one row
            if row - 1 > current_start:
                merge_range = f"{column_letter}{current_start}:{column_letter}{row - 1}"
                worksheet.merge_cells(merge_range)

                # Center align the merged cell
                merged_cell = worksheet[f"{column_letter}{current_start}"]
                merged_cell.alignment = Alignment(
                    horizontal="center", vertical="center")

            # Start new group
            current_value = value
            current_start = row

    # Handle the last group
    if start_row + len(values) - 1 > current_start:
        merge_range = f"{column_letter}{current_start}:{column_letter}{start_row + len(values) - 1}"
        worksheet.merge_cells(merge_range)

        # Center align the merged cell
        merged_cell = worksheet[f"{column_letter}{current_start}"]
        merged_cell.alignment = Alignment(
            horizontal="center", vertical="center")


def generate_excel_from_schema(json_file_path, output_file_path=None):
    """
    Generate Excel file from schema results JSON export.
    """
    print(f"📖 Reading schema results from: {json_file_path}")

    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    deals = data.get('deals', [])
    print(f"📊 Found {len(deals)} deals in the export file")

    # Collect all unique field keys across all deals
    all_fields = set()
    deal_data = {}

    for deal in deals:
        deal_id = deal['deal_id']
        acquire_name = deal.get('acquire_name', 'Unknown')
        target_name = deal.get('target_name', 'Unknown')
        deal_key = f"{target_name} vs {acquire_name}"

        schema_results = deal.get('schema_results', {})
        if schema_results:
            # Flatten the schema results
            flattened = dict(flatten_schema_results(schema_results))
            deal_data[deal_key] = flattened
            all_fields.update(flattened.keys())

    print(f"🔍 Found {len(all_fields)} unique fields across all deals")

    # Create the Excel data
    excel_data = []

    for field_key in sorted(all_fields):
        category, sub_category, field_name = parse_category_structure(
            field_key)

        row = {
            'Category': category,
            'Sub Category': sub_category,
            'Field Name': field_name
        }

        # Add values for each deal
        for deal_key in deal_data.keys():
            value = deal_data[deal_key].get(field_key, "")
            row[deal_key] = format_value(value)

        excel_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(excel_data)

    # Reorder columns: Category, Sub Category, Field Name, then deal columns
    deal_columns = [col for col in df.columns if col not in [
        'Category', 'Sub Category', 'Field Name']]
    df = df[['Category', 'Sub Category', 'Field Name'] + deal_columns]

    # Generate output filename if not provided
    if output_file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = f"schema_comparison_{timestamp}.xlsx"

    print(f"💾 Saving Excel file to: {output_file_path}")

    # Create Excel writer with formatting
    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Schema Comparison', index=False)

        # Get the workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Schema Comparison']

        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter

            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass

            adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
            worksheet.column_dimensions[column_letter].width = adjusted_width

        # Format header row
        from openpyxl.styles import Font, PatternFill, Alignment

        header_font = Font(bold=True)
        header_fill = PatternFill(
            start_color="366092", end_color="366092", fill_type="solid")
        header_alignment = Alignment(horizontal="center", vertical="center")

        for cell in worksheet[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment

        # Merge cells for same category and sub-category names
        print("🔗 Merging cells for same category names...")

        # Merge Category column cells
        merge_consecutive_cells(worksheet, 'A', df['Category'].tolist())

        # Merge Sub Category column cells
        merge_consecutive_cells(worksheet, 'B', df['Sub Category'].tolist())

        print("✅ Cell merging completed!")

    print(f"✅ Excel file generated successfully!")
    print(f"📁 File saved as: {output_file_path}")
    print(f"📊 Total rows: {len(df)}")
    print(f"📊 Total columns: {len(df.columns)}")

    # Print summary
    print(f"\n📈 Summary:")
    print(f"   • Categories found: {df['Category'].nunique()}")
    print(f"   • Sub-categories found: {df['Sub Category'].nunique()}")
    print(f"   • Total fields: {len(df)}")
    print(f"   • Deals compared: {len(deal_columns)}")

    return output_file_path


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description='Generate Excel sheet from schema results export')
    parser.add_argument(
        'input_file', help='Path to the schema results JSON export file')
    parser.add_argument(
        '-o', '--output', help='Output Excel file path (optional)')

    args = parser.parse_args()

    try:
        print("🚀 Starting Excel generation from schema results...")
        output_path = generate_excel_from_schema(args.input_file, args.output)
        print(f"\n🎉 Excel generation completed! File saved at: {output_path}")

    except Exception as e:
        print(f"❌ Error during Excel generation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # If no command line arguments, use the default file
    if len(sys.argv) == 1:
        # Look for the most recent schema export file
        schema_files = list(Path('.').glob('schema_results_export_*.json'))
        if schema_files:
            latest_file = max(schema_files, key=lambda x: x.stat().st_mtime)
            print(f"🔍 Using latest schema export file: {latest_file}")
            sys.argv = [sys.argv[0], str(latest_file)]

    main()
