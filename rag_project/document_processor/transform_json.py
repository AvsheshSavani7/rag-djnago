import json
import os
import re


def simplify_reference_section(ref_section):
    """
    Simplify reference sections to only include Section number and clauses.

    Examples:
    "ARTICLE Article III REPRESENTATIONS AND WARRANTIES OF THE COMPANY > Section 3.3 Authority Relative to Agreement > (b)"
    becomes "Section 3.3 (b)"

    "ARTICLE Article VIII GENERAL PROVISIONS > Section 8.11 Specific Performance"
    becomes "Section 8.11"

    "ARTICLE Article II EFFECT OF THE MERGER ON CAPITAL STOCK; EXCHANGE OF CERTIFICATES > Section 2.1 Effect on Securities > (a) > (ii)"
    becomes "Section 2.1 (a) (ii)"
    """
    if not ref_section:
        return ""

    # Extract the section number and name using regex
    section_match = re.search(r"Section\s+(\d+\.\d+)[^(>]*", ref_section)
    if not section_match:
        return ref_section

    section_num = section_match.group(1)
    result = f"Section {section_num}"

    # Extract all subsections/clauses (parts in parentheses)
    clause_matches = re.findall(r"\(([a-z0-9]+)\)", ref_section)
    if clause_matches:
        for clause in clause_matches:
            result += f" ({clause})"

    return result


def simplify_json(data):
    """
    Transform a complex JSON structure into a simplified format.
    Keeps original field names intact and extracts answer and reference_section.
    """
    # Create a new simplified structure
    simplified_data = {}

    # Process each main section
    for section_key, section_value in data.items():
        simplified_data[section_key] = {}

        # Handle sections that are lists (like conditions_to_closing)
        if isinstance(section_value, list):
            for item in section_value:
                field_name = item.get('field_name', '')
                if field_name:
                    # Keep the original field name
                    simplified_data[section_key][field_name] = {
                        "answer": item.get('answer', ''),
                        "clause_text": item.get('clause_text', ''),
                        "reference_section": simplify_reference_section(item.get('reference_section', '')),
                    }

        # Handle sections that are dictionaries
        elif isinstance(section_value, dict):
            # For dict-based sections with nested lists
            for subsection_key, subsection_value in section_value.items():
                # Initialize subsection
                simplified_data[section_key][subsection_key] = {}

                if isinstance(subsection_value, list):
                    # Process all fields
                    for item in subsection_value:
                        field_name = item.get('field_name', '')
                        if field_name:
                            # Keep the original field name
                            simplified_data[section_key][subsection_key][field_name] = {
                                "answer": item.get('answer', ''),
                                "clause_text": item.get('clause_text', ''),
                                "reference_section": simplify_reference_section(item.get('reference_section', ''))
                            }
                elif isinstance(subsection_value, dict):
                    # Handle direct dict mapping
                    field_name = subsection_value.get('field_name', '')
                    if field_name:
                        simplified_data[section_key][subsection_key][field_name] = {
                            "answer": subsection_value.get('answer', ''),
                            "clause_text": subsection_value.get('clause_text', ''),
                            "reference_section": simplify_reference_section(subsection_value.get('reference_section', ''))}

    return simplified_data


def transform_json_file(input_file, output_file):
    """
    Transform a JSON file using the simplify_json function.
    """
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Simplify the data
    simplified_data = simplify_json(data)

    # Write the simplified data to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(simplified_data, f, indent=2, ensure_ascii=False)

    print(f"Simplified JSON has been written to {output_file}")


if __name__ == "__main__":
    input_file = "Azek_04-06-25_Schema.json"
    output_file = "Azek_Schema.json"

    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
    else:
        transform_json_file(input_file, output_file)
