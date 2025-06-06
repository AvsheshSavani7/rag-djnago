import openai
from django_init import init_django
from document_processor.services import SchemaCategorySearch
import streamlit as st
from streamlit_ace import st_ace
import json
from pathlib import Path
import os
from document_processor.services import S3Service
import difflib
from typing import Dict, Any
import logging
from datetime import datetime
from copy import deepcopy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            f'schema_search_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

# Create a logger for this module
logger = logging.getLogger(__name__)

# Initialize Django first, before any other imports
init_django()

# Now we can safely import Django-related modules

# Set page config
st.set_page_config(
    page_title="Schema Enhancement Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
    theme="light"
)


# Add custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .json-output {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .json-viewer {
        font-family: monospace;
        white-space: pre;
        overflow-x: auto;
    }
    .diff-viewer {
        font-family: monospace;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .diff-added {
        background-color: #e6ffe6;
        color: #006400;
    }
    .diff-removed {
        background-color: #ffe6e6;
        color: #640000;
    }
    .diff-header {
        color: #666;
        margin-bottom: 10px;
    }
    .confirm-merge-btn {
        background-color: #ff4b4b;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .streamlit-expanderHeader {
        font-size: 1.1em;
        font-weight: 600;
    }
    .node-key {
        color: #2980b9;
        font-weight: bold;
    }
    .node-type {
        color: #7f8c8d;
        font-style: italic;
    }
    .node-value {
        color: #27ae60;
    }
    .node-container {
        margin-left: 20px;
        border-left: 1px solid #bdc3c7;
        padding-left: 10px;
    }
    .scrollable-prompt {
        max-height: 500px;
        overflow-y: auto;
        overflow-x: hidden;
        border: 1px solid #ccc;
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 5px;
        margin: 10px 0;
        white-space: pre-wrap;
        word-wrap: break-word;
        font-size: 14px;
    }
    code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        font-size: 14px !important;
    }
    .stMarkdown {
        font-size: 15px;
    }
    h1 {
        font-size: 26px !important;
    }
    h2 {
        font-size: 22px !important;
    }
    .streamlit-expanderHeader {
        font-size: 16px !important;
    }
    p {
        font-size: 15px !important;
        margin: 0 !important;
        padding: 2px 0 !important;
    }
    .sticky-top {
        position: sticky;
        top: 0;
        z-index: 999;
        background-color: white;
        padding: 10px 0;
        border-bottom: 1px solid #eee;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize S3 service
s3_service = S3Service()

# Initialize session state for schema and selected section
if 'edited_schema' not in st.session_state:
    st.session_state.edited_schema = {}
if 'full_schema' not in st.session_state:
    # Load the original schema from S3
    try:
        schema_url = "https://rag-mna.s3.eu-north-1.amazonaws.com/clauses_category_template/streamlit_schema_by_summary_sections.json"
        logger.info(f"Loading schema from {schema_url}")
        st.session_state.full_schema = s3_service.download_from_url(schema_url)
        logger.info("Successfully loaded schema from S3")
    except Exception as e:
        logger.error(f"Error loading schema: {str(e)}")
        st.error(f"Error loading schema: {str(e)}")
        st.session_state.full_schema = {}
if 'selected_section' not in st.session_state:
    st.session_state.selected_section = None


def display_json_node(key, value, level=0):
    """Display a JSON node with proper formatting and type information"""
    indent = "  " * level

    if isinstance(value, dict):
        st.markdown(
            f"{indent}📂 <span class='node-key'>{key}</span> <span class='node-type'>(object)</span>", unsafe_allow_html=True)
        container = st.container()
        with container:
            for k, v in value.items():
                display_json_node(k, v, level + 1)

    elif isinstance(value, list):
        st.markdown(
            f"{indent}📂 <span class='node-key'>{key}</span> <span class='node-type'>(array [{len(value)} items])</span>", unsafe_allow_html=True)
        container = st.container()
        with container:
            for i, item in enumerate(value):
                display_json_node(f"Item {i+1}", item, level + 1)

    else:
        value_str = str(value)
        if len(value_str) > 50:
            value_str = value_str[:47] + "..."
        st.markdown(
            f"{indent}📄 <span class='node-key'>{key}</span>: <span class='node-value'>{value_str}</span>",
            unsafe_allow_html=True
        )


def deep_merge_filtered_into_full(full, filtered):
    logger.info(f"Full: {json.dumps(filtered, indent=2)}")
    if isinstance(full, list) and isinstance(filtered, list):
        result = []
        for full_item in full:
            if isinstance(full_item, dict) and 'field_name' in full_item:
                match = next(
                    (f for f in filtered if f.get('field_name')
                     == full_item['field_name']),
                    None
                )
                if match:
                    merged = full_item.copy()
                    merged.update(match)
                    result.append(merged)
                else:
                    result.append(full_item)
        return result
    elif isinstance(full, dict) and isinstance(filtered, dict):
        merged = {}
        for key in full:
            if key in filtered:
                merged[key] = deep_merge_filtered_into_full(
                    full[key], filtered[key])
            else:
                merged[key] = full[key]
        return merged
    else:
        return filtered or full


def filter_schema_fields(schema):
    """Filter schema to show only specific fields"""
    FIELDS_TO_KEEP = ['instructions',
                      'recommended_prompt_type', 'question_query', 'field_name']

    if isinstance(schema, dict):
        filtered_schema = {}
        for key, value in schema.items():
            if isinstance(value, list):
                # For direct arrays (like conditions_to_closing)
                filtered_list = []
                for item in value:
                    if isinstance(item, dict):
                        filtered_item = {
                            k: v for k, v in item.items() if k in FIELDS_TO_KEEP}
                        if filtered_item:  # Only add if we have filtered fields
                            filtered_list.append(filtered_item)
                if filtered_list:  # Only add if the filtered list is not empty
                    filtered_schema[key] = filtered_list
            elif isinstance(value, dict):
                filtered_value = filter_schema_fields(value)
                if filtered_value:  # Only add if the filtered dict is not empty
                    filtered_schema[key] = filtered_value
            elif key in FIELDS_TO_KEEP:
                filtered_schema[key] = value
        return filtered_schema
    elif isinstance(schema, list):
        # For direct array inputs
        filtered_list = []
        for item in schema:
            if isinstance(item, dict):
                filtered_item = {k: v for k,
                                 v in item.items() if k in FIELDS_TO_KEEP}
                if filtered_item:  # Only add if we have filtered fields
                    filtered_list.append(filtered_item)
        return filtered_list if filtered_list else schema
    return schema


def filter_prompt_fields(schema):
    """Filter schema to show only prompt-related fields"""
    if isinstance(schema, dict):
        filtered_schema = {}
        for section_name, section_data in schema.items():
            if isinstance(section_data, dict):
                filtered_section = {}
                for field_name, field_items in section_data.items():
                    if isinstance(field_items, list):
                        filtered_items = []
                        for item in field_items:
                            if isinstance(item, dict):
                                # Keep all fields but ensure we have the prompt
                                filtered_item = item.copy()  # Keep all original fields
                                # Add prompt from the schema if available
                                if 'prompt' not in filtered_item:
                                    # Try to find prompt in parent levels
                                    if isinstance(section_data.get(field_name), dict):
                                        filtered_item['prompt'] = section_data[field_name].get(
                                            'prompt', '')
                                    elif isinstance(schema.get(section_name), dict):
                                        filtered_item['prompt'] = schema[section_name].get(
                                            'prompt', '')
                                filtered_items.append(filtered_item)
                        if filtered_items:
                            filtered_section[field_name] = filtered_items
                if filtered_section:
                    filtered_schema[section_name] = filtered_section
            elif isinstance(section_data, list):
                filtered_items = []
                for item in section_data:
                    if isinstance(item, dict):
                        filtered_item = item.copy()  # Keep all original fields
                        # Try to find prompt in parent level
                        if 'prompt' not in filtered_item:
                            filtered_item['prompt'] = schema.get(
                                section_name, {}).get('prompt', '')
                        filtered_items.append(filtered_item)
                if filtered_items:
                    filtered_schema[section_name] = filtered_items
        return filtered_schema
    return schema


def generate_diff(original: Dict[str, Any], modified: Dict[str, Any], section: str) -> str:
    """Generate a diff between two JSON objects"""
    original_json = json.dumps(original.get(
        section, {}), indent=2, sort_keys=True).splitlines()
    modified_json = json.dumps(modified.get(
        section, {}), indent=2, sort_keys=True).splitlines()

    diff = list(difflib.ndiff(original_json, modified_json))

    html_diff = []
    for line in diff:
        if line.startswith('+'):
            html_diff.append(f'<div class="diff-added">{line}</div>')
        elif line.startswith('-'):
            html_diff.append(f'<div class="diff-removed">{line}</div>')
        else:
            html_diff.append(f'<div>{line}</div>')

    return ''.join(html_diff)


def filter_schema_for_diff(schema):
    """Filter schema to show only specific fields for diff view"""
    FIELDS_TO_SHOW = ['instructions',
                      'recommended_prompt_type', 'question_query', 'field_name']

    if isinstance(schema, dict):
        filtered = {}
        for key, value in schema.items():
            if isinstance(value, dict):
                filtered_value = filter_schema_for_diff(value)
                if filtered_value:
                    filtered[key] = filtered_value
            elif isinstance(value, list):
                filtered_list = []
                for item in value:
                    if isinstance(item, dict):
                        filtered_item = {
                            k: v for k, v in item.items() if k in FIELDS_TO_SHOW}
                        if filtered_item:
                            filtered_list.append(filtered_item)
                if filtered_list:
                    filtered[key] = filtered_list
            elif key in FIELDS_TO_SHOW:
                filtered[key] = value
        return filtered
    elif isinstance(schema, list):
        filtered_list = []
        for item in schema:
            if isinstance(item, dict):
                filtered_item = {k: v for k,
                                 v in item.items() if k in FIELDS_TO_SHOW}
                if filtered_item:
                    filtered_list.append(filtered_item)
        return filtered_list
    return schema


def deep_merge_filtered_back(full_schema, filtered_schema):
    """Merge filtered schema back into full schema, preserving all original fields"""
    FIELDS_TO_MERGE = ['instructions',
                       'recommended_prompt_type', 'question_query', 'field_name']

    if isinstance(filtered_schema, dict) and isinstance(full_schema, dict):
        result = deepcopy(full_schema)
        for key, value in filtered_schema.items():
            if key in FIELDS_TO_MERGE:
                result[key] = value
            elif isinstance(value, dict) and key in result:
                result[key] = deep_merge_filtered_back(result[key], value)
            elif isinstance(value, list) and key in result:
                # Handle list of dictionaries
                result[key] = []
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        # Try to find matching item in original list
                        original_item = next((orig for orig in full_schema[key]
                                              if isinstance(orig, dict) and
                                              orig.get('field_name') == item.get('field_name')), None)
                        if original_item:
                            merged_item = deepcopy(original_item)
                            for field in FIELDS_TO_MERGE:
                                if field in item:
                                    merged_item[field] = item[field]
                            result[key].append(merged_item)
                        else:
                            result[key].append(item)
                    else:
                        result[key].append(item)
            else:
                result[key] = value
        return result
    return filtered_schema


# Title
st.title("📑 Schema Search Tool")

# Create a sticky container for dropdowns
st.markdown('<div class="sticky-top">', unsafe_allow_html=True)

# Create three columns for dropdowns and run button
col1, col2, col3 = st.columns([1, 1, 0.3])

with col1:
    # List of available deals with their names
    DEALS = {
        "68184d52478abf06ec1a28ec": "Azek",
        "682f00def21b9fca8e1d04fe": "Spirit Airlines",
        "68347651c88b3f7f9c69410a": "Catalent",
        "6836dbf3caf74b95439aeeba": "ChampionX Corp",
        "682f252ef21b9fca8e1d0530": "Silicon Motion",
        "684054f02e2e5aa5468773db": "United States Steel",
        "68412f11812d9ee0838c6fd4": "Celgene Corporation"
    }

    # Create the deal selector
    selected_deal = st.selectbox(
        "Select a Deal",
        options=list(DEALS.keys()),
        format_func=lambda x: f"{DEALS[x]} ({x})"
    )

with col2:
    # Available sections
    SECTIONS = [
        "termination",
        "ordinary_course",
        "board_approval",
        "party_details",
        "conditions_to_closing",
        "closing_mechanics",
        "specific_performance",
        "confidentiality_and_clean_room",
        "complex_consideration_and_dividends",
        "law_and_jurisdiction",
        "financing",
        "proxy_statement",
        "timeline",
        "material_adverse_effect",
        "non_solicitation",
        "best_efforts"
    ]

    # Create the section selector (single select)
    selected_section = st.selectbox(
        "Select Section to Search",
        options=SECTIONS,
        help="Choose a section to search through"
    )

with col3:
    # Add some vertical spacing to align with dropdowns
    st.write("")
    st.write("")
    # Add a search button
    run_button = st.button("🔍 Run", type="secondary", key="run_button")

# Close the sticky container
st.markdown('</div>', unsafe_allow_html=True)

# Store selected section in session state
st.session_state.selected_section = selected_section

# Create tabs for different functionalities
search_tab, edit_tab, review_tab = st.tabs(
    ["🔍 Results", "✏️ Edit Schema", "📋 Review Changes"])

if run_button:
    with search_tab:
        try:
            with st.spinner("Searching schema categories..."):
                logger.info(
                    f"Starting search for deal: {selected_deal}, section: {selected_section}")
                # Initialize the search service
                search_service = SchemaCategorySearch()

                # First try to use edited schema from session state
                search_schema = {}
                if selected_section in st.session_state.edited_schema:
                    logger.info(
                        f"Using edited schema for section: {selected_section}")
                    search_schema[selected_section] = st.session_state.edited_schema[selected_section]
                else:
                    logger.info(
                        f"Using original schema for section: {selected_section}")
                    search_schema[selected_section] = st.session_state.full_schema.get(
                        selected_section, {})

                search_service._schema = search_schema
                logger.debug(
                    f"Search schema: {json.dumps(search_schema, indent=2)}")

                # Perform the search
                results = search_service.search_all_schema_categories(
                    selected_deal)
                logger.info(f"Search completed for deal: {selected_deal}")

                if results:
                    st.success(
                        f"Search completed for {DEALS[selected_deal]}")

                    # Create tabs for different views
                    raw_tab, prompts_tab = st.tabs(["🔍 Raw JSON", "📝 Output"])

                    with raw_tab:
                        # Display raw JSON
                        st.json(results)

                    with prompts_tab:
                        # Filter and display only prompt-related fields
                        filtered_results = filter_prompt_fields(results)

                        def format_value(value):
                            """Format value based on its type"""
                            if isinstance(value, (int, float, bool)) or value is None:
                                return str(value)
                            return value

                        def display_prompts_hierarchy(data):
                            if isinstance(data, dict):
                                for section_name, section_data in data.items():
                                    st.markdown(f"# {section_name}")

                                    # Handle both dictionary and list section_data
                                    items_to_process = []
                                    if isinstance(section_data, dict):
                                        # For nested dictionary structure
                                        for field_name, items in section_data.items():
                                            if isinstance(items, list):
                                                for item in items:
                                                    if isinstance(item, dict):
                                                        item['parent_field'] = field_name
                                                        items_to_process.append(
                                                            item)
                                    elif isinstance(section_data, list):
                                        # For direct list structure
                                        items_to_process.extend(
                                            [item for item in section_data if isinstance(item, dict)])

                                    # Process all items
                                    for idx, item in enumerate(items_to_process, 1):
                                        parent_field = item.get(
                                            'parent_field', '')
                                        field_name = item.get(
                                            'field_name', 'Unknown Field')
                                        display_name = f"{parent_field} - {field_name}" if parent_field else field_name

                                        with st.expander(f"Field {idx} - {display_name}", expanded=False):
                                            # Display fields in single column
                                            fields_to_display = [
                                                ('Section', 'section'),
                                                ('Field Name', 'field_name'),
                                                # ('Answer', 'answer'),
                                                ('Summary', 'summary'),
                                                ('Confidence', 'confidence'),
                                                ('Reason', 'reason'),
                                                ('Reference Section',
                                                 'reference_section')
                                            ]

                                            for label, key in fields_to_display:
                                                if key in item and item[key]:
                                                    value = format_value(
                                                        item[key])
                                                    st.markdown(
                                                        f"**{label}:** {value}")

                                            # Display clause text if available
                                            if 'answer' in item and item['answer']:
                                                st.markdown("**Answer:**")
                                                st.markdown(
                                                    f"```\n{item['answer']}\n```")
                                            # Display clause text if available
                                            if 'clause_text' in item and item['clause_text']:
                                                st.markdown("**Clause Text:**")
                                                st.markdown(
                                                    f"```\n{item['clause_text']}\n```")

                                            # Display prompt in scrollable container if available
                                            if 'prompt' in item and item['prompt']:
                                                st.markdown("**Prompt:**")
                                                prompt_html = f"""<div class="scrollable-prompt">{item['prompt']}</div>"""
                                                st.markdown(
                                                    prompt_html, unsafe_allow_html=True)

                        display_prompts_hierarchy(filtered_results)

                else:
                    st.warning(
                        f"No results found for {selected_section} in deal {DEALS[selected_deal]}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Full error details")

with edit_tab:
    st.header("Edit Schema")

    if st.session_state.selected_section:
        edit_section = st.session_state.selected_section
        # Get current schema for the section
        full_schema = st.session_state.edited_schema.get(
            edit_section,
            st.session_state.full_schema.get(edit_section, {})
        )

        # Filter the schema to show only specific fields
        current_schema = filter_schema_fields(full_schema)

        st.subheader(f"JSON Editor - {edit_section}")
        # Create an ACE editor for JSON
        edited_content = st_ace(
            value=json.dumps(current_schema, indent=2),
            language="json",
            theme="monokai",
            # Add hash to force re-render
            key=f"ace_editor_{edit_section}_{hash(json.dumps(current_schema))}",
            height=500,
            show_gutter=True,
            wrap=True,
            auto_update=True,
            font_size=14,
        )

        # Save button
        if st.button("💾 Save Changes", type="primary"):
            try:
                logger.info(f"Saving changes for section: {edit_section}")
                edited_json = json.loads(edited_content)

                # Get the full original schema for this section
                full_section = deepcopy(
                    st.session_state.full_schema.get(edit_section, {}))

                # Merge edited filtered fields back into the full schema
                merged = deep_merge_filtered_back(full_section, edited_json)

                # Save to session state
                st.session_state.edited_schema[edit_section] = merged

                st.success("Successfully saved changes locally")

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON format: {str(e)}")
                st.error(f"Invalid JSON format: {str(e)}")
            except Exception as e:
                logger.error(f"Error saving changes: {str(e)}")
                st.error(f"Error saving changes: {str(e)}")

with review_tab:
    st.header("Review Changes")

    if not st.session_state.edited_schema:
        st.info("No changes to review. Make some changes in the Edit Schema tab first.")
    else:
        # Create columns for the action buttons
        col1, col2 = st.columns(2)

        # Reset button in first column
        if col1.button("🔄 Discard Changes"):
            logger.info("Discarding all changes")
            # Clear the edited schema
            st.session_state.edited_schema = {}
            # Clear any cached keys for the ACE editor
            for key in list(st.session_state.keys()):
                if key.startswith('ace_editor_'):
                    del st.session_state[key]
            logger.info("Successfully discarded all changes")
            st.success(f"Discarded all changes")
            st.rerun()

        # Merge button in second column
        if col2.button("🔄 Confirm and Merge Changes to S3", type="primary"):
            try:
                logger.info("Starting merge process to S3")
                # Update the main schema in S3
                schema_url = "clauses_category_template/streamlit_schema_by_summary_sections.json"
                main_schema = st.session_state.full_schema.copy()

                # Merge all changes
                for section, modified_schema in st.session_state.edited_schema.items():
                    logger.info(f"Merging changes for section: {section}")
                    main_schema[section] = modified_schema

                # Upload to S3
                logger.info("Uploading merged schema to S3")
                s3_url = s3_service.upload_json(main_schema, schema_url)

                # Update the full schema in session state
                st.session_state.full_schema = main_schema

                # Clear the edited schema after successful merge
                st.session_state.edited_schema = {}

                logger.info("Successfully merged all changes to S3")
                st.success("Successfully merged all changes to S3")
                st.rerun()

            except Exception as e:
                logger.error(f"Error merging changes: {str(e)}")
                st.error(f"Error merging changes: {str(e)}")

        # Show changes for each modified section
        st.markdown("### Changes by Section")
        logger.info("Displaying diffs for modified sections")

        for section in st.session_state.edited_schema:
            with st.expander(f"Changes in {section}", expanded=True):
                st.markdown(f"#### Comparing changes for section: {section}")

                # Get original and modified schemas
                original = st.session_state.full_schema.get(section, {})
                modified = st.session_state.edited_schema[section]

                # Filter schemas for diff display
                filtered_original = filter_schema_for_diff({section: original})
                filtered_modified = filter_schema_for_diff({section: modified})

                # Log the comparison of filtered schemas
                logger.debug(f"Comparing filtered section {section}:")

                # Generate and display diff using filtered schemas
                diff_html = generate_diff(
                    filtered_original,
                    filtered_modified,
                    section
                )

                # Display the diff with some styling
                st.markdown("""
                    <style>
                    .diff-container {
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        padding: 10px;
                        margin: 10px 0;
                        background-color: #f8f9fa;
                    }
                    </style>
                """, unsafe_allow_html=True)

                st.markdown('<div class="diff-container">',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<div class="diff-viewer">{diff_html}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

# Add helpful information
with st.sidebar.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Search Tab:**
       - Select a deal ID from the dropdown
       - Choose a section to search through
       - Click '🔍 Run' to view results
       - View results in  Raw JSON format
    
    2. **Edit Tab:**
       - Select a section to edit from the dropdown
       - Make changes in the JSON editor
       - The editor provides:
         • Syntax highlighting
         • Line numbers
         • Auto-indentation
         • Error detection
       - Click '💾 Save Changes' to save
       - Click '🔄 Reset to Original' to revert changes
       
    3. **Note on Saving:**
       - Changes are saved to a separate edited schema file
       - Original schema remains unchanged
       - Searches will use edited version when available
    """)

# Footer
st.sidebar.markdown("---")
