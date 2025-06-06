from document_processor.services import SchemaCategorySearch
from django.conf import settings
import json
import streamlit as st
import django
import os
import sys
from pathlib import Path
from streamlit_ace import st_ace

# Setup Django environment first
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')

# Only setup Django if it hasn't been set up yet
if not django.conf.settings.configured:
    django.setup()

# Set page config
st.set_page_config(
    page_title="Schema Search Tool",
    page_icon="🔍",
    layout="wide"
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
    </style>
""", unsafe_allow_html=True)

# Initialize session state for schema
if 'edited_schema' not in st.session_state:
    st.session_state.edited_schema = {}
if 'full_schema' not in st.session_state:
    # Load the original schema
    schema_path = os.path.join(current_dir, 'schema_by_summary_sections.json')
    try:
        with open(schema_path, 'r') as f:
            st.session_state.full_schema = json.load(f)
    except Exception as e:
        st.error(f"Error loading schema: {str(e)}")
        st.session_state.full_schema = {}


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


# Title
st.title("📑 Schema Search Tool")

# Create tabs for different functionalities
search_tab, edit_tab = st.tabs(["🔍 Search", "✏️ Edit Schema"])

with search_tab:
    # Sidebar with deal IDs and sections
    st.sidebar.header("Search Configuration")

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

    # Create the deal selector
    selected_deal = st.sidebar.selectbox(
        "Select a Deal",
        options=list(DEALS.keys()),
        format_func=lambda x: f"{DEALS[x]} ({x})"
    )

    # Create the sections multiselect
    selected_sections = st.sidebar.multiselect(
        "Select Sections to Search",
        options=SECTIONS,
        default=["ordinary_course"],
        help="Choose one or more sections to search through"
    )

    # Add a search button
    if st.sidebar.button("🔍 Search Schema", type="secondary"):
        if not selected_sections:
            st.warning("Please select at least one section to search")
        else:
            try:
                with st.spinner("Searching schema categories..."):
                    # Initialize the search service
                    search_service = SchemaCategorySearch()

                    # Use edited schema if available, otherwise use original
                    search_schema = {}
                    for section in selected_sections:
                        if section in st.session_state.edited_schema:
                            search_schema[section] = st.session_state.edited_schema[section]
                        else:
                            search_schema[section] = st.session_state.full_schema.get(
                                section, {})

                    search_service._schema = search_schema

                    # Perform the search
                    results = search_service.search_all_schema_categories(
                        selected_deal)

                    if results:
                        st.success(
                            f"Search completed for {DEALS[selected_deal]}")

                        # Create tabs for different views
                        tab1, tab2 = st.tabs(
                            ["📊 Formatted View", "🔍 Raw JSON"])

                        with tab1:
                            # Display results in a more structured way
                            for category, data in results.items():
                                with st.expander(f"Category: {category}"):
                                    if isinstance(data, dict):
                                        for key, value in data.items():
                                            st.markdown(f"**{key}:**")
                                            st.markdown(value)
                                    else:
                                        st.markdown(data)

                        with tab2:
                            # Display raw JSON
                            st.json(results)
                    else:
                        st.warning(
                            f"No results found for the selected sections in deal {DEALS[selected_deal]}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Full error details:", exc_info=True)

with edit_tab:
    st.header("Edit Schema")

    # Section selector for editing
    edit_section = st.selectbox(
        "Select Section to Edit",
        options=SECTIONS,
        help="Choose a section to edit its schema"
    )

    if edit_section:
        # Get current schema for the section
        current_schema = st.session_state.edited_schema.get(
            edit_section,
            st.session_state.full_schema.get(edit_section, {})
        )

        st.subheader("JSON Editor")
        # Create an ACE editor for JSON
        edited_content = st_ace(
            value=json.dumps(current_schema, indent=2),
            language="json",
            theme="monokai",
            key=f"ace_editor_{edit_section}",
            height=500,
            show_gutter=True,
            wrap=True,
            auto_update=True,
            font_size=14,
        )

        # Save options and buttons
        save_location = st.radio(
            "Save Location",
            ["Save to edited_schema.json", "Save to original schema file"],
            help="Choose where to save your changes. Be careful when saving to the original file!"
        )

        # Save and Reset buttons
        save_col, reset_col = st.columns([1, 1])
        with save_col:
            if st.button("💾 Save Changes", type="primary"):
                try:
                    # Parse the edited JSON to validate it
                    edited_json = json.loads(edited_content)

                    if save_location == "Save to edited_schema.json":
                        # Save to session state
                        st.session_state.edited_schema[edit_section] = edited_json

                        # Save to edited_schema.json
                        edited_schema_path = os.path.join(
                            current_dir, 'edited_schema.json')

                        # Load existing edited schema if it exists
                        existing_edited_schema = {}
                        if os.path.exists(edited_schema_path):
                            with open(edited_schema_path, 'r') as f:
                                existing_edited_schema = json.load(f)

                        # Update with new changes
                        existing_edited_schema[edit_section] = edited_json

                        # Save back to file
                        with open(edited_schema_path, 'w') as f:
                            json.dump(existing_edited_schema, f, indent=2)

                        st.success(
                            f"Successfully saved changes to edited_schema.json")
                    else:
                        # Save directly to original schema file
                        schema_path = os.path.join(
                            current_dir, 'schema_by_summary_sections.json')

                        # Load current schema
                        with open(schema_path, 'r') as f:
                            full_schema = json.load(f)

                        # Update the section
                        full_schema[edit_section] = edited_json

                        # Save back to file
                        with open(schema_path, 'w') as f:
                            json.dump(full_schema, f, indent=2)

                        # Update session state
                        st.session_state.full_schema = full_schema
                        if edit_section in st.session_state.edited_schema:
                            del st.session_state.edited_schema[edit_section]

                        st.success(
                            f"Successfully saved changes to original schema file")

                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON format: {str(e)}")
                except Exception as e:
                    st.error(f"Error saving changes: {str(e)}")

        with reset_col:
            if st.button("🔄 Reset to Original"):
                if edit_section in st.session_state.edited_schema:
                    del st.session_state.edited_schema[edit_section]
                st.success(f"Reset {edit_section} to original schema")
                st.rerun()

# Add helpful information
with st.sidebar.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Search Tab:**
       - Select a deal ID from the dropdown
       - Choose one or more sections to search through
       - Click '🔍 Search Schema' to view results
       - View results in either Formatted or Raw JSON format
    
    2. **Edit Tab:**
       - Select a section to edit from the dropdown
       - Make changes in the JSON editor
       - The editor provides:
         • Syntax highlighting
         • Line numbers
         • Auto-indentation
         • Error detection
       - Choose where to save your changes:
         • edited_schema.json (safe, keeps original intact)
         • Original schema file (permanent changes)
       - Click '💾 Save Changes' to save
       - Click '🔄 Reset to Original' to revert changes
       
    3. **Note on Saving:**
       - Saving to edited_schema.json keeps original file unchanged
       - Saving to original file modifies schema_by_summary_sections.json
       - Searches will use edited version when available
    """)

# Footer
st.sidebar.markdown("---")
