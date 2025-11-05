import openai
import os
import re
import sys
import json
from dotenv import load_dotenv
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from collections import defaultdict
import datetime
import dateutil.parser
from document_processor.pinecone_utils import PineconeSectionFetcher
import logging

# Configure logger
logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
except ImportError:
    genai = None
try:
    import anthropic
except ImportError:
    anthropic = None

# summary_engine.py
RUN_CONCISE_SUMMARIES = True
RUN_FULSOME_SUMMARIES = True

# =========================
# LLM Setup
# =========================


def load_api_keys():
    load_dotenv()
    api_keys = {
        'openai': os.getenv("OPENAI_API_KEY"),
        'google': os.getenv("GOOGLE_API_KEY"),
        'anthropic': os.getenv("ANTHROPIC_API_KEY")
    }
    return api_keys


# Initialize API keys
API_KEYS = load_api_keys()

# Configure OpenAI
if API_KEYS['openai']:
    openai.api_key = API_KEYS['openai']

# Configure Google Gemini
if API_KEYS['google'] and genai:
    genai.configure(api_key=API_KEYS['google'])

# Configure Anthropic
if API_KEYS['anthropic'] and anthropic:
    anthropic_client = anthropic.Anthropic(api_key=API_KEYS['anthropic'])
else:
    anthropic_client = None


def get_summary_mode_toggles():
    return RUN_CONCISE_SUMMARIES, RUN_FULSOME_SUMMARIES


def add_business_days(start_date, business_days):
    """
    Add business days to a given date, excluding weekends.

    Args:
        start_date (datetime): The starting date
        business_days (int): Number of business days to add

    Returns:
        datetime: The resulting date
    """
    current_date = start_date
    days_added = 0

    while days_added < business_days:
        current_date += datetime.timedelta(days=1)
        # Check if it's a weekday (Monday=0, Sunday=6)
        if current_date.weekday() < 5:  # Monday to Friday
            days_added += 1

    return current_date


def call_llm(prompt_text, model="gpt-4", temperature=0, provider="openai"):
    """
    Call LLM with specified provider and model

    Args:
        prompt_text (str): The prompt to send to the model
        model (str): The model name to use
        temperature (float): Temperature for response generation
        provider (str): The provider to use ('openai', 'google', 'anthropic')

    Returns:
        str: The model's response
    """
    system_message = "You are a legal summarization assistant."

    if provider.lower() == "openai":
        if not API_KEYS['openai']:
            raise ValueError("OpenAI API key not found")

        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt_text}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

    elif provider.lower() == "google":
        if not API_KEYS['google'] or not genai:
            raise ValueError(
                "Google API key not found or google.generativeai not installed")

        # Configure the model
        model_instance = genai.GenerativeModel(model)

        # Create the prompt with system message
        full_prompt = f"{system_message}\n\n{prompt_text}"

        response = model_instance.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=2048,
            )
        )
        return response.text.strip()

    elif provider.lower() == "anthropic":
        if not API_KEYS['anthropic'] or not anthropic_client or not anthropic:
            raise ValueError(
                "Anthropic API key not found or anthropic library not installed")

        response = anthropic_client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=temperature,
            system=system_message,
            messages=[
                {"role": "user", "content": prompt_text}
            ]
        )
        return response.content[0].text.strip()

    else:
        raise ValueError(
            f"Unsupported provider: {provider}. Supported providers are: openai, google, anthropic")

# =========================
# Utility to Traverse Nested Data
# =========================


def get_nested_value(data, path):
    keys = path.split(".")
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data

# =========================
# Utility to Extract Short Reference
# =========================


def shorten_reference(ref_string):
    ref_string = ref_string.replace("ARTICLE Article", "Article")
    match = re.search(
        r"(Section\\s\\d+(\\.\\d+)?([a-z])?)", ref_string, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ref_string.strip()


def extract_short_reference(references, data, fallback_to_section=True):
    short_refs = []
    for ref_path in references:
        val = get_nested_value(data, ref_path)
        if val is None:
            continue
        if ref_path.endswith("short_reference"):
            short_refs.append(val)
        elif fallback_to_section and isinstance(val, str):
            short_refs.append(shorten_reference(val))
        else:
            short_refs.append(val)
    return short_refs

# =========================
# Evaluate One Condition or Group
# =========================


def evaluate_condition_branch(condition, data):
    if "conditions" in condition:
        run_if = condition.get("run_if", "all")
        subresults = [evaluate_condition_branch(
            sub, data) for sub in condition["conditions"]]
        should_run = all("triggered" in res for res in subresults) if run_if == "all" else any(
            "triggered" in res for res in subresults)

        if should_run:
            output = {"triggered": True}
            for res in subresults:
                if "add_to_prompt" in res:
                    output.setdefault("add_to_prompt", {}).update(
                        res["add_to_prompt"])
                if "add_references" in res:
                    output.setdefault("add_references", []).extend(
                        res["add_references"])
            return output
        return {}

    value = get_nested_value(data, condition["if"])
    result = None
    output = {}

    if condition["type"] == "boolean":
        result = bool(value)

    elif condition["type"] == "enum":
        if isinstance(value, list):
            value_normalized = ", ".join(str(v) for v in value).strip()
        else:
            value_normalized = (value or "").strip()
        enum_cases = condition.get("enum_cases", {})
        branch = enum_cases.get(value_normalized, condition.get("default", {}))
        if branch:
            output["triggered"] = True

        if "text_output" in branch:
            output["text_output"] = branch["text_output"]

        if "add_to_prompt" in branch:
            resolved_prompt_fields = resolve_prompt_fields(
                branch["add_to_prompt"], data)
            output.setdefault("add_to_prompt", {}).update(
                resolved_prompt_fields)

        if "add_references" in branch:
            output.setdefault("add_references", []).extend(
                branch["add_references"])

        return output

    elif condition["type"] == "number":
        comparator = condition.get("comparator", "==")
        compare_value = condition.get("compare_to", condition.get("value"))
        if value is None or compare_value is None:
            result = False
        elif comparator == "==":
            result = value == compare_value
        elif comparator == ">":
            result = value > compare_value
        elif comparator == "<":
            result = value < compare_value
        elif comparator == ">=":
            result = value >= compare_value
        elif comparator == "<=":
            result = value <= compare_value
        else:
            result = False

    elif condition["type"] == "non_empty":
        result = value is not None and value != ""

    if result is True and "true" in condition:
        branch = condition["true"]
    elif result is False and "false" in condition:
        branch = condition["false"]
    elif "default" in condition:
        branch = condition["default"]
    else:
        branch = {}

    if branch:
        output["triggered"] = True
    if "text_output" in branch:
        output["text_output"] = branch["text_output"]
    if "add_to_prompt" in branch:
        resolved_prompt_fields = resolve_prompt_fields(
            branch["add_to_prompt"], data)
        output.setdefault("add_to_prompt", {}).update(resolved_prompt_fields)
    if "add_references" in branch:
        output.setdefault("add_references", []).extend(
            branch["add_references"])

    return output


def resolve_prompt_fields(prompt_dict, data):
    resolved = {}
    for k, v in prompt_dict.items():
        if isinstance(v, str) and v.startswith("{{") and v.endswith("}}"):
            key_path = v[2:-2].strip()  # Strip the double braces
            val = get_nested_value(data, key_path)

            if val and isinstance(val, str):
                match = re.search(
                    r"(\d+)\s+Business Days after.*(date of the Agreement|Signing Date)", val, re.IGNORECASE)
                if match:
                    num_days = int(match.group(1))
                    base_str = get_nested_value(
                        data, "timeline.agreement_signing_date.agreement_signing_date.clause_text")
                    try:
                        base_date = dateutil.parser.parse(base_str)
                        computed = add_business_days(base_date, num_days)
                        resolved[k] = computed.strftime("%B %d, %Y")
                    except Exception:
                        resolved[k] = val
                else:
                    resolved[k] = val
            else:
                resolved[k] = val if val is not None else ""
        else:
            # Treat as literal value
            resolved[k] = v
    return resolved


def normalize_to_string_list(value):
    if isinstance(value, str):
        return [value]
    elif isinstance(value, dict):
        return [str(value)]
    elif isinstance(value, list):
        return [str(item) if not isinstance(item, str) else item for item in value]
    else:
        return [str(value)]  # fallback for other types

# =========================
# Process a Single Clause Config
# =========================


def _resolve_reference_strings(reference_paths, schema_data):
    refs = []
    for path in reference_paths or []:
        val = get_nested_value(schema_data, path)
        if val:
            refs.append(val if isinstance(val, str) else str(val))
    return refs


def _maybe_fetch_pinecone_context(reference_paths, schema_data, deal_id=None):

    raw_refs = _resolve_reference_strings(reference_paths, schema_data)
    if not raw_refs:
        return {"context_text": "", "sections": []}

    fetcher = PineconeSectionFetcher()
    # Use provided deal_id, or fall back to environment variable
    deal_id_to_use = deal_id
    addl = {"deal_id": {"$eq": deal_id_to_use}} if deal_id_to_use else None
    return fetcher.get_context_for_references(raw_refs, top_k=8, additional_filter=addl)


def extract_and_match_definitions(pinecone_context_text, definitions_dict):
    """
    Extract capitalized terms (PascalCase, multi-word phrases) from text
    and match them against the definitions dictionary.

    Args:
        pinecone_context_text: List of text chunks or single text string
        definitions_dict: Dictionary of {term: definition}

    Returns:
        Dict of {matched_term: definition}
    """
    # Convert to string if it's a list
    if isinstance(pinecone_context_text, list):
        text = " ".join(pinecone_context_text)
    else:
        text = pinecone_context_text or ""

    matched_definitions = {}

    # Pattern 1: Multi-word acronyms (e.g., "PCI DSS")
    # Matches 2+ all-caps words with spaces
    acronym_pattern = r'\b([A-Z]{2,}(?:\s+[A-Z]{2,})+)\b'

    # Pattern 2: Mixed acronym + word patterns (e.g., "HSR Act", "NYSE Listed")
    # Matches acronym followed by capitalized word(s) or vice versa
    mixed_pattern = r'\b([A-Z]{2,}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z][a-z]+\s+[A-Z]{2,})\b'

    # Pattern 3: Number + word patterns (e.g., "102 Trustee", "409A Plan", "401K")
    # Matches numbers followed by optional capital letter and/or capitalized word(s)
    number_word_pattern = r'\b(\d+[A-Z]?(?:\s+[A-Z][a-z]+)*)\b'

    # Pattern 4: Section with numbers and optional words (e.g., "Section 102", "Section 102 Award")
    # Matches "Section" followed by numbers and optional capitalized words
    section_pattern = r'\b(Section\s+\d+(?:\s+[A-Z][a-z]*)*)\b'

    # Pattern 5: Multi-word capitalized phrases (e.g., "Parent Credit Facilities")
    # Matches 2+ consecutive capitalized words (mixed case only)
    multi_word_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'

    # Pattern 6: Single capitalized words or acronyms (e.g., "Order", "NYSE")
    # Matches single capital word or all-caps acronyms
    single_word_pattern = r'\b([A-Z]{2,}|[A-Z][a-z]+)\b'

    # Extract matches in priority order (most specific first)

    # 1. Multi-word acronyms (e.g., "PCI DSS")
    acronym_matches = re.findall(acronym_pattern, text)
    for term in acronym_matches:
        if term in definitions_dict:
            matched_definitions[term] = definitions_dict[term]

    # 2. Mixed patterns (e.g., "HSR Act", "NYSE Listed")
    mixed_matches = re.findall(mixed_pattern, text)
    for term in mixed_matches:
        if term in definitions_dict:
            matched_definitions[term] = definitions_dict[term]

    # 3. Number + word patterns (e.g., "102 Trustee", "409A Plan")
    number_word_matches = re.findall(number_word_pattern, text)
    for term in number_word_matches:
        # Filter out pure numbers or numbers with single letter (those are usually not definitions)
        # Only keep if it has at least one space + capitalized word, or has meaning in definitions
        if term in definitions_dict:
            matched_definitions[term] = definitions_dict[term]

    # 4. Section patterns (e.g., "Section 102", "Section 102 Award")
    section_matches = re.findall(section_pattern, text)
    for term in section_matches:
        if term in definitions_dict:
            matched_definitions[term] = definitions_dict[term]

    # 5. Multi-word capitalized phrases (e.g., "Parent Credit Facilities")
    multi_word_matches = re.findall(multi_word_pattern, text)
    for term in multi_word_matches:
        if term in definitions_dict:
            matched_definitions[term] = definitions_dict[term]

    # 6. Single capitalized words/acronyms (e.g., "Order", "NYSE")
    single_word_matches = re.findall(single_word_pattern, text)
    for term in single_word_matches:
        # Skip if already matched as part of multi-word phrase
        # Skip common words that are unlikely to be definitions
        skip_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'In', 'On', 'At',
                      'To', 'For', 'With', 'By', 'From', 'As', 'Of', 'And', 'Or', 'But',
                      'If', 'Then', 'When', 'Where', 'Who', 'What', 'Which', 'How', 'All',
                      'Each', 'Every', 'Any', 'Some', 'No', 'Not', 'Only', 'Also', 'Such',
                      'Other', 'Another', 'Both', 'Either', 'Neither', 'More', 'Most', 'Less',
                      'Least', 'Few', 'Many', 'Much', 'Several', 'Upon', 'Under', 'Over',
                      'Before', 'After', 'During', 'Between', 'Among', 'Through', 'Without',
                      'Within', 'About', 'Regarding', 'Concerning', 'Respect', 'Including',
                      'Except', 'Clause', 'Agreement', 'Party', 'Parties'}

        if term not in skip_words and term in definitions_dict:
            matched_definitions[term] = definitions_dict[term]

    # print(
    #     f"Extracted {len(matched_definitions)} matched definitions from context")
    # if matched_definitions:
    #     print(f"Sample matched terms: {list(matched_definitions.keys())[:10]}")

    return matched_definitions


def process_clause_config(clause_config, schema_data, provider="openai", model="gpt-4", temperature=0, deal_id=None, definitions_array=None, preamble_data=None):

    # Handle case where definitions_array might be None or empty
    if definitions_array:
        definitions_dict = {definition["label"]: definition["means"]
                            for definition in definitions_array}
    else:
        definitions_dict = {}
    # print(f"Total definitions available: {len(definitions_dict)}")

    # Extract preamble text if available
    preamble_text = ""
    if preamble_data and preamble_data.get('found'):
        preamble_text = preamble_data.get('preamble_text', '')
        # print(f"Preamble available: {len(preamble_text)} characters")

    prompt_fields = {}
    references = []
    final_text_output = None

    # Get model configuration if provided
    model_config = clause_config.get('model_config')

    for cond in clause_config.get("conditions", []):
        result = evaluate_condition_branch(cond, schema_data)
        if "add_to_prompt" in result:
            for k, v in result["add_to_prompt"].items():
                if k not in prompt_fields:
                    prompt_fields[k] = v
                else:
                    if isinstance(prompt_fields[k], list):
                        prompt_fields[k].append(v)
                    else:
                        prompt_fields[k] = [prompt_fields[k], v]
        if "add_references" in result:
            references.extend(result["add_references"])
        if "text_output" in result:
            final_text_output = result["text_output"]
    # Handle list-based prompt fields
    for k, v in prompt_fields.items():
        if isinstance(v, list):
            join_type = clause_config.get("join_type", "bullets")
            if join_type == "bullets":
                print(f"Processing field: {k}")
                print(f"Value (v): {v}")
                print(
                    f"Type of first item: {type(v[0]) if isinstance(v, list) and v else 'N/A'}")
                prompt_fields[k] = "\n- " + \
                    "\n- ".join(normalize_to_string_list(v))
            elif join_type == "sentences":
                prompt_fields[k] = " ".join(normalize_to_string_list(v))
            else:
                prompt_fields[k] = "\n".join(normalize_to_string_list(v))
    references.extend(clause_config.get("reference_fields", []))
    references = list(set(references))
    if clause_config.get("use_short_reference", True):
        short_refs = extract_short_reference(
            references, schema_data, fallback_to_section=True)
    else:
        short_refs = [
            get_nested_value(schema_data, path)
            for path in references
            if get_nested_value(schema_data, path)
        ]

     # === NEW: fetch Pinecone context using only the reference sections ===
    pinecone_ctx = _maybe_fetch_pinecone_context(
        references, schema_data, deal_id=deal_id)
    # print(f"pinecone_ctx: {pinecone_ctx}")

    # print(f"pinecone_ctx: {pinecone_ctx}")
    pinecone_context_text = pinecone_ctx.get("context_text", "")
    # print(f"pinecone_context_text: {pinecone_context_text}")

    # Extract and match definitions from the Pinecone context
    matched_definitions = extract_and_match_definitions(
        pinecone_context_text, definitions_dict)
    # print(f"Matched definitions: {matched_definitions}")

    pinecone_sections = pinecone_ctx.get("sections", [])

    # Build a standardized context preamble if we have any referenced chunks
    context_preamble = ""
    if pinecone_context_text:
        # Add preamble section if available
        preamble_section = ""
        if preamble_text:

            preamble_section += f"{preamble_text}\n"

        # Build excerpts with definitions per section
        excerpts_content = ""

        # pinecone_context_text is a list of strings formatted as "Section : Text"
        if isinstance(pinecone_context_text, list):
            for idx, chunk in enumerate(pinecone_context_text, 1):
                # Parse the chunk to extract section and content
                if " : " in chunk:
                    section_name, content = chunk.split(" : ", 1)
                else:
                    section_name = f"Section {idx}"
                    content = chunk

                # Add excerpt header
                excerpts_content += f"\nExcerpt {idx}\n"
                excerpts_content += f"{section_name}\n\n"
                excerpts_content += f"{content}\n"

                # Extract and match definitions specific to this chunk
                chunk_definitions = extract_and_match_definitions(
                    content, definitions_dict)

                # Add definitions for this section if any were found
                if chunk_definitions:
                    excerpts_content += f"\nRelevant Definitions for the above section:\n"
                    for term, definition in chunk_definitions.items():
                        excerpts_content += f"• {term}: {definition}\n"

                excerpts_content += "\n"  # Add spacing between excerpts
        else:
            # Fallback if it's a single string
            excerpts_content = f"\n{pinecone_context_text}\n"

            # Add all matched definitions at the end
            if matched_definitions:
                excerpts_content += "\n=== Relevant Definitions ===\n"
                for term, definition in matched_definitions.items():
                    excerpts_content += f"• {term}: {definition}\n"
                excerpts_content += "=== End Definitions ===\n"

        # Keep the preamble neutral and instructionally strong
        context_preamble = (
            "\n\nAbove schema value/clasuses are just directional and you should not rely on those answers and Use ONLY the following contract excerpts (by referenced sections) to answer.\n"
            "If information is not in the excerpts, say so rather than inferring.\n\n"
            "=== Actual Excerpts Start ===\n"
            f"{excerpts_content}"
            "=== Actual Excerpts End ===\n\n"
            "Below is the preamble section for the deal, which may be useful for resolving party aliases.\n"
            "=== Contract Preamble ===\n"
            f"{preamble_section}"
            "=== End Preamble ===\n"
            "----\n"
            "### Entity Resolution Requirement (Mandatory Step)\n"
            "Before you answer:\n"
            "1. Read the **Contract Preamble**.\n"
            "2. Identify and record all parties and their roles:\n"
            "- e.g., 'Parent' = James Hardie Industries plc\n"
            "'Merger Sub' = Juno Merger Sub Inc.\n"
            "'Company' = The AZEK Company Inc.\n"
            "3. Replace **every alias** in your reasoning and final answer with its specific name.\n"
            "Examples:\n"
            "- Use “The AZEK Company Inc.” instead of “Company.”\n"
            "- Use “James Hardie Industries plc” instead of “Parent.”\n"
            "4. This mapping must be performed **before writing the answer** and applied throughout.\n"

        )

    print(f"context_preamble: {context_preamble}")
    logger.info(f"context_preamble: {context_preamble}")

    # If prompt can be built
    if prompt_fields and "prompt_template" in clause_config:
        try:
            base_prompt = clause_config["prompt_template"].format(
                **prompt_fields) if prompt_fields else clause_config["prompt_template"]
            prompt = (base_prompt +
                      context_preamble) if context_preamble else base_prompt
            if clause_config.get("max_words"):
                prompt += f"\n\nLimit the response to {clause_config['max_words']} words."
            if clause_config.get("format_style"):
                prompt += f"\n\nFormat the response in a {clause_config['format_style']} style."
        except KeyError as e:
            if "fallback_prompt" in clause_config:
                prompt = clause_config["fallback_prompt"]
            else:
                prompt = f"[Missing field {str(e)} for prompt generation]"
        llm_result = call_llm(prompt, model=model,
                              temperature=temperature, provider=provider)
        return {
            "output": llm_result,
            "references": short_refs,
            "used_prompt": prompt if clause_config.get("view_prompt", False) else None,
            "summary_type": clause_config.get("summary_type"),
            "format_style": clause_config.get("format_style"),
            "summary_display_section": clause_config.get("summary_display_section"),
            # "summary_display_sub_section" : clause_config.get("summary_display_sub_section"),
            "summary_rank": clause_config.get("summary_rank"),
            "max_words": clause_config.get("max_words"),
            "matched_definitions": matched_definitions
        }
    # If no prompt was built, use fallback text_output
    if final_text_output:
        return {
            "output": final_text_output,
            "references": short_refs,
            "used_prompt": None,
            "summary_type": clause_config.get("summary_type"),
            "format_style": clause_config.get("format_style"),
            "summary_display_section": clause_config.get("summary_display_section"),
            # "summary_display_sub_section" : clause_config.get("summary_display_sub_section"),
            "summary_rank": clause_config.get("summary_rank"),
            "max_words": clause_config.get("max_words"),
            "matched_definitions": matched_definitions
        }
    return {
        "output": "No output generated.",
        "references": short_refs,
        "used_prompt": None,
        "summary_type": clause_config.get("summary_type"),
        "format_style": clause_config.get("format_style"),
        "summary_display_section": clause_config.get("summary_display_section"),
        # "summary_display_sub_section" : clause_config.get("summary_display_sub_section"),
        "summary_rank": clause_config.get("summary_rank"),
        "max_words": clause_config.get("max_words"),
        "matched_definitions": matched_definitions
    }


# =========================
# DOCX Writer
# =========================
SECTION_ORDER = ["Deal Structure", "Termination",
                 "Conditions", "Non Solicitation", "Regulatory", "General"]


def add_tab_stop(paragraph, position_inches):
    pPr = paragraph._element.get_or_add_pPr()
    tabs = pPr.find(qn('w:tabs'))
    if tabs is None:
        tabs = OxmlElement('w:tabs')
        pPr.append(tabs)

    # Avoid adding duplicate tab stops
    position_twips = str(int(position_inches * 1440))
    for existing_tab in tabs.findall(qn('w:tab')):
        if existing_tab.get(qn('w:pos')) == position_twips:
            return  # Already exists

    tab = OxmlElement('w:tab')
    tab.set(qn('w:val'), 'left')
    tab.set(qn('w:pos'), position_twips)
    tabs.append(tab)


def write_docx_summary(summaries, output_path, RUN_CONCISE_SUMMARIES, RUN_FULSOME_SUMMARIES):

    doc = Document()

    # Set default font
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Aptos"
    font.size = Pt(10.5)

    # Set margins to 1" all around
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Title
    title_para = doc.add_paragraph()
    title_run = title_para.add_run("Merger Agreement Clause Summaries")
    title_run.font.size = Pt(12)
    title_run.font.bold = True
    title_run.font.name = "Aptos"
    title_run.font.color.rgb = RGBColor(0, 0, 0)
    doc.add_paragraph("")

    # Concise / Fulsome grouping and content writing
    enabled_summary_types = []
    if RUN_CONCISE_SUMMARIES:
        enabled_summary_types.append("Concise")
    if RUN_FULSOME_SUMMARIES:
        enabled_summary_types.append("Fulsome")

    for s_type in enabled_summary_types:
        # Header
        p = doc.add_paragraph()
        run = p.add_run(f"{s_type} Summary")
        run.underline = True
        run.bold = True
        run.font.size = Pt(11)
        run.font.name = "Aptos"
        run.font.color.rgb = RGBColor(0, 0, 0)
        p.paragraph_format.space_after = Pt(4)

        already_print = []
        for s in [summary for summary in summaries if summary.get("summary_type") == s_type]:
            if s.get("summary_display_section") not in already_print:
                doc.add_heading(s.get("summary_display_section"), level=2)
                already_print.append(s.get("summary_display_section"))

            bullet_para = doc.add_paragraph()
            bullet_para.paragraph_format.left_indent = Inches(0.25)
            bullet_para.paragraph_format.first_line_indent = -Inches(0.25)
            bullet_para.paragraph_format.line_spacing = 1.16
            bullet_para.paragraph_format.space_before = Pt(1)
            bullet_para.paragraph_format.space_after = Pt(0)
            add_tab_stop(bullet_para, 0.25)

            bullet_run = bullet_para.add_run("+\t")
            bullet_run.font.name = "Aptos"
            bullet_run.font.size = Pt(10.5)

            bad_chars = '"\'“”--–—'  # quote marks + hyphen + en dash + em dash
            clean_output = s["output"].translate(
                str.maketrans('', '', bad_chars))
            # includes *
            bullet_class = r'[\u2022\u2023\u25E6\u25AA\u25CF\u25CB\u2043\u00B7\*\-]'
            clean_output = re.sub(
                rf'(?m)^\s*{bullet_class}\s*', '', s["output"])
            text_run = bullet_para.add_run(clean_output)
            text_run.font.name = "Aptos"
            text_run.font.size = Pt(10.5)

            if s.get("references") and s["references"][0] != '':
                ref_para = doc.add_paragraph()
                ref_para.paragraph_format.left_indent = Inches(1.0)
                ref_para.paragraph_format.first_line_indent = -Inches(0.25)
                ref_para.paragraph_format.line_spacing = 1
                add_tab_stop(ref_para, 1.0)

                ref_bullet = ref_para.add_run("○\t")
                ref_bullet.font.name = "Aptos"
                ref_bullet.font.size = Pt(8)
                ref_bullet.font.color.rgb = RGBColor(0, 0, 0)

                unique_refs = list(dict.fromkeys(s["references"]))
                ref_text = ref_para.add_run(
                    "References: " + "; ".join(unique_refs))
                ref_text.font.name = "Aptos"
                ref_text.font.size = Pt(10)

    doc.save(output_path)
    print(f"\n✅ DOCX summary written to: {output_path}")


# =========================
# Main Test Block
# =========================
if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))
    from clause_configs.ordinary_course_config import ORDINARY_COURSE_CLAUSES

    CLAUSE_CONFIG = ORDINARY_COURSE_CLAUSES

    with open("schema.json") as f:
        EXAMPLE_SCHEMA_DATA = json.load(f)

    print("Loaded clause configs:", list(CLAUSE_CONFIG.keys()))
    summary_outputs = []

    print(f"📝 Writing {len(summary_outputs)} summaries to DOCX")

    for clause_name, clause_config in CLAUSE_CONFIG.items():
        # Assert OFF clauses were filtered upstream (or flag if not)
        assert clause_config.get(
            "summary_type") != "OFF", f"OFF clause was not skipped: {clause_name}"

        print(f"→ Evaluating: {clause_name}")
        result = process_clause_config(
            clause_config, EXAMPLE_SCHEMA_DATA, provider="openai", model="gpt-4", temperature=0)
        print(f"→ Output preview: {result['output'][:100]}")
        if result["output"] and result["output"] != "No output generated.":
            filtered_result = result.copy()
            if not clause_config.get("view_prompt", False):
                filtered_result.pop("used_prompt", None)

            summary_outputs.append({
                "clause_name": clause_name,
                **filtered_result
            })

    write_docx_summary(summary_outputs)
