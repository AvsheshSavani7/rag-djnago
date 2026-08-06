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
    import anthropic
except ImportError:
    anthropic = None

# summary_engine.py
RUN_CONCISE_SUMMARIES = True
RUN_FULSOME_SUMMARIES = False  # True

# =========================
# "No answer" sentinels
# =========================
# Values the extractor writes to mean "this field has no real answer".
# Compared case-insensitively, with surrounding whitespace and a trailing
# period stripped — so "Not addressed." and "not addressed" both match.
NOT_FOUND_SENTINELS = {
    "not found", "na", "n/a", "not applicable", "not specified",
    "unspecified", "not addressed", "silent", "not inferred",
    "no mention", "none", "not_specified", "",
}


def _canon(v):
    """Normalize a value for sentinel testing only — never for output.

    Only None collapses to "". Other falsy values (False, 0, []) are real
    data and must stringify normally, or `_has_content` would treat a
    legitimate boolean False as a missing field.
    """
    if v is None:
        return ""
    return str(v).strip().rstrip(".").lower()


def is_not_found(v):
    """True if this value means 'no real answer'."""
    return _canon(v) in NOT_FOUND_SENTINELS


def _has_content(v):
    """True if a resolved prompt field carries usable text.

    Value-based, never path-based: `add_to_prompt` reads `.clause_text` in
    most configs and `.answer` in the rest, and either can be empty *or* the
    literal string "Not found". Both must count as no content.
    """
    if isinstance(v, list):
        return any(_has_content(x) for x in v)
    return bool(str(v).strip()) and not is_not_found(v)


# =========================
# Model registry + pricing
# =========================
# Prices are USD per 1,000,000 tokens (input, output).
# supports_temperature=False: the model accepts only its default temperature
# (1) and 400s on any explicit value, including 0 — so we omit the parameter.
MODEL_REGISTRY = {
    "gpt-5.6-sol":   {"id": "gpt-5.6-sol",   "provider": "openai",    "price_in": 5.00, "price_out": 30.00, "supports_temperature": False},
    "gpt-5.6-terra": {"id": "gpt-5.6-terra", "provider": "openai",    "price_in": 2.00, "price_out": 12.00, "supports_temperature": False},
    "gpt-5.6-luna":  {"id": "gpt-5.6-luna",  "provider": "openai",    "price_in": 0.20, "price_out": 1.20,  "supports_temperature": False},
    "gpt-5.2":       {"id": "gpt-5.2-2025-12-11", "provider": "openai",    "price_in": 1.25, "price_out": 10.00},
    "opus":          {"id": "claude-opus-4-8",    "provider": "anthropic", "price_in": 5.00, "price_out": 25.00},
    "sonnet":        {"id": "claude-sonnet-5",    "provider": "anthropic", "price_in": 3.00, "price_out": 15.00},
    "haiku":         {"id": "claude-haiku-4-5",   "provider": "anthropic", "price_in": 1.00, "price_out": 5.00},
}

# Which model to use, in precedence order:
#   1. per-call model_override (API/CLI forced model)      -> highest
#   2. clause config's "model" key
#   3. SUMMARY_MODEL env var
#   4. DEFAULT_MODEL_KEY below
DEFAULT_MODEL_KEY = "gpt-5.6-terra"
_ACTIVE_MODEL_KEY = None


def set_active_model(key):
    """Force the model used for every clause (CLI path only).

    Server/API paths must NOT use this global — pass model_override through
    process_clause_config instead, so concurrent deals cannot leak a model
    into one another. Validates eagerly so a bad name fails before any call.
    """
    global _ACTIVE_MODEL_KEY
    _resolve_model(key)
    _ACTIVE_MODEL_KEY = key


def _resolve_model(key=None):
    key = key or _ACTIVE_MODEL_KEY or os.getenv(
        "SUMMARY_MODEL", DEFAULT_MODEL_KEY)
    if key in MODEL_REGISTRY:
        return key, MODEL_REGISTRY[key]
    for k, spec in MODEL_REGISTRY.items():  # allow passing a raw model id
        if spec["id"] == key:
            return k, spec
    raise ValueError(
        f"Unknown model '{key}'. Choose from: {', '.join(MODEL_REGISTRY)}")


def model_for_clause(clause_config):
    """Model key this clause should run on, honouring a CLI override."""
    return _ACTIVE_MODEL_KEY or (clause_config or {}).get("model")


def summary_using_label(model_override=None):
    """Human label for the `summary_using` DB field.

    A forced model (API/CLI) is recorded by its registry key; otherwise the
    run is clause-config / default driven and no single model applies, so we
    record the literal "from clause config".
    """
    if model_override:
        return _resolve_model(model_override)[0]
    return "from clause config"


def validate_clause_models(clause_config_map):
    """Resolve every configured model before any paid call runs.

    A typo otherwise surfaces mid-run, after dozens of billed requests.
    """
    bad = []
    for name, cfg in clause_config_map.items():
        key = (cfg or {}).get("model")
        if key is None:
            continue
        try:
            _resolve_model(key)
        except ValueError as e:
            bad.append(f"  {name}: {e}")
    if bad:
        raise ValueError(
            "Invalid 'model' in clause config(s):\n" + "\n".join(bad))


def report_clause_models(clause_config_map, only_types=("Concise",)):
    """Log which model each clause will use, before spending anything."""
    if _ACTIVE_MODEL_KEY:
        logger.info(f"CLI override — ALL clauses forced to: {_ACTIVE_MODEL_KEY}")
    counts = defaultdict(int)
    unset = 0
    for name, cfg in clause_config_map.items():
        if (cfg or {}).get("summary_type") not in only_types:
            continue
        if not _ACTIVE_MODEL_KEY and (cfg or {}).get("model") is None:
            unset += 1
        counts[_resolve_model(model_for_clause(cfg))[0]] += 1
    logger.info("Model plan for this run:")
    for k, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        logger.info(f"   {n:3d} clause(s) -> {k}")
    if unset:
        logger.info(
            f"   ({unset} of these have no 'model' key — using the default"
            f" '{DEFAULT_MODEL_KEY}')")


# =========================
# Usage / cost tracking
# =========================
# Running token totals accumulated by call_llm, keyed by friendly model name.
USAGE_TOTALS = {}


def _record_usage(model_key, input_tokens, output_tokens):
    t = USAGE_TOTALS.setdefault(
        model_key, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
    t["calls"] += 1
    t["input_tokens"] += input_tokens or 0
    t["output_tokens"] += output_tokens or 0


def get_cost_summary():
    """Return (rows, grand_total_usd) where each row is a per-model breakdown."""
    rows = []
    grand_total = 0.0
    for key, t in USAGE_TOTALS.items():
        spec = MODEL_REGISTRY[key]
        cost = (t["input_tokens"] / 1_000_000) * spec["price_in"] \
            + (t["output_tokens"] / 1_000_000) * spec["price_out"]
        grand_total += cost
        rows.append({
            "model_key": key,
            "model_id": spec["id"],
            "calls": t["calls"],
            "input_tokens": t["input_tokens"],
            "output_tokens": t["output_tokens"],
            "cost_usd": cost,
        })
    return rows, grand_total


# =========================
# LLM Setup
# =========================


def load_api_keys():
    load_dotenv()
    api_keys = {
        'openai': os.getenv("OPENAI_API_KEY_SEC_FILING"),
        'anthropic': os.getenv("ANTHROPIC_API_KEY")
    }
    return api_keys


# Initialize API keys
API_KEYS = load_api_keys()

# Configure OpenAI
if API_KEYS['openai']:
    openai.api_key = API_KEYS['openai']

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


def call_llm(prompt_text, model=None, temperature=0):
    """Call the resolved LLM (OpenAI or Anthropic) for a single clause.

    The provider is derived from the registry entry, so callers never pass a
    provider. `model` may be a registry key, a raw model id, or None (in which
    case the precedence in _resolve_model decides).

    Args:
        prompt_text (str): The prompt to send to the model
        model (str|None): Registry key / raw id / None
        temperature (float): Used only for models that accept it

    Returns:
        str: The model's response text
    """
    model_key, spec = _resolve_model(model)
    provider = spec["provider"]
    model_id = spec["id"]
    system_message = "You are a legal summarization assistant."

    if provider == "openai":
        if not API_KEYS['openai']:
            raise ValueError("OpenAI API key not found")

        kwargs = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt_text},
            ],
        }
        # Some newer OpenAI models reject an explicit temperature and accept
        # only their default of 1 — omit the parameter entirely for those.
        if spec.get("supports_temperature", True):
            kwargs["temperature"] = temperature

        resp = openai.chat.completions.create(**kwargs)

        usage = getattr(resp, "usage", None)
        if usage:
            _record_usage(
                model_key,
                getattr(usage, "prompt_tokens", 0),
                getattr(usage, "completion_tokens", 0),
            )
        return resp.choices[0].message.content.strip()

    elif provider == "anthropic":
        if not API_KEYS['anthropic'] or not anthropic_client or not anthropic:
            raise ValueError(
                "Anthropic API key not found or anthropic library not installed")

        # NOTE: Opus/Sonnet reject `temperature` (400) — omit it entirely.
        resp = anthropic_client.messages.create(
            model=model_id,
            max_tokens=4096,
            system=system_message,
            messages=[{"role": "user", "content": prompt_text}],
        )

        usage = getattr(resp, "usage", None)
        if usage:
            input_tokens = (
                (getattr(usage, "input_tokens", 0) or 0)
                + (getattr(usage, "cache_read_input_tokens", 0) or 0)
                + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
            )
            _record_usage(
                model_key,
                input_tokens,
                getattr(usage, "output_tokens", 0),
            )
        # Models with extended thinking emit a ThinkingBlock *before* the
        # answer, so content[0] is not necessarily the text. Collect every
        # text block instead of assuming the first one.
        text_parts = [
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        ]
        if not text_parts:
            logger.warning(
                "No text block in %s response (blocks: %s) — returning empty",
                model_id,
                [getattr(b, "type", "?") for b in resp.content],
            )
        # Thinking tokens count against max_tokens, so a long reasoning pass
        # can truncate the answer mid-sentence — surface it rather than
        # silently writing a half-finished summary.
        if getattr(resp, "stop_reason", None) == "max_tokens":
            logger.warning(
                "%s hit max_tokens — summary may be truncated", model_id)
        return "".join(text_parts).strip()

    raise ValueError(f"Unknown provider '{provider}' for model '{model_id}'")


def add_table_spacing(doc, before_pt=6, after_pt=6):
    if before_pt:
        p_before = doc.add_paragraph()
        p_before.paragraph_format.space_after = Pt(before_pt)
        p_before.paragraph_format.space_before = Pt(0)

    if after_pt:
        p_after = doc.add_paragraph()
        p_after.paragraph_format.space_before = Pt(after_pt)
        p_after.paragraph_format.space_after = Pt(0)

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
        # Never print "Not found"/"NA" as though it were a section citation.
        if is_not_found(val):
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
        if isinstance(value, bool):
            result = value
        elif isinstance(value, str):
            v = value.strip().lower()
            if v in {"true", "1", "yes", "y"}:
                result = True
            elif v in {"false", "0", "no", "n", ""} or is_not_found(v):
                # "NA"/"Not found" means no answer was located, which must
                # not fire the true-branch.
                result = False
            else:
                # Extractor sometimes writes prose into a boolean field.
                # Degrade instead of raising: nothing wraps
                # process_clause_config, so an exception here kills the whole
                # run and discards every clause already summarised.
                logger.warning(
                    "Non-boolean value %r at %s — treating as False",
                    value,
                    condition.get("if"),
                )
                result = False
        else:
            result = bool(value)

    elif condition["type"] == "enum":
        if isinstance(value, list):
            value_normalized = ", ".join(str(v) for v in value).strip()
        elif isinstance(value, bool):
            value_normalized = "true" if value else "false"
        elif isinstance(value, (int, float)):
            value_normalized = str(value)
        else:
            # Convert value to string first to handle booleans, None, etc.
            value_normalized = str(value) if value is not None else ""
            value_normalized = value_normalized.strip()
        enum_cases = condition.get("enum_cases", {})

        # Tier 1: exact match — preserves all existing behaviour untouched.
        branch = enum_cases.get(value_normalized)

        # Tier 2: case-insensitive — catches "Not Found" vs "Not found".
        # Only string keys: a few configs use a literal None key, which
        # value_normalized (always a str) can never match anyway.
        if branch is None:
            ci = {k.lower(): v for k, v in enum_cases.items()
                  if isinstance(k, str)}
            branch = ci.get(value_normalized.lower())

        # Tier 3: sentinel collapse — "NA" uses whatever not-found branch
        # this config already declares. First match in declaration order.
        if branch is None and is_not_found(value_normalized):
            for k in enum_cases:
                if isinstance(k, str) and is_not_found(k):
                    branch = enum_cases[k]
                    break

        if branch is None:
            branch = condition.get("default", {})

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
        # "Is there something here?" must answer no for a string whose
        # literal meaning is "there is nothing here" ("NA", "Not found", ...).
        result = value is not None and not is_not_found(value)

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
        # A reference_section of "Not found"/"NA" is truthy but meaningless —
        # querying Pinecone for it returns whatever happens to be nearest.
        if val and not is_not_found(val):
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
    multi_word_pattern = r"\b([A-Z][a-z]+(?:[-\s][A-Z][a-z]+)+)\b"

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


def extract_section_references(text, current_sections=None):
    """

    Extract section references from text (e.g., "Section 7.3", "7.3", "§ 7.3").

    Excludes current sections to avoid duplicates.



    Args:

        text: String or list of strings to search for section references

        current_sections: List of current section IDs to exclude (e.g., ['7.1'])



    Returns:

        List of unique section references (normalized to section numbers like '7.3')

    """

    # Convert to string if it's a list

    if isinstance(text, list):

        text = " ".join(text)

    else:

        text = text or ""

    # Normalize current sections to just numbers (e.g., '7.1' from 'Section 7.1')

    current_section_nums = set()

    if current_sections:

        for sec in current_sections:

            # Extract just the number part

            match = re.search(r"(\d+(?:\.\d+)+|\d+)", str(sec))

            if match:

                current_section_nums.add(match.group(1))

    # Pattern to match section references:

    # - "Section 7.3", "Section 7.3.1", "section 7.3"

    # - "7.3", "7.3.1"

    # - "§ 7.3", "§7.3"

    # - "Sections 7.3 and 7.4"

    section_patterns = [
        r"(?:Section|section|Sections|sections)\s+(\d+(?:\.\d+)+|\d+)",  # "Section 7.3"
        r"§\s*(\d+(?:\.\d+)+|\d+)",  # "§ 7.3" or "§7.3"
        r"\b(\d+\.\d+(?:\.\d+)*)\b",  # "7.3" or "7.3.1" (standalone)
    ]

    found_sections = set()

    for pattern in section_patterns:

        matches = re.findall(pattern, text, re.IGNORECASE)

        for match in matches:

            # re.findall with a single capturing group returns a list of strings

            section_num = match if isinstance(match, str) else str(match)

            if section_num and section_num not in current_section_nums:

                found_sections.add(section_num)

    return list(found_sections)


def process_clause_config(clause_config, clause_name, schema_data, deal_id=None, definitions_array=None, preamble_data=None, model_override=None):

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

    # Extract section references from pinecone_context_text (excluding current sections)

    referenced_sections = extract_section_references(
        pinecone_context_text, current_sections=pinecone_sections
    )

    print(f"referenced_sections: {referenced_sections}")

    referenced_chunks = []

    if referenced_sections and pinecone_context_text:

        fetcher = PineconeSectionFetcher()

        addl = {"deal_id": {"$eq": deal_id}} if deal_id else None

        referenced_ctx = fetcher.get_context_for_references(
            referenced_sections, top_k=8, additional_filter=addl
        )

        referenced_chunks = referenced_ctx.get("context_chunks", [])

    # Build a standardized context preamble if we have any referenced chunks
    context_preamble = ""
    if pinecone_context_text:
        # Add preamble section if available
        preamble_section = ""
        if preamble_text:
            preamble_section = "\n=== Contract Preamble ===\n"
            preamble_section += f"{preamble_text}\n"
            preamble_section += "=== End Preamble ===\n\n"

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

        # Append referenced section chunks(from other sections mentioned in context)

        # This is done after processing the main context, regardless of list or string format

        if referenced_chunks:

            excerpts_content += "\n\n=== Additional Referenced Sections from Response ===\nPlease use the following additional sections to resolve any section references mentioned in your response, if possible.\n\n"

            # Calculate starting index based on how many excerpts we already have

            if isinstance(pinecone_context_text, list):

                start_idx = len(pinecone_context_text) + 1

            else:

                start_idx = 2  # If it was a string, we had 1 excerpt

            for idx, chunk in enumerate(referenced_chunks, start=start_idx):

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

                excerpts_content += "\n"  # Add spacing between excerpts

            excerpts_content += "=== End Additional Referenced Sections ===\n"

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
            "Before you answer, do the following SILENTLY / internally — do NOT write any of it in your response:\n"
            "1. Read the **Contract Preamble**.\n"
            "2. Work out each party and its role in your head (e.g., internally note that 'Parent' = James Hardie Industries plc, 'Merger Sub' = Juno Merger Sub Inc., 'Company' = The AZEK Company Inc.).\n"
            "3. In your final answer, replace **every alias** with its specific name.\n"
            "Examples:\n"
            "- Use “The AZEK Company Inc.” instead of “Company.”\n"
            "- Use “James Hardie Industries plc” instead of “Parent.”\n"
            "4. This mapping must be performed **before writing the answer** and applied throughout.\n"
            "\n"
            "### Output Rules (STRICT)\n"
            "- Your entire response must be ONLY the requested summary sentence(s).\n"
            "- Do NOT output the party-to-role mapping, an 'Entity mapping:' line, your reasoning, step numbers, headers, or any preamble/labels before or after the summary.\n"
            "- Do NOT prefix the answer with '*', '-', bullet markers, or notes about how you resolved the parties.\n"
        )

    print(f"context_preamble: {context_preamble}")
    logger.info(f"context_preamble: {context_preamble}")

    # If prompt can be built.
    # NOTE: test the *values*, not the dict. `prompt_fields` for a no-answer
    # clause is e.g. {"right_to_control_strategy": ""} — a non-empty dict
    # holding an empty string, which is truthy and used to slip through here,
    # producing an empty prompt and an LLM call that narrates the gap.
    meaningful = {k: v for k, v in prompt_fields.items() if _has_content(v)}
    if meaningful and "prompt_template" in clause_config:
        try:
            base_prompt = clause_config["prompt_template"].format(
                **prompt_fields) if prompt_fields else clause_config["prompt_template"]
            prompt = (base_prompt +
                      context_preamble) if context_preamble else base_prompt
            if clause_config.get("max_words"):
                prompt += f"\n\nLimit the response to {clause_config['max_words']} words."
            if clause_config.get("format_style"):
                prompt += f"\n\nFormat the response in a {clause_config['format_style']} style."
             # Appended DEAD LAST (after excerpts, preamble, and the generic entity-resolution
            # block) so a clause can override those with a final, highest-recency instruction.
            if clause_config.get("final_instruction"):
                prompt += f"\n\n{clause_config['final_instruction']}"
        except KeyError as e:
            if "fallback_prompt" in clause_config:
                prompt = clause_config["fallback_prompt"]
            else:
                prompt = f"[Missing field {str(e)} for prompt generation]"
        # Precedence: per-call override (API/CLI forced) -> clause "model"
        # key -> SUMMARY_MODEL env -> DEFAULT_MODEL_KEY (handled in call_llm).
        clause_model = model_override or model_for_clause(clause_config)
        llm_result = call_llm(prompt, model=clause_model)

        logger.info(f"llm_result: {llm_result[:100]}")
        logger.info(f"Clause Name Done: {clause_name}")
        return {
            "output": llm_result,
            "model": _resolve_model(clause_model)[0],
            "references": short_refs,
            "used_prompt": prompt if clause_config.get("view_prompt", False) else None,
            "summary_type": clause_config.get("summary_type"),
            "format_style": clause_config.get("format_style"),
            "summary_display_section": clause_config.get("summary_display_section"),
            # "summary_display_sub_section" : clause_config.get("summary_display_sub_section"),
            "summary_rank": clause_config.get("summary_rank"),
            "max_words": clause_config.get("max_words"),
            "clause_name": clause_name,
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
            "clause_name": clause_name,
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
        "clause_name": clause_name,
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


def write_docx_summary(summaries, output_path, RUN_CONCISE_SUMMARIES, RUN_FULSOME_SUMMARIES, From_Clause=False):

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

            if s.get("format_style") == "matrix_table":
                lines = [ln for ln in (
                    s.get("output") or "").splitlines() if ln.strip()]
                if len(lines) == 3:
                    header_line, row1_line, row2_line = lines

                    # Extract row labels
                    row_label1 = re.split(
                        r"\s+", row1_line.strip())[0] if row1_line.strip() else "Row 1"
                    row_label2 = re.split(
                        r"\s+", row2_line.strip())[0] if row2_line.strip() else "Row 2"
                    row_labels = ["", row_label1, row_label2]

                    # Split on 2+ spaces
                    header_cells = re.split(
                        r"\s{2,}", header_line.rstrip("\n"))
                    row1_cells = re.split(r"\s{2,}", row1_line.rstrip("\n"))
                    row2_cells = re.split(r"\s{2,}", row2_line.rstrip("\n"))

                    # Remove row labels from data rows
                    if row1_cells and row1_cells[0].strip() == row_label1:
                        row1_cells = row1_cells[1:]
                    if row2_cells and row2_cells[0].strip() == row_label2:
                        row2_cells = row2_cells[1:]

                    # --- KEY: shift Parent amounts if LLM left the first column empty ---
                    # Count leading spaces in the original row2_line (after the label)
                    after_label = row2_line
                    if row_label2 in row2_line:
                        idx = row2_line.index(row_label2) + len(row_label2)
                        after_label = row2_line[idx:]

                    leading_spaces = len(after_label) - \
                        len(after_label.lstrip(" "))

                    # Heuristic: if there is a gap wider than one typical space block,
                    # treat it as an intentionally empty first column
                    if leading_spaces >= 4:  # tweakable threshold
                        row2_cells = [""] + row2_cells

                    # Pad to equal length
                    max_data_cols = max(len(header_cells), len(
                        row1_cells), len(row2_cells))
                    header_cells += [""] * (max_data_cols - len(header_cells))
                    row1_cells += [""] * (max_data_cols - len(row1_cells))
                    row2_cells += [""] * (max_data_cols - len(row2_cells))

                    total_cols = max_data_cols + 1  # +1 for row label col

                    # Space before table
                    add_table_spacing(doc, before_pt=8)

                    table = doc.add_table(rows=3, cols=total_cols)
                    table.style = "Table Grid"
                    table.autofit = True

                    # Space after table
                    add_table_spacing(doc, after_pt=10)

                    for row_idx, row in enumerate(table.rows):
                        for cell_idx, cell in enumerate(row.cells):
                            for p in cell.paragraphs:
                                p.clear()

                            if cell_idx == 0:
                                cell.text = row_labels[row_idx]
                                for run in cell.paragraphs[0].runs:
                                    run.bold = True
                                cell.paragraphs[0].alignment = 0  # LEFT
                            else:
                                data_idx = cell_idx - 1
                                if row_idx == 0:
                                    cell.text = header_cells[data_idx]
                                elif row_idx == 1:
                                    cell.text = row1_cells[data_idx]
                                elif row_idx == 2:
                                    cell.text = row2_cells[data_idx]

                                if row_idx == 0 and cell.text:
                                    for run in cell.paragraphs[0].runs:
                                        run.bold = True
                                    cell.paragraphs[0].alignment = 1  # CENTER

                            for para in cell.paragraphs:
                                para.paragraph_format.space_before = Pt(0)
                                para.paragraph_format.space_after = Pt(0)

                    # Borders (unchanged)
                    table_part = table._tbl
                    tblPr = table_part.tblPr
                    tblBorders = OxmlElement("w:tblBorders")
                    for border_name in ["top", "left", "insideH", "insideV", "right", "bottom"]:
                        border = OxmlElement(f"w:{border_name}")
                        border.set(qn("w:val"), "single")
                        border.set(qn("w:sz"), "8")
                        border.set(qn("w:color"), "000000")
                        tblBorders.append(border)
                    tblPr.append(tblBorders)
                continue  # Skip bullets

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

            config_para = doc.add_paragraph()

            config_para.paragraph_format.left_indent = Inches(1.0)

            config_para.paragraph_format.first_line_indent = -Inches(0.25)

            config_para.paragraph_format.line_spacing = 1

            add_tab_stop(config_para, 1.0)

            if From_Clause:

                config_bullet = config_para.add_run("○\t")

                config_bullet.font.name = "Aptos"

                config_bullet.font.size = Pt(8)

                config_bullet.font.color.rgb = RGBColor(0, 0, 0)

                config_bullet = config_para.add_run(

                    "From : " + s.get("clause_name") + " - " + str(s.get("summary_rank")))

                config_bullet.font.name = "Aptos"

                config_bullet.font.size = Pt(10)

    doc.save(output_path)
    print(f"\n✅ DOCX summary written to: {output_path}")

    # for local testing - save both DOCX and JSON locally
    # local_docx_path = "summary_outputs.docx"
    # doc.save(local_docx_path)

    # with open("summary_outputs.json", "w", encoding="utf-8") as f:
    #     json.dump(summaries, f, indent=2, ensure_ascii=False)

    # print(f"\n✅ Local DOCX copy written to: {local_docx_path}")
    # print(f"\n✅ Summary outputs written to: summary_outputs.json")


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
            clause_config, clause_name, EXAMPLE_SCHEMA_DATA)
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
