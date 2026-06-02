from playwright.sync_api import sync_playwright
import requests as _requests_lib
import os
import re
import time
import logging
import json
import openai
import threading
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from anthropic import Anthropic

SEC_USER_AGENT = os.environ.get(
    "MNA_SEC_USER_AGENT",
    "Mozilla/5.0 (compatible; RAG_BE SEC scraper; contact: ashish.kachadiya@teqnodux.com)",
)

# Load environment variables from .env file
load_dotenv()


# Thread-local storage for worker-specific data
thread_local = threading.local()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------
# GLOBAL REGEX PATTERNS
# ---------------------------
metadata_patterns = [
    r"EX-\d+\.\d+\s+\d+\s+[^\s]+\.htm\s+EX-\d+\.\d+\s+Document",
    r"EXHIBIT\s+\d+\.\d+",
    r"Document\s*$",
    r"Filed by:.*",
    r"Date Filed:.*",
    r"SEC Accession No\..*"
]

priority_patterns = [
    r"\bLIST\s+OF\s+EXHIBITS\b",
    r"\bAPPENDICES\b",
    r"\bEXHIBITS\b",
    r"\bANNEX\b",
    r"\bANNEXES\b",
    r"\bAPPENDIX\b",
    r"\bEXHIBIT\b",
    r"\bSchedule\s+(?:[A-Z]|[IVXLCDM]|\d)\b"
]

SOFT_STOP_SCHEDULE_PATTERNS = [
    r"\bSchedules\b",
    r"\bSchedule\b",
]

agreement_pattern = r"AGREEMENT\s+AND\s+PLAN\s+OF\s+MERGER"
TOC_HEADER_PATTERN = r"TABLE\s+OF\s+CONTENTS|Contents"
TOC_COMBINED_PATTERN = re.compile(
    r"^ARTICLE\s+\d+|^Section\s+\d+\.\d+|^Table\s+of\s+Contents", re.IGNORECASE)

# START_PATTERN = r"(THIS\s+)?AGREEMENT\s+AND\s+PLAN\s+OF\s+MERGER"
# START_PATTERN =r"(?:THIS\s+AGREEMENT\s+AND\s+PLAN\s+OF\s+MERGER|THIS\s+MERGER\s+AGREEMENT)"
START_PATTERN = r"(?:THIS\s+)?(?:AGREEMENT\s+AND\s+PLAN\s+OF\s+MERGER|THIS\s+MERGER\s+AGREEMENT|This\s+Agreement\s+is)"

# END_PATTERN = r"agree\s+as\s+follows[:.]?"
END_PATTERN = r"agrees?\s*,?\s*as\s+follows[:.]?"
# END_PATTERN = r"agree\s*,?\s*as\s+follows\b"

# Load API keys from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY_SEC_FILING')
if not openai.api_key:
    raise ValueError(
        "OPENAI_API_KEY_SEC_FILING not found in environment variables. Please check your .env file.")

# Initialize Anthropic client
anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
if not os.getenv('ANTHROPIC_API_KEY'):
    raise ValueError(
        "ANTHROPIC_API_KEY not found in environment variables. Please check your .env file.")

# ---------------------------
# 17 NOISE PHRASES (TRUNCATE ONLY IN LAST SECTION)
# ---------------------------
TRUNCATE_PHRASES = [
    "Remainder of page intentionally left blank; signature pages follow.",
    "Signature Page Follows",
    "Signature page follows",
    "The remainder of this page has been intentionally left blank.",
    "The remainder of this page is intentionally left blank.",
    "Signature Pages Follow",
    "Remainder of Page Intentionally Blank; Signature Pages Follow",
    "REMAINDER OF THIS PAGE INTENTIONALLY LEFT BLANK",
    "The remainder of this page has been intentionally left blank; the next page is the signature page.",
    "REMAINDER OF PAGE INTENTIONALLY LEFT BLANK SIGNATURE PAGE FOLLOWS",
    "Remainder of page intentionally left blank",
    "IN WITNESS WHEREOF",
    "SIGNATURES TO FOLLOW ON THE NEXT PAGE",
]

# ---------------------------
# PREDEFINED KEYWORDS FOR DEFINITIONS
# ---------------------------
DEFINITION_KEYWORDS = [
    "definitions",
    "certain definitions",
    "Certain Terms Defined",
    "certain defined terms",
    "terms and definitions",
    "defined terms",
    "certain specified definitions",
    "(b) Certain Specified Definitions",
    "1.2 Certain Specified Definitions",
    "definitions & interpretations",
    "definitions and interpretations",
]
# REF_ONLY_DEF = re.compile(r'^\s*set\s+forth\s+in\s+(?:the\s+preamble(?:\s+hereto)?|Section\s+[A-Za-z0-9().\-]+|Article\s+[A-Za-z0-9().\-]+)\s*\.?\s*$', re.I)
REF_ONLY_DEF = re.compile(
    r'\bset\s+forth\s+in\s+'
    r'(?:'
    r'the\s+(?:preamble|recitals)(?:\s+hereto)?|'
    r'Section\s+[A-Za-z0-9().\-]+|'
    r'Article\s+[A-Za-z0-9().\-]+'
    r')',
    re.I
)
TERM_INDEX_INTRO_PATTERN = re.compile(
    r'(?:'
    # (b) Each of the following terms is defined in the Section set forth opposite such term...
    r'(?:\(\s*[a-z]\s*\)\s*)?Each of the following terms is defined in the (?:[Ss]ection|section|page) '
    r'set forth (?:opposite|after) such term'
    r'|'
    # The following capitalized terms / The following terms shall have / have the meanings / are defined...
    r'(?:\(\s*[a-z]\s*\)\s*)?(?:In addition,|The)\s+following(?: capitalized)? terms\b.*?'
    r'(?:'
    r'shall have the respective meanings'                 # <-- your explicit phrase
    r'|defined'
    r'|have\b.*?meanings?'
    r'|shall have\b.*?meanings?'
    r').*?(?:Section|sections?|page|Agreement)'
    r'|'
    # Terms Defined Elsewhere / Other Defined Terms / Index of Defined Terms / Certain Specified Definitions
    r'\bTerms Defined Elsewhere\b'
    r'|\bOther Defined Terms\b'
    r'|\bIndex of Defined Terms\b'
    r'|\bCertain Specified Definitions\b'
    r')',
    re.IGNORECASE
)
LAST_DEF_END_PATTERN = re.compile(
    r'(?:E\s*X\s*H\s*I\s*B\s*I\s*T|A\s*N\s*N\s*E\s*X|S\s*C\s*H\s*E\s*D\s*U\s*L\s*E)\s+[A-Z0-9]+\b',
    re.IGNORECASE,
)

OUTSIDE_DEF_HEADER = re.compile(
    r'(?im)(?:'
    r'^\s*(EXHIBIT|APPENDIX|ANNEX|SCHEDULE)\s+[A-Z0-9]+(?:\s*[–\-]\s*)?(?:\s+(DEFINITIONS?|DEFINED TERMS?|CERTAIN DEFINITIONS?))?\s*$'
    r'(?:\s*\r?\n\s*^\s*(DEFINITIONS?|DEFINED TERMS?|CERTAIN DEFINITIONS?)\s*$)?'
    r'|'
    r'^\s*(APPENDIX)\s+[A-Z0-9]+\s*$'
    r')'
)

SECTION_CERTAIN_DEFS_PATTERN = re.compile(
    r'(?im)^\s*(\d+(?:\.\d+)+)\s+Certain\s+Definitions\.?\b'
)

SECONDARY_DEFS_END_PATTERN = re.compile(
    r'(?m)^\s*(?:EXHIBIT|ANNEX)\s+[A-Z0-9]+\b(?!\s*-\s*\d+\b)',
    re.IGNORECASE | re.MULTILINE
)

CHARTER_STOP_PATTERN = re.compile(
    r'(?:'
    # 1) CONDITIONS TO THE OFFER
    r'^(?:C\s*O\s*N\s*D\s*I\s*T\s*I\s*O\s*N\s*S)\s+TO\s+THE\s+(?:O\s*F\s*F\s*E\s*R)'
    r'|'
    # 2) SURVIVING CORPORATION CERTIFICATE OF INCORPORATION
    r'^(?:S\s*U\s*R\s*V\s*I\s*V\s*I\s*N\s*G)\s+'
    r'(?:C\s*O\s*R\s*P\s*O\s*R\s*A\s*T\s*I\s*O\s*N)\s+'
    r'(?:C\s*E\s*R\s*T\s*I\s*F\s*I\s*C\s*A\s*T\s*E)\s+OF\s+'
    r'(?:I\s*N\s*C\s*O\s*R\s*P\s*O\s*R\s*A\s*T\s*I\s*O\s*N)'
    r'|'
    # 3) ARTICLES OF INCORPORATION (optionally "OF THE SURVIVING CORPORATION")
    r'^(?:A\s*R\s*T\s*I\s*C\s*L\s*E\s*S)\s+OF\s+'
    r'(?:I\s*N\s*C\s*O\s*R\s*P\s*O\s*R\s*A\s*T\s*I\s*O\s*N)'
    r'(?:\s+OF\s+THE\s+(?:S\s*U\s*R\s*V\s*I\s*V\s*I\s*N\s*G)\s+'
    r'(?:C\s*O\s*R\s*P\s*O\s*R\s*A\s*T\s*I\s*O\s*N))?'
    r'|'
    # 4) CERTIFICATE OF AMENDMENT TO AMENDED AND RESTATED CERTIFICATE OF INCORPORATION
    r'^(?:C\s*E\s*R\s*T\s*I\s*F\s*I\s*C\s*A\s*T\s*E)\s+OF\s+AMENDMENT\s+TO\s+AMENDED\s+AND\s+RESTATED\s+'
    r'(?:C\s*E\s*R\s*T\s*I\s*F\s*I\s*C\s*A\s*T\s*E)\s+OF\s+'
    r'(?:I\s*N\s*C\s*O\s*R\s*P\s*O\s*R\s*A\s*T\s*I\s*O\s*N)'
    r'|'
    # 5) SECOND AMENDED AND RESTATED CERTIFICATE OF INCORPORATION
    r'^(?:S\s*E\s*C\s*O\s*N\s*D)\s+AMENDED\s+AND\s+RESTATED\s+'
    r'(?:C\s*E\s*R\s*T\s*I\s*F\s*I\s*C\s*A\s*T\s*E)\s+OF\s+'
    r'(?:I\s*N\s*C\s*O\s*R\s*P\s*O\s*R\s*A\s*T\s*I\s*O\s*N)'
    r'|'
    # 6) CONSTRUCTION
    r'^(?:C\s*O\s*N\s*S\s*T\s*R\s*U\s*C\s*T\s*I\s*O\s*N)'
    r')',
    re.IGNORECASE,
)

# ---------------------------
# URLs (FULL LIST)
# ---------------------------
urls = [
    "https://www.sec.gov/Archives/edgar/data/1489096/000110465926018826/tm267070d1_ex2-1.htm"
]

# ---------------------------
# TOC CLEANUP + END POSITION
# ---------------------------


def clean_metadata_and_toc(text, url):
    for pattern in metadata_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    toc_start_match = re.search(
        TOC_HEADER_PATTERN, text[0:5000], re.IGNORECASE)
    if not toc_start_match:
        logger.warning(
            f"No TOC found in {url} , trying default start at pos 150")

        toc_start = 150
    else:
        toc_start = toc_start_match.start()
    stop_text = text[toc_start:]
    stop_pos_in_slice = len(stop_text)

    # --- Find soft stop patterns first --- temperary disable
    # m_start = re.search(START_PATTERN,  text[toc_start:], re.IGNORECASE)
    # demotext =  text[toc_start:toc_start+m_start.start()]
    # print("demotext=",demotext)

    for pattern in priority_patterns:
        match = re.search(pattern, stop_text, re.IGNORECASE)
        if match and match.start() < stop_pos_in_slice:
            stop_pos_in_slice = match.start()

    match = re.search(agreement_pattern, stop_text, re.IGNORECASE)
    if match and match.start() < stop_pos_in_slice:
        stop_pos_in_slice = match.start()

    stop_pos = toc_start + stop_pos_in_slice
    extracted_text = text[toc_start:stop_pos].strip()

    lines = extracted_text.splitlines()
    # cleaned_lines = [re.sub(r'\s*\d+$', '', line).strip() for line in lines if line.strip()]
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # ✅ only drop if the whole line is a page number
        if re.fullmatch(r"\d+", line):
            continue

        if (
            re.fullmatch(r"[ivxlcdm]+", line, re.IGNORECASE)
            or re.fullmatch(r"[-–—]\s*[ivxlcdm]+\s*[-–—]", line, re.IGNORECASE)
            or re.fullmatch(r"\(\s*[ivxlcdm]+\s*\)", line, re.IGNORECASE)
        ):
            continue

        cleaned_lines.append(line)

    extracted_text = "\n".join(cleaned_lines)
    extracted_text = re.sub(
        r'^[\s\xa0]*TABLE OF CONTENTS Page[\s\xa0]*', '', extracted_text, flags=re.IGNORECASE)

    return extracted_text if extracted_text else "[ERROR: No valid content extracted]", stop_pos

# ---------------------------
# FORMAT TOC
# ---------------------------


def format_toc_text(raw_toc: str) -> str:
    if not raw_toc:
        return raw_toc

    lines = [line.strip() for line in raw_toc.splitlines() if line.strip()]
    formatted_lines = []
    buffer = ""

    for line in lines:
        if TOC_COMBINED_PATTERN.match(line) or re.match(r"^[A-Z][A-Za-z\s'‘’]+ \d+$", line):
            if buffer:
                formatted_lines.append(buffer.strip())
            buffer = line
        else:
            if buffer:
                buffer += " " + line
            else:
                buffer = line

    if buffer:
        formatted_lines.append(buffer.strip())

    return "\n".join(formatted_lines)

# def format_toc_text(raw_toc: str) -> str:
#     if not raw_toc:
#         return raw_toc

#     lines = [line.strip() for line in raw_toc.splitlines() if line.strip()]
#     return "\n".join(lines)

# ---------------------------
# OPENAI CALL — CLEAN JSON OUTPUT
# ---------------------------


def call_openai_api(toc_text):

    system_prompt = """
    I am an expert in analyzing legal M&A agreement and corporate filings. 
    I specialize in identifying and structuring legal content—especially tables of contents—into clean, standardized data formats.
    """

    user_prompt_template = f"""
        You are a helpful assistant that extracts a structured Table of Contents (TOC) from raw text.
        here is my text : {toc_text}
        
        Your task is to convert the given TOC into a strict JSON format and output ONLY the JSON array—no additional commentary.

        Rules:
        1. Each entry should have a title. Ignore page numbers. for e.g. "Section 1.01 Finders or Brokers i Page" or "Section 4.2 Intellectual Property Rights i" or "Section 4.2 Stockholder of Merger Sub 1 and Merger Sub 2. 50" here i and page and 50 have to ignore.
        2. Maintain the hierarchy (main articles and their sections); maximum depth is 2 levels.
        3. Only include entries that look like actual document sections (skip repetitive or decorative lines).
        4. Sort entries by the order they appear in the text (not alphabetically).
        5. Remove any duplicate entries.
        6. Use indentation or numbering (e.g., 1.1, 2.2) to determine subsections.
        7. Remove the title "Table of Contents" if it appears.
        8. Do not modify the title formatting. Preserve all numbering, capitalization, punctuation, and spacing exactly as it appears.do not add anything extra (for e.g. period(.) after article or section index lable).
        9. Output ONLY the JSON array in the specified format. Do not include any explanation, introduction, or surrounding text.
        10. Do not infer or add article or section headers that are not explicitly present in the input. Only include content that exists exactly as written in the raw text.
        11. **CRITICAL: PRESERVE EXACT SECTION LABELING**
            - If raw text says: `Section 1.1` → Output: `"section": "Section 1.1"`
            - If raw text says: `1.1` → Output: `"section": "1.1"`
            - If raw text says: `SECTION 1.1` → Output: `"section": "SECTION 1.1"`
            - **Never standardize. Never remove "Section or Article". Never add it.**
            - **Match the raw text 100% exactly.**
        12. ARTICLE OR SECTION INDEX FORMAT
            The "article" or "section" field must only contain the index label itself, not the title.
            Incorrect example 1:
                "article": "SECTION 1 THE OFFER",
                "title": "THE OFFER"
            Correct example 1:
                "article": "SECTION 1",
                "title": "THE OFFER"
            Incorrect example 2:
                "article": "1. THE OFFER",
                "title": "THE OFFER"
            Correct example 2:
                "article": "1.",
                "title": "THE OFFER"    
        
                
            # Notes:
                - Do not add any periods if they are not in the raw text.
                - Preserve the numbering exactly as in the source text.
                - Never invent or standardize "Section" or "Article" labels. 

            The "article" or "section" field must follow Rule 11 for exact preservation of the original labeling.
        Output format must match exactly this JSON structure:
        [
            {{
              "article": "<exact article index label>",
              "title": "<article title>",
              "text": "",
              "sections": [
                {{
                  "section": "<exact section index label>",
                  "title": "<section title>",
                  "text": ""
                }}
              ]
            }}
        ]
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template}
            ]
        )

        usage = response.usage
        if usage:
            log_entry = f"Prompt:Tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}"
            print(log_entry)

        # === GET RAW OUTPUT ===
        raw = response.choices[0].message.content.strip()

        # === REMOVE MARKDOWN CODE BLOCKS ===
        if raw.startswith("```json"):
            raw = raw[7:]  # Remove ```json
        elif raw.startswith("```"):
            raw = raw[3:]  # Remove ```
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        # === FINAL VALIDATION ===
        try:
            json.loads(raw)  # Test if valid JSON
            return raw
        except json.JSONDecodeError as e:
            logger.error(
                f"OpenAI returned invalid JSON: {e}\nRAW OUTPUT:\n{raw[:500]}")
            return None

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return None

# ---------------------------
# CLEAN & TRUNCATE (ONLY FOR LAST SECTION)  ////remove this method later this is not in use
# ---------------------------


def clean_and_truncate_text(text):
    if not text or not text.strip():
        return ""
    text = re.sub(r'^\s*\.\s*', '', text).strip()

    earliest_pos = len(text)
    for phrase in TRUNCATE_PHRASES:
        pos = text.lower().rfind(phrase.lower())
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos

    if earliest_pos < len(text):
        clean_part = text[:earliest_pos]
        last_end = -1
        for char in '.);':
            pos = clean_part.rfind(char)
            if pos > last_end:
                last_end = pos
        if last_end >= 0:
            text = clean_part[:last_end + 1].strip()
        else:
            text = clean_part.strip()

    text = re.sub(r'\s*\d+\s*$', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    return text.strip()

# ---------------------------
# LEVEL 2 TEXT EXTRACTION (FUZZY + LAST SECTION TRUNCATE)
# ---------------------------


def text_clean(full_text: str, sequence: list) -> str:
    """Clean the text and save it to zone.txt"""
    if not full_text:
        return ""

    # Replace non-breaking space and remove non-printable characters

    text = full_text.replace('\xa0', ' ')
    text = ''.join(c if (c.isprintable() or c == '\n') else ' ' for c in text)

    # Normalize spaces
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)

    # Strip each line
    text = '\n'.join(line.strip() for line in text.splitlines())
    text = text.strip()

    if sequence:
        first_element = sequence[0]
        start_pattern = f"{first_element['id']} {first_element['title']}"
        dot_pattern = f"{first_element['id']}. {first_element['title']}"

        id_ = str(first_element["id"]).strip()
        title = str(first_element["title"]).strip()

        # ✅ title can be on SAME line or NEXT line; optional dot after id
        # Matches:
        #   "ARTICLE 1\nDEFINITIONS"
        #   "ARTICLE 1 DEFINITIONS"
        #   "ARTICLE 1.\nDEFINITIONS"
        #   "ARTICLE 1. DEFINITIONS"
        rx = re.compile(
            rf"{re.escape(id_)}(?:\s*\.)?(?:[ \t]+|\n)+{re.escape(title)}",
            re.IGNORECASE
        )

        m = rx.search(text)
        start_idx = m.start() if m else -1

        if start_idx != -1:
            text = text[start_idx:]
        else:
            logger.warning(
                f"Start pattern not found at creating zone text: {start_pattern}")

    # --- Truncate based on end patterns ---
    earliest_pos = len(text)
    for phrase in TRUNCATE_PHRASES:
        pattern = re.compile(
            rf"\[?\s*{re.escape(phrase)}\s*\]?", re.IGNORECASE)
        match = pattern.search(text)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()

    if earliest_pos < len(text):
        text = text[:earliest_pos]

    # --- 🔧 Fix spaces before punctuation marks (important for matching) ---

    # text = re.sub(r"\s+([’‘'“”.,;:!?])", r"\1", text)       # manage space before punctuation
    text = re.sub(r"[ \t]+([’‘'“”,;:!?])", r"\1", text)

    text = re.sub(r"([’‘'])\s+([a-z])(?![a-z])", r"\1\2", text)

    # Save to zone.txt

    # logger.info(f"Cleaned text saved to {zone_txt_path}, length: {len(text)}")
    return text
# ---------------------------
# 2️⃣ ARRAY EXTRACTION FUNCTION
# ---------------------------


def array_extraction(toc_json: list) -> list:

    if not toc_json:
        logger.error("No TOC JSON provided to array_extraction().")
        return []
    sequence = []
    for art in toc_json:
        art_id = str(art.get("article", "")).strip()
        art_title = str(art.get("title", "")).strip()
        if art_id:
            sequence.append(
                {"type": "article", "id": f"{art_id}", "title": art_title})
        for sec in art.get("sections", []):
            sec_id = str(sec.get("section", "")).strip()
            sec_title = str(sec.get("title", "")).strip()
            # If the LLM embedded the title inside the section ID field, strip it
            if sec_title and sec_id.lower().endswith(sec_title.lower()):
                sec_id = sec_id[:-len(sec_title)].strip().rstrip('.')
            sequence.append(
                {"type": "section", "id": sec_id, "title": sec_title})

    return sequence


def partition_zone_text(zone_text: str, sequence_array: list, toc_json: list, toc_end_pos) -> list:
    if not zone_text or not sequence_array:
        logger.error("No zone text or sequence array provided.")
        return []

    # Normalize text for consistent searching
    # zone_norm = re.sub(r'\s+', ' ', zone_text)

    zone_norm = re.sub(r'[ \t]+', ' ', zone_text)
    zone_norm = re.sub(r'\n{2,}', '\n', zone_norm)

    results = []
    positions = []
    search_pos = 0

    # --- 1️⃣ Find start positions for each ID+title ---
    for item in sequence_array:
        id_ = item.get("id", "").strip()
        title = item.get("title", "").strip()
        if not id_ or not title:
            logger.warning(
                f"Skipped TOC item (missing id/title): id='{id_}' title='{title}'")
            continue

        # Build regex pattern for "ID + TITLE" with punctuation flexibility

        title_rx = r"\s+".join(map(re.escape, title.split()))
        pattern = re.compile(
            rf"{re.escape(id_)}(?:\s*\.)?\s*{title_rx}\s*[:.\-–]*",
            re.IGNORECASE
        )
        match = pattern.search(zone_norm, search_pos)
        if match:
            # Use match.end() so text starts after title (removes header from text)
            positions.append((match.start(), match.end(), id_, title))
            search_pos = match.end()
        else:
            logger.warning(f"Pattern not found for: {id_} {title}")

    if not positions:
        logger.error("No valid section/article patterns found in text.")
        return []

    # --- 2️⃣ Extract text slices between positions and update toc_json ---
    is_true = False
    for i, (start_pos, end_pos, id_, title) in enumerate(positions):
        next_start_pos = positions[i + 1][0] if i + \
            1 < len(positions) else len(zone_norm)
        text_part = zone_norm[end_pos:next_start_pos].strip()

        # --- Check for Definitions keyword match ---
        is_definition = False
        def_result = None
        for keyword in DEFINITION_KEYWORDS:
            pattern = re.compile(
                rf"^\[?\s*{re.escape(keyword.lower())}\.?\s*\]?$", re.IGNORECASE)
            if pattern.search(title.lower().strip()):
                print(f"id_={id_}, title={title}")
                def_result, is_true = process_definitions_section(text_part)
                is_definition = True
                break

        # --- Skip if it's definitions ---
        if is_definition:
            if def_result and def_result.get("definitions"):
                defs = def_result["definitions"]

                for article in toc_json:
                    # Match article level
                    if article.get("article", "").strip() == id_:
                        article["definitions"] = defs

                    # Match section level under article
                    for section in article.get("sections", []):
                        if section.get("section", "").strip() == id_:
                            section["definitions"] = defs

            # Skip adding this block as a normal text section
            continue

        results.append({
            "id": id_,
            "title": title,
            "text": text_part
        })

        # --- NEW: Update toc_json ---
        for article in toc_json:
            # If this id_ matches the article number
            if article.get("article", "").strip() == id_:
                article["text"] = text_part
            # Check sections under the article
            for section in article.get("sections", []):
                if section.get("section", "").strip() == id_:
                    section["text"] = text_part

    if (is_true):
        print("definitions extracted. (inner check)")
    else:
        print("let's check outside definitions extraction.")
        defi, is_true = outside_definitions_extraction(toc_end_pos)
        if is_true and defi and defi.get("definitions"):
            print("outside definitions extraction SUCCESS")
        else:
            logger.warning("definitions extraction FAILED")
        preamble_entry = {"article": "Definitions",
                          "definitions": defi["definitions"]}
        toc_json.append(preamble_entry)

    return toc_json  # <-- Return updated toc_json


def process_definitions_section(text_part: str):
    text1 = (text_part or "").replace(
        "“", '"').replace("”", '"').replace("’’", '"')

    # normalize newlines (keep them!)
    text1 = text1.replace("\r\n", "\n").replace("\r", "\n")

    # remove non-printables but KEEP \n
    text1 = "".join(c if (c.isprintable() or c == "\n")
                    else " " for c in text1)

    # ✅ split into lines first
    arr = [line.strip() for line in text1.split("\n") if line.strip()]

    # ✅ now normalize spaces INSIDE each line (does not touch \n anymore)
    arr = [re.sub(r"[ \t]+", " ", line) for line in arr]

    leadin = (
        r'\b(?:'
        r'as of the time of reference|'
        r'refers to|'
        r'shall be|'
        r'and(?:\s+any)?\s+similar\s+phrase(?:s)?\s+mean(?:s)?|'
        r'or(?:\s+(?:any|a))?\s+similar\s+phrase(?:s)?\s+mean(?:s)?|'
        r'of the Company|'
        r'shall include|'
        r'shall be deemed to be|'
        r'will be deemed to be|'
        r'shall be deemed to have|'
        r'an Entity shall be deemed to be|'
        r'shall have the meaning(?:s)?(?: of)?|'
        r'have the(?:\s+respective)?\s+meaning(?:s)?(?: of)?|'
        r'has the(?:\s+same)?\s+meaning(?:s)?(?:\s+of)?|'
        r'has the given meaning(?:s)?(?: of)?|'
        r'of any entity mean(?:s)?|'
        r'of any party mean(?:s)?|'
        r'of a party mean(?:s)?|'
        r'of any(?:\s+particular)?\s+person\s+mean(?:s)?|'
        r'of any person shall mean(?:s)?|'
        r'of a specified person mean(?:s)?|'
        r'of a person(?:s)?|'
        r'of any person(?:s)?|'
        r'of a person mean(?:s)?|'
        r'of a Party mean(?:s)?|'
        r'by a Party mean(?:s)?|'
        r'with regard to |'
        r'(?:as used\s+)?with respect to(?:\s+(?:any|a))?\s+Person(?:s)?|'
        r'with respect to an Entity|'
        r'when used with respect to any(?:\s+specified)?\s+Person(?:s)?|'
        r'when used with respect to any party|'
        r'with respect to an Entity shall mean(?:s)?|'
        r'with respect to Parent or the Company|'
        r'(?:as applied\s+)?with respect to the Company|'
        r'shall mean(?:s)?|'
        r'is defined(?:s)?|'
        r'mean(?:s)?'
        r')\b'
    )

    header_pat = re.compile(
        r"^\d+(\.\d+)*\s+(Certain Definitions|Definitions|Defined Terms)\b", re.I)

    start_quoted = re.compile(
        rf'^(?:(?:\([A-Za-z0-9]+\)|[A-Za-z0-9]+[.)])\s*)?'
        rf'(?:(?:An|A|The)\s*)?'
        rf'(?P<head>"[^"]+"(?:\s*(?:(?:,|and|or)\s*|\s*)"[^"]+")*)\s*[,;:]?\s*'
        rf'(?:\([^)]*\))?\s*(?:or similar terms)?\s*(?P<leadin>{leadin})\s*[,;:]?\s*',
        re.I)

    start_label_quoted = re.compile(
        rf'^(?P<label>[A-Za-z][A-Za-z0-9_-]*(?:\s+[A-Za-z][A-Za-z0-9_-]*)*)\.\s*'
        rf'(?P<head>"[^"]+"(?:\s*(?:,|and|or)\s*"[^"]+")*)\s*'
        rf'(?:\([^)]*\))?\s*(?:or similar terms)?\s*(?P<leadin>{leadin})\s*[,;:]?\s*',
        re.I)

    start_heading_quoted = re.compile(
        rf'^(?P<label>[A-Za-z0-9&\-., ]+)\.\s*'
        rf'(?P<head>"[^"]+"(?:\s*(?:,|and|or)\s*"[^"]+")*)\s*'
        rf'(?:\([^)]*\))?\s*(?:or similar terms)?\s*(?P<leadin>{leadin})\s*[,;:]?\s*',
        re.I)

    start_unquoted = re.compile(
        rf'^(?!operations\b)(?P<term>[A-Za-z][A-Za-z0-9_-]*)\s+(?P<leadin>{leadin})\s*[,;:]?\s*', re.I)
    dup_pat = re.compile(
        r'^(?P<dup>[A-Za-z][A-Za-z0-9_-]*)\s+"(?P=dup)"\s+', re.I)

    definitions, current_terms, current_def = [], [], ""

    def flush(is_last=False):
        nonlocal current_terms, current_def
        if current_terms:
            d = re.sub(r"\s+", " ", current_def).strip()

            # ✅ apply TERM_INDEX_INTRO_PATTERN only for LAST definition (old behavior)
            if is_last:
                m_idx = TERM_INDEX_INTRO_PATTERN.search(d)
                if m_idx:
                    d = d[:m_idx.start()].rstrip()
                    lp = d.rfind('.')
                    if lp != -1:
                        d = d[:lp+1].strip()

            # if REF_ONLY_DEF.match(d):
            #     current_terms, current_def = [], ""
            #     return

            # for t in current_terms:
            #     if not REF_ONLY_DEF.search(d):
            #         definitions.append({"term": t, "definition": d})

            for t in current_terms:
                definitions.append({"term": t, "definition": d})
        current_terms, current_def = [], ""

    for idx, line in enumerate(arr):
        if header_pat.match(line):
            continue
        if line.strip().isdigit():   # single number line that is page number will be skipped
            continue

        if ((TERM_INDEX_INTRO_PATTERN.search(line) or re.search(r'\bTerm\b\s+\bSection\b', line, re.I)) and idx > 5):
            flush(is_last=True)
            break

        chunk = line[:120]
        m_q = start_quoted.search(chunk)
        m_u = start_unquoted.search(chunk)
        m_lq = start_label_quoted.search(chunk)
        m_hq = start_heading_quoted.search(chunk)

        if m_q or m_u or m_lq or m_hq:
            flush()

            if m_q:
                head = m_q.group("head")
                terms = re.findall(r'"([^"]+)"', head)

                # Variation 3: Code "Code" shall mean ... -> keep only quoted "Code"
                dm = dup_pat.match(line)
                if dm and terms and dm.group("dup").lower() == terms[0].lower():
                    # ok: ignore the unquoted duplicate; terms already correct
                    pass

                current_terms = terms
                current_def = line[m_q.end():].strip()
            elif m_lq:
                head = m_lq.group("head")
                terms = re.findall(r'"([^"]+)"', head)
                current_terms = terms
                current_def = line[m_lq.end():].strip()
            elif m_hq:
                terms = re.findall(r'"([^"]+)"', m_hq.group("head"))
                current_terms = terms
                current_def = line[m_hq.end():].strip()

            else:
                current_terms = [m_u.group("term")]
                current_def = line[m_u.end():].strip()
        else:
            if current_terms:
                current_def += " " + line

    # ✅ last definition truncation here
    flush(is_last=True)
    return {"definitions": definitions}, bool(definitions)


def outside_definitions_extraction(toc_end_pos):
    # Use thread-local variables instead of globals
    doc_text = getattr(thread_local, 'doc_text', None)
    preamble_end_pos = toc_end_pos

    if not doc_text or preamble_end_pos is None:
        return {"definitions": []}, False

    for pattern in metadata_patterns:
        text = re.sub(pattern, "", doc_text,
                      flags=re.IGNORECASE | re.MULTILINE)

    start_idx = None

    for phrase in TRUNCATE_PHRASES:
        # find this phrase AFTER preamble_end_pos
        m = re.search(re.escape(phrase),
                      text[preamble_end_pos:], re.IGNORECASE)
        if m:
            # absolute index in full text
            abs_pos = preamble_end_pos + m.end()
            if start_idx is None or abs_pos < start_idx:
                start_idx = abs_pos

    if start_idx is None:
        # fallback: use preamble_end_pos
        start_idx = preamble_end_pos

    process_text = text[start_idx:]

    final_def_text = ""

    # 1) Find starting point: OUTSIDE_DEF_HEADER
    m_start = OUTSIDE_DEF_HEADER.search(process_text)

    print("OUTSIDE_DEF_HEADER matched:", repr(m_start.group(0)))
    if m_start:
        start_idx = m_start.end()  # start after header

        # Slice from start for endpoint search
        tail = process_text[start_idx:]
        # print(f"defintions raw text : {tail}")``

        # 2) Find end point
        # m_end1 = TERM_INDEX_INTRO_PATTERN.search(tail[200:])          # 1st priority
        OFFSET = 200
        m_end2 = SECONDARY_DEFS_END_PATTERN.search(
            tail[OFFSET:])        # 2nd priority
        m_end3 = CHARTER_STOP_PATTERN.search(tail)

        # if m_end1:
        #     end_rel = m_end1.start()
        #     end_rel += 200  # adjust for offset
        # el
        if m_end2:
            end_rel = OFFSET + m_end2.start()
        elif m_end3:
            end_rel = m_end3.start()
        else:
            end_rel = len(tail)  # no endpoint → go to end

        end_idx = start_idx + end_rel
        final_def_text = process_text[start_idx:end_idx].strip()
        m = SECTION_CERTAIN_DEFS_PATTERN.search(final_def_text)
        if m:
            final_def_text = final_def_text[m.start():]
        # print(f"defintions raw text : {final_def_text}")

    def_result, is_true = process_definitions_section(final_def_text)
    return def_result, is_true


# ---------------------------
# 3️⃣ LEVEL 2 EXTRACTION FUNCTION
# ---------------------------
def extract_level2_text(full_text, toc_end_pos, toc_json):
    """Level 2 extraction: clean text + array extraction"""
    if not full_text or toc_end_pos is None:
        return "", []

    # 1️⃣ Extract array from sample.json
    sequence_array = array_extraction(toc_json)
    # 2️⃣ Clean the zone text

    zone_text = text_clean(full_text[toc_end_pos:], sequence_array)

    newjson = partition_zone_text(
        zone_text, sequence_array, toc_json, toc_end_pos)
    return newjson


# ---------------------------
# JSON VALIDATE FUNCTION
# ---------------------------
def validate_toc_numbering(toc_json, full_text):
    import re
    missing_articles, missing_sections = [], []

    # ---------------------------
    # 🔹 Level 1: Article check
    # ---------------------------
    article_numbers = []
    for art in toc_json:
        m = re.search(r'(\d+)', art.get("article", ""))
        if m:
            article_numbers.append(int(m.group(1)))

    if article_numbers:
        min_article = 1   # we expect Article 1 to start
        max_article = max(article_numbers)
        for i in range(min_article, max_article + 1):
            if i not in article_numbers:
                missing_articles.append(f"ARTICLE {i}")

    # ---------------------------
    # 🔹 Level 2: Section check
    # ---------------------------
    for art in toc_json:
        m = re.search(r'(\d+)', art.get("article", ""))
        if not m:
            continue
        art_num = int(m.group(1))
        section_nums = []

        for sec in art.get("sections", []):
            s = re.search(rf'{art_num}\.(\d+)', sec.get("section", ""))
            if s:
                section_nums.append(int(s.group(1)))

        if section_nums:
            min_sec = 1  # assume sections should start at .1
            max_sec = max(section_nums)
            for i in range(min_sec, max_sec + 1):
                if i not in section_nums:
                    missing_sections.append(f"Section {art_num}.{i}")

    result = {
        "missing_articles": missing_articles,
        "missing_sections": missing_sections
    }

    # toc_json = find_and_add_missing_articles(full_text, toc_json, result)

    return {
        "missing_articles": missing_articles,
        "missing_sections": missing_sections
    }


# ---------------------------
# DOCUMENT TYPE DETECTION
# ---------------------------
def _classify_from_content(text: str) -> str:
    """Check raw HTML/text for SEC SGML headers and common document phrases."""
    first_5k = text[:5000].upper()

    # SEC SGML headers (very reliable — always at top of EDGAR documents)
    if '<TYPE>EX-99' in first_5k or 'EXHIBIT 99' in first_5k:
        return 'press_release'
    if '<TYPE>8-K' in first_5k or '<TYPE>8K' in first_5k:
        return '8k'
    if '<TYPE>EX-2' in first_5k or 'EXHIBIT 2.1' in first_5k:
        return 'merger_agreement'

    # Common document phrases
    if 'AGREEMENT AND PLAN OF MERGER' in first_5k or 'MERGER AGREEMENT' in first_5k:
        return 'merger_agreement'
    if 'FORM 8-K' in first_5k or 'CURRENT REPORT' in first_5k:
        return '8k'
    if 'PRESS RELEASE' in first_5k or 'FOR IMMEDIATE RELEASE' in first_5k:
        return 'press_release'

    return ''


def _classify_with_llm(text: str) -> str:
    """Last-resort classification using Haiku (very cheap)."""
    try:
        snippet = text[:2000]
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=20,
            temperature=0,
            messages=[{"role": "user", "content": (
                "Classify this SEC filing document. Reply with EXACTLY one word: "
                "merger_agreement, press_release, or 8k\n\n" + snippet
            )}]
        )
        answer = response.content[0].text.strip().lower()
        if answer in ('merger_agreement', 'press_release', '8k'):
            print(f"  Haiku classified document as: {answer}")
            return answer
    except Exception as e:
        print(f"  Warning: Haiku classification failed: {e}")
    return ''


def detect_document_type(url: str, text: str = None) -> str:
    """
    Auto-detect document type from URL pattern, content headers, or LLM.
    Returns: 'merger_agreement', '8k', or 'press_release'
    """
    url_lower = url.lower()

    # 1. URL-based detection (free, instant)
    # SEC EDGAR uses zero-padded exhibit numbers: ex0201 = Exhibit 2.1, ex9901 = Exhibit 99.1
    if re.search(r'ex[\-_]?0?99[\-_]?0?1', url_lower):
        return 'press_release'
    if re.search(r'[\-_]8k\.htm', url_lower) or re.search(r'/0.*8-?k', url_lower):
        return '8k'
    if re.search(r'ex[\-_]?0?2[\-_]?0?1|dex2', url_lower):
        return 'merger_agreement'

    # 2. Content-based detection (check SGML headers + phrases)
    content = text
    if not content:
        # Quick lightweight fetch — just the first 3KB to check headers
        try:
            import requests as _req
            _sec_hdr = {
                'User-Agent': 'Academic Research Project mergerresearch@research.edu',
                'Accept-Encoding': 'gzip, deflate',
            }
            resp = _req.get(url, headers=_sec_hdr, timeout=10, stream=True)
            content = resp.raw.read(3000).decode('utf-8', errors='replace')
            resp.close()
        except Exception as e:
            print(f"  Warning: quick fetch for doc type detection failed: {e}")

    if content:
        result = _classify_from_content(content)
        if result:
            return result

    # 3. LLM fallback — Haiku (very cheap, ~$0.0001 per call)
    if content:
        result = _classify_with_llm(content)
        if result:
            return result

    # Default to merger agreement
    return 'merger_agreement'


# ---------------------------
# 8-K / PRESS RELEASE PROCESSING
# ---------------------------
def process_8k_press_release(url: str, accession: str,
                             pipeline_deal_id: str = None,
                             pipeline_doc_type: str = None) -> dict:
    """
    Lighter extraction path for 8-K filings and press releases.
    Extracts: fees, deal value, key terms, and whatever trigger info is available.

    When pipeline_deal_id and pipeline_doc_type are provided, outputs are uploaded
    to S3 instead of written locally.
    """
    import requests
    from bs4 import BeautifulSoup as BS4

    sec_headers = {
        'User-Agent': 'Academic Research Project mergerresearch@research.edu',
        'Accept-Encoding': 'gzip, deflate',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }

    print(f"\n{'='*60}")
    print(f"  8-K / PRESS RELEASE MODE: {accession}")
    print(f"{'='*60}")
    print(f"  URL: {url}")

    # Fetch and clean text
    try:
        resp = requests.get(url, headers=sec_headers, timeout=20)
        resp.raise_for_status()
        soup = BS4(resp.text, 'html.parser')
        for tag in soup(['script', 'style', 'head']):
            tag.decompose()
        doc_text = soup.get_text(separator='\n', strip=True)
        doc_text = re.sub(r'\n{3,}', '\n\n', doc_text)
        doc_text = doc_text[:20000]
    except Exception as e:
        logger.error(f"Failed to fetch 8-K/press release: {e}")
        return {"status": "error", "url": url, "reason": str(e)}

    detected_type = detect_document_type(url, doc_text)
    source_label = "press_release" if detected_type == "press_release" else "8k"
    print(f"  Document type: {source_label}")
    print(f"  Text length: {len(doc_text)} chars")

    # ---- TRACK 1: Fee Extraction ----
    print(f"\n  Extracting termination fees...")
    fee_prompt = f"""You are an expert M&A legal analyst. Extract ALL termination fee data from this
{source_label.replace('_', ' ')} about a merger/acquisition.

Press releases and 8-Ks state fees in plain English, e.g.:
- "The Company will pay a termination fee of $86 million..."
- "Parent will pay a reverse termination fee of $184 million..."
- "In the event of regulatory failure, Parent will pay $221 million..."

Return JSON in EXACTLY this format:

{{
  "company_termination_fee": {{
    "amount_usd": null,
    "amount_text": null,
    "triggers": [],
    "notes": null
  }},
  "reverse_termination_fee": {{
    "amount_usd": null,
    "amount_text": null,
    "triggers": [],
    "notes": null
  }},
  "parent_regulatory_termination_fee": {{
    "amount_usd": null,
    "amount_text": null,
    "triggers": [],
    "notes": null
  }},
  "expense_reimbursement": {{
    "amount_usd": null,
    "amount_text": null,
    "triggers": [],
    "notes": null
  }},
  "offer_price_per_share": {{
    "amount_usd": null,
    "amount_text": null,
    "consideration_type": null
  }},
  "deal_equity_value_usd": null,
  "deal_equity_value_text": null,
  "outside_date_months": null,
  "go_shop_period_days": null,
  "specific_performance_available": null,
  "willful_breach_carveout": null,
  "tail_provision_months": null,
  "sole_remedy_for_acquirer": null,
  "notes": null
}}

FIELD GUIDANCE:
- amount_usd: integer dollar amount (e.g. 272000000 for $272M). null if not mentioned.
- amount_text: the text as stated (e.g. "$272 million" or "$272,000,000")
- company_termination_fee: fee the TARGET/company pays to acquirer (break-up fee)
- reverse_termination_fee: fee the ACQUIRER/parent pays to target (reverse break fee)
- parent_regulatory_termination_fee: a SEPARATE, higher fee acquirer pays specifically for
  regulatory/antitrust failure. Only populate if explicitly a different amount from the standard RTF.
- triggers: list of scenarios, e.g.:
  ["fiduciary_out", "superior_proposal", "regulatory_block", "financing_failure", "acquirer_breach"]
- deal_equity_value_usd: total deal value as integer (e.g. 8800000000 for $8.8B)
- deal_equity_value_text: compact format like "$8.8B"
- outside_date_months: number of months until outside/drop-dead date
- go_shop_period_days: length of go-shop period in days (null if no-shop)
- If a field is not mentioned, use null.

Document text:
{doc_text}

Return ONLY the JSON object. No explanation."""

    fees_data = {}
    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            temperature=0,
            timeout=120.0,
            messages=[{"role": "user", "content": fee_prompt}]
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        fees_data = json.loads(raw.strip())
        fees_data["document_id"] = accession
        fees_data["extraction_source"] = f"8k_{source_label}"
        fees_data["source_url"] = url

        # Print summary
        ctf = fees_data.get("company_termination_fee", {}) or {}
        rtf = fees_data.get("reverse_termination_fee", {}) or {}
        reg = fees_data.get("parent_regulatory_termination_fee", {}) or {}
        print(
            f"  Company termination fee: {ctf.get('amount_text') or 'not found'}")
        print(
            f"  Reverse termination fee: {rtf.get('amount_text') or 'not found'}")
        if reg.get("amount_text"):
            print(f"  Regulatory term. fee:    {reg.get('amount_text')}")
        if fees_data.get("deal_equity_value_text"):
            print(
                f"  Deal value:              {fees_data.get('deal_equity_value_text')}")
    except Exception as e:
        logger.error(f"Fee extraction from 8-K failed: {e}")
        fees_data = {"document_id": accession, "error": str(e)}

    # Write fees output (tagged as 8K source)
    s3_urls = {}
    if pipeline_doc_type:
        from Termination_Embeddings_v1.termination_s3_utils import upload_json as _s3_upload_json
        _, fees_8k_url = _s3_upload_json(
            fees_data, accession, pipeline_doc_type, "fees_8k_json.json")
        s3_urls["fees_8k_json"] = fees_8k_url
        print(f"\n  Fees uploaded to S3: {fees_8k_url}")
    else:
        fees_filename = f"termination_response_{accession}_fees_8k.json"
        with open(fees_filename, "w", encoding="utf-8") as f:
            json.dump(fees_data, f, indent=2, ensure_ascii=False)
        print(f"\n  Fees output: {fees_filename}")

    # ---- TRACK 2: Trigger Extraction (best-effort from 8-K) ----
    print(f"\n  Extracting termination triggers (best-effort)...")
    trigger_prompt = f"""You are an expert M&A legal analyst. Extract termination trigger information
from this {source_label.replace('_', ' ')} about a merger/acquisition.

8-Ks and press releases describe termination rights at a high level. Extract whatever
trigger clauses are mentioned, even if summarized.

Return JSON in this format:
{{
  "document_id": "{accession}",
  "total_clauses": 0,
  "extraction_source": "8k_{source_label}",
  "clauses": [
    {{
      "clause_id": "1",
      "clause_type": "trigger",
      "trigger_type": "<mutual_consent|outside_date|regulatory_block|breach|fiduciary_out|superior_proposal|other>",
      "original_text": "<exact text from the document describing this trigger>",
      "processed_text": "<same as original_text>",
      "parties_replaced": []
    }}
  ]
}}

If the document mentions termination scenarios (e.g. "The agreement may be terminated by
either party if the merger is not completed by [date]"), extract each as a separate clause.

If NO termination trigger details are mentioned, return an empty clauses array.

Document text:
{doc_text}

Return ONLY the JSON object. No explanation."""

    triggers_data = {"document_id": accession, "total_clauses": 0, "clauses": [],
                     "extraction_source": f"8k_{source_label}"}
    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3000,
            temperature=0,
            timeout=120.0,
            messages=[{"role": "user", "content": trigger_prompt}]
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        triggers_data = json.loads(raw.strip())
        triggers_data["extraction_source"] = f"8k_{source_label}"
        n_clauses = len(triggers_data.get("clauses", []))
        print(f"  Found {n_clauses} trigger clause(s)")
    except Exception as e:
        logger.error(f"Trigger extraction from 8-K failed: {e}")

    # Write triggers output (tagged as 8K source)
    if pipeline_doc_type:
        from Termination_Embeddings_v1.termination_s3_utils import upload_json as _s3_upload_json
        _, triggers_8k_url = _s3_upload_json(
            triggers_data, accession, pipeline_doc_type, "triggers_8k_json.json")
        s3_urls["triggers_8k_json"] = triggers_8k_url
        print(f"  Triggers uploaded to S3: {triggers_8k_url}")
    else:
        triggers_filename = f"termination_response_{accession}_triggers_8k.json"
        with open(triggers_filename, "w", encoding="utf-8") as f:
            json.dump(triggers_data, f, indent=2, ensure_ascii=False)
        print(f"  Triggers output: {triggers_filename}")

    print(f"\n{'='*60}")
    print(f"  8-K / PRESS RELEASE EXTRACTION COMPLETE")
    print(f"{'='*60}\n")

    result = {
        "status": "success",
        "url": url,
        "accession": accession,
        "document_type": source_label,
    }
    if s3_urls:
        result["s3_urls"] = s3_urls
    else:
        result["fees_file"] = fees_filename
        result["triggers_file"] = triggers_filename
    return result


# ---------------------------
# WORKER FUNCTION
# ---------------------------
def worker(url, pipeline_deal_id: str = None, pipeline_accession: str = None,
           pipeline_doc_type: str = None):
    """
    Main entry per URL. Detects doc type, dispatches to merger agreement or 8-K path.

    When pipeline_deal_id / pipeline_accession / pipeline_doc_type are provided
    (new flow), outputs are uploaded to S3 instead of written locally, and
    the accession is taken from pipeline_accession rather than derived from the URL.
    """
    if pipeline_accession:
        accession = pipeline_accession
    else:
        file_stem = url.split('/')[-1].split('.')[0]
        if sum(1 for c in file_stem if c.isdigit()) < 6:
            cik = url.split('/')[-3]
            accession = f"{file_stem}_{cik}"
        else:
            accession = file_stem

    # Auto-detect document type from URL
    doc_type = detect_document_type(url)
    if doc_type in ('8k', 'press_release'):
        return process_8k_press_release(url, accession,
                                        pipeline_deal_id=pipeline_deal_id,
                                        pipeline_doc_type=pipeline_doc_type)

    # --- Merger agreement path (existing flow) ---
    filename = f"termination_response_{accession}_full.json"
    cv_filename = f"termination_response_{accession}_triggers_raw.json"
    log_txt_path = f"termination_response_{accession}.txt"
    thread_id = threading.get_ident()

    # reset thread-local
    thread_local.doc_text = None
    thread_local.preamble_end_pos = None

    thread_local.log_records = []
    log_records = thread_local.log_records

    class ThreadFilter(logging.Filter):
        def __init__(self, target_thread_id):
            super().__init__()
            self.target_thread_id = target_thread_id

        def filter(self, record: logging.LogRecord) -> bool:
            # Only accept records from this specific thread
            # record.thread is the thread ID (integer) from logging
            return getattr(record, 'thread', None) == self.target_thread_id

    class ListHandler(logging.Handler):
        def __init__(self, log_records_list):
            # Only capture WARNING and above
            super().__init__(level=logging.WARNING)
            self.setFormatter(logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            ))
            self.log_records_list = log_records_list

        def emit(self, record: logging.LogRecord) -> None:
            try:
                msg = self.format(record)
                self.log_records_list.append(msg)
            except Exception:
                # Avoid crashing on logging failures
                pass

    list_handler = ListHandler(log_records)
    list_handler.addFilter(ThreadFilter(thread_id))
    logger.addHandler(list_handler)

    # Store thread-local variables
    thread_local.doc_text = None
    thread_local.preamble_end_pos = None

    try:
        # =========================
        # FAST MODE: reuse existing full.json from covenant scrape
        # (skipped entirely when pipeline params are provided — always full extraction)
        # =========================
        enriched = None
        toc_json = None
        preamble = None
        loaded_from_cache = False

        existing_full = f"openai_response_{accession}_full.json"
        if not pipeline_doc_type:
            if not os.path.exists(existing_full):
                file_stem_only = url.split('/')[-1].split('.')[0]
                stem_only_path = f"openai_response_{file_stem_only}_full.json"
                if os.path.exists(stem_only_path):
                    existing_full = stem_only_path
        else:
            existing_full = None

        if existing_full and os.path.exists(existing_full):
            print(
                f"⚡ Fast mode: reusing {existing_full} (skipping scrape)")
            with open(existing_full) as _f:
                enriched = json.load(_f)
            # Rebuild minimal toc_json from enriched data
            toc_json = []
            for _art in enriched:
                _art_id = _art.get('article', '')
                if _art_id == 'Preamble':
                    preamble = _art.get('text', '')
                    continue
                toc_json.append({
                    'article': _art_id,
                    'sections': [
                        {'section': _s.get('section', ''),
                         'title': _s.get('title', '')}
                        for _s in _art.get('sections', [])
                    ]
                })
            n_secs = sum(len(a.get('sections', [])) for a in toc_json)
            print(f"  Loaded {n_secs} sections from cache")
            loaded_from_cache = True

        if not loaded_from_cache:
            # =========================
            # FETCH FULL DOCUMENT TEXT (Playwright + requests)
            # =========================
            full_text = None

            for attempt in range(3):
                print(f"[fetch] Attempt {attempt + 1}/3 loading page...")
                try:
                    with sync_playwright() as p:
                        browser = p.chromium.launch(
                            headless=True,
                            args=[
                                "--no-sandbox",
                                "--disable-dev-shm-usage",
                            ],
                        )
                        try:
                            context = browser.new_context()
                            page = context.new_page()

                            headers = {
                                "User-Agent": SEC_USER_AGENT,
                                "Accept-Encoding": "gzip, deflate",
                                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                            }
                            response = _requests_lib.get(
                                url, headers=headers, timeout=60)
                            html_content = response.text or ""

                            if (
                                response.status_code == 403
                                or "Undeclared Automated Tool" in html_content
                                or "Request Originates" in html_content
                            ):
                                raise ValueError(
                                    f"SEC blocked via requests (status={response.status_code})"
                                )

                            if len(html_content) < 1000:
                                raise ValueError(
                                    f"HTML too short ({len(html_content)} chars)")

                            page.set_content(
                                html_content, wait_until="domcontentloaded")
                            page.wait_for_timeout(1000)

                            text = page.inner_text("body") or ""
                            if not text:
                                raise ValueError("No text extracted from HTML")
                        finally:
                            browser.close()

                    # normalize whitespace
                    text = text.replace('\xa0', ' ')
                    text = re.sub(r'[\u2000-\u200A\u202F\u205F]', ' ', text)
                    text = re.sub(r'[ \t]+', ' ', text)
                    text = re.sub(r'(?im)^\s*exhibit\s*2\.1\s*$', '', text)
                    text = re.sub(r'\n{2,}', '\n', text).strip()

                    if len(text) < 100_000:
                        raise ValueError("Document too short")

                    thread_local.doc_text = text
                    full_text = text
                    print(
                        f"[fetch] OK — document length: {len(full_text):,} chars")
                    break

                except Exception as e:
                    logger.warning(f"Retry {attempt+1}: {e}")
                    print(f"[fetch] Retry {attempt+1} failed: {e}")
                    time.sleep(2)

            if not full_text:
                return {
                    "status": "error",
                    "url": url,
                    "reason": "Failed to load valid document",
                    "warnings": log_records
                }

            print(f"document fetch complete, start TOC extraction...")
            # =========================
            # TOC EXTRACTION
            # =========================

            toc_raw, toc_end_pos = clean_metadata_and_toc(full_text, url)
            if not toc_raw:
                logger.warning("TOC not detected")
                toc_end_pos = 0

            formatted_toc = format_toc_text(toc_raw) if toc_raw else None

            if not formatted_toc:
                return {
                    "status": "error",
                    "url": url,
                    "reason": "TOC formatting failed",
                    "warnings": log_records
                }

            # =========================
            # OPENAI / MOCK TOC JSON
            # =========================
            api_response = call_openai_api(toc_raw)
            # api_response = mock_openai_api_call(filename)

            if not api_response:
                return {
                    "status": "error",
                    "url": url,
                    "reason": "TOC → JSON failed",
                    "warnings": log_records
                }

            if not pipeline_doc_type:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(api_response)

            toc_json = json.loads(api_response)

            for art in toc_json:
                if isinstance(art.get("article"), str):
                    art["article"] = art["article"].strip().rstrip(".")
                for sec in (art.get("sections") or []):
                    if isinstance(sec.get("section"), str):
                        sec["section"] = sec["section"].strip().rstrip(".")

            # =========================
            # TOC NUMBERING VALIDATION (SAFE)
            # =========================
            toc_validation = {"missing_articles": [], "missing_sections": []}
            try:
                toc_validation = validate_toc_numbering(
                    toc_json, full_text) or toc_validation
                missing_articles = toc_validation.get(
                    "missing_articles", []) or []
                missing_sections = toc_validation.get(
                    "missing_sections", []) or []

                if missing_articles:
                    logger.warning("TOC missing articles: " +
                                   ", ".join(missing_articles))

                if missing_sections:
                    logger.warning("TOC missing sections: " +
                                   ", ".join(missing_sections))
            except Exception as e:
                logger.warning(f"TOC validation failed (non-fatal): {e}")

            # =========================
            # PREAMBLE DETECTION
            # =========================
            preamble = None
            preamble_end_pos = None

            if toc_end_pos:
                search_from = full_text[toc_end_pos:]
                m_start = re.search(START_PATTERN, search_from, re.IGNORECASE)
                if m_start:
                    abs_start = toc_end_pos + m_start.start()
                    m_end = re.search(
                        END_PATTERN, full_text[abs_start:], re.IGNORECASE)
                    if m_end:
                        abs_end = abs_start + m_end.end()
                        preamble = full_text[abs_start:abs_end].strip()
                        thread_local.preamble_end_pos = abs_end
                        preamble_end_pos = abs_end
                    else:
                        print(
                            "Preamble end pattern not found, attempting alternative extraction")
                        sequence_array = array_extraction(toc_json)
                        if sequence_array:
                            first_element = sequence_array[0]

                            id_ = str(first_element.get("id", "")).strip()
                            title = str(first_element.get("title", "")).strip()

                            # Work only inside the slice starting at abs_start
                            search_slice = full_text[abs_start:]

                            # Matches: "1 Title", "1. Title", "1\tTitle", "1\nTitle"
                            rx = re.compile(
                                rf"{re.escape(id_)}(?:\s*\.)?(?:[ \t]+|\n)+{re.escape(title)}",
                                re.IGNORECASE
                            )

                            m = rx.search(search_slice)
                            if m:
                                # This is where the first TOC element begins, relative to abs_start
                                rel_idx = m.start()

                                # Preamble is everything before that element
                                preamble = full_text[abs_start:abs_start +
                                                     rel_idx].strip()

                                # These must be ABSOLUTE positions
                                abs_preamble_end = abs_start + rel_idx
                                thread_local.preamble_end_pos = abs_preamble_end
                                preamble_end_pos = abs_preamble_end
                            else:
                                logger.warning(
                                    f"Preamble extraction failed: could not locate first TOC element: {id_} {title}"
                                )
                else:
                    logger.warning("Preamble start pattern not found")

            # =========================
            # LEVEL-2 EXTRACTION
            # =========================
            start_pos = preamble_end_pos or toc_end_pos

            enriched = extract_level2_text(full_text, start_pos, toc_json)
            if preamble:
                print(f"Preamble detected, length={len(preamble)}")
                toc_json.append({"article": "Preamble", "text": preamble})
            else:
                logger.warning("Preamble not detected")
            enriched = prune_empty_def_nodes(enriched)

        # end of: if not loaded_from_cache

# -------------------------Termination Extraction Start-------------------------

        termination_result = extract_termination_sections(toc_json)

        if termination_result:
            print(
                f"Termination section identification result: {json.dumps(termination_result, indent=2)}")

        replacement_result = extract_replacement_map_from_preamble(preamble) or {
        }
        replacement_map = replacement_result.get("replacement_map", {}) or {}

        # ------ TRACK 1: Termination Triggers (one or more sections) ------
        all_trigger_clauses = []

        # Support both new array format and old singular format for backward compat
        raw_trigger_sections = termination_result.get(
            "triggers_sections") or []
        if not raw_trigger_sections:
            # Fall back to old singular key if present
            old_single = termination_result.get("triggers_section") or {}
            if old_single.get("section_number"):
                raw_trigger_sections = [old_single]

        if not raw_trigger_sections:
            logger.warning(
                "No termination triggers section(s) identified in TOC")
        else:
            for ts in raw_trigger_sections:
                triggers_sec_no = (ts.get("section_number") or "").strip()
                triggers_sec_title = (ts.get("section_title") or "").strip()
                if not triggers_sec_no:
                    continue

                triggers_text = find_section_text(
                    enriched, triggers_sec_no) or ""
                if not triggers_text:
                    logger.warning(
                        f"No text found for termination triggers section {triggers_sec_no}")
                    continue

                print(
                    f"\nBreaking termination triggers {triggers_sec_no} into individual clauses...")
                trigger_clauses = break_termination_into_clauses(
                    triggers_text, triggers_sec_no, triggers_sec_title)

                if not trigger_clauses:
                    logger.warning(
                        f"Failed to break {triggers_sec_no} into trigger clauses, using full text")
                    trigger_clauses = [{
                        "clause_id": "1",
                        "clause_text": triggers_text,
                        "clause_type": "trigger",
                        "trigger_type": "other"
                    }]

                for clause in trigger_clauses:
                    clause_id = clause.get("clause_id", "")
                    clause_text = clause.get("clause_text", "")
                    clause_type = clause.get("clause_type", "")
                    trigger_type = clause.get("trigger_type", "other")

                    final_text, used_parties = replace_parties_in_text(
                        clause_text, replacement_map)

                    all_trigger_clauses.append({
                        "section_number": triggers_sec_no,
                        "section_title": triggers_sec_title,
                        "clause_id": clause_id,
                        "clause_type": clause_type,
                        "trigger_type": trigger_type,
                        "original_text": clause_text,
                        "processed_text": final_text,
                        "parties_replaced": used_parties
                    })

        # Write triggers output (for embedding pipeline)
        triggers_payload = {
            "document_id": accession,
            "total_clauses": len(all_trigger_clauses),
            "clauses": all_trigger_clauses
        }
        s3_urls = {}
        if pipeline_doc_type:
            from Termination_Embeddings_v1.termination_s3_utils import upload_json as _s3_upload_json
            _, triggers_url = _s3_upload_json(
                triggers_payload, accession, pipeline_doc_type, "triggers_json.json")
            s3_urls["triggers_json"] = triggers_url
            print(
                f"\n✅ Generated {len(all_trigger_clauses)} termination trigger clauses")
            print(f"📄 Triggers uploaded to S3: {triggers_url}")
        else:
            triggers_filename = f"termination_response_{accession}_triggers.json"
            with open(triggers_filename, "w", encoding="utf-8") as f:
                json.dump(triggers_payload, f, indent=2, ensure_ascii=False)
            print(
                f"\n✅ Generated {len(all_trigger_clauses)} termination trigger clauses")
            print(f"📄 Triggers output: {triggers_filename}")

        # ------ TRACK 2: Termination Fees (Section 8.3) ------
        fees_data = {}
        fees_section = termination_result.get("fees_section", {}) or {}
        fees_sec_no = (fees_section.get("section_number") or "").strip()

        if fees_sec_no:
            fees_text = find_section_text(enriched, fees_sec_no) or ""
            if fees_text:
                print(f"\nExtracting termination fees from {fees_sec_no}...")
                fees_data = extract_termination_fees(
                    fees_text, fees_sec_no, accession)
                fees_data = enrich_fees_from_definitions(
                    fees_data, enriched, accession)
                print(f"✅ Fee extraction complete")
                # Log key amounts found
                ctf = fees_data.get("company_termination_fee", {}) or {}
                rtf = fees_data.get("reverse_termination_fee", {}) or {}
                if ctf.get("amount_text"):
                    print(
                        f"   Company termination fee: {ctf.get('amount_text')}")
                if rtf.get("amount_text"):
                    print(
                        f"   Reverse termination fee: {rtf.get('amount_text')}")
                offer = fees_data.get("offer_price_per_share", {}) or {}
                if offer.get("amount_text"):
                    print(
                        f"   Offer price: {offer.get('amount_text')} per share ({offer.get('consideration_type', 'unknown')} consideration)")
            else:
                logger.warning(f"No text found for fees section {fees_sec_no}")
        else:
            logger.warning("No termination fees section identified in TOC")

        # Write fees output
        if pipeline_doc_type:
            from Termination_Embeddings_v1.termination_s3_utils import upload_json as _s3_upload_json
            _, fees_url = _s3_upload_json(
                fees_data, accession, pipeline_doc_type, "fees_json.json")
            s3_urls["fees_json"] = fees_url
            print(f"📄 Fees uploaded to S3: {fees_url}")

            _, raw_url = _s3_upload_json(
                {"clauses": all_trigger_clauses}, accession, pipeline_doc_type, "triggers_raw_json.json")
            s3_urls["triggers_raw_json"] = raw_url
        else:
            fees_filename = f"termination_response_{accession}_fees.json"
            with open(fees_filename, "w", encoding="utf-8") as f:
                json.dump(fees_data, f, indent=2, ensure_ascii=False)
            print(f"📄 Fees output: {fees_filename}")

            with open(cv_filename, "w", encoding="utf-8") as f:
                json.dump({"clauses": all_trigger_clauses},
                          f, indent=2, ensure_ascii=False)

# -------------------------Termination Extraction End-------------------------

        if pipeline_doc_type:
            from Termination_Embeddings_v1.termination_s3_utils import upload_json as _s3_upload_json
            _, full_url = _s3_upload_json(
                enriched, accession, pipeline_doc_type, "full_json.json")
            s3_urls["full_json"] = full_url
            logger.info(f"LEVEL 2 TEXT UPLOADED TO S3: {full_url}")
        else:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(enriched, f, indent=2, ensure_ascii=False)
            logger.info(f"LEVEL 2 TEXT EXTRACTED: {filename}")

        result = {
            "status": "success",
            "url": url,
            "accession": accession,
            "total_trigger_clauses": len(all_trigger_clauses),
            "warnings": log_records,
        }
        if s3_urls:
            result["s3_urls"] = s3_urls
        else:
            result["output_file"] = filename
            result["triggers_file"] = triggers_filename
            result["fees_file"] = fees_filename
            result["output"] = enriched
        return result

    except Exception as e:
        logger.error(f"Level 2 text extraction failed: {e}")
        return {
            "status": "error",
            "url": url,
            "reason": str(e),
            "warnings": log_records
        }

    finally:
        # Detach this handler from the global logger
        logger.removeHandler(list_handler)

        # Only write a log file if there was at least one WARNING/ERROR (standalone CLI only)
        if log_records and not pipeline_doc_type:
            with open(log_txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(log_records))

# ---------------------------
# CLEANER FUNCTION
# ---------------------------


def prune_empty_def_nodes(toc_json: list) -> list:
    """
    Remove nodes where:
      - title matches DEFINITION_KEYWORDS
      - AND text is empty
      - AND no definitions[] exists

    IMPORTANT:
      - Article nodes are removed ONLY if they also have NO sections.
      - Section nodes are removed using the base rule.
    """

    def is_empty_text(node: dict) -> bool:
        return not (node.get("text") or "").strip()

    def has_definitions(node: dict) -> bool:
        defs = node.get("definitions")
        return isinstance(defs, list) and len(defs) > 0

    def should_remove_section(node: dict) -> bool:
        title = str(node.get("title", "")).strip()
        return is_definition_title(title) and is_empty_text(node) and not has_definitions(node)

    def should_remove_article(node: dict) -> bool:
        title = str(node.get("title", "")).strip()
        sections = node.get("sections")

        # If article has sections (even empty text), DO NOT remove at article-level
        has_sections = isinstance(sections, list) and len(sections) > 0

        # Remove article only when:
        # - definition-title
        # - empty text
        # - no definitions
        # - AND no sections
        return (
            is_definition_title(title)
            and is_empty_text(node)
            and not has_definitions(node)
            and not has_sections
        )

    pruned = []

    for art in toc_json:
        # 1) First prune sections inside this article (if any)
        sections = art.get("sections")
        if isinstance(sections, list):
            art["sections"] = [
                sec for sec in sections if not should_remove_section(sec)]

        # 2) Now decide whether to remove the article
        if should_remove_article(art):
            continue

        pruned.append(art)

    return pruned


def is_definition_title(title: str) -> bool:
    if not title:
        return False
    t = title.lower().strip()
    return any(k.lower() in t for k in DEFINITION_KEYWORDS)


def extract_termination_sections(toc_json: list) -> dict:
    """
    Identify termination rights (Section 8.1) and termination fees (Section 8.3) from TOC.
    Returns structured JSON with separate triggers_section and fees_section.
    """

    if not toc_json:
        logger.error("No TOC JSON provided to termination extractor.")
        return {}

    # -----------------------------
    # 1️⃣ Flatten TOC into minimal input
    # -----------------------------
    flattened = []
    for art in toc_json:
        for sec in art.get("sections", []):
            flattened.append({
                "section_number": sec.get("section", "").strip(),
                "section_title": sec.get("title", "").strip()
            })

    if not flattened:
        logger.warning("No sections found in TOC.")
        return {}

    # -----------------------------
    # 2️⃣ Prepare Prompt
    # -----------------------------
    system_prompt = """
You are an expert M&A agreement analyst.
You specialize in identifying termination and fee sections from merger agreement TOC entries.
Return strict JSON only.
"""

    user_prompt = f"""
Goal:
From the provided Table of Contents (TOC), identify the termination triggers and fees sections.

1. TERMINATION TRIGGERS: Identify ALL sections that together define when each party may terminate.
   - Simple structure: one section named "Termination" (e.g. Section 8.1) lists all triggers.
   - Split structure: separate sections per party or per trigger type (e.g. Section 7.01
     "Termination by Mutual Consent", Section 7.02 "Termination by Either Party",
     Section 7.03 "Termination by Company", Section 7.04 "Termination by Parent").
     In this case return ALL of them in triggers_sections as an array.

2. TERMINATION FEES: The section that contains termination fee dollar amounts, break fees,
   reverse termination fees, expense reimbursements, and/or sole remedy language.
   Priority order — use the FIRST match found:
   a) Dedicated fee section: "Termination Fees", "Termination Fees and Expenses",
      "Break-Up Fee", "Fees and Expenses" (if it's in the termination article)
   b) Effect of termination section if it likely contains fees: "Effect of Termination
      and Abandonment", "Effect of Termination" — these often contain fee provisions
      when there is no separate fee section
   c) A standalone "Fees and Expenses" section anywhere in the agreement that covers
      termination fees

Return JSON only in this format:

{{
  "triggers_sections": [
    {{
      "section_number": "",
      "section_title": "",
      "confidence_score": 0
    }}
  ],
  "fees_section": {{
    "section_number": "",
    "section_title": "",
    "confidence_score": 0
  }}
}}

IMPORTANT:
- triggers_sections must be an array (even if only one section)
- If the triggers are split across multiple sections (7.01, 7.02, 7.03...), include ALL of them
- For fees_section: if there is no dedicated fee section, use the "Effect of Termination" section
- If a section is not found, set section_number to null (for fees) or return empty array (for triggers)

TOC Sections:
{json.dumps(flattened, indent=2)}
"""

    # -----------------------------
    # 3️⃣ Call OpenAI
    # -----------------------------
    try:
        response = openai.chat.completions.create(
            model="gpt-5.2-2025-12-11",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        raw = response.choices[0].message.content.strip()

        # Remove markdown blocks if present
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]

        raw = raw.strip()

        # Validate JSON
        parsed = json.loads(raw)
        return parsed

    except Exception as e:
        logger.error(f"Termination section extraction failed: {e}")
        return {}


def find_section_text(enriched_toc_json: list, section_number: str) -> str:
    if not enriched_toc_json or not section_number:
        return ""

    target = section_number.strip()

    for art in enriched_toc_json:
        # Search section level
        for sec in (art.get("sections") or []):
            if str(sec.get("section", "")).strip() == target:
                return (sec.get("text") or "").strip()

    return ""


def replace_definitions_in_text(
    text: str,
    definitions_array: List[Dict],
) -> tuple[str, List[Dict], List[Dict]]:

    original_text = text
    updated_text = text

    used_definitions: List[Dict] = []
    case_mismatch_definitions: List[Dict] = []
    placeholder_mapping: Dict[str, str] = {}

    for def_obj in definitions_array:
        term = def_obj.get("term", "")
        definition = def_obj.get("definition", "")

        if not term or not definition:
            continue

        # Match only whole words so "Effect" does not match inside "Effectiveness"
        word_boundary_pattern = re.compile(r"\b" + re.escape(term) + r"\b")
        matches = word_boundary_pattern.findall(original_text)

        if not matches:
            continue

        placeholder = f"${term}$"

        if all(m == term for m in matches):
            # Exact case match: replace whole-word occurrences only
            updated_text = word_boundary_pattern.sub(placeholder, updated_text)
            used_definitions.append({term: definition})
            placeholder_mapping[placeholder] = definition

        elif term.lower() in original_text.lower():
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            matches = pattern.findall(original_text)
            if matches:
                actual_match = matches[0]
                if actual_match != term:
                    case_mismatch_definitions.append({term: actual_match})

    # ✅ SAFE DEFAULTS
    score_map = {}
    scored_definitions = []

    if used_definitions:
        scoring_input = [
            {"term": list(d.keys())[0], "definition": list(d.values())[0]}
            for d in used_definitions
        ]

        scored_definitions = score_definitions_sequential(scoring_input)

        for sd in scored_definitions or []:
            t = (sd.get("term") or "").strip()
            try:
                s = float(sd.get("score") or 0)
            except Exception:
                s = 0
            if t:
                score_map[t] = s

    threshold = 7.0

    for placeholder, definition in placeholder_mapping.items():
        term = placeholder[1:-1]

        if score_map.get(term, 0) >= threshold:
            print(f"placeholder : {placeholder}")
            updated_text = updated_text.replace(placeholder, definition)
        else:
            updated_text = updated_text.replace(placeholder, term)

    return updated_text, scored_definitions, case_mismatch_definitions


def extract_replacement_map_from_preamble(preamble_text: str) -> dict:
    if not preamble_text or not preamble_text.strip():
        return {}

    system_prompt = (
        "You are an expert M&A legal document normalizer. "
        "Return ONLY valid JSON."
    )

    user_prompt = f"""
You are an expert M&A legal document normalizer.

You are given the PREAMBLE section of a merger or acquisition agreement (Exhibit 2.1).

Your task is to build a deterministic role-based replacement map that will be used to normalize covenant text for embedding similarity analysis.

Goal:
Replace actual legal entity names with standardized legal role nouns.

Instructions:
1. Identify all primary transaction parties from the preamble.
2. Determine each party’s role in the transaction.
3. Build a replacement map that converts every legal entity name and its defined shorthand into a standardized role noun.

Standard Role Nouns (use EXACTLY these values):

Target entity → "the Company"
Acquirer / Buyer / Parent → "Parent"
Merger Sub / Acquisition Sub → "Merger Sub"
Seller (if asset deal) → "Seller"
Purchaser (if asset deal) → "Purchaser"

Rules:
Include full legal name (e.g., "Udemy, Inc.")
Include short name versions (e.g., "Udemy")
Include punctuation variants if obvious
Include defined-term references from the preamble
Do NOT invent aliases not present in the preamble
Do NOT analyze covenants
Do NOT include commentary

Respond ONLY with valid JSON in this format:

{{
  "replacement_map": {{
    "Original Legal Name or Alias 1": "Standard Role Noun",
    "Original Legal Name or Alias 2": "Standard Role Noun"
  }}
}}

No explanation.
No additional text.
Only JSON.

preamble:
{preamble_text}
""".strip()

    try:
        response = openai.chat.completions.create(
            model="gpt-5.2-2025-12-11",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw = response.choices[0].message.content.strip()

        # remove markdown fences if present
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        return json.loads(raw)

    except Exception as e:
        logger.error(f"Replacement-map extraction failed: {e}")
        return {}


def replace_parties_in_text(text: str, replacement_map: dict):
    """
    Replace party terms using replacement_map.
    Returns: (updated_text, used_parties)
    Fixes artifacts like: "the Titanium" -> "the the Company".
    """
    if not text or not replacement_map:
        return text, []

    terms = sorted(
        [t for t in replacement_map.keys() if t and str(t).strip()],
        key=lambda s: len(str(s)),
        reverse=True
    )

    used_parties = []
    out = text

    for term in terms:
        role = replacement_map.get(term)
        if not role:
            continue

        escaped = re.escape(str(term))

        # 1) If role starts with "the ", replace "the <term>" first to avoid "the the ..."
        if str(role).lower().startswith("the "):
            out = re.sub(rf'(?<!\w)the\s+{escaped}(?!\w)', role, out)

        # 2) Then replace standalone "<term>"
        pattern = re.compile(rf'(?<!\w){escaped}(?!\w)')
        if pattern.search(out):
            out = pattern.sub(str(role), out)
            used_parties.append({"term": term, "standard_role": role})

    # 3) Minimal cleanup
    out = re.sub(r"\bthe\s+the\b", "the", out)

    return out, used_parties


def score_definitions_sequential(definitions_array: list) -> list:
    """
    OPTIMIZED: Batch definition scoring with GPT-4o-mini (99% cheaper than sequential GPT-5.2).
    Input:  [{"term": "...", "definition": "..."}, ...]
    Output: [{"term":"","definition":"","score":"","reason":""}, ...]
    """

    if not definitions_array:
        return []

    system_prompt = """
You are a legal analyst optimizing M&A clauses for semantic search and embedding models.
Return strict JSON only.
""".strip()

    # Build the batch prompt with all definitions
    definitions_list = []
    for idx, d in enumerate(definitions_array):
        term = (d.get("term") or "").strip()
        definition = (d.get("definition") or "").strip()
        if term and definition:
            definitions_list.append(
                f"{idx+1}. Term: {term}\n   Definition: {definition}")

    if not definitions_list:
        return [{"term": "", "definition": "", "score": "", "reason": ""} for _ in definitions_array]

    user_prompt = f"""
You are a legal analyst optimizing M&A clauses for semantic search and embedding models.

For each defined term below, evaluate whether replacing it inline would improve semantic completeness for an embedding model WITHOUT creating verbose noise that dilutes meaning.

Score from 0-10:
- 0-3: Simple cross-reference or boilerplate. Replacement adds no semantic value (e.g., "Business Day", "Subsidiary")
- 4-6: Contains some qualifying conditions but term name is self-explanatory. Replacement is optional.
- 7-10: Definition contains critical qualifiers, thresholds, or conditions that materially change interpretation. Replacement is needed for semantic completeness.

Consider:
- Does the term name itself convey sufficient meaning?
- Does the definition add material conditions/thresholds/qualifiers?
- Would embedding the definition help similarity matching or create noise?

Return a JSON array with one object per definition in this exact format:

[
  {{"term":"", "definition":"", "score":"", "reason":""}},
  ...
]

Definitions to score:
{chr(10).join(definitions_list)}

Return ONLY the JSON array. No explanation.
""".strip()

    def _clean_json(raw: str) -> str:
        raw = (raw or "").strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        # Try to find array boundaries
        i, j = raw.find("["), raw.rfind("]")
        return raw[i: j + 1].strip() if i != -1 and j != -1 and j > i else raw

    # Try batched call with GPT-4o-mini
    for attempt in range(3):
        try:
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw_out = (resp.choices[0].message.content or "").strip()
            js = _clean_json(raw_out)
            results = json.loads(js)

            if not isinstance(results, list):
                continue

            # Validate we got the right number of results
            if len(results) != len(definitions_array):
                logger.warning(
                    f"Batched scoring returned {len(results)} results, expected {len(definitions_array)}")
                continue

            # Map results back to original definitions
            out = []
            for idx, d in enumerate(definitions_array):
                orig_term = (d.get("term") or "").strip()
                orig_definition = (d.get("definition") or "").strip()

                if idx < len(results):
                    result = results[idx]
                    out.append({
                        "term": orig_term,
                        "definition": orig_definition,
                        "score": str(result.get("score", "")),
                        "reason": str(result.get("reason", "")),
                    })
                else:
                    out.append(
                        {"term": orig_term, "definition": orig_definition, "score": "", "reason": ""})

            print(
                f"✅ Batched definition scoring: {len(out)} definitions scored in 1 API call")
            return out

        except Exception as e:
            logger.warning(f"Batched scoring attempt {attempt+1} failed: {e}")
            continue

    # Fallback: return empty scores
    logger.error("Batched definition scoring failed after 3 attempts")
    return [
        {"term": d.get("term", ""), "definition": d.get(
            "definition", ""), "score": "", "reason": ""}
        for d in definitions_array
    ]


def break_termination_into_clauses(section_text: str, section_number: str, section_title: str) -> list:
    """
    Use Claude Sonnet 4.6 to break Section 8.1 (termination triggers) into individual clauses.
    Each clause = one distinct termination trigger right.
    Returns: [{"clause_id": "", "clause_text": "", "clause_type": "", "trigger_type": ""}, ...]
    """

    system_prompt = """You are an expert M&A legal analyst. Your task is to split merger agreement termination sections into individual clauses — one clause per distinct termination trigger or right."""

    user_prompt = f"""
Split the following termination section into individual clauses at the SEMANTIC TRIGGER level.
Each distinct termination right or circumstance that allows a party to terminate = one clause.

STEP 1 — IDENTIFY THE STRUCTURAL PATTERN:

PATTERN A — Each top-level letter is a self-contained termination right:
  Example: (a) mutual consent  (b) outside date  (c) regulatory block
  → Each letter = one clause.

PATTERN B — A top-level letter GROUPS multiple related triggers under one party:
  Example: (b) by either party if: (i) outside date passes; (ii) court order blocks deal
  → The wrapper intro letter gets a "preamble" clause; each sub-item = one clause.

STEP 2 — SPLITTING RULES:
1. PATTERN A: each top-level letter (a), (b), (c) = one clause
2. PATTERN B: wrapper intro = preamble clause; each numbered sub-item (i), (ii)... = one clause
3. Do NOT split within a sub-item's own sub-sub-clauses (A)(B)(C) or (1)(2)(3) — those qualify a single trigger
4. If the entire section has no structure, output ONE clause with clause_id "1"

CLAUSE ID NAMING:
- Pattern A: "a", "b", "c"
- Pattern B preamble: "b_preamble"
- Pattern B sub-items: "b_i", "b_ii", "b_iii"

TRIGGER TYPE — classify each clause as one of:
- "mutual_consent": either party can terminate by mutual agreement
- "outside_date": termination if deal hasn't closed by a deadline
- "regulatory_block": court order, injunction, or law permanently blocking the deal
- "shareholder_vote_failure": required stockholder approval not obtained
- "target_fiduciary_out": target board changes recommendation or accepts superior proposal
- "acquirer_fiduciary_out": acquirer board changes recommendation (stock deals only)
- "target_breach": target materially breaches representations, warranties, or covenants
- "acquirer_breach": acquirer materially breaches representations, warranties, or covenants
- "financing_failure": acquirer unable to obtain financing (triggers RTF)
- "preamble": wrapper intro clause (no independent trigger)
- "other": any other termination right not covered above

For each clause provide:
- clause_id: as described above
- clause_text: the COMPLETE text of that clause including internal qualifications
- clause_type: one of ["trigger", "preamble", "general"]
- trigger_type: one of the trigger types listed above

Section Number: {section_number}
Section Title: {section_title}

Termination Text:
{section_text}

Return ONLY the JSON array. No explanation. No additional text.
"""

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16000,
            temperature=0,
            timeout=600.0,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        raw = response.content[0].text.strip()

        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        clauses = json.loads(raw)

        if not isinstance(clauses, list):
            logger.error(
                "Claude returned invalid structure for termination triggers (not a list)")
            return []

        for clause in clauses:
            if not all(k in clause for k in ["clause_id", "clause_text", "clause_type"]):
                logger.error(
                    "Claude returned trigger clause missing required fields")
                return []
            # Ensure trigger_type exists
            if "trigger_type" not in clause:
                clause["trigger_type"] = "other"

        logger.info(
            f"Successfully broke termination section into {len(clauses)} trigger clauses")
        return clauses

    except Exception as e:
        logger.error(f"Failed to break termination section into clauses: {e}")
        return []


def extract_termination_fees(fee_text: str, section_number: str, accession: str) -> dict:
    """
    Use Claude Sonnet 4.6 to extract structured termination fee data from Section 8.3.
    Returns: dict with company_termination_fee, reverse_termination_fee, expense_reimbursement,
             offer_price_per_share, specific_performance_available, etc.
    """

    system_prompt = """You are an expert M&A legal analyst specializing in termination fee provisions.
Extract structured data from termination fee sections. Return strict JSON only."""

    user_prompt = f"""
Extract all termination fee data from the following merger agreement fee section.

Return JSON in EXACTLY this format:

{{
  "company_termination_fee": {{
    "amount_usd": null,
    "amount_text": null,
    "triggers": [],
    "notes": null
  }},
  "reverse_termination_fee": {{
    "amount_usd": null,
    "amount_text": null,
    "triggers": [],
    "notes": null
  }},
  "parent_regulatory_termination_fee": {{
    "amount_usd": null,
    "amount_text": null,
    "triggers": [],
    "notes": null
  }},
  "expense_reimbursement": {{
    "amount_usd": null,
    "amount_text": null,
    "triggers": [],
    "notes": null
  }},
  "offer_price_per_share": {{
    "amount_usd": null,
    "amount_text": null,
    "consideration_type": null
  }},
  "specific_performance_available": null,
  "willful_breach_carveout": null,
  "tail_provision_months": null,
  "sole_remedy_for_acquirer": null,
  "notes": null
}}

FIELD GUIDANCE:
- amount_usd: integer dollar amount (e.g. 450000000 for $450M). null if not present.
- amount_text: the exact text from the agreement (e.g. "$450,000,000" or "$450 million")
- triggers for company_termination_fee: list of trigger types that require company to pay, e.g.:
  ["fiduciary_out", "adverse_recommendation_change", "superior_proposal", "tail_provision", "company_breach"]
- triggers for reverse_termination_fee: list of trigger types that require acquirer/parent to pay, e.g.:
  ["financing_failure", "regulatory_block", "acquirer_breach", "acquirer_walkaway"]
- parent_regulatory_termination_fee: a SEPARATE, higher fee the acquirer pays specifically for regulatory/antitrust failure
  (distinct from the standard reverse_termination_fee). Common in large deals with antitrust risk.
  Only populate if there is an explicitly separate regulatory fee amount. If the RTF covers regulatory too, leave this null.
- offer_price_per_share: the per-share merger consideration (e.g. "$28.00 per share" → amount_usd: 28.0)
- consideration_type: "cash", "stock", "mixed"
- specific_performance_available: true if the agreement allows specific performance as a remedy
- willful_breach_carveout: true if willful breach liability survives payment of the termination fee
- tail_provision_months: integer months for the "tail" period after which company fee is owed if deal done with third party
- sole_remedy_for_acquirer: true if the RTF is explicitly the sole remedy for acquirer in case of financing/regulatory failure

If a field is not present in the text, use null.
If there is no RTF at all, set reverse_termination_fee.amount_usd to null.

Section: {section_number}

Fee Section Text:
{fee_text}

Return ONLY the JSON object. No explanation.
"""

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4000,
            temperature=0,
            timeout=120.0,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        raw = response.content[0].text.strip()

        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        parsed = json.loads(raw)
        parsed["document_id"] = accession
        parsed["section_number"] = section_number
        logger.info(f"Successfully extracted termination fees for {accession}")
        return parsed

    except Exception as e:
        logger.error(f"Failed to extract termination fees: {e}")
        return {"document_id": accession, "section_number": section_number, "error": str(e)}


def enrich_fees_from_definitions(fees_data: dict, enriched: list, accession: str = "") -> dict:
    """
    LLM-based enrichment: for any fee field that has a named term (e.g. 'Company Termination Fee')
    but no dollar amount, look up that term in the Article 1 definitions and use Claude to interpret
    the definition — handling fixed amounts, percentages, formulas, tiered structures, etc.

    Returns fees_data with amount_usd filled where possible, plus amount_type and amount_description
    added for non-fixed structures.
    """
    if not fees_data or not enriched:
        return fees_data

    # Build flat definitions lookup: {term_lower: definition_text}
    defs_lookup = {}
    for art in enriched:
        for d in art.get("definitions", []):
            term = (d.get("term") or "").strip()
            defn = (d.get("definition") or "").strip()
            if term and defn:
                defs_lookup[term.lower()] = defn
        for sec in art.get("sections", []):
            for d in sec.get("definitions", []):
                term = (d.get("term") or "").strip()
                defn = (d.get("definition") or "").strip()
                if term and defn:
                    defs_lookup[term.lower()] = defn

    fee_fields = ["company_termination_fee", "reverse_termination_fee",
                  "parent_regulatory_termination_fee", "expense_reimbursement"]

    # Collect all fields that need enrichment
    to_enrich = []
    for field in fee_fields:
        fee = fees_data.get(field) or {}
        if fee.get("amount_usd"):
            continue  # already resolved
        amount_text = (fee.get("amount_text") or "").strip()
        if not amount_text:
            continue

        # Look up definition — try exact match first, then partial
        # Strip leading articles ("the ", "a ", "an ") for better matching
        amount_text_clean = amount_text.lower().strip()
        for article in ["the ", "a ", "an "]:
            if amount_text_clean.startswith(article):
                amount_text_clean = amount_text_clean[len(article):]
                break
        defn_text = defs_lookup.get(
            amount_text_clean) or defs_lookup.get(amount_text.lower())
        if not defn_text:
            for k, v in defs_lookup.items():
                if amount_text_clean.endswith(k) or k in amount_text_clean:
                    defn_text = v
                    break

        if defn_text:
            to_enrich.append({
                "field": field,
                "term_name": amount_text,
                "definition": defn_text
            })
        else:
            logger.warning(
                f"Definition not found for fee term '{amount_text}' in {accession}")

    if not to_enrich:
        return fees_data

    # Single LLM call to interpret all fee definitions at once
    items_json = json.dumps(to_enrich, indent=2)
    user_prompt = f"""You are analyzing termination fee definitions from a merger agreement.

For each item below, the fee section referenced a named fee term but the dollar amount was defined
elsewhere. I have found the definition. Interpret each definition and extract the fee amount.

Return a JSON array with one object per item:
{{
  "field": "<same field name as input>",
  "amount_usd": <integer dollar amount, or null if not a fixed dollar figure>,
  "amount_type": "<one of: fixed | percentage | formula | tiered | unclear>",
  "amount_description": "<concise description: e.g. '$450,000,000', '3% of transaction equity value',
                          'greater of $X or 3% of deal value', '$15M if regulatory / $12M otherwise'>"
}}

GUIDANCE:
- fixed: a single stated dollar amount → set amount_usd to that integer value
- percentage: stated as X% of something → amount_usd null, describe the percentage
- formula: computed from other variables → amount_usd null, describe the formula
- tiered: different amounts for different trigger scenarios → amount_usd = the PRIMARY (most common) tier,
  describe all tiers in amount_description
- If the definition says "defined elsewhere" or cross-references another section without giving a number,
  set amount_type "unclear" and amount_usd null

Fee definitions to interpret:
{items_json}

Return ONLY the JSON array. No explanation.
"""

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            temperature=0,
            timeout=60.0,
            messages=[{"role": "user", "content": user_prompt}]
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        results = json.loads(raw.strip())

        for r in results:
            field = r.get("field")
            if field and field in fees_data and fees_data[field] is not None:
                if r.get("amount_usd"):
                    fees_data[field]["amount_usd"] = r["amount_usd"]
                fees_data[field]["amount_type"] = r.get(
                    "amount_type", "unclear")
                fees_data[field]["amount_description"] = r.get(
                    "amount_description", "")
                logger.info(
                    f"Definition enrichment for {field}: {r.get('amount_description', '')}")

    except Exception as e:
        logger.error(f"Fee definition enrichment failed for {accession}: {e}")

    return fees_data


# =========================
# ✅ helper: build replace_map from scored defs (score >= 7)
# =========================
def build_replace_map(scored_definitions: list, threshold: float = 7.0) -> dict:
    out = {}
    for d in scored_definitions or []:
        label = (d.get("label") or "").strip()
        means = (d.get("means") or "").strip()
        try:
            score = float(d.get("score") or 0)
        except Exception:
            score = 0.0
        if label and means and score >= threshold:
            out[label] = means
    return out


# =========================
# ✅ helper: replace ONLY selected used defs in text (no placeholders needed)
# =========================
def replace_scored_defs_in_text(text: str, replace_map: dict) -> tuple[str, list]:
    """
    replace_map: {term: means} only for score>=threshold
    Returns: (updated_text, used_replacements)
    """
    if not text or not replace_map:
        return text, []

    out = text
    used = []

    # longest first to avoid partial collisions
    terms = sorted(replace_map.keys(),
                   key=lambda s: len(s.strip()), reverse=True)

    for term in terms:
        means = replace_map.get(term, "")
        if not term or not means:
            continue
        if term in out:
            out = out.replace(term, means)
            used.append({"term": term, "definition": means})

    return out, used


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor

    # Configuration
    MAX_PARALLEL_WORKERS = 5  # Process up to 5 deals simultaneously

    # Load URLs from file if it exists, otherwise use hardcoded list
    urls_file = "urls.txt"
    if os.path.exists(urls_file):
        with open(urls_file, 'r') as f:
            urls_from_file = [
                line.strip() for line in f if line.strip() and not line.startswith('#')]
        if urls_from_file:
            urls = urls_from_file
            print(f"📂 Loaded {len(urls)} URLs from {urls_file}")
        else:
            print(f"⚠️  {urls_file} is empty, using hardcoded URLs")
    else:
        print(f"ℹ️  No {urls_file} found, using hardcoded URLs")

    if not urls:
        logger.error(
            "No URLs to process. Please add URLs to the 'urls' list or create a urls.txt file.")
        exit(1)

    print(f"\n{'='*80}")
    print(f"🚀 Starting termination extraction for {len(urls)} deal(s)")
    print(f"⚡ Parallel workers: {MAX_PARALLEL_WORKERS}")
    print(f"💰 Estimated cost: ${len(urls) * 0.35:.2f} (${0.35:.2f} per deal)")
    print(f"🎯 Output: termination_response_*_triggers.json + termination_response_*_fees.json")
    print(f"{'='*80}\n")

    start_time = time.time()

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        # Submit all URLs to the thread pool
        futures = {executor.submit(worker, url): url for url in urls}

        # Track progress
        completed = 0
        total = len(urls)
        print(f"Total URLs: {total}")

        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            completed += 1

            try:
                future.result()  # This will raise an exception if worker failed
                accession = url.split('/')[-1].split('.')[0]
                print(f"✅ [{completed}/{total}] Completed: {accession}")
            except Exception as e:
                accession = url.split('/')[-1].split('.')[0]
                print(f"❌ [{completed}/{total}] Failed: {accession}")
                logger.error(f"Error processing {url}: {e}", exc_info=True)

    elapsed_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(
        f"✅ All {len(urls)} document(s) processed in {elapsed_time/60:.1f} minutes")
    print(f"📊 Average time per deal: {elapsed_time/len(urls):.1f} seconds")
    print(f"{'='*80}\n")

    logger.info("All documents processed. Check output directory.")
