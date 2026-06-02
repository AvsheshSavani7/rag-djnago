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
    "https://www.sec.gov/Archives/edgar/data/1853513/000119312526056047/d113732dex21.htm"
    # "https://www.sec.gov/Archives/edgar/data/2076163/000149315226005670/ex2-1.htm"

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
# WORKER FUNCTION
# ---------------------------
def worker(url, pipeline_deal_id: str = None, pipeline_accession: str = None,
           pipeline_doc_type: str = None):
    """
    Main entry per URL. When pipeline_doc_type is provided (new flow),
    outputs are uploaded to S3 instead of written locally.
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

    filename = f"openai_response_{accession}_full.json"
    cv_filename = f"openai_response_{accession}_covenants.json"
    log_txt_path = f"openai_response_{accession}.txt"
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
        # FETCH FULL DOCUMENT TEXT
        # =========================
        full_text = None

        for attempt in range(3):
            print(f"[fetch] Attempt {attempt + 1}/3 loading page...")
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(
                        headless=True,
                        args=["--no-sandbox", "--disable-dev-shm-usage"],
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
                                f"SEC blocked via requests (status={response.status_code})")

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
            missing_articles = toc_validation.get("missing_articles", []) or []
            missing_sections = toc_validation.get("missing_sections", []) or []

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

        covenant_result = extract_company_interim_covenants(toc_json)

        if covenant_result:
            print(
                f"Covenant extraction result: {json.dumps(covenant_result, indent=2)}")

        # definitions_array removed — definitions substitution is disabled.
        # original_text is used for embeddings; definitions add deal-specific noise.

        replacement_result = extract_replacement_map_from_preamble(preamble) or {
        }
        replacement_map = replacement_result.get("replacement_map", {}) or {}

        all_clauses = []

        for item in covenant_result.get("matched_sections", []):
            sec_no = item.get("section_number", "").strip()
            sec_title = (item.get("title") or item.get(
                "section_title") or "").strip()
            sec_text = find_section_text(enriched, sec_no) or ""

            if not sec_text:
                logger.warning(f"No text found for section {sec_no}")
                continue

            # 0️⃣ FIRST: Break covenant into individual clauses using Claude Sonnet 4.6
            print(
                f"\nBreaking covenant {sec_no} into individual clauses using Claude Sonnet 4.6...")
            clauses = break_covenant_into_clauses(sec_text, sec_no, sec_title)

            if not clauses:
                logger.warning(
                    f"Failed to break section {sec_no} into clauses, using full text")
                # Fallback: treat entire section as one clause
                clauses = [{
                    "clause_id": "1",
                    "clause_text": sec_text,
                    "clause_type": "obligation"
                }]

            # Process each clause individually
            for clause in clauses:
                clause_id = clause.get("clause_id", "")
                clause_text = clause.get("clause_text", "")
                clause_type = clause.get("clause_type", "")

                # 1️⃣ Definitions substitution DISABLED — original_text is used for embeddings,
                #    definitions add deal-specific noise. processed_text = original_text.
                scored_definitions = []
                updated_text = clause_text

                # 2️⃣ Replace party names
                final_text, used_parties = replace_parties_in_text(
                    updated_text,
                    replacement_map
                )

                # 3️⃣ Build output for this clause
                all_clauses.append({
                    "section_number": sec_no,
                    "section_title": sec_title,
                    "clause_id": clause_id,
                    "clause_type": clause_type,
                    "original_text": clause_text,
                    "processed_text": final_text,
                    "definitions_used": [],
                    "parties_replaced": used_parties
                })

        # Write outputs
        clauses_payload = {
            "document_id": accession,
            "total_clauses": len(all_clauses),
            "clauses": all_clauses
        }
        s3_urls = {}

        if pipeline_doc_type:
            from Covenant_Embeddings_v1.covenant_s3_utils import upload_json as _s3_upload_json

            _, clauses_url = _s3_upload_json(
                clauses_payload, accession, "individual_clauses_json.json")
            s3_urls["individual_clauses_json"] = clauses_url
            print(f"\n✅ Generated {len(all_clauses)} individual clauses")
            print(f"📄 Clauses uploaded to S3: {clauses_url}")

            _, covenants_url = _s3_upload_json(
                {"clauses": all_clauses}, accession, "covenants_json.json")
            s3_urls["covenants_json"] = covenants_url

            _, full_url = _s3_upload_json(
                enriched, accession, "full_json.json")
            s3_urls["full_json"] = full_url
            logger.info(f"LEVEL 2 TEXT UPLOADED TO S3: {full_url}")
        else:
            clauses_filename = f"openai_response_{accession}_individual_clauses.json"
            with open(clauses_filename, "w", encoding="utf-8") as f:
                json.dump(clauses_payload, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Generated {len(all_clauses)} individual clauses")
            print(f"📄 Output: {clauses_filename}")

            with open(cv_filename, "w", encoding="utf-8") as f:
                json.dump({"clauses": all_clauses}, f,
                          indent=2, ensure_ascii=False)

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(enriched, f, indent=2, ensure_ascii=False)
            logger.info(f"LEVEL 2 TEXT EXTRACTED: {filename}")

# -------------------------Convenant Extraction End-------------------------

        result = {
            "status": "success",
            "url": url,
            "accession": accession,
            "total_clauses": len(all_clauses),
            "warnings": log_records,
        }
        if s3_urls:
            result["s3_urls"] = s3_urls
        else:
            result["output_file"] = filename
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


def extract_company_interim_covenants(toc_json: list) -> dict:
    """
    Identify Company interim operating covenant sections from TOC.
    Returns structured JSON result.
    """

    if not toc_json:
        logger.error("No TOC JSON provided to covenant extractor.")
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
You specialize in identifying Company interim operating covenants from TOC entries.
Return strict JSON only.
"""

    user_prompt = f"""
Goal:
From the provided Table of Contents (TOC), identify ONLY the sections that govern the COMPANY’s (Target’s) interim operations between signing and closing.

Company-only definition:
Include sections that impose duties/restrictions on the Company to operate in the ordinary course and restrict actions pending closing.

Exclude:
Parent conduct, regulatory efforts, financing, access, employee matters, no-shop, indemnification, termination, conditions, etc.

Return JSON only in this format:

{{
  "category": "company_interim_operating_covenants",
  "matched_sections": [
    {{
      "section_number": "",
      "section_title": "",
      "match_type": "direct | semantic | weak",
      "confidence_score": 0
    }}
  ]
}}

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
        logger.error(f"Covenant extraction failed: {e}")
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

        with open("scored_definitions.json", "w", encoding="utf-8") as f:
            json.dump(scored_definitions, f, indent=2, ensure_ascii=False)

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


def break_covenant_into_clauses(covenant_text: str, section_number: str, section_title: str) -> list:
    """
    Use Claude Sonnet 4.6 to break a covenant section into individual clauses.
    Cost optimized: Sonnet 4.6 is 40% cheaper than Opus with same quality for this structured task.
    Returns: [{"clause_id": "", "clause_text": "", "clause_type": ""}, ...]
    """

    system_prompt = """You are an expert M&A legal analyst. Your task is to split merger agreement covenant sections into individual clauses at the SEMANTIC OBLIGATION level — one clause per distinct legal obligation or restriction."""

    user_prompt = f"""
Split the following covenant section into individual clauses at the SEMANTIC OBLIGATION level.

STEP 1 — IDENTIFY THE STRUCTURAL PATTERN:

PATTERN A — Each top-level letter covers a distinct, self-contained obligation:
  Example: (a) don't issue securities  (b) don't incur debt  (c) don't pay dividends
  → The letters ARE the semantic units. Split at the letter level.

PATTERN B — A top-level letter INTRODUCES a LIST of distinct sub-obligations:
  Example: (b) The Company shall not do any of the following: (i) sell assets... (ii) merge... (xxix) agree to the foregoing
  → The roman-numeral sub-items ARE the semantic units. Split at the sub-item level.

STEP 2 — SPLITTING RULES:
1. PATTERN A: each top-level letter (a), (b), (c) = one clause
2. PATTERN B: the introductory wrapper letter gets its own "preamble" clause; each named sub-item (i), (ii), (iii)... = one separate clause
3. Affirmative obligation letters (e.g. "Company shall conduct business in ordinary course") = one clause each
4. Carveout or qualification letters at the top level (e.g., SpinCo carveout, closing catchall) = one clause each
5. Do NOT split within a sub-item's own sub-sub-items (A)(B)(C) or (1)(2)(3) — those elaborate a single restriction and stay together
6. If the entire section has no lettered or numbered structure, output ONE clause with clause_id "1"

CLAUSE ID NAMING:
- Pattern A letter clause: "a", "b", "c"...
- Pattern B preamble (the wrapper intro): "b_preamble" (or whichever letter the wrapper is)
- Pattern B sub-items: "b_i", "b_ii", "b_iii"... or "b_1", "b_2"... matching the document's numbering
- Standalone carveout/catchall letter: "c", "d"...

For each clause provide:
- clause_id: as described above
- clause_text: the COMPLETE text of that clause, including all its internal sub-sub-items
- clause_type: one of ["obligation", "prohibition", "preamble", "carveout", "catchall", "general"]

Section Number: {section_number}
Section Title: {section_title}

Covenant Text:
{covenant_text}

EXAMPLE OF PATTERN B (wrapper structure):
Input:
  "(a) Company shall operate in ordinary course.
   (b) Company shall not do any of the following: (i) sell assets; (ii) incur debt; (iii) agree to any of the foregoing.
   (c) SpinCo carveout applies."

Correct output (5 clauses — NOT 3):
[
  {{"clause_id": "a", "clause_text": "Company shall operate in ordinary course.", "clause_type": "obligation"}},
  {{"clause_id": "b_preamble", "clause_text": "Company shall not do any of the following:", "clause_type": "preamble"}},
  {{"clause_id": "b_i", "clause_text": "(i) sell assets", "clause_type": "prohibition"}},
  {{"clause_id": "b_ii", "clause_text": "(ii) incur debt", "clause_type": "prohibition"}},
  {{"clause_id": "b_iii", "clause_text": "(iii) agree to any of the foregoing", "clause_type": "catchall"}},
  {{"clause_id": "c", "clause_text": "SpinCo carveout applies.", "clause_type": "carveout"}}
]

Return ONLY the JSON array. No explanation. No additional text.
"""

    try:
        # Use Sonnet 4.6 for cost optimization (40% cheaper than Opus)
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16000,
            temperature=0,
            timeout=600.0,  # 10 minute timeout
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        raw = response.content[0].text.strip()

        # Remove markdown fences if present
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        # Parse and validate JSON
        clauses = json.loads(raw)

        # Validate structure
        if not isinstance(clauses, list):
            logger.error("Claude Opus returned invalid structure (not a list)")
            return []

        for clause in clauses:
            if not all(k in clause for k in ["clause_id", "clause_text", "clause_type"]):
                logger.error(
                    "Claude Opus returned clause missing required fields")
                return []

        logger.info(f"Successfully broke covenant into {len(clauses)} clauses")
        return clauses

    except Exception as e:
        logger.error(f"Failed to break covenant into clauses: {e}")
        return []


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
    print(f"🚀 Starting covenant extraction for {len(urls)} deal(s)")
    print(f"⚡ Parallel workers: {MAX_PARALLEL_WORKERS}")
    print(f"💰 Estimated cost: ${len(urls) * 0.28:.2f} (${0.28:.2f} per deal)")
    print(f"🎯 Cost optimized: Batched definitions + Sonnet 4.6 (78% cheaper)")
    print(f"{'='*80}\n")

    start_time = time.time()

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        # Submit all URLs to the thread pool
        futures = {executor.submit(worker, url): url for url in urls}

        # Track progress
        completed = 0
        total = len(urls)

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
