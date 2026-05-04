from playwright.sync_api import sync_playwright
import requests
import os
import re
import time
import logging
import json
import openai
import threading


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

# Load API key from environment (never hardcode in production)
_openai_key = os.environ.get("OPENAI_API_KEY")
if _openai_key:
    openai.api_key = _openai_key
else:
    # Optional: load from .env file for local dev (pip install python-dotenv)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        openai.api_key = os.environ.get("OPENAI_API_KEY") or ""
    except ImportError:
        openai.api_key = ""

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
    # "https://www.sec.gov/Archives/edgar/data/835324/000143774926002223/ex_912528.htm",
    "https://www.sec.gov/Archives/edgar/data/1823406/000119312526134773/d138375dex21.htm",
    # "https://www.sec.gov/Archives/edgar/data/1661460/000119312524265591/d881793dex21.htm",
]

# Output paths: use env vars or fall back to script-relative dirs (works on any OS)
_script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.environ.get(
    "MNA_OUTPUT_DIR") or os.path.join(_script_dir, "json")
output_logdir = os.environ.get(
    "MNA_LOG_DIR") or os.path.join(_script_dir, "logs")
SEC_USER_AGENT = os.environ.get(
    "MNA_SEC_USER_AGENT",
    "Mozilla/5.0 (compatible; RAG_BE SEC scraper; contact: ashish.kachadiya@teqnodux.com)",
)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_logdir, exist_ok=True)

# ---------------------------
# TOC CLEANUP + END POSITION
# ---------------------------


def clean_metadata_and_toc(text, url):
    for pattern in metadata_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    toc_start_match = re.search(
        TOC_HEADER_PATTERN, text[0:1000], re.IGNORECASE)
    if not toc_start_match:
        logger.warning(
            f"'Table Of Contents' word pattern not found in {url} , trying default start at pos 150")

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


def text_clean(full_text: str, sequence: list, output_dir: str) -> str:
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
            rf"(?:{re.escape(id_)}|(?:Section|ARTICLE)\s*{re.escape(id_.split()[-1])}(?:\s*\.)?|{re.escape(id_.split()[-1])}(?:\s*\.)?)"
            rf"(?:\s*[-–—]{{1,2}}\s*|(?:[ \t]+|\n)+){re.escape(title).replace(r'\ ', r'(?:[ \t]+|\n)+').replace(r'\-', r'[-–—]{1,2}')}",
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
    zone_txt_path = os.path.join(output_dir, "zone1.txt")
    with open(zone_txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    # text = re.sub(r"\s+([’‘'“”.,;:!?])", r"\1", text)       # manage space before punctuation
    text = re.sub(r"[ \t]+([’‘'“”,;:!?])", r"\1", text)
    zone_txt_path = os.path.join(output_dir, "zone1.txt")
    with open(zone_txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    text = re.sub(r"([’‘'])\s+([a-z])(?![a-z])", r"\1\2", text)

    # Save to zone.txt
    zone_txt_path = os.path.join(output_dir, "zone.txt")
    with open(zone_txt_path, "w", encoding="utf-8") as f:
        f.write(text)

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
            sequence.append(
                {"type": "section", "id": sec_id, "title": sec_title})

    # Save to zone_seq.txt
    zone_txt_path = os.path.join(output_dir, "zone_seq.txt")
    with open(zone_txt_path, "w", encoding="utf-8") as f:
        json.dump(sequence, f, ensure_ascii=False, indent=2)

    return sequence


def partition_zone_text(zone_text: str, sequence_array: list, output_dir: str, toc_json: list, toc_end_pos) -> list:
    if not zone_text or not sequence_array:
        logger.error("No zone text or sequence array provided.")
        return []
    print(
        f"[partition] Zone text length: {len(zone_text):,}, sequence items: {len(sequence_array)}")

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
            rf"(?:{re.escape(id_)}|(?:Section|ARTICLE)\s*{re.escape(id_.split()[-1])}(?:\s*\.)?|{re.escape(id_.split()[-1])}(?:\s*\.)?)"
            rf"(?:\s*[-–—]{{1,2}}\s*|\s*){title_rx.replace(r'\-', r'[-–—]{1,2}')}\s*[:.\-–—]*",
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
        print("[partition] ERROR: No section/article patterns matched")
        return []

    print(f"[partition] Matched {len(positions)} section/article headers")
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
                print(
                    f"[definitions] Processing definitions section: id_={id_}, title={title}")
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
        print("[definitions] Extracted from inner (TOC) definitions section")
    else:
        print("let's check outside definitions extraction.")
        defi, is_true = outside_definitions_extraction(toc_end_pos)
        if is_true and defi and defi.get("definitions"):
            print(
                f"[definitions] Outside extraction SUCCESS ({len(defi['definitions'])} terms)")
        else:
            logger.warning("definitions extraction FAILED")
            print("[definitions] Outside extraction failed or empty")
            # preamble_entry = {"article": "Definitions",
            #               "definitions": defi["definitions"]}
        # Safe access: defi may be None or missing "definitions" on failure
        preamble_entry = {"article": "Definitions",
                          "definitions": defi["definitions"]}
        toc_json.append(preamble_entry)

    # --- 3️⃣ Save to zonewise.txt for demo ---
    zonewise_path = os.path.join(output_dir, "zonewise.txt")
    with open(zonewise_path, "w", encoding="utf-8") as f:
        for idx, r in enumerate(results, 1):
            f.write(f"==== ZONE {idx}: {r['id']} - {r['title']} ====\n")
            f.write(r['text'])
            f.write("\n\n")

    print(f"[partition] Wrote {len(results)} zones to zonewise.txt")
    return toc_json  # <-- Return updated toc_json


def process_definitions_section(text_part: str):
    text1 = (text_part or "").replace(
        "“", '"').replace("”", '"').replace("’’", '"')

    # normalize newlines (keep them!)
    text1 = text1.replace("\r\n", "\n").replace("\r", "\n")

    # remove non-printables but KEEP \n
    text1 = "".join(c if (c.isprintable() or c == "\n")
                    else " " for c in text1)
    def_txt_path = os.path.join(output_dir, "def_text.txt")
    with open(def_txt_path, "w", encoding="utf-8") as f:
        f.write(text1)
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
        r'with respect to [A-Z][A-Za-z0-9&,\- ]+,\s*mean(?:s)?|'
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
        rf'(?:(?:The\s+term)\s*)?'
        rf'(?P<head>"[^"]+"(?:\s*(?:,|and|or)\s*"[^"]+")*)\s*'
        rf'(?:\([^)]*\))?\s*(?:or similar terms)?\s*(?P<leadin>{leadin})\s*[,;:]?\s*',
        re.I)

    start_heading_quoted = re.compile(
        rf'^(?P<label>[A-Za-z0-9&\-., ]+)\.\s*'
        rf'(?:(?:The\s+term)\s*)?'
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

            for t in current_terms:
                definitions.append({"term": t, "definition": d})
        current_terms, current_def = [], ""

    for idx, line in enumerate(arr):
        if header_pat.match(line):
            continue
        if line.strip().isdigit():   # single number line that is page number will be skipped
            continue

        if ((TERM_INDEX_INTRO_PATTERN.search(line) or re.search(r'\bTerm\b\s+\bSection\b', line, re.I)) and idx > 5 and (current_terms or definitions)):

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
    zone_txt_path = os.path.join(output_dir, "zone_norm_before.txt")
    with open(zone_txt_path, "w", encoding="utf-8") as f:
        f.write(doc_text)
    for pattern in metadata_patterns:
        text = re.sub(pattern, "", doc_text,
                      flags=re.IGNORECASE | re.MULTILINE)
    zone_txt_path = os.path.join(output_dir, "zone_norm_after.txt")
    with open(zone_txt_path, "w", encoding="utf-8") as f:
        f.write(text)
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

    if m_start:
        print("[definitions] OUTSIDE_DEF_HEADER matched:",
              repr(m_start.group(0)))
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

    zone_text = text_clean(full_text[toc_end_pos:], sequence_array, output_dir)

    newjson = partition_zone_text(
        zone_text, sequence_array, output_dir, toc_json, toc_end_pos)
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
# MOCK OPENAI CALL (READ SAMPLE.JSON)
# ---------------------------
def mock_openai_api_call(filename):
    sample_path = os.path.join(output_dir, filename)
    if not os.path.exists(sample_path):
        logger.error(f"{filename} not found at {sample_path}")
        return None

    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded sample.json from {sample_path}")
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to load sample.json: {e}")
        return None

# ---------------------------
# MOCK RAW TOC READING (READ raw_toc.txt)
# ---------------------------


def raw_toc_text_file():
    """
    Temporary function to read pre-saved TOC text from raw_toc.txt
    instead of reformatting or regenerating it.
    """
    raw_toc_path = os.path.join(output_dir, "raw_toc.txt")

    if not os.path.exists(raw_toc_path):
        logger.error(f"raw_toc.txt not found at {raw_toc_path}")
        return None

    try:
        with open(raw_toc_path, "r", encoding="utf-8") as f:
            raw_text = f.read().strip()
        logger.info(
            f"Loaded TOC text from {raw_toc_path}, length={len(raw_text)}")
        return raw_text
    except Exception as e:
        logger.error(f"Failed to read raw_toc.txt: {e}")
        return None


# ---------------------------
# WORKER FUNCTION
# ---------------------------
def worker(url):
    accession = url.split('/')[-1].split('.')[0]
    print(f"[worker] Starting for URL: {url}")
    print(f"[worker] Accession: {accession}")

    filename = os.path.join(output_dir, f"openai_response_{accession}.json")
    log_txt_path = os.path.join(
        output_logdir, f"openai_response_{accession}.txt")
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
                        args=[
                            "--no-sandbox",
                            "--disable-dev-shm-usage",
                        ],
                    )
                    try:
                        context = browser.new_context()
                        page = context.new_page()

                        # Download HTML with `requests` (SEC tends to allow this more reliably),
                        # then render locally (no Playwright network request => avoids headless blocks).
                        headers = {
                            "User-Agent": SEC_USER_AGENT,
                            "Accept-Encoding": "gzip, deflate",
                            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        }
                        response = requests.get(
                            url, headers=headers, timeout=60)
                        html_content = response.text or ""

                        # Detect SEC bot-block pages.
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
                full_text_path = os.path.join(output_dir, "full_text.txt")
                with open(full_text_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                print(
                    f"[fetch] OK — document length: {len(full_text):,} chars")
                break

            except Exception as e:
                logger.warning(f"Retry {attempt+1}: {e}")
                print(f"[fetch] Retry {attempt+1} failed: {e}")
                time.sleep(2)

        if not full_text:
            print("[worker] ERROR: Failed to load valid document")
            return {
                "status": "error",
                "url": url,
                "reason": "Failed to load valid document",
                "warnings": log_records
            }
        print("[TOC] Extracting table of contents...")
        full_text_path = os.path.join(output_dir, "zone_norm_1.txt")
        with open(full_text_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"document fetch commplete, start TOC extraction...")
        # =========================
        # TOC EXTRACTION
        # =========================

        toc_raw, toc_end_pos = clean_metadata_and_toc(full_text, url)
        print(
            f"[TOC] Raw TOC length: {len(toc_raw) if toc_raw else 0} chars, toc_end_pos: {toc_end_pos}")
        if not toc_raw:
            logger.warning("TOC not detected")
            toc_end_pos = 0

        rawToc_path = os.path.join(output_dir, "raw_toc1.txt")
        with open(rawToc_path, "w", encoding="utf-8") as f:
            f.write(toc_raw)
        formatted_toc = format_toc_text(toc_raw) if toc_raw else None
        rawToc_path = os.path.join(output_dir, "raw_toc.txt")
        with open(rawToc_path, "w", encoding="utf-8") as f:
            f.write(formatted_toc)
        if not formatted_toc:
            print("[worker] ERROR: TOC formatting failed")
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
            print("[worker] ERROR: TOC → JSON (OpenAI) failed")
            return {
                "status": "error",
                "url": url,
                "reason": "TOC → JSON failed",
                "warnings": log_records
            }
        print("[OpenAI] TOC JSON received successfully")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(api_response)

        toc_json = json.loads(api_response)
        print(f"[TOC] Parsed {len(toc_json)} top-level articles")

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
                print(f"[validate] Missing articles: {missing_articles}")

            if missing_sections:
                logger.warning("TOC missing sections: " +
                               ", ".join(missing_sections))
                print(f"[validate] Missing sections: {missing_sections}")
        except Exception as e:
            logger.warning(f"TOC validation failed (non-fatal): {e}")
            print(f"[validate] Validation error (non-fatal): {e}")

        # =========================
        # PREAMBLE DETECTION
        # =========================
        print("[preamble] Detecting preamble...")
        preamble = None
        preamble_end_pos = None

        if toc_end_pos:
            search_from = full_text[toc_end_pos:]
            m_start = re.search(START_PATTERN, search_from, re.IGNORECASE)
            if m_start:
                abs_start = toc_end_pos + m_start.start()
                m_end = re.search(
                    END_PATTERN, full_text[abs_start:abs_start+5000], re.IGNORECASE)
                if m_end:
                    abs_end = abs_start + m_end.end()
                    preamble = full_text[abs_start:abs_end].strip()
                    thread_local.preamble_end_pos = abs_end
                    preamble_end_pos = abs_end
                    print(
                        f"[preamble] Found (pattern), end pos: {abs_end}, length: {len(preamble)}")
                else:
                    print(
                        "[preamble] End pattern not found, attempting alternative extraction")
                    sequence_array = array_extraction(toc_json)
                    if sequence_array:
                        first_element = sequence_array[0]

                        id_ = str(first_element.get("id", "")).strip()
                        title = str(first_element.get("title", "")).strip()

                        # Work only inside the slice starting at abs_start
                        search_slice = full_text[abs_start:]

                        # Matches: "1 Title", "1. Title", "1\tTitle", "1\nTitle"
                        rx = re.compile(
                            rf"(?:{re.escape(id_)}|(?:Section|ARTICLE)\s*{re.escape(id_.split()[-1])}(?:\s*\.)?|{re.escape(id_.split()[-1])}(?:\s*\.)?)"
                            rf"(?:\s*[-–—]{{1,2}}\s*|(?:[ \t]+|\n)+){re.escape(title).replace(r'\ ', r'(?:[ \t]+|\n)+').replace(r'\-', r'[-–—]{1,2}')}",
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
                            print(
                                f"[preamble] Found (fallback), end pos: {abs_preamble_end}, length: {len(preamble)}")
                        else:
                            logger.warning(
                                f"Preamble extraction failed: could not locate first TOC element: {id_} {title}"
                            )
                            print(
                                f"[preamble] Fallback failed: first TOC element not found")
            else:
                logger.warning("Preamble start pattern not found")
                print("[preamble] Start pattern not found")

        # =========================
        # LEVEL-2 EXTRACTION
        # =========================
        start_pos = preamble_end_pos or toc_end_pos
        print(f"[level2] Extracting section text from position {start_pos}...")

        enriched = extract_level2_text(full_text, start_pos, toc_json)
        if preamble:
            print(f"[preamble] Added to output, length={len(preamble)}")
            toc_json.append({"article": "Preamble", "text": preamble})
        else:
            logger.warning("Preamble not detected")
        print("[prune] Removing empty definition nodes...")
        enriched = prune_empty_def_nodes(enriched)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(enriched, f, indent=2, ensure_ascii=False)
        logger.info(f"LEVEL 2 TEXT SAVED: {filename}")
        print(f"[worker] Output saved to: {filename}")

        logger.info(f"Log_records: {log_records}")

        # ---------------------------
        # Red-flag validation
        # ---------------------------
        # We don't treat every WARNING as a hard failure, but we do prevent
        # returning "success" when the extraction is clearly incomplete.
        pattern_not_found_count = sum(
            1 for msg in log_records if "Pattern not found for" in msg
        )
        preamble_not_detected = any(
            "Preamble not detected" in msg for msg in log_records
        )

        if preamble_not_detected or pattern_not_found_count > 5:
            red_flags = []
            if preamble_not_detected:
                red_flags.append("preamble_not_detected")
            if pattern_not_found_count > 5:
                red_flags.append(
                    f"pattern_not_found_count={pattern_not_found_count} (>5)"
                )

            reason = "Red-flag validation failed: " + ", ".join(red_flags)
            return {
                "status": "error",
                "url": url,
                "accession": accession,
                "output_file": filename,
                "reason": reason,
                "warnings": log_records,
                "output": enriched
            }

        return {
            "status": "success",
            "url": url,
            "accession": accession,
            "output_file": filename,
            "warnings": log_records,
            "output": enriched
        }

    except Exception as e:
        logger.error(f"Level 2 text extraction failed: {e}")
        print(f"[worker] ERROR: {e}")
        return {
            "status": "error",
            "url": url,
            "reason": str(e),
            "warnings": log_records
        }

    finally:
        # Detach this handler from the global logger
        logger.removeHandler(list_handler)

        # Only write a log file if there was at least one WARNING/ERROR
        if log_records:
            with open(log_txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(log_records))


def parse_ex21(url: str):
    """
    Parse an SEC ex-2.1 document from URL. Intended for use by deal_creation and other callers
    that expect a dict with key "articles" (list of article/section content).

    Returns:
        On success: dict with key "articles" (list of enriched article/section nodes).
        On failure: None.
    """
    result = worker(url)
    if result.get("status") == "success" and result.get("output") is not None:
        return {"articles": result["output"]}
    return None


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

# ---------------------------
# MAIN
# ---------------------------


if __name__ == "__main__":
    print(f"[main] Processing {len(urls)} URL(s)...")
    for idx, url in enumerate(urls, start=1):
        print(f"\n[main] --- Document {idx}/{len(urls)} ---")
        worker(url)
    print("\n[main] All documents processed. Check output directory.")
    logger.info("All documents processed. Check output directory.")
