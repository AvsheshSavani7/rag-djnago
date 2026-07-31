"""Shared utility for unique file naming across SEC summarizer modules."""
import re
import hashlib


def filing_uid(url) -> str:
    """Extract a short unique ID from an SEC filing URL.

    Uses the accession number from the URL path, which is unique per filing.
    Example: .../edgar/data/1434868/000162828025057944/... → '25057944'
    """
    if isinstance(url, list):
        url = url[0]

    # SEC URLs: /Archives/edgar/data/{CIK}/{accession}/{filename}
    m = re.search(r'/Archives/edgar/data/\d+/(\d+)/', url)
    if m:
        acc = m.group(1).lstrip('0') or '0'
        return acc[-8:]
    # Fallback: hash of URL
    return hashlib.md5(url.encode()).hexdigest()[:8]


def sanitize_filename_part(value, default="UNKNOWN") -> str:
    """Sanitize a value for use in S3/docx filenames.

    Handles None/empty from LLM JSON (dict.get default only applies when the
    key is missing, not when the value is explicitly null).
    """
    if value is None or value == "":
        value = default
    return re.sub(r"[^\w\-\.]", "_", str(value))


def sanitize_date_part(value, default="unknown-date") -> str:
    """Sanitize a filing date for filenames. Handles None from LLM."""
    if value is None or value == "":
        return default
    return str(value).replace("/", "-")
