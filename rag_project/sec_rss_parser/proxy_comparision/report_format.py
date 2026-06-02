"""
Shared parsing helpers for proxy comparison reports (DOCX and HTML).
"""

import re

from .config import _PRESERVE_UPPER


def smart_title(text: str) -> str:
    """Title-case text but preserve known acronyms (HSR, SH, etc.)."""
    words = text.split()
    result = []
    for w in words:
        if w.upper() in _PRESERVE_UPPER:
            result.append(w.upper())
        elif w == "&":
            result.append("&")
        else:
            result.append(w.capitalize())
    return " ".join(result)


def parse_summary_sections(text: str):
    """Parse summary/change text into (opening_paragraph, [(header, content), ...])."""
    lines = text.split("\n")
    opening_lines = []
    sections = []
    current_header = None
    current_lines = []
    found_first_header = False

    skip_headers = {"RULES"}

    for line in lines:
        stripped = line.strip()
        is_header = (
            stripped
            and re.match(r"^[A-Z][A-Z &/]+$", stripped)
            and len(stripped) >= 3
            and stripped not in skip_headers
        )
        if not is_header and stripped:
            colon_match = re.match(r"^([A-Z][A-Z &/]+):?$", stripped)
            if colon_match and colon_match.group(1) not in skip_headers:
                candidate = colon_match.group(1)
                if len(candidate) >= 3 and candidate == candidate.upper():
                    is_header = True
                    stripped = candidate

        if is_header:
            if not found_first_header:
                found_first_header = True
            if current_header is not None:
                sections.append((current_header, "\n".join(current_lines).strip()))
            current_header = stripped
            current_lines = []
        elif not found_first_header:
            if stripped:
                opening_lines.append(stripped)
        else:
            current_lines.append(line.rstrip())

    if current_header is not None:
        sections.append((current_header, "\n".join(current_lines).strip()))

    opening = " ".join(opening_lines)
    return opening, sections
