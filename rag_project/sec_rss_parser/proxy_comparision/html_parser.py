"""
html_parser.py — HTML parsing into canonical blocks.
"""

import re
from typing import List, Optional

from bs4 import BeautifulSoup, Tag, NavigableString

from .models import Block, CanonicalDocument
from .config import get_form_family
from .sec_fetcher import fetch_html, guess_form_type


def normalize_text(s: str) -> str:
    """Normalize unicode and whitespace."""
    # Smart quotes -> straight quotes
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    # Em/en dashes
    s = s.replace("\u2014", " -- ").replace("\u2013", " - ")
    # Non-breaking spaces and other whitespace
    s = s.replace("\u00a0", " ").replace("\u200b", "")
    # Collapse whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def is_noise_block(text: str) -> bool:
    """Detect noise blocks to drop (page numbers, continued lines, TOC headers)."""
    stripped = text.strip()
    if not stripped:
        return True
    # Standalone page numbers
    if re.match(r"^\d{1,4}$", stripped):
        return True
    # "Table of Contents" repeated headers
    if stripped.lower() in ("table of contents", "index"):
        return True
    # "continued on next page" / "see notes" type lines
    if re.match(r"^(continued|see accompanying notes|page \d+)", stripped, re.I):
        return True
    return False


def serialize_table(table_el: Tag) -> str:
    """Convert an HTML table to a stable markdown-like format."""
    rows = []
    for tr in table_el.find_all("tr"):
        cells = []
        for td in tr.find_all(["td", "th"]):
            cell_text = normalize_text(td.get_text(" ", strip=True))
            cells.append(cell_text)
        if any(c.strip() for c in cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


# Patterns that look like headings but are noise
_HEADING_NOISE_PATTERNS = [
    r"^\(page[s]?\s+[\d\s,and\u2013-]+\)\s*$",      # (page 25)
    r"^\(see page[s]?\s+[\d\s,and\u2013-]+\)\s*$",   # (see page 65)
    r"^\[\s*\]\s*,?\s*\d{4}",                    # [  ], 2025
    r"^\d{1,4}\s*$",                              # standalone numbers
    r"^[A-Z]-\d+\s*$",                           # exhibit refs like A-1
    r"^UNITED STATES$",                           # cover page
    r"^SECURITIES AND EXCHANGE COMMISSION$",
    r"^WASHINGTON",
    r"^SCHEDULE \d",
    r"^Proxy Statement Pursuant",
    r"^Securities Exchange Act",
    r"^\(Name of Registrant",
    r"^\(Amendment No\.",
]


def _is_heading_noise(text: str) -> bool:
    """Check if text matches noise patterns that shouldn't be section headings."""
    stripped = text.strip()
    for pat in _HEADING_NOISE_PATTERNS:
        if re.match(pat, stripped, re.I):
            return True
    return False


def detect_heading_level(el: Tag) -> Optional[int]:
    """Detect heading level from HTML element. Conservative -- only real section headers."""
    tag_name = el.name if hasattr(el, 'name') else ""

    # Explicit heading tags (h1-h6)
    if tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        return int(tag_name[1])

    text = el.get_text(strip=True)
    if not text or len(text) > 150 or len(text) < 4:
        return None

    # Skip noise patterns
    if _is_heading_noise(text):
        return None

    # Check CSS class for heading indicators (e.g., BRDSX_h1, BRDSX_h2, etc.)
    classes = el.get("class", [])
    for cls in (classes or []):
        cls_lower = cls.lower()
        if re.search(r'_h1\b|_head1|heading1', cls_lower):
            return 1
        if re.search(r'_h2\b|_head2|heading2', cls_lower):
            return 2
        if re.search(r'_h3\b|_head3|heading3', cls_lower):
            return 3
        if re.search(r'_h4\b|_head4|heading4', cls_lower):
            return 4

    # Check styling for bold elements
    style = el.get("style", "")
    is_bold = ("font-weight" in style and ("bold" in style or "700" in style))
    has_bold_child = el.find("b") is not None or el.find("strong") is not None

    if not (is_bold or has_bold_child):
        return None

    # All-caps bold text = section heading
    if text.isupper() and len(text) >= 8:
        size_match = re.search(r"font-size:\s*(\d+)", style)
        if size_match and int(size_match.group(1)) >= 14:
            return 1
        return 2

    # Bold text with large font-size
    size_match = re.search(r"font-size:\s*(\d+)", style)
    if size_match:
        size = int(size_match.group(1))
        if size >= 16:
            return 1
        if size >= 13:
            return 2

    return None


def parse_html_to_blocks(html: str) -> List[Block]:
    """Parse SEC filing HTML into canonical blocks."""
    soup = BeautifulSoup(html, "lxml")

    # Remove script and style tags
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    blocks = []
    idx = 0

    # Walk through body elements
    body = soup.find("body") or soup
    for el in body.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                              "li", "table", "span"]):
        # Skip nested elements (only process top-level of each type)
        if el.parent and el.parent.name in ("li", "td", "th"):
            continue

        # Tables
        if el.name == "table":
            table_text = serialize_table(el)
            if table_text.strip() and not is_noise_block(table_text):
                row_count = len(table_text.split("\n"))
                blocks.append(Block(
                    type="table",
                    text=normalize_text(table_text),
                    meta={"row_count": row_count},
                    index=idx,
                ))
                idx += 1
            continue

        # List items
        if el.name == "li":
            text = normalize_text(el.get_text(" ", strip=True))
            if text and not is_noise_block(text):
                blocks.append(Block(type="list_item", text=text, index=idx))
                idx += 1
            continue

        # Headings and paragraphs
        text = normalize_text(el.get_text(" ", strip=True))
        if not text or is_noise_block(text):
            continue

        heading_level = detect_heading_level(el)
        if heading_level is not None:
            blocks.append(Block(
                type="heading",
                text=text,
                meta={"level": heading_level},
                index=idx,
            ))
        else:
            blocks.append(Block(type="paragraph", text=text, index=idx))
        idx += 1

    return blocks


def extract_filing_date_from_html(html: str) -> str:
    """Try to extract the filing date from SEC HTML headers."""
    # Look for common date patterns in the first 5000 chars
    header = html[:5000]
    # Pattern: "Filed as of Date: 2025-02-10"
    m = re.search(r"Filed\s+(?:as of\s+)?Date:\s*(\d{4}-\d{2}-\d{2})", header)
    if m:
        return m.group(1)
    # Pattern: "FILED ON 02/10/2025"
    m = re.search(r"FILED\s+ON\s+(\d{2}/\d{2}/\d{4})", header, re.I)
    if m:
        parts = m.group(1).split("/")
        return f"{parts[2]}-{parts[0]}-{parts[1]}"
    return "unknown"


def ingest_filing(url: str) -> CanonicalDocument:
    """Fetch and parse a filing into a CanonicalDocument (Phase 1 only -- no sections or facts yet)."""
    print(f"    Fetching: {url.split('/')[-1][:50]}...")
    html = fetch_html(url)

    form_type = guess_form_type(url)
    filing_date = extract_filing_date_from_html(html)
    family = get_form_family(form_type)

    print(f"    Parsing blocks...")
    blocks = parse_html_to_blocks(html)
    print(f"    Form: {form_type}, Family: {family}, Blocks: {len(blocks)}")

    return CanonicalDocument(
        form_type=form_type,
        filing_date=filing_date,
        source_url=url,
        doc_type_family=family,
        blocks=blocks,
    )
