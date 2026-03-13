"""
section_mapper.py — Section detection and LLM mapping to canonical IDs.
"""

import re
import json
from typing import List, Dict, Tuple, Optional

from anthropic import Anthropic

from .models import Block, CanonicalSection, CanonicalDocument
from .config import SECTION_MAPPING_PROMPT, MODEL_STANDARD


def _is_section_heading_noise(title: str) -> bool:
    """Filter headings that shouldn't become sections."""
    t = title.strip()
    # Too short to be a real section
    if len(t) < 5:
        return True
    # Page references
    if re.match(r"^\(page", t, re.I):
        return True
    if re.match(r"^\(see page", t, re.I):
        return True
    # Cover page items
    cover_noise = [
        "united states", "securities and exchange commission", "washington",
        "schedule 14a", "schedule 14d", "form s-4", "form f-4",
        "proxy statement pursuant", "securities exchange act",
        "name of registrant", "amendment no",
    ]
    t_lower = t.lower()
    for cn in cover_noise:
        if t_lower.startswith(cn):
            return True
    # Dates that are just standalone dates
    if re.match(r"^\[?\s*\]?\s*,?\s*\d{4}\s*$", t):
        return True
    # Just a company name repeated (common in SEC filings)
    if re.match(r"^[A-Z][a-z]+(\s+[A-Z][a-z]+){0,2},?\s*(Inc\.?|Corp\.?|LLC|L\.P\.)?\s*$", t):
        # This could be a real heading like "Dayforce, Inc." used as section start
        # Only filter if it's very short (< 3 words)
        if len(t.split()) <= 3:
            return True
    return False


def detect_sections_from_blocks(blocks: List[Block]) -> List[Tuple[int, str]]:
    """Find section boundaries from heading blocks. Returns [(block_index, title)]."""
    sections = []
    for i, block in enumerate(blocks):
        if block.type == "heading":
            title = block.text.strip()
            if title and not _is_section_heading_noise(title):
                sections.append((i, title))

    # Fallback: detect "Item X" patterns (SC TO / 14D-9)
    if len(sections) < 5:
        for i, block in enumerate(blocks):
            if block.type in ("paragraph", "heading"):
                m = re.match(r"^Item\s+\d+[A-Za-z]?\.\s+", block.text)
                if m:
                    if not any(idx == i for idx, _ in sections):
                        sections.append((i, block.text.strip()))

    sections.sort(key=lambda x: x[0])
    return sections


def map_sections_with_llm(client: Anthropic, headings: List[str]) -> Dict[str, str]:
    """Use LLM to map raw section titles to canonical IDs. Returns {title: section_id}.

    Batches large heading lists to avoid token overflow.
    """
    if not headings:
        return {}

    # Process in batches of 80 headings max
    BATCH_SIZE = 80
    all_mappings = {}

    for batch_start in range(0, len(headings), BATCH_SIZE):
        batch = headings[batch_start:batch_start + BATCH_SIZE]
        headings_text = "\n".join(f"- {h}" for h in batch)

        response = client.messages.create(
            model=MODEL_STANDARD,
            max_tokens=8000,
            messages=[{
                "role": "user",
                "content": SECTION_MAPPING_PROMPT.format(headings_list=headings_text)
            }]
        )

        raw_text = response.content[0].text.strip()

        try:
            # Strip markdown code fences if present
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)
            mappings = json.loads(raw_text)
            for m in mappings:
                all_mappings[m["heading"]] = m["section_id"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    Warning: Failed to parse section mapping batch: {e}")
            # Fallback: try to use keyword matching for this batch
            for h in batch:
                all_mappings[h] = _keyword_section_match(h)

    # Safety net: keyword-override for critical sections the LLM may mismap
    for heading, sid in list(all_mappings.items()):
        kw_match = _keyword_section_match(heading)
        # Positive override: keyword match is definitive, override LLM
        if not kw_match.startswith("other") and sid != kw_match:
            print(f"    Section override: '{heading[:60]}' LLM='{sid}' -> keyword='{kw_match}'")
            all_mappings[heading] = kw_match
        # Negative override for "background" only: LLM sometimes maps unrelated
        # headings (e.g., "Representations and Warranties") to background
        elif sid == "background" and "background" not in heading.lower():
            fallback = f"other:{re.sub(r'[^a-z0-9]+', '_', heading.lower())[:40]}"
            print(f"    Section reject: '{heading[:60]}' LLM='background' rejected -> '{fallback}'")
            all_mappings[heading] = fallback

    return all_mappings


def _keyword_section_match(title: str) -> str:
    """Fallback keyword-based section matching when LLM fails."""
    t = title.lower()
    if "background" in t:
        return "background"
    if "consideration" in t or "merger consideration" in t or "exchange ratio" in t:
        return "consideration_summary"
    if "vote" in t or "record date" in t or "special meeting" in t or "stockholder" in t:
        return "vote_info"
    if "regulat" in t or "antitrust" in t or "hsr" in t or "cfius" in t:
        return "regulatory"
    if "financ" in t or "source of funds" in t or "commitment" in t:
        return "financing"
    if "condition" in t and ("closing" in t or "merger" in t or "offer" in t):
        return "closing_conditions"
    if "terminat" in t or "break" in t or "go-shop" in t or "no-shop" in t:
        return "termination"
    if "risk factor" in t:
        return "risk_factors"
    if "interest" in t and ("director" in t or "officer" in t or "conflict" in t):
        return "interests_conflicts"
    if "projection" in t or "forecast" in t or "prospective" in t:
        return "projections"
    if "appraisal" in t or "dissent" in t:
        return "appraisal_rights"
    if "merger agreement" in t and "summary" in t:
        return "merger_agreement_summary"
    if t.startswith("summary") or "questions and answers" in t:
        return "summary"
    return f"other:{re.sub(r'[^a-z0-9]+', '_', t)[:40]}"


def build_sections(doc: CanonicalDocument, client: Anthropic) -> None:
    """Detect sections and map them to canonical IDs. Modifies doc in place."""
    print(f"    Detecting sections...")
    raw_sections = detect_sections_from_blocks(doc.blocks)
    print(f"    Found {len(raw_sections)} section headings")

    if not raw_sections:
        # Fallback: treat entire document as one section
        doc.sections = [CanonicalSection(
            section_id="other:full_document",
            raw_title="Full Document",
            blocks=doc.blocks,
            start_block_idx=0,
            end_block_idx=len(doc.blocks),
        )]
        return

    # Map headings via LLM
    headings = [title for _, title in raw_sections]
    print(f"    Mapping sections via LLM...")
    mapping = map_sections_with_llm(client, headings)

    # Build section objects
    sections = []
    for i, (block_idx, title) in enumerate(raw_sections):
        # Determine end boundary
        if i + 1 < len(raw_sections):
            end_idx = raw_sections[i + 1][0]
        else:
            end_idx = len(doc.blocks)

        section_id = mapping.get(title, f"other:{re.sub(r'[^a-z0-9]+', '_', title.lower())[:40]}")
        section_blocks = doc.blocks[block_idx:end_idx]

        sections.append(CanonicalSection(
            section_id=section_id,
            raw_title=title,
            blocks=section_blocks,
            start_block_idx=block_idx,
            end_block_idx=end_idx,
        ))

    doc.sections = sections
    print(f"    Mapped {len(sections)} sections to canonical IDs")


def get_section_by_id(doc: CanonicalDocument, section_id: str) -> Optional[CanonicalSection]:
    """Get a section by its canonical ID. Returns the first match."""
    for s in doc.sections:
        if s.section_id == section_id:
            return s
    return None


def get_sections_by_ids(doc: CanonicalDocument, section_ids: List[str]) -> List[CanonicalSection]:
    """Get all sections matching any of the given IDs."""
    return [s for s in doc.sections if s.section_id in section_ids]
