"""
extractor.py — Priority fact extraction via LLM.
"""

import json
import re
from typing import Any, List

from anthropic import Anthropic

from .models import CanonicalDocument
from .config import (
    MODEL_STANDARD, MODEL_THINKING, THINKING_BUDGET,
    DATES_EXTRACTION_PROMPT, CONSIDERATION_EXTRACTION_PROMPT,
    FINANCING_EXTRACTION_PROMPT, SH_VOTES_EXTRACTION_PROMPT,
    REGULATORY_EXTRACTION_PROMPT, CLOSING_GUIDANCE_EXTRACTION_PROMPT,
    _SECTION_ID_TO_TOPICS,
)
from .classifier import _get_blocks_by_topic
from .section_mapper import get_section_by_id, get_sections_by_ids


def _call_llm_json(client: Anthropic, prompt: str, use_thinking: bool = False) -> Any:
    """Call LLM and parse JSON response."""
    kwargs = {
        "model": MODEL_THINKING if use_thinking else MODEL_STANDARD,
        "max_tokens": 16000 if use_thinking else 4000,
        "messages": [{"role": "user", "content": prompt}],
    }
    if use_thinking:
        kwargs["thinking"] = {"type": "enabled",
                              "budget_tokens": THINKING_BUDGET}

    response = client.messages.create(**kwargs)

    # Extract text (skip thinking blocks)
    text_parts = [b.text for b in response.content if b.type == "text"]
    raw = "\n".join(text_parts).strip()

    # Strip markdown fences
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        m = re.search(r"[\[{].*[\]}]", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        print(f"    Warning: Failed to parse JSON from LLM response")
        return {}


def _get_section_text_for_extraction(doc: CanonicalDocument,
                                     section_ids: List[str],
                                     max_chars: int = 60000) -> str:
    """Get combined text from relevant sections for extraction.

    Uses a two-layer approach:
    1. Topic-tagged blocks (from classify_blocks) -- content-based, not heading-dependent
    2. Section-based fallback (from build_sections) -- heading-dependent
    """
    parts = []

    # Layer 1: Gather topic-tagged blocks (primary -- works even in mega-sections)
    topics_needed = set()
    for sid in section_ids:
        topics_needed.update(_SECTION_ID_TO_TOPICS.get(sid, []))

    if topics_needed:
        topic_text = _get_blocks_by_topic(
            doc, list(topics_needed), max_chars=max_chars)
        if topic_text.strip():
            parts.append(
                f"[Topic-tagged content: {', '.join(sorted(topics_needed))}]\n{topic_text}")

    # Layer 2: Section-based fallback (still useful when sections are well-mapped)
    sections = get_sections_by_ids(doc, section_ids)
    for fallback_id in ["summary", "questions_and_answers"]:
        if fallback_id not in section_ids:
            s = get_section_by_id(doc, fallback_id)
            if s:
                sections.append(s)

    if sections:
        section_text = "\n\n---\n\n".join(
            f"[Section: {s.raw_title}]\n{s.text}" for s in sections
        )
        # Only add section text if it brings new content (avoid pure duplication)
        if not parts:
            parts.append(section_text)
        else:
            # Add section text up to remaining budget
            remaining = max_chars - sum(len(p) for p in parts)
            if remaining > 5000 and section_text.strip():
                if len(section_text) > remaining:
                    section_text = section_text[:remaining] + "\n[TRUNCATED]"
                parts.append(section_text)

    if not parts:
        # Last resort: use all sections
        text = "\n\n---\n\n".join(
            f"[Section: {s.raw_title}]\n{s.text}" for s in doc.sections
        )
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[TRUNCATED]"
        return text

    combined = "\n\n---\n\n".join(parts)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n[TRUNCATED]"
    return combined


def extract_priority_facts(doc: CanonicalDocument, client: Anthropic) -> None:
    """Extract all priority facts from the document. Modifies doc in place."""
    print(f"    Extracting priority facts...")

    # Dates -- look in summary, vote_info, questions_and_answers
    print(f"      Dates...")
    dates_text = _get_section_text_for_extraction(
        doc, ["summary", "vote_info", "questions_and_answers", "closing_conditions"])
    doc.priority_facts.dates = _call_llm_json(
        client, DATES_EXTRACTION_PROMPT.format(section_text=dates_text))

    # Consideration -- look in consideration_summary, summary, merger_agreement_summary
    print(f"      Consideration...")
    consideration_text = _get_section_text_for_extraction(
        doc, ["consideration_summary", "summary", "merger_agreement_summary"])
    doc.priority_facts.consideration = _call_llm_json(
        client, CONSIDERATION_EXTRACTION_PROMPT.format(section_text=consideration_text))

    # Financing -- look in financing, summary
    print(f"      Financing...")
    financing_text = _get_section_text_for_extraction(
        doc, ["financing", "summary", "merger_agreement_summary"])
    doc.priority_facts.financing = _call_llm_json(
        client, FINANCING_EXTRACTION_PROMPT.format(section_text=financing_text))

    # SH Votes -- look in vote_info, summary
    print(f"      Shareholder votes...")
    votes_text = _get_section_text_for_extraction(
        doc, ["vote_info", "summary", "questions_and_answers"])
    doc.priority_facts.sh_votes = _call_llm_json(
        client, SH_VOTES_EXTRACTION_PROMPT.format(section_text=votes_text))

    # Regulatory -- look in regulatory, summary, closing_conditions
    print(f"      Regulatory...")
    reg_text = _get_section_text_for_extraction(
        doc, ["regulatory", "summary", "closing_conditions"])
    reg_result = _call_llm_json(
        client, REGULATORY_EXTRACTION_PROMPT.format(section_text=reg_text))
    doc.priority_facts.regulatory = reg_result if isinstance(
        reg_result, list) else []

    # Closing Guidance -- look in summary, closing_conditions, questions_and_answers
    print(f"      Closing guidance...")
    closing_text = _get_section_text_for_extraction(
        doc, ["summary", "closing_conditions", "questions_and_answers"])
    doc.priority_facts.closing_guidance = _call_llm_json(
        client, CLOSING_GUIDANCE_EXTRACTION_PROMPT.format(section_text=closing_text))

    print(f"    Priority facts extracted.")
