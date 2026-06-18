"""
report_writer.py — Summary and change report generation.
"""

import re
import json
from typing import List, Tuple
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from anthropic import Anthropic

from .models import CanonicalDocument, ChangeEvent
from .config import (
    MODEL_STANDARD, MODEL_OPUS,
    SECTION_CONFIGS, SECTION_PROMPTS, SECTION_ORDER, SECTION_HEADERS,
    _TOPIC_TO_SECTION_IDS, CHANGE_OPENING_PROMPT, _PRESERVE_UPPER,
)
from .classifier import _get_blocks_by_topic
from .extractor import _get_section_text_for_extraction


def _generate_section(client: Anthropic, config: dict, doc: CanonicalDocument,
                      ticker: str, target: str, acquirer: str, facts_json: str) -> Tuple[str, str, str]:
    """Generate one output section using focused topic-tagged content.

    Returns (key, llm_output, source_text).
    """
    # Gather text: topic-tagged blocks for this section's topics
    section_text = _get_blocks_by_topic(
        doc, config["topics"], max_chars=config["max_chars"])

    # Also add section-based fallback text if topic text is thin
    fallback_ids = _TOPIC_TO_SECTION_IDS.get(config["key"], [])
    if fallback_ids and len(section_text) < 2000:
        fallback = _get_section_text_for_extraction(
            doc, fallback_ids, max_chars=config["max_chars"])
        section_text = section_text + "\n\n---\n\n" + fallback

    # Termination fee-amount supplement: if no dollar amounts captured,
    # scan for blocks pairing dollar amounts with fee/termination language
    if config["key"] == "termination":
        _HAS_DOLLAR_AMT = re.compile(r'\$[\d,]{4,}')
        _has_amt = _HAS_DOLLAR_AMT.search(section_text)
        if _has_amt:
            print(
                f"      termination fee scan: already has '{_has_amt.group()}' — skipped")
        if not _has_amt:
            _TERM_FEE_PHRASE = re.compile(
                r'(?:Company|Parent|Reverse)\s+Termination\s+Fee', re.IGNORECASE)
            _LARGE_AMT = re.compile(r'\$[\d,]{7,}')
            candidates = []
            for b in doc.blocks:
                if not b.text.strip() or b.type == "heading":
                    continue
                t = b.text.strip()
                if _TERM_FEE_PHRASE.search(t) and _LARGE_AMT.search(t):
                    candidates.append(t)
            candidates.sort(key=len)
            extra_parts = []
            extra_total = 0
            for t in candidates:
                if extra_total + len(t) > 10000:
                    continue
                extra_parts.append(t)
                extra_total += len(t)
            if extra_parts:
                print(
                    f"      termination fee scan: added {extra_total:,} chars from {len(extra_parts)} blocks")
                section_text = section_text + \
                    "\n\n---\n\n" + "\n\n".join(extra_parts)

     # Closing supplement: ensure outside/termination date is in the text
    if config["key"] == "closing":
        _OUTSIDE_DATE_DEF_RE = re.compile(
            r'(?:outside\s+date|["\u201c]\s*(?:Termination|End)\s+Date\s*["\u201d])',
            re.IGNORECASE
        )
        _DATE_VALUE_RE = re.compile(
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
            r'\s+\d{1,2},?\s+\d{4}')
        # Check if we already have the outside date with an actual date value
        has_date_with_value = False
        for m in _OUTSIDE_DATE_DEF_RE.finditer(section_text):
            context = section_text[max(0, m.start()-100):m.end()+300]
            if _DATE_VALUE_RE.search(context):
                has_date_with_value = True
                break
        if not has_date_with_value:
            candidates = []
            for b in doc.blocks:
                t = b.text.strip()
                if not t or b.type == "heading":
                    continue
                if _OUTSIDE_DATE_DEF_RE.search(t) and _DATE_VALUE_RE.search(t):
                    candidates.append(t)
            candidates.sort(key=len)
            extra = []
            extra_total = 0
            for t in candidates[:3]:
                if extra_total + len(t) > 5000:
                    continue
                extra.append(t)
                extra_total += len(t)
            if extra:
                print(
                    f"      closing outside-date scan: added {extra_total:,} chars from {len(extra)} blocks")
                section_text = section_text + \
                    "\n\n---\n\n" + "\n\n".join(extra)

    # Clean source text for reference output (strip topic/section tags)
    source_clean = re.sub(
        r'^\[(?:Topic|Section): [^\]]+\]\n?', '', section_text, flags=re.MULTILINE)
    source_clean = re.sub(r'\n---\n', '\n', source_clean)
    source_clean = source_clean.strip()

    prompt = SECTION_PROMPTS[config["key"]].format(
        ticker=ticker, target=target, acquirer=acquirer,
        section_text=section_text, facts_json=facts_json,
    )

    model = MODEL_OPUS if config["model"] == "opus" else MODEL_STANDARD
    kwargs = {
        "model": model,
        "max_tokens": 4000,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    if config.get("thinking"):
        kwargs["max_tokens"] = 8000
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 5000}
        # temperature not supported with extended thinking
        del kwargs["temperature"]

    response = client.messages.create(**kwargs)
    text_parts = [b.text for b in response.content if b.type == "text"]
    raw = "\n".join(text_parts).strip()
    # Strip markdown the LLM may have added
    raw = re.sub(r'\*\*([^*]+)\*\*', r'\1', raw)
    raw = re.sub(r'(?<!\w)\*([^*]+)\*(?!\w)', r'\1', raw)
    raw = re.sub(r'^#{1,6}\s+', '', raw, flags=re.MULTILINE)
    return config["key"], raw, source_clean


def _generate_opening(client: Anthropic, section_results: dict,
                      ticker: str, target: str, acquirer: str) -> str:
    """Generate a 2-4 sentence opening paragraph from extracted section results."""
    context = "\n\n".join(f"{k.upper()}:\n{v}" for k,
                          v in section_results.items())
    prompt = f"""Write 2-4 sentences stating the key facts for {ticker} ({target}) being acquired by {acquirer}.
Cover: SH vote date (or that it has not been announced), HSR status, other regulatory status, expected closing.
Concise and direct. Reference specific dates. No bullet points. Plain text, no markdown.

RULES:
- State ONLY what the filing discloses. Do not interpret, infer, or editorialize.
- Do NOT use phrases like "notably", "significantly", "suggesting", "indicating",
  "which would", "appears to", "expanded to reveal", or similar editorial language.
- Use near-verbatim filing language where possible.

EXTRACTED SECTIONS:
{context}"""

    response = client.messages.create(
        model=MODEL_STANDARD, max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def _assemble_summary(opening: str, section_results: dict, source_texts: dict = None) -> str:  # noqa: ARG001
    """Assemble section results into the final summary text."""
    parts = [opening, ""]
    for key in SECTION_ORDER:
        header = SECTION_HEADERS[key]
        body = section_results.get(key, "Not disclosed.")
        # Clean up any echoed headers from LLM output
        body = re.sub(rf'^{re.escape(header)}\s*\n?',
                      '', body, flags=re.IGNORECASE)
        # Strip echoed COMPANY: / ACQUIRER: lines
        body = re.sub(r'(?m)^COMPANY:.*\n?', '', body)
        body = re.sub(r'(?m)^ACQUIRER:.*\n?', '', body)
        body = body.strip()
        parts.append(header)
        parts.append(body)
        parts.append("")
    return "\n".join(parts).strip()


def generate_full_summary(client: Anthropic, doc: CanonicalDocument,
                          ticker: str, target: str, acquirer: str) -> str:
    """Generate a full summary using parallel per-section LLM calls."""
    facts_json = json.dumps(asdict(doc.priority_facts), indent=2, default=str)

    # Run all section extractions in parallel
    section_results = {}
    source_texts = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {}
        for config in SECTION_CONFIGS:
            future = executor.submit(
                _generate_section, client, config, doc,
                ticker, target, acquirer, facts_json
            )
            futures[future] = config["key"]

        for future in as_completed(futures):
            key, text, source = future.result()
            section_results[key] = text
            source_texts[key] = source
            print(f"      {key}: done ({len(text)} chars)")

    # Generate opening paragraph last -- using section results as context
    opening = _generate_opening(
        client, section_results, ticker, target, acquirer)
    print(f"      opening: done ({len(opening)} chars)")

    # Assemble final output
    return _assemble_summary(opening, section_results, source_texts)


def _parse_summary_sections(text: str) -> tuple:
    """Parse new-format summary text into (opening_paragraph, [(header, content), ...]).

    The opening paragraph is all text before the first ALL-CAPS header line.
    Section headers are lines that are entirely uppercase letters, spaces, &, /.
    """
    lines = text.split("\n")
    opening_lines = []
    sections = []
    current_header = None
    current_lines = []
    found_first_header = False

    SKIP_HEADERS = {"RULES"}

    for line in lines:
        stripped = line.strip()
        is_header = (stripped and re.match(r'^[A-Z][A-Z &/]+$', stripped)
                     and len(stripped) >= 3 and stripped not in SKIP_HEADERS)
        if not is_header and stripped:
            colon_match = re.match(r'^([A-Z][A-Z &/]+):?$', stripped)
            if colon_match and colon_match.group(1) not in SKIP_HEADERS:
                candidate = colon_match.group(1)
                if len(candidate) >= 3 and candidate == candidate.upper():
                    is_header = True
                    stripped = candidate

        if is_header:
            if not found_first_header:
                found_first_header = True
            if current_header is not None:
                sections.append(
                    (current_header, "\n".join(current_lines).strip()))
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


_SUMMARY_HEADER_TO_CATEGORY = {
    "DATES": "dates",
    "CONSIDERATION": "consideration",
    "FINANCING": "financing",
    "SH APPROVAL": "sh_votes",
    "HSR": "hsr",
    "OTHER REGULATORY": "regulatory",
    "CONDITIONS": "conditions",
    "CONDITIONS TO CLOSING": "conditions",
    "CLOSING": "closing",
    "TERMINATION & FEES": "termination",
    "TERMINATION": "termination",
}

_CATEGORY_TO_SUMMARY_HEADER = {v: k for k, v in _SUMMARY_HEADER_TO_CATEGORY.items()
                               if k not in ("CONDITIONS TO CLOSING", "TERMINATION")}


def _summary_text_to_category_dict(summary_text: str) -> dict:
    """Parse summary text into {category_key: section_content} dict.

    Returns dict with opening paragraph under key '_opening'.
    """
    if not summary_text or not summary_text.strip():
        return {}
    opening, sections = _parse_summary_sections(summary_text)
    result = {}
    if opening:
        result["_opening"] = opening
    for header, content in sections:
        cat_key = _SUMMARY_HEADER_TO_CATEGORY.get(header.strip())
        if cat_key:
            result[cat_key] = content.strip()
    return result


def _merge_changes_into_summary(base_summary: str, events: List[ChangeEvent]) -> str:
    """Update the base summary text with change events for chain propagation.

    Simple text manipulation (no LLM):
    - For newly_disclosed events: append bullet to the appropriate section
    - For updated events: try to replace old value with new, else append
    """
    if not events or not base_summary:
        return base_summary

    from .config import SECTION_HEADERS, SECTION_ORDER
    cat_dict = _summary_text_to_category_dict(base_summary)
    opening = cat_dict.pop("_opening", "")

    events_by_cat: dict = {}
    for e in events:
        if e.category in ("background", "other_material"):
            continue
        events_by_cat.setdefault(e.category, []).append(e)

    for cat, cat_events in events_by_cat.items():
        section_text = cat_dict.get(cat, "")
        for event in cat_events:
            if event.change_type == "newly_disclosed":
                label = event.field or ""
                val = event.new_value or event.summary or ""
                if label and val:
                    new_line = f"- {label}: {val}"
                elif val:
                    new_line = f"- {val}"
                else:
                    continue
                section_text = (section_text + "\n" +
                                new_line) if section_text else new_line
            elif event.change_type in ("updated", "changed"):
                old_val = event.old_value or ""
                new_val = event.new_value or ""
                if old_val and new_val and old_val in section_text:
                    section_text = section_text.replace(old_val, new_val, 1)
                elif event.field and new_val:
                    new_line = f"- {event.field}: {new_val}"
                    section_text = (section_text + "\n" +
                                    new_line) if section_text else new_line
        cat_dict[cat] = section_text

    parts = []
    if opening:
        parts.append(opening)
        parts.append("")
    for key in SECTION_ORDER:
        header = SECTION_HEADERS.get(key, key.upper())
        cat_key = _SUMMARY_HEADER_TO_CATEGORY.get(header, key)
        content = cat_dict.get(cat_key, "")
        if content:
            parts.append(header)
            parts.append(content)
            parts.append("")

    return "\n".join(parts)


def _generate_change_opening(client: Anthropic, events: List[ChangeEvent],
                             ticker: str, old_label: str, new_label: str) -> str:
    """Generate a 2-4 sentence opening paragraph for a change report."""
    # Build a concise summary of changes for the LLM
    summary_parts = []
    for e in events:
        if e.category == "background":
            if e.change_type == "other":
                summary_parts.append("Background: No material changes.")
            else:
                summary_parts.append(f"Background: Changes detected.")
            continue
        if e.field:
            if e.change_type == "newly_disclosed":
                summary_parts.append(
                    f"{e.category}/{e.field}: {e.new_value} [NEW]")
            elif e.old_value and e.new_value:
                summary_parts.append(
                    f"{e.category}/{e.field}: {e.old_value} -> {e.new_value}")
        elif e.summary:
            summary_parts.append(f"{e.category}: {e.summary[:200]}")

    changes_summary = "\n".join(
        summary_parts) if summary_parts else "No significant changes detected."

    response = client.messages.create(
        model=MODEL_STANDARD,
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": CHANGE_OPENING_PROMPT.format(
                old_label=old_label,
                new_label=new_label,
                ticker=ticker,
                changes_summary=changes_summary,
            )
        }]
    )
    return response.content[0].text.strip()


def _is_hsr_event(event: ChangeEvent) -> bool:  # noqa: F841
    """Check if a regulatory event is HSR-related."""
    field_lower = (event.field or "").lower()
    val_lower = ((event.new_value or "") + (event.old_value or "")).lower()
    return ("hsr" in field_lower or "hart-scott" in field_lower
            or "hsr" in val_lower or "hart-scott" in val_lower)


def generate_change_report(client: Anthropic, events: List[ChangeEvent],
                           ticker: str, old_label: str, new_label: str,
                           base_summary_text: str = "") -> str:
    """Generate a change report with opening paragraph + clean section format.

    If base_summary_text is provided, each section shows the base summary first
    (under [From ...]) followed by changes (under [Changes in ...]).
    """
    lines = []

    # Parse base summary into category dict for merged output
    base_dict = _summary_text_to_category_dict(
        base_summary_text) if base_summary_text else {}

    # Opening paragraph (LLM-generated)
    opening = _generate_change_opening(
        client, events, ticker, old_label, new_label)
    for para_line in opening.split("\n"):
        para_line = para_line.strip()
        if para_line:
            lines.append(para_line)
    lines.append("")

    # Helper: humanize snake_case field names, preserving acronyms
    def _humanize(field_name):
        words = field_name.replace("_", " ").title().split()
        return " ".join(w.upper() if w.upper() in _PRESERVE_UPPER else w for w in words)

    # Helper: format events for a category
    def _format_fact_events(cat_events):
        if not cat_events:
            return ["No changes."]
        result = []
        for event in cat_events:
            if event.field:
                label = _humanize(event.field)
                if event.change_type == "newly_disclosed":
                    new = event.new_value or "[not disclosed]"
                    result.append(f"- {label}: {new}  [NEW]")
                elif event.change_type == "removed":
                    old = event.old_value or "[removed]"
                    result.append(f"- {label}: {old}  [REMOVED]")
                else:
                    old = event.old_value or "[not disclosed]"
                    new = event.new_value or "[not disclosed]"
                    if len(old) + len(new) < 120:
                        result.append(f"- {label}: {old} -> {new}")
                    else:
                        result.append(f"- {label}:")
                        result.append(f"    Was: {old}")
                        result.append(f"    Now: {new}")
            elif event.summary:
                for sl in event.summary.split("\n"):
                    sl = sl.strip()
                    if sl:
                        result.append(sl if sl.startswith(
                            "- ") or sl.startswith("\u2022") else f"- {sl}")
        return result if result else ["No changes."]

    # Helper: add a section with base summary + changes (merged format)
    def _add_section(header, cat_key, cat_events):
        lines.append(header)
        base_content = base_dict.get(cat_key, "")
        if base_content:
            # Strip LLM throat-clearing and truncation notes from baseline
            cleaned_lines = []
            for bl in base_content.split("\n"):
                bl = bl.strip()
                if not bl:
                    continue
                bl_lower = bl.lower()
                if (bl_lower.startswith("based on a careful review")
                        or bl_lower.startswith("based on the provided filing text")
                        or bl_lower.startswith("note: the filing text provided is truncated")
                        or bl_lower.startswith("note: the specific dollar amounts")
                        or bl_lower.startswith("to complete this extraction")):
                    continue
                cleaned_lines.append(bl)
            lines.append(f"  [From {old_label}:]")
            for bl in cleaned_lines:
                lines.append(f"  {bl}")
            lines.append("")
            lines.append(f"  [Changes in {new_label}:]")
        for fl in _format_fact_events(cat_events):
            lines.append(f"  {fl}" if base_content else fl)
        lines.append("")

    # --- Structured sections (ALL-CAPS headers for DOCX parser) ---
    _add_section("DATES", "dates",
                 [e for e in events if e.category == "dates"])

    _add_section("CONSIDERATION", "consideration",
                 [e for e in events if e.category == "consideration"])

    _add_section("FINANCING", "financing",
                 [e for e in events if e.category == "financing"])

    _add_section("SH APPROVAL", "sh_votes",
                 [e for e in events if e.category == "sh_votes"])

    _add_section("HSR", "hsr",
                 [e for e in events if e.category == "hsr"])

    _add_section("OTHER REGULATORY", "regulatory",
                 [e for e in events if e.category == "regulatory"])

    _add_section("CONDITIONS", "conditions",
                 [e for e in events if e.category == "conditions"])

    _add_section("CLOSING", "closing",
                 [e for e in events if e.category == "closing"])

    _add_section("TERMINATION & FEES", "termination",
                 [e for e in events if e.category == "termination"])

    # --- OTHER MATERIAL (only if there are events \u2014 no base summary shown) ---
    other_mat_events = [e for e in events if e.category == "other_material"]
    if other_mat_events:
        lines.append("OTHER MATERIAL")
        lines.extend(_format_fact_events(other_mat_events))
        lines.append("")

    # --- BACKGROUND ---
    bg_events = [e for e in events if e.category == "background"]
    lines.append("BACKGROUND")

    bg_event = bg_events[0] if bg_events else None
    if bg_event and bg_event.summary:
        for sl in bg_event.summary.split("\n"):
            sl = sl.strip()
            if sl:
                lines.append(sl)
    elif not bg_events:
        lines.append("Background section not found in one or both filings.")

    return "\n".join(lines)


def format_txt_header(ticker: str, target: str, doc_label: str,
                      report_type: str, timestamp: str) -> str:
    """Create a formatted TXT header."""
    lines = [
        "=" * 72,
        f"  {ticker} -- {target}",
        f"  {doc_label}",
    ]
    if report_type:
        lines.append(f"  {report_type}")
    lines.append(f"  Generated: {timestamp}")
    lines.append("=" * 72)
    lines.append("")
    return "\n".join(lines) + "\n"
