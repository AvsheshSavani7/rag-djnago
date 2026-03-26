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

    try:
        response = client.messages.create(**kwargs)
    except TypeError as e:
        if "thinking" in str(e):
            kwargs.pop("thinking", None)
            kwargs["temperature"] = 0
            response = client.messages.create(**kwargs)
        else:
            raise
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
    prompt = f"""Write 2-4 sentences summarizing the key facts for {ticker} ({target}) being acquired by {acquirer}.
Cover: SH vote date, HSR status, other regulatory status, expected closing.
Analyst voice -- concise and direct. Reference specific dates. No bullet points.
Plain text, no markdown.

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
                           ticker: str, old_label: str, new_label: str) -> str:
    """Generate a change report with opening paragraph + clean section format."""
    lines = []

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

    # Helper: add a section with its events
    def _add_section(header, cat_events):
        lines.append(header)
        lines.extend(_format_fact_events(cat_events))
        lines.append("")

    # --- Structured sections (ALL-CAPS headers for DOCX parser) ---
    _add_section("DATES",
                 [e for e in events if e.category == "dates"])

    _add_section("CONSIDERATION",
                 [e for e in events if e.category == "consideration"])

    _add_section("FINANCING",
                 [e for e in events if e.category == "financing"])

    _add_section("SH APPROVAL",
                 [e for e in events if e.category == "sh_votes"])

    _add_section("HSR",
                 [e for e in events if e.category == "hsr"])

    _add_section("OTHER REGULATORY",
                 [e for e in events if e.category == "regulatory"])

    _add_section("CLOSING",
                 [e for e in events if e.category == "closing"])

    _add_section("TERMINATION & FEES",
                 [e for e in events if e.category == "termination"])

    # --- OTHER MATERIAL (only if there are events) ---
    other_mat_events = [e for e in events if e.category == "other_material"]
    if other_mat_events:
        _add_section("OTHER MATERIAL", other_mat_events)

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
