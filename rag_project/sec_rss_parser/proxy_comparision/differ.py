"""
differ.py — Diff engine: fact diff, background diff, category comparison.
"""

import re
import json
import hashlib
import difflib
import os
from typing import List, Dict, Any, Optional, Tuple

from anthropic import Anthropic

from .models import PriorityFacts, CanonicalSection, CanonicalDocument, ChangeEvent
from .config import (
    MODEL_STANDARD, OUTPUT_FOLDER,
    TIER1_SECTION_IDS, TIER1_CATEGORIES,
    FACT_CATEGORY_MAP, _COMPARISON_PROMPTS,
    _COMPARISON_RULES,
    _CATEGORY_TO_SECTIONS, _CATEGORY_TO_TOPICS,
    BACKGROUND_DIFF_INTERPRET_PROMPT,
    get_form_label,
)
from .classifier import _get_blocks_by_topic
from .section_mapper import get_section_by_id, get_sections_by_ids


# =============================================================================
# 4.1: Structured Fact Diff
# =============================================================================

def _normalize_fact_val(val):
    """Normalize a fact value for comparison. Returns None for empty/unknown."""
    if val is None:
        return None
    if isinstance(val, str):
        v = val.strip()
        if v.lower() in ("null", "", "n/a", "none", "not required", "not applicable"):
            return None
        return v
    if isinstance(val, list):
        return val if val else None
    return val


def _values_are_semantically_same(old_val, new_val) -> bool:
    """Check if two values are semantically the same despite minor wording differences."""
    if old_val is None and new_val is None:
        return True
    if old_val is None or new_val is None:
        return False
    if isinstance(old_val, str) and isinstance(new_val, str):
        def normalize(s):
            s = re.sub(r'[,;.\-\u2013\u2014()"]', ' ', s.lower())
            return re.sub(r'\s+', ' ', s).strip()
        n_old = normalize(old_val)
        n_new = normalize(new_val)
        # Sequence-level similarity (catches minor edits)
        ratio = difflib.SequenceMatcher(None, n_old, n_new).ratio()
        threshold = 0.70 if len(n_old) > 100 or len(n_new) > 100 else 0.85
        if ratio > threshold:
            return True
        # Word-overlap similarity (catches reordering -- LLM extraction phrasing varies)
        STOP_WORDS = {"the", "a", "an", "of", "to", "and", "in", "for", "by", "at",
                      "or", "is", "are", "was", "were", "be", "been", "its", "that",
                      "with", "from", "on", "as", "not", "must", "shall", "may",
                      "which", "have", "has", "had", "into", "upon", "certain",
                      "subject", "terms", "conditions", "pursuant", "set", "forth",
                      "such", "any", "all", "each", "other", "their", "this",
                      "will", "would", "could", "should", "under", "including",
                      "respect", "entered", "also", "than"}
        old_words = set(n_old.split()) - STOP_WORDS
        new_words = set(n_new.split()) - STOP_WORDS
        if old_words and new_words:
            overlap = len(old_words & new_words) / \
                max(len(old_words), len(new_words))
            # Lower threshold for long descriptions -- LLM extraction adds/omits details
            overlap_threshold = 0.60 if len(
                n_old) > 80 or len(n_new) > 80 else 0.70
            if overlap > overlap_threshold:
                return True
        return False
    if isinstance(old_val, list) and isinstance(new_val, list):
        if len(old_val) != len(new_val):
            return False
        return all(_values_are_semantically_same(str(a), str(b))
                   for a, b in zip(sorted(str(x) for x in old_val),
                                   sorted(str(x) for x in new_val)))
    return old_val == new_val


def _format_reg_summary(reg: dict) -> str:
    """Format a regulatory item as a readable one-line summary."""
    parts = []
    agency = reg.get("agency", "Unknown")
    jurisdiction = reg.get("jurisdiction", "")
    if jurisdiction:
        parts.append(f"{jurisdiction} - {agency}")
    else:
        parts.append(agency)
    status = reg.get("status")
    if status:
        parts.append(f"Status: {status}")
    filed = reg.get("filed_date")
    if filed:
        parts.append(f"Filed: {filed}")
    approved = reg.get("approval_date")
    if approved:
        parts.append(f"Approved: {approved}")
    details = reg.get("details")
    if details:
        parts.append(details)
    return " | ".join(parts)


def _diff_list_values(key: str, old_list: list, new_list: list, category: str) -> List[ChangeEvent]:
    """Diff two list-valued fact fields item by item. Returns change events for added/removed items."""
    events = []
    old_strs = [str(x).strip() for x in old_list]
    new_strs = [str(x).strip() for x in new_list]

    # Match items by semantic similarity
    matched_old = set()
    matched_new = set()
    for i, o in enumerate(old_strs):
        for j, n in enumerate(new_strs):
            if j not in matched_new and _values_are_semantically_same(o, n):
                matched_old.add(i)
                matched_new.add(j)
                break

    added = [new_strs[j] for j in range(len(new_strs)) if j not in matched_new]
    removed = [old_strs[i]
               for i in range(len(old_strs)) if i not in matched_old]

    for item in added:
        events.append(ChangeEvent(
            tier=1, category=category, change_type="newly_disclosed",
            field=f"{key} (added)", new_value=item,
        ))
    for item in removed:
        events.append(ChangeEvent(
            tier=1, category=category, change_type="removed",
            field=f"{key} (removed)", old_value=item,
        ))
    return events


def diff_facts(old: PriorityFacts, new: PriorityFacts) -> List[ChangeEvent]:
    """Compare priority facts field by field. Returns Tier-1 change events."""
    events = []

    # Compare dict-based categories
    for attr in ["dates", "consideration", "financing", "sh_votes", "closing_guidance"]:
        old_dict = getattr(old, attr) or {}
        new_dict = getattr(new, attr) or {}
        category = FACT_CATEGORY_MAP[attr]

        # Skip fields that duplicate other fact categories or are too noisy
        # duplicates regulatory + closing_conditions
        SKIP_FIELDS = {"gating_items"}

        all_keys = set(list(old_dict.keys()) + list(new_dict.keys()))
        for key in all_keys:
            if key in SKIP_FIELDS:
                continue
            old_val = _normalize_fact_val(old_dict.get(key))
            new_val = _normalize_fact_val(new_dict.get(key))

            # Skip if semantically identical
            if _values_are_semantically_same(old_val, new_val):
                continue

            # Skip "removed" if new is None -- likely extraction miss, not actual removal
            if old_val is not None and new_val is None:
                continue

            # List-valued fields: diff item by item instead of dumping two arrays
            if isinstance(old_val, list) and isinstance(new_val, list):
                events.extend(_diff_list_values(
                    key, old_val, new_val, category))
                continue

            if old_val is None and new_val is not None:
                change_type = "newly_disclosed"
            else:
                change_type = "updated"

            events.append(ChangeEvent(
                tier=1,
                category=category,
                change_type=change_type,
                field=key,
                old_value=str(old_val) if old_val else None,
                new_value=str(new_val) if new_val else None,
            ))

    # Compare regulatory -- use fuzzy agency name matching
    old_regs = old.regulatory or []
    new_regs = new.regulatory or []

    # Build lookup by normalized agency name
    def _reg_key(r):
        return re.sub(r'[^a-z0-9]+', '', ((str(r.get("agency") or "")) + (str(r.get("jurisdiction") or ""))).lower())

    old_reg_map = {_reg_key(r): r for r in old_regs}
    new_reg_map = {_reg_key(r): r for r in new_regs}

    all_reg_keys = set(list(old_reg_map.keys()) + list(new_reg_map.keys()))
    for rk in sorted(all_reg_keys):
        old_reg = old_reg_map.get(rk)
        new_reg = new_reg_map.get(rk)
        agency_name = (new_reg or old_reg or {}).get("agency", rk)

        if old_reg is None and new_reg is not None:
            events.append(ChangeEvent(
                tier=1, category="regulatory", change_type="newly_disclosed",
                field=agency_name,
                new_value=_format_reg_summary(new_reg),
            ))
        elif old_reg is not None and new_reg is None:
            # Skip removals -- likely just LLM naming the agency differently
            continue
        elif old_reg != new_reg:
            # Find substantive field changes (status, filed_date, approval_date)
            for fname in ["status", "filed_date", "approval_date", "details"]:
                ov = old_reg.get(fname)
                nv = new_reg.get(fname)
                if ov != nv and not _values_are_semantically_same(str(ov), str(nv)):
                    events.append(ChangeEvent(
                        tier=1, category="regulatory", change_type="updated",
                        field=f"{agency_name} -- {fname}",
                        old_value=str(ov) if ov else "[not disclosed]",
                        new_value=str(nv) if nv else "[not disclosed]",
                    ))

    return events


# =============================================================================
# 4.2: Background Sentence-Level Diff
# =============================================================================

def _clean_filing_artifacts(text: str) -> str:
    """Remove SEC filing artifacts (page numbers, TOC markers, running headers)."""
    # Remove page number + TABLE OF CONTENTS artifacts
    text = re.sub(r'\d+\s*\n?\s*TABLE OF CONTENTS\s*\n?',
                  ' ', text, flags=re.I)
    # Remove standalone page numbers on their own line
    text = re.sub(r'\n\s*\d{1,3}\s*\n', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, handling common abbreviations."""
    # Clean SEC filing artifacts first
    text = _clean_filing_artifacts(text)

    # Protect abbreviations
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Sr|Jr|Inc|Corp|Ltd|Co|vs|etc|approx)\.',
                  r'\1<PERIOD>', text, flags=re.I)
    text = re.sub(r'(\d)\.\s*(\d)', r'\1<PERIOD>\2', text)  # Decimal numbers

    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)

    # Restore protected periods
    sentences = [s.replace('<PERIOD>', '.').strip() for s in sentences]
    return [s for s in sentences if s and len(s) > 10]


def sentence_hash(s: str) -> str:
    """Normalize and hash a sentence for comparison."""
    normalized = _clean_filing_artifacts(s).lower()
    return hashlib.md5(normalized.encode()).hexdigest()


def diff_background_with_llm(client: Anthropic,
                             old_section: Optional[CanonicalSection],
                             new_section: Optional[CanonicalSection],
                             old_label: str, new_label: str) -> Dict[str, Any]:
    """Use Haiku to compare two Background sections and identify real changes.

    Returns dict with:
        has_changes: bool
        summary: str (bullet-point summary of changes, or "No material changes.")
    """
    if old_section is None and new_section is not None:
        return {
            "has_changes": True,
            "summary": "Background section is entirely new in the later filing.",
        }
    if new_section is None or old_section is None:
        return {"has_changes": False, "summary": "No material changes."}

    # Haiku context is 200K tokens (~800K chars). Background is typically 40-50K words (~250K chars).
    # Two sections fit comfortably. Use 300K per section as a generous limit.
    max_chars = 300_000
    old_text = old_section.text[:max_chars]
    new_text = new_section.text[:max_chars]

    prompt = f"""Compare these two versions of a "Background of the Merger" section from SEC filings. Your job is purely mechanical: identify any text that appears in one version but not the other.

EARLIER VERSION ({old_label}):
{old_text}

---

LATER VERSION ({new_label}):
{new_text}

---

INSTRUCTIONS:
- Report paragraphs or sentences that are ADDED in the later version (new factual content not in earlier).
- Report paragraphs or sentences that are REMOVED in the later version (factual content present in earlier but deleted).
- Report sentences where the SUBSTANCE meaningfully changed (different facts, dates, names, dollar amounts, or deal terms).
- IGNORE these non-substantive differences — do NOT report them:
  * List item renumbering (e.g., "(i)" → "(a)", "(ii)" → "(b)")
  * Punctuation changes (commas, semicolons, periods)
  * Formatting, whitespace, page numbers, "TABLE OF CONTENTS" headers
  * Legal entity name corrections that refer to the same entity (e.g., "Citibank, N.A." → "Citigroup Global Markets Inc." when both are referred to as "Citi")
  * Cross-reference or section number updates

If the two versions contain exactly the same text (ignoring formatting), respond with exactly:
NO_CHANGES

If there is ANY text difference, respond with:
CHANGES
- ADDED: [quote or summarize the added text, with key facts/dates/names]
- REMOVED: [quote or summarize the removed text]
- CHANGED: [describe what was changed]

Be precise. Include specific facts, dates, names, and dollar amounts from the changed text."""

    old_chars = len(old_text)
    new_chars = len(new_text)
    diag = []
    diag.append(
        f"Background comparison: old={old_chars:,} chars, new={new_chars:,} chars, diff={new_chars - old_chars:+,} chars")
    diag.append(f"OLD ends with: \"...{old_text[-200:].strip()[:100]}\"")
    diag.append(f"NEW ends with: \"...{new_text[-200:].strip()[:100]}\"")
    print(f"    Comparing Background sections via Haiku...")
    for d in diag:
        print(f"    {d}")
    try:
        response = client.messages.create(
            model=MODEL_STANDARD, max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.content[0].text.strip()
        diag.append(f"Haiku comparison response:\n---\n{answer}\n---")
        print(
            f"    Background comparison Haiku response ({len(answer)} chars):")
        for line in answer.split('\n'):
            print(f"      {line}")

        if "NO_CHANGES" in answer.upper():
            diag.append("RESULT: no changes")
            return {"has_changes": False, "summary": "No material changes.", "_diag": diag}
        else:
            summary = answer
            # Strip "CHANGES" prefix if present
            if summary.upper().startswith("CHANGES"):
                summary = summary[len("CHANGES"):].strip()
            diag.append("RESULT: changes found")
            return {"has_changes": True, "summary": summary, "_diag": diag}

    except Exception as e:
        diag.append(f"ERROR: {e}")
        print(f"    Background comparison failed: {e}")
        return {"has_changes": False, "summary": "Background comparison unavailable.", "_diag": diag}


def word_level_diff(old_text: str, new_text: str) -> str:
    """Compute word-level diff markup between two text strings."""
    old_words = old_text.split()
    new_words = new_text.split()

    matcher = difflib.SequenceMatcher(None, old_words, new_words)
    parts = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(" ".join(old_words[i1:i2]))
        elif tag == "delete":
            parts.append(f"[-{' '.join(old_words[i1:i2])}-]")
        elif tag == "insert":
            parts.append(f"[+{' '.join(new_words[j1:j2])}+]")
        elif tag == "replace":
            parts.append(f"[-{' '.join(old_words[i1:i2])}-]")
            parts.append(f"[+{' '.join(new_words[j1:j2])}+]")

    return " ".join(parts)


def interpret_background_diff(client: Anthropic, diff_result: Dict[str, Any],
                              doc1_label: str, doc2_label: str) -> str:
    """Send background diff to LLM for interpretation."""
    if not diff_result["inserted"] and not diff_result["deleted"] and not diff_result["modified"]:
        return "Background sections are substantially identical. No material changes."

    inserted_text = "\n".join(
        f"- {s}" for s in diff_result["inserted"]) or "None"
    deleted_text = "\n".join(
        f"- {s}" for s in diff_result["deleted"]) or "None"

    modified_parts = []
    for old_s, new_s in diff_result["modified"]:
        diff_markup = word_level_diff(old_s, new_s)
        modified_parts.append(
            f"- OLD: {old_s}\n  NEW: {new_s}\n  DIFF: {diff_markup}")
    modified_text = "\n".join(modified_parts) or "None"

    response = client.messages.create(
        model=MODEL_STANDARD,
        max_tokens=3000,
        messages=[{
            "role": "user",
            "content": BACKGROUND_DIFF_INTERPRET_PROMPT.format(
                doc1_label=doc1_label,
                doc2_label=doc2_label,
                inserted_text=inserted_text,
                deleted_text=deleted_text,
                modified_text=modified_text,
                unchanged_count=diff_result["unchanged_count"],
            )
        }]
    )
    return response.content[0].text.strip()


# =============================================================================
# 4.5: Direct Category Comparison (10K/10Q approach)
# =============================================================================

def _gather_category_text(doc: CanonicalDocument, category: str, max_chars: int = 30000) -> str:
    """Gather text from category-specific sections for comparison.

    Unlike _get_section_text_for_extraction (which adds summary/Q&A fallbacks
    for comprehensive first-filing extraction), this function gathers only
    targeted text so comparisons stay focused on real differences.
    """
    section_ids = _CATEGORY_TO_SECTIONS.get(category, [])
    parts = []
    total = 0

    # Layer 1: Topic-tagged blocks (primary -- content-based, survives mega-sections)
    topics_needed = _CATEGORY_TO_TOPICS.get(category, [])

    if topics_needed:
        topic_text = _get_blocks_by_topic(
            doc, topics_needed, max_chars=max_chars // 2)
        if topic_text.strip():
            parts.append(topic_text)
            total += len(topic_text)

    # Layer 2: Section text -- prioritize category-specific sections,
    # add merger_agreement_summary only if budget remains.
    primary_ids = [sid for sid in section_ids if sid !=
                   "merger_agreement_summary"]
    secondary_ids = [sid for sid in section_ids if sid ==
                     "merger_agreement_summary"]

    for sid_group in [primary_ids, secondary_ids]:
        if total >= max_chars:
            break
        sections = get_sections_by_ids(doc, sid_group)
        for s in sections:
            remaining = max_chars - total
            if remaining < 2000:
                break
            chunk = f"[Section: {s.raw_title}]\n{s.text}"
            if len(chunk) > remaining:
                chunk = chunk[:remaining] + "\n[TRUNCATED]"
            parts.append(chunk)
            total += len(chunk)

    return "\n\n---\n\n".join(parts) if parts else ""


def _parse_comparison_response(raw: str, category: str) -> List[ChangeEvent]:
    """Parse LLM JSON response into ChangeEvents."""
    text = raw.strip()

    # Strategy 1: Extract JSON from markdown fences -- use LAST block (LLM sometimes
    # self-corrects mid-response, so the final block is the most accurate)
    fence_matches = list(re.finditer(
        r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL))
    if fence_matches:
        for fm in reversed(fence_matches):
            try:
                items = json.loads(fm.group(1))
                if isinstance(items, list):
                    return _build_change_events(items, category)
            except json.JSONDecodeError:
                continue

    # Strategy 2: Direct JSON parse
    try:
        items = json.loads(text)
        if isinstance(items, list):
            return _build_change_events(items, category)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Find first balanced JSON array
    bracket_start = text.find('[')
    if bracket_start >= 0:
        depth = 0
        for i in range(bracket_start, len(text)):
            if text[i] == '[':
                depth += 1
            elif text[i] == ']':
                depth -= 1
                if depth == 0:
                    try:
                        items = json.loads(text[bracket_start:i+1])
                        if isinstance(items, list):
                            return _build_change_events(items, category)
                    except json.JSONDecodeError:
                        pass
                    break

    # No JSON found -- treat as empty
    if not text or text == "[]":
        return []
    print(f"    Compare ({category}): failed to parse JSON response")
    return []


def _build_change_events(items: list, category: str) -> List[ChangeEvent]:
    """Convert parsed JSON items into ChangeEvent objects."""
    events = []
    for item in items:
        if not isinstance(item, dict):
            continue
        change_type_raw = item.get("type", "")
        field = item.get("field", category)

        if change_type_raw == "new":
            events.append(ChangeEvent(
                tier=1,
                category=category,
                change_type="newly_disclosed",
                field=field,
                new_value=str(item.get("value", "")),
            ))
        elif change_type_raw == "changed":
            events.append(ChangeEvent(
                tier=1,
                category=category,
                change_type="updated",
                field=field,
                old_value=str(item.get("was", "")),
                new_value=str(item.get("now", "")),
            ))
        elif change_type_raw == "removed":
            events.append(ChangeEvent(
                tier=1,
                category=category,
                change_type="removed",
                field=field,
                old_value=str(item.get("value", "")),
            ))
    return events


def _compare_category_direct(client: Anthropic, category: str,
                             old_doc: CanonicalDocument, new_doc: CanonicalDocument,
                             old_label: str, new_label: str,
                             diag_path: str = None) -> List[ChangeEvent]:
    """Compare both filings' sections for a category in one LLM call.

    This is the 10K/10Q approach: send both texts to one call, ask what changed.
    Eliminates false positives from independent extraction.
    """
    prompt_template = _COMPARISON_PROMPTS.get(category)
    if not prompt_template:
        print(f"    Compare ({category}): no prompt template -- skipping")
        return []

    old_text = _gather_category_text(old_doc, category)
    new_text = _gather_category_text(new_doc, category)

    if not old_text.strip() and not new_text.strip():
        print(
            f"    Compare ({category}): no text in either filing -- skipping")
        return []

    if not old_text.strip():
        old_text = "[Section not found in this filing]"
    if not new_text.strip():
        new_text = "[Section not found in this filing]"

    print(
        f"    Compare ({category}): old={len(old_text):,} chars, new={len(new_text):,} chars")

    prompt = prompt_template.format(
        old_label=old_label,
        new_label=new_label,
        old_text=old_text,
        new_text=new_text,
    )

    try:
        response = client.messages.create(
            model=MODEL_STANDARD,
            max_tokens=4000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.content[0].text.strip()

        # Diagnostic: save raw response to file + print summary
        if diag_path:
            with open(diag_path, "a") as df:
                df.write(f"\n{'='*60}\n")
                df.write(
                    f"CATEGORY: {category} | old={len(old_text):,} chars, new={len(new_text):,} chars\n")
                df.write(f"RAW RESPONSE:\n{answer}\n")

        events = _parse_comparison_response(answer, category)
        print(f"    Compare ({category}): {len(events)} changes detected")
        if events:
            for i, e in enumerate(events[:3]):
                field_preview = (e.field or "")[:50]
                val_preview = (
                    e.new_value or e.old_value or e.summary or "")[:60]
                print(
                    f"      [{i}] {e.change_type}: {field_preview} = {val_preview}")
            if len(events) > 3:
                print(f"      ... and {len(events) - 3} more")
        return events
    except Exception as e:
        print(f"    Compare ({category}): LLM call failed: {e}")
        return []


def _compare_other_material(client: Anthropic,
                            old_doc: CanonicalDocument,
                            new_doc: CanonicalDocument,
                            old_label: str, new_label: str,
                            diag_path: str = None) -> List[ChangeEvent]:
    """Catch-all sweep for material, deal-specific changes in 'general' blocks.

    Finds paragraphs in the new filing that are new or substantially changed
    compared to the old filing, then screens them for materiality.
    """
    # Collect general-tagged blocks from both filings
    old_texts = [b.text.strip() for b in old_doc.blocks
                 if b.topic == "general" and b.text.strip() and len(b.text.split()) >= 20]
    new_texts = [b.text.strip() for b in new_doc.blocks
                 if b.topic == "general" and b.text.strip() and len(b.text.split()) >= 20]

    if not new_texts:
        print(f"    Compare (other_material): no general blocks in new filing — skipping")
        return []

    # Build lookup of old blocks: fingerprint (first 200 chars) -> full text
    # This lets us detect both brand-new paragraphs AND paragraphs whose
    # opening is the same but whose body was substantially modified.
    old_by_fingerprint: Dict[str, str] = {}
    for t in old_texts:
        fp = t[:200].strip().lower()
        old_by_fingerprint[fp] = t

    # Find new or substantially changed paragraphs, scored by degree of change
    # so we can prioritize the most-changed content within the token budget.
    delta_scored = []  # (change_score, text)  — higher = more changed
    for t in new_texts:
        fp = t[:200].strip().lower()
        old_match = old_by_fingerprint.get(fp)
        if old_match is None:
            # Brand-new paragraph — high priority
            delta_scored.append((1.0, t))
        else:
            # Same opening — check if body changed substantially
            len_ratio = abs(len(t) - len(old_match)) / max(len(old_match), 1)
            if len_ratio > 0.10:
                # Score by degree of change (larger length delta = more new content)
                delta_scored.append((len_ratio, t))

    if not delta_scored:
        print(f"    Compare (other_material): no new/changed general blocks — skipping")
        return []

    # Split into brand-new vs modified paragraphs, then allocate budget
    # so both types get representation. Modified paragraphs (existing text
    # with new content inserted) are easy to miss if brand-new blocks
    # consume the entire budget.
    brand_new = [(s, t) for s, t in delta_scored if s >= 1.0]
    modified = [(s, t) for s, t in delta_scored if s < 1.0]
    modified.sort(key=lambda x: x[0], reverse=True)

    max_budget = 60000
    if brand_new and modified:
        mod_budget = max_budget // 2
    elif modified:
        mod_budget = max_budget
    else:
        mod_budget = 0

    delta_text_parts = []
    total_chars = 0

    # Fill modified bucket first (existing paragraphs with new content added)
    for _score, p in modified:
        if total_chars + len(p) > mod_budget:
            continue
        delta_text_parts.append(p)
        total_chars += len(p)

    # Fill remaining budget with brand-new paragraphs
    for _score, p in brand_new:
        if total_chars + len(p) > max_budget:
            continue
        delta_text_parts.append(p)
        total_chars += len(p)

    delta_text = "\n\n---\n\n".join(delta_text_parts)
    print(f"    Compare (other_material): {len(delta_scored)} new/changed general blocks, "
          f"sending {len(delta_text_parts)} ({total_chars:,} chars) for materiality screen")

    prompt = f"""\
You are screening NEW or CHANGED paragraphs from an SEC merger filing for material, deal-specific disclosures that fall outside the standard categories (dates, consideration, financing, shareholder approval, HSR, regulatory, closing conditions, termination fees, background).

These paragraphs appeared in the LATER filing ({new_label}) but NOT in the EARLIER filing ({old_label}), or were substantially changed.

NEW/CHANGED PARAGRAPHS:
{delta_text}

INSTRUCTIONS:
Only flag items that meet BOTH criteria:
(a) DEAL-SPECIFIC: directly about this transaction, the parties, or consequences of the merger — not generic market/industry risks
(b) MATERIAL: could meaningfully affect deal value, timing, certainty, or post-closing operations — not trivial or immaterial

Examples of what TO flag:
- Potential significant tax liabilities arising from the merger
- New litigation or regulatory investigations related to the deal
- Material adverse changes in either party's business disclosed for the first time
- New material conditions or commitments not covered by other sections
- Significant cost estimates or write-downs tied to the transaction

Examples of what NOT to flag:
- Generic risk factors about market conditions, interest rates, cyber risk
- Boilerplate forward-looking statement disclaimers
- Procedural/administrative disclosures (how to submit proxies, etc.)
- Minor cost items or immaterial amounts
- Risk factors that simply restate deal terms already covered elsewhere

{_COMPARISON_RULES}

Format:
- New material item: {{"field": "short descriptive name", "type": "new", "value": "concise description of the material disclosure (1-2 sentences)"}}

Return a JSON array. If nothing meets BOTH materiality AND deal-specificity thresholds, return: []"""

    try:
        response = client.messages.create(
            model=MODEL_STANDARD,
            max_tokens=4000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.content[0].text.strip()

        # Diagnostic logging
        _diag = diag_path or os.path.join(OUTPUT_FOLDER, "compare_diagnostic_latest.txt")
        try:
            os.makedirs(os.path.dirname(_diag), exist_ok=True)
            with open(_diag, "a") as df:
                df.write(f"\n{'='*60}\n")
                df.write(f"CATEGORY: other_material | delta={total_chars:,} chars, "
                         f"{len(delta_text_parts)} paragraphs\n")
                df.write(f"RAW RESPONSE:\n{answer}\n")
        except OSError as diag_err:
            print(f"    Compare (other_material): diagnostic write failed: {diag_err}")

        events = _parse_comparison_response(answer, "other_material")
        print(
            f"    Compare (other_material): {len(events)} material items flagged")
        if events:
            for i, e in enumerate(events[:3]):
                field_preview = (e.field or "")[:50]
                val_preview = (e.new_value or e.summary or "")[:60]
                print(
                    f"      [{i}] {e.change_type}: {field_preview} = {val_preview}")
            if len(events) > 3:
                print(f"      ... and {len(events) - 3} more")
        return events
    except Exception as e:
        print(f"    Compare (other_material): LLM call failed: {e}")
        return []


# =============================================================================
# 4.6: Full Pairwise Diff
# =============================================================================
# Background section detection patterns (promoted to module level)
_BG_START_PATTERNS = [
    re.compile(r"^Background of the Mergers?$", re.IGNORECASE),
    re.compile(r"^Background of the Transactions?$", re.IGNORECASE),
    re.compile(r"^Background of the Offers?$", re.IGNORECASE),
    re.compile(r"^Background of the Acquisitions?$", re.IGNORECASE),
    re.compile(r"^Background of the Proposed", re.IGNORECASE),
    re.compile(r"^Background$", re.IGNORECASE),
]

_BG_INLINE_PATTERNS = [
    re.compile(r"(?:^|\n)\s*Background of the Mergers?\s*(?:\n|$)",
               re.IGNORECASE),
    re.compile(
        r"(?:^|\n)\s*Background of the Transactions?\s*(?:\n|$)", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*Background of the Offers?\s*(?:\n|$)", re.IGNORECASE),
    re.compile(
        r"(?:^|\n)\s*Background of the Acquisitions?\s*(?:\n|$)", re.IGNORECASE),
]

_BG_END_PATTERNS = [
    re.compile(r"^Reasons? for the Mergers?", re.IGNORECASE),
    re.compile(r"^Reasons? for the Transactions?", re.IGNORECASE),
    re.compile(r"^Reasons? for the Offers?", re.IGNORECASE),
    re.compile(r"^Recommendation of", re.IGNORECASE),
    re.compile(r"^.{0,60}Board.{0,10}Recommend", re.IGNORECASE),
    re.compile(r"^.{0,60}Recommend.{0,20}Reasons? for", re.IGNORECASE),
    re.compile(r"^Opinion of", re.IGNORECASE),
    re.compile(r"^Fairness Opinion", re.IGNORECASE),
    re.compile(r"^Certain (?:Unaudited )?Prospective", re.IGNORECASE),
    re.compile(r"^Financial Projections?", re.IGNORECASE),
    re.compile(r"^The Merger Agreement", re.IGNORECASE),
    re.compile(r"^Position of .* Regarding Fairness", re.IGNORECASE),
    re.compile(r"^Interests of .* Directors", re.IGNORECASE),
    re.compile(r"^Purpose and Reasons", re.IGNORECASE),
    re.compile(r"^Merger Consideration", re.IGNORECASE),
]


def _extract_bg_deterministic(doc: CanonicalDocument, label: str,
                              bg_log=None) -> Optional[CanonicalSection]:
    """Find the Background section using deterministic text matching.

    No LLM calls -- uses regex patterns to identify start and end
    boundaries across all filing types (PREM14A, DEFM14A, S-4, F-4).

    Returns a CanonicalSection or None.
    """
    def log(msg):
        if bg_log:
            bg_log(msg)
        else:
            print(msg)

    blocks = doc.blocks
    bg_start = None
    method = None

    # Strategy 1: Look for a heading block whose text matches a BG pattern
    for i, block in enumerate(blocks):
        if block.type == "heading":
            text = block.text.strip()
            for pat in _BG_START_PATTERNS:
                if pat.match(text):
                    bg_start = i
                    method = f"heading_exact: \"{text}\""
                    break
            if bg_start is not None:
                break

    # Strategy 2: Look for inline sub-heading within paragraph blocks
    if bg_start is None:
        for i, block in enumerate(blocks):
            text = block.text.strip()
            for pat in _BG_INLINE_PATTERNS:
                m = pat.search(text)
                if m:
                    before = text[:m.start()].lower()
                    if any(kw in before for kw in ["captioned", "see page", "see the section", "entitled"]):
                        continue
                    bg_start = i
                    method = f"inline: block {i}, \"{text[:60]}\""
                    break
            if bg_start is not None:
                break

    # Strategy 3: Check for "Anchor Background of the Merger" pattern
    if bg_start is None:
        for i, block in enumerate(blocks):
            text = block.text.strip()
            if re.match(r"^(?:Anchor\s+)?Background of the Mergers?\s*$", text, re.IGNORECASE):
                bg_start = i
                method = f"anchor: \"{text}\""
                break
            m = re.match(
                r"^(?:Anchor\s+)?Background of the Mergers?\s+(?:The |In |On |During )", text, re.IGNORECASE)
            if m:
                bg_start = i
                method = f"anchor_with_text: \"{text[:80]}\""
                break

    if bg_start is None:
        log(f"    Background ({label}): not found (no start heading detected)")
        return None

    log(f"    Background ({label}): START via {method}")

    # Find the END boundary
    bg_end = len(blocks)  # default: end of document
    search_start = bg_start + 3  # skip a few blocks after start

    for i in range(search_start, len(blocks)):
        block = blocks[i]
        text = block.text.strip()
        if not text:
            continue

        # Strategy A: Check ANY short block against end patterns
        if len(text) < 150:
            for pat in _BG_END_PATTERNS:
                if pat.match(text):
                    bg_end = i
                    break
            if bg_end != len(blocks):
                break

        # Strategy B: Stop at heading blocks at same or higher level
        if block.type == "heading" and i > search_start:
            bg_level = blocks[bg_start].meta.get("level", 3)
            this_level = block.meta.get("level", 3)
            if this_level <= bg_level and len(text) > 5:
                bg_end = i
                break

    # Build the CanonicalSection from the detected range
    bg_blocks = blocks[bg_start:bg_end]
    content_blocks = [b for b in bg_blocks if b.text.strip()]
    bg_text = "\n\n".join(b.text for b in content_blocks)

    result = CanonicalSection(
        section_id="background",
        raw_title="Background of the Merger",
        blocks=content_blocks,
        start_block_idx=bg_start,
        end_block_idx=bg_end,
    )

    log(f"    Background ({label}): blocks [{bg_start}:{bg_end}] = {len(content_blocks)} content blocks, {len(bg_text):,} chars")
    first_text = content_blocks[0].text.strip(
    )[:100] if content_blocks else "N/A"
    last_text = content_blocks[-1].text.strip()[:100] if content_blocks else "N/A"
    log(f"    Background ({label}): FIRST: \"{first_text}...\"")
    log(f"    Background ({label}): LAST:  \"{last_text}...\"")
    if bg_end < len(blocks):
        next_text = blocks[bg_end].text.strip()[:100]
        log(f"    Background ({label}): NEXT:  \"{next_text}\"")

    return result


def route_tier(event: ChangeEvent) -> int:
    """Ensure correct tier assignment."""
    if event.category == "background":
        return 1
    if event.category in TIER1_CATEGORIES:
        return 1
    return 2


def compute_pairwise_diff(client: Anthropic,
                          old_doc: CanonicalDocument,
                          new_doc: CanonicalDocument,
                          deal_output_dir: str = None,
                          filing_id: str = None) -> List[ChangeEvent]:
    """Compute all change events between two filings.

    Uses direct comparison (10K/10Q approach): send both filings' relevant
    sections to one LLM call per category and ask what changed.

    deal_output_dir and filing_id: if provided, diagnostic files are written
    to deal_output_dir/{filing_id}_compare_diagnostic.txt etc.
    Otherwise falls back to OUTPUT_FOLDER.
    """
    old_label = get_form_label(old_doc.form_type)
    new_label = get_form_label(new_doc.form_type)

    print(f"\n  Diffing: {old_label} -> {new_label}")
    events = []

    # Determine diagnostic file paths
    if deal_output_dir and filing_id:
        diag_path = os.path.join(
            deal_output_dir, f"{filing_id}_compare_diagnostic.txt")
        bg_diag_path = os.path.join(
            deal_output_dir, f"{filing_id}_bg_diagnostic.txt")
    else:
        diag_path = os.path.join(
            OUTPUT_FOLDER, "compare_diagnostic_latest.txt")
        bg_diag_path = os.path.join(OUTPUT_FOLDER, "bg_diagnostic_latest.txt")

    # Clear comparison diagnostic file for this run
    with open(diag_path, "w") as df:
        df.write(f"Comparison diagnostics: {old_label} -> {new_label}\n")

    # 4.1: Direct category comparison (replaces independent extraction + diff + verification)
    comparison_categories = [
        "dates", "consideration", "financing", "sh_votes",
        "hsr", "regulatory", "closing", "termination",
    ]
    for cat in comparison_categories:
        cat_events = _compare_category_direct(client, cat, old_doc, new_doc,
                                              old_label, new_label,
                                              diag_path=diag_path)
        events.extend(cat_events)

    print(
        f"    Direct comparison: {len(events)} changes across {len(comparison_categories)} categories")
    print(f"    Raw LLM responses saved to: {diag_path}")

    # 4.1b: Other material changes (catch-all for deal-specific material items in general blocks)
    other_events = _compare_other_material(
        client, old_doc, new_doc, old_label, new_label,
        diag_path=diag_path)
    events.extend(other_events)

    # 4.2: Background diff (deterministic boundary detection + LLM comparison)
    _bg_diag_lines = []

    def _bg_log(msg):
        """Print and capture diagnostic line."""
        print(msg)
        _bg_diag_lines.append(msg)

    old_bg = _extract_bg_deterministic(old_doc, "old", bg_log=_bg_log)
    new_bg = _extract_bg_deterministic(new_doc, "new", bg_log=_bg_log)
    if old_bg or new_bg:
        bg_result = diff_background_with_llm(
            client, old_bg, new_bg, old_label, new_label)

        # Capture comparison diagnostics
        if "_diag" in bg_result:
            _bg_diag_lines.extend(bg_result["_diag"])

        if bg_result["has_changes"]:
            events.append(ChangeEvent(
                tier=1,
                category="background",
                change_type="updated",
                summary=bg_result["summary"],
            ))
        else:
            events.append(ChangeEvent(
                tier=1,
                category="background",
                change_type="other",
                summary=bg_result["summary"],
            ))

    # Write all background diagnostics to file
    with open(bg_diag_path, "w") as f:
        f.write("\n".join(_bg_diag_lines))
    print(f"    Background diagnostics saved to: {bg_diag_path}")

    # 4.3: Tier-2 section diffs (lightweight -- only recognized canonical sections)
    # Aggregate word counts by canonical ID to avoid double-counting sub-sections
    TIER2_SECTION_WHITELIST = {
        "risk_factors", "interests_conflicts", "projections",
        "appraisal_rights", "merger_agreement_summary", "summary",
        "questions_and_answers",
    }
    TIER2_LABELS = {
        "risk_factors": "Risk Factors",
        "interests_conflicts": "Interests & Conflicts",
        "projections": "Projections",
        "appraisal_rights": "Appraisal Rights",
        "merger_agreement_summary": "Merger Agreement Summary",
        "summary": "Summary",
        "questions_and_answers": "Q&A",
    }

    def _aggregate_wc(doc_obj, sid):
        """Get word count for a canonical section. Use the largest single section
        to avoid inflated counts from many sub-sections mapped to the same ID."""
        counts = [s.word_count for s in doc_obj.sections if s.section_id == sid]
        return max(counts) if counts else 0

    seen_tier2 = set()
    for section in new_doc.sections:
        if section.section_id in TIER1_SECTION_IDS:
            continue
        if section.section_id not in TIER2_SECTION_WHITELIST:
            continue
        if section.section_id in seen_tier2:
            continue
        seen_tier2.add(section.section_id)

        old_wc = _aggregate_wc(old_doc, section.section_id)
        new_wc = _aggregate_wc(new_doc, section.section_id)
        if old_wc > 200 and new_wc > 200 and abs(new_wc - old_wc) / old_wc > 0.25:
            label = TIER2_LABELS.get(section.section_id, section.raw_title)
            events.append(ChangeEvent(
                tier=2,
                category=section.section_id,
                change_type="updated",
                summary=f"{label}: {old_wc:,} -> {new_wc:,} words ({new_wc - old_wc:+,})",
            ))

    print(f"    Total change events: {len(events)} "
          f"(Tier 1: {sum(1 for e in events if e.tier == 1)}, "
          f"Tier 2: {sum(1 for e in events if e.tier == 2)})")

    return events
