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
    FACT_CATEGORY_MAP, _COMPARISON_PROMPTS,_COMPARISON_RULES,
    _CATEGORY_TO_SECTIONS, _CATEGORY_TO_TOPICS,
    BACKGROUND_DIFF_INTERPRET_PROMPT,_CATEGORY_KEYWORD_PATTERNS,
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
            overlap = len(old_words & new_words) / max(len(old_words), len(new_words))
            # Lower threshold for long descriptions -- LLM extraction adds/omits details
            overlap_threshold = 0.60 if len(n_old) > 80 or len(n_new) > 80 else 0.70
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
    removed = [old_strs[i] for i in range(len(old_strs)) if i not in matched_old]

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
        SKIP_FIELDS = {"gating_items"}  # duplicates regulatory + closing_conditions

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
                events.extend(_diff_list_values(key, old_val, new_val, category))
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
    text = re.sub(r'\d+\s*\n?\s*TABLE OF CONTENTS\s*\n?', ' ', text, flags=re.I)
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


def _normalize_for_bg_diff(text: str) -> str:
    """Normalize background text so only substantive differences survive diffing."""
    text = re.sub(r'\d+\s*\n?\s*TABLE OF CONTENTS\s*\n?', ' ', text, flags=re.I)
    text = re.sub(r'TABLE OF CONTENTS\s*', '', text, flags=re.I)
    text = re.sub(r'\n\s*\d{1,3}\s*\n', ' ', text)
    # Normalize ALL list numbering to (X)
    text = re.sub(r'\((?:[ivxlc]{1,6})\)', '(X)', text, flags=re.I)
    text = re.sub(r'\((?:[a-z])\)', '(X)', text, flags=re.I)
    text = re.sub(r'\((?:\d{1,2})\)', '(X)', text)
    # Normalize punctuation noise
    text = re.sub(r'(\b\d{4}),(\s)', r'\1\2', text)
    text = re.sub(r'[,;]\s*to\b', ' to', text)
    text = re.sub(r'[,;]\s*provided\b', ' provided', text)
    text = re.sub(r'[,;]\s*and\b', ' and', text)
    text = re.sub(r'[,;]\s*due\b', ' due', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _bg_paragraphs_from_section(section: CanonicalSection) -> List[str]:
    """Extract deduplicated, cleaned paragraphs from a Background section."""
    seen_fps = set()
    paragraphs = []
    for b in section.blocks:
        t = b.text.strip()
        if not t:
            continue
        if len(t) < 15 and re.match(r'^[A-Za-z]?-?\d+$', t):
            continue
        if t.startswith("TABLE OF CONTENTS"):
            continue
        if t.lower().startswith("background of the merger"):
            if len(t) < 40:
                continue
        fp = t[:120]
        if fp in seen_fps:
            continue
        seen_fps.add(fp)
        paragraphs.append(t)
    return paragraphs


def _is_ignorable_bg_change(old_text: str, new_text: str) -> bool:
    """Return True if ALL word changes between two paragraphs are cosmetic."""
    old_words = old_text.split()
    new_words = new_text.split()
    sm = difflib.SequenceMatcher(None, old_words, new_words)

    _IGNORABLE_WORD_PATTERNS = [
        re.compile(r'^\([ivxlca-z\d]{1,6}\),?$', re.I),
        re.compile(r'^\d{4},?$'),
        re.compile(r'^[,;]$'),
    ]
    _ENTITY_ALIASES = {
        frozenset(["citibank,", "n.a."]): frozenset(["citigroup", "global", "markets", "inc."]),
    }

    for tag, a1, a2, b1, b2 in sm.get_opcodes():
        if tag == "equal":
            continue
        old_span = " ".join(old_words[a1:a2]).lower()
        new_span = " ".join(new_words[b1:b2]).lower()
        old_match = any(p.match(old_span) for p in _IGNORABLE_WORD_PATTERNS)
        new_match = any(p.match(new_span) for p in _IGNORABLE_WORD_PATTERNS)
        if old_match and new_match:
            continue
        if not old_span and new_match:
            continue
        if not new_span and old_match:
            continue
        old_set = frozenset(old_span.split())
        new_set = frozenset(new_span.split())
        alias_match = any(
            (old_set == k and new_set == v) or (old_set == v and new_set == k)
            for k, v in _ENTITY_ALIASES.items()
        )
        if alias_match:
            continue
        return False
    return True


def diff_background_structural(old_section: Optional[CanonicalSection],
                                new_section: Optional[CanonicalSection],
                                old_label: str, new_label: str) -> Dict[str, Any]:
    """Deterministic normalized diff of Background sections. No LLM needed."""
    if old_section is None and new_section is not None:
        return {"has_changes": True, "summary": "Background section is entirely new in the later filing."}
    if new_section is None or old_section is None:
        return {"has_changes": False, "summary": "No material changes."}

    diag = []
    old_paras = _bg_paragraphs_from_section(old_section)
    new_paras = _bg_paragraphs_from_section(new_section)
    diag.append(f"Background structural diff: old={len(old_paras)} paragraphs, new={len(new_paras)} paragraphs")

    old_normalized = [_normalize_for_bg_diff(p) for p in old_paras]
    new_normalized = [_normalize_for_bg_diff(p) for p in new_paras]

    matcher = difflib.SequenceMatcher(None, old_normalized, new_normalized, autojunk=False)
    changes = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "insert":
            for j in range(j1, j2):
                changes.append({"type": "added", "text": new_paras[j], "detail": f"New paragraph in {new_label}"})
        elif tag == "delete":
            for i in range(i1, i2):
                changes.append({"type": "removed", "text": old_paras[i], "detail": f"Removed from {new_label}"})
        elif tag == "replace":
            old_chunk = old_paras[i1:i2]
            new_chunk = new_paras[j1:j2]
            old_norm = old_normalized[i1:i2]
            new_norm = new_normalized[j1:j2]
            paired = set()
            for oi, on in enumerate(old_norm):
                best_ratio, best_nj = 0, -1
                for nj, nn in enumerate(new_norm):
                    if nj in paired:
                        continue
                    ratio = difflib.SequenceMatcher(None, on, nn).ratio()
                    if ratio > best_ratio:
                        best_ratio, best_nj = ratio, nj
                if best_ratio > 0.85 and best_nj >= 0:
                    if old_norm[oi] == new_norm[best_nj]:
                        paired.add(best_nj)
                        continue
                    paired.add(best_nj)
                    diff_markup = word_level_diff(old_chunk[oi], new_chunk[best_nj])
                    changes.append({"type": "modified", "old_text": old_chunk[oi],
                                    "new_text": new_chunk[best_nj], "diff": diff_markup,
                                    "detail": "Modified paragraph"})
                else:
                    changes.append({"type": "removed", "text": old_chunk[oi],
                                    "detail": f"Removed from {new_label}"})
            for nj in range(len(new_chunk)):
                if nj not in paired:
                    changes.append({"type": "added", "text": new_chunk[nj],
                                    "detail": f"New paragraph in {new_label}"})

    # Filter cosmetic modifications
    filtered = []
    ignored = 0
    for c in changes:
        if c["type"] == "modified" and _is_ignorable_bg_change(c.get("old_text", ""), c.get("new_text", "")):
            ignored += 1
            continue
        filtered.append(c)
    if ignored:
        print(f"    Filtered out {ignored} ignorable modifications")
    changes = filtered

    # Cancel out moved paragraphs (same text added+removed)
    added_texts, removed_texts = {}, {}
    for i, c in enumerate(changes):
        key = _normalize_for_bg_diff(c.get("text", "")[:120])
        if c["type"] == "added":
            added_texts[key] = i
        elif c["type"] == "removed":
            removed_texts[key] = i
    moved = set()
    for key in added_texts:
        if key in removed_texts:
            moved.add(added_texts[key])
            moved.add(removed_texts[key])
    if moved:
        changes = [c for i, c in enumerate(changes) if i not in moved]
        print(f"    Filtered out {len(moved)//2} moved paragraphs")

    if not changes:
        print(f"    Background structural diff: no changes detected")
        return {"has_changes": False, "summary": "No material changes.", "_diag": diag}

    _IGNORABLE_DIFF_RE = re.compile(r'^\((?:[ivxlc]{1,4}|[a-h]|\d{1,2})\),?$', re.I)
    summary_parts = []
    seen_change_pairs = set()

    for c in changes:
        if c["type"] == "added":
            summary_parts.append(f"- {c['text'][:500]}  [NEW]")
        elif c["type"] == "removed":
            summary_parts.append(f"- {c['text'][:500]}  [REMOVED]")
        elif c["type"] == "modified":
            old_words = c.get("old_text", "").split()
            new_words = c.get("new_text", "").split()
            sm2 = difflib.SequenceMatcher(None, old_words, new_words)
            real_changes = []
            for op_tag, a1, a2, b1, b2 in sm2.get_opcodes():
                if op_tag == "equal":
                    continue
                old_span = " ".join(old_words[a1:a2])
                new_span = " ".join(new_words[b1:b2])
                if _IGNORABLE_DIFF_RE.match(old_span) and _IGNORABLE_DIFF_RE.match(new_span):
                    continue
                if not old_span and _IGNORABLE_DIFF_RE.match(new_span):
                    continue
                if not new_span and _IGNORABLE_DIFF_RE.match(old_span):
                    continue

                def _sent_start(words, pos):
                    limit = max(0, pos - 40)
                    for i in range(pos - 1, limit - 1, -1):
                        if i < 0:
                            return 0
                        if words[i].endswith(('.', '!', '?', '."', ".'", '.”')):
                            return i + 1
                    return limit

                def _sent_end(words, pos):
                    limit = min(len(words), pos + 40)
                    for i in range(pos, limit):
                        if words[i].endswith(('.', '!', '?', '."', ".'", '.”')):
                            return i + 1
                    return limit

                old_sentence = " ".join(old_words[_sent_start(old_words, a1):_sent_end(old_words, a2)])
                new_sentence = " ".join(new_words[_sent_start(new_words, b1):_sent_end(new_words, b2)])
                real_changes.append((old_span, new_span, old_sentence, new_sentence))

            for old_span, new_span, old_sentence, new_sentence in real_changes:
                pair_key = (old_span.lower(), new_span.lower())
                if pair_key in seen_change_pairs:
                    continue
                seen_change_pairs.add(pair_key)
                if old_span and new_span:
                    label = f'"{old_span}" changed to "{new_span}"'
                elif new_span:
                    label = f'addition of "{new_span}"'
                else:
                    label = f'removal of "{old_span}"'
                summary_parts.append(f"- {label}:")
                summary_parts.append(f"    Was: {old_sentence}")
                summary_parts.append(f"    Now: {new_sentence}")

    diag.append(f"RESULT: {len(changes)} changes found")
    print(f"    Background structural diff: {len(changes)} changes detected")
    for c in changes:
        ctype = c["type"].upper()
        snippet = c.get("text", c.get("old_text", ""))[:80]
        print(f"      [{ctype}] {snippet}...")

    return {"has_changes": True, "summary": "\n".join(summary_parts), "_diag": diag}


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

    inserted_text = "\n".join(f"- {s}" for s in diff_result["inserted"]) or "None"
    deleted_text = "\n".join(f"- {s}" for s in diff_result["deleted"]) or "None"

    modified_parts = []
    for old_s, new_s in diff_result["modified"]:
        diff_markup = word_level_diff(old_s, new_s)
        modified_parts.append(f"- OLD: {old_s}\n  NEW: {new_s}\n  DIFF: {diff_markup}")
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

def _get_keyword_matched_blocks(doc: CanonicalDocument, category: str,
                                 exclude_indices: set,
                                 max_chars: int = 5000) -> str:
    """Scan ALL blocks for category-specific keywords, excluding already-included blocks.

    This is the belt-and-suspenders layer: catches blocks that both Haiku classification
    AND the keyword override missed. Independent of topic tags — pure text search.
    """
    patterns = _CATEGORY_KEYWORD_PATTERNS.get(category, [])
    if not patterns:
        return ""

    matched_texts = []
    total = 0
    for b in doc.blocks:
        if b.index in exclude_indices:
            continue
        if not b.text.strip() or b.type == "heading":
            continue
        text_preview = b.text[:1500]
        for pat in patterns:
            if pat.search(text_preview):
                if total + len(b.text) > max_chars:
                    break
                matched_texts.append(b.text)
                total += len(b.text)
                exclude_indices.add(b.index)
                break

    return "\n\n".join(matched_texts) if matched_texts else ""

_SECTION_TITLE_FILTERS = {
    "dates": [
        r"notice.*(?:meeting|stockholder)", r"special meeting", r"record date",
        r"summary.*term", r"questions and answers",
    ],
    "consideration": [
        r"(?:merger|offer)\s*consideration", r"exchange ratio", r"treatment.*(?:securit|stock|unit)",
        r"summary.*term", r"terms of the (?:merger|offer|transaction)",
    ],
    "financing": [
        r"financ", r"source.*funds", r"commitment", r"summary.*term",
    ],
    "sh_votes": [
        r"(?:special|annual)\s*meeting", r"(?:stockholder|shareholder)\s*(?:approval|vote)",
        r"record date", r"quorum", r"(?:merger|transaction)\s*proposal",
        r"summary.*term", r"questions and answers",
    ],
    "hsr": [
        r"(?:HSR|Hart.Scott)", r"regulatory\s+approv", r"antitrust",
    ],
    "regulatory": [
        r"regulatory\s+approv", r"antitrust", r"(?:HSR|Hart.Scott)", r"(?:CFIUS|FIRB|SAMR|CMA|ACCC)",
        r"condition", r"consummation", r"status.*(?:regulatory|approv)",
    ],
    "conditions": [
        r"condition", r"closing", r"consummation",
    ],
    "closing": [
        r"closing", r"consummation", r"expected.*tim", r"outside.*date",
        r"summary.*term", r"questions and answers",
    ],
    "termination": [
        r"terminat", r"(?:break|termination)\s*fee", r"go.shop", r"no.shop",
    ],
}

_COMPILED_TITLE_FILTERS = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in _SECTION_TITLE_FILTERS.items()
}


def _section_title_relevant(section: "CanonicalSection", category: str) -> bool:
    """Check if a section's title is relevant to the given comparison category."""
    filters = _COMPILED_TITLE_FILTERS.get(category)
    if not filters:
        return True  # no filter defined = accept all
    title = section.raw_title
    for pat in filters:
        if pat.search(title):
            return True
    return False


def _gather_category_text(doc: CanonicalDocument, category: str, max_chars: int = 30000) -> str:
    """Gather text for comparison — deterministic layers first, topic tags as safety net.

    Priority order:
        Layer 1: Section-bounded text with relevant titles (deterministic).
        Layer 2: Keyword-matched blocks — regex patterns (deterministic).
        Layer 2.5: Termination fee amount scan (termination category only).
        Layer 3: Topic-tagged blocks — NON-DETERMINISTIC safety net (< 3K chars only).
    """
    section_ids = _CATEGORY_TO_SECTIONS.get(category, [])
    topics_needed = _CATEGORY_TO_TOPICS.get(category, [])
    parts = []
    total = 0
    used_block_indices = set()

    # Per-section cap (termination sections are long, need higher cap for fee amounts)
    _CATEGORY_SECTION_CAPS = {"termination": 20000, "regulatory": 15000}
    PER_SECTION_CAP = _CATEGORY_SECTION_CAPS.get(category, 10000)

    # === LAYER 1: Section-bounded text (deterministic) ===
    primary_ids = [sid for sid in section_ids if sid not in ("merger_agreement_summary", "summary")]
    sections = get_sections_by_ids(doc, primary_ids)
    relevant_sections = [s for s in sections if _section_title_relevant(s, category)]

    for s in relevant_sections:
        remaining = max_chars - total
        if remaining < 2000:
            break
        section_parts = []
        section_total = 0
        cap = min(PER_SECTION_CAP, remaining)
        for b in s.blocks:
            t = b.text.strip()
            if not t or b.type == "heading":
                continue
            if t.startswith("TABLE OF CONTENTS"):
                continue
            if len(t) < 15 and re.match(r'^[A-Za-z]?-?\d+$', t):
                continue
            if section_total + len(t) > cap:
                break
            section_parts.append(t)
            section_total += len(t)
            used_block_indices.add(b.index)
        if section_parts:
            parts.append("\n\n".join(section_parts))
            total += section_total

    # === LAYER 2: Keyword-matched blocks (deterministic) ===
    kw_budget = min(max_chars - total, 15000) if total < max_chars - 2000 else 0
    if kw_budget > 2000:
        exclude_indices = set(used_block_indices)
        kw_text = _get_keyword_matched_blocks(doc, category, exclude_indices, max_chars=kw_budget)
        if kw_text.strip():
            parts.append(kw_text)
            total += len(kw_text)
            used_block_indices.update(exclude_indices)

    # === LAYER 2.5: Targeted fee-amount scan for termination ===
    if category == "termination" and total < max_chars - 2000:
        gathered_so_far = "\n".join(parts)
        _HAS_LARGE_AMT = re.compile(r'\$[\d,]{7,}')
        if not _HAS_LARGE_AMT.search(gathered_so_far):
            _TERM_FEE_PHRASE = re.compile(r'(?:Company|Parent|Reverse)\s+Termination\s+Fee', re.IGNORECASE)
            candidates = []
            for b in doc.blocks:
                if b.index in used_block_indices:
                    continue
                t = b.text.strip()
                if not t or b.type == "heading":
                    continue
                if _TERM_FEE_PHRASE.search(t) and _HAS_LARGE_AMT.search(t):
                    candidates.append((b.index, t))
            candidates.sort(key=lambda x: len(x[1]))
            amt_budget = min(max_chars - total, 10000)
            amt_total = 0
            for idx, t in candidates:
                if amt_total + len(t) > amt_budget:
                    continue
                parts.append(t)
                amt_total += len(t)
                total += len(t)
                used_block_indices.add(idx)
            if amt_total:
                print(f"    Layer 2.5: added {amt_total:,} chars of fee-amount blocks for {category}")

    # === LAYER 3: Topic-tagged blocks (NON-DETERMINISTIC safety net) ===
    # Only used when deterministic layers produced < 3K chars
    SAFETY_NET_THRESHOLD = 3000
    if total < SAFETY_NET_THRESHOLD and topics_needed:
        haiku_budget = min(max_chars - total, 15000)
        haiku_text = _get_blocks_by_topic(doc, topics_needed,
                                           max_chars=haiku_budget,
                                           exclude_indices=used_block_indices)
        if haiku_text.strip():
            parts.append(haiku_text)
            total += len(haiku_text)

    return "\n\n---\n\n".join(parts) if parts else ""


def _parse_comparison_response(raw: str, category: str) -> List[ChangeEvent]:
    """Parse LLM JSON response into ChangeEvents."""
    text = raw.strip()

    # Strategy 1: Extract JSON from markdown fences -- use LAST block (LLM sometimes
    # self-corrects mid-response, so the final block is the most accurate)
    fence_matches = list(re.finditer(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL))
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


def _haiku_has_changes(client: Anthropic, category: str,
                       old_text: str, new_text: str,
                       old_label: str, new_label: str) -> bool:
    """Cheap Haiku pre-check: are there material differences in this category?

    Sends truncated text (first 4K chars per side) to Haiku asking yes/no.
    Returns True if changes likely exist (or on any error — fail open).
    """
    PREVIEW_CHARS = 4000
    prompt = f"""Compare these two filing excerpts for the "{category}" category.
Are there ANY material differences in substance (new items, changed values, removed items)?
Ignore formatting, word order, and boilerplate differences.

EARLIER FILING ({old_label}) — first {PREVIEW_CHARS} chars:
{old_text[:PREVIEW_CHARS]}

LATER FILING ({new_label}) — first {PREVIEW_CHARS} chars:
{new_text[:PREVIEW_CHARS]}

Reply with ONLY "yes" or "no"."""

    try:
        from .config import MODEL_HAIKU
        response = client.messages.create(
            model=MODEL_HAIKU,
            max_tokens=8,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.content[0].text.strip().lower()
        return "yes" in answer
    except Exception as e:
        print(f"    Haiku pre-check ({category}): failed ({e}) — assuming changes exist")
        return True  # fail open


def _suppress_false_new_events(events: List[ChangeEvent],
                                base_summary_dict: dict) -> List[ChangeEvent]:
    """Post-validation: suppress [NEW] events whose key values already appear in the base summary."""
    if not base_summary_dict:
        return events

    full_summary = " ".join(v for v in base_summary_dict.values() if v)

    _DATE_PAT = re.compile(r'(?:January|February|March|April|May|June|July|August|'
                            r'September|October|November|December)\s+\d{1,2},?\s+\d{4}')
    _DOLLAR_PAT = re.compile(r'\$[\d,.]+(?:\s*(?:billion|million))?')
    _PCT_PAT = re.compile(r'\d+(?:\.\d+)?%')
    _BIG_NUM_PAT = re.compile(r'(?<!\$)\b\d{1,3}(?:,\d{3})+\b')

    def _extract_values(text: str) -> set:
        vals = set()
        for pat in [_DATE_PAT, _DOLLAR_PAT, _PCT_PAT, _BIG_NUM_PAT]:
            for m in pat.finditer(text):
                vals.add(m.group().strip().rstrip(","))
        return vals

    kept = []
    for event in events:
        if event.change_type != "newly_disclosed":
            kept.append(event)
            continue

        event_text = " ".join(filter(None, [event.field, event.new_value, event.summary]))
        event_vals = _extract_values(event_text)

        if not event_vals:
            kept.append(event)
            continue

        date_vals = {d.strip().rstrip(",") for d in _DATE_PAT.findall(event_text)}
        non_date_vals = event_vals - date_vals

        found_specific = False
        match_val = None
        for val in non_date_vals:
            if val in full_summary:
                found_specific = True
                match_val = val
                break

        if found_specific:
            print(f"    [suppress] False NEW suppressed: {event.field} "
                  f"(value '{match_val}' found in base summary)")
            continue

        if non_date_vals:
            kept.append(event)
        elif date_vals:
            if all(d in full_summary for d in date_vals) and len(date_vals) >= 2:
                print(f"    [suppress] False NEW suppressed: {event.field} "
                      f"(all {len(date_vals)} dates found in base summary)")
            else:
                kept.append(event)
        else:
            kept.append(event)

    return kept


def _compare_category_direct(client: Anthropic, category: str,
                              old_doc: CanonicalDocument, new_doc: CanonicalDocument,
                              old_label: str, new_label: str,
                              diag_path: str = None,
                              base_summary_section: str = "") -> List[ChangeEvent]:
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
        print(f"    Compare ({category}): no text in either filing -- skipping")
        return []

    if not old_text.strip():
        old_text = "[Section not found in this filing]"
    if not new_text.strip():
        new_text = "[Section not found in this filing]"

    print(f"    Compare ({category}): old={len(old_text):,} chars, new={len(new_text):,} chars")

    # Haiku pre-check: skip expensive Sonnet call if no material differences.
    # Skip when one side is missing (guaranteed new/removed content).
    # Skip for cross-form comparisons (PREM→DEFM) — almost always have real changes.
    old_base_form = old_doc.form_type.replace("/A", "").strip()
    new_base_form = new_doc.form_type.replace("/A", "").strip()
    same_form_family = (old_base_form == new_base_form)

    if (same_form_family and old_text.strip() and new_text.strip()
            and "[Section not found" not in old_text and "[Section not found" not in new_text):
        if not _haiku_has_changes(client, category, old_text, new_text, old_label, new_label):
            print(f"    Compare ({category}): Haiku says no changes — skipping Sonnet call")
            return []
    elif not same_form_family:
        print(f"    Compare ({category}): cross-form ({old_doc.form_type}→{new_doc.form_type}) — skipping Haiku pre-check")

    prompt = prompt_template.format(
        old_label=old_label,
        new_label=new_label,
        old_text=old_text,
        new_text=new_text,
    )

    # Prepend known deal terms from base summary to reduce false [NEW]
    if base_summary_section:
        summary_preamble = (
            f"KNOWN DEAL TERMS (from {old_label} summary — already disclosed):\n"
            f"{base_summary_section}\n\n"
            f"Use the above as context for what was previously disclosed.\n"
            f"- If a specific value (date, dollar amount, percentage) appears above with its actual value, "
            f"do NOT flag the same value as [NEW] in the later filing.\n"
            f"- However, if the summary says 'not disclosed', 'truncated', 'placeholder', '[•]', "
            f"'amount not specified', or similar — that item is NOT yet known. "
            f"If the later filing provides the actual value, you MUST report it as newly_disclosed.\n\n"
        )
        prompt = summary_preamble + prompt

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
                df.write(f"CATEGORY: {category} | old={len(old_text):,} chars, new={len(new_text):,} chars\n")
                df.write(f"RAW RESPONSE:\n{answer}\n")

        events = _parse_comparison_response(answer, category)
        print(f"    Compare ({category}): {len(events)} changes detected")
        if events:
            for i, e in enumerate(events[:3]):
                field_preview = (e.field or "")[:50]
                val_preview = (e.new_value or e.old_value or e.summary or "")[:60]
                print(f"      [{i}] {e.change_type}: {field_preview} = {val_preview}")
            if len(events) > 3:
                print(f"      ... and {len(events) - 3} more")
        return events
    except Exception as e:
        print(f"    Compare ({category}): LLM call failed: {e}")
        return []

def _compare_other_material(client: Anthropic,
                             old_doc: CanonicalDocument,
                             new_doc: CanonicalDocument,
                             old_label: str, new_label: str) -> List[ChangeEvent]:
    """Side-by-side comparison of 'general' blocks for material new disclosures.

    Same approach as structured categories: send old and new text to the LLM
    together, let the LLM decide what's genuinely new vs already present.
    """
    # Topics covered by structured comparison categories — exclude from "other"
    _STRUCTURED_TOPICS = {"dates", "consideration", "financing", "sh_approval",
                          "hsr", "regulatory", "closing", "termination", "background"}

    def _is_other_block(b):
        if not b.text.strip() or b.type == "heading":
            return False
        if len(b.text.split()) < 20:
            return False
        return b.topic == "general" or b.topic not in _STRUCTURED_TOPICS

    old_general = [b.text.strip() for b in old_doc.blocks if _is_other_block(b)]
    new_general = [b.text.strip() for b in new_doc.blocks if _is_other_block(b)]

    if not new_general:
        print(f"    Compare (other_material): no general blocks in new filing — skipping")
        return []

    per_side_budget = 30000

    def _fill_budget(texts, budget):
        out, total = [], 0
        for t in texts:
            if total + len(t) > budget:
                continue
            out.append(t)
            total += len(t)
        return "\n\n---\n\n".join(out), total, len(out)

    old_text, old_chars, old_count = _fill_budget(old_general, per_side_budget)
    new_text, new_chars, new_count = _fill_budget(new_general, per_side_budget)

    print(f"    Compare (other_material): old={old_chars:,} chars ({old_count} blocks), "
          f"new={new_chars:,} chars ({new_count} blocks)")

    prompt = f"""\
You are comparing two versions of an SEC merger filing to identify material, deal-specific disclosures in the LATER filing ({new_label}) that are GENUINELY NEW — not present in any form in the EARLIER filing ({old_label}).

The text below contains "general" paragraphs (not already covered by dates, consideration, financing, shareholder approval, HSR, regulatory, closing conditions, termination fees, or background sections).

EARLIER FILING ({old_label}) — GENERAL PARAGRAPHS:
{old_text}

LATER FILING ({new_label}) — GENERAL PARAGRAPHS:
{new_text}

Compare the two sets of paragraphs. Only flag items from the LATER filing that meet ALL THREE criteria:
(a) GENUINELY NEW: The disclosure does NOT appear in the earlier filing in any form — not even with placeholder values like [•]. If the earlier filing contains the same information with blanks/placeholders, it is NOT new.
(b) DEAL-SPECIFIC: Directly about this transaction, the parties, or consequences of the merger — not generic market/industry risks.
(c) MATERIAL: Could meaningfully affect deal value, timing, certainty, or post-closing operations — not trivial or immaterial.

Examples of what TO flag:
- New litigation, regulatory investigations, or stockholder demands related to the deal
- Material tax liabilities or structuring issues arising from the merger
- Fairness opinion valuation details (specific multiples, ranges, methodologies) not in the earlier filing
- New equity compensation or severance arrangements triggered by the merger
- Financial advisor fee structures or conflict disclosures not in the earlier filing

Examples of what NOT to flag:
- ANY content that appears in the earlier filing, even with different formatting or filled-in placeholders
- Procedural/administrative disclosures (proxy submission, meeting logistics, passcodes, registration deadlines, document request deadlines, proxy statement dates, virtual meeting URLs)
- Current stock/market prices, trading data, or price comparisons
- Ownership percentages or share counts where only a placeholder was filled in
- Terms from the merger agreement (termination fees, financing commitments, regulatory requirements, closing conditions, go-shop provisions, etc.)
- Standard definitive proxy sections that are always absent from preliminary filings: U.S. federal income tax consequences, appraisal/dissenters' rights procedures, "who can help answer your questions" sections, information about the acquirer/parent entity (founding date, AUM, address)
- Premium calculations over pre-announcement stock price (these are standard disclosure, not deal term changes)

{_COMPARISON_RULES}

Format:
- New material item: {{"field": "short descriptive name", "type": "new", "value": "concise description of the material disclosure (1-2 sentences)"}}

Return a JSON array. If nothing meets ALL THREE criteria, return: []"""

    try:
        response = client.messages.create(
            model=MODEL_STANDARD,
            max_tokens=4000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.content[0].text.strip()

        diag_path = os.path.join(OUTPUT_FOLDER, "compare_diagnostic_latest.txt")
        with open(diag_path, "a") as df:
            df.write(f"\n{'='*60}\n")
            df.write(f"CATEGORY: other_material | old={old_chars:,}, new={new_chars:,} chars\n")
            df.write(f"RAW RESPONSE:\n{answer}\n")

        events = _parse_comparison_response(answer, "other_material")
        print(f"    Compare (other_material): {len(events)} material items flagged")
        if events:
            for i, e in enumerate(events[:3]):
                print(f"      [{i}] {e.change_type}: {(e.field or '')[:50]} = {(e.new_value or e.summary or '')[:60]}")
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
    re.compile(r"(?:^|\n)\s*Background of the Mergers?\s*(?:\n|$)", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*Background of the Transactions?\s*(?:\n|$)", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*Background of the Offers?\s*(?:\n|$)", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*Background of the Acquisitions?\s*(?:\n|$)", re.IGNORECASE),
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
            m = re.match(r"^(?:Anchor\s+)?Background of the Mergers?\s+(?:The |In |On |During )", text, re.IGNORECASE)
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
    first_text = content_blocks[0].text.strip()[:100] if content_blocks else "N/A"
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
                           filing_id: str = None,
                           base_summary_text: str = "",
                           category_filter: List[str] = None) -> List[ChangeEvent]:
    """Compute all change events between two filings.

    Uses direct comparison (10K/10Q approach): send both filings' relevant
    sections to one LLM call per category and ask what changed.

    base_summary_text: if provided, injected into comparison prompts to reduce
    false [NEW] events and used for post-validation suppression.

    category_filter: if provided, only these categories are compared.

    deal_output_dir and filing_id: if provided, diagnostic files are written
    to deal_output_dir/{filing_id}_compare_diagnostic.txt etc.
    Otherwise falls back to OUTPUT_FOLDER.
    """
    old_label = get_form_label(old_doc.form_type)
    new_label = get_form_label(new_doc.form_type)

    # Parse base summary into per-category dict for prompt injection + validation
    base_summary_dict = {}
    if base_summary_text:
        from .report_writer import _summary_text_to_category_dict
        base_summary_dict = _summary_text_to_category_dict(base_summary_text)
        print(f"    Base summary available: {len(base_summary_dict)} sections")

    print(f"\n  Diffing: {old_label} -> {new_label}")
    events = []

    # Determine diagnostic file paths
    if deal_output_dir and filing_id:
        diag_path = os.path.join(deal_output_dir, f"{filing_id}_compare_diagnostic.txt")
        bg_diag_path = os.path.join(deal_output_dir, f"{filing_id}_bg_diagnostic.txt")
    else:
        diag_path = os.path.join(OUTPUT_FOLDER, "compare_diagnostic_latest.txt")
        bg_diag_path = os.path.join(OUTPUT_FOLDER, "bg_diagnostic_latest.txt")

    # Clear comparison diagnostic file for this run
    with open(diag_path, "w") as df:
        df.write(f"Comparison diagnostics: {old_label} -> {new_label}\n")

    # 4.1: Direct category comparison
    all_categories = [
        "dates", "consideration", "financing", "sh_votes",
        "hsr", "regulatory", "conditions", "closing", "termination",
    ]
    comparison_categories = [c for c in all_categories if c in category_filter] if category_filter else all_categories
    if category_filter:
        print(f"    Category filter: {comparison_categories}")

    for cat in comparison_categories:
        cat_summary = base_summary_dict.get(cat, "")
        cat_events = _compare_category_direct(client, cat, old_doc, new_doc,
                                               old_label, new_label,
                                               diag_path=diag_path,
                                               base_summary_section=cat_summary)
        events.extend(cat_events)

    # Post-validate: suppress false [NEW] events using base summary
    if base_summary_dict:
        pre_count = len([e for e in events if e.change_type == "newly_disclosed"])
        events = _suppress_false_new_events(events, base_summary_dict)
        post_count = len([e for e in events if e.change_type == "newly_disclosed"])
        if pre_count != post_count:
            print(f"    Post-validation: {pre_count - post_count} false [NEW] suppressed")

    # Cross-category dedup: remove events with identical (old_value, new_value) across categories
    _CATEGORY_PRIORITY = {
        "dates": 1, "consideration": 2, "financing": 3, "sh_votes": 4,
        "hsr": 5, "regulatory": 6, "conditions": 7, "closing": 8, "termination": 9,
    }
    seen_values = {}
    dedup_remove = set()
    for idx, e in enumerate(events):
        was_val = (e.old_value or "").strip().lower().rstrip('.,;: ')[:80]
        now_val = (e.new_value or "").strip().lower().rstrip('.,;: ')[:80]
        if not was_val and not now_val:
            continue
        key = (was_val, now_val)
        pri = _CATEGORY_PRIORITY.get(e.category, 99)
        if key in seen_values:
            prev_idx, prev_pri = seen_values[key]
            if pri < prev_pri:
                dedup_remove.add(prev_idx)
                seen_values[key] = (idx, pri)
            else:
                dedup_remove.add(idx)
        else:
            seen_values[key] = (idx, pri)
    if dedup_remove:
        removed_cats = [events[i].category for i in dedup_remove]
        events = [e for i, e in enumerate(events) if i not in dedup_remove]
        print(f"    Cross-category dedup: removed {len(dedup_remove)} duplicate(s) from {removed_cats}")

    print(f"    Direct comparison: {len(events)} changes across {len(comparison_categories)} categories")
    print(f"    Raw LLM responses saved to: {diag_path}")

    # 4.1b: Other material changes (catch-all for deal-specific material items in general blocks)
    if not category_filter or "other_material" in category_filter:
        other_events = _compare_other_material(client, old_doc, new_doc, old_label, new_label)
        events.extend(other_events)

    # 4.2: Background diff (deterministic — no LLM cost)
    if category_filter and "background" not in category_filter:
        print(f"    Skipping background (not in category filter)")
        return events

    _bg_diag_lines = []

    def _bg_log(msg):
        """Print and capture diagnostic line."""
        print(msg)
        _bg_diag_lines.append(msg)

    old_bg = _extract_bg_deterministic(old_doc, "old", bg_log=_bg_log)
    new_bg = _extract_bg_deterministic(new_doc, "new", bg_log=_bg_log)
    if old_bg or new_bg:
        bg_result = diff_background_structural(old_bg, new_bg, old_label, new_label)

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
