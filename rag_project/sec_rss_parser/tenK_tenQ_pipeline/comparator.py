"""Three-pass comparison engine with deterministic pre-matching and recency-prioritized strategy."""

import difflib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import requests

from .anthropic_debug_log import handle_anthropic_http_error, raise_if_anthropic_billing_blocked
from .config import (
    TIMING_PROMPT, REGULATORY_PROMPT, LEGAL_LANGUAGE_PROMPT, SINGLE_PASS_PROMPT,
)
from .json_utils import parse_json_response

# =============================================================================
# DETERMINISTIC PARAGRAPH PRE-MATCHING
# =============================================================================

def _normalize_for_matching(text: str) -> str:
    """Normalize text for similarity comparison: lowercase, collapse whitespace, strip dates."""
    t = text.lower()
    t = re.sub(r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s*\d{4}', ' ', t)
    t = re.sub(r'\d{4}-\d{2}-\d{2}', ' ', t)
    t = re.sub(r'\$\s*[\d,.]+\s*(?:million|billion|thousand)?', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def _section_overlap(sec_a: str, sec_b: str) -> bool:
    """Check if two section headers refer to the same filing section."""
    a = sec_a.lower().strip()[:80]
    b = sec_b.lower().strip()[:80]
    if not a or not b:
        return False
    item_a = re.search(r'item\s+\d+[a-z]?', a)
    item_b = re.search(r'item\s+\d+[a-z]?', b)
    if item_a and item_b:
        return item_a.group() == item_b.group()
    words_a = set(re.findall(r'\b\w{4,}\b', a))
    words_b = set(re.findall(r'\b\w{4,}\b', b))
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
    return overlap > 0.5


def _word_overlap_ratio(text_a: str, text_b: str) -> float:
    """Fraction of words unchanged between two texts (word-level diff)."""
    words_a = text_a.split()
    words_b = text_b.split()
    if not words_a or not words_b:
        return 0.0
    matcher = difflib.SequenceMatcher(None, words_a, words_b, autojunk=False)
    equal_words = sum(i2 - i1 for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag == 'equal')
    return equal_words / max(len(words_a), len(words_b))


def pre_match_excerpts(
    current_excerpts: list,
    prior_excerpts: list,
    match_threshold: float = 0.5,
    section_boost: float = 0.15,
    section_mismatch_penalty: float = 0.25,
    min_word_overlap: float = 0.25,
) -> tuple:
    """
    Deterministically match current paragraphs to prior paragraphs using text similarity
    and section location as the primary matching signal.

    Returns:
        matched_pairs: list of dicts with keys:
            current_idx, prior_idx, current_excerpt, prior_excerpt, similarity
        new_disclosures: list of current excerpts with no match in prior
    """
    if not current_excerpts or not prior_excerpts:
        return [], list(current_excerpts)

    current_normed = [_normalize_for_matching(e.get("text", "")) for e in current_excerpts]
    prior_normed = [_normalize_for_matching(e.get("text", "")) for e in prior_excerpts]

    scores = []
    for ci, c_text in enumerate(current_normed):
        c_section = current_excerpts[ci].get("section", "")
        for pi, p_text in enumerate(prior_normed):
            p_section = prior_excerpts[pi].get("section", "")
            ratio = difflib.SequenceMatcher(None, c_text.split(), p_text.split(), autojunk=False).ratio()

            sections_known = bool(c_section.strip()) and bool(p_section.strip())
            same_section = _section_overlap(c_section, p_section)

            if same_section:
                ratio = min(1.0, ratio + section_boost)
            elif sections_known:
                ratio = max(0.0, ratio - section_mismatch_penalty)

            scores.append((ratio, ci, pi))

    scores.sort(reverse=True)
    matched_current = set()
    matched_prior = set()
    matched_pairs = []

    for ratio, ci, pi in scores:
        if ratio < match_threshold:
            break
        if ci in matched_current or pi in matched_prior:
            continue

        raw_overlap = _word_overlap_ratio(
            current_excerpts[ci].get("text", ""),
            prior_excerpts[pi].get("text", ""),
        )
        if raw_overlap < min_word_overlap:
            continue

        matched_current.add(ci)
        matched_prior.add(pi)
        matched_pairs.append({
            "current_idx": ci,
            "prior_idx": pi,
            "current_excerpt": current_excerpts[ci],
            "prior_excerpt": prior_excerpts[pi],
            "similarity": round(ratio, 3),
        })

    matched_pairs.sort(key=lambda p: p["current_idx"])
    new_disclosures = [current_excerpts[i] for i in range(len(current_excerpts)) if i not in matched_current]

    return matched_pairs, new_disclosures


# =============================================================================
# LLM HELPERS
# =============================================================================

def call_claude_comparison(prompt: str, user_msg: str, anthropic_key: str,
                           max_tokens: int = 4000) -> dict:
    """Claude API call with robust JSON parsing."""
    for attempt in range(3):
        raise_if_anthropic_billing_blocked()
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": f"{prompt}\n\n{user_msg}"}],
                    "temperature": 0.1,
                },
                timeout=180,
            )
            response.raise_for_status()
            raw = response.json()["content"][0]["text"]
            return parse_json_response(raw)

        except requests.exceptions.HTTPError as e:
            if e.response is not None:
                handle_anthropic_http_error(
                    e.response, "10-K/10-Q comparison (Claude)")
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            print(f" FAILED: {e}")
        except requests.exceptions.Timeout:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            print(f" FAILED: timeout")
        except Exception as e:
            print(f" ERROR: {e}")
            break

    return {"findings": [], "error": "Analysis failed"}


def _build_pairs_block(matched_pairs: list, new_disclosures: list) -> str:
    """Build the text block sent to the LLM from pre-matched pairs + new disclosures."""
    lines = []
    for i, pair in enumerate(matched_pairs, 1):
        c_section = pair["current_excerpt"].get("section", "Unknown")
        c_text = pair["current_excerpt"]["text"][:2500]
        p_text = pair["prior_excerpt"]["text"][:2500]
        sim = pair["similarity"]
        lines.append(
            f"[PAIR-{i}] (Section: {c_section}) (Similarity: {sim})\n"
            f"CURRENT:\n{c_text}\n\n"
            f"PRIOR:\n{p_text}"
        )

    for i, excerpt in enumerate(new_disclosures, 1):
        section = excerpt.get("section", "Unknown")
        text = excerpt["text"][:2500]
        lines.append(
            f"[NEW-{i}] (Section: {section}) — No matching paragraph in prior filing\n"
            f"CURRENT:\n{text}"
        )

    separator = "\n\n" + "=" * 40 + "\n\n"
    return separator.join(lines)


# =============================================================================
# THREE-PASS COMPARISON (pre-matched pairs)
# =============================================================================

def run_three_passes(matched_pairs: list, new_disclosures: list,
                     deal_meta: dict, anthropic_key: str) -> dict:
    """Run timing, regulatory, and legal language analysis on pre-matched pairs."""
    ticker = deal_meta.get("ticker", "")
    target = deal_meta.get("target", "")
    acquirer = deal_meta.get("acquirer", "")

    pairs_block = _build_pairs_block(matched_pairs, new_disclosures)

    user_msg = f"""PRE-MATCHED PAIRS ({len(matched_pairs)} pairs, {len(new_disclosures)} new):

{pairs_block}"""

    passes = [
        ("TIMING", TIMING_PROMPT, 4000),
        ("REGULATORY", REGULATORY_PROMPT, 4000),
        ("LEGAL LANGUAGE", LEGAL_LANGUAGE_PROMPT, 8000),
    ]

    def _run_pass(pass_name, prompt_template, max_tok):
        print(f"  Pass: {pass_name} ...", end="", flush=True)
        prompt = prompt_template.format(ticker=ticker, target=target, acquirer=acquirer)
        response = call_claude_comparison(prompt, user_msg, anthropic_key, max_tokens=max_tok)

        findings = response.get("findings", [])
        changes = [f for f in findings if f.get("changed")]

        key = pass_name.lower().replace(" ", "_")
        result = {
            "findings": findings,
            "total_findings": len(findings),
            "changes_detected": len(changes),
        }

        if response.get("error"):
            print(f"  Pass: {pass_name} -> ERROR: {response['error']}")
        elif not findings:
            print(f"  Pass: {pass_name} -> no changes detected")
        else:
            sevs = [f.get("severity", "none") for f in findings]
            sig, mod, minor = sevs.count("significant"), sevs.count("moderate"), sevs.count("minor")
            parts = []
            if sig: parts.append(f"{sig} significant")
            if mod: parts.append(f"{mod} moderate")
            if minor: parts.append(f"{minor} minor")
            print(f"  Pass: {pass_name} -> {', '.join(parts) if parts else 'no changes'}")

        return key, result

    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_run_pass, *p): p[0] for p in passes}
        for future in as_completed(futures):
            key, result = future.result()
            results[key] = result

    return results


# =============================================================================
# MERGE RESULTS
# =============================================================================

def merge_results(pass_results: dict, matched_pairs: list, new_disclosures: list) -> list:
    """
    Merge findings from three passes into a unified list.
    Uses PAIR-N / NEW-N references from pre-matched comparison.
    Each result carries both current and prior text for redline rendering.
    """
    para_lookup = {}

    for i, pair in enumerate(matched_pairs, 1):
        ref = f"PAIR-{i}"
        ce = pair["current_excerpt"]
        pe = pair["prior_excerpt"]
        is_flagged = ce.get("timing_flag") or ce.get("regulatory_flag")
        para_lookup[ref] = {
            "paragraph_index": ce.get("paragraph_index"),
            "section": ce.get("section", ""),
            "relevance_score": ce.get("relevance_score"),
            "timing_flag": ce.get("timing_flag", False),
            "regulatory_flag": ce.get("regulatory_flag", False),
            "tier": "critical" if is_flagged else "review",
            "text": ce["text"],
            "prior_text": pe["text"],
            "similarity": pair["similarity"],
            "is_new": False,
            "timing": None, "regulatory": None, "legal_language": None,
            "overall_severity": "none",
        }

    for i, excerpt in enumerate(new_disclosures, 1):
        ref = f"NEW-{i}"
        is_flagged = excerpt.get("timing_flag") or excerpt.get("regulatory_flag")
        para_lookup[ref] = {
            "paragraph_index": excerpt.get("paragraph_index"),
            "section": excerpt.get("section", ""),
            "relevance_score": excerpt.get("relevance_score"),
            "timing_flag": excerpt.get("timing_flag", False),
            "regulatory_flag": excerpt.get("regulatory_flag", False),
            "tier": "critical" if is_flagged else "review",
            "text": excerpt["text"],
            "prior_text": None,
            "similarity": 0.0,
            "is_new": True,
            "timing": None, "regulatory": None, "legal_language": None,
            "overall_severity": "none",
        }

    severity_rank = {"significant": 3, "moderate": 2, "minor": 1, "none": 0}
    for pass_name, pass_data in pass_results.items():
        for finding in pass_data.get("findings", []):
            ref = finding.get("ref", "")
            if ref not in para_lookup:
                continue
            para_lookup[ref][pass_name] = finding
            sev = finding.get("severity", "none")
            if severity_rank.get(sev, 0) > severity_rank.get(para_lookup[ref]["overall_severity"], 0):
                para_lookup[ref]["overall_severity"] = sev

    results = []
    for ref, data in para_lookup.items():
        has_finding = any(data[k] is not None for k in ["timing", "regulatory", "legal_language"])
        if has_finding:
            results.append(data)

    results.sort(key=lambda x: -severity_rank.get(x["overall_severity"], 0))
    return results


# =============================================================================
# SINGLE-PASS COMPARISON
# =============================================================================

def run_single_pass_comparison(
    current_excerpts: list,
    prior_filing_groups: list,
    deal_meta: dict,
    anthropic_key: str,
) -> tuple:
    """
    Single LLM call comparison: send all current + prior paragraphs, let the LLM
    do matching, analysis, and quote selection in one shot.

    Returns same structure as run_recency_prioritized_comparison():
        (pass_results dict, matched_pairs list, new_disclosures list)
    """
    ticker = deal_meta.get("ticker", "")
    target = deal_meta.get("target", "")
    acquirer = deal_meta.get("acquirer", "")

    lines = ["CURRENT FILING PARAGRAPHS:"]
    for i, exc in enumerate(current_excerpts, 1):
        section = exc.get("section", "Unknown")
        text = exc["text"][:2500]
        lines.append(f"\n[CUR-{i}] (Section: {section})\n{text}")

    prior_idx = 0
    prior_lookup = {}
    for label, excerpts in prior_filing_groups:
        lines.append(f"\n\n{'='*40}\nPRIOR FILING: {label}\n{'='*40}")
        for exc in excerpts:
            prior_idx += 1
            section = exc.get("section", "Unknown")
            text = exc["text"][:2500]
            lines.append(f"\n[PRIOR-{prior_idx}] (Section: {section})\n{text}")
            prior_lookup[f"PRIOR-{prior_idx}"] = (label, exc)

    paragraphs_block = "\n".join(lines)

    prompt = SINGLE_PASS_PROMPT.format(ticker=ticker, target=target, acquirer=acquirer)
    user_msg = f"PARAGRAPHS ({len(current_excerpts)} current, {prior_idx} prior):\n\n{paragraphs_block}"

    print(f"  [Single-pass] Sending {len(current_excerpts)} current + {prior_idx} prior paragraphs to LLM...")

    response = call_claude_comparison(prompt, user_msg, anthropic_key, max_tokens=16000)

    findings = response.get("findings", [])
    print(f"  [Single-pass] LLM returned {len(findings)} findings")

    matched_pairs = []
    new_disclosures = []
    pass_results = {
        "timing": {"findings": [], "total_findings": 0, "changes_detected": 0},
        "regulatory": {"findings": [], "total_findings": 0, "changes_detected": 0},
        "legal_language": {"findings": [], "total_findings": 0, "changes_detected": 0},
    }

    pair_idx = 0
    new_idx = 0

    for finding in findings:
        cur_ref_str = finding.get("current_ref", "")
        prior_ref_str = finding.get("prior_ref")
        severity = finding.get("severity", "minor")

        cur_num = int(cur_ref_str.replace("CUR-", "")) if cur_ref_str.startswith("CUR-") else None
        if cur_num is None or cur_num < 1 or cur_num > len(current_excerpts):
            continue
        cur_excerpt = current_excerpts[cur_num - 1]

        if prior_ref_str and prior_ref_str in prior_lookup:
            pair_idx += 1
            ref = f"PAIR-{pair_idx}"
            prior_label, prior_excerpt = prior_lookup[prior_ref_str]

            matched_pairs.append({
                "current_idx": cur_num - 1,
                "prior_idx": 0,
                "current_excerpt": cur_excerpt,
                "prior_excerpt": prior_excerpt,
                "similarity": 0.0,
                "_prior_label": prior_label,
            })
        else:
            new_idx += 1
            ref = f"NEW-{new_idx}"
            prior_label = None
            new_disclosures.append(cur_excerpt)

        timing_analysis = (finding.get("timing_analysis") or "").strip()
        regulatory_analysis = (finding.get("regulatory_analysis") or "").strip()
        ll_analysis = (finding.get("legal_language_analysis") or "").strip()
        notable = finding.get("notable_changes") or []

        has_timing = bool(timing_analysis)
        has_regulatory = bool(regulatory_analysis)
        has_ll = bool(ll_analysis) or bool(notable)

        _base = {"ref": ref, "changed": True, "mismatch": False, "severity": severity}
        if prior_label:
            _base["_prior_label"] = prior_label

        if has_timing:
            pass_results["timing"]["findings"].append({**_base, "analysis": timing_analysis})

        if has_regulatory:
            pass_results["regulatory"]["findings"].append({**_base, "analysis": regulatory_analysis})

        if has_ll or notable:
            pass_results["legal_language"]["findings"].append({
                **_base, "analysis": ll_analysis, "notable_changes": notable,
            })

        if not has_timing and not has_regulatory and not has_ll and not notable:
            pass_results["legal_language"]["findings"].append({
                **_base,
                "analysis": finding.get("legal_language_analysis", "Change identified"),
                "notable_changes": notable,
            })

    for pk in ["timing", "regulatory", "legal_language"]:
        f_list = pass_results[pk]["findings"]
        pass_results[pk]["total_findings"] = len(f_list)
        pass_results[pk]["changes_detected"] = sum(1 for f in f_list if f.get("changed"))

    sev_counts = {}
    for finding in findings:
        s = finding.get("severity", "none")
        sev_counts[s] = sev_counts.get(s, 0) + 1
    parts = []
    for s in ["significant", "moderate", "minor"]:
        if sev_counts.get(s):
            parts.append(f"{sev_counts[s]} {s}")
    print(f"  [Single-pass] Result: {len(findings)} findings ({', '.join(parts)})")

    return pass_results, matched_pairs, new_disclosures


# =============================================================================
# RECENCY-PRIORITIZED COMPARISON
# =============================================================================

def run_recency_prioritized_comparison(
    current_excerpts: list,
    prior_filing_groups: list,
    deal_meta: dict,
    anthropic_key: str,
) -> tuple:
    """
    Compare current filing against prior filings from most recent to oldest.
    Uses deterministic pre-matching before sending to LLM.

    Returns:
        (pass_results dict, matched_pairs list, new_disclosures list)
    """
    pending_indices = list(range(len(current_excerpts)))

    all_matched_pairs = []
    all_pass_results = {pk: {"findings": [], "total_findings": 0, "changes_detected": 0}
                        for pk in ["timing", "regulatory", "legal_language"]}

    for prior_label, prior_excerpts in reversed(prior_filing_groups):
        if not pending_indices:
            break
        if not prior_excerpts:
            continue

        current_subset = [current_excerpts[i] for i in pending_indices]

        matched_pairs, unmatched = pre_match_excerpts(current_subset, prior_excerpts)

        print(f"    [Recency] vs '{prior_label}': {len(matched_pairs)} matched, "
              f"{len(unmatched)} unmatched of {len(current_subset)} pending")

        if not matched_pairs:
            continue

        pass_results = run_three_passes(matched_pairs, [], deal_meta, anthropic_key)

        # LLM mismatch validation: if 2+ passes flag mismatch=true for a PAIR,
        # the deterministic matcher paired paragraphs about different topics.
        mismatch_local_indices = set()
        for local_idx, pair in enumerate(matched_pairs):
            ref = f"PAIR-{local_idx + 1}"
            mismatch_votes = 0
            for pk in ["timing", "regulatory", "legal_language"]:
                for finding in pass_results.get(pk, {}).get("findings", []):
                    if finding.get("ref") == ref and finding.get("mismatch"):
                        mismatch_votes += 1
            if mismatch_votes >= 2:
                mismatch_local_indices.add(local_idx)
                print(f"    [Mismatch] PAIR-{local_idx + 1} rejected by LLM "
                      f"('{pair['current_excerpt']['text'][:50]}...' vs "
                      f"'{pair['prior_excerpt']['text'][:50]}...')")

        if mismatch_local_indices:
            for pk in ["timing", "regulatory", "legal_language"]:
                pass_results[pk]["findings"] = [
                    f for f in pass_results.get(pk, {}).get("findings", [])
                    if not (
                        f.get("ref", "").startswith("PAIR-") and
                        int(f["ref"].split("-")[1]) - 1 in mismatch_local_indices
                    )
                ]

            matched_pairs = [
                pair for local_idx, pair in enumerate(matched_pairs)
                if local_idx not in mismatch_local_indices
            ]

            old_to_new = {}
            new_idx = 1
            for local_idx in range(len(matched_pairs) + len(mismatch_local_indices)):
                if local_idx not in mismatch_local_indices:
                    old_to_new[local_idx + 1] = new_idx
                    new_idx += 1
            for pk in ["timing", "regulatory", "legal_language"]:
                for finding in pass_results.get(pk, {}).get("findings", []):
                    ref = finding.get("ref", "")
                    if ref.startswith("PAIR-"):
                        try:
                            old_num = int(ref.split("-")[1])
                            if old_num in old_to_new:
                                finding["ref"] = f"PAIR-{old_to_new[old_num]}"
                        except (IndexError, ValueError):
                            pass

        if not matched_pairs:
            continue

        ref_offset = len(all_matched_pairs)
        if ref_offset > 0:
            for pk in ["timing", "regulatory", "legal_language"]:
                for finding in pass_results.get(pk, {}).get("findings", []):
                    old_ref = finding.get("ref", "")
                    if old_ref.startswith("PAIR-"):
                        try:
                            num = int(old_ref.split("-")[1])
                            finding["ref"] = f"PAIR-{num + ref_offset}"
                        except (IndexError, ValueError):
                            pass

        for pk in ["timing", "regulatory", "legal_language"]:
            for finding in pass_results.get(pk, {}).get("findings", []):
                finding["_prior_label"] = prior_label

        for pair in matched_pairs:
            pair["_prior_label"] = prior_label
        all_matched_pairs.extend(matched_pairs)

        for pk in ["timing", "regulatory", "legal_language"]:
            all_pass_results[pk]["findings"].extend(
                pass_results.get(pk, {}).get("findings", [])
            )

        matched_current_indices = set()
        for pair in matched_pairs:
            local_idx = pair["current_idx"]
            matched_current_indices.add(pending_indices[local_idx])

        pending_indices = [i for i in pending_indices if i not in matched_current_indices]

    final_new_disclosures = [current_excerpts[i] for i in pending_indices]

    if final_new_disclosures:
        print(f"    [Recency] {len(final_new_disclosures)} new disclosures (no match in any prior)")
        new_pass_results = run_three_passes([], final_new_disclosures, deal_meta, anthropic_key)
        for pk in ["timing", "regulatory", "legal_language"]:
            all_pass_results[pk]["findings"].extend(
                new_pass_results.get(pk, {}).get("findings", [])
            )

    for pk in ["timing", "regulatory", "legal_language"]:
        findings = all_pass_results[pk]["findings"]
        all_pass_results[pk]["total_findings"] = len(findings)
        all_pass_results[pk]["changes_detected"] = sum(
            1 for f in findings if f.get("changed")
        )

    return all_pass_results, all_matched_pairs, final_new_disclosures

# =============================================================================
# EXEC SUMMARY BULLETS (unchanged from v1)
# =============================================================================

def summarize_findings_to_bullets(
    all_comparison_steps: list,
    deal,
    filing_labels: List[str],
    anthropic_key: str,
) -> List[dict]:
    """
    Calls Claude once with all findings and returns a synthesized, deduplicated list:
        [{"category": "Timing"|"Regulatory"|"Business/Risk", "bullet": str}, ...]
    """
    raw_findings = []
    for step in all_comparison_steps:
        for result in step.get("merged_results", []):
            section = result.get("section", "")[:100]

            flags = []
            parts = []
            for pk in ["timing", "regulatory", "legal_language"]:
                f = result.get(pk)
                if not f or not f.get("changed"):
                    continue
                flags.append(pk)
                analysis = (f.get("analysis") or "").strip()
                if analysis:
                    parts.append(analysis)
                notable = f.get("notable_changes") or []
                for change in notable:
                    old_p = (change.get("old_phrase") or "").strip()
                    new_p = (change.get("new_phrase") or "").strip()
                    interp = (change.get("interpretation") or "").strip()
                    if old_p or new_p:
                        parts.append(f'"{old_p}" → "{new_p}"')
                    if interp:
                        parts.append(interp)

            if flags and parts:
                seen = set()
                deduped = []
                for p in parts:
                    if p not in seen:
                        seen.add(p)
                        deduped.append(p)

                primary_cat = "Timing" if "timing" in flags else (
                    "Regulatory" if "regulatory" in flags else "Legal"
                )
                raw_findings.append({
                    "id": len(raw_findings),
                    "category": primary_cat,
                    "section": section,
                    "detail": " | ".join(deduped[:4]),
                })

    if not raw_findings:
        return []

    findings_text = "\n".join(
        f'[{i["id"]}] ({i["category"]}) {i["section"]}: {i["detail"]}'
        for i in raw_findings
    )

    prompt = f"""You are a merger arb analyst briefing a portfolio manager verbally. You have 20 seconds. Tell them what changed in this filing and why it matters.

DEAL: {deal.acquirer_company} / {deal.target_company} ({deal.ticker})
FILINGS: {' → '.join(filing_labels)}

Below are raw findings from a redline comparison. Many overlap or repeat. Synthesize them into what a PM actually needs to know — not a list of every change, just the ones that move the needle on timing, risk, or deal certainty.

OUTPUT RULES:
1. Write one "headline" — a single sentence (≤20 words) capturing the single most important takeaway. If nothing material changed, say so directly (e.g. "No material changes — deal tracking to plan").
2. Write 4–8 bullets, one sentence each (≤20 words). Each bullet = one distinct fact. No duplicates.
3. Category must be one of: "Timing", "Regulatory", "Business/Risk"
4. Plain English only. No legalese. No filler ("The filing indicates...", "This reflects..."). Start with the fact.
5. Omit anything trivial — boilerplate language updates, minor rewording with no substantive meaning.
6. Order bullets by importance: timing changes first, then regulatory, then business/risk.

Return a JSON object only — no prose, no markdown fences:
{{
  "headline": "...",
  "bullets": [
    {{"category": "Timing", "bullet": "..."}},
    {{"category": "Regulatory", "bullet": "..."}},
    {{"category": "Business/Risk", "bullet": "..."}}
  ]
}}

RAW FINDINGS:
{findings_text}"""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": anthropic_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 3000,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=60,
        )
        response.raise_for_status()
        raw_resp = response.json()["content"][0]["text"]
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_resp)
        json_str = fence_match.group(1).strip() if fence_match else raw_resp.strip()
        obj_start = json_str.find("{")
        arr_start = json_str.find("[")
        if obj_start != -1 and (arr_start == -1 or obj_start < arr_start):
            json_str = json_str[obj_start:]
        elif arr_start != -1:
            json_str = json_str[arr_start:]
        parsed = json.loads(json_str)
        if isinstance(parsed, dict):
            headline = parsed.get("headline", "")
            bullets_list = parsed.get("bullets", [])
        elif isinstance(parsed, list):
            headline = ""
            bullets_list = parsed
        else:
            headline = ""
            bullets_list = []
    except Exception as e:
        print(f"  [exec summary] Summarization failed: {e} — falling back to raw analysis")
        headline = ""
        bullets_list = []

    result = []
    if headline:
        result.append({"category": "__headline__", "bullet": headline})
    for item in bullets_list:
        if isinstance(item, dict) and "bullet" in item:
            result.append({
                "category": item.get("category", "Other"),
                "bullet": item["bullet"],
            })
    return result
