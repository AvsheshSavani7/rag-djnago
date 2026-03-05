"""Three-pass comparison engine with recency-prioritized matching."""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import requests

from .config import TIMING_PROMPT, REGULATORY_PROMPT, LEGAL_LANGUAGE_PROMPT
from .json_utils import parse_json_response


def build_labeled_block(excerpts: list, label_prefix: str, include_tier: bool = False) -> str:
    lines = []
    for i, p in enumerate(excerpts, 1):
        section = p.get("section", "Unknown")
        source = p.get("_filing_source", "")
        tier = ""
        if include_tier:
            is_flagged = p.get("timing_flag") or p.get("regulatory_flag")
            tier = " [CRITICAL]" if is_flagged else " [REVIEW]"
        header = f"[{label_prefix}-{i}]{tier}"
        if source:
            header += f" (Source: {source})"
        header += f" (Section: {section})"
        lines.append(f"{header}\n{p['text'][:2500]}")
    return "\n\n---\n\n".join(lines)


def call_claude_comparison(prompt: str, user_msg: str, anthropic_key: str,
                           max_tokens: int = 4000) -> dict:
    """Claude API call with robust JSON parsing."""
    for attempt in range(3):
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "claude-opus-4-6",
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": f"{prompt}\n\n{user_msg}"}],
                    "temperature": 0.1,
                },
                timeout=180,
            )
            response.raise_for_status()
            raw = response.json()["content"][0]["text"]
            return parse_json_response(raw)

        except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            print(f" FAILED: {e}")
        except Exception as e:
            print(f" ERROR: {e}")
            break

    return {"findings": [], "error": "Analysis failed"}


def run_three_passes(current_excerpts: list, prior_excerpts: list,
                     deal_meta: dict, anthropic_key: str) -> dict:
    ticker = deal_meta.get("ticker", "")
    target = deal_meta.get("target", "")
    acquirer = deal_meta.get("acquirer", "")

    current_block = build_labeled_block(current_excerpts, "CURRENT", include_tier=True)
    prior_block = build_labeled_block(prior_excerpts, "PRIOR", include_tier=False)

    user_msg = f"""CURRENT FILING PARAGRAPHS ({len(current_excerpts)} total):

{current_block}

{'='*60}

PRIOR FILING PARAGRAPHS ({len(prior_excerpts)} total):

{prior_block}"""

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
        changes = [f for f in findings if f.get("changed") or f.get("match_type") == "new"]

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


def merge_results(pass_results: dict, current_excerpts: list) -> list:
    para_lookup = {}
    for i, p in enumerate(current_excerpts, 1):
        ref = f"CURRENT-{i}"
        is_flagged = p.get("timing_flag") or p.get("regulatory_flag")
        para_lookup[ref] = {
            "paragraph_index": p.get("paragraph_index"),
            "section": p.get("section", ""),
            "relevance_score": p.get("relevance_score"),
            "timing_flag": p.get("timing_flag", False),
            "regulatory_flag": p.get("regulatory_flag", False),
            "tier": "critical" if is_flagged else "review",
            "text": p["text"],
            "timing": None, "regulatory": None, "legal_language": None,
            "overall_severity": "none",
        }

    severity_rank = {"significant": 3, "moderate": 2, "minor": 1, "none": 0}
    for pass_name, pass_data in pass_results.items():
        for finding in pass_data.get("findings", []):
            ref = finding.get("current_ref", "")
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


def run_recency_prioritized_comparison(
    current_excerpts: list,
    prior_filing_groups: list,   # [(label, excerpts), ...] ordered OLDEST first
    deal_meta: dict,
    anthropic_key: str,
) -> dict:
    """
    Compare current filing against prior filings from most recent to oldest.

    For each current paragraph:
    - Compare against the most recent prior filing first.
    - If found in that prior (any pass has match_type != 'new'): record finding, stop.
    - If not mentioned at all: treat as unchanged, stop.
    - If only 'new' match_type returned across all passes: not found → try next older prior.
    - If not found in ANY prior: record as a new disclosure.
    """
    combined = {pk: {"findings": [], "total_findings": 0, "changes_detected": 0}
                for pk in ["timing", "regulatory", "legal_language"]}

    pending_indices = list(range(len(current_excerpts)))
    first_not_found: dict = {}   # (original_idx, pk) → finding_copy

    for prior_label, prior_excerpts in reversed(prior_filing_groups):
        if not pending_indices:
            break
        if not prior_excerpts:
            continue

        current_subset = [current_excerpts[i] for i in pending_indices]
        print(f"    [Recency] {len(current_subset)} para(s) vs '{prior_label}' "
              f"({len(prior_excerpts)} prior excerpts)")

        pass_results = run_three_passes(current_subset, prior_excerpts, deal_meta, anthropic_key)

        still_pending = []

        for local_idx, original_idx in enumerate(pending_indices):
            local_ref = f"CURRENT-{local_idx + 1}"
            orig_ref  = f"CURRENT-{original_idx + 1}"

            pass_findings: dict = {}
            for pk in ["timing", "regulatory", "legal_language"]:
                for f in pass_results.get(pk, {}).get("findings", []):
                    if f.get("current_ref") == local_ref:
                        pass_findings[pk] = f
                        break

            if not pass_findings:
                # Not mentioned → unchanged vs this prior → settled
                first_not_found.pop((original_idx, "timing"), None)
                first_not_found.pop((original_idx, "regulatory"), None)
                first_not_found.pop((original_idx, "legal_language"), None)
                continue

            found_non_new = any(f.get("match_type", "new") != "new"
                                for f in pass_findings.values())

            if found_non_new:
                for pk, f in pass_findings.items():
                    if f.get("match_type", "new") != "new" or f.get("changed"):
                        f_copy = dict(f)
                        f_copy["current_ref"]     = orig_ref
                        f_copy["_prior_excerpts"] = prior_excerpts
                        f_copy["_prior_label"]    = prior_label
                        combined[pk]["findings"].append(f_copy)
                for pk in ["timing", "regulatory", "legal_language"]:
                    first_not_found.pop((original_idx, pk), None)
            else:
                for pk, f in pass_findings.items():
                    key = (original_idx, pk)
                    if key not in first_not_found:
                        f_copy = dict(f)
                        f_copy["current_ref"]     = orig_ref
                        f_copy["_prior_excerpts"] = prior_excerpts
                        f_copy["_prior_label"]    = prior_label
                        first_not_found[key] = f_copy
                still_pending.append(original_idx)

        pending_indices = still_pending

    # Commit truly-new disclosures
    for (original_idx, pk), f in first_not_found.items():
        combined[pk]["findings"].append(f)

    # Rebuild stats
    for pk in ["timing", "regulatory", "legal_language"]:
        findings = combined[pk]["findings"]
        combined[pk]["total_findings"] = len(findings)
        combined[pk]["changes_detected"] = sum(
            1 for f in findings if f.get("changed") or f.get("match_type") == "new"
        )

    return combined


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
                if not f or not (f.get("changed") or f.get("match_type") == "new"):
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
