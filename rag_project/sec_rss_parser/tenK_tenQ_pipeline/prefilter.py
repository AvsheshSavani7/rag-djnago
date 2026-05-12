"""Stage 1: Haiku pre-filter — cheap relevance classification before Sonnet scoring."""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import requests

from .config import HAIKU_CLASSIFY_PROMPT, HAIKU_BATCH_SIZE
from .models import DealContext, ParsedParagraph


KEYWORD_OVERRIDE_PATTERNS = [
    r"(?i)\bmerger\s+agreement\b",
    r"(?i)\bmerger\s+sub\b",
    r"(?i)\bclosing\s+condition",
    r"(?i)\btermination\s+fee",
    r"(?i)\bbreak[\s-]?up\s+fee",
    r"(?i)\breverse\s+termination",
    r"(?i)\bHart[\s-]Scott[\s-]Rodino\b",
    r"(?i)\bHSR\s+Act\b",
    r"(?i)\bHSR\b.*\b(?:waiting|expir|filing)\b",
    r"(?i)\bCFIUS\b",
    r"(?i)\bantitrust\b.*\b(?:approv|review|clear)",
    r"(?i)\bstockholder.*\b(?:vote|approv|adopt|meeting)\b",
    r"(?i)\bspecial\s+meeting\b",
    r"(?i)\beffective\s+time\b.*\bmerger\b",
    r"(?i)\bmerger\s+consideration\b",
    r"(?i)\bper\s+share\b.*\$\s*\d",
    r"(?i)\bFINRA\b.*\b(?:approv|Rule|change\s+in\s+control)\b",
    r"(?i)\binsurance\s+regulator",
    r"(?i)\b(?:DOJ|FTC|Department\s+of\s+Justice|Federal\s+Trade)\b",
    r"(?i)\b(?:proxy\s+statement|DEFM14A|PREM14A)\b",
    r"(?i)\bchange\s+(?:of|in)\s+control\b",
    r"(?i)\b(?:outside|drop[\s-]?dead)\s+date\b",
    r"(?i)\b(?:pending|proposed)\s+transaction\b",
    r"(?i)\bthe\s+transaction[s]?\b.*\b(?:close|closing|condition|approv|regulat)",
    r"(?i)\b(?:pending|proposed)\s+(?:merger|acquisition)\b",
    r"(?i)\bin\s+connection\s+with\s+the\s+(?:merger|pending|proposed)",
    r"(?i)\btransaction[\s-]+related\s+cost",
]


def _check_keyword_override(text: str, section: str = "") -> bool:
    """Return True if text matches any keyword override pattern.
    Section-aware: paragraphs in key deal sections (Notes, Risk Factors)
    have a broader set of patterns that trigger inclusion.
    """
    for pattern in KEYWORD_OVERRIDE_PATTERNS:
        if re.search(pattern, text):
            return True
    sec_lower = section.lower()
    in_key_section = (
        "risk factor" in sec_lower or "item 1a" in sec_lower
        or ("notes" in sec_lower and "financial" in sec_lower)
    )
    if in_key_section:
        section_patterns = [
            r"(?i)\brestructuring\s+initiative\b",
            r"(?i)\bstrategic\s+(?:review|alternative|initiative)",
            r"(?i)\boperating\s+model\s+(?:optim|transform)",
            r"(?i)\bseparation[\s-]+related",
            r"(?i)\b(?:strategic\s+review\s+committee|discontinued\s+effective)\b",
        ]
        for pattern in section_patterns:
            if re.search(pattern, text):
                return True
    return False


def classify_batch_haiku(batch: List[ParsedParagraph], deal: DealContext,
                         api_key: str, retries: int = 3) -> List[bool]:
    """Haiku classifies a batch as relevant/not-relevant. Returns list of bools."""
    para_block = ""
    for i, p in enumerate(batch):
        para_block += f"\n--- P{i+1} ---\n{p.text[:2000]}\n"

    user_msg = f"""DEAL: {deal.acquirer_company} acquiring {deal.target_company} ({deal.ticker})

PARAGRAPHS:
{para_block}

Classify each paragraph (P1 through P{len(batch)}). Return JSON array."""

    for attempt in range(retries + 1):
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 512,
                    "system": HAIKU_CLASSIFY_PROMPT,
                    "messages": [{"role": "user", "content": user_msg}],
                    "temperature": 0.0,
                },
                timeout=60,
            )
            response.raise_for_status()
            raw = response.json()["content"][0]["text"]
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
            json_str = (match.group(1) if match else raw).strip()
            arr_match = re.search(r'\[[\s\S]*\]', json_str)
            if arr_match:
                json_str = arr_match.group(0)
            results = json.loads(json_str)
            if isinstance(results, list):
                bools = []
                for r in results[:len(batch)]:
                    val = r.get("relevant", "no") if isinstance(r, dict) else "no"
                    bools.append(val.lower().strip() in ("yes", "true", "1"))
                while len(bools) < len(batch):
                    bools.append(True)
                return bools
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 401:
                raise RuntimeError("FATAL: Anthropic API key is invalid or expired.") from e
            if status == 429:
                wait = min(60, 5 * (2 ** attempt))
                print(f"    Rate limited (429), waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
        except (requests.exceptions.Timeout, json.JSONDecodeError, Exception) as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            print(f"    Warning: Haiku classify error: {e}")

    return [True] * len(batch)


def prefilter_with_haiku(paragraphs: List[ParsedParagraph], deal: DealContext,
                         api_key: str, batch_size: int = HAIKU_BATCH_SIZE,
                         max_workers: int = 10) -> List[ParsedParagraph]:
    """
    Stage 1: Haiku classifies all paragraphs as relevant/not-relevant.
    Keyword overrides force-include paragraphs with obvious merger terms.
    Returns only paragraphs that pass the filter.
    """
    scoreable = [p for p in paragraphs if not p.is_header]
    batches = [scoreable[i:i + batch_size] for i in range(0, len(scoreable), batch_size)]
    total_batches = len(batches)
    print(f"  [Stage 1] Haiku pre-filter: {len(scoreable)} paragraphs")
    print(f"  Batches ({batch_size}/batch): {total_batches} | workers: {max_workers}")

    haiku_relevant = set()
    keyword_relevant = set()

    for p in scoreable:
        if _check_keyword_override(p.text, section=p.section):
            keyword_relevant.add(p.index)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(classify_batch_haiku, batch, deal, api_key): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            batch = batches[idx]
            results = future.result()

            batch_yes = 0
            for para, is_relevant in zip(batch, results):
                if is_relevant:
                    haiku_relevant.add(para.index)
                    batch_yes += 1
            print(f"  Haiku batch {idx + 1}/{total_batches}: {batch_yes}/{len(batch)} relevant")

    all_relevant = haiku_relevant | keyword_relevant
    override_only = keyword_relevant - haiku_relevant

    passed = [p for p in paragraphs if p.index in all_relevant or p.is_header]

    print(f"  [Stage 1] Results: {len(haiku_relevant)} from Haiku, "
          f"{len(override_only)} added by keyword override, "
          f"{len(all_relevant)} total → Sonnet")

    return passed
