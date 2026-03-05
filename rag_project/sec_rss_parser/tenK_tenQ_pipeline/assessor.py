"""Claude second-pass assessment: timing + regulatory flags."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import requests

from .config import OPUS_ASSESSMENT_PROMPT
from .json_utils import parse_json_response
from .models import DealContext, ParsedParagraph


def assess_with_claude(paragraphs: List[ParsedParagraph], deal: DealContext,
                       anthropic_api_key: str, threshold: int = 6,
                       max_workers: int = 10) -> List[ParsedParagraph]:
    relevant = [p for p in paragraphs if p.relevance_score >= threshold and not p.is_header]
    if not relevant:
        print("  No relevant paragraphs for Claude assessment.")
        return paragraphs

    print(f"  Assessing {len(relevant)} paragraphs with Claude (sonnet, {max_workers} workers)...")
    system_prompt = OPUS_ASSESSMENT_PROMPT.format(
        acquirer=deal.acquirer_company, target=deal.target_company,
        ticker=deal.ticker, deal_type=deal.deal_type,
        deal_value=deal.deal_value or "N/A", expected_close=deal.expected_close or "N/A",
    )
    total = len(relevant)

    def _assess_one(idx_para):
        i, para = idx_para
        prefix = f"  [{i+1}/{total}] ¶{para.index} (score={para.relevance_score})"
        user_msg = (
            f"SECTION: {para.section}\n"
            f"SCORE: {para.relevance_score}/10 ({para.category})\n\n"
            f"EXCERPT:\n{para.text[:3000]}"
        )

        for attempt in range(3):
            try:
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": anthropic_api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 500,
                        "messages": [{"role": "user", "content": f"{system_prompt}\n\n{user_msg}"}],
                        "temperature": 0.1,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                raw = response.json()["content"][0]["text"]
                result = parse_json_response(raw)

                timing = result.get("timing", {})
                para.timing_flag = timing.get("relevant", False)
                para.timing_assessment = timing.get("assessment", "")
                regulatory = result.get("regulatory", {})
                para.regulatory_flag = regulatory.get("relevant", False)
                para.regulatory_assessment = regulatory.get("assessment", "")

                flags = []
                if para.timing_flag: flags.append("TIMING")
                if para.regulatory_flag: flags.append("REGULATORY")
                suffix = f" -> {', '.join(flags)}" if flags else " -> done"
                print(f"{prefix}{suffix}")
                break
            except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                print(f"{prefix} -> FAILED: {e}")
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                    continue
                print(f"{prefix} -> ERROR: {e}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_assess_one, (i, para)): para
                   for i, para in enumerate(relevant)}
        for future in as_completed(futures):
            future.result()

    return paragraphs
