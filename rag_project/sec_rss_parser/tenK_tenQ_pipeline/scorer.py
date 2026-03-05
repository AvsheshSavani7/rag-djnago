"""Batch LLM scoring of paragraphs using Claude Sonnet."""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import requests

from .config import BATCH_SIZE, BATCH_SCORING_PROMPT
from .models import DealContext, ParsedParagraph


def score_batch(batch: List[ParsedParagraph], deal: DealContext,
                api_key: str, retries: int = 3) -> List[dict]:
    para_block = ""
    for i, p in enumerate(batch):
        para_block += f"\n--- P{i+1} ---\n{p.text[:2000]}\n"

    user_msg = f"""DEAL CONTEXT:
Acquirer: {deal.acquirer_company}
Target: {deal.target_company} ({deal.ticker})
Deal Type: {deal.deal_type} | Value: {deal.deal_value or 'N/A'}
Expected Close: {deal.expected_close or 'N/A'}

PARAGRAPHS TO SCORE:
{para_block}

Score each paragraph (P1 through P{len(batch)}) for merger relevance. Return JSON array."""

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
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 1024,
                    "system": BATCH_SCORING_PROMPT,
                    "messages": [{"role": "user", "content": user_msg}],
                    "temperature": 0.1,
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
            if isinstance(results, list) and len(results) >= len(batch):
                return results[:len(batch)]
            elif isinstance(results, list):
                while len(results) < len(batch):
                    results.append({"score": 3, "rationale": "Not scored", "category": "general", "key_info": []})
                return results
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 401:
                raise RuntimeError("FATAL: Anthropic API key is invalid or expired.") from e
            if status == 429:
                wait = min(60, 5 * (2 ** attempt))
                print(f"    Rate limited (429), waiting {wait}s before retry {attempt + 1}/{retries}...")
                time.sleep(wait)
                continue
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            print(f"    Warning: API error: {e}")
        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            print(f"    Warning: Timeout on batch scoring")
        except json.JSONDecodeError as e:
            if attempt < retries:
                time.sleep(1)
                continue
            print(f"    Warning: JSON parse error: {e}")
        except Exception as e:
            print(f"    Warning: {type(e).__name__}: {e}")
            break

    return [{"score": 3, "rationale": "Scoring failed", "category": "general", "key_info": []}
            for _ in batch]


def score_all_paragraphs(paragraphs: List[ParsedParagraph], deal: DealContext,
                         api_key: str, batch_size: int = BATCH_SIZE,
                         max_workers: int = 10) -> List[ParsedParagraph]:
    scoreable = [p for p in paragraphs if not p.is_header]
    batches = [scoreable[i:i + batch_size] for i in range(0, len(scoreable), batch_size)]
    total_batches = len(batches)
    print(f"  Paragraphs to score: {len(scoreable)}")
    print(f"  Batches ({batch_size}/batch): {total_batches} | workers: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(score_batch, batch, deal, api_key): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            batch = batches[idx]
            results = future.result()

            for para, result in zip(batch, results):
                para.relevance_score = int(result.get("score", result.get("relevance_score", 3)))
                para.rationale = result.get("rationale", "")
                para.category = result.get("category", "general")
                para.key_info = result.get("key_info", [])

            scores = [r.get("score", r.get("relevance_score", "?")) for r in results]
            high = sum(1 for s in scores if isinstance(s, int) and s >= 6)
            print(f"  Batch {idx + 1}/{total_batches}: scores={scores} ({high} relevant)")

    return paragraphs
