"""Robust JSON parsing helpers for LLM responses."""

import json
import re
from typing import Optional


def parse_json_response(raw: str) -> dict:
    """
    Parse JSON from LLM response with multiple fallback strategies.
    Handles truncated responses, markdown fences, and nested objects.
    """
    # Strategy 1: Extract from markdown code fence
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    json_str = (match.group(1) if match else raw).strip()

    # Strategy 2: Try direct parse first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Find balanced braces (handles nested objects correctly)
    result = _extract_balanced_json(json_str)
    if result is not None:
        return result

    # Strategy 4: Try the raw text directly
    result = _extract_balanced_json(raw)
    if result is not None:
        return result

    # Strategy 5: Truncated JSON repair — find last complete object/array
    repaired = _repair_truncated_json(json_str)
    if repaired is not None:
        return repaired

    return {"findings": [], "error": "JSON parse failed"}


def _extract_balanced_json(text: str) -> Optional[dict]:
    """Find the first balanced JSON object in text."""
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None


def _repair_truncated_json(text: str) -> Optional[dict]:
    """
    Attempt to repair truncated JSON by closing unclosed brackets/braces.
    Common when max_tokens cuts off the response mid-JSON.
    """
    start = text.find('{')
    if start == -1:
        return None

    json_str = text[start:]

    # Count unmatched quotes — if we're inside a string, close it
    quote_count = 0
    escape = False
    for c in json_str:
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"':
            quote_count += 1

    if quote_count % 2 != 0:
        last_quote = json_str.rfind('"')
        if last_quote > 0:
            json_str = json_str[:last_quote + 1]

    attempts = [
        json_str,
        re.sub(r',\s*\{[^}]*$', '', json_str),
        re.sub(r',\s*\{[^{}]*$', '', json_str),
    ]

    for attempt in attempts:
        cleaned = attempt.rstrip()
        cleaned = re.sub(r',\s*"[^"]*"\s*:\s*"[^"]*$', '', cleaned)
        cleaned = re.sub(r',\s*"[^"]*"\s*:\s*\[[^\]]*$', '', cleaned)
        cleaned = re.sub(r',\s*"[^"]*"\s*:\s*$', '', cleaned)
        cleaned = re.sub(r',\s*"[^"]*$', '', cleaned)
        cleaned = re.sub(r',\s*$', '', cleaned)

        depth_brace = 0
        depth_bracket = 0
        in_str = False
        esc = False
        for c in cleaned:
            if esc: esc = False; continue
            if c == '\\': esc = True; continue
            if c == '"': in_str = not in_str; continue
            if in_str: continue
            if c == '{': depth_brace += 1
            elif c == '}': depth_brace -= 1
            elif c == '[': depth_bracket += 1
            elif c == ']': depth_bracket -= 1

        if depth_brace >= 0 and depth_bracket >= 0:
            closed = cleaned + ']' * depth_bracket + '}' * depth_brace
            try:
                return json.loads(closed)
            except json.JSONDecodeError:
                continue

    return None
