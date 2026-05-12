"""Fetch deal context from Perplexity API."""

import json
import re

import requests

from .config import DEAL_SETUP_PROMPT
from .models import DealContext


def fetch_deal_context(ticker: str, api_key: str, company_name: str = "") -> DealContext:
    print(f"  Querying Perplexity for {ticker}...")
    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "sonar",
            "messages": [
                {"role": "system", "content": DEAL_SETUP_PROMPT},
                {"role": "user", "content": f"""{ticker} {f'({company_name}) ' if company_name else ''}merger acquisition announcement 2024 2025 2026

Search for recent merger, acquisition, or takeover news involving the US-listed company {f'{company_name} ' if company_name else ''}with ticker {ticker} (SEC filer). Include unsolicited bids or hostile proposals if any. Find:
1. Who is acquiring {ticker}, or who has proposed to acquire them? What company or consortium?
2. What is the merger subsidiary name?
3. What is the deal value and per-share price?
4. When was it announced and when is it expected to close?
5. What regulatory approvals are required (HSR, CFIUS, etc.)?
6. Has the target adopted any defense measures (poison pill, rights plan)?"""},
            ],
            "temperature": 0.1,
        },
        timeout=60,
    )
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]

    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    json_str = match.group(1) if match else raw
    try:
        data = json.loads(json_str.strip())
    except json.JSONDecodeError:
        data = {"target_company": ticker, "acquirer_company": "Unknown"}

    return DealContext(
        ticker=ticker.upper(),
        target_company=data.get("target_company") or ticker,
        target_aliases=data.get("target_aliases") or [],
        acquirer_company=data.get("acquirer_company") or "Unknown",
        acquirer_aliases=data.get("acquirer_aliases") or [],
        merger_sub=data.get("merger_sub"),
        deal_value=data.get("deal_value"),
        announcement_date=data.get("announcement_date"),
        expected_close=data.get("expected_close"),
        deal_type=data.get("deal_type") or "unknown",
        key_terms=data.get("key_terms") or [],
        raw_response=raw,
    )
