"""
Post-process LLM summary results with known deal tickers.

When deal_context.primary_ticker is available (filing-company ticker resolved
from deal CIK), replace the leading ticker in L1_headline. Otherwise leave the
LLM output unchanged.
"""

from __future__ import annotations

import re


_L1_TICKER_RE = re.compile(
    r"^(\+?\s*)([A-Za-z][A-Za-z0-9.\-]{0,9})(\s*[–—-]\s*)"
)


def apply_known_tickers(result: dict, deal_context: dict | None = None) -> dict:
    """Replace L1 leading ticker with deal primary_ticker when available.

    Also updates result['ticker'] when that key is present.
    No-op when deal_context / primary_ticker is missing or L1_headline is empty.
    """
    if not isinstance(result, dict):
        return result

    ctx = deal_context or {}
    primary = ctx.get("primary_ticker")
    if primary is None:
        return result
    primary = str(primary).strip()
    if not primary or primary.upper() == "N/A":
        return result

    headline = result.get("L1_headline")
    if headline and str(headline).strip():
        original = str(headline).strip()
        replaced, n = _L1_TICKER_RE.subn(
            rf"\1{primary}\3", original, count=1
        )
        if n:
            result["L1_headline"] = replaced
        else:
            # No recognizable ticker prefix — prepend filing ticker
            body = original.lstrip("+").strip()
            result["L1_headline"] = f"+ {primary} – {body}"

    if "ticker" in result:
        result["ticker"] = primary

    return result


def resolve_l1_for_email(
    l1_headline: str | None,
    *,
    primary_ticker: str | None = None,
    matched_cik_label: str | None = None,
    target_ticker: str | None = None,
    acquirer_ticker: str | None = None,
) -> str | None:
    """Safety-net L1 rewrite at email send time using filer ticker.

    Prefer explicit primary_ticker (CIK-resolved). Else derive from
    matched_cik_label + target/acquirer tickers.
    """
    primary = (primary_ticker or "").strip() or None
    if not primary:
        label = (matched_cik_label or "").strip()
        if label == "(acquirer)":
            primary = (acquirer_ticker or "").strip() or None
        elif label == "(target)":
            primary = (target_ticker or "").strip() or None
        else:
            primary = (
                (target_ticker or "").strip()
                or (acquirer_ticker or "").strip()
                or None
            )
    if not primary or not l1_headline:
        return l1_headline
    return apply_known_tickers(
        {"L1_headline": l1_headline},
        {"primary_ticker": primary},
    ).get("L1_headline", l1_headline)
