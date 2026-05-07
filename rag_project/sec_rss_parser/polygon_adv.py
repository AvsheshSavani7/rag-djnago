"""
polygon_adv.py
--------------
Fetch Average Daily Volume in dollar terms (ADV $) for a ticker using Polygon.io.

Importable:
    from polygon_adv import get_adv
    result = get_adv("AAPL", days=20, api_key="your_key")

CLI:
    python polygon_adv.py --ticker AAPL
    python polygon_adv.py --ticker MSFT --days 30
    POLYGON_API_KEY=your_key python polygon_adv.py --ticker TSLA
"""

import os
import argparse
from datetime import date, timedelta

import requests


# ── Core function ─────────────────────────────────────────────────────────────

def get_adv(ticker: str, days: int = 20, api_key: str = None) -> dict:
    """
    Return ADV in dollar terms for `ticker` over the last `days` trading days.

    Parameters
    ----------
    ticker  : str  — e.g. "AAPL", "SPY", "X:BTCUSD"
    days    : int  — number of trading days to average (default 20)
    api_key : str  — Polygon API key; falls back to POLYGON_API_KEY env var

    Returns
    -------
    dict with keys:
        ticker, days_requested, days_found, adv_dollars,
        adv_dollars_fmt, from_date, to_date
    """
    api_key = os.environ.get(
        "POLYGON_API_KEY", "IXD26asVR4bNq5thZmCRwV8CBKx8E74O")
    if not api_key:
        raise EnvironmentError(
            "Polygon API key not provided. Pass api_key= or set POLYGON_API_KEY env var."
        )

    ticker = ticker.upper().strip()
    if not ticker:
        raise ValueError("ticker must be a non-empty string.")

    # Request extra calendar days to ensure we get enough trading days back
    # (~1.5× to cover weekends + holidays)
    calendar_buffer = int(days * 1.5) + 10
    # yesterday (today may not be settled)
    to_date = date.today() - timedelta(days=1)
    from_date = to_date - timedelta(days=calendar_buffer)

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day"
        f"/{from_date}/{to_date}"
    )
    params = {
        "adjusted": "true",
        "sort": "desc",          # newest first so we can slice the last N easily
        "limit": days + 20,      # a little headroom
        "apiKey": api_key,
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()

    data = resp.json()

    if data.get("status") == "ERROR" or data.get("resultsCount", 0) == 0:
        raise ValueError(
            f"No data returned for '{ticker}'. "
            f"Check the ticker symbol and your Polygon subscription."
        )

    bars = data["results"]          # sorted newest → oldest
    bars = bars[:days]              # keep only the last N trading days

    dollar_volumes = []
    for bar in bars:
        volume = bar.get("v", 0)
        # fallback to close if VWAP missing
        vwap = bar.get("vw") or bar.get("c")
        if volume and vwap:
            dollar_volumes.append(volume * vwap)

    if not dollar_volumes:
        raise ValueError(
            f"Could not compute dollar volume for '{ticker}' — missing v/vw fields.")

    adv = sum(dollar_volumes) / len(dollar_volumes)

    return {
        "ticker":          ticker,
        "days_requested":  days,
        "days_found":      len(dollar_volumes),
        "adv_dollars":     adv,
        "adv_dollars_fmt": _fmt_dollars(adv),
        "from_date":       str(from_date),
        "to_date":         str(to_date),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_dollars(amount: float) -> str:
    """Format a large dollar amount into a human-readable string."""
    if amount >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.2f}B"
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.2f}M"
    if amount >= 1_000:
        return f"${amount / 1_000:.2f}K"
    return f"${amount:.2f}"


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get ADV in $ via Polygon.io")
    parser.add_argument("--ticker", default="CCRN",
                        help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--days",   type=int, default=30,
                        help="Trading days to average (default 20)")
    parser.add_argument("--api-key", dest="api_key", default=None,
                        help="Polygon API key (or set POLYGON_API_KEY env var)")
    args = parser.parse_args()

    try:
        result = get_adv(ticker=args.ticker, days=args.days,
                         api_key=args.api_key)
        print(f"\nTicker         : {result['ticker']}")
        print(f"Period         : {result['from_date']} → {result['to_date']}")
        print(
            f"Trading days   : {result['days_found']} (of {result['days_requested']} requested)")
        print(f"ADV ($)        : {result['adv_dollars_fmt']}")
        print(f"ADV (raw)      : ${result['adv_dollars']:,.0f}")
    except (ValueError, EnvironmentError) as e:
        print(f"Error: {e}")
