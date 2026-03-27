#!/usr/bin/env python3
"""
Standalone script to verify URL date extraction for SEC 10-K/10-Q filings.
Run: python check_date_match.py [url1 [url2 ...]]
With no args, runs built-in test URLs.
"""

import re
import sys


# Pattern 1 (old): date must be immediately followed by .htm — misses x10k/x10q filenames
PATTERN_OLD = re.compile(r'(\d{4})(\d{2})(\d{2})\.htm')

# Pattern 2 (current): allows optional x10k/x10q or _10k/_10q before .htm — use this for result
PATTERN_NEW = re.compile(
    r'(\d{4})(\d{2})(\d{2})(?:[x_]10[kq])?\.htm', re.IGNORECASE)


def extract_with_pattern(url: str, pattern: re.Pattern) -> str:
    """Return period_date (YYYY-MM-DD) or 'unknown' if no match."""
    match = pattern.search(url)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return "unknown"


def main():
    builtin_urls = [
        "https://www.sec.gov/Archives/edgar/data/1413837/000110465926028501/ffwm-20251231x10k.htm",
        "https://www.sec.gov/Archives/edgar/data/1438231/000143774926007793/dmrc20251231_10k.htm",
        "https://www.sec.gov/Archives/edgar/data/123/000123456789012345/abc-20240630.htm",
        "https://www.sec.gov/Archives/edgar/data/2065779/000182912626002312/dboralarcacq1_10k.htm",
        "https://www.sec.gov/Archives/edgar/data/456/000456789012345678/xyz-20230930x10q.htm",
        "https://www.sec.gov/Archives/edgar/data/1851909/000149315226009210/form10-k.htm",
        "https://www.sec.gov/Archives/edgar/data/1898496/000162828025013103/gety-20241231.htm",
        "https://www.sec.gov/Archives/edgar/data/1438231/000143774926007793/dmrc20251231_10k.htm"
    ]

    urls = sys.argv[1:] if len(sys.argv) > 1 else builtin_urls

    print("URL date match check (pattern 1 = old, pattern 2 = used)\n" + "=" * 70)
    for url in urls:
        date_old = extract_with_pattern(url, PATTERN_OLD)
        date_new = extract_with_pattern(url, PATTERN_NEW)
        used = date_new
        status = "OK" if used != "unknown" else "NO MATCH"
        print(f"  Pattern 1 (old):  period_date = {date_old}")
        print(f"  Pattern 2 (used): period_date = {date_new}  [{status}]")
        print(f"  URL: {url[:75]}{'...' if len(url) > 75 else ''}")
        print()
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
