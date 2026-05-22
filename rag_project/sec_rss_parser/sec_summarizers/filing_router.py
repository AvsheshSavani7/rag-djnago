"""
SEC Filing Router — Auto-detects filing type and routes to the correct summarizer.
Intended to run from the Django app only. Uses settings.ANTHROPIC_API_KEY.
Usage: from sec_rss_parser.sec_summarizers.filing_router import route_and_summarize
"""

from __future__ import annotations
import importlib
import sys

import anthropic
import re

from ._config import get_anthropic_api_key

# ──── Default URL (override when calling route_and_summarize(url)) ────
FILING_URL = "https://www.sec.gov/Archives/edgar/data/1577552/000110465926024354/tm268212d1_6k.htm"
# ─────────────────────────────────

# ──── Filing type → module mapping ────
FILING_MAP = {
    "8-K":       "8k_summary",
    "99.1":      "991_summary",
    "6-K":       "6k_summary",
    "FORM 4":    "form4_summary",
    "FORM 144":  "form144_summary",
    "424B5":     "424b5_summary",
    "SC 13D":    "sc13d_summary",
    "SC 13G":    "sc13d_summary",
    "SC 13D/A":  "sc13d_summary",
    "SC 13G/A":  "sc13d_summary",
    "425":       "425_summary",
    "S-8":       "s8_summary",
    "SC TO-T":   "sc_to_summary",
    "SC TO-T/A": "sc_to_summary",
    "SC TO-C":   "sc_to_summary",
    "SC TO-C/A": "sc_to_summary",
    "SC 14D-9":  "sc_to_summary",
    "SC 14D-9/A": "sc_to_summary",
    "S-4":       "s4_summary",
    "S-4/A":     "s4_summary",
    "F-4":       "f4_summary",
    "F-4/A":     "f4_summary",
    "FORM 25":   "form25_summary",
    # ── Press releases ──
    "PRESS_RELEASE": "PRNewswire_summary",
    # ── Additional types from email filings ──
    "8-K/A":     "8k_summary",
    "10-Q":      "10k_summary",
    "10-K":      "10k_summary",
    "10-K/A":    "10k_summary",
    "10-Q/A":    "10k_summary",
    "DEF 14A":   "sec_filing_summary",
    "DEFA14A":   "sec_filing_summary",
    "DEFM14A":   "sec_filing_summary",
    "S-3":       "sec_filing_summary_sonnet",
    "S-3/A":     "sec_filing_summary_sonnet",
    "S-1":       "sec_filing_summary_sonnet",
    "S-1/A":     "sec_filing_summary_sonnet",
    "FORM 3":    "sec_filing_summary_sonnet",
    "NT 10-Q":   "sec_filing_summary",
    "NT 10-K":   "sec_filing_summary",
    "ARS":       "sec_filing_summary_sonnet",
    "EFFECT":    "sec_filing_summary_sonnet",
    "CORRESP":   "sec_filing_summary_sonnet",
    "UPLOAD":    "sec_filing_summary_sonnet",
}

# ──── URL patterns for fast detection ────
URL_PATTERNS = [
    (r'/sc13d',          "SC 13D"),
    (r'/sc13g',          "SC 13G"),
    (r'/sc13da',         "SC 13D/A"),
    (r'/sc13ga',         "SC 13G/A"),
    (r'424b5',           "424B5"),
    (r'424b\d',          "424B5"),      # other 424 variants route here too
    (r'/form4',          "FORM 4"),
    (r'_form4',          "FORM 4"),
    (r'/144',            "FORM 144"),
    (r'form144',         "FORM 144"),
    (r'/8-?k[^a-z]',     "8-K"),
    (r'/8k[^a-z]',       "8-K"),
    (r'/6-?k[^a-z]',     "6-K"),
    (r'/6k[^a-z]',       "6-K"),
    (r'ex99[\-_\.]?1',   "99.1"),
    (r'ex-?99',          "99.1"),
    (r'/425[^0-9]',      "425"),
    (r'_425\.',          "425"),
    (r'/s-?8[^0-9]',     "S-8"),
    (r'_s8\.',           "S-8"),
    (r'sc-?toc',         "SC TO-C"),
    (r'sc-?to',          "SC TO-T"),
    (r'sc14d',           "SC 14D-9"),
    (r'sc-?14d',         "SC 14D-9"),
    (r'/s-?4[^0-9]',     "S-4"),
    (r'_s4\.',           "S-4"),
    (r'/f-?4[^0-9]',     "F-4"),
    (r'_f4\.',           "F-4"),
    (r'form-?25',        "FORM 25"),
    (r'/25-nse',         "FORM 25"),
    (r'/10-?q[^a-z]',    "10-Q"),
    (r'_10q\.',          "10-Q"),
    (r'/10-?k[^a-z]',    "10-K"),
    (r'_10k\.',          "10-K"),
    (r'def14a',          "DEF 14A"),
    (r'defa14a',         "DEFA14A"),
    (r'defm14a',         "DEFM14A"),
    (r'/s-?3[^0-9]',     "S-3"),
    (r'/s-?1[^0-9]',     "S-1"),
    (r'nt10-?[qk]',      "NT 10-Q"),
    (r'xsleffect',       "EFFECT"),
]

# ──── Press release domains (detect before URL patterns) ────
PRESS_RELEASE_DOMAINS = [
    "prnewswire.com",
    "globenewswire.com",
    "businesswire.com",
    "accesswire.com",
    "newswire.com",
]


CLASSIFY_PROMPT = """You are an SEC filing classifier. Given the first portion of a filing's text, identify the filing type.

Respond ONLY with one of these exact labels (no extra text):
8-K
99.1
6-K
FORM 4
FORM 144
424B5
SC 13D
SC 13G
425
S-8
SC TO-T
SC TO-C
SC 14D-9
S-4
F-4
FORM 25
10-Q
10-K
DEF 14A
DEFA14A
S-3
FORM 3
OTHER

If the filing is an amendment (e.g., SC 13D/A, 8-K/A), use the base type (e.g., SC 13D, 8-K).
If you see "Exhibit 99" or "EX-99" in the header, classify as 99.1.
If the document is NOT an SEC filing (e.g., a foreign government document, court filing, regulatory dispatch, news article, or any non-SEC source), respond with OTHER.
If you are not confident the document matches a specific SEC filing type, respond with OTHER rather than guessing.

FILING TEXT (first ~1000 words):
"""


def detect_from_url(url: str) -> str | None:
    """Try to detect filing type from URL patterns."""
    url_lower = url.lower()

    # Check for press release domains first
    for domain in PRESS_RELEASE_DOMAINS:
        if domain in url_lower:
            return "PRESS_RELEASE"

    for pattern, filing_type in URL_PATTERNS:
        if re.search(pattern, url_lower):
            return filing_type
    return None


def fetch_preview(source: str, word_limit: int = 1000) -> str:
    """Fetch just enough text to classify the filing type (supports HTML + PDF)."""
    from .fetch_utils import fetch_text
    return fetch_text(source, word_limit=word_limit)


def classify_with_claude(text: str) -> str:
    """Use Claude Opus to classify the filing type from text content."""
    api_key = get_anthropic_api_key()
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Set it in .env or Django settings.")
    client = anthropic.Anthropic(api_key=api_key)

    msg = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=20,
        messages=[{
            "role": "user",
            "content": CLASSIFY_PROMPT + "\n\n" + text
        }]
    )

    label = msg.content[0].text.strip().upper()

    # Normalize common variations
    label = label.replace("SCHEDULE ", "SC ")
    label = label.replace("EXHIBIT ", "")
    if "13D" in label and "13G" not in label:
        label = "SC 13D"
    if "13G" in label:
        label = "SC 13G"

    return label


def route_and_summarize(url: str | list[str], deal_context: dict | None = None):
    """Detect filing type, import the right summarizer, and run it.

    Args:
        url: Single URL string or list of URLs. For a list, the first URL is
             used for classification and the full list is passed to the
             summarizer module as FILING_URL.
        deal_context: Optional pre-confirmed deal metadata dict with keys:
            primary_ticker, target_ticker, target_name, acquirer_ticker,
            acquirer_name. When provided, these values are injected into the
            summarizer prompt so the LLM uses them directly rather than
            inferring them from the filing text.
    """
    classify_url = url[0] if isinstance(url, list) else url

    # ── Step 1: Try URL-based detection (free, instant) ──
    filing_type = detect_from_url(classify_url)
    if filing_type:
        print(f"📎 Detected filing type from URL: {filing_type}")
    else:
        # ── Step 2: Fetch preview and classify with Claude ──
        print("🔍 Could not detect type from URL — fetching preview for classification...")
        preview = fetch_preview(classify_url)
        print(f"   Extracted {len(preview.split())} words for classification")

        print("   Classifying via Claude Opus 4.5...")
        filing_type = classify_with_claude(preview)
        print(f"📎 Classified as: {filing_type}")

    # ── Step 3: Route to the correct summarizer ──
    module_name = FILING_MAP.get(filing_type)

    if module_name is None:
        if filing_type == "OTHER" or filing_type not in FILING_MAP:
            print(
                f"⚠️  No dedicated summarizer for '{filing_type}' — using catch-all")
            module_name = "sec_filing_summary"
        else:
            module_name = "sec_filing_summary"

    print(f"📂 Routing to: {module_name}.py")
    if isinstance(url, list):
        print(f"   ({len(url)} documents to combine)")
    print("=" * 70)

    # ── Step 4: Import and run the summarizer ──
    module = importlib.import_module(
        f"sec_rss_parser.sec_summarizers.{module_name}")

    # Set the URL(s) and deal context in the target module
    module.FILING_URL = url
    if hasattr(module, "DEAL_CONTEXT"):
        module.DEAL_CONTEXT = deal_context or None

    # Run it
    result = module.main()

    print("result:2 ", result)

    return result


def main():
    url = FILING_URL

    if not url:
        print("❌ No URL provided. Paste a URL into FILING_URL at the top of the script.")
        sys.exit(1)

    if not get_anthropic_api_key():
        print("❌ ANTHROPIC_API_KEY not set. Set it in Django settings (e.g. from .env).")
        sys.exit(1)

    print(f"🚀 SEC Filing Router")
    print(f"   URL: {url}")
    print()

    result = route_and_summarize(url)

    print("result: ", result)
    return result


if __name__ == "__main__":
    main()
