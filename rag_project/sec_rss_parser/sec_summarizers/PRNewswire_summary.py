"""
PRNewswire Merger Press Release Summarizer — Multi-level summaries via Claude API
Usage: python PRNewswire_summary.py
"""

import anthropic
import re
import json
import sys
import os
import io
from pathlib import Path

try:
    from ._naming import filing_uid
    from ._deal_context import inject_deal_context
except ImportError:
    from _naming import filing_uid
    from _deal_context import inject_deal_context

# ──── PASTE YOUR PRNEWSWIRE URL HERE ────
FILING_URL = ""
DEAL_CONTEXT = None
# ──── OUTPUT FOLDER (local fallback when not using S3) ────
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "Output Summaries"
# ─────────────────────────────────


try:
    import requests
    from bs4 import BeautifulSoup
    from docx import Document as DocxDocument
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                          "requests", "beautifulsoup4", "python-docx", "-q"])
    import requests
    from bs4 import BeautifulSoup
    from docx import Document as DocxDocument
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

try:
    import cloudscraper as _cloudscraper_mod
except ImportError:
    _cloudscraper_mod = None

try:
    from playwright.sync_api import sync_playwright as _sync_playwright
    _playwright_available = True
except ImportError:
    _sync_playwright = None
    _playwright_available = False

try:
    from ._config import get_anthropic_api_key
except ImportError:
    from _config import get_anthropic_api_key

ANTHROPIC_API_KEY = get_anthropic_api_key()
if not ANTHROPIC_API_KEY and __name__ == "__main__":
    print("❌ ANTHROPIC_API_KEY not found. Set it in .env or Django settings (ANTHROPIC_API_KEY).")
    sys.exit(1)

# ──── Perplexity config ────
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_MODEL = "sonar-pro"


SUMMARY_PROMPT = """You are an expert M&A analyst summarizing merger and acquisition press releases from PRNewswire for a merger arbitrage desk.

Given the press release text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "acquirer": "<name of acquiring company>",
  "target": "<name of target company>",
  "acquirer_ticker": "<acquirer ticker, or null if not mentioned>",
  "target_ticker": "<target ticker, or null if not mentioned>",
  "announcement_date": "<MM/DD/YY>",
  "deal_type": "<Merger | Acquisition | Tender Offer | Merger of Equals | Other>",

  "L1_headline": "+ <TARGET TICKER or NAME> – <key event in ≤8 words>. | <date>",

  "L2_brief": "<2-3 sentence summary covering: who is acquiring whom, deal value and structure, and key terms>",

  "L3_detailed": {
    "deal_value": "<total deal value, e.g. '$2.1B'>",
    "deal_structure": "<cash, stock, or mix — include ratio/price per share if stated>",
    "premium": "<premium to last close or unaffected price, if stated>",
    "timeline": "<expected close date and key milestone dates>",
    "conditions": ["<regulatory approvals needed, shareholder votes, financing conditions>"],
    "advisors": ["<investment banks, law firms advising each side>"],
    "strategic_rationale": "<stated reason for the deal>",
    "risks_flagged": ["<breakup fee, regulatory risk, competing bids, litigation, financing risk>"]
  }
}

Rules:
- CRITICAL — FACTS ONLY: Every statement in your summary must be directly traceable to the filing text. Report ONLY what the document says. Do NOT add analysis, assess significance, interpret motives, predict outcomes, evaluate probability, or editorialize. Do NOT state what is "not disclosed" or "not mentioned" — simply omit fields where the filing is silent. If the filing does not say it, do not write it.
  GOOD: "CADE requested revenue data for 2021-2025 across four markets."
  BAD: "The broad scope of information requested indicates potentially detailed competitive analysis ahead."
  GOOD: "The offer expires June 10, 2026."
  BAD: "This tight timeline may create pressure on shareholders to tender quickly."
- PRECISION: Use the filing's exact terminology for legal, regulatory, and financial terms. Do NOT paraphrase in ways that broaden or narrow the stated meaning. GOOD: "All 14 Pennsylvania PUC hearings have concluded." BAD: "Regulatory proceedings concluded in Pennsylvania."
- L1 format MUST be: + <TICKER or NAME> – <event>. | <date>
- Use the TARGET's ticker for L1 if available, otherwise use target company name
- Extract exact dollar amounts, per-share prices, premiums, and exchange ratios
- Flag any termination/breakup fees, go-shop periods, or matching rights
- Note any regulatory bodies that must approve (DOJ, FTC, CFIUS, EU, etc.)
- If advisors are not mentioned, set "advisors" to an empty list

PRESS RELEASE TEXT:
"""


COMPANY_RESEARCH_PROMPT = """You are a financial research analyst. Based on your knowledge and the Perplexity research context provided, assess whether the following company is a US publicly traded company with a market cap greater than $100 million.

Company: {company}
Ticker (if known): {ticker}
Context / Perplexity Research:
{context}

Respond ONLY in valid JSON (no markdown fences):
{{
  "is_us_publicly_traded": <true|false|null>,
  "market_cap_over_100m": <true|false|null>,
  "estimated_market_cap": "<estimated market cap range, e.g. '$500M–$1B', or 'Private' or 'Unknown'>",
  "exchange": "<NYSE|NASDAQ|AMEX|OTC|Not Listed|Private|Unknown>",
  "ticker_confirmed": "<confirmed ticker symbol, or null>",
  "confidence": "<High|Medium|Low>",
  "rationale": "<1-2 sentence explanation of the assessment>"
}}

Use null only when information is genuinely insufficient to determine."""


ARTICLE_QUESTIONS_PROMPT = """You are an expert M&A and regulatory analyst. Read the press release text below and answer the two questions that follow.

Respond ONLY in valid JSON (no markdown fences):
{{
  "announces_new_merger": {{
    "answer": "<Yes|No>",
    "explanation": "<1-2 sentence explanation>"
  }},
  "significant_regulatory_development": {{
    "answer": "<Yes|No>",
    "explanation": "<1-2 sentence explanation>"
  }}
}}

Question 1: Does this press release announce a new merger or acquisition transaction?
Question 2: Does this press release discuss a significant regulatory development (e.g., antitrust review, regulatory approval or rejection, government investigation, new compliance requirement, or material regulatory risk)?

PRESS RELEASE TEXT:
{text}"""


def _fetch_with_jina(url: str) -> str:
    """Fetch a page via Jina Reader (r.jina.ai).

    Jina fetches the page through their own trusted infrastructure, bypassing
    Cloudflare bot protection that blocks direct requests and headless browsers.
    Returns HTML for BeautifulSoup to parse.
    """
    print("  ⚠️  Retrying with Jina Reader (r.jina.ai)...")
    jina_url = f"https://r.jina.ai/{url}"
    resp = requests.get(
        jina_url,
        headers={"Accept": "text/html", "X-Return-Format": "html"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.text


def _fetch_with_playwright(url: str) -> str:
    """Fetch a page using a real headless Chromium browser (last-resort fallback)."""
    print("  ⚠️  Retrying with Playwright (headless Chromium)...")
    with _sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="en-US",
        )
        page = context.new_page()
        page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        html = page.content()
        browser.close()
    return html


def fetch_article_text(source: str) -> str:
    """Fetch and extract article text from a press release URL or local file.

    Supports PRNewswire, GlobeNewswire, and BusinessWire URLs.
    For bot-protected sites (BusinessWire) cloudscraper is used as a fallback.
    """
    if source.startswith("http"):
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
        }
        resp = requests.get(source, headers=headers, timeout=30)

        html = None
        if resp.status_code == 200:
            html = resp.text
        elif resp.status_code in (403, 429, 503):
            # Tier 2: cloudscraper (handles older Cloudflare JS challenges)
            if _cloudscraper_mod is not None:
                try:
                    print(
                        f"  ⚠️  HTTP {resp.status_code} — retrying with cloudscraper...")
                    scraper = _cloudscraper_mod.create_scraper()
                    cs_resp = scraper.get(source, headers=headers, timeout=30)
                    if cs_resp.status_code == 200:
                        html = cs_resp.text
                    else:
                        print(
                            f"  ⚠️  cloudscraper returned HTTP {cs_resp.status_code}")
                except Exception as e:
                    print(f"  ⚠️  cloudscraper failed: {e}")

            # Tier 3: Jina Reader — routes through trusted infrastructure,
            # bypassing Cloudflare bot protection that blocks headless browsers.
            if html is None:
                try:
                    html = _fetch_with_jina(source)
                except Exception as e:
                    print(f"  ⚠️  Jina Reader failed: {e}")

            # Tier 4: Playwright headless browser (last resort)
            if html is None:
                if _playwright_available:
                    try:
                        html = _fetch_with_playwright(source)
                    except Exception as e:
                        print(f"  ⚠️  Playwright failed: {e}")
                if html is None:
                    print(
                        f"  ❌  All fetch methods failed for {source}\n"
                        "  Save the page as HTML from your browser and pass the file path instead."
                    )
                    resp.raise_for_status()
        else:
            resp.raise_for_status()
    else:
        html = Path(source).read_text()

    soup = BeautifulSoup(html, "html.parser")

    # Site-specific selectors FIRST (before stripping boilerplate, since some
    # sites wrap article content inside <header> tags — e.g. GlobeNewswire)
    article = (
        # GlobeNewswire
        soup.find("div", id="main-body-container")
        or soup.find("div", class_=re.compile(r"main-body-container", re.I))
        # BusinessWire
        or soup.find("div", class_=re.compile(r"bw-release-story|bw-release-body", re.I))
        or soup.find("section", class_=re.compile(r"bw-press-release", re.I))
        # PRNewswire
        or soup.find("section", class_=re.compile(r"release-body", re.I))
        or soup.find("div", class_=re.compile(r"article-body|press-release-body|news-release", re.I))
        # Generic
        or soup.find("article")
    )

    if article:
        # Remove boilerplate only within the found article container
        for tag in article(["script", "style", "meta", "link", "nav", "footer", "header"]):
            tag.decompose()
        text = article.get_text(separator="\n", strip=True)
    else:
        # Fallback: strip boilerplate from full page
        for tag in soup(["script", "style", "meta", "link", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)

    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    # Truncate to ~10k words to stay within token budget
    words = text.split()
    if len(words) > 10000:
        text = " ".join(words[:10000])

    return text


def _s(value, default: str = "N/A") -> str:
    """Coerce optional/null LLM fields to a safe string for display."""
    if value is None:
        return default
    return str(value)


def _extract_response_text(msg) -> str:
    """Collect text from all Claude content blocks."""
    parts = []
    for block in msg.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _parse_json_response(raw: str, *, context: str = "Claude response") -> dict:
    """Parse JSON from Claude output, stripping markdown fences."""
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    if not raw:
        raise ValueError(f"{context}: empty response")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{context}: invalid JSON: {raw[:500]!r}") from e


def summarize(text: str, model: str = "claude-opus-4-6") -> dict:
    """Call Claude API to produce multi-level summary."""
    if not text or not text.strip():
        raise ValueError("Cannot summarize: no article text extracted")

    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Set it in .env or Django settings (ANTHROPIC_API_KEY).")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    msg = client.messages.create(
        model=model,
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": inject_deal_context(SUMMARY_PROMPT, DEAL_CONTEXT) + "\n\n" + text
        }]
    )

    raw = _extract_response_text(msg)
    if msg.stop_reason != "end_turn":
        if raw.count("{") > raw.count("}"):
            raw += '"' + "}" * (raw.count("{") - raw.count("}"))
        if raw.count("[") > raw.count("]"):
            raw += "]" * (raw.count("[") - raw.count("]"))

    return _parse_json_response(raw, context="summarize")


def search_perplexity(query: str):
    """Query Perplexity's online search model and return the response text."""
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": PERPLEXITY_MODEL,
        "messages": [{"role": "user", "content": query}],
        "max_tokens": 600,
    }

    try:
        resp = requests.post(PERPLEXITY_API_URL,
                             headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  ⚠️  Perplexity search failed: {e}")
        return None


def check_target_company(acquirer: str, target: str, target_ticker: str = None) -> dict:
    """Research the target company via Perplexity and Claude Opus 4.6.

    Assesses whether the target is a US publicly traded company with market cap > $100M.
    """
    print(f"\nResearching target company: {target}...")

    ticker_hint = f" (ticker: {target_ticker})" if target_ticker else ""
    perplexity_context = ""

    # Step 1: Perplexity web search for real-time market data
    if os.getenv("PERPLEXITY_API_KEY"):
        print("  Querying Perplexity for current market data...")
        query = (
            f"Is {target}{ticker_hint} a publicly traded company listed on a US stock exchange "
            f"(NYSE, NASDAQ, or AMEX)? What is its current market capitalization? "
            f"Is it a private company? Please provide its ticker symbol if it trades publicly."
        )
        perplexity_context = search_perplexity(query) or ""
        if perplexity_context:
            print(
                f"  ✓ Perplexity: {len(perplexity_context.split())} words received")
    else:
        print("  ⚠️  PERPLEXITY_API_KEY not set — skipping Perplexity search")

    # Step 2: Claude Opus 4.6 assessment
    print("  Querying Claude Opus 4.6 for company assessment...")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    context_block = f"Acquiring company: {acquirer}\n"
    if perplexity_context:
        context_block += f"\nPerplexity web search result:\n{perplexity_context}"
    else:
        context_block += "\n(No Perplexity data available — use training knowledge only)"

    prompt = COMPANY_RESEARCH_PROMPT.format(
        company=target,
        ticker=target_ticker or "not mentioned in press release",
        context=context_block,
    )

    try:
        msg = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        result["perplexity_raw"] = perplexity_context or None
        print("  ✓ Claude Opus 4.6 assessment complete")
        return result
    except Exception as e:
        print(f"  ⚠️  Company check failed: {e}")
        return {
            "is_us_publicly_traded": None,
            "market_cap_over_100m": None,
            "estimated_market_cap": "Unknown",
            "exchange": "Unknown",
            "ticker_confirmed": None,
            "confidence": "Low",
            "rationale": f"Assessment failed: {e}",
            "perplexity_raw": perplexity_context or None,
        }


def answer_article_questions(text: str) -> dict:
    """Ask two specific questions about the press release via Claude Opus 4.6."""
    print("\nAnswering article-specific questions via Claude Opus 4.6...")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Limit text length for this prompt
    truncated = " ".join(text.split()[:4000])
    prompt = ARTICLE_QUESTIONS_PROMPT.format(text=truncated)

    try:
        msg = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        print("  ✓ Article questions answered")
        return result
    except Exception as e:
        print(f"  ⚠️  Article Q&A failed: {e}")
        return {
            "announces_new_merger": {"answer": "Unknown", "explanation": f"Assessment failed: {e}"},
            "significant_regulatory_development": {"answer": "Unknown", "explanation": f"Assessment failed: {e}"},
        }


def _bool_label(val) -> str:
    if val is True:
        return "YES"
    if val is False:
        return "NO"
    return "UNKNOWN"


def print_summary(s: dict):
    """Pretty-print the multi-level summary."""
    print("\n" + "=" * 70)
    print("  PRNEWSWIRE MERGER PRESS RELEASE SUMMARY")
    print("=" * 70)

    acquirer = s.get("acquirer", "N/A")
    target = s.get("target", "N/A")
    a_tick = s.get("acquirer_ticker") or "—"
    t_tick = s.get("target_ticker") or "—"

    print(
        f"\n   Deal:    {acquirer} ({a_tick})  acquires  {target} ({t_tick})")
    print(f"   Type:    {s.get('deal_type', 'N/A')}")
    print(f"   Date:    {s.get('announcement_date', 'N/A')}")

    # L1 — Headline
    print(f"\n📌 L1 | HEADLINE")
    print(f"   {s['L1_headline']}")

    # L2 — Brief
    print(f"\n📋 L2 | BRIEF")
    print(f"   {s['L2_brief']}")

    # L3 — Detailed
    d = s["L3_detailed"]
    print(f"\n📊 L3 | DETAILED")
    print(f"   Deal Value:  {d.get('deal_value', 'N/A')}")
    print(f"   Structure:   {d.get('deal_structure', 'N/A')}")
    print(f"   Premium:     {d.get('premium', 'N/A')}")
    print(f"   Timeline:    {d.get('timeline', 'N/A')}")
    print(f"   Rationale:   {d.get('strategic_rationale', 'N/A')}")

    if d.get("conditions"):
        print(f"   Conditions:")
        for c in d["conditions"]:
            print(f"     • {c}")

    if d.get("advisors"):
        print(f"   Advisors:")
        for a in d["advisors"]:
            print(f"     • {a}")

    if d.get("risks_flagged"):
        print(f"   Risks:")
        for r in d["risks_flagged"]:
            print(f"     • {r}")

    # ── Intelligence Check: Target Company ──
    cc = s.get("company_check")
    if cc:
        print(f"\n🔍 TARGET COMPANY INTELLIGENCE CHECK  (Perplexity + Claude Opus 4.6)")
        pub = _bool_label(cc.get("is_us_publicly_traded"))
        cap = _bool_label(cc.get("market_cap_over_100m"))
        flag = " ✅" if (cc.get("is_us_publicly_traded")
                        and cc.get("market_cap_over_100m")) else " ❌"
        print(f"   US Publicly Traded:      {pub}")
        print(f"   Market Cap > $100M:      {cap}{flag}")
        print(
            f"   Estimated Market Cap:    {cc.get('estimated_market_cap', 'Unknown')}")
        print(f"   Exchange:                {cc.get('exchange', 'Unknown')}")
        if cc.get("ticker_confirmed"):
            print(f"   Ticker (confirmed):      {cc['ticker_confirmed']}")
        print(f"   Confidence:              {cc.get('confidence', 'N/A')}")
        print(f"   Rationale:               {cc.get('rationale', '')}")

    # ── Article-Specific Questions ──
    qa = s.get("article_questions")
    if qa:
        print(f"\n❓ ARTICLE-SPECIFIC QUESTIONS  (Claude Opus 4.6)")

        merger = qa.get("announces_new_merger", {})
        print(f"\n   Q1. Does this announce a new merger or acquisition?")
        print(f"       Answer:  {merger.get('answer', 'Unknown')}")
        print(f"       Detail:  {merger.get('explanation', '')}")

        reg = qa.get("significant_regulatory_development", {})
        print(f"\n   Q2. Does this discuss a significant regulatory development?")
        print(f"       Answer:  {reg.get('answer', 'Unknown')}")
        print(f"       Detail:  {reg.get('explanation', '')}")

    print("=" * 70)


def export_docx(s: dict, s3_key_suffix: str):
    """Build summary as Word doc, upload to S3 (summary_docx/), return (s3_path, s3_url)."""
    from .s3_utils import upload_docx_bytes

    target = _s(s.get("target"), "UNKNOWN")
    acquirer = _s(s.get("acquirer"), "UNKNOWN")
    date = _s(s.get("announcement_date"), "")

    doc = DocxDocument()

    # -- Styles --
    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    # -- Title --
    title = doc.add_heading(f"M&A Press Release Summary", level=0)
    title.runs[0].font.size = Pt(20)

    # Deal metadata
    meta = doc.add_paragraph()
    meta.add_run("Acquirer: ").bold = True
    meta.add_run(f"{acquirer} ({s.get('acquirer_ticker') or '—'})")
    meta.add_run("    Target: ").bold = True
    meta.add_run(f"{target} ({s.get('target_ticker') or '—'})")

    meta2 = doc.add_paragraph()
    meta2.add_run("Announcement Date: ").bold = True
    meta2.add_run(date)
    meta2.add_run("    Deal Type: ").bold = True
    meta2.add_run(_s(s.get("deal_type")))

    # -- L1: Headline --
    doc.add_heading("L1 — Headline", level=1)
    p = doc.add_paragraph()
    run = p.add_run(_s(s.get("L1_headline")))
    run.bold = True
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(0, 51, 102)

    # -- L2: Brief --
    doc.add_heading("L2 — Brief", level=1)
    doc.add_paragraph(_s(s.get("L2_brief")))

    # -- L3: Detailed --
    doc.add_heading("L3 — Detailed", level=1)
    d = s.get("L3_detailed") or {}

    doc.add_heading("Deal Terms", level=2)
    terms = doc.add_paragraph()
    terms.add_run("Deal Value: ").bold = True
    terms.add_run(_s(d.get("deal_value")) + "\n")
    terms.add_run("Structure: ").bold = True
    terms.add_run(_s(d.get("deal_structure")) + "\n")
    terms.add_run("Premium: ").bold = True
    terms.add_run(_s(d.get("premium")) + "\n")
    terms.add_run("Timeline: ").bold = True
    terms.add_run(_s(d.get("timeline")))

    doc.add_heading("Strategic Rationale", level=2)
    doc.add_paragraph(_s(d.get("strategic_rationale")))

    if d.get("conditions"):
        doc.add_heading("Closing Conditions", level=2)
        for c in d["conditions"]:
            doc.add_paragraph(c, style="List Bullet")

    if d.get("advisors"):
        doc.add_heading("Advisors", level=2)
        for a in d["advisors"]:
            doc.add_paragraph(a, style="List Bullet")

    if d.get("risks_flagged"):
        doc.add_heading("Risks Flagged", level=2)
        for r in d["risks_flagged"]:
            doc.add_paragraph(r, style="List Bullet")

    # -- Intelligence Check: Target Company --
    cc = s.get("company_check")
    if cc:
        doc.add_heading("Target Company Intelligence Check", level=1)

        is_pub = cc.get("is_us_publicly_traded")
        is_cap = cc.get("market_cap_over_100m")
        qualifies = is_pub is True and is_cap is True

        intro = doc.add_paragraph()
        intro.add_run("Sources: ").bold = True
        intro.add_run("Perplexity web search + Claude Opus 4.6")

        verdict = doc.add_paragraph()
        verdict.add_run(
            "Qualifies (US public, market cap > $100M): ").bold = True
        vrun = verdict.add_run("YES" if qualifies else "NO" if (
            is_pub is False or is_cap is False) else "UNCERTAIN")
        vrun.font.color.rgb = RGBColor(
            0, 128, 0) if qualifies else RGBColor(192, 0, 0)
        vrun.bold = True

        details = doc.add_paragraph()
        details.add_run("US Publicly Traded: ").bold = True
        details.add_run(_bool_label(is_pub) + "\n")
        details.add_run("Market Cap > $100M: ").bold = True
        details.add_run(_bool_label(is_cap) + "\n")
        details.add_run("Estimated Market Cap: ").bold = True
        details.add_run(_s(cc.get("estimated_market_cap"), "Unknown") + "\n")
        details.add_run("Exchange: ").bold = True
        details.add_run(_s(cc.get("exchange"), "Unknown") + "\n")
        if cc.get("ticker_confirmed"):
            details.add_run("Confirmed Ticker: ").bold = True
            details.add_run(_s(cc.get("ticker_confirmed")) + "\n")
        details.add_run("Confidence: ").bold = True
        details.add_run(_s(cc.get("confidence")) + "\n")
        details.add_run("Rationale: ").bold = True
        details.add_run(_s(cc.get("rationale"), ""))

        if cc.get("perplexity_raw"):
            doc.add_heading("Perplexity Raw Response", level=2)
            doc.add_paragraph(cc["perplexity_raw"])

    # -- Article-Specific Questions --
    qa = s.get("article_questions")
    if qa:
        doc.add_heading(
            "Article-Specific Questions (Claude Opus 4.6)", level=1)

        merger = qa.get("announces_new_merger", {})
        doc.add_heading(
            "Q1: Does this announce a new merger or acquisition?", level=2)
        p_m = doc.add_paragraph()
        p_m.add_run("Answer: ").bold = True
        ans_run = p_m.add_run(merger.get("answer", "Unknown"))
        ans_run.bold = True
        if merger.get("answer") == "Yes":
            ans_run.font.color.rgb = RGBColor(0, 128, 0)
        doc.add_paragraph(merger.get("explanation", ""))

        reg = qa.get("significant_regulatory_development", {})
        doc.add_heading(
            "Q2: Does this discuss a significant regulatory development?", level=2)
        p_r = doc.add_paragraph()
        p_r.add_run("Answer: ").bold = True
        ans_run2 = p_r.add_run(reg.get("answer", "Unknown"))
        ans_run2.bold = True
        if reg.get("answer") == "Yes":
            ans_run2.font.color.rgb = RGBColor(0, 128, 0)
        doc.add_paragraph(reg.get("explanation", ""))

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    path, url = upload_docx_bytes(buf.read(), s3_key_suffix)
    return path, url


def main():
    source = FILING_URL

    print(f"Fetching PRNewswire press release from: {source}")

    text = fetch_article_text(source)
    print(f"Extracted {len(text.split())} words of text")

    print("Generating summary via Claude Opus 4.5...")
    result = summarize(text)

    # ── Intelligence Check: target company ──
    company_check = check_target_company(
        acquirer=result.get("acquirer", ""),
        target=result.get("target", ""),
        target_ticker=result.get("target_ticker"),
    )
    result["company_check"] = company_check

    # ── Article-specific Q&A ──
    result["article_questions"] = answer_article_questions(text)

    print_summary(result)

    uid = filing_uid(FILING_URL)
    from .s3_utils import upload_json

    s3_json_path, s3_json_url = upload_json(
        result, f"PRNewswire_summary_{uid}.json")
    print(f"\nJSON uploaded to S3: {s3_json_url}")

    if not isinstance(result.get("L3_detailed"), dict):
        result["L3_detailed"] = {}

    target_name = result.get("target") or "UNKNOWN"
    date = result.get("announcement_date") or ""
    safe_target = re.sub(r'[^\w\-\.]', '_', target_name)
    safe_date = date.replace("/", "-") if date else "unknown-date"
    docx_suffix = f"PRNewswire_Summary_{safe_target}_{safe_date}_{uid}.docx"
    s3_docx_path, s3_docx_url = export_docx(result, docx_suffix)
    print(f"DOCX uploaded to S3: {s3_docx_url}")

    result["s3_docx_path"] = s3_docx_path
    result["s3_docx_url"] = s3_docx_url
    result["s3_json_path"] = s3_json_path
    result["s3_json_url"] = s3_json_url
    return result


if __name__ == "__main__":
    main()
