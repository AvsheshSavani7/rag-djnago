"""
Classify RSS news items as merger-related: either pertaining to a deal we follow (follow)
or new merger news (new). For new deals, extract deal details via web search and create a new deal.
"""
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Optional OpenAI for LLM classification and extraction
try:
    import openai
except ImportError:
    openai = None

# Optional Playwright for 403 fallback (BusinessWire, etc.)
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


# Browser-like headers to reduce 403 from sites like BusinessWire/PR Newswire
_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def _fetch_html_playwright(url: str, timeout_ms: int = 20000) -> Optional[str]:
    """Fetch HTML using headless Chrome."""
    if not sync_playwright:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage",
                      "--disable-blink-features=AutomationControlled"],
            )
            try:
                context = browser.new_context(
                    user_agent=_FETCH_HEADERS["User-Agent"],
                    locale="en-US",
                )
                page = context.new_page()
                page.goto(url, timeout=timeout_ms,
                          wait_until="domcontentloaded")
                page.wait_for_timeout(2000)
                return page.content()
            finally:
                browser.close()
    except Exception as e:
        logger.warning("Playwright fetch failed for %s: %s", url, e)
        return None


def fetch_html_from_url(url: str, timeout: int = 15) -> Optional[str]:
    """
    Fetch HTML content from a URL using Playwright (headless Chrome).
    Falls back to requests only if Playwright is not installed.

    Args:
        url: Source URL to fetch.
        timeout: Request timeout in seconds (converted to ms for Playwright).

    Returns:
        HTML string or None on failure.
    """
    if not url or not url.strip().startswith(("http://", "https://")):
        return None
    timeout_ms = min(timeout * 1000, 60000)
    if sync_playwright:
        return _fetch_html_playwright(url, timeout_ms=timeout_ms)
    try:
        resp = requests.get(url, timeout=timeout, headers=_FETCH_HEADERS)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        logger.warning("Failed to fetch URL %s: %s", url, e)
        return None


def get_deals_record_string() -> str:
    """
    Fetch all deals from DB and build a pipe-separated record per deal:
    deal_id|target_name|acquirer_name|target_aliases|parent_aliases
    Aliases are comma-separated (target_aliases and parent_aliases from ProcessingJob).
    Only include deals whose `deal_status` is one of:
      - "Open"
      - "Unknown"
      - null/None
      - "" (empty string)
    Deals that don't have a `deal_status` attribute are also included.

    Returns:
        Newline-separated string of records.
    """
    try:
        from document_processor.models import ProcessingJob
    except ImportError:
        logger.warning(
            "document_processor not available; no deals for classification")
        return ""

    records: List[str] = []
    # Prefer DB-side filtering when possible, but fall back to Python filtering
    # if the model/field isn't compatible in the current environment.
    try:
        from django.db.models import Q  # type: ignore

        jobs_qs = ProcessingJob.objects.filter(
            Q(deal_status__in=["Open", "Unknown"]) |
            Q(deal_status__isnull=True) |
            Q(deal_status__exact=""))
    except Exception:
        jobs_qs = ProcessingJob.objects.all()

    for job in jobs_qs:
        # Include if:
        # - `deal_status` attribute is missing (older/newer schema), or
        # - value is exactly one of: "Open", "Unknown", None, "".
        if hasattr(job, "deal_status"):
            deal_status_val = getattr(job, "deal_status", None)
            if deal_status_val not in ("Open", "Unknown", None, ""):
                continue
        # else: no deal_status attribute -> include

        deal_id = str(job.id)
        target = (job.target_name or "").strip() or "N/A"
        acquirer = (job.acquire_name or "").strip() or "N/A"
        target_aliases = getattr(job, "target_aliases", None) or []
        parent_aliases = getattr(job, "parent_aliases", None) or []
        target_aliases_str = ",".join(str(a).strip()
                                      for a in target_aliases if a)
        parent_aliases_str = ",".join(str(a).strip()
                                      for a in parent_aliases if a)
        records.append(
            f"{deal_id}|{target}|{acquirer}|{target_aliases_str}|{parent_aliases_str}")
    return "\n".join(records)


# --- Prompt 1: Does this article say anything about a deal we follow? Return true+deal_id or false. ---
PROMPT_1_DEAL_WE_FOLLOW = """We have a list of deals we follow.

 Use web search to open and read the article at this URL. 
 
 DEAL RECORDS WE FOLLOW (one per line, 
 format: deal_id|target_name|acquirer_name|target_aliases|parent_aliases):
 
  {deals_record} 
  
  ARTICLE URL: {article_url} 
  
  Does this article say anything about any of these deals (e.g. news, update, or mention of one of these target/acquirer names or aliases)? 
  
  If YES, return the matching deal_id and provide additional details about the match.
  
  If NO, return match false. 
  
  Return ONLY a JSON object with these exact keys:
  - "match": true or false
  - "deal_id": the matching deal_id string or null
  - "matched_side": one of "target", "acquirer", or "both" (which company names/aliases were mentioned: target company, acquirer/parent company, or both). Return null if no match.
  - "match_keywords": an array of strings—the specific names, terms, or phrases from the deal record that appeared in the article and triggered the match (e.g. company names, tickers, aliases). Return null or empty array if no match.
  
  Example response format:
  {{"match": true, "deal_id": "abc123", "matched_side": "target", "match_keywords": ["XYZ Corp", "XYZ Corporation"]}}
  
  Use deal_id, matched_side, and match_keywords only when match is true.
"""

# --- Prompt 2: Is this article a self-announce of a new merger? Extract deal fields. ---
PROMPT_2_SELF_ANNOUNCE_EXTRACT = """Use web search or a browser to open and read the article at this URL.

ARTICLE URL: {article_url}

1. Determine whether this article is a formal announcement of a NEW merger or acquisition transaction.

Return true ONLY if the primary purpose of the article is to announce:
• A company acquiring another company,
• A merger agreement between companies,
• A company acquiring a meaningful business unit, core operating assets, or intellectual property of another company.

Return false if the article is only about:
• Real estate purchases unrelated to acquiring a business,
• Partnerships or collaborations,
• Financing or debt transactions,
• Executive hires,
• Product launches,
• Growth strategy commentary,
• Retrospective discussion of past deals,
• Industry trend commentary,
• Regulatory filings without a new transaction announcement.

2. If and only if (1) is true, extract the deal details.

Return ONLY a JSON object with these exact keys (use null for unknown):

- "is_it_self_announce_merger": true or false
- "target_name": Legal name of the company being acquired
- "acquire_name": Legal name of the acquiring company / parent / buyer
- "cik": Target company CIK (10 digits, leading zeros) if public; otherwise null
- "acquirer_cik": Acquirer company CIK (10 digits, leading zeros) if public; otherwise null
- "announce_date": Official transaction announcement or signing date in YYYY-MM-DD format
    • This must be the deal announcement/signing date.
    • Do NOT use article publish date unless explicitly stated as announcement date.
    • If multiple dates exist, prefer the date of entry into the merger agreement.
- "sec_ex_2_1_url": Direct URL to SEC EX-2.1 merger agreement document (.htm) on sec.gov
    • Only return URL if document type is EX-2.1.
    • Do NOT return 8-K index pages.
    • Do NOT return S-4 cover pages.
    • Do NOT return press releases.
    • If no EX-2.1 exists, return null.

Return only the JSON object. No explanation.
"""

# --- Prompt 3: Is target company US listed and market cap > $100M? ---
PROMPT_3_US_LISTED_MARKET_CAP = """Use web search to determine listing and market capitalization information for the TARGET company.

ARTICLE URL: {article_url}

Company details (from the article):
- Target company: {target_name}
- Acquirer company: {acquire_name}

Determine the following about the TARGET company (the one being acquired):

1. "is_us_listed":
   Return true ONLY if the target company itself (not its parent unless clearly the same entity) is publicly traded on a US exchange such as:
   • NYSE
   • NASDAQ
   • NYSE American
   • NYSE Arca
   • OTC Markets (if publicly traded in the US)

   Return false if:
   • The company is private
   • The company is listed only on a non-US exchange
   • Only its parent company is US listed
   • Listing cannot be confirmed

2. "is_market_cap_gt_100m":
   Return true ONLY if reliable sources confirm that the target company's current or most recently reported market capitalization exceeds USD $100,000,000.
   • Use recent financial sources (exchange site, Yahoo Finance, SEC filings, etc.).
   • If the company is not publicly traded, return false.
   • If market cap cannot be reliably determined, return false.

Important:
• Verify the correct legal entity before answering.
• Do not assume based on name similarity.
• If uncertain, return false.

Return ONLY a JSON object:
{{"is_us_listed": true|false, "is_market_cap_gt_100m": true|false}}

No explanation.
"""


# --- Title/description only: does this feed item mention a deal we follow? (no URL fetch) ---
PROMPT_TITLE_DESC_DEAL_MATCH = """We have a list of deals we follow.

DEAL RECORDS WE FOLLOW (one per line, format: deal_id|target_name|acquirer_name|target_aliases|parent_aliases):

{deals_record}

Feed item title: {title}

Feed item description:
{description}

Based ONLY on the title and description above (no web search), does the title or description say anything about any of these deals (e.g. mention of target/acquirer names or aliases, or news about one of these deals)?

If YES, return the matching deal_id.
If NO, return match false.

Return ONLY a JSON object: {{"match": true|false, "deal_id": "<id>"|null}}
Use deal_id only when match is true.
"""


def _call_llm_json_simple(prompt: str, model: str = "gpt-5.2") -> Optional[Dict[str, Any]]:
    """Call OpenAI API without web search; parse first JSON object from output_text."""
    if not openai or not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": "low"},
        )
        result_text = None
        for item in response.output:
            if getattr(item, "type", None) == "message" and hasattr(item, "content"):
                for content_item in item.content:
                    if getattr(content_item, "type", None) == "output_text":
                        result_text = getattr(content_item, "text", None)
                        break
            if result_text:
                break
        if not result_text:
            return None
        match = re.search(r"\{[\s\S]*?\}", result_text)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        logger.warning("LLM simple call failed: %s", e)
    return None


def classify_feed_item_by_title_description(
    deals_record_string: str,
    title: str,
    description: str,
) -> Dict[str, Any]:
    """
    Classify a feed item by title and description only (no URL fetch).
    Ask LLM: does title or description say anything about a deal we follow? If yes, return deal_id.

    Returns: {"match": bool, "deal_id": str|None}
    """
    out = {"match": False, "deal_id": None}
    if not (deals_record_string or "").strip():
        return out
    parsed = _call_llm_json_simple(
        PROMPT_TITLE_DESC_DEAL_MATCH.format(
            deals_record=deals_record_string or "(no deals)",
            title=(title or "").strip() or "(no title)",
            description=(description or "").strip() or "(no description)",
        ),
        model="gpt-5.2",
    )
    if not parsed or not isinstance(parsed, dict):
        return out
    out["match"] = bool(parsed.get("match"))
    did = parsed.get("deal_id")
    if did is not None:
        out["deal_id"] = str(did).strip() or None
    return out


def _call_llm_json_with_web_search(
    prompt: str, model: str = "gpt-5.2"
) -> Optional[Dict[str, Any]]:
    """Call OpenAI Responses API with web_search tool; parse first JSON object from output_text."""
    if not openai or not os.environ.get("OPENAI_API_KEY"):
        return None
    try:

        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.responses.create(
            model=model,
            tools=[{"type": "web_search"}],
            input=prompt,
            reasoning={"effort": "medium"},
        )
        result_text = None
        for item in response.output:
            if getattr(item, "type", None) == "message" and hasattr(item, "content"):
                for content_item in item.content:
                    if getattr(content_item, "type", None) == "output_text":
                        result_text = getattr(content_item, "text", None)
                        break
            if result_text:
                break
        if not result_text:
            return None
        match = re.search(r"\{[\s\S]*?\}", result_text)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        logger.warning("LLM web search call failed: %s", e)
    return None


def prompt_1_deal_we_follow(
    article_url: str,
    deals_record_string: str,
) -> Dict[str, Any]:
    """
    Prompt 1: Does this article say anything about a deal we follow?
    Returns: { "match": bool, "deal_id": str|None, "matched_side": str|None, "match_keywords": list|None }
    """
    out = {"match": False, "deal_id": None,
           "matched_side": None, "match_keywords": None}
    parsed = _call_llm_json_with_web_search(
        PROMPT_1_DEAL_WE_FOLLOW.format(
            deals_record=deals_record_string or "(no deals)",
            article_url=article_url or "",
        )
    )
    if not parsed or not isinstance(parsed, dict):
        return out
    out["match"] = bool(parsed.get("match"))
    did = parsed.get("deal_id")
    if did is not None:
        out["deal_id"] = str(did).strip() or None

    # Parse and normalize matched_side
    matched_side = parsed.get("matched_side")
    if matched_side and isinstance(matched_side, str):
        matched_side_lower = matched_side.strip().lower()
        if matched_side_lower in ("target", "acquirer", "both"):
            out["matched_side"] = matched_side_lower

    # Parse and normalize match_keywords (array of strings)
    match_keywords = parsed.get("match_keywords")
    if match_keywords is not None and isinstance(match_keywords, list):
        keywords = [str(k).strip()
                    for k in match_keywords if k is not None and str(k).strip()]
        if keywords:
            # Cap at 20 keywords to avoid bloat
            out["match_keywords"] = keywords[:20]

    return out


def _normalize_p2_parsed(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Prompt 2 parsed response: CIK padding, key names."""
    out = {
        "is_it_self_announce_merger": False,
        "target_name": None,
        "acquire_name": None,
        "cik": None,
        "acquirer_cik": None,
        "announce_date": None,
        "sec_ex_2_1_url": None,
    }
    if not parsed:
        return out
    out["is_it_self_announce_merger"] = bool(
        parsed.get("is_it_self_announce_merger"))
    for key in ("target_name", "acquire_name", "cik", "acquirer_cik", "announce_date", "sec_ex_2_1_url"):
        if key in parsed and parsed[key] is not None:
            out[key] = str(parsed[key]).strip() or None
    if out["cik"]:
        out["cik"] = re.sub(r"\D", "", out["cik"]).zfill(10)[:10]
    if out["acquirer_cik"]:
        out["acquirer_cik"] = re.sub(
            r"\D", "", out["acquirer_cik"]).zfill(10)[:10]
    return out


def prompt_2_self_announce_extract(article_url: str) -> Dict[str, Any]:
    """
    Prompt 2: Is this article a self-announce of a new merger? Extract deal fields.
    Returns: is_it_self_announce_merger, target_name, acquire_name, cik, acquirer_cik, announce_date, sec_ex_2_1_url.
    """
    parsed = _call_llm_json_with_web_search(
        PROMPT_2_SELF_ANNOUNCE_EXTRACT.format(article_url=article_url or "")
    )
    if not parsed or not isinstance(parsed, dict):
        return _normalize_p2_parsed({})
    return _normalize_p2_parsed(parsed)


def prompt_3_us_listed_market_cap(
    article_url: str,
    target_name: str,
    acquire_name: str,
) -> Dict[str, Any]:
    """
    Prompt 3: Is target US listed and market cap > $100M?
    Returns: { "is_us_listed": bool, "is_market_cap_gt_100m": bool }
    """
    out = {"is_us_listed": False, "is_market_cap_gt_100m": False}
    parsed = _call_llm_json_with_web_search(
        PROMPT_3_US_LISTED_MARKET_CAP.format(
            article_url=article_url or "",
            target_name=target_name or "—",
            acquire_name=acquire_name or "—",
        )
    )
    if not parsed or not isinstance(parsed, dict):
        return out
    out["is_us_listed"] = bool(parsed.get("is_us_listed"))
    out["is_market_cap_gt_100m"] = bool(parsed.get("is_market_cap_gt_100m"))
    return out


EXTRACT_NEW_DEAL_PROMPT = """You are extracting M&A deal information from a news article. 

Use web search to open and read the article at this URL. 
If necessary, also search SEC EDGAR and official filings to verify deal details.

ARTICLE URL: {article_url}

Extract the following information. Return ONLY a JSON object with these exact keys (use null for unknown):

- "target_name": Legal name of the company being acquired.
- "acquire_name": Legal name of the acquiring company / parent / buyer.
- "cik": Target company CIK (10 digits, leading zeros).
- "acquirer_cik": Acquirer company CIK (10 digits, leading zeros).

- "announce_date": 
    The official deal announcement date or signing date of the merger/acquisition.
    This must be the date the transaction was publicly announced or signed.
    • Do NOT use the article publish date unless it clearly states the deal was announced that same day.
    • Prefer the date stated in the press release body (e.g., "Company A announced on March 5, 2026...")
    • If available, prefer the date from the SEC filing (e.g., 8-K filing date describing entry into merger agreement).
    • Format strictly as YYYY-MM-DD.
    • If unclear, return null.

- "sec_url": 
    Direct URL to SEC Exhibit 2.1 (EX-2.1) merger agreement document.

    Rules:
    1. The URL must be on sec.gov.
    2. The filing must include document type EX-2.1.
    3. Return the direct exhibit document link (e.g., ex2-1.htm or similar).
    4. Do NOT return:
        - 8-K index pages
        - S-4 cover pages
        - Press releases
        - Investor relations pages
        - Non-SEC domains
        - EX-99.1
    5. If EX-2.1 cannot be found, return null.

Return ONLY the JSON object and no additional text."""


def _parse_extract_response(text: str) -> Dict[str, Any]:
    """Parse LLM extraction response."""
    out = {
        "target_name": None,
        "acquire_name": None,
        "cik": None,
        "acquirer_cik": None,
        "sec_url": None,
        "announce_date": None,
    }
    if not text or not isinstance(text, str):
        return out
    match = re.search(
        r"\{[^{}]*(?:target_name|acquire_name|cik|sec_url|announce)[^{}]*\}", text, re.DOTALL)
    if not match:
        match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                for key in out:
                    if key in parsed and parsed[key] is not None:
                        out[key] = str(parsed[key]).strip() or None
        except json.JSONDecodeError:
            pass
    # Normalize CIKs to 10 digits
    for key in ("cik", "acquirer_cik"):
        if out[key]:
            out[key] = re.sub(r"\D", "", out[key]).zfill(10)[:10] or None
    return out


def extract_new_deal_with_web_search(
    article_html: str,
    article_url: str,
) -> Dict[str, Any]:
    """
    Use LLM with web search to extract target/acquirer names, CIKs, SEC URL, announce date.
    Reads the article via web search (article_url only; article_html is unused).

    Returns:
        Dict with keys: target_name, acquire_name, cik, acquirer_cik, sec_url, announce_date (strings or None).
    """
    if not openai or not os.environ.get("OPENAI_API_KEY"):
        return {
            "target_name": None,
            "acquire_name": None,
            "cik": None,
            "acquirer_cik": None,
            "sec_url": None,
            "announce_date": None,
        }

    prompt = EXTRACT_NEW_DEAL_PROMPT.format(article_url=article_url or "")

    try:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.responses.create(
            model="gpt-5.2",
            tools=[{"type": "web_search"}],
            input=prompt,
            reasoning={"effort": "low"},
        )
        result_text = None
        for item in response.output:
            if getattr(item, "type", None) == "message" and hasattr(item, "content"):
                for content_item in item.content:
                    if getattr(content_item, "type", None) == "output_text":
                        result_text = getattr(content_item, "text", None)
                        break
            if result_text:
                break
        return _parse_extract_response(result_text or "")
    except Exception as e:
        logger.warning("Extract new deal (web search) failed: %s", e)
        return {
            "target_name": None,
            "acquire_name": None,
            "cik": None,
            "acquirer_cik": None,
            "sec_url": None,
            "announce_date": None,
        }


def _parse_announce_date(value: Optional[str]) -> Optional[datetime]:
    """Parse announce_date string YYYY-MM-DD to datetime."""
    if not value or not isinstance(value, str):
        return None
    value = value.strip()[:10]
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None


def create_deal_from_extracted(extracted: Dict[str, Any]) -> Optional[Any]:
    """
    Create a new ProcessingJob (deal) in the deals collection from extracted fields.

    Args:
        extracted: Dict with target_name, acquire_name, cik, acquirer_cik, sec_url, announce_date.

    Returns:
        ProcessingJob instance or None on failure.
    """
    try:
        from document_processor.models import ProcessingJob
    except ImportError:
        logger.warning("document_processor not available; cannot create deal")
        return None

    target_name = (extracted.get("target_name") or "").strip() or None
    acquire_name = (extracted.get("acquire_name") or "").strip() or None
    if not target_name and not acquire_name:
        logger.warning("Cannot create deal: no target_name or acquire_name")
        return None

    job = ProcessingJob(
        target_name=target_name,
        acquire_name=acquire_name,
        cik=extracted.get("cik") or None,
        acquirer_cik=extracted.get("acquirer_cik") or None,
        sec_url=extracted.get("sec_url") or None,
        announce_date=_parse_announce_date(extracted.get("announce_date")),
        embedding_status="PENDING",
        summary_status="PENDING",
        summary_using="openai-gpt-4",
        file_url="https://placeholder.com",
        pdf_url="https://placeholder.com",
        deal_status="Unknown"
    )
    job.save()
    logger.info("Created new deal from RSS news: %s / %s (ID: %s)",
                acquire_name, target_name, job.id)
    return job


def get_deal_info_for_email(deal_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a single deal by ID and return a dict suitable for email template (deal_info).

    Returns:
        Dict with id, target_name, acquire_name, cik, acquirer_cik, sec_url, announce_date, or None.
    """
    try:
        from bson import ObjectId
        from document_processor.models import ProcessingJob
    except ImportError:
        return None

    if not deal_id:
        return None
    try:
        job = ProcessingJob.objects.get(id=ObjectId(deal_id))
    except Exception:
        return None

    return {
        "id": str(job.id),
        "target_name": job.target_name or "",
        "acquire_name": job.acquire_name or "",
        "cik": job.cik or "",
        "acquirer_cik": job.acquirer_cik or "",
        "sec_url": job.sec_url or "",
        "announce_date": job.announce_date.strftime("%Y-%m-%d") if job.announce_date else "",
    }


def deal_info_from_extracted(
    extracted: Dict[str, Any],
    in_db: bool = False,
    deal_id: Optional[str] = None,
    is_target_us_listed: Optional[bool] = None,
    is_target_market_cap_gt_100m: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build deal_info dict from extracted fields (for email). Optional US listed / market cap for display."""
    sec_url = extracted.get("sec_url") or extracted.get("sec_ex_2_1_url") or ""
    info = {
        "id": deal_id or "",
        "target_name": (extracted.get("target_name") or "").strip() or "—",
        "acquire_name": (extracted.get("acquire_name") or "").strip() or "—",
        "cik": extracted.get("cik") or "",
        "acquirer_cik": extracted.get("acquirer_cik") or "",
        "sec_url": sec_url,
        "announce_date": extracted.get("announce_date") or "",
        "in_db": in_db,
    }
    if is_target_us_listed is not None:
        info["is_target_us_listed"] = is_target_us_listed
    if is_target_market_cap_gt_100m is not None:
        info["is_target_market_cap_gt_100m"] = is_target_market_cap_gt_100m
    return info


def _extracted_to_create_payload(p2: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Prompt 2 response to payload for create_deal_from_extracted (sec_ex_2_1_url -> sec_url)."""
    return {
        "target_name": p2.get("target_name"),
        "acquire_name": p2.get("acquire_name"),
        "cik": p2.get("cik"),
        "acquirer_cik": p2.get("acquirer_cik"),
        "announce_date": p2.get("announce_date"),
        "sec_url": p2.get("sec_ex_2_1_url"),
    }


# Result keys: skip_email, deal_id, deal_info, email_note
# email_note: "existing_deal" | "new_deal_in_db" | "new_deal_not_in_db"
def resolve_rss_item_flow(
    item: Dict[str, Any],
    deals_record_string: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run the 3-prompt flow for one RSS item.

    Prompt 1: Does article mention a deal we follow? -> true+deal_id → attach deal in email, save item with deal_id.
    Prompt 2 (if False): Is it self-announce new merger? If not self-announce -> send email with not_merger_related, do not save, do not run Prompt 3.
    Prompt 3 (if self-announce): US listed and market cap > $100M? If true -> create deal, attach deal_id. If false -> no deal in DB but send email with deal info + US listed/market cap.

    Returns: skip_email, deal_id, deal_info, email_note.
    """
    source_url = (item.get("url") or "").strip()
    result = {
        "skip_email": True,
        "deal_id": None,
        "deal_info": None,
        "email_note": None,
    }
    if not source_url:
        return result

    # Prompt 1: Does this article say anything about a deal we follow?
    p1 = prompt_1_deal_we_follow(
        article_url=source_url, deals_record_string=deals_record_string)
    match = p1.get("match", False)
    deal_id = p1.get("deal_id")
    logger.info(f"Prompt 1 match: {match}, deal_id: {deal_id}")

    if match and deal_id:
        result["skip_email"] = False
        result["deal_id"] = deal_id
        result["deal_info"] = get_deal_info_for_email(deal_id)
        result["email_note"] = "existing_deal"
        if result["deal_info"]:
            result["deal_info"]["in_db"] = True

        # Add match details from Prompt 1
        match_details = {}
        if p1.get("matched_side"):
            match_details["matched_side"] = p1["matched_side"]
        if p1.get("match_keywords"):
            match_details["match_keywords"] = p1["match_keywords"]
        if match_details:
            result["match_details"] = match_details

        logger.debug("Prompt 1 match deal_id: %s, matched_side: %s",
                     deal_id, p1.get("matched_side"))
        return result

    # Prompt 2: Is it self-announce new merger? Extract deal fields.
    p2 = prompt_2_self_announce_extract(article_url=source_url)
    logger.info(f"Prompt 2: {p2}")
    is_self_announce = p2.get("is_it_self_announce_merger", False)
    logger.debug(
        "Prompt 2 is_it_self_announce_merger: %s, extracted: %s", is_self_announce, p2)

    if not is_self_announce:
        result["skip_email"] = False
        result["email_note"] = "not_merger_related"
        return result

    result["skip_email"] = False

    # Prompt 3: US listed and market cap > $100M?
    p3 = prompt_3_us_listed_market_cap(
        article_url=source_url,
        target_name=p2.get("target_name") or "",
        acquire_name=p2.get("acquire_name") or "",
    )
    is_us_listed = p3.get("is_us_listed", False)
    is_market_cap_gt_100m = p3.get("is_market_cap_gt_100m", False)
    logger.info("Prompt 3 is_us_listed: %s, is_market_cap_gt_100m: %s",
                is_us_listed, is_market_cap_gt_100m)

    if is_us_listed and is_market_cap_gt_100m and not dry_run:
        payload = _extracted_to_create_payload(p2)
        logger.info(f"Payload: {payload}")
        new_deal = create_deal_from_extracted(payload)
        logger.info(f"New deal: {new_deal}")
        if new_deal:
            result["deal_id"] = str(new_deal.id)
            result["deal_info"] = get_deal_info_for_email(str(new_deal.id))
            result["email_note"] = "new_deal_in_db"
            if result["deal_info"]:
                result["deal_info"]["in_db"] = True
                result["deal_info"]["is_target_us_listed"] = is_us_listed
                result["deal_info"]["is_target_market_cap_gt_100m"] = is_market_cap_gt_100m
        else:
            logger.info(f"New deal not in db")
            result["deal_info"] = deal_info_from_extracted(
                p2, in_db=False, is_target_us_listed=is_us_listed, is_target_market_cap_gt_100m=is_market_cap_gt_100m
            )
            result["email_note"] = "new_deal_not_in_db"
    else:
        result["deal_info"] = deal_info_from_extracted(
            p2, in_db=False, is_target_us_listed=is_us_listed, is_target_market_cap_gt_100m=is_market_cap_gt_100m
        )
        result["email_note"] = "new_deal_not_in_db"

    logger.info(f"Final result: {result}")
    return result
