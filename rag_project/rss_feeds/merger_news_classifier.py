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
    for job in ProcessingJob.objects.all():
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


# --- Prompt 1: Is this merger-related? Is it a self-announce of a new merger? (web search to read article) ---
PROMPT_1_MERGER_CHECK = """You are classifying a news article. Use web search or a browser to open and read the full article at this URL.

ARTICLE URL: {article_url}
ARTICLE TITLE: {article_title}

Then answer two questions. Return ONLY a JSON object with no other text.

1. "merger_related":
   Is this article related to any merger, acquisition, divestiture, asset purchase, business purchase, corporate transaction, or strategic investment activity?
   Include corporate acquisitions, business unit sales, property acquisitions, majority stake purchases, and buyouts.
   true or false.

2. "is_self_announce_new_merger":
   Does this article ITSELF formally announce a specific new corporate merger or acquisition between companies?
   Only return true if the primary purpose is to announce a company acquiring or merging with another company.
   Exclude property-only purchases, financing transactions, internal restructurings, or commentary on past deals.

Format: {{"merger_related": true|false, "is_self_announce_new_merger": true|false}}"""


# --- Prompt 2: If merger-related, is this deal in our database? Return deal_id. (web search to read article) ---
PROMPT_2_MATCH_DEAL_ID = """We have a merger-related article. Use web search to open and read the article, then check if the deal is in our database.

DEAL RECORDS IN OUR DATABASE (one per line, format: deal_id|target_name|acquirer_name|target_aliases|parent_aliases).
{deals_record}

ARTICLE URL: {article_url}
ARTICLE TITLE: {article_title}

If the article is about ONE of the deals in the list above, return that deal's deal_id. If NOT in our database, return null for deal_id.
Return ONLY a JSON object: {{"deal_id": "<id>"|null}}"""


def _truncate_html(html: str, max_chars: int = 12000) -> str:
    """Truncate HTML for the prompt to avoid token limits."""
    if not html:
        return ""
    text = re.sub(r"\s+", " ", html).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


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
        logger.warning("LLM web search call failed: %s", e)
    return None


def prompt_1_merger_check(
    article_html: str,
    article_url: str,
    article_title: str,
) -> Dict[str, Any]:
    """
    Prompt 1: Is this merger-related? Is it a self-announce of a new merger?
    Uses web search to read the article. Returns: { "merger_related": bool, "is_self_announce_new_merger": bool }
    """
    out = {"merger_related": False, "is_self_announce_new_merger": False}
    parsed = _call_llm_json_with_web_search(
        PROMPT_1_MERGER_CHECK.format(
            article_url=article_url or "",
            article_title=article_title or "",
        )
    )
    if not parsed or not isinstance(parsed, dict):
        return out
    out["merger_related"] = bool(parsed.get("merger_related"))
    out["is_self_announce_new_merger"] = bool(
        parsed.get("is_self_announce_new_merger"))
    return out


def prompt_2_match_deal_id(
    article_html: str,
    article_url: str,
    article_title: str,
    deals_record_string: str,
) -> Optional[str]:
    """
    Prompt 2: If merger-related, is this deal in our database? Return deal_id or None.
    Uses web search to read the article.
    """
    parsed = _call_llm_json_with_web_search(
        PROMPT_2_MATCH_DEAL_ID.format(
            deals_record=deals_record_string or "(no deals)",
            article_url=article_url or "",
            article_title=article_title or "",
        )
    )
    if not parsed or not isinstance(parsed, dict):
        return None
    did = parsed.get("deal_id")
    if did is None:
        return None
    return str(did).strip() or None


EXTRACT_NEW_DEAL_PROMPT = """You are extracting M&A deal information from a news article. Use web search if needed to find official details (SEC filing, company names, CIKs, announcement date).

ARTICLE URL: {article_url}

ARTICLE HTML (excerpt):
{article_html_excerpt}

Extract the following. Use web search to find SEC filing and CIKs when possible. Return ONLY a JSON object with these exact keys (use null for unknown):
- "target_name": target company name (being acquired)
- "acquire_name": acquirer/parent/buyer company name (acquiring)
- "cik": target company CIK (10 digits, leading zeros)
- "acquirer_cik": acquirer company CIK (10 digits, leading zeros)
- "sec_url": URL of main SEC filing for this deal (e.g. merger agreement or EX-2.1), or null
- "announce_date": announcement date in YYYY-MM-DD format, or null

Return only the JSON object, no other text."""


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

    excerpt = _truncate_html(article_html or "", max_chars=10000)
    prompt = EXTRACT_NEW_DEAL_PROMPT.format(
        article_url=article_url or "",
        article_html_excerpt=excerpt or "(no content)",
    )

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


def deal_info_from_extracted(extracted: Dict[str, Any], in_db: bool = False, deal_id: Optional[str] = None) -> Dict[str, Any]:
    """Build deal_info dict from extracted fields (for email when deal not in DB or for display)."""
    return {
        "id": deal_id or "",
        "target_name": (extracted.get("target_name") or "").strip() or "—",
        "acquire_name": (extracted.get("acquire_name") or "").strip() or "—",
        "cik": extracted.get("cik") or "",
        "acquirer_cik": extracted.get("acquirer_cik") or "",
        "sec_url": extracted.get("sec_url") or "",
        "announce_date": extracted.get("announce_date") or "",
        "in_db": in_db,
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

    When dry_run=True, never create a new deal in DB (still extract and return deal_info from extraction).

    Returns dict with:
      - skip_email: bool. If True, do not save item and do not send email.
      - deal_id: Optional[str]. Set when matched or when we created a new deal (for saving on FeedItem).
      - deal_info: Optional[dict]. For email template.
      - email_note: "existing_deal" | "new_deal_in_db" | "new_deal_not_in_db" | None (when skip_email).
    """
    source_url = (item.get("url") or "").strip()
    article_title = (item.get("title") or "").strip()
    result = {
        "skip_email": True,
        "deal_id": None,
        "deal_info": None,
        "email_note": None,
    }
    if not source_url:
        return result

    # We always use web search to read the article (no HTML fetch)
    html = ""

    # Prompt 1
    p1 = prompt_1_merger_check(
        article_html=html or "", article_url=source_url, article_title=article_title)
    merger_related = p1.get("merger_related", False)
    is_self_announce = p1.get("is_self_announce_new_merger", False)
    print(
        f"merger_related: {merger_related}, is_self_announce: {is_self_announce}")
    if not merger_related:
        return result

    result["skip_email"] = False

    # Prompt 2: match deal in DB
    deal_id = prompt_2_match_deal_id(
        article_html=html or "",
        article_url=source_url,
        article_title=article_title,
        deals_record_string=deals_record_string,
    )
    print(f"deal_id: {deal_id}")

    if deal_id:
        result["deal_id"] = deal_id
        result["deal_info"] = get_deal_info_for_email(deal_id)
        result["email_note"] = "existing_deal"
        if result["deal_info"]:
            result["deal_info"]["in_db"] = True
        return result

    # Prompt 3: extract deal info (deal not in DB)
    extracted = extract_new_deal_with_web_search(
        article_html=html or "", article_url=source_url)
    print(f"extracted: {extracted}")
    if is_self_announce and not dry_run:
        new_deal = create_deal_from_extracted(extracted)
        if new_deal:
            result["deal_id"] = str(new_deal.id)
            result["deal_info"] = get_deal_info_for_email(str(new_deal.id))
            result["email_note"] = "new_deal_in_db"
            if result["deal_info"]:
                result["deal_info"]["in_db"] = True
        else:
            result["deal_info"] = deal_info_from_extracted(
                extracted, in_db=False)
            result["email_note"] = "new_deal_not_in_db"
    else:
        result["deal_info"] = deal_info_from_extracted(extracted, in_db=False)
        result["email_note"] = "new_deal_not_in_db"
    print(f"result: {result}")
    return result
