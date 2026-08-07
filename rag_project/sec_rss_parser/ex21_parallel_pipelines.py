"""
Non-blocking side pipelines kicked off after EX-2.1 parse + deal creation.

1) Extract Notices section text from parsed JSON.
2) Find the most recent 10-K for the deal's target CIK before announce_date
   via sec_Last_Year.print_filings (SEC submissions API).

Results are logged and emailed via send_direct_email → N8N_WEBHOOK_INTERNAL.
Failures must never affect the main EX-2.1 / process_document flow.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from html import escape
from typing import Any, Optional

import requests
from bson import ObjectId

from document_processor.models import ProcessingJob
from sec_rss_parser.email_service.email_dispatch_service import send_direct_email
from sec_rss_parser.sec_Last_Year import print_filings
from sec_rss_parser.utils_8k import normalize_cik

logger = logging.getLogger(__name__)

LOG_PREFIX = "ex21_side_pipeline"
# Look back this far when filtering for 10-Ks before announce_date (annual filings).
PREVIOUS_10K_LOOKBACK_DAYS = 1095

N8N_WEBHOOK_INTERNAL = os.environ.get(
    "N8N_WEBHOOK_INTERNAL",
    "https://n8n.arbintel.cloud/webhook/80830c6d-ff5b-45e3-9ef3-a061db1fbf0c",
)
SIDE_PIPELINE_EMAIL_RECIPIENTS = [
    "avshesh.savani@teqnodux.com",
    "kaushal.devani@gmail.com",
]
NOTICES_EMAIL_TEXT_CHARS = 800


def _coerce_announce_datetime(value) -> Optional[datetime]:
    """Accept datetime or string announce_date from Mongo deals."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if hasattr(value, "strftime") and hasattr(value, "year"):
        # date-like without being datetime
        try:
            return datetime(value.year, value.month, value.day)
        except Exception:
            pass
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y"):
        try:
            return datetime.strptime(text[:19] if "T" in text or " " in text else text[:10], fmt)
        except ValueError:
            continue
    # ISO with timezone / fractional seconds — take date prefix
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            pass
    return None


def _log_json(pipeline: str, payload: dict) -> None:
    message = f"{LOG_PREFIX}:{pipeline}: {json.dumps(payload, default=str)}"
    logger.info(message)
    print(message)


def _normalize_section_title(title: str) -> str:
    text = (title or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.:;,\-–—]+$", "", text).strip()
    return text


def _notices_match_score(title: str) -> tuple[int, str]:
    """
    Return (score, match_type). Higher score = better Notices match.
    Prefer exact Notices over fuzzy Notice / starts-with.
    """
    norm = _normalize_section_title(title)
    if not norm:
        return 0, ""
    if norm in ("notices", "notice"):
        return 100 if norm == "notices" else 90, "exact"
    if norm.startswith("notices ") or norm.startswith("notice "):
        # Avoid clause titles like "Notice of Competing Offer"
        if "of " in norm or "competing" in norm or "termination" in norm:
            return 0, ""
        return 70, "fuzzy_prefix"
    if norm.endswith(" notices") or norm.endswith(" notice"):
        return 60, "fuzzy_suffix"
    return 0, ""


def _iter_sections(parsed: Any):
    """Yield (article_name, section_dict) from common parsed-JSON shapes."""
    articles = None
    if isinstance(parsed, list):
        articles = parsed
    elif isinstance(parsed, dict):
        articles = parsed.get("articles")
        if articles is None and "sections" in parsed:
            for section in parsed.get("sections") or []:
                if isinstance(section, dict):
                    yield parsed.get("article") or parsed.get("title") or "", section
            return

    if not isinstance(articles, list):
        return

    for article in articles:
        if not isinstance(article, dict):
            continue
        article_name = article.get("article") or article.get("title") or ""
        for section in article.get("sections") or []:
            if isinstance(section, dict):
                yield article_name, section


def _load_parsed_json(parsed_json_url: str) -> Any:
    """Load parsed JSON from an HTTP(S) URL or a local filesystem path."""
    source = (parsed_json_url or "").strip()
    if not source:
        raise ValueError("missing parsed_json_url")
    if source.startswith("http://") or source.startswith("https://"):
        resp = requests.get(source, timeout=60)
        resp.raise_for_status()
        return resp.json()
    # Local path (useful for management-command testing)
    from pathlib import Path

    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"parsed JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_notices_from_parsed_json(deal_id: str, parsed_json_url: str) -> dict:
    """Download/load parsed JSON and extract the best Notices / Notice section."""
    result = {
        "status": "FAILED",
        "deal_id": deal_id,
        "parsed_json_url": parsed_json_url,
        "matched_title": None,
        "section": None,
        "article": None,
        "text": None,
        "match_type": None,
        "error": None,
    }
    try:
        if not parsed_json_url:
            result["status"] = "SKIPPED"
            result["error"] = "missing parsed_json_url"
            return result

        parsed = _load_parsed_json(parsed_json_url)

        best = None
        best_score = 0
        for article_name, section in _iter_sections(parsed):
            title = section.get("title") or ""
            score, match_type = _notices_match_score(title)
            if score > best_score:
                best_score = score
                best = {
                    "matched_title": title,
                    "section": section.get("section"),
                    "article": article_name,
                    "text": section.get("text") or "",
                    "match_type": match_type,
                }

        if not best or not (best.get("text") or "").strip():
            result["status"] = "NOT_FOUND"
            result["error"] = "No Notices/Notice section found in parsed JSON"
            return result

        result.update(best)
        result["status"] = "COMPLETED"
        result["error"] = None
        return result
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
        logger.exception(
            "%s:notices: error for deal_id=%s", LOG_PREFIX, deal_id
        )
        return result


def find_previous_10k_by_cik(
    target_cik: str,
    announce_date: Optional[datetime] = None,
    deal_id: Optional[str] = None,
) -> dict:
    """
    Use sec_Last_Year.print_filings to find the latest 10-K before announce_date.
    """
    result = {
        "status": "FAILED",
        "deal_id": deal_id,
        "target_cik": None,
        "announce_date": None,
        "form": None,
        "filing_date": None,
        "accession_number": None,
        "primary_document": None,
        "link": None,
        "source_url": None,
        "error": None,
    }
    try:
        cik = normalize_cik(target_cik) if target_cik else ""
        result["target_cik"] = cik or None
        announce_dt = _coerce_announce_datetime(announce_date)
        announce_str = announce_dt.strftime("%Y-%m-%d") if announce_dt else None
        if announce_str:
            result["announce_date"] = announce_str

        if not cik:
            result["status"] = "SKIPPED"
            result["error"] = "deal has no target cik"
            return result

        # print_filings keeps filing_date >= start_date; look back then filter
        # strictly before announce_date in Python.
        if announce_dt:
            lookback = (
                announce_dt - timedelta(days=PREVIOUS_10K_LOOKBACK_DAYS)
            ).strftime("%Y-%m-%d")
        else:
            lookback = (
                datetime.utcnow() - timedelta(days=PREVIOUS_10K_LOOKBACK_DAYS)
            ).strftime("%Y-%m-%d")

        source_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        result["source_url"] = source_url

        filings = print_filings(
            cik,
            start_date=lookback,
            form_types=["10-K"],
        ) or []

        candidates = []
        for filing in filings:
            filing_date = (filing.get("filing_date") or "").strip()
            if not filing_date:
                continue
            if announce_str and filing_date >= announce_str:
                continue
            candidates.append(filing)

        if not candidates:
            result["status"] = "NOT_FOUND"
            result["error"] = (
                "No 10-K filing found before announce_date"
                if announce_dt
                else "No 10-K filing found via sec_Last_Year.print_filings"
            )
            return result

        best = max(candidates, key=lambda f: f.get("filing_date") or "")
        result.update(
            {
                "status": "COMPLETED",
                "form": best.get("form") or "10-K",
                "filing_date": best.get("filing_date"),
                "accession_number": best.get("accession_number"),
                "primary_document": best.get("primary_document"),
                "link": best.get("url"),
                "error": None,
            }
        )
        return result
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
        logger.exception(
            "%s:previous_10k: error for deal_id=%s cik=%s",
            LOG_PREFIX,
            deal_id,
            target_cik,
        )
        return result


def find_previous_10k_for_deal(deal_id: str) -> dict:
    """Load deal target CIK + announce_date, then find previous 10-K."""
    try:
        job = ProcessingJob.objects.get(id=ObjectId(deal_id))
    except Exception as e:
        return {
            "status": "FAILED",
            "deal_id": deal_id,
            "target_cik": None,
            "announce_date": None,
            "form": None,
            "filing_date": None,
            "accession_number": None,
            "primary_document": None,
            "link": None,
            "source_url": None,
            "error": f"deal not found: {e}",
        }
    return find_previous_10k_by_cik(
        target_cik=job.cik or "",
        announce_date=job.announce_date,
        deal_id=deal_id,
    )


# Backward-compatible aliases (older test/command imports).
find_previous_10q_by_cik = find_previous_10k_by_cik
find_previous_10q_for_deal = find_previous_10k_for_deal


def _run_notices_pipeline(deal_id: str, parsed_json_url: str) -> dict:
    payload = extract_notices_from_parsed_json(deal_id, parsed_json_url)
    # Avoid dumping huge clause text into a single unreadable log line when long
    log_payload = dict(payload)
    text = log_payload.get("text") or ""
    if isinstance(text, str) and len(text) > 500:
        log_payload["text"] = text[:500] + f"... [truncated, total_chars={len(text)}]"
        log_payload["text_length"] = len(text)
    _log_json("notices", log_payload)
    return payload


def _run_previous_10k_pipeline(deal_id: str) -> dict:
    payload = find_previous_10k_for_deal(deal_id)
    _log_json("previous_10k", payload)
    return payload


def _status_color(status: Optional[str]) -> str:
    s = (status or "").upper()
    if s == "COMPLETED":
        return "#0a7a2f"
    if s in ("NOT_FOUND", "SKIPPED"):
        return "#b36b00"
    return "#b00020"


def _build_side_pipeline_email_html(
    deal_id: str,
    notices: dict,
    tenk: dict,
) -> str:
    notices_text = notices.get("text") or ""
    notices_preview = notices_text[:NOTICES_EMAIL_TEXT_CHARS]
    if len(notices_text) > NOTICES_EMAIL_TEXT_CHARS:
        notices_preview += f"... [truncated, total_chars={len(notices_text)}]"

    def row(label: str, value: Any) -> str:
        display = "—" if value is None or value == "" else escape(str(value))
        return (
            f"<tr>"
            f"<td style='padding:6px 10px;border:1px solid #ddd;font-weight:600;"
            f"width:180px;background:#f7f7f7'>{escape(label)}</td>"
            f"<td style='padding:6px 10px;border:1px solid #ddd'>{display}</td>"
            f"</tr>"
        )

    n_status = notices.get("status")
    k_status = tenk.get("status")
    html = f"""
    <div style="font-family:Arial,sans-serif;font-size:14px;color:#222">
      <h2 style="margin:0 0 12px">EX-2.1 Side Pipelines Result</h2>
      <p style="margin:0 0 16px">deal_id: <code>{escape(str(deal_id))}</code></p>

      <h3 style="margin:18px 0 8px;color:{_status_color(n_status)}">
        1) Notices Extract — {escape(str(n_status or "UNKNOWN"))}
      </h3>
      <table style="border-collapse:collapse;width:100%;max-width:900px">
        {row("matched_title", notices.get("matched_title"))}
        {row("section", notices.get("section"))}
        {row("article", notices.get("article"))}
        {row("match_type", notices.get("match_type"))}
        {row("parsed_json_url", notices.get("parsed_json_url"))}
        {row("error", notices.get("error"))}
      </table>
      <pre style="white-space:pre-wrap;background:#f5f5f5;padding:12px;
                  border:1px solid #ddd;max-width:900px;margin-top:8px">{escape(notices_preview) or "—"}</pre>

      <h3 style="margin:24px 0 8px;color:{_status_color(k_status)}">
        2) Previous 10-K — {escape(str(k_status or "UNKNOWN"))}
      </h3>
      <table style="border-collapse:collapse;width:100%;max-width:900px">
        {row("target_cik", tenk.get("target_cik"))}
        {row("announce_date", tenk.get("announce_date"))}
        {row("form", tenk.get("form"))}
        {row("filing_date", tenk.get("filing_date"))}
        {row("accession_number", tenk.get("accession_number"))}
        {row("primary_document", tenk.get("primary_document"))}
        {row("link", tenk.get("link"))}
        {row("source_url", tenk.get("source_url"))}
        {row("error", tenk.get("error"))}
      </table>
    </div>
    """
    return html


def _send_side_pipeline_email(
    deal_id: str,
    notices: dict,
    tenk: dict,
) -> bool:
    """Email pipeline results via send_direct_email → N8N_WEBHOOK_INTERNAL."""
    try:
        n_ok = notices.get("status") == "COMPLETED"
        k_ok = tenk.get("status") == "COMPLETED"
        subject = (
            f"[EX21 Side Pipelines] deal={deal_id} "
            f"notices={'OK' if n_ok else notices.get('status')} "
            f"10K={'OK' if k_ok else tenk.get('status')}"
        )
        html = _build_side_pipeline_email_html(deal_id, notices, tenk)
        ok = send_direct_email(
            recipients=list(SIDE_PIPELINE_EMAIL_RECIPIENTS),
            payload={
                "subject": subject,
                "html": html,
                "deal_id": deal_id,
                "email_type": "ex21_side_pipelines",
                "notices_status": notices.get("status"),
                "previous_10k_status": tenk.get("status"),
            },
            webhook_url=N8N_WEBHOOK_INTERNAL,
        )
        logger.info(
            "%s:email: sent=%s deal_id=%s recipients=%s",
            LOG_PREFIX,
            ok,
            deal_id,
            SIDE_PIPELINE_EMAIL_RECIPIENTS,
        )
        print(
            f"{LOG_PREFIX}:email: sent={ok} deal_id={deal_id} "
            f"to={SIDE_PIPELINE_EMAIL_RECIPIENTS}"
        )
        return bool(ok)
    except Exception as e:
        logger.error("%s:email: failed for deal_id=%s: %s", LOG_PREFIX, deal_id, e)
        print(f"{LOG_PREFIX}:email: failed for deal_id={deal_id}: {e}")
        return False


def _run_side_pipelines_and_email(deal_id: str, parsed_json_url: str) -> None:
    """Run Notices + previous 10-K in parallel, then email combined result."""
    notices: dict = {}
    tenk: dict = {}
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_notices = executor.submit(
                _run_notices_pipeline, deal_id, parsed_json_url or ""
            )
            fut_tenk = executor.submit(_run_previous_10k_pipeline, deal_id)
            notices = fut_notices.result()
            tenk = fut_tenk.result()
    except Exception as e:
        logger.exception(
            "%s: error running side pipelines for deal_id=%s", LOG_PREFIX, deal_id
        )
        notices = notices or {
            "status": "FAILED",
            "deal_id": deal_id,
            "error": str(e),
        }
        tenk = tenk or {
            "status": "FAILED",
            "deal_id": deal_id,
            "error": str(e),
        }

    _send_side_pipeline_email(deal_id, notices, tenk)


def start_ex21_side_pipelines(deal_id: str, parsed_json_url: str) -> None:
    """
    Fire-and-forget both pipelines + result email.
    Safe to call from process_8k_document_async; never raises to the caller.
    """
    try:
        if not deal_id:
            logger.warning("%s: skip side pipelines — missing deal_id", LOG_PREFIX)
            return

        t = threading.Thread(
            target=_run_side_pipelines_and_email,
            args=(deal_id, parsed_json_url or ""),
            name=f"ex21-side-pipelines-{deal_id}",
            daemon=True,
        )
        t.start()
        logger.info(
            "%s: started notices + previous_10k + email thread for deal_id=%s",
            LOG_PREFIX,
            deal_id,
        )
        print(
            f"{LOG_PREFIX}: started notices + previous_10k + email thread "
            f"for deal_id={deal_id}"
        )
    except Exception as e:
        # Must never break main EX-2.1 flow
        logger.error("%s: failed to start side pipelines: %s", LOG_PREFIX, e)
        print(f"{LOG_PREFIX}: failed to start side pipelines: {e}")
