"""Main pipeline orchestrator: run_pipeline() — MongoDB + S3, importable entry point."""

from sec_rss_parser.utils_10k_10q import N8N_WEBHOOK_URL_10K_10Q
from sec_rss_parser.utils_8k import send_webhook_notification
from sec_rss_parser.email_templates import generate_10k_10q_comparison_summary_email_html
from .s3_utils import upload_file, upload_json
from .summary_db import SummaryDB
from .sec_fetcher import detect_filing_metadata, fetch_sec_filing, make_filing_label
from .scorer import score_all_paragraphs
from .models import DealContext
from .html_parser import parse_html_to_paragraphs
from .excerpts import generate_excerpts_json, load_excerpts, load_excerpts_from_url
from .docx_builder import (
    generate_client_report,
    generate_exec_summary_report,
    generate_full_comparison_json,
    generate_redline_report,
    generate_single_filing_report_fulsome,
)
from .deal_context import fetch_deal_context
from .config import BATCH_SIZE
from .comparator import merge_results, run_recency_prioritized_comparison
from .assessor import assess_with_claude
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# Email for final comparison summary (JSON + DOCX URLs)


def _get_ticker_for_deal(deal_id: str) -> str:
    """Resolve ticker from deal_id via ProcessingJob.target_ticker."""
    from document_processor.models import ProcessingJob
    job = ProcessingJob.objects(id=deal_id).first()
    if not job:
        raise ValueError(f"No ProcessingJob found for deal_id={deal_id}")
    ticker = getattr(job, "target_ticker", None) or getattr(
        job, "target_name", None)
    if not ticker:
        raise ValueError(
            f"ProcessingJob {deal_id} has no target_ticker or target_name")
    return str(ticker).strip().upper() if len(str(ticker)) <= 10 else str(ticker).strip()


def run_pipeline(
    urls: List[str],
    deal_id: str,
    output_dir: Optional[Path] = None,
    env_path: Optional[Path] = None,
    threshold: int = 6,
    batch_size: int = BATCH_SIZE,
    skip_assessment: bool = False,
) -> dict:
    """
    MongoDB + S3 pipeline. Only urls and deal_id are required.

    1. Resolve ticker from deal_id (ProcessingJob.target_ticker).
    2. Upsert all input URLs into SECFilingSummary (SummaryDB).
    3. Detect metadata for records missing it.
    4. Fetch deal context (cached by deal_id or Perplexity).
    5. Process unprocessed filings: fetch → parse → score → assess → save excerpts + DOCX to temp → upload to S3 (10K_10Q/) → update ten_k_ten_q.
    6. Run comparison: load excerpts from S3 URLs → generate reports to temp → upload to S3 → update newest record's ten_k_ten_q.

    Returns:
        {
            "processed": [urls newly processed],
            "skipped": [urls already processed],
            "comparison_outputs": { "redline": s3_url|None, "client_report": ..., "comparison_json": ..., "exec_summary": ... }
        }
    """
    if env_path and env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            pass

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    perplexity_key = os.environ.get("PERPLEXITY_API_KEY", "")

    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY is required")

    logger.info("10-K/10-Q pipeline: resolving ticker for deal_id=%s", deal_id)
    ticker = _get_ticker_for_deal(deal_id)
    logger.info(
        "10-K/10-Q pipeline: ticker=%s, starting DB upsert for %s URL(s)", ticker, len(urls))

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SEC PIPELINE: {ticker} | deal_id={deal_id}")
    print(f"{'='*60}")
    print(f"  Input URLs: {len(urls)}")

    db = SummaryDB()

    for url in urls:
        db.upsert_by_url(url, {"deal_id": deal_id})

    for url in urls:
        record = db.get_by_url(url)
        if record and not record.get("period_date"):
            period_date, filing_type = detect_filing_metadata(url)
            label = make_filing_label(period_date, filing_type)
            db.update(record["_id"], {
                "period_date": period_date,
                "filing_type": filing_type,
                "label": label,
            })

    deal_context_path = output_dir / f"{deal_id}_deal_context.json"
    if deal_context_path.exists():
        logger.info("10-K/10-Q pipeline: loading deal context from cache")
        deal = DealContext.load(deal_context_path)
        print(
            f"\n[DEAL CONTEXT] Loaded from cache: {deal.target_company} / {deal.acquirer_company}")
    else:
        if not perplexity_key:
            raise ValueError(
                "PERPLEXITY_API_KEY is required to fetch deal context")
        logger.info("10-K/10-Q pipeline: fetching deal context from Perplexity")
        print("\n[DEAL CONTEXT] Fetching from Perplexity...")
        deal = fetch_deal_context(ticker, perplexity_key)
        deal.save(deal_context_path)
        print(f"  {deal.target_company} / {deal.acquirer_company}")
    logger.info(
        "10-K/10-Q pipeline: deal context ready, starting processing loop")

    processed_urls: List[str] = []
    skipped_urls: List[str] = []

    print(f"\n[PROCESSING]")
    for url in urls:
        record = db.get_by_url(url)
        if not record:
            continue

        if record.get("processed"):
            print(f"  SKIP (already processed): {url}")
            skipped_urls.append(url)
            continue

        print(f"\n  --- Processing: {url} ---")
        logger.info("10-K/10-Q pipeline: fetching SEC filing: %s", url)

        html = fetch_sec_filing(url)
        period_date, filing_type = detect_filing_metadata(url, html)
        label = make_filing_label(period_date, filing_type)
        print(f"  Detected: {label} (period: {period_date})")

        paragraphs = parse_html_to_paragraphs(html)
        print(f"  Parsed: {len(paragraphs)} paragraphs")

        logger.info(
            "10-K/10-Q pipeline: scoring %s paragraphs (Anthropic)", len(paragraphs))
        paragraphs = score_all_paragraphs(
            paragraphs, deal, anthropic_key, batch_size)

        if not skip_assessment:
            logger.info("10-K/10-Q pipeline: running assessment (Claude)")
            paragraphs = assess_with_claude(
                paragraphs, deal, anthropic_key, threshold)

        safe_ticker = ticker.replace("/", "_")
        safe_filing = filing_type.replace("-", "").replace("/", "_")
        basename = f"{safe_ticker}_{period_date.replace('-', '')}_{safe_filing}"
        excerpts_path = output_dir / f"{basename}_excerpts.json"
        generate_excerpts_json(
            paragraphs, deal, threshold, excerpts_path,
            source_url=url, period_date=period_date, filing_type=filing_type,
        )

        excerpts_data, _ = load_excerpts(excerpts_path, threshold)
        fulsome_path = output_dir / f"{basename}_fulsome_report.docx"
        generate_single_filing_report_fulsome(
            excerpts_data, deal, label, fulsome_path)

        accession = record.get("accession_number") or basename
        excerpts_dict = json.loads(excerpts_path.read_text(encoding="utf-8"))
        s3_json_url = upload_json(excerpts_dict, f"{accession}/excerpts.json")
        s3_docx_url = upload_file(
            fulsome_path,
            f"{accession}/fulsome_report.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

        db.update(record["_id"], {
            "period_date": period_date,
            "filing_type": filing_type,
            "label": label,
            "processed": True,
            "processed_at": datetime.now(tz=timezone.utc).isoformat(),
            "s3_json_url": s3_json_url,
            "s3_docx_url": s3_docx_url,
        })

        processed_urls.append(url)
        print(f"  Done: {label}")
        logger.info("10-K/10-Q pipeline: completed one filing: %s", label)

    logger.info(
        "10-K/10-Q pipeline: processing loop done, starting comparison step")
    print(f"\n[COMPARISON]")
    cik_number = None
    for url in urls:
        rec = db.get_by_url(url)
        if rec and rec.get("cik_number"):
            cik_number = rec["cik_number"]
            break
    if not cik_number:
        raise ValueError(
            f"Could not determine CIK number from input URLs for deal_id={deal_id}")
    all_records = db.get_by_deal_id_and_cik(deal_id, cik_number)
    processed_records = [r for r in all_records if r.get(
        "processed") and r.get("s3_json_url")]

    comparison_outputs: dict = {
        "redline": None,
        "client_report": None,
        "comparison_json": None,
        "exec_summary": None,
    }

    # Only run comparison and send email when we have at least 2 processed filings.
    if len(processed_records) < 2:
        print(
            f"  Only {len(processed_records)} processed filing(s) — skipping comparison (no comparison summary, no email)")
    else:
        def _sort_key(r):
            pd = r.get("period_date") or ""
            if not pd or pd == "unknown":
                pd = "0000-00-00"
            # Amendments (e.g. 10-K/A) are filed after their base (10-K)
            # and must sort later when period_date ties.
            is_amendment = 1 if "/A" in (r.get("filing_type") or "") else 0
            return (pd, is_amendment)

        sorted_records = sorted(processed_records, key=_sort_key)
        newest_record = sorted_records[-1]
        prior_records = sorted_records[:-1]

        print(f"  Newest: {newest_record.get('label')}")
        print(f"  Priors: {[r.get('label') for r in prior_records]}")

        newest_excerpts, newest_meta = load_excerpts_from_url(
            newest_record["s3_json_url"], threshold, source_label=newest_record.get(
                "label")
        )

        prior_filing_groups = []
        for rec in prior_records:
            p_excerpts, _ = load_excerpts_from_url(
                rec["s3_json_url"], threshold, source_label=rec.get("label")
            )
            tagged = []
            for p in p_excerpts:
                p_copy = dict(p)
                p_copy["_filing_source"] = rec.get("label", "prior")
                tagged.append(p_copy)
            prior_filing_groups.append((rec.get("label", "prior"), tagged))

        deal_meta = {
            "ticker": deal.ticker,
            "target": deal.target_company,
            "acquirer": deal.acquirer_company,
        }

        pass_results = run_recency_prioritized_comparison(
            newest_excerpts, prior_filing_groups, deal_meta, anthropic_key
        )
        merged = merge_results(pass_results, newest_excerpts)

        all_comparison_steps = [{
            "current_label": newest_record.get("label", "current"),
            "prior_labels": [r.get("label", "") for r in prior_records],
            "merged_results": merged,
            "pass_results": pass_results,
            "prior_filing_groups": prior_filing_groups,
        }]

        filing_labels = [r.get("label", "") for r in sorted_records]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{deal_id}_{timestamp}"

        client_path = output_dir / f"{base}_client_report.docx"
        generate_client_report(all_comparison_steps, deal,
                               filing_labels, client_path)
        s3_client = upload_file(
            client_path,
            f"comparison/{base}_client_report.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

        redline_path = output_dir / f"{base}_redline.docx"
        generate_redline_report(all_comparison_steps,
                                deal, filing_labels, redline_path)
        s3_redline = upload_file(
            redline_path,
            f"comparison/{base}_redline.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

        json_path = output_dir / f"{base}_comparison.json"
        generate_full_comparison_json(
            all_comparison_steps, deal, filing_labels, json_path)
        s3_comparison = upload_json(
            json.loads(json_path.read_text(encoding="utf-8")),
            f"comparison/{base}_comparison.json",
        )

        exec_path = output_dir / f"{base}_exec_summary.docx"
        exec_bullets = generate_exec_summary_report(
            all_comparison_steps, deal, filing_labels, exec_path, anthropic_key
        )
        s3_exec = upload_file(
            exec_path,
            f"comparison/{base}_exec_summary.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

        comparison_outputs = {
            "redline": s3_redline,
            "client_report": s3_client,
            "comparison_json": s3_comparison,
            "exec_summary": s3_exec,
        }

        db.update(newest_record["_id"], {
            "s3_comparison_json_url": s3_comparison,
            "s3_redline_docx_url": s3_redline,
            "s3_client_report_docx_url": s3_client,
            "s3_exec_summary_docx_url": s3_exec,
        })

        sig = sum(1 for r in merged if r["overall_severity"] == "significant")
        print(f"\n  Comparison: {len(merged)} changes ({sig} significant)")

        # Send email for final summary with JSON and DOCX URLs (same pattern as utils_10k_10q)
        try:
            subject, html = generate_10k_10q_comparison_summary_email_html(
                ticker=deal.ticker,
                target_company=deal.target_company,
                filing_labels=filing_labels,
                s3_comparison_json_url=s3_comparison,
                s3_exec_summary_docx_url=s3_exec,
                s3_redline_docx_url=s3_redline,
                s3_client_report_docx_url=s3_client,
                exec_summary_bullets=exec_bullets,
            )
            payload = {
                "subject": subject,
                "html": html,
                "company_name": deal.target_company or deal.ticker,
                "email_type": "10k_10q_comparison_summary",
                "s3_comparison_json_url": s3_comparison,
                "s3_exec_summary_docx_url": s3_exec,
                "s3_redline_docx_url": s3_redline,
                "s3_client_report_docx_url": s3_client,
            }
            send_webhook_notification(
                N8N_WEBHOOK_URL_10K_10Q, payload, "10-K/10-Q comparison summary email")
            print(f"  Email sent: final summary with JSON and DOCX links")
        except Exception as email_e:
            print(
                f"  Warning: failed to send comparison summary email: {email_e}")

    logger.info("10-K/10-Q pipeline: DONE | processed=%s | skipped=%s",
                len(processed_urls), len(skipped_urls))
    print(f"\n{'='*60}")
    print(
        f"DONE | processed={len(processed_urls)} | skipped={len(skipped_urls)}")
    print(f"{'='*60}\n")

    return {
        "processed": processed_urls,
        "skipped": skipped_urls,
        "comparison_outputs": comparison_outputs,
    }
