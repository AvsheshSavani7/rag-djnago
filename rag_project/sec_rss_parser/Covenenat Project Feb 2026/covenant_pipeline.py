"""
Covenant Pipeline Runner
Orchestrates: URL -> covenant_scraping.py -> Stages 6/7/8/9/10 -> dashboard HTML

S3 storage, MongoDB tracking. Called from services.py or management command.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

COVENANT_PROJECT_DIR = Path(__file__).resolve().parent
COVENANT_EMBEDDINGS_DIR = COVENANT_PROJECT_DIR / "Covenant_Embeddings_v1"

_RAG_PROJECT_DIR = COVENANT_PROJECT_DIR.parent.parent
_PROJECT_ENV_PATH = _RAG_PROJECT_DIR / ".env"


def run_covenant_pipeline_s3(url: str, accession_number: str,
                             deal_id: str = "", deal_name: str = "",
                             send_email: bool = True):
    """
    Run the full covenant pipeline: scraping -> classify -> assess ->
    benchmark comparison -> provision checks -> dashboard.
    All outputs go to S3; progress tracked in the covenant_analysis MongoDB collection.

    Args:
        url:              SEC EDGAR URL (required)
        accession_number: SEC accession / document identifier (required)
        deal_id:          from deals collection (optional)
        deal_name:        human-readable name for dashboard header (optional)
    """
    if _PROJECT_ENV_PATH.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_PROJECT_ENV_PATH)
        except ImportError:
            pass

    embeddings_dir = str(COVENANT_EMBEDDINGS_DIR)
    if embeddings_dir not in sys.path:
        sys.path.insert(0, embeddings_dir)

    try:
        from sec_rss_parser.models import CovenantAnalysis

        record = CovenantAnalysis.save_or_update(
            accession_number=accession_number,
            sec_url=url,
            deal_id=deal_id or None,
        )
        print(f"[Covenant Pipeline] MongoDB record: {record.id}")

        # ── Step 1: Scraping ──
        print("[Covenant Pipeline] Step 1/6: Scraping...")
        cwd_backup = os.getcwd()
        os.chdir(str(COVENANT_PROJECT_DIR))
        try:
            from covenant_scraping import worker as scraping_worker
            scrape_result = scraping_worker(
                url,
                pipeline_deal_id=deal_id or None,
                pipeline_accession=accession_number,
                pipeline_doc_type="2.1",
            )
        finally:
            os.chdir(cwd_backup)

        if scrape_result.get("status") != "success":
            error_msg = scrape_result.get("reason", "Unknown scraping error")
            print(f"[Covenant Pipeline] ERROR in scraping: {error_msg}")
            return

        s3_urls = scrape_result.get("s3_urls", {})

        record = CovenantAnalysis.save_or_update(
            accession_number=accession_number,
            **s3_urls,
        )

        clauses_url = s3_urls.get("individual_clauses_json")
        if not clauses_url:
            print("[Covenant Pipeline] ERROR: No individual_clauses output produced")
            return

        # ── Step 2: Stage 6 — Classify ──
        print("[Covenant Pipeline] Step 2/6: Stage 6 — Classify...")
        from importlib import import_module
        stage6 = import_module("6_classify_new_deal")
        stage6_result = stage6.run_stage6(clauses_url, accession_number)

        classification_url = stage6_result["classification_json"]
        summary_csv_url = stage6_result["summary_csv"]

        record = CovenantAnalysis.save_or_update(
            accession_number=accession_number,
            classification_json=classification_url,
            summary_csv=summary_csv_url,
        )

        # ── Step 3: Stage 7 — Assess ──
        print("[Covenant Pipeline] Step 3/6: Stage 7 — Assess...")
        stage7 = import_module("7_assess_new_deal")
        stage7_result = stage7.run_stage7(classification_url, accession_number)

        assessment_url = stage7_result["assessment_json"]

        record = CovenantAnalysis.save_or_update(
            accession_number=accession_number,
            assessment_json=assessment_url,
        )

        # ── Step 4: Stage 8 — Benchmark Comparison ──
        print("[Covenant Pipeline] Step 4/6: Stage 8 — Benchmark Comparison...")
        benchmark_comparison_url = None
        benchmark_summary_url = None
        try:
            stage8 = import_module("8_compare_to_benchmark")
            stage8_result = stage8.run_stage8(assessment_url, accession_number)

            benchmark_comparison_url = stage8_result.get(
                "benchmark_comparison_json")
            benchmark_summary_url = stage8_result.get("benchmark_summary_csv")

            record = CovenantAnalysis.save_or_update(
                accession_number=accession_number,
                benchmark_comparison_json=benchmark_comparison_url,
                benchmark_summary_csv=benchmark_summary_url,
            )
        except Exception as e:
            print(
                f"[Covenant Pipeline] Warning: Stage 8 failed (non-fatal): {e}")

        # ── Step 5: Stage 9 — Provision Checks ──
        print("[Covenant Pipeline] Step 5/6: Stage 9 — Provision Checks...")
        specific_provisions_url = None
        try:
            stage9 = import_module("9_specific_provision_checks")
            stage9_result = stage9.run_stage9(clauses_url, accession_number)

            specific_provisions_url = stage9_result.get(
                "specific_provisions_json")

            record = CovenantAnalysis.save_or_update(
                accession_number=accession_number,
                specific_provisions_json=specific_provisions_url,
            )
        except Exception as e:
            print(
                f"[Covenant Pipeline] Warning: Stage 9 failed (non-fatal): {e}")

        # ── Step 6: Stage 10 — Dashboard ──
        print("[Covenant Pipeline] Step 6/6: Stage 10 — Dashboard...")
        stage10 = import_module("10_generate_dashboard")
        stage10_result = stage10.run_stage10(
            accession=accession_number,
            classification_url=classification_url,
            assessment_url=assessment_url,
            comparison_url=benchmark_comparison_url,
            provisions_url=specific_provisions_url,
            deal_name=deal_name,
        )

        record = CovenantAnalysis.save_or_update(
            accession_number=accession_number,
            dashboard_html=stage10_result["dashboard_html"],
        )

        print(
            f"\n[Covenant Pipeline] Complete! Dashboard: {stage10_result['dashboard_html']}")

        if send_email:
            _send_dashboard_email(
                dashboard_url=stage10_result["dashboard_html"],
                deal_id=deal_id,
                deal_name=deal_name,
                accession_number=accession_number,
                sec_url=url,
            )

    except Exception as e:
        print(f"[Covenant Pipeline] ERROR: {e}")
        import traceback
        traceback.print_exc()


def start_covenant_pipeline_s3(url: str, accession_number: str,
                               deal_id: str = "", deal_name: str = "") -> dict:
    """Launch the covenant pipeline in a background thread."""
    try:
        from sec_rss_parser.models import CovenantAnalysis
        existing = CovenantAnalysis.objects(
            accession_number=accession_number).first()
        if existing and existing.dashboard_html:
            return {"status": "already_complete", "dashboard_html": existing.dashboard_html}
    except Exception:
        pass

    thread = threading.Thread(
        target=run_covenant_pipeline_s3,
        args=(url, accession_number, deal_id, deal_name),
        daemon=True,
    )
    thread.start()
    return {"status": "started"}


def _send_dashboard_email(dashboard_url: str, deal_id: str, deal_name: str,
                          accession_number: str, sec_url: str):
    """Send covenant dashboard HTML link via n8n webhook email."""
    try:
        import requests
        N8N_WEBHOOK_URL = "https://n8n-xwx1.onrender.com/webhook/80830c6d-ff5b-45e3-9ef3-a061db1fbf0c"

        title = deal_name or deal_id or accession_number
        subject = f"Covenant Analysis — {title}"

        html_body = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px;">
            <h2 style="color: #1a1a2e;">Covenant Analysis Complete</h2>
            <table style="border-collapse: collapse; width: 100%; margin: 16px 0;">
                <tr><td style="padding: 6px 12px; color: #666;">Deal</td>
                    <td style="padding: 6px 12px; font-weight: 600;">{title}</td></tr>
                <tr><td style="padding: 6px 12px; color: #666;">Accession</td>
                    <td style="padding: 6px 12px;">{accession_number}</td></tr>
                <tr><td style="padding: 6px 12px; color: #666;">SEC Filing</td>
                    <td style="padding: 6px 12px;"><a href="{sec_url}">View on SEC EDGAR</a></td></tr>
            </table>
            <a href="{dashboard_url}"
               style="display: inline-block; padding: 12px 24px; background: #1a1a2e;
                      color: #87d96c; text-decoration: none; border-radius: 4px;
                      font-weight: 600; margin: 8px 0;">
                Open Covenant Dashboard
            </a>
            <p style="color: #999; font-size: 12px; margin-top: 20px;">
                Dashboard URL: <a href="{dashboard_url}">{dashboard_url}</a>
            </p>
        </div>
        """

        payload = {
            "subject": subject,
            "html": html_body,
            "company_name": title,
            "form_type": "EX-2.1 (Covenants)",
            "dashboard_url": dashboard_url,
        }

        print(f"[Covenant Pipeline] Sending dashboard email via n8n webhook...")
        response = requests.post(
            N8N_WEBHOOK_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        print(
            f"[Covenant Pipeline] Email sent successfully (status {response.status_code})")

    except Exception as e:
        print(
            f"[Covenant Pipeline] Warning: failed to send dashboard email: {e}")
