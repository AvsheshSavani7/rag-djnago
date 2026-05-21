"""
Termination Pipeline Runner
Orchestrates: URL -> termination_scraping.py -> Stages 6/7/9/10 -> dashboard HTML

S3 storage, MongoDB tracking. Called from services.py or management command.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

COVENANT_PROJECT_DIR = Path(__file__).resolve().parent
TERMINATION_EMBEDDINGS_DIR = COVENANT_PROJECT_DIR / "Termination_Embeddings_v1"

_RAG_PROJECT_DIR = COVENANT_PROJECT_DIR.parent.parent
_PROJECT_ENV_PATH = _RAG_PROJECT_DIR / ".env"


# =========================================================================
# PIPELINE
# =========================================================================

def run_termination_pipeline_s3(url: str, accession_number: str, doc_type: str,
                                deal_id: str = "", deal_name: str = "",
                                send_email: bool = True):
    """
    Run the full termination pipeline: scraping -> classify -> assess ->
    provision checks -> dashboard. All outputs go to S3; progress tracked
    in the termination_analysis MongoDB collection.

    Args:
        url:              SEC EDGAR URL (required)
        accession_number: SEC accession / document identifier (required)
        doc_type:         "8-K", "99.1", or "2.1" (required)
        deal_id:          from deals collection (optional)
        deal_name:        human-readable name for dashboard header (optional)
    """
    if _PROJECT_ENV_PATH.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_PROJECT_ENV_PATH)
        except ImportError:
            pass

    embeddings_dir = str(TERMINATION_EMBEDDINGS_DIR)
    if embeddings_dir not in sys.path:
        sys.path.insert(0, embeddings_dir)

    try:
        from sec_rss_parser.models import TerminationAnalysis

        # ── Create / find MongoDB record ──
        record = TerminationAnalysis.save_or_update(
            accession_number=accession_number,
            doc_type=doc_type,
            sec_url=url,
            deal_id=deal_id or None,
        )
        print(f"[Pipeline] MongoDB record: {record.id}")

        # ── Step 1: Scraping ──
        print("[Pipeline] Step 1/5: Scraping...")
        cwd_backup = os.getcwd()
        os.chdir(str(COVENANT_PROJECT_DIR))
        try:
            from termination_scraping import worker as scraping_worker
            scrape_result = scraping_worker(
                url,
                pipeline_deal_id=deal_id or None,
                pipeline_accession=accession_number,
                pipeline_doc_type=doc_type,
            )
        finally:
            os.chdir(cwd_backup)

        if scrape_result.get("status") != "success":
            error_msg = scrape_result.get("reason", "Unknown scraping error")
            print(f"[Pipeline] ERROR in scraping: {error_msg}")
            return

        s3_urls = scrape_result.get("s3_urls", {})

        record = TerminationAnalysis.save_or_update(
            accession_number=accession_number,
            doc_type=doc_type,
            **s3_urls,
        )

        triggers_url = s3_urls.get(
            "triggers_json") or s3_urls.get("triggers_8k_json")
        fees_url = s3_urls.get("fees_json") or s3_urls.get("fees_8k_json")

        if not triggers_url:
            print("[Pipeline] ERROR: No trigger output produced by scraping")
            return

        # ── Step 2: Stage 6 — Classify ──
        print("[Pipeline] Step 2/5: Stage 6 — Classify...")
        from importlib import import_module
        stage6 = import_module("6_classify_new_deal_termination")
        stage6_result = stage6.run_stage6(
            triggers_url, accession_number, doc_type)

        classification_url = stage6_result["classification_json"]
        summary_csv_url = stage6_result["summary_csv"]

        record = TerminationAnalysis.save_or_update(
            accession_number=accession_number,
            doc_type=doc_type,
            classification_json=classification_url,
            summary_csv=summary_csv_url,
        )

        # ── Step 3: Stage 7 — Assess ──
        print("[Pipeline] Step 3/5: Stage 7 — Assess...")
        stage7 = import_module("7_assess_new_deal_termination")
        stage7_result = stage7.run_stage7(
            classification_url, accession_number, doc_type)

        assessment_url = stage7_result["assessment_json"]

        record = TerminationAnalysis.save_or_update(
            accession_number=accession_number,
            doc_type=doc_type,
            assessment_json=assessment_url,
        )

        # ── Step 4: Stage 9 — Provision Checks ──
        provision_checks_url = None
        if fees_url:
            print("[Pipeline] Step 4/5: Stage 9 — Provision Checks...")
            stage9 = import_module("9_specific_provision_checks_termination")
            stage9_result = stage9.run_stage9(
                fees_url, triggers_url,
                accession_number, doc_type)

            provision_checks_url = stage9_result["provision_checks_json"]

            record = TerminationAnalysis.save_or_update(
                accession_number=accession_number,
                doc_type=doc_type,
                provision_checks_json=provision_checks_url,
            )
        else:
            print("[Pipeline] Step 4/5: Stage 9 — Skipped (no fees data)")

        # ── Step 5: Stage 10 — Dashboard ──
        print("[Pipeline] Step 5/5: Stage 10 — Dashboard...")
        stage10 = import_module("10_generate_dashboard_termination")

        stage10_result = stage10.run_stage10(
            accession=accession_number,
            doc_type=doc_type,
            classification_url=classification_url,
            assessment_url=assessment_url,
            provision_checks_url=provision_checks_url,
            fees_url=s3_urls.get("fees_json"),
            fees_8k_url=s3_urls.get("fees_8k_json"),
            triggers_url=s3_urls.get("triggers_json"),
            triggers_8k_url=s3_urls.get("triggers_8k_json"),
            deal_name=deal_name,
        )

        record = TerminationAnalysis.save_or_update(
            accession_number=accession_number,
            doc_type=doc_type,
            dashboard_html=stage10_result["dashboard_html"],
        )

        print(
            f"\n[Pipeline] Complete! Dashboard: {stage10_result['dashboard_html']}")

        if send_email:
            _send_dashboard_email(
                dashboard_url=stage10_result["dashboard_html"],
                deal_id=deal_id,
                deal_name=deal_name,
                accession_number=accession_number,
                doc_type=doc_type,
                sec_url=url,
            )

    except Exception as e:
        print(f"[Pipeline] ERROR: {e}")
        import traceback
        traceback.print_exc()


def start_termination_pipeline_s3(url: str, accession_number: str, doc_type: str,
                                  deal_id: str = "", deal_name: str = "") -> dict:
    """Launch the termination pipeline in a background thread."""
    try:
        from sec_rss_parser.models import TerminationAnalysis
        existing = TerminationAnalysis.objects(
            accession_number=accession_number, doc_type=doc_type
        ).first()
        if existing and existing.dashboard_html:
            return {"status": "already_complete", "dashboard_html": existing.dashboard_html}
    except Exception:
        pass

    thread = threading.Thread(
        target=run_termination_pipeline_s3,
        args=(url, accession_number, doc_type, deal_id, deal_name),
        daemon=True,
    )
    thread.start()
    return {"status": "started"}


# =========================================================================
# EMAIL NOTIFICATION
# =========================================================================

def _send_dashboard_email(dashboard_url: str, deal_id: str, deal_name: str,
                          accession_number: str, doc_type: str, sec_url: str):
    """Send termination dashboard HTML link via n8n webhook email."""
    try:
        import requests

        N8N_WEBHOOK_URL = "https://n8n.arbintel.cloud/webhook/80830c6d-ff5b-45e3-9ef3-a061db1fbf0c"

        title = deal_name or deal_id or accession_number
        subject = f"Termination Analysis — {title}"

        html_body = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px;">
            <h2 style="color: #1a1a2e;">Termination Analysis Complete</h2>
            <table style="border-collapse: collapse; width: 100%; margin: 16px 0;">
                <tr><td style="padding: 6px 12px; color: #666;">Deal</td>
                    <td style="padding: 6px 12px; font-weight: 600;">{title}</td></tr>
                <tr><td style="padding: 6px 12px; color: #666;">Accession</td>
                    <td style="padding: 6px 12px;">{accession_number}</td></tr>
                <tr><td style="padding: 6px 12px; color: #666;">Doc Type</td>
                    <td style="padding: 6px 12px;">{doc_type}</td></tr>
                <tr><td style="padding: 6px 12px; color: #666;">SEC Filing</td>
                    <td style="padding: 6px 12px;"><a href="{sec_url}">View on SEC EDGAR</a></td></tr>
            </table>
            <a href="{dashboard_url}"
               style="display: inline-block; padding: 12px 24px; background: #1a1a2e;
                      color: #5ccfe6; text-decoration: none; border-radius: 4px;
                      font-weight: 600; margin: 8px 0;">
                Open Termination Dashboard
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
            "form_type": f"EX-{doc_type}",
            "dashboard_url": dashboard_url,
        }

        print(f"[Pipeline] Sending dashboard email via n8n webhook...")
        response = requests.post(
            N8N_WEBHOOK_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        print(
            f"[Pipeline] Email sent successfully (status {response.status_code})")

    except Exception as e:
        print(f"[Pipeline] Warning: failed to send dashboard email: {e}")
