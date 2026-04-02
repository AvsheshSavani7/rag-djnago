"""
Termination Pipeline Runner
Orchestrates: URL -> termination_scraping.py -> Stages 6/7/9/10 -> dashboard HTML

S3 storage, MongoDB tracking. Called from services.py or management command.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

# termination_pipeline.py lives in: sec_rss_parser/Covenenat Project Feb 2026/
COVENANT_PROJECT_DIR = Path(__file__).resolve().parent
TERMINATION_EMBEDDINGS_DIR = COVENANT_PROJECT_DIR / "Termination_Embeddings_v1"
TERMINATION_OUTPUT_DIR = COVENANT_PROJECT_DIR / "data" / "termination"
PIPELINE_DIR = TERMINATION_OUTPUT_DIR / "pipeline"

# rag_project/ is two levels up from this file's directory
_RAG_PROJECT_DIR = COVENANT_PROJECT_DIR.parent.parent

# Main project .env — contains AWS, MongoDB, API keys.
_PROJECT_ENV_PATH = _RAG_PROJECT_DIR / ".env"


# =========================================================================
# STATUS HELPERS
# =========================================================================

def _status_path(deal_id: str) -> Path:
    return PIPELINE_DIR / deal_id / "status.json"


def _write_status(deal_id: str, status: str, step: str = "", error: str = ""):
    p = _status_path(deal_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "status": status,
        "step": step,
        "error": error,
        "updated_at": datetime.now().isoformat(),
    }))


def get_pipeline_status(deal_id: str) -> dict:
    p = _status_path(deal_id)
    if p.exists():
        return json.loads(p.read_text())
    return {"status": "idle", "step": "", "error": ""}


# =========================================================================
# PIPELINE — S3 + MongoDB
# =========================================================================

def run_termination_pipeline_s3(deal_id: str, url: str, accession_number: str,
                                doc_type: str, deal_name: str = ""):
    """
    Run the full termination pipeline: scraping -> classify -> assess ->
    provision checks -> dashboard. All outputs go to S3; progress tracked
    in the termination_analysis MongoDB collection.

    Input:
        deal_id          — from deals collection
        url              — SEC EDGAR URL
        accession_number — SEC accession / document identifier
        doc_type         — "8-K", "99.1", or "2.1"
        deal_name        — human-readable deal name for dashboard header (optional)
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
        # ── Create / find MongoDB record ──
        _write_status(deal_id, "running", "init")
        from sec_rss_parser.models import TerminationAnalysis
        record = TerminationAnalysis.save_or_update(
            accession_number=accession_number,
            doc_type=doc_type,
            sec_url=url,
            deal_id=deal_id,
        )
        print(f"[Pipeline] MongoDB record: {record.id}")

        # ── Step 1: Scraping ──
        _write_status(deal_id, "running", "scraping")
        cwd_backup = os.getcwd()
        os.chdir(str(COVENANT_PROJECT_DIR))
        try:
            from termination_scraping import worker as scraping_worker
            scrape_result = scraping_worker(
                url,
                pipeline_deal_id=deal_id,
                pipeline_accession=accession_number,
                pipeline_doc_type=doc_type,
            )
        finally:
            os.chdir(cwd_backup)

        if scrape_result.get("status") != "success":
            error_msg = scrape_result.get("reason", "Unknown scraping error")
            _write_status(deal_id, "error", "scraping", error_msg)
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
            _write_status(deal_id, "error", "scraping",
                          "No trigger output produced by scraping")
            return

        # ── Step 2: Stage 6 — Classify ──
        _write_status(deal_id, "running", "stage_6")
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
        _write_status(deal_id, "running", "stage_7")
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
            _write_status(deal_id, "running", "stage_9")
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

        # ── Step 5: Stage 10 — Dashboard ──
        _write_status(deal_id, "running", "generating_dashboard")
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

        _write_status(deal_id, "complete", "done")
        print(
            f"\n[Pipeline] Complete! Dashboard: {stage10_result['dashboard_html']}")

        _send_dashboard_email(
            dashboard_url=stage10_result["dashboard_html"],
            deal_id=deal_id,
            deal_name=deal_name,
            accession_number=accession_number,
            doc_type=doc_type,
            sec_url=url,
        )

    except Exception as e:
        _write_status(deal_id, "error", "unknown", str(e))
        import traceback
        traceback.print_exc()


def start_termination_pipeline_s3(deal_id: str, url: str,
                                  accession_number: str, doc_type: str,
                                  deal_name: str = "") -> dict:
    """Launch the termination pipeline in a background thread."""
    status = get_pipeline_status(deal_id)
    if status.get("status") == "running":
        return {"status": "already_running", "step": status.get("step", "")}

    thread = threading.Thread(
        target=run_termination_pipeline_s3,
        args=(deal_id, url, accession_number, doc_type, deal_name),
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

        N8N_WEBHOOK_URL = "https://n8n-xwx1.onrender.com/webhook/d50502ea-6746-4d4b-8dfe-fb7bd71e0a1f"

        title = deal_name or deal_id
        subject = f"Termination Analysis — {title}"

        html_body = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px;">
            <h2 style="color: #1a1a2e;">Termination Analysis Complete</h2>
            <table style="border-collapse: collapse; width: 100%; margin: 16px 0;">
                <tr><td style="padding: 6px 12px; color: #666;">Deal</td>
                    <td style="padding: 6px 12px; font-weight: 600;">{title}</td></tr>
                <tr><td style="padding: 6px 12px; color: #666;">Deal ID</td>
                    <td style="padding: 6px 12px;">{deal_id}</td></tr>
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
            "company_name": deal_name or deal_id,
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
