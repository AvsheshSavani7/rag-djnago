"""
Email Dispatch Service
======================
Organisation-aware email sending via the n8n webhook.

Flow
----
send_report_email(report_type, payload, org_id=None)
  1. Find active organization(s).
  2. Check organization_notification_settings — keep only those where
     enabled_report_types contains the given report_type.
  3. Find active organization_email_recipients for each qualifying org
     — keep only those whose report_types list contains the report_type.
  4. POST one webhook request per org (with its recipients list).

MongoDB collections (default DB)
---------------------------------
- organizations             : _id (ObjectId), status ("active"|...), name
- organization_notification_settings
                            : organization_id (str), enabled_report_types (list)
- organization_email_recipients
                            : organization_id (str), email, name,
                              is_active (bool), report_types (list, may be absent)
"""

import logging
import os
from pathlib import Path
from typing import Optional

import requests
from bson import ObjectId

from rag_project.db_utils import get_default_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dedicated email audit logger
# Writes to email/daily/{IST-date}/email.log via the existing pipeline handler.
# Uses propagate=False + its own handler so PipelineContextFilter on the root
# logger does not overwrite pipeline="email" with the calling pipeline's name.
# ---------------------------------------------------------------------------

class _ForceEmailPipelineFilter(logging.Filter):
    """Stamps every record with pipeline='email' so it routes to email/ log folder."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.pipeline = "email"
        if not hasattr(record, "run_id"):
            record.run_id = "-"
        if not hasattr(record, "accession"):
            record.accession = "-"
        if not hasattr(record, "doc_type"):
            record.doc_type = "EMAIL"
        return True


def _build_email_audit_logger() -> logging.Logger:
    log = logging.getLogger("email_audit")
    if log.handlers:
        return log
    log.propagate = False
    log.setLevel(logging.INFO)
    log_root = os.environ.get(
        "LOG_ROOT",
        str(Path(__file__).resolve().parents[3] / "logs"),
    )
    try:
        from core.dynamic_pipeline_handler import DynamicPipelineHandler
        handler = DynamicPipelineHandler(log_root=log_root)
        handler.addFilter(_ForceEmailPipelineFilter())
        log.addHandler(handler)
    except Exception:
        # Fallback: plain stderr when running outside Django (tests, scripts)
        log.addHandler(logging.StreamHandler())
    return log


_email_audit_logger = _build_email_audit_logger()

# Static CC addresses appended to every outgoing org email (excluded for our
# internal org to avoid duplicate notifications to the Hyperion team).
CC_EMAILS = [
    "kaushal@hyperiontechnologies.ai",
    "josh@hyperiontechnologies.ai",
]

# org_id that should NOT receive the static CC (internal Hyperion org)
_INTERNAL_ORG_ID = "6a031d87e4f1d72367bd2f92"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_active_orgs(db, org_id: Optional[str] = None) -> list:
    """
    Return active organization documents.

    If org_id is provided, scope to that single org;
    otherwise return all active orgs.
    """
    query = {"status": "active"}
    if org_id:
        try:
            query["_id"] = ObjectId(org_id)
        except Exception:
            logger.error("Invalid org_id format: %s", org_id)
            return []
    return list(db["organizations"].find(query))


def _is_report_type_enabled(db, organization_id: str, report_type: str) -> bool:
    """Return True when the org's notification settings include report_type."""
    settings = db["organization_notification_settings"].find_one(
        {"organization_id": organization_id}
    )
    if not settings:
        return False
    return report_type in settings.get("enabled_report_types", [])


def _get_recipients(db, organization_id: str, report_type: str) -> list:
    """
    Return active recipients for this org that have subscribed to report_type.

    A recipient is included only when:
    - is_active is True
    - report_types field exists and contains report_type
    """
    cursor = db["organization_email_recipients"].find(
        {
            "organization_id": organization_id,
            "is_active": True,
            "report_types": report_type,
        }
    )
    return list(cursor)


def _send_to_webhook(webhook_url: str, payload: dict) -> bool:
    """POST payload to the n8n webhook. Returns True on success."""
    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        logger.info(
            "Email dispatched via webhook [%s] — status %s",
            webhook_url,
            response.status_code,
        )
        return True
    except requests.exceptions.RequestException as exc:
        logger.error("Webhook POST failed: %s", exc)
        if hasattr(exc, "response") and exc.response is not None:
            logger.error("Response body: %s", exc.response.text[:300])
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_direct_email(
    recipients: list,
    payload: dict,
    webhook_url: Optional[str] = None,
) -> bool:
    """
    Send an email directly to a fixed recipient list — no org lookup,
    no report_type check, no MongoDB queries.

    Use this for emails that are not part of the report_type taxonomy:
    internal alerts, pipeline error notifications, admin-only sends, etc.

    Parameters
    ----------
    recipients : list[str]
        Flat list of email addresses, e.g. ["admin@example.com", "josh@example.com"].
    payload : dict
        Must include at minimum:
            "subject" (str)  — email subject line
            "html"    (str)  — HTML body
        Any additional keys are forwarded to the n8n webhook as-is.
    webhook_url : str, optional
        Override the default TESTING_N8N_HOOK.

    Returns
    -------
    bool
        True if the webhook call succeeded, False otherwise.

    Example
    -------
    send_direct_email(
        recipients=["admin@teqnodux.com"],
        payload={
            "subject": "Pipeline Error",
            "html": "<p>Something went wrong in the 10-K pipeline.</p>",
        },
        webhook_url=os.environ.get("N8N_WEBHOOK_INTERNAL"),
    )
    """
    hook = webhook_url or os.environ.get("TESTING_N8N_HOOK", "")
    if not hook:
        raise EnvironmentError(
            "No webhook URL provided and TESTING_N8N_HOOK is not set."
        )

    if not recipients:
        logger.warning(
            "send_direct_email called with empty recipients list — skipping.")
        return False

    webhook_payload = {
        **payload,
        "recipients": recipients,
    }

    logger.info("send_direct_email | recipients=%d | subject=%s",
                len(recipients), payload.get("subject", ""))
    return _send_to_webhook(hook, webhook_payload)


def send_report_email(
    report_type: str,
    payload: dict,
    org_id: Optional[str] = None,
    webhook_url: Optional[str] = None,
) -> dict:
    """
    Send an email for the given report_type to all eligible recipients.

    Parameters
    ----------
    report_type : str
        Key from the REPORT_TYPES taxonomy (e.g. "sec_standard_summary").
    payload : dict
        Must include at minimum:
            "subject" (str)  — email subject line
            "html"    (str)  — HTML body
        Any additional keys are forwarded to the n8n webhook as-is.
    org_id : str, optional
        Target a single organisation. If omitted, all active orgs are processed.
    webhook_url : str, optional
        Override the default TESTING_N8N_HOOK (useful for per-report-type routing
        in future; for now the service always falls back to TESTING_N8N_HOOK).

    Returns
    -------
    dict
        {
          "report_type": str,
          "orgs_processed": int,
          "orgs_sent": int,
          "orgs_skipped_no_setting": int,
          "orgs_skipped_no_recipients": int,
          "results": [
              {"org_id": str, "org_name": str, "recipients": [...], "sent": bool},
              ...
          ]
        }
    """
    hook = webhook_url or os.environ.get("TESTING_N8N_HOOK", "")
    if not hook:
        raise EnvironmentError(
            "TESTING_N8N_HOOK is not set. "
            "Add it to your .env file before calling send_report_email."
        )

    db, client = get_default_db()

    summary = {
        "report_type": report_type,
        "orgs_processed": 0,
        "orgs_sent": 0,
        "orgs_skipped_no_setting": 0,
        "orgs_skipped_no_recipients": 0,
        "results": [],
    }

    try:
        active_orgs = _get_active_orgs(db, org_id)
        logger.info(
            "send_report_email | report_type=%s | active orgs found=%d",
            report_type,
            len(active_orgs),
        )

        for org in active_orgs:
            org_id_str = str(org["_id"])
            org_name = org.get("name", org_id_str)
            summary["orgs_processed"] += 1

            # Step 2 – check notification settings
            if not _is_report_type_enabled(db, org_id_str, report_type):
                logger.info(
                    "Org '%s' does not have '%s' in enabled_report_types — skipping.",
                    org_name,
                    report_type,
                )
                summary["orgs_skipped_no_setting"] += 1
                summary["results"].append(
                    {
                        "org_id": org_id_str,
                        "org_name": org_name,
                        "skipped_reason": "report_type not in enabled_report_types",
                        "sent": False,
                    }
                )
                continue

            # Step 3 – get subscribed recipients
            recipients = _get_recipients(db, org_id_str, report_type)
            if not recipients:
                logger.info(
                    "Org '%s' has no active recipients for '%s' — skipping.",
                    org_name,
                    report_type,
                )
                summary["orgs_skipped_no_recipients"] += 1
                summary["results"].append(
                    {
                        "org_id": org_id_str,
                        "org_name": org_name,
                        "skipped_reason": "no active recipients for this report_type",
                        "sent": False,
                    }
                )
                continue

            # Step 4 – build webhook payload and send
            recipient_list = [r["email"] for r in recipients]
            webhook_payload = {
                **payload,
                "report_type": report_type,
                "org_id": org_id_str,
                "org_name": org_name,
                "recipients": recipient_list,
                "cc": CC_EMAILS if org_id_str != _INTERNAL_ORG_ID else [],
            }

            logger.info(
                "Sending '%s' email to org '%s' (%d recipients).",
                report_type,
                org_name,
                len(recipient_list),
            )
            sent = _send_to_webhook(hook, webhook_payload)

            # Audit log — goes to email/daily/{date}/email.log
            _email_audit_logger.info(
                "report_type=%s | org_id=%s | org_name=%s | "
                "recipients=%s | cc=%s | sent=%s | subject=%s",
                report_type,
                org_id_str,
                org_name,
                recipient_list,
                webhook_payload.get("cc", []),
                sent,
                payload.get("subject", ""),
            )

            summary["results"].append(
                {
                    "org_id": org_id_str,
                    "org_name": org_name,
                    "recipients": recipient_list,
                    "sent": sent,
                }
            )
            if sent:
                summary["orgs_sent"] += 1

    finally:
        client.close()

    logger.info(
        "send_report_email done | report_type=%s | orgs_sent=%d / orgs_processed=%d",
        report_type,
        summary["orgs_sent"],
        summary["orgs_processed"],
    )
    return summary
