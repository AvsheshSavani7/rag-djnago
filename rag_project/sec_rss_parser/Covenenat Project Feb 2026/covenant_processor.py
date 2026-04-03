"""
Covenant Processor
Integrates covenant analysis pipeline with the backend.

Looks up the covenant_analysis MongoDB record and regenerates
the dashboard from S3-stored stage outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_COVENANT_DIR = Path(__file__).resolve().parent
_EMBEDDINGS_DIR = _COVENANT_DIR / "Covenant_Embeddings_v1"

_RAG_PROJECT_DIR = _COVENANT_DIR.parent.parent
_PROJECT_ENV_PATH = _RAG_PROJECT_DIR / ".env"


def get_covenant_dashboard_url(deal_id: str) -> Optional[str]:
    """Return the S3 dashboard URL from the MongoDB record, or None."""
    try:
        from sec_rss_parser.models import CovenantAnalysis
        record = CovenantAnalysis.objects(deal_id=deal_id).first()
        if record and record.dashboard_html:
            return record.dashboard_html
    except Exception:
        pass
    return None


def build_covenant_dashboard_s3(deal_id: str, deal_name: str = "") -> Optional[str]:
    """
    Look up the covenant_analysis MongoDB record for deal_id,
    download all JSONs from S3, generate the dashboard, upload to S3.
    Returns the S3 dashboard URL, or None if no record found.
    """
    if _PROJECT_ENV_PATH.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_PROJECT_ENV_PATH)
        except ImportError:
            pass

    try:
        from sec_rss_parser.models import CovenantAnalysis
    except Exception:
        return None

    record = CovenantAnalysis.objects(deal_id=deal_id).first()
    if not record:
        return None

    if not record.classification_json or not record.assessment_json:
        return None

    embeddings_dir = str(_EMBEDDINGS_DIR)
    if embeddings_dir not in sys.path:
        sys.path.insert(0, embeddings_dir)

    from importlib import import_module
    stage10 = import_module("10_generate_dashboard")

    result = stage10.run_stage10(
        accession=record.accession_number,
        classification_url=record.classification_json,
        assessment_url=record.assessment_json,
        comparison_url=record.benchmark_comparison_json,
        provisions_url=record.specific_provisions_json,
        deal_name=deal_name,
    )

    dashboard_url = result.get("dashboard_html")
    if dashboard_url:
        record.update(set__dashboard_html=dashboard_url)

    return dashboard_url
