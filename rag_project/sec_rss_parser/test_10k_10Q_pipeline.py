#!/usr/bin/env python3
"""
Test script for the 10-K/10-Q pipeline (MongoDB + S3).

Entry: run_pipeline(urls, deal_id). Ticker is resolved from deal_id (ProcessingJob.target_ticker).
Run from rag_project with Django configured so SECFilingSummary and ProcessingJob are available.
"""
import os
import sys
from pathlib import Path

# Ensure we can import sec_rss_parser and Django is set up when run as script
if __name__ == "__main__":
    _rag_root = Path(__file__).resolve().parent.parent.parent
    if str(_rag_root) not in sys.path:
        sys.path.insert(0, str(_rag_root))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rag_project.settings")
    import django
    django.setup()

from sec_rss_parser.models import SECFilingSummary
from sec_rss_parser.tenK_tenQ_pipeline.orchestrator import run_pipeline

# ── Configuration ──
DEAL_ID = "68f1d30397173821e21c541c"
URLS = [
    "https://www.sec.gov/Archives/edgar/data/794619/000079461925000107/amwd-20250731.htm",
    "https://www.sec.gov/Archives/edgar/data/794619/000079461925000115/amwd-20251031.htm",
    "https://www.sec.gov/Archives/edgar/data/794619/000079461926000005/amwd-20260131.htm"
]
ENV_PATH = Path(".env")

if __name__ == "__main__":
    result = run_pipeline(
        urls=URLS,
        deal_id=DEAL_ID,
        env_path=ENV_PATH if ENV_PATH.exists() else None,
    )

    print("\n=== PIPELINE RESULT ===")
    print(f"Processed:  {result['processed']}")
    print(f"Skipped:    {result['skipped']}")
    print("Comparison outputs (S3 URLs):")
    for k, v in result["comparison_outputs"].items():
        print(f"  {k}: {v}")

    # ── Verify MongoDB ──
    records = list(
        SECFilingSummary.objects(
            deal_id=DEAL_ID,
            form_type__in=["10-K", "10-Q"],
        )
    )
    print(f"\n=== SECFilingSummary records ({len(records)} total) ===")
    for r in records:
        tq = r.ten_k_ten_q or {}
        print(
            f"  [{tq.get('label', '?')}] processed={tq.get('processed')} "
            f"s3_json={tq.get('s3_json_url') or 'none'}"
        )
