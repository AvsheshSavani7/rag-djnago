#!/usr/bin/env python3
"""
Create a new deal in real time (Django/MongoDB).
- Parse ex-2.1 via parse_ex21, upload only parsed JSON to S3 (no PDF/HTML upload or download dir).
- Upsert deal via ProcessingJob (MongoEngine) in the same process; no external API.
"""

import json
import os
import re
import argparse
import tempfile
from datetime import datetime
from typing import Optional

# Load parse_ex21 from extraction_2-1 (filename has hyphen, so use importlib)
import importlib.util
_extraction_21_path = os.path.join(
    os.path.dirname(__file__), "extraction_2-1.py")
_spec = importlib.util.spec_from_file_location(
    "extraction_21", _extraction_21_path)
_extraction_21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_extraction_21)
parse_ex21 = _extraction_21.parse_ex21


def sanitize_filename(name: str) -> str:
    """
    Sanitize a filename by replacing invalid filesystem characters with underscore.
    Same logic as documentProcessor.js sanitizeFilename.
    """
    return re.sub(r'[\\/:"*?<>|]+', "_", name or "")


def format_company_name_for_s3(company_name: str) -> str:
    """Format company name for S3 key: non-alphanumeric -> underscore, lowercase."""
    return re.sub(r"[^a-zA-Z0-9]", "_", (company_name or "").strip()).lower()


def upload_json_to_s3(
    file_path: str,
    company_name: str,
    date_str: str,
    bucket: Optional[str] = None,
    region: Optional[str] = None,
) -> Optional[str]:
    """
    Upload a JSON file to S3 (parsed_jsons/ only). Returns public URL or None if S3 disabled/failed.
    """
    if os.environ.get("ENABLE_S3_UPLOADS", "").lower() == "false":
        return None
    try:
        import boto3
    except ImportError:
        return None
    bucket = bucket or os.environ.get("AWS_S3_BUCKET")
    region = region or os.environ.get("AWS_REGION", "us-east-1")
    if not bucket:
        return None
    key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not key_id or not secret:
        return None

    formatted = format_company_name_for_s3(company_name)
    date_part = date_str or datetime.utcnow().strftime("%Y-%m-%d")
    s3_key = f"parsed_jsons/{formatted}_{date_part}.json"

    s3 = boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
    )
    with open(file_path, "rb") as f:
        s3.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=f.read(),
            ContentType="application/json",
        )
    return f"https://{bucket}.s3.{region}.amazonaws.com/{s3_key}"


def _is_ex21_url(url: str) -> bool:
    """URL (lower) contains ex-2.1, ex21, ex0201, ex2-1, e21, ex2_1, exh21, ex2d1."""
    lower = (url or "").lower()
    return (
        "ex-2.1" in lower
        or "ex21" in lower
        or "ex0201" in lower
        or "ex2-1" in lower
        or "e21" in lower
        or "ex2_1" in lower
        or "exh21" in lower
        or "ex2d1" in lower
    )


def parse_and_upload_json(
    url: str,
    company_name: str,
    filing_date_str: str,
    is_from_ui: Optional[bool] = None,
) -> dict:
    """
    Parse ex-2.1 from URL (parse_ex21) and upload only the parsed JSON to S3.
    No PDF/HTML download or upload; no download dir.
    - When is_from_ui is False: always run parse + upload.
    - When is_from_ui is True: only run if URL looks like ex-2.1.
    Returns: { success, json_url }.
    """
    result = {"success": False, "json_url": ""}
    from_ui = is_from_ui is True or str(is_from_ui).lower() == "true"
    is_ex21_url = _is_ex21_url(url)
    if from_ui and not is_ex21_url:
        result["success"] = True
        return result

    try:
        parsed_data = parse_ex21(url)
        if not parsed_data:
            result["success"] = True
            return result
        articles = (
            getattr(parsed_data, "articles", None)
            or (parsed_data.get("articles") if isinstance(parsed_data, dict) else None)
        )
        if articles is None:
            result["success"] = True
            return result

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(articles, f, indent=2)
            tmp_path = f.name
        try:
            result["json_url"] = (
                upload_json_to_s3(tmp_path, company_name,
                                  filing_date_str) or ""
            )
            result["success"] = True
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return result
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["errorType"] = type(e).__name__
        result["url"] = url
        return result


def _normalize_date(announce_data: str) -> str:
    """Return YYYY-MM-DD from announce_data."""
    try:
        dt = datetime.fromisoformat(announce_data.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(announce_data.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return announce_data.strip()[:10]


def _announce_date_to_datetime(date_str: str):
    """Convert YYYY-MM-DD to datetime for ProcessingJob.announce_date."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str.strip()[:10], "%Y-%m-%d")
    except ValueError:
        return None


def create_new_deal(
    url: str,
    announce_data: str,
    target_name: str,
    target_cik: Optional[str] = None,
    acquired_name: Optional[str] = None,
    sec_filing_id: Optional[str] = None,
    acquirer_cik: Optional[str] = None,
    is_from_ui: Optional[bool] = None,
) -> dict:
    """
    Create or update a deal in real time (Django): parse ex-2.1, upload only JSON to S3,
    then upsert into MongoDB via ProcessingJob. No PDF/HTML upload, no download dir.
    """
    from .models import ProcessingJob

    date_str = _normalize_date(announce_data)
    parse_result = parse_and_upload_json(
        url, target_name, date_str, is_from_ui
    )
    if not parse_result.get("success"):
        return {
            "message": "Parse/upload failed",
            "error": parse_result.get("error", "Unknown error"),
            "status": False,
        }

    parsed_json_url = parse_result.get("json_url", "") or ""
    file_url = None
    pdf_url = None

    announce_dt = _announce_date_to_datetime(date_str)
    cik = (target_cik or "").strip() or None
    if cik:
        cik = "".join(filter(str.isdigit, cik)).zfill(10) or None
    acq_cik = (acquirer_cik or "").strip() or None
    if acq_cik:
        acq_cik = "".join(filter(str.isdigit, acq_cik)).zfill(10) or None

    existing = ProcessingJob.objects(sec_url=url).first() if url else None
    if existing:
        existing.pdf_url = pdf_url
        existing.file_url = file_url
        existing.parsed_json_url = parsed_json_url or None
        existing.target_name = target_name or existing.target_name
        existing.acquire_name = acquired_name or existing.acquire_name
        existing.cik = cik or None
        existing.acquirer_cik = acq_cik or None
        existing.sec_filing_id = sec_filing_id or None
        existing.announce_date = announce_dt or existing.announce_date
        existing.embedding_status = "PENDING"
        existing.updatedAt = datetime.utcnow()
        existing.save()
        job = existing
    else:
        job = ProcessingJob(
            cik=cik or None,
            acquirer_cik=acq_cik or None,
            acquire_name=acquired_name or None,
            target_name=target_name,
            announce_date=announce_dt,
            embedding_status="PENDING",
            summary_status="PENDING",
            file_url=file_url,
            pdf_url=pdf_url,
            parsed_json_url=parsed_json_url or None,
            sec_url=url or None,
            sec_filing_id=sec_filing_id or None,
        )
        job.save()

    return {
        "status": True,
        "deal_id": str(job.id),
        "data": {
            "deal_id": str(job.id),
            "jsonUrl": parsed_json_url or None,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create/upsert a deal in real time (download, parse ex-2.1, S3, MongoDB)."
    )
    parser.add_argument("--url", required=True, help="SEC filing HTML URL")
    parser.add_argument("--announce-data", required=True,
                        help="Announcement date (e.g. YYYY-MM-DD)")
    parser.add_argument("--target-name", required=True,
                        help="Target company name")
    parser.add_argument("--target-cik", help="Target company CIK")
    parser.add_argument("--acquired-name", help="Acquired company name")
    parser.add_argument("--sec-filing-id", help="SEC filing ID")
    parser.add_argument("--acquirer-cik", help="Acquirer CIK")
    parser.add_argument("--is-from-ui", action="store_true",
                        help="Request from UI")
    args = parser.parse_args()

    try:
        result = create_new_deal(
            url=args.url,
            announce_data=args.announce_data,
            target_name=args.target_name,
            target_cik=args.target_cik or None,
            acquired_name=args.acquired_name or None,
            sec_filing_id=args.sec_filing_id or None,
            acquirer_cik=args.acquirer_cik or None,
            is_from_ui=args.is_from_ui if args.is_from_ui else None,
        )
        print(result)
        return 0 if result.get("status") else 1
    except Exception as e:
        print(f"Deal creation failed: {e}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    exit(main())
