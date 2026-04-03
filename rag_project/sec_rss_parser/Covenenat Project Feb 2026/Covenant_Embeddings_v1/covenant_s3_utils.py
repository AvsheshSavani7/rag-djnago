"""
S3 helpers for the covenant analysis pipeline.

Key pattern:
    covenant_analysis/{accession_number}/{filename}

Uses same AWS credentials as the rest of the project (AWS_S3_BUCKET, etc.).
"""

import json
import os
from pathlib import Path
from typing import Tuple, Optional

S3_PREFIX = "covenant_analysis"

_PROJECT_ENV = Path(__file__).resolve().parent.parent.parent.parent / ".env"
if _PROJECT_ENV.exists() and not os.environ.get("AWS_S3_BUCKET"):
    try:
        from dotenv import load_dotenv
        load_dotenv(_PROJECT_ENV)
    except ImportError:
        pass


def _get_client_and_bucket():
    try:
        import boto3
    except ImportError:
        raise ValueError("boto3 is required for S3. Install with: pip install boto3")
    bucket = os.environ.get("AWS_S3_BUCKET")
    if not bucket:
        raise ValueError("AWS_S3_BUCKET is not set in environment")
    region = os.environ.get("AWS_REGION", "us-east-1")
    client = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=region,
    )
    return client, bucket, region


def build_s3_key(accession: str, filename: str) -> str:
    return f"{S3_PREFIX}/{accession}/{filename}"


def upload_json(data: dict, accession: str, filename: str) -> Tuple[str, str]:
    """Upload JSON dict to S3. Returns (s3_key, s3_url)."""
    client, bucket, region = _get_client_and_bucket()
    key = build_s3_key(accession, filename)
    body = json.dumps(data, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    return key, url


def upload_text(content: str, accession: str, filename: str,
                content_type: str = "text/plain; charset=utf-8") -> Tuple[str, str]:
    """Upload text content (CSV, HTML, plain text) to S3. Returns (s3_key, s3_url)."""
    client, bucket, region = _get_client_and_bucket()
    key = build_s3_key(accession, filename)
    body = content.encode("utf-8")
    client.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)
    url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    return key, url


def download_json(s3_url: str) -> Optional[dict]:
    """Fetch an S3 HTTPS URL and parse as JSON."""
    if not s3_url:
        return None
    import urllib.request
    req = urllib.request.Request(s3_url, headers={"User-Agent": "CovenantPipeline/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_text(s3_url: str) -> Optional[str]:
    """Fetch an S3 HTTPS URL and return as string."""
    if not s3_url:
        return None
    import urllib.request
    req = urllib.request.Request(s3_url, headers={"User-Agent": "CovenantPipeline/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")
