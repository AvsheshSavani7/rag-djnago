"""
S3 helpers for proxy comparison pipeline.
Uses same buckets/folders as sec_summarizers: summary_json/ and summary_docx/.
Naming: proxy_comp_{deal_id}_{record_id}_*.json | *.txt | *.docx
Works with plain Python (no Django). Requires boto3 and requests.
"""

import json
import os
from typing import Tuple

FOLDER_DOCX = "summary_docx"
FOLDER_JSON = "summary_json"


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


def upload_json(data: dict, s3_key_suffix: str) -> Tuple[str, str]:
    """Upload JSON to S3. Key = summary_json/{s3_key_suffix}. Returns (key, url)."""
    client, bucket, region = _get_client_and_bucket()
    key = f"{FOLDER_JSON}/{s3_key_suffix}"
    body = json.dumps(data, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    return key, url


def upload_docx_bytes(data: bytes, s3_key_suffix: str) -> Tuple[str, str]:
    """Upload DOCX bytes to S3. Key = summary_docx/{s3_key_suffix}. Returns (key, url)."""
    client, bucket, region = _get_client_and_bucket()
    key = f"{FOLDER_DOCX}/{s3_key_suffix}"
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    return key, url


def upload_text(s3_key_suffix: str, content: str) -> Tuple[str, str]:
    """Upload plain text to S3. Key = summary_json/{s3_key_suffix} (e.g. ..._change_report.txt). Returns (key, url)."""
    client, bucket, region = _get_client_and_bucket()
    key = f"{FOLDER_JSON}/{s3_key_suffix}"
    body = content.encode("utf-8")
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="text/plain; charset=utf-8",
    )
    url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    return key, url


def download_json_from_url(url: str) -> dict:
    """Fetch URL and parse as JSON. Works with S3 HTTPS URLs. Uses stdlib urllib. Raises on failure."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "ProxyComp/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def proxy_comp_key_suffix(deal_id: str, record_id: str, suffix: str) -> str:
    """Build S3 key suffix for proxy comparison: proxy_comp_{deal_id}_{record_id}_{suffix}."""
    return f"proxy_comp_{deal_id}_{record_id}_{suffix}"
