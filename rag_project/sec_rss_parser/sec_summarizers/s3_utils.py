"""
S3 upload helpers for sec_summarizers. Keys under summary_docx/ and summary_json/.
Returns S3 path (key) and full S3 URL; use get_s3_url(key) to build URL from a key.
"""

import json
import os
from pathlib import Path
from typing import Union

FOLDER_DOCX = "summary_docx"
FOLDER_JSON = "summary_json"


def _get_client_and_bucket():
    try:
        import boto3
    except ImportError:
        raise ValueError("boto3 is required for S3 upload. Install with: pip install boto3")
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


def get_s3_url(key: str) -> str:
    """Build full S3 URL for a given key. Requires AWS_S3_BUCKET (and optionally AWS_REGION) in env."""
    bucket = os.environ.get("AWS_S3_BUCKET")
    if not bucket:
        raise ValueError("AWS_S3_BUCKET is not set in environment")
    region = os.environ.get("AWS_REGION", "us-east-1")
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


def upload_docx_bytes(data: bytes, s3_key_suffix: str) -> tuple[str, str]:
    """
    Upload DOCX bytes to S3. Key = summary_docx/{s3_key_suffix}.
    Returns (s3_path, s3_url).
    """
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


def upload_file(
    local_path: Union[str, Path],
    s3_key_suffix: str,
    content_type: str = None,
) -> tuple[str, str]:
    """
    Upload a file to S3. Key = summary_docx/{s3_key_suffix}.
    Returns (s3_path, s3_url).
    """
    client, bucket, region = _get_client_and_bucket()
    key = f"{FOLDER_DOCX}/{s3_key_suffix}"
    extra = {"ContentType": content_type} if content_type else {}
    client.upload_file(str(local_path), bucket, key, ExtraArgs=extra)
    url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    return key, url


def upload_json(data: dict, s3_key_suffix: str) -> tuple[str, str]:
    """
    Upload JSON to S3. Key = summary_json/{s3_key_suffix}.
    Returns (s3_path, s3_url).
    """
    client, bucket, region = _get_client_and_bucket()
    key = f"{FOLDER_JSON}/{s3_key_suffix}"
    body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    return key, url
