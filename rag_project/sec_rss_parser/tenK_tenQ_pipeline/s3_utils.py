"""
S3 upload helpers for 10-K/10-Q pipeline. All keys are under prefix 10K_10Q/.
"""

import json
import os
from pathlib import Path
from typing import Union

S3_PREFIX = "10K_10Q"


def _get_client_and_bucket():
    try:
        import boto3
    except ImportError:
        raise ValueError("boto3 is required for S3 upload. Install with: pip install boto3")
    bucket = os.environ.get("AWS_S3_BUCKET")
    if not bucket:
        raise ValueError("AWS_S3_BUCKET is not set in environment")
    client = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )
    return client, bucket


def upload_file(
    local_path: Union[str, Path],
    s3_key_suffix: str,
    content_type: str = None,
) -> str:
    """
    Upload a file to S3. Full key = 10K_10Q/{s3_key_suffix}.
    Returns the S3 URL.
    """
    client, bucket = _get_client_and_bucket()
    key = f"{S3_PREFIX}/{s3_key_suffix}"
    extra = {"ContentType": content_type} if content_type else {}
    client.upload_file(str(local_path), bucket, key, ExtraArgs=extra)
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def upload_json(data: dict, s3_key_suffix: str) -> str:
    """
    Upload JSON to S3. Full key = 10K_10Q/{s3_key_suffix}.
    Returns the S3 URL.
    """
    client, bucket = _get_client_and_bucket()
    key = f"{S3_PREFIX}/{s3_key_suffix}"
    body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    return f"https://{bucket}.s3.amazonaws.com/{key}"
