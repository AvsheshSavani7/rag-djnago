import os
from pathlib import Path
from typing import Optional


def _get_s3_client():
    """
    Create a boto3 S3 client using the same env vars as other modules.
    """
    # Allow disabling uploads without crashing (used in multiple code paths)
    if os.environ.get("ENABLE_S3_UPLOADS", "").lower() == "false":
        return None

    try:
        import boto3  # local import so module import doesn't require boto3
    except ImportError:
        return None

    bucket = os.environ.get("AWS_S3_BUCKET")
    if not bucket:
        return None

    region = os.environ.get("AWS_REGION", "us-east-1")
    key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")

    # If creds are missing, boto3 will error on use; return None so caller can fallback.
    if not key_id or not secret:
        return None

    return (
        boto3.client(
            "s3",
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            region_name=region,
        ),
        bucket,
        region,
    )


def build_parsed_jsons_s3_key(ex21_url: str, sec_filing_accession_number: str) -> str:
    """
    Build S3 key for parsed JSON.

    Example:
      sec_filing_accession_number=000110465926029067
      ex21_url=https://.../tm268896d3_ex2-1.htm
    ->
      parsed_jsons/000110465926029067_tm268896d3_ex2-1.json
    """
    last = (ex21_url or "").split("/")[-1]
    tm_id = last.split(".")[0]  # strip .htm/.html
    prefix = "parsed_jsons"
    if not sec_filing_accession_number:
        # Caller should treat this as failure; keep key deterministic.
        sec_filing_accession_number = "unknown_accession"
    return f"{prefix}/{sec_filing_accession_number}_{tm_id}.json"


def upload_json_file_to_s3(
    local_file_path: str,
    s3_key: str,
    content_type: str = "application/json",
) -> Optional[str]:
    """
    Upload a JSON file to S3. Returns the public URL, or None if uploads are disabled/failed.
    """
    client_info = _get_s3_client()
    if not client_info:
        return None

    client, bucket, _region = client_info

    p = Path(local_file_path)
    if not p.exists():
        return None

    # Use upload_file to match patterns in other modules (supports large files too).
    client.upload_file(
        str(p),
        bucket,
        s3_key,
        ExtraArgs={"ContentType": content_type},
    )
    return f"https://{bucket}.s3.amazonaws.com/{s3_key}"

