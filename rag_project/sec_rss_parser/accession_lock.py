import os
import uuid
from datetime import datetime, timedelta

from mongoengine.errors import NotUniqueError

from sec_rss_parser.models import (
    AccessionLookedUp,
    AccessionProcessingLock,
    SECFiling,
    SECFilingSummary,
)


DEFAULT_LOCK_TTL_SECONDS = int(os.environ.get("ACCESSION_LOCK_TTL_SECONDS", "900"))


def _materialized_output_exists(accession_number):
    """
    Heuristic for crash recovery:
    if filing/summary already exists, treat as processed to avoid duplicate side effects.
    """
    if not accession_number:
        return False
    if SECFilingSummary.objects(accession_number=accession_number).first():
        return True
    if SECFiling.objects(accession_number=accession_number).first():
        return True
    return False


def acquire_accession_lock(accession_number, source="unknown", ttl_seconds=None):
    """
    Try to acquire processing lock for accession_number.

    Returns owner_id string on success, else None.
    """
    if not accession_number:
        return None

    # Fast path: already finalized
    if AccessionLookedUp.objects(accession_number=accession_number).first():
        return None

    owner_id = f"{source}:{uuid.uuid4().hex}"
    ttl = ttl_seconds if ttl_seconds is not None else DEFAULT_LOCK_TTL_SECONDS
    expires_at = datetime.utcnow() + timedelta(seconds=max(60, int(ttl)))

    try:
        AccessionProcessingLock(
            accession_number=accession_number,
            owner_id=owner_id,
            expires_at=expires_at,
        ).save()
        return owner_id
    except (NotUniqueError, Exception) as e:
        # Duplicate key => someone else likely holds the lock.
        # If stale lock exists, attempt stale recovery exactly once.
        msg = str(e).lower()
        if "duplicate" not in msg and "e11000" not in msg:
            return None

        existing = AccessionProcessingLock.objects(
            accession_number=accession_number
        ).first()
        if not existing:
            return None

        now = datetime.utcnow()
        if existing.expires_at and existing.expires_at > now:
            return None

        # Stale lock found: if output exists, finalize looked_up and do not reprocess.
        if _materialized_output_exists(accession_number):
            try:
                AccessionLookedUp(accession_number=accession_number).save()
            except Exception:
                pass
            try:
                existing.delete()
            except Exception:
                pass
            return None

        # No materialized output; delete stale lock and retry once.
        try:
            existing.delete()
        except Exception:
            return None

        try:
            AccessionProcessingLock(
                accession_number=accession_number,
                owner_id=owner_id,
                expires_at=expires_at,
            ).save()
            return owner_id
        except Exception:
            return None


def mark_accession_processed(accession_number):
    if not accession_number:
        return
    try:
        AccessionLookedUp(accession_number=accession_number).save()
    except Exception:
        pass


def release_accession_lock(accession_number, owner_id):
    if not accession_number or not owner_id:
        return
    try:
        lock = AccessionProcessingLock.objects(
            accession_number=accession_number, owner_id=owner_id
        ).first()
        if lock:
            lock.delete()
    except Exception:
        pass

