"""
Regeneration Pipeline for DMA, Termination, Covenant, and MAE.

Called from RegeneratePipelineView. Each step reuses existing service functions —
no business logic is duplicated.

DMA hierarchy (top-down cascade):
    parsing → embedding → schema

Independent pipelines (run as selected):
    termination, covenant, mae

API contract:
    POST  /api/files/regenerate/          — start a run
    GET   /api/files/regenerate/<run_id>/  — poll progress
"""

import logging
import sys
import time
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from bson import ObjectId
from mongoengine.errors import DoesNotExist

from document_processor.models import ProcessingJob, DealSchemaResults
from document_processor.transform_json import simplify_json

logger = logging.getLogger(__name__)

DMA_STEPS = ["parsing", "embedding", "schema"]
INDEPENDENT_STEPS = ["termination", "covenant", "mae"]
ALL_VALID_STEPS = DMA_STEPS + INDEPENDENT_STEPS


# ---------------------------------------------------------------------------
#  In-memory run tracker (thread-safe, no DB model needed)
# ---------------------------------------------------------------------------
_tracker_lock = threading.Lock()
_active_runs: dict = {}  # run_id → run dict
_MAX_COMPLETED_RUNS = 200  # auto-evict oldest completed runs to bound memory


def _create_run(run_id: str, deal_id: str, plan: dict, requested_steps: list):
    """Register a new regeneration run."""
    run = {
        "run_id": run_id,
        "deal_id": deal_id,
        "status": "processing",
        "requested_steps": requested_steps,
        "plan": plan,
        "current_step": None,
        "steps": {},
        "error": None,
        "started_at": datetime.utcnow().isoformat(),
        "finished_at": None,
    }
    with _tracker_lock:
        _active_runs[run_id] = run
        _evict_old_runs()
    return run


def _update_step(run_id: str, step: str, step_status: str, **extra):
    """Update a step's status inside a run."""
    with _tracker_lock:
        run = _active_runs.get(run_id)
        if not run:
            return
        run["current_step"] = step if step_status == "processing" else run.get(
            "current_step")
        entry = run["steps"].get(step, {})
        entry["status"] = step_status
        entry.update(extra)
        run["steps"][step] = entry


def _finish_run(run_id: str, final_status: str = "completed", error: str = None):
    """Mark a run as finished (completed / failed)."""
    with _tracker_lock:
        run = _active_runs.get(run_id)
        if not run:
            return
        run["status"] = final_status
        run["current_step"] = None
        run["finished_at"] = datetime.utcnow().isoformat()
        if error:
            run["error"] = error


def get_run_status(run_id: str) -> dict | None:
    """Read a run's current state (called from the polling view)."""
    with _tracker_lock:
        return _active_runs.get(run_id)


def get_latest_run_for_deal(deal_id: str) -> dict | None:
    """Return the most recent run for a deal (convenience for frontend)."""
    with _tracker_lock:
        candidates = [r for r in _active_runs.values()
                      if r["deal_id"] == deal_id]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["started_at"])


def _evict_old_runs():
    """Keep memory bounded by removing oldest completed runs."""
    completed = [
        (rid, r) for rid, r in _active_runs.items() if r["status"] != "processing"
    ]
    if len(completed) > _MAX_COMPLETED_RUNS:
        completed.sort(key=lambda x: x[1].get("finished_at") or "")
        for rid, _ in completed[: len(completed) - _MAX_COMPLETED_RUNS]:
            _active_runs.pop(rid, None)


def generate_run_id() -> str:
    return uuid.uuid4().hex[:16]


def resolve_execution_plan(requested_steps: list) -> dict:
    """
    Determine what to actually run from a list of user-selected checkboxes.

    DMA steps follow a top-down cascade: selecting an earlier step implies
    all later ones.  Independent steps run as-is.

    Returns:
        {
            "dma_start_from": "embedding" | None,
            "independent": ["termination", "covenant"],
        }
    """
    dma_start = None
    for step in DMA_STEPS:
        if step in requested_steps:
            dma_start = step
            break

    independent = [s for s in requested_steps if s in INDEPENDENT_STEPS]
    return {"dma_start_from": dma_start, "independent": independent}


def validate_steps(steps: list) -> list:
    """Return list of invalid step names (empty list == all valid)."""
    return [s for s in steps if s not in ALL_VALID_STEPS]


# ---------------------------------------------------------------------------
#  DMA step runners
# ---------------------------------------------------------------------------

def _step_parsing(deal_id: str, job: ProcessingJob) -> str:
    """
    Re-parse the SEC document into flattened JSON chunks and upload to S3.
    Returns the new flattened_json_url.
    """
    from document_processor.services import FlattenProcessor

    sec_url = job.sec_url
    if not sec_url:
        raise ValueError(
            f"Deal {deal_id} has no sec_url — cannot re-parse. "
            "Upload the document first via /api/files/process/."
        )

    logger.info("[regenerate] parsing: deal_id=%s url=%s", deal_id, sec_url)
    processor = FlattenProcessor(file_url=sec_url)
    result = processor.process()

    flattened_url = result.get("flattened_json_url")
    if not flattened_url:
        raise RuntimeError("FlattenProcessor returned no flattened_json_url")

    job.flattened_json_url = flattened_url
    job.save()
    logger.info("[regenerate] parsing complete: %s", flattened_url)
    return flattened_url


def _step_embedding(deal_id: str, job: ProcessingJob):
    """
    Delete existing Pinecone vectors for the deal, then re-embed from the
    flattened JSON stored on S3.
    """
    from document_processor.services import EmbeddingService, S3Service

    flattened_url = job.flattened_json_url
    if not flattened_url:
        raise ValueError(
            f"Deal {deal_id} has no flattened_json_url — run parsing first."
        )

    logger.info("[regenerate] embedding: deal_id=%s", deal_id)
    job.update_embedding_status("PROCESSING")

    embedding_service = EmbeddingService()

    # Delete old vectors before re-upserting
    try:
        logger.info(
            "[regenerate] deleting old Pinecone vectors for deal_id=%s", deal_id)
        embedding_service.index.delete(filter={"deal_id": str(deal_id)})
        logger.info("[regenerate] old vectors deleted")
    except Exception as e:
        logger.warning(
            "[regenerate] vector deletion failed (continuing): %s", e)

    # Download chunks and re-embed
    s3_service = S3Service()
    chunks = s3_service.download_from_url(flattened_url)
    logger.info("[regenerate] downloaded %d chunks", len(chunks))

    embedding_service.process_chunks(chunks, str(deal_id))

    job.update_embedding_status("COMPLETED")
    logger.info("[regenerate] embedding complete")


def _step_schema(deal_id: str, job: ProcessingJob):
    """Regenerate schema results from Pinecone vectors."""
    from document_processor.services import SchemaCategorySearch

    logger.info("[regenerate] schema: deal_id=%s", deal_id)

    schema_search = SchemaCategorySearch()
    category_results = schema_search.search_all_schema_categories(
        deal_id=str(deal_id))
    simplified_data = simplify_json(category_results)

    job.save_json_to_db(simplified_data)

    DealSchemaResults.save_or_update(
        deal_id=ObjectId(deal_id),
        schema_results=simplified_data,
        schema_processing_completed=job.schema_processing_completed,
        schema_processing_timestamp=job.schema_processing_timestamp,
    )
    logger.info("[regenerate] schema complete")


# ---------------------------------------------------------------------------
#  Independent pipeline runners
# ---------------------------------------------------------------------------

def _extract_accession_from_url(sec_url: str) -> str:
    """Extract formatted accession (e.g. 0001193125-26-134889) from an SEC URL."""
    if not sec_url:
        return ""
    for segment in sec_url.split("/"):
        if len(segment) == 18 and segment.isdigit():
            return f"{segment[:10]}-{segment[10:12]}-{segment[12:]}"
    return ""


def _build_deal_name(job: ProcessingJob) -> str:
    if getattr(job, "acquire_name", None) and getattr(job, "target_name", None):
        return f"{job.acquire_name} / {job.target_name}"
    return getattr(job, "target_name", None) or ""


def _run_termination(deal_id: str, job: ProcessingJob):
    sec_url = job.sec_url
    if not sec_url:
        logger.warning(
            "[regenerate] termination skipped — no sec_url for deal %s", deal_id)
        return

    accession = _extract_accession_from_url(sec_url)
    if not accession:
        logger.warning(
            "[regenerate] termination skipped — could not extract accession from %s", sec_url)
        return

    _covenant_dir = Path(__file__).resolve().parent.parent / \
        "sec_rss_parser" / "Covenenat Project Feb 2026"
    if str(_covenant_dir) not in sys.path:
        sys.path.insert(0, str(_covenant_dir))

    from termination_pipeline import run_termination_pipeline_s3

    logger.info(
        "[regenerate] termination: deal_id=%s accession=%s", deal_id, accession)
    run_termination_pipeline_s3(
        url=sec_url,
        accession_number=accession,
        doc_type="2.1",
        deal_id=str(deal_id),
        deal_name=_build_deal_name(job),
        send_email=False,
    )
    logger.info("[regenerate] termination complete")


def _run_covenant(deal_id: str, job: ProcessingJob):
    sec_url = job.sec_url
    if not sec_url:
        logger.warning(
            "[regenerate] covenant skipped — no sec_url for deal %s", deal_id)
        return

    accession = _extract_accession_from_url(sec_url)
    if not accession:
        logger.warning(
            "[regenerate] covenant skipped — could not extract accession from %s", sec_url)
        return

    _covenant_dir = Path(__file__).resolve().parent.parent / \
        "sec_rss_parser" / "Covenenat Project Feb 2026"
    if str(_covenant_dir) not in sys.path:
        sys.path.insert(0, str(_covenant_dir))

    from covenant_pipeline import run_covenant_pipeline_s3

    logger.info("[regenerate] covenant: deal_id=%s accession=%s",
                deal_id, accession)
    run_covenant_pipeline_s3(
        url=sec_url,
        accession_number=accession,
        deal_id=str(deal_id),
        deal_name=_build_deal_name(job),
        send_email=False,
    )
    logger.info("[regenerate] covenant complete")


def _run_mae(deal_id: str):
    logger.info("[regenerate] mae: deal_id=%s", deal_id)
    from document_processor.MAE.run_full_pipeline import run_pipeline_for_deal_id

    results = run_pipeline_for_deal_id(str(deal_id))
    if results:
        logger.info("[regenerate] mae complete — risk_level=%s",
                    results.get("risk_assessment", {}).get("risk_summary", {}).get("final_risk_level", "N/A"))
    else:
        logger.warning(
            "[regenerate] mae found no MAE text for deal %s", deal_id)


# ---------------------------------------------------------------------------
#  Main orchestrator
# ---------------------------------------------------------------------------

INDEPENDENT_RUNNERS = {
    "termination": _run_termination,
    "covenant": _run_covenant,
    "mae": _run_mae,
}

STEP_DESCRIPTIONS = {
    "parsing": "Parsing SEC document into chunks",
    "embedding": "Generating embeddings and upserting to Pinecone",
    "schema": "Extracting schema fields from vector store",
    "termination": "Running termination analysis (5 sub-steps, may take several minutes)",
    "covenant": "Running covenant analysis (5 sub-steps, may take several minutes)",
    "mae": "Running MAE extraction pipeline",
}


def run_regeneration_pipeline(deal_id: str, requested_steps: list, run_id: str = None) -> dict:
    """
    Main entry point — called from the view's background thread.

    Args:
        deal_id: ProcessingJob / deal _id
        requested_steps: list of step names selected by the admin
        run_id: tracker ID (created by the view before dispatching)

    Returns:
        dict with per-step status and any error messages
    """
    results = {}
    plan = resolve_execution_plan(requested_steps)

    try:
        job = ProcessingJob.objects.get(id=ObjectId(deal_id))
    except DoesNotExist:
        if run_id:
            _finish_run(run_id, "failed", error=f"Deal {deal_id} not found")
        return {"error": f"Deal {deal_id} not found"}

    # --- DMA cascade ---
    if plan["dma_start_from"]:
        idx = DMA_STEPS.index(plan["dma_start_from"])
        steps_to_run = DMA_STEPS[idx:]
        logger.info("[regenerate] DMA cascade for deal %s: %s",
                    deal_id, steps_to_run)

        for step in steps_to_run:
            started = time.time()
            desc = STEP_DESCRIPTIONS.get(step, step)
            if run_id:
                _update_step(run_id, step, "processing", message=desc)
            try:
                if step == "parsing":
                    _step_parsing(deal_id, job)
                    job.reload()
                elif step == "embedding":
                    _step_embedding(deal_id, job)
                elif step == "schema":
                    _step_schema(deal_id, job)

                duration = int((time.time() - started) * 1000)
                results[step] = {"status": "completed",
                                 "duration_ms": duration}
                if run_id:
                    _update_step(run_id, step, "completed",
                                 duration_ms=duration)
            except Exception as e:
                logger.error("[regenerate] DMA step '%s' failed: %s", step, e)
                logger.error(traceback.format_exc())
                results[step] = {"status": "failed", "error": str(e)}
                if run_id:
                    _update_step(run_id, step, "failed", error=str(e))
                for remaining in steps_to_run[steps_to_run.index(step) + 1:]:
                    results[remaining] = {
                        "status": "skipped", "reason": f"{step} failed"}
                    if run_id:
                        _update_step(run_id, remaining, "skipped",
                                     reason=f"{step} failed")
                break

    # --- Independent pipelines ---
    for step in plan["independent"]:
        started = time.time()
        desc = STEP_DESCRIPTIONS.get(step, step)
        if run_id:
            _update_step(run_id, step, "processing", message=desc)
        try:
            runner = INDEPENDENT_RUNNERS[step]
            if step == "mae":
                runner(deal_id)
            else:
                runner(deal_id, job)

            duration = int((time.time() - started) * 1000)
            results[step] = {"status": "completed", "duration_ms": duration}
            if run_id:
                _update_step(run_id, step, "completed", duration_ms=duration)
        except Exception as e:
            logger.error("[regenerate] step '%s' failed: %s", step, e)
            logger.error(traceback.format_exc())
            results[step] = {"status": "failed", "error": str(e)}
            if run_id:
                _update_step(run_id, step, "failed", error=str(e))

    # --- Finalize ---
    any_failed = any(r.get("status") == "failed" for r in results.values())
    final_status = "failed" if any_failed else "completed"
    if run_id:
        _finish_run(run_id, final_status)

    logger.info("[regenerate] finished for deal %s — status=%s results=%s",
                deal_id, final_status, results)
    return results
