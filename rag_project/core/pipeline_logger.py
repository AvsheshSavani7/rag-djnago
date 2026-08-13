import uuid
from core.logging_context import set_pipeline_context

# ---------------------------------------------------------------------------
# Pipeline name constants — import and use these instead of plain strings so
# folder names are consistent across the whole codebase.
# ---------------------------------------------------------------------------
SEC_8K       = "sec_8k"
SEC_FEED     = "sec_feed"
EX21         = "ex21"
SEC_SUMMARY  = "sec_summary"
PROXY        = "proxy"
PROXY_COMP   = "proxy_comparison"
TEN_K_TEN_Q  = "ten_k_ten_q"
COVENANT     = "covenant"
TERMINATION  = "termination"
DMA          = "dma"
DMA_SUMMARY  = "dma_summary"
MAE          = "mae"
REGENERATION      = "regeneration"
RSS               = "rss"
EMAIL             = "email"
GLOBAL_FORM_FEED  = "global_form_feed"
SEC_FEED_POLLER   = "sec_feed_poller"
SEC_FEED_BACKFILL = "sec_feed_backfill"


def start_pipeline(
    pipeline: str,
    accession: str = None,
    doc_type: str = "UNKNOWN",
) -> str:
    """
    Call once at the start of processing each accession/item.

    Sets pipeline context via ContextVar so every log line emitted in the
    current thread — and any threads spawned from it — automatically carries
    pipeline, run_id, accession, and doc_type.

    Returns the 6-char run_id so you can reference it in your own log lines.

    Usage:
        from core.pipeline_logger import start_pipeline, SEC_8K

        run_id = start_pipeline(SEC_8K, accession=accession_number, doc_type="8K")
        logger.info("Processing started")   # → pipeline=sec_8k | run_id=a8f91c | ...
    """
    run_id = uuid.uuid4().hex[:6]
    set_pipeline_context(
        pipeline=pipeline,
        run_id=run_id,
        accession=accession or "-",
        doc_type=doc_type,
    )
    return run_id
