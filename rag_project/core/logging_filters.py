import logging
from core.logging_context import get_pipeline, get_run_id, get_accession, get_doc_type


class PipelineContextFilter(logging.Filter):
    """
    Stamps every LogRecord with pipeline context fields.
    No changes to existing logger.info() / logger.error() calls are needed —
    this filter runs automatically for every handler it is attached to.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.pipeline  = get_pipeline()
        record.run_id    = get_run_id()
        record.accession = get_accession()
        record.doc_type  = get_doc_type()
        return True
