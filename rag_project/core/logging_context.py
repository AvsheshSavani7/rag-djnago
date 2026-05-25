from contextvars import ContextVar

_pipeline  = ContextVar("pipeline",  default="app")
_run_id    = ContextVar("run_id",    default="-")
_accession = ContextVar("accession", default="-")
_doc_type  = ContextVar("doc_type",  default="UNKNOWN")


def set_pipeline_context(
    pipeline: str,
    run_id: str = "-",
    accession: str = "-",
    doc_type: str = "UNKNOWN",
):
    _pipeline.set(pipeline)
    _run_id.set(run_id)
    _accession.set(accession)
    _doc_type.set(doc_type)


def get_pipeline()  -> str: return _pipeline.get()
def get_run_id()    -> str: return _run_id.get()
def get_accession() -> str: return _accession.get()
def get_doc_type()  -> str: return _doc_type.get()
