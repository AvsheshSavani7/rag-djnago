from django.urls import path
from .views import (
    PipelineListView,
    GlobalSearchView,
    PipelineLogStreamView,
    TraceDateListView,
    TraceFileListView,
    TraceFileDetailView,
)

app_name = "logs_api"

urlpatterns = [
    # List all pipeline folders
    # GET /api/logs/pipelines/
    path("pipelines/", PipelineListView.as_view(), name="pipeline_list"),

    # Search across ALL pipelines at once
    # GET /api/logs/search/?accession=XXX&level=ERROR&run_id=a8f91c&search=text
    path("search/", GlobalSearchView.as_view(), name="global_search"),

    # Rolling log for one pipeline (with filters)
    # GET /api/logs/sec_8k/stream/?tail=200&accession=XXX&level=ERROR
    path("<str:pipeline>/stream/", PipelineLogStreamView.as_view(), name="pipeline_stream"),

    # List dates that have trace files for a pipeline
    # GET /api/logs/sec_8k/traces/
    path("<str:pipeline>/traces/", TraceDateListView.as_view(), name="trace_dates"),

    # List per-accession trace files for a pipeline+date
    # GET /api/logs/sec_8k/traces/2026-05-25/
    path("<str:pipeline>/traces/<str:date>/", TraceFileListView.as_view(), name="trace_file_list"),

    # Read one per-accession trace file
    # GET /api/logs/sec_8k/traces/2026-05-25/0001193125-26-126362_EX21_a8f91c.log/
    path("<str:pipeline>/traces/<str:date>/<str:filename>/",
         TraceFileDetailView.as_view(), name="trace_file_detail"),
]
