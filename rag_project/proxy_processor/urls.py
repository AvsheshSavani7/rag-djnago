from django.urls import path
from . import views

app_name = 'proxy_processor'

urlpatterns = [
    # Process SEC document
    path('proxry-processor/', views.process_sec_document,
         name='process_sec_document'),

    # List all processing jobs
    path('jobs/', views.list_processing_jobs, name='list_processing_jobs'),
]
