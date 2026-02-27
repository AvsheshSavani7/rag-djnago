from django.urls import path
from .views import (
    ProcessSECFeedView,
    SECFilingListView,
    SECFilingDetailView,
    SEC8KFilingListView,
    SECFeedStatusView,
    SECFilingStatsView,
)

app_name = 'sec_rss_parser'

urlpatterns = [
    # Process SEC RSS feed
    path('process-feed/', ProcessSECFeedView.as_view(), name='process_feed'),

    # List SEC filings
    path('filings/', SECFilingListView.as_view(), name='filings_list'),

    # Get specific filing details
    path('filings/<str:filing_id>/',
         SECFilingDetailView.as_view(), name='filing_detail'),

    # List 8-K filings with HTM files
    path('8k-filings/', SEC8KFilingListView.as_view(), name='8k_filings'),

    # Get feed status
    path('feed-status/', SECFeedStatusView.as_view(), name='feed_status'),

    # Get filing statistics
    path('stats/', SECFilingStatsView.as_view(), name='filing_stats'),

]
