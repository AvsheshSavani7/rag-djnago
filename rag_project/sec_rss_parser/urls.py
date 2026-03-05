from django.urls import path
from .views import (
    ProcessSECFeedView,
    SECFilingListView,
    SECFilingDetailView,
    SEC8KFilingListView,
    SECFeedStatusView,
    SECFilingStatsView,
    FetchSECFeedByDealCIKView,
)

app_name = 'sec_rss_parser'

urlpatterns = [
    # Process SEC RSS feed
    path('process-feed/', ProcessSECFeedView.as_view(), name='process_feed'),

    # Fetch SEC feed by deal CIK
    path('fetch-feed-by-deal-cik/', FetchSECFeedByDealCIKView.as_view(),
         name='fetch_feed_by_deal_cik'),
    # demo RSS file curl "http://localhost:8000/sec-rss-parser/fetch-feed-by-deal-cik/?use_demo=true&limit_deals=5"
    # live SEC feeds curl "http://localhost:8000/sec-rss-parser/fetch-feed-by-deal-cik/?limit_deals=10"
    # custom output path:curl "http://localhost:8000/sec-rss-parser/fetch-feed-by-deal-cik/?use_demo=true&limit_deals=5&output_path=/path/to/output.json"

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
