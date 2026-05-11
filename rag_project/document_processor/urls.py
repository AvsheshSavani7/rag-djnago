from django.urls import path
from .views import (
    ProcessFileView,
    ProcessingJobDetailView,
    ProcessEmbeddingsView,
    ListAllDealsView,
    ListAllDealsNoPaginationView,
    ExportDealsExcelView,
    PineconeVectorListView,
    UpdatePineconeVectorView,
    ChatWithAIView,
    SummaryGenerationView,
    SummaryEngineView,
    JobStatusView,
    HighValueFollowersView,
    TweetsView,
    RedditPostsView,
    RedditScraperTaskView,
    RegeneratePipelineView,
    RegeneratePipelineStatusView,
)

urlpatterns = [
    path('process/', ProcessFileView.as_view(), name='process_file'),
    path('embed/', ProcessEmbeddingsView.as_view(), name='process_embeddings'),
    path('deals/', ListAllDealsView.as_view(), name='list_deals'),
    path('deals/all/', ListAllDealsNoPaginationView.as_view(), name='list_all_deals'),
    path('deals/export/', ExportDealsExcelView.as_view(),
         name='export_deals_excel'),
    path('deals/<str:id>/',
         ProcessingJobDetailView.as_view(), name='job_detail'),
    path('jobs/<str:job_id>/',
         JobStatusView.as_view(), name='job_status'),
    path('vectors/<str:deal_id>/',
         PineconeVectorListView.as_view(), name='vector_list'),
    path('vectors/update/<str:vector_id>/',
         UpdatePineconeVectorView.as_view(), name='update_vector'),
    path('chat/', ChatWithAIView.as_view(), name='chat_with_ai'),
    path('summary/', SummaryGenerationView.as_view(), name='generate_summary'),
    path('summary/engine/', SummaryEngineView.as_view(),
         name='generate_summary_engine'),
    path('highvaluefollowers/<str:deal_id>/',
         HighValueFollowersView.as_view(), name='high_value_followers'),
    path('tweets/<str:deal_id>/',
         TweetsView.as_view(), name='tweets'),
    path('redditposts/<str:deal_id>/',
         RedditPostsView.as_view(), name='reddit_posts'),
    path('reddit-scraper/tasks/',
         RedditScraperTaskView.as_view(), name='reddit_scraper_tasks'),
    path('regenerate/',
         RegeneratePipelineView.as_view(), name='regenerate_pipeline'),
    path('regenerate/<str:run_id>/',
         RegeneratePipelineStatusView.as_view(), name='regenerate_status'),
    path('regenerate/deal/<str:deal_id>/',
         RegeneratePipelineStatusView.as_view(), name='regenerate_status_by_deal'),
]


# /api/files/process/                      direct file upload
# /api/files/embed/                        direct flatten file processing
# /api/files/deals/                        list all deals
# /api/files/deals/<str:id>/               get deal by id
# /api/files/vectors/<str:deal_id>/        get vectors by deal id
# /api/files/vectors/update/<str:vector_id>/  update vector by id
# /api/files/chat/                         chat with ai by deal id
# /api/files/summary/                      generate summary by deal id
# /api/files/summary/engine/               generate summary via engine
# /api/files/regenerate/                   POST — start pipeline regeneration
# /api/files/regenerate/<run_id>/          GET  — poll run progress by run_id
# /api/files/regenerate/deal/<deal_id>/    GET  — poll latest run for a deal
# /api/files/highvaluefollowers/<deal_id>/ get high value followers
# /api/files/tweets/<deal_id>/             get tweets for a deal
# /api/files/redditposts/<deal_id>/        get reddit posts for a deal
