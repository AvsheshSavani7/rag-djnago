from django.urls import path
from .views import (
    ProcessFileView,
    ProcessingJobDetailView,
    ProcessEmbeddingsView,
    ListAllDealsView,
    PineconeVectorListView,
    UpdatePineconeVectorView,
    ChatWithAIView,
    SummaryGenerationView,
    SummaryEngineView
)

urlpatterns = [
    path('process/', ProcessFileView.as_view(), name='process_file'),
    path('embed/', ProcessEmbeddingsView.as_view(), name='process_embeddings'),
    path('deals/', ListAllDealsView.as_view(), name='list_deals'),
    path('deals/<str:id>/',
         ProcessingJobDetailView.as_view(), name='job_detail'),
    path('vectors/<str:deal_id>/',
         PineconeVectorListView.as_view(), name='vector_list'),
    path('vectors/update/<str:vector_id>/',
         UpdatePineconeVectorView.as_view(), name='update_vector'),
    path('chat/', ChatWithAIView.as_view(), name='chat_with_ai'),
    path('summary/', SummaryGenerationView.as_view(), name='generate_summary'),
    path('summary/engine/', SummaryEngineView.as_view(),
         name='generate_summary_engine')
]


# /api/v1/process/  direct file upload
# /api/v1/embed/  direct flattern file processing
# /api/v1/deals/  list all deals
# /api/v1/deals/<str:id>/  get deal by id
# /api/v1/vectors/<str:deal_id>/  get vectors by deal id
# /api/v1/vectors/update/<str:vector_id>/  update vector by id
# /api/v1/chat/  chat with ai by deal id
# /api/v1/summary/  generate summary by deal id
