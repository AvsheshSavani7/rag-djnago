from django.urls import path
from .views import ThreadCreateView, ThreadListView, MessageListView, MessageSendView, ThreadDeleteView

urlpatterns = [
    path('threads/', ThreadCreateView.as_view(), name='create_thread'),
    path('threads/<str:user_id>/', ThreadListView.as_view(), name='get_threads'),
    path('threads/delete/<str:thread_id>/',
         ThreadDeleteView.as_view(), name='delete_thread'),
    path('messages/<str:thread_id>/',
         MessageListView.as_view(), name='get_messages'),
    path('send-message/', MessageSendView.as_view(), name='send_message'),
]
