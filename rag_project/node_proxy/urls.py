from django.urls import path
from .views import ProcessView, ItemsListView

urlpatterns = [
    path('process/', ProcessView.as_view(), name='process'),
    # path('items/', ItemsListView.as_view(), name='items'),
]


# /api/v1/process/  excel file upload
