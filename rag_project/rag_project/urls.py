"""
URL configuration for rag_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path, include
from django.http import JsonResponse
from .example import ExampleAPIView
# Django admin removed
# from rest_framework.documentation import include_docs_urls

urlpatterns = [
    # Admin path removed
    path("", lambda request: JsonResponse({"status": "API is live ✅"})),
    path("api/example/", ExampleAPIView.as_view()),  
    path("api/", include("node_proxy.urls")),
    path("api/files/", include("document_processor.urls")),
    path("api/auth/", include("user_auth.urls")),
    path('api/gpt/', include('gpt_chat.urls')),
    # path("docs/", include_docs_urls(title="RAG API")),
]
