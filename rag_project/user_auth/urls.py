from django.urls import path
from .views import UserRegistrationView, UserLoginView, ProtectedTestView
from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
    path('register/', UserRegistrationView.as_view(), name='register'),
    path('login/', UserLoginView.as_view(), name='login'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    # path('protected/', ProtectedTestView.as_view(), name='protected'),
]
