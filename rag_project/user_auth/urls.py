from django.urls import path
from .views import (
    UserRegistrationView,
    UserLoginView,
    ProtectedTestView,
    ChangePasswordView,
    AdminResetPasswordView,
    ListUsersView,
    ChangeRoleView,
)
from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
    path('register/', UserRegistrationView.as_view(), name='register'),
    path('login/', UserLoginView.as_view(), name='login'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('change-password/', ChangePasswordView.as_view(), name='change_password'),
    path('admin-reset-password/', AdminResetPasswordView.as_view(), name='admin_reset_password'),
    path('users/', ListUsersView.as_view(), name='list_users'),
    path('change-role/', ChangeRoleView.as_view(), name='change_role'),
    # path('protected/', ProtectedTestView.as_view(), name='protected'),
]
