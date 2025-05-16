from django.contrib.auth.backends import BaseBackend
from .models import User


class CustomAuthBackend(BaseBackend):
    """
    Custom authentication backend to work with our MongoDB-based User model
    """

    def authenticate(self, request, username=None, password=None, **kwargs):
        """
        Authenticate a user based on email/password.
        """
        # Get the email from username parameter (Django auth uses 'username' for the identifier)
        email = username

        if not email or not password:
            return None

        # Find the user by email
        user = User.find_by_email(email)

        # Check if user exists and password is correct
        if user and user.check_password(password):
            return user

        return None

    def get_user(self, user_id):
        """
        Get a user by ID for authentication purposes.
        """
        # The user_id could be either the ObjectId or the email, depending on settings
        # Try to find by _id first
        user = User.find_by_id(user_id)

        # If not found, try by email
        if not user:
            user = User.find_by_email(user_id)

        return user
