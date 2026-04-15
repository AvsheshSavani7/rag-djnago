from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny, IsAuthenticated
from .serializers import (
    UserRegistrationSerializer,
    UserLoginSerializer,
    ChangePasswordSerializer,
    AdminResetPasswordSerializer,
    ChangeRoleSerializer,
)
from django.contrib.auth.hashers import make_password
from user_auth.models import User


class UserRegistrationView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response({
                'message': 'User registered successfully.',
                'user': {
                    'email': user.email,
                    'role': user.role,
                    '_id': user._id,
                }
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserLoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = UserLoginSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            user = serializer.validated_data['user']
            return Response({
                'message': 'Login successful',
                'user_id': user._id,
                'user_email': user.email,
                'role': user.role,
                'refresh': serializer.validated_data['refresh'],
                'access': serializer.validated_data['access'],
            }, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_401_UNAUTHORIZED)


class ChangePasswordView(APIView):
    permission_classes = [IsAuthenticated]

    def put(self, request):
        serializer = ChangePasswordSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            user = request.user
            user.password = make_password(serializer.validated_data['new_password'])
            user.save()
            return Response({'message': 'Password changed successfully.'}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class AdminResetPasswordView(APIView):
    permission_classes = [IsAuthenticated]

    def put(self, request):
        if request.user.role != 'admin':
            return Response({'error': 'Only admins can reset passwords.'}, status=status.HTTP_403_FORBIDDEN)

        serializer = AdminResetPasswordSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            target_user = User.objects(email=email).first()
            target_user.password = make_password(serializer.validated_data['new_password'])
            target_user.save()
            return Response(
                {'message': f'Password reset successfully for {email}.'},
                status=status.HTTP_200_OK,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ListUsersView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.role != 'admin':
            return Response({'error': 'Only admins can view users.'}, status=status.HTTP_403_FORBIDDEN)

        users = User.objects.all()
        users_list = [
            {
                '_id': str(user._id),
                'email': user.email,
                'role': user.role,
                'createdAt': user.createdAt.isoformat() if user.createdAt else None,
                'last_login': user.last_login.isoformat() if user.last_login else None,
            }
            for user in users
        ]
        return Response({'users': users_list}, status=status.HTTP_200_OK)


class ChangeRoleView(APIView):
    permission_classes = [IsAuthenticated]

    def put(self, request):
        if request.user.role != 'admin':
            return Response({'error': 'Only admins can change roles.'}, status=status.HTTP_403_FORBIDDEN)

        serializer = ChangeRoleSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            target_user = User.objects(email=email).first()
            target_user.role = serializer.validated_data['role']
            target_user.save()
            return Response(
                {'message': f'Role updated to "{target_user.role}" for {email}.'},
                status=status.HTTP_200_OK,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ProtectedTestView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        # user will be an instance of your custom User model (MongoEngine Document)
        # But DRF might not inject it correctly unless you've patched DRF auth backend
        try:
            user = User.objects(email=request.user.email).first()
            return Response({
                'message': 'You have access to protected content',
                'user_email': user.email,
                'role': user.role
            }, status=status.HTTP_200_OK)
        except Exception:
            return Response({'error': 'User not found or unauthorized'}, status=401)
