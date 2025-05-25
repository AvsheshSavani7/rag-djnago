from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny, IsAuthenticated
from .serializers import UserRegistrationSerializer, UserLoginSerializer
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
