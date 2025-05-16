from rest_framework import serializers
from django.utils import timezone
from .models import MongoUser
from rest_framework_simplejwt.tokens import RefreshToken


class UserRegistrationSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)
    password = serializers.CharField(write_only=True, required=True, style={
                                     'input_type': 'password'})
    role = serializers.CharField(required=False, default='admin')

    def create(self, validated_data):
        user = MongoUser.objects.create_user(
            email=validated_data['email'],
            password=validated_data['password'],
            role=validated_data.get('role', 'admin')
        )
        return user

    def to_representation(self, instance):
        return {
            'email': instance.email,
            'role': instance.role
        }


class UserLoginSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)
    password = serializers.CharField(write_only=True, required=True, style={
                                     'input_type': 'password'})

    def validate(self, data):
        email = data.get('email')
        password = data.get('password')

        if email and password:
            # Find the user by email
            user = MongoUser.find_by_email(email)

            # Check if user exists and password is correct
            if not user or not user.check_password(password):
                raise serializers.ValidationError(
                    'Unable to login with provided credentials.')
        else:
            raise serializers.ValidationError(
                'Must include "email" and "password".')

        # Update last login time (if needed)
        user.updatedAt = timezone.now()
        user.save()

        # Generate JWT token - this part stays the same as JWT doesn't depend on ORM
        # Create a dictionary that JWT token generation expects
        refresh = RefreshToken.for_user(user)

        data['user'] = user
        data['refresh'] = str(refresh)
        data['access'] = str(refresh.access_token)

        return data
