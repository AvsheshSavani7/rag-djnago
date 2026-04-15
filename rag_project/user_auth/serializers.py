from rest_framework import serializers
from user_auth.models import User
from rest_framework_simplejwt.tokens import RefreshToken
from django.utils import timezone
from django.contrib.auth.hashers import make_password, check_password
import jwt
from datetime import datetime, timedelta
from django.conf import settings


class UserRegistrationSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)
    password = serializers.CharField(write_only=True, required=True, style={'input_type': 'password'})
    role = serializers.CharField(default='admin')

    def create(self, validated_data):
        hashed_password = make_password(validated_data['password'])

        user = User(
            email=validated_data['email'],
            password=hashed_password,
            role=validated_data.get('role', 'admin')
        )
        user.save()
        return user


class UserLoginSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)
    password = serializers.CharField(write_only=True, required=True, style={'input_type': 'password'})

    def validate(self, data):
        email = data.get('email')
        password = data.get('password')

        user = User.objects(email=email).first()
        if not user or not check_password(password, user.password):
            raise serializers.ValidationError("Invalid email or password")

        user.last_login = timezone.now()
        user.save()

        access_token_payload = {
            'user_id': str(user._id),
            'email': user.email,
            'role': user.role,
            'exp': datetime.utcnow() + timedelta(hours=1000000),
            'iat': datetime.utcnow()
        }

        refresh_token_payload = {
            'user_id': str(user._id),
            'exp': datetime.utcnow() + timedelta(days=7000),
            'iat': datetime.utcnow()
        }

        access = jwt.encode(access_token_payload, settings.JWT_SECRET_KEY, algorithm='HS256')
        refresh = jwt.encode(refresh_token_payload, settings.JWT_SECRET_KEY, algorithm='HS256')

        data['user'] = user
        data['access'] = access
        data['refresh'] = refresh
        return data

class ChangePasswordSerializer(serializers.Serializer):
    old_password = serializers.CharField(required=True, write_only=True, style={'input_type': 'password'})
    new_password = serializers.CharField(required=True, write_only=True, style={'input_type': 'password'}, min_length=8)

    def validate_old_password(self, value):
        user = self.context['request'].user
        if not check_password(value, user.password):
            raise serializers.ValidationError("Old password is incorrect.")
        return value

    def validate(self, data):
        if data['old_password'] == data['new_password']:
            raise serializers.ValidationError({"new_password": "New password must differ from the old password."})
        return data


class AdminResetPasswordSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)
    new_password = serializers.CharField(required=True, write_only=True, style={'input_type': 'password'}, min_length=8)

    def validate_email(self, value):
        user = User.objects(email=value).first()
        if not user:
            raise serializers.ValidationError("No user found with this email.")
        return value


ROLE_CHOICES = ['admin', 'user']


class ChangeRoleSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)
    role = serializers.ChoiceField(choices=ROLE_CHOICES, required=True)

    def validate_email(self, value):
        user = User.objects(email=value).first()
        if not user:
            raise serializers.ValidationError("No user found with this email.")
        return value


# class UserLoginSerializer(serializers.Serializer):
#     email = serializers.EmailField(required=True)
#     password = serializers.CharField(write_only=True, required=True, style={
#                                      'input_type': 'password'})

#     def validate(self, data):
#         email = data.get('email')
#         password = data.get('password')

#         if email and password:
#             user = authenticate(request=self.context.get(
#                 'request'), username=email, password=password)
#             if not user:
#                 raise serializers.ValidationError(
#                     'Unable to login with provided credentials.')
#         else:
#             raise serializers.ValidationError(
#                 'Must include "email" and "password".')

#         # Update last_login time
#         user.last_login = timezone.now()
#         user.save(update_fields=['last_login'])
#         # Generate JWT token
#         refresh = RefreshToken.for_user(user)

#         data['user'] = user
#         data['refresh'] = str(refresh)
#         data['access'] = str(refresh.access_token)

#         return data
