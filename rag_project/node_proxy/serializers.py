from rest_framework import serializers
from .models import ApiRequestLog


class ApiRequestLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = ApiRequestLog
        fields = '__all__'
        read_only_fields = fields
