from rest_framework import serializers
from .models import ApiRequestLog


class ApiRequestLogSerializer(serializers.Serializer):
    endpoint = serializers.CharField()
    method = serializers.CharField()
    status_code = serializers.IntegerField()
    request_data = serializers.DictField(required=False, allow_null=True)
    response_data = serializers.DictField(required=False, allow_null=True)
    created_at = serializers.DateTimeField(read_only=True)

    def create(self, validated_data):
        return ApiRequestLog.objects.create(**validated_data)

    def update(self, instance, validated_data):
        # This log document should be read-only, but implement if needed
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        return instance
