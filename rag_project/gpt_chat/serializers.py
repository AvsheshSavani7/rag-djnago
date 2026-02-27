from rest_framework import serializers
from .models import Thread


class ThreadSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    openai_thread_id = serializers.CharField(read_only=True)
    name = serializers.CharField()
    created_at = serializers.DateTimeField(read_only=True)

    def create(self, validated_data):
        return Thread.objects.create(**validated_data)

class MessageSerializer(serializers.Serializer):
    thread_id = serializers.CharField()
    message = serializers.CharField()
