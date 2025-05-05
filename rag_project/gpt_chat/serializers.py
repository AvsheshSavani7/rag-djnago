from rest_framework import serializers
from .models import Thread


class ThreadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Thread
        fields = ['user_id', 'openai_thread_id', 'name', 'created_at']
        read_only_fields = ['openai_thread_id', 'created_at']


class MessageSerializer(serializers.Serializer):
    thread_id = serializers.CharField()
    message = serializers.CharField()
