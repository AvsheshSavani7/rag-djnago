from rest_framework import serializers


class ThreadSerializer(serializers.Serializer):
    _id = serializers.CharField(read_only=True)
    user_id = serializers.CharField()
    openai_thread_id = serializers.CharField(read_only=True)
    name = serializers.CharField()
    created_at = serializers.DateTimeField(read_only=True)


class MessageSerializer(serializers.Serializer):
    thread_id = serializers.CharField()
    message = serializers.CharField()
