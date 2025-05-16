from django.shortcuts import render
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings
from openai import OpenAI
from .models import Thread
from .serializers import ThreadSerializer, MessageSerializer
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
assistant_id = os.getenv('OPENAI_ASSISTANT_ID')


def format_message(message):
    """Format OpenAI message to a more readable structure"""
    content = message.content[0].text.value if message.content else ""
    return {
        "id": message.id,
        "role": message.role,
        "content": content,
        "created_at": message.created_at,
        "thread_id": message.thread_id
    }


class ThreadCreateView(APIView):
    """
    API endpoint to create a new thread
    """

    def post(self, request):
        serializer = ThreadSerializer(data=request.data)
        if serializer.is_valid():
            try:
                # Create thread in OpenAI
                openai_thread = client.beta.threads.create()

                # Save to MongoDB
                thread_data = serializer.validated_data
                thread_data['openai_thread_id'] = openai_thread.id

                # Create the Thread instance and save it
                thread_instance = Thread(**thread_data)
                thread_instance.save()

                response_data = ThreadSerializer(thread_instance).data
                return Response(response_data, status=status.HTTP_201_CREATED)
            except Exception as e:
                logger.error(f"Error creating thread: {str(e)}")
                return Response({"error": "Failed to create thread"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ThreadListView(APIView):
    """
    API endpoint to get all threads for a user
    """

    def get(self, request, user_id):
        try:
            threads = Thread.find_by_user_id(user_id)
            # Sort threads by created_at (descending)
            threads.sort(key=lambda x: x.created_at, reverse=True)
            serializer = ThreadSerializer(threads, many=True)
            return Response({
                'user_id': user_id,
                'threads': serializer.data
            })
        except Exception as e:
            logger.error(f"Error fetching threads: {str(e)}")
            return Response({"error": "Failed to fetch threads"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ThreadDeleteView(APIView):
    """
    API endpoint to delete a thread
    """

    def delete(self, request, thread_id):
        try:
            # Find all threads with this OpenAI thread ID
            threads = Thread.find_by_query({'openai_thread_id': thread_id})

            if not threads:
                return Response({"error": "Thread not found"}, status=status.HTTP_404_NOT_FOUND)

            thread = threads[0]  # Get the first matching thread

            # Delete thread from OpenAI
            try:
                client.beta.threads.delete(thread_id=thread_id)
            except Exception as e:
                logger.warning(f"Error deleting OpenAI thread: {str(e)}")
                # Continue with MongoDB deletion even if OpenAI deletion fails

            # Delete thread from MongoDB
            Thread.delete_by_id(thread._id)

            return Response({
                'message': 'Thread deleted successfully',
                'thread_id': thread_id
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error deleting thread: {str(e)}")
            return Response({"error": "Failed to delete thread"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class MessageListView(APIView):
    """
    API endpoint to get messages for a thread
    """

    def get(self, request, thread_id):
        try:
            # Find threads with this OpenAI thread ID
            threads = Thread.find_by_query({'openai_thread_id': thread_id})

            if not threads:
                return Response({"error": "Thread not found"}, status=status.HTTP_404_NOT_FOUND)

            thread = threads[0]  # Get the first matching thread

            # Get query parameters
            limit = request.query_params.get('limit')
            order = request.query_params.get(
                'order', 'desc')  # Default to descending order

            # Get messages from OpenAI
            messages = client.beta.threads.messages.list(
                thread_id=thread_id,
                limit=int(limit) if limit else None,
                order=order
            )
            messages.data.reverse()
            print(messages.data)

            # Format messages
            formatted_messages = [format_message(msg) for msg in messages.data]

            # # Sort messages based on order parameter
            # if order == 'asc':
            #     formatted_messages.sort(key=lambda x: x['created_at'])
            # else:  # desc
            #     formatted_messages.sort(
            #         key=lambda x: x['created_at'], reverse=True)

            return Response({
                'user_id': thread.user_id,
                'thread_id': thread_id,
                'messages': formatted_messages
            })
        except Exception as e:
            logger.error(f"Error fetching messages: {str(e)}")
            return Response({"error": "Failed to fetch messages"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class MessageSendView(APIView):
    """
    API endpoint to send a message to a thread
    """

    def post(self, request):
        serializer = MessageSerializer(data=request.data)
        if serializer.is_valid():
            thread_id = serializer.validated_data['thread_id']
            message = serializer.validated_data['message']

            try:
                # Find threads with this OpenAI thread ID
                threads = Thread.find_by_query({'openai_thread_id': thread_id})

                if not threads:
                    return Response({"error": "Thread not found"}, status=status.HTTP_404_NOT_FOUND)

                thread = threads[0]  # Get the first matching thread

                # Add message to thread
                client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=message
                )

                # Run the assistant
                run = client.beta.threads.runs.create(
                    thread_id=thread_id,
                    assistant_id=assistant_id
                )

                # Wait for completion
                while True:
                    run_status = client.beta.threads.runs.retrieve(
                        thread_id=thread_id,
                        run_id=run.id
                    )
                    if run_status.status == 'completed':
                        break

                # Get the latest messages
                messages = client.beta.threads.messages.list(
                    thread_id=thread_id)

                # Format messages
                formatted_messages = [format_message(
                    msg) for msg in messages.data]

                return Response({
                    'user_id': thread.user_id,
                    'thread_id': thread_id,
                    'messages': formatted_messages
                })

            except Exception as e:
                logger.error(f"Error sending message: {str(e)}")
                return Response({"error": "Failed to send message"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
