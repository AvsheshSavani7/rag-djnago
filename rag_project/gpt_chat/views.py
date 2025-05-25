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
from mongoengine.errors import DoesNotExist


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
                thread = client.beta.threads.create()

                # Save to MongoDB
                thread_data = serializer.validated_data
                thread_data['openai_thread_id'] = thread.id
                thread_instance = Thread.objects.create(**thread_data)

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
            threads = Thread.objects.filter(
                user_id=user_id).order_by('-created_at')
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
            # Get the thread from MongoDB
            thread = Thread.objects.get(openai_thread_id=thread_id)

            # Delete thread from OpenAI
            try:
                client.beta.threads.delete(thread_id=thread_id)
            except Exception as e:
                logger.warning(f"Error deleting OpenAI thread: {str(e)}")
                # Continue with MongoDB deletion even if OpenAI deletion fails

            # Delete thread from MongoDB using _id
            Thread.objects.filter(_id=thread._id).delete()

            return Response({
                'message': 'Thread deleted successfully',
                'thread_id': thread_id
            }, status=status.HTTP_200_OK)

        except DoesNotExist:
            return Response({"error": "Thread not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error deleting thread: {str(e)}")
            return Response({"error": "Failed to delete thread"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



class MessageListView(APIView):
    """
    API endpoint to get messages for a thread
    """

    def get(self, request, thread_id):
        try:
            thread = Thread.objects.get(openai_thread_id=thread_id)

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
        except DoesNotExist:
            return Response({"error": "Thread not found"}, status=status.HTTP_404_NOT_FOUND)
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
                thread = Thread.objects.get(openai_thread_id=thread_id)

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

            except DoesNotExist:
                return Response({"error": "Thread not found"}, status=status.HTTP_404_NOT_FOUND)
            except Exception as e:
                logger.error(f"Error sending message: {str(e)}")
                return Response({"error": "Failed to send message"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
