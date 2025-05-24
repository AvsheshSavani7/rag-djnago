from django.http import JsonResponse
from django.views import View

class ExampleAPIView(View):
    def get(self, request):
        return JsonResponse({"message": "API is working fine in production!"})