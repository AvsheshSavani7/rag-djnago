from django.contrib import admin
from .models import ApiRequestLog


@admin.register(ApiRequestLog)
class ApiRequestLogAdmin(admin.ModelAdmin):
    list_display = ('method', 'endpoint', 'status_code', 'created_at')
    list_filter = ('method', 'status_code')
    search_fields = ('endpoint', 'request_data', 'response_data')
    readonly_fields = ('method', 'endpoint', 'status_code',
                       'request_data', 'response_data', 'created_at')

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False
