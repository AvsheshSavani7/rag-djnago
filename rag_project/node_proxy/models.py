from mongoengine import Document, StringField, IntField, DictField, DateTimeField
from datetime import datetime

class ApiRequestLog(Document):
    """
    Document to log API requests to the Node.js API
    """
    endpoint = StringField(max_length=255, required=True)
    method = StringField(max_length=10, required=True)
    status_code = IntField(required=True)
    request_data = DictField(null=True)   # Equivalent to JSONField
    response_data = DictField(null=True)
    created_at = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'api_request_logs',  # Custom MongoDB collection name
        'ordering': ['-created_at'],
        'verbose_name': 'API Request Log',
        'verbose_name_plural': 'API Request Logs'
    }

    def __str__(self):
        return f"{self.method} {self.endpoint} - {self.status_code}"
