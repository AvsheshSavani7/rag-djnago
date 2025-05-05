from django.db import models
from djongo import models as djongo_models
from bson import ObjectId


def generate_object_id():
    return str(ObjectId())


class Thread(djongo_models.Model):
    _id = djongo_models.CharField(primary_key=True, max_length=24,
                                  default=generate_object_id, editable=False)
    user_id = djongo_models.CharField(max_length=100)
    openai_thread_id = djongo_models.CharField(max_length=100)
    name = djongo_models.CharField(max_length=255)
    created_at = djongo_models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'threads'
        indexes = [
            djongo_models.Index(fields=['user_id']),
        ]

    def __str__(self):
        return f"Thread {self.name} ({self._id})"
