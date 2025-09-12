from django.apps import AppConfig


class ProxyProcessorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "proxy_processor"
    verbose_name = "SEC Proxy Processor"

    def ready(self):
        """
        Import signal handlers when the app is ready.
        """
        pass
