from django.apps import AppConfig


class NodeProxyConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "node_proxy"
    verbose_name = "Node.js API Proxy"
