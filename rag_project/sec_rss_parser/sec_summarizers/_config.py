"""
Django-compatible config for sec_summarizers.
Resolves ANTHROPIC_API_KEY from Django settings when running inside the app,
otherwise from os.environ (e.g. from .env via load_dotenv in standalone use).
Same pattern as document_analyzer_new.py using os.environ.get("OPENAI_API_KEY").
"""

import os


def get_anthropic_api_key() -> str | None:
    """
    Return ANTHROPIC_API_KEY for Claude summarizers.
    - When running under Django: uses settings.ANTHROPIC_API_KEY (from env).
    - When standalone: uses os.environ (caller should load_dotenv if needed).
    """
    try:
        from django.conf import settings
        key = getattr(settings, "ANTHROPIC_API_KEY", None)
        if key:
            return key
    except Exception:
        pass
    return (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY_TEST")
    )
