import os
import sys
import django
from pathlib import Path


def init_django():
    # Get the parent directory (project root)
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent

    # Add the project root to the Python path
    sys.path.insert(0, str(parent_dir))

    # Set up Django settings
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')

    if not django.conf.settings.configured:
        django.setup()
