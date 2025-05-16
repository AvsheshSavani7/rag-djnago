#!/usr/bin/env bash
# Exit on error
set -o errexit

# Move to the project directory
cd rag_project

# Apply migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic --no-input 