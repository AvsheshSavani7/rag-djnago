#!/usr/bin/env bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Collecting static files..."
cd rag_project
python manage.py collectstatic --noinput || echo "Static files skipped"

echo "Running migrations..."
python manage.py migrate || echo "Migration skipped"

echo "Current directory: $(pwd)"
ls -la