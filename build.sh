#!/usr/bin/env bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Collecting static files..."
cd rag_project
python manage.py collectstatic --noinput

echo "Running migrations..."
python manage.py migrate