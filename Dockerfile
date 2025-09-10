# Use the official Playwright image (includes Chromium, Firefox, WebKit + deps)
FROM mcr.microsoft.com/playwright/python:v1.40.0-focal

# Set work directory
WORKDIR /app

# Install system dependencies required for Django + psycopg2/Postgres
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies first (better caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy Django project code
COPY . /app/

# Collect static files for Django
RUN python manage.py collectstatic --noinput

# Expose Render port
EXPOSE 8000

# Start Django with Gunicorn + UvicornWorker
CMD ["gunicorn", "rag_project.asgi:application", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
