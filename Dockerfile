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


# Add this line to install browsers!
RUN playwright install --with-deps

# Copy Django project code
COPY . /app/

# Collect static files for Django
RUN python manage.py collectstatic --noinput

# Expose Render port
EXPOSE 8000

# Start Django with Gunicorn + UvicornWorker
CMD ["sh", "-c", "uvicorn rag_project.asgi:application --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 120 --reload"]
