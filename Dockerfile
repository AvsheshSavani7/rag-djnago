# Use the official Playwright image (includes Chromium, Firefox, WebKit + deps)
FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

RUN python --version

# Install system dependencies required for Django + psycopg2/Postgres
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory at /app
WORKDIR /app

# Copy Python dependencies first (for Docker layer cache)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install --with-deps

# Copy the whole repo into /app
COPY . /app/

# Move into Django project directory
WORKDIR /app/rag_project

# Optional: Debug
RUN ls -al .

# Collect static files for Django
RUN python manage.py collectstatic --noinput

# Expose Render port
EXPOSE 8000

# Start Django with Gunicorn + UvicornWorker
CMD ["sh", "-c", "uvicorn rag_project.asgi:application --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 120 --reload"]
