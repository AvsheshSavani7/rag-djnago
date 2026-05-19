FROM python:3.12-slim

# System deps for Django + Playwright + Postgres
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libpq-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*



# (Optional) Show version for debug
RUN python --version

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright & browsers
RUN pip install --no-cache-dir playwright && playwright install --with-deps


COPY . /app/

WORKDIR /app/rag_project

RUN ls -al .


EXPOSE 8000

CMD ["sh", "-c", "uvicorn rag_project.asgi:application --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 120"]
