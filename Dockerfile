FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY alembic.ini .

EXPOSE 8000

# Shell form (not exec-form JSON) so ${PORT} expands: Render's Docker
# runtime injects PORT and requires the container to bind to it; local
# docker-compose doesn't set PORT, so this falls back to 8000 unchanged.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
