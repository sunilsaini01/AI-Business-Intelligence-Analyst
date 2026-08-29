FROM python:3.11-slim

WORKDIR /frontend

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Shell form so ${PORT} expands (Render's Docker runtime injects PORT and
# requires the container to bind to it; local docker-compose doesn't set
# PORT, so this falls back to 8501 unchanged) and headless mode is
# explicit rather than relying on Streamlit's no-TTY auto-detection.
CMD exec streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true
