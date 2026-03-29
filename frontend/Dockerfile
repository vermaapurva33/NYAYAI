# ── Stage: Frontend (Streamlit — lightweight, no GPU needed) ──────────────────
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# poppler for PDF rendering via PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ui.py .
COPY finalised_logo.png .
COPY .env.example .env

EXPOSE 8501

# BACKEND_URL is set at runtime by docker-compose or by the host environment.
# Default: http://localhost:8000 (overridden for Docker by docker-compose.yml)
CMD ["streamlit", "run", "ui.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
