# ── Stage: Backend API (FastAPI + ML stack) ───────────────────────────────────
# Built on CUDA 12.4 base for RTX 4050 / any Ampere/Ada GPU
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    libglib2.0-0 libgl1-mesa-glx \    
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
# Install PyTorch first with the correct CUDA index
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "torch==2.6.0+cu124" \
        "torchvision==0.21.0+cu124" \
        --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY src/ ./src/
COPY .env.example .env

# Models are mounted at runtime via docker-compose volume.
# We just create the expected mount point here.
RUN mkdir -p /models

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
