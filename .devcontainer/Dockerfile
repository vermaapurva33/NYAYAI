# -------------------------------
# Base Image (CUDA 11.8 + cuDNN 8 Devel)
# -------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Consolidated Library Paths
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# -------------------------------
# System Dependencies
# -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    poppler-utils \
    libleptonica-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Fix: Symlink cuDNN so Paddle finds the generic .so name
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/local/cuda/lib64/libcudnn.so

# Make python default
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspaces/NyayAI

# -------------------------------
# Python Dependencies
# -------------------------------
COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir paddlepaddle-gpu==2.6.1.post118 \
       -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html \
    && pip install --no-cache-dir -r requirements.txt

# -------------------------------
# Application Code
# -------------------------------
COPY . .

CMD ["sleep", "infinity"]