FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface \
    DIFFUSERS_CACHE=/runpod-volume/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
