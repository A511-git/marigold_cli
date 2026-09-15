FROM python:3.10-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_HTTP_TIMEOUT=600 \
    UV_NETWORK_ATTEMPTS=5 \
    UV_CONCURRENT_INSTALLS=3

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv /opt/venv

COPY requirements.txt /tmp/requirements.txt
RUN uv pip install \
    --python /opt/venv/bin/python \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip install \
    --python /opt/venv/bin/python \
    -r /tmp/requirements.txt

FROM python:3.10-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY standalone_marigold/ /app/standalone_marigold/
COPY app.py /app/app.py

RUN chmod +x /app/app.py /app/standalone_marigold/infer_panorama.py \
    && ln -s /app/standalone_marigold/infer_panorama.py /usr/local/bin/infer_panorama

RUN mkdir -p /checkpoints /data

ENV CHECKPOINT_DIR=/checkpoints \
    MARIGOLD_OUTPUT=/data/outputs \
    MARIGOLD_CHECKPOINT=huawei-bayerlab/marigold-v2-0 \
    MARIGOLD_DEVICE=cuda \
    MARIGOLD_SPLIT_RESOLUTION=512 \
    MARIGOLD_BATCH_SIZE=1 \
    MARIGOLD_DEPTH_NPY=true

ENTRYPOINT ["python", "app.py"]
CMD []
