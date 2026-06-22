FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN pip install --no-cache-dir uv
# headless GL for PyVista/VTK offscreen rendering (pixel-perfect 3D plots)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libxrender1 libxext6 libsm6 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# deps cached as a layer separate from source
COPY pyproject.toml uv.lock* README.md ./
RUN uv sync --no-install-project || true

COPY . .
RUN uv sync

ENV TORCHINDUCTOR_CACHE_DIR=/caches/inductor \
    HF_HOME=/caches/hf \
    PYVISTA_OFF_SCREEN=true \
    PYTHONUNBUFFERED=1
