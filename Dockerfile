FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN pip install --no-cache-dir uv
# headless GL for PyVista/VTK offscreen rendering (pixel-perfect 3D plots).
#  - libegl1/libgles2/libglvnd0/libglx0: EGL + GLES + vendor dispatch so VTK's vtkEGLRenderWindow can
#    render ON THE GPU headlessly (no X). With the NVIDIA EGL ICD (provided by the container runtime when
#    NVIDIA_DRIVER_CAPABILITIES includes `graphics`) this is ~10-50x faster than software rendering.
#  - libosmesa6: software OpenGL fallback if EGL is unavailable (PYVISTA_OFF_SCREEN). Slow (CPU llvmpipe).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglx0 libglvnd0 libegl1 libgles2 libxrender1 libxext6 libsm6 libosmesa6 libxcursor1 xvfb \
    && rm -rf /var/lib/apt/lists/*
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
