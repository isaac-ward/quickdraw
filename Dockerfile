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
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=osmesa
#   MUJOCO_GL=osmesa: mujoco/robosuite offscreen rendering, chosen BEFORE mujoco initialises its GL
#   context. Without it `robosuite.make(has_offscreen_renderer=True)` dies looking for a display.

# THE SIMULATOR IS NOT IN THIS IMAGE, deliberately. robosuite + robocasa are editable installs of the
# source checkouts under /caches/sim -- a runtime VOLUME, because they carry ~23 GB of scene assets and
# because robocasa needs robosuite from SOURCE (the PyPI wheel raises `unexpected keyword argument
# 'load_model_on_init'`). /caches is not mounted at build time, so the Dockerfile cannot reach them.
# After ANY rebuild, relink them with:
#
#     ./utils/setup_sim.sh
#
# Everything else -- including the numpy==2.2.5 and mujoco==3.3.1 that robocasa ASSERTS on exactly --
# is declared in pyproject.toml and baked in by the `uv sync` above, so a rebuild reproduces it.
