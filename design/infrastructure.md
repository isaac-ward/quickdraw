# Infrastructure — Dockerized, GPU-ready

Goal: `docker compose up` on any machine with an NVIDIA GPU brings up a ready-to-run environment —
no local Python, CUDA, or dependency setup. Deps are pinned via `uv.lock`; the image tag is pinned;
runs are reproducible.

## Prerequisites (host only)

- NVIDIA driver + Docker Engine + **NVIDIA Container Toolkit** (gives Docker the `nvidia` runtime).
- Nothing else — no Python, CUDA, or uv on the host.

## Files

```
Dockerfile
docker-compose.yml
.dockerignore            # .git, logs/, datasets/, __pycache__, *.ckpt
.env.template             # WANDB_API_KEY=, HUGGINGFACE_TOKEN=
```

## Dockerfile

CUDA + PyTorch base (devel image so `torch.compile`/FlexAttention have `nvcc`/`g++`); `uv` installs
the locked deps; deps cached as a layer separate from source.

```dockerfile
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project        # cached unless deps change
COPY . .
RUN uv sync --frozen
ENV TORCHINDUCTOR_CACHE_DIR=/caches/inductor \
    HF_HOME=/caches/hf
```

Deps to add to `pyproject.toml`: `torch`, `lightning`, `wandb`, `lerobot`, `torchcodec`,
`imageio[ffmpeg]` (plus the existing `numpy scipy matplotlib tqdm hydra-core`).

## docker-compose.yml

```yaml
services:
  app:
    build: .
    runtime: nvidia
    deploy:
      resources: { reservations: { devices: [{ capabilities: [gpu] }] } }
    env_file: .env
    shm_size: "16gb"                 # DataLoader workers need it
    volumes:
      - ./conf:/app/conf             # edit configs without rebuilding
      - ./datasets:/app/datasets     # generated once, reused
      - ./logs:/app/logs             # run folders, checkpoints, media
      - caches:/caches               # persist torch.compile + HF caches across runs
    command: uv run python -m quickdraw.train_world_model
volumes:
  caches:
```

The dataset is generated on first run if `datasets/torus/v1` is absent, then reused.

## Quickstart (this block is the README, ≤5 lines)

```bash
cp .env.template .env          # add WANDB_API_KEY
docker compose up --build     # builds image, generates data, trains + evals on GPU
```

## Steps (the container is idle; shell in with `docker compose exec app bash`)

```bash
uv run python -m quickdraw.data_generation experiment=$RUN
uv run python -m quickdraw.train_world_model          experiment=$RUN              # val + in-dist open-loop
uv run python -m quickdraw.eval_ood       experiment=$RUN checkpoint=logs/train_<ts>_$RUN
uv run python -m quickdraw.eval_control   experiment=$RUN checkpoint=logs/train_<ts>_$RUN
```
Config overrides pass straight through, e.g. `... quickdraw.train_world_model model.depth=6 trainer.max_epochs=200`.
Runs land in `logs/<step>_<timestamp>_<experiment>/`; the evals resolve the train dir's `best.ckpt`.

## Speed & reproducibility notes

- GPU reaches the container via the NVIDIA runtime; `bf16-mixed`, `torch.compile`, FlexAttention,
  and fused AdamW (`training.md`) all run inside it.
- The `caches` volume persists the `torch.compile`/Inductor cache, so recompiles are skipped between
  runs — first run pays the compile cost once.
- Pinned `uv.lock` + pinned base-image tag = byte-reproducible environment. `datasets/` and `logs/`
  are bind-mounted so data and results survive container teardown.
