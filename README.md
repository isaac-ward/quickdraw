# quickdraw

A world-models shoot-out: long-horizon consistency as staying on a torus data-manifold, with an MPPI control eval. Needs an NVIDIA GPU + the NVIDIA Container Toolkit.

## Usage

Bring the container up and shell in:

```bash
# copy the keys template and set WANDB_API_KEY
cp .env.template .env
# build the image and start the container (idle)
docker compose up -d --build
# confirm the GPU is visible inside the container
docker compose exec app python -c "import torch; print(torch.cuda.is_available())"
# shell in
docker compose exec app bash
```

Then, inside the container, run the steps in order (`RUN` is your experiment name). Everything for a
run lands in `logs/<step>_<timestamp>_<experiment>/` — dataset, plots, checkpoints, all in one place.

The dataset is always **local**: `data.root` points at a run directory on disk. Either generate one
(step 1a) or pull the published dataset from the Hub (step 1b) — both give you a local `data.root`.

```bash
# 1a. generate the dataset locally + sample plots/summary (all under one logs/ run dir; prints the path)
uv run python -m quickdraw.data_generation experiment=$RUN
#     -> set DATA=logs/data_generation_<ts>_$RUN

# 1b. ...OR pull the published dataset from the Hub into a local dir, and point DATA at it
uv run huggingface-cli download isaac-ronald-ward/quickdraw-torus --repo-type dataset --local-dir data/torus
#     -> set DATA=data/torus   (a split loads as LeRobotDataset("torus/<split>", root="$DATA/<split>"))

# 2. train: train/val + the subscribed in-loop evals every N epochs   ->   set CKPT=logs/train_<ts>_$RUN
uv run python -m quickdraw.train_world_model            experiment=$RUN data.root=$DATA

# 3. post-hoc evals at the best checkpoint (each writes its own logs/ run dir of plots/videos)
uv run python -m quickdraw.eval_ood_horizon experiment=$RUN data.root=$DATA checkpoint=$CKPT  # long-horizon open-loop rollout
uv run python -m quickdraw.eval_ood         experiment=$RUN data.root=$DATA checkpoint=$CKPT  # OOD splits (visual/geometric/dynamics)
uv run python -m quickdraw.eval_control     experiment=$RUN data.root=$DATA checkpoint=$CKPT  # dual-MPPI control (oracle vs learned)
uv run python -m quickdraw.eval_manifold    experiment=$RUN data.root=$DATA checkpoint=$CKPT  # recovered-manifold projections
uv run python -m quickdraw.eval_diffusion   experiment=$RUN data.root=$DATA checkpoint=$CKPT  # denoising viz (diffusion models only)
uv run python -m quickdraw.eval_interpret   experiment=$RUN data.root=$DATA checkpoint=$CKPT  # VLM-labeled latent manifolds (needs OPENAI_API_KEY; vision only)

# (optional) push a generated dataset run to the Hub as one repo
uv run python -m quickdraw.push_to_hub data.root=$DATA +hub.name=quickdraw-torus +hub.private=true
```

Override any Hydra field on the CLI.

## Supported methods

All share one modality-parameterized space-time transformer spine; they differ along independent axes —
where they predict, how each step is produced, how latent collapse is prevented, and what shapes training.

- **Autoregression mechanisms** — where the recurrence lives:
  - **Data-space autoregression (DSAR)** — predict the next *observation* and re-encode it each step; the rollout lives in data space.
  - **Latent-space autoregression (LSAR)** — predict the next *latent* (`z_t + Δ`) and roll forward in latent space. Cheaper and more expressive, but needs a collapse-prevention strategy:
    - **Regularizers**
      - [SIGReg](https://arxiv.org/abs/2511.08544) (LeJEPA) — push the batch latent distribution toward an isotropic Gaussian via random-projection sketches.
      - [VICReg](https://arxiv.org/abs/2105.04906) — per-dimension variance hinge (a std floor) plus covariance decorrelation.
      - Data-space reconstruction penalty — decode the latents back to observations, grounding the encoder so it can't collapse.
      - Naked — no collapse term (baseline / probe; expected to degenerate).
      - [EMA](https://arxiv.org/abs/2006.07733) (BYOL / I-JEPA style) — target is a slow exponential-moving-average, stop-gradient copy of the encoder, with an online predictor.

- **Prediction mechanisms** — how each step's next state/latent is produced:
  - Direct — a single deterministic readout maps the transformer output to the next state/latent (a point prediction; used by DSAR and LSAR today).
  - Implicit distribution — sample the next latent from a distribution defined *implicitly* by an iterative sampler (no closed-form density), rather than a parametric form:
    - Rectified-flow diffusion (a flow-matching method) — learn a velocity field that transports noise to the next latent along near-straight paths; sample by integrating the ODE over K steps.
    - Shortcut models (few-step) — a self-consistency objective lets the same flow sample in one-to-few steps instead of K, for fast rollouts (the `shortcut` flag).
  - Parametric distribution, [Dreamer](https://arxiv.org/abs/2301.04104)-style *(planned, not yet implemented)* — the spine parameterizes an *explicit* distribution over the next latent (a categorical or Gaussian), which is then **sampled** rather than emitted as a point. The latent state is just carried forward (no separate recurrent hidden state); trained by the likelihood of the true next latent, and composable with the same collapse regularizers as LSAR (no Dreamer-style KL required).
  
- **Loss variations** — orthogonal train-time shaping terms, composable with any of the above (all off by default):
  - Physical — penalize predictions that leave the torus surface or violate its tangent constraint (optionally a kinematic `v = dp/dt` continuity term).
  - Contraction — hinge on the largest singular value of the one-step Jacobian, encouraging contractive, drift-resistant dynamics.
  - Noise injection — add Gaussian noise to the normalized observation inputs during training (robustness to the model's own rollout error).

## Links

- [Hugging Face Dataset](https://huggingface.co/datasets/isaac-ronald-ward/quickdraw-torus)
- [GitHub](https://github.com/isaac-ward/quickdraw)
