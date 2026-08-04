# quickdraw

A world-models shoot-out. Needs an NVIDIA GPU + the NVIDIA Container Toolkit.

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

From here, two docs tell you where to go:

- **[docs/workflow.md](docs/workflow.md)** — the full pipeline end-to-end, one `uv run` line per step: generate data → push to the Hub → train the world model → action model → interpret (VLM labeling) → reward model → language control.
- **[docs/byo_environment.md](docs/byo_environment.md)** — run that same pipeline on your *own* environment: any Gymnasium env (zero code) or a first-class `WorldEnv`.

### RoboCasa offline world-model training

The certified Scene 4 LeRobot package is mounted read-only at `/datasets/robocasa`; decoded 128px
eye-in-hand frames are cached under the project-local `.cache/quickdraw-runtime` directory. Verify the
adapter and prescribed model first:

```bash
docker compose exec app uv run python -m quickdraw.smoke.robocasa_dataset /datasets/robocasa
```

The supported from-scratch launch uses one GPU, an episode-disjoint 234/27 train/validation split, train-only
normalization, generic offline validation, and W&B project `quickdraw` / group `robocasa-world-model`:

```bash
docker compose exec app uv run python -m quickdraw.train_world_model \
  --config-name robocasa_world_model \
  data.root=/datasets/robocasa \
  run_summary.problem='Establish the first RoboCasa offline world-model baseline' \
  run_summary.tried='Certified and validated the source trajectory package' \
  run_summary.trying='Train the existing QuickDraw multimodal flow world model on RoboCasa' \
  run_summary.trying_detail='Use 16-D state, 12-D action, one eye-in-hand camera, and an episode holdout' \
  run_summary.rationale='The adapter presents the same canonical transition-window contract as torus data'
```

Set `data.hf_repo=madang6/quickdraw-robocasa-scene4-4h` instead of `data.root` to use the Hugging Face
snapshot path. Do not set both.

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

- [Hugging Face Dataset](https://huggingface.co/datasets/isaac-ronald-ward/torus-world)
- [GitHub](https://github.com/isaac-ward/quickdraw)
