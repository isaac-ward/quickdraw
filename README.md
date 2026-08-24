# quickdraw

A world-models shoot-out. Needs an NVIDIA GPU + the NVIDIA Container Toolkit.

## Usage

Bring the container up and shell in:

```bash
# copy the keys template and set the key values
cp .env.template .env
# build the image and start the container (idle)
docker compose up -d --build
# confirm the GPU is visible inside the container
docker compose exec app python -c "import torch; print(torch.cuda.is_available())"
# shell in
docker compose exec app bash
```

From here, the fastest start is the **wizard** — point your AI assistant (Claude, Codex, …) at it and it
drives the whole setup:

- **[wizard/prompt.md](wizard/prompt.md)** — *"look at wizard/prompt.md and talk me through the choices for `<my dataset or env>`"*. It interviews you about your data/env and model and **resolves all the dims for you** — either **inspecting a HuggingFace dataset** (cameras, dims, fps) if you bring *recorded data*, or wiring up **`data_generation`** from the simulator if you bring an *environment* (a Gym env or a full `WorldEnv`). It applies the documented learnings, verifies the setup (`check_dataset` for the data, `model_summary` for the model), then compiles a runnable pipeline script to `wizard/scripts/`.

Or read the docs directly:

- **[docs/workflow.md](docs/workflow.md)** — the full pipeline end-to-end, one `uv run` line per step: generate data → push to the Hub → train the world model → action model → interpret (VLM labeling) → reward model → language control.
- **[docs/byo.md](docs/byo.md)** — run that same pipeline on your *own* environment: recorded data, a required-contract env (or any Gymnasium env, zero code), or a full `WorldEnv`.

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
  - Parametric distribution, [DreamerV3](https://arxiv.org/abs/2301.04104)-style *(in active implementation)* — the spine parameterizes an *explicit* distribution over the next latent (a categorical or Gaussian), which is then **sampled** rather than emitted as a point. The latent state is just carried forward (no separate recurrent hidden state); trained by the likelihood of the true next latent, and composable with the same collapse regularizers as LSAR (no Dreamer-style KL required).
  
- **Loss variations** — orthogonal train-time shaping terms, composable with any of the above (all off by default):
  - Physical — penalize predictions that leave the torus surface or violate its tangent constraint (optionally a kinematic `v = dp/dt` continuity term).
  - Contraction — hinge on the largest singular value of the one-step Jacobian, encouraging contractive, drift-resistant dynamics.
  - Noise injection — add Gaussian noise to the normalized observation inputs during training (robustness to the model's own rollout error).

## Links

- [Hugging Face Dataset](https://huggingface.co/datasets/isaac-ronald-ward/torus-world)
- [GitHub](https://github.com/isaac-ward/quickdraw)

## Feature compatibility

**Prediction mechanisms** (table columns). Written out on first use; the abbreviation is used in the tables.

*Generic autoregression* — a single deterministic readout predicts the next state/latent as a point:
- **Generic Data-Space Autoregression (DSAR)** — predict the next *observation* and re-encode it each step; the rollout lives in data space.
- **Generic Latent-Space Autoregression (LSAR)** — predict the next *latent* as a point and roll forward in latent space.

*Implicit distribution* — sample the next latent from a distribution defined only through an iterative solver (no closed-form density):
- **Flow** — a [rectified-flow](https://arxiv.org/abs/2209.03003) / [flow-matching](https://arxiv.org/abs/2210.02747) velocity field, integrated over K steps (one step with the shortcut objective) to draw the next latent.

*Parametric distribution* — the spine outputs the explicit parameters of a distribution over the next latent, which is then sampled. **In active implementation** — design in [design/models/probabilistic_heads.md](design/models/probabilistic_heads.md); the cells below are intended behavior, verified against code as each head lands:
- **Categorical** — discrete one-hot latent per [DreamerV3](https://arxiv.org/abs/2301.04104).
- **Gaussian-KL (G-KL)** — stochastic Gaussian latent, KL-trained ([PlaNet](https://arxiv.org/abs/1811.04551) / [Dreamer](https://arxiv.org/abs/1912.01603)).
- **Gaussian-NLL (G-NLL)** — deterministic latent, predicted Gaussian, NLL-trained ([Ward et al. 2026](https://arxiv.org/abs/2603.06987)).
- **Multivariate normal (MVN)** — a Gaussian with full or low-rank covariance; KL- or NLL-trained like G-KL / G-NLL.

**Collapse strategies** (top-table rows):
- **Naked** — no collapse term (baseline / probe; expected to degenerate).
- **Reconstruction** — decode the latent back to observations, grounding the encoder.
- **EMA** — slow exponential-moving-average, stop-gradient target encoder ([BYOL](https://arxiv.org/abs/2006.07733) / [I-JEPA](https://arxiv.org/abs/2301.08243)).
- **SIGReg** — push the batch latent toward an isotropic Gaussian via random-projection sketches ([LeJEPA](https://arxiv.org/abs/2511.08544)).
- **VICReg** — per-dimension variance hinge + covariance decorrelation ([VICReg](https://arxiv.org/abs/2105.04906)).
- **Prior/posterior KL** — pull a predicted prior toward the encoder's posterior; KL + reconstruction is the anti-collapse ([DreamerV3](https://arxiv.org/abs/2301.04104)).

**Symbols.** ✓ works · ✗ mutually exclusive / disallowed (fails fast) · ⚠ works with a caveat · — not applicable / not wired (see footnote).

### Prediction mechanism × collapse strategy

| collapse | DSAR | LSAR | Flow | Categorical | G-KL | G-NLL | MVN |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Naked | — | ✓ | — | ✗ | ✗ | ⚠ | ✗ |
| Reconstruction | — | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| EMA | — | ✓ | — | ✗ | ✗ | ✗ | ✗ |
| SIGReg ᵃ | — | ⚠ | — | ✗ | ✗ | ✓ | ✓ |
| VICReg ᵃ | — | ⚠ | — | ✗ | ✗ | ✓ | ✓ |
| Prior/posterior KL | — | — | — | ✓ | ✓ | — | ✓ |

**Notes — collapse table.**
- Collapse strategies are needed only by the **point-prediction latent heads**. Generic-LSAR predicts the next latent as a point, which can trivially collapse — encode every observation to a constant and predict that constant: the loss is zero while the representation is empty. So LSAR **needs** an explicit strategy.
- **DSAR (—):** grounded inherently by its data-space re-encode (a constant latent can't reproduce the diverse next observation); no strategy is wired (`setup.py` DSAR branch never calls `make_collapse` — note a `collapse=` config on `mm_dsar`/`mm_flow` is currently *silently ignored*, a fail-fast TODO).
- **Flow (—):** the anti-collapse is the **decode reconstruction**, not the residual objective — a collapsed constant latent makes `Δz≡0`, which the flow head satisfies *trivially*, so grounding comes from the recon/round-trip (and the `dynamics_detach_encoder` escape hatch), not from `Δz`. No separate strategy is wired.
- ᵃ **LSAR × SIGReg/VICReg ⚠:** these require `model.latent_norm=none` (the model raises otherwise), but the default latent-norm is `layernorm` — so the naive combo fails fast; set `latent_norm=none` to use them.
- **Prior/posterior KL is not a swappable knob** — it needs a posterior `q(z|o)` + prior `p(z|h)` *pair*, which only the parametric heads have; adding one to a point head *is* building the G-KL/Categorical head. It is also dual-purpose (it is how the dynamics learns to predict). So the posterior heads (Categorical, G-KL, MVN-KL) take **Reconstruction + Prior/posterior KL together** and reject the rest (hence **Naked ✗** for them — a probe-only mode makes no sense without recon; EMA contradicts the posterior-as-target design; SIGReg/VICReg on embedded one-hots regulate a statistic the embedding table owns). **G-NLL** is deterministic (no posterior/KL), so it behaves LSAR-like: Naked ⚠ (probe), SIGReg/VICReg compose. **MVN** rows assume KL-mode for the KL/Naked rows and NLL-mode for the SIGReg/VICReg rows.

### Prediction mechanism × everything else

| feature | DSAR | LSAR | Flow | Categorical | G-KL | G-NLL | MVN |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Physical loss | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Contraction ¹ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Noise injection (input) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Diffusion forcing (pre-fusion) ² | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Shortcut (few-step) ² | — | — | ✓ | — | — | — | — |
| latent-norm (none/affine/layernorm) ³ | ✓ | ✓ | ✓ | ⚠ | ⚠ | ✓ | ⚠ |
| relative-position encoding | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Fourier features (proprio/action) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `obs_keep` subsetting | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `subsample` stride | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| proprio (vector) modality | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| image — bespoke ViT-AE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| image — pretrained TAESD + adapters | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| encode arch (vit/conv) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| decode kind (mse/flow) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| decode arch (vit/unet) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| decode shortcut | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| multi-image ⁹ | ⚠ | ⚠ | ⚠ | ⚠ | ⚠ | ⚠ | ⚠ |
| `num_tokens` (image compression) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| round-trip anchor (`latent_loss_weight`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| action-token conditioning | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| action Fourier / squash | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| action-flow head (MPPI prior) ² | — | — | ✓ | — | — | — | — |
| physics prior ⁴ | ⚠ | ⚠ | ✓ | ⚠ | ⚠ | ⚠ | ⚠ |
| physics-chained training ⁴ | ⚠ | ⚠ | ✓ | ⚠ | ⚠ | ⚠ | ⚠ |
| KV-cache rollout ⁵ | ⚠ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `compile_rollout` ⁶ | ✓ | ✓ | ✓ | ⚠ | ⚠ | ⚠ | ⚠ |
| `grad_checkpoint` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| teacher-forcing schedule (`p_tf`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `detach_every` BPTT truncation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| shared-encode fast path ⁷ | — | — | ✓ | — | — | — | — |
| sliding-attention `window` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| autobatch | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| MPPI control ⁸ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| action model (post-hoc) ¹⁰ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| reward / caption-contrastive model ⁸ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| language steering ⁸ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| interpretability (VLM latent labeling) ⁸ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| OOD / anomaly eval ⁸ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**Notes — compatibility table.** (`—` = not applicable *or* not wired — the footnote says which.)
1. **Contraction** ✗ wherever the one-step map runs through a sampler or ODE (no double-backward Jacobian) — supported only on DSAR/LSAR. Also mutually exclusive with `compile_rollout`.
2. **Flow-specific.** Diffusion forcing and the [shortcut](https://arxiv.org/abs/2410.12557) objective are *intrinsically* flow mechanisms (they noise the flow field's context / self-consistency-train the flow ODE), so they don't exist for other heads (`—`/✗). The action-flow prior is a separate flow head wired only on Flow; portable but not wired elsewhere (`—`), and currently *silently ignored* on other `model.name`s rather than fail-fast — a guard is a TODO.
3. **latent-norm.** All heads support none/affine/layernorm, but affine requires a frozen pretrained trunk (raises otherwise, any model). ⚠ on the stochastic-latent heads (Categorical/G-KL/MVN): the affine "0 dB round-trip" identity is void under a sampled bottleneck; G-NLL keeps it (deterministic latent).
4. **Physics prior** — a **repo-specific** residual-dynamics hook (spacecraft-docking), not a general feature. The `dynamics_prior` hook is *wired for every model* via the shared config, but it is **implemented and verified only on Flow**; ⚠ elsewhere = unverified, and a categorical bottleneck complicates the absolute-unit physics.
5. **KV-cache** ⚠ DSAR: the backbone caches, but DSAR's per-step decode + re-encode dominates cost, so the speedup is small; full benefit on the latent-AR variants.
6. **`compile_rollout`** ⚠ on the parametric heads: Gumbel-max / reparameterized sampling should be compile-safe but is unvalidated — opt-in with a parity A/B.
7. **shared-encode fast path.** A `lit._step` optimization (encode each window once, reuse for context + loss + round-trip); gated on the flow head (`—` = not wired elsewhere, not impossible; identical result, a little more compute).
8. **Downstream stacks.** MPPI, interpretability, and OOD eval consume *decoded* observations; the reward and language-steering heads score the *rolled latent bag decode-free*. Both paths are base-class, so all are prediction-mechanism-agnostic for the implemented models — but the decode-free reward/language transfer to a categorical embedded-one-hot latent is not yet designed.
9. **multi-image** ⚠: the modality registry accepts N image streams, but the data loader is hard-coded to a single image head (`setup.py` `window_loaders` / `MMWindowLoader`), so a second image modality is not supported end-to-end yet. Independent of prediction mechanism.
10. **action model (post-hoc)** ✗ on non-Flow: `train_action_model` asserts a flow world model (it *is* the action-flow head, trained post-hoc), so it fails fast elsewhere.
