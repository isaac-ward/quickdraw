# LSAR — Latent-Space Autoregressor (JEPA-style, pluggable anti-collapse)

The latent sibling of `data_space_autoregressor.md` (DSAR). Instead of predicting the observation
directly, LSAR **encodes each observation into a latent, predicts the next latent autoregressively,
and computes its loss in latent space.** This is the JEPA/Dreamer family — but with Dreamer's
generative RSSM machinery dropped for now, leaving a clean deterministic latent and a single axis of
interest: **how to stop the encoder from collapsing.** One LSAR class realizes several anti-collapse
mechanisms (naked, reconstruction, EMA, SIGReg, VICReg) by flipping three orthogonal config knobs.
The stochastic, generative Dreamer-v3 alternative is a separate sibling — `rssm.md`, planned as the
**second milestone** after this family works.

See `high_level.md` for the overview, comparison table, and taxonomy.

## Why a latent model at all

DSAR predicts raw coordinates; its supervision is always grounded in true observations, so it cannot
collapse, but it must model every detail of the observation including dynamically irrelevant ones.
LSAR's bet (the JEPA thesis) is that predicting in a *learned* space lets the model abstract away
unpredictable/irrelevant detail and represent only what matters for the dynamics — potentially better
long-horizon manifold-tracking and extrapolation. The catch: when the prediction target is itself
learned, the trivial minimizer is "encode everything to a constant," which makes prediction perfect
and the representation useless. **Collapse prevention is the entire design problem**, and comparing
its mechanisms fairly is the point of this model.

## Shared ancestor — what LSAR inherits

LSAR is a subclass of `SequenceWorldModel` (specified in `data_space_autoregressor.md`). It inherits
**unchanged**: `enc (6→dz)`, `enc_a (2→d)`, the `TokenStreamFuser`, the causal `Transformer`
(RoPE + sliding window `W`), and the single autoregressive rollout (the `p_tf` teacher-forcing
curriculum + truncated BPTT `detach_every`). The rollout's four hooks are filled as:

| Hook | LSAR |
|---|---|
| `seed_state(o)` | `z = enc(o)` — carried state is the latent |
| `to_token(z, a)` | `action_fuser(z, enc_a(a))` — `z` already in ℝ^d, **not re-encoded** |
| `next_state(h, z_prev)` | `z_prev + predictor(h)` — latent delta |
| `to_obs(z)` | `dec(z)` — decode for obs-space metrics |

**Fairness fact.** DSAR's `enc` also outputs `dz` and the shared fuser up-projects, so this holds at
any `dz`: in teacher-forced mode (`p_tf = 1`) LSAR's tokens are `action_fuser(enc(oₜ), enc_a(aₜ))` —
*bit-identical to DSAR's* — so the encoder, fuser, transformer, and hidden `hₜ` are the same
computation. LSAR and DSAR diverge only at the head (`predictor` vs `delta_head`), the loss space
(latent vs obs), and during free-running rollout (LSAR feeds back the predicted `dz`-latent and never
re-encodes; DSAR feeds back the obs and re-encodes). LSAR's only structural extras are the `dz`-dim
predicted/carried latent and the decoder — exactly the latent-vs-data variable we mean to isolate.

## Two fusion roles — observation fusion vs. action conditioning

The single `TokenStreamFuser` quietly does two conceptually different jobs; making LSAR (and the EMA
boundary) precise means pulling them apart:

- **Role (a) — observation fusion** → `z_t`. Collapse *all observation modalities* into one
  observation latent, **action-free and time-free**: `z_t = obs_fuser(enc(o_vec), enc_img(o_img), …)`.
  `z_t` is "what the world looks like now" — the thing that is **predicted at t+1, EMA-targeted, and
  carried/fed-back** in the latent rollout.
- **Role (b) — action conditioning** → token. Fuse `z_t` with the action into the Transformer's input
  token: `x_t = action_fuser(z_t, enc_a(a_t))`. This feeds the *dynamics*; it is not part of the
  world-state representation.

**Today (single obs modality) they're merged:** role (a) is trivial (one stream → `z = enc(o)`, no
fusion) and the existing `TokenStreamFuser` is really `action_fuser`. When image/lidar arrive, role
(a) becomes a real multi-stream `obs_fuser`. This pins down the EMA boundary exactly: **`enc_ema`
shadows the entire role-(a) stack** (every per-obs-stream encoder + `obs_fuser`) — the action
encoder, `action_fuser`, Transformer, and predictor are the online dynamics model and are **never**
EMA'd. (In I-JEPA terms: role (a) = the EMA'd patch encoder; everything else = the un-EMA'd predictor.)

## Architecture — modules added on top of the ancestor

`d` is the **Transformer hidden width** (every token/hidden vector is in ℝ^d; `d=96` in
`conf/model/base.yaml`). `dz` is the **latent width**, and we set **`dz < d`** (illustratively
`dz ≈ 16`): the latent is a *compact world-state*, smaller than the backbone — essential once
observations are high-dimensional (images), and even now closer to the torus's intrinsic **4-D**
state than `d` is. No standalone `proj_z` is needed — the `dz ↔ d` projections are absorbed by modules
we already have: the **(shared) fuser up-projects** `z (dz)` to token width `d` on the input side; the
**predictor head maps `d → dz`** and the **decoder `dz → 6`** on the output side. `dz` is **fixed and
identical for every model** (DSAR + all LSAR mechanisms) — see *Shared vs. per-mechanism
hyperparameters*.

- **Encoder** `enc: ℝ⁶ → ℝ^{dz}` — inherited; its output *is* the latent `z` (role (a)).
- **Predictor** `predictor: ℝ^d → ℝ^{dz}` — an MLP head reading the hidden `hₜ` and emitting a
  **latent delta**: `ẑ_{t+1} = zₜ + predictor(hₜ)`. Mirror of DSAR's `delta_head` (`d→6`), here `d→dz`.
- **Decoder** `dec: ℝ^{dz} → ℝ⁶` — an MLP that maps a latent back to observation space. **Always
  present** (the shoot-out's metrics — `manifold_distance_error`, MPPI reward, OOD scoring — all live
  in ℝ⁶, so a pure no-decoder JEPA cannot be scored) and **kept small + identical across mechanisms**
  (see *Decoder & obs-space readout*). Whether its gradient reaches `enc` is the `λ_rec` knob below.
- **EMA encoder** `enc_ema` — a non-trainable, exponential-moving-average copy of the **role-(a)
  stack** (`enc` + `obs_fuser`), updated after each optimizer step:
  `enc_ema ← τ·enc_ema + (1−τ)·enc`. Instantiated **only** for the EMA mechanism; absent otherwise.

## Prediction & targets — delta, in latent space

The predictor outputs a residual exactly like DSAR. The prediction loss compares the predicted next
latent against a **target latent** built from the *true* next observation:

```
ẑ_{t+1} = zₜ + predictor(hₜ)
L_pred  = dist( ẑ_{t+1},  sg( target(o_{t+1}) ) )      target(o) ∈ { enc(o), enc_ema(o) }
```

The target always comes from the true next obs `o_{t+1}` (teacher forcing changes only what is *fed
back* during rollout, never the target). `sg(·)` is stop-gradient. Which encoder produces the target
— the online `enc` (gradient flows, symmetric) or the slow `enc_ema` (no gradient, asymmetric) — is
the `target` knob.

**Prediction metric — standardize the target (`pred_metric`).** Raw MSE on `ẑ` is *not* comparable
across mechanisms: SIGReg/VICReg pin the latent scale, but EMA leaves it free, and MSE is partly
gameable by shrinking the target norm. So we apply a **non-affine LayerNorm to the target latent
before the loss** (predict the standardized target) — what JEPA-family methods do in practice;
equivalent to a cosine objective and it removes scale as a confound. Exposed as
`pred_metric ∈ {mse, cosine, normed_mse}` so it can be ablated, but it is **fixed to the same value
for every mechanism** and never varied within the shoot-out. Note the asymmetry with the regularizer:
the prediction loss sees the *standardized* target, while `L_reg` operates on the **raw** `z` (VICReg's
variance term needs the un-normalized scale) — each term sees the representation it was designed for.

## The collapse problem & the three-knob framework

Every LSAR variant minimizes the same objective:

```
L  =  L_pred  +  λ_reg · L_reg  +  λ_dec · L_dec
```

- **`L_pred`** — the latent delta-prediction loss above (always on).
- **`L_reg`** — a distributional regularizer on the batch of latents: `none`, `SIGReg`, or `VICReg`.
- **`L_dec`** — the decoder's reconstruction loss `‖ dec(zₜ) − oₜ ‖²`, **always trained** (the decoder
  must work to read out metrics). The decisive toggle is whether its gradient reaches `enc`:
  - **probe mode** — `L_dec = ‖ dec(sg(zₜ)) − oₜ ‖²` (stop-grad on the latent): the decoder learns to
    read latents into obs *for measurement only*; `enc` is untouched by it.
  - **reconstruction mode** — `L_dec = ‖ dec(zₜ) − oₜ ‖²` (no stop-grad): the decoder's gradient
    shapes `enc`, and reconstruction becomes an anti-collapse force.

A "mechanism" is one setting of three orthogonal knobs — **`target`** (online / EMA),
**`reg`** (none / SIGReg / VICReg), **`recon_grad`** (probe / flows-in). Off the *naked* control
(which collapses), each named mechanism flips exactly one knob:

| Mechanism | `target` | `reg` | `recon_grad` | what stops collapse |
|---|---|---|---|---|
| **naked** (control) | online | none | probe | nothing — expect collapse |
| **reconstruction** | online | none | **flows-in** | decoder gradient keeps `enc` informative |
| **EMA** | **EMA, sg** | none | probe | fast-online / slow-target asymmetry |
| **SIGReg** | online | **SIGReg** | probe | latents → isotropic Gaussian |
| **VICReg** | online | **VICReg** | probe | per-dim variance floor + decorrelation |

### Mechanism detail

- **naked** — `L = L_pred` only (decoder is a stop-grad probe contributing nothing to `enc`). With a
  shared online target and no penalty, the optimizer drives `enc` to a constant: target and
  prediction both become that constant, `L_pred → 0`, representation destroyed. Negative control; its
  job is to demonstrate the failure and bound the others from below.

- **reconstruction** — turn `recon_grad` on. To rebuild varying observations, `enc` must keep latents
  informative; a constant cannot decode to different obs, so collapse is blocked. Classic
  autoencoder-style world model. Risk: the latent is forced to encode *everything* about the obs, not
  just the predictable/dynamics-relevant part — the representation JEPA argues against.

- **EMA** (BYOL / I-JEPA) — targets from the slow `enc_ema` (stop-grad), no penalty term. The fast
  online encoder forever chases a lagging, still-structured target; the collapsed fixed point is
  never stable. One-directional (online predicts target; target never predicts online; target gets no
  gradient). Key hyperparameter: the decay `τ`. Per-step hook: the EMA weight update.

- **SIGReg** (LeJEPA) — single online encoder (symmetric, no EMA, no stop-grad asymmetry, no
  negatives) plus a penalty that pushes the **distribution** of latents toward an isotropic Gaussian,
  estimated by sketched 1-D random projections + a characteristic-function/normality statistic
  (Epps–Pulley style). Single hyperparameter `λ_reg`, linear cost. `L_reg = SIGReg(z_batch)`.

- **VICReg** — single online encoder plus two explicit terms on the **raw** latent batch (not the
  standardized one): a **variance** hinge keeping each dimension's std above a floor
  (`max(0, γ − std_j)`, stops dimensional collapse) and a **covariance** term driving off-diagonal
  entries of the latent covariance to zero (decorrelates dimensions). `L_pred` is the invariance term.
  `L_reg = var(z) + cov(z)`, weighted.

Future mechanisms slot in as new `reg` options (SimSiam = stop-grad + extra predictor head; InfoNCE =
contrastive with negatives drawn from other timesteps/trajectories; Barlow Twins = cross-correlation
matrix → identity) without touching the skeleton.

## Cross-cutting variations

The two optional toggles in `variations.md` — **physical loss** (`λ_phys`) and **noise injection**
(`σ`) — apply to every LSAR mechanism. LSAR's model-specific detail: the physical loss penalizes the
**decoded** prediction `dec(ẑ)` (decoder in the physics path; gradient reaches `enc` only if
`recon_grad` is on), and noise is injected on `o` before `enc` while the latent **target** stays
clean. Neither is an anti-collapse force — they compose with the collapse mechanisms, they don't
replace them.

## Decoder & obs-space readout

The decoder serves two roles: (1) **eval/control readout** — every rollout step's predicted latent
is decoded to ℝ⁶ via `to_obs` so the env metrics and MPPI reward apply unchanged; (2) optionally the
**anti-collapse mechanism** itself (reconstruction mode). In probe mode the latent is stop-gradded
into the decoder so the representation is shaped *only* by `L_pred + L_reg` — keeping the
mechanism-vs-mechanism comparison apples-to-apples. The decoder shares no weights with `enc`.

**Can a strong decoder hide collapse?** Mostly no, with one caveat:
- In **probe mode** the decoder trains on `sg(z)`, so a *fully* collapsed (constant) `z` simply
  cannot be decoded to varying obs → the obs-space metric correctly looks terrible. Probe mode
  **surfaces** total collapse rather than masking it.
- The real risk is **partial** collapse — the latent keeps a little residual structure that an
  over-powered decoder can exploit, flattering the obs metric while most latent dimensions are dead.
- Defense (two layers): (i) keep `dec` **small and byte-identical across all mechanisms** so it is
  never a confound; (ii) **always log latent diagnostics regardless of the obs metric** — per-dim
  variance, **effective rank** of the latent covariance (how many of the `d` dims are actually used),
  off-diagonal covariance mass, and `L_pred` magnitude. The obs metric is the *downstream score*; the
  latent diagnostics are the *is-it-actually-degenerate* check — you need both. **Collapse signature:**
  near-zero `L_pred` + high obs-space error (+ low effective rank). Effective rank is especially
  pointed here because even the compact latent (`dz ≈ 16`) is over-complete for the torus's intrinsic
  **4-D** state (`θ, φ, θ̇, φ̇`), so the live question is *how many of the `dz` dims each mechanism uses*.

## Training

- **Loss:** `L = L_pred + λ_reg·L_reg + λ_dec·L_dec`, all in normalized space (train-split stats from
  `data.md`). `L_pred`/`L_dec` reduced over the rollout horizon and batch.
- **Long-horizon, same curriculum as DSAR (honest comparison).** LSAR rolls **multiple latent steps
  autoregressively** during training, using the *same* `p_tf` teacher-forcing curriculum and
  truncated BPTT (`detach_every`) as DSAR — implemented once in the shared ancestor, operating on
  whatever the carried state is. Teacher forcing feeds back the true latent `zₜ = enc(oₜ)` with prob
  `p_tf`, else the predicted `ẑₜ`; `detach_every` detaches the fed-back latent every N steps.
- **EMA update** (EMA mechanism only): after each optimizer step, `enc_ema ← τ·enc_ema + (1−τ)·enc`.
- Windows from the `delta_timestamps` loader (`P` past, `F` future), as DSAR.

## Rollout / eval

Seed `P` context steps → `enc` → latent buffer; feed the true `action` sequence; predict `ẑ`
autoregressively in **latent space** for 2048 steps; `dec` each predicted latent to ℝ⁶ and score with
the `environment.md` errors (`manifold_distance_error`, `pointwise_error`, `tangent_velocity_error`).
MPPI control reuses the same compiled rollout, decoding latents to obs for the reward — no
control-code change vs DSAR. **Collapse diagnostics** to log alongside the standard metrics: latent
variance / effective rank, off-diagonal covariance mass, and `L_pred` magnitude (a near-zero
`L_pred` with a high obs-space error is the collapse signature). These are logged as a `collapse/`
panel, once per validation epoch — see `logging.md` for the exact plots and captions.

## Shared vs. per-mechanism hyperparameters (fair comparison)

To keep "SIGReg beat EMA" from secretly meaning "SIGReg was tuned harder," **fix everything we can**
and let each mechanism tune **only its own intrinsic knobs**, reported explicitly.

- **Shared and frozen across all mechanisms (and DSAR):** backbone (`d`, depth, heads, window `W`,
  RoPE), `dz` (fixed, `< d`), encoder/predictor/decoder shapes, the `dec` (small, byte-identical),
  optimizer + LR + weight decay + schedule + grad clip, batch/`P`/`F`, rollout horizon, the `p_tf`
  teacher-forcing curriculum, `detach_every`, and `pred_metric`.
- **Unique per mechanism (the only things allowed to vary):**
  - **naked** — none (it's the control).
  - **reconstruction** — `λ_dec` (reconstruction weight; `recon_grad = flows-in`).
  - **EMA** — the decay `τ` (and optionally a predictor-head width, if we add the BYOL predictor).
  - **SIGReg** — `λ_reg` (single hyperparameter) + n sketch projections.
  - **VICReg** — the variance/covariance/invariance weights + variance floor `γ`.

Each mechanism's unique hyperparameters are recorded in its `collapse:` preset and logged with the
run, so any result is reproducible and the shared substrate is auditable.

## Implementation

- `LatentSpaceAR(SequenceWorldModel)`: inherits encoders + fuser + transformer + rollout; adds
  `predictor (d→dz)`, `dec (dz→6)`, and (conditionally) `enc_ema`; fills the four hooks above.
- **Collapse mechanism = composed strategy, not a subclass.** A small `CollapseStrategy` object holds
  the three knobs and exposes: `target_encoder()` (online `enc` or `enc_ema`), `reg_loss(z_batch)`
  (0 / SIGReg / VICReg), `recon_detach` (bool), `on_optimizer_step()` (EMA update or no-op). Selected
  by a `collapse:` block in `conf/model/latent_space_autoregressor.yaml` (alongside `pred_metric`).
  Switching mechanism is a one-line config edit; named presets set the three knobs to the rows above.
- Lightning `LightningModule` is shared with DSAR (it only calls the model's `loss(...)` and the
  shared rollout); it additionally calls `strategy.on_optimizer_step()` and logs the collapse
  diagnostics.

## Resolved decisions

- **Latent prediction metric → standardized target.** Non-affine LayerNorm on the target before
  `L_pred` (cosine-equivalent), fixed identically for all mechanisms; `L_reg` still on raw `z`. See
  *Prediction & targets*.
- **`dz < d`, fixed (not swept).** The latent is a compact world-state (`dz ≈ 16`, smaller than the
  backbone), which is what we'll need for image observations and is closer to the torus's intrinsic
  4-D state. A bottleneck only confounds the comparison if it's *tuned per mechanism* — so we **fix
  one `dz` and share it across DSAR and every LSAR mechanism**. The fuser absorbs the `dz→d`
  up-projection; the predictor (`d→dz`) and decoder (`dz→6`) handle the output side. We **log
  effective rank** to see how many of the `dz` dims each mechanism actually uses.
- **Decoder small + identical + always-on diagnostics.** See *Decoder & obs-space readout*.
- **Fairness protocol.** Shared substrate frozen; only intrinsic per-mechanism knobs vary, logged
  explicitly. See *Shared vs. per-mechanism hyperparameters*.

## Still open

- Exact `pred_metric` default (`cosine` vs `normed_mse`) — pick after a smoke run; whichever, it's the
  same for everyone.
- SIGReg sketch count and VICReg weight ratios — to be swept *within* each mechanism, not across.
