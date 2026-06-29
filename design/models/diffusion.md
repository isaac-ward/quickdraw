# Diffusion — latent autoregressive flow-matching world model

A fourth model class for the shoot-out, alongside **DSAR** (data-space) and **LSAR** (latent-space,
deterministic). Where the deterministic models predict a *point estimate* of the next state, the
diffusion model predicts the **distribution** of the next state by learning a **velocity field** that
transports noise onto the next-state manifold. Sampling that field is, literally, a vector field that
pushes a noisy guess onto the surface of the torus — the project's central theme, made generative.

It is **latent diffusion**: the field lives entirely in the latent `z`; the encoder/decoder own the
modality (a vector MLP today; a CNN/ViT encoder + image decoder later). The diffusion never touches
pixels, so it is modality-agnostic and image-ready by construction.

---

## Where it sits (see high_level.md)

Subclass of `SequenceWorldModel`. It shares the action encoder + causal transformer backbone + the ONE
autoregressive rollout, and reuses three of the four state hooks. The only structural changes are:

| hook / piece        | DSAR / LSAR (deterministic)            | Diffusion                                            |
|---------------------|----------------------------------------|-----------------------------------------------------|
| `encode_state`      | obs → z                                | **reused** (obs → z, LayerNorm-bounded; see below)  |
| `to_token`          | (z, a) → token                         | **reused**                                          |
| transformer         | → context hidden `h`                    | **reused** (h conditions the flow field)            |
| `readout(h, z_t)`   | `z_t + predictor(h)` (point)           | **flow SAMPLE**: integrate the ODE → `z_t + Δẑ`     |
| `to_obs`            | dec(z)                                 | **reused** (decoder; also the grounding — below)    |
| `loss_terms`        | `pred_latent` (+ reg)                  | **flow-matching** loss                              |

Because `readout` keeps its signature `(h, prev_state) → next_state`, the shared `_rollout` and
`imagine_eval` work **unchanged** — they just call a `readout` that samples instead of one that adds a
delta.

---

## Grounding — the flow is NOT anti-collapse on its own

The flow-matching loss is computed in latent space. A collapsed latent (every `z` equal → `Δz ≈ 0`) is
*trivially* flow-predictable (loss → 0) yet decodes to garbage. So the flow **admits the collapse
solution** and cannot be the only objective.

The latent must therefore be obs-grounded. The default (and simplest) grounding is **reconstruction**:
the decoder decodes the predicted rollout and the unified `loss_pred_obs` flows **full-grad into the
encoder** (`pred_obs_in_loss = True`, exactly as `LSAR-reconstruction`). This forces `z` to carry the
real state, which prevents collapse. So the standalone model is:

> **reconstruction-grounded encoder + a flow prediction head.**

This is precisely the "prediction head" framing: **flow** is the prediction mechanism; **grounding**
(reconstruction by default) is the orthogonal axis. Keeping reconstruction means we are *not* adding a
new collapse mechanism — we reuse the one that already works.

---

## Rectified flow (the parameterization)

We use **rectified flow** (the modern, schedule-free formulation; score/DDPM are a curved-path special
case, DDIM is just their ODE sampler — we build flow only). We diffuse the **residual** `Δz` (mirrors
the deterministic residual head; the field then nudges the *current* latent onto the next-state surface).

```
target      Δz = z_{t+1} − z_t                      (residual; "absolute" z_{t+1} is a config option)
noise       ε ∼ N(0, I)
flow time   τ ∈ [0, 1]
path        x_τ = (1 − τ)·Δz + τ·ε                  (linear / rectified — straight ⇒ few steps)
velocity    u  = dx_τ/dτ = ε − Δz                   (constant along a straight path)
LOSS        L_flow = E_{τ, ε}  ‖ v_θ(x_τ, τ, h) − (ε − Δz) ‖²
SAMPLE      integrate dx/dτ = v_θ(x, τ, h) from τ=1 (x=ε) → τ=0, Euler with K steps;  Δẑ = x(0)
            next state = z_t + Δẑ
```

`v_θ` is the velocity field — the "vector field". The data `Δz` lies on (the residual to) the valid
next-state manifold, so the integrated field projects a noise sample onto that manifold.

**Carried-latent LayerNorm.** `encode_state` and `readout` LayerNorm the carried latent (as
`LSAR-reconstruction` does), so `z` lives on the unit sphere — a clean, bounded surface for the flow to
target, and it stops magnitude drift over the rollout.

---

## The flow field `v_θ` (the only new module)

**Decision: the flow field is a separate small MLP** (the transformer is unchanged — it stays the
deterministic context encoder → `h`). A unified transformer-as-denoiser is a future image-era option,
not what we build (see "Could the field just *be* the transformer?" below).

A small MLP, **shared across all rollout steps**:

```
v_θ : [ x_τ ‖ emb(τ) ‖ h ]  →  dz-dim velocity
```

- It is **not** the decoder (the decoder is `dz → 6`; the field is `dz → dz`). The decoder is reused for
  grounding + metrics + the physical loss; the field is the one genuinely new net. Width mirrors
  `dec_hidden`; `emb(τ)` is a small Fourier/MLP time embedding.
- **Conditioning `h → v_θ`**: `concat` (default — glue `h` and `emb(τ)` onto the input) or `adaln`
  (per-block FiLM/AdaLN-zero modulation, the DiT/SD3 default; stronger but more params — an upgrade if
  concat underperforms).

### Blocks, backbone vs diffusion-specific

| Block | Function | Shared or diffusion-specific |
|---|---|---|
| obs encoder `enc` | obs → latent `z` (dz=16); the modality boundary (MLP now, CNN/ViT later) | **shared** |
| action encoder `act_enc` | action → d-dim embedding | **shared** |
| fuser `to_token` | (z, action) → step token (d) | **shared** |
| transformer backbone | depth × [LN → sliding-window RoPE FlexAttention → LN → MLP] → context `h` (d) | **shared** |
| time embedding | flow time `τ` → embedding (Fourier + small MLP) | **diffusion** |
| **flow field `v_θ`** | MLP on `[x_τ ‖ emb(τ) ‖ h]` → latent velocity (dz→dz) | **diffusion** (only new net) |
| decoder `dec` (`to_obs`) | latent `z` → obs (6 / image); reconstruction grounding **and** render | **shared** |

### Could the field just *be* the transformer? (yes — but only for images)

The field could be unified with the backbone — a "diffusion transformer": append the noisy next-latent
as a token, condition the blocks on `τ` (AdaLN), read the velocity off that token, and KV-cache the
(fixed) past so each of the K denoising steps is one cheap single-token pass. That's the modern, more
powerful design, **but the deciding factor is latent dimensionality.** For a **16-d** latent the hard
part — understanding the past + action — is already done by the transformer into `h`; denoising 16
numbers given `h` needs almost no capacity, so a small MLP is plenty *and* keeps sampling cheap (K
tiny-MLP evals, which matters because the sampler runs K×T times per rollout). The unified
transformer-as-denoiser only earns its cost when the **latent is high-dimensional and tokenized — i.e.
images**, where the denoiser needs attention. So: **separate small field now; the unified diffusion
transformer is the natural upgrade when we move to image latents** — same modality-generality story as
the rest of the model.

---

## Training

- **One-step (default; `shortcut: false`).** Teacher-forced denoising: given the true context → `h`,
  take the true `Δz`, noise it at a random `τ`, regress the velocity. One `v_θ` forward per position —
  as cheap as the deterministic head. *Caveat:* the model only sees true contexts, so it doesn't
  practice recovering from its own rollout drift (a real disadvantage on the long-horizon metric).
- **In-rollout (`shortcut: true`, `train_rollout: auto`).** Sample `Δẑ` each rollout step (K=1 via the
  shortcut), feed it forward, so the model sees its own drift — apples-to-apples with DSAR/LSAR. Uses
  the **same `detach_every` truncation** as the other models. Cost: backprop runs through the sampler
  *and* the AR chain, so it is O(T·K) — only sane at **K=1**, which only the shortcut provides. Hence the
  coupling: in-rollout ⇔ shortcut.

### Shortcut (an option on top of plain flow — build plain flow first)

"One Step Diffusion via Shortcut Models" (Frans et al. 2024). Condition `v_θ` on the **step size `d`**
too, and add a **self-consistency** term: one step of size `2d` must equal two chained steps of size
`d` (bootstrap target from the model's own outputs, stop-grad). This teaches accurate large (even
single) steps, so **one trained model samples at any K** — chosen at inference, no distillation. We
build/validate **plain flow first**; shortcut is a strictly additive mode (extra loss term + step-size
conditioning) that unlocks K=1 (and therefore cheap in-rollout training). We will compare both.

---

## Sampling / evaluation

- **K Euler steps** per AR step: 4–8 for plain flow, 1 for shortcut.
- **Deterministic ODE from a FIXED starting noise** for the headline metrics (MDE/OOD/control) — the
  *same* noise every eval (a fixed seed, or `ε=0` the noise-mean), **not a fresh random draw**. So it's a
  single reproducible "best guess" (same input → same prediction each epoch), directly comparable to the
  deterministic models and golden-testable. This *same* fixed-seed path is the one highlighted in the
  streamline viz, so the picture shows exactly the prediction the metrics are computed on.
- Optionally render a few **stochastic** samples (fresh noise) in the *video only*, to visualize the
  predicted spread / multimodality.

---

## Visualizing the flow field (the headline diffusion artifact)

The field lives in the latent, but we render it **through the decoder into observation space** on the
torus, so you can literally see the agent at a point on its arc with the field sweeping onto the surface
and pinching to the next spot. The convergence is strongest at high noise level `τ` (far from the answer
it pulls hard; near it, fine adjustments) and is **action-conditioned** (via `h`) — it points where the
agent goes *given its action*. At **~4 fixed prediction steps** along episode 0 (fixed so you can watch
it sharpen across training epochs), we render two complementary views, each a **PNG via the
`TorusRenderer`** with the extra geometry drawn on top of the usual torus + current-dot + action-arrow:

- **Streamlines** (`diffusion/streamline/example_{0,1,2,3}`): sample N≈16 noise vectors, integrate each
  through `v_θ` (conditioned on this step's `h`, `z_t`), **decode every integration step** → N paths that
  start off-manifold and flow *onto* the torus, converging to the predicted next position. Overlay the
  **deterministic-ODE sample** (the single committed prediction used for metrics) as a highlighted path,
  and mark the **true next position** — so you see the cloud of possibilities, the one it commits to, and
  whether it aims true.
- **Quiver** (`diffusion/quiver/example_{0,1,2,3}`): a small grid of candidate positions on/just-off the
  torus near the agent; at each, evaluate `v_θ` at a fixed mid-`τ` and map the latent velocity to an
  **obs-space arrow** by finite-difference through the decoder (`dec(z+δv) − dec(z)`). The arrows
  converge on the next spot — the literal "field pointing where to go."

Cost is dominated by the *render*, not the flow (the field + decoder are tiny; a frame is a few hundred
evals). 4 steps × 2 views is cheap enough to log every eval epoch.

**What it looks like.** A funnel/teardrop: a diffuse cloud of decoded-noise points floating off the
torus, ~16 threads arcing down and pinching to a tight knot at the next position (comets homing on a
landing site), the bright committed path down its spine, and a ring at the true next spot. Beam *width*
= confidence (tight = sure; split = multimodal); whether threads land *on* the skin vs float above it
= on-manifold vs drift (the failure mode, made visible). Across epochs (steps are fixed) you watch the
fat scattered funnel narrow into a clean on-surface beam that bullseyes the ring.

---

## Logging (wandb)

**Losses** (train + val, in the `{tag}/loss/` breakdown):
`{tag}/loss/total`, `{tag}/loss/flow` (flow-matching prediction), `{tag}/loss/pred_obs` (reconstruction
grounding, full-grad), `{tag}/loss/flow_consistency` (shortcut mode only).

**Metrics** (shared with all models, computed on the deterministic ODE sample):
`val/manifold_distance_error`, `val/pointwise_error`, `val/tangent_velocity_error`, `val/obs_error`.

**Collapse diagnostics** (confirm reconstruction grounding holds): `collapse/effective_rank`,
`collapse/latent_norm`, … (reused).

**Diffusion-specific scalar**: `diffusion/sample_spread` — std across stochastic samples of the predicted
next-position (a read on predicted uncertainty / multimodality).

**Flow-field PNGs**: `diffusion/streamline/example_{0,1,2,3}` and `diffusion/quiver/example_{0,1,2,3}`
(above), on top of the usual `eval_control/control_video_0`, `eval_ood_horizon` videos, etc.

---

## Composability with the variations (design/models/variations.md)

- **noise_injection** — works unchanged (perturbs obs inputs before `encode_state`).
- **physical_loss** — works (acts on the decoded ODE sample via the frozen-decoder `physical_state`).
- **contraction** — **HARD ERROR.** Combining contraction with a diffusion model must `assert`/raise at
  construction: the contraction penalty differentiates the one-step state map, which for diffusion runs
  through the ODE sampler — out of scope. The two are mutually exclusive by design.

---

## Modality / images

Because the field and the flow loss live in the latent, moving to images changes **only** `enc`/`dec`:
encoder → CNN/ViT, decoder → image decoder, and the reconstruction grounding becomes image
reconstruction. `v_θ`, the flow loss, the sampler, and the rollout are untouched. This is the core
reason to diffuse the latent rather than pixels.

---

## Implementation shape (head-ready, instantiated standalone)

Encapsulate the flow as a self-contained `FlowField` component owning `sample()` (the ODE integration)
and `loss()` (flow-matching, + shortcut self-consistency when on). The standalone diffusion model is a
thin `SequenceWorldModel` subclass (reconstruction-grounded) that plugs `FlowField` into `readout` and
`loss_terms`. Promoting it later to a **swappable prediction head** (a `deterministic | flow` axis,
orthogonal to the collapse/grounding axis) is then a small refactor — but for now **only the standalone
model runs in the shoot-out** (no extra head × grounding cells).

---

## Config

```yaml
diffusion:
  parameterization: flow      # SUPPORTED: "flow" only (subsumes ddpm/ddim). assert otherwise.
  path: linear                # SUPPORTED: "linear" (rectified) only, for now.
  predict: residual           # SUPPORTED: "residual" (diffuse Δz; default) | "absolute" (diffuse z_{t+1}).
  cond: concat                # SUPPORTED: "concat" (default, simple) | "adaln" (per-layer, stronger upgrade).
  shortcut: false             # false = plain flow (needs sampling_steps 4–8).
                              # true  = step-size-conditioned net + self-consistency loss -> K=1 works.
  sampling_steps: 6           # K Euler steps / AR step at inference. off -> 4–8 ; on -> 1 (any K). >= 1.
  train_rollout: auto         # SUPPORTED: "auto" (= shortcut: off->one-step, on->in-rollout) | true | false.
  stochastic_eval: false      # false = deterministic ODE from fixed noise (reproducible metrics).
                              # true  = fresh-noise samples (video spread only).
  time_sampling: uniform      # SUPPORTED: "uniform" (default) | "logit_normal" (SD3-style, mid-noise weighted).
  # detach_every: inherited from model.detach_every (same truncation as the other models).
  # GROUNDING: reconstruction (pred_obs full-grad) — required; the flow alone is not anti-collapse.
  # HARD ERROR: variations.contraction + diffusion -> assert at construction.
```

---

## Smoke tests (to write with the implementation)

- flow-matching loss is finite with a finite gradient; the sampler returns finite latents.
- both training paths run: one-step (shortcut off) and in-rollout (shortcut on, K=1).
- deterministic-ODE sampling is reproducible (same seed → identical sample); stochastic differs.
- shortcut self-consistency: a 2d step ≈ two d steps within tolerance.
- reconstruction grounding prevents collapse (effective_rank stays > 1 over training).
- contraction + diffusion raises at construction.
