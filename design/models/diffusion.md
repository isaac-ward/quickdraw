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

### Why a separate field at all (vs "just use the transformer")?

We *do* use the transformer — the field is **conditioned on its output `h`**. They're two different
functions run at different *frequencies*:
- **Transformer** = "read the (state, action) history → summarize what comes next" (`h`). A sequence
  model, run **once** per prediction.
- **Flow field** = "given a noisy guess `x_τ` at noise level `τ`, nudge it toward the answer." Takes the
  *in-progress guess* + `τ` (which the transformer pass doesn't), and runs **K times** per prediction.

The crux is compute frequency: "understand the past" is expensive and needed **once**; "refine the guess"
is needed **K times**. A small dedicated denoiser keeps the K repetitions cheap. Folding denoising into
the transformer means re-running the transformer K times (the history hasn't changed between denoising
steps — wasted work) or adding KV-cache machinery — only worth it when the *denoising itself* is hard,
i.e. high-dim tokenized (image) latents. For a 16-number latent the hard part (the past) is already in
`h`, so a small MLP has ample capacity. Hence: **transformer for context (once) + cheap field for the K
refinement steps.**

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

### Why ε=0 is the fair deterministic prediction

The **prior** is the noise distribution the flow ODE starts from: `ε ~ N(0, I)`, a standard Gaussian in
the latent. To sample, draw `ε` from it and integrate the ODE to a prediction; `ε=0` is the **mean** (the
centre) of that prior. We score the metrics on the `ε=0` path, and that is the fair choice for the
head-to-head shoot-out:

1. **Cost-matched & reproducible** — it is *one* deterministic forward pass, exactly like the deterministic
   models (DSAR/LSAR) produce one prediction. Same input → same output, so the metric is stable and
   golden-testable.
2. **No information leakage** — `ε=0` is the prior's centre, *independent of the target*; it carries zero
   information about the true next state. The prediction comes entirely from the learned field conditioned
   on the past, so it isn't "cheating."
3. **The alternatives are each unfair one way** — scoring a *random* sample would judge diffusion on noise
   (too harsh); scoring the *posterior mean* (averaging N samples) is the MSE-optimal estimate but hands
   diffusion an N×-sampling advantage the deterministic baselines don't get (too generous). `ε=0` is the
   neutral, single-pass middle.

Point estimate and uncertainty are kept separate: `val/pointwise_error` scores the `ε=0` prediction, while
`eval_diffusion/std_of_samples` reports the predicted spread (std of the swarm's final positions).

---

## Visualizing the flow field (the headline diffusion artifact)

The field lives in the latent, but we render it **through the decoder into observation space** on the
torus, so you literally watch a swarm of noise samples flow onto the surface and pinch to the next spot.
At **3 FIXED prediction steps** along episode 0 (fixed so you can watch it sharpen across epochs), rendered
as the full 4-panel atlas (iso + 3 axial) via the `TorusRenderer`:

- **Quiver** (`eval_diffusion/quiver/example_{0,1,2}`, ~2 s mp4): the torus, current dot, action arrow,
  the agent's black history tail AND future path, and a black truth ring (the true next position) stay
  fixed; a **GREY SWARM** of ~16 decoded ODE paths flows from off-surface noise (τ=1) onto the torus (τ=0),
  each leaving a growing tail so the accumulating tails trace the field. (No field-probe arrows and no
  committed/red particle — the swarm itself shows the flow.)
- **Multistep** (`eval_diffusion/quiver_multistep`, ~4 s mp4): the SAME single prediction with the agent
  and its history held FIXED, the swarm integrated with a fine 16-step ODE and played over 4 s so each
  denoising step reads clearly.
- **Shortcut models:** the swarm viz uses a fine many-step integration regardless of the prediction's K
  (the model still has a fine field), so the picture is smooth even when inference is K=1.

How the geometry is computed: each swarm path is `dec(z_t + x_k)[:3]` recorded at every ODE step, from
`dec(z_t + ε)` (decoded noise, **off-surface**) to `dec(z_t + Δẑ)` (on the torus), curved because the
decoder is nonlinear. Swarm = ~16 random `ε` (fixed generator seed → reproducible). Cost is dominated by
the render.

**What it looks like.** A funnel/teardrop: a cloud of decoded-noise points off the torus, ~16 grey threads
arcing down and pinching to a tight knot at the next position, with a black ring at the true next spot.
Beam *width* = confidence (tight = sure; split = multimodal); whether threads land *on* the skin vs float
above = on-manifold vs drift. Across epochs (steps are fixed) you watch the fat scattered funnel narrow
into a clean on-surface beam that bullseyes the ring.

---

## Logging (wandb)

**Losses** (train + val, in the `{tag}/loss/` breakdown):
`{tag}/loss/total`, `{tag}/loss/flow` (flow-matching prediction), `{tag}/loss/pred_obs` (reconstruction
grounding, full-grad), `{tag}/loss/flow_consistency` (shortcut mode only).

**Metrics** (shared with all models, computed on the deterministic ODE sample):
`val/manifold_distance_error`, `val/pointwise_error`, `val/tangent_velocity_error`, `val/obs_error`.

**Collapse diagnostics** (confirm reconstruction grounding holds): `collapse/effective_rank`,
`collapse/latent_norm`, … (reused).

**Diffusion-specific scalars**:
- `eval_diffusion/std_of_samples` — std across the swarm's FINAL (fully-denoised) next-position samples
  (predicted uncertainty / multimodality). No deterministic-model equivalent.
- `eval_diffusion/time/{sample_s, sample_ms_per_euler_step}` — wall time of one K-step deterministic
  prediction (quantifies flow-K6 vs shortcut-K1 inference cost).

Pointwise accuracy is NOT duplicated here — it's the shared `val/pointwise_error` rollout metric.

**Flow-field viz**: `eval_diffusion/quiver/example_{0,1,2}` (~2 s) and `eval_diffusion/quiver_multistep`
(~4 s) mp4s, on top of the usual `eval_control/control_video_0`, `eval_ood_horizon` videos, etc.

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

## Implementation notes (as built — 2026-06-29)

Files: `models/flow.py` (`FlowField`), `models/diffusion.py` (`Diffusion`), `conf/model/diffusion.yaml`,
the `diffusion` branch of `training/setup.build_model`, the `eval_diffusion_field` routine + the
`diffusion/{streamline,quiver}` viz producers in `logging/viz.py`, `smoke/diffusion.py` (greenlight A–H,
all green). Shared code stayed bit-identical (loss_refactor / variations / collapse_validation pass;
viz.py is append-only). Two faithful-to-spirit choices worth recording:

- **The flow-matching LOSS is computed teacher-forced (the "one-step" training) in BOTH plain and
  shortcut modes** — `loss_terms` re-runs the backbone over the true latent window to get the context
  `h` at every transition and regresses the velocity toward the (detached) true transition. The spec's
  *in-rollout* drift exposure is still achieved operationally: `readout` SAMPLES the ODE, so during the
  `p_tf < 1` rollout the reconstruction grounding (`loss_pred_obs`, full-grad) backprops through the
  sampler **and** the AR chain — the model is trained on its own drift via grounding. Shortcut remains
  strictly additive: step-size conditioning on the field + the self-consistency loss term
  (`{tag}/loss/flow_consistency`), which unlocks K=1 sampling. (A dedicated in-rollout *flow* loss, vs.
  the teacher-forced one, is the one place the build is simpler than the doc; revisit only if the
  one-step flow loss underperforms on the long-horizon metric.)
- **Deterministic sample = `eps = 0` (the noise mean)**, not a fixed RNG seed — fully reproducible
  (golden-able) and `to_obs(readout)` equals the committed streamline endpoint by construction.
- `cond: adaln` is **not yet implemented** (raises); `concat` is the default and all tests use it. The
  config also keeps `parameterization: flow` / `path: linear` as the only supported values (asserted).
- One additive contract change: `SequenceWorldModel.loss_terms` gained an optional trailing
  `act_seq=None` (diffusion needs actions to rebuild `h`; DSAR/LSAR ignore it → bit-identical).

Launch (equivalent settings to the variation campaign): `scripts/launch_diffusion.sh` runs plain flow +
shortcut as a 2-run mini-shootout (recon-equivalent backbone, `detach_every=16`, `diffusion_field` viz on).

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

## Greenlight checklist — smoke tests that MUST pass before a training run

All of these are mechanical (tiny canned model + data, like `smoke/loss_refactor.py` /
`smoke/variations.py`). When every box is green, the model is ready for a real training run.

**A. Contract + construction**
- builds and satisfies the `SequenceWorldModel` hook surface (`encode_state`/`to_token`/`readout`/`to_obs`);
  `physical_state` returns a `[...,6]` vector; `one_step_states` exists.
- **contraction + diffusion → raises at construction** (the documented hard error).

**B. Flow loss + grounding**
- flow-matching loss is finite with a finite gradient that reaches the flow field.
- reconstruction grounding: the `pred_obs` full-grad reaches the **encoder**, and `effective_rank`
  stays > 1 over a few canned train steps (the flow alone is not anti-collapse — this proves recon holds).

**C. Sampling (the ODE loop)**
- `readout`/sample returns finite latents of the right shape; the K-step Euler loop runs.
- **deterministic: a fixed seed → byte-identical sample on two calls** (the reproducible committed prediction).
- stochastic: fresh noise → different samples.

**D. Rollout**
- `imagine_eval` over a horizon returns finite obs; the shared `_rollout` works with a sampling `readout`.

**E. Training modes**
- one-step (`shortcut:false`): a train step backprops finitely.
- in-rollout (`shortcut:true`, K=1): sampling-in-rollout backprops finitely and respects `detach_every`.
- shortcut: self-consistency term finite; a 2d step ≈ two d steps within tolerance; K=1 sampling works.

**F. Variations + logging**
- noise + diffusion works; physical + diffusion works (acts on the decoded deterministic sample).
- `{tag}/loss/{flow, pred_obs[, flow_consistency]}` present on **both** train and val.

**G. Visualization**
- streamline renderer → a finite PNG: N paths start off-surface and converge, committed path overlaid,
  true-next marker present.
- quiver renderer → a finite τ-sweep gif/mp4: **the committed particle's position changes across frames**
  (assert frame-to-frame motion — not a static grid merely rescaling) and ends near the prediction; arrow
  scale constant across frames.
- **consistency**: the committed path's endpoint equals the deterministic prediction the metrics are
  computed on (same fixed seed) — the picture matches the numbers.
- **determinism**: with a fixed seed the whole viz is reproducible (identical frames on two calls) →
  golden-testable.
- shortcut: the fine-field viz renders smoothly at K=1; the committed path is the actual K-step leap.
- **metric goal**: `eval_diffusion/pointwise_error` (mean over the 3 viz steps of ‖committed prediction −
  true next position‖, in tube-radii) is logged and **trends down over training** — the quantitative
  companion to "the funnel sharpens onto the ring."

**H. Integration + regression**
- a handful of real `LitWorldModel` steps (one-step AND in-rollout): finite objective, no NaN, all keys
  present, end-to-end backward finite.
- regression: existing `smoke/loss_refactor.py` + `smoke/variations.py` still pass (diffusion is additive).
