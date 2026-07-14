# Flow heads — one flow core for the dynamics AND every decoder (+ diffusion forcing)

Status: **design / implementation plan** (2026-07). Extends `design/models/diffusion.md` (the latent
dynamics flow) by (a) generalizing the flow into a **reusable generative-head core** shared by the
dynamics and the modality decoders, (b) adding an opt-in **diffusion-forcing** training regime, and (c)
recording where we sit vs *Interactive World Simulator* (IWS, arXiv 2603.08546) and canonical CTM.
Default behavior is unchanged: `decode_kind: mse` + `diffusion_forcing: off` is **bit-identical to today**.

---

## 0. TL;DR

- Today: dynamics is generative (rectified flow in latent), decode is **deterministic MSE** → blur.
- Plan: make **decode a generative flow head too**, reusing the *same* flow core, per trunk
  (`decode_kind: mse | flow`). One-step via **shortcut** (not CTM — same inference speed, far simpler).
- Add **diffusion forcing** as a gated variation: noise the *context*, not just the prediction target,
  at independent levels → the model trains on noisy history → robust long rollouts.
- **Naming, done right (not grandfathered):** the family is **flow**. Rename `MultiModalDiffusion →
  MultiModalFlow`, `eval_diffusion → eval_flow`, `flow_consistency → shortcut`; log losses as
  `flow/<target>`. Breaking saved dashboards is fine.

---

## 1. Taxonomy — where flow / rectified / shortcut / diffusion / CTM sit

```
      Continuous-time generative transport   (move a noise sample -> the data manifold via an ODE/SDE)
                                   |
          +------------------------+-------------------------+
   DIFFUSION / score-based                              FLOW MATCHING
   forward noising SDE; learn the SCORE;                learn a VELOCITY field v(x,tau) directly
   PF-ODE path is CURVED; needs a noise schedule        along a chosen path; schedule-free
   (DDPM, DDIM, EDM)                                          |
          | few-step accel (distill-free):            +-------+------------------------+
   CONSISTENCY MODELS (CM)                      RECTIFIED FLOW            few-step accel:
   learn a MAP: any noisy point -> x0           straight-line path        SHORTCUT MODELS
          |                                     (what quickdraw uses)     condition v on step size d;
   CTM: generalizes CM ->                                                 self-consistency (our K->1)
   jump to ANY level s (not just 0)
```

Two axes: **(1) base process** — diffusion (curved, learn score, needs a schedule) vs flow matching
(learn velocity directly, schedule-free). **(2) few-step accelerator** (single-model, distillation-free)
— CM/CTM on the diffusion branch vs shortcut on the flow branch. Notes:
- **rectified flow** = flow matching with straight paths. A *base-process* choice, **not** itself a
  few-step trick — it makes few steps easier; true K=1 still needs shortcut.
- **shortcut** = flow-branch accelerator (what we have).
- **consistency / CTM** = diffusion-branch accelerator (the analogue of shortcut). **Not flow.** CTM ⊃ CM
  (CM = CTM with target level `s=0`).
- At the pure-ODE level diffusion's PF-ODE is just a *curved* flow, so "everything is a flow" holds
  abstractly — but the lineages differ in what you learn (score vs velocity) and the path.
- Our 3b idea = import CTM's "jump to any `s`" into **flow** coordinates → still the flow branch.

**Examples per node** (canonical references):
- Diffusion / score: **DDPM** (Ho et al. 2020), **score-SDE/NCSN** (Song et al. 2021), **DDIM** (Song et
  al. 2021, deterministic sampler), **EDM** (Karras et al. 2022).
- Consistency: **Consistency Models** (Song et al. 2023) → **CTM** (Kim et al. 2024).
- Flow matching: **Flow Matching for Generative Modeling** (Lipman et al. 2023), **Stochastic
  Interpolants** (Albergo & Vanden-Eijnden 2023).
- Rectified flow: **"Flow Straight and Fast" / Rectified Flow** (Liu et al. 2023); used in **SD3** (Esser
  et al. 2024). ← quickdraw.
- Shortcut: **One-Step Diffusion via Shortcut Models** (Frans et al. 2024). ← quickdraw's K→1 path.

**Correct names for our code:** the high-level umbrella that contains *both* branches is **continuous
transport** (a.k.a. continuous-time generative transport). Our branch is **flow**. The reusable class is
`TransportHead` (broad enough to host a future diffusion-branch CTM); the objective/log family is `flow`.

### 1.1 Flow vs diffusion — velocity vs score (two sides of one coin)

Both are the **same object**: a continuous-time path between noise and data traversed by a learned vector
field. They differ in *how the path is specified* and *what is parameterized*.

- **Diffusion (score-based):** you *define a forward noising SDE* `dx = f(x,t)dt + g(t)dW`. Those `f,g`
  **are** the noise schedule (VP/DDPM, VE, EDM's σ(t)) — *required*, because they define, at each `t`, the
  noised distribution `p_t = data ⊛ N(0, σ(t)²)`. You learn the **score** `∇ₓ log p_t(x)` (in practice by
  predicting ε). Sampling reverses the SDE (or its probability-flow ODE) along that schedule.
- **Flow matching:** you *directly pick a path* `p_t` (e.g. straight line `x_t=(1−t)x₀+tε`) and learn the
  **velocity** `v(x,t)=E[dx_t/dt | x_t]`. Sampling integrates `dx/dt=v`. **Rectified flow** = the straight
  path → constant velocity `(ε−x₀)`; nothing schedule-like to tune → "schedule-free."

**Why one needs a schedule and the other doesn't:** diffusion is *defined relative to* a forward
corruption process, and "how fast variance grows" is part of that definition (no schedule → no process).
Flow matching folds that choice into the *path*; rectified flow picks the trivial linear path, so there is
no separate variance profile to specify.

**The coin (advisor: yes):** every diffusion model has a **probability-flow ODE**
`dx/dt = f(x,t) − ½g(t)²·score` — which *is* a flow, with a **curved** path set by the schedule and a
velocity built from the score. Conversely a Gaussian-path flow *is* a diffusion; **score and velocity are
related by an invertible linear map** given `(α_t, σ_t)`, so they carry the same information. Concrete
bridge: DDPM predicts **ε**, rectified flow predicts **ε−x₀** — both linear in `(x_t, x₀, ε)`; ε-/x₀-/v-
/score-prediction are all interconvertible. So: **same continuous transport, two coordinate systems
(score vs velocity), two path conventions (curved+scheduled vs straight+schedule-free).**

---

## 2. Architecture: transformer backbone + per-target velocity nets

Dynamics is a **transformer**, not an MLP. The flow field is a small **denoiser** conditioned on the
transformer output. Two functions at two frequencies:

- **Transformer backbone** — reads (state, action) history → per-token context `h`, run **once** per
  prediction. This *is* the dynamics model.
- **Velocity net `v_θ(x_τ, τ, cond, d)`** — "given a noisy guess at level τ, nudge it toward the answer,"
  run **K times** per token during sampling. Small on purpose.

Velocity-net size scales with the **target's internal structure** — the key to reuse:

| Flow head | Target `x₀` | Conditioning `cond` | Velocity net |
|---|---|---|---|
| **dynamics** | next-bag residual `Δz` (token-dim `d`) | backbone `h_state` | **small MLP** (current `FlowField`) |
| **image decode** | image (H×W×3) | predicted image tokens | **ViT** (repurposed decoder) |
| **proprio decode** | 6-D proprio | predicted proprio token | **small MLP** |

The transformer stays the single dynamics model; each head plugs a different velocity net into the
**same** flow algorithm. (This is exactly IWS's structure — one loss reused across decoder + dynamics.)

---

## 3. The reuse refactor: split `FlowField` into (algorithm) + (velocity net)

`FlowField` currently bundles a specific MLP velocity net **and** the flow algorithm. Split them:

- **`flow_core`** (pure algorithm): `loss(velocity_fn, target, cond)`, `sample(velocity_fn, cond, steps)`,
  `shortcut_consistency(velocity_fn, …)`. Operates on any `velocity_fn(x_τ, τ, cond, d) → v`.
- **velocity nets**: token MLP (dynamics), ViT (image decode), proprio MLP.
- **`GenerativeHead`** = (velocity net) + a thin wrapper exposing `.loss(target, cond)` / `.sample(cond,
  steps)` that call `flow_core`. The dynamics head and each decode head are instances.

Result: **one** implementation of flow + shortcut, N heads.

---

## 4. Decode heads: conditioning, target, config

- **Conditioning = the predicted next-step state tokens** (`preds` in `lit.py`, what today's
  `to_obs(src)` consumes). NOT the encoded ground-truth — decoding the *predicted* tokens is what makes
  training match inference. Target = the **clean** future obs.
- **Per-trunk switch `decode_kind: mse | flow`.** `mse` → the exact current path (`ae.decode` +
  `F.mse_loss`) → bit-identical. `flow` → the head's flow loss; render via **1-step** sample (shortcut).
- **Start config: `decode_kind: flow` on EVERY trunk** (image *and* proprio) to confirm the generative
  decode works **cross-trunk** before settling per-trunk defaults. (Proprio-flow is for code uniformity,
  not accuracy — MSE proprio is already near-perfect; one decode contract for all trunks.)
- **`image_fpv` is just a trunk's config name — nothing hardcodes it.** Everything keys off
  `for name, _ in m.layout`, so heads/losses/logs are `flow/<trunk_name>` for whatever trunks exist.
  Future `image_wrist`, `image_overhead`, … each get their own `decode_kind` and namespace automatically.

---

## 5. Losses (every term inside the L2 defined)

Notation: `x₀` clean target; `ε ∼ N(0,I)`; `τ ∈ [0,1]` (0 clean, 1 noise); `x_τ = (1−τ)x₀ + τε`;
`c` = conditioning (dynamics: `h_state`; decode: predicted tokens); `sg[·]` = stop-grad.

**Flow (have, unchanged):**
```
L_flow = E‖ v_θ(x_τ, τ, c) − (ε − x₀) ‖²
         └── predicted velocity ──┘   └ true straight-path velocity u = ε − x₀ ┘
```

**Shortcut (have; the K→1 unlock):** condition `v_θ` on step size `d`; a 2d-step = two d-steps.
```
v₁ = v_θ(x_τ, τ, c, d)                      # first small step's velocity
v₂ = v_θ(x_τ − v₁·d, τ − d, c, d)           # second small step's velocity (from the landed point)
L_shortcut = E‖ v_θ(x_τ, τ, c, 2d) − sg[ ½(v₁ + v₂) ] ‖²
             └── velocity for one big 2d step ┘   └ stop-grad avg of the two d-steps ┘
```
Inference (1 NFE): one eval of `v_θ(ε, 1, c, 1)`; `x̂₀ = ε − v·1`.

**CTM (NOT building now; here for §7).** Learn a finite-jump map `G_θ(x_τ, τ, s, c)` to level `s`:
```
canonical (Kim): L = E‖ G_θ(x_τ,τ,s,c) − sg[ G_θ(G_θ(x_τ,τ,r,c), r, s, c) ] ‖²   (self-bootstrap, τ≥r≥s)
                     └ direct jump τ->s ┘   └ stop-grad two-hop jump τ->r->s ┘     + EMA teacher / DSM / GAN
IWS (data-grounded): L = w(t)·‖ G_θ(x_t,t->s,c) − sg(x_s) ‖²  +  w(s)·‖ G_θ(x_s,s->0,c) − sg(x₀) ‖²
                          └ jump t->s ┘   └ TRUE data noised to s ┘   └ jump s->0 ┘   └ clean data ┘
```

---

## 6. Logging (bit-identical default; breaking renames are fine)

- Dispatch at the `recon` term in `lit.py` on `decode_kind`. `mse` → current path, byte-identical.
- **Renames (do now):** dynamics `loss/flow → loss/flow/latent`, `loss/flow_consistency →
  loss/shortcut/latent`. Decode heads: `loss/flow/<trunk>` (+ `loss/shortcut/<trunk>`). `eval_diffusion →
  eval_flow`; `MultiModalDiffusion → MultiModalFlow`.
- **Eval metrics unchanged** (`val/metric/<trunk>/{mse,l1,psnr,ssim}`, proprio manifold/pointwise/tangent)
  — just fed the **1-step sample** instead of the deterministic decode. So decode-head quality rides the
  **existing** recon eval in `eval_ood_horizon`; no new eval folder. The `eval_flow` swarm viz stays
  **dynamics-only** (a decode-denoising viz is a later nice-to-have).

### 6.1 Rename map — `Transport` is ONLY the head class; everything else is `Flow`

Key distinction: **`Transport`** names the abstract umbrella (continuous transport) → used *only* for the
reusable head class. Our model *is* a flow model, so the model/eval/config rename to **`Flow`**, not
Transport.

| Current | New | Kind |
|---|---|---|
| *(new)* | `TransportHead` | the reusable generative-head class (velocity-net + `flow_core`) — the **only** `Transport` name |
| `MultiModalDiffusion` | `MultiModalFlow` | model class (our branch is flow) |
| `eval_diffusion` (routine, wandb namespace, viz producers) | `eval_flow` | eval |
| `flow_consistency` (loss key) | `shortcut` | loss log key |
| `loss/flow` (dynamics) | `loss/flow/latent` | loss log key |
| `models/flow.py: FlowField` | split → `flow_core` + velocity nets + `TransportHead` | module |
| `conf/model/mm_diffusion.yaml`, `model=mm_diffusion` | `mm_flow.yaml`, `model=mm_flow` | config group |
| `build_model` branch `"diffusion"` | `"flow"` | dispatch |
| `smoke/diffusion.py` | `smoke/flow.py` | smoke |

`design/models/diffusion.md` may stay (dynamics-head specialization) or be renamed `flow_dynamics.md` —
cosmetic, decide later.

---

## 7. Where we sit: canonical CTM vs IWS vs quickdraw

| Axis | Canonical CTM (Kim 2024) | **IWS (this paper)** | **quickdraw (plan)** |
|---|---|---|---|
| What's learned | jump map `G(x,t,s)` | jump map `G(x,t,s)`, **same loss for decoder & dynamics** | velocity `v(x,τ,d)` |
| Regression target | self-bootstrap (two-hop) + EMA teacher, often **+ DSM + GAN** | **ground-truth noised samples** `x_s, x₀` — no teacher, no GAN | flow velocity + shortcut self-consistency (teacher-free) |
| Geometry | EDM/diffusion (curved) | discretized diffusion timesteps | **rectified flow (straight)** |
| Context/history | diffusion forcing (indep per-frame levels) | **diffusion forcing** (`prev_frame_noise_scale`, `uniform`) | clean today; **DF opt-in** (§8) |
| Few-step | anytime 1..N | 1..N via stop-level `s` | K→1 via shortcut `d` |
| Staging | (method) | **3 stages** (enc+dec → dyn → dec-finetune) | **1 stage** (joint enc+dec+dyn) |
| Extra losses | DSM + adversarial common | just weighted MSE (`loss_s + loss_u`) | flow + shortcut MSE |

Takeaway: IWS's edge over canonical CTM is the **data-grounded, teacher/GAN-free** consistency target
reused across decoder & dynamics; its stability lever is **diffusion forcing**. We already have the reuse
(one flow core) and one-stage; the two things we borrow are **generative decode** (§4) and **diffusion
forcing** (§8). We stay in **flow** geometry (straight + shortcut), not diffusion+CTM — same 1-step speed,
simpler training.

---

## 8. Diffusion forcing (opt-in, gated)

### `s` (context) vs `h` (backbone) — the injection site
- `s = z[:, :-1]` — the **clean encoded context bags** (encoder outputs for history frames), the
  **inputs** to the transformer.
- `h = backbone(_to_input(s, act))` — the **transformer output**, the per-token context that conditions
  the flow field. `h` is *derived* from `s` (+ actions).
- **You noise `s`, not `h`.** DF corrupts the context *inputs* at independent levels and *tells* the
  backbone each token's level (a noise-level embedding), so `h` carries it forward. No separate "noise on
  `h`" — `h` inherits it.

### Today vs DF (noise sites)
- **Today:** the only diffusion noise is on the **prediction target** (next-step residual) inside
  `flow.loss`; context `s` and `h` are clean. (The separate `noise_std` input noise is the
  `noise_injection` variation — additive obs noise, unrelated to flow τ.)
- **DF:** additionally noise each **context** frame in `s` at an **independent** level before the
  backbone. At inference, set committed-history level = 0 and denoise only the new frame → the payoff
  (robustness to the model's own rollout drift).

### Exactly where/when the noise goes (the pipeline, super-explicit)

The noise is injected on the **encoded per-modality tokens, PRE-FUSION** (before the transformer). This is
a **training-time augmentation**; at inference nothing is re-noised by default.

**TRAINING** (a window of true data, teacher-forced, one backbone pass):
1. raw obs → encode per modality → per-frame token bag `s` (**pre-fusion**).
2. **inject `observed_token_noise_scale` on the CONTEXT tokens of `s`** (independent levels per the
   `granularity`), and record each token's level.
3. fuse: transformer over `[s (noised, + level embedding), action]` → context `h`.
4. flow loss: predict the next frame's tokens (their own target τ).
→ gradient teaches the model to condition on imperfect history.

**INFERENCE** (autoregressive imagination):
1. P true context frames → encode → tokens, presented at **level 0 (clean)** — **NOT noised**.
2. fuse → `h` → flow-sample the next latent bag.
3. **carry** the predicted latent back as new context — **NOT re-noised**.
4. repeat, **never re-noising**; predicted latents are always presented at level 0.

So: **the noise is applied every TRAINING step to the (true-data) context; at inference it is off.** Your
mental model is right about "carried predictions are never re-noised" — and by default the *initial* true
context isn't noised at inference either. (Optionally, at inference you *could* present a fresh true
context at a small nonzero level to match training — an inference knob, off by default. IWS keeps history
at level 0.) The robustness comes from having *trained* on noisy context, not from re-noising at rollout.

- **Does it matter for flow (vs IWS-diffusion)?** Yes — the benefit is parameterization-agnostic
  ("condition on noisy history" helps any next-step generative model). Diffusion-vs-flow only changes
  *how* you noise (schedule σ vs straight-path τ), not *whether* it helps. Caveat: quickdraw already
  exposes drift via in-rollout training (`p_tf<1` + grounding through the sampler), so DF **partially
  overlaps** — it's a cheaper, more direct way to hit the same robustness.

### Unified noise-injection abstraction (`variations.noise`)
`noise_injection` and diffusion forcing are **two points in one design space**, unified along two axes:

- **Site** — where the noise enters. Two toggleable robustness sites:
  - **`observations_raw`** — raw obs, pre-encode (today's `noise_injection` + the per-modality `noise_std`).
  - **`observations_encoded_pre_fusion`** — encoded per-modality tokens, pre-transformer (diffusion forcing).
  - (Not knobs: the **prediction target's** τ is the flow *objective*, always on; **carried latents at
    inference** is an off-by-default inference option; the **action token** is a possible future site.)
- **`conditioned`** — does the site **tell the predictor its noise level**?
  - `false` = "corrupt-and-hide": plain additive Gaussian, level hidden (pure augmentation; model-agnostic).
  - `true`  = "corrupt-and-tell": flow-level τ + a backbone level-embedding (the model denoises knowing the
    level). **Requires a flow model → the gate.**

```yaml
variations:
  noise:
    # site 1 — CORRUPT-AND-HIDE: additive Gaussian on the raw obs; the level is NOT given to the predictor
    #          (pure augmentation, model-agnostic). This is today's noise_injection + the per-modality noise_std.
    observations_raw:                {scale: 0.0, conditioned: false}
    # site 2 — CORRUPT-AND-TELL: flow-level noise on the encoded pre-fusion tokens; the level IS given to the
    #          predictor (a backbone noise-level embedding) so it denoises knowing the level. This is diffusion
    #          forcing; REQUIRES a flow model (gate). `scale` here == `observed_token_noise_scale`.
    observations_encoded_pre_fusion: {scale: 0.0, conditioned: true, granularity: timestep, observed_only: true}
```
`conditioned` selects the noise *mode* under the hood (corrupt-and-hide = additive Gaussian, level hidden;
corrupt-and-tell = flow-interpolation-to-a-level, level conditioned), so one schema subsumes both
`noise_injection` and the per-modality `noise_std`. Gate: `conditioned: true` on any site ⇒ assert a flow
model at construction. `granularity: timestep | modality` applies to the (token) site.

**Logging when a noise variation is on** — keep it minimal + self-documenting:
- `noise/<site>/scale` (+ `conditioned`, `granularity`) logged as scalars each run, so the run records
  exactly what noise was applied (no silent config).
- No new *quality* metric needed: the payoff of `observations_encoded_pre_fusion` (diffusion forcing) shows
  up in the **existing long-horizon rollout error** (`eval_ood_horizon`) — that curve should drop; and the
  standard recon metrics are unaffected. Optionally log the histogram of sampled context noise levels once
  (a sanity check that the schedule is what you set).

### The rollout it makes robust (how AR works)
Rollout is **latent-space AR — no DSAR, no pixel round-trip.** `_rollout` carries **latent bags** in a
buffer; each step it slides the window and re-runs the **transformer over that latent window** to get
fresh `h`, samples the next latent, appends, slides. "Context recomputed each step" = the causal
transformer re-attends over the current *latent* window (an impl detail; KV-cacheable). Training computes
`h` **once** over the whole window (teacher-forced, parallel).

### Is it a variation? Partly — policy in the suite, mechanism in the model
The **policy** (on/off, scale, granularity) lives in `variations`; the **mechanism** (a per-token
noise-level embedding the backbone consumes + noising `s` in `loss_terms`) is an architecture capability
that must live in `MultiModalFlow`. Unlike `noise_injection`, DF is **not** a pure model-agnostic hook →
hence the hard gate.

### Config + knobs
```yaml
variations:
  diffusion_forcing:
    enabled: false             # off = today (clean context), bit-identical
    observed_token_noise_scale: 0.25   # noise on the PRE-FUSION tokens of INGESTED TRUE observations only
    #                            (never on carried/imagined latents during AR). Fraction of the full noise range;
    #                            0 = clean; 1 = as noisy as the target. (= IWS `prev_frame_noise_scale`.)
    granularity: timestep      # timestep | modality  (token NOT supported — see below). timestep = default.
```
- **`observed_token_noise_scale`** (← IWS `prev_frame_noise_scale`): the name encodes both properties you
  wanted — **`observed`** = applied only to ingested *true observations* (never to carried/imagined latents
  during autoregression), **`token`** = on the *pre-fusion* per-modality tokens (after encode, before the
  transformer). Caps how corrupted the ingested context gets in training; bigger = more drift practiced =
  more robust, harder to fit. (Alt names considered: `ingest_token_noise_scale`, `grounding_token_noise`.)
- **`granularity`** — valid inputs `timestep | modality` (**`token` is NOT supported** — too many tokens,
  rarely worth it). The τ_ctx tensor is `(B, T_ctx, G, 1)` with `G ∈ {1, n_modalities}`, broadcast over the
  bag; the backbone's level-embedding is added per token.
  - **`timestep`** — one level per timestep (all `n_state` tokens of that step's bag share it). Matches
    IWS's single 2D latent. **Default.** ("timestep", not "frame" — the bag has a proprio token too.)
  - **`modality`** — one level per modality-*group* per timestep (proprio token one level, all image patch
    tokens another). The interesting one: enables **cross-modal conditioning** (proprio clean + image noised)
    without per-patch chaos — a capability the heterogeneous token-bag unlocks that IWS's single-map can't.
- **Gate:** `if enabled and not isinstance(model, MultiModalFlow): raise ConfigError("diffusion_forcing
  requires a flow dynamics; got {type}")`. LSAR/DSAR/deterministic → hard error at construction.

---

## 9. Implementation phases

1. **Refactor `FlowField` → `flow_core` + velocity nets + `GenerativeHead`.** No behavior change; dynamics
   uses the token-MLP net. Do the renames (`flow`→`flow/latent`, `flow_consistency`→`shortcut/latent`,
   `MultiModalDiffusion`→`MultiModalFlow`, `eval_diffusion`→`eval_flow`). Existing diffusion smoke green.
   *(Effort: M. Everything hangs off this.)*
2. **Generative decode heads (ALL trunks = flow), shortcut 1-step.** `decode_kind` on the modality +
   `lit.py` recon dispatch; ViT velocity net for images, MLP for proprio; log `flow/<trunk>`
   (+`shortcut/<trunk>`); render via 1-step; eval metrics unchanged; `mse` bit-identical. Confirm it works
   cross-trunk. *(Effort: M.)*
3. **Per-trunk defaults** once validated (likely image→flow, proprio→mse in production). *(Effort: S.)*
4. **Diffusion-forcing variation.** Backbone noise-level embedding; noise `s` in `loss_terms`; inference
   schedule; gate + config. *(Effort: M–L — touches the backbone, not just a loss.)*
5. **(Optional) CTM mode** — only if 1-step image quality disappoints; copy IWS's data-grounded loss.

### File touchpoints
- `models/flow.py` — split into `flow_core` + velocity nets + `GenerativeHead`.
- `models/multimodal.py` — decode heads on modalities; `to_obs`/decode dispatch; DF context-noising +
  backbone level-embedding; gate; class rename `MultiModalDiffusion → MultiModalFlow`.
- `models/vision.py` — ViT decoder gains the `(noised image, τ, cond, d) → velocity` mode.
- `training/lit.py` — `recon` dispatch on `decode_kind`; log-key renames.
- `evaluation/routines.py` + viz — `eval_diffusion → eval_flow`.
- `conf/model/*.yaml` — `decode_kind` per trunk; `conf/config.yaml` `variations.diffusion_forcing`;
  **rename trunk `image_fpv → image`** in every `mm_*.yaml` layout.
- `smoke/` — extend diffusion smoke for decode-flow (all trunks) + DF (mse-bit-identical + gate asserts);
  update `smoke/train_mm.py` + `smoke/mm_loader.py` (they hardcode `"image_fpv"`).

**Trunk rename caveat:** `image_fpv` is keyed generically in code (everything reads `layout` names) BUT
is baked into (a) existing checkpoint state-dict keys `model.modalities.image_fpv.*` and (b) the dataset
loader's `image_head`. So the current `vis_refactor3_diffusion` checkpoint **won't load** under a renamed
trunk — a retrain (or a load-time key-remap) is required. Since flow-heads forces a retrain anyway, bundle
the rename with it; do NOT rename in isolation expecting the old checkpoint to load.

**Future-proofing (stable IDs vs display names):** to make renames free going forward, **decouple a stable
`id` from the display `name`**. Assign each modality a stable `id` (an integer or short slug, set once,
never changed) and key state-dict + dataset stores as `modalities.<id>.*`; `name` becomes purely
cosmetic (config + logging: `flow/<name>`) and renamable without touching checkpoints. Cost: less-readable
state-dict keys. Do this in the flow-heads refactor so `image_fpv → image` (and future multi-image trunks)
never break a checkpoint again.

---

## 10. Decisions (resolved)
- **Doc = `flow_heads.md`.** `diffusion.md` stays the dynamics-head specialization.
- **Family = flow.** Rename `MultiModalDiffusion→MultiModalFlow`, `eval_diffusion→eval_flow`,
  `flow_consistency→shortcut`; loss namespace `flow/<target>`. **Breaking dashboards is fine.**
- **decode_kind: `flow` on all trunks to start** (validate cross-trunk), settle per-trunk defaults after.
- **One-step via shortcut** (not CTM); inference speed identical, training far simpler.
- **Rename trunk `image_fpv → image`**, bundled with the flow-heads retrain (checkpoint keys change; see
  §9 caveat). The per-trunk system handles future multi-image trunks (`image_wrist`, …) by name.
- **DF knobs:** `observed_token_noise_scale` (training-only; noise on the pre-fusion tokens of ingested
  TRUE obs, never re-noised during AR), `granularity: timestep | modality` (timestep default; modality =
  the cross-modal middle; **`token` not supported**).
- **Umbrella term = "continuous transport"**; our branch = flow; reusable class `TransportHead`.
- **Stable `id` vs display `name`** for trunks — decouple so `image_fpv → image` and future renames never
  break checkpoints (§9).
- **Unify noise injection** under `variations.noise` (site × `conditioned`): `observations_raw` (was
  `noise_injection`) + `observations_encoded_pre_fusion` (was `diffusion_forcing`). `conditioned:true` ⇒ flow-model gate.
  Subsumes the per-modality `noise_std` too. (Revises the earlier "keep distinct" — the unified schema is
  cleaner; the two noise *modes* still differ under the hood, selected by `conditioned`.)
- Open: image velocity-net cost = one ViT forward per frame per rollout step (K=1, ~1× the decoder) —
  confirm acceptable before scaling horizons.
