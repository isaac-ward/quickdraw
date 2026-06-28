# Variations — cross-cutting shoot-out axis

Three **orthogonal toggles** that can be turned on for **any** model (DSAR, any LSAR mechanism, RSSM)
and combined freely. They are *not* model classes or collapse mechanisms — they are an extra axis of
the shoot-out: "does adding this help, across the board?" All default **off**, so a baseline run is
unaffected, and all are set in a `variations:` config block:

```yaml
variations:
  noise_injection: { std: 0.0 }                                   # σ_noise; 0 = off
  physical_loss:   { weight: 0.0 }                                # λ_phys;  0 = off
  contraction:     { weight: 0.0, target: 1.02, power_iters: 2,   # λ_contract, τ; weight 0 = off
                     n_sample_steps: 4, wrt: last_state }
```

Run the shoot-out with each toggle off/on (and optionally combined) to read its marginal effect on
long-horizon manifold-tracking, OOD, and control. All three are train-time shaping only — run-time
inference (eval / rollout / control) is unaffected (with at most optional diagnostic logging).

## Noise injection (`noise_injection.std = σ_noise`)

Add i.i.d. Gaussian noise to the **observation inputs** during training, for off-manifold robustness.

- **Where — one site, same for every model:** perturb the raw observation *before the shared encoder
  `enc`* (the role-(a) stack): `õ = o + ε`, `ε ~ N(0, σ_noise²)`, in **normalized** space (so `σ_noise`
  is in std units), resampled each step. This is a single injection point — every model consumes `o`
  through the same `enc`, so DSAR's "before tokenization", LSAR's "before `enc`", and RSSM's "before
  the posterior" are all *the same place*, named in each model's local vocabulary. (Downstream differs
  — DSAR re-encodes raw obs each rollout step, LSAR carries the encoded `z`, RSSM feeds a posterior —
  but the noise enters at the same point.)
- **Multimodal (future):** with image/lidar streams the raw inputs have very different natures
  (ImageNet-normalized pixels vs. the 6-vector), so a single scalar `σ_noise` won't fit all — expect a
  **per-stream `σ_noise`** when those modalities arrive. The config's single `std` is the vector-only case.
- **What is *not* noised:** prediction **targets** stay clean (DSAR's true next `o`, LSAR's
  `enc(o_{t+1})` / `enc_ema(o_{t+1})`), and the model's own fed-back predictions are left as-is. So the
  model learns *noisy/off-manifold input → clean on-manifold target* — i.e. to **denoise and project
  back onto the manifold**.
- **Training only — eval is always clean.** Noise is an input augmentation for learning off-manifold
  recovery; every reported metric is measured on unperturbed observations.
- **Why it should help:** autoregressive rollout inevitably feeds the model slightly off-manifold
  inputs (its own imperfect predictions), but teacher-forced training only ever shows on-manifold
  inputs — a train/rollout distribution shift that compounds into long-horizon drift. Injecting input
  noise exposes the model to off-manifold inputs and teaches recovery, attacking compounding error.
  (Same idea as denoising autoencoders / DART / scheduled-sampling, applied uniformly across models.)
- **Composability:** independent of the `p_tf` curriculum (it adds off-manifold exposure even at
  `p_tf = 1`, cheaply) and of every collapse mechanism. When used as a shoot-out axis, `σ_noise` is a
  single fixed value shared across the models being compared.
- **Logging:** `σ_noise` recorded in config/provenance; its effect shows up in the standard eval metrics.

## Physical loss (`physical_loss.weight = λ_phys`)

A physics-informed penalty added to the loss, `L += λ_phys · L_phys`, that penalizes predictions
leaving the world's physics: the predicted position should lie **on the torus** and the velocity
should be **tangent** to it. It reuses the **exact** `environments/torus.py` functions (`signed_dist`,
`normal`, `angles_from_point`) — already verified to be **pure, batched, differentiable torch** with
gradients flowing — so the *same* code is the eval metric and the training penalty (no separate impl).

Definition (one canonical form — squared **distance**, not the residual):

```
ô = (p̂; ṗ̂)                                                       denormalize to physical units first
d_off(p̂) = signed_dist(p̂) = √((ρ̂ − R)² + ẑ²) − r ,  ρ̂ = √(x̂²+ŷ²)   normal distance off the surface
v_off(p̂, ṗ̂) = ṗ̂ · n̂(p̂)                                            velocity's normal component
L_phys = ( d_off / r )²  +  ( v_off / v_scale )²                    both terms DIMENSIONLESS
```

- `d_off` is exactly `torus.py:signed_dist` (the env's `manifold_distance_error` is its `.abs()`);
  squaring `signed_dist` gives a smooth `dist²` with no kink at 0. It is **not** the implicit residual
  `ρ² − r²` (a mistaken earlier formula).
- **Normalized throughout (fixes the unit mismatch).** The raw terms have different physical units
  (length vs velocity), so each is **non-dimensionalized**: distance ÷ tube radius `r`, normal velocity
  ÷ a characteristic speed `v_scale` (the velocity scale from the `data.md` norm stats). Both are then
  dimensionless and O(1), so a single `λ_phys` balances them and they're directly comparable; log the
  dimensionless `d_off/r` and `v_off/v_scale`.
- Compute the geometry in **physical** space (denormalize first) — per-dim normalization distorts the
  torus, so the torus distance must *not* be computed in per-dim-normalized coordinates. The
  non-dimensionalization above is what makes the scalars normalized.

- **Per-model application:**
  - **DSAR** penalizes its prediction `ô` directly — the physics term is *exact*.
  - **LSAR** penalizes the **decoded** prediction `dec(ẑ)`, so the penalty is **mediated by the
    decoder**: (1) it's only as accurate as `dec`, so ramp `λ_phys` in after a short warmup once the
    decode is trustworthy; (2) it threads `dec` into the gradient path (`L_phys → dec → ẑ → predictor
    → enc`). To keep `dec` a clean, passive readout (trained only by reconstruction/probe), **freeze
    `dec`'s parameters in the physics term** — its Jacobian still carries the gradient back to the
    prediction/latent, but its own weights aren't updated by physics. *(Implementation: a functional
    call with detached `dec` params, or a frozen `dec` copy — **not** a plain `.detach()` on the
    output, which would also kill the gradient to `ẑ`.)* This is not a blocker and not a confound
    (applied uniformly across mechanisms); it's the one place the latent path genuinely differs from DSAR.
  - **RSSM** penalizes its decoded prediction likewise (milestone 2).
- **Not an anti-collapse force.** A constant on-manifold latent satisfies `L_phys = 0`, so physics
  shapes *prediction quality* (less manifold drift), not representational diversity — it composes with
  the collapse axis rather than belonging to it.
- **Expected effect:** less long-horizon drift off the manifold — the central failure mode.
- **Logging (normalized / dimensionless).** Log the dimensionless terms, and the caption below each
  plot must state they're normalized — following the `viz.py:CAPTIONS` convention
  `name · equation · plain-language question · [range]`:
  - `loss_phys` — total `(d_off/r)² + (v_off/v_scale)²` (dimensionless; the penalty actually optimized).
  - `loss_phys/d_off` — caption: *`d_off/r · (√((ρ−R)²+z²) − r)/r · how far the prediction floated off
    the torus, in tube-radii (dimensionless) · [0, ∞)`*.
  - `loss_phys/v_off` — caption: *`v_off/v_scale · ⟨ṗ̂, n̂(p̂)⟩/v_scale · the predicted velocity's
    off-surface component, in units of the characteristic speed (dimensionless) · [0, ∞)`*.
  Logged on `train/` and `val/`; a loss term, not a collapse diagnostic.

**Normalization policy (decided): a metric is logged dimensionless iff it feeds `loss_phys`.**
- `manifold_distance_error` and `tangent_velocity_error` *are* the `loss_phys` terms → logged
  **dimensionless**, normalized **per-split** (÷ that split's own `r` / characteristic speed `v_scale`)
  so OOD splits with different geometry/dynamics stay comparable. In-distribution this is just a
  constant rescale, so model selection via `val/manifold_distance_error` is unaffected. This implies
  the eval computations in `environments/torus.py` divide by the per-split scale, and their
  `viz.py:CAPTIONS` entries (the explanation **below the plot**) must state the normalization:
  - `manifold_distance_error` → *`manifold_distance_error · |signed_dist(p̂)|/r = |√((ρ−R)²+z²) − r|/r,
    ρ=√(x²+y²) · how far the prediction floated off the torus surface, in tube-radii (dimensionless,
    ÷ this split's r) · [0, ∞)`*.
  - `tangent_velocity_error` → *`tangent_velocity_error · |⟨ṗ̂, n̂(p̂)⟩|/v_scale · the predicted
    velocity's off-surface (normal) component, in characteristic speeds (dimensionless, ÷ this split's
    v_scale) · [0, ‖ṗ̂‖/v_scale]`*.
- `pointwise_error` is **not** in `loss_phys` (it needs ground truth — an accuracy metric, not a
  physics one) → stays **raw physical units** (its global-position length scale is arbitrary anyway).
- `eval/control` metrics (`final_distance`, `tol`, `r_settle`) stay **physical** — operational
  thresholds a controller reasons about in real units.

## Contractivity (`contraction.weight = λ_contract`, `target = τ`)

Caps how much the one-step map can *amplify a state error*, to stop compounding error from blowing the
rollout off the manifold over long horizons. The lever is the **spectral norm `σ_max`** (largest
singular value) of the **one-step state-Jacobian** — the worst-case factor by which a perturbation of
the fed-back state grows in one step. If `σ_max ≤ τ ≈ 1`, errors stay bounded across the rollout
instead of exploding as `σ_max^T`.

**"One step"** is a single advance of the shared rollout —
`one_step(state_t, action_t, window) = next_state(transformer(to_token(state_t, action_t), window), state_t)`
— mapping the current carried state + action to the next carried state. It's model-defined through the
hooks: `state` is `o` for DSAR (obs-space Jacobian, ~`6×6`) and `z` for LSAR (latent-space Jacobian,
`dz×dz`), so one implementation covers both.

**Differentiate w.r.t. the state; condition on the action.** This is the key framing:
- `∂(next state)/∂(state)` — *what we penalize.* In the rollout it's the **state** that loops back and
  compounds, so this is the error-propagation Jacobian. It is evaluated **at the real action** (the
  rollout feeds the true action in), so the Jacobian is **action-conditioned** `J(state_t, a_t)` —
  error propagation legitimately depends on the control regime (e.g. hard thrust near `a_max`).
- `∂(next state)/∂(action)` — *what we do not touch.* That measures **controllability** (how strongly
  actions move the state), which we *want* to be large; penalizing it would make the world
  uncontrollable. (A `∂/∂action` penalty would be a separate, intentional *control-smoothness* variant
  — a different goal, not this one.)

**One-sided hinge, not a pull-to-zero.** `L_contract = max(0, σ_max − τ)²`. Below `τ`: zero penalty,
zero gradient — every well-behaved and orbital (`σ ≈ 1`, conservative/cyclic) direction is left free;
only runaway above `τ` is clipped. A direct `λ·σ_max` would be wrong — it damps legitimate expansion
*and* (for LSAR) **colludes with collapse**, since a dead/constant latent has `σ_max = 0` and would be
"rewarded." The hinge avoids both.

**Choosing `τ` — and it means different things for DSAR vs LSAR; do not reuse the number.**
- **DSAR** — the Jacobian is in **obs space**, so the simulator-measured one-step `σ_max ≈ 1.01–1.02`
  is a *directly meaningful* anchor (slightly *above* 1 because position integrates velocity, a shear
  that transiently stretches errors even in a stable system; `τ < 1` would over-constrain the real
  physics).
- **LSAR** — the Jacobian is in **latent units** (`dz×dz`); the simulator number does **not** transfer.
  The principled anchor is the delta structure `J = I + J_Δ` — an inert predictor gives exactly
  `σ_max = 1` — so use `τ ≈ 1` and sweep upward from there.

In both cases sweep a small range (`1.0–1.1`); `τ` is this variation's intrinsic knob (reported per the
fairness protocol). Too low → over-damps (orbit decays to a fixed point); too high → under-constrains
(runaway returns).

**Estimating `σ_max` — power iteration over the whole `one_step` map.** Never form the Jacobian.
Power-iterate over the *entire* `one_step` computation (encoder + fuser + transformer + head),
differentiated w.r.t. the state input: keep a probe `v` in carried-state space, iterate
`v ← normalize(Jᵀ(J v))` 2–3× via `torch.func.jvp`/`vjp`; `σ_max ≈ ‖J v‖`. This is the end-to-end
map's Jacobian — **not** a single weight matrix's spectral norm — so unlike weight
spectral-normalization it captures residuals/attention exactly. **Use power iteration uniformly** —
including DSAR's small obs-space Jacobian, where an exact Jacobian would be cheap. One estimator path
keeps the code uniform and, more importantly, scales to the high-`dz` **image** latents (milestone
goal) where exact Jacobians are infeasible.

**Autodiff caveat — verify before implementing.** The penalty needs forward-mode AD (`jvp`) **and**
second-order backward (`create_graph=True`) through `one_step`. The backbone uses **FlexAttention**
(`torch.nn.attention.flex_attention`, torch 2.6, *no SDPA fallback by design*), which does **not**
support double-backward or forward-mode AD — and certainly not under `torch.compile`. So `one_step` for
the penalty's sampled steps must run on a **plain eager attention path** — a manual `softmax(QKᵀ + mask)`
with the same causal + sliding-window mask — which fully supports `jvp`/double-backward. This is cheap
(`W = 64`, a handful of steps) and only the penalty uses it; training/eval/rollout keep the fast
FlexAttention path. Confirm with a one-line `create_graph` smoke test first.

**Operating-point bookkeeping.** Treat the sampled `state_t` as a **detached input leaf** (`requires_grad`,
no history) so the Jacobian is purely `∂ one_step / ∂ state_t`; for `wrt: last_state`, **detach the
`window_prefix`** too (hold the history fixed). Persist one probe `v` per call-site **across training
iterations** (the warm start — the Jacobian drifts slowly); since sampled steps are random each batch,
the warm start is approximate but adequate.

**Sampled along the rollout (→ action-conditioned for free).** Compute the penalty at the visited
`(state_t, a_t)` pairs from the actual rollout — that samples the real action distribution. Don't
synthesize off-distribution actions for the penalty (it would skew the constraint).

**Why subsampling.** Each penalized step costs 2–3 power iterations (≈ 4–6 extra transformer passes)
*plus* second-order backprop (`create_graph=True`, since `σ_max` is itself built from gradients). Over
a 2048-step rollout that's prohibitive in compute and memory. So enforce the constraint at only a
**handful of randomly sampled steps per batch** (`n_sample_steps`), relying on stochastic coverage
across batches/epochs to enforce it everywhere — the same logic as sampling random points for a
gradient penalty.

**Train time vs. run time.** Train: an added loss term `λ_contract · L_contract` that shapes the
weights so the learned dynamics are non-expansive (active only in training). Run time
(eval / rollout / control): **nothing extra** — stability is baked into the weights; the plain rollout
runs at full speed. Optionally log `σ_max` at eval as a diagnostic, but it is not part of inference.

**Per-model / images.** DSAR caps the obs-space step Jacobian; LSAR the latent-space (`dz×dz`) one.
Because the latent Jacobian is `dz×dz` regardless of input modality, contraction is **modality-blind
for latent models** — identical for vector or image inputs. Data-space (DSAR) with image observations
is the awkward case (a Jacobian over pixel space), another reason images favor the latent formulation.

**Not anti-collapse.** Like the other variations it shapes dynamics stability, not representational
diversity (the hinge specifically avoids rewarding the dead-latent `σ_max = 0`); it composes with the
collapse axis. Logged: `train/loss_contract` and the `σ_max` diagnostic.

## Where this sits

These compose with everything in `high_level.md`: a shoot-out cell is *(model class) × (collapse
mechanism, if LSAR) × (noise on/off) × (physical loss on/off) × (contraction on/off)*. Keep each
toggle at a single fixed value (and `τ` at one swept setting) when used as an axis, so its marginal
effect is attributable.
