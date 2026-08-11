# robocasa-scene4-4h — running log

Every run on this dataset in order, what it showed, and what we decided next. Newest at the bottom.
Dataset: `isaac-ronald-ward/robocasa-scene4-4h`, 128px, camera `robot0_agentview_left`, `RecordedEnv`
(replayed trajectories — it **cannot** `step`, so control eval is permanently off).

**The goal, as stated by the user (2026-08-10):** *slightly sharper predictions that stay coherent for
32–64 steps, with confidence that motion is actually being modelled.* 32–64 is the trained rollout length
(`F=64`). Everything before 2026-08-10 was scored on a 791–2048 step horizon, which is ~12× past that and
was actively misleading.

---

## 1. The 14.3 dB wall (pre-2026-08-09)

Four configurations — varying encoder, decoder, BPTT window, sampler — all pinned at **14.3–14.8 dB** image
PSNR. The bespoke conv/U-Net autoencoder's own reconstruction floor was ~14.3 dB.

**Decision:** the world model was saturating its tokenizer; all headroom was in the tokenizer, not the
dynamics. Move to a pretrained one.

## 2. TAESD + a broken adapter (issue #12, first attempt)

Frozen TAESD bridged to the token bag by a *learned Perceiver* adapter. Round-tripped at **10.49 dB after
nine epochs** — while a *randomly initialised* projection scores 11.10. Training the adapter bought
literally nothing and capped every image metric far below the tokenizer's ceiling.

**Decision:** rewrite the adapter as a parameter-free index rearrangement + zero-init residual, so it is
bit-exact identity at init. Verified 23.92 dB at init = the raw frozen AE on the same frames.

## 3. `taesd_exact` — the wall breaks, then collapses (08-09 04:56)

`logs/robocasa-tok/train_world_model_2026_08_09_04_56_48_taesd_exact`

| epoch | one-step PSNR | open-loop PSNR |
|---|---|---|
| 0 | 11.76 | — |
| 1–3 | 17.80 → 18.36 | 12.21 → 12.92 |
| 4 | **18.39** | 12.30 |
| 5 | **11.63** | 10.55 ← collapse, never recovered |

**Two findings that shaped everything after.**
- **Single-step and long-horizon are decoupled.** The tokenizer swap bought **+6.6 dB single-step and
  +0.1 dB at horizon**. The largest per-step win available moved the horizon metric not at all.
- **`loss/roundtrip/image` read exactly `0.0` at every epoch** — it was measuring a norm-free path the model
  never runs, so the adapter had never been trained to invert LayerNorm.

**Decisions:** (a) any lever that only improves single-step is a poor bet for horizon — dropped
`flow_hidden` and `depth`; (b) fix the round-trip to route through `encode_state`/`to_obs`; (c) the collapse
was un-diagnosable because `last.ckpt` had frozen at epoch 4 — fixed so it is written unconditionally
every epoch.

## 4. num_tokens ↔ d see-saw (08-09 22:15, fit-start only)

`logs/robocasa-tok/*nt{4,8,16,32,64}*` — floors at init, frozen TAESD:

| bag | mode | floor | LN tax |
|---|---|---|---|
| 4×256 | EXACT | 20.79 | 3.13 |
| **8×128** | EXACT | **20.41** | 3.51 |
| 16×64 | EXACT | 20.01 | 3.91 |
| 32×32 | EXACT | 19.72 | 4.20 |
| 64×16 | EXACT | 18.68 | 5.24 |
| 32×128 | **PADDED** | **16.03** | 7.89 |

**Decisions:** (a) `num_tokens × d == L` exactly, **never pad** — padding is *active damage*, not idle
width, because LayerNorm is per-token so the zeros enter the statistics the real floats are divided by
(−4.4 dB, worse than any token-count choice); (b) storage is definitively not the constraint — 8×128 holds
all 1024 latent floats; (c) no clean `num_tokens` A/B exists at a fixed budget, since raising tokens always
shrinks the backbone (`~d²`). Written up in `design/capacity.md`.

## 5. `seesaw_wide` vs `seesaw_ctrl` — capacity (08-09 22:35, killed ep1/ep4)

4×256 (~4.2M backbone) vs 8×128 (1.06M) at a fixed 1024-float budget.

Control led at **every** epoch (ep0 15.22 vs 13.71). **Decision:** dynamics capacity via the width/token
see-saw is not the lever; killed `wide`. Also first sighting of `codec/roundtrip_image` **rising**
(0.0043 → 0.0058) — the dynamics dragging the adapter off fidelity.

## 6. `df_off` vs `df_on` — diffusion forcing (08-10, killed ep4/30)

`logs/robocasa-df/train_world_model_2026_08_10_08_58_57_{df_off,df_on}` (an earlier 04:03 pair of the same
arms died at ep1 — see *incidents*). Full results are in `robocasa-df.sh`.

|  | ep0 | ep1 | ep2 | ep3 |
|---|---|---|---|---|
| one-step, off | 12.50 | 17.68 | 17.95 | **18.20** |
| one-step, on | 11.04 | 17.38 | 17.65 | 17.79 |
| open-loop, off | 9.20 | 12.05 | 12.28 | **12.70** |
| open-loop, on | 8.82 | 12.48 | 12.82 | 12.25 |
| **val `dynamics/latent`, off** | 0.2833 | **0.2961** | **0.2961** | **0.2966** |
| val `dynamics/latent`, on | 0.8644 | 0.3870 | 0.4202 | 0.5256 |
| **ae_floor (codec NOW), off** | 23.52 | 22.23 | 21.88 | **21.63** |
| ae_floor, on | 23.02 | 21.75 | 21.58 | 21.29 |

**Three findings.**
1. **DF at 0.1 does not help.** Trails on one-step at every epoch, LPIPS a wash, and its train-noised vs
   val-clean gap *widens* every epoch — reproducing what `conf/config.yaml` already recorded for 0.25/1.0.
   It also does **not** simulate autoregression: it mixes in *isotropic Gaussian* noise, while real rollout
   error is structured and drift-shaped.
2. **The dynamics stopped learning at epoch 1.** `dynamics/latent` flat to 4 s.f. across three epochs, and
   it is ~96% of the objective. Every gain in `val loss total` came from `decode/image`, not the dynamics.
3. ***The codec is eroding.*** `eval_ae_floor` — the *current* encode→decode quality — falls ~2.3 dB in four
   epochs, and `closed_loop_1_steps` tracks it almost exactly (22.45 → 21.13). **One-step prediction was
   pinned to a ceiling training was destroying.** `dynamics/latent` and `codec/roundtrip` share the same
   0.134M trainable adapter params and the dynamics wins.

**Decisions:** drop DF. Attack the erosion. And fix the measurement, because —

## 7. The measurement gap (08-10)

Readouts fired at **quarters of H** only, so at H=791 they were `@+197/394/591/791` — **no number at step 32
or 64 anywhere.** Every headline figure quoted for two days was a mean over 791 steps. And nothing tested
whether *motion* was modelled: no persistence baseline existed, so a model predicting "next = current"
would score respectably.

**Landed:** readouts at `@+1,8,16,32,64` ∪ quarters; `eval.horizon` 2048 → 128; `psnr_frozen` (hold frame 0
— the do-nothing baseline); `motion_ratio` (‖Δpred‖/‖Δtrue‖, 1 = right amount of motion, <1 = freezing).
Validated: a frozen predictor scores `motion_ratio` 0.000 and `psnr` *exactly* equal to `psnr_frozen`; a
perfect one scores 1.000.

## 8. `tf_ln` vs `tf_affine` — token-mixing denoiser (08-10 19:36, KILLED ep1/ep2)

`logs/robocasa-tfaff/train_world_model_2026_08_10_19_36_25_{tf_ln,tf_affine}` — see
`robocasa-tf-affine.sh`. Both `flow_arch=transformer`; arms differ only in `latent_norm`
(layernorm floor 20.41 vs affine floor 23.92). Score arm B as **gap-to-floor**.

Rationale: a dynamics loss flat to 4 s.f. is what a *representational ceiling* looks like, and the
per-token MLP denoiser (0.072M) treats the 8 tokens independently given frozen conditioning — it cannot
represent correlated motion across the bag. The transformer is also **the only lever with evidence from
outside this machine** (reported significantly better than mlp on the user's other setup). Affine removes
the 3.51 dB LayerNorm tax, which is the only reason the adapter residual must learn anything at all.

---

## 9. `tf_affine` vs `tf_bespoke` — frozen vs learned tokenizer (08-11 03:31, killed ep3/ep2)

`logs/robocasa-tfaff/train_world_model_2026_08_11_03_31_45_{tf_affine,tf_bespoke}`

Both `flow_arch=transformer`. Arm A frozen TAESD + affine; arm B a bespoke conv/U-Net AE trained from scratch
(+ layernorm, because affine presupposes a FIXED latent a learned encoder does not have).

**The inversion, at ep0.** The bespoke codec is **8.6 dB worse** (15.29 vs 23.88) and it was **2 dB BETTER at 64
steps** (9.06 vs 7.06), with `motion_ratio@+64` **1.117 vs 0.589**. Sharp-but-static versus blurry-but-moving,
exactly as read off the videos. A reconstruction-optimised frozen latent is a hostile prediction target.

**Then BOTH diverged**, and the cause was not the tokenizer:

```
val/loss/total    tf_affine  0.711 0.420 0.863 1.825      tf_bespoke  0.824 1.502 3.202
grad/norm/flow    tf_affine  0.98  1.48  766   21         tf_bespoke  2.10  65.6  10151
ae_floor          tf_affine  23.88 22.42 18.94 16.37      tf_bespoke  15.29 10.69 11.47
lpips@+1          tf_affine  0.173 0.254 0.450 0.627
```

`_TokenMixBlock` did not zero-init its residual branches, so it perturbed the velocity field from step 0 inside
a graph ~32 BPTT steps x 6 ODE steps deep — gain >= 1 per application, compounded ~192 times.
`gradient_clip_val=1.0` then **laundered** it: `norm_postclip` was exactly 1.000 every epoch, so training took
full-size confident steps along the exploded direction and degraded SMOOTHLY instead of NaN-ing. That is why it
read as a modelling failure for two days.

**Decisions:** zero-init both residual branches (identity at init, wakes after one step); log
`grad/clip_ratio = preclip/clip_val` as the number that makes this legible; and the per-token MLP denoiser is
the only one with four epochs of stable gradients (0.43-0.85), so it is the safe baseline.

## 10. `mlp_affine` vs `tfz_affine` — mlp vs FIXED transformer (08-11 16:03, RUNNING)

`logs/robocasa-tfaff/train_world_model_2026_08_11_16_03_52_{mlp_affine,tfz_affine}`

Byte-identical except `flow_arch`. **The first clean mlp-vs-transformer test** — every earlier comparison was
confounded by normalization and/or batch size.

|  | mlp ep0 -> ep1 | transformer(zero-init) ep0 -> ep1 |
|---|---|---|
| **ae_floor (CODEC)** | 24.31 -> **21.52**  (-2.8) | 25.06 -> **24.51**  (-0.55) |
| val `codec/roundtrip` | 0.0038 -> 0.0070 (+84%) | 0.0033 -> 0.0037 (+12%) |
| **val 1-step PSNR (PREDICTIVE)** | 12.47 -> 16.74 | 15.55 -> **18.29** |
| closed_loop_1 | 22.61 -> 20.95 | 22.92 -> **22.67** |
| open_loop @+32 | 10.39 -> 13.91 | 11.17 -> **14.39** |
| open_loop @+64 | 8.11 -> 12.97 | 9.83 -> **13.64** |
| (bar: `psnr_frozen@+64`) | 10.40 | 10.40 |
| motion_ratio @+64 | 0.769 -> 0.192 | 0.700 -> **0.126** |
| lpips @+1 | 0.207 -> 0.384 | 0.132 -> **0.166** |
| val loss total | 0.708 -> 0.517 | 0.484 -> **0.254** |
| grad/norm/flow | 0.602 -> 0.906 | 0.593 -> **0.571** |

**1. The zero-init fix holds so far.** `grad/norm/flow` 0.593 -> 0.571, where the broken run went
0.98 -> 1.48 -> 766. It is now BELOW the mlp. ep2 is the real test (that is where 766 appeared).

**2. A STRONGER DENOISER PROTECTS THE CODEC — the headline result.** The transformer holds `ae_floor` at 24.51
(-0.55 dB) while the mlp erodes to 21.52 (-2.8 dB), and `codec/roundtrip` rises 12% vs 84%. Interpretation: the
weaker per-token MLP cannot fit the dynamics, so it leans on the shared adapter to reshape the latent into
something easier — paying codec fidelity for dynamics loss. The transformer is capable enough not to need that.
This also re-frames section 8: what was called "the dynamics destroying the codec" is really "an underpowered
denoiser destroying the codec".

**3. Both arms now BEAT the do-nothing baseline at 64 steps** for the first time: 12.97 and 13.64 against
`psnr_frozen@+64 = 10.40`. Predictive PSNR is genuinely better, and the transformer leads everywhere
(+1.55 dB one-step, +0.67 dB at 64 steps, half the val loss, less than half the lpips@+1).

**4. But motion is COLLAPSING as they train** — `motion_ratio@+64` 0.769 -> 0.192 (mlp) and 0.700 -> 0.126
(transformer). They are getting SHARPER AND MORE STATIC. Beating persistence by predicting a
slightly-less-frozen scene is not the goal, and the transformer is the *worse* of the two on motion.

**Consequence -> action conditioning** (design/action_conditioning.md): `grad/norm/act_enc` was 0.0010 against
the flow's 0.5932, i.e. **0.17% of the total gradient**. `readout()` was slicing off the action token's output
entirely, so actions reached a prediction ONLY via attention onto 1 of 10 slots.
`diffusion.concat_action_embedding` (default ON) now concatenates the RAW pre-backbone `act_enc(act)` onto every
state token; `model.action_fourier_freqs` and `modalities.<i>.fourier_freqs` (both default 0/off) add sin/cos
bands. NOT bit-identical to these runs — they predate it.

## Appendix — folded in from wizard/scripts/*.md (2026-08-11)

These lived next to the launch scripts, where `.gitignore` kept them unsynced. Content preserved verbatim.

### from `wizard/scripts/handoff-eval-action-dist.md`

# Handoff: 4 fixes in the action-distribution eval + one config declaration

Context: running `mm_flow` on a **recorded, no-simulator** dataset with a **12-dim** action space
(`isaac-ronald-ward/robocasa-scene4-4h`, `environments=recorded`). Line numbers are against the commit
`2c1ea51`. Items 1–3 are bugs; item 4 is a feature request with a design.

Motivating measurement — the 12 action dims over all 259,299 train frames. Note how degenerate a real
robot action space is, which is what makes the current magnitude-only products uninformative:

| dim | std | n_unique | reading |
|---|---|---|---|
| 0,1,2 | 0.23 / 0.21 / 0.19 | ~550–700 | EEF delta position |
| **3** | **0.000** | **1** | exactly constant — dead |
| **4** | 0.71 | **2** | binary flag (~85% at −1) |
| 5,6,7 | 0.31 / 0.24 / 0.30 | ~700 | rotation deltas |
| 8,9,10 | 0.081 / 0.086 / 0.072 | ~710 | near-dead (mobile base, unused in this scene) |
| **11** | 0.97 | **2** | binary — gripper |

---

## 1. `model.compile_rollout` is undeclared, so the bare override is rejected

`src/quickdraw/training/setup.py:81` reads it with `bool(m.get("compile_rollout", False))`, but
`conf/model/mm_flow.yaml` never declares the key. So:

```
model.compile_rollout=true    -> ERROR: "Key 'compile_rollout' is not in struct"
+model.compile_rollout=true   -> works
```

This is the same trap class as the old `trainer.fast_dev_run` (read but undeclared ⇒ needs `+`). It is
easy to hit because every *other* model knob in that file takes the bare form.

**Fix:** declare it in `conf/model/mm_flow.yaml` next to `grad_checkpoint`, defaulting off, with the
guard conditions in the comment:

```yaml
compile_rollout: false   # true -> torch.compile the AR rollout STEP (~6x, accelerations.md Exp 9).
#                          Requires head_dim (d/heads) >= 16 and is mutually exclusive with
#                          variations.contraction (its Jacobian power-iteration needs the eager graph).
```

Measured on this dataset for reference: **3.90 s/batch eager → 0.67 s/batch compiled** at
`d=128 depth=4 F=64 batch=32` (5.8x), and 1.07 s/batch at `d=256 depth=6`.

---

## 2. CRASH: bare `ecfg.a_max` in `eval_action_distribution` (3 sites)

`src/quickdraw/evaluation/routines.py` lines **679, 690, 694**:

```python
fig = viz.fig_action_by_state(arr, x, ecfg.a_max, ...)              # 679
frames = viz.anim_action_distribution(true_a, pred_a, ecfg.a_max, ...)   # 690
frames_bx = viz.anim_action_by_state(true_a, pred_a, x, ecfg.a_max, ...) # 694
```

`a_max` is a **torus-only** knob (`conf/environments/torus.yaml:7`, `a_max: 4.0`).
`conf/environments/recorded.yaml` has no such key ⇒ `AttributeError` the moment this eval runs on any
recorded/non-torus env. This is the identical bare-attribute pattern already fixed at `_ood_axis`
(line 166), just further down the same file.

`a_max` is only ever used as a histogram **x-limit** — `viz.py:1197`, `1287`:
`hi = min(float(a_max), <data max> * 1.05)`. So it has a natural data-driven fallback.

**Fix:** pass `getattr(ecfg, "a_max", None)` and make the three viz functions treat `None` as "use the
data range":

```python
a_max = getattr(ecfg, "a_max", None)   # torus-only knob; None -> derive the limit from the data
```
and in `viz.py` (`fig_action_distribution:1160`, `anim_action_distribution:1185`,
`anim_action_by_state:1225`, `fig_action_by_state:1276`):
```python
data_hi = max(float(tm.max()), float(pm.max())) * 1.05
hi = data_hi if a_max is None else min(float(a_max), data_hi)
```

### 2b. Same bug, one line up, still live: `ecfg.init_speed`

`routines.py:168` — inside the function whose `R`/`r` default you *just* made lazy on line 166:

```python
se.get("init_speed", ecfg.init_speed),   # <-- bare attribute, evaluated EAGERLY
```

`se.get(k, default)` always evaluates `default`, so this raises on any env without `init_speed` —
`recorded.yaml` has none — even when the dataset card supplies `init_speed`. Line 166 was fixed with
`or {...getattr...}`; line 168 was missed.

**Fix:** `se.get("init_speed", getattr(ecfg, "init_speed", None))`, and confirm `_openloop_split`
tolerates `None` (it is reached only by the `ood_visual/geometric/dynamics` axes, so this is latent
rather than blocking — but it is the same footgun).

---

## 3. The by-state ("by-x") products must be torus-only

`routines.py:663`:

```python
x = np.stack([o[1:L, 0] for o, _, _ in eps]).astype(np.float32)   # ambient x at each action's state
```

This hardcodes **observation dim 0** and the two by-state products then split rows on `x<0` / `x>=0`,
captioned as the torus's *slow basin / fast basin* (`viz.py:1226`: "TOP x<0 slow, BOTTOM x>=0 fast").

On any other env, obs dim 0 is an unrelated quantity — for this robocasa state it is a joint value with
mean 2.86, std 1.16, so `x<0` selects almost nothing and the split is silently meaningless. It does not
crash, which is worse: the panels render and look authoritative.

**Minimal fix** — gate both by-state products on geometry, exactly like the denoising routines now do:

```python
has_geom = getattr(ecfg, "R", None) is not None      # by-x split is torus semantics
...
if has_geom:
    for name, arr in (("true", true_a), ("pred", pred_a)):
        fig = viz.fig_action_by_state(...)           # 678-681
    frames_bx = viz.anim_action_by_state(...)        # 694-695
```

**Preferred fix** — make it an *optional env hook*, matching the 7-hook contract in `docs/byo.md` so
each hook independently unlocks one product with a graceful fallback. Add to the `WorldEnv` protocol in
`src/quickdraw/environments/base.py`:

```python
def action_dist_split(self, obs):        # OPTIONAL
    """-> (labels: bool array over obs rows, low_name: str, high_name: str) | None.
    Splits action-distribution panels by a MEANINGFUL state feature. None -> panels are pooled only."""
```

`TorusEnv` returns `(obs[..., 0] < 0, "slow", "fast")`; every other env inherits `None` and the by-state
products self-skip. That also deletes the hardcoded `o[..., 0]` and lets the panel captions come from the
env instead of being baked into `viz.py`.

---

## 4. FEATURE: per-dim action marginals (dataset/env-agnostic)

**Why the current products are not enough.** Every existing action plot reduces to a **magnitude**:
`np.linalg.norm(..., axis=-1)` at `viz.py:1192-1193`, `1231-1232`, `1283`. That is dimension-agnostic (so
12-D does *not* crash once item 2 is fixed) but for a real action space it is dominated by whichever dims
happen to have the largest scale. Here |a| is driven by the two **binary** dims (std 0.97 and 0.71),
while the six dims that actually carry continuous control have std 0.07–0.31. So the headline plot mostly
shows the gripper toggling, and a head could match |a| well while getting every continuous dim wrong.

Likewise `true_pred_w1` (`routines.py:684`) is a single scalar on pooled |a| — it cannot say *which* dim
is wrong.

### 4a. New viz function

In `src/quickdraw/logging/viz.py`, beside the existing action functions (~1160–1320):

```python
def fig_action_marginals(true_a, pred_a, names=None, max_cols=4, discrete_max=10, q=(0.001, 0.999)):
    """Per-dim action marginals: recorded (filled) vs head (outline) on SHARED bins, one panel per dim.
    true_a/pred_a: (E, T, A) physical actions. names: list[str] | None -> 'a[i]'.
    Dimension-agnostic and env-agnostic: no a_max, no state split, no geometry."""
```

Per-dim behaviour, decided from the **true** actions so panels stay comparable across runs:

- **constant** (`n_unique <= 1`): don't fake a histogram — render a text tile `"a[3] constant @ 0.000"`.
  Keeps the grid aligned and makes degeneracy *visible* instead of hidden.
- **discrete** (`n_unique <= discrete_max`): grouped bar chart of value frequencies, true vs pred.
  A 60-bin histogram of a ±1 flag is unreadable; this is the correct rendering for dims 4 and 11.
- **continuous**: shared bins over `[lo, hi]` = the `q` quantiles of the true dim, padded ~5%. Overlay
  true (filled, alpha) and pred (step outline). Robust quantiles, not min/max, so one outlier can't
  flatten the panel.

Shared bins between true and pred are the important detail — separately-binned histograms are not
visually comparable.

### 4b. Per-dim quantitative metric

Reuse the existing quantile-difference Wasserstein already at `routines.py:683-684` (same `q` grid),
applied per dim instead of to pooled magnitude:

```python
q = np.linspace(0.0, 1.0, 512)
w1_per_dim = [float(np.mean(np.abs(np.quantile(true_a[..., i], q) - np.quantile(pred_a[..., i], q))))
              for i in range(true_a.shape[-1])]
live = [i for i in range(true_a.shape[-1]) if true_a[..., i].std() > 1e-6]   # skip dead dims
writer.scalars({f"eval_action_distribution/w1/dim_{i}": w for i, w in enumerate(w1_per_dim)}, step)
writer.scalar("eval_action_distribution/w1_mean", float(np.mean([w1_per_dim[i] for i in live])), step)
```

**`live` matters:** a constant dim has W1 ≈ 0 by construction. Averaging it in would flatter the head —
on this dataset 4 of 12 dims are dead or near-dead, so a naive mean is diluted by a third. Keep the
existing pooled-magnitude `true_pred_w1` unchanged for continuity with prior runs.

`writer.scalars(dict, step)` already exists (`logging/writer.py:135`).

### 4c. Dim names, from the dataset (not the env)

lerobot carries them: `<root>/train/meta/info.json` → `features.action.names`. It is `null` for this
dataset, so the fallback must be graceful — but wiring it now means a dataset that *does* label its
actions gets readable panels for free. Read once in the routine (root via `resolve_data_root(cfg)`,
already imported at `routines.py:20`), fall back to `None` ⇒ `a[i]`.

### 4d. Call site

In `eval_action_distribution`, right after `true_a` / `pred_a` are built (`routines.py:671-672`) and
**unconditionally** — it needs no geometry, no `a_max`, no state split, so it works for torus (2-D),
pendulum (1-D) and robocasa (12-D) alike:

```python
fig = viz.fig_action_marginals(true_a, pred_a, names=action_names)
writer.figure("eval_action_distribution/marginals", fig, step); plt.close(fig)
```

This should become the **primary** product of the eval; the magnitude animations stay as secondary
views, and the by-state ones become torus-only per item 3.

---

## Acceptance criteria

1. `python -m quickdraw.eval_action_distribution checkpoint=<run> data.root=<root> data.repo_id=robocasa-scene4-4h data.cam=robot0_agentview_left environments=recorded environments.obs_dim=16 environments.action_dim=12` completes with **no `AttributeError`**.
2. `eval_action_distribution/marginals` shows **12 panels**: histograms for dims 0,1,2,5,6,7,8,9,10; bar
   charts for dims 4 and 11; a "constant" tile for dim 3.
3. No `by_state_*` / `animation_byx` products are emitted for `environments=recorded`; they still are for torus.
4. `w1_mean` excludes dim 3, and per-dim `w1/dim_*` scalars exist for all 12.
5. Torus runs are **unchanged** — same products, same `true_pred_w1`, `a_max` still respected as the limit.

Note this eval requires an action head, which is trained **post-hoc** on a frozen WM via
`train_action_model` (joint training is locked off — it killed control and destabilised the WM). So
reproducing needs a `train_action_model` run first; there is no action prior in a stock WM checkpoint.

### from `wizard/scripts/robocasa-scene4-4h.md`

# Wizard record — `isaac-ronald-ward/robocasa-scene4-4h`

Script: `wizard/scripts/robocasa-scene4-4h.sh` (both gitignored). Dataset inspected 2026-08-04;
**experiment revised 2026-08-05** from a decode-objective A/B to a **capacity sweep** at the user's
direction ("do the vit mse on both, a smaller 3.3M on one and a larger one on the other… 128x128 on both").

## Part A — what the dataset actually is

**Inspected before asking anything.** The key finding: this repo is **not a raw robocasa dump** — it is
already a **processed quickdraw recording**, pushed via `push_to_hub` from
`logs/recording_2026_08_04_06_54_40_robocasa-scene4-4h`. It has `train/` + `val/` split dirs each with
`meta/info.json`, `data/chunk-000/`, `videos/observation.images.<cam>/`, plus top-level `summary.json`,
`dataset_card.json`, `normalization_stats.json`, `progress.log`, and a `media/` dir of per-episode mp4s.

**So there is no `data_generation` stage and no `data/processors.py` stage.** Those are already done. (The
raw source was `madang6/quickdraw-robocasa-scene4-4h`, which the `robocasa` processor reads.)

| property | value | source |
|---|---|---|
| `obs_dim` | **16** | `train/meta/info.json` → `observation_vector.shape` |
| `action_dim` | **12** | `train/meta/info.json` → `action.shape` |
| `dt` / fps | **0.05 / 20** | `summary.json`, `dataset_card.json` |
| camera key | **`robot0_agentview_left`** — the ONLY one | `videos/observation.images.robot0_agentview_left` |
| image | **256×256×3**, av1 / yuv420p | `info.json` video info |
| train | 235 eps · 259,299 frames · **3.60 h** · len 615/1103/1749 | `summary.json` + `check_dataset` |
| val | 26 eps · 29,294 frames · **0.41 h** · len 644/1126/2066 | `summary.json` + `check_dataset` |
| windows @ P=8 F=64 | **242,614 train / 27,448 val** | `check_dataset` |
| robot_type | `null` | `info.json` |

**Column of `docs/byo.md`: 1 — recorded data, no simulator.** Confirmed by the run's own contract report:

```
[env-contract] recorded | obs_dim=16 action_dim=12 | reset ✗ no-sim | step ✗ no-sim | reward ✗ no-control |
render_obs ✗ dataset-images-only | rollout_metrics ✗ default(pointwise) | render_diagnostics ✗ filmstrip |
control_goals ✗ reward-only | physical_loss ✗ off | checkpoint_metric=pointwise_error | policies=-
```

- **Available:** WM training + validation, the `ood_horizon` pointwise metric, the `pred`/GT image
  filmstrip, and (deferred, not in this script) `eval_interpret` + `train_reward_model`, which need only the
  frozen WM + data + a VLM.
- **Unavailable:** MPPI control — both the goal race *and* language steering — plus the oracle baseline.
  All three must *step* the env. Hence `eval.during_train.evals.control=false`.

**Only one camera**, so the multi-trunk (one `image` modality per camera) question is moot. Multiple
cameras would mean re-running the processor against the source repo with `+source.camera=<leaf>`; the
`robocasa` docstring notes the source has 3 cameras.

## The user's choices

| topic | chosen | notes |
|---|---|---|
| image decode | **ViT-MSE on BOTH arms** | `decode_kind=mse`, `decode_arch=vit` — the header's winner |
| image resolution | **128×128 on both** (from 256) | the resolution the recipe's numbers were measured at |
| action head | **off on both** | WM only; no `train_action_model` |
| model size | **arm A ~3.2M / arm B ~14.2M** | the sweep axis |
| pipeline scope | **`train_world_model` only** | interpret + reward head deferred to the winning arm |
| GPUs | **both — one arm each** | GPU 0 = base, GPU 1 = large, otherwise byte-identical |

## The experiment: capacity, with everything else held fixed

The single independent variable is `d`/`depth`/`heads`. Everything else — data, schedule, decode objective,
teacher forcing, epochs — is identical between arms, so a difference in outcome is attributable to capacity
alone.

| arm | `d` | `depth` | `heads` | `head_dim` | params | GPU |
|---|---|---|---|---|---|---|
| base | 128 | 4 | 8 | 16 ✓ | **3.22M** | 0 |
| large | 256 | 6 | 16 | 16 ✓ | **14.21M** | 1 |

`head_dim = d/heads` must be a power of 2 for FlexAttention — both arms land on 16. (`d=192, heads=8`
would give 24 ✗; `d=256, heads=16` was chosen over `d=192, heads=12` for a cleaner 4.4× scale-up.)

`model_summary` for the large arm — the growth is almost entirely backbone, since `num_tokens=8` fixes the
per-frame token budget and the ViT heads scale only with `d`:

```
[train] MultiModalFlow 14.21M params
[arch] d=256 window=32 | per-step bag = 9 state token(s) + 1 action = 10 tokens
```

Base arm, per component (the 3.22M reference):

```
  proprio encoder (mlp)              (B,T,16) -> (B,T,1,128)                              0.009M
  image encoder (vit)                (B,T,128,128,3) -> (B,T,8,128)                       0.968M
  action_enc                         (B,T,12) -> (B,T,1,128)                              0.018M
  space-time backbone                (B,T,10,128) -> same [spatial 10 tok/step + causal]   1.060M
  predict_next: flow (rectified, per-token) (B,T,9,128) -> same                            0.072M
  proprio decode (mlp flow)          (B,T,1,128) -> (B,T,16)                              0.019M
  image decode (vit mse/no-noise)    (B,T,8,128) -> (B,T,128,128,3)                       1.072M
```

**Why 128×128 and not native 256.** At 256 the same config is 3.27M (encoder 0.992M / decode 1.097M) —
parameters barely move, but `patch=16` means **256 patches/frame vs 64**, ~4× the ViT encode+decode work
across the F=64 rollout, and the GPU-resident frame store grows 12.75 GB → 51 GB. The cost is compute and
memory, not parameters. 128 also matches the resolution every quoted metric was measured at.

## Learnings applied (quoted from `conf/model/mm_flow.yaml`)

- **`decode_kind=mse` + `decode_arch=vit`** — "ViT-MSE image decode (decode_kind=mse, decode_arch=vit)
  slightly BEAT U-Net/flow on EVERY axis here": val PSNR **18.9 vs 18.3**, OOD pointwise **0.24 vs 0.32**,
  OOD PSNR **14.5 vs 12.6**, control **4.12 vs 3.5**. The image is a deterministic render, so "the
  conditional mean IS the target."
- **`action_head.enabled=false`** — "the JOINT action head KILLS control (goals ~0 vs 3.88 baseline) AND
  destabilizes the WM (unetflow NaN'd ~ep31, vitmse collapsed to a frozen 11.1)." The control half of that
  doesn't even apply here (no steppable env); the destabilization half does.
- **`p_tf_end=0.0` (in-rollout), `p_tf_warmup_epochs=4`, Diffusion Forcing OFF** — full teacher forcing
  causes "autoregressive MEAN-COLLAPSE (proprio→origin, images→one frozen frame)"; in-rollout took "val
  0.81→0.32, pointwise 0.61." DF "at scale 0.25/1.0 it made collapse WORSE" — "The anti-collapse lever is
  in-rollout, not DF."
- **`diffusion.shortcut=true`** — K=1 sampling via self-consistency, safe because next-state dynamics are
  near-deterministic (a straight noise→target field, so "1×(2d) == 2×(d)").
- **`recon_frac=0.25`, `detach_every=16`, `num_tokens=8`, `encode_arch=vit`** — from the FULL recipe line;
  the header also forbids "NO conv encoder, NO num_tokens>8, NO grad-accum, NO window_stride>1".
- **`lr_warmup_steps=300`, `weight_decay=1e-4`** — already the defaults in `conf/optim/adamw.yaml`, so no
  override was added. Warmup exists because "flow heads regress a clean target from near-pure noise →
  high-variance early gradients."
- **Forbidden levers left alone:** `trainer.accumulate_grad_batches` (LOCKED at 1 — Lightning SUMS
  micro-batch grads → ~N× effective LR → mean-collapse) and `data.window_stride` (LOCKED at 1).
- **`trainer.max_epochs=20`** — the header's metrics are "@ep10-15" and it records collapse past ep20
  (NaN ~ep31, frozen 11.1), so 20 is the informative budget. Config default is 50; raise it only to study
  the collapse itself.

## Repo gotchas found while wiring this up

These are the reason the script looks the way it does. **None required a code change.**

1. **`data.root`, not `data.hf_repo`.** Training resolves either (`resolve_data_root`), but every eval
   routine reads `cfg.data.root` **directly** — `evaluation/routines.py:89,163,218,264,361,414,636,639`.
   With `hf_repo` alone, `data.root` stays `null` and `ood_horizon` would die at the first eval epoch (5).
   The script resolves the snapshot path itself and passes `data.root`.
2. **Frame-cache race.** `load_fpv_frames` writes `<root>/<split>/<cam>_128.npy` with a non-atomic
   `np.save` (`data/dataset.py:69,108`). Two simultaneous cold starts would both decode 288k frames and
   could tear the file. Stage 0 warms it serially: **train 12.75 GB in 158s, val 1.44 GB in 19s**. This is
   the single most important reason the two arms cannot simply be launched back-to-back from cold.
3. **`manifold` / `denoising_*` must be disabled.** `eval_manifold` (`routines.py:218`) and both denoising
   routines call `load_split_episodes_mm(cfg.data.root, "val", img_size=img_size)` with **no `cam` and no
   `repo_id`**, so they default to `fpv` / `"torus"` and cannot find this dataset. Worse, the denoising
   routines gate on `isinstance(m, MultiModalFlow)` — which **is** our model, so they do *not* self-skip —
   and then read torus-only `ecfg.R` / `ecfg.r`, absent from `recorded.yaml`.
4. **`ood_horizon` is safe** and stays on: it passes `cam` + `repo_id` (`routines.py:89`) and takes
   geometry via `getattr(ecfg, "R", None)`, so the torus scene path degrades gracefully. "OOD" here means
   horizon ≫ trained (H up to 2048 vs F=64) on the **val** split — not a distribution-shifted split, which
   this dataset doesn't have.
5. **`recorded.yaml` ships starling's dims AND its rate.** `obs_dim: 16` happens to match, but
   `action_dim: 4` and `dt: 0.0333333` (30 Hz) do not — this dataset is 12-dim at 20 Hz. `dt` feeds
   `fps = round(1.0/ecfg.dt)` in `eval_ood_horizon`, so without `environments.dt=0.05` the filmstrip and
   rollout videos play 1.5× too fast.
6. **`trainer.fast_dev_run` does not exist in this repo.** `Trainer(...)`
   (`train_world_model.py:178-190`) never passes it, so plain `trainer.fast_dev_run=true` is rejected by
   the struct and `+trainer.fast_dev_run=true` would be silently ignored. The wired equivalents are
   `+trainer.limit_train_batches` / `+trainer.limit_val_batches`, which the script uses instead.
   **`wizard/prompt.md`'s pre-flight step 3 is wrong for this codebase** and should be reworded.
7. **Hydra override quoting.** `run_summary` prose containing `(` or `,` fails the override grammar
   ("mismatched input ' ('"). Each arg is shell-single-quoted with the value double-quoted.
8. **The duplicate-summary check is inert under `QUICKDRAW_LOG_ROOT`.** `_assert_summary_unique` is called
   with its default `root="logs"` (`train_world_model.py:90`), globbing `logs/*/auto_run_summary.txt`,
   while runs land in `logs/robocasa-scene4-4h/<run>/`. The summaries here are unique anyway; the smoke
   stage carries `allow_duplicate=true` so re-running the script is safe.
9. **Stale startup banner.** `[startup] data inventory` echoes `conf/data/torus.yaml`'s `splits:` block
   (256 traj × 256 steps, `eval_ood_visual/geometric/dynamics`) — meaningless on this path. The real
   numbers are the line above: `data ready in 65.9s: 242614 train / 27448 val windows`.
10. **`snapshot_download` pulls the 1.13 GB `media/` dir** (2.54 GB total) that the loader never reads —
    the lerobot videos it uses are under `train/videos/`. Cosmetic bandwidth waste only.
11. **`docker compose exec` does not forward host env** (found 2026-08-05). The script `export`s
    `QUICKDRAW_LOG_ROOT`, but that only affects the *host* shell; without an explicit `-e` on every `exec`,
    `make_run_dir` falls back to plain `logs/` inside the container and the two arms scatter outside the
    grouped subfolder. Every `exec` in the script now passes `-e QUICKDRAW_LOG_ROOT=…`.

Items 1, 3 and 5 are latent bugs on the recorded+non-torus path rather than user error — worth fixing in
the repo (thread `resolve_data_root` and `cam`/`repo_id` through the eval routines; make the denoising
routines require torus geometry rather than just `MultiModalFlow`) if this dataset becomes a regular target.
Item 6 is a documentation bug in `wizard/prompt.md` itself.

## Pre-flight results

**1. `check_dataset`** — passed:

```
[check] data=/caches/hf/hub/datasets--isaac-ronald-ward--robocasa-scene4-4h/snapshots/5a3df71e...
        P=8 F=64  (an episode needs >= P+F = 72 steps for 1 window)  repo_id=robocasa-scene4-4h
[check] env 'recorded': obs_dim=16 action_dim=12
[check] train:  235 episodes | len min/mean/max = 615/1103/1749 | obs_dim=16 action_dim=12 -> 242614 training windows
[check] val  :   26 episodes | len min/mean/max = 644/1126/2066 | obs_dim=16 action_dim=12 -> 27448 training windows

[check] OK — dataset fits the config
```

**2. `model_summary`** — 3.22M (base) / 14.21M (large); see the arch tables above.

**3. Smoke (2 train + 2 val batches)** — both configs pass, no nonfinite grads:

| arm | run dir | train_loss | val_loss | nonfinite grads |
|---|---|---|---|---|
| base (3.22M, ViT-MSE) | `train_world_model_2026_08_04_20_52_19_rc4h_smoke_mse` | **2.5908** | **2.3219** | 0 |
| large (14.21M, ViT-MSE) | `train_world_model_2026_08_05_03_07_14_rc4h_smoke_large` | **2.7071** | **2.3088** | 0 |

Large-arm grad norms at step 1 — the image heads dominate and warmup is doing its job (preclip 2.62 clipped
to 1.0): `decode_image 2.03`, `encode_image 1.63`, `decode_proprio 0.218`, `encode_proprio 0.073`,
`flow 0.064`, `backbone 0.045`, `act_enc 0.0008`. The 14.2M arm fits at `batch=32` alongside the 14.2 GB
resident frame store, which was the open risk.

## Expected cost

242,614 windows ÷ batch 32 = **7,582 steps/epoch**; the base smoke measured ~0.5 s/batch → **~63 min/epoch**,
plus validation every 4 epochs ("~as long as the train epoch"). So **20 epochs ≈ 1 day for the base arm**;
the large arm is wider and will be somewhat slower. Both run in parallel on separate GPUs. `best.ckpt`
tracks `pointwise_error` (no `checkpoint_metric` hook on `RecordedEnv`). Resume with
`+resume=<run_dir>/checkpoints/last.ckpt`.

## What to compare when they finish

Both runs write to `logs/robocasa-scene4-4h/train_world_model_<ts>_rc4h_{base,large}/`. The decisive
products, all under `eval_ood_horizon/`:

- `<head>/psnr|ssim|mse|l1` error-vs-step curves and the `@+x` scalars — the primary capacity readout. A
  large PSNR gap means the base arm was underfitting the kitchen scene.
- `image/filmstrip_<i>` — decoded vs ground-truth frames. Read this **alongside** PSNR: a conditional-mean
  decoder is structurally favoured by PSNR, so check whether the larger model actually resolves movable
  objects and the gripper rather than just lowering average error.
- `eval_ood_horizon` / `pointwise_error` — the proprio-side scalar `best.ckpt` is chosen on.
- Watch both for the documented failure mode: autoregressive **mean-collapse** (images → one frozen frame,
  proprio → origin) and any nonfinite-grad skips, especially past ep10.

**Then:** run `eval_interpret` (needs a drafted `conf/interpret/robocasa.yaml` — semantic factors + VLM
prompt, the one non-automatic interpret input — plus `OPENAI_API_KEY`) and `train_reward_model` on whichever
arm wins. Both are env-free and therefore available on this recorded dataset.

## Standing decisions

- `num_tokens × d == L` exactly. **Never pad.**
- Latent normalization is **required**; only the mechanism is a choice (`layernorm | affine | none`).
- `p_tf_end=0` (in-rollout). Never full teacher forcing.
- `accumulate_grad_batches=1` and `window_stride=1` are **LOCKED**.
- Never `uv sync` (strips umap/sklearn).
- **Never edit `src/` while a run is live** — see incidents.
- Action head **off** on this dataset: collapsed 2/2 (ep15, ep7, identical signature).
- Control eval **off**: `RecordedEnv` cannot `step`.

## Incidents worth not repeating

- **Live source edit poisoned a run.** The 04:03 df pair lost *every* eval to
  `TypeError: fig_error_vs_step() got an unexpected keyword 'split_bottom'`. `logging/viz` was imported at
  process start with the old signature; `evaluation/products` imports lazily at the first eval and got the
  new code — one process running two generations. It would have burned 46 GPU-hours emitting nothing, as a
  "FAILED non-fatally ... CONTINUING training" line. **Fix:** 2 consecutive failures of one routine is now
  fatal, plus `smoke/eval_products.py` covering that seam.
- **Epoch-0 evals were skipped** as an "untrained baseline" — wrong, `on_train_epoch_end` fires after a full
  epoch (6066 batches), and ep0 is the `p_tf=1` teacher-forced baseline every later epoch should be read
  against. Now evaluated.
- **`autobatch` miscalibrated for the AR path.** It probes at `p_tf=1` (61 GB) but the AR epochs ran at
  88.7/93 GB, over the 75 GB it targeted. Headroom raised 0.25 → 0.35.
- **Metrics that measured the wrong path.** `eval_ae_floor` certified 23.92 dB while the model ran at 20.41;
  `roundtrip` read 0.0 for every run; `kvcache/latent_max_abs_diff` became pure sampling noise the moment
  `stochastic_eval` defaulted true. All three were "correct code measuring a path the model never executes".
