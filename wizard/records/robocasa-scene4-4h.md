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
