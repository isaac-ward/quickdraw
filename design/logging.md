# Logging — W&B + local artifacts

W&B via Lightning's `WandbLogger`, driven by one `LoggingCallback`. Every scalar, plot, and video is
written to the run folder **and** sent to W&B through one helper, so the two never diverge. Same
metric keys across splits so the workspace auto-aligns.

## Cadence

| What | When |
|---|---|
| `train/` scalars, `diag/` | every 50 steps |
| `train/` media (windowed report) | once per epoch, fixed sample |
| `val/` scalars + media | every validation (per epoch) |
| subscribed eval routines (`long_horizon`/`ood`/`control`) | every `during_train.every_epochs` |
| checkpoints, config, norm stats | run start + per checkpoint |

## The rollout report (shared block)

For any rollout — a training window or a full eval trajectory — log the same set. Each plot carries a
caption = equation + one sentence (the definitions in `environment.md`):

- scalars: `manifold_distance_error`, `pointwise_error`, `tangent_velocity_error` (mean over rollout)
- `…/error_vs_step` — the three errors vs rollout step (plot)
- `…/trajectory_plot` — predicted vs true trajectory (torus atlas if geometry; else a geometry-free 3D path — GT black / pred grey / context black-dashed — for any env with an explicit `position_idx`) (plot)
- `…/trajectory_axes` — per-axis position vs step (GT vs pred), the readable companion to the 3D path (plot, geometry-free envs)
- `…/rollout_video` — the same animated on the torus surface (video)
- `…/obs_image_video` — predicted vs ground-truth ego frames (image stage)

## Loss keys (renamed 2026-08-10)

ROLE first; the decode PARAMETERIZATION never appears in a name. Under `train/loss/` and `val/loss/`:

| key | was | what it is |
|---|---|---|
| `dynamics/latent` | `flow/latent` | the transition function (teacher-forced rectified-flow term) |
| `dynamics/latent_shortcut` | `shortcut/latent` | its self-consistency term (`diffusion.shortcut`) |
| `decode/<mod>` | `<mod>` (mse) or `flow/<mod>` (flow) | that modality's decode recon |
| `decode/<mod>_shortcut` | `shortcut/<mod>` | flow decoders' self-consistency |
| `codec/roundtrip_<mod>` | `roundtrip/<mod>` | adapter encode->decode identity; NOT a decode loss |
| `action/flow`, `action/shortcut` | `flow/action`, `shortcut/action` | action-head prior |

The old scheme emitted `image` but `flow/proprio` for the SAME role, and grouped `flow/latent` (the
dynamics) with `flow/proprio` (a decoder) under one `flow/` panel. The three groups now correspond to the
three things that actually compete in the objective: dynamics vs decode vs codec.

`recon_losses` returns `(losses, weights)` -- the same contract as `loss_terms` -- so no caller infers a
weight by parsing a key. That parsing was a live bug: `wts[k.split("/")[-1]]` mapped `roundtrip/image` to the
IMAGE DECODE weight, so ablating a head's decode recon also silently deleted its adapter supervision.

NOTE this forks every series: runs before 2026-08-10 log `flow/latent`, after log `dynamics/latent`. Charts
spanning the rename split silently rather than erroring.

## Chart families

- **`train/`** = rollout report on one window (`P=32→F=32`) each epoch, plus `train/loss_total`,
  `train/loss_obs_vector`, `train/loss_obs_image` (image stage).
- **`val/`** = exactly `train/`, on val data. Drives model selection.
- **`eval/long_horizon/`** = in-distribution full-trajectory open-loop rollout report, plus summary
  scalars `…/manifold_distance_error@500`, `@1000`, `@2000`, `…/manifold_distance_error_auc`.
- **`eval/ood/<split>/`** = the same rollout report per OOD split (`ood_visual`, `ood_geometric`,
  `ood_dynamics`), each scored on its own geometry.
- **`eval/control/`** = closed-loop MPPI to the 16 torus targets (routine in `training.md`).
  Per target `eval/control/<target>/`: `time_to_completion`, `final_distance`, `success`,
  `distance_to_target_vs_step` (plot), `control_video`. Aggregate `eval/control/`: `control_hz`,
  `success_rate`, `mean_time_to_completion`, the **target atlas** figure, and a 16-row summary table.
  The aggregate scalars are the control scoreboard for the shoot-out.
- **`diag/`** = `lr`, `grad_norm`, `weight_norm`, `throughput_samples_per_s`, `epoch_time_s`,
  param/grad histograms.
- **`data/`** = `normalization_stats` table, `obs_vector_hist`, `action_hist`, `dataset_card` artifact.

Eval logs the same things as train; it differs only by: full trajectory not a window, repeated across
cases, summary scalars added, losses dropped.

Shoot-out: tag each run and group runs in W&B; the shared `eval/*/manifold_distance_error@*`
scalars drive automatic comparison across runs × cases.

## `diag`/`grad/` — and the one number that actually screams

`grad/norm_preclip`, `grad/norm_postclip`, per-module `grad/norm/<part>`, `grad/num_nans`, `grad/num_infs` and
`grad/nonfinite_skipped` have existed since the beginning. On 2026-08-11 a **767x gradient explosion in the
transformer denoiser went unnoticed for two days** with all of them logging correctly, because the pair reads
as two unremarkable numbers unless you divide them.

**`grad/clip_ratio` = norm_preclip / gradient_clip_val is the number to watch.**

| value | meaning |
|---|---|
| ~1 | clipping never engages. Healthy. |
| a few | clipping engages sometimes. Normal for a spiky loss. |
| >> 1 | **the update is direction-only and the magnitude is junk.** Clipping discards the norm but keeps the DIRECTION, which is dominated by whatever exploded, so the optimizer takes full-size confident steps into garbage. The run degrades SMOOTHLY instead of NaN-ing -- it looks like a modelling failure, not an optimizer one. |

`norm_postclip` pinned at exactly the clip value while `preclip` grows is the signature. Watch clip_ratio's
TREND, not any single value -- a legitimately spiky loss can sit above 1 and train fine, so what matters is
whether it is climbing. Usual causes when it is: a residual branch that is not zero-init'd, too long a
BPTT x ODE chain (detach_every x sampling_steps), or too high an LR.

MEASURED for reference -- the same config, two denoisers:

```
grad/norm/flow        ep0    ep1     ep2      ep3
mlp                  0.43   0.85    0.49     0.59     stable, <1 throughout
transformer          0.98   1.48  766.36    21.21     exploded; clip_ratio 767
transformer+bespoke  2.10  65.62 10151.31     -
```

## `normalization/` — how the latent is made scale-free, and whether it is drifting

One folder describing the ACTIVE latent-normalization mechanism (`model.latent_norm`) and the statistic it
acts on. Logged at fit start AND **every validation epoch**, probed on a FIXED set of 8 val frames so the
numbers are comparable across epochs (a moving probe set would make a trend meaningless).

| key | meaning |
|---|---|
| `normalization/is_layernorm` | 1.0 when `latent_norm: layernorm` — per-token non-affine LN on the carried bag |
| `normalization/is_affine` | 1.0 when `latent_norm: affine` — fixed per-channel scale+shift on the AE latent |
| `normalization/is_invertible` | 1.0 for `affine`/`none`. `layernorm` is 0: it DISCARDS 2 scalars per token |
| `normalization/<mod>/pre_norm_std_mean` | mean per-token std of the encoded bag BEFORE normalization |
| `normalization/<mod>/pre_norm_std_min` | the smallest such std — the first token to degenerate |
| `normalization/<mod>/pre_norm_absmean_mean` | mean \|per-token mean\| before normalization |
| `normalization/<mod>/mean_c{i}`, `std_c{i}` | `affine` ONLY: the calibrated per-channel parameters (frozen after fit start; re-logged each epoch so the folder is self-contained) |
| `normalization/<mod>/n_frames` | frames the affine calibration used |

**Read `ln_gain_max` as a collapse tripwire.** Under `layernorm`, `_ln` divides every token by its own
std. If the encoder — or, in rollout, the dynamics — drifts toward emitting near-constant tokens, that std
falls and LayerNorm amplifies whatever remains by up to `1/sqrt(eps)` ~ 316x. That is a positive feedback
loop, and it is the leading suspect for the un-diagnosed epoch-5 collapse of `taesd_exact` (val PSNR
18.39 -> 11.63 in one epoch, never recovered). A falling `pre_norm_std_mean` should be visible BEFORE the
loss moves. Under `affine` the mechanism does not exist, so the series is informational only.

The probe is fail-soft but NOT silent: if it raises, it logs `[latent_norm] per-epoch probe disabled (...)`
once to progress.log. A bare `except: pass` here previously hid a real bug for a full verify cycle.

`progress.log` additionally carries a one-line `[latent_norm] <type>: <what it costs>` at startup.

## Image metrics: LPIPS sits alongside psnr/ssim/mse/l1

`evaluation/openloop.image_curves` returns `{psnr, ssim, mse, l1, lpips}` per timestep, and EVERY consumer
picks the new key up automatically (`emit_horizon_readouts` iterates the dict; `products.log_error_curves`
plots it) — so it appears in both `eval_ood_horizon/` and `eval_ae_floor/` with no per-call-site change.

**LOWER lpips is better**, unlike psnr/ssim. It exists because every other image metric here is pixelwise
and therefore cannot distinguish a prediction blurred toward the dataset mean from one that is sharp but
wrong — the two failures need opposite fixes, and long-horizon rollouts on this repo look like the former.
Measured separation on synthetic inputs: heavy blur 0.7635 vs sharp-but-noisy 0.0159.

Backbone is SqueezeNet (cheapest of the three LPIPS variants, ~0.1 GFLOP/frame at 128px — negligible beside
the rollout that produced the frames), cached per device, and fail-soft: if the weights cannot be fetched
the key is simply absent rather than the eval dying.

## Collapse diagnostics (latent models only)

A `collapse/` panel describing **this run's** latent health (one model — not a cross-model view).
Logged **once per validation epoch** — never a per-step bar plot, and not worth a 50-step cadence.
These read the run's latent batch stacked as `Z ∈ ℝ^{N×dz}` (`N` = batch × timesteps) **directly,
bypassing the decoder**, so they reveal collapse the obs-space metrics can hide (a strong decoder can
flatter a partially-collapsed latent — see `models/latent_space_autoregressor.md`). Each plot carries
a caption = definition + one sentence, in the same style as the rollout report's `error_vs_step`.

Let `C = (1/(N−1))·Z̄ᵀZ̄` be the latent covariance (`Z̄` = mean-centered `Z`), `σⱼ = std(Z[:,j])`.

- `collapse/effective_rank` — participation ratio `PR = (Σλᵢ)² / Σλᵢ²` of `C` (`λᵢ` its eigenvalues),
  vs epoch. Caption: *how many of the `dz` dims carry variance; `PR→1` = all variance in one direction
  (collapse), `PR→dz` = fully used.*
- `collapse/per_dim_std` — the per-dimension stds `σⱼ`, sorted (bar plot). If **this run's mechanism
  defines a variance floor** (only VICReg's `γ`), draw it as a reference line; otherwise just the bars.
  Caption: *std of each latent dim; dims pinned near 0 are dead.*
- `collapse/offdiag_cov_mass` — mean absolute off-diagonal of the **correlation** matrix
  `Corr_ij = C_ij/(σᵢσⱼ)` (diagonal = 1, entries in `[−1,1]`), i.e. `mean_{i≠j}|Corr_ij|`, vs epoch.
  Caption: *redundancy between dims; →0 = decorrelated, high = dims duplicate each other (a precursor
  to falling effective rank).*
- `collapse/l_pred` — the (standardized) latent prediction loss vs epoch. Caption: *near-zero `L_pred`
  alongside high obs-space error is the collapse signature — the target became trivially predictable.*

The **collapse signature** across the panel: `L_pred ≈ 0` ∧ obs error high ∧ effective rank low.
**Mechanism-specific extras** live under the mechanism's namespace (a run is one mechanism): EMA logs
`collapse/ema/online_gap` (`‖enc − enc_ema‖`); RSSM (milestone 2) logs
`collapse/rssm/{dyn_kl, rep_kl, posterior_entropy}`. Keys are stable, so overlaying several shoot-out
runs later aligns the panels for free — but the spec above is for a single run.

The **physical-loss variant** (`λ_phys`, see `models/data_space_autoregressor.md`) adds the scalar
`train/loss_phys` (and `val/loss_phys`) — a loss term, not a collapse diagnostic.

## Rendering media cheaply

Toroid visuals are matplotlib 3D — no game engine. Precompute the torus mesh once and reuse it.
`Agg` backend, `Axes3D`, translucent surface tinted with the rainbow hue, true vs predicted 3D
lines + markers. The static plot is logged via `wandb.Image(fig, caption=...)`. The rollout video
animates the same figure over subsampled steps (2048 → ~200 frames, ~480p) via `imageio` → MP4 and
`wandb.Video`. At the image stage, `obs_image_video` tiles model output beside torchcodec-decoded
ground truth — no rendering. The static plot is one frame of the video; one shared helper.

**Control media.** The **target atlas** is `GridSpec(4,3)` — top 3×3 merged = large isometric view
labeling all 16 targets and their reached paths; bottom row = 1×3 axial views (XY, XZ, YZ). Each
**control video** is the true closed-loop trajectory + target marker + running distance. Same torus
helper as above.

## Local mirror

```
logs/run_<workflow>_<YYYY_MM_DD_HH_MM_SS>/<model_name>/
├── checkpoints/{config.resolved.yaml, top_k=<N>/, last.ckpt}
├── normalization_stats.json
├── metrics.csv          # every scalar
├── plots/               # trajectory_plot / error_vs_step PNGs
├── media/               # rollout / obs_image MP4s
└── summary.json         # final eval summary scalars per case
```

## Provenance

At run start: `wandb.config = OmegaConf.to_container(cfg, resolve=True)`, same written to
`config.resolved.yaml`; log `data/dataset = "torus@v1"` and the dataset card. Rollout scoring and
torus plotting import the `environment.md` functions — one source of truth.
