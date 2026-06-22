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
- `…/trajectory_plot` — predicted vs true trajectory on the torus (plot)
- `…/rollout_video` — the same animated on the torus surface (video)
- `…/obs_image_video` — predicted vs ground-truth ego frames (image stage)

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
