# Reading the outputs

Every entrypoint writes a run folder `logs/<step>_<timestamp>_<experiment>/`. All logging goes through one
writer that fans out to **wandb and a local mirror identically** — same tags, same step (= epoch):

- scalars → wandb + `<run>/logs/metrics.jsonl`
- config → wandb + `<run>/logs/config.json`
- figures/videos → wandb + `<run>/logs/epoch_<i>/<tag>.{png,mp4}` (epoch first, then the tag path)

So anything you see on wandb is browsable on disk at the same tag path, and vice versa.

## Eval metrics (scalars)

**The three rollout errors** (proprio, defined once in `environments/torus.py`; logged as
`error_vs_step_avg_{linear,log}` curves + `<metric>_mean` scalars under `eval_ood_horizon/` and each OOD
split's own prefix):

- **`manifold_distance_error`** — `|signed_dist(p̂)| / r`: perpendicular distance off the torus surface,
  dimensionless (in tube-radii). 0 = the prediction stays ON the data manifold; this is the long-horizon
  *consistency* metric — a model can drift along the surface and still score 0 here.
- **`pointwise_error`** — `‖p̂ − p‖` in raw physical units: distance to the true point at the same step.
  The *accuracy* metric; grows with horizon for any imperfect model.
- **`tangent_velocity_error`** — `|⟨v̂, n̂⟩| / v_scale`: the predicted velocity's off-surface component,
  dimensionless. 0 = motion stays tangent to the manifold.

**OOD-horizon future-step scalars** (image heads): each image stat (`psnr`/`ssim`/`mse`/`l1`) is read out
at quarter-horizon steps into the future plus the full horizon, one scalar apiece —
`eval_ood_horizon/<head>/<stat>/@+<x>` (`x` = steps ahead). This makes the accuracy DECAY vs rollout depth
trackable across epochs in wandb without opening the curve figures.

**Control** (`eval_control/`): `{true,pred,diff}/{success_rate, mean_goals_reached,
mean_steps_to_complete, mean_seconds_to_complete}`. `true` is the oracle — MPPI planning with the REAL
dynamics — i.e. the ceiling for this env + planner budget; `pred` is the same MPPI planning through the
world model; `diff` is the gap. Read `pred/mean_goals_reached` against the oracle's: a healthy world model
closes most of the gap.

**Action distribution** (`eval_action_distribution/true_pred_w1`): 1D-Wasserstein between the pooled true
(recorded) and head-sampled action-magnitude distributions. Lower = the learned prior matches the data's
behavior; a bimodal dataset with a unimodal head shows up here before you ever open a plot.

## Eval-viz videos and figures

**Open-loop rollout** (`eval_ood_horizon/`, per-episode for the first `n_plot`): `trajectory_plot_<i>`,
`trajectory_video_<i>`, `scene_<i>` — a BLACK agent on the true path and a GREY agent on the model's
predicted path; they share the context, then diverge at the fork step. Watch WHERE the grey agent fails:
leaving the surface (manifold error), lagging/leading on the surface (pointwise error), or both.
Image heads add `<head>/rollout_<i>` (pred-vs-true video) and `<head>/filmstrip_<i>` (pred top / GT bottom
at sampled steps; the raw frames are also saved as `raw_filmstrip_frames_<i>.npz` so sharpness can be
judged at native resolution). For an env without `render_diagnostics`, this filmstrip IS the eval-viz.

**Control** (`eval_control/`): `control_video_<i>` per episode plus `control_video_combined` (all agents
on one torus). Black = oracle planner, grey = learned planner, gold ring = current goal. In language mode
add `reward_trace_<i>.png` — four lines, {imagined, achieved} x {reward head, ground truth}: imagined
tracking achieved means the WM's imagination is calibrated; the reward-head line tracking ground truth
means the language head is grounded.

**Manifold** (`eval_manifold/<method>/latent_space_to_{2,3}d`): the carried latent space projected by PCA
(global geometry), UMAP (neighborhoods), and t-SNE (local clusters). A healthy latent recovers torus-like
structure; a collapsed one degenerates to a blob or filament.

**Action-distribution plots** (produced by `train_action_model`, which runs this eval on its trained
head; also standalone via `python -m quickdraw.eval_action_distribution checkpoint=<run> data.root=<data>`):
`by_state_{true,pred}` (|a| histograms split by ambient-x state and time), `animation_pooled` and
`animation_byx` (recorded actions in green vs head samples in red, over time). For bimodal data, check
that BOTH magnitude basins appear in red and that the by-x split keeps each basin's mode crisp.

## Where each step's outputs land

| producer | scalars/plots under | notes |
|---|---|---|
| `train_world_model` | `train/`, `val/`, plus every in-loop eval's prefix | in-loop evals at epochs {5, 10, 20, 40, ...}, media in `logs/epoch_<i>/` |
| standalone `eval_*` | the routine's prefix, at `epoch_0000/` | each writes its own `logs/eval_<name>_<ts>_<exp>/` run dir + `summary.json` |
| `train_action_model` | `eval_action_distribution/` | full-model checkpoints (frozen WM + head) |
| `train_reward_model` | grounding/probe plots in its run dir | plus `reward_head.pt` |
| `eval_control` (language) | `eval_control/` + latent-space animation subdirs | see step 7 of [docs/workflow.md](workflow.md) |
