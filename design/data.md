# Data — Datasets, Generation, Loading

How we generate, store, and load data for the shoot-out. Storage layer is the **`lerobot` package**
(`LeRobotDataset`): Parquet for low-dim, MP4 + torchcodec for images, JSON meta. This gives video
compression, fast windowed reads, and direct interop with real robot datasets (Open-X / LeRobot).

## Schema (locked)

What the model sees is an observation, not state:

| Key | Shape | lerobot feature |
|---|---|---|
| `observation_vector` | `(6,)` float32 | `observation_vector` |
| `observation.images.fpv` | `(256, 256, 3)` uint8 (video) | `observation.images.fpv` |
| `action` | `(2,)` float32 | `action` |

The **FPV image** (egocentric RGB, rendered per trajectory and stored as MP4 / torchcodec-decoded) is
the image modality and is now generated on **every** run, aligned 1:1 with the vector frames. Per-frame
meta: `timestamp`, `frame_index`, `episode_index` (lerobot provides these).

## Dataset taxonomy

Train and val share the in-distribution params **A**. Eval splits long-horizon rollouts across
in-distribution and three OOD axes.

| Split | Distribution | # traj | steps/traj | Purpose |
|---|---|---|---|---|
| `train` | A | 256 | 256 | fit models (sliced into windows) |
| `val` | A, new seeds | 64 | 256 | model selection |
| `eval_ood_horizon` | A, new seeds | 64 | 1024 | long-horizon, no shift |
| `eval_ood_visual` | A + recolor | 64 | 256 | image generalization |
| `eval_ood_geometric` | new `(R,r)` | 64 | 256 | manifold-shape generalization |
| `eval_ood_dynamics` | new `γ, mass` | 64 | 256 | dynamics generalization |

Values above are the current `conf/data/torus.yaml`. `eval_ood_horizon` is the long-horizon split
(1024 steps) because long-horizon drift is the measured thing; the other eval splits are 256 steps and
each isolate ONE OOD axis (visual = recolor only, geometric = `(R,r)` only, dynamics = `γ, mass` only).

## Generation

1. `TorusEnv` steps all of a split's trajectories at once; **actions come from `data.action_sampler`** —
   `ornstein_uhlenbeck` (unimodal, default; deprecated alias `ou`) or `bimodal` (`BimodalActionSampler`: a two-basin
   action-**magnitude** process, temporally smoothed, leaning to the low-thrust basin). An 8-tile
   **`media/action_distribution.png`** preview of the action distribution over time is rendered every run.
2. **FPV render (the heavy step, decoupled for parallelism):** each trajectory's 256×256 egocentric
   clip is rendered offscreen (OSMesa) at full worker parallelism, then ingested as `observation.images.fpv`.
3. Write each split's `LeRobotDataset` (`add_frame` / `save_episode`), then meta + `summary.json`, then
   stitch per-split composite grids.
4. Each split is this pipeline with its own trajectory count / steps / seed / env overrides.

Each generated dataset is a `logs/data_generation_<timestamp>_<experiment>/` run folder (lerobot on-disk
format, one sub-dataset per split + `media/` + `summary.json`), reused by every model via `data.root=`.
The canonical copy is published to the HF Hub (`isaac-ronald-ward/torus-world`) via
`quickdraw.push_to_hub`, which clears-and-reuploads the whole folder on every push.

## Loading

- **Train / val (windowed):** windows cover past `P=8` and future `F=64` frames → one window per item
  (`conf/data/torus.yaml`). The default loader is **GPU-resident** (`data.fast_gpu`): all windows held
  on-device and batched by index (no per-item torchcodec workers) for throughput; the lerobot
  `delta_timestamps` path is the CPU alternative. Each batch carries the past context + the `F`-step target.
- **Eval (full trajectory):** iterate whole episodes; feed the model the first `P` frames + the true
  `action` sequence; roll out the split's horizon (1024 for `eval_ood_horizon`); score with the
  `environment.md` errors.
- **Normalization:** compute mean/std on `train` only; apply those stats to every split (including
  OOD), so OOD shift stays real. Override lerobot's per-dataset stats with the train stats.

## Run folders (provenance, seamstress parity)

```
logs/run_<workflow>_<YYYY_MM_DD_HH_MM_SS>/<model_name>/
├── checkpoints/config.resolved.yaml   # resolved OmegaConf
├── normalization_stats.json           # train-only stats used
└── ...                                 # (logging.md)
```
A run records the dataset `name@version` it used; the dataset records its generation params. Both
reproducible from the record alone.

## Milestones

1. `TorusEnv` + OU actions; verify every true trajectory has `manifold_distance_error ≈ 0`.
2. Generation → `LeRobotDataset` for all splits; verify per-episode seed regenerates identical data.
3. Train-only normalization + windowed `delta_timestamps` loader; verify train stats ≈ N(0,1).
4. Full-trajectory eval loader + rollout scoring on `eval_ood_horizon`.
5. OOD splits (visual / geometric / dynamics).
6. Image stage: render → MP4; torchcodec decode flows through the unchanged loader.
