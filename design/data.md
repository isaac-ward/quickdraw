# Data — Datasets, Generation, Loading

How we generate, store, and load data for the shoot-out. Storage layer is the **`lerobot` package**
(`LeRobotDataset`): Parquet for low-dim, MP4 + torchcodec for images, JSON meta. This gives video
compression, fast windowed reads, and direct interop with real robot datasets (Open-X / LeRobot).

## Schema (locked)

What the model sees is an observation, not state:

| Key | Shape | lerobot feature |
|---|---|---|
| `observation_vector` | `(6,)` float32 | `observation.vector` |
| `observation_image` | `(3, 72, 128)` uint8 | `observation.image` (video) |
| `action` | `(2,)` float32 | `action` |

`observation_image` is absent until the image stage; the schema and loader do not change when it
arrives. Per-frame meta: `timestamp`, `frame_index`, `episode_index` (lerobot provides these).

## Dataset taxonomy

Train and val share the in-distribution params **A**. Eval splits long-horizon rollouts across
in-distribution and three OOD axes.

| Split | Distribution | # traj | steps/traj | Purpose |
|---|---|---|---|---|
| `train` | A | 1000 | 256 | fit models (sliced into windows) |
| `val` | A, new seeds | 128 | 256 | model selection |
| `eval_ood_horizon` | A, new seeds | 32 | 2048 | long-horizon, no shift |
| `eval_ood_visual` | A + recolor | 32 | 2048 | image generalization |
| `eval_ood_geometric` | new `(R,r)` | 32 | 2048 | manifold-shape generalization |
| `eval_ood_dynamics` | new `γ,a_max` | 32 | 2048 | dynamics generalization |

Train trajectories are short and many (broad coverage, ~193 windows each). Eval trajectories are
long (2048 steps ≈ 68 s @ 30 Hz) because long-horizon drift is the measured thing.

## Generation

1. `TorusEnv` steps `B` envs at once on GPU; actions from an OU process; per-episode seed
   `fold_in(base_seed, episode_id)` stored in meta.
2. Write each episode with `LeRobotDataset.add_frame` / `save_episode`.
3. Image stage: batched render → encode to MP4 per episode (lerobot's video writer).
4. Each split is this pipeline with its own params block.

Datasets are immutable and versioned (`datasets/torus/v1/`, lerobot on-disk format), generated once,
reused by every model — separate from training runs.

## Loading

- **Train / val (windowed):** `LeRobotDataset` with `delta_timestamps` covering past `P=32` and
  future `F=32` relative frames → one window per item. Batch keys `past_*` / `future_*`. torchcodec
  decodes the image window on demand (image stage). DataLoader: `num_workers>0`, `pin_memory=True`,
  `persistent_workers=True`, `prefetch_factor=4`.
- **Eval (full trajectory):** iterate whole episodes; feed the model the first `P` frames + the true
  `action` sequence; roll out 2048 steps; score with the `environment.md` errors.
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
