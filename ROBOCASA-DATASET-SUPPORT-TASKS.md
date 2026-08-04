# RoboCasa Dataset Support — Development Tasks

## Status

**Implemented and locally validated on one RTX PRO 6000. Full training launch is pending.**

Validated on 2026-08-04:

- Hydra resolves the complete prescribed `robocasa_world_model` configuration.
- All 261 episodes load as aligned float32 16-D state / 12-D action sequences.
- The deterministic task-stratified split is 234 train / 27 validation episodes and 270,062 total windows.
- The selected eye-in-hand stream decodes and the full 13.2 GiB 128px cache is reusable.
- Focused model forward/backward, bounded Lightning, batch-32 memory, generic validation, W&B, and
  best/last checkpoint smokes pass on one RTX PRO 6000 (26.85 GB measured peak at batch 32).
- Existing model configs compose with the torus 6-D state / 2-D action defaults, and the modality regression
  smoke remains green.

Still pending from this checklist: remote Hugging Face snapshot parity and completion of the full from-scratch
training run. Checklist boxes below remain the reviewable acceptance inventory rather than a claim that every
remote/full-run condition has already completed.

This checklist scopes the minimum work required to train QuickDraw's existing, task-agnostic world model on
the certified RoboCasa Scene 4 dataset in either of these forms:

- local package root:
  `.cache/robocasa-data-generation/quickdraw-hf-robocasa-scene4-certified-4h`; or
- Hugging Face dataset repo:
  `madang6/quickdraw-robocasa-scene4-4h`.

The target end state is that selecting a RoboCasa **data config** is enough to adapt the external dataset to
QuickDraw's existing internal batch contract. This is dataset support, not a new task abstraction and not a
new world-model architecture.

## Scope lock

### In scope

- Read the existing LeRobot v2.1 RoboCasa package without rewriting its source files.
- Map `observation.state` to QuickDraw's internal `proprio` modality.
- Map the 12-dimensional `action` feature to the existing action-token encoder.
- Select exactly one of the three packaged RGB camera streams as the existing `image` modality.
- Split the single packaged `train` split into deterministic, disjoint train and validation episode sets.
- Compute normalization statistics from the derived train episode set only.
- Train any existing QuickDraw world-model method against the adapted batches.
- Validate and checkpoint using environment-neutral prediction metrics.
- Support both `data.root=...` and `data.hf_repo=...` through the same code path.
- Add smoke tests and usage documentation for the supported path.

### Explicitly out of scope

- No task-conditioning token, task embedding, or task-specific model head.
- No use of `task_index`, human annotation columns, rewards, or success predicates as model inputs or targets.
- No separate model per RoboCasa task.
- No RoboCasa simulator or `WorldEnv` implementation.
- No control, MPPI, online rollout, OOD-environment, reward-model, language-control, or policy-training work.
- No action retargeting, replay, re-rendering, or data regeneration.
- No multi-camera model in the first implementation.
- No changes to the certified source states, actions, videos, certificates, or Hugging Face package contents.

The eight RoboCasa activities are simply sources of transition coverage inside one offline trajectory corpus.
They are not QuickDraw tasks. Their episode labels may be consulted only when constructing a representative
validation holdout and reporting dataset inventory.

## Source data contract

The adapter must validate the package contract before allocating GPU tensors.

| Item | Required value |
|---|---|
| Dataset format | LeRobot v2.1 run root with one `train/` dataset |
| Episodes | 261 |
| Frames | 288,593 |
| FPS | 20 |
| State key | `observation.state` |
| State shape | `(16,)` |
| State source dtype | `float64`; cast to `float32` at the loader boundary |
| Action key | `action` |
| Action shape | `(12,)` |
| Action source dtype | `float64`; cast to `float32` at the loader boundary |
| Default image key | `observation.images.robot0_eye_in_hand` |
| Alternate image keys | `observation.images.robot0_agentview_left`, `observation.images.robot0_agentview_right` |
| Source image shape | `(256, 256, 3)` RGB video at 20 FPS |
| Video layout | `train/videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4` |
| Episode inventory | `train/meta/episodes.jsonl` |
| Feature and video templates | `train/meta/info.json` |
| Task text codebook | `train/meta/tasks.jsonl`; inventory/split use only, never a model input |

All packaged episodes are at least 615 frames, so every episode can produce a dense QuickDraw window with
the initial `P=8`, `F=64`, `L=72` settings. With all episodes included, the package contains 270,062 such
dense windows before creating the validation holdout.

## Internal batch contract — unchanged

The world model must continue receiving the same canonical batch structure it receives today:

```text
batch["obs_seq"]  float32 (B, P + F, 16)          normalized RoboCasa state
batch["act_seq"]  float32 (B, P + F, 12)          normalized RoboCasa action
batch["image"]    float32 (B, P + F, 128, 128, 3) RGB in [0, 1]
```

Source feature names must stop at the dataset adapter. `LitWorldModel` and the multimodal model should still
refer to the canonical internal names `proprio`, `action`, and `image`.

## Configuration work

QuickDraw training configuration is Hydra YAML, not `config.json`. Do **not** introduce an unrelated
training `config.json`. The JSON work required for this integration is limited to reading the LeRobot metadata
files and writing reproducibility artifacts into the training run directory, as specified below.

### 1. Add `conf/data/robocasa.yaml`

Add a data config with an explicit external-to-internal schema. The exact shape should be equivalent to:

```yaml
root: null
hf_repo: null
source_split: train

schema:
  state_key: observation.state
  state_dim: 16
  action_key: action
  action_dim: 12
  image_key: observation.images.robot0_eye_in_hand
  image_name: image
  image_size: 128
  fps: 20

validation:
  fraction: 0.10
  seed: 0
  stratify_by_episode_task: true
  minimum_per_group: 1

P: 8
F: 64
window_stride: 1
batch: 32
workers: 0
fast_gpu: true
```

Requirements:

- [ ] `root` and `hf_repo` retain their current mutually exclusive meanings.
- [ ] `source_split` names the physical LeRobot directory; both derived logical splits read `train/`.
- [ ] `schema.image_key` selects one packaged stream without renaming or copying videos.
- [ ] `schema.image_name` remains `image` so the existing model and Lightning batch contract stay unchanged.
- [ ] `schema.state_dim` and `schema.action_dim` are validated against `train/meta/info.json`.
- [ ] `schema.fps` is validated against metadata rather than used to reinterpret timestamps.
- [ ] `window_stride` remains `1`.
- [ ] Start with `batch=32`, matching the documented image/flow recipe; tune only after a measured GPU smoke run.
- [ ] Do not copy torus-only split, coloring, camera-FOV, or action-sampler fields into this config.

### 2. Make model dimensions data-driven

The model remains dataset-independent, but its input widths must come from the selected data schema.

- [ ] Add `state_dim: 6` and `action_dim: 2` to `conf/data/torus.yaml` so the existing dataset states its contract.
- [ ] Change existing multimodal model configs to source the widths from the data config:
      `action_dim: ${data.schema.action_dim}` and proprio `dim: ${data.schema.state_dim}`.
- [ ] Add the corresponding `schema` block to the torus data config so all current model choices still compose.
- [ ] Do not add `mm_flow_robocasa`, `mm_lsar_robocasa`, or other dataset-specific model classes.
- [ ] In `training/setup.py`, fail before model construction if configured dimensions differ from dataset metadata.

The intended first model is still `model=mm_flow`. Apply the repository's documented recipe through ordinary
model overrides or a model recipe config; do not encode it as RoboCasa behavior:

```text
d=128, depth=4, heads=8, window=32
F=64, batch=32, recon_frac=0.25
diffusion.shortcut=true, p_tf_end=0.0, p_tf_warmup_epochs=4, detach_every=16
image num_tokens=8, encode_arch=vit, decode_arch=vit, decode_kind=mse
action_head.enabled=false
```

### 3. Add one complete proposed-run config

Add `conf/robocasa_world_model.yaml` as the primary Hydra config for the first supported RoboCasa run. It
must compose the normal QuickDraw config and then select/lock the approved data, model, evaluation, trainer,
and single-GPU settings in one reviewable file. The intended structure is:

```yaml
defaults:
  - config
  - override /model: mm_flow
  - override /data: robocasa
  - override /eval: offline
  - _self_

experiment: robocasa-scene4

data:
  P: 8
  F: 64
  window_stride: 1
  batch: 32
  fast_gpu: true

model:
  d: 128
  depth: 4
  heads: 8
  window: 32
  action_dim: ${data.schema.action_dim}
  p_tf_start: 1.0
  p_tf_end: 0.0
  p_tf_warmup_epochs: 4
  recon_frac: 0.25
  detach_every: 16
  dynamics_detach_encoder: false
  grad_checkpoint: false
  diffusion:
    shortcut: true
    sampling_steps: 6
    predict: residual
    stochastic_eval: false
    time_sampling: uniform
    flow_hidden: 0
  action_head:
    enabled: false
    weight: 1.0
    shortcut: true
    detach_gradient: false
  modalities:
    - name: proprio
      kind: vector
      dim: ${data.schema.state_dim}
      weight: 1.0
      decode_kind: flow
      decode_param: x0
      decode_arch: mlp
    - name: image
      kind: image
      num_tokens: 8
      img_size: 128
      patch: 16
      ae_depth: 4
      weight: 1.0
      decode_kind: mse
      encode_arch: vit
      decode_arch: vit

trainer:
  devices: 1
  accumulate_grad_batches: 1
  monitor: val/loss/total
  monitor_mode: min

variations:
  noise_injection:
    std: 0.0
    observations_encoded_pre_fusion:
      scale: 0.0
      granularity: timestep
  physical_loss: {weight: 0.0, continuity: 0.0, warmup_epochs: 10}
  contraction: {weight: 0.0, target: 1.02, power_iters: 2, n_sample_steps: 4}
```

Requirements:

- [ ] `python -m quickdraw.train_world_model --config-name robocasa_world_model --cfg job` composes cleanly.
- [ ] The resolved config contains exactly the prescribed recipe; no required setting exists only in a shell
      command or prose comment.
- [ ] `trainer.devices=1` is honored by `train_world_model.py`; do not add DDP or distributed-loader work.
- [ ] The config initializes the complete model from scratch unless `checkpoint=` is explicitly supplied for a
      deliberate resume operation.
- [ ] Data location and the five run-summary strings remain CLI inputs because they identify a particular launch,
      not the reusable training method.
- [ ] This config is the source of truth for the first full run and is saved verbatim through the existing
      resolved-config artifact.

### 4. Add an offline evaluation config

Add `conf/eval/offline.yaml` for dataset-only world-model training.

- [ ] Set `during_train.every_epochs: 0` and `during_train.at_epochs: []`.
- [ ] Set every existing torus/control/manifold/denoising routine under `during_train.evals` to `false`.
- [ ] Do not make the RoboCasa data config silently select an evaluation config; invocation should state
      `eval=offline` until QuickDraw has a general experiment-composition group.
- [ ] Normal Lightning validation remains enabled; only environment-dependent evaluation callbacks are disabled.

### 5. Make checkpoint selection configurable

- [ ] Add `monitor` and `monitor_mode` fields to trainer configuration.
- [ ] Preserve the torus default:
      `monitor=val/metric/proprio/manifold_distance_error`, `monitor_mode=min`.
- [ ] For offline RoboCasa training use `monitor=val/loss/total`, `monitor_mode=min`.
- [ ] Replace the literal checkpoint metric in `train_world_model.py` with these fields.
- [ ] Verify that `best.ckpt` and `last.ckpt` are both written by a smoke run.

## JSON metadata and run-artifact work

### Source JSON files — read-only

| File | Required handling |
|---|---|
| `train/meta/info.json` | Validate format version, FPS, state/action dimensions, selected video key, and video path template. |
| `train/meta/episodes.jsonl` | Read authoritative episode indices, lengths, and top-level task descriptions for holdout stratification. |
| `train/meta/tasks.jsonl` | Optional lookup for readable inventory. Never feed codes or strings into the model. |
| `normalization_stats.json` | Validate finite source values, but do not use these full-corpus statistics for the derived holdout experiment. |
| `dataset_card.json` | Validate package status is `complete`; otherwise do not train. |
| `summary.json` | Record dataset identity/counts in the run manifest; do not depend on torus-only fields. |

Do not edit any of these files in a local package or Hugging Face snapshot.

### New run-local `dataset_split.json`

Write `<train_run>/dataset_split.json` before training. It must contain at least:

```json
{
  "schema_version": 1,
  "source": {
    "root": "<resolved local or HF snapshot path>",
    "hf_repo": "madang6/quickdraw-robocasa-scene4-4h",
    "source_split": "train"
  },
  "policy": {
    "fraction": 0.1,
    "seed": 0,
    "stratify_by_episode_task": true,
    "minimum_per_group": 1
  },
  "train_episode_indices": [],
  "val_episode_indices": [],
  "groups": {}
}
```

Requirements:

- [ ] The two episode lists are sorted, disjoint, and their union is all 261 episode indices.
- [ ] Split at the episode level before windowing; no episode or frame may appear in both sets.
- [ ] Group using the single top-level task description in each `episodes.jsonl` record.
- [ ] For each group, choose `max(minimum_per_group, round(group_size * fraction))` validation episodes,
      capped at `group_size - 1`, from a seeded deterministic permutation.
- [ ] The expected default partition is 234 train episodes and 27 validation episodes.
- [ ] Each of the eight top-level task descriptions must have at least one train and one validation episode.
- [ ] Include per-group episode and frame counts so the task imbalance is visible without becoming a model input.
- [ ] Re-running with the same dataset revision and seed must produce byte-identical episode lists.

### Run-local `normalization_stats.json`

Compute statistics after applying `dataset_split.json`, using train episodes only, and write them into the
training run directory. Preserve QuickDraw's canonical keys so existing `Normalizer` call sites continue to
work:

```json
{
  "observation_vector": {
    "mean": ["16 float values"],
    "std": ["16 positive float values"]
  },
  "action": {
    "mean": ["12 float values"],
    "std": ["12 positive float values"]
  },
  "source_keys": {
    "observation_vector": "observation.state",
    "action": "action"
  }
}
```

- [ ] Use an epsilon floor of `1e-6` for constant or nearly constant dimensions.
- [ ] Use the same train-derived statistics for both logical splits.
- [ ] Validate all means and standard deviations are finite and every standard deviation is positive.
- [ ] Do not use the package's existing full-corpus normalization file for training after deriving validation.
- [ ] Save the exact statistics used by the model next to the resolved config and checkpoints.

## Development tasks

### Phase 1 — Configurable LeRobot schema adapter

Primary file: `src/quickdraw/data/dataset.py`.

- [ ] Change `load_split_episodes` to accept source split, state key, action key, and an explicit episode-index set.
- [ ] Remove the literal `torus/<split>` repo identifier from the load path; repo ID is metadata only when `root`
      already identifies the local dataset.
- [ ] Read `episode_index`, configured state, and configured action columns.
- [ ] Cast state/action arrays to contiguous `float32` exactly once at the dataset boundary.
- [ ] Preserve variable episode lengths; do not pad and do not concatenate windows across episodes.
- [ ] Assert each selected episode length agrees with `meta/episodes.jsonl`.
- [ ] Assert all selected episodes are at least `P + F` frames and report any excluded episode explicitly.
- [ ] Keep `stack_windows` and `MMWindowLoader` model-facing output names unchanged.

Acceptance:

- [ ] A state-only load returns 261 episodes with shapes `(T, 16)` and `(T, 12)` before holdout filtering.
- [ ] Dense `P=8`, `F=64` windowing never crosses an episode boundary.
- [ ] Local-package and downloaded-HF loads return array-identical state/action sequences.

### Phase 2 — Configurable single-camera loading

Primary file: `src/quickdraw/data/dataset.py`.

- [ ] Replace the torus-only `observation.images.fpv` glob with the `video_path` template from
      `train/meta/info.json` and the configured image key.
- [ ] Decode videos in authoritative episode-index order, not incidental filesystem order.
- [ ] Validate every selected video has exactly the episode's declared frame count.
- [ ] Area-downsample 256×256 RGB to configured 128×128 RGB once, preserving uint8 storage.
- [ ] Name the decoded frame cache using dataset identity, selected image key, image size, and split-manifest hash;
      choosing a different camera must never reuse another camera's cache.
- [ ] Keep only one full uint8 frame store on the GPU and gather image windows by frame index as today.
- [ ] Continue converting gathered batches to float `[0, 1]` immediately before yielding them.
- [ ] Reject more than one configured image source with a clear message; multi-camera loading is deferred.

Acceptance:

- [ ] The default eye-in-hand loader sees exactly 288,593 frames before holdout filtering.
- [ ] A decoded frame batch has shape `(B, 72, 128, 128, 3)`, dtype float32, and range `[0, 1]`.
- [ ] State, action, and image frames share the same episode/frame indices.
- [ ] The raw resident frame store is approximately 13.2 GiB for the complete one-camera corpus.

### Phase 3 — Deterministic logical train/validation splits

Primary files: `src/quickdraw/data/dataset.py`, `src/quickdraw/training/setup.py`.

- [ ] Add one function that derives the episode partition from `episodes.jsonl` and validation config.
- [ ] Build the partition once, then pass the resulting index sets to both state-only and multimodal loaders.
- [ ] Never create train/validation partitions independently inside separate loader calls.
- [ ] Return inventory metadata with loaders: episode count, frame count, transition count, and window count.
- [ ] Replace `train_world_model.py`'s assumed fixed `cfg.data.splits.*.steps` inventory with the returned inventory.
- [ ] Persist `dataset_split.json` and train-only `normalization_stats.json` before Lightning starts.

Acceptance:

- [ ] Default split is 234 train / 27 validation episodes with no overlap.
- [ ] Repeated runs with seed 0 produce identical manifests and window counts.
- [ ] Changing only the split seed changes membership but not group totals or overall counts.

### Phase 4 — Environment-neutral training validation

Primary files: `src/quickdraw/training/lit.py`, `src/quickdraw/train_world_model.py`,
`src/quickdraw/training/setup.py`.

- [ ] Add a configured validation-metric mode: `torus` or `generic`.
- [ ] Preserve the current torus metrics byte-for-byte when mode is `torus`.
- [ ] In `generic` mode, do not call `manifold_distance_error`, `pointwise_error`, or
      `tangent_velocity_error`.
- [ ] Continue logging `val/loss/total` and normalized decoded state MSE.
- [ ] Add generic decoded state MAE for interpretability; image MSE/L1/PSNR already remain valid.
- [ ] Do not construct or require `TorusConfig` for a generic offline training run.
- [ ] Reject torus-only physical-loss variations when validation mode is generic.
- [ ] Use the configurable checkpoint monitor from trainer config.
- [ ] Run with `eval=offline` so no environment-dependent callback reads nonexistent OOD splits.

Acceptance:

- [ ] A 16-dimensional validation batch completes without shape/broadcast errors.
- [ ] Validation logs total loss, state MSE/MAE, and image MSE/L1/PSNR.
- [ ] The smoke run produces a best checkpoint selected by `val/loss/total`.
- [ ] Existing torus smoke tests and torus checkpoint-monitor behavior remain unchanged.

### Phase 5 — Hugging Face and runtime plumbing

Primary files: `src/quickdraw/training/setup.py`, `.env.template`, `.dockerignore`, documentation.

- [ ] Keep `resolve_data_root` as the one local/HF convergence point.
- [ ] Record the requested repo ID and resolved snapshot path in `dataset_split.json`.
- [ ] Ensure all initial training code uses the resolved root; no loader should read `cfg.data.root` directly.
- [ ] Use `HF_TOKEN` consistently in `.env.template` and code.
- [ ] Add `.cache/` to `.dockerignore`; the current workspace cache is roughly 250 GiB and must not enter the
      Docker build context.
- [ ] Document either `data.hf_repo=...` inside Docker or an explicit read-only local dataset bind mount.
- [ ] Do not require the omitted remote provenance `extras/` files for training; required metadata is present in
      `train/meta/`.

Acceptance:

- [ ] Public HF loading succeeds without a token.
- [ ] Private HF loading succeeds with `HF_TOKEN`.
- [ ] A Docker rebuild does not copy the workspace `.cache` tree.
- [ ] Local and HF smoke loaders produce identical split manifests, state arrays, actions, and selected frames.

### Phase 6 — Smoke and regression coverage

Add `src/quickdraw/smoke/robocasa_dataset.py` or equivalent focused tests.

- [ ] Metadata smoke: validate package status, version, FPS, dimensions, selected video key, and totals.
- [ ] Split smoke: validate the 234/27 deterministic partition, group coverage, and absence of overlap.
- [ ] Normalization smoke: train state/action normalize to approximately zero mean and unit standard deviation.
- [ ] Alignment smoke: compare selected episode/frame indices across state, action, and video.
- [ ] Loader smoke: fetch one state-only batch and one multimodal batch with exact expected shapes.
- [ ] Model smoke: build `mm_flow` with state dim 16/action dim 12 and complete one forward/loss/backward step.
- [ ] Trainer smoke: run two train batches and two validation batches on one GPU, with external eval disabled.
- [ ] Checkpoint smoke: verify `last.ckpt`, selected best checkpoint, resolved config, split manifest, and
      normalization stats exist.
- [ ] Regression smoke: existing torus data/model smokes still pass without config overrides.

## First supported launch

After the tasks above pass, the supported command should be:

```bash
uv run python -m quickdraw.train_world_model \
  --config-name robocasa_world_model \
  data.hf_repo=madang6/quickdraw-robocasa-scene4-4h \
  run_summary.problem='Establish the first RoboCasa offline world-model baseline' \
  run_summary.tried='Certified and validated the source trajectory package' \
  run_summary.trying='Train the existing QuickDraw multimodal flow world model on RoboCasa' \
  run_summary.trying_detail='Use 16-D state, 12-D action, one eye-in-hand camera, and an episode holdout' \
  run_summary.rationale='The adapter now presents the same canonical transition-window contract as torus data'
```

`conf/robocasa_world_model.yaml` is the reviewed source of truth for every method, data-shape, validation,
checkpoint, and device setting in this launch. The CLI supplies only dataset location and run identity.

## Definition of done

RoboCasa dataset support is complete when all of the following are true:

- [ ] The command above starts from either a local package or HF repo without modifying source data.
- [ ] The resolved config exactly matches `conf/robocasa_world_model.yaml` and the prescribed `mm_flow` recipe.
- [ ] The trainer uses exactly one GPU and does not initialize a distributed strategy.
- [ ] The existing world-model classes require no RoboCasa-specific branches.
- [ ] Batches contain aligned 16-D state, 12-D action, and one selected RGB camera.
- [ ] Train and validation are deterministic, episode-disjoint, and represented in run artifacts.
- [ ] Normalization is computed only from derived train episodes.
- [ ] Generic validation and checkpoint selection run without importing torus metrics.
- [ ] A bounded GPU smoke run completes forward, backward, validation, and checkpoint creation.
- [ ] Existing torus behavior remains unchanged.
- [ ] No task label, reward, success predicate, simulator, or controller is required anywhere in the core
      RoboCasa world-model training path.
