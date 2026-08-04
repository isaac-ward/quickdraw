# Training on the owm-envs ISS docking datasets

The `outofthisworldmodel-envs` project generates ISS docking-transition datasets in
quickdraw's LeRobot layout and publishes them to the Hub. They are **recorded data**: there is no ISS
simulator inside quickdraw, so the world model trains purely on the dataset. Everything runs through the
existing `name: recorded` path (`src/quickdraw/environments/recorded.py`) — no new code.

Each dataset ships `train` + `val` splits written with the repo prefix `iss` (`iss/train`, `iss/val`), a
camera stream named `fpv`, and 20 Hz transitions (`dt = 0.05`). Actions are 6-dim body-frame
`[force(3), torque(3)]`.

## Config groups

| Group | Observation | Contents |
| --- | --- | --- |
| `environments=owm_iss` | 13 | chaser state: `pos(3), vel(3), quat(4), body rate(3)` |
| `environments=owm_iss_goal` | 25 | the 13-dim state + goal errors `pos(3), vel(3), attitude(3), rate(3)` |
| `data=owm_iss` | — | `repo_id: iss`, `cam: fpv`, P/F/batch, and the `hf_repo` hook |

## The environment config does NOT set the model's width

This is the one thing that will bite you. `environments.obs_dim` sizes the `WorldEnv`; the model's input
width comes from the **model** config (`_modality_specs` reads `model.modalities[].dim`, `build_model`
reads `model.action_dim`). Nothing cross-checks the two, so selecting only `environments=owm_iss` leaves
the torus defaults (`dim: 6`, `action_dim: 2`) in place and you get a linear-layer shape error, not a
readable config error. Always pass both model overrides:

```
model.modalities.0.dim=13   # or 25 for owm_iss_goal
model.action_dim=6
```

## Train

13-dim state:

```bash
uv run python -m quickdraw.train_world_model experiment=iss \
    environments=owm_iss data=owm_iss data.hf_repo=<namespace>/owm-iss-<variant> \
    model.modalities.0.dim=13 model.action_dim=6 \
    eval.during_train.every_epochs=0 'eval.during_train.at_epochs=null' \
    +run_summary.problem=... +run_summary.tried=... +run_summary.trying=... \
    +run_summary.trying_detail=... +run_summary.rationale=...
```

25-dim state + goal errors — identical but for the two dim-carrying flags:

```bash
uv run python -m quickdraw.train_world_model experiment=iss_goal \
    environments=owm_iss_goal data=owm_iss data.hf_repo=<namespace>/owm-iss-<variant> \
    model.modalities.0.dim=25 model.action_dim=6 \
    eval.during_train.every_epochs=0 'eval.during_train.at_epochs=null' \
    +run_summary.problem=... +run_summary.tried=... +run_summary.trying=... \
    +run_summary.trying_detail=... +run_summary.rationale=...
```

`data.hf_repo` downloads and caches the dataset repo; use `data.root=<local dir>` instead to train from a
local copy. The `run_summary` fields are mandatory and must be unique per run (`train_world_model` fails
before any setup otherwise). `best.ckpt` monitors `val/metric/proprio/pointwise_error` — `RecordedEnv`
declares no `checkpoint_metric`, so the generic full-observation L2 is used.

## Keep these off

- **The during-train eval routines** (hence `eval.during_train.every_epochs=0` +
  `at_epochs=null` above). Two independent reasons, each covering routines that are on by default.
  `control` plans with MPPI, which reads `env.a_max` and calls `env.step` — neither exists on
  `RecordedEnv`, which raises `NotImplementedError` by design. Separately, `ood_horizon` and `manifold`
  load their episodes from `cfg.data.root` directly (`evaluation/routines.py:89,218`) rather than the
  resolved HF snapshot, so they cannot find the data when it comes from `data.hf_repo`. Train and val run
  normally regardless; only the extra routines are disabled.
- **`variations.physical_loss`** (already `0.0` by default — leave it). It evaluates *torus* geometry
  against `R`/`r`, which on a recorded env are the inert `1.0` placeholders from `RecordedConfig`. On ISS
  data it does not merely produce a meaningless number, it raises: `physical_state` decodes a readout at
  the proprio modality's own width (13 or 25), and the tangent term multiplies `obs_phys[..., 3:]` by a
  3-component surface normal (`training/variations.py:104`), so the two do not broadcast.

## `data.splits` is a log, not a loader

`conf/data/owm_iss.yaml` carries `splits` only because `train_world_model` prints a startup data inventory
(trajectories / frames / hours / windows) from it. The loaders read the dataset itself, so if the published
counts differ from the values there, the inventory line is wrong but the run is unaffected. The committed
numbers mirror owm-envs `configs/generation_default.yaml`, where `steps` is that config's `max_steps` cap —
real docking episodes terminate early, so actual episodes are typically shorter.
