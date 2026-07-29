# Gym-environment refactor — make the codebase env-agnostic

Turn torus-world into a proper Gymnasium environment, and generalize the pipeline so **any** gym env can be
plugged in: mine play data with a policy → push to HuggingFace → train the world model on the HF dataset →
evaluate control in the real env (using its reward). Torus becomes the reference implementation.

## Target workflow (what a user does)
```
1. bring a gymnasium.Env (or use TorusWorld-v0)      # reset/step/spaces/reward/render(rgb_array)
2. python -m quickdraw.data_generation env=<name> policy=<random|bimodal|...>   # mine episodes -> lerobot/parquet
3. python -m quickdraw.push_to_hub data.root=<run> +hub.name=<name>            # -> HF dataset (parquet+mp4)
4. python -m quickdraw.train_world_model data.hf_repo=<user>/<name>            # train on the HF dataset (remote)
5. eval_control runs the LEARNED plan in the REAL env, scored by env.reward     # MBRL control eval
```

## Guiding constraints
- **Torus outputs stay bit-identical.** The refactor is a re-plumbing behind an interface; the torus data,
  renders (proven by `smoke/render_golden`), metrics, and run dirs must not change. New abstraction, same numbers.
- **General via config, specific via implementation.** One `WorldEnv` interface; torus + a generic gym adapter
  implement it. No env-specific `if` branches in the pipeline.
- **We wrap gym, we do not fork it.** Gymnasium already provides `reset`/`step`/spaces/`reward`/`render`.

## Core abstraction — `WorldEnv` (environments/base.py, NEW)
A minimal **batched** protocol the whole pipeline talks to (batched because data-gen + internal rollouts are
batched-torch — a big speed win; gym is single-env, adapted in Phase 6):
```python
class WorldEnv(Protocol):
    action_dim: int
    obs_dim: int                       # proprio/state vector width
    def reset(self, generator) -> Tensor: ...            # (B, obs_dim)
    def step(self, action: Tensor) -> Tensor: ...        # (B, obs_dim)
    def reward(self, obs, goal=None) -> Tensor: ...      # (B,) — for control eval (gym: env reward)
    def render_obs(self, obs) -> Tensor: ...             # (B,H,W,3) THE IMAGE MODALITY (model input; FPV for torus)
    def render_diagnostics(self, obs, pred=None) -> dict[str, Tensor]: ...  # {view: (B,H,W,3)} EVAL-VIZ ONLY, optional
```
- **`render_obs`** = the image modality (always present; for torus = FPV). **Never skipped.**
- **`render_diagnostics`** = the eval-video multi-view (torus: `{scene, axial_x, axial_y, axial_z}`); OPTIONAL —
  a generic gym env returns `{}` or just its single `render()`, and eval-viz falls back gracefully. This keeps
  the modality (what the model sees) cleanly separated from diagnostics (what we watch).

## Phases

### Phase 1 — interface + TorusEnv implements it + register `TorusWorld-v0`
- Add `environments/base.py` (`WorldEnv` protocol) and `environments/registry.py` (`make_env(name, cfg)`).
- `TorusEnv` (already has `reset`/`step`) → implement `reward` (goal-distance, from the current control logic),
  `render_obs` (wrap `viz.fpv_frames` / the fast FPV plotter), `render_diagnostics` (wrap the pyvista scene +
  axial renderers). Pure re-plumbing — call the existing byte-identical renderers.
- Register a single-env **`TorusWorld-v0`** subclassing `gymnasium.Env` (thin wrapper over batch-1 `TorusEnv`)
  for the public API + control eval.
- `conf/environments/torus_world.yaml` names the env class + geometry (R, r, dt, gamma, a_max, …).

### Phase 2 — generalize data generation (policy-driven, env-agnostic)
- `data/generate.py`: `generate_episodes(env_cfg: TorusConfig, …)` → `generate_episodes(env: WorldEnv, policy, …)`.
  Roll ANY `WorldEnv` with a pluggable **policy** (default `random`; `bimodal`/`ou` from the current samplers;
  user-supplied). Collect `(obs, action, reward, render_obs frames)` → `write_lerobot_split` (unchanged; already
  parquet+mp4). `conf/data/policy=<...>`.
- `data_generation.py`: build the env from config instead of hardcoding torus; the FPV render step becomes
  `env.render_obs`. Torus path stays byte-identical (same renderer, same frames).

### Phase 3 — train on the HF dataset (remote)
- `data/dataset.py`: `load_split_episodes[_mm]` currently does `LeRobotDataset("torus/<split>", root=<local>)`.
  Add a `data.hf_repo` option → `LeRobotDataset("<user>/<name>", ...)` which lerobot downloads+caches from HF.
  Local root stays the default (bit-identical). One small branch, no new datamodule.

### Phase 4 — control eval via the env's reward
- `eval_control` currently plans MPPI (learned vs oracle) toward torus **goals** with a goal-distance cost.
  Generalize: the cost/return uses **`env.reward`** (gym-native); "oracle" = MPPI with the true `env.step`,
  "learned" = MPPI with the WM. Torus keeps its goal-sequence reward (implemented as `TorusEnv.reward`), so
  torus control numbers are unchanged; a generic env brings its own reward.

### Phase 5 — rendering split (already designed into the interface)
- Eval-viz (`emit_openloop`, filmstrips, rollout videos) call `render_diagnostics` when available, else
  `render_obs`. Torus returns its rich `{scene, axial_*}` → identical videos. Generic env → single-view video.

### Phase 6 — `GymBatchAdapter` (bring your own gym env)
- `environments/gym_adapter.py`: wrap any `gymnasium.Env` into `WorldEnv` — vectorize B copies (loop or
  `gymnasium.vector`), map `observation_space`→`obs_dim`, `action_space`→`action_dim`, `step`→reward+obs,
  `render(rgb_array)`→`render_obs`. `render_diagnostics` = `{}` (falls back to the single view). This is the
  "provide your own env" entry point.

## Naming (proposed — needs confirmation)
- gym env id: **`TorusWorld-v0`** (Gymnasium convention: CamelCase + `-vN`).
- HF dataset repo: **`torus-world`** (kebab-case) — replaces `quickdraw-torus`.
- display: **"Torus World"**.
- **python package / code repo: stays `quickdraw`** → all imports, run dirs, wandb, and outputs bit-identical.
  Only the *environment/dataset* is renamed (the `torus/<split>` lerobot repo_id → `torus_world/<split>`, and
  the HF dataset name). This is the one place the rename is visible; it does not alter learned outputs.

## What stays torus-specific
The pyvista torus mesh + FPV/scene/axial renderers, the torus geometry config, the OU/bimodal samplers (they
become *policies* usable by any env, but were written for torus dynamics). Everything else (data-gen loop,
lerobot/HF I/O, datamodule, world model, training, MPPI, eval scaffolding) becomes env-agnostic.

## Parquet note
Already satisfied: LeRobotDataset writes the vector/action data as **parquet** (+ mp4 video); the HF dataset
card already points the viewer at `**/*.parquet`. No change needed.

## Open decisions / what I need from you
1. **Confirm naming** (`TorusWorld-v0` / `torus-world` / package `quickdraw`).
2. Confirm the **batched `WorldEnv` + gym adapter** split (vs forcing everything through single-env gym — the
   batched path is much faster for data-gen and is what the code already does).
3. Execution order: I'd do Phase 1 (interface + torus implements it, prove byte-identical) → 3 (remote HF) →
   2 (policy-driven gen) → 4 (control reward) → 6 (gym adapter), verifying torus parity at each step.
