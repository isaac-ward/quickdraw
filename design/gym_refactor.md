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
    def render_diagnostics(self, overlay: SceneOverlay, views: list[str]) -> dict[str, np.ndarray]: ...  # OPTIONAL diagnostic
```
- **`render_obs`** = the image modality (always present; for torus = FPV). **Never skipped.**
- **`render_diagnostics`** = the ONE optional diagnostic renderer, and the trick that stops it exploding into
  per-eval methods (MPPI / long-horizon / language each want different actors). It is **declarative**: the eval
  routine hands the env a `SceneOverlay` (what to draw, in world coords + a role) and a list of `views`, and the
  env draws its geometry + those overlays from those cameras. The env NEVER knows about "MPPI" or "language
  steering" — only "draw these labelled points/paths/fields in my world from these views."
```python
@dataclass
class SceneOverlay:
    agents:  dict[str, Tensor] = {}   # role -> (T,3) world PATH (roles: "true"=black, "pred"=grey, ...)
    markers: dict[str, Tensor] = {}   # role -> (K,3) POINTS   (roles: "goal"=gold ring, "concept"=cross, ...)
    field:   Tensor | None = None     # optional scalar field over the manifold (e.g. language reward field)
```
  Role→color/style lives in ONE shared `viz` style map (env-agnostic). Each eval just fills the overlay:
  `eval_ood_horizon` → `agents={true,pred}`; `eval_control` → `agents={true,pred}, markers={goal}`;
  language steering → `agents={agent}, field=reward, markers={concept}`. So ONE env method serves all three.
  **Optional + graceful:** if an env doesn't implement `render_diagnostics` (or ignores overlays/views it can't do),
  eval-viz falls back to the `render_obs` pred-vs-true filmstrip. Torus implements it fully (scene + 3 axial).

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
- **Internal lerobot split id STAYS `torus/<split>`** (baked into every local dataset path — renaming it would
  break loading + not be bit-identical). Only the PUBLIC HF dataset name (`quickdraw-torus` → `torus-world`) and
  the gym env id (`TorusWorld-v0`) change. CONFIRMED with the user.

## What stays torus-specific
The pyvista torus mesh + FPV/scene/axial renderers, the torus geometry config, the OU/bimodal samplers (they
become *policies* usable by any env, but were written for torus dynamics). Everything else (data-gen loop,
lerobot/HF I/O, datamodule, world model, training, MPPI, eval scaffolding) becomes env-agnostic.

## Parquet note
Already satisfied: LeRobotDataset writes the vector/action data as **parquet** (+ mp4 video); the HF dataset
card already points the viewer at `**/*.parquet`. No change needed.

## Policies & planners — how "black oracle vs grey learned" hooks in
Two distinct notions, both env-agnostic:
- **Behavior policy** (data-gen): `policy(obs, generator) -> action`, config `data.policy=<name>`, used only to
  mine play data. **`random` (sample the env's `action_space`) is the ONLY env-agnostic default** — it works for
  any env. `bimodal`/`ou` are **torus-specific** (they encode torus action semantics) and can't be applied to an
  arbitrary env; they ship as torus's policies. A BYO-env uses `random` (or supplies its own policy).
- **Control planner** (eval): the "oracle" (black) and "learned" (grey) are the SAME MPPI code parameterized by
  the *rollout source* — `MPPI(rollout_fn, reward_fn, action_dim)`. oracle: `rollout_fn = env.step` (true
  dynamics); learned: `rollout_fn = WM.rollout`. `reward_fn = env.reward` for both. So swapping oracle↔learned is
  swapping one callable; nothing torus-specific. Their paths become the `agents={true→oracle, pred→learned}`
  overlay for `render_diagnostics`.

## Bit-identical verification — the test loop I'll run and iterate on
The refactor is re-plumbing (same renderers/sim/model called through an interface), so parity should hold by
construction; these tests catch accidental drift, run after EVERY phase, iterate until zero diff:
1. **Renders** — `smoke/render_golden` already asserts pixel-identical FPV + scene + axial. Run as-is.
2. **Data-gen parity** (`smoke/refactor_parity.py`, NEW) — regen a tiny fixed-seed torus dataset with the
   pre-refactor commit vs HEAD; assert **exact** equality of the parquet obs/act arrays, the rendered frames
   (pixel-exact), and `normalization_stats.json`.
3. **Train-step parity** — fixed seed + config, build model, run 1 training step old vs new; assert the loss +
   a forward output match (bit-exact under the same seed/precision path; `torch.use_deterministic_algorithms`
   where feasible).
4. **Eval parity** — run `eval_ood_horizon` + `eval_control` on a FIXED checkpoint old vs new; diff the metric
   scalars (exact) and the rendered videos (pixel-exact via the golden renderer).
Green on 1–4 = torus is byte-identical. I fix any diff and re-run until all four pass.

## GPU budget
**Minimal.** The refactor is CPU-side plumbing + docs. Renders/`render_golden` run offscreen (OSMesa, no GPU).
Data-gen parity is a 2-trajectory sim (negligible). Only train-step parity (1 step) and eval parity (1 eval on a
fixed ckpt) touch a GPU, briefly. No long training runs are needed for the refactor. Caveat: both GPUs are
currently held by the `repro_ptf0_*` runs, so I'll run the tiny GPU checks on CPU-fallback or squeeze them in
when a card frees — the repro runs keep priority.

## Phase 7 — documentation (linked from README)
- **`docs/byo.md`** — how to bring your own gym env: the minimal `WorldEnv`/gym contract, `render_obs`
  for the image modality, the OPTIONAL `render_diagnostics(overlay, views)` (what a full diagnostic renderer must
  accept: the `SceneOverlay` roles + view names) with the graceful fallback, reward/goal conventions
  (`env.reward`, gym goal-conditioned pattern), and a worked minimal example env.
- **`docs/workflow.md`** — every main workflow end-to-end with commands, IN ORDER:
  `data_generation` (mine play data) → `push_to_hub` (parquet+mp4 to HF) → **`train_world_model`** (with the
  during-training val + eval routines — ood_horizon, control, manifold — documented as SUB-points here) →
  `train_action_model` (post-hoc, frozen WM) → **`eval_interpret`** (VLM-labeled latent interpretability) →
  `train_reward_model` (language reward head) → **language control examples** (steering MPPI by a request).
- **`docs/interpret.md`** — its OWN doc for `eval_interpret`: how to DEFINE the concepts/factors and the VLM
  prompts for a given env (the labeling contract), since that's env-specific and non-obvious. Referenced from
  workflow.md at the interpret step.
- **`docs/byo.md`** dedicated section "**Diagnostic renders (optional)**": exactly what an env must
  implement to get the rich eval videos — the `render_diagnostics(overlay, views)` signature, which `SceneOverlay`
  roles/views it should honor, the shared role→style map, and what you lose if you skip it (fallback to the
  `render_obs` filmstrip). Clear "you need X for Y" table.
- **`README.md`** — link `docs/workflow.md`, `docs/byo.md`, `docs/interpret.md` (+ this plan and
  `design/accelerations.md`).

## Decisions (all CONFIRMED 2026-07-29)
- Naming: `TorusWorld-v0` / HF `torus-world` / package `quickdraw` / internal `torus/<split>` kept.
- Batched `WorldEnv` + single-env `GymBatchAdapter` split — yes.
- Declarative `render_diagnostics(overlay, views)` + `SceneOverlay` — yes.
- `random` is the only env-agnostic behavior policy; `bimodal`/`ou` are torus-specific.
- Docs: `workflow.md` (order above) + `interpret.md` + `byo.md` (with the diagnostic-render section),
  all linked from README.

## Execution order (parity-verified at each step)
Phase 1 (interface + TorusEnv implements it + `TorusWorld-v0`, prove byte-identical) → 3 (remote HF) →
2 (policy-driven gen) → 4 (control reward) → 5 (eval-viz uses render_diagnostics) → 6 (gym adapter) → 7 (docs).
See the checkbox tracker below.

## Checkbox tracker (execute in order; parity-verify each phase)

### Phase 1 — WorldEnv interface + TorusEnv implements it + TorusWorld-v0
- [x] 1.1 `environments/base.py`: `WorldEnv` Protocol + `SceneOverlay` dataclass + shared `ROLE_STYLE` map.
- [x] 1.2 `environments/registry.py`: `make_env(name, cfg)` (torus_world -> TorusEnv; later gym adapter).
- [x] 1.3 `TorusEnv.reward(obs, goal)` — extract the goal-distance/settle logic from eval_control into the env.
- [x] 1.4 `TorusEnv.render_obs(obs)` — wrap the existing FPV renderer (byte-identical).
- [x] 1.5 `TorusEnv.render_diagnostics(overlay, views)` — wrap the existing pyvista scene + axial renderers,
        driven by the overlay's agents/markers/field (byte-identical to today's rollout/control videos).
        DEFERRED to Phase 5 (co-implemented with the eval-viz rewire + video parity); `wants_diagnostics`
        correctly returns False until then.
- [x] 1.6 `TorusWorld-v0`: register a single-env `gymnasium.Env` (batch-1 TorusEnv) + `action_space`/`observation_space`.
- [x] 1.7 `conf/environments/torus_world.yaml` (geometry + policy defaults). Keep `conf/environments/torus.yaml` values.
- [~] 1.8 PARITY: covered piecewise — render_golden pixel-identical, smoke/refactor_parity_datagen.py
        byte-identical datagen, Phase 4 control bit-identical, Phase 5 videos pixel-identical, 3.3 HF-load
        exact. WM train path is untouched by the refactor; no separate combined smoke added.
        (superseded) run `smoke/render_golden`; add + run `smoke/refactor_parity.py` (data-gen 2-traj, train 1 step,
        eval on a fixed ckpt) — all pixel/array/scalar exact vs the pre-Phase-1 commit. Iterate until zero diff.

### Phase 3 — train on the HF dataset (remote)
- [x] 3.1 `data/dataset.py`: `data.hf_repo` option -> `LeRobotDataset("<user>/<name>")` (HF download/cache); local root default unchanged.
- [x] 3.2 Thread `data.hf_repo` through `setup.window_loaders` + the eval loaders.
- [x] 3.3 PARITY: local-root load == hf_repo load for the same dataset (array-exact).
        Verified on the live torus-world push: all 6 splits' observation_vector + action parquet arrays
        array-exact (max|diff|=0) and every video mp4 byte-identical (push_to_hub uploads the folder
        verbatim; snapshot_download returns the same bytes).

### Phase 2 — policy-driven, env-agnostic data generation
- [x] 2.1 `environments/policies.py`: `RandomPolicy` (samples action_space) + wrap OU/Bimodal samplers as policies.
- [x] 2.2 `generate_episodes(env: WorldEnv, policy, ...)`; collect (obs, action, reward, render_obs frames).
- [x] 2.3 `data_generation.py`: build env+policy from config; FPV step -> `env.render_obs`. Torus path byte-identical.
- [x] 2.4 PARITY: torus dataset regen (fixed seed) == pre-refactor dataset (arrays + frames + norm stats).

### Phase 4 — control eval via env.reward
- [x] 4.1 `MPPI(rollout_fn, reward_fn, action_dim)`: oracle `rollout_fn=env.step`, learned `rollout_fn=WM.rollout`; reward=`env.reward`.
        `_score` now sums a per-step `reward_fn(obs, goal)` (default `env.reward` via `make_env` in
        run_and_log_control; config beta_vel/r_settle bound when the env's reward exposes them). The
        language `dist` fast path (1 - reward on the rolled bag) is kept inline.
- [x] 4.2 `eval_control` builds the SceneOverlay (agents={true,pred}, markers={goal}) for render_diagnostics.
        DEFERRED to Phase 5 with 1.5 (render_diagnostics itself).
- [x] 4.3 PARITY: torus control scalars + videos unchanged vs pre-refactor (fixed ckpt).
        Verified on repro_ptf0_mse last.ckpt (4 eps, 400 steps, GPU): all eval_control scalars AND the
        sha256 of the full per-step dist_curves/paths of both controllers are bit-identical old vs new
        (and across an old-repeat determinism control). Video/render code untouched by this phase.

### Phase 5 — eval-viz uses render_diagnostics with graceful fallback
- [x] 5.1 `emit_openloop`/filmstrips/rollout videos call `render_diagnostics(overlay, views)`; fall back to `render_obs` filmstrip if `{}`.
- [x] 5.2 PARITY: torus ood_horizon + control videos pixel-identical.

### Phase 6 — GymBatchAdapter (bring your own env)
- [x] 6.1 `environments/gym_adapter.py`: wrap any `gymnasium.Env` (vectorize B), map spaces, step->reward+obs, render(rgb_array)->render_obs, render_diagnostics->{}.
- [x] 6.2 Smoke: a stock gym env (e.g. `Pendulum-v1`) end-to-end: gen -> (push) -> train 1 epoch -> control eval, core diagnostics present.

### Phase 7 — docs + naming
- [x] 7.1 `docs/workflow.md` (order: datagen -> pushhub -> wm[+val/eval subpoints] -> am -> interpret -> rm -> language-control).
- [x] 7.2 `docs/interpret.md` (defining concepts/factors + VLM prompts).
- [x] 7.3 `docs/byo.md` (+ the "Diagnostic renders (optional)" section: exact contract + what-you-lose table).
- [x] 7.4 `README.md` links to the three docs + this plan + accelerations.md.
- [x] 7.5 Public rename: HF dataset `quickdraw-torus` -> `torus-world`; gym id `TorusWorld-v0`. Internal `torus/<split>` unchanged.
