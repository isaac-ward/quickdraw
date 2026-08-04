# Bring your own environment

The whole pipeline runs against **one** interface — the `WorldEnv` protocol
(`src/quickdraw/environments/base.py`), a `@runtime_checkable Protocol`. You don't subclass anything; you
provide an object with the right methods and `make_env(name)` hands it to the pipeline. There are no
env-specific branches downstream.

There are **three ways** to provide one, in increasing effort. This table is the whole story — what each
path demands of you, and what it unlocks (**✓** works · **✗** unavailable / falls back):

| | **Recorded data** | **Just the environment** | **Full implementation** |
|---|:---:|:---:|:---:|
| | *no simulator* | *required contract* | *+ optional hooks* |
| **You provide** | | | |
| `obs_dim` / `action_dim` / `dt` | ✓ | ✓ | ✓ |
| a logged dataset (obs, action, frames) | ✓ | ✗ | ✗ |
| `reset` / `step` | ✗ | ✓ | ✓ |
| `reward` | ✗ *(zeros)* | ✓ | ✓ |
| `render_obs` | ✗ *(frames from data)* | ✓ | ✓ |
| the 7 optional hooks ¹ | ✗ | ✗ | ✓ |
| **You get** | | | |
| WM training + validation | ✓ | ✓ | ✓ |
| pointwise rollout metrics (`ood_horizon`) | ✓ | ✓ | ✓ |
| `pred`/GT image filmstrip | ✓ | ✓ | ✓ |
| interpret + reward head (latent labeling) ² | ✓ | ✓ | ✓ |
| MPPI control — goal race + language steering | ✗ | ✓ | ✓ |
| env-meaningful metrics + `best.ckpt` on them | ✗ | ✗ | ✓ |
| diagnostic scene video | ✗ | ✗ | ✓ |
| goal-race control + oracle baseline | ✗ | ✗ | ✓ |
| physics-informed loss term | ✗ | ✗ | ✓ |
| scripted play policies | ✗ | ✗ | ✓ |

¹ `rollout_metrics`, `checkpoint_metric`, `render_diagnostics`, `control_goals`, `physical_loss`,
`POLICIES`, `fork` — each independent, each with a graceful fallback (the ✗ rows above).
² interpret + the reward head use only the frozen WM + data + a VLM — **no env stepping** — so recorded
(no-simulator) data can do them (they need a `conf/interpret/<env>.yaml` + a VLM). Only the **control** row
needs a steppable env: the goal race *and* language steering both execute plans in the env.

Every run prints an `[env-contract]` ✓/✗ report to its `progress.log`, so you can always see which column
you're in. Two full-contract reference envs ship in `environments/examples/`: `pendulum.py` and
`torus.py`; `base.py` is the spec. The three sections below walk up the ladder left-to-right.

**Data source is a separate axis.** *Where* the training data comes from — `data_generation` (the env
generates it) or a `data/processors.py` processor (an existing dump) — is independent of *which env* you
provide. The **Recorded data** column is specifically the *no-env* case. If you have pre-generated data
**and** an env (gym or full), you're in column 2 or 3: skip `data_generation`, point `data.root` at the
processed dataset, and set `environments.name` to your env — you get that column's **full** evals, trained
on your own data. (`env = make_env(environments.name)` and the dataset loader are wired independently, so
nothing requires the data to have come from that env.)

---

## 1. Recorded data (no simulator)

You have logged trajectories (states, actions, camera frames) and **no simulator**. A **processor** in
`data/processors.py` fixes up the raw dump into the standard lerobot layout, and training routes through
`RecordedEnv` (`environments/recorded.py`) — a `WorldEnv` that provides `obs_dim`/`action_dim`/`dt`
(`conf/environments/recorded.yaml`) and serves camera frames from the dataset, but raises on
`step`/`reset`/`render_obs` and stubs `reward` to zeros (all `@not_provided`, so the contract report flags
them ✗):

```bash
python -m quickdraw.data.processors +processor=starling +source.dir=<path> +source.name=<name>
#    -> logs/recording_<ts>_<name>; then verify the fit, then train:
python -m quickdraw.check_dataset data.root=<run_dir> data.repo_id=<name> environments=recorded
python -m quickdraw.train_world_model data.root=<run_dir> data.repo_id=<name> data.cam=<cam> \
    environments=recorded environments.obs_dim=<D> environments.action_dim=<A>   # (+ model/ run_summary)
```

Use **`environments=recorded`** — the config *group*, which carries the dims — **not**
`environments.name=recorded`, which only renames the default (torus) env and leaves the dims unresolved.
`conf/environments/recorded.yaml` defaults to starling's `16`/`4`, so override `environments.obs_dim` /
`action_dim` (and `model.action_dim` / `modalities.0.dim`) for other datasets. `check_dataset` counts the
`P+F` training windows and flags any obs/action-dim mismatch **before** you burn a run.

Each dataset gets a small bespoke processor (`starling`, `robocasa`, …) that parses its quirks and emits
the same layout via a shared builder; **non-image** datasets are supported too (a processor yields
`frames=None` → a proprio-only WM). You get WM training + validation, the `ood_horizon` pointwise metric,
the `pred`/GT filmstrip (image head decode vs the dataset frames), **and the whole interpret stack** —
`eval_interpret` (latent-space interpretation + VLM labeling) and `train_reward_model` (the language reward
head), since those use only the frozen WM + data + a VLM and never step the env. (The interpret **factors +
VLM prompt are env-specific** — author a `conf/interpret/<env>.yaml`; `conf/interpret/pendulum.yaml` is a
copyable example. This is the one interpret input that isn't automatic.) What you **can't** get is anything
that must *step* the world — MPPI **control** (the goal race *and* language steering/execution) and the
oracle baseline. So on recorded data you can *interpret and label* the latent space, just not *act* with it.

Training reads this **local** `data.root` directly — pushing to the Hub is optional
(`push_to_hub data.root=<run_dir> +hub.name=<name>`), and re-pushing is a clean **clear-and-reupload** to
the same repo each time (`delete_patterns="*"`), for sharing or for training elsewhere via
`data.hf_repo=<repo>`.

Dataset knobs the loader threads (any dataset, recorded or generated): `data.cam` — the camera key
`videos/observation.images.<cam>` (default `fpv`; recordings use `ego`); `data.repo_id` — the lerobot repo
prefix (default `torus`); non-square images via `modalities.i.img_size: [H, W]`.

---

## 2. Just the environment (the required contract)

Six members, and the pipeline can already **train, plan, and render**:

| member | signature | role |
|---|---|---|
| `obs_dim` / `action_dim` | `int` | width of the state / action vector |
| `reset` | `(generator) -> (B, obs_dim)` | deterministic given a `torch.Generator` |
| `step` | `action (B, action_dim) -> (B, obs_dim)` | next observation |
| `reward` | `(obs, goal=None) -> (B,)` | per-step control return (higher = better); scores control |
| `render_obs` | `(obs) -> (B, H, W, 3)` | uint8 — THE image modality the model consumes |

Pendulum's versions (`examples/pendulum.py`): `obs = [cosθ, sinθ, θ̇]`, `action = [torque]`, `step`
integrates the ODE, `reward` is swing-up height (or `−dist` to a goal), and `render_obs` draws the rod —
**that rod image IS the image modality** the world model predicts. With just these you get WM training,
reward-scored **MPPI control**, and the filmstrip; env-specific metrics, diagnostic scenes, goal control,
physics loss, and scripted policies all fall back until rung 3.

> **Zero-code shortcut.** Already have a `gymnasium.Env`? Skip writing `reset`/`step`:
> `environments.name=gym:<EnvId>` wraps it via `GymBatchAdapter` (`environments/gym_adapter.py`) —
> `observation_space` flattened → `obs_dim`, flat `Box` action, gym-native `reward`, `render()` →
> `render_obs`. That lands you exactly in this column, for free.

Register a name in `environments/registry.make_env` (pendulum is `name=pendulum`) and add a
`conf/environments/<name>.yaml`.

**Train pre-generated data through this env.** The data source and the env are wired independently
(`env = make_env(environments.name)` vs `root = resolve_data_root(data.root | data.hf_repo)`), so you can
skip `data_generation` entirely: process a logged dump once (§1's `data/processors.py`), then point
`data.root` at it while naming your *real* env — you get this column's full evals (control, oracle, …),
trained on your own data:

```bash
# 1. one-time: your dump -> recorded lerobot layout
python -m quickdraw.data.processors +processor=starling +source.dir=<dump> +source.name=mydata
# 2. train YOUR data through YOUR env  (NOT environments=recorded)
uv run python -m quickdraw.train_world_model data.root=logs/recording_<ts>_mydata \
    environments.name=<your-registered-env> +run_summary.problem=...   # (all 5 fields)
```

The only requirement: the dataset's `obs_dim`/`action_dim` (and camera view) match the env's. `RecordedEnv`
(§1) is only for when you have *no* env — give it a real env and you leave the recorded column.

---

## 3. Full implementation (add the optional hooks)

Continuing with the pendulum, implement the optional hooks — each is **independent**, add the ones whose
eval you want. This is the extra-in → extra-out:

| add this hook | extra code (pendulum's) | unlocks | fallback if skipped |
|---|---|---|---|
| `rollout_metrics` | `angle_error` = wrapped `|θ_pred − θ_true|` | env-meaningful val + `ood_horizon` curves | generic `pointwise_error` |
| `checkpoint_metric` | returns `"angle_error"` | `best.ckpt` chosen on the *meaningful* metric | `pointwise_error` |
| `render_diagnostics` | draw the rod overlay (below) | the diagnostic **scene video** | `render_obs` filmstrip |
| `control_goals` | 4 named tip targets (upright/right/down/left) | **goal-race** control eval | reward-only control |
| `physical_loss` | off-circle + energy-drift + continuity residuals | physics-informed training term | variation unavailable |
| `POLICIES` | `swingup` (bang-bang) + `sinusoid` | scripted play policies (`data.action_sampler=swingup`) | `random` only |
| `fork` | copy `θ`/`θ̇`/torque into a `k`-batch clone | the **oracle** rollout baseline in control | control skips the oracle |

Implement all seven and pendulum runs the **entire** pipeline.

**The diagnostic scene (`render_diagnostics`).** Optional, and *declarative*: the eval hands the env a
`SceneOverlay` — `agents` (world-space paths, e.g. `true`/`pred`), `markers` (points, e.g. `goal`),
optional `field_` (a scalar field), `extras` (presentation hints) — plus a list of `views`, and the env
draws its own geometry with those overlays. One method serves open-loop rollout, control, and language
steering; roles map to a shared style (`true`/`oracle` black, `pred`/`learned` grey, `goal` gold,
`concept` red). The design rule both examples follow — **draw the overlay in the env's OWN view**:

| example | what it draws | true-vs-pred readout |
|---|---|---|
| `TorusEnv` | the 3D torus surface with the `agents` paths *on the manifold* + fork marker + action arrows | pred path diverging across the surface |
| `PendulumEnv` | the pendulum's own 2D view — pivot + one swinging rod *per agent, overlaid* at each step's angle, goal a dashed target rod | grey pred rod tracking/drifting from the black true rod |

Because the pendulum's rod render *is* its image modality, `ood_horizon` gives the true-vs-pred comparison
twice: the **scene** (`trajectory_video_i`, rods superimposed in one analytic panel — tests the *dynamics
geometry*) and the **image filmstrip** (`image/filmstrip_i`, two stacked rows of decoded vs ground-truth
frames — tests the *pixel decoder*). Torus's static atlas PNG + 3D scene JSON are torus-only bonuses (they
need the `R/r` geometry) and are gated off for other envs.

---

## Worked example: pendulum end-to-end

The full-contract pendulum runs **every** step of [docs/workflow.md](workflow.md) — the only difference
from torus is `environments.name=pendulum`:

```bash
# 1. mine play data (scripted swing-up policy, shipped via PendulumEnv.POLICIES)
uv run python -m quickdraw.data_generation 'environments.name=pendulum' data.action_sampler=swingup experiment=pendulum
#    -> set DATA=logs/data_generation_<ts>_pendulum
# 2. push the dataset to the Hub
uv run python -m quickdraw.push_to_hub data.root=$DATA +hub.name=pendulum
# 3. train the world model (best.ckpt tracked on angle_error via checkpoint_metric)
uv run python -m quickdraw.train_world_model experiment=pendulum data.root=$DATA 'environments.name=pendulum' +run_summary.problem=... # (all 5 fields)
#    -> set CKPT=logs/train_world_model_<ts>_pendulum
# 4. action model (post-hoc, on the frozen WM)
uv run python -m quickdraw.train_action_model checkpoint=$CKPT data.root=$DATA experiment=pendulum 'environments.name=pendulum'
# 5. control eval — goal-race (control_goals) + oracle baseline (fork), rod diagnostic scene (render_diagnostics)
uv run python -m quickdraw.eval_control checkpoint=$CKPT data.root=$DATA 'environments.name=pendulum'
```

Swap `name=gym:Pendulum-v1` back in and the *same* commands run on the zero-code adapter, dropping to the
rung-2 fallbacks (pointwise metric, filmstrip, reward-only control) — a direct before/after of what the
optional hooks buy.
