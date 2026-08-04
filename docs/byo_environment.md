# Bring your own environment

The whole pipeline is written against **one** interface — the `WorldEnv` protocol
(`src/quickdraw/environments/base.py`) — and builds environments purely by config through
`environments/registry.make_env`. `WorldEnv` is a `@runtime_checkable Protocol`, **not** a base class:
you don't subclass anything, you just provide an object with the right methods and `make_env(name)` hands
it to the pipeline. There are no env-specific branches downstream.

There are **three ways** to provide a `WorldEnv`, and they form a **ladder** — each rung is more code and
unlocks more of the pipeline. We walk the *same* env (a pendulum) up rungs 1 → 2 so you can see exactly
what each optional hook **costs** and what it **buys**:

| path | you write | you get | you can't get |
|---|---|---|---|
| **1. Just the environment** (required contract) | `reset`/`step`/`reward`/`render_obs` + 2 dims | WM training, reward-scored MPPI control, `render_obs` filmstrips | env-specific metrics, diagnostic scenes, goal-race control, physics loss, scripted policies |
| **2. Full implementation** (+ optional hooks) | + the 7 optional hooks | **+** env-meaningful metrics & best-ckpt, diagnostic scene videos, goal-race control, physics-informed loss, scripted play policies, oracle rollouts | — (the whole pipeline runs) |
| **3. Recorded data** (no simulator) | a logged dataset (obs, action, frames) + dims/`dt` | WM training + pointwise rollout metrics + filmstrips | anything that must *step* the world: control, goals, oracle, interpret/language |

Every run prints an `[env-contract]` ✓/✗ report to its `progress.log`, so you can see at a glance which
rung you're on and what fell back. Two **full-contract reference envs** ship in `environments/examples/`:
`pendulum.py` (walked through below) and `torus.py` (the original, richer 3D reference). `base.py` is the
spec.

---

## Path 1 — Just the environment (the required contract)

Six members, and the pipeline can already **train, plan, and render**:

| member | signature | role |
|---|---|---|
| `obs_dim` | `int` | width of the proprio/state vector |
| `action_dim` | `int` | width of the action vector |
| `reset` | `(generator) -> (B, obs_dim)` | deterministic given a `torch.Generator` |
| `step` | `action (B, action_dim) -> (B, obs_dim)` | next observation |
| `reward` | `(obs, goal=None) -> (B,)` | per-step control return (higher = better); scores the control eval |
| `render_obs` | `(obs) -> (B, H, W, 3)` | uint8 — THE image modality the model consumes. Required for image world models |

The pendulum's versions (`examples/pendulum.py`): `obs = [cosθ, sinθ, θ̇]` (`obs_dim = 3`),
`action = [torque]` (`action_dim = 1`), `step` integrates the pendulum ODE, `reward` is swing-up height
(or `−dist` to a goal tip), and `render_obs` draws the rod on a white field — **that rod image IS the
image modality** the world model learns to predict.

**With just these you get:**
- world-model training on datagen'd play data,
- reward-scored **MPPI control** (the eval maximizes `env.reward`),
- the `render_obs` `pred(top)/GT(bottom)` **filmstrip** as eval-viz.

**What you don't get yet** (every one a graceful fallback, never an error): env-specific metrics (only a
generic pointwise L2), a diagnostic scene (only the filmstrip), goal-race control (only reward-only
control), the physics-loss term, and scripted play policies (only `random`).

> **Zero-code shortcut.** Already have a `gymnasium.Env`? Skip writing `reset`/`step` entirely —
> `environments.name=gym:<EnvId>` wraps it via `GymBatchAdapter` (`environments/gym_adapter.py`), giving
> exactly this required contract for free:
> - **obs**: `observation_space` flattened → `obs_dim`; **action**: flat `Box` dim (a `Discrete` env takes
>   a `(B, n)` score row and steps its argmax).
> - **reward()**: the gym-native reward from the last `step` — what control scores with.
> - **render_obs()**: each env's `render()` (`rgb_array`) stacked to `(B, H, W, 3)` — needs
>   `render_mode="rgb_array"` (classic control needs `pygame`).
> - **determinism**: `reset(generator)` seeds env `i` with `base + i`; done envs auto-reset, still
>   deterministic. You land exactly here on rung 1 (`wants_diagnostics` False → filmstrip fallback).

---

## Path 2 — The full implementation (add the optional hooks)

Same pendulum, now implement the optional hooks. Each is **independent** — add the ones whose eval you
want; each has a graceful fallback if you skip it. This is the extra-in → extra-out:

| add this hook | extra code (pendulum's) | unlocks | fallback if skipped |
|---|---|---|---|
| `rollout_metrics` | `angle_error` = wrapped `|θ_pred − θ_true|` | env-meaningful val + `ood_horizon` metric curves | generic `pointwise_error` |
| `checkpoint_metric` | returns `"angle_error"` | `best.ckpt` chosen on the *meaningful* metric | `pointwise_error` |
| `render_diagnostics` | draw the rod overlay (below) | the diagnostic **scene video** | `render_obs` filmstrip |
| `control_goals` | 4 named tip targets (upright/right/down/left) | **goal-race** control eval | reward-only control |
| `physical_loss` | off-circle + energy-drift + continuity residuals | physics-informed training term | variation unavailable |
| `POLICIES` | `swingup` (bang-bang) + `sinusoid` | scripted play policies (`data.action_sampler=swingup`) | `random` only |
| `fork` | copy `θ`/`θ̇`/torque into a `k`-batch clone | the **oracle** rollout baseline in control | control skips the oracle |

Implement all seven and the pendulum runs the **entire** pipeline — the same as the torus reference, minus
the torus-only 3D atlas bonus. The header of `examples/pendulum.py` lists these in one place; `base.py`
gives the full signatures. Then register a name in `environments/registry.make_env` (pendulum is already
`name=pendulum`) and add a `conf/environments/<name>.yaml` with the env's parameters.

### The diagnostic scene (`render_diagnostics`)

Two render concerns, kept separate:

- **`render_obs(obs)`** — required (rung 1). The image modality: what the model trains on, and what drives
  the `pred`-vs-`true` filmstrip.
- **`render_diagnostics(overlay, views)`** — optional. The ONE rich eval-video renderer, and it is
  **declarative**: the eval hands the env a `SceneOverlay` (WHAT to draw, in world coordinates) and a list
  of camera `views`; the env draws its own geometry plus those overlays. The env never knows which eval
  called it — one method serves open-loop rollout, control, and language steering. Returns
  `{view_name: np.ndarray frames}`.

`SceneOverlay` (`environments/base.py`):

| field | shape | meaning |
|---|---|---|
| `agents` | `{role: (T, 3)}` | world-space PATHS (e.g. `true`, `pred`) |
| `markers` | `{role: (K, 3)}` | world-space POINTS (e.g. `goal`) |
| `field_` | `Tensor \| None` | optional scalar field over the manifold (e.g. a language reward field) |
| `extras` | `dict` | presentation hints (title, action arrows, candidate fan, fork step, ...) — honor what you like |

Roles map to a shared, env-agnostic style (`ROLE_STYLE`): `true`/`oracle` = black, `pred`/`learned` =
grey, `goal` = gold, `concept` = red cross. Honor the roles you can; ignore what you can't. The wiring
that calls it (Phase 5 of [design/gym_refactor.md](../design/gym_refactor.md)) asks for the view
`"scene"`; the open-loop rollout video and the control video both go through it and fall back to the
`render_obs` filmstrip when `wants_diagnostics(env)` is False or the env returns `{}`.

The two reference examples show the design rule — **draw the overlay in the env's OWN view**, don't invent
a second visualization:

| example | what `render_diagnostics` draws | true-vs-pred readout |
|---|---|---|
| `TorusEnv` (`examples/torus.py`) | the 3D torus surface with the `agents` paths *on the manifold* (true black + sphere-ended, pred grey), fork marker, ambient action arrows | pred path diverging from the true path across the surface |
| `PendulumEnv` (`examples/pendulum.py`) | the pendulum's own 2D view — pivot + one swinging rod *per agent, overlaid in the same frame* (true black, pred grey) at each step's angle, goal a faint dashed target rod | watch the grey pred rod track/drift from the black true rod |

Because the pendulum's rod render *is* its image modality, a pendulum `ood_horizon` run gives you the
true-vs-pred comparison **twice**, and they test different things:

- **the diagnostic scene** (`trajectory_video_i`) — *one analytic panel, rods superimposed*: pivot + the
  true (black) and pred (grey) rods drawn together at each step's angle. Tests the **dynamics geometry**.
- **the image filmstrip** (`image/filmstrip_i`) — *two stacked rows of decoded images*: top row = the
  WM's **decoded** rod image per timestep, bottom row = the ground-truth frame. Tests the **pixel decoder**.

Torus's static atlas PNG + interactive 3D scene JSON are **torus-only** bonus products (they need the
`R/r` geometry) and are gated off for other envs.

---

## Path 3 — Recorded data (no simulator)

The third rung is data-only: you have logged trajectories (states, actions, camera frames) and **no
simulator at all**. `recording_to_lerobot.py` converts the dump into the standard lerobot run layout, and
training routes through `RecordedEnv` (`environments/recorded.py`) — a `WorldEnv` that provides the
obs/action dims + `dt` (`conf/environments/recorded.yaml`) and serves the camera frames from the dataset,
but raises on `step`/`reset`/`render_obs` (nothing to simulate or render) and stubs `reward` to zeros
(no goal semantics in logged data) — all marked `@not_provided`, so the contract report flags them ✗:

```bash
python -m quickdraw.recording_to_lerobot +recording.dir=<path> +recording.name=<name>
#    -> logs/recording_<ts>_<name>; then train on it with
#       environments.name=recorded data.repo_id=<name> data.cam=<cam>
```

- **You provide:** the logged dataset + `obs_dim`/`action_dim`/`dt`. That's it.
- **You get:** world-model training, validation, the `ood_horizon` rollout metrics (generic
  `pointwise_error`), and the `pred`-vs-`true` filmstrip.
- **You can't get:** control, goals, the oracle baseline, interpret, or language steering — all of which
  need a live env to *step*. This is the "data-only corner" of the contract.

Dataset knobs the loader threads (for any dataset, recorded or generated): `data.cam` — the camera key
`videos/observation.images.<cam>` (default `fpv`; recordings use `ego`); `data.repo_id` — the lerobot repo
prefix the splits were written with (default `torus`; a recorded dataset uses its recording name); and
non-square images via `modalities.i.img_size: [H, W]` on the model's image modality.

---

## Policies

Two kinds of behavior policy mine your play data (`data.action_sampler=<name>`,
`environments/policies.py`):

- **`random`** — free and env-agnostic: uniform over the env's action range. The only default that works
  for ANY env; a rung-1 env starts here.
- **env-specific** — ship behavior policies WITH your env via a `POLICIES` class registry (rung 2):
  `{name: factory(env, device) -> policy}`, where a policy has `reset(generator)` and
  `sample(obs, generator) -> (B, action_dim)`. `make_policy` resolves `random` itself and delegates every
  other name to the env's registry. Pendulum registers `swingup` + `sinusoid`; `TorusEnv` registers
  `ornstein_uhlenbeck` + `bimodal`.

---

## Worked example: pendulum end-to-end

The full-contract pendulum (rung 2) runs **every** step of [docs/workflow.md](workflow.md) — the only
difference from torus is `environments.name=pendulum` threaded through:

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

# 5. control eval — goal-race (control_goals) + oracle baseline (fork), diagnostic rod scene (render_diagnostics)
uv run python -m quickdraw.eval_control checkpoint=$CKPT data.root=$DATA 'environments.name=pendulum'
```

Because the pendulum implements the optional hooks, step 5 runs the **goal race** (not just reward-only)
with the **oracle** baseline and renders the **rod diagnostic scene**. The interpret / reward /
language-control steps (5–7 of [docs/workflow.md](workflow.md)) also run — they need per-env semantic
factors, which pendulum supplies in `conf/interpret/pendulum.yaml`, plus a VLM. Swap `name=gym:Pendulum-v1`
back in and the *same* commands run on the zero-code adapter instead, dropping to the rung-1 fallbacks
(pointwise metric, filmstrip, reward-only control) — a direct before/after of what the optional hooks buy.
