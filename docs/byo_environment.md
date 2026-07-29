# Bring your own environment

The pipeline is written against **one** interface — the `WorldEnv` protocol
(`src/quickdraw/environments/base.py`) — and builds environments purely by config through
`environments/registry.make_env`. There are no env-specific branches downstream, so plugging in a new
environment just means giving the pipeline a `WorldEnv`.

There is only that one interface; you have two ways to provide it, differing only in effort:

- **use the built-in gym adapter** (zero code) — for any `gymnasium.Env`, or
- **implement the protocol yourself** (full control) — for a batched, first-class env like the torus.

The adapter is simply a pre-written `WorldEnv` implementation, so everything downstream is identical either
way.

## Quickest: any registered `gymnasium.Env` (the built-in adapter)

```bash
uv run python -m quickdraw.data_generation 'environments.name=gym:Pendulum-v1' \
    data.action_sampler=random experiment=pendulum
```

`environments.name=gym:<EnvId>` routes through `GymBatchAdapter`
(`src/quickdraw/environments/gym_adapter.py`), which wraps B copies of
`gymnasium.make(env_id, render_mode="rgb_array")` into a batched `WorldEnv`:

- **obs**: `observation_space` flattened → `obs_dim`.
- **action**: flat `Box` dim; a `Discrete` env takes a `(B, n)` score row and steps its argmax.
- **reward()**: the gym-native reward from the last `step` — this is what the control eval scores with.
- **render_obs()**: each env's `render()` (rgb_array) stacked to `(B, H, W, 3)` uint8 — the env must
  support `render_mode="rgb_array"` (classic control needs `pygame` installed).
- **determinism**: `reset(generator)` seeds env `i` with `base + i`; done envs auto-reset (vector
  semantics), still deterministic.

For free you get the `random` behavior policy, the env's own reward for control eval, and the
`render_obs` pred-vs-true filmstrip as eval-viz (`render_diagnostics` is absent → `wants_diagnostics` is
False → graceful fallback). From here the rest of [docs/workflow.md](workflow.md) applies unchanged.

## Full control: implement the `WorldEnv` protocol yourself

For a batched, first-class env like the torus reference (`environments/torus.py`), implement the protocol
directly — batched-torch `reset`/`step` is a big speed win for data-gen and internal rollouts:

| member | contract |
|---|---|
| `action_dim: int` | width of the action vector |
| `obs_dim: int` | width of the proprio/state vector |
| `reset(generator) -> Tensor` | `(B, obs_dim)`; deterministic given a `torch.Generator` |
| `step(action) -> Tensor` | action `(B, action_dim)` → next obs `(B, obs_dim)` |
| `reward(obs, goal=None) -> Tensor` | `(B,)` per-step control return (higher = better); scores the control eval |
| `render_obs(obs) -> Tensor` | `(B, H, W, 3)` uint8 — THE image modality the model consumes (torus: FPV). Required for image world models |
| `render_diagnostics(overlay, views)` | OPTIONAL — see below |

Then register a name for it in `environments/registry.make_env` and add a
`conf/environments/<name>.yaml` with `name: <registry-name>` plus your env's parameters (see
`conf/environments/torus_world.yaml`).

## Policies

Two kinds of behavior policy mine your play data (`data.action_sampler=<name>`,
`environments/policies.py`):

- **`random`** — free and env-agnostic: uniform over the env's action range. The only default that works
  for ANY env; a BYO env starts here.
- **env-specific** — ship behavior policies WITH your env via a `POLICIES` class registry:
  `{name: factory(env, device) -> policy}`, where a policy has `reset(generator)` and
  `sample(obs, generator) -> (B, action_dim)`. `make_policy` resolves `random` itself and delegates every
  other name to the env's registry. Reference: `TorusEnv.POLICIES` registers `ornstein_uhlenbeck` and
  `bimodal`.

## Diagnostic renders (optional)

Two render concerns, kept separate:

- **`render_obs(obs)`** — required. The observation/image modality (FPV for torus): it is what the model
  trains on, and it drives the pred-vs-true filmstrip eval-viz.
- **`render_diagnostics(overlay, views)`** — optional. The ONE rich eval-video renderer, and it is
  **declarative**: the eval routine hands the env a `SceneOverlay` (WHAT to draw, in world coordinates)
  and a list of camera `views`; the env draws its own geometry plus those overlays. The env never knows
  which eval called it — one method serves them all. Returns `{view_name: np.ndarray frame}`.

`SceneOverlay` (`environments/base.py`):

| field | shape | meaning |
|---|---|---|
| `agents` | `{role: (T, 3)}` | world-space PATHS (e.g. `true`, `pred`) |
| `markers` | `{role: (K, 3)}` | world-space POINTS (e.g. `goal`) |
| `field_` | `Tensor \| None` | optional scalar field over the manifold (e.g. a language reward field) |
| `extras` | `dict` | optional presentation hints from the eval routine (title, action arrows, candidate fan, fork step, ...) — honor what you like, ignoring them wholesale is fine |

Roles map to a shared, env-agnostic style map (`ROLE_STYLE`): `true`/`oracle` = black path,
`pred`/`learned` = grey path, `goal` = gold ring, `concept` = red cross. Honor the roles you can; ignore
what you can't.

What you need for what:

| you implement | you get |
|---|---|
| `render_obs` only | image-modality training + the pred-vs-true filmstrip eval videos (`wants_diagnostics` → False, automatic fallback) |
| + `render_diagnostics` | the rich multi-view diagnostic videos: open-loop rollout (`agents={true, pred}`), control (`agents={oracle, learned}, markers={goal}`), language steering (`agents`, `markers={concept}`, `field_`) |

The eval-viz wiring that CALLS `render_diagnostics` (Phase 5 of
[design/gym_refactor.md](../design/gym_refactor.md)) asks for the view `"scene"`: the open-loop rollout
video and the control video both go through it, and fall back to the `render_obs` pred-vs-true filmstrip
when `wants_diagnostics(env)` is False or the env returns `{}`. Reference implementation:
`TorusEnv.render_diagnostics` (`environments/torus.py`) — a thin wrapper over the torus scene renderers,
pixel-identical to the pre-interface videos.

## Worked example: a stock gym env end-to-end

Each step is its own `uv run` line, exactly as in [docs/workflow.md](workflow.md) — the only difference is
the `environments.name=gym:Pendulum-v1` override threaded through:

```bash
# 1. mine play data with the env-agnostic random policy
uv run python -m quickdraw.data_generation 'environments.name=gym:Pendulum-v1' data.action_sampler=random experiment=pendulum
#    -> set DATA=logs/data_generation_<ts>_pendulum

# 2. push the dataset to the Hub (its own step)
uv run python -m quickdraw.push_to_hub data.root=$DATA +hub.name=pendulum

# 3. train the world model
uv run python -m quickdraw.train_world_model experiment=pendulum data.root=$DATA 'environments.name=gym:Pendulum-v1' +run_summary.problem=... # (all 5 fields)
#    -> set CKPT=logs/train_world_<ts>_pendulum

# 4. train the action model (post-hoc, on the frozen WM)
uv run python -m quickdraw.train_action_model checkpoint=$CKPT data.root=$DATA experiment=pendulum 'environments.name=gym:Pendulum-v1' +run_summary.problem=...

# 5. control eval — MPPI scored by the env's own reward
uv run python -m quickdraw.eval_control checkpoint=$CKPT data.root=$DATA 'environments.name=gym:Pendulum-v1'
```

Control eval scores plans with the env's own `reward`; eval-viz uses the `render_obs` filmstrip until you
implement `render_diagnostics`. The interpret / reward / language-control steps (5–7 of
[docs/workflow.md](workflow.md)) need per-env semantic factors (`conf/interpret/<env>.yaml`) and a VLM, so
they're env-specific — add them when you want language steering. To go beyond `random` play data or
single-view eval videos, implement the protocol yourself (above).
