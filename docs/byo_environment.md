# Bring your own environment

The pipeline is written against one interface — the `WorldEnv` protocol
(`src/quickdraw/environments/base.py`) — and builds environments purely by config through
`environments/registry.make_env`. There are no env-specific branches downstream, so plugging in a new
environment means satisfying that protocol one of two ways.

## Path A (quickest): any registered `gymnasium.Env`

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

## Path B (full): implement the `WorldEnv` protocol

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

Roles map to a shared, env-agnostic style map (`ROLE_STYLE`): `true`/`oracle` = black path,
`pred`/`learned` = grey path, `goal` = gold ring, `concept` = red cross. Honor the roles you can; ignore
what you can't.

What you need for what:

| you implement | you get |
|---|---|
| `render_obs` only | image-modality training + the pred-vs-true filmstrip eval videos (`wants_diagnostics` → False, automatic fallback) |
| + `render_diagnostics` | the rich multi-view diagnostic videos: open-loop rollout (`agents={true, pred}`), control (`agents={oracle, learned}, markers={goal}`), language steering (`agents`, `markers={concept}`, `field_`) |

Note: the eval-viz wiring that CALLS `render_diagnostics` is Phase 5 of
[design/gym_refactor.md](../design/gym_refactor.md) and may land shortly after this doc; the protocol
above (as defined in `environments/base.py`) is the extension point to build against.

## Worked example: a stock gym env end-to-end

```bash
# 1. mine play data with the env-agnostic random policy
uv run python -m quickdraw.data_generation 'environments.name=gym:Pendulum-v1' \
    data.action_sampler=random experiment=pendulum
#    -> DATA=logs/data_generation_<ts>_pendulum

# 2. onward exactly as docs/workflow.md: push_to_hub -> train_world_model -> evals
uv run python -m quickdraw.train_world_model experiment=pendulum data.root=$DATA \
    'environments.name=gym:Pendulum-v1' +run_summary.problem=... # (all 5 fields)
```

Control eval scores plans with the env's own `reward`; eval-viz uses the `render_obs` filmstrip until you
implement `render_diagnostics`. To go beyond `random` play data or single-view eval videos, graduate to
Path B.
