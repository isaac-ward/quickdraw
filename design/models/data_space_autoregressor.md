# DSAR — Data-Space Autoregressor (Causal Transformer, Direct Observation Prediction)

The control model for the shoot-out: a decoder-only transformer that predicts `observation_vector`
**directly** (no latent world model, unlike seamstress's RSSM), action-conditioned, autoregressive
to arbitrary horizon. PyTorch + Lightning + Hydra. Inspired by `architecture_old`'s transformer
block; differs by being causal, pre-norm, temporal, and data-space.

See `high_level.md` for the shoot-out overview, comparison table, and taxonomy. The sibling
`latent_space_autoregressor.md` (LSAR) predicts in a learned latent space; both descend from the
same `SequenceWorldModel` ancestor (below). In teacher-forced mode DSAR and LSAR share the **entire
forward pass up to the prediction head** — identical encoder, fuser, transformer, hidden `hₜ`; they
diverge only at the head/loss and in free-running rollout (DSAR feeds back the observation and
re-encodes it; LSAR feeds back the latent and never re-encodes). That divergence *is* the
data-space/latent-space distinction.

## Shared ancestor — `SequenceWorldModel`

DSAR and LSAR are subclasses of one ancestor that owns everything common: the action encoder
`enc_a (2→d)`, the observation encoder `enc (6→dz)` (shared with LSAR; the fuser up-projects `dz→d`),
the `TokenStreamFuser`, the causal `Transformer`
(RoPE + sliding window `W`), and **one** autoregressive rollout loop (the `p_tf` teacher-forcing
curriculum + truncated BPTT). The rollout is written against four hooks the subclass fills in:

| Hook | DSAR (this model) | LSAR (sibling) |
|---|---|---|
| `seed_state(o)` | obs as-is (carried state = obs) | `enc(o)` (carried state = latent) |
| `to_token(state, a)` | `fuse(enc(o), enc_a(a))` | `fuse(z, enc_a(a))` |
| `next_state(h, prev)` | `prev + delta_head(h)` | `prev + predictor(h)` |
| `to_obs(state)` | as-is | `dec(state)` |

Only `seed_state`/`to_obs` are identity for DSAR — and only because the carried state already *is*
the observation. `enc`/`delta_head` are real layers running inside `to_token`/`next_state`. Keeping
the rollout, curriculum, and BPTT in the ancestor means the eval and MPPI control code never changes
when swapping DSAR ↔ LSAR.

## Tokenization — full fusion, one token per timestep

Every input is a **stream** with its own encoder; one fuser collapses them to a single step-token.

- Encoders → a per-stream feature vector: `observation_vector` → `MLP(6→dz)` (the latent `z`, shared
  with LSAR); `action` → `MLP(2→d)`; later `observation_image` → ViT (`architecture_old`) pooled to
  one vector; lidar → its encoder. The fuser's per-stream `Linear(·→d)` projects each up to token
  width `d`.
- Fuse (seamstress `TokenStreamFuser`): per-stream `LayerNorm → Linear(·→d)`, concat, `MLP → d`.
  Per-stream norm absorbs differing scales (action vs obs).
- Result: stream `x₀, x₁, …`, **one fused token per timestep** `xₜ = fuse(oₜ, aₜ)`.

Adding a modality = adding a stream. No block-causal, no interleaving.

## Backbone

- **Causal self-attention decoder**: token `t` attends to `0…t` within a **sliding window `W=64`**.
- **RoPE** on the timestep index → arbitrary-length rollout, consistent relative positions.
- Reuse the `architecture_old` block (Q/K/V, GELU FFN, LayerNorm) but **pre-norm + causal**.
- Attention = **FlexAttention only** — a causal + sliding-window (`W`) `mask_mod` compiled to a
  block-sparse kernel that skips out-of-window blocks. No SDPA fallback: if FlexAttention is
  unavailable the model fails hard. (See `training.md` for the speed stack.)
- Defaults: `d=128, depth=4, heads=4 (head_dim 32), mlp_ratio=4`.

## Prediction — delta, read off the step-token

Output head on each step-token's hidden state `hₜ`: `Δ = Linear(d→6)`, `ô_{t+1} = oₜ + Δ`.
`hₜ` is the causal summary of `x₀…xₜ`, so it already conditions on the action `aₜ` inside `xₜ`.
This is plain next-token prediction (position `t` predicts `t+1`).

## Training

- Loss: MSE on `ô_{t+1}` in normalized observation space (train-split stats from `data.md`).
- **Teacher-forcing knob `p_tf ∈ [0,1]`**: at each horizon step, feed the true `oₜ` with prob
  `p_tf`, else the model's own `ôₜ`. `1.0` = pure teacher forcing (one parallel causal pass);
  `<1.0` = sequential rollout, use **truncated BPTT** (detach every few steps; teacher-forced steps
  also break the chain).
- Windows from the `delta_timestamps` loader (`P=32` past, `F=32` future).

## Cross-cutting variations

Two optional toggles apply to DSAR (and every other model) — **physical loss** (`λ_phys`) and **noise
injection** (`σ`) — specified in `variations.md`. DSAR's only model-specific detail: the physical loss
penalizes its prediction `ô` directly (no decode step needed).

## Rollout / eval

Seed with `P` context steps, feed the true `action` sequence, predict `Δ` autoregressively for
2048 steps. Score with the `environment.md` errors (`manifold_distance_error`, `pointwise_error`,
`tangent_velocity_error`). Sliding window + RoPE make the long horizon well-defined.

## Future-proofing

- **More modalities:** add an encoder + stream to the fuser (input); add a head (output). Core
  transformer unchanged.
- **Policy / inverse dynamics:** add an action head that decodes the step-token — the output-side
  mirror of fusion.
- **Multimodal fidelity:** if one fused token bottlenecks images, split that modality into multiple
  tokens → adopt block-causal masking (FlexAttention) at that point only.

## Implementation

- `DataSpaceAR(SequenceWorldModel)`: inherits stream encoders + `TokenStreamFuser` + `Transformer` +
  the shared rollout; adds the `delta_head (d→6)` and fills the four hooks above. Hydra-configured
  (`d, depth, heads, W, p_tf`). (Refactor note: this generalizes today's `BaseWorldModel` — the
  rollout loop currently in `BaseWorldModel._rollout` moves up into `SequenceWorldModel`.)
- Lightning `LightningModule` owns the `p_tf` curriculum, truncated BPTT, and the `logging.md`
  metrics. Metric functions imported from the env — single source of truth.
