# RSSMLatentAR — Stochastic Latent World Model (Dreamer-v3 style)

> **Milestone 2.** A planning doc. The deterministic LSAR family (`latent_space_autoregressor.md`) is
> milestone 1 and the tight controlled experiment; RSSMLatentAR is the *heavier generative reference*
> we add once that family works. Specified here so the shared ancestor is designed with it in mind.

The third sibling under `SequenceWorldModel` (see `high_level.md`). DSAR predicts the observation;
LSAR predicts a *deterministic* latent and prevents collapse with a swappable knob. RSSMLatentAR
predicts a **stochastic** latent through a learned **prior/posterior** pair with a KL-balanced loss
plus reconstruction — the Dreamer-v3 recipe, with the GRU replaced by our causal Transformer
("TransDreamer" substitution). It is **not** a knob combination of the LSAR framework: stochastic
latents, distributional prediction, and KL-based anti-collapse are structurally different, so it is
its own subclass.

## What question it answers

LSAR asks "which cheap trick prevents collapse best?" RSSMLatentAR asks a different question:
"**does a fully generative, stochastic latent world model beat the deterministic-JEPA family and the
data-space baseline on long-horizon torus tracking and control?**" It is the strongest, most
capable reference in the shoot-out — and the most machinery. Keeping it as a *separate, later*
milestone stops its complexity from contaminating the clean LSAR mechanism comparison.

## What it shares with the spine (fairness)

To make the comparison as fair as possible, RSSMLatentAR reuses **everything that isn't intrinsic to
being stochastic/generative**:

- the **same `enc` (6→dz)** feeding the latent (role (a) — observation fusion, per
  `latent_space_autoregressor.md`);
- the **same causal Transformer backbone** (RoPE + sliding window `W`) — this *is* the
  TransDreamer substitution: the Transformer produces the deterministic recurrent summary `h_t` that
  Dreamer's GRU normally would;
- the **same `dec` (·→6)** and the **same obs-space metrics + MPPI control harness** — eval decodes
  latents to ℝ⁶ exactly like the others, so `manifold_distance_error` / control success are directly
  comparable;
- the **same shared rollout loop, optimizer, `d`, depth, heads, horizon, batch/`P`/`F`.**

Crucially, **control uses the same MPPI on decoded obs** as every other model — we do **not** use
Dreamer's actor-critic. That keeps the *control* comparison apples-to-apples; RSSMLatentAR competes
purely as a learned dynamics model.

## What it adds / overrides (why it can't be an LSAR knob)

- **Stochastic latent.** A categorical latent `z_t` (e.g. 32 categoricals × 32 classes) with
  straight-through gradients, in place of LSAR's deterministic `z ∈ ℝ^d`.
- **Posterior** `q(z_t | h_t, o_t)` — infers `z_t` from temporal context **and** the current
  observation (consumes the role-(a) encoding). This is the obs-grounded latent.
- **Prior** `p̂(z_t | h_t)` — predicts `z_t` from temporal context **alone**, no observation. This is
  the latent-space dynamics predictor (the "imagination" model) — the analog of LSAR's `predictor`,
  but it outputs a *distribution*, not a delta.
- **KL-balanced loss** (below) instead of `L_pred`. There is no residual/delta target vector to
  regress to, so the delta-prediction code path does not apply.
- **Reconstruction is integral**, not optional: the decoder gradient always flows into `enc`
  (`recon_grad = flows-in`), and reconstruction + KL *are* the anti-collapse mechanism. (In LSAR terms
  it is nearest the **reconstruction** baseline, but with a stochastic latent and a learned prior.)
- We **drop** Dreamer's reward and continue heads and its actor-critic — there is no reward in the
  world-model objective (reward is computed externally from decoded obs at MPPI time).

## Loss

Per step, summed over the sequence:

```
L  =  L_rec                                   ‖ dec(h_t, z_t) − o_t ‖²   (z_t ~ posterior)
   +  β_dyn · KL[ sg(q(z_t|h_t,o_t)) ‖ p̂(z_t|h_t) ]      dynamics loss: train the prior toward the posterior
   +  β_rep · KL[ q(z_t|h_t,o_t) ‖ sg(p̂(z_t|h_t)) ]      representation loss: train the posterior toward the prior
```

- **KL balancing:** `β_dyn > β_rep` (Dreamer-v3 defaults ≈ 0.5 / 0.1) — move the prior toward the
  posterior faster than the reverse, so the representation stays informative.
- **Free bits:** clip each KL term below a floor (~1 nat) so the model doesn't over-regularize a
  latent that is already easy to predict.
- Collapse is prevented structurally: reconstruction forces `z` to stay informative; the prior/posterior
  KL trains the dynamics without letting the posterior degenerate. No SIGReg/VICReg/EMA needed.

## Teacher forcing unifies with `p_tf`

The shared rollout's `p_tf` curriculum maps cleanly onto Dreamer's posterior/prior split:

- **posterior step** (uses the true observation) = **teacher-forced** — feed back `z_t ~ q(·|h_t,o_t)`.
- **prior step** (imagined, no observation) = **free-running** — feed back `z_t ~ p̂(·|h_t)`.

So `p_tf` = probability of taking a posterior (obs-grounded) step vs a prior (imagined) step, and
`detach_every` applies to the fed-back latent exactly as in LSAR/DSAR. Open-loop eval is the
`p_tf = 0` limit: roll the **prior** forward from context, decode, score. Same machinery, no new
rollout code.

## Hook mapping — state-bundling keeps the ancestor unchanged

Dreamer decodes from the **concat of the deterministic hidden `h_t` and the stochastic `z_t`**, so the
readout needs `h_t`, which LSAR/DSAR never carry. Rather than widen the shared `to_obs` signature
(which would leak RSSM's needs into a contract the other two share), we keep **all hooks unary** and
let the **state be model-defined**: RSSM's carried state is the **tuple `(h_t, z_t)`**. The rollout
already has `h_t` in scope right after the Transformer call, so `next_state` simply bundles it in.

| Hook | RSSMLatentAR (state = `(h, z)`) |
|---|---|
| `seed_state(o)` | `h` from context, `z ~ q(·|h, o)` (posterior); carry `(h, z)` |
| `to_token(s, a)` | `action_fuser(embed(s.z), enc_a(a))` — unpack `z` from the state |
| `next_state(h, s_prev)` | `z ~ p̂(·|h)` (prior) — or `q(·|h, o)` when teacher-forced; carry `(h, z)` |
| `to_obs(s)` | `dec(concat(s.h, s.z))` |

So the `SequenceWorldModel` contract is **untouched** — `state` stays an opaque, model-defined object
(`o` for DSAR, `z ∈ ℝ^{dz}` for LSAR, `(h, z)` for RSSM) and every hook stays unary. The only RSSM
cost is unpacking `z` from its state in `to_token`. No ancestor refactor at milestone 2.

## Implementation (planned)

- `RSSMLatentAR(SequenceWorldModel)`: inherits `enc`, `enc_a`, fuser, Transformer, rollout; adds the
  posterior head `q`, the prior head `p̂`, a latent embedding `embed(z)→ℝ^d`, and `dec`.
- Loss overrides the LSAR objective entirely (KL-balanced + recon + free bits); no `CollapseStrategy`.
- Reuses the shared `LightningModule` rollout, `p_tf` curriculum, BPTT, and obs-space metrics; logs
  KL terms, posterior entropy, and prior/posterior agreement as the RSSM-specific diagnostics.

## Open / deferred to milestone 2

- Categorical dims (`32×32`?) and whether a smaller discrete latent suits the 4-D torus state.
- Whether to keep a separate deterministic recurrent channel beyond the Transformer hidden `h_t`, or
  let `h_t` serve as Dreamer's `h` directly (current plan: the latter).
- KL-balance weights and free-bits floor — swept *within* RSSM, reported explicitly, exactly as the
  LSAR fairness protocol prescribes.
