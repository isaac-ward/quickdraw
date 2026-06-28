# Models — High-Level Overview (the world-model shoot-out)

The shoot-out compares **how a world model should represent and predict the world** on the torus
benchmark. Every model shares one spine — stream encoders, a `TokenStreamFuser`, a causal
Transformer backbone (RoPE + sliding window), and a single autoregressive rollout (teacher-forcing
curriculum `p_tf` + truncated BPTT `detach_every`). Models differ in exactly two places: **where the
prediction lives** (raw observation space vs. a learned latent space) and, for latent models, **what
prevents representational collapse**. Holding the spine fixed makes the comparison honest: any
difference in long-horizon manifold-tracking, OOD behaviour, or control performance is attributable
to the representation/collapse choice, not to backbone luck.

Three model classes descend from the shared ancestor:

- **DSAR — Data-Space Autoregressor** — predicts the observation directly. *(Milestone 1.)*
- **LSAR — Latent-Space Autoregressor** — predicts a *deterministic* latent (JEPA style); one class,
  several anti-collapse mechanisms selected by config. *(Milestone 1.)*
- **RSSMLatentAR** — predicts a *stochastic* latent via a learned prior/posterior with KL +
  reconstruction (Dreamer-v3 style). A separate, heavier sibling kept as a reference point.
  *(Milestone 2 — see `rssm.md`.)*

A crucial fairness fact: **in teacher-forced mode DSAR and LSAR share the entire forward pass up to
the prediction head** — identical encoder, fuser, transformer, hidden `h_t`. They diverge only at
the head/loss and during free-running rollout (DSAR feeds back the observation and re-encodes; LSAR
feeds back the latent and never re-encodes). That single divergence *is* the data-space/latent-space
distinction.

## The models

Each model below: **what it does · why it's different · what we expect.**

- **DSAR — Data-Space Autoregressor.**
  Predicts the next observation directly in ℝ⁶ as a residual on the current observation
  (`ô_{t+1} = o_t + Δ`), trained by MSE against ground truth. It has no latent bottleneck and no
  collapse failure mode — supervision is always grounded in real data — which makes it the reference
  point for the whole shoot-out. We expect a strong, stable baseline that nails short horizons but
  may accumulate error and drift off the manifold over long rollouts, since it models raw coordinates
  rather than abstracting the dynamics.

- **LSAR: naked** *(negative control)*.
  Encodes each observation into a latent, predicts the next latent as a residual, and trains only
  against its own encoder's output for the true next observation. It is distinguished by having **no
  collapse-prevention mechanism at all** — it exists to prove the problem is real. We expect it to
  collapse: the encoder maps everything to a near-constant latent that is trivially predictable, the
  prediction loss craters toward zero, and the obs-space readout becomes useless.

- **LSAR: reconstruction** *(positive baseline)*.
  Adds a decoder whose gradient flows back into the encoder, forcing each latent to retain enough
  information to reconstruct its observation while the predictor learns latent dynamics. It differs
  by using **reconstruction itself** as the anti-collapse force — the classic autoencoder-style world
  model — rather than a regularizer or target asymmetry. We expect it to avoid collapse and work
  reasonably, but to spend latent capacity faithfully encoding every detail of the observation
  (relevant to the dynamics or not), which may not be the most predictive representation.

- **LSAR: EMA** *(BYOL / I-JEPA style)*.
  Builds prediction targets from a slow exponential-moving-average copy of the encoder
  (stop-gradient), so the fast online encoder forever chases a lagging, still-structured target. It
  prevents collapse **purely architecturally** — through the fast/slow asymmetry — with no penalty
  term and no decoder gradient. We expect a compact, predictive latent that abstracts the dynamics
  and may extrapolate better than reconstruction, at the cost of sensitivity to the EMA decay rate.

- **LSAR: SIGReg** *(LeJEPA)*.
  Keeps a single (online) encoder and adds the SIGReg penalty, which pushes the batch distribution of
  latents toward an isotropic Gaussian via sketched 1-D normality tests. It differs by preventing
  collapse through an **explicit distributional regularizer with a single hyperparameter** — no EMA,
  no stop-gradient, no negatives. We expect stable, well-spread latents and a principled anti-collapse
  signal; the open question is whether the Gaussian-isotropy prior helps or hurts on this
  low-dimensional torus dynamics.

- **LSAR: VICReg.**
  Keeps a single (online) encoder and adds VICReg's variance + covariance penalties: a hinge keeping
  each latent dimension's standard deviation above a floor, plus an off-diagonal covariance term that
  decorrelates dimensions. It differs by attacking collapse **explicitly per-dimension** (variance
  stops dimensional collapse; covariance stops redundancy) rather than via a global distributional
  test or architectural asymmetry. We expect it to reliably prevent collapse and produce decorrelated,
  well-used latent dimensions, with performance sensitive to the variance/covariance/invariance
  weighting.

- **RSSMLatentAR** *(Dreamer-v3 style — milestone 2)*.
  Encodes each observation into a *stochastic* (categorical) latent and learns a prior `p̂(z_t|h_t)`
  that predicts the next latent from temporal context alone, matched to an observation-grounded
  posterior `q(z_t|h_t,o_t)` via a KL-balanced loss plus reconstruction. It differs from the LSAR
  family by being **generative and stochastic** — collapse is prevented structurally (reconstruction
  + prior↔posterior KL with free bits), not by a swappable knob, so it is a separate subclass rather
  than a knob combination. We expect it to be a strong long-horizon world model and the most capable
  reference, at the cost of more machinery (categorical sampling, straight-through, KL balancing) and
  a heavier training recipe.

## Comparison table

All LSAR variants share encoder, predictor, decoder, backbone, and the delta-prediction loss
`L_pred`; a mechanism is one setting of three orthogonal knobs — **`target`** (where prediction
targets come from), **`reg`** (distributional penalty), **`λ_rec`** (does the decoder's gradient
reach the encoder). Everything *not* in those three knobs is frozen across the whole sweep — in
particular the **latent width `dz` (`< d`) is a single fixed value used by DSAR and every LSAR
mechanism**, never tuned per mechanism (a swept `dz` would confound "did the mechanism help" with
"did this latent size help"; a fixed `dz` is just part of the shared substrate).

| Model | predicts | `target` | `reg` | `λ_rec` (decoder→enc) | what stops collapse |
|---|---|---|---|---|---|
| **DSAR** | `o_t + Δ` (obs) | — (true `o_{t+1}`) | — | — | grounded in true obs; collapse impossible |
| **LSAR: naked** | `z_t + Δ` (latent) | online | none | 0 (probe) | nothing — negative control, should collapse |
| **LSAR: reconstruction** | `z_t + Δ` | online | none | **>0 (flows in)** | decoder gradient keeps `enc` informative |
| **LSAR: EMA** | `z_t + Δ` | **EMA, stop-grad** | none | 0 (probe) | fast-online / slow-target asymmetry |
| **LSAR: SIGReg** | `z_t + Δ` | online | **SIGReg** | 0 (probe) | latents pushed to isotropic Gaussian |
| **LSAR: VICReg** | `z_t + Δ` | online | **VICReg** | 0 (probe) | per-dim variance floor + decorrelation |
| **RSSMLatentAR** † | stochastic `z ~ q(·\|h,o)` | posterior | KL (balanced + free bits) | flows-in | reconstruction + prior↔posterior KL |

† Not a knob combination — a separate generative/stochastic subclass (Dreamer-v3 style), milestone 2.
See `rssm.md`. The three-knob framing applies only to the deterministic LSAR family.

## Taxonomy

```
SequenceWorldModel  (shared ancestor)
│   enc (6→dz) · enc_a (2→d) · fuser · causal Transformer (RoPE, window W)
│   one autoregressive rollout: p_tf curriculum + truncated BPTT (detach_every)
│   delta/residual prediction · obs-space metrics via to_obs()
│
├── DSAR  — Data-Space Autoregressor
│     carried state = obs o_t  (re-encoded every step)
│     readout = delta_head (d→6):  ô_{t+1} = o_t + Δ
│     loss = MSE on obs
│     collapse: not applicable (supervised by true obs)
│     role: reference / control for the shoot-out
│
├── LSAR  — Latent-Space Autoregressor   (carried state = latent z_t = enc(o_t) ∈ ℝ^dz, dz < d)
│     readout = predictor (d→dz):  ẑ_{t+1} = z_t + Δ ;  dec (dz→6) for obs readout
│     loss = L_pred + λ_reg·L_reg + λ_rec·L_rec        [milestone 1]
│     collapse: a real risk → distinguished entirely by 3 knobs (target / reg / λ_rec):
│     │
│     ├── naked          target=online · reg=none   · λ_rec=0    → negative control (should collapse)
│     ├── reconstruction target=online · reg=none   · λ_rec>0    → positive baseline (decoder-gradient cure)
│     ├── EMA            target=EMA,sg · reg=none   · λ_rec=0    → architectural asymmetry cure
│     ├── SIGReg         target=online · reg=SIGReg · λ_rec=0    → distributional cure (LeJEPA)
│     └── VICReg         target=online · reg=VICReg · λ_rec=0    → variance/covariance cure
│           (future easy adds as new reg options: SimSiam stop-grad+predictor, InfoNCE, Barlow Twins)
│
└── RSSMLatentAR — stochastic latent (Dreamer-v3 style)   [milestone 2 — separate subclass, see rssm.md]
      stochastic z ~ posterior q(z|h,o); prior p̂(z|h) is the latent-space predictor
      loss = reconstruction + KL-balanced(prior ‖ posterior) + free bits
      collapse: prevented structurally (recon + prior/posterior KL), not by a knob
```

## Cross-cutting variations & diagnostics

- **Variations axis (`variations.md`).** Three orthogonal toggles, all off by default, addable to
  **any model** and combinable: **physical loss** (`λ_phys` — penalize predictions leaving the torus /
  non-tangent velocity, using the `environment.md` geometry), **noise injection** (`σ_noise` —
  Gaussian noise on the observation inputs for off-manifold robustness), and **contraction**
  (`λ_contract`, `τ` — cap the spectral norm `σ_max` of the one-step state-Jacobian via a one-sided
  hinge, so errors don't compound over long rollouts). A shoot-out cell is *(model class) × (collapse
  mechanism, if LSAR) × (noise on/off) × (physical loss on/off) × (contraction on/off)*. None is an
  anti-collapse force; all three shape prediction robustness/stability and compose with the collapse
  axis. All are train-time only — inference is unaffected.
- **Collapse diagnostics (latent models).** Each run logs a `collapse/` panel — effective rank,
  per-dim std, off-diagonal correlation mass, `L_pred` — **once per validation epoch** (never a
  per-step bar plot; cheap enough that 50-step cadence isn't worth it). Full plot/caption spec and the
  per-mechanism extras (`collapse/<mechanism>/…`) are in `logging.md`; plain-language explanations of
  the four diagnostics are in `collapse_explanations.md`.

## Detailed specs

- `data_space_autoregressor.md` — DSAR architecture, tokenization, backbone, training, rollout.
- `latent_space_autoregressor.md` — LSAR architecture, the 3-knob collapse framework, per-mechanism
  math, decoder/probe handling, shared-ancestor refactor, fairness protocol.
- `rssm.md` — RSSMLatentAR (Dreamer-v3 style), the stochastic/generative sibling; milestone 2.
- `variations.md` — cross-cutting toggles (physical loss, noise injection) for any model.
- `collapse_explanations.md` — plain-language reference for the four `collapse/` diagnostics.
