# Combatting rollout drift (Ceiling B)

The open-loop latent random-walk: rolled the dynamics forward, the latent cosine decays toward orthogonal by
~60 steps and object identity is lost in transit (§8.15, §8.25). Every one-step regularizer tried lands at the
same floor-corrected residual `OL − floor ≈ 0.063` (band 0.057–0.069) and nothing has moved it.

## Root cause (the framing the whole literature converges on)

**Exposure bias / covariate shift.** We train the dynamics loss on CLEAN ground-truth context (teacher forced,
`p_tf_dynamics=1`), so the v-field is only ever asked to step from *true* latents. At rollout it steps from its
*own* latents, which it has (mostly) never seen. Correction to an earlier overstatement: our dynamics loss is
NOT naive mean-seeking MSE — `flow.loss` is `F.mse_loss(v, eps−target)` on the velocity at a random τ, i.e. the
rectified-flow objective, which is **distributional** (it samples `p(z_{t+1}|z_t,a)`). The gap is narrower and
sharper than "MSE → mean":

- Our loss constrains each **one-step conditional**. Every transition is individually plausible.
- Nothing constrains the **joint trajectory** `p(z_{1:H}|z_0,a_{1:H})`. A chain of individually-plausible steps
  walks off-manifold: 97%-right, 60 times, compounds to garbage, and the per-step loss never sees it.

We DID try scheduled-sampling-lite (`p_tf_dynamics=0.8`, ran to ep13, §8.17): it feeds own predictions up to the
training horizon **F=64**, and it landed at the baseline. Two reasons it wasn't enough: (1) **depth** — it sees
drift only to F=64, never the eval horizon 1651 where identity actually dies; (2) **the objective is still
per-step** — plausible-per-transition, unconstrained-per-trajectory.

So the family of fixes below all move from **per-step** to **sequence-level** supervision, and/or expose the
model to its own DEEP drift cheaply (roll deep in the forward, truncate the backward).

## The stop-grad deep-rollout mechanism (shared by all of these)

The recurring trick for "train on deep drift without an H-deep backprop graph": roll the model H steps feeding
its OWN predictions (via the KV-cache, exactly like inference), but treat the history as **fixed context under
stop-grad** — the forward pass sees deep drift, the backward graph stays short (gradient enters only through the
current step / the final loss). This is what lets any of the losses below train in the drifted regime. Option
(a) folds it in directly (its own `overshoot_detach_every`); option (c) pairs it with a critic.

---

## The ladder (cheapest → heaviest)

Baseline `bs_ss10_2cam` optimizes **3 loss terms** (counting all per-head `decode/*` as one `decode` term and
all `codec/roundtrip_*` as one `roundtrip` term): `dynamics/latent`, `decode`, `roundtrip`. Totals below are
*for an arm built on that baseline*. (For reference: straight03 = 3 + `latent_straightness` = 4; DF = 3 + 0 = 3,
since diffusion forcing adds no loss term.)

### (a) Latent overshooting (PlaNet, arXiv 1811.04551) — lowest risk, do first

Roll `k` steps in latent space feeding own predictions and match the rolled `ẑ_{t+k}` to the TRUE `z_{t+k}` with
`flow.loss` (reuses the existing flow — no new network). Constrains multi-step *marginals*: not the full joint,
but far more than per-step.

- **How many steps (`k`)?** The overshoot horizon = how far you roll before matching to truth. Start modest and
  sweep: `k ∈ {4, 8, 16}`. You can supervise a single horizon (`k=16`) or a few (dense, `{4,8,16}`). Larger k =
  deeper drift exposure, but a longer forward AND a harder target (the correction back to truth grows with k).
- **Why its OWN `detach_every`?** The main recon rollout's `detach_every=32` is LOCKED (§8: 8 froze the model,
  32 is load-bearing). The overshoot rollout is a *separate, possibly deeper* rollout whose backprop graph we
  must bound independently — so it gets `overshoot_detach_every`. Set it to 1 for a **pure-forward, fully
  stop-grad** overshoot (cheapest; the DAgger/"data-as-demonstrator" regime — see deep drift, zero long
  backprop) up to `k` for full BPTT through it. Decoupling it means we tune the overshoot's gradient depth
  WITHOUT touching the locked recon truncation.
- **`overshoot_weight` (λ):** scalar multiplying the overshoot loss term in the total, balancing it against the
  1-step `dynamics/latent` term and the recon/anchor terms. Start ~0.1–0.5; watch `motion_ratio` (too high →
  the freeze/mean-collapse attractor).
- **New config (`MultiModalFlow.__init__`):** `overshoot_k` (int or list), `overshoot_weight` (float),
  `overshoot_detach_every` (int). All 0/None → off, bit-identical.
- **Loss terms TOTAL: 3 + 1 = 4** (one aggregated `dynamics/latent_overshoot`; `3 + len(k)` if logged per-horizon).
- **Runs to validate:** 2–3 arms to ep13 — a small `(k, weight)` sweep vs the straight03/DF references, read
  `OL − floor` and `motion_ratio`. Cheap in code, slower per-epoch (extra rollout in the loss).
- **Blast radius: MEDIUM.** Core dynamics loss (`multimodal.py`), but reuses tensors `rollout_train` already
  computes + the existing `flow.loss`; no new modules, no training-loop change. Must leave the 1-step loss intact.

### (b) Trajectory moment / MMD matching — cheap, isolated, non-adversarial

Roll H steps and match the DISTRIBUTION of trajectory statistics (generated vs real) — specifically the step
deltas `‖Δz_t‖` and latent norms `‖z_t‖`, which are the walk's signature (drift = deltas that don't cancel +
growing norm).

- **Plain moment loss:** match a few LOW-ORDER MOMENTS — e.g. `E[‖Δz‖]`, `Var[‖Δz‖]`, `E[‖z‖]` — of generated vs
  real, via MSE on those scalars. Simple, cheap, differentiable; only constrains the moments you pick.
- **MMD (Maximum Mean Discrepancy):** a kernel distance between the two *whole distributions*:
  `MMD² = E[k(x,x')] + E[k(y,y')] − 2E[k(x,y)]`, `x`~generated stats, `y`~real stats. With an RBF kernel it
  captures ALL moments implicitly (the full distribution shape), not just mean/variance. Stronger than moment
  matching (moment ⊂ MMD in what it constrains), at the cost of a batch of samples + a kernel bandwidth to set.
- **New config (a `Variation`, sibling of `latent_straightness`):** `variations.trajectory_mmd.weight`, `.stats`
  (`delta_norm|latent_norm`), `.kernel`/`.bandwidth`. Off by default = bit-identical.
- **Loss terms TOTAL: 3 + 1 = 4** (one aggregated term; `3 + 2` if `Δ‖`-dist and `‖z‖`-dist are logged separately).
- **Runs:** 1–2 weight arms; watch `motion_ratio` (matching norms can freeze OR inflate).
- **Blast radius: LOW** *if* the variation hook exposes both the rolled and the true latents. `LatentStraightness`
  already runs on the rolled latent, so this is a near-sibling — but it needs the *true* latent too, which
  straightness doesn't use; if the hook doesn't already pass it, that's a small contract extension. Otherwise a
  new isolated `Variation`, off by default.

### (c) GAN on latent trajectories (the real Self-Forcing move, arXiv 2506.08009) — heaviest live option

A discriminator `D` over latent-bag SEQUENCES (small temporal transformer/conv on `(H, n_state, d)`): real =
encoded true val trajectories, fake = our rolled `ẑ_{1:H}`. Train the flow to fool `D`, train `D` to catch it.
`D` learns which trajectory features betray "generated" (drift, colour swaps, frozen motion) — a *learned*,
sequence-level distribution matcher. Can't mean-collapse (a mean trajectory is trivially fake to `D`); can't
point-blow-up (no per-sample target). Paired with the stop-grad deep-rollout above (history stop-grad; the
adversarial gradient enters only through the current step's generation). **This is where §"stop-grad rollout +
critic" lives** — the rollout mechanism is folded into (a); the *critic* is the piece unique to (c).

- **New config:** `gan_weight`, D arch/width, D lr, D:G update ratio, warmup, spectral-norm on D.
- **New module + training-loop change:** a discriminator network, and `lit.py` must switch from automatic to
  **manual optimization** (two optimizers, alternating G/D).
- **Loss terms TOTAL: 3 + 2 = 5** (`dynamics/adv_generator` + `discriminator`), plus a whole D network on a
  second optimizer.
- **Runs: MANY.** GANs need tuning (lr ratio, weight, warmup, D capacity); expect several arms just to stabilize
  before the effect is even readable.
- **Blast radius: HIGH.** New module + training-loop surgery + adversarial instability, and it risks perturbing
  the already-good 1-step behavior. Last rung — only after (a)/(b) show multi-step supervision moves the floor.

### (d) DMD (distribution matching distillation) — SKIP

Two score nets (a "real" score from a teacher, a "fake" score tracking the generator) + a KL-matching gradient.
**Requires a bidirectional/teacher score model we don't have** (we'd have to train one first). Loss terms 3 + 2 = 5,
but the prerequisite makes it VERY HIGH blast radius and least natural for our regime. Not pursued.

---

## Recommended order

1. **(a) overshoot on a stop-grad deep rollout** — 1 new term (total 4), MEDIUM blast, no adversary, folds in the
   deep-drift mechanism. The honest first test of "does constraining multi-step marginals move 0.063?".
2. **(b) trajectory-MMD variation** — 1 new term (total 4), LOW blast. Targets the walk's signature directly.
3. **(c) latent-trajectory GAN** — 2 new terms (total 5) + a network + manual-optimization surgery. Only if (a)/(b)
   plateau. This is the full Self-Forcing fix.
4. **(d) DMD** — skip.

All four are flow-preserving (no paradigm change). None of them touch the DECODE side (Ceiling A / identity
render) — for that, see `design/identity_preservation.md`.
