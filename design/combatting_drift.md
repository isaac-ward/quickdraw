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

**Explicit terms (ungrouped), for the record:**
- **Baseline `bs_ss10_2cam` (6):** `dynamics/latent`, `decode/proprio`, `decode/cam_scene`, `decode/cam_wrist`,
  `codec/roundtrip_cam_scene`, `codec/roundtrip_cam_wrist`. (No `codec/roundtrip_proprio` — proprio
  `latent_loss_weight=0`. No `decode/*_shortcut` — x0 decode, shortcut off. No `derivative/*` — weight 0. No
  `dynamics/latent_shortcut` — dynamics shortcut off.)
- **straight03 (7):** baseline + `variations/latent_straightness`.
- **DF (6):** baseline (diffusion forcing noises the context + adds a level embedding, but scores through the
  existing `dynamics/latent` — no new term).
- **(a) overshoot (6):** GENERALIZES `dynamics/latent` over depths — same 6 terms (no new named term); a
  separate `dynamics/latent_overshoot` key is optional, for logging the deep part only.
- **(b) moment/MMD (7):** baseline + `variations/trajectory_mmd` (or `+2` if `Δ‖z‖` and `‖z‖` logged separately).
- **(c) latent GAN (8):** baseline + `dynamics/adv_generator` + `discriminator`.

### (a) Latent overshooting (PlaNet, arXiv 1811.04551) — lowest risk, do first

Roll `k` steps in latent space feeding own predictions and match the rolled `ẑ_{t+k}` to the TRUE `z_{t+k}` with
`flow.loss` (reuses the existing flow — no new network). Constrains multi-step *marginals*: not the full joint,
but far more than per-step.

**This is the DEPTH-GENERALIZATION of the current `dynamics/latent`, not a separate term.** Today's loss is the
`k=1` case: at `p_tf_dynamics=1` the rollout is teacher-forced, so `dynamics/latent` is `L_flow` on the TRUE
1-step context. Generalize it over a SET of rollout depths `D` (default `{1}` = bit-identical to today); for
`k>1` the only change is that the context is the model's own `k`-step rollout instead of the truth:

    L_dynamics = Σ_{k in D} w_k · E_t L_flow( cond(rollout_own(z_t,a,k−1), a),  z_{t+k} − ẑ_{t+k-1} )
                 # D={1}        -> current loss, unchanged
                 # D={1,4,8,16} -> keep the stable 1-step anchor (w_1=1) + weighted deep supervision

So it adds NO new named term — `dynamics/latent` just becomes a sum over depths (a separate `*_overshoot` key is
only for readability if you want to log the deep part). It is the SAME axis as `p_tf_dynamics` (own-vs-true
context), made DETERMINISTIC and explicit-depth rather than a Bernoulli mix — and it reuses the existing
machinery (`rollout_train(return_feeds=True)` already yields what each step stood on; `dynamics_loss` already
conditions the flow loss on those feeds). Keep depth-1 as the always-on stable anchor; deep depths get
`overshoot_detach_every` to bound backprop. Why it still matters at p_tf=1: today NOTHING compounds — both
`dynamics/latent` (1-step latent) and `decode/*` (1-step pixel) are trained from truth — so the deep depths are
the only place the model sees, and is corrected on, its own compounding drift in latent space.

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
- **Loss terms TOTAL: 3** — it GENERALIZES `dynamics/latent` over depths, so no new named term (becomes 4 only
  if you choose to log the deep-depth part as a separate `*_overshoot` key).
- **Relation to `p_tf<1` (why it's safe, and why it's not the same).** `p_tf<1` REPLACES the context mix at
  every position with a Bernoulli own/true draw — near steps included — which is why `p_tf→0` blows up /
  mean-collapses and `p_tf=0.8` only gave shallow, rare drift. Overshoot instead keeps the clean depth-1
  true-context term (the **always-on anchor**, `w_1=1`, = today's loss) fixed and ADDS a deterministic,
  controlled-depth deep term. So it reaches reliable deep-drift exposure WITHOUT entering the unstable pure-AR
  regime. "Anchor" = the loss that works never leaves the sum.
- **Implementation (reuses existing machinery, no new module, no loop change):**

      # MultiModalFlow.__init__: overshoot_depths=(1,)  overshoot_weight=0.0  overshoot_detach_every=1
      z = encode_state(obs, anchor)                          # true latents (already computed)
      L = flow.loss(cond_true_t, z[:,1:] - z[:,:-1])         # depth-1 anchor (w_1=1) == CURRENT loss, unchanged
      if overshoot_weight > 0 and training:
          for k in [d for d in overshoot_depths if d > 1]:
              drifted = _rollout_from(true_window, actions, steps=k-1,     # own-fed (p_tf=0), OWN truncation
                                      p_tf=0.0, detach_every=overshoot_detach_every)
              L += overshoot_weight * flow.loss(cond(drifted, a_k), z[:,k:] - drifted)  # 1-step correction

  `_rollout_from` / `rollout_train(return_feeds=True)` ALREADY do own-fed rollouts with a `detach_every` and
  return what each step stood on; `flow.loss` is the same primitive as the anchor. So it's a loop over depths
  adding `flow.loss` on reused tensors — no new network, no manual-optimization surgery (unlike the GAN).
- **Runs to validate:** 2–3 arms to ep13 — a small `(k, weight)` sweep vs the straight03/DF references, read
  `OL − floor` and `motion_ratio`. Cheap in code, slower per-epoch (extra own-rollout in the loss).
- **Blast radius: MEDIUM.** One function in `multimodal.py`; reuses `rollout_train`'s tensors + the existing
  `flow.loss`; no new modules, no training-loop change. `overshoot_weight=0` / `depths=(1,)` is bit-identical.

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

## Math / pseudocode for the new terms

Shared primitive — our existing rectified-flow loss (`param="v"`), the thing `dynamics/latent` already is:

    L_flow(c, y) = E_{τ~U(0,1), ε~N(0,I)}  || v_θ(x_τ, τ, c) − (ε − y) ||²,   x_τ = (1−τ)·y + τ·ε

`c` = backbone conditioning from context+action; `y` = residual target `z_{t+1} − z_t`. Current 1-step term:
`L_1 = E_t L_flow(c_t^true, z_{t+1} − z_t)` — TRUE context, so it never compounds (p_tf=1).

Write `enc(·)` for the encoder (true latent `z_t = enc(o_t)`), `step(s, a)` for one flow rollout step producing
the next latent from state `s` (own-fed), `cond(s, a)` for the backbone conditioning at state `s`.

### (a) Latent overshoot — `dynamics/latent_overshoot`

Roll `k` OWN steps from a true anchor, then score the compounded latent against truth in latent space:

    # anchor at true z_t; roll own predictions forward k steps
    s = z_t                                        # true window end
    for j in 1..k:
        ẑ = step(s, a_{t+j-1})                     # OWN prediction fed forward (this is the compounding)
        if j % overshoot_detach_every == 0:
            ẑ = ẑ.detach()                         # truncate backprop; =1 -> pure-forward (DAgger regime)
        s = slide(s, ẑ)                            # advance the window with the OWN pred
    L_overshoot = λ_over · E_t [ L_flow( cond(s, a_{t+k-1}),  z_{t+k} − ẑ_{t+k-1} ) ]
    # target = correction from the DRIFTED own-latent ẑ_{t+k-1} back to the TRUE next z_{t+k}

(Optionally sum over several k, e.g. {4,8,16}.) Contrast with `L_1`: the conditioning `cond(s,·)` is built from
the model's OWN rolled context, not the true one — that's the whole point.

### (b) Trajectory moment / MMD — `variations/trajectory_mmd`

Roll H own steps → `ẑ_{1:H}`; encode truth → `z_{1:H}`. Form per-step statistics (the walk's signature):

    d̂_t = ‖ẑ_{t+1} − ẑ_t‖,  n̂_t = ‖ẑ_t‖      (generated)
    d*_t = ‖z_{t+1} − z_t‖,  n*_t = ‖z_t‖      (real)

**Plain moment loss** (match low-order moments):

    L_moment = λ · [ (mean d̂ − mean d*)² + (var d̂ − var d*)²
                   + (mean n̂ − mean n*)² + (var n̂ − var n*)² ]

**MMD loss** (match whole distributions; RBF kernel k(a,b)=exp(−‖a−b‖²/2σ²); X=generated stat set, Y=real):

    MMD²(X,Y) = mean_{i,i'} k(x_i,x_{i'}) + mean_{j,j'} k(y_j,y_{j'}) − 2·mean_{i,j} k(x_i,y_j)
    L_mmd = λ · MMD²( {[d̂_t, n̂_t]},  {[d*_t, n*_t]} )

No decoder, no adversary — just statistics of the rolled vs real latent trajectory.

### (c) Latent-trajectory GAN — `dynamics/adv_generator` + `discriminator`

A discriminator `D_φ` over a latent-bag SEQUENCE (small temporal transformer/conv, (H,n_state,d)→scalar).
Generator = the flow rolling `ẑ_{1:H}` (history stop-grad; gradient enters through the current step). Hinge form:

    ẑ_{1:H} = rollout_own(z_0, a)          # own-fed; deep history detached, see the stop-grad mechanism above
    z_{1:H} = enc(true frames)
    L_D = E[ relu(1 − D_φ(z_{1:H})) ] + E[ relu(1 + D_φ(ẑ_{1:H})) ]      # train D (its own optimizer)
    L_G = − E[ D_φ(ẑ_{1:H}) ]                                            # train the flow to look real
    L_adv_generator = λ_gan · L_G

`D_φ` is the learned, sequence-level distribution matcher: can't mean-collapse (a mean/frozen trajectory reads
as fake), can't point-explode (no per-sample target). Alternates G/D steps → manual optimization in `lit.py`.

## Recommended order

1. **(a) overshoot on a stop-grad deep rollout** — GENERALIZES `dynamics/latent` over depths (no new term, total
   3 grouped), MEDIUM blast, no adversary, folds in the deep-drift mechanism. The honest first test of "does
   constraining multi-step marginals move 0.063?".
2. **(b) trajectory-MMD variation** — 1 new term (total 4), LOW blast. Targets the walk's signature directly.
3. **(c) latent-trajectory GAN** — 2 new terms (total 5) + a network + manual-optimization surgery. Only if (a)/(b)
   plateau. This is the full Self-Forcing fix.
4. **(d) DMD** — skip.

All four are flow-preserving (no paradigm change). None of them touch the DECODE side (Ceiling A / identity
render) — for that, see `design/identity_preservation.md`.
