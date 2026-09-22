# Combatting rollout drift (Ceiling B)

The open-loop latent random-walk: rolled the dynamics forward, the latent cosine decays toward orthogonal by
~60 steps and object identity is lost in transit (§8.15, §8.25). Every one-step regularizer tried lands at the
same floor-corrected residual `OL − floor ≈ 0.063` (band 0.057–0.069) and nothing has moved it.

## Root cause (the framing the whole literature converges on)

**Exposure bias / covariate shift — but ONLY on the latent dynamics loss.** Important nuance (the recipe runs
`p_tf_end=0, p_tf_warmup_epochs=1`, `p_tf_dynamics=1.0`):
- The **recon (pixel) rollout is at p_tf=0** after epoch 1 — fully own-fed. So the recon loss ALREADY trains on
  the model's own deep rollout (over F=64), decoded. Drift IS seen — in pixel space, *through the decoder*.
- The **latent dynamics loss is clean** (`p_tf_dynamics=1.0`): it steps only from TRUE latents, 1-step. It never
  sees its own drifted latents. This is where exposure bias actually lives.

So the v-field is only ever asked to step from true latents, while the recon signal that DOES see drift reaches
it only through a decoder that can absorb latent error. Correction to an earlier overstatement: the dynamics
loss is NOT naive mean-seeking MSE — `flow.loss` is `F.mse_loss(v, eps−target)` on the velocity at a random τ,
the rectified-flow objective, which is **distributional** (it samples `p(z_{t+1}|z_t,a)`). The gap is narrower:

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
- **(a) overshoot (7):** baseline + `dynamics/latent_overshoot`. Conceptually it's the depth-generalization of
  `dynamics/latent` (clean anchor = the depth-1 case), but the SHIPPED implementation keeps the clean anchor as
  `dynamics/latent` and logs the deep feeds-conditioned correction as a separate `dynamics/latent_overshoot`
  key (clearer + reuses the p_tf=0 rollout's `feeds`). Grouped-count: 4.
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
`overshoot_detach_every` to bound backprop.

**Why it still matters even though the recon rollout is at p_tf=0.** The recon rollout ALREADY runs own-fed at
p_tf=0, so the deep own-drifted contexts (`feeds`) are already computed and the recon loss already supervises
them — but only in PIXEL space, through the decoder (which can absorb latent error). The LATENT dynamics loss is
kept clean (`p_tf_dynamics=1.0`), so nothing supervises those deep contexts DIRECTLY in latent space. Overshoot
is exactly that missing direct-latent term. And because the deep contexts are already produced by the p_tf=0
recon rollout, overshoot **reuses `feeds`** (request `return_feeds=True`) rather than running a new rollout, and
it is COMPATIBLE with `compile_rollout` (the bs recipe's setting) because it only adds a PARALLEL loss-term
backbone pass — the compiled rollout is untouched. So it keeps the compiled ~1.1h rollout and adds one eager
parallel pass -> **~1.3-1.5h/epoch, NOT DF's 2.9h** (DF is eager because it noises the rollout itself).

**Relation to `p_tf_dynamics`, precisely (why 0.8 failing doesn't doom this).** Same raw material (the p_tf=0
own-contexts); the difference is what supervises the latent loss: `p_tf_dynamics=1.0` = clean only (today);
`=0.8` (ran, FAILED) = 80/20 clean/own mix — shallow (a deep coherent own-context is ~0.2^k ≈ 0) AND no separate
anchor; `→0` = deep but no anchor → blow-up/mean-collapse. Overshoot = the unexplored **deep AND anchored**
point, which the single `p_tf_dynamics` scalar cannot express (it trades anchor against depth; overshoot
decouples them into two weights). RISK, stated honestly: the recon already trains on these deep contexts through
the decoder and did NOT fix drift, so overshoot's whole bet is that a DIRECT latent target (bypassing the
forgiving decoder) is the missing piece — plausible, not guaranteed (freeze attractor / unlearnable far-drift
correction are the failure modes to watch via `motion_ratio`).

**Why the anchor prevents the `p_tf→0` blow-up (`w_1` = weight on the clean depth-1 term).** `p_tf→0` is a
SINGLE loss whose context is fully own-fed, so it has two runaway modes with nothing to counteract them:
(1) FREEZE/mean-collapse — predict a near-constant transition so the latent barely moves, so drift and the
correction target vanish (trivially low loss, `motion_ratio→0`); (2) BLOW-UP — own predictions feed the context,
errors compound, gradients explode (the §19 Jacobian mode). Overshoot KEEPS the clean depth-1 term at full
weight (`w_1=1`) and ADDS the deep terms (`w_{k>1}`, detached) on top — like a residual/skip connection, the
stable objective never leaves the sum. Freeze is now directly penalized (reproducing the TRUE 1-step transition
requires real motion, so a constant prediction has HIGH anchor loss); blow-up is damped (a stable full-weight
base gradient dominates the noisier deep terms, and `overshoot_detach_every` truncates the long-rollout Jacobian
product). Not a guarantee — if `w_{k>1}` is set too high it can still freeze — but it converts the failure mode
from "blows up" into "safely does nothing" (lands back at ~baseline), which is the right failure mode to test with.

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

#### Implementation plan

**Relation to the two teacher-forcing knobs (they are NOT stacked):**
- `p_tf` (scheduled start→end) governs the RECON (pixel) rollout; overshoot does not touch it.
- `p_tf_dynamics` (default 1.0) governs the DYNAMICS (latent) loss: `q_dyn = p_tf if p_tf_dynamics is None else
  p_tf_dynamics`. At 1.0 the latent transition loss is always clean 1-step.
- The overshoot chain is rolled PURE-OWN (`p_tf=0`), deterministically, to a controlled depth. Rolling it with a
  Bernoulli true/pred prob would just recreate `p_tf_dynamics<1` (the stochastic, shallow, no-anchor version we
  ran at 0.8 and that failed). So overshoot is the deterministic, depth-controlled, ANCHORED replacement for
  `p_tf_dynamics<1` — hence guard (1) below.

**Config (`MultiModalFlow.__init__`, threaded in `setup.py`):** `overshoot_depths: list[int] = [1]` (=[1] bit-
identical), `overshoot_weight: float = 0.0` (w for k>1; 0=off), `overshoot_detach_every: int = 1` (its own
truncation, independent of the locked recon `detach_every=32`).

**Incompatibilities — raise at build in `setup.py` (mirror the `df_scale` / `compile_rollout` guards):**
1. `p_tf_dynamics != 1.0` (`<1` or `None`) → raise. Overshoot needs the depth-1 CLEAN anchor; `p_tf_dynamics<1`
   already drifts the whole dynamics loss (no clean anchor) and is the stochastic twin. Mutually exclusive.
2. non-flow model (`name ∉ {mm_flow, flow}`) → raise (needs the flow + `_rollout_from`).
3. `compile_rollout` — **COMPATIBLE, no guard.** Unlike DF (which noises the ROLLOUT per step and so forces
   eager), overshoot does NOT modify the rollout: it reuses the (compiled) rollout's `feeds` and adds a PARALLEL
   loss-term backbone pass, exactly like the existing anchor pass. The bs recipe runs `compile_rollout=True`
   (that's why straight03 is 1.1h compiled and DF is 2.9h eager); overshoot KEEPS the compiled rollout and just
   adds one eager parallel pass. (Overshoot is latent-only — no decode.)
4. `df_rollout_level > 0` → raise for v1 (two different fed-back-past modifications, untested interaction;
   `df_scale>0` training-noise-only is fine).
5. prior mode (`dynamics_prior` set / `_proprio_prior != "none"`) → raise (proprio dynamics is chained physics,
   not the flow).
6. validation: `max(overshoot_depths) ≤ data.F`; `overshoot_weight ≥ 0`; `overshoot_detach_every ≥ 1`;
   `1 ∈ overshoot_depths`.

**Code changes:**
- `setup.py`: read the 3 knobs, apply guards (1)-(6), pass into `MultiModalFlow(...)`.
- `MultiModalFlow.__init__`: store the 3 knobs.
- `dynamics_loss` (or a `_overshoot_loss` called from the same site): keep the depth-1 term unchanged; for each
  `k>1` in `overshoot_depths`, roll `k-1` own steps from the true anchor via `_rollout_from(..., p_tf=0.0,
  detach_every=overshoot_detach_every)`, then `L += overshoot_weight * flow.loss(cond(drifted, a_k),
  z[:,k:]-drifted)`. Reuses `_rollout_from` / `rollout_train(return_feeds=True)`.
- Optional: log `dynamics/latent_overshoot` as a separate key (readability only).

**Smoke (`smoke/overshoot.py`):** (i) `overshoot_depths=[1]` OR `overshoot_weight=0` is bit-identical to the
baseline loss; (ii) `depths=[1,8], weight>0` adds a nonzero term and a gradient to `self.flow`; (iii) each of
guards (1)-(5) raises; (iv) `overshoot_detach_every` bounds the backprop graph (grad-graph depth check).

**Runs:** 2-3 arms to ep13 — small `(overshoot_depths, overshoot_weight)` sweep (e.g. `[1,8]`/`[1,16]` × w∈{0.1,0.5},
`overshoot_detach_every=1`) vs the straight03/DF references; read `OL − floor` and `motion_ratio`.

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
