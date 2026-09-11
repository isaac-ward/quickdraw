# A first-order (derivative) loss term: match the CHANGE, not just the state

Status: **IDEA + IMPLEMENTATION PLAN. Nothing implemented.** Written 2026-09-11.

Motivated by the **block-stack environment**, where predicted rollouts flicker: content appears, vanishes,
and reappears displaced. Each frame is individually plausible; the sequence is jointly nonsense.

One sentence: today the loss asks *"does each predicted frame look like the true frame?"*; this adds
*"does the prediction CHANGE the way the truth changes?"*

---

## 1. Why flicker is currently free

The objective is `sum_t d(x_hat_t, x_t)` — **separable over t**. No term's value depends on the PAIR
(t, t+1), so temporal incoherence costs nothing beyond the per-frame error it happens to incur.

Worked, on a block that is stationary in the ground truth:

| | frame 5 | frame 6 | frame 7 |
|---|---|---|---|
| truth | block there | block there | block there |
| prediction | block there | **gone** | block there |

* **Per-frame loss today:** frame 5 matches, frame 6 is wrong, frame 7 matches. **A flicker costs ONE
  frame of error.**
* **With a derivative term:** truth's change 5->6 is *nothing*, the prediction's is *a blob vanished* —
  penalty. Then 6->7: truth *nothing*, prediction *a blob appeared* — penalty again. **Two large
  penalties, aimed exactly at the defect.**

And the part that matters most: a prediction that put the block slightly in the WRONG PLACE but held it
there scores about the same as the flicker today, and very differently with this term. **The objective
currently cannot distinguish wrong-but-coherent from wrong-and-incoherent.**

Supporting evidence that first-order behaviour is unconstrained: we already MEASURE it and never optimise
it. `latent_motion_ratio@+8` was **0.50** on starling — the model predicts half the true motion magnitude
— and no term in the loss touches that quantity.

## 2. The formulation, and the one that is wrong

```
WRONG   embedding of the difference   d( f(x_hat_{t+1} - x_hat_t) , f(x_{t+1} - x_t) )
RIGHT   difference of embeddings      d( f(x_hat_{t+1}) - f(x_hat_t) , f(x_{t+1}) - f(x_t) )
```

`f` is nonlinear. In the RIGHT form every input to the network is a real image — in domain — and the
subtraction happens in feature space, where it means "how did the content change". In the WRONG form you
hand VGG a signed, sparse, near-zero tensor it was never trained on, and the features are meaningless.

**For the PIXEL terms the two forms are identical** (`f` = identity, differencing commutes), which is why
the distinction is easy to miss. It bites only on the nonlinear feature networks.

So, writing `dp = p[:,1:] - p[:,:-1]` and `dg = g[:,1:] - g[:,:-1]`:

```
L_dt =  w_l1    * L1(dp, dg)
      + w_lpips * sum_l  || (phi_l(p_{t+1}) - phi_l(p_t)) - (phi_l(g_{t+1}) - phi_l(g_t)) ||^2
```

### 2.1 STRIDE: the same term over longer gaps, and it is nearly free

Generalise the difference to a stride `k`:  `Delta_k x_t = x_{t+k} - x_t`, and sum over a set of strides:

```
L_dt = sum_k  beta_k * d( Delta_k pred , Delta_k true )
```

**Default is `strides=(1,)`** — the plain single-step difference, which is all the flicker fix needs. The
generalisation exists because each stride measures a DIFFERENT error structure:

| stride | catches |
|---|---|
| k=1 | local motion — FLICKER |
| k=8 | accumulated displacement over ~1.3 s at 6 Hz — DRIFT that each adjacent step hides |
| k=F | total displacement across the window — did the trajectory end up in the right place RELATIVE to where it started |

These are not redundant with the per-frame loss, and the reason is sharp. A prediction offset by a CONSTANT
`c` at every frame pays `c` per frame under the per-frame loss and **exactly zero** under every `Delta_k`,
because the offset cancels. A prediction that DRIFTS LINEARLY pays a growing per-frame cost, a small
constant `Delta_1` cost, and a large `Delta_F` cost. So:

```
order 0 (existing)   absolute state at each time
Delta_1              local smoothness / motion
Delta_k, k large     accumulated displacement over that horizon
```

a temporal multi-resolution decomposition, mirroring the `@+1 / @+8 / @+16 / @+32 / @+64 / @+128` structure
the eval already reports.

**Why it is nearly free:** compute `phi_l` ONCE for a contiguous block of frames and every stride within
that block is then indexing and subtraction — no extra network passes. The number of strides costs nothing;
only the feature computation does. This is what supersedes pair-sampling in §6.

**The caveat, and it grows with k:** the mean-seeking risk of §7 gets WORSE at large stride. The average
64-step displacement of "the block might go left or right" is zero, so a large `beta_k` at large `k` pushes
hard toward a frozen prediction. `beta_k` should decay with `k`; start with `k` in {1} and only widen to
{1, 4, 16} if drift shows up separately from flicker.

**A second, DIFFERENT axis, deferred:** ORDER, i.e. `Delta^2 x_t = (x_{t+2} - x_{t+1}) - (x_{t+1} - x_t)`,
which is acceleration-like rather than displacement-like. Both are "multiple steps" and they are not the
same generalisation. For flicker, first order is the right tool; order 2 is over-engineering until there is
a measured reason.

**Use RAW unit-normalised VGG features, not LPIPS's learned per-layer weights.** Those weights were fitted
to match HUMAN JUDGEMENTS OF IMAGE SIMILARITY. Nothing calibrates them for temporal-difference similarity;
borrowing a calibration across tasks is the kind of thing that looks rigorous and is not. LPIPS already
normalises activations to unit length per spatial position before its linear layer — stop there. Make the
weighting a flag if it is worth ablating.

## 3. Where it goes

Two separable questions, and conflating them is what made this take four passes to design:

* **Who owns the term and its weight** -> a PEER term per modality, `derivative/<m>`, emitted from
  `recon_losses` where the `(B, F, ...)` shapes still exist.
* **What distance it uses** -> the modality's own, via the polymorphism that `recon_loss` already has.

**Decode site ONLY, never the roundtrip anchor.** The anchor decodes REAL encoded frames, so there is no
predicted temporal structure there to be incoherent about — a derivative term there would measure codec
smoothness at weight 10, which is not the thing we are trying to fix. This is the decisive reason it is a
peer term rather than a fourth term inside `VisualLoss`: inside, it would land at BOTH sites automatically.

### 3.1 The symmetric form: composition, and NO overrides at all

The first draft of this had `Modality.derivative_loss` as a base method with an `ImageModality` override,
mirroring `recon_loss`. That works, but it preserves an asymmetry that is already in the code and worth
removing instead: **images get a configurable distance (`VisualLoss`: `w_l2`, `w_l1`, `w_lpips`, backbone,
frame subsample) while vectors get a hardcoded `F.mse_loss` with no configurability at all.**

Give vectors a peer class and the asymmetry disappears, along with every override:

```python
# models/vector_loss.py   (NEW, sibling of visual_loss.py)
class VectorLoss(nn.Module):
    """w_l2*MSE + w_l1*L1 on a vector modality. Defaults (w_l2=1.0) are EXACTLY F.mse_loss,
    so this lands as a no-op until a weight is set -- the same contract VisualLoss shipped with."""
    def forward(self, pred, target, site="decode"): ...
    def temporal(self, pred, target, strides=(1,)): ...      # the same mix, applied to Delta_k

# visual_loss.py gains the matching method
class VisualLoss(nn.Module):
    def forward(self, pred, target, site="decode"): ...      # unchanged
    def temporal(self, pred, target, strides=(1,)): ...      # pixel Delta + feature Delta

# modalities.py -- ONE implementation, no subclass overrides
class Modality:
    def __init__(self, spec, ...):
        self.dist = build_distance(spec)       # VectorLoss or VisualLoss, chosen from spec fields

    def recon_loss(self, pred, target, site="decode"):
        return self.dist(pred, target, site=site)

    def derivative_loss(self, pred, target):
        return self.dist.temporal(pred, target, strides=self.strides)
```

`ImageModality` stops overriding loss methods entirely. **The polymorphism moves from INHERITANCE (which
method runs) to COMPOSITION (which distance object was built)** — the right call when the interface is
identical and only the implementation differs per instance.

What this buys, concretely:

* **A new modality needs no code.** A force/torque channel or a gripper-state stream declares
  `vector_l1: 1.0` in its spec and gets BOTH the reconstruction and the derivative term.
* **Multi-camera is already handled.** `cam_scene`/`cam_wrist`, and block-stack's FOUR cameras, each build
  their own `VisualLoss` and each get `temporal` with no per-camera work.
* **Proprio becomes configurable** for the first time — Huber, L1, or a mix, instead of MSE by fiat.

The cost is that it touches `recon_loss`, the most load-bearing loss path in the project with 25+
historical runs behind it. The precedent for doing it safely is `VisualLoss` itself, whose docstring
records that its defaults are "EXACTLY `F.mse_loss` ... bit-identical to the previous behaviour until a
weight is set". `VectorLoss` must land the same way, and the gate is non-negotiable (see step 0).

### 3.2 Why `temporal()` is a METHOD on the distance, not a separate `derivative_loss.py`

The criterion is *what is genuinely shared between the proprio and image temporal losses*. Answer: only
the differencing, one line. Everything else differs completely — MSE on a 16-vector versus pixel-L1 plus
VGG-feature differences. A shared module whose shared content is one subtraction is the over-abstraction
`CLAUDE.md` warns about.

The IMAGE temporal loss, by contrast, shares a great deal with the image RECONSTRUCTION loss: the cached
net, the sanitise-then-clamp guard, the frame subsample, the `w_l1 : w_lpips` ratio. That is where the
module boundary belongs. So one file owns image distances, and the modality-agnostic part lives where
polymorphism already lives.

**There is no second feature net.** `_lpips_net(device, net_type)` (`evaluation/openloop.py:29`) is a
process-level cache keyed by `(device, net_type)`; `VisualLoss` holds a REFERENCE from it
(`visual_loss.py:123-124`), it does not own one. The only refactor is a private `_layer_features(x)` that
`_lpips_term` and `temporal` both call — extraction INSIDE the file, not across files. The
sanitise-then-clamp guard in particular must stay in exactly one place: it is the fix that cost
`torus_vl128` 7.5 hours, it is non-obvious (`1e8 -> 0.0` by catastrophic cancellation, `inf -> NaN`), and a
second copy would drift.

## 4. Total loss, before and after

```
BEFORE                                        AFTER
L = lambda_flow * L_dyn                       L = lambda_flow * L_dyn
  + sum_m  w_m * D_m(rolled decode)             + sum_m  w_m * D_m(rolled decode)
  + sum_m  a_m * D_m(roundtrip)                 + sum_m  a_m * D_m(roundtrip)
                                                + sum_m  d_m * Dot_m(rolled decode)   <-- new
```

`m` over `{proprio, image, ...}`; `w_m` = modality weight (1.0); `a_m` = `latent_loss_weight` (1.0 proprio,
10 image); `d_m` = the new `derivative_weight`, **default 0.0 so every existing run stays bit-identical**.

For vl128 today `D_image = 3.0*L1 + 1.0*LPIPS-vgg + 0.0*L2`, and `Dot_image` inherits that same ratio.

## 5. Implementation plan

Each step has a verification, per `CLAUDE.md` §4. Step 0 is the symmetry refactor, 1-3 are the term,
4-5 are the experiment. **Do step 0 first**, so the derivative term lands into a symmetric structure
rather than adding a fourth thing to an asymmetric one.

0. **`VectorLoss` + `Modality.dist`, removing every loss override (§3.1).** New
   `models/vector_loss.py` with `forward` and `temporal`; `Modality.__init__` builds `self.dist` from spec
   fields; `recon_loss` becomes one non-overridden method; `ImageModality`'s `recon_loss` override is
   DELETED. New spec fields `vector_l2` (default 1.0) and `vector_l1` (default 0.0) so the constructed
   default is exactly `F.mse_loss`.
   -> **verify, and this gate is non-negotiable:** a 2-epoch run on a fixed seed is **bit-identical** to
   `main` — same `train/loss/*` and `val/loss/*` at every step, not merely close. This path carries 25+
   historical runs and a silent change to it would invalidate every comparison in
   `wizard/records/*.md`. Also: `smoke/visual_loss.py` 22/22, and a new assert that
   `VectorLoss(w_l2=1.0)(a, b) == F.mse_loss(a, b)` exactly.

1. **`VisualLoss._layer_features(x)`** — extract the per-layer unit-normalised VGG maps that `_lpips_term`
   already computes implicitly through torchmetrics (`net.net` is `_NoTrainLpips` with `.net = Vgg16`,
   `.L = 5`, `.chns = [64,128,256,512,512]`, plus `scaling_layer`). Refactor `_lpips_term` to use it.
   -> **verify:** `smoke/visual_loss.py` still passes 22/22 and `_lpips_term` returns bit-identical values
   on a fixed seed. This step must change no number.
2. **`temporal(pred, target, strides=(1,))` on BOTH distance classes** — the §2 formula for `VisualLoss`,
   the same mix on `Delta_k` for `VectorLoss`. `(B, F, ...)` input, **contiguous-segment** subsample (§6).
   -> **verify:** four checks, and the third is the one that proves the term measures MOTION and not
   POSITION.
     (a) identical sequences -> exactly 0.
     (b) a sequence against itself shifted one step -> > 0.
     (c) **a constant-offset sequence (`p = g + c`) -> ~0**, because a constant offset has zero temporal
         difference. If this is not ~0 the implementation is measuring position, not motion.
     (d) `strides=(1,)` on a 2-frame input equals the hand-written `mse(p[:,1:]-p[:,:-1], ...)`.
3. **`Modality.derivative_loss` + `derivative/<m>` in `recon_losses`**, with `derivative_weight` and
   `derivative_strides` spec fields (defaults `0.0` and `[1]`). No overrides — step 0 removed the need.
   -> **verify:** with all `derivative_weight = 0`, a 2-epoch run is bit-identical to `main`. Non-zero
   weight makes `derivative/<m>` appear in `metrics.jsonl` for **every** modality, proprio included, with
   no per-modality code.
4. **Get the block env into the pipeline.** The dataset is **`isaac-ronald-ward/block-stack`** (NOT
   `swoosh-data/lego_assemblies`, which was a wrong guess). It was written by quickdraw's own recorder --
   root `normalization_stats.json` / `summary.json` / `dataset_card.json`, lerobot splits -- so it should
   need **no processor at all**, only a `conf/data/block_stack.yaml`, exactly as `starling` did.
       fps 30 | observation_vector 17-dim | action 5-dim | 4.64 GB
       FOUR cameras at 144x192 (non-square, 4:3): scene_left, scene_right,
                                                  gripper_right_bottom, gripper_right_top
       train 43 eps x 4019 steps = 172,835 transitions (169,782 windows)
       val    2 eps x 17400      =  34,799
       eval_purple_play  5 eps   =   9,048      <- extra OOD-ish splits
       eval_purple_stack 6 eps   =   5,323
   Note the val split is only **2 episodes** (very long ones) -- thin for evaluation, so read val numbers
   with that in mind. `img_size` must be the `[144, 192]` tuple form, paired with an `ae_bottleneck` that
   divides both axes. Four cameras means `vl128_2cam`'s N-arbitrary multi-head path applies unchanged.
   -> **verify:** `check_dataset` reports the window count and no dim mismatch.
5. **Baseline first, then the arm.** Train the existing recipe on the block env and read `motion_ratio` /
   `latent_motion_ratio` across horizon BEFORE adding the term.
   -> **verify:** if those are far below 1.0 the model under-moves and this term targets it; if they are
   near 1.0 while the frames still flicker, the motion is INCOHERENT rather than INSUFFICIENT and the
   correspondence idea in §8 is the better fit. Either reading is worth having before spending a GPU.

## 6. Gotchas, all of them real

**`_flatten` destroys the time axis, at BOTH sites.** Confirmed live: the anchor passes
`(B,F,H,W,C)` and `_flatten` collapses it to `(B*F,H,W,C)`; the AR decode site arrives ALREADY flat because
`decode_loss` does `tok.reshape(-1, ...)`. So the loss currently sees `B*F` rows with no idea which are
adjacent in time. `temporal()` needs the unflattened shape — this is the one real piece of surgery.

**`visual_frames=128` INVALIDATES the term.** `_subsample` is `torch.randperm(M)[:n]` — a random
permutation of the flattened rows. Pick 128 of `B*F = 1344` and you get `(b=3,t=17)`, `(b=0,t=52)`,
`(b=14,t=6)`...; differencing consecutive entries computes `frame(b=3,t=17) - frame(b=0,t=52)`, two frames
from DIFFERENT EPISODES at unrelated times. Not a derivative — noise.

Fix: **sample contiguous SEGMENTS, not pairs.** e.g. 8 segments of 16 consecutive frames = 128 frames,
exactly the existing budget. Compute `phi_l` once per segment and then EVERY stride up to 15 is free
indexing (§2.1), which is strictly better than the 64-pairs scheme an earlier draft of this document
proposed — same cost, and it unlocks the multi-stride generalisation instead of foreclosing it. Segment
count vs length is the knob: more segments = more episodes represented, longer segments = larger strides
available. (A pixel-only version can just use all frames, since a subtraction and an L1 are free; the
feature part cannot.)

**Never difference adjacent rows of the FLAT tensor.** In the flat layout row `b*F + (F-1)` is followed by
`(b+1)*F + 0`, which crosses an episode boundary and fabricates a huge spurious delta at every seam.
Reshape to `(B, F, ...)` and difference along dim 1 — the bug is then impossible by construction rather
than masked after the fact.

**`recon_frac` coupling.** The term needs adjacent decoded frames. `recon_frac=1.0` (our setting) decodes
every rollout frame so pairs exist; anything less, or non-contiguous, and the term is invalid. Assert it.

## 7. The risk, and its tripwire

**A difference-matching term is mean-seeking on a stochastic quantity.** If the next frame is genuinely
uncertain, matching `delta` in expectation means predicting the AVERAGE change — and the average of "the
block might go left or right" is **no motion**. A heavy `d_m` therefore pushes toward a FROZEN prediction:
the mirror image of the blur failure, arrived at from the other side.

This is not hypothetical here. `latent_motion_ratio@+8 = 0.50` says the model ALREADY under-moves.

**Tripwire: `motion_ratio` and `latent_motion_ratio`, which we already log.** If either falls below its
`d_m = 0` baseline as `d_m` rises, the term is over-weighted. Start small and sweep upward, watching that
column rather than the loss value.

## 8. Rejected alternatives, and why (the valuable part of this document)

* **Embedding of the difference** (`d(f(dp), f(dg))`) — §2. Feeds VGG an out-of-domain tensor. This was the
  first formulation written down here and it is wrong.
* **LPIPS on the difference image** — the same error wearing the existing API. `VisualLoss(dp, dg)` is
  exactly this, which is why `temporal()` must be a separate method and not a call to `forward`.
* **Symmetric pooling over time.** "Embed both trajectories, compare the embeddings" is the natural reading
  of *"a loss across a predicted trajectory vs another trajectory"*, and it is the one design that is
  GUARANTEED to fail: a mean (or max, or sum) over T frames is **invariant to permuting them**, so `[A,B,C]`
  and `[C,B,A]` receive the same loss. Any trajectory-level representation must be order-sensitive — the
  difference sequence, a causal temporal model, or an alignment.
* **A separate `derivative_loss.py` module** — §3.2. Its shared content across modalities is one
  subtraction; the real sharing is between each modality's RECONSTRUCTION and TEMPORAL distance, which is
  why `temporal()` is a method on the distance class.
* **`Modality.derivative_loss` base + `ImageModality` override** — the first design here, and workable. It
  was superseded by §3.1 because it preserves the existing asymmetry (configurable distance for images,
  hardcoded MSE for vectors) rather than removing it, and because composition gives a new modality both
  terms for free where inheritance needs a decision per modality.
* **64 contiguous PAIRS as the subsample** — superseded by contiguous SEGMENTS (§6), which cost the same
  and make every stride free rather than only stride 1.
* **Higher ORDER differences** (`Delta^2`, acceleration-like) — a different axis from stride, deferred
  until there is a measured reason; first order is what flicker calls for (§2.1).
* **LPIPS's learned per-layer weights** — §2. Calibrated for a different task.
* **Patch-correspondence consistency** (feature-space flow: match the ground truth's patch->patch
  correspondence field). Strictly more expressive — it catches *incoherent* motion, not just *wrong*
  motion, and answers "did this content persist" without object slots. Set aside as the follow-on if §5
  says the motion is incoherent rather than insufficient, and because it costs an encoder pass plus a
  256x256 similarity per frame pair.
* **Soft-DTW over the trajectory.** Aligns the sequences before scoring, so a correct-but-LATE rollout is
  not punished twice. Addresses TIMING error, which is a different failure from flicker; the user confirmed
  flicker. Keep in reserve.

## 9. Expected value, honestly

**More on the image head than on proprio.** starling's obs already CONTAINS velocity (dims 3:6) and the
per-step loss already supervises those dims, so `delta(position)` and the velocity channels are two views of
the same quantity — the proprio term still adds something (it forces the position SEQUENCE to move
correctly, which velocity supervision only does indirectly) but the image has no explicit motion channel at
all. Proprio is where this is cheapest to add and least likely to matter.

**Adjacent prior art in-repo:** `environments/base.continuity_residual` already computes a first-order
residual for proprio — but a DIFFERENT one. It compares the prediction's velocity channels against the
central difference of the prediction's OWN positions: pure self-consistency, no ground truth. This term
compares `delta(pred)` to `delta(true)`: ground-truth matching. It is also env-gated (needs
`position_idx`/`velocity_idx` and physical units) and OFF for recorded datasets, so there is no conflict
today — but check they do not fight if it is ever switched on.

**Environment-specific, so a per-recipe flag and never a global default.** A fixed-camera env (block-stack)
is the ideal case: the frame difference IS the object motion. On the drone it is dominated by camera
egomotion and would be far noisier — which is the opposite of where
`design/gaussian_splat_decoder.md` applies.
