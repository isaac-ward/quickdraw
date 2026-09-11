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

```python
# modalities.py -- THE modality-agnostic interface, mirroring recon_loss exactly
class Modality:
    def derivative_loss(self, pred, target):          # (B, F, ...) -> scalar
        """First-order term: match the CHANGE between consecutive steps, not just each step.
        Base = MSE on the temporal difference; correct for ANY vector modality."""
        return F.mse_loss(pred[:, 1:] - pred[:, :-1],
                          target[:, 1:] - target[:, :-1])

class ImageModality(Modality):
    def derivative_loss(self, pred, target):
        """The ONE shared VisualLoss in temporal mode -- same instance, same net, same mix ratio."""
        return self.visual.temporal(pred, target)
```

### Why `VisualLoss.temporal` and NOT a separate `derivative_loss.py`

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

Each step has a verification, per `CLAUDE.md` §4. Steps 1-3 are the term; 4-5 are the experiment.

1. **`VisualLoss._layer_features(x)`** — extract the per-layer unit-normalised VGG maps that `_lpips_term`
   already computes implicitly through torchmetrics (`net.net` is `_NoTrainLpips` with `.net = Vgg16`,
   `.L = 5`, `.chns = [64,128,256,512,512]`, plus `scaling_layer`). Refactor `_lpips_term` to use it.
   -> **verify:** `smoke/visual_loss.py` still passes 22/22 and `_lpips_term` returns bit-identical values
   on a fixed seed. This step must change no number.
2. **`VisualLoss.temporal(pred, target)`** — the §2 formula, on `(B, F, H, W, C)` input, with a
   **pair-aware** subsample (see §6).
   -> **verify:** identical sequences give exactly 0; a sequence differenced against itself shifted by one
   step gives > 0; a constant-offset sequence (`p = g + c`) gives ~0 on the pixel part, since a constant
   offset has zero temporal difference. That last check is the one that proves it measures MOTION and not
   POSITION.
3. **`Modality.derivative_loss` + `ImageModality` override + `derivative/<m>` in `recon_losses`** with a
   `derivative_weight` spec field defaulting to 0.0.
   -> **verify:** with all `derivative_weight = 0`, a 2-epoch run is bit-identical to `main` (same seed,
   same loss curve). Non-zero weight makes `derivative/<m>` appear in `metrics.jsonl` for every modality.
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
Fix: **sample 64 contiguous PAIRS `(t, t+1)` = 128 frames**, which is exactly the existing budget. Same VGG
cost as today, adjacency preserved. (For a pixel-only version you can simply use all frames, since a
subtraction and an L1 are free — but the feature part cannot afford that.)

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
* **A separate `derivative_loss.py` module** — §3. Its shared content across modalities is one subtraction.
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
