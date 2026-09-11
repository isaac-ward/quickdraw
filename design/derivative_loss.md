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

### 2.1 STRIDE IS LOCKED AT 1 — settled by prior art AND by our own loss structure

The difference generalises to a stride `k`:  `Delta_k x_t = x_{t+k} - x_t`. Both forms are just "the
difference between two frames"; `k` only changes **how far apart those two frames are**, and therefore
**which timescale of motion the term can see**:

* **`Delta_1`, a short gap** — fast changes show strongly, slow ones barely register (a slow drift moves
  almost nothing in one step). A HIGH-PASS filter on motion: sensitive to jitter and FLICKER, nearly blind
  to drift.
* **`Delta_k`, a long gap** — a slow drift has had k steps to accumulate and becomes visible, while fast
  jitter partly cancels across the gap. A LOW-PASS filter: sensitive to DRIFT, insensitive to jitter.

**`derivative_strides` SHOULD BE LOCKED AT `[1]`.** Two independent reasons, and neither is a guess:

1. **Measured, in the literature.** `Frame Difference-Based Temporal Loss for Video Stylization`
   (arXiv 2102.05822) is the paper that formalises this loss, and §4.5 tests exactly this knob: *"using the
   difference between two frames that are separated by an interval of K frames where K > 1 should work as
   well as using K = 1. This was verified by the experiments... the stylization results of models trained
   with different K are almost identical to each other."* The generalisation has been tried and does not
   pay.
2. **Our own loss structure already covers what large `k` would add.** A large `k` catches DRIFT — but a
   drifting prediction IS in the wrong place, and the existing per-frame term penalises precisely that. So
   `per-frame + Delta_1` already spans both failures: the per-frame term catches WRONG PLACE, `Delta_1`
   catches WRONG MOTION. `Delta_k` is redundant with a term we already have.

So `derivative_strides: [1]` is a **locked config value**, in the same category as
`accumulate_grad_batches=1` and `data.window_stride=1`: written down, justified, and not to be swept. The
plural signature `strides=(1,)` stays in the API only so a future reader does not have to re-derive the
generalisation to know it was considered and rejected.

**A SECOND, DIFFERENT AXIS — order, deferred.** `Delta^2 x_t = (x_{t+2} - x_{t+1}) - (x_{t+1} - x_t)` is
acceleration-like rather than displacement-like. Both are "multiple steps" and they are NOT the same
generalisation. For flicker, first order is the right tool; order 2 is over-engineering until something
measured asks for it.

### 2.2 PRIOR ART: this loss is not novel, and that is useful

Searched 2026-09-11. The formulation is established, has a name, and its open questions are already answered.

* **`Frame Difference-Based Temporal Loss for Video Stylization`** (arXiv 2102.05822) — THE closest prior
  art, essentially identical. Their eq (5)-(7):
  `L_temp = (1/2N(T-1)) sum_t || phi(I~_t) - phi(I_t) ||^2` with `phi(x_t) = f_l(x_{t+1}) - f_l(x_t)`.
  `l = 0` is the pixel frame difference (**P-FDB**), `l > 0` takes the difference in FEATURE space
  (**F-FDB**), and the weighted sum is **C-FDB**. That is our formulation exactly, including the
  pixel/feature split — and note `f_l(x_{t+1}) - f_l(x_t)` is the DIFFERENCE OF EMBEDDINGS, independently
  confirming §2's choice. Two further findings worth having: they positioned FDB as the cheap FLOW-FREE
  replacement for the optical-flow-based (OFB) temporal loss and found the two roughly level in a human
  study (38.6% vs 40.7% preference); and they observed the anti-flicker property EMERGENTLY — a region
  occluded by a pillar and then reappearing was stylized consistently with no explicit long-term
  mechanism, which they attribute to *"learned resistance to the disturbance of input: if the input stays
  the same, the stylized output is also trained to stay the same."*
* **`Temporal Gradient Matching`**, in Video Depth Anything (arXiv 2501.12375) — the same idea for depth:
  the change in depth between adjacent predicted frames should match the change in the ground truth,
  explicitly without optical flow.
* **NOT this, despite the name:** `Gradient Difference Loss` (Mathieu, Couprie, LeCun, arXiv 1511.05440,
  the canonical video-prediction-loss paper) is a loss on SPATIAL image gradients for edge sharpness. It is
  about blur, not temporal coherence. Easy to conflate; they are unrelated.

**What appears genuinely untested is the SETTING, not the loss.** Every use found — stylization, depth,
restoration — applies it where the network SEES BOTH input frames and transforms them. Ours is a
free-running autoregressive rollout with no access to the truth, at the site that is simultaneously the
only autoregressive gradient into a learned dynamics model (`design/flow.md`). Whether the term behaves
the same there is open, and the interesting claim would be about what it does to the DYNAMICS rather than
to the frames. Do not describe the loss itself as novel.

**And if it underperforms, the better-targeted next step is NOT a fancier derivative.** For autoregressive
rollout drift specifically, the 2025-26 direction is CYCLE CONSISTENCY — roll forward from ground truth,
then reverse-generate back to reconstruct the initial state and penalise the error (`Cycle-World`,
arXiv 2607.11836; `LIVE`, arXiv 2602.03747, which claims forward drift can be strictly bottlenecked by the
cycle objective). That constrains the whole trajectory rather than adjacent pairs. Persistent spatial
memory is the other direction (`Persistent Robot World Models`, arXiv 2603.25685), and is the same bet as
`design/gaussian_splat_decoder.md`.

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
  `vector_l1: 1.0` in its spec and gets BOTH the reconstruction and the derivative term — provided it is
  `decode_kind: mse`, which is the eligibility rule for every head alike (§5.1).
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

### 5.0 ONE DECODER FORWARD PER HEAD, consumed by both terms

The decoded prediction currently never escapes: `decode_loss` (`modalities.py:223`) flattens `(B,F)` away
and calls `decode_head.loss(...)`, which builds `pred` internally and returns only a scalar. A second
forward is not an option -- the image decoder is **~78% of per-sample memory**.

So decode OUTSIDE the loss and let both terms consume the one tensor. **The roundtrip anchor already works
this way** (`roundtrip_losses`: `to_obs(...)` then `recon_loss(rec[n], ...)`), so the decode site is the
odd one out and this removes an inconsistency rather than adding a mechanism.

```python
# flow.py -- a pure extraction. loss()'s no_noise branch and _sample()'s no_noise branch already
#            compute this IDENTICAL tensor in two places; after this there is ONE copy and loss()
#            calls it.
def predict(self, cond, target):
    """The clean prediction the decode loss scores. no_noise ONLY."""
    if not self.no_noise:
        raise ValueError(
            "predict() requires a no_noise decoder (decode_kind='mse'). A noised parameterisation "
            "has no clean single-pass prediction: it denoises from x_tau at a tau sampled PER "
            "ELEMENT, so differencing its predictions across time measures the tau draw rather than "
            "the motion -- measured at 62% tau noise (see 5.1).")
    return self._chunked_velocity(torch.zeros_like(target),
                                  self._temb(target.new_ones(self._tau_shape(target))), cond, None)

# modalities.py -- IDENTICAL code for every modality. No per-head special cases.
def decode_loss(self, tok, target):
    lead = tok.shape[:-2]                                      # (B, F)
    flat = tok.reshape(-1, tok.shape[-2], tok.shape[-1])
    tgt  = target.reshape(-1, *target.shape[len(lead):])
    cond = self._decode_cond(flat)

    if self.derivative_weight > 0.0 and not self.decode_head.no_noise:
        raise ValueError(
            f"derivative_weight > 0 on modality {self.name!r} requires decode_kind='mse' "
            f"(it is {self.decode_kind!r}). Set model.modalities.<i>.decode_kind=mse. See 5.1.")

    if self.decode_head.no_noise:
        pred  = self.decode_head.predict(cond, tgt)             # ONE forward, and it escapes
        recon = self.recon_loss(pred, tgt, site="decode")
        deriv = (self.derivative_loss(pred.view(*lead, *tgt.shape[1:]),
                                      target.view(*lead, *tgt.shape[1:]))
                 if self.derivative_weight > 0.0 else None)     # time axis RESTORED
        return recon, None, deriv

    main, sc = self.decode_head.loss(cond, tgt, recon_loss=self.recon_loss)
    return main, sc, None                                       # noised heads: unchanged path
```

Beats threading a `lead` kwarg through `TransportHead.loss` and returning a dict: no signature change on
`recon_loss`, no Tensor-vs-dict polymorphism, and the noised path is untouched. The only churn is
`decode_loss`'s arity 2 -> 3, which has ONE call site.

For a `no_noise` head, `predict()` + `recon_loss()` is the SAME two operations in the SAME order that
`loss()` performed, so the route change is bit-identical by construction -- that is the step-2 gate.

### 5.1 THE DERIVATIVE TERM REQUIRES `decode_kind: mse`. Noised heads RAISE.

`no_noise = (decode_kind == "mse")`. Measured across every recipe (2026-09-11):

| head | `decode_kind` | `no_noise` |
|---|---|---|
| `image` / `cam_scene` / `cam_wrist` | **mse** | True |
| `proprio` (all recipes as shipped) | **flow** | False |

**Why noised heads cannot take this term.** The noised `x0` branch denoises from `x_tau` at a `tau`
sampled PER ELEMENT with fresh `eps`, so frames `t` and `t+1` are denoised from DIFFERENT noise levels and
the difference of their predictions carries the difference in estimation quality as well as the motion.
Measured on a trained checkpoint, one real window, 24 independent draws:

```
true |Delta| per step            0.229      (normalised units)
noised |Delta| mean              0.286
SPREAD across tau draws   std    0.178   <-- the contamination
deterministic |Delta|            0.231

contamination / true motion    = 0.776
contamination / noised Delta   = 0.622     <-- 62% of the difference is WHICH TAU WAS DRAWN
```

62% sampling noise is not a term worth optimising. So: **`derivative_weight > 0` on a noised head
RAISES**, with a message naming `decode_kind`. It is never silently tolerated and never worked around.

**To use it on proprio, set `decode_kind: mse` on that modality.** Verified (2026-09-11) that
`model.modalities.0.decode_kind=mse` builds, yields `no_noise=True` for proprio, and `recon_losses` runs
normally. That is a ONE-LINE recipe override and the intended way to enable the term -- NOT bit-identical,
because it changes that head's training objective from "denoise at random levels" to "predict directly",
so it is its own decision and belongs in the recipe, not hidden in the loss code.

**Explicitly rejected: a second deterministic forward for noised heads.** It would work (the deterministic
decode of a noised `x0` head is byte-for-byte `_chunked_velocity(zeros, temb(1), cond)` -- verified exactly
equal to `Modality.decode(commit=True)`, which the anchor already calls), and for proprio's tiny MLP it
would be nearly free. Rejected anyway, on the user's call (2026-09-11): it costs a second forward and it
makes proprio and image take different paths. **One forward per head, identical code for every
modality**, and the config decides whether a head is eligible.

### 5.2 The checklist

**BIT-IDENTICAL IS THE GOVERNING CONSTRAINT.** Steps 1-4 must not change a single logged number, and step
5 must not when `derivative_weight = 0`. The gate is a BYTE comparison of the loss series, not a visual
check, because 25+ historical runs and every finding in `wizard/records/*.md` are written against these
paths.

- [ ] **1. `TransportHead.predict()`** — extract from `loss()`'s `no_noise` branch and have `loss()` call
      it, so there is ONE copy. Raise for noised parameterisations (message names `decode_kind`). Must use
      `_chunked_velocity` so `decode_chunk_train` memory behaviour is unchanged.
      - [ ] verify: `predict()` is bit-identical to what `loss()` computed internally, fixed seed
      - [ ] verify: `predict()` raises on a `decode_kind=flow` head
      - [ ] verify: `smoke/multimodal.py` 10/10; `smoke/decode_recon.py` no NEW failures (1 pre-existing)
- [ ] **2. `decode_loss` decodes once, returns `(recon, shortcut, deriv)`**; `recon_losses` unpacks the
      third value and ignores `None`. `no_noise` heads take the predict route; noised heads keep the
      existing single-call path.
      - [ ] verify: **2-epoch run, fixed seed, `metrics.jsonl` loss series BYTE-identical to `main`**
- [ ] **3. `Modality.derivative_loss`** — base = MSE on the temporal difference, `ImageModality` override
      = `self.visual.temporal(...)`. New spec fields `derivative_weight` (default **0.0**) and
      `derivative_strides` (**LOCKED `[1]`**, §2.1). Emit `derivative/<m>` from `recon_losses`.
      - [ ] verify: default config -> 2-epoch run still BYTE-identical to `main`
      - [ ] verify: `derivative_weight > 0` on a `decode_kind=flow` head raises, message names `decode_kind`
      - [ ] verify: with `modalities.0.decode_kind=mse` and both weights > 0, `derivative/proprio` AND
            `derivative/image` both appear in `metrics.jsonl` — from the SAME code path, no special case
- [ ] **4. `VisualLoss._layer_features(x)`** — extract the per-layer unit-normalised VGG maps that
      `_lpips_term` computes implicitly through torchmetrics (`net.net` is `_NoTrainLpips`, `.net = Vgg16`,
      `.L = 5`, `.chns = [64,128,256,512,512]`). Refactor `_lpips_term` to use it.
      - [ ] verify: `smoke/visual_loss.py` 22/22 and `_lpips_term` bit-identical on a fixed seed
- [ ] **5. `VisualLoss.temporal(pred, target, strides=(1,))`** — `w_l1 * L1(dp, dg)` plus the
      DIFFERENCE-OF-EMBEDDINGS feature term (§2); contiguous-PAIR subsample and reshape-then-difference,
      never flat-row differencing (§6).
      - [ ] verify: identical sequences -> **exactly 0**
      - [ ] verify: one-step-shifted sequence -> **> 0**
      - [ ] verify: **constant offset `p = g + c` -> ~0** <- THE check that it measures MOTION, not POSITION
      - [ ] verify: `strides=(1,)` equals the hand-written `l1(p[:,1:]-p[:,:-1], g[:,1:]-g[:,:-1])`
      - [ ] verify: episode-seam check — a `(B=2, F=4)` input yields 3 differences per episode, never 7
- [ ] **6. `smoke/derivative_loss.py`** — all of step 5's checks, plus: the term appears for every
      eligible head; `derivative_weight=0` contributes exactly 0.0 to the total; a noised head raises.
- [ ] **7. `conf/data/block_stack.yaml`** — `isaac-ronald-ward/block-stack`, `img_size: [144, 192]` paired
      with `ae_bottleneck: 9` (144/16 = 9, 192/16 = 12 -> exact 9x12 grid at 4 levels). Probably NO
      processor needed (recorder-written, root `normalization_stats.json`).
      - [ ] verify: `check_dataset` reports the window count, no obs/action dim mismatch (17 / 5)
      - [ ] verify: frames load, `frames == steps` per split, normalizer round-trips
- [ ] **8. `conf/model/<recipe>_blockstack.yaml`** with `decode_kind: mse` on BOTH modalities, so both
      heads are eligible for the term from the start.
      - [ ] verify: `no_noise=True` on every head; a 2-epoch run trains and logs `decode/proprio`
- [ ] **9. BASELINE run with `derivative_weight = 0`**, reading `motion_ratio` / `latent_motion_ratio`
      across horizon BEFORE the arm.
      - [ ] verify: far below 1.0 -> under-moving, this term targets it. Near 1.0 while frames still
            flicker -> the motion is INCOHERENT not INSUFFICIENT, and patch-correspondence (§8) is the
            better fit. **Either reading is worth having before spending a second GPU.**
- [ ] **10. The arm.** Sweep `derivative_weight` upward from small.
      - [ ] verify: `@+128` against the step-9 baseline at matched epochs (the objective)
      - [ ] verify: **`motion_ratio` is the TRIPWIRE** — if it falls below the `d=0` baseline the term is
            over-weighted and is freezing the prediction (§7)
- [ ] **11. Record it** as a new numbered section in `wizard/records/block-stack.md` (a new file; the
      convention is one record per dataset), whichever way it goes.

### 5.3 Deferred, deliberately out of this build

* **`VectorLoss` + the composition refactor (§3.1).** Drops OFF the critical path under 5.0 — the only
  polymorphism needed is `derivative_loss`, so the symmetry cleanup is an independent later change with
  its own bit-identical gate. Doing it first would put the riskiest edit in front of the experiment.
* **Multi-stride, higher order, cycle consistency, optical-flow warping** — §8.

## 6. Gotchas, all of them real

**`_flatten` destroys the time axis, at BOTH sites.** Confirmed live: the anchor passes
`(B,F,H,W,C)` and `_flatten` collapses it to `(B*F,H,W,C)`; the AR decode site arrives ALREADY flat because
`decode_loss` does `tok.reshape(-1, ...)`. So the loss currently sees `B*F` rows with no idea which are
adjacent in time. `temporal()` needs the unflattened shape — this is the one real piece of surgery.

**`visual_frames=128` INVALIDATES the term.** `_subsample` is `torch.randperm(M)[:n]` — a random
permutation of the flattened rows. Pick 128 of `B*F = 1344` and you get `(b=3,t=17)`, `(b=0,t=52)`,
`(b=14,t=6)`...; differencing consecutive entries computes `frame(b=3,t=17) - frame(b=0,t=52)`, two frames
from DIFFERENT EPISODES at unrelated times. Not a derivative — noise.

Fix: **sample contiguous PAIRS `(t, t+1)`.** 64 pairs = 128 frames, exactly the existing budget, same
VGG cost as today, adjacency preserved. With `derivative_strides` LOCKED at `[1]` (§2.1) pairs are all
that is needed — an earlier draft of this document proposed contiguous SEGMENTS so that larger strides
would be available for free, which is true but pointless once the stride is locked. Prefer pairs: they are
simpler and they spread across more episodes for the same frame budget. (A pixel-only version can just use
all frames, since a subtraction and an L1 are free; the feature part cannot.)

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
* **Multi-stride (`Delta_k`, k > 1)** — REJECTED on two independent grounds (§2.1): arXiv 2102.05822 §4.5
  tested it and found results "almost identical" across K, and our per-frame term already catches the
  drift that large K would add. `derivative_strides` is LOCKED at `[1]`.
* **Contiguous SEGMENTS as the subsample** — an intermediate draft, motivated by making every stride free.
  Pointless once the stride is locked; contiguous PAIRS are simpler and spread over more episodes (§6).
* **Higher ORDER differences** (`Delta^2`, acceleration-like) — a different axis from stride, deferred
  until there is a measured reason; first order is what flicker calls for (§2.1).
* **Optical-flow warping loss** — the established STRONGER temporal loss, which FDB was introduced to
  replace cheaply (roughly level in a human study, 38.6% vs 40.7%). Deferred because it needs a flow
  estimator in the training loop; revisit if the derivative term underdelivers.
* **Cycle consistency** (`Cycle-World` arXiv 2607.11836, `LIVE` arXiv 2602.03747) — roll forward from
  ground truth, reverse-generate back, penalise the reconstruction of the initial state. Constrains the
  WHOLE trajectory rather than adjacent pairs, and is the better-targeted method for autoregressive drift
  specifically. **This is the next step if the derivative term underperforms**, not a fancier derivative.
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
all. Proprio is least likely to matter, and it now costs a deliberate config choice
(`decode_kind: mse`, §5.1) rather than coming for free, so enable it on purpose or not at all.

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
