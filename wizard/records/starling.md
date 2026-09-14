# starling / starling-2 — recorded egocentric drone flight

Two HF datasets written by quickdraw's own recorder: `isaac-ronald-ward/starling` (124 train eps x 1726
steps, 1.98 h) and `isaac-ronald-ward/starling-2` (60 x 3539, 1.97 h, plus a 49-episode `eval` split).
30 Hz, `observation_vector` 16-dim, `action` 4-dim, one camera `ego` at **112x192 (NOT square, 12:7)**.
Configs: `conf/data/starling.yaml`, `starling2.yaml`, `starling_ctx.yaml`. Recipes: `conf/model/
vl128_starling.yaml` and its arms. Frame caches pre-built by `quickdraw._build_starling_cache`.

**THE OBJECTIVE, as everywhere in this project: raw `eval_ood_horizon/open_loop/image/lpips/@+128`.**
Nothing below counts as a result unless it moves that.

---

## 1. THE FAILURE IS POSITIONAL, AND IT IS THE DYNAMICS, NOT THE CODEC (09-05/07)

### 1.1 The recipe transfers, and the bottleneck moved

`vl128` carried over with only what the data forces (obs 16, act 4, `img_size=[112,192]` paired with
`ae_bottleneck=7`, `decode_out_act=sigmoid`), `subsample=5`. Both runs improved, then plateaued:

| | best @+128 | best floor | floor/OL ratio |
|---|---|---|---|
| `starling_vl128` | **0.2754** (ep7) | **0.0283** (ep14) | **8.1x** |
| `starling2_vl128` | 0.3505 (ep11) | 0.0424 (ep14) | 7.2x |
| robocasa `twocam_full` (reference) | 0.1161 | 0.0580 | **2.0x** |

The codec is the best this project has produced -- 0.0283 against robocasa's best-ever 0.0565 -- and the
open-loop score is 2.4x WORSE. On robocasa the rollout costs 2x the floor; here it costs 8x.

### 1.2 It is not long-horizon drift. 80% of the error is spent by step 8

Horizon curve, `starling_vl128` ep10:

| horizon | +1 | +8 | +32 | +128 |
|---|---|---|---|---|
| open-loop | 0.1229 | 0.2691 | 0.2994 | 0.2900 |
| AE floor | 0.0573 | 0.0670 | 0.0486 | 0.0358 |
| **dynamics share** | 0.066 | **0.202** | 0.251 | 0.254 |

120 further steps after +8 add 0.05. Two more measurements pin it to ONE step: closed-loop-1 (re-ground
EVERY step) still scores 0.148 against a 0.036 floor, so a single prediction costs 4x the codec; and
`latent_motion_ratio@+8` is 0.50 with `latent_cos@+8` 0.265 -- half the true motion in a barely
correlated direction, the signature of HEDGING.

### 1.3 THE DECOMPOSITION THAT SETTLES IT

Sharpness recovery = fraction of pixels with |grad| > 0.08, predicted / ground truth. Measured on the
runs' own logged filmstrip frames:

| | AE floor (codec only) | open-loop (codec + dynamics) |
|---|---|---|
| starling depth2 ep5 | **0.855** | **0.174** |
| starling depth4 ep5 | **0.882** | 0.174 |
| robocasa twocam ep29 | 0.890 | **0.856** |

**The codec renders sharp structure as well as robocasa's. The rollout throws 80% of it away, where
robocasa's keeps 96%.** And ground-truth sharpness is the SAME in both (18.78% vs 20.29% strong edges),
so it is not the scene either. The one distinguishing fact: robocasa's camera is FIXED, so most pixels
are unchanged step to step and sharp structure survives by being carried forward; here the camera IS the
drone, so every thin line must be re-localised at a new position every step.

**What the user called "ghosting" -- repeated copies of thin filaments -- is therefore a SYMPTOM of the
short-horizon positional failure, not an artifact in its own right.** Edge statistics show a deficit in
BOTH bands vs truth (weak -10pp, strong -15.5pp), i.e. strong structure DISSOLVING, not phantom
structure appearing. A hedging decoder that does not know where a light strip is renders a faint smear.

### 1.4 The sigmoid was free insurance here, and we now know it was unnecessary

`decode_out_act=sigmoid` was enabled because it was worth 3.4x on torus. Measured before launch: 0.26% /
0.09% of starling target pixels sit at exactly 1.0 -- robocasa-like (0.4%), not torus-like (66.9%). The
range diagnostics confirmed it across 22 epochs: `frac_hi` 0.0000-0.0001, `frac_lo` 0.0000-0.0036, `max`
pinned at 1.0000, zero alarms. So the torus ratchet is SATURATION-SPECIFIC, not general. Keep the flag on
(a bounded output cannot be wrong) but do not expect it to do anything on natural imagery.

### 1.5 `flow_arch_depth: 4` — NOT WINNING on the objective (`*_heavy`, 09-07)

The first arm, aimed at 1.2's hedging signature. It does move the latent tracking decisively -- 
`latent_cos@+8` better at EVERY matched epoch on both sets (+24% to +37% on starling; 2.3x at ep0 on
starling-2) -- and it does not convert:

| @+128, epochs better out of matched | starling | starling-2 |
|---|---|---|
| depth4 vs depth2 | **2/7** | **2/6** |

At @+8 on starling-2 it wins 5/6, so the gain is real and short-horizon only. The lesson is that
`latent_cos` is NOT a proxy for the objective on this dataset -- it moved 30% and @+128 did not follow.
Do not rank arms on it.

### 1.6 THE FRAME STRIDE WAS WRONG — copied from robocasa, never measured (09-08)

> **CORRECTION (09-10): every number in this section is diluted.** It was measured on the 30 Hz
> build, which was resampled UP from a ~17 Hz camera and so carried **44% duplicate consecutive
> frames**. A duplicate contributes a per-step delta of exactly zero, so every RMSE below is pulled
> toward zero by roughly that fraction, and the stride recommendation drawn from it is wrong. On the
> rebuilt 15 Hz data (1.9% duplicates) the same physical step durations measure: stride 1 = 0.104
> (67 ms), stride 2 = 0.138 (133 ms), stride 3 = 0.159 (200 ms), stride 4 = 0.174 (267 ms). The
> argument of this section SURVIVES -- stride 1 at 0.104 is within 4% of robocasa's working 0.0997,
> so no striding is needed -- but the winning stride does not: what §1.7.3 targets is stride 3, the
> whole stride nearest the 167 ms every historical flight number was actually trained at.

`subsample=5` was chosen for TRACTABILITY (epoch parity with robocasa: 34k windows vs 205k at stride 1),
and `conf/data/starling.yaml`'s header said in as many words that §13's SNR argument did not transfer and
the delta "has to be measured here before a stride is chosen". It never was. Measured now, per-step image
RMSE on val frames:

| | stride 1 | stride 2 | stride 3 | stride 5 |
|---|---|---|---|---|
| **starling** (30 Hz) | 0.0751 | **0.1053** | 0.1248 | **0.1502** <- what we ran |
| **robocasa** (20 Hz) | 0.0426 | — | — | **0.0997** <- what the recipe was tuned on |

§13 raised robocasa to stride 5 to lift its per-step signal from 0.0389 -- 0.61x its own codec floor,
where predicting zero motion was the correct answer -- ABOVE that floor. **Starling at stride 1 is already
0.0751, nearly 2x robocasa's stride 1 and already in the regime §13 was reaching for.** Striding 5 on top
made every step a 0.1502 jump, 1.5x harder than where this recipe works. That is a plausible first-order
cause of 1.2's finding that 80% of the dynamics error is spent by step 8, and of 1.3's positional failure:
we asked the transition head to predict a much larger physical change than it was ever tuned for.

**stride 2 (0.1053) lands almost exactly on robocasa's working difficulty (0.0997).**

READING IT REQUIRES CARE: the stride changes what a horizon MEANS. At stride 5 `@+128` spans 21.3 s; at
stride 2 it spans 8.5 s. Compare at matched PHYSICAL duration -- stride-5 `@+51` against stride-2 `@+128`
-- never at matched step count. Cost ~2.5x the epoch.

### 1.7 Queue

1. ~~**`visual_l1: 10.0` on the DEPTH-2 base**~~ (`model=vl128_starling_l1x10`) -- **RUN, TIED, CLOSED.**
   11 evals on the OLD 30 Hz build, best @+128 **0.2761 at ep9** against the base's 0.2754, and the AE
   floor did NOT degrade. That is the refutation condition stated below firing in the inert direction:
   the pixel/perceptual balance is flat across 3.0->10.0, so the mix is not the lever. Original argument
   kept verbatim below because it was a good argument that lost to a measurement.
   THE loss arm, and the argument is 1.3. `VisualLoss` trains two sites and the AR decode loss is, per
   design/flow.md, the ONLY autoregressive gradient in the model -- so the mix is a lever on the DYNAMICS,
   not just the codec. vl64.yaml's header already names this failure: "a loss blind to spatial
   displacement tells the dynamics that LANDING IN THE WRONG PLACE IS CHEAP", and L1 is the only term
   that prices displacement. The mix has NEVER been varied on this dataset -- all four runs are
   l1=3.0/lpips=1.0, inherited from vl128 where 3.0 was solved for ROBOCASA's codec. Measured on the
   STARLING codec (natural ratio 3.97:1):
       l1= 3.0 -> 43% pixel / 57% perceptual   <- every run so far
       l1= 5.0 -> 56 / 44
       l1=10.0 -> 72 / 28                      <- the arm
   Spending codec quality we are not using (0.88 of ceiling) to buy positional accuracy we badly need
   (0.174). REFUTED IF the floor degrades and @+128 does not improve.
2. **`data=starling_ctx`** (`P: 8 -> 24`, 1.33 s -> 4.0 s of history; written, not run). Directly targets
   positional lock: more visual history is more egomotion evidence. torus.yaml argues history is
   unnecessary because obs carries velocity -- true of the PROPRIO state, false of the VISUAL one, since
   what appears next depends on geometry currently out of frame. Better motivated after 1.3 than when it
   was first ranked third.
3. **`data.subsample=3` on the rebuilt 15 Hz build** (`experiment=s2_sub3`) -- **QUEUED 09-10, waiting on
   a GPU.** Nothing is pre-empted for it: a host-side watcher polls both accelerators every 120 s and
   launches on the first one that has sat under 20 GB for three consecutive polls, i.e. only once
   `s2_sub1` or `s2_sub4` has finished by itself. Log: `scratch/queue_s2_sub3.log`.
   WHY 3, having argued for 2. §1.6 was measured on the 30 Hz build, where 44% of consecutive frames
   were duplicates; the per-step deltas in that table are diluted and the stride it recommends is
   therefore wrong. On the rebuilt 15 Hz data the OLD leaderboard's own step duration (167 ms, from
   subsample 5 at 30 Hz) is **stride 2.5**, which is not an integer. Stride 3 (200 ms, per-step image
   RMSE 0.159, @+128 spanning 25.6 s) is the whole stride nearest it, and it brackets from the HARDER
   side -- which is where the matched-duration comparison between the two running arms already points:
   stride 4 beats stride 1 at every physical duration past ~2 s (4.27 s: 0.2580 vs 0.3093; 8.53 s:
   0.3298 vs 0.3533). It is also the only stride whose 128-step score can be quoted against the
   historical 21.3 s numbers without a duration caveat.
   ~~`starling_stride2`~~ was launched 09-08 and STOPPED at ep3 (3 evals, best 0.3473) to free its
   accelerator for the two current arms. Stride 2 is unmeasured, not refuted.
   REFUTED IF the three strides are non-monotone in @+128 at matched physical duration -- that would
   mean per-step difficulty is not the axis and §1.6's whole framing is wrong.
4. `flow_arch_heads: 4 -> 8` / `encode_base: 32 -> 48`. Capacity on axes never varied. DEPRIORITISED --
   1.5 is evidence that transition-head capacity is not the binding constraint.

**NOT queued, deliberately: `num_tokens`.** Proposed down (32->16) on an unsupported argument, withdrawn,
then listed up (32->64) without flagging the reversal. §21.2 of the robocasa record has 64 open and
leading at matched eval, but there is no starling evidence in either direction. Leave it alone.

---

## 2. THE ACTION PRIOR: GROUPED CONTEXT + PERCENTILE TARGET, CHUNK 8, 16 STEPS (09-13/14)

**THE SETTLED RECIPE, if you read nothing else: `action_head.context: grouped` (LOCKED),
`target_transform: pit`, `chunk: 8`, `sampling_steps: 16`, `data.action_aggregate: concat`, post-hoc on a
frozen WM. 2.7 has the defaults and their consequences; 2.6 has the experiment that decided it.**

**AND A WARNING ABOUT EVERY NUMBER IN 2.1 AND 2.2:** they are W1 against the recorded actions, and 2.5 shows
a model that ignores its context entirely scores 0.0006 on that metric while the trained heads score
0.063-0.099. Those two sections are statements about MARGINAL MATCHING. They are kept because the reasoning
was sound given what was measurable at the time, and because the sampling-step curve in 2.1 is still a valid
statement about the marginal -- but do not quote them as conditional skill, and do not re-derive the chunk
length from 2.2. The conditional metrics start at 2.5.

Two post-hoc arms on the frozen `s2_sub4_concat` world model (`ah_base_ep38.ckpt`), 60 epochs each,
autobatch (420 / 419), `data.subsample=4`, `action_aggregate=concat` so one post-subsample action is
16 wide. Launchers: `scratch/launch/s2_ah_chunk8.sh`, `s2_ah_chunk16.sh`. Only the prior trains.

### 2.1 The sampling-step default was wrong in BOTH directions

`sampling_steps` was borrowed from the dynamics flow (6) and then set to 64 by hand for these two runs.
Neither number was measured on the prior. Swept on the FINAL checkpoints with the head frozen and the
SAME noise draw at every step count, so the integrator is the only thing that varies
(`scratch/sweep_action_steps.py`, W1 against the recorded actions, mean over lead times):

    chunk8    steps    2      4      6      8     12     16     24     32     48     64     96    128
              W1    .3695  .2202  .1517  .1095  .0941  .0731  .0656  .0632  .0633  .0649  .0662  .0674
              ms/1k   0.6    0.3    0.5    0.6    0.8    1.0    1.5    2.0    2.9    3.9    5.8    7.7

    chunk16   W1    .4921  .3201  .2354  .1884  .1536  .1278  .1056  .0919  .0802  .0746  .0685  .0658

**32 is the default now** (`conf/model/mm_flow.yaml`). The chunk8 curve TURNS AROUND after 48 -- 64, the
value it trained with, is worse than 32 -- so past the knee you are paying time for a worse answer, not
trading speed against quality. 24 is within 5% at 1.5 ms/1k if MPPI's sampling budget binds.

The turnaround is a FAR-LEAD effect and it is the one genuinely surprising result here: leads 0-2 improve
monotonically all the way to 128, while leads 6-7 bottom out at **16** steps (.0721/.0795) and degrade
~20% by 128 (.0884/.0898). Finer integration sharpens the far-lead marginals into a worse fit. Unexplained.

`p99 spread` (head's 99th pct |a| / recorded) sits at 1.06-1.13 near both knees, i.e. the prior now
reaches PAST the recorded extremes. The old "not sharp" complaint was under-integration, not timidity --
at 6 steps the same head scored .1517 against .0632, 2.4x its true error.

### 2.2 Chunk 8, not 16 -- and the comparison is dirty in 16's favour

Final quality is a wash (best W1 **.0632** at K=8 vs .0658 at K=16), but K=8 gets there with a 128-wide
head and a quarter of the sampling steps; K=16 was still improving at 128 steps and never got ahead.
4.27 s of joint prediction buys nothing over 2.13 s on this data.

CAVEAT, recorded because the launcher's own rationale states it wrongly: the two arms differ in TWO
things. `s2_ah_chunk16.sh:18` sets `action_head.hidden=512` to keep the 256-number target off a 128-wide
bottleneck, and its RATIONALE claims this matches the arms on capacity per output. It does not -- K=8 is
128 numbers through 128 (ratio 1.0), K=16 is 256 through 512 (ratio 2.0). The longer arm got DOUBLE the
capacity per output and still lost, so the conclusion survives the flaw, but "K=16 needs ~4x the steps"
is entangled with "the 512-wide field is harder to integrate" and is NOT established. A K=16 arm at
width 128 would settle it; not queued, because the quality result does not depend on the answer.

### 2.3 THE PRIOR COULD NOT REPRESENT ITS OWN TARGET (09-13)

The finished chunk-8 head matches the recorded action distribution's SHAPE badly in one specific way, and
the reason is structural rather than a tuning failure. Measured on the recorded train actions:

    axis | unique values | mass at EXACTLY 0.0
     a0  |     1811      |   37.8%          the stick at rest is an ATOM, not a narrow peak
     a1  |     1748      |   23.4%
     a2  |     1726      |   44.5%
     a3  |     1030      |   41.6%

A rectified flow integrates a velocity field with finite Lipschitz constant, so noise -> action is a
DIFFEOMORPHISM and its terminal law is absolutely continuous: it cannot place an atom, only a bump. Putting
38% of the mass on one point needs the Jacobian determinant to vanish, i.e. |dv/dx| -> infinity, and the
flow-matching MSE (a conditional-MEAN regression) has no term that pays for that singularity -- the loss is
smooth in the bump's width and the gradient for narrowing it vanishes as it narrows.

Measured on the finished head at 32 steps (`scratch/check_action_atom.py`), mass within +-tol of zero:

    a0   true 43.9% within +-0.05, head 9.0%    | needs +-0.204 to hold the atom's mass
    a3   true 58.2% within +-0.05, head 11.5%   | NEVER reaches it, even at +-0.5

a2's case is the benign one the theory predicts (right mass, +-0.09 bump standing in for a spike). a3 is
NOT: the head cannot find that mass anywhere, which is a conditional failure on top of the representational
one. Rough arithmetic: a Dirac-vs-bump costs W1 ~ mass x displacement ~ 0.38 x 0.1 = 0.04 against a total
W1 of 0.063, so most of what we had been optimising was the unrepresentable atom.

### 2.4 THE FIX: A PERCENTILE TARGET (PIT), AND THE MACHINERY IT NEEDED

    z = Phi^-1(F(a))     train the flow on z     a = F^-1(Phi(z))      F = empirical CDF, TRAIN split only

Percentiles are uniform by construction, so EVERY sharp marginal feature is flattened in proportion to its
mass, with no assumption about where the sharp parts are -- it reads them off the data. A zero-inflated gate
was considered and rejected: it encodes WHERE one peak is, and the peaks are not only at rest. Measured
z-width per feature (the room the flow gets), which tracks MASS and not position:

    a3  rest atom  +0.002, 41.7% of mass -> 3.009 z-units      a0 saturation +0.893, 0.8% -> 0.810
    a3  peak at -0.967,     8.9% of mass -> 0.780 z-units      a typical 0.32% bin      -> 0.008

An atom becomes a SLAB: the flow only has to get the slab's total mass right, and every z inside it inverts
to the atom's exact value. Round trip on the recorded actions: max 7.4e-5 (0.004% of the action range),
mean 1.6e-8, every atom exact.

MACHINERY, all defaulting to prior behaviour and asserted bit-identical (`smoke/transforms.py`, 38 checks):
  * `data/transforms.py` -- ONE `Transform` protocol (fit/apply/invert/state_dict) with `ZScore`, `Symlog`,
    `PIT`, `Compose`. `Normalizer` now composes `ZScore` instead of inlining it (92 call sites untouched,
    bitwise equal); `symlog` moved out of `models/features.py` after a bitwise `FourierMLP` forward check.
  * knots fitted in `compute_norm_stats` (1024 per raw axis, ~97 KB in normalization_stats.json);
    `quickdraw.data.backfill_pit` adds them to datasets built earlier, including HF cache snapshots (where the
    stats file is a symlink into a content-addressed blob and must be REPLACED, not written through).
  * `model.action_head.target_transform: none|pit`, applied to the target in `action_pairs` and inverted in
    `sample_action`, so the head's boundary, the WM's conditioning and every downstream metric are unchanged.
  * hard errors on: a checkpoint whose transform disagrees, `action_aggregate=sum`, and
    `sample_action(deterministic=True)` (inverting the mean of z gives the MEDIAN action, not the mean).

TWO BUGS, both of which produce plausible-looking wrong answers and cost a run between them:
  1. Knots fitted on RAW actions but `action_pairs` transforms the NORMALIZED tensor -- 38.6% of the target
     pinned at the clamp, z std 3.34. The knots must be z-scored into the space the target lives in.
  2. Nearest-knot inversion lost 18% of the action range on one axis's thin upper tail. The quantile
     function is piecewise LINEAR now, which is exact everywhere AND still returns atoms exactly (an atom is
     a flat run of identical knots, and interpolating between two identical values is that value).

RESULT, on a partially trained (ep9) checkpoint: mass at exactly 0.0 -- a0 35.7% vs recorded 37.6%, a2 37.4%
vs 34.1%, a3 44.1% vs 58.1%, against 0.0% for every axis before. The sampling knee also moved 32 -> 12 steps
(the field no longer has to approximate a singularity) and p99 spread went 1.12 -> 0.99.

### 2.5 W1 WAS NEVER MEASURING CONDITIONAL SKILL. IT IS 0.0006 FOR A MODEL THAT IGNORES ITS CONTEXT

`eval_action_distribution`'s W1 compares POOLED histograms. Shuffling the recorded val chunks ACROSS contexts
leaves the pooled histogram untouched, so a model that ignores its context entirely scores:

    w1_blind_null 0.0006      vs the trained heads' 0.063 - 0.093

The control beats every model by 30x or more. This is not a PIT artefact -- the hole was always there, and
PIT merely makes it easier to fall into by handing the head the right marginal shape. CONSEQUENCE: the
chunk8-over-chunk16 call in 2.2 and the step-count knee in 2.1 are statements about MARGINAL MATCHING, not
about conditional skill, and should not be quoted as the latter.

REPLACED BY, in `evaluation/conditional.py`, logged every eval under `eval_action_distribution/`
(`energy_skill_vs_blind`, `lead_XX/energy_skill`, `rank_calibration_dev`, `rest_auc/dim_i`, `w1_blind_null`),
~1 s on top of a ~700 s eval, validated against known answers in `smoke/conditional.py`:

  * ENERGY SKILL   1 - ES_model/ES_null with ES = E||X-y|| - 0.5 E||X-X'||, a strictly proper rule needing
                   only ONE observation per context. 0 = context-blind. Interpretation: skill is the
                   fractional reduction in the SCALE of the uncertainty, so R^2 ~ 1 - (1-skill)^2 -- verified
                   on synthetic data where the context explains 94.1% of the variance and skill measured
                   +0.758 (1 - (1-0.758)^2 = 0.941).
  * RANK CALIB.    where the observation falls among the draws; uniform = calibrated, piled at the ends =
                   overconfident, in the middle = overdispersed. The only one that says WHICH WAY it is wrong.
  * REST SKILL     AUC/Brier of P(at rest | context). The smoke case that matters: a model with the rest RATE
                   exactly right (40% vs 40%) scores AUC 0.494 -- knowing HOW OFTEN is not knowing WHEN.

### 2.6 THE BINDING CONSTRAINT WAS THE CONTEXT POOLING, NOT THE OBJECTIVE (09-13)

`quickdraw.evaluation.action_context_ceiling`: ridge from the FROZEN `h_ctx` to the recorded chunk, fit on 20 TRAIN flights and
tested on all 7 VAL flights (no flight in both; an episode-level split of val alone fails on the per-flight
mean shift and reports -0.30 everywhere, which measures the shift and nothing else).

    context                     lead 0 R^2    leads >= 8 mean R^2
    pooled  (128, the default)     0.2695           0.1116
    grouped (384)                  0.7940           0.3338

`action_context` averaged ALL 34 tokens (32 image + 1 proprio + 1 action) into one d=128 vector, so the
vehicle state was 2.9% of what the prior reads AND SO WAS THE ACTION TOKEN -- the command currently being
held, which is the most informative single feature about the command coming next. `grouped` averages within
each modality and keeps the two single tokens whole: [mean(image), proprio, action] = (len(layout)+1)*d.
`model.action_head.context: pooled|grouped`, one `pool_context` used by BOTH call sites (loss_terms and
action_context, which had the mean written out twice).

Per-axis at lead 0 under grouped: a0 0.916, a1 0.908, a2 0.799, a3 0.927 -- all four, where pooled gave
0.366/0.458/0.082/0.441. At lead 31 (8.5 s) grouped still holds R^2 0.277, with a3 at 0.611.

THE HEAD ITSELF WAS NOT THE PROBLEM: its conditional MEAN reaches R^2 0.209 against the pooled probe's
0.270, i.e. it recovers ~78% of what a linear readout finds in the same vector. The averaging was the loss.

CONFIRMED IN TRAINING at epoch 5 of 60, grouped + PIT, against pooled's best over 34 epochs:

    lead 0 energy skill   +0.552 (chunk 8)   +0.475 (chunk 32)     vs +0.110 ... +0.130 pooled
    lead 16 energy skill                     +0.258 (chunk 32)     vs  ~0.000        pooled

Lead 0 is at the linear ceiling (+0.546 implied) and lead 4 (+0.323) is ABOVE it (+0.223 implied), i.e. the
head is finding nonlinear structure. The long-horizon claim in an earlier draft of this record -- that skill
dies past lead 8 -- was an artefact of the pooled context and is WITHDRAWN.

### 2.7 RUNNING / QUEUED

  * `s2_ah_pit` (GPU 0) and `s2_ah_pit_chunk32` (GPU 1), both PIT + grouped, 60 epochs, batch 421/422,
    launched 09-13 23:43. chunk32 carries hidden=512 to hold capacity-per-output at 1.0, the ratio chunk8
    sits at by accident and the ratio the old chunk16 arm (2.0) did not.
  * THE 2x2 IS COMPLETE (09-14), all four arms read at epoch 29, same data, same frozen WM, same batch,
    chunk 8, only the two flags differing. Conditional metrics, not W1:

        config              lead-0   cond-mean   rest AUC d0/d1/d2/d3      calibration     pooled W1
                            skill    R^2 lead0
        no-PIT + pooled     +0.168     0.247    0.50 0.50 0.50 0.50    overdisp   0.034     0.0994
        PIT    + pooled     +0.115     0.187    0.52 0.52 0.61 0.84    calibrated 0.022     0.0796
        no-PIT + grouped    +0.654     0.872    0.50 0.50 0.50 0.50    overdisp   0.063     0.0955
        PIT    + grouped    +0.653     0.868    0.87 0.82 0.95 1.00    overconf   0.117     0.0200

    THE CONTEXT IS THE FIX: 4x either way (0.168 -> 0.654 without PIT, 0.115 -> 0.653 with), consistent
    across both rows. `action_head.context: grouped` should become the default once a WM run confirms it.

    THE TWO FACTORS INTERACT, which is why the square was worth completing instead of testing one factor at
    a time: PIT costs 0.053 skill on the POOLED context and 0.001 on the GROUPED one -- free once the
    bottleneck is gone -- while buying atom recovery in both. A one-at-a-time study run on the pooled context
    would have concluded PIT was harmful and dropped it, and with it the only mechanism that lets the prior
    emit the value the pilot holds 38-58% of the time (rest AUC 0.999 vs 0.500, Brier skill +0.935 vs -1.389).

    RECIPE: grouped + PIT. It is also the only arm where the marginal AND the conditional are right (pooled
    W1 0.0200 against 0.0796-0.0994) -- though note the null still scores 0.0006, so W1 remains a floor test
    and not a discriminator.

    THE OPEN BLEMISH IS CALIBRATION, and it is the only thing left that is clearly wrong. The two grouped
    arms fail in OPPOSITE directions -- PIT overconfident (deviation 0.117, and rising monotonically over
    training: 0.119/0.106/0.136/0.199/0.216 at ep5/9/15/19/29 while skill plateaued after ep19), no-PIT
    overdispersed (0.063). Skill is done improving; calibration is not done getting worse. THIS is the first
    honest case for the energy-score term removed below, since overconfidence is exactly what a strictly
    proper scoring rule taxes and flow-matching MSE does not.

    The head's conditional mean at lead 0 (0.868-0.872) is ABOVE the grouped ridge probe's ceiling (0.794),
    so the flow finds nonlinear structure a linear readout cannot, and "the objective is under-extracting"
    is refuted for good.
  * ENERGY-SCORE TRAINING: BUILT, CHECKED, REMOVED UNUSED (09-14). `action_energy_loss` plus
    flow_weight/energy_weight/energy_draws/energy_steps existed for a few hours and are gone. It was built on
    the hypothesis that the head was under-extracting relative to a linear probe; the grouped-context result
    refuted that (the head matches the linear ceiling at lead 0 and BEATS it at leads 7 and 31), so the
    objective was never the bottleneck. Removed rather than left dormant because it was numerically verified
    but never TRAINED with, and untrained-with code in a loss path is what breaks silently later.
    THE DESIGN, if a head ever does stall below the probe ceiling: ES = E||X-y|| - 0.5 E||X-X'|| as a second
    weighted entry in the loss dict (the mechanism already supports named terms with weights, so flow-only,
    energy-only and any blend are config, not a code path). Unbiased for draws >= 2 -- the pair term is a
    U-statistic over ordered pairs -- measured ES mean 1.775/1.798/1.779 at draws 2/4/16 with sd
    0.103/0.081/0.024, so draws is a variance knob only; 16 is the sweet spot since the real cost is the
    autograd graph over draws x steps sampler forwards, which trades against batch size via autobatch.
    Use `torch.cdist`, not an explicit difference: the (..., M, M, D) intermediate is 3.5 GB at our batch.
    TWO CAVEATS that would have to be handled: the energy score loses discriminating power in high dimensions
    and is insensitive to misspecified DEPENDENCE (our target is 128 numbers at chunk 8, 512 at chunk 32), so
    per-lead terms or the variogram score come first; and under PIT it MUST be scored in z-space, because the
    inverse is flat inside an atom's slab and an action-space score has exactly zero gradient for moving mass
    into or out of the rest position.
  * SAMPLING STEPS RE-SWEPT on the settled recipe (grouped + PIT, ep29), and the two metrics disagree in a
    way worth keeping: STEPS BUY MARGINAL FIDELITY, NOT CONDITIONAL SKILL.
        steps      6      8     12     16     24     32     48     64     96    128
        W1     .0921  .0843  .0787  .0774  .0764  .0761  .0758  .0757  .0756  .0756
        skill  +.408         +.405  +.404         +.403         +.403
        ms/1k    0.51   0.63   0.87   1.13   1.59   2.09   3.04   3.98   5.92   7.87
    Energy skill is FLAT from 6 to 64 and fractionally BETTER at 6; coarse integration distorts the SHAPE of
    the distribution without moving where it is centred or how wide it is per context. The count is therefore
    chosen on W1 -- the one job W1 is still good for -- and the default is now 16 (within 2.4% of the
    asymptote at half the cost of 32). The curve is MONOTONE here, unlike the pre-PIT field which turned
    around after 48 while the flow approximated an unreachable singularity.
  * DEFAULTS CHANGED 2026-09-14, all three in conf/model/mm_flow.yaml:
      `action_head.context: grouped`      LOCKED (user). The 2x2 above; 4x conditional skill either way.
      `action_head.target_transform: pit` The prior cannot emit an atom without it, and 23-58% of every
                                          recorded axis IS an atom. Free on the grouped context.
      `action_head.sampling_steps: 16`    Steps buy marginal fidelity, not conditional skill; 16 is within
                                          2.4% of the asymptote at half the cost of 32.
    CONSEQUENCE OF THE PIT DEFAULT, accepted deliberately: an action-head run on a dataset without fitted
    knots, or under `action_aggregate=sum`, now RAISES rather than silently training a head that cannot
    represent its target. Both errors name the fix (`python -m quickdraw.data.backfill_pit <root>`, or set concat, or set
    target_transform=none). torus and robocasa need the backfill before their next action-head run.

    WHY PIT REQUIRES concat, which is two separate reasons and only the second is about PIT:
      1. SEMANTIC, true regardless: starling actions are ABSOLUTE stick positions and summing `subsample` of
         them is meaningless -- it scales the std by ~s against stats computed on raw commands (4.21 z-std at
         stride 4, data/dataset.py). concat is the right aggregation for this data on its own merits.
      2. MECHANICAL: the knots are fitted ONCE at dataset build time, but aggregation happens at LOAD time
         and depends on `data.subsample`, a training-time knob the dataset cannot know. Under concat every
         slot holds the RAW action distribution whatever the stride, so tiled raw knots are exactly right;
         under sum the stored action is a sum whose distribution changes with the stride, so no pre-fitted
         table describes it. Fitting knots post-aggregation would lift this restriction entirely.

    WHAT A KNOT IS, since the word is now in three configs: `action_pit` in normalization_stats.json holds
    1024 numbers per RAW action axis -- the action value at the 0th, 1/1023rd, ... 100th percentile of the
    training data. Those points ARE the empirical CDF, stored as a lookup table; between them the quantile
    function is read piecewise-linearly. An atom is a long RUN of identical knots, which is exactly why
    interpolating inside one returns the atom's value bit-for-bit.
  * OLD, superseded: re-sweep `sampling_steps` on the grouped heads -- 32 was measured on the pre-PIT field and 12 on the
    pooled PIT field; neither describes this one.
  * `data=starling_ctx` (P 8 -> 24) is better motivated now than in 1.7, but it should be re-argued against
    the grouped ceiling rather than the pooled one.
