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

1. **`visual_l1: 10.0` on the DEPTH-2 base** (`model=vl128_starling_l1x10`, written 09-07, NOT yet run).
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
3. ~~`subsample: 5 -> 3`~~ SUPERSEDED by 1.6 -- measured, and **stride 2** is the principled target, not
   3. Running as `starling_stride2`.
4. `flow_arch_heads: 4 -> 8` / `encode_base: 32 -> 48`. Capacity on axes never varied. DEPRIORITISED --
   1.5 is evidence that transition-head capacity is not the binding constraint.

**NOT queued, deliberately: `num_tokens`.** Proposed down (32->16) on an unsupported argument, withdrawn,
then listed up (32->64) without flagging the reversal. §21.2 of the robocasa record has 64 open and
leading at matched eval, but there is no starling evidence in either direction. Leave it alone.
