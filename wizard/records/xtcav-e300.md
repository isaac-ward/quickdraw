# xtcav-e300 — FACET-II XTCAV L2-phase scans (E300/TEST, 2026-04-15 shift)

Real accelerator data: DTOTR2 (XTCAV-streaked LPS screen) + 132 BSA scalar channels, 3 runs ×
~4915 matched shots @ 10 Hz, action = [L2 phase setpoint, TCAV phase, TCAV amp] at shot t+1.
Full setup rationale + interview: `wizard/scripts/xtcav-e300.md` (gitignored, local).
Dataset: `logs/recording_2026_08_18_16_02_20_xtcav_e300` — standardized-obs re-convert, used from run 2 on
(run 1 used the retired raw-unit `logs/recording_2026_08_17_15_22_44_xtcav_e300`). Processor: `xtcav` in
`data/processors.py`; channel table + physical obs mean/std in `xtcav_channels_xtcav_e300.json` in the run_dir.

## 0. Dataset setup + pre-flight (2026-08-17) — DONE

- Conversion 179 s: 24 train / 3 val / 3 held-out `eval` episodes (last-6-scan-step tails),
  obs_dim 132, action_dim 3, frames native 184×894 grey×3, fixed scale 2000 counts → 255.
- `check_dataset`: 9018 train / 1132 val / 2468 eval windows at P8+F64 — OK.
- `model_summary` (mm_flow, TAESD@[96,448], num_tokens=21 EXACT 2688, action head off,
  shortcut off + flow_arch=transformer, symlog, position_idx=null): 4.19M params.
- `fast_dev_run`: passed, grads finite (`logs/train_world_model_2026_08_17_15_28_01_xtcav_e300`).

| number | value | unit |
|---|---|---|
| matched shots (train+val+eval) | 14,748 | shots |
| raw frozen-TAESD floor on these frames | **47.06** | dB PSNR (vs ~31 dB robocasa gate) |
| dropped channels | 6 of 138 | 2 dead-NaN, 2 laser-PWR >10% NaN, 2 ragged |

DECISION: dataset + config are sound; proceed to the first full training
(`wizard/scripts/xtcav-e300.sh` stage 3). First checks in run 1: `motion_ratio` / per-step image
delta vs the 47 dB floor (subsample is the lever if sub-floor), and which blocks landed in val.

## 1. Run 1 — repo-default pace levers, epoch-3 assessment (2026-08-17 23:11 → killed 2026-08-18) — KILLED

Run: `logs/train_world_model_2026_08_17_23_11_51_xtcav_e300` (stage-3 command as compiled in §0:
`recon_frac=1.0` default, `compile_rollout=false` default, raw-unit obs). Killed by Ryan mid-epoch 4
after the epoch-3 checkpoint assessment (standalone eval:
`logs/eval_ood_horizon_2026_08_18_15_49_08_xtcav_e300_ep3`, on `last.ckpt` = epoch 3, val split, H=439).

| number | value | unit |
|---|---|---|
| autobatch result | 4 | batch (probe: batch 8 = 71.9 GB > 64 GB budget — the p_tf=1 parallel path decodes all 72 window frames at recon_frac=1.0) |
| epoch 0 (p_tf=1, parallel) | 28 | min |
| epochs 1–3 (0<p_tf<1, eager AR) | ~4.2 | h/epoch (15.1k s) |
| epoch 4 (p_tf=0, eager — compile off) | ~5 | h/epoch (pace at 50%) |
| trainer ETA at ep3 | 218.9 | h (finish 08-27) |
| ep3 train / val loss | 2.5603 / 3.1498 | normalized |
| ep3 monitored `pointwise_error` | 1.27e9 | raw units — TMIT-dominated, see bug below |
| open-loop PSNR @+1 / @+16 / @+64 / @+109 | 36.8 / 35.0 / 25.2 / 18.8 | dB |
| frozen-frame baseline PSNR @+8..+109 | 34–36 | dB |
| motion_ratio @+1 / @+16 / @+64 | 0.34 / 0.22 / 1.05 | — |

Findings:
- **Pre-competent, as the p_tf schedule predicts**: at/below the frozen-frame baseline through +16
  (mean-ish motion), rollouts leave the manifold by ~+64 (filmstrip: saturated latent garbage vs dark
  GT). Epochs 0–3 were the teacher-forcing ramp; in-rollout training had only just begun. Not a
  model-quality verdict.
- **Motion SNR is healthy**: frozen-PSNR ~35 dB vs the 47 dB TAESD floor ⇒ true shot-to-shot change
  sits ~12 dB above the codec floor. The robocasa sub-floor trap does NOT apply; `data.subsample`
  stays 1.
- **Checkpoint-metric bug**: `pointwise_error` is an L2 over DENORMALIZED obs; our 132-D vector mixes
  TMIT (~1e9 counts) with degrees/mm, so best.ckpt was selecting on TORO/BPM-charge error alone
  (monitored score 1.27e9). The normalized `obs_error` was fine but unmonitored.
- Pace: eager AR rollout (64 steps × 6 ODE samples/step) at batch 4 ⇒ 2255 steps/epoch ⇒ 8–9 days.

DECISION → run 2 with three changes: (1) `model.recon_frac=0.25` (best-recipe value; 4× less decode
memory ⇒ autobatch ≫ 4); (2) `model.compile_rollout=true` (documented ~6× on the p_tf=0 rollout;
head_dim=16 OK, contraction off); (3) dataset re-converted with obs PRE-STANDARDIZED to z-scores on
head-shot stats (physical mean/std in the channels json) so `pointwise_error` weighs channels equally
→ `logs/recording_2026_08_18_16_02_20_xtcav_e300` (same 24/3/3 episodes, 9018 windows; builder stats
on it ≈ (0,1)). Compiled-rollout path pre-flighted via fast_dev_run with `p_tf_start=0`
(`logs/train_world_model_2026_08_18_16_07_11_xtcav_e300_devrun`): compile 157.8 s one-time, grads
finite. Run-1 numbers stay comparable in normalized `obs_error` / image dB, NOT in `pointwise_error`
(unit change).

**Independent review before run 2 (2026-08-18): SAFE TO LAUNCH, no blockers.** The reviewer re-derived
the whole obs/action pipeline from the raw `.mat`/h5 and matched the stored parquets for every episode of
all three splits (max obs deviation 3e-3 = float32 quantization; actions exact), confirmed the join /
background / orientation against the reference reader (streak verified along the 894-col axis: TCAV sign
moves the column centroid 15–35 px, row ~1 px), and validated every stage-3 override against its config
contract. Corrections + accepted limitations it produced:
- TCAV ±90° flips in LONG BLOCKS (~3% of consecutive pairs), not shot-to-shot as §0 assumed; ~51–57
  TCAV-off shots per run are correctly labeled by the amp channel. Conditioning design unaffected.
- `XTCAV_SCALE=2000` clips the peaks of ~1.7% of frames (p99 of frame maxima ~2210, camera saturates
  4095); pixel-level clip fraction 3e-7 — negligible, comment fixed, revisit at next re-convert.
- ≤4 of 14,748 frames may be the wrong shot's image (duplicate PIDs inside a step file, first taken —
  same as the reference reader); the processor now logs collisions.
- `dt=0.1 s` is fiction exactly at the 32 step boundaries (6–9 s gaps) — the only places the L2 action
  moves, so any settling transient is compressed into one model step. Inherent to the framing.
- The TCAV action dims are READBACKS at t+1 (leak ~0.55° phase jitter; also duplicated in 2 obs dims,
  trivially flattering obs metrics). For control use later, switch to setpoints.
- The `eval` tail split is knob EXTRAPOLATION (L2 6.75–8.0° never in train), not an iid holdout.
- Final-row duplicated action can be wrong (even sign-flipped) but is verifiably never consumed by
  windowing/rollout; use true next-shot knobs at the next re-convert.

## 2. Run 2 — speed+metric fixes; best @ep27; late flow-head collapse (2026-08-18 19:08 → 2026-08-20 02:09) — COMPLETE

Run: `logs/train_world_model_2026_08_18_19_08_49_xtcav_e300` (standardized-obs dataset, recon_frac=0.25,
compile_rollout=true, flow_arch=transformer — the repo's FIRST at-scale run of transformer+no-shortcut).
Batch 8 (autobatch, compiled-confirm), ~33 min/epoch, 50 epochs in ~31 h. **best.ckpt = epoch 27.**

| number | value | unit |
|---|---|---|
| best val `pointwise_error` (ep27) | 6.48 | z-L2 over 132 ch (mean-predictor ref √132 = 11.5) |
| best val `obs_error` (ep27) | 0.4375 | per-dim z-MSE (≈56% variance explained) |
| proprio ae_floor at ep29 eval | 5.54 | z-L2 — but PCA says a 128-dim LINEAR code loses only 0.011: the floor is the stochastic flow DECODE head, not capacity |
| closed-loop-1 image @+64 (ep19/29) | 37.4 / 37.0 | dB vs frozen-frame 36.2 — real one-step skill |
| closed-loop-1 image @+64 trend | 38.0 → 34.1 | dB, ep5 → ep49, MONOTONE decline (image skill peaked ep5) |
| open-loop stability | ~+109 → escape ~+200 | steps (physically vacuous beyond drift time; stability check only) |
| collapse onset / endpoint | ep~33 / ep47 | train dynamics/latent 1.9 → 3.8 (ep27) → 38.4; val_loss 41.6 |
| grad/norm/flow | 1.02 → 6.77 → 92 | ep31 → 35 → 47, while decode_image held ~1e-4 — flow head blew FIRST |
| codec erosion (symptom) | roundtrip ×8; ae_floor 5.72→6.54, 39.1→34.2 dB | dragged by the flow blowup |

Diagnosis (fork-reviewed, code-grounded): the transformer flow head's documented explosion mode
(`flow.py:159-179`, 2026-08-11 incident: grad/norm/flow 0.98→766, clip=1.0 "launders" it into smooth
degradation — exactly run 2's no-NaN phenomenology) recurred at ep~33 under constant LR 1e-3 (warmup-only,
no decay). `latent_loss_weight=1` let the blowup drag the codec (symptom). Early-warning channel for any
future run: `grad/norm/flow` (1.0 @ep31 → 6.8 @ep35).

Physics framing (see analyses discussion): shot-to-shot linac data is a static knob→LPS response surface
+ machine-state drift/feedbacks + white jitter — NOT Markovian beam dynamics. Judge the surrogate on
closed-loop/short-horizon, response-surface fidelity, and jitter calibration; open-loop long-horizon PSNR
is uninformative (a perfect model asymptotes to the frozen baseline). Ep15-stretched filmstrips: the model
places energy/centroid correctly but predicts a compact blob, not the S-curve morphology.

DECISION (joint with fork review): evals-first, no immediate rerun.
1. Physics-eval suite vs ep27 best.ckpt (reuse xtcav_metrics + yiheng extractors): (a) response-surface
   sweep of the L2 action vs the measured scan curve, µm-scored, incl. the 6.75–8° tail as extrapolation;
   (b) jitter calibration — histograms + per-channel PSDs of sampled rollouts vs data; (c) per-channel-group
   skill (RF / BLEN / TMIT+orbit); (d) virtual-diagnostic probe (ground proprio, sample image).
2. Gate: evals pass → ep27 IS the surrogate, run 3 optional. Fail → the failing axis defines run 3.
3. Run 3 package (if warranted): `optim.lr=3e-4` (primary stability lever), `dynamics_detach_encoder=true`,
   `modalities.1.latent_loss_weight=10` (belt-and-braces), `modalities.0.decode_kind=mse` (primary quality
   lever — collapses the proprio floor; monitor values will RE-SCALE, not comparable to run 2), keep
   flow_arch=transformer, keep 50 epochs. Bundle v3 re-convert: true next-shot knobs for the final action
   row; reconsider BC11 BLEN (57% NaN = real dropout; validity-mask channel rather than ffill → +2 dims).
4. Optional 10-line change if run 3 happens: checkpoint monitor on closed-loop-1 error instead of
   horizon-averaged open-loop pointwise_error.

## 3. Physics-eval suite + L2-slew verdict on run-2 best.ckpt (2026-08-20) — MODEL FAILS RESPONSE SURFACE

Suite: `wizard/scripts/xtcav_physics_eval.py` (v2 after an adversarial physics-critic review that found
4 blockers in v1: loose TCAV gating admitted 54% off-phase shots; pooling 3 working points fabricated the
curve; model-res extractor had phi-correlated 5-95% rejection and 2-10x underestimation; ps cal assumed
S-band — the TCAV is X-BAND 11.424 GHz, 1 deg = 0.2431 ps, ~4936 um/ps at 1200 um/deg).
Outputs: `logs/physics_eval_xtcav_e300/` (measured.json, validate.json, slew.json, score.json, PNGs).

**Measured target (now validated, a standing asset):** strict deployed on-shot gating (||phase|-90|<1 deg,
|amp-mean_on|<0.25 MV) -> 6,799 shots (Tier-0: ~6,706), energy-gated extraction 90.3%, per-setpoint medians
agree with Tier-0 native-res truth: ratio med 1.00 (IQR 0.96-1.08) 15673, 1.09 TEST, 1.04 (noisy —
separations ~8 model px) 15671. Per-run compression V-curves with zero crossings: 15671 4.2-5.6 deg,
15673 6.6-7.1, TEST 5.8-6.6 (linear-chirp fit |s0(1+c*phi)|, s0 = 1.3 / 4.2-5.1 / 3.0-4.0 mm screen).

**Slew protocol:** 12 strict-gated contexts (3 runs x phi {0.5,2,4,6}), +-2 deg ramps at 0.25 deg/setpoint,
dwell 10 keep last 5, 2 stochastic reps, up/down separate. 99% of decoded frames pass extraction.

| verdict metric | value | bar |
|---|---|---|
| in-range RMS vs measured medians | 4.0-5.0 mm (all runs) | pass <=150 um, fail >=300 um |
| model response-fit zero crossing | 43-64 deg (=flat), TEST -0.25/deg (noise) | within +-0.5 deg of per-run zc |
| median \|sep error\| at ~5 rollout steps | 1.8 mm | — kills the drift explanation |
| error vs rollout depth (5-85 steps) | flat 1.9-3.2 mm | not drift-dominated |

**Conclusion:** run-2 ep27 decoded frames do NOT carry quantitative two-bunch separation, from the first
predicted steps; the L2 response surface is absent from the image head (consistent with blob-not-streak
morphology, PSNR-near-frozen, motion_ratio ~0.2 = latent mean-collapse). Per §2's gate: the failing axis
defines run 3 — image-side LPS structure under action changes. Candidate levers (in addition to §2's
stability+decode package): boundary-window enrichment (~28% of training windows contain any action change),
raising image recon emphasis / physics-feature-aware eval-driven selection, and testing the
virtual-diagnostic mode (ground proprio, sample image) to separate "can't render structure" from "can't
infer it from context".

**1-step boundary response addendum (--part boundary, same day):** 20 strict-gated real boundaries
(8/5/7 per run), exactly one predicted step with the real action + a hold-L2 counterfactual, 8 samples
each. Result — the discriminator resolves toward "can't MAINTAIN under rollout", not "can't render":
with real measured context, TEST_15668's 1-step predictions track the measured curve at low-mid phi
(median |dev| 304 um, sensitivity corr 0.86), 15673 partial (median 1004 um, corr 0.50), 15671 poor
(1278 um — its separations are ~8 model px, extraction-limited). Real-vs-hold action deltas are
right-signed in 2/3 runs but at ~0.25-deg scale (~100 um) they sit below extraction noise. Contrast
with the slew's flat 1.9-3.2 mm error from 5 steps on: self-generated context degrades physics content
far faster than pixel PSNR suggested. Also large stochastic-decode spread (per-boundary IQR up to
+-1.5 mm). Run-3 emphasis updated: (1) rollout-conditioning robustness + decode-variance reduction on
the image head, (2) boundary enrichment for action sensitivity, (3) n=20 is thin — a staircase
counterfactual protocol (several 0.25-deg increments within <=16 steps of real context) would give
denser response coverage next time. As a re-grounded (per-shot measured context) 1-step surrogate,
ep27 is marginally useful at TEST-like working points but nowhere near the 150-um bar.

**Boundary coverage revision (same day):** the low-phi 15673 gap was a CONTEXT-GATE artifact: (a) TCAV
phase-ramp recovery shots at step edges (correctly excluded), plus (b) stable blocks parked at ~88 deg
(sin 88 = 0.9994 streak strength) rejected by the literal +-1 deg deployed window. Context gate relaxed to
"stable streaked block" (||ph|-90|<5, ph std <1.5 deg, amp window; measured-curve population stays
deployed-strict) -> 42 boundaries (14/run), full 0-8 deg coverage. REVISED verdict: the n=20 result was a
favorable subset. At n=42: median |dev| 1.10-1.45 mm in ALL runs, sensitivity corr collapsed (-0.16..+0.64
noise), median real-vs-hold action delta ~0 at the 0.25-deg scale, per-boundary stochastic spread +-2-4 mm.
Some individual boundaries track the curve well (TEST low-phi, 15673 ~2 deg), but the aggregate says: the
1-step conditional is HIGH-VARIANCE and only sporadically calibrated, with no measurable 0.25-deg action
sensitivity — before any rollout. Unified diagnosis: decode-variance + weak conditioning of the image head
is the primary defect (slew's 1.9-3.2 mm = this + rollout degradation on top). Run-3 priority order
updated accordingly: image-head calibration/variance first, boundary enrichment second, rollout robustness
third.

## 4. Run 3 pre-registration — critic-reviewed package (2026-08-20) — RUNNING (arm 1b)

**Arm 1a aborted at ep11** (`logs/train_world_model_2026_08_20_15_27_55_xtcav_e300_r3`, ~2 h): the run-2
flow-head blowup signature appeared HOT FROM THE START despite p_tf=1 — grad/norm/flow 2.04 → 3.03 → 4.81
at eps 3/7/11 (run 2 sat at ~1.0 until ep31), dynamics/latent 1.25→1.59, clip_ratio 4.9 (laundered-blowup
regime), val obs_error 0.96→2.51 — while the 1-step decode losses still improved. Suspected accelerant:
recon_frac=1.0 backprops the decode loss through the flow-sample ODE unroll at EVERY window step at batch
24 — a much hotter version of the documented unstable path. Killed on the pre-registered early-warning
channel; ep-3 best.ckpt preserved. **Arm 1b** = identical config at `optim.lr=1e-4`
(`logs/train_world_model_2026_08_20_17_28_20_xtcav_e300_r3b`, batch 24, ~10 min/epoch). If the signature
recurs at 1e-4, next levers in order: train-time sampling_steps 6→4 (shorter unroll), flow_arch_depth=1,
then recon_frac compromise (0.5 + boundary-frame-guaranteed supervision would need a code change).

**Arm 1b also aborted at ep11**: grad/norm/flow 2.31 → 2.62 → 3.89 — the lr cut damped growth for 4
epochs (+13%) then re-accelerated to arm-1a's rate (+48%), train/val turning up with it. The lr lever
delays, not fixes. **Arm 1c** (= 1b + train sampling_steps 4) **aborted at ep7**: 3.60 → 4.78 — shorter
unroll made it WORSE (coarser samples -> larger decode loss through fewer steps). Three arms across two
lrs and two unroll depths = the instability is intrinsic to BACKPROPAGATING THE RECON LOSS THROUGH
flow.sample AT ALL. Flow-grad ladder: 1a 2.04/3.03/4.81, 1b 2.31/2.62/3.89, 1c 3.60/4.78 (eps 3/7/11).

**Arm 1d — the principled fix (RUNNING):** `model.pred_obs_in_loss=false` — a newly config-exposed flag
riding lit.py's existing EMA/JEPA "decoder-only probe" path (`recon_src = preds.detach()`): the recon
loss trains ONLY the decode heads (on realistic detached samples); the flow head + backbone train purely
by VELOCITY MATCHING — the theoretically correct conditional estimator — and the adapters by the x10
roundtrip. The unstable gradient path no longer exists. Code: setup.py build_model exposes the flag
(default true = bit-identical), declared in mm_flow.yaml. Config = arm 1b + this flag (sampling_steps
back to 6): `logs/train_world_model_2026_08_20_20_47_49_xtcav_e300_r3d`. Caveat recorded: with the
decode-through-samples pressure gone, the variance gate rides on flow matching's own calibration — if
the variance ratio still fails, the M1 fallback (image weight/recon emphasis) is NOT available in this
configuration and the honest next lever is a mean-of-K decode policy as a named deliverable.

**Arm 1d killed at ep3 for a COMPOSITION BUG** (caught by inspection, not metrics): with pred detach +
dynamics_detach_encoder + detached flow targets + no proprio roundtrip (roundtrip is gated on
latent_loss_weight>0, set only on the image modality), the PROPRIO ENCODER received zero gradient from
any loss — a frozen-random encoder. **Arm 1e** = 1d + `+model.modalities.0.latent_loss_weight=1`
(append form — the proprio modality dict doesn't declare the key, the yaml's own documented struct
trap): the proprio encoder/decoder train as a supervised autoencoder (PCA says the 138->128 squeeze is
lossless), dynamics predicts in that anchored latent. Grad-flow audit of the 1e configuration: flow head
<- velocity matching only; backbone <- flow loss; proprio enc/dec <- roundtrip; image adapters <-
roundtrip w=10; decode heads <- recon on detached samples. Run:
`logs/train_world_model_2026_08_20_21_34_17_xtcav_e300_r3e`, batch 24.

**1e in-flight findings (ep7):** (i) the `+latent_loss_weight=1` append did NOT plumb to the
VectorModality module attr `roundtrip_losses` gates on — `grad/norm/encode_proprio` = 0, no
`roundtrip_proprio` term: the proprio encoder is a FIXED RANDOM projection. Quasi-benign (PCA: a
128-dim code is lossless, and decode/proprio 1.15→0.56 shows the decoder learning to invert it) but
unintended — plumbing fix queued for a 1f only if gates fail. (ii) flow-grad still grows (2.24→3.86)
with the recon path fully severed — the growth is intrinsic to the velocity-matching fit (targets are
also non-stationary while the image adapters train). Unlike 1a/1b the LOSS falls (dynamics/latent
1.12→1.00), so this is steep fitting, not the spiral. REVISED abort criterion: kill only if
dynamics/latent turns UPWARD while flow-grad grows (the actual 1a/1b signature). Letting 1e run to 50.
Side note: the box's NVML/driver userspace went mismatched mid-day (nvidia-smi broken, CUDA fine) —
monitoring via torch only until a reboot realigns it.

Plan reviewed by an independent critic agent: GO-WITH-CHANGES; all amendments adopted. Key correction to
the plan's own reasoning (M1): the flow-matching loss was ALWAYS teacher-forced (`loss_terms` re-encodes
real contexts regardless of p_tf) — p_tf only changes the decode/recon conditioning and the gradient path
through chained samples (the run-2 blowup path). So p_tf=1 attacks overdispersion via decode-through-
samples pressure on real contexts, not via the flow loss; if the variance gate fails, the pre-registered
fallback lever is raising image recon emphasis vs lambda_flow (a mean-of-K decode policy would be a
separately named deliverable, not a silent fix).

**Dataset v3** `logs/recording_2026_08_20_15_20_44_xtcav_e300`: 6-step blocks (13 train / 2 val / 3 tail
eval; val = 15673-block0 + TEST-block1), TCAV settle-head drops (7-17/run) + off/ramp drops (~900/run,
18% — the off-phase contaminant population), masked-channel band (BLEN:IN10:596 + 2 laser PWR as value +
validity, mean-imputed not ffilled; obs 138-D), true next-shot final action rows. Extractor re-validated
on v3 (ratio med 0.99-1.10 vs Tier-0; measured fits unchanged: zc 4.2-5.6 / 6.6-7.2 / 5.8-6.6 deg).

**Arm 1 config** (`wizard/scripts/xtcav-e300.sh` stage 3, fast_dev_run passed incl. the intentional
full-TF trip-wire): p_tf=1.0 constant (1-step conditional IS the model at 10 Hz; run 2 proved the defect
exists before rollout), F=16 + window=24 (train/eval span parity), recon_frac 1.0 (0.25 would drop the
boundary frame from decode supervision 75% of the time), `data.boundary_frac=0.35` (loader duplicates
boundary-in-supervised-region windows; 944/8412 = 11.2% x4 -> 39%/epoch; superset of dense coverage),
mse proprio decode (monitor RE-SCALES again — not comparable to runs 1-2), lr 3e-4 +
dynamics_detach_encoder + latent_loss_weight=10, sampling_steps 6 at train / 12 at eval (the TF decode
loss backprops through the sample unroll — raising it at train time is counterproductive).

**Pre-registered acceptance gates** (physics suite, v3-rebased): scored on **val+eval boundaries only**
(enrichment oversamples train boundaries — pooled numbers could pass on memorization) + the staircase
re-grounded protocol (`--part staircase`: 0.25-deg increments every 4 shots within 16 steps of real
context). PASS = variance ratio <= 2x AND single-sample dev <= 1.5x fair null AND median response curve
zc within +-0.5 deg / <= 300 um RMS on re-grounded protocols. Slew = secondary rollout-stability check
(looser bar; fallback if closed-loop-16 PSNR drops below the frozen baseline: short in-rollout fine-tune
at lr 3e-4 with abort on grad/norm/flow > 2). Training metrics are a fresh baseline (horizon, decode,
obs_dim all changed) — cross-run comparison happens ONLY inside the fixed physics suite.

**Arm 2** (only if arm 1 passes calibration gates): history-blind control — contexts shuffled
within-setpoint on BOTH streams (proprio + frames; per-setpoint means would leak future shots and blur
the image context). Plus a zero-training "marginal sampler" null (random real same-setpoint shot) in the
eval. If arm 1 doesn't beat both, history conditioning isn't earning its keep and the static-surrogate
route wins.

## 5. Run-3 arm 1e verdict vs pre-registered gates (2026-08-21) — CALIBRATION PASS, RESPONSE FAIL → arm 1f

Arm 1e aborted at ep27 on the (revised) spiral criterion: dynamics/latent bottomed ep11 (0.982), rose to
1.138 by ep27 with flow-grad envelope 4.1→6.5 — but NO codec drag this time (roundtrip_image kept
improving; the pred-detach did its structural job). Best models = snapshot ladder eps 3-15. Gate battery
(v3-rebased suite; boundary n_rep=32 on eps 3 and 15; staircase; slew):

| pre-registered gate | ep3 | ep15 | bar |
|---|---|---|---|
| variance ratio (per run) | 1.42 / 0.79 / 1.50 | 2.17 / 1.25 / 2.40 | <= 2 -> **ep3 PASS** (run 2: 5.7-13x) |
| boundary gate dev (val+eval) vs fair null | 183/335, 152/365, 350/548 um | 91, 122, 670 um | <= 1.5x null -> **ep3 PASS all** |
| response curve (staircase+boundary): model zc | 21.8 / 9.8 / 11.3 deg | 55 / 9.6 / 14.6 deg | measured 4.2-7.1 +-0.5 -> **FAIL** |
| response slope vs measured | 23% / 70% / 55% of true c | similar | — partial response, right sign+ordering |
| in-range RMS (re-grounded medians) | 1.10-1.70 mm | 1.01-1.70 mm | <= 300 um -> FAIL |

**Reading:** the two defects that disqualified run 2 as a stochastic surrogate are FIXED — the conditional
is CALIBRATED (variance at the jitter scale, boundary predictions at the single-shot noise floor) — but
the model still under-responds to the L2 action away from its grounded context (context-leaning). Train
boundaries score WORSE than val/eval (no enrichment-memorization artifact). Sensitivity corrs remain
noise at the 0.25-deg scale, as pre-registered.

**Arm 1f (RUNNING, `logs/train_world_model_2026_08_21_02_08_25_xtcav_e300_r3f`):** the pre-registered
escalation for exactly this failure — `boundary_frac` 0.35→0.5 — bundled with the queued VectorModality
`latent_loss_weight` plumbing fix (bit-identical default-None scheme in modalities.py; devrun-verified:
roundtrip_proprio fires, grad/norm/encode_proprio != 0), so action conditioning can route through TRAINED
readback features instead of a frozen-random projection. Two changes, flagged: one is the registered
escalation, one is a bug fix. If 1f still fails the response gate, next is the history-blind control +
marginal-sampler null to size how much of the remaining skill is context-copying, and the loss-side
question (physics-feature-weighted recon) reopens.

**Arm 1f/1g addendum (2026-08-21 morning):** 1f (enrichment 0.5 + proprio-roundtrip fix bundled) blew up
by ep7 (flow-grad 76): a MOVING proprio latent makes the velocity targets non-stationary — the roundtrip
fix requires a pretrain-freeze stage (deferred; the modalities.py plumbing itself is correct and default-
off). 1g (single variable: 1e + boundary_frac 0.5) is the healthiest arm yet (dyn floor ~1.0 at ep15-23,
no spiral through ep23). Early gate on its ep15 snapshot: calibration PASS again (varratio 1.13-1.80,
gate devs 183/198/730 vs nulls 335/365/548) but **response slopes UNCHANGED (37/64/55% of measured)** —
doubling boundary emphasis did not move the response: data frequency is NOT the binding constraint.
Working hypothesis: HEDGING — at a boundary the context (8 shots at phi_old) and the action (phi_new)
conflict, and the conditional shrinks toward context; a distance-from-context probe shows near-zero bias
at the grounded setpoint growing to ~0.5-0.8 mm positive bias 1-2 steps out (suggestive; up/down pooling
makes it non-decisive). **Marginal-null insight**: an unconditioned same-setpoint draw scores dev=fair-null
and varratio=1 BY DEFINITION — the WM's sub-null gate deviations therefore already prove history
conditioning adds value at grounded 1-step; the deficit is specifically the ACTION channel. Candidate
next levers (post-1g): context-dropout training (classifier-free-guidance style, forces action reliance),
action-guidance at inference (extrapolate real-vs-hold branches), staged proprio-AE pretrain.

**RUN-3 CAMPAIGN CLOSE-OUT (2026-08-21 ~11:30).** 1g stopped at ep35 on the criterion (dyn 1.04→1.38
from ep23). ep23 battery: varratio degraded past the dyn floor (2.4-3.4), slopes unchanged (29/58/41%).
**CHAMPION: arm 1g ep15** (`.../xtcav_e300_r3g/checkpoints/snap_ep15.ckpt`, copy also at snaps/):
calibration PASS (varratio 1.13-1.80; gate devs 183/198/730 um vs nulls 335/365/548 — sub-null on 2/3
runs, 1.33x on TEST), beats the marginal-sampler null (= history conditioning demonstrably adds value),
response gate FAIL at 37/64/55% of measured slope. Campaign findings, each isolated by one arm:
(1a-1c) recon-through-flow-samples gradient is intrinsically unstable at any lr/unroll -> fixed
structurally via config-exposed pred_obs_in_loss (in codebase, default-safe); (1f) a moving proprio
latent under the flow head is untenable -> roundtrip fix needs pretrain-freeze staging; (1e vs 1g)
boundary enrichment 0.35->0.5 changes nothing about response -> the action deficit is STRUCTURAL
(hedging-consistent), and its levers are conditioning-side: context-dropout training, inference-time
action guidance, staged AE pretrain. Seven arms, ~12 GPU-h total (vs run 2's 31 for one arm).
Status vs the program: a calibrated, history-conditioned 1-step stochastic surrogate EXISTS (usable
for re-grounded jitter-aware prediction now); a knob-response surrogate does NOT yet.

**Boundary-blur diagnosis (2026-08-21, `blur_diagnosis_r3g_ep15.png`):** the diffuse blur in champion
samples at setpoint changes is BOUNDARY-SPECIFIC and NOT an ODE-integration or decoder-sharpness
artifact: no-boundary controls (action=hold) produce crisp committed streaks at 12 ODE steps, and 48
steps at boundaries is no sharper. Confirmed mechanism: under context-action CONFLICT the flow's
conditional smears across the hypothesis region; single samples are latent SUPERPOSITIONS which the
frozen TAESD renders as diffuse structure (off-grey chroma = off-manifold tell). Same phenomenon as the
40-65% response slope — the slope deficit is the hedge's MEAN, the blur its SPREAD. This promotes
context-dropout (CFG-style) training to the clearly-indicated run-4 lever, with inference-time guidance
as its natural companion; extra ODE steps / longer training are ruled out as fixes.

## 6. Run-4 design proposals (2026-08-21, pre-critic draft)

Unifying diagnosis from runs 2-3: ONE residual defect — the flow under-commits wherever its conditional
is broad — triggered two ways: context-action conflict at setpoint changes (the 40-65% response slope =
the hedge's mean; boundary blur = its spread) and intrinsic jitter breadth within-step (closed-loop-1
blur on curved-streak shots; sign/boundary proximity ruled out). Width calibration is solved; MODE
COMMITMENT is not.

**Arm A (leading): scalar-conditioned factorized image head ("virtual-diagnostic factorization").**
Factorize p(obs' | ctx, a) = p(scalars' | ctx, a) x p(image' | scalars', ctx, a). Train the image
pathway conditioned on the TARGET-step proprio (teacher-forced realized scalars) so jitter is EXPLAINED
AWAY — image targets become near-deterministic given inputs, the decode pathway gets sharp consistent
supervision, and jitter-hedging in the image head is removed by construction. Inference = ancestral:
sample scalars from the (already calibrated, best-trained) proprio conditional, then image given
scalars. Expected second benefit: the knob->beam response burden RELOCATES to scalar space where the
per-shot SNR is far better (e.g. BC14 BLEN's L2 response) and where run-3 models already perform best.
Precedent in current inputs: the TCAV t+1 readbacks are already conditioning and never hedge.
Known risks: (i) exposure bias — image head trained on true scalars, run on sampled ones (mild, 1-step;
consider scalar-noise augmentation at train); (ii) response masking — the direct action->image path
gets even less gradient, so the ACTION-SENSITIVITY GATE MUST BE SCORED ON THE SCALAR CONDITIONAL
explicitly, and context-dropout applies to the scalar model where hedging then concentrates.

**Arm B (comparison): context-dropout (CFG-style) on the current joint architecture.** Randomly mask
the context (both streams; keep the action) during training so the model must learn a committed
action-conditioned answer; classifier-free-guidance-style sharpening/extrapolation available at
inference. Directly targets the boundary-hedge; does NOT by itself address within-step jitter-breadth
blur.

**Supporting levers (either arm):** inference-time action guidance from real-vs-hold branches
(w ~ 1/slope-fraction ~ 1.5-2.5, zero-training, validate against the measured curve); staged proprio-AE
pretrain-freeze (unblocks the roundtrip fix that blew up arm 1f when trained jointly); careful
shortcut/consistency term as a mode-commitment device for the jitter-breadth case (known mean-collapse
risk — the opposite failure — so gate on variance ratio staying >= ~0.7).

**Gates:** same suite, same val+eval discipline; add (a) scalar-conditional action-sensitivity gate
(BLEN/energy channels response vs measured per-setpoint medians), (b) image-given-true-scalars
sharpness/fidelity gate (virtual-diagnostic mode — separation extraction on images rendered from
measured scalars vs realized frames), (c) ancestral-vs-teacher-forced gap gate (exposure-bias monitor).

### §6 SHARPENED (post-critic review, 2026-08-21 — adopted in full)

**Verdicts:** Arm A GO-WITH-CHANGES; Arm B (context-dropout, both streams) **NO-GO as drafted** —
the CFG direction amplifies CONTEXT reliance, backwards against the §5 diagnosis (the deficit is the
action channel; sub-null drift-tracking is the campaign's proven asset and the first casualty), and the
uncond branch is quasi-ill-posed under predict=residual. Re-scoped to ACTION guidance (zero-training
first). **Shortcut/consistency lever STRUCK**: the repo's own yaml documents it as mean-seeking on
broad conditionals — it manufactures the exact latent superposition being fought.

**Three zero-training probes BEFORE any run-4 training (hours, can re-scope both arms):**
- **P1 jitter-explainability** (CPU, data only): per-setpoint R² of extracted separation on the 138
  scalars (strict-gated shots; EXCLUDE the 2 TCAV-readback-duplicate dims). Sets Arm A's breadth-blur
  ceiling (residual 1-R² stays blurry) and selects gate-(a) channels.
- **P2 virtual-diagnostic clamp on the champion**: slot-0 RePaint-style clamping in a bespoke Euler
  loop over m.flow.velocity (mirror _rollout_step; exact under latent_norm=affine, predict=residual;
  ~50 eval-script lines, zero model changes). Crisp+responsive -> the joint flow already learned the
  scalar-image correlation (strong Arm A prior); still blurry -> Arm A viable only as a trained
  conditional.
- **P3 action guidance**: v_hold + w(v_real - v_hold) with SHARED eps (FlowField.sample exposes eps;
  predict_next doesn't — mirror in script), w in {1,1.5,2,3}, ±1-2 deg counterfactual deltas (0.25 deg
  sits below extraction noise). If slope recovers inside calibration bounds -> deployable partial fix,
  trained CFG largely redundant.

**Arm A corrected design** (one mechanism, all champion stability settings frozen):
- Condition on the RAW normalized 138-D scalars via a new learned scalar_cond_enc — NOT the frozen
  random-projected token (random 138->128 loses ~7%/channel in expectation and forces the denoiser to
  unmix an arbitrary rotation; raw conditioning also DECOUPLES Arm A from the staged AE pretrain).
- **Leak fix (critical)**: a single-pass _cond extension leaks the answer to the proprio token both
  directly (broadcast) and through the token-mix attention (flow.py:184-189). Correct form = one joint
  FlowField, extra cond channel (h_dim 384->512), TWO masked flow-loss passes: pass U (zero channel,
  loss on slot 0 only) trains the leak-free scalar conditional; pass C (scalar_cond_enc(true next
  scalars + sigma·noise), loss on slots 1-21) trains p(image'|scalars',ctx,a). Needs a ~6-line
  loss_mask in TransportHead.loss (None = bit-identical).
- Inference = SPLICE-ancestral predict_next: pass-1 (zero channel) -> take slot 0, decode to scalars;
  pass-2 (conditioned on decoded scalars) -> take slots 1-21; splice. forward() then automatically
  trains decode heads on ancestral samples (free alignment with gate c).
- Plumbing via build_model with default-off fallbacks (old configs rebuild bit-identically; h_dim
  change means champion ckpts can't load into scalar_cond=true builds — loud fail, evals unaffected
  since they rebuild from each run's own config.json). Declare keys in mm_flow.yaml (the 1e struct
  trap); devrun-verify plumbing as in 1f.
- **Eval contract**: all response/calibration gates score the full ANCESTRAL chain; any protocol fed
  realized target scalars is a rendering diagnostic and can never satisfy a response gate. The chain
  slope = scalar-slope x image-given-scalar-slope — a flat scalar conditional with crisp images fails
  the mission while passing sharpness metrics, so gate (a) on the SAMPLED scalar conditional is
  PRIMARY.

**Gate set with thresholds:** (a) sampled scalar conditional, P1-selected response channels, val+eval
boundaries+staircase: slope ratio >= 0.8, right sign 3/3 runs (chain today: 37-65%). (b) image-given-
TRUE-scalars vs the realized shot (not setpoint median): acceptance >= 90%, median |sep-realized| <=
~300 um, fixed-scalar sample IQR <= 1.5x P1 residual sigma, band-sigma blur tell <= 1.2x — DIAGNOSTIC
only. (c) ancestral-vs-TF gap <= 30% relative — attributes failure, never decides; the decider is the
ancestral chain passing the standing response bar (zc ±0.5 deg, <= 300 um RMS re-grounded, <= 1.5x
fair null single-sample). (d) chain variance ratio in [0.7, 2.0] (Arm A can now fail over-confident).
(e) boundary dev <= 1.5x null on val+eval, train-split diagnostic kept. (f) val proprio obs_error
non-regression. (g) any guidance w: variance >= 0.7x AND boundary dev stays sub-1.5x-null. (h) slew
unchanged as secondary stability check. n_rep >= 32 pre-registered.

**Order:** P1 -> P2 -> P3 -> Arm A (snapshot ladder, spiral abort criterion, gate at ~ep15; training
metrics re-baseline — image velocity targets get easier) -> trained action-dropout CFG only if P3
shows a real-but-weak direction. Parallel arms rejected (one-lesson-per-arm at ~10 min/epoch).

### §6 PROBE RESULTS (2026-08-21 afternoon) — BOTH ARMS RE-SCOPED BY EVIDENCE

`wizard/scripts/xtcav_run4_probes.py`; outputs `run4_p{1,2,3}_*.json`.
- **P1 (jitter explainability): the ceiling is LOW.** Out-of-fold R² of within-setpoint separation
  residuals on the 138 scalars (strict-gated, TCAV-dup dims excluded, ridge 5-fold): **-0.01 / -0.01 /
  0.33** (15671 / 15673 / TEST; sigma 465/538/336 um). In 2 of 3 runs the jitter is NOT measured by
  the BSA channels (plausibly unmeasured L2 phase + streak-cal jitter). Consequences: (i) Arm A cannot
  explain jitter away — the within-step blur is mostly IRREDUCIBLE given current inputs; (ii) the
  model's width there is CORRECT, and the only fixable rendering defect is COMMITMENT (sharp sample
  from a wide conditional), not width; (iii) sharpness gates at broad conditionals must not demand
  what the data cannot support.
- **P2 (virtual-diagnostic clamp): the joint flow NEVER LEARNED scalar->image coupling.** Clamping the
  proprio token to the encoded TRUE next scalars along the ODE changes nothing: median |sep-realized|
  304 vs 304 um, band-sigma 707 vs 711 um (15 val+eval boundaries x 16 reps). Arm A would build the
  conditional from scratch — against P1's low ceiling. **Arm A DEMOTED**: its surviving value
  (response relocation to scalar space) does not require the image factorization; gate the scalar
  conditional directly instead.
- **P3 (action guidance): DEAD.** v_hold + w(v_real - v_hold), shared eps, ±1-2 deg deltas, 15
  contexts x 42 bins: pred-vs-meas slope stuck at **0.51 for every w in {1,1.5,2,3}**; var-ratio ~2.7
  at these off-distribution jumps. The under-response is NOT a linear shrinkage in velocity space —
  the flow's action dependence SATURATES, so no inference-time arithmetic recovers it, and the trained
  CFG premise weakens accordingly.

**Post-probe assessment.** The champion is closer to the data ceiling than assumed: within-setpoint
width is right and largely irreducible; grounded 1-step location is sub-null. The two REAL remaining
gaps: (1) response slope 0.5 vs 1.0 — the information exists in the data across setpoints (the
measured curve is derived from it), so this is a credit-assignment/training problem in the action
pathway, not an information problem; candidate levers for a redesigned run 4: feed the action as an
explicit DELTA vs the context setpoint (the model currently must infer the change by comparing action
to readbacks), an auxiliary loss on response-carrying scalar channels, or physics-feature-aware
supervision (the §5 reopened question). (2) mode commitment in rendering — with width irreducible,
the honest options are mean-of-K decode as a named deliverable for pointwise use, or accepting
committed-sample quality as a research question. NO run-4 training launched on the old design;
next design round should start from these two gaps.

**Emittance-confound analysis (2026-08-21, Ryan's observation — adopted as a gate):** an uncommitted
sample is indistinguishable, at the single-image level, from a real transverse-emittance increase
(screen image = LPS ⊗ transverse spot; superposed displaced streaks ≡ one wider streak). A surrogate
leaking uncertainty into apparent beam size is systematically biased toward high-ε_n wherever it is
unsure — unacceptable for an ε_n-targeting program. COMMITMENT IS THE DECOUPLER: emittance =
persistent per-sample width; jitter uncertainty = cross-sample spread of sharp samples; uncommitted
models collapse both before extraction. New pre-registered **width-calibration gate**: distribution of
per-sample band-sigma must match the real per-shot band-sigma distribution per setpoint (alongside the
existing cross-sample variance gate). CAVEAT adopted: **mean-of-K decode is admissible only for
location-type features — for width features it manufactures the emittance artifact** (strike it from
any width-sensitive path). Commitment routes ranked: (1) best-of-K manifold filtering (immediate;
also measures native commitment rate; select on manifold membership, NEVER on feature value);
(2) retrieval/local-projection onto the empirical latent manifold, conditioning-aware (deliverable
mechanism; must project within the commanded-setpoint bin or it undoes the response); (3) discrete
mode-variable head (principled redesign; commitment at a categorical choice, flow refines within-mode
where it already behaves). Ruled out: reflow (bakes current smear into the coupling), consistency
distillation (mean-seeking, §6), adversarial decoder sharpening (fake commitment + reopens the
unstable sample-gradient path), low-temperature sampling (destroys calibrated width). Standing program
rule: single uncommitted samples never feed emittance-sensitive consumers.

**Best-of-K measured (2026-08-21, `--part bestk`, K=32, champion; `run4_bestk_{gallery,widths}.png`):**
NATIVE COMMITMENT RATE 30% at tight conditionals / 21% at boundaries (val+eval; 13 controls + 15
boundaries) — the sharp/diffuse MIXTURE hypothesis confirmed; K=32 yields ~7-10 committed samples per
context (3-5x throughput cost). Rejections: dominated by too-wide (superpositions), but 61 samples
rejected TOO-NARROW — the two-sided band cuts both directions. **Bias check: accepted-width median =
1.11-1.13x the real per-shot median — NOT biased toward low emittance**; the accepted distribution
tracks the real per-shot width distribution incl. its 1200-1600 um tail, while the unfiltered model
distribution skews wide with a long superposition tail. The anti-bias construction (two-sided +
conditioning-local bands) is what prevents an "imagine low emittance" filter; the residual risks to
monitor are (i) drift of the accepted/real width ratio away from ~1 and (ii) COVERAGE bias — per-context
commitment rates vary (some contexts near 0/32), so filtered outputs under-represent the states where
the model is least committed; report per-context acceptance alongside any filtered product. Post-filter
location: median |sep dev| 365 um at controls (fair-null territory) and 730 um at boundaries (improved
from ~1.1 mm unfiltered; response deficit remains — filtering does not fix the slope).

**RETRACTION (same day, Ryan's visual audit of the simplified gallery):** the width-only acceptance
criterion is INVALID — "kept" samples are dim smears whose faint skirts fall below the extractor noise
threshold, leaving an in-band width on the surviving core. Multi-feature re-test (real p5-p95 bands per
run on width + TOTAL INTENSITY + peak brightness, 480 boundary samples): width alone passes 77%,
total-intensity 2%, peak 10%, **all three — honest commitment — 1-2%. Best-of-K is DEAD as a practical
mechanism** (the earlier "21-30% commitment" was an in-band-width rate; the morphology-level bias check
is void). New crisp defect isolated: samples are systematically DIM — per-sample integrated intensity
(∝ charge) almost never reaches the real band: superpositions spread energy into sub-threshold haze,
i.e. per-sample charge non-conservation. What survives: the two-sided conditioning-local real-band
methodology as the acceptance GATE, now JOINT (intensity+peak+width, ideally + latent NN distance).
Re-ranked routes: (1) retrieval/local projection onto real latents = near-term mechanism (passes the
joint gate by construction; must stay conditioning-aware); (2) discrete-mode head = research track;
(3) per-sample charge-conservation as a first-class gate (renormalization is NOT a fix — a bright
smear is still a smear). Lesson recorded: single-feature manifold proxies are insufficient; visual
audit caught what the metric passed.

**Fair-null correction (scoring fairness audit):** the measured curve is a many-shot median while model
points are few-sample medians — single-draw scoring has a nonzero floor even for a perfect model. Measured
floor: a REAL single shot deviates from its setpoint median by median 426 um (jitter + single-shot
extraction noise; n=23 realized boundary shots). Against that corrected bar: model single samples deviate
1248 um from the median (2.9x the floor) and 1461 um from the realized next shot (calibrated would be
~sqrt(2)*426 ~ 600 um). Variance calibration: model per-boundary 8-sample IQR = 2115 um vs real
per-setpoint shot IQR = 297 um -> the model's conditional distribution is ~7x WIDER than real shot-to-shot
jitter (some of this is extraction noise on smeared model frames — itself a defect, but it conflates
structural blur with separation error). The averaging asymmetry therefore explains ~1/3 of the raw error
scale, not the failure. It also explains the n=20 "clean subset" optimism: with 8 samples of a ~2.1 mm-IQR
distribution, each per-boundary median carries ~+-0.7 mm SE — a handful of points landing on the curve is
expected by chance, and sensitivity correlations from <=7 pairs of ~100-um-scale quantities are meaningless.
Scoring rules going forward: (a) model MEDIAN curves (many samples/contexts) vs measured medians -> the
150-um bar applies; (b) single-sample scoring -> the ~430-um fair-null bar applies; (c) report the variance
ratio (target ~1x) as a first-class calibration metric. Note: closed-loop-1 pixel PSNR beating frozen (37-38 dB) coexists with this — pixel
metrics were exactly as blind to the physics as §"loss/eval reformulation" predicted.

## 7. Run-5 recipe — bsp32mse-based bespoke codec (2026-08-22, REGISTERED, NOT YET LAUNCHED)

Ryan-directed design: base the architecture on the robocasa `bsp32mse` recipe (its §16 winner; executable
form `conf/model/bsp32mse.yaml`) — from-scratch conv encoder + U-Net decoder replacing the frozen TAESD
trunk, roundtrip anchor w=10, num_tokens=32 (no TAESD budget constraint), mse U-Net decode,
`flow_arch=transformer` KEPT (Ryan-confirmed; measured to protect a learned codec: -0.55 dB erosion vs
-2.8 dB mlp; consequence: eval_flow's denoising_multistep/aggregate self-skip — accepted),
`latent_norm=layernorm` (forced; affine raises on a learned encoder; -3.5 dB ceiling, non-invertible).

**Decisions taken:** backbone depth 4 (recipe value; small dataset, deficits proven non-capacity,
one-mechanism-per-arm — the codec swap is this run's mechanism; depth 6 = follow-up arm, +0.53M).
Proprio stays 1 token (Ryan asked whether 138-D needs more: PCA says a 128-D linear code loses 0.011
z-units; run-3 proprio was the best component through 1 token; VectorModality is hard-wired to 1 token so
multi-token = code change = its own future arm). Proprio Fourier features ON (+model.modalities.0.
fourier_freqs=16 -> 4,554 encoder input dims; addresses small step-to-step differences). Proprio decode
mse (run-3 finding). `ae_bottleneck=16` (§17 robocasa: the hardcoded 8x8 conv bottleneck was the binding
floor constraint; 16 won PSNR on fewer params; known trade: -LPIPS, acceptable — our gates are
moment-based). Deviations from the robocasa recipe, both dataset-specific: `action_squash=symlog` kept
(TCAV-amp z-score blowup hazard) and `recon_frac=1.0` kept (critic M5: 0.25 randomly drops boundary
frames from decode supervision). `compile_rollout` dropped (dead under p_tf=1).

**Training schedule = the campaign-validated stability recipe, NOT robocasa's**: p_tf=1.0 constant,
`pred_obs_in_loss=false` (bespoke encoder trains via the x10 roundtrip, decoder via roundtrip + detached
recon, flow via velocity matching), lr 1e-4, F=16, window=24, boundary_frac=0.5,
dynamics_detach_encoder=true, v3 dataset. **REGISTERED RISK: 32 jointly-trained image tokens = moving
flow targets — the arm-1f blowup mechanism at scale (robocasa's own tf_bespoke arm hit flow-grad
2.1→65→10,151).** Mitigations: the roundtrip anchor, lr 1e-4, flow-grad monitor with the standing kill
criterion (dynamics/latent rising while flow-grad grows), per-val checkpoint snapshots. Early look
mandatory: ep0 `ae_floor` (bespoke will land FAR below TAESD's 47 dB; robocasa bespoke ~20 dB; if <~30 dB
the extractor-based gates harden — record, judge at the physics gates, don't hard-kill on it alone).

**Architecture @ depth 4 (model_summary, verified): 3.78M params** — image enc (conv) 0.439M, U-Net
decoder (bott16) 1.437M, proprio Fourier-MLP enc 0.300M, backbone 1.063M, transformer flow denoiser
(depth 2) 0.485M, proprio mse decode 0.034M; bag = 33 state + 1 action = 34 tokens; dynamics share 41%
(vs robocasa's 24% — the non-square frames + bott16 shrink the decoder; §17's rebalance critique mostly
moot here).

**Launch command (stage 3; run when ready):**
```
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run --no-sync python -m quickdraw.train_world_model \
    experiment=xtcav_e300_r5 \
    data.root=logs/recording_2026_08_20_15_20_44_xtcav_e300 data.repo_id=xtcav_e300 data.cam=dtotr2 \
    data.F=16 data.boundary_frac=0.5 \
    environments=recorded environments.obs_dim=138 environments.action_dim=3 \
    'environments.position_idx=null' \
    model=bsp32mse model.depth=4 model.action_dim=3 \
    model.modalities.0.dim=138 model.modalities.0.decode_kind=mse \
    '+model.modalities.0.fourier_freqs=16' \
    'model.modalities.1.img_size=[96,448]' '+model.modalities.1.ae_bottleneck=16' \
    model.window=24 model.action_squash=symlog model.recon_frac=1.0 \
    model.compile_rollout=false \
    model.p_tf_start=1.0 model.p_tf_end=1.0 model.pred_obs_in_loss=false \
    model.dynamics_detach_encoder=true optim.lr=1e-4 \
    eval.during_train.evals.control=false \
    '+run_summary.problem="the champion TAESD codec cannot be trained on-domain and its samples do not commit; robocasa evidence says a bespoke codec is the only configuration whose motion ratio rose with training"' \
    '+run_summary.tried="runs 1-3 (frozen TAESD champion, calibrated but half-slope response and 1-2 percent sample commitment), run-4 probes P1-P3, best-of-K retraction"' \
    '+run_summary.trying="run 5 - bsp32mse bespoke conv/U-Net codec at 96x448 with ae_bottleneck 16, proprio fourier 16, mse decodes, 1 proprio token, depth 4, our stability schedule"' \
    '+run_summary.trying_detail="registered risk: 32 moving image tokens = arm-1f mechanism at scale; roundtrip w10 anchor, lr 1e-4, flow-grad kill criterion, per-val snapshots; ep0 ae_floor early look"' \
    '+run_summary.rationale="on-domain codec with no natural-image prior may commit where TAESD superposes; parameter split here is 41 percent dynamics so the robocasa decoder-heavy critique does not transfer"'
```
Pre-flight before the real run: fast_dev_run of the exact command (+trainer.fast_dev_run=true,
data.autobatch=false data.batch=4, throwaway run_summary), confirm `[loader] boundary enrichment` and the
fourier/mse lines in the arch table, then launch with monitors (flow-grad readout per val, bracket-trick
pkill patterns) and the snapshotter. **Acceptance = the standing pre-registered physics gates (§4-6):
val+eval boundary location vs fair null, variance ratio [0.7,2], response zc/RMS on re-grounded
protocols, plus the §6 width-calibration and charge-conservation commitment gates — the headline question
for run 5 is whether the bespoke codec's samples PASS the joint commitment gate where TAESD's 1-2% did.**

### §7 AMENDED — dataset v4: COM-centered crop (2026-08-24, Ryan-directed, implemented + verified)

**Change:** DTOTR2 frames are now COM-CENTERED 128x384 native crops (yiheng `preprocess_shot` style: COM
on a median-3 + threshold-5 cleaned copy; window cut from the raw frame, zero-padded), stored native,
trained at **64x192** (2x downsample; isotropic 61.0 um/px; 3.5x fewer input pixels than the old 96x448).
The removed position becomes two obs channels (`DTOTR2_COM_X/Y_um`) -> **obs 138 -> 140**; positional
jitter moves to the proprio pathway and the image conditional narrows (direct attack on commitment).

**Window choice, measured:** retention saturates from 384 wide up (p1 94.6% == 512's 95.3%); the safety
criterion is corr(charge kept, L2) — 384 scores -0.01 (no setpoint-correlated clipping) vs +0.12 @320 and
+0.29 @256 (reject: fake knob response); 96 rows doubles the >1%-loss fraction (energy axis has no slack).
Conversion verified: BEAM-charge retention median 99.99%, 9-13% of frames lose >1% (diffuse halo), p1
outliers are near-empty shots (beam sum <1k vs median 58k; geometric-center fallback fires). NOTE the
first conversion's "64% kept" alarm was a METRIC bug (raw-sum retention counts the sub-threshold pedestal,
~2/3 of raw counts over the full frame — discarding it is the point); metric fixed to thresholded beam
charge. **Dataset v4 = `logs/recording_2026_08_24_16_12_29_xtcav_e300`** (13/2/3 episodes, 8412 windows,
check_dataset OK at obs 140). Eval suite rebased (DATA, IMG_SIZE 64x192, PX/ROW 61.0 um) and re-validated:
measured fits invariant vs v3 (zc 4.2-7.1 deg), Tier-0 ratios 1.01-1.09.

**Run-5 architecture at v4 dims (model_summary): 3.53M params** — image enc 0.193M, U-Net dec 1.437M,
proprio Fourier-MLP 0.304M (140-D in), backbone 1.063M, flow 0.485M; dynamics share 44%.

**Updated launch command (supersedes §7's):** identical to §7 except
`data.root=logs/recording_2026_08_24_16_12_29_xtcav_e300`, `environments.obs_dim=140`,
`model.modalities.0.dim=140`, `'model.modalities.1.img_size=[64,192]'`, and experiment name unchanged
(`xtcav_e300_r5`). Everything else (depth 4, transformer flow, fourier 16, bott16, symlog, recon 1.0,
stability schedule, gates incl. the joint commitment gate) stands.

### §7 RUN 5 LAUNCHED + VERIFIER REPORT (2026-08-25)

**Run:** `logs/train_world_model_2026_08_25_15_41_19_xtcav_e300_r5`, GPU 0, batch 37 (new autobatch
linear-fit probe), ~6 min/epoch, boundary enrichment 53%, monitors (flow-grad/dyn/roundtrip per val) +
per-val snapshotter armed. Verifier (adversarial, empirical): **LET RUN CONTINUE — no blockers.**
Verified clean: v4 crops bit-exact vs recomputation from v3 (codec noise only), COM channels no x/y swap
(destd median 0.39 px vs recomputed), obs order correct with first 138 columns BIT-IDENTICAL to v3,
zero-padding correct on 26 edge frames, resolved config == §7 spec key-for-key, fourier/bott16/
pred_obs_in_loss all plumbed (arch table matches 3.53M).

**Verifier findings, actioned:**
- **[MAJOR] Border-flash COM bias (v4 carries it):** transient frame-border flashes (also found
  independently by daq2npz) hold >0.1% of cleaned signal on ~10-22% of frames and shift the COM median
  ~4-5 px (max ~16) — ~2 model-px of residual centering jitter on ~10% of frames + matching noise on the
  COM obs channels; bottom-edge flashes can appear inside crops (beam sits low). Not fatal (corr(kept,L2)
  gate passed with the bias included; transient, not setpoint-locked). FIXED IN CODE for the next
  re-convert (16-px border zeroed in the COM COPY only, processors.py); run-5 results must be read with
  this caveat.
- **[MAJOR] "ep0 ae_floor early look" never scheduled** — `_eval_due` fires at {5,9,15,...} under
  every=10/at=[5,15], and the startup log falsely claimed ep0 eval runs. Log line + docstring FIXED
  (truthful, conditional on _eval_due(0)). First bespoke ae_floor lands at ep5; recorded below when in.
- **[MINOR] 30 beam-off frames (0.25%) train as noise crops** (COM_Y at -4.3 sigma marks them);
  acceptable at this incidence; gate them (daq2npz-style beam-signal gate) in any v5.
- **[MINOR] eval-script staleness** (CKPT default, docstring geometry) FIXED -> r5 best.ckpt.
- **[NOTE]** daq2npz's 227-px half-width "requirement" includes E331 + un-repaired border junk; on the
  beam core our 192 half-width retains median ~100% (indep. check p1 89.9%) and the operative
  no-L2-correlation gate passed. Documented tradeoff, not a bug.

### §7 RUN 5 RESULTS + GATE VERDICTS (2026-08-26)

**Training (COMPLETE, first fully healthy run):** all 50 epochs, zero instability — flow-grad
oscillated 1.3–3.2 with no trend (no spiral; sailed past ep35, run-3's collapse point), dyn/latent held
its 0.24–0.38 floor, and the bespoke ae_floor ROSE 32.3→33.7 dB PSNR through training (robocasa bespoke
arms eroded; roundtrip anchor + on-domain 64×192 crops reversed that here). Built-in rollout monitor
anti-aligned as always (best.ckpt=ep3).

**Checkpoint triage** (boundary n_rep=8 over ep3/15/31/47/49, `r5_triage/`): **ep15 wins** — same
sweet-spot epoch as r3g. Variance ratio 0.72–0.95 at ep15 vs 0.17–0.42 at ep31+ (late training collapses
sample dispersion → over-deterministic) and ep3 under-trained. Snaps needed symlinking into
`checkpoints/` (loader assumes config two levels up — same trick as r3g).

**Champion = `snap_ep15.ckpt`. Full battery (`r5_ep15/`):**
- **Boundary (n_rep=32):** location beats fair null on both E300 runs — medabs_val 122 vs null 305
  (15671), 229 vs 305 (15673); TEST fails (839 vs 427). Variance-ratio gate [0.7,2]: 1.00 / 0.65
  (marginal) / 0.80 → calibration essentially held.
- **Response (staircase, ±1°):** well-sampled rows (n≥48) give 20–43% of measured slope
  (15671−: 40%, 15673−: 20%, TEST−: 43%); sign=+1 rows too thin/unstable to score. Boundary
  sensitivity slopes 0.05–0.12. **Under-response NOT fixed** — at or below r3g's 37–65%.
- **JOINT COMMITMENT GATE (headline, `r5_commit.json`, new `--part commit` in probes script; K=32,
  two-sided conditioning-local [p05,p95] bands on width+peak+total-intensity):**
  control 4% / boundary 0.4% joint (TAESD champion: 1–2%). **Gate FAILS — but the decomposition is
  the run's real finding: per-sample charge non-conservation is FIXED.** Intensity in-band 45–47%
  (was ~2%), median sample/real intensity 0.93–0.94 (was systematically dim). Width in-band 13–26%,
  peak 12–16% → samples now conserve charge but still spread it: "correct-charge blur."
- **Slew (secondary):** fails as expected (in-range RMS 1.3–2.3 mm, extraction ok 9.4% — open-loop
  long-horizon remains physically vacuous and degrades).

**Interpretation:** the codec-level pathology (dim haze / charge loss) is eliminated — the remaining
blur is upstream, in the flow conditional itself (mean-seeking averaging over jitter), now cleanly
attributable to the DYNAMICS head, not the decoder. Run 5 therefore de-confounds §6's re-ranked routes:
the next lever must act on the conditional's entropy allocation — (1) retrieval/local projection onto
real latents, (2) mode-variable/categorical commitment head, (3) jitter-explanation moved to the proprio
pathway so the image conditional narrows. Caveat on all v4 numbers: border-flash COM bias (~2 model-px
on ~10% of frames, fixed in code for v5).

Artifacts: `logs/physics_eval_xtcav_e300/r5_triage/` (ladder), `r5_ep15/` (boundary_score, staircase,
slew, score, boundary_response.png, sep_vs_l2_with_model.png), `r5_commit.json` + `r5_commit_hists.png`.

### §7 DIAGNOSIS: SMEAR ARTIFACT + POLARITY-FLIP CONFUSION (2026-08-26)

Prompted by the boundary-example strips (`r5_ep15/boundary_examples.png`): faint full-width
horizontal bands in model samples, and a suspicion that simultaneous TCAV-polarity flips at L2 step
boundaries confuse the model. Diagnostic: `wizard/scripts/xtcav_r5_polarity_diag.py` →
`r5_ep15/polarity_diag.{json,png}`.

**Dataset fact first:** the DAQ scans polarity within each setpoint block (~40–52-shot single-sign
stretches), so 60/81 of ALL L2 boundaries (74%) and **8/15 gated val+eval boundaries are simultaneous
two-knob changes** (L2 step + sign flip). 302 shot-to-shot flips total (2.5% of steps, 80% mid-block);
36% of F=16 training windows contain one — flip→mirror supervision was abundant.

**Finding 1 — the model provably IGNORES the TCAV-phase action dim.** Real ± images at the same
(run, φ) are x-mirrors (median corr(+,−)=0.12 direct vs 0.55 mirrored). Counterfactual on 13 clean
controls: flipping the action phase leaves the prediction unchanged — corr(flip-sample, true-sample)
= **0.991**; both correlate ~0.73 with the context-sign real mean and ~0.23 with the opposite-sign
mean. The model always continues context polarity. Not a data-sparsity effect (see above); most
plausible cause: an x-mirror is a global token-permutation the flow must gate on one scalar —
representationally awkward, worth only ~2.5% of steps, so SGD settles on ignoring it. This is the
extreme end of the same action-under-response pathology (20–43% L2 slope).

**Finding 2 — the smear/banding is dynamics-side, uniform, NOT polarity-specific.** Banding metric
(full-width row-median baseline / total intensity): real 0.454, **AE roundtrip 0.454 (codec exactly
clean)**, flow samples 0.50–0.51 identically at controls, no-flip and flip boundaries. Visually the
bands sit at the energy rows of the streak TAIL — the band is the tail's jitter range marginalized
along the streak axis instead of committed. Same mean-seeking mechanism as the commitment failure;
polarity flips are NOT the cause of the bands.

**Finding 3 — eval bug (flatters the model):** `part_boundary` tags boundaries with the CONTEXT sign
and scores against that sign's measured curve; at the 8 flip boundaries the realized shot belongs to
the other sign. Rescored with realized-sign medians: E300_15671 flip subset 122→244 µm, E300_15673
flip subset 122→305 µm (no-flip subsets unchanged: 183 / 671 TEST). The model actually matches the
WRONG-sign curve better — direct confirmation it predicts old-polarity beam through the flip. The §7
"sub-null location" margin on E300 mostly evaporates (244–305 vs null 305) once scored correctly.
Staircase/slew response numbers are unaffected (those protocols hold phase constant). TODO: patch
part_boundary to use the sign of the action at t (or report flip/no-flip split).

**Fix routes (for run 6 / v5, not yet implemented):** (a) preferred — **polarity canonicalization in
the processor**: x-mirror all φ− frames to a canonical orientation (negate COM_X accordingly), keep
sign as a scalar; removes the mirror from image space entirely and merges the two per-sign image
distributions (~2× effective density); (b) mirror-equivariance augmentation (flip images + negate
phase + COM_X) to force the action channel to carry polarity; (c) eval-only: score flips against
realized sign and report the flip/no-flip split.

### §7 ADDENDUM: ZERO-POLARITY / SETTLING TAIL + POLARITY FIX PLAN (2026-08-26)

**Zero-polarity facts (converted v4 data):** the processor excises amp<5 and ||φ|−90|>15° shots, so
the physical ramp through φ=0 never appears — each flip is a splice with a hidden multi-second gap
(no time/PID channel exposes it). But a **settling tail survives**: kept-shot phase-dev bands are
<1°: 7151, 1–5°: 4114, **5–15°: 734**, and 59% of the 5–15° shots sit within 6 shots after a flip
(78% of flips have a tail, median 2 shots; first post-flip shot median dev 9.7°). The settle-drop
protects scan-step heads but was never applied to mid-block flips. Consequences: flip-step training
targets are typically HALF-SETTLED states labeled by continuous mid-ramp action values (+80.3 not
+90) — flip supervision is rare, maximal-residual, AND inconsistently labeled. Canonicalization of
images alone would still leave a large bimodal proprio discontinuity (TCAV:LI20:2400:P, dim 129) at
flips, and the ± mirror is only approximate (corr 0.55; per-sign measured zc differ, e.g. 15671
5.85° vs 4.20° — real machine asymmetry).

**Qualitative evidence:** `wizard/scripts/xtcav_r5_polarity_examples.py` →
`r5_ep15/polarity_flip_examples.png`: 5 mid-block flips (settled contexts, val/eval), tiles
[context | realized | x-mirror(realized) | 3× samples w/ real flip action | 2× HOLD-phase
counterfactual]. Model samples reproduce the CONTEXT morphology in every row; real-action and
hold-phase tiles indistinguishable. 158 usable mid-block flips exist (45 val/eval) — enough for a
pre-registered gate population.

**POLARITY FIX PLAN (v5/run 6, pre-registered): — SUPERSEDED 2026-08-26 by "REVISED V5/RUN-6 SPEC"
below (canonicalization demoted to fallback after the S-conditioning discussion).**
1. **Canonicalize images** in the processor: x-mirror all sign(φ)<0 frames, negate COM_X to match.
   One image distribution; no mirror circuit needed; ~2× effective density per morphology.
2. **Fold phase scalars** (obs dim 129 AND action): φ → (sign ∈ {−1,+1}, dev = |φ|−90). Action
   becomes 4-D [L2, sign, dev, amp]. KEEP sign as conditioning — it must carry the real ± asymmetry
   residual the mirror can't (folding it away would widen the conditional, feeding the commitment
   problem).
3. **Extend the settle-drop to polarity flips** (drop dev>5° tail; ~430 shots, 0.4%), so flip
   transitions are settled→settled like scan-step heads.
4. **Eval patches:** (a) part_boundary sign attribution → sign of the action AT t, report
   flip/no-flip split; (b) new pre-registered polarity-response gate on the ~45 held-out mid-block
   flips: flipping the sign action must change the prediction (corr(flip, hold) well below the
   current 0.991) and move it toward the other sign's real mean; in canonical coordinates: flip
   steps must score like no-flip steps.
5. Rationale for data-level over model-level: the flow predicts a RESIDUAL on the previous bag and
   tokens carry absolute patch identity + learned slot embeddings — a mirror is a scalar-gated
   global token permutation, the hardest transform the architecture can express, worth 2.5% of
   steps. Augmentation/equivariance tricks fight the architecture; canonicalization deletes the
   problem class. (Action-dropout/readback-masking against the broader setpoint/readback redundancy
   trap is a SEPARATE, undecided lever.)

### §7 REVISED V5/RUN-6 SPEC (2026-08-26) — supersedes the polarity fix plan above

Driven by the S-continuum reframe: TCAV polarity is not a binary but the sign of a continuous
signed streak strength **S ∝ amp·sin(φ)** (off = 0, ramp = intermediate, full streak = ±S₀).
Canonicalization is demoted to fallback: it implements only S↔−S at |S|=max, bakes in
"streaked near ±90" as a domain assumption, cannot represent TCAV-off, and the mirror is
approximate anyway (corr 0.55).

**1. TCAV representation (v5 processor, action + obs):**
- Action: replace (phase, amp) with **S_next (signed streak strength)**; keep amp only if it ever
  varies when on (it doesn't: ~22 constant). Candidate action = [L2, S] (2-D) or [L2, S, amp].
- Obs: fold the readback dims (TCAV:LI20:2400:A=128, :P=129) the same way → S_readback (+ a
  dev/|S|-settling coordinate so half-settled shots are EXPLAINED, not excised).

**2. Data retention (v5 processor):**
- **Un-drop the off/ramp shots: 2,711 shots (~18% of recorded; 885/922/904 per run) currently
  excised.** They are the S-continuum: S≈0 anchors the unstreaked (betatron-spot) end, ramps
  connect the ± branches through zero — decomposing the mirror into a chain of small residuals,
  exactly what the residual predictor learns well. The disconnected ± clusters were the
  learnability problem.
- Replace the TCAV-state gate with a **beam-presence gate** (TMIT-based; also covers the verifier's
  30 beam-off noise crops). TCAV-off shots WITH beam are now valid training data.
- KEEP the L2 settle-head drop at scan-step boundaries (that one is about L2/klystron settling and
  underpins the "settled response is the 1-step object" eval framing). KEEP strict eval gating
  (‖φ∓90‖<1°) for measured curves.
- Carried v4→v5 items: border-strip COM fix (already in code), COM-vs-L2 correlation check.

**3. Redundancy / duplication mitigations (the setpoint↔readback causal-confusion trap):**
- **Audit + de-duplicate obs channels that copy the action**: TCAV:LI20:2400:A/P are exact
  duplicates (P1 already excluded them); audit for L2-setpoint echoes among klystron/phase
  readbacks. Default: drop exact duplicates from obs; keep genuinely jittered readbacks.
- One-time **channel-semantics audit** of the 138 BSA scalars: flag categorical/binary channels
  (status words, shutters) and circular ones (wrapping phases) — z-scored linear encoding is wrong
  for both.
- Training-time option (run 6): **trained CFG / action dropout** (randomly null the action so an
  unconditional branch exists; guide w>1 at inference — P3 showed inference-only guidance saturates
  without it). Optional stronger lever: **adaLN/FiLM action modulation** in the flow blocks
  (multiplicative leverage; the concat input path is already healthy post-readout-fix).
- Enrichment: extend `boundary_frac` to trigger on **|ΔS| changes as well as L2 changes** in the
  supervised region.

**4. New obs channel: Δt gap.** Add time-since-previous-kept-shot (log-scaled) so splices (dropped
shots, scan-step settle gaps) stop being invisible — directly mitigates the dt fiction at
boundaries and the hidden dead time at ramps.

**5. Eval battery additions (pre-registered):**
- part_boundary sign attribution → sign at t; report flip/no-flip splits (contamination measured
  2026-08-26: flip subsets 122→244/305 µm when scored on the realized sign).
- **Polarity-response gate** on the ~45 val/eval mid-block flips: flipping S's sign in the action
  must change the prediction (corr(flip, hold) well below the current 0.991) and move it toward the
  other sign's real mean.
- **S-smoothness check**: predictions vary monotonically/smoothly along an S sweep at fixed
  context; at S≈0 the model should produce an unstreaked spot (now testable because off shots are
  in-distribution).
- Everything already standing: boundary n=32, staircase response, variance-ratio [0.7,2], joint
  commitment gate (width+peak+intensity), width-calibration gate, fair-null scoring.

**6. Fallbacks / ablations / external:**
- Canonicalization (mirror φ− frames): fallback only if S-conditioning + retained continuum fails.
- STN/warp-structured decoder (x-axis scale/reflect by S; exactly: betatron blur ⊗ S-scaled
  t-density, degenerates correctly at S=0): tier-2 physics-native option.
- `model.predict=absolute`: diagnostic ablation only (removes the copy prior but likely hurts
  small-signal response and 1-step fidelity; the retained S-continuum makes residual an asset).
- Counterfactual-consistency loss: high risk of manufacturing fake response; not planned.
- **DAQ request for next E300/TEST shift: randomized excitation** — per-shot/short-block randomized
  polarity, small PRBS dither on L2 around setpoints, deliberate amp ramps. System-ID persistent
  excitation; breaks the redundancy trap for the whole action vector (also the L2 under-response).
  More passive same-schedule data does NOT help: causal confusion is distribution-, not
  sample-limited.

**Open scope decision for run 6:** this spec is the ACTION-RESPONSE arm; the COMMITMENT levers
(§6 re-ranked routes: retrieval/local latent projection, mode-variable head, jitter→proprio
narrowing) are orthogonal. Decide whether run 6 combines both or isolates the response arm on the
v5 dataset first.

### §8 RUN 6 PROPOSAL (2026-08-26, pre-audit)

**Scope decision: the ACTION-RESPONSE arm in isolation** (commitment levers from §6 deferred to run
7) — the v5 data representation changes are large enough that combining arms would destroy
attribution.

**v5 dataset (new conversion, per §7 revised spec + action-space amendment):**
- **Action = [L2, S, dev, amp] (4-D)** — S = amp·sin(φ) signed streak strength; dev = |φ|−90
  settling coordinate; φ, amp kept via (S, dev, amp) which is a lossless reparameterization of
  (φ, amp). Rationale for keeping raw-equivalent channels: intra-action redundancy is safe (the
  causal-confusion hazard is obs→action echo, not action-internal), and S is a modeled feature —
  raw knobs let the net learn residuals if streak ∝ sin(φ) is imperfect (centroid kick ∝ cos(φ),
  TCAV-induced energy spread ∝ amp separately).
- **Obs = 141-D**: 138 BSA with TCAV:LI20:2400:A/P (dims 128/129) REPLACED by S_readback +
  dev_readback (readbacks are jittered near-duplicates, kept but folded to match action semantics)
  + Δt-gap channel (log-scaled time since previous kept shot) + COM_X + COM_Y.
- **Retention:** beam-presence gate (TMIT) replaces the TCAV-state gate → the ~2,711 off/ramp shots
  (18%) enter training labeled by (S, dev, amp). L2 settle-head drop stays. Border COM fix active.
  Beam-off frames gated. COM-vs-L2 correlation check re-run on the enlarged population.
- Channel-semantics audit (categorical/circular flags) executed BEFORE conversion.

**Training (run 6 config, diff vs run-5 recipe — everything else inherited unchanged:**
bsp32mse, depth 4, d128, transformer flow d2, fourier 16 on proprio, ae_bottleneck 16, window 24,
F=16, 64×192, symlog squash, recon_frac 1.0, p_tf 1.0, pred_obs_in_loss=false,
dynamics_detach_encoder=true, lr 1e-4, 50 ep, GPU 0):
- `environments.action_dim=4`, `model.action_dim=4`, `environments.obs_dim=141`,
  `model.modalities.0.dim=141`.
- **Enrichment extended:** `boundary_frac` triggers on L2 change OR sign(S) change OR |ΔS| above
  threshold in the supervised region (dataset.py change).
- **Trained CFG (action dropout):** null the action embedding with p=0.15 during training (learned
  null token); inference guidance weight w becomes an eval knob (P3 showed inference-only guidance
  saturates without a trained unconditional branch). Code change in multimodal.py act_enc path,
  behind a config flag, default OFF for exact run-5 reproducibility elsewhere.
- adaLN action modulation: NOT in run 6 (second lever only if CFG+representation fails; one lever
  at a time).

**Eval (pre-registered, before training starts):**
- Existing battery (boundary n=32 with FIXED sign attribution + flip/no-flip split, staircase,
  variance ratio [0.7,2], joint commitment gate, width calibration, fair null).
- NEW polarity-response gate: ~45 val/eval mid-block flips; corr(flip-action sample, hold-action
  sample) must drop well below 0.991 and predictions must move toward the realized sign's real
  mean.
- NEW S-smoothness check: S sweep at fixed context → monotone/smooth image response; S≈0 must
  produce an unstreaked spot (in-distribution now).
- CFG guidance-weight sweep on the staircase response slope (does w>1 recover slope INSIDE the
  variance-ratio calibration gate).

**Success criteria (falsifiable):** polarity-response gate passes; staircase response ≥70% of
measured slope on well-sampled rows (vs 20–43% now) at w chosen WITHOUT peeking at test runs;
variance ratio stays in [0.7,2]; joint commitment not worse than run 5 (4%/0.4%).

### §8.1 AUDIT VERDICT: REDESIGN → REDESIGNED RUN-6 SPEC (2026-08-26)

Independent adversarial audit (empirical, against raw DAQ npz + code) returned **REDESIGN**. The two
central data moves of §8 are refuted by measurement; the conditioning/eval/CFG pieces survive.

**Refuted (with the auditor's evidence):**
- **[BLOCKER] The S-continuum premise is false on this data.** ~86% of the 2,711 excised shots are
  full-amplitude phase-slew shots whose S LABELS populate 1<|S|<20 but whose IMAGES do not follow:
  corr(|S|, streak extent) = 0.031 over 103 on-frame excised shots; within L2-constant ramp blocks
  the streak stays flat (rms_x 7–10 px) while S sweeps +19→−20. The cavity field during/after a
  transition is a latent variable not carried by (S, dev, amp) — readback-implied deflection ≥97%
  where the observed streak is down ~70%, including on KEPT half-settled shots. Un-dropping these
  shots would teach the model ~2,200 more times that the action can be ignored. The homotopy
  argument collapses; a real S-continuum can only come from the randomized-excitation DAQ request
  (deliberate amp ramps at locked φ).
- **[BLOCKER] TMIT beam gate is a no-op.** All 2,711 excised shots have healthy TMIT on every
  reliable toroid (ratio 1.000); 11% of excised frames are blank + 22% heavily smeared,
  kick-correlated (beam scraped downstream of the last toroid — TMIT-blind). The verifier's 30
  beam-off kept crops are TMIT-invisible too. Gate must be IMAGE-SIGNAL (thresholded frame charge,
  daq2npz-style), optionally AND TORO:LI20:1988:TMIT.
- **[MAJOR] Readbacks are BIT-EXACT action copies**, not jittered near-duplicates: obs dims 128/129
  and action dims 1/2 read the same scalar columns with obs[t+1] ≡ action[t]. Keeping them (even
  folded) hollows out CFG's unconditional branch.
- **[MAJOR] The "L2 settle-head drop" is itself a TCAV-state test in code** — §8's "keep it, drop
  the TCAV gate" was self-contradictory.
- **[MAJOR] S-smoothness gate ill-posed:** (|S|, dev) support is a 1-D curve + the ±22 clusters
  (corr(|S|, amp·cos dev)=1.0000 on excised full-amp shots); no data at (intermediate S, dev≈0).
- Also: eval scripts hardcode the 3-D action schema; boundary_frac triggers would run on NORMALIZED
  actions (sign(z-S) ≠ sign(S)); Δt gaps live at scan-step boundaries, NOT ramps (flips are
  contiguous 10 Hz; pulseID.SLAC_time exists to compute Δt); dev readback is noise when amp≈0.

**Verified true by the audit:** (S, dev, amp) lossless + wrap-free (a genuine improvement over raw
circular φ); 45 val/eval mid-block flips (14+31); dims 128/129 = TCAV:LI20:2400:A/P; no L2-setpoint
echo elsewhere in obs (TCAV A/P is the only exact echo); ~150 true-off shots (amp<5) are on-screen
full-charge — clean unstreaked S=0 anchors; kept-range S is binary ±22 (frac 2<|S|<20 ≤ 0.001).

**REDESIGNED SPEC (supersedes §8 items it contradicts):**
1. Action = [L2, S, dev, amp] (4-D), with dev zeroed/masked when amp < threshold. UNCHANGED.
2. Obs: **drop** dims 128/129 entirely → 136 BSA + Δt + COM_X + COM_Y = 139 nominal; obs_dim
   pre-registered as DERIVED from conversion output (nan/variance censuses shift), gated by
   check_dataset.
3. Retention: **revert to settled-only supervision** — extend the settle criterion to polarity
   flips (image-consistent streak check, or dev>5° tail drop as in the original §7 ADDENDUM). Add
   ONLY the ~150 image-verified true-off shots as S=0 anchors. Image-signal beam gate (+ TORO:LI20:
   1988:TMIT), kept_frac floor gate on crops. The 2,200 phase-slew shots stay excised.
4. Polarity mechanism: sign bit in the action + **trained CFG with a single per-(sample,step) null
   mask** threaded to BOTH act_enc call sites (_to_input token AND _cond raw concat), learned null
   embedding at act_enc output, nulling only the supervised-transition action; guidance
   v = v_u + w·(v_c − v_u) pre-registered. (Obs echo already removed by item 2, so the
   unconditional branch is genuinely unconditional.) Canonicalization returns as first fallback.
5. Enrichment: sign(S)/|ΔS| triggers computed on DENORMALIZED actions; |ΔS| threshold 1.0
   (kept jitter p50 0.13, ramp steps ≈ 6).
6. Eval: port the whole battery to the 4-D schema + re-validate measured-curve invariance BEFORE
   training. Polarity-response gate with numbers: median corr(flip-sample, hold-sample) < 0.8 over
   the 45 flips AND corr(sample mean, realized-sign settled class mean) > context-sign class mean
   on ≥60% of flips (score vs SETTLED class means). S gate reduced to its two defensible endpoints:
   (S≈0, amp≈0) → unstreaked spot; (S=±22, dev<1°) → correct-sign streak. CFG w chosen by val-only
   staircase sweep subject to val variance-ratio ∈ [0.7,2], frozen, eval reported once;
   "well-sampled rows" fixed at n≥48 up front. Commitment comparison vs run 5 reported nominally
   with the bands-recomputed-on-v5 caveat stated.
7. Success criteria: polarity gate (above) passes; staircase response ≥70% of measured slope on
   n≥48 rows at the frozen w; variance ratio in [0.7,2]; joint commitment nominally ≥ run 5.

### §8.2 TCAV SLEW-SETTLE MEASUREMENT: 35% OF KEPT SHOTS ARE STREAK-SUPPRESSED (2026-08-26)

Follow-up to the audit's finding-2 (delivered deflection decouples from readbacks through
transitions). Measured on all 11,999 kept v4 shots: per-shot streak extent (baseline-subtracted
x-projection RMS) normalized to the settled reference at the same (run, sign, L2-bin) — settled =
dev<1° and >5 shots from any flip. "Suppressed" = norm < 0.6 (the dev<1° population is tight: only
1.8% below the cut, so the threshold separates cleanly from normal jitter).

**Findings:**
- **4,219 kept shots (35.2%) are streak-suppressed** (typically 3–4× reduced extent) at nominal
  actions.
- **Post-flip recovery takes median 14 shots (~1.4 s), p95 = 25** (until 3 consecutive clean) — an
  order of magnitude longer than the 2-shot dev>5° tail previously identified. P(suppressed) stays
  ~78% flat for k=0..5 after a flip while readback dev decays 9.7°→2.0°: dev is a LOCK-STATE
  SYMPTOM, not the deflection physics (cos(2°)=0.9994 predicts no loss; observed ~70%).
- **Intermittent suppression exists away from all knob changes**: 14.2% of shots >15 from any flip
  AND any L2 step are suppressed, in 294 short episodes (median 2, max 19 shots) — an unlabeled
  recurring mechanism (RF re-lock/gating), almost always dev-flagged.
- **Criterion performance** (drop rule vs image ground truth): dev>5° recall 13% (the old §7 plan —
  nearly useless); since-flip≤2 recall 16%; dev>2° recall 63%; **dev>1° recall ~97%, precision
  ~84%** (dev bands 1–2°: 83% suppressed, 2–5°: 88%, 5–15°: 76%; stealth suppressed at dev<1° far
  from flips: only 83 shots, 0.7%). An image-side streak-consistency check is exact by construction
  and catches the stealth 83.
- Eval was always protected: strict_on (dev<1°) keeps measured curves and commitment bands 98%
  clean. TRAINING was not: the model's conditional saw the full mix.

**Interpretation — this connects the polarity bug to the commitment failure:** at nominal (±90,
amp 22) conditioning, training targets are bimodal (settled full streak vs 3×-suppressed) for 35%
of shots. The flag that separates the modes is the phase channel (dev) — the very channel the model
provably ignores. Ignoring TCAV phase therefore costs the model the ability to explain a third of
its training frames, which legitimately widens the image conditional → mean-seeking width smear.
Making the model read the TCAV action (CFG) and/or removing suppressed shots are BOTH commitment
levers, not just polarity levers.

**Spec change (amends §8.1 item 3):** the flip/settle criterion is **dev>1° drop** (simple,
label-only, 97/84) **plus the image streak-consistency check** to catch the 83 stealth shots and
audit the 16% false-drop rate. Expected cost: ~4.8k shots (40%) → ~7.2k training shots (~5k windows
at F=16) — acceptable, and the removed shots were actively training the wrong conditional. The
commitment-gate real bands stay dev<1° (unchanged). Revisit keeping suppressed shots WITH dev
conditioning only after CFG demonstrates the model actually reads the action.

### §8.3 BSA CHANNEL AUDIT (partial execution of §8.1 item, 2026-08-26)

Scanned all 138 obs channels for action echoes (|corr| vs L2/sign/amp/dev within train episodes),
clock correlation, and cardinality.

- **Echoes to drop: only dims 128/129** (TCAV:LI20:2400:A corr 1.00 w/ amp; :P corr 1.00 w/ sign) —
  confirmed the only exact copies; already removed in the v5 spec.
- **Physics responders — KEEP, do not confuse with echoes:** BLEN:LI14:888:BRAW (|r|=0.71 vs L2 —
  bunch length responds to L2 chirp via BC14 compression), LI14 BPM X (dims 89/92, r≈0.62 —
  dispersive orbit response to L2 energy change), LI20 BPM X (dims 116/119/122, r≈0.5–0.6 vs TCAV
  SIGN — residual centroid kick flips with polarity), DTOTR2_COM_X (r=0.43 vs sign, same physics).
  These are downstream beam consequences measured at context times — legitimate state the surrogate
  needs (drift/jitter info). They differ from echoes in kind: causally lagged, noisy, beam-derived.
  Note they do give the model context-time polarity redundantly with the images, but removing them
  would not change the shortcut structure (the images carry it anyway); CFG nulls the ACTION, which
  is the correct intervention point.
- **Categorical finding:** WIRE:LI20:3179:POSN (dim 131) is binary (wire parked/inserted) with zero
  within-episode variance → dead weight or run fingerprint; drop in v5. Validity bits 135–137 are
  by-design binary; keep.
- **Clock note:** the L2-responders necessarily correlate with shot index (0.5–0.7) because the
  scan ramps L2 monotonically — schedule leakage rides on real channels; nothing droppable, broken
  only by the randomized-excitation DAQ request.

### §8.4 E331 EXTENSION RECON (2026-08-26)

Inspected all 15 E331 mats (16019–16035) + E300/TEST for the all-data extension. 16026 is dead
(0 matched shots). ~16k additional usable shots across 14 datasets.

**Transfers cleanly (verified):** same scan knob in every dataset (L2_PHASE.MKB → action semantics
unchanged); same DTOTR2 camera, 30.5 µm/px, same rotation; same BSA families (TCAV:2400:A/P, WIRE,
masked-band members present); E331 is SINGLE-POLARITY per dataset with 99% of shots at dev<1° —
essentially no TCAV settle/suppression contamination (§8.2 drop costs ~1% there).

**Changes to anticipate:**
1. **Scan range ±15° (half the datasets NEGATIVE) vs E300's [0,8]** — linear-chirp theory fit and
   zero-crossing framing won't hold over the full range (over-compression territory); refit
   measured references per branch/range. Big response-learning win: 2–4× excitation range.
2. **Per-dataset ROIs** (894×184 / 830×256 / 786×238 / 826×208 oriented) — COM-crop absorbs this by
   design, but E331 has 238–256 energy rows vs E300's 184 and daq2npz flags E331 beams wider; the
   128-row energy crop + energy walk at ±15° (~3–4% ΔE) must pass per-dataset kept_frac +
   corr(kept, L2) gates; be prepared to enlarge the crop for all datasets (one shared geometry).
3. **Block structure 61 steps × 20 shots** (vs 33×150): episode/blocking + val/eval split logic must
   generalize; boundary density is ~7× higher (≈854 new L2 boundaries vs E300's 81) — a major gain
   in step-response supervision.
4. **Polarity:** E331 adds ZERO flip supervision (sign constant per dataset, context-inferable);
   polarity gates stay E300/TEST-based; between-dataset sign variation is confounded with run ID.
5. **Cross-shift drift:** 17 runs over different machine epochs — per-run standardization handles
   offsets, but consider run/working-point conditioning or shift differences fold into conditional
   width. Normalization + channel censuses recomputed over the union (obs_dim stays derived).
6. **nonBSA klystron lists (LI11–19 PDES/ADES/SBST_PDES) exist in the mats** — these are SETPOINTS
   that echo the L2 scan directly. Obs must remain BSA-only, or any nonBSA addition must exclude
   *_PDES/*_ADES (action-echo audit applies).
7. Extractor validation (part_validate-style vs Tier-0) must re-run per experiment before trusting
   separation evals on E331 morphology; crop-extent probe inconclusive (border junk) — defer to the
   conversion gates.

### §8.5 RUN 6 BUILT + VERIFIER REPORT + LAUNCH (2026-08-26)

**v5 dataset:** `logs/recording_2026_08_26_12_10_49_xtcav_e300` — 13/2/3 episodes, 4900 train windows,
obs 138-D (de-echoed + DT_LOG1P_S + COM), action 4-D [L2, S, dev, amp]. Gate landed on §8.2's
predictions: kept_on 6958 + 158 off anchors; drop_dev ~5.3k, drop_nobeam ~1.0k (image-signal gate),
drop_suppressed ~80 (streak-consistency). Measured-curve invariance vs v4: 195/195 bins, median Δ=0
(only 1-px quantization diffs); extractor re-validated vs Tier-0 (ratio 1.01–1.09).

**Code (all verified):** processors.py v5 gate + 4-D actions + dt + cross-run PV intersection (E331-
ready); dataset.py sign/|ΔS| enrichment triggers on DENORMALIZED values + FIXED duplication formula
(old one over-duplicated when natural coverage ≥ target — v5 natural coverage is 60%, so k=0);
multimodal.py trained-CFG action dropout (p flag, learned act_null, ONE mask per (sample,step) at
BOTH injection sites, dynamics loss only, bit-identical off); eval battery ported to 4-D schema
(schema-aware shot_knobs, knobs_to_action with off-mask, part_boundary realized-sign fix +
flip/no-flip split).

**Independent verifier: FIX-THEN-LAUNCH.** Verified correct empirically: action rows match raw mats
(≤2e-3), t+1 alignment exact vs SLAC_time, S round-trip 2e-6, suppressed fraction in kept data
0.42% (was 35%), CFG masks bit-equal at both sites + eval bit-identical, enrichment k=0 confirmed,
E331_16019 BSA lists byte-identical to E300 (intersection path safe — crashes, never misaligns).
Fixes applied: (1) load_checkpoint act_null migration (pre-v5 ckpts load again, zero-seeded = inert);
(2) part_boundary now EXCLUDES boundaries whose realized next shot is TCAV-off (extractor fires
spuriously on unstreaked spots — 4/16 gate boundaries were poisoned); (3) NEW
wizard/scripts/xtcav_r6_gates.py — the pre-registered POLARITY-RESPONSE GATE (median corr(flip,hold)
< 0.8 AND realized-sign class-mean wins ≥ 60%) + the S=0 OFF-ENDPOINT gate, both with a CFG guidance
path (v = v_u + w·(v_c − v_u) via the trained act_null; --w swept on VAL only, then frozen);
(4) knobs_to_action off-mask. **Re-registered flip-gate population on v5: 25 (7 val + 18 eval)**
(strict-context gathering; the verifier's 34 counted kept-adjacent pairs without context gates).
Verifier notes logged: CFG nulled actions also null context steps within the TF window (standard
sequence-CFG; accepted); E331 traps — exclude dead 16026, eval RUNS/BLOCKS_PER_RUN hardcodes need
porting before E331 evals; mean_on estimator assumes on-dominant runs.

**Launch (GPU 0):** run 5 recipe carried unchanged (bsp32mse, depth 4, 64x192, bott16, fourier-16
proprio, window 24, F=16, symlog, recon 1.0, p_tf 1.0, pred_obs_in_loss=false, detach-encoder,
lr 1e-4, 50 ep) + v5 data + action_dim 4 + obs 138 + `model.diffusion.action_dropout=0.15` +
`data.boundary_sign_dim=1 data.boundary_delta=1.0` (natural coverage 60% → no duplication).
Smoke (fast_dev_run): params 3.53M (= run 5), act_null gradient flowing, proprio fourier on.
**Acceptance (§8 as amended):** polarity gate on the 25 flips; staircase response ≥70% at w frozen
from a val-only sweep; variance ratio [0.7,2]; boundary location vs fair null (off-next excluded,
flip/no-flip split); S=0 off-endpoint; joint commitment nominally ≥ run 5's 4%/0.4% (v5-bands caveat).

### §8.6 RUN 6 RESULTS + GATE VERDICTS (2026-08-26)

Training: all 50 epochs healthy (flow-grad 1.6–3.4 no trend, dyn/latent 0.24–0.33, ae_floor
31.0→32.6 dB despite 40% less data; one act_null grad spike at ep47, benign). Triage: ep15 is the
BAD rung this time (samples unextractable); ep47≈ep49 lead; **champion = ep49 (last.ckpt)**.
Fair nulls TIGHTENED on v5 (92/183/183 µm vs v4's 305/427) — settled-only data has less jitter.

**THE HEADLINE WIN — L2 response gate PASSES (2/3 rows):** staircase slopes vs measured, well-
sampled rows (n≥48): E300_15673− **88%**, TEST_15668− **95%**, E300_15671+ 59% (gate ≥70%).
Run 5 was 20–43%, r3g 37–65%. At w=1 (no guidance), so the gain is the V5 DATA CLEANING, not CFG:
removing the 35% streak-suppressed transients removed the unexplained variance that diluted the
action signal. The campaign's longest-standing failure (action under-response on the continuous
knob) is substantially fixed by supervision hygiene alone.

**Boundary location (n_rep=8 ladders, realized-sign + off-next exclusion):** no-flip subsets at or
below null on both E300 runs (61–122 µm); flip subsets remain bad (except sporadic); TEST fails at
every rung as always. Variance ratio at ep49: 0.42/0.75/0.50 — over-deterministic, calibration gate
only passes on 15673 (regression vs run 5's 0.65–1.0; plausibly the same mechanism that lifted the
slopes — a tighter conditional — traded away dispersion).

**FAILED GATES — the TCAV action channel is still unread:**
- Polarity gate: corr(flip-sample, hold-sample) = 0.985–0.995 at EVERY guidance weight
  w ∈ {1, 1.5, 2, 3, 5}; realized-sign wins ≤11% (val 0%). Guidance is mechanically live (small
  monotone drift at w=5) but (v_c − v_u) simply does not carry polarity: **the trained
  unconditional branch is nearly identical to the conditional** — nulling the action creates no
  pressure while the 138-D context readbacks + 8 context frames still pin the prediction.
- S=0 off-endpoint: unstreaked fraction 0% (9 contexts) — off actions ignored too.
- Joint commitment: 0% joint, intensity in-band 0–9%, sample/real intensity 0.79 (run 5: 0.93 —
  but v5 bands are tighter settled-only; nominal regression with the pre-registered caveat.
  Candidate causes: CFG dropout cost 15% of conditional supervision, and/or the sharper conditional
  redistributed error into dimming; needs a no-CFG ablation to attribute).

**Verdict:** v5 data hygiene = big win (response), CFG-on-actions = clean negative (polarity).
The refutation is informative: action-nulling cannot make the action necessary when the CONTEXT
carries the shortcut — the run-7 lever is CONTEXT-side (context/readback dropout, shortened or
perturbed image context) and/or the §6 commitment routes. A no-CFG v5 arm (run 6b: identical config,
action_dropout=0) would cleanly attribute the commitment regression and costs ~2 h.
Artifacts: r6_triage/ (ladders, polarity w-sweep, commits), r6_ep49/ (staircase, off gate).

### §8.7 POLARITY-FLIP LEARNABILITY PROBES: THE NATIVE MECHANISM ISOLATED (2026-08-26)

Two independent probe agents on the r6 champion (ep49), scripts/results in session scratchpad
(probeA_*, probeB_*). Question: what is NECESSARY for the model to learn flips NATIVELY from the
real flip transitions (no mirroring/canonicalization)?

**Probe A (action pathway + loss economics):**
- The S signal SURVIVES to the flow head at full strength: sign flip = the largest embedding change
  act_enc ever emits (‖Δemb‖/‖emb‖ = 1.26, 37× the L2±0.25° change the model provably uses), and
  >100% change on the raw-action cond channel.
- It dies in the head's USE: the velocity net applies a uniform weak ~10% LINEAR transfer gain to
  the action channel — identical for S and L2 (0.10 vs 0.10). Enough for L2's small physical
  residual; a ~10× shortfall vs the mirror residual (flip targets 8.9× non-flip).
- Copy basin confirmed: the model captures only ~9% of the flip residual, paying 91% of the
  predict-zero penalty at every flip, all run. Flips = 1.1% of steps, ~3.1% of loss mass.
- **The r6 sign enrichment was a NO-OP**: natural boundary coverage on v5 is 76% > the 50% target
  → duplication factor k = 0; flip windows received ZERO extra weight. (And window duplication
  dilutes 15:1 per-step anyway.)
- Why CFG was inert, mechanistically: guidance amplifies (v_c − v_u), but both branches run through
  the same weak linear action gain — there is no flip mode to amplify.

**Probe B (latent geometry + offline learnability — DECISIVE):**
- Polarity is a dominant, near-linear latent direction (sign logistic 0.998 on val; flip jumps 3.4×
  the within-setpoint scatter; realized flip shots land at the opposite-sign centroid).
- Fresh 2-layer heads trained OFFLINE on frozen (prev_bag, action)→residual, evaluated on the 25
  val+eval strict flips vs a copy baseline: per-token MLP 0.68, transformer 0.71, transformer with
  flips at 30% loss mass **0.55** (val-only 0.216, cos 0.886); action-ablation: real beats hold on
  **25/25** — the heads genuinely read sign(S). Cost on normal steps ~3%.
- A static mirror template FAILS (oracle mirror-cluster ratio 1.5, cos ≈ 0): flip residuals are
  STATE-DEPENDENT — any fix must flow through the conditional pathway, and hard-coded mirror
  operators would not have worked.
- Interpretation cell: **representation + data are SUFFICIENT (59 train flips! same architecture
  class!) — the blocker is purely joint-training gradient allocation.**
- Next ceiling flagged: within-run drift on eval tails (sign decode 0.998 val → 0.65 eval tails).

**ANSWER — what is necessary (and NOT):**
NOT necessary: mirror augmentation, canonicalization, equivariant codec, more flip data, input
encoding changes, CFG/guidance. NECESSARY (and per Head C, likely sufficient): **concentrated
per-STEP gradient on flip transitions in the dynamics loss** — importance-weight steps whose action
flips sign(S) so flips carry ~25–30% of dynamics loss mass (window-level enrichment mathematically
cannot deliver this: no-op at 76% natural coverage + 15:1 in-window dilution). Also remove the
dilution: action_dropout=0 for the flip arm (CFG nulls 15% of the already-rare flip actions and
demonstrably buys nothing).

**RUN 7 (native-flip arm) SPEC:** v5 data UNCHANGED (no flipping anywhere); loss_terms gains a
per-step weight vector — steps with sign(S_t) != sign(S_{t-1}) get weight w such that flips ≈ 25–30%
of the dynamics loss mass (w ≈ 30–40 given 1.1% frequency; normalize to keep total loss scale);
action_dropout=0; everything else = run-6 config. Pre-registered predictions from Head C: val-flip
residual MSE ≈ 0.2–0.5× copy → polarity gate corr(flip,hold) drops well below 0.8; L2 response and
non-flip metrics unchanged (offline cost ~3%); eval-tail flips remain partial (drift ceiling).

### §8.8 RUN 7 + RUN 8 (E331-COMBINED) BUILT, AUDITED, LAUNCHED (2026-08-26)

**Run 7 implementation (per §8.7 spec):** per-step flip importance weighting in the dynamics loss —
flow.py FlowField.loss gained optional per-lead-position `weights` (weighted mean normalized by Σw,
broadcast over the token axis, bit-identical when None/all-ones — numerically verified vs F.mse_loss
on the real (B,23,33) shapes); loss_terms flags steps whose action crosses the z-image of physical
S=0 (flip_zero_z from the dataset normalizer, 0.05-z dead-band = 1.09 S-units excludes off anchors)
with weight flip_loss_weight; config model.diffusion.{flip_loss_weight,flip_dim}. Detection verified
77/77 against physical ground truth, 0 false positives (77 flags = 59 strict flips + settled block-
splice sign changes, which the audit argues are equally valid supervision). action_dropout=0 (CFG
removed — it diluted flip supervision and act_null stays zero/unused).

**Audit (independent): FIX-THEN-LAUNCH.** Verified: inert-by-default bit-identity, weighted-mean
math + axis placement, step semantics (weights exactly the transition containing the sign swing),
flip_zero_z units, config merge, k=0 enrichment non-interaction, act_null grad-None safe. Actioned:
fresh run_summary (blocker), yaml mass numbers corrected to 51%/26% at the detector's f=1.48%,
train-curve incomparability documented. Registered protocol change: **run 7 gates score at w=1
only** (act_null untrained → the val w-sweep clause is void for this run). Watch item: initial ~51%
flip mass — monitor grad/norm/flow early. Coverage note: flips at window position 0 unweighted
(~4.3% coverage loss, acceptable).

**Run 8 = the same recipe on the E331-COMBINED dataset** `logs/recording_2026_08_26_20_15_05_xtcav_all`
(17 runs, 16026 excluded): 139/16/17 episodes, 16,430 train windows, obs 138-D (census stable:
TORO:LI14:890 nan_frac 0.904 → still dropped), action 4-D. Per-run v5 gates: E300/TEST as before;
E331 on-runs keep 98–99.5% (dev-drop ~0.5%, confirming §8.4's clean-lock finding); **E331_16035 is
a genuine TCAV-OFF L2 scan (amp ≈ 0.02 on 100% of shots)** — its 1,189 shots enter as S=0 anchors
spanning the full L2 range, a large clean population for the off-endpoint gate. E331 contributes no
polarity flips (single-sign runs); all flip supervision remains E300/TEST.

Launch: run 7 GPU 0, run 8 GPU 1, both bsp32mse run-6 recipe + flip_loss_weight=25 +
action_dropout=0; monitors + snapshotters armed. Pre-registered predictions (§8.7) apply to run 7;
run 8 additionally tests cross-experiment generalization (17-run drift, ±15° L2 range — measured
references must be refit per §8.4 before gate scoring on E331 runs).

### §8.9 RUN 7 FIRST VERDICT + EXTENSION TO 100 EPOCHS (2026-08-26)

Run 7 (flip weight 25, 50 ep) — the polarity gate moved FOR THE FIRST TIME in the campaign:
corr(flip,hold) 0.993 (ep31) → 0.931 (ep47) → **0.926 (ep49)**, monotone and still falling at
training end (every prior arm: 0.985–0.995 at all guidance weights). realized-sign wins still 4–8%
→ gate FAILS but the mechanism engaged. Side effects at ep49: variance ratio IMPROVED to
0.67/1.10/0.80 (r6: 0.42–0.75); L2 response MIXED — 15673− held 94%, TEST− dropped 95→54%,
boundary location slipped (15671 medabs 183→214 vs null 92) — consistent with the audited transient
~51% flip loss-mass diverting capacity early (asymptote 26% as flips converge).

**Decision: RESUME run 7 to 100 epochs** (+resume=<last.ckpt> trainer.max_epochs=100, ~2 h) — the
cheapest test of whether the still-falling flip corr keeps closing toward the <0.8 gate while the
decaying flip mass share lets the response rows recover. If the trend saturates above the gate,
next levers: longer schedule from scratch, w tuning, or flip-loss curriculum. (Note: the r7
flip-capture measurement was NOT taken — the probe script ran on its hardcoded r6 ckpt; re-measure
on the 100-ep model with an explicit ckpt.) Run 8 (E331-combined) still training on GPU 1.

### §8.10 RUN 7 EXTENSION: POLARITY GATE PASSES — AND THE FRONTIER IT EXPOSES (2026-08-27)

**THE POLARITY GATE PASSES FOR THE FIRST TIME — natively, from real flips, no data mirroring:**
corr(flip,hold) trajectory 0.926 (ep49) → 0.712 (ep59) → 0.568 (ep79) → **0.410 (ep95, realized
wins 60% — GATE PASS)** → 0.442/68% (ep99). Exactly the §8.7 probe prediction: per-step loss
allocation was the necessary and sufficient lever.

**But the extension exposed a polarity↔response FRONTIER** (constant w=25, 100 ep, 4.9k windows;
val loss drifted 1.41→1.89 over the extension):
ep49 pol 0.93 / resp 55–94% / vr 0.67–1.10 → ep59 pol 0.71 / resp 48–91% / vr 0.71–0.75(0.40) →
ep79 0.57 / 38–72% → ep95 0.41✓ / 20–35% / vr 0.33–0.50 → ep99 0.44✓ / 18–42%. Monotone trade:
as the flip circuit strengthens the continuous L2 response and dispersion calibration decay
(capacity reallocation + overfitting at sustained ~26% flip mass on a small dataset). ep59 is the
knee; NO single rung passes both the polarity and response gates. Off-endpoint gate: unstreaked
frac 7–9% at ep95/99 (up from 0% — the off action is beginning to register too).

**Two-phase anneal probe (launched):** resume from ep99 (flip circuit fully built) with
flip_loss_weight=3 for +30 epochs (to ep130). Hypothesis: circuit MAINTENANCE is far cheaper than
construction (at w=3 flips still carry ~4% mass), so response/calibration recover while polarity
holds. If it works → the run-7 final recipe is a weight SCHEDULE (build w=25 ~60–100 ep, maintain
w≈3); if polarity collapses at low w → the objectives genuinely compete at this capacity and the
levers are model size or separate flip-specialized phases.

### §8.11 ANNEAL RESULT: TWO-PHASE SCHEDULE VALIDATED (2026-08-27)

Maintenance phase (ep99→129 at w=3): **polarity gate HELD and sharpened** — corr(flip,hold) 0.399
(ep115, wins 68%) → 0.349 (ep129, wins 64%), both GATE PASS. Circuit maintenance is indeed far
cheaper than construction (~4% loss mass suffices once built). Response PARTIALLY recovered:
ep115 15671+ 79%, TEST− 62%; ep129 TEST− 77% (from 20–35% at ep95) — trend positive but not yet at
the ≥70%-on-all-rows bar; vr still 0.33–0.60. Maintenance extended to ep160 (anneal2) to complete
the recovery test.

**Qualitative milestone** (`r7_ep129/polarity_flip_examples.png`): per-flip correlations INVERTED
for the first time — corr(model, realized) > corr(model, x-mirror-of-realized) on 4/5 examples
(clean rows 0.39 vs 0.01 and 0.52 vs 0.10; runs 5/6 read ~0.03 vs ~0.4, i.e. the model used to
match the MIRROR of the truth). The model now predicts the actual flipped beam natively.

**Emerging final recipe:** flip-weight SCHEDULE — build w=25 for ~100 ep, maintain w≈3 thereafter.
Open at this writing: full response/calibration recovery depth (ep160 scoring pending), and run 8's
(E331-combined) battery.

### §8.12 RUN 8 (E331-COMBINED) VERDICT: SCALE WITHOUT REBALANCING REGRESSES (2026-08-27)

Eval port: NEW wizard/scripts/xtcav_all_eval.py (E300 scripts untouched) — deterministic episode→run
mapping reconstructed from per-run n_steps + run-major emission + builder rng(0) split, content-
verified 172/172 episodes; measured references recomputed for all 17 runs (E331 medians only, no
linear fit at ±15°); model loaded with its own xtcav_all stats. Outputs: r8_ep49/.

**Run 8 ep49 vs its parents (E300-subset gates):** boundary medabs 229/183/1037 µm (nulls 92/183/183),
vr 0.33/0.90/0.33; staircase response 17–47% (r6: 59–95%); **polarity corr(flip,hold) 0.997, wins 0%
— the flip circuit never engaged** (r7 at the same epoch count: 0.926); **S=0 off gate 6.1%
unstreaked — the 1,189 off anchors did NOT teach the off endpoint**; E331 1-step corr ≤ persistence
baseline on negative-scan runs (0.55 vs 0.76). Broad regression.

**Why (diagnosis):** the combined data DILUTED the polarity mechanism exactly as §8.4 warned —
E300's 77 flips are the only polarity supervision and now sit at ~0.2% of ~35k transitions (flip
loss mass at w=25 ≈ 14%, below run 7's build-phase 51%), while 14 single-sign E331 runs strengthen
the context-polarity shortcut. Also 50 epochs on 3.3× windows ≠ run 7's effective schedule (which
needed ~95 epochs + anneal). Lesson: flip weight must be set from the DATASET's flip fraction
(rebalance to the same target mass, here w≈80–100), or the flip phase must run on the E300 subset
with E331 mixed in afterward (curriculum). Caveat: ep49-only scoring; the snaps ladder can be
re-scored with the same script.

**Open items for the next session:** anneal2 (r7 maintenance to ep160) scoring — the two-phase
schedule's final joint numbers; commitment gate on r7's champion; decision on run 9 (combined data
with rebalanced flip mass + longer schedule, or E300-champion + E331 fine-tune).

### §8.13 MAINTENANCE COLLAPSE AT EP159 + RUN-7 CHAMPION DESIGNATION (2026-08-27)

The second maintenance stretch (ep129→159 at w=3) COLLAPSED the flip circuit: polarity
corr(flip,hold) 0.349 (ep129, PASS) → **0.997 (ep159, fully reverted to the copy shortcut)**, while
response only partially recovered (53/44/63%, vr 0.25–0.85). §8.11's "maintenance validated" is
hereby CORRECTED: w=3 (~4% flip mass) holds the circuit for ~30 epochs but erodes it on a 30–60
epoch horizon — below the true maintenance threshold. (No snapshots exist in ep130–158 — the
anneal2 snapshotter was not armed — so the collapse cannot be localized; ep107–127 snaps survive.)

**Run-7 champion: snap_ep115** (the best JOINT checkpoint of the campaign): polarity 0.399 /
realized wins 68% (GATE PASS), response 79/52/62% best rows, vr 0.33–0.50. ep127 (unscored,
adjacent to the last passing score) is the backup.

**Revised schedule understanding:** build w=25 (~ep60–100) → the circuit forms; maintenance needs
either a HIGHER weight (est. w≈8–12, flip mass ~10–15%) or EARLY STOPPING at the joint knee
(~ep110–130). The polarity circuit is a fragile minority mode: it decays whenever its loss share
drops near the noise floor — consistent with the §8.7 economics throughout.

**Run-8 diagnostics (figures r8_ep49/):** boundary_examples — best sample fidelity of any run
(35 dB codec; low-L2 4.5 mm separations tracked to ~5%) but real≡hold everywhere and a NEW blocky
background-haze artifact (17-run brightness hedging); polarity_flip_examples — total flip
blindness, model matches the MIRROR of truth on clean rows (0.01/−0.03 vs 0.23/0.36), identical to
run 6. "Longer?" answered in-record: ep49 already = 3.3× run-6 gradient steps at zero flip
engagement — the dilution is structural (flip mass ~14% < build threshold; 14 single-sign E331 runs
feed the shortcut); the indicated continuation is a rebalanced build phase (w≈90 → ~50% initial
flip mass) + anneal, not plain extra epochs.

### §8.14 TWO AGENT STUDIES: AUTO-BALANCED FLIP LOSS + ARTIFACT FORENSICS (2026-08-27)

**A. Composition-invariant flip loss (design memo, agent-verified algebra).** The flip share of
dynamics loss mass m = w·f·r/(w·f·r + (1−f)) reproduces EVERY measured regime (51% build / 14%
stall / 4% collapse / 26% response-damage). w was always a proxy for m; the hand-tuning pain is
that m depends on dataset composition f and phase r. **Winner: per-group balanced loss
L = α·mean(flip) + (1−α)·mean(non-flip)** — deletes f entirely (same shares on E300 and combined),
and SELF-ANNEALS: share starts at α·r/(α·r+1−α) ≈ 28% (build) and pins at exactly α as r→1
(maintenance floor that CANNOT decay — the ep159 collapse becomes structurally impossible).
**α = 0.12** derived from the four measured thresholds; equals w_eff≈9 on E300 (matching §8.13's
independent estimate) and w_eff≈62 on combined (matching §8.12's rebalancing estimate). Rejected
with campaign-specific reasons: group-DRO (chases loss equality against an intrinsically-harder
group, n=77 noise), focal/self-weighted (hard-codes the collapse dynamic: weight ∝ loss decays as
the circuit learns), balanced sampling (mathematically capped at 11.6% share < 26% build threshold),
gradient surgery (allocation was the problem, not conflict geometry). Runner-up escalation: ratio-
constrained Lagrangian (L_flip ≤ c·L_nonflip, dual ascent) if the α window proves empty.
Implementation: per-batch w_b = α·n_nf/((1−α)·n_f) through the EXISTING weights arg — no flow.py
change; log per-group means + realized share as collapse early-warning. Pre-registered run-9 arms +
falsifiers in the memo (session scratchpad + this section). Caveat kept honest: capacity competition
is managed, not removed — if no α in [0.10, 0.20] passes both gates, the frontier is real at 3.5M
params and the lever is capacity/architectural separation.

**B. Artifact forensics (probeC) — REVISES §7's codec exoneration.**
- **BLOCKS are CODEC-SIDE**, present in every generation (r6/r7/r8 alike — r8's figures just made
  them conspicuous): the AE roundtrip already contains the full block pattern (block metric: real
  0.0002, AE 0.171, samples 0.156). §7's row-median banding metric was BLIND to half-width blocks
  (it reproduces exactly: banding real 0.4535 / AE 0.4499). Mechanism: the mse-decode U-Net gets all
  its spatial structure from a 2×2 seed (cond_to_spatial) nearest-upsampled to the 16×48 bottleneck
  → 32×96-px half-quadrant plateaus (vertical-edge energy at col 96 = 1.8–2.0× elsewhere; a single
  seed-cell perturbation decodes to exactly the observed rectangle). The decoder is hedging the ~1 u8
  camera noise floor it cannot paint at pixel resolution. NOT 17-run hedging (r6/r7 ≥ r8; E300≈E331
  contexts).
- **STRIPES are FLOW-SIDE**: per-token bag noise renders through the decoder's row-band background
  basis; amplitude ∝ eps (banding 0.446 at eps=0 → 0.506 at 1.5), rows context-fixed (corr 0.52
  across draws). §7's "marginalized tail" reading WEAKENED (stripe rows anti-correlate with dim-row
  maps). Architecture correction: these bespoke-conv checkpoints have NO GridToTokens — tokens are
  Perceiver-query slots with GLOBAL decode footprints, not spatial patches.
- **Mitigations:** stripes — eps=0/low-temperature for committed point predictions (measured
  no-cost), or latent-noise-augmented decode loss in training; blocks — larger/bilinear decoder seed
  (2×2 → 4×12), or zero the ~1 u8 camera floor in the processor. Both artifacts ≤~5 u8 = at the
  extractor's threshold → separation physics largely unaffected; eval hygiene: score sample
  backgrounds against the paired AE roundtrip, not real frames.

### §8.15 RUN 9 PROPOSAL (full combined dataset, pre-review)

**Goal:** one model on the FULL 17-run corpus that passes the polarity gate WITHOUT collapse, holds
L2 response, and removes the two artifact mechanisms — using the composition-invariant loss so no
per-dataset weight tuning survives into future data.

**1. Data (v6 conversion of the 17-run corpus, same gates as v5 plus):**
- Camera-floor zeroing: subtract/zero the ~1 u8 uniform noise floor (pixels < 2 u8 → 0) in the
  STORED crops — removes the background the decoder provably hedges into blocks (§8.14B). Gate:
  per-frame beam-charge change < 1% (the floor is sub-threshold for the extractor by construction).
- Everything else identical to the existing xtcav_all conversion (settled-only, 4-D S-actions,
  de-echoed 138-D obs, dt channel, 16035 off anchors).

**2. Loss (the §8.14A mechanism):**
- Per-group BALANCED dynamics loss via per-batch weights through the existing weights arg:
  α_flip = 0.15 nominal on combined (empty-batch correction: realized ≈ 0.8α at P(no flip)≈20%),
  giving initial share ~34%, pinned floor ~12% — inside the measured [build ≥26%, no-damage ≤26%,
  no-collapse ≥10%] window at the realized level.
- OPEN (reviewer to decide empirically): two groups {flip, rest} vs three {flip, L2-change, rest}
  with α_bnd ≈ 0.10. Discriminating measurement: per-step dynamics loss AT L2-BOUNDARY steps,
  r6-ep49 (response healthy) vs r7-ep95 (response collapsed). Boundary loss flat while response
  fell → capacity damage, three-group buys nothing structural (include only as cheap insurance or
  drop); boundary loss rose → allocation damage, three-group directly indicated.
- Logging: per-group mean losses + realized shares every epoch (collapse early-warning: alarm if
  EMA(L̄_f/L̄_nf) rises >1.5× its post-build minimum). action_dropout = 0.
**3. Decoder seed fix (§8.14B blocks):** replace the 2×2 seed's nearest-upsample with BILINEAR (and
optionally enlarge the seed 2×2 → 4×12 if the change is cheap in vision.py) — removes the hard
col-96 plateau edge. Pre-register an ae_floor comparison vs run 8 (the codec recipe is measured;
any erosion > 1 dB aborts the arch change and falls back to data-floor-zeroing alone).
**4. Eval hygiene:** committed point predictions at eps=0 (stripes measured gone there; calibration
metrics stay stochastic); block/banding metrics scored against the PAIRED AE roundtrip; battery via
xtcav_all_eval (episode→run mapping verified 172/172) + polarity gate (E300 flips) + S=0 off gate +
E331 1-step vs persistence.
**5. Schedule:** max_epochs 150 (~17 h on one A100; flips/epoch = 77 regardless of corpus size, and
run 7 needed ~95 epochs at higher share), per-val snapshots, ladder scoring every ~10 epochs from
ep60; EARLY-STOP rule pre-registered: stop when polarity gate passes AND staircase ≥70% on n≥100
rows at the same rung, or at ep150.
**6. Success criteria (pre-registered):** polarity corr(flip,hold) < 0.8 AND realized wins ≥ 60%,
sustained (no collapse) through end of training — the pinned-share mechanism's headline falsifiable
claim; staircase ≥70% on E300 n≥100 rows; vr [0.7,2]; AE-roundtrip block metric < 0.05 (vs 0.17);
E331 1-step corr > persistence baseline; S=0 off gate improvement over 6%.
**Falsifiers/escalations:** no build by ep150 → α 0.15→0.20; response < 60% at pinned 12% → the
feasible α window is empty at 3.5M params → capacity/architectural separation (§8.10) or the
Lagrangian (§8.14A C6); blocks persist after both fixes → bottleneck cross-attention on tokens.

### §8.16 RUN 9 REVIEW (APPROVE-WITH-CHANGES) + AMENDED FINAL SPEC (2026-08-27)

**Discriminating measurement (reviewer-run, decides §8.15's open item): TWO GROUPS.** Per-step
dynamics loss r6-ep49 vs r7-ep95, paired contexts: boundary-step loss rose 1.24× — EXACTLY the
ordinary-step rise (1.24×; boundary-specific excess = 1.00) — while L2 response fell 3× and flip
loss halved. Response damage is DIFFUSE capacity reallocation, not boundary gradient starvation;
α_bnd would allocate by a non-discriminating signal. DELETED. Corollary recorded: train flow loss
is nearly blind to the response surface (24% uniform rise ↔ 3× response collapse) — response must
be protected by LADDER/GATE scoring, never loss-based early stopping.

**Audit corrections adopted:**
- **α = 0.12 nominal** (not 0.15): the memo's empty-batch correction was wrong — with stride-1
  windows a flip lands in ~21 windows → flip windows are 8% of the pool, P(no flip in batch)=4.8%,
  realized ≈ 0.95α. α=0.12 → initial share ~27%, floor ~13.5% (at measured r=1.20). Escalation
  ladder 0.12→0.15→0.20. Per-batch counts kept (bounds any batch's flip influence at α — the
  anti-collapse guarantee); n_f=0 fallback = plain mean; log per-group means/share/n_f/w_b.
- **Decoder seed change REMOVED from run 9** (deferred, contingent): bilinear is a code change (not
  config) that silently alters every historical checkpoint's decode; 4×12 hard-fails old loads; and
  with the v6 floor zeroed the hedging mechanism loses its target — blocks likely vanish from data
  alone. Becomes a config-gated follow-up ablation only if v6's AE-roundtrip block metric still
  fails (<0.05).
- **v6 gate strengthened:** camera floor ~1 u8/px carries ~35% of pixel mass (sub-threshold for all
  v5 gates and the extractor — structurally safe to zero) but breaks cross-run ae_floor and
  commitment-`tot` comparability; gate = measured-curve BYTE-INVARIANCE vs the v5 xtcav_all
  conversion + extractor re-validation; commitment tot-bands marked v6-internal.
- **Tripwire:** corr(flip,hold) not <0.97 at the ep60 ladder → resume at next α immediately (run 7
  moved by ep47). Schedule ep150 (~16 h at measured 388 s/epoch; per-epoch flip mass ≈ 2× run 7's
  build), pre-registered +50 resume for "trending but unmet". Snapshotter armed across ALL resumes
  = launch blocker (the §8.13 blind gap). Gates at w=1 only. Enrichment config dropped (verified
  k=0; prevents future duplication fighting the α mechanism). Monitoring adds: per-run polarity
  scores + E331-tail sign-decode probe (the shortcut is managed, not removed — flips/epoch on
  combined = 69 train flips, not 77).

### §8.17 RUN 9 IMPLEMENTED + LAUNCHED (2026-08-27)

**Implementation (per §8.16 amended spec):**
- processors.py: XTCAV_FLOOR_ZERO_U8=2 — stored-crop pixels <2 u8 zeroed (v6). Recorded in the
  channels json (v5_gate.floor_zero_u8).
- flow.py: weights docstring precision fix (all-ones = numerically equivalent to None only to float
  summation order ~1e-7; the inert path uses None and IS bit-identical) + `_last_per` side-channel.
- multimodal.py: `flip_loss_alpha` two-group balanced path (per-batch w_b = α·n_nf/((1−α)·n_f)
  through the flow-loss weights arg; n_f=0 → plain mean) + group-loss/realized-share telemetry
  (weight-0 raw entries: dynamics/flip_group_loss, nonflip_group_loss, flip_realized_share).
- setup.py: alpha plumbing; alpha and legacy flip_loss_weight mutually exclusive (assert).
- UNIT-VERIFIED on the real FlowField at run shapes: balanced loss ≡ α·mean_f+(1−α)·mean_nf to 0.0;
  realized share 0.124 at α=0.12; weights=None bit-identical.

**v6 dataset** `logs/recording_2026_08_27_10_08_28_xtcav_all`: gate counts identical to v5 (zeroing
is post-gate); check_dataset OK (139/16/17 eps, 16,430 windows, 138-D/4-D).
**Invariance gate — AMENDED after characterization:** NOT byte-invariant (310/937 bins moved; the
§8.16 reviewer claim that the extractor is blind to the floor was WRONG — energy_gated_sep sums raw
grey, so removing 35% baseline mass shifts marginal extractions). Deviations: p50=0, p90=61 µm
(1 px), >3 px in only 4/937 bins, all marginal E331 zero-crossing bins; gate-scored E300/TEST
subset: max 3 px, 7/185 bins >1 px. ACCEPTED with the documented amendment: v6 references
recomputed and used for ALL run-9 scoring (self-consistent); the v6 curves are arguably cleaner
(charge-fraction denominators de-floored).

**Launch:** `xtcav_all_r9`, GPU 0, 150 epochs (~16 h), α=0.12, action_dropout=0, no enrichment
config (verified k=0 anyway), fresh 5-point note. Watchers: health monitor, resume-proof
snapshotter, ep60 polarity tripwire (corr not <0.97 → resume at α=0.15).

### §8.18 RUN 9 ep59 TRIPWIRE: THE COMPOSITION-INVARIANT MECHANISM ENGAGES (2026-08-27)

**Tripwire does NOT fire — run 9 continues untouched.** ep59 polarity gate (31 flips, v6 data,
w=1): corr(flip,hold) = **0.938** (threshold was "not <0.97 → resume at α=0.15"), realized wins
9.7%. Gate not yet passed (needs <0.8 + 60%), but the circuit is engaging on schedule.

**Direct refutation of the run-8 failure, same corpus:**
| run | corpus | mechanism | epoch | corr(flip,hold) |
|---|---|---|---|---|
| 8 | combined (v5) | constant w=25 (share 14%) | 49 | 0.997 — never engaged |
| 7 | E300 (v5) | constant w=25 (share 51%→26%) | 47 | 0.931 |
| **9** | **combined (v6)** | **balanced α=0.12 (share 20–22%)** | **59** | **0.938** |
Run 9 tracks run 7's build trajectory on a 3.3× larger corpus where the hand-tuned constant weight
produced literally zero engagement — the composition-invariance claim of §8.14A, confirmed live.
(Exposure calibration: run 9 ep59 ≈ run 7 ep52 in weighted flip supervisions.)

**Live share telemetry** (the instrument runs 7/8 lacked; matches m = α·r/(α·r+1−α) to 3 decimals
at every read): ep3 r=1.84 share=0.201 | ep19 r=2.13 0.225 | ep39 r=1.87 0.203 | ep59 r=1.70 0.188.
Share peaked as the copy shortcut formed, now decaying toward the α=0.12 floor as the flip circuit
builds — the designed self-anneal, observed rather than inferred.

**Two operational errors, both caught before contaminating results:**
1. The launch-time tripwire watcher invoked xtcav_all_eval.py WITHOUT a data override; that script's
   module default is the **v5** combined recording, so a v6-trained model would have been scored on
   un-zeroed frames at the campaign's key decision point. ROOT-CAUSE FIXED: `--data` CLI flag added
   (rebuilds RUNS17/mapping); the omission-is-silent hazard is now impossible from the CLI.
2. Killing that watcher hit the documented pkill self-match trap (exit 144, own wrapper shell
   matched). Training run verified unharmed. Use the bracket trick.
Also re-confirmed: snapshot ckpts must be symlinked into `checkpoints/` before eval (_load_model
resolves run_dir two levels up).
