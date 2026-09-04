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

### §8.19 RUN 9 COMPLETE + EXPERIMENT LOG DRAFTED (2026-08-28)

**Run 9 finished (150 ep).** Polarity ladder: ep59 0.938/9.7% → ep79 0.838/32.3% → ep99 0.835/9.7%
→ ep119 0.990/6.5% → ep139 0.862/25.8% → **ep149 (final) 0.995/0%**. OSCILLATES around a marginal
equilibrium; gate never passed; final checkpoint sits at the top of the oscillation. The α=0.12
share (0.19–0.24 measured, matching the closed form to 3 decimals at every read) is below the
consolidation threshold on this corpus, but the restoring force works exactly as designed — share
rises automatically when the circuit breaks (0.241 at ep119, the run's max) — which is why run 9
never suffered run 7's PERMANENT ep159-style collapse. Pre-registered escalation α→0.15 (resume,
not retrain) is the indicated next step and has NOT been run.

**Artifact gate (contingent decoder arm now TRIGGERED):** v6 floor-zeroing cut the codec's painted
background 1.29 → 0.45 counts (real 0.07) but left BLOCK AMPLITUDE unchanged (background block std
1.04 vs run 8's 1.05, 8×24-px blocks over beam-free regions). The data fix alone is insufficient;
per §8.16's trigger the seed-grid change (bilinear / 4×12, config-gated) is indicated as an
isolated ablation on identical v6 data.

**Cross-run figure suite generated** (`wizard/scripts/xtcav_figgen_samples.py`, one model per
process since the eval modules hold global corpus state; matched physical events verified across
corpora — E300_15673 L2 6.25 +→− and TEST_15668 L2 2.25 for flips, E300_15673 6.00→6.25 for the L2
step). Run 5 is excluded from the matched grids (its v4 corpus gates shots differently, so the same
physical events are not present) and appears only in the codec figure. Eight figures in
`lab-notebook/images/2026-08-28_xtcav-wm-r5to9_*`.

**Experiment log drafted:** `lab-notebook/claude/drafts/2026-08-28_xtcav-worldmodel-runs5-9.md`
(runs 5–9; for Ryan to promote). Quantitative claims cross-checked against §§7–8.18.

### §8.20 RESPONSE IN PHYSICAL UNITS + A METHODOLOGICAL CAUTION (2026-08-28)

Added to the runs 5–9 log: the L2 response in µm/deg of bunch separation, and two new figures
(`_sep-vs-l2.png`, `_sep-filmstrip.png`) from `wizard/scripts/xtcav_figgen_sweep.py`.

**Counterfactual slopes (staircase; rows with n≥100 and |measured slope|≥300 µm/deg):** run 5
−123/−297 (20/43%); **run 6 −533/−666 vs measured −604/−704 (88/95%)**; run 7 ep49 94/54%, ep95
20/35%, ep115 52/62%, ep159 44/63%; run 8 −107/−303 (17/44%). Filter rationale: the +90° branches
sit near the separation zero crossing where measured slopes fall to 140–540 µm/deg and ratios become
unstable (the same checkpoint scores +152% and −158% on adjacent working points).

**CAUTION — surface tracking is NOT causal response.** A per-setpoint 1-step sweep (real context
supplied at each setpoint) has every model tracing the measured curve, with full-range chord slopes
−521 to −664 µm/deg vs measured −419 — i.e. models that recover only 17–95% of the CAUSAL slope
appear to over-respond on the surface, because the supplied context already displays the separation
being asked about. The gap between the two measurements is the quantitative signature of the context
shortcut. Gates stay on counterfactuals; curve-matching must never be used for scoring.

**Also fixed while building these:** the first sweep implementation over-extended the validated
staircase protocol (a 100-step open-loop walk vs the campaign's ±1° from a real context) and
extracted on only 3/25 setpoints; rewritten as per-setpoint 1-step predictions it extracts 29/29.
The r9 staircase under xtcav_all_eval needs `--part measured` run into the same --out dir first
(it reads measured.json from there).

### §8.21 RUN 9 FINAL STATUS: DOMINATED — AND THE COMBINED CORPUS ITSELF COSTS RESPONSE (2026-08-28)

**Run 9 complete** (150 ep, both GPUs idle; best.ckpt = ep107, monitor-picked and as always suspect).
Response now scored (the earlier staircase attempt wrote its raw data but failed its slope step for a
missing measured.json in the out dir; scored here against the v6 refs per §8.17):

| rung | polarity corr / wins | response, well-defined rows |
|---|---|---|
| ep79 (best polarity) | 0.838 / 32% | 22%, 52% (median 22%) |
| ep149 (final) | 0.995 / 0% | 17%, 27% (median 22%) |

**Run 9 is DOMINATED on both axes**: run 7 ep115 beats it on polarity (0.399/68% PASS vs 0.838/32%)
AND on response (52/62% vs 22/52%); run 6 beats it on response 4× (88/95%). The mechanism claim of
§8.18 stands (engagement restored where run 8 had none) but run 9 produced no useful checkpoint.

**NEW FINDING — the corpus, not the flip loss, is the response killer.** Response by corpus:
- E300 only: run 6 88/95%, run 7 ep49 94/54%, ep115 52/62%.
- 17-run combined: run 8 17/44% (median 31%), run 9 22/52% at ep79 and 17/27% at ep149 (median 22%
  at BOTH rungs).
Run 8's flip circuit NEVER engaged, so its poor response cannot be flip-capacity competition — the
degradation tracks the corpus itself. Both combined-corpus runs sit at 20–30% while every E300 run
sits at 50–95%. Adding 14 single-polarity E331 runs (3.3× windows, ±15° ranges, per-shift ROIs and
drift) costs roughly a factor of 3 in L2 response at fixed capacity (3.53M params).

**Consequence for the pre-registered escalation.** α→0.15 addresses only the polarity axis and would,
if anything, take more capacity from response. The run-10 decision is therefore NOT simply "raise α":
the candidates are (a) capacity — the §8.14A caveat that the feasible window may be empty at 3.53M
params is now the leading hypothesis, and a width/depth increase is the cheapest test; (b) curriculum
— build both circuits on E300 (where both demonstrably reach useful levels) then adapt to the
combined corpus; (c) per-run conditioning — give the model an explicit run/working-point embedding so
17 shifts stop competing for one shared conditional. Escalating α on the current 3.53M model is
likely to reproduce run 9 with a slightly better polarity number and a worse response.

### §8.22 CORRECTION TO §8.21 — THE RESPONSE DEFICIT IS AN ACTION-NORMALIZATION ARTIFACT (2026-08-28)

§8.21 concluded that the combined corpus "costs a factor of 3 in L2 response at fixed capacity" and
ranked a capacity increase as the first run-10 action. **That diagnosis was WRONG in its mechanism
and the ranking is withdrawn.**

**Measured cause.** The action normalizer's L2 std is corpus-dependent: **1.962 on E300-only
(range 0–8°) vs 7.014 on the combined corpus (range ±15°) — a factor of 3.57.** The model never sees
degrees; it sees z-scored actions, so an identical physical 0.25° step presents as 0.127 normalized
units on E300 and 0.036 on the combined corpus. Probe A (§8.7) established that the flow head applies
a roughly FIXED ~10% linear gain to the action channel, uniform across dimensions — the gain is a
property of the head, not fitted per dimension — so a 3.57× smaller input yields a 3.57× smaller
physical response.

**Decisive test — response re-expressed per NORMALIZED action unit (µm per z-unit):**
| model | E300_15673 −90 | TEST_15668 −90 |
|---|---|---|
| run 6 ep49 (E300) | −1046 | −1307 |
| run 7 ep49 (E300) | −1099 | −734 |
| run 7 ep115 (E300) | −627 | −839 |
| run 8 ep49 (combined) | −751 | **−2127** |
| run 9 ep79 (combined) | −943 | **−2529** |
| run 9 ep149 (combined) | −761 | −1286 |
All six sit in one band (~600–2500) with NO systematic corpus split; on TEST the combined-corpus
models are the STRONGEST. The per-degree deficit (88–95% → 17–27%) is therefore almost entirely the
3.57× normalization change, not capacity, not corpus heterogeneity, not flip-loss competition.
(Naive response-averaging across corpora — E300 343 vs E331 230 µm/deg, shot-weighted blend 263 —
predicts only 77%, i.e. it is a minor term. E331 is genuinely two-bunch with comparable separations
and 0.85–0.99 extraction acceptance, so "different beam topology" is also excluded.)

**Deeper implication.** Every model in this campaign has learned roughly the same NORMALIZED-space
action gain rather than the physically correct µm/deg. The action pathway is under-fitted in a way
that makes physical response inversely proportional to the action-normalization scale — which also
means run-to-run response comparisons are only valid within a fixed corpus normalization.

**Contributing config detail:** `action_fourier_freqs=0` in every run to date (startup log
"[action] squash=symlog | fourier=OFF") while the PROPRIO carries 16 Fourier bands. The action goes
through a plain MLP on a symlog'd scalar, which is exactly the regime where a 3.57× smaller input
gives a 3.57× smaller output; Fourier features would make small normalized deltas linearly separable.

**Revised run-10 priorities (replacing §8.21's):**
1. **Corpus-independent action scaling** — normalize L2 by a FIXED physical scale (~2°) instead of
   the corpus std, so a degree means the same thing in every dataset. Cheapest possible change,
   directly targets the measured cause. Pre-registered prediction: combined-corpus response rises
   ~3.6× toward 60–95%, i.e. into run-6 territory, with no other change.
2. **Action Fourier features** (`action_fourier_freqs=16`) — removes the fixed-gain bottleneck
   itself, and is the run-5 proprio treatment never applied to the action.
3. Capacity / per-run conditioning / curriculum: DEMOTED to after 1 and 2, since the evidence no
   longer implicates capacity.
4. The α escalation remains a separate, polarity-only lever.

### §8.23 METRIC AUDIT AFTER §8.22 — WHAT THE NORMALIZATION FINDING DOES AND DOES NOT INVALIDATE

Per-dimension action-normalization ratios (combined / E300-only) and the shrink applied to each
operative step:
| action dim | std E300 | std combined | operative step | shrink |
|---|---|---|---|---|
| L2 setpoint | 1.962 | 7.014 | 0.25° step | **3.57×** |
| S (polarity) | 21.754 | 21.193 | ±22 → ∓22 flip | 0.97× (none) |
| dev | 0.341 | 0.351 | 1° | 1.03× (none) |
| amp | 3.319 | 4.968 | 1 MV | 1.50× (not a scored knob) |

**IMPLICATED (cross-corpus comparisons only):**
- Staircase response % — the headline case. §8.12's "run 8 17–47% vs run 6 59–95%" and §8.21's
  "combined corpus costs 3× response at fixed capacity" both mis-attributed a normalization artifact
  to the corpus/capacity. WITHIN a fixed corpus the metric remains sound (run 7's ep49→ep95 response
  decay under flip weighting is real; it was measured at constant normalization).
- Boundary location, only mildly: under-response makes the prediction land short of the new setpoint,
  but the whole 0.25° step is worth just ~157 µm, so it can contribute at most ~130 µm of the
  229–1037 µm deviations observed. Not the dominant term there.

**NOT IMPLICATED:**
- **The entire polarity story.** S's std is within 3% across corpora (a flip is 2.02 z-units on E300,
  2.08 combined), so run 8's 0.997, run 9's oscillation, and run 7's gate pass are all genuine and
  directly comparable. This matters: the polarity conclusions are the campaign's main results.
- Variance ratio, commitment gate, codec/ae_floor metrics — no action-scale dependence (they carry
  their own separate v6 caveats).

**PHYSICAL VERDICT UNCHANGED — run 9 really is a worse surrogate.** Per real 0.25° step the machine
moves 157 µm; run 6 moves 133 µm (85%), run 7 ep115 80 µm (51%), run 8 27 µm (17%), run 9 27–34 µm
(17–21%). For anyone operating on these predictions run 9 is ~4× less responsive than run 6. The
normalization finding explains the CAUSE and makes the fix cheap; it does not make the current model
usable. Also note the within-run spread that normalization does NOT explain: run 9's TEST −90 gain is
2529 µm/z at ep79 vs 1286 at ep149, a 2× swing across checkpoints of one run.

### §8.24 PROPRIO-HEAD ACCURACY: BUNCH LENGTH AND BPM READBACKS (2026-08-28)

New probe `wizard/scripts/xtcav_figgen_proprio.py`: 400 held-out strict-gated contexts per model,
1-step prediction, 64 channels (3 real BLEN + masked BLEN + 60 BPM X/Y), scored against
persistence (copy previous shot) and the setpoint block mean. Figures
`lab-notebook/images/2026-08-28_xtcav-wm-r5to9_proprio-{skill,mechanism}.png`.

**1. NO model beats persistence on any channel group** (skill = 1 − RMSE_model/RMSE_copy):
| model | BLEN | BPM X | BPM Y |
|---|---|---|---|
| run 6 (E300) | **−1.44** | −0.32 | −0.73 |
| run 7 ep115 (E300) | −1.39 | −0.30 | −0.72 |
| run 8 (17 runs) | −0.38 | −0.30 | −0.60 |
| run 9 (17 runs) | −0.47 | −0.37 | −0.68 |
Deterministic (eps=0) predictions score the same as the 16-draw mean, so this is NOT sampling noise.

**2. Variance collapse (scale-free, unconfounded):** predicted/realized spread over all 64 channels
— run 6 **0.40**, run 7 **0.38**, run 8 **0.86**, run 9 **0.87**. The E300 models under-disperse by
~2.5×; this is the scalar analogue of the image-side commitment failure.

**3. The worst channel is the most action-coupled one.** BLEN:LI14:888:BRAW (corr 0.71 with L2,
§8.3) — E300 models predict ≈ the corpus mean (corr 0.14/0.19, bias −2.2σ) while the held-out shots
sit at +2.3σ; persistence gets corr 0.96. Same under-response pathology as the image head, visible
in a scalar.

**4. DISSOCIATION — the combined corpus HELPS the proprio side** while (per §8.22) appearing to hurt
image response: BLEN:LI14:888 corr 0.61 (r8) / 0.47 (r9) vs 0.14–0.19 for E300 models, and near-
calibrated spread (0.86 vs 0.39). CONFOUND, stated: the E300 held-out split is genuinely harder
(eval = run tails = extrapolation; persistence itself scores 0.645 there vs 0.365 on combined), so
absolute nRMSE is NOT comparable across corpora — the same trap as §8.22. Skill partially controls
and still mildly favours combined; the spread ratio is scale-free and favours it clearly.

**5. Artifact to avoid quoting:** BLEN:IN10:596 [masked] is the ONLY channel with positive skill
(+0.08…+0.34), but it is 43%-valid (E300) / 13%-valid (combined) and imputed to the mean where
invalid — the model "wins" by predicting the imputation constant. Not a real capability.

**Implication.** The proprio head is not currently a usable virtual diagnostic for BLEN or BPMs:
copy-last-shot is strictly better. Its failure mode (regression to the mean + under-dispersion) is
the same one the image head shows, and is worst exactly where the action coupling is strongest —
further evidence that the weak action pathway (§8.7/§8.22) is a single root cause expressing itself
in both heads.

### §8.25 ROOT CAUSE OF THE PROPRIO FAILURE: THE PROPRIO ENCODER NEVER TRAINS (2026-08-28)

Not capacity. **`grad/norm/encode_proprio` is logged as exactly 0.000e+00 at the first AND last step
of every run**, and a direct weight comparison confirms it: over run 6's epochs 7→47 the proprio
encoder's 299,840 parameters have relative |Δweight| = **0.000e+00**, while the proprio decoder moved
0.389 and the backbone 0.360. The proprio encoder is FROZEN AT RANDOM INIT in runs 5–9. That is
299,840 parameters — 8.5% of the 3.53M model — doing nothing but a fixed random projection.

**Why (union of three individually-justified stability decisions):**
1. `dynamics_detach_encoder=True` (run-3 anti-collapse) → no dynamics gradient reaches encoders.
2. `pred_obs_in_loss=False` (run-3 fix for the flow blow-up) → the recon loss sees DETACHED predicted
   bags, so no gradient returns through the decode path either.
3. `latent_loss_weight` unset for the proprio modality → VectorModality class default **0** → no
   roundtrip anchor. The IMAGE modality has 10, which is exactly why `grad/norm/encode_image` is
   nonzero (8.6e-3) while proprio is identically zero.
This is the arm-1d composition bug rediscovered: it was found in run 3, fixed with
latent_loss_weight=1, then REVERTED when arm 1f showed a moving proprio latent destabilized the
flow — and the revert has silently persisted through runs 5, 6, 7, 8 and 9.

**Decisive measurement (`wizard/scripts/xtcav_proprio_forensics.py`), codec vs dynamics:**
| model | roundtrip decode(encode(TRUE obs)) | full 1-step prediction |
|---|---|---|
| run 6 | 0.888 | 0.896 |
| run 9 | 0.535 | 0.529 |
Handed the true observation with no dynamics involved, the model still cannot reproduce BLEN/BPM.
The dynamics contributes essentially nothing to the error — the loss is entirely codec-side, and the
codec's proprio half was never trained end to end.

**Capacity is NOT the limit:** 64 PCs capture 99.36% of obs variance and 128 capture 99.999%, so the
128-float proprio token is ample. The pathway is untrained, not too small. (Consistent detail: the
combined corpus scores BETTER on the roundtrip, 0.535 vs 0.888 — 3.3× more data lets the DECODER
learn to invert the frozen random projection better.)

**Scope — what this does and does not invalidate.** A random nonlinear projection largely PRESERVES
information, so the backbone can still condition on proprio context; the image-side results
(separation response, polarity gates, commitment) are NOT invalidated. What is broken is decoding
back to observation space, i.e. the proprio head as a virtual diagnostic — plus the model is
effectively ~8.5% smaller than its parameter count suggests.

**Fix candidates (cheap, testable):** (a) set `model.modalities.0.latent_loss_weight=1` — the arm-1d
fix, now on a far more stable recipe than the one arm 1f rejected, and the flow-grad kill criterion
exists to catch a relapse; (b) a proprio-only roundtrip loss; (c) make `dynamics_detach_encoder`
per-modality; (d) pretrain and freeze a proprio autoencoder — a TRAINED frozen encoder strictly
dominates a random one. Pre-registered prediction for (a): roundtrip nRMSE should fall well below
persistence (0.645 on E300), and the proprio skill score should turn positive.

### §8.26 FIGURE: THE TCAV READBACK vs THE DELIVERED DEFLECTION (2026-08-28)

`wizard/scripts/xtcav_readback_vs_reality.py` → `logs/physics_eval_xtcav_e300/cross_run_figs/
readback_vs_reality.png`. Built on the **v4** corpus deliberately (v5/v6 removed exactly these shots).
Makes §8.2's finding visual:
- **Panel A**: readback deviation vs measured streak extent. The cos(dev) curve the readback implies
  sits at ~1.0 across the whole range; the data spans 0.05–1.3×.
- **Panel B**: suppression rate by readback band — 0–0.5°: 1% (n=6024), 0.5–1°: 5% (n=1127),
  **1–2°: 83% (n=1720)**, 2–5°: 88%, 5–10°: 74%, 10–16°: 79%. cos(2°) predicts a 0.06% loss.
- **Panel C**: after a polarity flip the delivered streak sits at ~0.28× settled and needs ~15–18
  shots to recover, while the readback-implied cos(dev) reads ~1.0 the entire time — the readback
  recovers in about two shots, the cavity does not.
- **Panel D**: three matched pairs from different runs/setpoints where the readbacks are
  indistinguishable at **dev ≈ 0.01°, amp ≈ 22 MV** yet the streak differs 2–4× (1.28× vs 0.44×;
  0.94× vs 0.23×; 1.04× vs 0.34×). These are post-flip shots whose phase readback has fully returned
  to nominal while the delivered field has not.

Cross-run model figures also relocated out of the notebook to `cross_run_figs/` per user request.

### §8.27 MAJOR RETRACTION — THE STAIRCASE RESPONSE METRIC IS NOT A COUNTERFACTUAL (2026-08-28)

An adversarial review of §8.22 rejected both the hypothesis AND the observation it explained. I
verified its central claims independently against artifacts already on disk; they hold.

**The metric flaw.** `part_slew`'s staircase builds ~4 contexts per (run,sign) at L2 ≈ 0.5/2/4/6°,
walks ±1° from each, then the scoring pools ALL points and fits one slope across the full 0.5–6°
span. The reported number is therefore dominated by the BETWEEN-context offset — i.e. how well the
imagined separation reflects which real context was handed in. My own decomposition:
| model | row | pooled (reported) | within-context | between-context |
|---|---|---|---|---|
| run 6 ep49 | E300_15673 | −533 | −620 | −435 |
| run 6 ep49 | TEST_15668 | −666 | −327 | **−673** |
| run 7 ep115 | E300_15673 | −320 | −639 | **−305** |
| run 8 ep49 | E300_15673 | −107 | −185 | **−105** |
| run 9 ep149 | TEST_15668 | −183 | +88 | **−219** |
Pooled ≈ between-context in 7 of 8 rows. And the within-context walks are themselves confounded: the
ramp is monotone in TIME, so open-loop rollout drift is collinear with the commanded change (the
reviewer measured δ=0 drift of −412 to −1661 µm over 8 held steps, 3–10× the 157 µm signal).

**THE CONTRADICTION WAS IN MY OWN RECORD.** §8.6 reported, in the same entry, "L2 response gate
PASSES 88–95%" AND "boundary sensitivity slopes 0.05–0.12". `part_boundary`'s `sensitivity_slope`
IS the correct paired real-vs-hold counterfactual, and it has read ≈0 for every model all campaign:
run 6 across triage rungs −0.251…+0.380, mean ≈ +0.03; run 5 −0.24…+0.39; run 8 −0.087…+0.058.
The only checkpoint with a consistently positive counterfactual is **run 7 ep115 (+0.016/+0.192/
+0.219)** — matching the reviewer's independent held-context probe (r6 2±4%, r8 1±1%, r9 0±1%,
**r7 ep115 14–39%**). Two independent estimators agree; I reported both numbers and never
reconciled them.

**WITHDRAWN:**
- §8.22/§8.23's normalization hypothesis — it explains an artifact. Retained as fact: the σ ratio
  (1.962 vs 7.014) and the 27%/73% within/between variance split, both independently reproduced;
  and the general warning about cross-corpus comparison of action-scaled metrics.
- §8.6's "the v5 data cleaning raised response to 88–95%", §8.16's "flip weighting damaged response
  94%→20%", §8.21's "the corpus costs 3× response". All are staircase-derived and measure context
  tracking under drift, not knob response.
- §8.20's "gates stay on counterfactuals" — the staircase is not one; that caution applies to it too.

**Additional errors the review found in my hypothesis, independent of the metric:** symlog is applied
AFTER z-scoring, so the true input shrink is 2.51× (geomean), not 3.57×, and is setpoint-dependent
(1.7–4.2×); the measured per-degree action→velocity gain ratio r9/r6 is **0.97**, not the predicted
0.28 — run 9 largely COMPENSATED the normalizer change by learning a ~3.4× larger normalized gain;
the "one band, no corpus split" claim had power to detect only ~2× (p≈0.23) and was a null presented
as a positive; and the r6 staircase retains only 6–7 of 22 setpoints (29.6% extraction), biasing its
fit to the steep part of the curve.

**WHAT SURVIVES.** Run 7 ep115 is the only checkpoint with measurable causal action response, so the
lever that built it is the flip-loss SCHEDULE, not the data cleaning and not the corpus. The matched
pair r7@ep47 vs r8@ep49 (same recipe/epoch, different corpus) is ~0% causal for both while reported
as 94% vs 17% — the corpus never mattered for response.

**NEXT (adopting the review's D-1 before any training):** re-score the whole ladder with a paired
fixed-context counterfactual (δ=±0.5/±1/±2°, paired noise, δ=0 drift reported alongside) plus the
existing `sensitivity_slope`; declare a response number unmeasurable when |drift| > 0.5×|signal|;
report extraction ok-rate and retained setpoints with every slope; bootstrap CIs mandatory. The
proposed fixed-scale normalization arm (§8.22 item 1) is DEFERRED — it would change a normalizer on
a model whose causal response is 0±1%, and its own prediction was mis-derived (×2.51 not ×3.6, and
the MEAN must be specified too, not just the scale).

### §8.28 COUNTERFACTUAL RE-SCORING COMPLETE + CORRECTED FIGURES + REWRITTEN DRAFT (2026-08-28)

**New official response gate:** `wizard/scripts/xtcav_counterfactual_gate.py`. Fixed real context,
symmetric ±δ (0.5/1/2°) held H steps, PAIRED noise across arms, δ=0 drift reported alongside; a
response is unmeasurable when |drift| > 0.5×|signal|. The staircase (`part_slew` + slope scoring) is
RETIRED for response.

**Demonstration of the confound (decisive):** substituting a PURE CONTEXT-COPIER into the old
staircase scoring — a model that ignores the knob and merely reproduces the separation of the real
context handed to it — yields −488 µm/deg vs the machine's −638 (**76%**) on E300_15673 and −721 vs
−687 (**105%**) on TEST. The metric certified our models at 88–95%; a knob-blind copier scores
76–105% on it.

**Re-scored ladder (E300_15673, h=4, % of the machine's local slope):**
| checkpoint | old staircase | paired counterfactual | held-knob drift |
|---|---|---|---|
| run 5 ep15 | 20% | −4.0 ± 2.6% | −1464 µm |
| run 6 ep49 | 88% | −3.9 ± 2.3% | −244 µm |
| run 7 ep47 | 94% | −5.7 ± 2.1% | +183 µm |
| run 7 ep95 | 20% | **+25.5 ± 4.8%** | −305 µm |
| **run 7 ep115** | 52% | **+39.2 ± 8.3%** | **−61 µm** |
| run 8 ep49 | 17% | +3.1 ± 1.7% | −1403 µm |
| run 9 ep79 | 22% | +2.9 ± 1.1% | −274 µm |
| run 9 ep149 | 17% | −0.9 ± 0.7% | −244 µm |
(TEST row where measured: r7 ep115 +15.5±2.0, r7 ep47 +0.3±0.9, r6 +1.8±1.2, r8 +0.3±0.7.)

**THE FRONTIER DOES NOT EXIST — the two capabilities are ALIGNED.** Polarity corr vs causal
response: 0.931→−5.7% (r7 ep47), 0.410→+25.5% (ep95), 0.399→+39.2% (ep115); every blind model
(0.99+) sits at −4…+3%. Run 7's action response builds MONOTONICALLY under the flip loss while the
polarity gate passes. §8.10/§8.16's "flip weighting damaged response" is INVERTED: it appears to
have CREATED it, for both knobs. Corollary: run 7's anneal to w=3 was the wrong move, and run 9
should be continued at higher α rather than annealed.

**Champion: run 7 snap_ep115** — passes polarity (0.399 / 68%) AND is the only checkpoint with
measurable causal L2 response (+39%), with the smallest drift in the set. The record's "no
checkpoint passes both gates" problem was an artifact and is dissolved.

**Figures** (`logs/physics_eval_xtcav_e300/cross_run_figs/`): NEW `counterfactual_rescore.png` and
`polarity_response_alignment.png` (replaces the deleted, retracted `_frontier.png`); the sweep
figures re-captioned as demonstrations OF the confound rather than evidence of response. Unaffected
and retained: polarity-flip/trajectories, loss-share, proprio, artifacts, readback_vs_reality.

**Draft rewritten from scratch** at `lab-notebook/claude/drafts/2026-08-28_xtcav-worldmodel-runs5-9.md`
(figure links point into this repo; the vault holds no copies, per user).

**Open — NOT yet run:** the discriminating ablation (matched schedule, flip_loss_weight=0, scored on
the new gate) that decides whether the flip loss CAUSES the action pathway or merely coincides with
it. Also unfinished: TEST-row counterfactuals for r7 ep95 / r9 ep79 / r9 ep149 (batch was killed).

### §8.29 DOES THE CODEC BLOCKINESS BIAS THE SEPARATION MEASUREMENT? (2026-08-30)

`wizard/scripts/xtcav_block_bias.py` → `cross_run_figs/block_bias.png`. The concern is well posed:
the decoder's plateaus are HALF-WIDTH along the streak axis, so any pedestal they add is asymmetric
about the very axis the separation is measured on.

**Two protections exist in the estimator and were quantified, not assumed:** `frame_grey` hard-
thresholds at NOISE_U8 = 5 counts (blocks measure 0.4–1.8), and the band centroid is a MEDIAN of the
projection, which is robust to a low pedestal. The residual risk is PROMOTION — a block riding on the
beam's sub-threshold tail lifting those pixels over the threshold on one side only.

**Injection control (clean causal test: synthetic half-width plateau added to REAL block-free frames):**
| amplitude | median Δsep | p90 |Δsep| | frames changed |
|---|---|---|---|
| 0.5 counts | 0 µm | 0 | 5% |
| 1.0 | 0 | 61 (1 px) | 18% |
| 2.0 | 0 | 98 | 40% |
| **4.0** | **+671** | **1745** | **100%** |
| 6.0 | +1037 | 1928 | 100% |
There is a CLIFF just below the 5-count threshold. At the measured block amplitude the MEDIAN bias is
**zero** — but 18–40% of individual frames move by 1–2 px, and we sit only a factor of ~2–3 below the
amplitude at which the measurement collapses entirely.

**On the samples we actually score** (pedestal removed vs not; note this upper-bounds the artifact
because the removal also takes some genuine beam tail):
| model | sample pedestal | px promoted over threshold | median Δsep | p90 |Δsep| |
|---|---|---|---|---|
| run 6 | 0.59 counts | 1.07% | 0 µm | 439 |
| run 7 | 0.72 | 2.51% | −122 | 854 |
| run 8 | 0.73 | 2.09% | −61 | 915 |
| **run 9 (v6 data)** | **0.08** | **0.00%** | **0** | **122** |

**VERDICT.** Median-aggregated separations (measured curves, boundary/staircase medians, the
counterfactual gate, which differences paired arms) are SAFE — median bias is zero at the measured
amplitude. PER-FRAME numbers are not: the commitment gate's per-sample width/peak/intensity features
and any single-shot comparison carry a 1–2 px tail on v5-era models. And the margin to catastrophic
failure is only 2–3×, so anything that raises the painted background (a codec change, a brighter
working point, a new corpus) could cross it.

**The v6 camera-floor zeroing already fixed most of this in practice** — run 9's samples carry a 9×
lower pedestal and promote ZERO pixels, versus 2.1% for run 8 — even though §8.19 found the block
STRUCTURE amplitude unchanged in the AE roundtrip. Level fell, structure did not; for the estimator
it is the level that matters.

**Actions:** (a) keep v6 floor-zeroing for every future corpus — it is the cheapest mitigation and is
already validated; (b) the §8.16 contingent decoder-seed change (bilinear / larger seed) remains
indicated to remove the structure at source and restore margin; (c) do NOT raise the extractor
threshold for model frames — it would break comparability with real frames, which is the one thing
the estimator must preserve; (d) report per-frame separations with the caveat above until (b) lands.

**§8.29 addendum — visual evidence** (`cross_run_figs/block_v5_vs_v6.png`, script
`wizard/scripts/xtcav_block_v5_vs_v6.py`). Same sample shown at normal gain (artifact invisible) and
at a 0–3 count stretch (where it lives), plus the fitted pedestal field and the REAL frame from each
corpus at the same stretch. v5: real frames carry a uniform 1.00-count floor, the decoder imitates it
with a mottled 0.61-count pedestal spanning the frame. v6: real background 0.00, model pedestal
**0.00**, pedestal field uniformly zero, and no pixels promoted over the extractor threshold.
PRECISION NOTE: a residual step across the seed boundary survives at about half amplitude (v5 0.59 →
v6 0.28 counts), consistent with §8.19's finding that floor-zeroing lowers the LEVEL but not the
structure. For the estimator only the level matters, so the measurement risk is removed; the
structure argues for still doing the decoder-seed fix.

### §8.30 ROOT CAUSE OF BOTH ARTIFACTS: THE DECODER HAS A 2×2 SPATIAL BOTTLENECK (2026-08-30)

User observation from the v5/v6 figure: the blockiness is not fully gone, and the bunch "looks
chopped off at the edges" versus ground truth. Both are the same architectural cause, now traced in
`src/quickdraw/models/vision.py::ConditionalUNet.velocity`.

**In an mse decode `x = zeros`.** Therefore `in_conv` and every down-path skip are spatially
CONSTANT, and `g = cond.mean(1)` is a spatially UNIFORM FiLM vector. The only spatially varying
signal in the entire decode is
`seed = cond_to_spatial(cond.flatten()).reshape(M, ch, 2, 2)` → `F.interpolate(..., mode="nearest")`.
Confirmed geometry at 64×192 with ae_bottleneck=16: bottleneck grid (16,48), seed 2×2, so **each seed
cell owns 32×96 pixels**, and the path contains **three successive NEAREST upsamples** (seed→bottleneck,
then two ×2 up levels). All image structure — beam position, curvature, tail taper — is synthesised by
convolutions from FOUR spatial locations.

**Measured on the BEAM (not the background):** mean |gradient| at the seed-cell boundaries (row 32,
col 96) divided by the same elsewhere, restricted to beam pixels:
| source | at boundary | elsewhere | ratio |
|---|---|---|---|
| REAL frames | 8.93 | 8.54 | **1.05** (no structure — the grid has no physical meaning) |
| run 6 (v5) | 6.16 | 4.01 | **1.54** |
| run 8 (v5) | 6.20 | 4.32 | **1.44** |
| run 9 (v6) | 5.98 | 5.55 | **1.08** |
Figure `cross_run_figs/decoder_seed_grid.png` overlays the grid: in the tail-only stretch the v5
models terminate the streak exactly on the boundary, which is the "chopped off" appearance.

**Why v6 helped so much.** The spatial pathway is a scarce resource. On v5 the decoder had to paint a
uniform ~1-count camera floor everywhere, and its only spatial handle is the seed — so the quadrant
structure imprinted on everything including the beam. With the floor zeroed the background is the
default output and the whole 4-location budget goes to the beam (ratio 1.54 → 1.08). v6 reduced the
DEMAND on a weak pathway rather than increasing its capacity, so the fragility remains.

**Aspect-ratio bug:** `seed_hw = 2` is hardcoded SQUARE while the image is 64×192 (1:3), so the cells
are 32×96 — three times wider than tall, i.e. the artifact is worst along the STREAK axis, the very
axis separation is measured on.

**Recommended fix (cheaper than the status quo). ⚠ SUPERSEDED — see §8.31 (measured insufficient:
4×8 recovers only 4% extraction) and §8.32 for the spec that replaced it.** `cond_to_spatial` is currently
`Linear(T·d → ch·2·2)` = Linear(4096→512) ≈ 524k params for 4 locations (the code comment notes an
8×8 dense map was ~8M and was cut for cost). Instead reshape the **32 tokens into a 4×8 spatial grid**
and project each with a SHARED `Linear(d→ch)` = 128×128 ≈ **16k params** — 32× fewer parameters for
**8× more spatial locations**. Then interpolate bilinearly, and switch the two up-path
`mode="nearest"` calls to bilinear. Config-gate all of it (`ae.seed_hw`, `ae.seed_upsample`) so
historical checkpoints decode unchanged. This supersedes §8.16's "larger/bilinear seed" as the
concrete form of the contingent decoder arm, and it now has a measured beam-level justification
rather than only a background one.

### §8.31 CAPACITY VERDICT: THE DECODER CANNOT RENDER THE BUNCH WELL ENOUGH TO MEASURE IT (2026-08-30)

Follow-up to §8.30 (user: "will that add sufficient resolution? the tail cutting seems super relevant
— ensure this has sufficient representative capacity"). Measured, and the §8.30 fix proposal
(2×2 → 4×8 seed) is INSUFFICIENT. Retracted as stated.

**1. How much of the latent the decoder actually receives.** cond is (32 tokens × 128) = 4096
numbers. The decoder sees only `g = cond.mean(1)` (**128**) plus `seed = cond_to_spatial(...)`
reshaped to (ch=64, 2, 2) (**256**). **Total 384 of 4096 = 9.4%**, rendering 12,288 pixels — 32:1.
The encoder's information is not the constraint; the decoder discards 90% of it.

**2. What spatial resolution the separation measurement needs** (real frames reduced to an N×M
layout, bilinear back, re-extracted — conservative, since the conv stack can synthesise some detail):
| layout grid | px/cell | median sep error | p90 | extraction ok |
|---|---|---|---|---|
| 2×2 (current) | 32×96 | — | — | **0%** |
| 2×6 | 32×32 | — | — | 0% |
| **4×8 (§8.30 proposal)** | 16×24 | 946 µm | 1104 | **4%** |
| 4×12 | 16×16 | 244 | 390 | 3% |
| 8×24 | 8×8 | 183 | 671 | 62% |
| **16×48 (= bottleneck)** | 4×4 | **61 µm (1 px)** | 61 | **99%** |
| 32×96 | 2×2 | 0 | 61 | 100% |
The two-lobe structure that DEFINES bunch separation survives only at bottleneck resolution.

**3. The end-to-end capacity test — AE round trip, true latent, no dynamics:**
| model | extraction ok | median |sep error| | p90 | median |width error| |
|---|---|---|---|---|
| run 6 | 99% | **244 µm (4 px)** | 1110 µm | 138 µm |
| run 9 | 92% | **244 µm (4 px)** | 1342 µm | 216 µm |
**The codec's own separation error (244 µm) EXCEEDS the entire physical effect being measured** — a
0.25° L2 step is worth 157 µm, and the single-shot fair null is 92–305 µm. Handed a perfect latent
and asked only to draw the image back, the decoder misplaces the bunches by more than the signal.
This is a hard capacity verdict and it is upstream of every dynamics result.

**4. Consequence for the fix.** Reaching bottleneck resolution from a 32-token bag cannot be done
with a bigger seed (768 locations, 32 tokens). The principled route is **spatial cross-attention**:
a 16×48 query grid attending to the 32 tokens, so every location gets a learned mixture of the full
4096 numbers, replacing `cond.mean(1)` + the 2×2 seed. Cost ≈ 25k params for q/k/v at d=64 — still
**40× cheaper than the current Linear(4096→256) at 1.05M**. Keep bilinear upsampling and the
aspect-correct grid. Injecting at one up level as well is the standard further step.

**5. Scope.** Model-sample separations carry a ~244 µm codec noise floor. Median-aggregated
quantities average it down; per-sample features (the commitment gate) sit on top of it. The
counterfactual response gate survives because it differences paired arms against a ~1200 µm expected
signal at δ=1°, which is why r7 ep115's +39 ± 8% was resolvable at all — but the floor is a real part
of its error bar, and no separation-based gate can be sharper than the codec that renders the frames.

### §8.32 DECODER ARCHITECTURE CHANGE — SPEC (2026-08-30)

Consolidates the fix implied by §8.30–§8.31 into an implementable form. Nothing here is implemented
yet; this is the spec of record, and it supersedes both §8.16's "larger/bilinear seed" and §8.30's
4×8-seed proposal.

**Target.** The decode path must expose the beam at the bottleneck grid (16,48) at 64×192 with
`ae.bottleneck=16` — §8.31's layout test puts 99% extraction and 61 µm (1 px) median separation error
there, versus 4% and 946 µm at 4×8. Anything coarser than the bottleneck fails the measurement the
campaign is built on, so the seed grid is not a tunable in the "try a few" sense: it has a floor.

**Change 1 — replace the global mean + 2×2 seed with spatial cross-attention.** In
`ConditionalUNet` (`src/quickdraw/models/vision.py`), `cond_to_spatial = Linear(T·d → ch·2·2)`
(Linear(4096→256) ≈ **1.05M params** for 4 locations) is replaced by a single-head cross-attention
block: a learned query grid `q_pos` of shape (bott_h·bott_w, ch) attends over the 32 tokens
projected to k/v. Every one of the 768 spatial locations then receives its own learned mixture of
the full 4096-number bag instead of the 384 numbers §8.31 measured. Cost at ch=64, d=128:
q/k/v ≈ 3·128·64 ≈ **25k params**, i.e. **~40× cheaper than what it replaces** while giving 192×
more spatial locations. The `q_pos` table itself is 768·64 ≈ 49k. This also removes the aspect-ratio
bug for free: the query grid is built from `self.bott_hw`, which is already (16,48), so there is no
hardcoded square.

**Change 2 — keep `g = cond.mean(1)` only as the FiLM vector, not as the sole content path.** The
mean is a reasonable global film signal; the failure in §8.31 is that it was carrying *content*
because nothing else could. With change 1 in place it keeps its FiLM role unchanged.

**Change 3 — bilinear, not nearest, on the two up-path `F.interpolate` calls.** Nearest is what
turns any residual coarse structure into visible plateaus (§8.29–§8.30). The seed→bottleneck
interpolate disappears entirely, since the attention output is already at `bott_hw`.

**Change 4 — inject at one up level as well** (a second, cheaper cross-attention at the first
up-level grid). Standard, and it is what lets fine tail structure be addressed rather than
synthesised by convolutions from the bottleneck alone. Treat as the second increment, not the first.

**Config gating.** All of it behind `ae.decoder_cond` (`"seed"` = current behaviour, default;
`"xattn"` = the above) plus `ae.seed_upsample` (`"nearest"`/`"bilinear"`). Historical checkpoints
must decode byte-identically under the defaults — every number in §7–§8.31 was produced by the
`"seed"` path and stays comparable only if that path is untouched.

**Acceptance gates, in order.** (a) AE round trip on the true latent — median |sep error| must fall
from the measured 244 µm to under the 61 µm single-pixel floor, since §8.31's layout test says the
information is there at this grid; (b) the seed-boundary gradient ratio on beam pixels returns to
the real-frame value of 1.05 (from 1.44–1.54 on v5, 1.08 on v6); (c) no regression in the
counterfactual response gate for a retrained run-7 recipe. Gate (a) is the one that matters — it is
the capacity claim, and it is measurable before any dynamics training at all, on an AE-only run.

**Why this is now first in line.** §8.31's verdict is that the codec's own separation error exceeds
the physical effect (244 µm vs 157 µm for a 0.25° L2 step). Every loss-side and data-side change
under discussion is scored through this renderer, so its floor caps all of them. Ordering: decoder
first, then the proprio-encoder fix (`latent_loss_weight` > 0, §8.25), then the step-response work.

### §8.33 RUN 10 PLAN — THREE GOALS, TWO STAGES (2026-08-30)

User-set goals for the next run: (1) decoder stability/blockiness, (2) L2/XTCAV transition fidelity,
(3) proprio fidelity. Goals 1 and 3 are the SAME class of defect and 2 is a different one, which
sets the staging.

**The structural point.** §8.31 (image) and §8.25 (proprio) are both CODEC failures, and both were
measured with the dynamics removed — §8.31's AE round trip (244 µm on a true latent) and §8.25's
`decode(encode(TRUE obs))` (0.888 nRMSE vs 0.896 for the full 1-step prediction, i.e. the dynamics
contributes ~0.008 of the error). Neither needs a dynamics run to score. Goal 2 is the only one that
does. So: fit the codec first, gate it, then train dynamics on a codec that can express the answer.

**Why the proprio encoder is gradient-free, restated as the fix condition.** Verified in code:
`roundtrip_losses` is modality-agnostic and gated ONLY on `latent_loss_weight > 0`; `decode_loss`
trains the decoder on the dynamics' PREDICTED bag, which `pred_obs_in_loss=False` detaches. So the
roundtrip anchor is the ONLY gradient path to any encoder — which is exactly why image (llw=10) has
`grad/norm/encode_image` = 8.6e-3 and proprio (llw unset → 0) is identically 0.

**STAGE A — codec only.** `lambda_flow=0` **and** `model.modalities.*.weight=0` (both needed: with
the flow untrained, the decode loss would otherwise train the decoders against garbage predicted
bags), `model.modalities.0.latent_loss_weight` > 0, image keeps 10, plus the §8.32 decoder
(`ae.decoder_cond=xattn`, bilinear up-path). Only `codec/roundtrip_*` is live, which trains BOTH
adapters through the real encode→decode path. Gates:
- (1a) AE round-trip median |sep error| **244 µm → below the 61 µm single-pixel floor** (§8.31);
- (1b) seed-boundary gradient ratio on beam pixels **1.44–1.54 → 1.05**, the real-frame value (§8.30);
- (3a) proprio round-trip nRMSE **0.888 → well below persistence at 0.645** (§8.25 pre-registered);
- (3b) predicted/realized spread **0.38–0.40 → ~1.0** (§8.24's variance collapse). NOT guaranteed by
  the encoder fix — mean-regression is consistent with a frozen random projection, but it is not
  proven to be caused by it. Report it either way; if it survives stage A it is a separate defect.

**STAGE B — dynamics, warm-started from A** via the existing `load_checkpoint` path. Set proprio
`latent_loss_weight` back to **0** and leave `dynamics_detach_encoder=True` and
`pred_obs_in_loss=False` untouched. The proprio encoder then receives zero gradient exactly as in
runs 5–9 — but it is now a TRAINED frozen encoder instead of a random one, which strictly dominates
(§8.25 fix (d)). **This is what makes goal 3 safe:** arm 1f rejected `latent_loss_weight=1` because a
MOVING proprio latent destabilized the flow; here the latent does not move during dynamics training,
so that failure mode is not merely mitigated, it is absent. Stage A carries no such risk either,
because it has no dynamics loss to destabilize.

**Goal 2 rides on stage B** with the flip-loss family (§8.7/§8.16) plus whatever the running
action-representation study returns (arms: raw / raw-weighted / delta / fourier-8 / f8-weighted /
f8-delta, weighted for ~30% boundary loss mass on the combined corpus). Fold its result in before
fixing the stage-B config. Standing input: the zero-flip-weight ablation decides whether the flip
loss CAUSES run 7 ep115's +39.2 ± 8.3% or merely coincides with it.

**Caveats, stated up front.**
- Stage A trains decoders on TRUE bags; stage B trains them on PREDICTED bags. A is a warm start, not
  a final answer — re-run gate (1a) after B and expect some drift.
- §8.25's error decomposition (codec 0.888, dynamics +0.008) holds while the CODEC dominates. Once it
  is fixed the dynamics term need not stay negligible, so (3a) passing at stage A does not guarantee
  the 1-step proprio prediction beats persistence. That is a stage-B measurement.
- Run 10 will not be numerically comparable to runs 5–9 on codec-side metrics, by construction. The
  counterfactual gate (§8.27) stays comparable — it is scored on separations, in µm, through
  whatever renderer the run has.

### §8.34 STEP-RESPONSE STUDY: THE CORPUS HAS ONE STEP SIZE (2026-08-30)

Reviewer-agent study (12 measurements, GPU 1; scratch `m1`–`m17_*`). Three findings change §8.33.
Load-bearing claims re-verified independently before adoption; each is marked.

**M-A — the commanded L2 step is a monoculture. [VERIFIED independently.]** Reading the action
parquets directly: of 25,227 transitions in the combined v6 corpus, 936 are non-zero in L2, and
**|ΔL2| min = p50 = p90 = max = 0.2500°** — every commanded step in the entire corpus is exactly
±0.25°. Episode L2 span is **exactly 1.500° for all 172 episodes** (p50 = p90 = max), which is
`XTCAV_BLOCK_STEPS = 6` × 0.25° (`processors.py:338`). Direction: 516 up / 420 down on combined;
E300 alone supplies only +0.25 (L2 never decreases there), E331 supplies the down-ramps. A 24-step
training window spans at most 0.25–0.50°.

**Consequence, and it is severe: the official gate (§8.27) commands δ = ±0.5/1/2° held 8 steps —
2× to 8× beyond the largest action excursion in any training window, and on E300 the −δ arm is a
direction never commanded.** The gate is an extrapolation probe. The model has never been shown the
data from which a response *function* (magnitude) could be learned — only "a step happened".

**M-B — run 7 ep115's response is LARGE-SIGNAL ONLY.** Paired ±δ at h=1 (drift-free by
construction), n≈220 pairs, bootstrap CI:
| model | ±0.25° (in-distribution) | ±1° | ±2° |
|---|---|---|---|
| run 6 | −5.8% [+1.1, −15.8] | +1.8% | +2.3% |
| **run 7 ep115** | **−2.8% [+14.5, −23.3]** | **+12.1% [3.7, 20.4]** | **+21.2% [12.4, 30.0]** |
| run 8 | +1.7% [+14.1, −7.5] | +0.1% | −0.1% |
| run 9 | +0.5% [+1.4, −0.2] | −0.1% | −0.2% |
This does NOT retract §8.28 — the gate was computed correctly; it resolves what the number means. The
+39.2 ± 8.3% at h=4 pools δ = ±0.5/1/2, i.e. it is an average over rungs that are all out of
distribution. **Stated precisely: at ±0.25° the measurement is UNDERPOWERED (CI spans ±19 points at
n=223; resolving 5% needs ~750 pairs, ~10 GPU-min), so this is "not resolvable", not "proven zero".**
What makes the distinction moot for planning is M-G4 below: run 7 ep115 is strongly NONLINEAR in the
action (v/act 0.49 at 0.25–1° → 1.92 at 4°) while every other model is linear and flat. A nonlinear
large-signal mode cannot be extrapolated down to the operating step size. **Every future quotation of
+39.2% must carry "at δ = ±0.5–2°, out of distribution".**

**M-D — E331 is also 0%.** Same protocol, scored on `measured_all.json` per-(run,sign,L2) medians:
r8 on E331_16023 −0.2…+0.4%; r9 on E331_16023 +0.6…−0.1%; r9 on E331_16034 −0.1…+2.2%. The response
failure is not E300-specific — combined-corpus models are blind on their own corpus. An E331 gate
needs no new ground truth, but must pre-register |L2| ≤ 8 (45.9% of E331 shot mass sits at |L2| > 8
where the measured slope collapses to 122 µm/deg) and |local slope| ≥ 200 µm/deg, and report the
extraction ok-rate (67–80% on E331 vs 92–99% on E300).

**M-I — THE FORK: the information is present; the forward model does not use it.** Classifier on
(prev bags, realized latent residual) → was a 0.25° L2 step commanded?
| corpus / encoder | held-out AUC | 95% CI | n_pos |
|---|---|---|---|
| E300 / run-6 encoder | 0.732 | wide | 15 |
| **combined / run-9 encoder** | **0.913** | **[0.880, 0.942]** | **153** |
| combined / run-9 encoder, polarity flip *(control)* | 0.992 | [0.985, 0.997] | 43 |
A 0.25° step is clearly identifiable from the realized transition **in the same latent space where
run 9's forward model has 0% causal response.** This is the exact analogue of probe B's polarity
verdict — representation and data are sufficient, the blocker is forward-side gradient allocation —
and that analogue correctly predicted run 7. The earlier E300-only null (§8.3-era) was underpowered
at 15 positives, which is why the deficit looked like an observability problem.

**M-J — input representation beats loss weighting, and the ordering is measured.** Forward
learnability (probe-B protocol, combined, 581 train / 160 held-out boundary steps, binomial SE 3.95%).
Decisive metric = paired action ablation ("A-wins": does the true commanded L2 beat knob-held-fixed?).
| arm | A-wins | z vs chance |
|---|---|---|
| **raw symlog scalar (what runs 5–9 use)** | **50.0%** | **0.0** |
| raw + boundary weight (w=12.5, 30% mass) | 45.6% | −1.1 |
| + Δaction channel | 51.2% | 0.3 |
| **Fourier(8, f_max 16)** | **58.1%** | **+2.05 (p≈0.04)** |
| Fourier + boundary weight | 53.1% | +0.8 |
| **Fourier + Δaction** | **60.6%** | **+2.7 (p≈0.007)** |
The raw scalar the models actually consume is at **exactly chance**. Boundary loss weighting is
null-to-negative and is below its unweighted twin in all three pairings — weighting reallocates
gradient toward a target the input cannot resolve. **Ordering claim, evidence-backed: fix the action
input representation FIRST, then weight — not the reverse.** This contradicts the natural instinct to
extend the flip-loss lever to L2 boundaries by analogy. Effect is small (60.6% vs 100% for polarity
in probe B): Fourier is necessary, not sufficient.

**M-H — the repo's Fourier ladder is mis-conditioned for this knob. [VERIFIED independently.]**
`fourier_freqs(n_freq, f_max=100.0)` (`features.py:36`) has exactly one caller,
`multimodal.py:76`, which **never passes f_max** — so it is hardcoded at 100.0 and the top band
sweeps ~55 cycles across the E300 L2 range at any `n_freq`. Measured effect: ‖Δfeature‖ is nearly
step-size independent (4.53 at 0.25° vs 5.40 at 4°), destroying the metric. An 8-band f_max = 16
ladder grows monotonically (1.95 → 4.62) and still amplifies a 0.25° step 30×. **Exposing
`fourier_fmax` is a required small code change if Fourier is used** — shipping `action_fourier_freqs`
with the default f_max would be worse than the scalar.

**M-G — conditioning geometry, and a cheap monitor.** At the flow head, ΔL2 = 0.25° moves the whole
conditioning vector by only 0.16–0.47%; the action block carries ≤9–18% of the conditioning norm
despite occupying 1/3 of its width (`_cond` concatenates `[h_state | slot | act_enc]`). **`v/cond` is
the only internal quantity that tracks causal response** (r = 0.76, n = 6; r6 1.94, r8 2.03, r9 1.23,
r7ep47 3.42, r7ep95 3.36, r7ep115 4.81) and it is *leading* — ep47 had 3.42 with zero response. Use
as a 3-GPU-min monitor every 4 epochs; necessary, not sufficient.

**M-G3 — the action-normalizer arm is refuted, not merely deferred.** Run 9 has 2.7× LARGER
rel_cond than run 8 and 1.5× larger than run 6, with 0% response: it self-compensated the combined
normalizer by learning a 2.1× larger action-block norm share (0.181 vs 0.086). A rescale will be
absorbed. Also the combined L2 variance is 73.1% BETWEEN-run, so the within-run sd is 3.94 not 7.014
— a fixed physical scale is a 1.7× change, not 3.57×. §8.22/§8.23's withdrawal is now affirmative.

**A12 — `p_tf < 1` cannot fix rollout drift. [VERIFIED in code.]** `loss_terms` always teacher-forces
the flow (velocity matching against true encoded states at every lead), so `p_tf` touches only the
decode/recon conditioning. Drift control needs a genuine latent multi-step consistency loss — a real
code change, not a config knob. Relevant because M-C puts median |δ=0 drift| ≤ 61 µm only at h ≤ 3;
**an in-distribution gate exists only at h ≤ 3.**

**M-K — what E331 actually contributes.** +840 L2-boundary transitions (9.75× E300's 96), **ZERO
polarity flips, ZERO TCAV on↔off transitions**. E331_16035's ~1,200 off anchors are a single all-off
run with no transition in it — which is exactly why run 8's off gate stalled at 6.1% despite the
anchors. E331 is 3.7× denser in action changes per unit time (19 vs 70 shots/setpoint), lifting
P(24-step window contains an L2 boundary) from 29.2% to 73.5%. **Keep the corpus combined**: the
corpus never mattered for response (§8.27's matched r7ep47/r8ep49 pair is ~0% for both; M-D confirms
on E331), it mattered for polarity, and the composition-invariant α already solves that. Do not drop
E331 — it holds the only well-powered L2 supervision in the campaign.

**Operational.** The r7abl ablation had **no snapshotter and an empty `snaps/`** — Lightning writes
`last.ckpt` only at validation with `save_top_k=2` by val loss, so it would have finished with ep119
plus two early-epoch checkpoints and the pre-registered ep47/95/115 comparison would have been
impossible (§8.13's blind gap repeating). A non-invasive copier is running
(`scratchpad/snapshotter.sh`, verified read-only w.r.t. the job; 5 snaps captured, ablation at
ep21/120 at 18:48, ETA ~21:50). Separately, **run 8's `snaps/` has no symlinks into `checkpoints/`**
and needs them before its ladder can be re-scored.

### §8.35 RUN 10 PLAN, REVISED AGAINST §8.34 (2026-08-30)

§8.33's two-stage codec plan (goals 1 and 3) is UNCHANGED — §8.34 touches nothing in the codec path,
and stage A is action-agnostic. What changes is goal 2, and one arm is superseded.

**Goal 2 is re-scoped by M-A.** The prior framing — "make the model respond to the L2 step" — assumed
the training data could teach a response function. It cannot: one step size, ±0.25°, and ≤0.50° per
window. So goal 2 splits:
- **2a (modelling, now):** make the model use the step information it demonstrably has. M-I puts
  inverse-dynamics AUC at 0.913 in run 9's own latent space, so this is forward-side gradient
  allocation, not observability. Lever order is measured (M-J): **input representation first**
  (Fourier with an exposed `fourier_fmax`), weighting last or not at all.
- **2b (data, has lead time):** a DAQ scan that randomizes the L2 step (±0.25 to ±2°, both
  directions, interleaved). This is the only thing that makes a response *function* learnable and the
  gate's rungs in-distribution. Outside the modelling critical path, so it should be requested now
  rather than after 2a resolves. Joins the standing randomized-polarity request.

**Gate protocol must be fixed BEFORE the ablation is scored.** The current gate cannot detect the
improvement goal 2 targets, because its rungs are OOD. Add a δ = ±0.25°, h ∈ {1,2,3} rung at
n ≥ 750 pairs with bootstrap CIs, per corpus, and keep the §8.28 rungs for comparability. h ≤ 3 is
forced by M-C's drift budget. ~30 GPU-min per checkpoint. Pre-registered prediction: every existing
checkpoint, run 7 ep115 included, reads 0 ± 5%.

**Superseded / dropped arms.**
- The reviewer's E3 (proprio `latent_loss_weight=1` inside a dynamics run) is **superseded by
  §8.33's stage A**: pretrain-and-freeze reaches the same trained encoder with the arm-1f
  instability structurally absent rather than merely monitored, and costs no separate arm.
- The action-normalizer rescale is **dropped**, not deferred (M-G3 refutes it).
- Boundary-group α is **scheduled last and conditionally** (M-J measures it null-to-negative before
  the input fix). §8.16's deletion of α_bnd rested on a staircase-derived premise that §8.28
  inverted, so it is reopened in principle — but only after the input change moves the gate.
- Capacity (width/depth) stays deferred; §8.21's claim was withdrawn and stage A recovers the
  8.5% of parameters the frozen proprio encoder was wasting.
- `p_tf < 1` is dropped as a drift fix (A12, verified). If drift blocks the gate, the replacement is
  a latent multi-step consistency loss — a real code change, sized ~1 day.

**Revised sequence.** Stage A (codec, goals 1+3) → E1 the running ablation, scored on the FIXED gate
→ stage B = run 10a warm-started from A, whose single action-side change is the Fourier ladder with
`fourier_fmax` exposed (default 100.0 preserved so every existing checkpoint stays bit-identical).
Do not combine the Fourier ladder with any other action-input change; `n_freq` and `f_max` are one
variable, never two.

**The most likely outcome, stated in advance (the reviewer's F2).** Run 10a passes the OOD gate and
fails the in-distribution one — i.e. the response stays large-signal-only, an artifact of the
0.25° monoculture. **If that happens the next action is 2b (data), not more modelling.** Pre-register
it so the result is not re-interpreted after the fact.

### §8.36 MAKING THE GATE FAIR: RAMP COUNTERFACTUALS + SCENARIO RANDOMIZATION (2026-08-30)

Ryan: "are there ways to do scenario randomization or change the validation procedure so it's more
fair between training data and counterfactual data?" Two families, answering DIFFERENT questions —
worth keeping separate. Changing the gate measures *what the model learned from what it saw* (fair
attribution). Scenario randomization changes *what it saw* (capability). §8.34 conflated them.

**KEY STRUCTURAL FACT, measured.** Grouping the combined v6 parquets by `episode_index`:
**all 172 episodes are 100% monotone in L2**, with a median of **6** non-zero steps each (min 1,
max 6) — an episode literally IS a monotone 6-step ramp of ±0.25° spanning 1.5°. Therefore:

**V1 — THE CUMULATIVE-RAMP COUNTERFACTUAL (recommended primary fix).** The gate's OOD-ness comes from
the single-step MAGNITUDE, not from the total excursion. Replace the single δ = ±1° jump with
**4 consecutive in-distribution ±0.25° steps**: total excursion 1.0°, every individual transition
exactly what training contains, and the trajectory shape (a monotone ramp) is the training episode's
own shape. Keeps the paired-symmetric design, so drift still cancels; keeps ~600 µm of expected
signal (4 × 157 µm), which clears both the codec floor and M-C's drift budget in a way the
in-distribution single step does not. **Score it on the COMBINED corpus** — E300 has only +0.25
(L2 never decreases), so a ±ramp is fully in-distribution only where E331's 420 down-steps live.
This is strictly fairer than both the current gate (OOD magnitude) and a bare ±0.25° rung (signal
below the codec floor per §8.31).

**V2 — report the excursion percentile with every gate number.** State where the commanded δ sits in
the training action-excursion distribution. "δ = ±2° — never observed; max training window excursion
0.50°" makes the extrapolation visible in the result instead of implicit in the protocol. This is the
discipline whose absence let §8.28's number stand for two days. Zero cost.

**V3 — a matched real-data null through the same extractor.** Take real held-out shot pairs at L2 and
L2+δ, push them through the identical `energy_gated_sep` path, and report the model against that
rather than against the fitted machine slope alone. Codec bias, extractor threshold effects and
shot noise then appear in BOTH arms. §8.31's 92–305 µm single-shot null exists but is not integrated
into the gate.

**V4 — n ≥ 750 pairs is not optional, and it is what makes an in-distribution rung possible at all.**
A 0.25° step is worth 157 µm; the codec's own separation error is 244 µm median (§8.31). Per-sample
the signal is BELOW the renderer's noise. The mean beats it down as 1/√n (244/√750 ≈ 9 µm), so the
rung is resolvable in aggregate but only in aggregate. **Goal 1 (decoder) is therefore a prerequisite
for any per-sample or small-n response gate**, not merely a parallel goal.

**V5 — hold out SCENARIOS, not run tails.** val/eval are currently run tails, i.e. extrapolation,
which §8.24 already showed makes persistence itself score differently across corpora (0.645 vs 0.365)
and makes absolute nRMSE non-comparable. Holding out whole runs or whole setpoint blocks gives eval
the same action statistics as train at different working points — comparable by construction.

---

**Scenario randomization (training side) — and its honest limit.**

**S0 — the limit.** No resampling can synthesise a step size the machine never commanded. Subsampling
creates *aggregate* excursions (4 × 0.25° collapsed into one 1.0° action label), which is useful, but
the underlying physical transition remains a 4-step ramp, not a jump. The model would learn "1.0° per
4 time units", which is self-consistent and matches V1's gate — but it is NOT a single-shot 1° jump,
and it must not be reported as if it were.

**S1 — ⚠ `data.subsample` IS NOT CONFIG-ONLY FOR THIS CORPUS; AS WRITTEN IT WOULD CORRUPT THE ACTION
SPACE. [VERIFIED.]** `_subsample_episodes` (`dataset.py:83`) **SUMS** actions over each group, taking
last only on dims it detects as near-binary (`len(unique) <= 2`). Measured on the combined corpus, all
four XTCAV dims are **absolute setpoints** with thousands of unique values —
dim 0 (L2 phase) −15…+15, 121 unique; dim 1 (S) ±22.5, 23,082 unique; dim 2 (dev) ±1.0, 23,806;
dim 3 (amp) 0…22.5, 22,464. **None is detected as binary, so all four would be summed**: four absolute
L2 setpoints of ~4° each aggregate to ~16°, outside the physical range entirely. The correct rule here
is TAKE-LAST on all four (the setpoint that ends the group is the setpoint that drove it), with the
excursion carried by an explicit Δ channel if wanted. This needs an aggregation-mode flag —
a small code change, not a config toggle. The reviewer's A4 is corrected on this point.

**S2 — `XTCAV_BLOCK_STEPS` 6 → 12–20** lifts the 1.5° episode cap to 3–5°. Must compose with S1:
a 24-frame window still covers only ~1–2 scan steps without stride, so neither knob delivers range
alone. Re-convert ≈ 20 min, and it is a new corpus version (do not combine with a model change).

**S3 — inference-time classifier-free guidance is trained but unused.** `act_null` already trains a
genuine unconditional branch (`multimodal.py:730`), and the comment states the intent —
`v = v_u + w·(v_c − v_u)` — but also "Train-only; eval/rollout untouched". Adding the guided sampler
gives a direct response-amplification knob at zero training cost. Caveats: runs 7/8/9 used
`action_dropout=0.0` so their unconditional branch is untrained (only run 6 at 0.15 has one), and any
guidance scale w must be fixed in advance and reported with every number, or it becomes a free
parameter tuned against the gate.

**S4 — inverse-dynamics auxiliary head.** M-I's AUC 0.913 says the target is learnable in the existing
latent, so the head will fit; the open question is whether its gradient reshapes the FORWARD model.
Note `model.action_head` is a forward prior p(a|h) and is the wrong direction. ~20 lines.

**Ordering.** V1 + V2 + V3 are protocol-only and should land BEFORE anything is re-scored, including
the cancelled ablation's surviving checkpoints — they cost no training and they change what every
subsequent number means. S1/S2 are a corpus change and belong with, not before, run 10.

### §8.37 RUN 10 — FULL SPEC (2026-08-30)

Consolidates §8.32–§8.36 into the plan of record. Supersedes §8.33 and §8.35 where they differ.
Ryan's three goals: (1) decoder stability/blockiness, (2) L2/XTCAV transition fidelity, (3) proprio
fidelity. Goals 1 and 3 are codec defects measurable WITHOUT dynamics; goal 2 is not. That sets the
staging. Base config = run 9 verbatim (read from its `logs/config.json`), one group of changes per
stage.

---
#### PHASE 0 — protocol + code, no GPU. Blocks everything downstream.

Nothing may be re-scored (including the cancelled ablation's ep3–19 snapshots) until P0-1..3 land:
they change what every number means.

**P0-1 `fourier_fmax` exposed.** `features.fourier_freqs(n_freq, f_max=100.0)` has ONE caller,
`multimodal.py:76`, which never passes it (§8.34 M-H, verified). Thread it
`features → FourierMLP.__init__ → setup.py`, **default 100.0** so every existing checkpoint stays
bit-identical. Without this, `action_fourier_freqs=8` ships a 55-cycle top band and is worse than the
scalar.

**P0-2 decoder `ae.decoder_cond` + `ae.seed_upsample`.** The §8.32 spec: single-head spatial
cross-attention, query grid built from `self.bott_hw` (= (16,48) at 64×192/bottleneck 16), replacing
`cond.mean(1)`-as-content and the 2×2 seed; bilinear on the two up-path interpolates. Config-gated
(`"seed"`/`"nearest"` = current behaviour = default).

**P0-3 ramp counterfactual (§8.36 V1) + reporting discipline (V2, V3).** Add to
`xtcav_counterfactual_gate.py` a `--ramp K` mode: K consecutive ±0.25° steps instead of one ±δ jump.
Justified by the measured fact that all 172 episodes are 100% monotone in L2 with a median of 6
steps — a K=4 ramp IS the training episode's own shape. Score on the COMBINED corpus (E300 has no
down-steps). Every emitted number carries: n pairs, bootstrap CI, extraction ok-rate, δ=0 drift at
the same horizon, and **the percentile of the commanded excursion in the training distribution**.

**P0-4 run 8 `snaps/` symlinks** into `checkpoints/` so its ladder is scoreable (every other run has
them).

*Deliberately NOT in P0:* the `data.subsample` aggregation-mode flag (§8.36 S1). It is a real bug for
this corpus — all four action dims are absolute setpoints and would be SUMMED — but it is only needed
for S2's corpus change, which is not in run 10.

---
#### STAGE A — codec only. Goals 1 and 3. No dynamics.

```bash
uv run --no-sync python -m quickdraw.train_world_model \
  experiment=xtcav_all_r10A \
  data.root=logs/recording_2026_08_27_10_08_28_xtcav_all data.repo_id=xtcav_all data.cam=dtotr2 data.F=16 \
  environments=recorded environments.obs_dim=138 environments.action_dim=4 environments.position_idx=null \
  model=bsp32mse model.depth=4 model.action_dim=4 model.window=24 \
  model.lambda_flow=0 model.lambda_consistency=0 \
  model.modalities.0.weight=0 model.modalities.1.weight=0 \
  +model.modalities.0.latent_loss_weight=1 model.modalities.1.latent_loss_weight=10 \
  model.modalities.0.dim=138 model.modalities.0.decode_kind=mse +model.modalities.0.fourier_freqs=16 \
  model.modalities.1.img_size=[64,192] +model.modalities.1.ae_bottleneck=16 \
  +model.modalities.1.decoder_cond=xattn +model.modalities.1.seed_upsample=bilinear \
  model.recon_frac=1.0 model.p_tf_start=1.0 model.p_tf_end=1.0 \
  model.pred_obs_in_loss=false model.dynamics_detach_encoder=true \
  optim.lr=1e-4 eval.during_train.evals.control=false
```

**Why both zeroings.** `lambda_flow=0` alone is not enough: with the flow untrained, `decode_loss`
would train the decoders against garbage PREDICTED bags (`pred_obs_in_loss=false` detaches them but
does not make them meaningful). Zeroing the modality weights leaves only `codec/roundtrip_*`, which
runs the REAL encode→decode path and therefore trains both adapters. Verified in
`roundtrip_losses`: it is modality-agnostic and gated solely on `latent_loss_weight > 0`.

**Why this fixes proprio without the arm-1f risk.** Run 9's `modalities.0` has no
`latent_loss_weight` key at all → VectorModality default 0 → the roundtrip anchor, which is the ONLY
gradient path to any encoder, is off. Hence `grad/norm/encode_proprio` ≡ 0 and 299,840 params frozen
at random init. Stage A turns it on where there is **no dynamics loss to destabilize** — arm 1f's
failure mode is structurally absent, not merely monitored.

**Stage-A gates** (all measurable with zero dynamics compute):
| # | quantity | baseline | pass |
|---|---|---|---|
| **A1** | AE round-trip median \|sep error\|, true latent | **244 µm** (r6, r9) | **< 61 µm** (1 px) |
| A2 | seed-boundary grad ratio, beam pixels | 1.44–1.54 (v5), 1.08 (r9) | **≤ 1.10**, target 1.05 (real) |
| A3 | AE round-trip p90 \|sep error\| | 1110 / 1342 µm | < 300 µm |
| **A4** | proprio round-trip nRMSE | **0.888** (r6), 0.535 (r9) | **< 0.645** (persistence) |
| A5 | predicted/realized spread, 64 ch | 0.38–0.40 (E300), 0.86–0.87 | ∈ [0.7, 1.3] |
| A6 | ae_floor vs run 9 | — | no regression > 1 dB |

A1 and A4 are the decisive ones. **A5 is reported but NOT pass/fail**: mean-regression is consistent
with a frozen random projection but not proven caused by it; if it survives stage A it is a separate
defect and gets its own investigation.

---
#### STAGE B — dynamics, warm-started from A. Goal 2. TWO ARMS IN PARALLEL.

Warm-start via the existing `load_checkpoint` path. Set proprio `latent_loss_weight` **back to 0** and
leave `dynamics_detach_encoder=true` / `pred_obs_in_loss=false` untouched: the proprio encoder then
receives zero gradient exactly as in runs 5–9, but is now TRAINED-frozen rather than random-frozen.

Both GPUs are free, so run the control and the treatment **concurrently** — a clean one-variable
contrast at the same wall-clock as a single run. Do not skip the control; this campaign has already
retracted one result to a confounded comparison (§8.28) and one to a two-segment schedule (§8.34).

| arm | GPU | action input | everything else |
|---|---|---|---|
| **10B-ctrl** | 0 | `action_fourier_freqs=0` (raw symlog scalar, as run 9) | identical |
| **10B-fourier** | 1 | `action_fourier_freqs=8` + `action_fourier_fmax=16.0` | identical |

```bash
# shared; ARM = ctrl | fourier
uv run --no-sync python -m quickdraw.train_world_model \
  experiment=xtcav_all_r10B_${ARM} trainer.max_epochs=150 \
  model.init_from=logs/<stage-A run dir>/checkpoints/best.ckpt \
  data.root=logs/recording_2026_08_27_10_08_28_xtcav_all data.repo_id=xtcav_all data.cam=dtotr2 data.F=16 \
  data.boundary_frac=0.0 data.boundary_sign_dim=null data.boundary_delta=0.0 \
  environments=recorded environments.obs_dim=138 environments.action_dim=4 environments.position_idx=null \
  model=bsp32mse model.depth=4 model.action_dim=4 model.window=24 model.action_squash=symlog \
  model.modalities.0.dim=138 model.modalities.0.decode_kind=mse +model.modalities.0.fourier_freqs=16 \
  model.modalities.1.img_size=[64,192] +model.modalities.1.ae_bottleneck=16 \
  +model.modalities.1.decoder_cond=xattn +model.modalities.1.seed_upsample=bilinear \
  model.modalities.1.latent_loss_weight=10 \
  model.recon_frac=1.0 model.compile_rollout=false \
  model.p_tf_start=1.0 model.p_tf_end=1.0 model.pred_obs_in_loss=false model.dynamics_detach_encoder=true \
  model.diffusion.action_dropout=0.0 model.diffusion.flip_loss_alpha=0.12 model.diffusion.flip_loss_weight=0.0 \
  model.diffusion.concat_action_embedding=true \
  optim.lr=1e-4 eval.during_train.evals.control=false
  # ARM=fourier adds:  model.action_fourier_freqs=8 +model.action_fourier_fmax=16.0
```
`concat_action_embedding` stays EXPLICIT: `setup.py:219` falls back to **False** for configs that omit
it, and runs 7/8/9 all had it true. `flip_loss_alpha=0.12` is carried over from run 9 UNCHANGED so the
Fourier ladder is the only action-side variable; α is the polarity lever and is not part of the
response story (§8.34 M-J).

**Stage-B gates.** Primary is the ramp gate, because it is the only in-distribution one.
| # | gate | baseline | pass |
|---|---|---|---|
| **B1** | **ramp counterfactual**, K=4 × ±0.25°, combined corpus, n ≥ 750, bootstrap CI | none exists yet — establish on run 9 + r7ep115 first | **≥ +25%** of local slope, CI excluding 0, on ≥ 3 of 4 rows |
| B2 | §8.28 gate (δ=±0.5/1/2, h=4), **labelled OOD** | r7ep115 +39.2 ± 8.3 | report only; comparability, not a target |
| B3 | single-step in-distribution δ=±0.25°, h ≤ 3, n ≥ 750 | −5.8…+1.7%, all CIs spanning 0 | ≥ +15%, CI excluding 0 |
| B4 | median \|δ=0 drift\| | h3 ≤ 61 µm; h8 183–1830 | ≤ 100 µm at h=3 |
| B5 | polarity: corr(flip, hold) / realized wins | r9 0.995 / 0% | < 0.8 and ≥ 60% |
| B6 | E331 transfer: B1 on ≥ 1 E331 row, \|L2\| ≤ 8, \|slope\| ≥ 200 µm/deg | 0% (M-D) | as B1 |
| B7 | no codec regression vs stage A | A1–A6 | A1 within 2× after dynamics training |
| **M** | `v/cond` at ΔL2 = 0.25°, every 4 ep (~3 GPU-min) | r9 1.23, r8 2.03, r7ep115 4.81 | alarm if < 2.5 by ep60 |

---
#### HOW EACH GOAL IS ADDRESSED

**Goal 1 — decoder stability/blockiness.** Root cause is not the data floor (v6 already fixed the
pedestal) but the decode path: 384 of 4096 latent numbers reach it, through a 2×2 seed hardcoded
square on a 1:3 image, upsampled nearest three times. Fixed by P0-2, gated by A1/A2/A3 in stage A
**before any dynamics compute is spent**. Note the coupling that makes this first in line rather than
parallel: a 0.25° step is worth 157 µm and the codec's own error is 244 µm, so at the in-distribution
step size the physical signal is below the renderer's noise. Only the n-averaged mean survives
(244/√750 ≈ 9 µm). **Goal 1 is a prerequisite for goals 2's fair gate, not a sibling of it.**

**Goal 2 — L2/XTCAV transition fidelity.** Split, because the corpus cannot support the original
framing. (2a) *Use the information already present*: inverse dynamics reads a 0.25° step from the
realized transition at AUC 0.913 in run 9's own latent space, where its forward model scores 0% — so
this is forward-side gradient allocation, and the measured lever order is input representation first
(Fourier 60.6% A-wins vs the raw scalar's 50.0% = exactly chance), weighting last or never (boundary
weighting is null-to-negative in all three pairings). Addressed by 10B-fourier vs 10B-ctrl.
(2b) *Fair measurement*: the old gate commanded 2–8× the largest training-window excursion; the ramp
gate delivers the same 1.0° total using only ±0.25° transitions, in the ramp shape training uses.
Addressed by P0-3/B1. **XTCAV specifically:** polarity keeps the composition-invariant α; but E331
contains **zero** polarity flips and **zero** TCAV on↔off transitions, so the off-endpoint cannot be
learned causally from any current data — that is a DAQ item, not a config.

**Goal 3 — proprio fidelity.** Root cause is that the encoder never trains — verified again in run
9's own config, which has no `latent_loss_weight` key on `modalities.0`. Stage A trains it where the
instability that caused the original revert cannot occur; stage B freezes it trained. Gated by
A4/A5. Side benefit: recovers 8.5% of the model's parameters from doing a fixed random projection.

---
#### NOT IN THIS PLAN, AND WHY

Action-normalizer rescale — **refuted**, run 9 self-compensated it (2.7× larger rel_cond, 0%
response). Boundary-group α — measured null-to-negative before the input fix; revisit only if B1
moves. Capacity/width — §8.21 withdrawn; stage A recovers 8.5% for free. `p_tf < 1` — cannot touch
dynamics drift, `loss_terms` always teacher-forces. Curriculum E300→combined — no measurement
indicates it. Dropping E331 — it holds the only well-powered L2 supervision AND the only down-steps,
without which the ramp gate has no symmetric arm. `data.subsample` / re-block — deferred to a corpus
change after run 10, and blocked on the S1 aggregation bug.

#### PRE-REGISTERED FALSIFIERS
- **F1** 10B-fourier ≈ 10B-ctrl on B1 → input representation is not the constraint; escalate to
  adaLN/FiLM action modulation (the action block carries ≤9–18% of the conditioning norm).
- **F2 (most likely)** B2 passes, B1 and B3 fail → response is large-signal-only, an artifact of the
  0.25° monoculture. **Next action is DATA, not modelling.** Stated in advance so it cannot be
  re-interpreted after the fact.
- **F3** `v/cond` rises above 2.5 but B1 stays 0 → same escalation as F1.
- **F4** B1 improves but B4 fails → the number is unmeasurable; latent multi-step consistency loss
  becomes the blocker (~1 day of code).
- **F5** Stage A fails A1 → cross-attention at bottleneck resolution is not sufficient either;
  re-derive from the §8.31 layout table (8×24 gives 62% extraction, 183 µm — the next rung down).
- **F6** B1 passes on E300 rows but fails on E331 → pathway is corpus-local; revisit a per-run
  working-point embedding.

#### STANDING DAQ REQUESTS (lead time; outside the modelling path)
1. **Randomized L2 step size** — ±0.25 to ±2°, both directions, interleaved. The only thing that
   makes a response *function* learnable and the OOD rungs in-distribution.
2. **Randomized TCAV polarity**, and explicit **on↔off transitions** — no corpus run contains an
   on↔off transition; E331_16035 is a single all-off run with anchors but no transition.

### §8.38 WHY THE RAW SCALAR SCORES EXACTLY CHANCE: THE ACTION VECTOR IS 96% NOISE (2026-08-30)

Ryan: "why does raw scalar produce 50%? this really seems to indicate that some info is getting lost
in the action input." Measured on the combined v6 corpus against its own
`normalization_stats.json`. The instinct is correct and the cause is not the scalar's expressiveness.

**The action path is z-score → symlog → MLP** (`Normalizer.norm_act` → `FourierMLP.forward`, which
applies symlog BEFORE the MLP; with `action_fourier_freqs=0` there is no Fourier branch). Normalizer
std = [L2 7.014, S 21.193, dev 0.351, amp 4.968].

**Median |change| in z-units per transition — what the encoder input actually moves by:**
| dim | at an L2 boundary | at an ordinary step |
|---|---|---|
| L2 | **0.0356** | 0.0000 |
| S | 0.0051 | 0.0048 |
| **dev** | **0.8758** | **0.9689** |
| amp | 0.0213 | 0.0206 |

**`dev` redraws itself from its entire distribution every single shot.** Its per-step change is
0.966 z against a full spread of 1.000 z, and its lag-1/2/5 autocorrelation is
**−0.044 / −0.023 / 0.010** — i.i.d. white, shot to shot.

**Consequences, all measured:**
1. At an L2 boundary the 4-D action vector moves by **54.9% of its own norm**, and the commanded knob
   is **3.7% of that displacement**. The other 96.3% is `dev`.
2. Variance ratio at the boundary: 0.966² / 0.0356² = **743×**. The informative channel arrives at
   1/27th the amplitude of a channel that predicts nothing.
3. **symlog is exonerated**: at 0.0356 z it maps to 0.0350 — near-identity, loses nothing. The
   squash is not the problem.

**`dev` is physically inert.** dev = |phase| − 90 with a measured physical range of ±1.0°, so it
modulates the streak by sin(89°)/sin(90°) = **0.99985 — a 0.015% effect**, against the 2–4× streak
variation §8.26 documented. It is a real measurement of something that does essentially nothing to
the beam, injected at full variance.

**And it is exactly redundant.** corr(S, amp·sin(|dev|+90)·sign(S)) = **1.0000** — S is an exact
function of amp and dev, so the TCAV state is already carried by the smooth composite coordinate the
v5 schema introduced to fix the polarity discontinuity. `dev` was kept alongside it and contributes
only its noise. (Its sign is not recoverable from S/amp, but sin is flat there, so that is
physically meaningless information.)

**What this does and does not claim.** Information is NOT destroyed in the information-theoretic
sense — the dims are linearly separable and a first Linear layer *could* zero dim 2. What is
destroyed is **learnability**: the action input's SNR at initialisation is **0.037**, the network must
learn to null a 743×-variance nuisance channel before it can exploit the signal, and there is little
gradient pressure to do so because the action block ends up carrying only 9–18% of the conditioning
norm anyway (§8.34 M-G). That chain terminates exactly where M-J measured it: dpred 0.005 and A-wins
50.0%. Note the action is not *ignored* — 0.005 is nonzero — its effect is simply smaller than the
residual it must explain.

**This also explains M-J's two puzzles.** Fourier is the only above-chance arm because it AMPLIFIES:
30× at f_max = 16 takes the SNR from 0.037 to ≈1.1. And the Δ-action channel barely helped (51.2%)
because it was carried in setpoint units, where ΔL2 = 0.25° is still ~0.036.

**A cheaper and larger lever than Fourier, from the same measurement:**
- std(ΔL2) over all transitions = **0.0481°**, so a 0.25° step normalized by its OWN delta statistics
  is **5.19 units** vs 0.0356 through the setpoint normalizer — **146× amplification**, ~5× more than
  Fourier's 30×, and interpretable rather than a frequency ladder.
- **Drop `dev`** (redundant, inert, 96.3% of the displacement noise) → the nuisance channel is gone
  rather than suppressed.

**Status: diagnosis is measured, the causal claim needs the probe.** That `dev` is white, inert and
96.3% of the displacement is established. That it CAUSES the 50% is inference. The decisive test is
cheap — rerun the M-J forward-learnability probe (581 train / 160 held-out, ~10 GPU-min) with arms:
(i) raw as-is, (ii) `dev` zeroed, (iii) ΔL2 added in its own units, (iv) both, (v) Fourier as the
reference. **Run this BEFORE fixing run 10's action-side config**, because if (iv) beats (v) the
right change is a corpus/schema fix, not the Fourier ladder — and that reverses §8.37's E2 arm.
Caveat: dropping dev and adding ΔL2 changes `action_dim` and the normalizer, i.e. a new corpus
version and checkpoint-incompatible, which is exactly why it must be probed before it is built.

### §8.39 M18 PROBE: THE DELTA CHANNEL WORKS, BUT A-WINS COULD NEVER HAVE SHOWN IT (2026-08-30)

Ran the §8.38 probe. `wizard`-side scratch `scratchpad/m18_action_repr.py`; results
`m18_results.json`. Head/optimiser/split/early-stopping are m17's verbatim; **3 seeds per arm**
(m17 ran one). First, a construction fact that invalidates part of §8.38: **m17 reconstructed action
dims 1–3 from constants and set `dev` ≡ 0**, so its "raw" arm never contained the noise channel at
all. This probe recovers the REAL 4-D rows from the corpus (alignment asserted against m15's stored
L2 column) and tests raw-with-dev for the first time.

| arm | A-wins | 95% CI | dpred | **cos(Δact, resid)** | β | RMSE/copy |
|---|---|---|---|---|---|---|
| ctrl (action zeroed) | 0.0% | — | 0.000 | +0.0000 | — | 0.853 |
| **raw_true** (what runs 5–9 eat) | 51.5% | [47.0, 55.9] | 0.004 | −0.0004 ± 0.0160 | 8.07 | 0.845 |
| raw_nodev (= m17's "raw") | 49.4% | [44.9, 53.8] | 0.003 | +0.0021 ± 0.0121 | −5.41 | 0.842 |
| **dl2** (ΔL2 in its own units) | 47.1% | [42.6, 51.5] | **0.155** | **+0.0972 ± 0.0146 (6.7σ)** | 0.46 | 0.837 |
| **dl2_nodev** (§8.38 proposal) | 51.0% | [46.6, 55.5] | 0.140 | **+0.0898 ± 0.0131 (6.9σ)** | 0.50 | **0.831** |
| f8 (Fourier, f_max 16) | 53.1% | [48.7, 57.6] | 0.018 | +0.0204 ± 0.0149 (1.4σ) | 0.97 | 0.847 |
| f8_nodev (= m17's "f8") | 53.5% | [49.1, 58.0] | 0.014 | +0.0133 ± 0.0153 (0.9σ) | 1.24 | 0.843 |

Measured delta units: std of the physical action delta = [0.0452, 3.3471, 0.509, 2.0561], so a 0.25°
step is **5.54 delta-units vs 0.0356 setpoint-z — 155×**.

**1. RETRACTED — §8.38's causal claim that `dev`'s noise produces the 50%.** raw_true 51.5% vs
raw_nodev 49.4%, and dl2 +0.0972 vs dl2_nodev +0.0898: removing `dev` changes nothing, in either
metric, in either pairing. `dev` remains white, physically inert (0.015% streak effect) and exactly
redundant (corr 1.0000) — those measurements stand — but it is **not what blocks the action pathway**.
Dropping it is hygiene, not a fix. The audit (§8.40) independently corroborates the inertness by a
different route: `dev`'s max |partial r| with any of the 138 obs channels is 0.015.

**2. RETRACTED — M-J's headline that Fourier is the only above-chance treatment (58.1%, p≈0.04).**
It does not replicate. Three seeds give 50.6 / 50.6 / 57.5% (f8) and 54.4 / 50.6 / 55.6% (f8_nodev),
pooled 53.1% and 53.5% with CIs spanning 50%. **The seed-to-seed spread (~7 points) is larger than
the claimed effect**; m17's p-value was a single-seed artifact. §8.37's E2 arm — "Fourier is the one
change to launch first" — loses its evidential basis, and so does M-J's ordering claim ("input
representation first, then weighting"), since the weighting arms were scored the same way.

**3. THE METRIC WAS THE PROBLEM — again.** A-wins asks whether the true action beats the held action
in *total MSE*. Every arm's RMSE/copy sits at 0.831–0.853 including ctrl at 0.853, i.e. **the action
explains at most ~2% of the boundary residual variance**. A binary win/loss on total MSE at n=160
cannot resolve a 2%-of-variance effect — so 50% was structurally guaranteed regardless of the input
representation. This is the **third metric failure in this campaign** (staircase → §8.27
counterfactual; now A-wins → direction-aware), and the same shape each time: a statistic dominated by
something other than the quantity of interest.

**4. WHAT ACTUALLY WORKS, on a metric that can see it.** Project the action-induced change in the
prediction, Δact = pred(a_true) − pred(a_hold), onto what the counterfactual prediction misses,
resid = target − pred(a_hold). **The ΔL2 channel in its own units gives cos = +0.097 at 6.7σ**, while
raw is 0.0σ and Fourier 1.4σ. dpred rises 35–40× (0.004 → 0.155). RMSE/copy also improves most for
dl2_nodev (0.831 vs 0.853 ctrl). So the delta channel genuinely makes the model's action response
point the right way — Fourier does not, at any distinguishable level.

**5. CAVEAT, stated: the effect is directionally right but small and mis-scaled.** cos = 0.097 is a
weak alignment, and β ≈ 0.46–0.50 means the optimal gain on the response is about half what the model
produces — it **over-responds by ~2×** in magnitude while being only weakly aligned in direction.
This is a real signal to build on, not a solved problem, and it does not by itself predict that the
gate will pass.

**Consequences for §8.37.** The action-side change for run 10 becomes **a ΔL2 channel normalized by
std(ΔL2), not the Fourier ladder**. `fourier_fmax` should still be exposed (P0-1) because §8.40
finding 3 needs it on the OBS path, but it is no longer the action-side treatment. Any future
action-representation scoring must use the direction-aware statistic; A-wins is retired.

### §8.40 NORMALIZATION/SAMPLING AUDIT: THE CHECKPOINT METRIC IS BLIND TO THE KNOB (2026-08-30)

Audit agent, CPU/data-only, combined v6 corpus + r9's resolved config and `best.ckpt`. Scratch in
`scratchpad/audit/`. Ranked; the top finding is larger than anything in §8.38–§8.39.

**F1 — `best.ckpt` is selected by a metric that provably cannot see the L2 response. [VERIFIED.]**
`train_world_model.py:248` monitors `val/metric/proprio/{checkpoint_metric}` — the prefix is
**hardcoded `proprio/`** — and `environments/base.py:143` defines the default as full-observation L2
error on the proprio vector. **The image never enters checkpoint selection.** Partial correlation of
each of the 138 obs channels with the commanded L2, computed WITHIN episode with a quadratic time
trend removed:
| | |
|---|---|
| max \|r\| over 138 channels | **0.068** (`BPMS:LI20:3156:Y`) |
| channels with \|r\| > 0.1 | **0 / 138** |
| obs variance explained by L2 | **0.0359%** |
| `DTOTR2_COM_X` / `COM_Y` | +0.021 / −0.020 |
The same knob on the IMAGE observable (band-median separation, 2,764 valid TCAV-on eval shots,
within (episode, TCAV-sign) strata): **73.4% of separation variance is L2-driven**, per-shot SNR
1.66, L2-driven range 443 µm vs per-shot sd 199 µm. So the response is strong in the image and
absent from the proprio vector — and only the proprio vector picks the checkpoint. This plausibly
explains why hand-picked snapshots (r7 ep115) have outperformed `best.ckpt` all campaign.
*Corroboration for §8.38/§8.39 by an independent route:* `dev`'s max \|r\| with any obs channel is
**0.015**, `amp` 0.019, while `S` reaches **0.842** — only S couples to the machine at all.
**Fix:** put image-derived observables (band separation, per-band charge fraction, streak σ — the
extractor already computes all three) into the obs vector and/or the checkpoint metric.

**F2 — 28% of the obs vector is one quantity; half the gradient is RF shot noise.**
| group | n | share of input var | share of *irreducible* residual |
|---|---|---|---|
| BPM position | 60 | 43.5% | 27.4% |
| **TMIT (charge)** | **39** | **28.3%** | **2.3%** |
| **RF klystron readbacks** | 24 | 17.4% | **49.4%** |
| validity flags | 3 | 2.2% | 6.6% |
| **DTOTR2 COM** | **2** | **1.45%** | **0.63%** |
The 39 TMIT channels have top-PC share **0.9636** and participation ratio **1.08** — one physical
quantity in 39 unit-variance copies. Whole-vector participation ratio **3.43**; 20/29/47 dims carry
90/95/99% of variance; **337 channel pairs above |corr| 0.98**. The asymmetry is the defect: charge
is 28.3% of the input but 2.3% of the learning signal, RF readbacks are 17.4% of the input but half
the gradient, and the only channels describing the beam on the screen being modelled are 1.45%.

**F3 — the proprio Fourier ladder converts a structured input into a majority-white one.**
`fourier_freqs=16` at the hardcoded `f_max=100` gives 138 raw + 4,416 sin/cos. Measured within-episode
lag-1 autocorrelation: raw z **+0.356**, band 7 (f=8.6) +0.118, band 15 (f=100) **+0.012**.
**2,652 / 4,416 fourier features (60.1%) are shot-to-shot white = 58.2% of the encoder's input
width**, while the 138 structured channels are 3% of it. Fix: `f_max ≈ 8`, or drop the top 6 bands.
Same measurement that sets the action-side ladder — one code change (P0-1) serves both.

**F4 — two obs channels are i.i.d. DAQ-dropout flags (the `dev` defect, replicated).**
`PMTR:HT10:950:PWR [validity]` and `LASR:LT10:930:PWR [validity]`: **identical on 99.81% of shots**
(corr 0.9908), valid fraction 87.98%, invalid stretches of length 1 in **98.7%** of cases, 1-step
linear **R² = 0.0203 / 0.0204 — the two least predictable of all 138 channels** — and 6.6% of the
irreducible residual. Each is z-scored to unit variance. Physical inertness is asserted from channel
semantics + R², not an independent physics test.

**F5 — `BLEN:IN10:596:BRAW` is 88% imputed and 2.76×-amplified.** Valid fraction fell **43.62%
(E300) → 11.79% (combined)**; nan_frac 0.882 survives `XTCAV_MASK_MAX = 0.90` by 0.018. Obs are
z-scored TWICE (processor, then `Normalizer`), and because 88% of rows are the imputed exact 0, the
second std is 0.362 — real values re-enter at **2.76×** and 1.8% of shots exceed the squash clamp.
This is the §8.24-point-5 artifact channel, now much worse. Physical importance not established.

**F6 — `_proj_median` quantizes the separation observable to 61 µm.** It returns
`float(np.searchsorted(...))`, an integer pixel index. All 2,764 `sep_um` values are exact multiples
of 61.0 (132 distinct values). Per shot it is harmless (310 µm² vs 57,685 µm² of real jitter), but
**100% of the 105 (episode, L2, sign) cell medians land on the grid** — a median of quantized data
does not average it away. |integer − interpolated|: p50 **15.0 µm**, p90 30.9, max 54.1 — 0.41× those
cells' sampling SE. Fix: interpolate the 50% crossing, ~3 lines.

**F7 — the mp4 hop is lossy on a scientific image.** Direct proof: the processor executes
`crops[crops < 2] = 0`, so value 1 cannot exist in the source, yet **1.47% of decoded pixels are
exactly 1**. Gen-2 round trip: RMSE 0.228 u8 overall, **0.842 u8 on beam pixels** (~3% relative),
0.93% of true zeros become nonzero, total frame signal drifts −1.36%/generation. Low rank given the
1.66 per-shot SNR, but it is free to fix (ffv1 or npy).

**NEGATIVE RESULTS — checked and clean (valuable; these close off hypotheses).**
1. **`NOISE_U8=5` / `XTCAV_FLOOR_ZERO_U8=2` do NOT discard beam tail.** Sweeping the threshold over
   {0,2,5,8,12} on 3,000 eval frames: corr ≥ **0.9986** with the deployed setting, median Δ = 0.00 µm
   in every arm. The discarded mass is background — only 1.9% of u8∈[1,5) mass lies within 3 px of a
   core pixel vs 3.8% of frame area, i.e. *less* concentrated near the beam than chance. **This
   retires the "chopped-off tail is a thresholding artifact" hypothesis; §8.31's decoder verdict is
   the remaining explanation.**
2. `XTCAV_SCALE=2000` is not clipping (zero pixels at 255).
3. The Fourier `squash=4.0` clamp is nearly inert (0.3675% of entries).
4. Image MSE is not background-dominated: pixels lit <1% of the time are 62.7% of area, 0.36% of
   per-pixel variance.
5. Cross-run PV intersection is clean — E300-only and 17-run corpora carry the SAME 138 channels.
6. `_subsample_episodes` is inert at `data.subsample=1` (r9's setting), so §8.36 S1 never bit. It
   stays a live hazard for any future stride change.
7. **The action token's small norm is exonerated.** Squared norm 5.6 vs 128 per `_ln`'d state token,
   but 3.03 at random init (so not a learned collapse), and `SpaceTimeBlock` is pre-norm — the action
   token is renormalized before attention. The magnitude relation is discarded, but no suppression
   claim survives.
8. Near-constant BPM channels are real structure (1-step R² 0.908), not amplified dither.
9. Imputation is not what makes the flag channels noisy (R² 0.178→0.196 restricted to valid pairs).

**Ordering.** F1 first and alone — nothing else changes what `best.ckpt` optimises for, and it is
cheap. F2/F3/F4 are one corpus-schema change together (obs_dim shrinks, normalizer changes,
checkpoint-incompatible) and must be probed with the §8.39 direction-aware statistic before being
built. F6/F7 are cheap and independent.

### §8.41 SYNTHESIS: WHY THE RESPONSE PREDICTION FAILED (2026-08-30)

Ryan asked for the mechanism, now that the single-cause hypotheses are all refuted. It was never one
bug. Five independent attenuations compound, each measured.

**THE ROOT PHYSICAL FACT — one commanded step is below the machine's own noise.** From
`measured_all.json` (939 usable (run, sign, L2) cells, per-shot sd recovered as se·√n; the grid is
0.25°):
| | |
|---|---|
| per-shot separation sd | **254 µm** |
| per-0.25°-step \|Δsep\| | **152 µm** (p25 61, p75 305) |
| **per-step SNR** | **0.60** |
| 4-step ramp | 610 µm → **SNR 2.40** |

**This reconciles the two numbers that look contradictory.** §8.40 F1 measures 73.4% of separation
variance as L2-driven; §8.39 measures the action at ~2% of the one-step latent residual. Both are
right: the L2 response is a STRONG effect integrated over a scan and a **0.6σ effect per step** — and
the model is trained, scored and gated per step. The signal accumulates over the 6-step episode ramp;
any single increment is buried.

**The five attenuations, in causal order:**
1. **Physics** — per-step SNR 0.60 (above). Not a defect; the operating point.
2. **Target dilution** — the action explains ~2% of the boundary residual variance (RMSE/copy 0.853
   with the action zeroed vs 0.831–0.845 with it, §8.39). A perfect action encoder buys ~2% MSE, so
   gradient descent has almost no incentive to build the pathway.
3. **Input resolution** — the step arrives at 0.0356 z (§8.38). Real, but **secondary**: amplifying it
   155× via the delta channel yields only cos = 0.097 alignment (§8.39). This is where my §8.38
   framing was wrong — resolution was never the binding constraint.
4. **Selection blindness** — `best.ckpt` monitors `val/metric/proprio/...`, and L2 explains **0.0359%**
   of proprio variance with 0/138 channels above |r| 0.1 (§8.40 F1). Even if training produced a
   responsive model, the saved checkpoint is not selected for it.
5. **Measurement blindness** — both instruments were broken. The staircase scored context-copying
   (76–105% for a pure copier, §8.27). A-wins is a binary MSE comparison on a 2%-of-variance effect,
   so 50% was structurally guaranteed regardless of input (§8.39).

**Why every single-cause hypothesis kept failing.** Normalizer scale (§8.34 M-G3), capacity (§8.21),
corpus composition (§8.27), `dev` noise (§8.39), Fourier representation (§8.39) — each was a real
observation and none was sufficient, because no one of them was the bottleneck. Removing any single
attenuation leaves the other four. That is the pattern to expect from the remaining fixes too.

**What each planned fix actually buys, stated honestly:**
- Ramp gate (§8.36 V1): attacks (5). SNR 0.60 → **2.40** by accumulating 4 in-distribution steps.
  This is the largest single measurable improvement available and costs no training.
- Image observables / checkpoint metric (§8.40 F1): attacks (4). Cheap, isolated, and until it lands
  every run selects on a criterion blind to the knob.
- ΔL2 channel (§8.39): attacks (3). 6.7σ directional alignment, but cos 0.097 and β ≈ 0.5 (the model
  over-responds ~2× while weakly aligned). Real, partial.
- Decoder cross-attention (§8.32): attacks the *measurement* floor — the codec's 244 µm roundtrip
  error is comparable to the machine's own 254 µm shot noise, i.e. the renderer roughly doubles the
  noise on every model-side separation.
- **Nothing in the plan addresses (2).** Target dilution is set by the step size the DAQ commanded.
  That is what the randomized-step-size request is for, and it is why §8.37's F2 remains the most
  likely outcome.

**r7 ep115, restated in this frame.** Its +39.2% lives at δ = ±0.5–2°, where the accumulated signal
clears the noise; the flip loss built a nonlinear large-signal mode (v/act 0.49 → 1.92). At the
in-distribution 0.25° step, where SNR is 0.60, it is indistinguishable from every other checkpoint.

**§8.41 addendum — is the jitter averageable? Partially. [MEASURED]**
(`scratchpad/audit/jitter_structure.py`; 201 (run, sign, L2) cells with ≥12 shots, per-shot
extraction on real frames, residual = sep − cell median. Median within-cell residual sd **207 µm**.)

Shot-to-shot autocorrelation of the residual at a FIXED setpoint:
| lag | r | n pairs |
|---|---|---|
| 1 | **+0.227** | 4412 |
| 2 | +0.157 | 4196 |
| 5 | +0.075 | 3552 |

So the jitter is **not white** — it carries a slowly-decaying correlated component. How a k-shot mean
actually averages down, normalised to the k=1 pooled sd of 268 µm:
| k | measured sd | white-noise ideal | penalty |
|---|---|---|---|
| 1 | 268 | 268 | 1.00 |
| 2 | 205 | 190 | 1.08 |
| 4 | 169 | 134 | 1.26 |
| 8 | 123 | 95 | **1.30** |

**Consequences.** (a) Averaging still works, at ~1.3× worse sd than the √k ideal by k=8, i.e. **~1.7×
more shots than white-noise math predicts** — §8.36 V4's n ≥ 750 should be **n ≈ 1300**. (b) The
correlated component makes the paired ±-arm design MORE valuable, not less: correlated jitter is
common to both arms and cancels in the difference, which is exactly what the ramp gate does. (c) It
does NOT rescue the training problem — a correlated nuisance is still a nuisance in a per-sample
gradient, and averaging helps an estimator, not a gradient.

**The two consequences of dilution are separable and only one is cheap.**
- *Measurement* dilution: fixed now, for free — accumulate 4 in-distribution steps (SNR 0.60 → 2.40)
  and average ~1300 pairs. No training required.
- *Learning* dilution: NOT fixed by averaging. Every training sample's gradient is dominated by
  jitter and the action is ~2% of the target variance. This is the residual hard problem, and it is
  set by the commanded step size — the randomized-step DAQ request is the only lever on it.

For scale: the codec's 244 µm roundtrip error (§8.31) against the machine's 207 µm within-cell jitter
means the renderer inflates the per-shot noise by ~1.55× in quadrature, on top of everything above.

### §8.42 THE LOSS UNDERWEIGHTS THE PHYSICS BY ~100× — AND THE RAMP GATE IS RETRACTED (2026-08-30)

Three measurements on the cached r9-encoder bags (`scratchpad/audit/kstep_share.py`,
`residual_vs_absolute.py`), run to design the next suite. Two proposals of mine die; the survivor is
a different change than anything in §8.37.

**1. Longer targets do NOT concentrate the action. [MEASURED]** Fraction of the k-step latent-target
variance linearly explained by the cumulative commanded ΔL2 over those k steps:
| k | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| R² (windows containing a step) | 0.196% | 0.229% | 0.203% | 0.163% | 0.070% | 0.078% | 0.127% |
| median cumulative \|ΔL2\| | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.50 |
**The action's share is flat at ~0.1–0.2% at EVERY horizon from 1 to 64 frames.** The reason is in
the last row: the setpoint changes once per ~27 frames (936 steps / 25,227 transitions), so even a
64-frame window accumulates a median of only 0.50°. Signal grows with the number of steps in the
window, but so does drift, and the ratio never improves.

**2. RETRACTED — the §8.36 V1 cumulative-ramp gate.** It commands 4 × 0.25° in 4 consecutive model
steps = 1.0° in 4 frames, while training contains at most **0.50° in 24 frames**. I fixed the
step-MAGNITUDE extrapolation and introduced a step-RATE extrapolation ~12× beyond the data — the same
error I criticised in §8.34, one axis over. Episodes are monotone 6-step ramps in *scan* time, not in
frame time, and I conflated the two.
**Replacement:** a single in-distribution ±0.25° step (in distribution in BOTH magnitude and rate),
h ≤ 3 per §8.34 M-C's drift budget, paired, averaged over n ≈ 1300 (§8.41 addendum). Power check:
per-instance SNR 0.60, correlation penalty 1.3 → mean SNR ≈ 0.60·√1300/1.3 ≈ **16σ**. Comfortably
resolvable; no ramp needed. (Subsampling to the scan timescale is also dead as a fix — row 1 shows
k≈27 is the WORST horizon, 0.07–0.08%.)

**3. THE ACTUAL DEFECT — the loss underweights the physics by ~100×. [MEASURED]**
| target | predictor | R² (all) | R² (moving) |
|---|---|---|---|
| absolute state z_t | L2_t | **1.211%** | 1.372% |
| transition z_t − z_{t−1} | ΔL2 | 0.010% | 0.198% |
| context-residual (ridge on z_{t−1} removed) | ΔL2 | 0.015% | **0.313%** |
| — | **context alone explains** | **72.9% of the state** | |

Read against §8.40 F1's **73.4% of separation variance is L2-driven**, the resolution is:
**the L2 response is a LOW-DIMENSIONAL direction in a 4224-dim latent.** L2 explains 73% of the
separation but only 1.2% of the latent state, because the bag encodes beam shape, position,
background and noise as well. The training loss is a **uniform MSE over all 4224 dims**, so the
separation direction — the only thing the campaign measures — receives on the order of **1% of the
gradient**, and the action's incremental contribution beyond context is **0.313%** against a context
shortcut worth 72.9%. That is a **~240:1 shortcut-to-signal ratio**, and it is a property of the
LOSS, not of the data or the architecture.

This finally explains the campaign's persistent shape: the models are good image predictors and bad
response predictors because that is exactly what the objective asks for. It also explains why every
input-side fix (normalizer, Fourier, ΔL2 channel, dropping `dev`) produced small or null effects —
they improve what the action *can* contribute to a target in which it is worth 0.3%.

**Switching `predict_residual` to absolute-state prediction is NOT the fix**: it raises the action's
share only 0.2% → 1.4%, still negligible, and it makes the context shortcut *easier* — the failure
mode §8.27 already retracted a result to.

**The fix that follows: put the physics in the supervised target.** Add image-derived observables —
band-median separation, per-band charge fraction, per-band streak σ (the extractor already computes
all three) — to the obs vector, so they are decoded and supervised directly rather than left as a
~1% direction inside an image MSE. This single change serves three ends at once:
- it re-weights the gradient toward the quantity the campaign measures (this §);
- it fixes checkpoint selection, since `best.ckpt` monitors the proprio vector (§8.40 F1);
- it gives the proprio head something L2-coupled to predict, where today max |partial r| over all
  138 channels is 0.068.
Caveat, stated: it does NOT change the per-step physical SNR of 0.60 (§8.41). It changes what
fraction of the GRADIENT reaches the physics direction. Those are different problems and both are
real; this addresses only the second.

### §8.43 THE PHYSICS TARGET — DESIGN AND GRADIENT ANALYSIS (2026-08-30)

Concrete form of §8.42's fix. The gradient analysis is what picks the design: the obvious
implementation is inert.

**THE THREE GRADIENT PATHS (verified in code).**
| path | route | trains | reaches `act_enc`? |
|---|---|---|---|
| decode loss | pred bag → `decode_loss` | decoder only | **NO** — `pred_obs_in_loss=false` detaches (`lit.py:131`) |
| roundtrip anchor | obs → enc → bag → dec | encoder + decoder | **NO** — no flow in the path |
| **dynamics flow loss** | act → `act_enc` → `a_emb` → backbone → `h_state` → `flow.loss` | backbone + **act_enc** | **YES** |
In `loss_terms` the context `s` is detached (`dynamics_detach_encoder`) and `target` is detached, but
`a_emb = self.act_enc(a_ctx)` is **not** — so the flow loss is the ONLY route to the action pathway.

**⚠ THE TRAP: putting the physics observables in the 138-D obs vector does nothing for response.**
They would be supervised by `decode/proprio`, which sees a DETACHED predicted bag. That trains the
proprio decoder to read separation off a bag and delivers **zero** gradient to the dynamics or the
action encoder. It fixes §8.40 F1 (checkpoint selection) and nothing else. The physics must enter
the **flow** loss.

**THE DESIGN.**
1. **Compute offline, at conversion.** Run the existing extractor per frame in `processors.py`;
   store band-median separation (µm), per-band charge fraction, per-band streak σ. They are needed
   only as TARGETS, so **no differentiable extractor is required** — this removes a whole class of
   complexity. Carry the extractor's `ok` flag and **mask the loss on invalid rows** rather than
   imputing a value (§8.40 F4/F5: impute-plus-mask is itself a documented hazard here).
2. **Its own modality, not appended to proprio.** `VectorModality` is hardcoded to **1 token**
   (`modalities.py:127`), so a physics modality adds one token: layout 33 → **34**. Appending to the
   138-D proprio vector buries it inside a shared token where it cannot be weighted separately.
3. **Per-token flow weighting — ALREADY SUPPORTED, no `flow.py` change.** The dynamics flow has
   `event_dims=1`, so `per = (v-u)².mean(last dim)` has shape **(B, L−1, n_state)** and the assert
   `w.shape == per.shape[:w.ndim]` accepts a per-TOKEN weight. Today's flip weights are (B, L−1) and
   broadcast over tokens (`flow.py:117`); a (B, L−1, n_state) weight upweights the physics token
   directly. Uniform weighting gives it 1/34 = 2.9%; for a 30% share use w ≈ 14, expressed in the
   composition-invariant α form of §8.16 so it survives bag-size changes.
4. **`latent_loss_weight` > 0 on it — MANDATORY, not optional.** `VectorModality` defaults it to
   **0** (`modalities.py:129`), which is precisely the §8.25 bug that froze the proprio encoder at
   random init for five runs. With `dynamics_detach_encoder=true` the roundtrip anchor is the only
   gradient path to any encoder. Repeating this omission would silently make the physics token a
   random projection.
5. **Point the checkpoint monitor at it.** The `proprio/` prefix is hardcoded
   (`train_world_model.py:248`); small code change.

**WHY THIS IS ~85× BETTER, quantitatively.** The action's share of the target is what starves the
pathway (§8.42: 0.313% of the context-residual in the raw 4224-dim latent). In the SEPARATION
coordinate the same step is worth SNR 0.60 (§8.41), i.e. a variance share of
0.60²/(1+0.60²) = **26%**. Upweighting a token whose target is action-rich is therefore worth about
**85×** more gradient-per-unit-loss-mass than upweighting anything in the raw latent.

**COROLLARY — boundary weighting is reopened.** §8.34 M-J measured boundary loss weighting as
null-to-negative, and §8.39 then showed that whole probe was underpowered. But the deeper reason it
could not work is now clear: it reallocated gradient toward a target in which the action is worth
0.3% regardless of weighting. **In an action-rich target it should work**, so M-J's null does not
transfer to this design.

**TWO REAL BENEFITS BEYOND GRADIENT WEIGHT.**
- **Training targets never touch the codec.** They are extracted from REAL frames, so §8.31's 244 µm
  roundtrip error — which is comparable to the machine's own 207 µm jitter — is absent from the
  training signal entirely.
- **The response gate can read the predicted physics token directly**, with no renderer in the loop,
  removing the codec floor from the measurement too. Keep the image-based gate as a cross-check;
  do not replace it, since a trained token can be a copier.

**RISKS, STATED.**
- **Context shortcut.** Putting separation in the context makes copying easier, and context already
  explains 72.9% of the state (§8.42). This does not invalidate the design — the paired counterfactual
  gate is built to defeat copying — but it means the training change must NOT be scored by anything
  other than that gate.
- **Optimisation shock.** A heavily upweighted single token is the flip-loss situation (w=25 was a
  large perturbation, §8.34). Use the α form, ramp it, and keep the existing `grad/norm/flow` kill
  criterion armed.
- **It does not change the physics.** Per-step SNR stays 0.60. This changes what fraction of the
  GRADIENT reaches the physics direction, not what the machine delivers. §8.41's two dilutions remain
  distinct and only the second is addressed here.

### §8.44 IMPLEMENTATION — WHAT IS DONE AND VERIFIED (2026-08-30)

All defaults preserved: every change below is bit-identical to runs 5–9 unless explicitly enabled.

**DONE + VERIFIED**
1. **`fourier_fmax` exposed** (§8.34 M-H, §8.40 F3). Threaded `features.fourier_freqs` →
   `FourierMLP(f_max=)` → `MultiModalSequenceModel(action_fourier_fmax=)` → both subclasses →
   `setup.py`, plus `ModalitySpec.fourier_fmax` → `VectorModality`. Default **100.0** everywhere.
   Verified: default ladder bit-identical; `f_max=16` gives a top band of 16; spec path works.
2. **Checkpoint monitor un-hardcoded** (§8.40 F1). `trainer.checkpoint_modality` (default `"proprio"`
   = bit-identical) replaces the hardcoded `val/metric/proprio/…` prefix, and the resolved monitor key
   is now PRINTED at startup — a monitor key that is never logged makes ModelCheckpoint silently never
   write `best.ckpt`, which is this campaign's signature failure mode.
3. **Extractor moved to `src/quickdraw/data/xtcav_features.py`** — the single source of truth, so the
   physics modality's TRAINING targets and the EVAL gate are literally the same code. Verified
   **identical to 1e-9 on 60 synthetic two-bunch frames plus identical rejection on 2 degenerate ones**.
   `wizard/scripts/xtcav_physics_eval.py` re-exports `PX_UM/ROW_UM/NOISE_U8/QMIN_FRAC/IMIN_SUM/
   frame_grey/energy_gated_sep/_proj_median` so every dependent wizard script is untouched (verified by
   import). Adds `extract_features(frames) -> (N,6) float32, ok (N,) bool` and an **opt-in**
   `interpolate=True` sub-pixel median (§8.40 F6; default off, since every §7–§8.42 number used the
   integer version — it shifts `sep_um` by a median of 5.5 µm).
4. **`ModalitySpec.obs_slice`** — a vector modality can read a `[start, stop)` slice of the shared
   `observation_vector` instead of needing its own batch stream. This matters because the data path
   carries exactly TWO streams (`observation_vector` + one camera): a genuine third modality would
   mean a new LeRobot column, a new loader tuple and new window stacking. Slicing gives the physics
   features their **own bag token** — the entire point, since only a token's flow-loss weight reaches
   `act_enc` — at zero cost to the pipeline. Wired via a `_obs_dict` helper at both `lit.py` sites.
5. **Physics-token flow weighting** (§8.43) — `model.diffusion.physics_modality` +
   `physics_alpha` (0.0 = inert), composition-invariant closed form, composes with the flip weighting,
   with a validity mask that ZEROES the token's weight where extraction failed rather than imputing.
   **No `flow.py` change was needed**, as predicted: `event_dims=1` leaves `per` at
   `(B, L−1, n_state)` and the existing assert accepts a per-token weight.

**THE DECISIVE TEST** (`scratchpad/test_physics_grad.py`), which is what §8.43's whole design rests on:
| check | result |
|---|---|
| flow loss reaches `act_enc` | **0.0201 grad norm at alpha=0** — confirmed, not assumed |
| physics weighting re-allocates that gradient | 0.0201 → 0.0225 (**1.12×** at alpha=0.60 in a 2-token bag; the real 34-token bag reweights the token 2.9% → alpha) |
| `flow.loss` receives a per-TOKEN weight | shape **(B, L−1, n_state)** ✓ |
| realized share == requested alpha, any bag size | 0.30/0.60/0.80 exact ✓ |
| validity mask | zeroes ONLY the physics token on ONLY invalid rows (1.5 → 0.0, proprio untouched) ✓ |
| physics encoder trains **iff** `latent_loss_weight > 0` | 4 params with gradient at llw=1, **0 at llw=0** — the §8.25 trap is live and must be set explicitly |

That last row is worth stating plainly: the §8.25 frozen-encoder bug reproduces exactly for any new
vector modality. `latent_loss_weight` is not optional here.

**NOT YET DONE** — the decoder cross-attention (§8.32), the processor-side feature extraction and
append (needs a re-convert = new corpus version), the ΔL2 action channel (§8.39), the gate rework
(§8.42), and the Stage A / run-10 configs. The corpus-changing items are deliberately held until the
plan review returns, since a re-convert is the expensive, hard-to-undo step.

### §8.45 PLAN REVIEW — THE ~85× AND THE 16σ ARE BOTH WRONG (2026-08-30)

Adversarial review of §8.42/§8.43/§8.37 before the corpus-changing work. It confirmed the gradient
analysis and broke almost everything built on top of it. Load-bearing items re-verified by me.

**CONFIRMED — the gradient analysis (§8.43 claim 1).** Built a real `MultiModalFlow` and backwarded
each term separately: `decode/physics` → decoder ONLY; `codec/roundtrip_physics` → decoder + encoder;
`dynamics/latent` → `act_enc` (0.0349), backbone (0.170), flow (0.274), **no encoder**. The trap is
real. Two caveats to add: `dynamics/latent_shortcut` WOULD reach `act_enc` (r9 has `shortcut:false`),
and **the action head is a fourth live path** via `h_ctx = h.mean(-2)` when
`action_head.detach_gradient=false` (r9 has it disabled). `lambda_pred_obs` is dead code.

**RETRACTED — the ~85× (§8.43). [MEASURED against a validated pipeline.]** The comparison was
apples-to-oranges three ways: a multivariate incremental R² vs a univariate variance share; a
PER-SHOT SNR (0.60) applied to a per-TRANSITION target (the noise is a difference of two shots →
0.48, and 0.29 measured on the train corpus); and `sep_um` is 1 of 6 channels with no evidence the
others are L2-coupled. Reviewer reproduced §8.40 F1 to within a point on the eval split (72.7% vs
73.4%, n=2748 vs 2764) and then measured on TRAIN, where the gradient actually lives:
| quantity | measured |
|---|---|
| within-(episode, sign) separation variance that is L2-driven, **TRAIN** | **45.7%** (the 73.4% headline is an EVAL-split number) |
| one-step transition sd at HOLD, train | 513 µm → per-transition SNR **0.29**, variance share **7.7%** |
| context shortcut on the physics TRANSITION | **25–32%** (vs 72.9% on the raw latent — better, but see below) |
| **token weight alone** | **~8×** (~25× if one accepts the record's optimistic 26%) |
**The advertised 85× silently credited the token weight with the PRODUCT of the token weight and the
boundary weighting** — and §8.43 explicitly deferred the latter. This is the campaign's characteristic
error committed one turn after naming it. Boundary weighting is therefore not "reopened, maybe later":
**most of the advertised gain lives there**, and that must be said plainly.

**RETRACTED — the 16σ power claim (§8.42).** Overstated 3–13×. (a) 16σ is σ for detecting a
100%-of-slope response; the bars are +25% (B1) and +15% (B3), so divide by 4 and 6.7. (b) **n≈1300 is
not 1300 independent units**: `xtcav_counterfactual_gate.py` sets `N_CTX, N_DRAW = 3, 6` with `H = 8`
over a 12-point grid = **36 distinct contexts**; draws and horizons are within-context repeats on the
same frames and the same codec error, and raising them does not shrink context-level variance.
(c) §8.41's 1.3× penalty was measured on real shots at a fixed setpoint and does not describe model
draws. Empirical anchor from M-B (n≈223, CI [+14.5,−23.3] → per-unit sd ≈143 points): at n=1300
**independent** units, **+25% is 6.3σ and +15% is 3.8σ**. Quote those. "No ramp needed" survives for
B1; **B3 at +15% is marginal**. `N_CTX` must go 3 → ~110.

**BLOCKERS I VERIFIED MYSELF.**
- **`model.init_from` DOES NOT EXIST** — `grep -rn "init_from" src/ conf/` returns **0**. §8.37's
  Stage-B warm-start command would have started from random init and said nothing. The only mechanism
  is `+resume=<ckpt>`, a full Lightning resume that continues **in the Stage-A run_dir**, so the two
  Stage-B arms would clobber each other's checkpoints.
- **Adding a modality breaks the warm start anyway**: `n_input` 34→35 changes `backbone.slot_emb`
  from (1,1,34,d) to (1,1,35,d) and `load_state_dict(strict=False)` **raises**. ⇒ **the physics
  modality must be present in Stage A.** The A-vs-B attribution question is settled by force.
- **Changes (3)+(4) together produce a run with NO `best.ckpt`. [VERIFIED, `lit.py:206-219`.]**
  `rollout_metrics` is logged only on the proprio branch as `val/metric/proprio/{mk}`; a non-proprio
  head logs only `mse`/`l1`/`psnr` — and on a **`.clamp(0,1)`'d** decode, which for a z-scored physics
  vector clamps away exactly the excursions of interest and then reports a PSNR. So
  `checkpoint_modality=physics` yields a monitor key that is never logged and ModelCheckpoint silently
  never writes best.ckpt — the precise failure the comment I added two lines above it warns about.
- **12 sites dispatch on `!= "proprio"` and assume image** (`evaluation/routines.py` ×6,
  `manifold.py` ×2, `lit.py`, `controller/*`, `scripts/bench_batch.py`). `eval_ae_floor` and
  `eval_ood_horizon` would hand a `(B,T,64,192,3)` tensor to `VectorModality.encode`. Must become
  `kind == "image"`.
- **P8 — the extractor is RESOLUTION-COUPLED, so §8.43 item 1 was wrong.** `PX_UM`, `IMIN_SUM`,
  `find_peaks(distance=3)` and `gaussian_filter1d(σ=1)` are all in 64×192 px. Measured on 1811 frames:
  `sep_um` median **610 (64×192) vs 1159 (128×384)**, ratio 1.90, and a different ok-mask (97.3% vs
  98.2%). **Extract from the decoded 64×192 store, not in `processors.py`.** Consequence: the targets
  then DO carry §8.40 F7's mp4 loss and F6's 61 µm grid — "never touches the codec" is true of the AE
  only. And `interpolate` must be flipped for gate and targets in the SAME commit.
- **P4 — §8.43's "mandatory `latent_loss_weight`>0" contradicts §8.37's Stage B**, which sets vector
  llw back to 0 precisely to keep arm 1f's moving-latent instability structurally absent. Resolution
  is the one already chosen for proprio: **train the physics encoder in Stage A, freeze it in Stage B.**
- **P7 — `noise_std` is INERT for the dynamics loss**: `lit.py` noises `obs_in`, but `loss_terms` is
  called with the CLEAN `obs`. There is no config-only way to withhold the context shortcut; a
  target-only variant needs a code change masking the physics slot in `_to_input`.
- **Confound — the ΔL2 channel as a corpus column breaks everything**: `action_dim` 4→5 means a new
  corpus, a new normalizer and an `act_enc` width change, so the Stage-A checkpoint will not load.
  **Better: derive Δa INSIDE the model** from `act_seq[:,t] − act_seq[:,t−1]`, which `loss_terms` and
  `imagine_eval` both already hold. No corpus rebuild, no checkpoint break. Adopted.

**WHAT MY IMPLEMENTATION ALREADY HANDLES (§8.44), tested against the reviewer's predictions.**
| predicted failure | actual |
|---|---|
| flip telemetry crashes on a 3-D `w_step` | **does not** — the `w_lead = w_step.mean(-1)` guard works; `flip_realized_share` 0.1165 vs α=0.12 |
| physics weight silently dropped on empty-flip batches (~4.8%) | **survives** — `w_tok` is built fresh, not composed onto a possibly-None `w_step` |
| the weight is train-only, so validation runs uniform | **CONFIRMED** — `weights=None` in eval; `val/loss/dynamics/latent` will not reflect the change. Open decision. |

**THE MOST IMPORTANT NEGATIVE RESULT.** On 11,220 train transitions (310 moving), the physics-vector
transition shows **no linearly recoverable signed ΔL2 component**: incremental R² 0.0003% against a
permutation null of 0.0059 ± 0.0054%, **z = −1.0** — while an assumption-free variance-excess test
(STEP vs matched HOLD) confirms the magnitude at **148 µm, 95% CI [−73, 232]**, consistent with the
record's 152 µm. The effect is real in magnitude but its **sign structure is working-point dependent**
— which is exactly what §8.34 M-D's `|slope| ≥ 200 µm/deg` pre-registration was already saying.
**A plain MSE on the physics token will not automatically represent that**, and
`val/metric/physics/*` improving would not tell you whether it had.

**REVISED ORDER.** Phase 0 (no GPU): 0a re-measure the target's action share and replace the 85×
with the measured number BEFORE building it; 0b make the modality registry `kind`-aware and add a
vector-stream path + per-modality `pointwise_error`; 0c precompute from the 64×192 store; 0d gate
rework + re-baseline r9/r7ep115, reporting effective n = distinct contexts; 0e add a real weights-only
`model.init_from`; 0f decide whether the token weight applies at validation.
Phase A: ONE Stage-A run carrying the decoder, proprio llw, and the physics modality (`weight=0`,
llw>0, placed last), keeping proprio `fourier_fmax=100` so A4 measures one thing.
Phase B: `10B-ctrl` → `10B-tokw` → `10B-dl2`, one variable each, never `tokw` and `dl2` together.

### §8.46 THE ~8× DOES NOT REPLICATE EITHER — 0a's ESTIMATORS ARE BOTH BLIND (2026-08-30)

Re-ran §8.45's step 0a independently on the reviewer's own cached physics features
(`scratchpad/review2/phys_train.npz`: 139 episodes, per-episode `v (T,6)`, `ok`, `a (T,4)` — the
artifacts are real and were verified present). The reviewer's headline **does not replicate**, and
the failure is instructive.

**Variance-excess estimator (STEP vs HOLD transitions), d(sep_um):**
| gate | n | moving | sd hold | sd step | excess magnitude |
|---|---|---|---|---|---|
| all transitions | 17462 | 676 | 452.2 | 404.5 | **−202 µm** |
| TCAV on (\|S\|>0 both ends) | 16493 | 622 | 451.6 | 384.4 | **−237 µm** |
| on + same polarity | 16429 | 609 | 451.0 | 384.0 | −237 µm |
| on + same polarity + amp ≥ 15 | 16429 | 609 | 451.0 | 384.0 | −237 µm |
Bootstrap (2000 resamples) on the strictest gate: **95% CI [−295, −135] µm**, i.e. significantly
NEGATIVE. The reviewer reported **+148 µm [−73, 232]** — a CI that already included zero — and then
quoted the point estimate as "consistent with the record's 152 µm". Under every gate I can construct,
**step transitions are LESS variable than hold transitions.**

That sign is not a measurement of the step; it means the two populations differ in something other
than the commanded step. The likely confound is settling: §8.2/§8.26 established the cavity needs
~15–18 shots to recover, so the HOLD population is contaminated with post-step settling shots whose
variance is inflated. **The variance-excess estimator is confounded for this target and neither
+148 nor −237 estimates the step magnitude.**

**Linear estimator (incremental R² on signed ΔL2), permutation null, 200 draws:**
| set | observed | null | z |
|---|---|---|---|
| all transitions | 0.0019% | 0.0054 ± 0.0059% | **−0.6** |
| moving only | 0.0451% | 0.1200 ± 0.1270% | **−0.6** |
Agrees with the reviewer in conclusion (they got −1.0 / −1.3): **no linearly recoverable signed-ΔL2
component.** But this estimator is also the wrong instrument, for a reason already in the record —
the sign structure is working-point dependent (§8.34 M-D's `|slope| ≥ 200 µm/deg` pre-registration is
exactly that statement), and a single signed regressor cannot represent it.

**CONSEQUENCE — RETRACT the ~8× from §8.45.** It was 0.298 × 0.03 × **8%**, where the 8% came from
the variance share implied by a per-transition SNR of 0.29, which came from the +148 µm excess that
does not replicate. With the excess estimator confounded and the linear estimator blind,
**the action's share of the physics target has not been established by ANY method.** The honest
status is not "8×" and not "85×" — it is **unmeasured**.

**This is the campaign's signature failure, a third time in one day**: 0a was proposed as the step
that would catch variance-based self-deception, and it was itself run with two estimators that cannot
see the effect. Both my 85× and the reviewer's 8× rest on it.

**THE RIGHT INSTRUMENT — the one that already worked.** §8.39's M18 probe resolved an effect that
A-wins could not, by being *direction-aware* and *nonlinear*: fit a predictor of the target from
(context, action), then project the action-induced change in the prediction, Δact = pred(a_true) −
pred(a_hold), onto what the counterfactual misses, resid = target − pred(a_hold). That measured
6.7σ where a total-MSE comparison read exactly chance. Applied to the physics transition it (a) lets
the predictor learn the working-point-dependent sign from context, which the linear regressor cannot,
and (b) is paired, so the settling confound that breaks the variance-excess estimator cancels.
**Nothing may be built on the physics target until that number exists.** Until then §8.43's design is
mechanically verified (§8.44) and quantitatively unjustified.

**§8.46 addendum — does the physics TOKEN actually carry the physics? [MEASURED]**
Encoded the real extracted features through the actual `FourierMLP(6→128)` at random init, with and
without the bag's `_ln`:
| | latent_norm off | latent_norm ON (r9's setting) |
|---|---|---|
| linear read-back R² of each of the 6 features from the token | **1.000** (all six) | 0.995–0.998 |
| corr(‖Δz_token‖, ‖Δphysics‖) | **0.987** | **0.919** |
| corr(‖Δz_token‖, \|Δsep_um\|) | 0.566 | 0.550 |
So the correspondence is **by construction and verified**: the encoder's only input is the 6-vector,
a 6→128 projection is injective, and the token's latent CHANGE — which is exactly the flow target on
that token — tracks the physics change at r = 0.92 even after LayerNorm. Upweighting the token is
therefore genuinely upweighting the physics transition, not a proxy for it.

**Two honest caveats.** (1) Raw feature scales span 8000× (`sep_um` std 1238 vs `q_lo` std 0.152), so
unnormalized the token would be *entirely* `sep_um` and the charge fractions invisible — the
`obs_slice` design fixes this for free, since `Normalizer.norm_obs` z-scores every obs channel
(this resolves the reviewer's P1 normalization concern without the vector-stream work).
(2) The token carries **all six** features, so `corr(‖Δz‖, |Δsep|)` is only 0.55 — roughly **1/6 of
the reweighted gradient reaches separation**, the rest goes to charge fractions and widths. If
separation is the target, either weight the channels or use a narrower feature set; the current
6-feature token dilutes the intended effect ~6×, which is a further correction to any "×" estimate.

### §8.47 AVERAGING / DOWNSELECTION / PHYSICS LOSSES — ASSESSED (2026-08-31)

Ryan: would averaging, clean-data downselection, or physics-based loss terms help the model learn
transitions? Measured where possible (reviewer's cached features, `review2/phys_train.npz`).

**1. AVERAGING — yes, in FEATURE space, and it is the first estimator that recovers the step
cleanly. [MEASURED]** Average the EXTRACTED sep over each settled plateau, difference adjacent
plateaus; split-half within-plateau null at the same averaging depth (√2-adjusted):
| settling shots dropped | plateau steps | median \|Δsep\| | plateau-pair noise | SNR | variance share |
|---|---|---|---|---|---|
| 0 | 594 | 154 µm | 197 µm | 0.78 | 38% |
| 3 | 591 | 159 µm | 190 µm | 0.83 | 41% |
| 8 | 578 | 160 µm | 185 µm | **0.86** | **43%** |
| *(single-shot baseline)* | — | ~152 µm | 452 µm | **0.34** | **10%** |
Three consequences. (a) The step magnitude survives plateau averaging at 154–160 µm — the FIRST
estimator on the physics features that recovers the record's 152 µm/step (the variance-excess gave
−237, the linear regressor gave z≈−0.6). The signal is in the features; the per-shot estimators
were the problem. (b) The action's variance share at plateau level is **~40% vs ~10% per shot** —
a measured 4× improvement in learnability of the target. (c) The correlated-jitter ceiling is
visible: 19 shots of averaging bought 452→190 µm (2.4×), not √19 = 4.4× — §8.41's penalty, again.
**Caveats:** averaging RAW FRAMES would be actively harmful — sep jitters 3–7 px shot-to-shot,
comparable to the lobe width, so a mean frame smears the very structure the extractor measures.
Feature-space only. And a plateau-mean target is a SCAN-level response model, not the 10 Hz
shot-to-shot conditional (the stated product) — so it belongs as an AUXILIARY head/target (predict
the settled plateau mean of sep alongside the per-shot bag), not as a replacement.

**2. CLEAN-DATA DOWNSELECTION — mostly spent; one measured candidate left.** The big win was
already taken: v5's settled-only gate (35% of shots dropped) was load-bearing. Remaining headroom:
- **Downweight flat-slope working points.** 45.9% of E331 shot mass sits at \|L2\| > 8 where the
  measured slope collapses to 122 µm/deg, and E300_15671's slope is 141/38 µm/deg (§8.34 M-K).
  Boundary transitions there genuinely teach "this knob does nothing" — they dilute the ~600
  usable boundary events with counter-examples. Downweighting (not dropping — the flat region is
  real physics the model should know) by local \|slope\| is defensible and cheap.
- **Do NOT drop post-step settling shots** — they ARE the response the model must learn (the
  cavity's 15–18-shot recovery, §8.26). Tag them (a settling-age channel exists implicitly via dt)
  rather than delete. Note the plateau table above shows dropping them barely moves the
  plateau-mean estimator (0.78→0.86), so they are not the noise bottleneck anyway.
- Downselection CANNOT raise the per-step SNR at a given working point — that is machine-set.
  It only stops diluting boundary supervision. Expectations should be sized accordingly.

**3. PHYSICS-BASED LOSS TERMS — the token is the weak form; the strong form is a PAIRED
COUNTERFACTUAL LOSS, and it is the only proposal that structurally cancels the context shortcut.**
Once the physics modality exists, sep is a decoded scalar. Run the dynamics twice from the SAME
context with actions a±δ and PAIRED noise (the §8.27 gate's trick, moved into training), difference
the decoded sep, and supervise the difference against the machine's measured local slope:
`L_cf = || [sep(a+δ) − sep(a−δ)]/(2δ) − slope_meas(L2) ||`. Because both arms share the context,
**the 72.9%/240:1 context shortcut cancels exactly** — a context-copier scores zero response and
gets full gradient pressure, which no reweighting of a single-arm MSE can achieve. This directly
supervises the quantity the campaign measures.
Caveats, pre-registered: (a) **gate circularity** — this trains on `measured.json` slopes, so the
counterfactual gate is no longer independent; the gate must move to HELD-OUT runs/setpoints
(candidates: TEST_15668 + an E331 run excluded from slope supervision). (b) Only defined where the
slope is measured and \|slope\| ≥ 200 µm/deg. (c) Cost: a second rollout per training step on the
counterfactual arm (paired noise, short horizon h≤3). (d) It supervises the LOCAL derivative, not
global dynamics — keep the standard flow loss as the base term. (e) It inherits the sign problem's
solution for free: the slope carries the working-point-dependent sign explicitly.
**Ordering:** this is a Stage-B+ arm (`10B-cf`), after `10B-tokw`, because it depends on the physics
modality existing and on the gate rework (held-out split) landing first. It is also the arm most
likely to move B1, precisely because it optimizes B1's quantity — which is both its strength and
why the held-out gate is non-negotiable.

**Synthesis.** The three compose rather than compete: feature-space plateau averaging gives an
auxiliary target with a measured 4× better action share; slope-weighting concentrates boundary
supervision where response exists; the paired counterfactual loss is the only term that defeats
copying. None of them changes per-step SNR 0.60 — F2 (randomized-step DAQ data) remains the
most-likely-needed lever.

**§8.47 addendum — HOW plateau averaging is implemented (2026-08-31).** Status: currently an offline
estimator only (`review2` features script); nothing in training or eval. Ryan asked where it should
live. Working the design through kills two naive routes and leaves one sound one:

*Eval:* effectively already there. The gate's references are per-(run, sign, L2) cell medians from
`measured.json` — plateau-level aggregates — and the model side medians over draws. The 0d gate
rework formalizes it (n = distinct contexts, plateau-mean references); no new mechanism needed.

*Training — the two routes that DON'T work:*
1. **As an obs channel in the physics token: leaks.** The full-plateau mean at shot t includes
   FUTURE shots. The flow's context and target are built by the SAME encoder, so a channel cannot be
   in the target without also entering the context — "target-only obs channel" is structurally
   impossible in this architecture. As context it is (a) future information and (b) uncomputable
   online at deployment (an RL agent cannot know the settled mean mid-settling).
2. **As a decode-head target: inert.** Decode losses see the DETACHED predicted bag
   (`pred_obs_in_loss=false`) — the §8.43 trap again; zero gradient to dynamics or `act_enc`.
   Un-detaching per-modality would send gradient through `flow.sample`'s ODE unroll, the exact
   mechanism that blew up run 3 (arms 1a–1c). Not acceptable.

*The sound route — an AUXILIARY HEAD on the backbone context (the `action_head` pattern):*
`h = backbone(_to_input(s, a_ctx, act_emb=a_emb))` contains the undetached action embedding, and
§8.45 verified an aux head on `h` reaches `act_enc` (it is "the fourth live path"). So:
- **Labels:** at the 0c precompute, for each shot store the settled-plateau mean of sep for the
  plateau the NEXT shot belongs to (drop 3–8 settling shots, extraction-ok only) + a validity flag.
  Appended to the obs vector but **covered by NO modality's `obs_slice`** (proprio must set
  `obs_slice=(0,138)`), extracted in `_obs_dict` as a loss label — the `_phys_valid` mechanism,
  already built. Never encoded ⇒ leak-free by construction.
- **Head:** MLP on `h` (per-step, `detach_gradient=false`) → predicted next-plateau sep. ~30 lines
  on the `action_head` template.
- **Loss:** masked by validity; boundary-weighted (this is the setting where boundary weighting
  SHOULD work per §8.45's corollary — the target is action-rich, 43% share measured, unlike the raw
  latent's 0.3%). Within-plateau steps make it a denoising task (context-inferable, fine); boundary
  steps are the action-informative case — predicting the NEXT plateau's settled mean requires the
  action, and the context does NOT contain it.
- **What it does not do:** cancel the context shortcut at boundaries entirely (the machine's settled
  response is also predictable from run identity etc.) — the paired counterfactual loss (§8.47 pt 3)
  remains the only structurally shortcut-immune term. The aux head and the token compose: token =
  per-shot physics in the bag (context + gate readout), aux head = clean plateau supervision on the
  action pathway.

### §8.48 OPINION OF RECORD: HOW TO RESOLVE THE SPACING ERRORS (2026-08-31)

Ryan: "how should we resolve the physics spacing errors? fix the codec first and then see?"

**The spacing error is three different errors** (all measured):
| source | size | nature |
|---|---|---|
| codec roundtrip (decoder 2×2 seed) | **244 µm** median, p90 1100–1340 | model-side only; renderer |
| extraction quantization (integer median) | 61 µm grid | both sides; FIXED (opt-in interp, §8.44) |
| machine jitter + real-frame extraction noise | 207–254 µm/shot, corr 0.227 | physics; average, don't "fix" |
| *(the signal all of these sit on)* | *152 µm per 0.25°* | |

**Codec-first: yes, but for a sharper reason than measurement hygiene — the decoder is the
encoder's only teacher.** Under r9's config (`dynamics_detach_encoder=true`,
`pred_obs_in_loss=false`), the image encoder's ONLY gradient is the roundtrip anchor, which flows
through the decoder. A decoder that can only express a 384-number summary rewards the encoder for
information that survives that summary. [INFERRED, with measured counter-evidence bounding it:
inverse-dynamics AUC 0.913 shows the step info reaches the bag anyway, so the effect is marginal,
not fundamental.] Still: fixing the decoder is not cosmetic — it changes the representation's
training signal, and it is the cheap, independently-gated piece (AE-only Stage A, gate A1 244→<61 µm
before any dynamics compute).

**But the codec fix will NOT make the model learn transitions, and should not gate the response
work.** The dynamics loss lives in latent space; the decoder never touches its gradient (decode is
detached). Fixing the renderer sharpens the image-side INSTRUMENTS and goal 1's rendering quality —
attenuations 1–4 of §8.41 are untouched. And the physics token already provides a renderer-bypass
readout for the primary gate (sep decoded from the vector head, no U-Net in the loop), with the
image gate retained as the anti-copy cross-check — the cross-check is what needs the codec fix.

**So the answer to "fix codec first, then see" is: run it first, don't WAIT on it.** The three
streams are parallel, not serial:
1. Zero-GPU decision stream — the M18-style physics probe + gate rework (held-out, plateau
   references, N_CTX~110) + re-baseline. This, not the codec, is what decides the response strategy.
2. Stage A (one run) — decoder xattn + proprio llw + physics modality, which must ship TOGETHER
   anyway (slot_emb 34→35 makes a later addition checkpoint-breaking, §8.45), each with its own gate
   (A1–A3 image, A4/A7/A8 vector) so attribution survives the bundling.
3. DAQ requests — outstanding regardless of 1 and 2; per-step SNR 0.60 is machine-set and F2
   remains the most likely endpoint.
If A1 fails (xattn insufficient), the response campaign does not stall — the physics-token readout
carries the gate while the decoder iterates (F5's fallback ladder: 8×24 grid, 183 µm, 62%).

### §8.49 EXECUTION: M19 PROBE, v7phys CORPUS, xattn DECODER, STAGE A LAUNCHED (2026-08-31)

**M19 — the physics-target probe (first pass, 3 seeds).** `scratchpad/m19_physics_probe.py`;
M18's head/protocol; targets = (a) per-shot Δphysics (the token's flow target), (b) settled
next-plateau mean sep (the aux head's target); held-out boundary events n=139/155; paired
counterfactual, direction-aware.
| target | arm | cos(Δact, resid), 3 seeds | pooled | corr_sep |
|---|---|---|---|---|
| shot | ctrl | 0 exactly | 0 | — |
| shot | raw_true | −.063/−.168/−.013 | **−0.082** | +0.145 |
| shot | **dl2** | +.118/+.070/+.022 | **+0.070** | **+0.155** |
| plateau | **raw_true** | +.136/+.097/+.136 | **+0.123** | +0.131 |
| plateau | dl2 | +.148/−.032/+.007 | +0.041 | +0.078 |
Honest read: ctrl is exactly 0 (instrument clean); `corr_sep` is consistently POSITIVE (+0.08…+0.16)
across five of six action arms — the action IS weakly informative conditional on context — but the
pooled cos effects are M18-sized (~0.07–0.12), i.e. **the physics target does not dramatically
improve conditional action recoverability over the raw latent by this instrument**, consistent with
§8.41's per-step SNR and only ~600 boundary events. raw_true is anti-aligned on the 6-dim shot
target while positive on the plateau target — arm-dependent, thin-n behaviour. The within-seed z's
(±2.7–3.0) overstate certainty (between-seed spread is comparable to the effect); a **10-seed
follow-up (m19b) using the between-seed t-stat is running** on ctrl/raw_true/dl2 × both targets.
Plateau RMSE/copy ≈ 1.9–2.1 is expected, not damning: the head estimates a plateau mean from 4 noisy
shots against a copy baseline that inherits the settled old mean; the paired cos is the metric.

**v7phys corpus** (`logs/recording_2026_08_31_xtcav_all_v7phys`, builder `scratchpad/build_v7.py`):
6 physics channels + ok flag appended to obs (138 → 145), extracted from the decoded 64×192 store
(§8.45 P8), invalid rows imputed to the train valid-row mean (z=0 after the Normalizer — the F5 trap
avoided), flag normalizer stats FORCED to (0,1) so `_phys_valid`'s >0 test is exact. Extraction ok:
train 92.8% / val 89.1% / eval 94.1%. Verified end-to-end through `load_split_episodes_mm`: stored
physics == fresh extraction on every ok row; flag exact; invalid rows at z=0.00e+00. Free
cross-check: train stats match the reviewer's independent cache to 3 decimals (1517.53/1237.67).

**xattn decoder implemented** (`vision.py::ConditionalUNet`, gated by `ModalitySpec.decoder_cond` /
`seed_upsample`, threaded through `VisionAEConfig`): learned 768-query grid at the (16,48) bottleneck
attending over the 32 tokens; `cond.mean(1)` demoted to FiLM-only. Verified: default seed path keeps
the EXACT historical parameter count (1,435,267 — old checkpoints load bit-identically); xattn is
**983k params smaller**; all 32 tokens receive gradient (the seed path's `cond.mean` starved
per-token structure); q_pos/k/v train. Also fixed en route: `lit.py` val-metrics denorm is now
obs_slice-aware (the 138-slice of 145-D stats — would have crashed at the first val epoch).

**Stage A LAUNCHED** (`xtcav_all_r10A`, GPU 1, 60 epochs, `conf/model/xtcav_physA.yaml`):
bsp32mse + xattn/bilinear decoder + proprio llw=1 + physics modality (obs_slice 138:144, llw=1),
`lambda_flow=0` + all modality weights 0 → roundtrip-only training; during-train evals off (the
§8.45 P1 `!= "proprio"` eval sites are NOT yet kind-aware — deliberately sidestepped, not fixed).
Two smokes passed first: full 1-epoch (457 batches) and a val-path smoke (66 val batches, sliced
denorm exercised, monitor key printed: `val/metric/proprio/pointwise_error`). Model 2.57M params
(vs r9 3.53M; xattn −983k). Gates on completion: A1 (sep 244→<61 µm), A2 (grad ratio →1.05),
A3 (p90 <300), A4 (proprio roundtrip <0.645), A7/A8 (physics roundtrip, µm), A6 (ae_floor ±1 dB).

### §8.50 GATE DESIGN: ENSEMBLE (MULTI-DRAW) BOUNDARY EVALUATION (2026-08-31)

Ryan: "in the future, can we evaluate the world model with multiple jittered draws over the boundary
instead of single median draws to compute physics summary parameters?" Adopted into the 0d gate
rework. Design of record:

**What exists vs what changes.** The counterfactual gate already draws N_DRAW=6 stochastic samples
with paired noise per context — but everything is COLLAPSED to a mean response, and the filmstrip/
sweep figures reduced draws to a median before extracting physics. The change: keep the per-draw
physics values as an ENSEMBLE and report distributional statistics, per boundary context:
1. **Distributional response**: the full Δsep distribution over draws for each (context, ±δ) arm —
   shift of the ensemble, not a point estimate. Report per-context **d′ = paired shift / pooled
   ensemble spread** alongside the mean response.
2. **Calibration**: model ensemble spread vs the MACHINE's shot ensemble at the same setpoint (the
   existing variance-ratio gate, vr ∈ [0.7, 2], generalized from a scalar to the physics parameters).
   This matters because §8.24 measured the E300 models UNDER-dispersed 2.5× — an ensemble is only
   interpretable as machine jitter if this gate passes, so calibration is a PRECONDITION gate, not a
   nice-to-have.
3. **Per-draw ok-masking**: extraction failures (1–33% of rendered frames) drop draws, not contexts.

**Three distinct "jitters", spent in the right order.**
- *Sampling noise (eps)*: already there. Averaging over K draws shrinks model-sampling noise as 1/√K
  — but the reviewer's P6 point stands and is the binding constraint: **draws are within-context
  repeats on the same frames and the same codec bias; raising N_DRAW does not shrink context-level
  variance.** Diminishing returns beyond K ≈ 10–20; the campaign-level error is set by the number of
  DISTINCT CONTEXTS (3 today, ~110 needed).
- *Context jitter*: resampling real contexts at the same setpoint IS raising N_CTX — the statistically
  binding unit. Spend budget here first.
- *Action jitter (new, optional mode)*: instead of the two-point ±δ arms, draw a ~ N(a₀, σ) with the
  SAME paired noise and REGRESS the per-draw physics parameter on the drawn action → a per-context
  response SLOPE with its own error bar, compared to the machine's measured local slope. The
  two-point symmetric design is more efficient for a LINEAR response; the regression mode tests
  shape/linearity (run 7 ep115's nonlinearity, §8.34 M-G4, is exactly what it would expose). Keep
  ±δ as the primary, action-jitter as the diagnostic.

**Why this becomes cheap after Stage A/B: the physics token.** Per-draw physics currently costs a
render + extraction (fragile, 244 µm floor). Once the physics head exists, sep is a decoded SCALAR
per draw — hundreds of draws per context at negligible cost, no renderer in the loop. The rendered-
frame ensemble stays as the anti-copy cross-check at small K.

**Pre-registered reporting per boundary**: n contexts / n draws / per-draw ok-rate; ensemble median
response + bootstrap CI over CONTEXTS (never over draws); d′; calibration ratio; δ=0 drift ensemble.
Nothing may be averaged over draws before the paired differencing — pairing first, then aggregate.

### §8.51 M19b: THE PHYSICS-TARGET PROBE PASSES AT 10 SEEDS (2026-09-01)

10 seeds, between-seed t-stat (the honest statistic after m17's single-seed fiasco), same protocol:
| target | arm | pooled cos | t (between-seed) | seeds positive | corr_sep |
|---|---|---|---|---|---|
| shot (token flow target) | ctrl | 0.0000 | — | — | — |
| shot | raw_true | **−0.061** | **−3.9** | 1/10 | +0.094 |
| shot | **dl2** | **+0.083** | **+4.3** | 10/10 | **+0.172** |
| plateau (aux-head target) | raw_true | **+0.112** | **+5.0** | 9/10 | +0.129 |
| plateau | **dl2** | **+0.099** | **+3.8** | 10/10 | +0.154 |

**The pre-registered necessary condition PASSES on both targets**: the commanded action is
recoverable from the physics targets conditional on context, at >3.8σ, with ctrl pinned at exactly 0.
The §8.43 token design and the §8.47 aux-head design are both quantitatively justified — at the
measured (modest) effect size: cos ≈ 0.08–0.11, corr_sep ≈ 0.13–0.17. This does NOT promise the
gate passes after training (sufficiency is a Stage-B question); it rules out dead-on-arrival.

**A robust, surprising negative: the raw setpoint input is ANTI-aligned on the fine-grained target**
(−0.061 at t=−3.9, 9/10 seeds negative) while positive on the plateau target. The raw z-input does
not merely fail to help at the shot level — it learns a wrong-direction response, presumably fitting
the settling transient's sign structure. The ΔL2-in-own-units channel fixes the direction on BOTH
targets (10/10 seeds positive on each). **Consequence for Stage B: the Δa channel (derived in-model,
§8.45) is now evidence-backed as REQUIRED for the token pathway, not optional** — with raw input the
fine-grained physics gradient would push the response the wrong way.

### §8.52 STAGE A GATES: ALL PASS — THE §8.31 CAPACITY VERDICT IS RESOLVED (2026-09-01)

Run `train_world_model_2026_08_31_17_45_17_xtcav_all_r10A` (60 epochs, codec-only, last.ckpt).
Gate script `wizard/scripts/xtcav_stageA_gates.py`; r9 baselines RECOMPUTED on the identical frame
sample (600 val+eval frames), not quoted from the record.

| gate | quantity | Stage A | r9 (same frames) | bar | verdict |
|---|---|---|---|---|---|
| **A1** | AE roundtrip \|sep err\| median | **61 µm** | 183 µm | < 61 | **AT THE 1-PX FLOOR** |
| **A3** | AE roundtrip \|sep err\| p90 | **244 µm** | 1220 µm | < 300 | **PASS** (5× better tail) |
| A1x | extraction-ok on decodes | 86.8% | 87.0% | report | par |
| A2 | seed-boundary grad ratio (beam px) | **1.042** | 1.031 | ≤ 1.10 (real 1.07) | **PASS** — no grid imprint |
| A6 | image roundtrip PSNR | **43.6 dB** | 38.6 dB | within 1 dB | **PASS** (+5.0 dB) |
| **A4** | proprio roundtrip nRMSE | **0.113** | 0.493 | < persistence 0.343 | **PASS** (3× below persistence) |
| A7 | physics roundtrip nRMSE (z) | 0.005 | — | report | ~perfect |
| A8 | physics roundtrip \|sep err\| | **3.0 µm** | — | < 61 | **PASS** |

Notes. (a) A1's 61.000 is the extraction grid: with the integer median, errors are multiples of
61 µm, so a median of exactly 61 = the single-pixel floor — the decoder now places the bunches to
≤1 px at the median, from 4 px (244 µm recorded; 183 µm on this sample) for the seed decoder.
(b) A4's persistence bar is 0.343 here, not §8.24's 0.645, because this gate scores ALL 138 channels
while §8.24 used the 64-channel `sel`; the comparison is internally consistent (all three columns on
the same channels) and the §8.25 pre-registered prediction — roundtrip well below persistence once
the encoder trains — is confirmed either way. (c) A8 = 3 µm says the physics TOKEN decodes sep
essentially exactly: the renderer-bypass readout for the ensemble gate (§8.50) is real. (d) All of
this at **2.57M params vs r9's 3.53M** — the xattn decoder is 983k smaller and strictly better.

**Consequences.** §8.31's "no separation-based gate can be sharper than the codec" cap is lifted:
the model-side measurement floor drops from ~244 µm to the extractor's own ~61 µm grid (and to ~3 µm
via the physics head). The Stage-B warm start now exists with a trained proprio encoder, a trained
physics codec, and a 34-token layout. Remaining known gaps carried into Stage B: `model.init_from`
still doesn't exist (§8.45 P3 — must be added before Stage B launches, `+resume` is NOT acceptable),
and the §8.45 P1 eval-site refactor is still owed if during-train evals are wanted.

**§8.52 addendum — figures** (`wizard/scripts/xtcav_stageA_figs.py` →
`<r10A run dir>/figs/`, all AE roundtrips on identical frames, r9 recomputed alongside):
- `stageA_roundtrip_filmstrip.png` — REAL vs Stage-A vs r9 at 6 sep quantiles (10–90%; the 2%/98%
  tails were deliberately excluded — degenerate morphology where extraction is ambiguous for every
  codec; the CDF panel carries the full distribution so nothing is hidden). Stage A matches the real
  morphology at all six; sep EXACT in 4/6. r9 chops the beam at the col-96 seed boundary and loses
  the tails at wide separations.
- `stageA_quant.png` — (a) sep-error CDFs: **39% of Stage-A frames land EXACTLY on the true sep**
  (error 0 on the 61 µm grid) vs run 9's median 183/p90 1275; (b) gate bars; (c) proprio per-channel:
  **138/138 channels better** than the frozen encoder; (d) physics-token readout scatter: identity
  line, median |err| 3.0 µm over 536 shots.
- `stageA_background.png` — the §8.29/§8.30 blockiness check on a representative wide-sep frame
  (0.85 quantile, not the max — the extreme tail is degenerate morphology and would misstate typical
  quality): at the 0–3 count stretch, r9 paints its rectangular seed plateaus and chops the streak at
  col 96; Stage A's field is smooth with the full tail, grad ratio 1.04 vs real 1.07.

**§8.52 addendum 2 — why the proprio roundtrip is nonzero (0.112), measured
(`scratchpad/proprio_rt_budget.py`).** Hypothesis going in: the 64-wide `_mlp` hidden layer (enc AND
dec pinch at h=64) is the binding constraint, and the residual concentrates on the incompressible
white channels. **Both parts REFUTED:**
| | median per-channel nRMSE |
|---|---|
| Stage-A model roundtrip | **0.112** |
| best 64-dim LINEAR code (PCA on train, scored held-out) | **0.034** |
A plain rank-64 linear map beats the trained nonlinear codec **3.3×**, so capacity is NOT the limit;
corr(model err, PCA-64 err) over channels is only 0.386; and the worst model channels are BPM X's
and TMIT duplicates (0.23–0.28 where PCA-64 gets 0.03) while the flags/imputed channels are among
the BEST (near-constant = trivially reproduced). The nonzero loss decomposes as:
- a small structural floor (~0.03): 138 ch → one 128-float LayerNormed token through 64-wide MLPs;
- **the other ~3× is FITTING, not capacity**: the roundtrip weight is llw=1 (vs the image's 10), 60
  epochs, and the encoder input is 138 raw + 4,416 fourier features of which 58% are shot-to-shot
  white at the deliberately-held f_max=100 (§8.40 F3) — the first Linear(4554→64) spends its
  conditioning fighting noise features.
Levers if the 3× matters later: proprio llw 1→~5, `fourier_fmax` 100→8, or longer Stage A — each a
one-knob rerun, deferred (0.112 already sits 3× below the persistence bar and 4.4× below r9).

### §8.53 PROPRIO ROUNDTRIP: MECHANISM FOUND, FIX = LINEAR SKIP, STAGE A RELAUNCHED (2026-09-01)

Ryan wants accurate imagined BPMs, so the 0.112 roundtrip (vs a linear floor) was attacked with
staged hypothesis probes (`scratchpad/m20{,b,c}_proprio_rt.py`) before any retraining.

**m20 — the suspected knobs ALL fail.** Free-standing training of the exact modality path (real
`roundtrip_losses`, full gradient, converged): repl **0.128** ≈ the in-run 0.112 — so the gap is NOT
objective competition (llw 1 vs 10) and NOT under-training. Single knobs: fourier f_max 8 → 0.130,
fourier off → 0.120, hidden 256 → 0.103, no-LN → 0.110. Nothing approaches the floor.

**m20b — the mechanism, and a correction to addendum 2.** The PCA-64 "floor" of 0.034 was the wrong
reference: **PCA-128 = 0.0000** — the token is 128-wide and the obs vector has ≤128 effective dims,
so the true linear floor is ~zero. Scored on BOTH a train-holdin slice and the real (run-tail)
heldout:
| arm | holdin | heldout |
|---|---|---|
| pure linear 138→128→138 | **0.0016** | **0.0043** |
| the MLP (h=64, GELU) | 0.0544 | 0.0878 |
| **linear skip + MLP residual (zero-init)** | 0.0029 | **0.0128** |
Two compounding causes, now separated: the hidden-64 GELU MLP **pinches a near-linear signal**
(0.054 in-distribution ≈ the rank-64 bound) **and extrapolates 1.6× worse onto the run-tail heldout**
(0.054 → 0.088). SGD is exonerated (the linear arm reaches 0.004 under the same optimizer).

**m20c — under the model's bag LayerNorm** (locked model-wide; the dynamics needs the scale-free
bag): linskip+LN **0.032** heldout, lin+LN 0.033. LN costs ~3× (it discards the token's mean/std and
the sphere hurts extrapolation — consistent with the documented −3.51 dB on the image trunk), and
with LN the MLP residual adds nothing over pure linear. **0.032 is the achievable in-model target**
= 3.5× better than 0.112, 10× below persistence.

**Implemented: `ModalitySpec.linear_skip`** (default False = bit-identical, verified: identical
param set when off). Parallel `nn.Linear` enc/dec beside the MLP trunk/head; the decode head learns
the RESIDUAL target − dec_lin(token), so `decode()` and `decode_loss()` stay consistent (verified:
head-zeroed decode == the pure linear path; gradients reach dec_lin/enc_lin/head through both losses).

**Stage A relaunched as `xtcav_all_r10A2`** — identical to r10A except `modalities.0.linear_skip=true`
(one variable; physics left unchanged at 0.005). Expected: A4 0.112 → ~0.03, A1–A3/A6–A8 unchanged.
r10A2 supersedes r10A as the Stage-B warm start if the gates hold.

### §8.54 LN-CARRIER DIMS: THE BAG LAYERNORM MADE LOSSLESS (2026-09-01)

Ryan: "can you encode info which would otherwise be lost in the layer norm in additional dimensions?"
Yes — and the construction is EXACTLY invertible, not a learned compensation.

**Construction** (`m20d_carrier.py`, now `ModalitySpec.ln_carrier`): the encoder emits d−2 content
dims, normalizes them INTERNALLY to mean 0 / biased std 1, and writes the two scalars the outer LN
would destroy — the content's mean and log-std — into the last 2 dims as ordinary direction
components. Because the content dims then have KNOWN statistics, the outer LN's unknowns (m, s) are
solvable from its own output: s = 1/std(y[:d−2]), m = −mean(y[:d−2])·s. The decoder undoes the LN in
closed form and recovers the content exactly.

**Measured:**
| | holdin | heldout (run tails) |
|---|---|---|
| linskip under bag LN (r10A2's config) | 0.0092 | 0.0319 |
| **linskip + carrier under bag LN** | **0.0029** | **0.0111** |
| (no-LN linskip reference, m20b) | 0.0029 | 0.0128 |
Inversion exact to **7.2e-7** (fp32, incl. F.layer_norm's eps). The carrier arm under LN MATCHES the
no-LN reference — the LN cost (3×) is fully recovered. Total vs the r10A in-run 0.112: **~10×**.

**Properties worth recording:** (a) the bag token is still exactly unit-LN'd — the dynamics'
scale-free anti-collapse geometry is untouched; (b) the inversion is norm-agnostic (identity when
the token was never LN'd), so `latent_norm=none/affine` models decode unchanged; (c) it degrades
gracefully on PREDICTED tokens (the constraints are read from the prediction itself); (d) default
off = bit-identical (verified: param set unchanged); (e) the same trick applies in principle to the
image tokens (LN's documented −3.51 dB on the image trunk) — not done, one modality at a time.

**Execution:** implemented + unit-verified (full encode → bag `_ln` → `_decode_cond` recovery at
3.6e-7). r10A2 (linskip only, 30 min in) KILLED and superseded — attribution for each piece is
already held by the standalone probes (linskip: m20b; carrier: m20d), so the combined Stage A loses
nothing. **r10A3 launched** (proprio `linear_skip` + `ln_carrier`; physics untouched at 0.005).
Expected gates: A4 0.112 → **~0.011**, A1–A3/A6–A8 unchanged. r10A3 becomes the Stage-B warm start.
(Also: the pkill self-match trap fired AGAIN killing my own wrapper (exit 144, third occurrence) —
the yaml edit it swallowed was redone and verified.)

### §8.55 LN-CARRIER EXTENDED TO THE IMAGE TOKENS; r10A3/r10A4 A/B RUNNING (2026-09-01)

Ryan: apply §8.54's carrier to the image tokens too, and audit the workflow for other such losses.

**Image implementation** (`ImageModality`, flag `ln_carrier`): each of the 32 tokens gives up 2 dims
to carry its own (mean, log std). One wrinkle the vector case didn't have: the ConvImageEncoder's
attention needs d % heads == 0 and 126 is not divisible by 8, so the AE keeps its native d=128 and a
modality-level `carrier_proj = Linear(128→126)` produces the content; the decode head is built at
the content width via `dataclasses.replace(cfg, d=126)`. The rank-126 projection is a STATIC learned
subspace the encoder co-adapts to — unlike the LN it replaces, nothing per-sample is destroyed.
Verified: recovery through the real bag `_ln` exact to 4.8e-7; decode/decode_loss train
(carrier_proj grad 2.5); default off bit-identical; smoke (6 batches + full val pass) clean.

**Runs — a clean A/B on the two free GPUs:**
| run | GPU | proprio | image | decides |
|---|---|---|---|---|
| r10A3 | 1 | linskip + carrier | plain (xattn) | A4 (0.112 → ~0.011 expected) |
| r10A4 | 0 | linskip + carrier | **+ carrier** | A1/A3/A6 image side vs r10A3 |
Same corpus, same every-other-knob. Whichever wins the image gates becomes the Stage-B warm start.
Note the image side is NOT guaranteed to improve: A1 already sits at the extraction grid floor
(61 µm median) — the room is in A3's p90 (244) and A6's PSNR, and the documented −3.51 dB LN cost
was measured on the TAESD trunk, not this bespoke codec. The A/B answers it either way.

**Audit agent dispatched** (CPU-only): sweep the workflow for other irreversible information drops —
the rollout's per-step re-LN interaction with carrier tokens, SpaceTimeBlock pre-norm discarding
cross-token magnitude relations, action-side fourier clamp saturation, encoder avg_pool, decode
clamps, GroupNorm — each to be verified in code and measured where cheap. Report pending.

### §8.56 INFO-LOSS AUDIT: IMAGE CARRIER RETRACTED (NULL, NOT WRONG); STAGE-B LANDMINES (2026-09-01)

Audit agent report (CPU-only, on r10A/r10A3 checkpoints + the shipped corpus; scratch
`scratchpad/audit2/`). Load-bearing claims re-verified before acting.

**F1 — r10A4 KILLED; §8.55's image carrier was a NULL experiment, and the motivation was my
provenance error. [VERIFIED in code + measured by the agent.]** `ConvImageEncoder.encode`'s LAST op
is its own `nn.LayerNorm(d)` (`vision.py:359/371` — re-verified), so image tokens reach the bag `_ln`
already per-token normalized: the bag LN removes a near-constant (across-frame std of the removed
mean/log-std: **1.0e-3 / 1.3e-3**, vs proprio's 8.3e-2 / 4.3e-2 where the carrier win is real).
End-to-end through the real trained decode head: **46.70 dB vs 46.70 dB — 0.002 dB difference**.
The −3.51 dB LN tax I cited comes from `robocasa-scene4-4h.md` measured on the **frozen TAESD
trunk, which has no internal LayerNorm** — it does not transfer to the bespoke conv codec. The
correct reading for the record: not "the carrier doesn't help images" but **"there was nothing to
recover"** — the conv encoder's own LN already makes the bag LN a no-op for image tokens. The
implementation stays in-tree (correct, default-off); r10A4 killed at ~40 min; **r10A3 (proprio
carrier only) is the Stage-A of record**, pending its gates. The proprio result is unaffected: the
FourierMLP encoder has NO internal LN, m20c measured the real cost (0.012→0.032) and m20d recovered
it (0.011).

**F2 — carrier sensitivity concentration: the roundtrip win must be re-verified on PREDICTED
tokens.** The inversion exponentiates the log-sd carrier: its decode sensitivity is **8.8×** a
median content dim (mu: 5.8×) while `flow.loss` weights it at 1/128 (the event-dim mean at
`flow.py:110` averages d FIRST, so per-dim weights are inexpressible today). The agent's linear
replica (which reproduces §8.53/§8.54's numbers: 0.0100 vs 0.0111, 0.0446 vs 0.032) puts the
carrier-vs-plain crossover at ~20% of one-step motion left unexplained. Closed negative: no
conditioning blow-up (inversion gain ≤1.014 even perturbed), and the sphere→(content,μ,logσ) map is
a bijection so predicted tokens are always structurally valid. **Pre-registered for Stage B:**
report imagined-proprio nRMSE r10A3-warm vs r10A-warm at the first eval; if the carrier arm is
worse despite the 10× roundtrip, F2 is why, and the cheap fix is scaling the carrier dims at encode.

**F7 — STAGE-B LANDMINE: `physics_valid_idx` is wired in code but appears in NO config.** 7.23% of
train rows have the physics block constant-imputed (z=0), flag at dim 144. Enabling `physics_alpha`
without `+model.diffusion.physics_valid_idx=144` gives full alpha-boosted flow weight to a CONSTANT
target on those steps — training the dynamics toward the dataset mean exactly where the extractor
failed. **The flag is now a mandatory line in the Stage-B launch command.**

**F9 — `best.ckpt` in codec-only Stage A is an 8.9 dB WORSE image codec than `last.ckpt`** (37.8 vs
46.7 dB PSNR) — expected, since the monitor is a garbage metric when the flow is untrained (§8.49
noted this), but it means **Stage B must warm-start from `last.ckpt`, never `best.ckpt`** — a
constraint on the still-unbuilt `model.init_from`.

**Closed negative (no action):** F3 action path destroys nothing (fourier off on actions; symlog
keeps everything inside the clamp; 0% saturation); F4 the encoder's avg_pool grid is NOT a
separation bound (linear sep readout flat at 239–281 µm RMSE from native down to 8×24 pooling — the
244 µm §8.31 bound was the decoder's seed path alone); F5 eval clamps cost 0.0 µm on the gate
(decoded range [−0.04, 0.47], upper clamp never fires) — though `NOISE_U8` removes 2.1× more mass
from decoded frames than real ones (21.8% vs 10.3%), so threshold changes move the decoded gate arm
~2× harder; F6 the physics token needs NO carrier (worst-channel readback 0.9992 post-LN — 128 dims
carrying 6 numbers leaves ample LN-invariant subspace; a carrier would import F2's concentration
onto the token `physics_alpha` protects); F8 SpaceTimeBlock pre-norm and U-Net GroupNorms are
harmless (residual streams bypass; the raw `act_enc` output reaches `_cond` un-normalized).

**Also surfaced:** the physics TRAINING target at obs 138 inherits `proj_median`'s 61 µm grid, and
gate A8's bar is exactly that grid step; label noise ~15 µm p50 vs 1606 µm signal std — not a
limiter today, but the target and the gate share a grid (flip `interpolate` for both in one commit,
§8.45 P8 discipline).

### §8.57 STAGE B SETUP: init_from + action_delta BUILT, SMOKED; LAUNCH PLAN (2026-09-01)

Ryan away ~10 h; instruction: set up run B, review, launch, guide. r10A3 ETA ~18:11.

**Built and unit-tested:**
1. **`model.init_from`** (`setup.py::warm_start` + call site + `mm_flow.yaml` key): weights-only warm
   start into a FRESH run dir — loads name+shape matches, SKIPS mismatches with a loud report, never
   touches optimizer state; a run-dir path resolves to `last.ckpt`, never best (§8.56 F9). Verified:
   act_enc widened by action_delta is skipped and reported; everything else loads.
2. **`model.action_delta`** (§8.51-required: raw input is ANTI-aligned on the physics target).
   `_act_feats` appends (a[t]−a[t−1])/act_delta_scale as extra act_enc channels, derived IN-MODEL —
   no corpus change. Applied ONCE per entry point on the full sequence (loss_terms / `_rollout_from`
   funnel (covers parallel+KV-cached+imagine) / forward / one_step_states / action_context); flip
   detection reads RAW actions (`a_ctx_raw`); action_flow targets stay raw. Scale from config
   (`action_delta_scale`, REQUIRED when on), measured on v7 train: **[0.007045, 0.157653, 1.447905,
   0.414392]** → a 0.25° step = **5.06 delta-units** (vs 0.0356 raw). Verified: off = bit-identical
   widths; step Δ-feature = 5.53 (synthetic check); parallel vs KV-cached rollout parity at the
   documented fp level (1.7e-3, same as the off-model's 4.4e-3) under paired seeds; flip telemetry
   intact.
3. **Two real bugs caught by the smoke** (the reason smokes exist): (a) `_obs_dict`'s `_phys_valid`
   key crashed `m.modalities[k]` lookups in lit.py's noise/shared-encode guards — guarded; (b) the
   §8.43 mask read `future_obs` (the F-frame slice) — misaligned with the flow's L−1 transitions;
   now reads the full-window `obs` dict sliced `[:, 1:Lm1+1]`. Smoke 2 ran WITH
   `physics_alpha=0.3 + physics_valid_idx=144` live to prove the mask branch end to end: clean
   train + full val pass.
4. Smoke-2's warm_start report shape-skipped the proprio codec pieces — EXPECTED: the stand-in
   source was r10A (no carrier). **Launch assertion: from r10A3 the report must show ZERO skips
   (ctrl) / exactly `act_enc.net.0.weight` (dl2).** Note r10A3's checkpoint predates the
   `act_delta_scale` buffer → missing key → warm_start leaves the CONFIGURED value. Correct by
   construction; reviewer asked to confirm.

**Plan of record for tonight** (pending the reviewer agent + r10A3 gates):
| step | what | gate |
|---|---|---|
| 1 | r10A3 finishes (~18:11) | — |
| 2 | Stage-A gates on r10A3 `last.ckpt` | A4 ≈ 0.011; A1 ≤ 61 µm; A2 ≤ 1.10; A6 within 1 dB; A8 < 61 µm |
| 3 | launch **10B-ctrl** (GPU 0) + **10B-dl2** (GPU 1), 150 ep, `scratchpad/r10B_launch.sh` | warm_start report assertion above |
| 4 | one watcher per run (`r10B_watch.sh`): 20-min snapshot ladder + tripwire (nan, or val_loss > 50 after ep 20) whose exit notifies me | — |
| 5 | on completion: F2 pre-registered check (imagined-proprio nRMSE, carrier vs r10A-warm), then the §8.42/§8.50 gate suite on the ladder | — |
Both arms carry `physics_valid_idx=144` (inert at alpha 0, never forgotten — §8.56 F7);
physics_alpha stays 0 (tokw is the NEXT arm, on dl2's base, §8.51). Fallback if r10A3 gates FAIL:
launch both arms from r10A instead (its gates passed; costs the proprio-carrier improvement, not the
campaign) and record the divergence.

### §8.58 STAGE-B LAUNCH REVIEW: NO-GO → FIXED → GO-PENDING-GATES (2026-09-01)

Reviewer verdict: NO-GO as scripted, GO after four fixes — no defect in the new code (it verified
the warm start by SIMULATING the exact load: ctrl 245/245 tensors, dl2 244/245 with exactly the
widened `act_enc` skipped, `act_delta_scale` survives because r10A3's checkpoint predates the
buffer). All applied:

**P0 fixes (launch script `scratchpad/r10B_launch.sh`, now fully executable):**
- **P0-1 batch confound**: autobatch sizes against the CARD, not free memory, and probe-OOM silently
  steps the batch down — an arm launched onto a busy GPU would train at a tiny batch for 17 h.
  Fixed: `data.batch=36 data.autobatch=false` (r9's validated value) + a <500 MiB GPU gate in the
  script.
- **P0-2**: `experiment=` early in the overrides (the watcher pgreps `"experiment=<tag> "`),
  `CUDA_VISIBLE_DEVICES` per arm, per-arm stdout logs (warm_start/action-delta lines go to stdout
  only), run_summary blocks inline.
- **P0-3 PRE-REGISTERED INTERPRETATION**: with `physics_alpha=0` the physics token carries 1/34 ≈
  2.9% of a uniform flow loss, so **tonight's dl2-vs-ctrl tests "Δa under a uniform-bag loss", NOT
  the m19b token mechanism.** A dl2 null tonight must NOT be written up as "Δa doesn't work" — the
  mechanism arm is `10B-tokw` (dl2 + physics_alpha + the two-sided mask). Named before launch so it
  cannot be re-interpreted after.
- **P0-4**: `ae_floor` + `manifold` during-train evals also disabled — they crash on the 145-D obs
  (`evaluation/` is not obs_slice-aware, §8.45 P1 still open; caught + self-disabled, but noisy).
  Also recorded: `manifold.py:143` / `routines.py:475/682` / `xtcav_run4_probes.py:114` /
  `xtcav_r6_gates.py` call `_to_input` with RAW actions → loud width crash on any dl2 checkpoint;
  fix before scoring dl2.

**Code fixes from P1/P2:** warm_start now genuinely REFUSES `best.ckpt` (the docstring had claimed a
refusal that didn't exist — and §8.37's own Stage-B command reads `.../best.ckpt`, a copy-paste
trap); `act_delta_scale` excluded from warm_start loading (config-owned; a future warm start FROM a
Stage-B ckpt would have silently overwritten it); the `_phys_valid` mask is now TWO-SIDED
(`v[:, :Lm1] * v[:, 1:Lm1+1]` — a residual target depends on both endpoints; one-sided would have
put alpha-boosted weight on ~7% garbage residuals the moment tokw launches); `_rollout_from`
docstring order. Watcher upgraded with the reviewer's tested §8.16-aware tripwire (flip-ratio
1.5× post-warmup-min rule, realized-share band, nan/inf grads, clip blowup) + a 30-min staleness
trip.

**Accepted with eyes open:** 150 ep ≈ 16–17 h (finish ~11:00 tomorrow — epoch ~85 at the 10-h
mark); `last.ckpt`/snapshot granularity is 4 epochs (val cadence) and snap epoch-labels are
approximate — read the epoch from the ckpt at scoring time; `grad/norm/encode_proprio` is MISLABELED
under linear_skip (dec_lin buckets into it — use `encode_physics == 0.0` as the frozen-encoder
witness); `val/metric/physics/*` is garbage (clamped z-scores) — never read it, never point the
monitor at it; ctrl and dl2 differ by one random act_enc draw on top of the intended variable
(between-seed spread is comparable to effect sizes — say so when reporting); §8.56 F2's
pre-registration (r10A3-warm vs r10A-warm) has NO arm tonight — both warm from r10A3; it needs its
own arm later or a retraction.

**Reviewer negatives worth keeping:** every act_enc-reaching path featurizes exactly once (swept:
MPPI, openloop, contraction, lit, the counterfactual gate — the B1/B3 gate is action_delta-safe);
r10A3's backbone/flow/act_enc are bit-frozen at random init (max|Δ| = 0.000e+00 ep3→ep7, AdamW never
touched them: grads None) so the warm-started dynamics is distributionally a fresh init; OOM
headroom 16 GB at the realized peak; disk ~2.6 GB total vs 11 TB.

### §8.59 r10A3 GATES + STAGE B LAUNCHED (2026-09-01 17:40)

**r10A3 gates** (`<r10A3>/stageA_gates.json`, r9 recomputed on identical frames):
| gate | r10A3 | r10A (prior Stage A) | r9 | bar | verdict |
|---|---|---|---|---|---|
| **A4 proprio roundtrip** | **0.018** | 0.112 | 0.493 | < 0.343 | **PASS — 6.2× better than r10A** (linskip+carrier; m20c predicted 0.011) |
| A1 sep err median | 61 | 61 | 183 | < 61 | at the grid floor |
| **A3 sep err p90** | **305** | 244 | 1220 | < 300 | **MARGINAL: one 61 µm grid step over the bar.** Errors are quantized to 61 µm multiples, so p90 can only land on 244 or 305 — a 1-px difference within grid resolution, still 4× better than r9. Accepted; recorded, not hidden. |
| A2 grid imprint | 1.014 | 1.042 | 1.031 | ≤ 1.10 | PASS |
| A6 PSNR | 42.97 | 43.56 | 38.56 | ±1 dB | PASS (−0.6 dB vs r10A, +4.4 vs r9) |
| A7/A8 physics | 0.004 / **2.97 µm** | 0.005 / 2.98 | — | <61 | PASS |
**r10A3 is the Stage-B warm start.**

**Stage B LAUNCHED 17:38** from `r10B_launch.sh` (§8.58 P0 fixes in): both GPUs verified idle first;
`10B-ctrl` GPU 0, `10B-dl2` GPU 1; batch pinned 36, autobatch off. Checklist verified:
- ctrl warm_start: **245/245 params, zero shape-skips** (left-at-init = 5 non-persistent freqs
  buffers + config-owned act_delta_scale — exactly the reviewer's simulation);
- dl2 warm_start: **SHAPE-SKIPPED exactly `act_enc.net.0.weight (128,4)→(128,8)`**;
  `[action-delta] scale=[0.00704, 0.15765, 1.4479, 0.41439]`;
- both monitor `val/metric/proprio/pointwise_error`; one PID per GPU (66.4 GB each).
Watchers running as tracked tasks (20-min snapshot ladder + §8.16 tripwire + staleness); their exit
notifies the agent. ETA ~150 ep ≈ 16–17 h → ~10:30 tomorrow.
Runs: `logs/train_world_model_2026_09_01_17_37_53_xtcav_all_r10B_{ctrl,dl2}`.
**Standing interpretation (§8.58 P0-3): physics_alpha=0 in both — tonight tests Δa under a
uniform-bag loss, not the token mechanism; the mechanism arm (10B-tokw) launches on the dl2 base.**

**§8.59 addendum — first-night event (20:2x): FALSE ALARM, tripwire recalibrated.** The ctrl watcher
tripped on `grad/clip_ratio=684.5`. Diagnosis before action: the run itself was healthy (train_loss
0.82 ↓, val improving, dl2 clean at median 3.2), and **r9's own successful run hit transient
clip-ratio spikes of 29,879 and 35,360** (median 1.4) — single-point spikes are this recipe's normal
behaviour, and the >50 threshold was adopted without checking that baseline (my miss). Fixed to a
PERSISTENCE rule (last 3 logged values all > 200); verified quiet on ctrl's current metrics; both
watchers restarted with the fix — the snapshot ladder was the real thing at stake, since a tripped
watcher stops snapshotting. (Also: the pgrep/pkill self-match trap fired a FOURTH time on the watcher
cleanup — killed my own wrapper, exit 144; the bracket pattern `r10B_[w]atch.sh` is now the habit.)
Training was never touched; both arms at ep ~23/150, ETA ~12:00 tomorrow.

**§8.59 addendum 2 (21:2x) — second false trip, flip rule fixed AS SPECIFIED.** The dl2 watcher
tripped on `flip/nonflip ratio 1.531 > 1.5× run-min 0.943`. Diagnosis: I had implemented §8.16's
rule on RAW per-val-epoch points; the raw series bounces ±30% (…1.10, 0.94, 1.53) and the "min" was
itself a noise excursion — §8.16 specifies the rule on an **EMA**, and the collapse signature is the
ABSOLUTE flip loss RISING while nonflip falls; here both fall (flip 0.183→0.12 band, share 0.173
in-band, clip 2.8, val healthy). Tripwire now: EMA(0.5) ratio > 1.5× post-warmup EMA-min **AND**
flip loss > 1.3× its recent min. Verified quiet on both arms' live metrics; watchers restarted.
Training untouched throughout; dl2 at ep 31/150, ctrl in step, ETA ~12:00.
Meta-note for the record: two false trips in one night, both from adopting a reviewer threshold
without back-testing it against a healthy run's history — the same class of error as the metric
failures this campaign keeps cataloguing, now in the monitoring layer. Rule adopted going forward:
**no tripwire arms without a back-test against r9 + one healthy Stage-A/B run.**

**§8.59 addendum 3 (09-02 00:53) — a REAL event this time, assessed: intermittent gradient
blow-ups in BOTH arms, with recovery; no intervention.** The corrected §8.16 tripwire fired on dl2
with corroboration (flip EMA 1.62 > 1.5× min AND flip rising; val_loss 1.57 → 4.65; best stuck at
ep 35). Cross-arm series pulled before acting:
| | ctrl | dl2 |
|---|---|---|
| clip_ratio spikes | 6360, **8.2e9**, then 2.1–3.2 (healed) | 658, **1.3e12** (in progress) |
| proprio_pw | 13.5 spike → **8.03 (its best)** | 29.7 → 18.0 (recovering) |
| flip / nonflip | mild, recovering | 0.663/0.43 → 0.575/0.311 (recovering) |
**Both arms show the same event class — so it is the RECIPE (fresh dynamics at full lr=1e-4 on a
trained codec, no LR warmup segment; r9's segmented schedule re-warmed each time and still spiked to
35k), not the Δa channel.** Grad-clip at 1.0 contains it; both arms recover; best.ckpt + the 20-min
snap ladder preserve pre-event states. Decision: LET RUN. The assessed flip-EMA alarm is now
log-only; added a sustained-divergence trip instead (2 consecutive val proprio_pw > 25); nan /
staleness / persistent-clip trips unchanged. Morning items: (a) compare arms at matched epochs ON
THE LADDER, not just at 150 — these events add noise that snapshot-ladder scoring averages over;
(b) if Stage-B-style warm starts recur, add an LR warmup (r9 had one per segment; these runs have
p_tf_warmup only).

### §8.60 dl2 DIVERGENCE: ROOT-CAUSED TO THE DELTA PATHWAY; dl2b RELAUNCHED WITH ZERO-INIT (09-02 01:5x)

Escalation of §8.59 addendum 3: dl2's event did NOT heal — `grad/nonfinite_skipped=1`, clip_ratio
658 → 1.3e12 → **inf**, val 1.57 → 4.65 → 5.68, best stuck at ep 35. Per-module gradient norms
localize it EXACTLY:
| module | dl2 last-6 | ctrl |
|---|---|---|
| **act_enc** | 0.060, 0.036, 0.043, **28.7, 3.2e11, inf** | quiet |
| backbone | 0.16 → 658 → 1.2e12 → inf (downstream of act_enc) | healed |
| every decode/encode head | sane throughout (flow ≤ 7.2) | sane |
**The Δa channel concentrated gradient into `act_enc` — as designed — and at lr 1e-4 with no warmup
that pathway is unstable.** ctrl's superficially similar event healed; dl2's did not: same event
class, different severity, arm-specific origin. This is itself evidence the channel engages the
action pathway (nothing else in five runs ever moved act_enc's gradient off ~0.05).

**Action (01:30–01:55):** dl2 killed at ep 67 (ctrl untouched); **delta input columns of
`act_enc.net[0]` are now ZERO-INITIALIZED** whenever `action_delta` is on — the channel starts
exactly inert (verified: output invariant to the delta features at init; setpoint columns keep
standard init, so `action_delta=false` remains bit-identical) and grows in, a warmup by
initialization — the same zero-init-residual trick as linear_skip/carrier_proj. No lr change, so
the A/B against ctrl keeps exactly one variable. Relaunched as **`r10B_dl2b`** (GPU 1; warm-start
report identical to dl2's: 244/251, act_enc skipped; delta scale confirmed); watcher up. New ETA
~11:30 for ctrl (ep 150), dl2b ~17:30 — **matched-epoch LADDER scoring (already the §8.59 plan) is
how the arms compare tomorrow**, not end-of-run. One shell note: the relaunch initially failed
because zsh does not word-split unquoted $SHARED (bash does) — launched via bash script.
dl2 attempt 1's run dir + ladder (through ep ~63) are retained for the instability post-mortem.

**§8.60 addendum (03:50) — ATTRIBUTION CORRECTED: the act_enc blow-up is in BOTH arms.** ctrl's
watcher tripped on a skipped nonfinite step at ep ~87; its per-module series shows `act_enc`
0.04 → 6.6e7 → **inf** — the SAME signature as dl2, in the arm with NO delta channel. §8.60's
"arm-specific origin" is therefore wrong: **the origin is act_enc in both arms; the r10B recipe
(warm-started codec + fresh dynamics, no LR warmup) is the cause.** dl2's delta channel made the
event earlier (ep 63 vs 83) and non-healing; it did not create the class. The zero-init fix remains
correct for dl2b (it softens exactly this pathway early). ctrl healed its first event, best.ckpt is
its run-best (ep 59), and the skip guard withheld the bad step — LET RUN. Nonfinite trip moved to a
persistence rule (2 consecutive) to match. Morning question queued: why does the ONLY module both
loss groups' action gradients flow through blow up every ~20–30 epochs in this recipe — flip-batch
weight spikes (w_b ≈ 6–7 on 1–2% of steps) interacting with a fresh act_enc is the first suspect;
an LR warmup or a flip-α ramp is the likely fix for future warm starts.

**§8.60 addendum 2 (04:20) — ctrl entered the non-healing state too; let limp, snapshot-only.**
ctrl's event turned persistent (nonfinite_skipped [1,1], act_enc inf ×2 cycles, proprio_pw 8.3 →
11.8, best stuck at ep 59). **Both arms therefore reach the same act_enc instability endpoint at
ep ~60–90 — the r10B warm-start recipe has a hard stability defect, independent of the Δa channel.**
Disposition: ctrl runs out its 150 (clip-at-inf zero-scales bad batches — it stalls rather than
corrupts; ladder ≤ ep 83 + best ep 59 already banked; a restart would reproduce the same endpoint),
its watcher dropped to snapshot-only (NOALARM mode). dl2b (zero-init delta) stays fully armed — at
~ep 35 clean so far; whether zero-init DELAYS or PREVENTS the endpoint is exactly what its watcher
now measures. **The A/B conclusion will come from the ladder at matched pre-event epochs; the
instability post-mortem (flip-batch × fresh act_enc; LR-warmup fix) is the first morning workstream,
and the §8.61 run-11 recipe must include it.**

### §8.61 MORNING REPORT: ctrl COMPLETE; dl2b STABLE; THE UNANCHORED-DECODER DRIFT (2026-09-02 11:00)

**Run status.** ctrl finished all 150 epochs cleanly (52-snap ladder; two act_enc instability events,
first healed, second persistent from ~ep 87 — best.ckpt = ep 59, pre-event). dl2b (zero-init delta)
at ep 82+: ONE spike (act_enc 134, clip 2.2e3) that self-healed within a single val cycle, zero
nonfinite skips, proprio_pw stable ~8.5 — through the ep 60–90 window where both prior arms
diverged, **the healthiest arm of the three**; finishes ~17:30.

**NEW FINDING (F2-adjacent, probed on ctrl's ladder with the new `xtcav_r10b_onestep.py`): Stage B
destroys the vector codecs' roundtrip because llw=0 leaves the DECODERS unanchored.**
| ctrl checkpoint | proprio roundtrip floor | 1-step dynamics | persistence | sep floor (physics tok) | sep 1-step |
|---|---|---|---|---|---|
| Stage A (r10A3) | **0.018** | — | 0.343 | **3.0 µm** | — |
| ep 3 | 0.297 | 0.494 | 0.370 | 68.6 µm | 203 |
| ep 100 | 0.271 | 0.394 | — | 119 µm | 333 |
| best (ep 59) | **0.286** | **0.429** | 0.341 | **100 µm** | **201** |
The drift is immediate (by ep 3) and permanent: with proprio/physics llw=0, the vector DECODERS are
trained only by `decode/<name>` on DETACHED PREDICTED bags, so they leave the roundtrip optimum to
fit the dynamics' outputs — the Stage-A codec quality (0.018 / 3 µm) is lost 16×/33× within epochs.
The frozen-ENCODER intent (arm-1f safety) was right; freezing by llw=0 also unanchors the decoder —
a distinction §8.37/§8.45 P4 never drew. **Consequence: 1-step imagined proprio (0.429) is again
worse than persistence (0.341), but now the FLOOR (0.286) is the dominant term, not the dynamics gap
(+0.14).** The imagined-BPM goal needs the floor back.

**RUN-11 FIX (pre-specified): a DECODER-ONLY roundtrip anchor** — keep llw>0 on the vector
modalities but detach the encoder inside `roundtrip_losses` (a per-modality `roundtrip_detach_enc`
flag). The latent stays frozen (arm-1f hazard structurally absent, unchanged) while the decoder is
held at the codec optimum. Expected: floor ≈ Stage-A 0.018/3 µm; the 1-step dynamics number then
measures the DYNAMICS (+0.14-ish), which is the honest quantity for the imagined-BPM claim.

**Instability post-mortem (queued):** both arms' recipe reaches an act_enc-origin blow-up at
ep ~60–90 (§8.60 + addenda); dl2b's zero-init damps it (one small self-healing event so far).
Run-11 additionally needs an LR warmup or flip-α ramp for warm starts.

**Comparison discipline for today:** A/B ctrl-vs-dl2b on LADDER rungs at matched epochs, pre-event
only (ctrl ≤ ep 83); response gates still owed the §8.42 rework (±0.25 in-distribution rung,
N_CTX ~110, held-out rows) before any response number is quoted.

### §8.62 dl2b ENDPOINT + THE MATCHED-EPOCH A/B (2026-09-02 14:50)

**dl2b reached the same act_enc endpoint at ep ~107–115** (act_enc inf ×2 cycles, skips [1,1],
proprio_pw 16.3, best pre-event at ep 91). The zero-init **delayed the endpoint 44 epochs (63 → 107)
and kept every earlier event self-healing, but did not prevent it** — the r10B recipe's instability
needs the real fix (run 11: LR warmup / flip-α ramp), not initialization alone. dl2b STOPPED at
ep 115 (unlike ctrl's 4am let-limp, the GPU now has queued daytime work); 38-rung ladder + best
(ep 91) retained.

**Matched-epoch A/B, 1-step probe (400 held-out contexts each):**
| arm @ rung | proprio 1-step / floor | sep 1-step / floor (µm) |
|---|---|---|
| ctrl ep 56 | 0.425 / 0.289 | 209 / 104 |
| dl2b ep 57 | 0.431 / 0.272 | 212 / 116 |
| dl2b best (ep 91) | 0.430 / 0.288 | 279 / 128 |
| persistence | 0.341 | — |
**On prediction metrics the arms are indistinguishable — exactly as pre-registered (§8.58 P0-3):
under a uniform-bag flow loss the Δa channel was not expected to move prediction quality, and these
metrics do not measure response.** The Δa verdict is a RESPONSE question and awaits the §8.42 gate
rework (in-distribution ±0.25° rung, N_CTX ~110, held-out rows). The dominant defect in both arms
remains the §8.61 unanchored-decoder floor (~0.28 vs Stage A's 0.018).

**State handed back:** GPUs free; ctrl + dl2b ladders complete; run-11 recipe requirements collected
(decoder-only roundtrip anchor §8.61; warm-start stability fix §8.60–§8.62; physics_alpha arm with
the two-sided mask §8.58; gate rework before any response claim §8.42/§8.45/§8.58).

### §8.63 24-H AUTONOMOUS ITERATION — PREP (2026-09-02 ~15:40)

Ryan away 24 h; mandate: pending the results/fixes reviewer, iterate on PREDICTION ACCURACY (image
physics steps + proprio/BPM). Success metric fixed up front: the extended 1-step probe
(`xtcav_r10b_onestep.py`, now with boundary oversampling + image-side sep) against ctrl-best
baselines — proprio 0.429 (bnd 0.667) / floor 0.286 / persistence 0.341 (bnd 0.755); sep token
201 µm / floor 100; sep image 183 µm. Response gates are OUT OF SCOPE for this window (still owed
the §8.42 rework). Noted at small n=10: ctrl's dynamics BEATS boundary persistence on proprio
(0.667 < 0.755) — first proprio-prediction win anywhere; re-scored with oversampling next pass.

**Built, unit-verified, default-off (launch pending the reviewer's verdict):**
1. `roundtrip_detach_enc` (per-modality): decoder-only roundtrip anchor — enc grad exactly 0, dec
   anchored (the §8.61 fix).
2. `flip_weight_cap`: the per-batch flip weight is UNBOUNDED as n_f→1 — measured **112.8** on a
   single-flip batch at batch 36 (α=0.12). A state-dependent, rare-batch amplifier is consistent
   with the LATE blow-ups. Cap verified binding.
3. **LR warmup is refuted as the §8.60 stability fix**: r10B's resolved config already had
   `lr_warmup_steps: 300` — the blow-ups fired at ep 60–115 THROUGH a warmup. My §8.60/§8.61
   proposal was wrong on that point; the flip-α ramp likely shares the flaw (early-phase fix, late
   trigger). The w_b cap is the surviving in-house candidate, pending the reviewer's C2 evidence.

**Plan for the window** (each launch smoked first; watchers + persistence tripwires; one variable
per arm; ladders scored at matched epochs): integrate reviewer verdict → run-11 arms on both GPUs
(~11 h at 100 ep) → ladder scoring → one follow-up iteration if the window allows. All records here;
nothing written to lab-notebook (offer stands for Ryan's return).

### §8.64 RESULTS REVIEW: BOTH §8.61/§8.62 DIAGNOSES OVERTURNED; THE REAL CAUSES, MEASURED (2026-09-02 ~16:30)

Adversarial review of the Stage-B story; its load-bearing claims re-verified by me at the artifacts.
This section RETRACTS or corrects four of my own §8.59–§8.62 conclusions.

**1. THE INSTABILITY IS A bf16 FlexAttention BACKWARD OVERFLOW — not a recipe/warm-start defect,
not flip batches. [MEASURED, re-verified.]** At ctrl's dead ep-87 checkpoint, same batch: loss
0.1817 identically, total grad norm **1.76e10 under bf16 autocast vs 4.25 in fp32** (reviewer:
inf vs 4.22 over 16 batches; hooks localize to `blocks.3.t_attn` backward). The clock is unbounded
temporal attention-LOGIT growth (block-3 max |logit| 42 → 17,152 across the run); the kernel's bf16
backward overflows past a threshold. Flip batches are excluded three ways (share sum-normalized and
bounded at α by construction; **n_f=0 batches blow up identically**; realized share in-band at every
event). LR warmup was already deployed (300 steps) and irrelevant. **The same signature exists in
r9** — fresh init, the champion recipe — so §8.60–§8.62's "warm-start recipe defect" is RETRACTED;
this defect predates run 10 and plausibly IS r9's ep-159 maintenance collapse. My w_b-cap hypothesis
(§8.63): mechanism real (112.8 measured) but NOT the trigger — refuted by the n_f=0 blow-ups.
**Fix implemented + verified: `model.temporal_attn_fp32`** (temporal attention pinned fp32 under
autocast, both forward paths incl. KV-cache; default off = bit-identical). At the dead checkpoint,
under the training's own autocast: 8.9e10 → **4.3**. Root-cause option (QK-norm / logit soft-cap)
deferred — architecture change, needs its own arm.

**2. ctrl WAS DEAD FOR 62 OF ITS 150 EPOCHS. [Re-verified: ep87 vs ep147 max|Δweight| = 0.00e+00
across 25,592 optimizer steps.]** My "let it limp — it stalls rather than corrupts" (§8.60 add. 2)
was half right and operationally wrong: it corrupted nothing but also TRAINED NOTHING — 6.4 GPU-h of
no-ops, while `grad/norm_postclip = 0` was being logged the whole time. Dead-run watchdog added to
the tripwire. Also from the review: the `grad/*` series are SINGLE-BATCH samples, not epoch
aggregates (`reduce_fx` isn't applying — norm_preclip ≡ clip_ratio at all 37 epochs), so every §8.59–
§8.62 timing statement carries that caveat; and each "self-healing" spike cost ~3 dB of image codec
(ctrl ep56, dl2b ep71 are −3.3 dB PSNR outliers).

**3. THE DECODER "DRIFT" IS A CORRECT ADAPTATION — §8.61's fix is REJECTED. [Reviewer-measured,
decisive: the decoder-splice experiment.]** Splicing r10A3's decoders into ctrl ep56 restores the
floor exactly (0.0197 / 2.5 µm) and makes 1-step imagined proprio **45% WORSE** (0.425 → 0.594) and
sep 38% worse (209 → 289). Mechanism = **§8.56 F2 realized**: the flow over-predicts the ln_carrier
log-σ dim by ~1.30× [1.15, 1.45], and the exact inversion turns that into a 1.3× multiplicative gain
error on all 138 channels (fixing just those 2 dims of a predicted token halves its decode error,
1.00 → 0.53). The drifted decoder traded inversion exactness for carrier robustness — correctly.
Positive control: the image modality (llw=10, no carrier exp()) held its roundtrip flat all run.
§8.61's "floor + dynamics-gap" decomposition is RETRACTED (not additive), and the §8.63
`roundtrip_detach_enc` anchor is DEMOTED to a measurement tool (it would un-fix prediction).
F2's pre-registration was the answer all along and went untested — the §8.58-named failure mode,
one section later.

**4. THE §8.62 A/B WAS UNINFORMATIVE, NOT CONFIRMATORY.** Paired bootstrap Δ(1-step) = +0.011
[−0.008, +0.032] — the CI is 3× the difference; within-arm rung-to-rung sd (~0.009) ≥ the between-arm
gap; the ctrl ep56 rung was a post-event −3.3 dB outlier; and a NEGATIVE-CONTROL quantity (the
decoder floor, which Δa cannot touch) differs between arms MORE than the metric of interest → both
offsets are seed noise. One seed per arm. "Indistinguishable as pre-registered" conflated a
predicted null with an uninformative one. Nothing was learned about Δa in either direction.

**AMENDED FIX LIST (supersedes §8.61/§8.63):**
| # | fix | status |
|---|---|---|
| 1 | `temporal_attn_fp32` | implemented, verified at the dead ckpt |
| 2 | dead-run watchdog (postclip==0) | in the tripwire |
| 3 | carrier handling for PREDICTION (scale carrier dims / per-dim flow weight / drop carrier) | next-iteration arm; needs Stage-A rerun or flow.py event-dim weighting |
| 4 | two-decoder split: frozen Stage-A decoder for MEASUREMENT (post-hoc splice suffices), adapted head for prediction | adopted as an EVAL convention, no run change |
| 5 | physics-token readout floor (2.5 → 104 µm in Stage B) must be resolved before the §8.50 gate uses it | queued with the gate rework |
| 6 | epoch budget 150 → ~50 (both arms converged by ep 20–30); ≥2 seeds per arm | adopted for run 11 |
| 7 | QK-norm / logit cap (root cause) + per-block logit telemetry + reduce_fx logging fix | deferred, own arm |
| 8 | flip-α ramp / LR warmup / w_b cap as stability fixes | RETRACTED (trigger mismatch) |

**RUN 11 (launching): 2 seeds × (ctrl recipe + `temporal_attn_fp32`), 50 epochs.** One variable vs
10B-ctrl (the precision fix); the second seed measures the seed-noise scale that made §8.62
uninterpretable — the prerequisite for EVERY future A/B on these metrics. Scored on the multi-draw
extended probe at matched rungs.

**§8.64 addendum — run 11 launched (16:5x); iteration-2 pre-registration.** r11_s0/r11_s1 up (2
seeds × ctrl-recipe + `temporal_attn_fp32`, 50 ep, watchers + dead-run watchdog), ETA ~23:00. The
probe now uses an 8-draw ensemble median (§8.50). Scoring at matched rungs with bootstrap CIs over
contexts; the seed pair gives the seed-noise scale directly.

Pre-registered iteration-2 (launch ~23:30, decided by r11's outcome, both mechanism-matched to the
reviewer's measured 1.30× carrier-log-σ over-prediction — the largest identified imagined-BPM error):
- **(a) proprio-token flow upweighting** (the §8.43 per-token machinery pointed at `proprio`,
  α ≈ 0.2): make the flow predict the proprio token — including its carrier dims — better; 2 seeds ×
  50 ep on GPU 1 sequentially.
- **(b) no-carrier Stage A + Stage B chain** on GPU 0 (~12.8 h): rerun Stage A with `ln_carrier`
  OFF (keep linear_skip; accepts m20c's 0.032 floor instead of 0.011) then a 50-ep Stage B — removes
  the exp(log-σ) amplification path entirely.
Interpretation rules, stated in advance: r11-vs-10B-ctrl at matched rungs measures ONLY the fp32 fix
(and its healthy-tail benefit); s0-vs-s1 measures seed noise, and NO between-arm difference smaller
than that spread may be claimed in any later A/B; if (a) improves imagined proprio but degrades
image-side prediction (token-weight reallocation, §8.45 B7 risk), report both — no cherry-picking
the moved metric.

### §8.65 RUN 11 RESULTS + ITERATION 2 LAUNCHED (2026-09-02 ~23:00)

**Run 11 (2 seeds × ctrl-recipe + `temporal_attn_fp32`, 50 ep): the fix works, and every mandate
metric improved.**
- **Stability by intervention**: max clip_ratio **3.15 / 3.65 across the entire runs, zero nonfinite
  events** — the identical recipe minus the flag produced 1e9–inf spikes and terminal freezes in all
  three r10B arms. The bf16-FlexAttention-backward diagnosis (§8.64) is confirmed causally.
- **8-draw-ensemble probe, boundary-oversampled (n=360, 162 boundary), 10B-ctrl re-scored with the
  IDENTICAL probe:**
| arm | proprio dyn (bnd) | sep tok µm (bnd) | sep img µm (bnd) |
|---|---|---|---|
| 10B-ctrl best | 0.406 (0.508) | 232 (289) | 244 (305) |
| **r11 s0 / s1 best** | **0.397/0.395 (0.490/0.486)** | **214/214 (247/245)** | **183/183 (244/244)** |
| persistence (this mix) | 0.467 (0.570) | — | — |
- **Seed spread ~0.002 / ~1 µm** — the ensemble+oversampling probe is ~4× tighter than §8.62's
  single-draw rungs, and the improvements (−0.010 proprio, −18/−43 µm sep tok, −61 µm sep img) are
  3–10× the seed noise → real. Achieved in ONE THIRD of 10B's epochs.
- Note on baselines: persistence is 0.467 on THIS boundary-heavy mix (vs 0.341 uniform), and 10B-ctrl
  also beats it here — the r11-vs-ctrl delta, same probe, is the claim; "beats persistence" is
  sampling-dependent framing and is NOT claimed as new.

**Iteration 2 launched (~23:00), per the §8.64-addendum pre-registration:**
- **r12a** (GPU 1): 2 seeds × (r11 + proprio-token flow upweighting, `physics_modality=proprio`,
  α=0.2, NO validity mask — proprio has no ok flag). Smoked first (alpha>0 in-train for the first
  time; config verified in the resolved yaml). One variable vs r11 at matched seed.
- **r12b** (GPU 0): chained no-carrier codec — Stage A rerun with `ln_carrier=false` (keeps
  linear_skip; accepts m20c's 0.032 floor) → auto-gate (A4 < persistence, else abort) → 50-ep
  Stage B. Endpoint test of the §8.64 carrier trade-off.
Both done ~12:00–12:30 tomorrow; scoring + final window report before Ryan's return.

**§8.65 addendum (23:55–00:15) — r12a_s0 nan at ep 7: LATENT telemetry bug exposed by the token
weighting; fixed; chain relaunched.** Chain: (1) the §8.16 flip telemetry computes
`per[flip].mean()` — nan on a ZERO-flip batch; (2) historically unreachable (zero-flip batches had
`w_step=None`); (3) token weighting makes `w_step` always non-None → the block ran on ~5% of batches;
(4) `lit.py:176` sums `w[k]*raw[k]` without filtering zero weights and **0.0 × nan = nan** poisoned
the TRAINING loss — with perfectly healthy gradients (clip 2.2, postclip 1; the nonfinite guard
skipped the poisoned steps). Fixed with a `flip.any() and (~flip).any()` guard; verified finite on
zero-flip + alpha-on batches with a realistic bag. The nan tripwire caught it inside one epoch. The
relaunch was then blocked once by the run-note uniqueness assert (notes rewritten to describe the
restart — the honest note anyway). New ETA: s0 ~06:30, s1 ~12:50; r12b chain unaffected on GPU 0.

### §8.66 OVERNIGHT: NO-CARRIER STAGE A PASSES EVERYTHING; CHAIN ADVANCED (2026-09-03 05:40)

**r12bA (Stage A, `ln_carrier=false`, linear_skip kept) — ALL GATES PASS, two bests-of-campaign:**
| gate | r12bA | r10A3 (carrier) | bar |
|---|---|---|---|
| A4 proprio roundtrip | **0.038** | 0.011-class (0.018 measured) | < 0.337 — and within 20% of m20c's predicted 0.032 |
| A1 / A3 sep err | 61 / **244** | 61 / 305 | <61 / <300 — **A3 back under the bar** |
| A2 grid imprint | **0.995** | 1.014 | ≤1.10 |
| A6 image PSNR | **44.06 dB** | 42.97 | **best of campaign** |
| A8 physics sep | 2.85 µm | 2.97 | <61 |
The chain's automatic gate (`A4 < persistence`) passed → **r12bB (Stage B on the no-carrier codec)
launched itself at ~05:20**, watcher attached. The §8.64 trade-off question — does giving up 3× on
the roundtrip floor (0.011 → 0.038) buy back the 2.7× carrier penalty on PREDICTED tokens — is now
an empirical head-to-head: r12bB vs r11 at matched rungs, ~12:00.

**r12a_s0 (proprio-token upweighting, post-nan-fix) completed 50 epochs cleanly**; the chain rolled
into s1 automatically. Scoring of everything (r12a s0/s1, r12bB, vs r11 s0/s1) queued for ~12:30 on
the 8-draw probe; final window report before Ryan's return.

### §8.67 24-H WINDOW CLOSE-OUT: FINAL SCORES (2026-09-03 20:30)

All runs completed ~11:15 (scoring executed at Ryan's return; the completed-chain notifications did
not re-wake the session — the ~9 h gap was idle, not compute). 8-draw ensemble probe, identical 360
contexts (162 boundary), r11 seed noise = ~0.002 proprio / ~1–2 µm sep:

| arm | proprio dyn (bnd) | sep tok µm (bnd) | sep img µm (bnd) | floors (pro / tok) |
|---|---|---|---|---|
| 10B-ctrl best (pre-window) | 0.406 (0.508) | 232 (289) | 244 (305) | 0.266 / 117 |
| **r11 s0/s1** (fp32 fix) | 0.397/0.395 (0.490/0.486) | 214/214 (247/245) | **183/183 (244)** | 0.24 / 91–107 |
| **r12a s0/s1** (+proprio-token α=0.2) | 0.401/**0.390** (**0.480/0.479**) | 233/205 (282/254) | 183/183 (244) | 0.23 / 105–112 |
| **r12bB** (no-carrier codec, 1 seed) | 0.411 (0.508) | **176 (237)** | 214 (274) | 0.25 / **81** |

**Verdicts (against the pre-registered rules):**
- **r12a**: a real, seed-consistent BOUNDARY-proprio gain (0.480/0.479 vs r11's 0.490/0.486; both
  seeds agree; ~4× r11's seed noise) at no image cost. Pooled-proprio and sep-token moves are inside
  r12a's own (larger) seed spread — not claimed. The mechanism arm did what its mechanism predicts,
  modestly.
- **r12bB**: the carrier trade-off is REAL AND TWO-SIDED — the physics-token readout improves
  substantially (sep 214 → **176**, floor 91 → **81**; and its Stage A had the best image PSNR of the
  campaign) but proprio dynamics regresses to ctrl level (0.508 bnd) and image-side sep worsens
  (183 → 214). Single seed; the direction split is bigger than r11-scale seed noise but unconfirmed.
  **Neither carrier setting dominates: carrier-on wins proprio+image, carrier-off wins the physics
  token.** §8.64's fix-3 options (carrier-dim scaling / per-dim flow weights) remain the candidates
  for getting both.
- **Best current models:** r12a_s1 for proprio/BPM (0.390 pooled / 0.479 boundary — campaign best),
  r11 (either seed) for the balanced profile, r12bB where the physics-token readout matters.

**Window totals vs the pre-window baseline (10B-ctrl), same probe:** proprio 0.406 → 0.390
(boundary 0.508 → 0.479), sep image 244 → 183 µm, sep token 232 → 176–205 µm, floors improved,
**zero instability events across five run-halves** after the fp32 fix, at a third of the prior epoch
budget per run. Plus: the bf16 root cause (also r9's), the carrier trade-off quantified from both
ends, the seed-noise scale established, three watchdog classes armed and back-tested, and two latent
bugs (telemetry-nan, dead-run) found and fixed by the monitoring itself.

**§8.67 addendum — figures.**
- `logs/physics_eval_xtcav_e300/r12_figs/r12_bpm_accuracy.png` — imagined-BPM accuracy across
  10B-ctrl / dl2b / r11×2 / r12a×2 / r12bB on identical contexts (8-draw ensemble, 95% CI over
  contexts), pooled + boundary, roundtrip floors and persistence marked; per-channel panel:
  **109/138 channels improved** (r12a_s1 vs 10B-ctrl).
- `logs/physics_eval_xtcav_e300/r12_figs/r12_step_prediction.png` — 1-step separation prediction
  across scan-step boundaries, physics-token vs image-side readouts, hold-vs-boundary breakdown.
  Titled explicitly as PREDICTION, not the causal response gate (§8.42 pending). Context-level CIs
  are wide (between-context spread); the paired seed-noise comparison in §8.65/§8.67 is the tighter
  inter-arm statistic.
- `<r12bA run dir>/figs/stageA_{roundtrip_filmstrip,quant,background}.png` — codec roundtrip
  validation for the NO-CARRIER codec vs r9 on identical frames: 39% exact-sep frames, 138/138
  proprio channels better, physics token at 2.9 µm; note its p90 reads 305 on this 502-frame figure
  sample vs 244 on the 300-frame gate sample — the 61 µm grid again; both are 4–5× better than r9's
  1220. (r10A3's equivalent figures exist from §8.52.)

### §8.68 THE PAIRED COUNTERFACTUAL AT REAL ±0.25° STEPS: EVERY MODEL'S RESPONSE IS ZERO (2026-09-03 21:30)

Ryan: "this doesn't seem very significant — visualize the imagined images at these steps — are the step
sizes fair per our prior finding?" Built `xtcav_r12_stepviz.py`: 100 held-out BOUNDARY contexts (real
±0.25° steps, valid physics), 8-draw paired ensembles under the TRUE action vs the SAME context with
the last action's L2 REVERTED (no step). Figures in `logs/physics_eval_xtcav_e300/r12_figs/`
(`r12_step_filmstrip.png`, `r12_step_counterfactual.png`).

**Fairness: the step sizes are fair — all real, in-distribution ±0.25° steps (§8.34). And that makes
the result worse, not better.**

| model | image-side Δsep_cf median / sign-agree | token-side Δsep_cf median / sign-agree |
|---|---|---|
| 10B-ctrl | **0 µm** / 5% | 0 µm (|cf| 2) / 27% |
| r11 s1 | **0 µm** / 9% | 0 µm (|cf| 5) / 30% |
| r12a s1 | **0 µm** / 8% | 0 µm (|cf| 2) / 27% |
| r12bB | **0 µm** / 2% | 0 µm (|cf| 0) / 15% |
| real |Δsep| at these steps | **244 µm** median | |
**Stepping the knob changes the predicted image by nothing measurable, in every model.** The
difference images are speckle with no coherent structure. Sign agreement at or below chance.

**Consistency check from the prediction data (§8.67 dumps):** if a model IGNORED the step, its
boundary error would be ≈ √(hold² + step²) = √(183² + 152²) = **238 µm**; r11/r12a's measured
boundary error is **244 µm**. Proprio boundary/hold error ratios: models 1.59–1.72 vs persistence
1.88 — modest extra absorption at boundaries, not response. **The window's prediction gains are
better CONTEXT TRACKING, not better step response.** Ryan's read is correct.

**What this settles.** The full modelling ladder has now been applied — codec at the pixel floor,
stability fixed, carrier trade-off characterized, decoder handled, Δa channel, proprio-token
weighting, no-carrier codec — and **not one of them moved the in-distribution response off zero**.
This is §8.37's pre-registered F2 outcome, demonstrated directly rather than inferred, and it matches
§8.34 M-B (r7 ep115 also ≈0 at ±0.25°). At this step size the signal (152–244 µm) sits inside the
per-shot jitter (254 µm) and inside every model's 1-step error; the corpus contains no other step
size. **The lever is the DAQ (randomized, larger L2 steps), not the model.** The single in-house
candidate that targets response rather than tracking is the paired-counterfactual training loss
(§8.47 pt 3), which forces a nonzero Δsep_cf against the measured slope — and it is bounded by the
same SNR. This measurement (100 contexts, paired, in-distribution) is also, in effect, the §8.42 gate
rework's core, delivered: it needs the direction-aware scoring and a held-out split before being
called the official gate, but its answer will not change.

### §8.69 IS THE SEPARATION EXTRACTOR BRITTLE? TESTED — AND THE §8.68 VERDICT MADE EXTRACTOR-FREE (2026-09-04)

Ryan, on the §8.68 filmstrip: "many of these look subjectively very similar between context and next, however
the separation change can change often by very large amounts. this actually seems like an error with how
separation is being computed in a brittler manner than it should be — the wm predictions look very similar to
xtcav but simply measuring the delta sep makes it seem like there's huge changes which aren't there."

Two questions, tested separately: (1) is the deployed extractor brittle? (2) does the §8.68 zero-response
verdict depend on it? Scripts: `wizard/scripts/xtcav_sep_brittleness.py` (T1–T5, REAL held-out frames only,
5816 frames / 5337 valid, no model), `xtcav_sep_t6_fixed_split.py` + `xtcav_sep_t6b_direction.py` (T6),
`xtcav_sep_t7_pixel_lag1.py` (T7), `xtcav_r12_pixel_response.py` (extractor-free scoring of the stepviz
counterfactual). `xtcav_r12_stepviz.py` patched: BOTH frames must be valid (see bug below), pixel NCC in the
column titles, a third `true_b` arm (true action, fresh seeds = the seed-noise floor), images dumped to the npz.
Figures: `r12_figs/sep_brittleness.png`, `r12_figs/r12_pixel_response.png`, regenerated `r12_step_filmstrip.png`.

**(1a) Per frame the extractor is NOT brittle (T1).** Same frame, perturbations no eye would see:
| perturbation | \|Δsep\| p50 | p90 | >300 µm | became invalid |
|---|---|---|---|---|
| additive noise σ=2 u8 | 0 | 61 (1 px) | 0.7% | 0.6% |
| intensity ×0.97 / ×1.03 | 0 / 0 | 61 / 0 | 0.1% / 0.0% | 0.1% / 0.0% |
| 1-row shift | 0 | 0 | 0.0% | 0.0% |
| 0.5-px blur | 0 | 0 | 0.3% | 0.6% |

**(1b) Look-alike frames DO produce small Δsep (T2).** 5155 consecutive valid pairs (139 boundary). Pixel
similarity of consecutive REAL frames is low — NCC p50 **0.906** within plateau, 0.766 at boundaries. Among the
20% of plateau pairs that are genuinely near-identical (NCC > 0.97): |Δsep| p50 **61 µm**, >300 µm in **1.4%**.
Among all plateau pairs: p50 122 µm, >300 µm in 27%. So large Δsep values come with real pixel change; the eye
discounts intensity redistribution along a streak, the band MEDIAN does not. **Every one of Ryan's look-alike
columns had NCC 0.57–0.77** (regenerated filmstrip: the NCC 0.93–0.96 columns carry |Δsep| 61–183 µm, the NCC
0.57 / 0.76 columns carry +915 / +1281 µm).

**(1c) The brittleness that DOES exist, and its size (T3, T6).** The 1.4% cases are real and the mechanism is
in the figure's middle row: on a continuous curved streak the smoothed energy projection has an almost flat
valley, so the argmin split row jumps 7–11 rows between frames with NCC 0.97–0.98 (peak rows barely move), and
on a diagonal streak each row of split shifts both half-medians along the streak axis — a ~60–100 µm/row lever
arm, ~600 µm per event. Quantified over all plateau pairs: the peak PAIR changes in 23% of pairs (robust sd of
Δsep 452 µm there vs 181 µm when the pair is stable); split-row jitter p50 1 row, p90 4. Recomputing with the
split row HELD at the plateau median:
| variant | plateau \|Δ\| p50 | >300 µm | boundary \|Δ\| p50 | bnd/plateau | valid |
|---|---|---|---|---|---|
| deployed | 122 | 27.0% | 305 | 2.50 | 92.7% |
| deployed + interpolate | 128 | 24.0% | 265 | 2.07 | 92.7% |
| fixed split (plateau median) | 122 | 24.5% | 244 | 2.00 | 100% |
| fixed split + interpolate | 125 | 21.5% | 252 | 2.02 | 100% |
Fixing the split removes ~10% of the >300 µm jumps and REDUCES the boundary/plateau contrast (part of the
split movement at a boundary is the physical response). **Verdict on (1): a real but minor defect (~10% of the
large jumps); ~90% of the "jitter" is beam, not extractor.** Also (T4): 44% of valid frames are CONTINUOUS
streaks (valley depth ≥ 0.5 of the lower peak) with no genuine second energy band; their "separation" is a
charge-distribution proxy, not a two-bunch spacing — identical jitter statistics to two-band frames (p50 122,
>300 in 24–25%), but a different physical meaning. Not a bug; a caveat on what `sep_um` measures on E331-like frames.

**(1d) A bug in MY §8.68 figure, fixed.** The stepviz boundary filter required only frame t to be valid. For an
INVALID frame the store carries the imputation constant (z = 0 → the corpus mean, ≈1518 µm), and the old
column 6 displayed it as a measured "1518 → 366 µm, Δ −1152". 2/100 contexts were affected; median |real Δsep|
244 → 214 µm on the affected set, **244 µm on the regenerated both-valid set** (unchanged). No conclusion moved.

**(2) The §8.68 verdict re-scored WITHOUT the extractor (`r12_pixel_response.py`, 100 both-valid held-out
±0.25° boundaries, 8-draw medians, paired seeds).** RMSE in u8 counts over the 64×192 grey frame:
| model | RMSE→real: true / held (persistence 4.36) | benefit of the true step, p50 [95% CI] / sign>0 | response RMSE(true,hold) vs seed floor RMSE(true,true_b) | cos(true−hold, real Δ) |
|---|---|---|---|---|
| 10B-ctrl | 4.02 / 4.01 | −0.001 [−0.004, +0.003] / 47% | 0.16 vs 0.79 (**0.22**) | +0.002 |
| r11 s1 | 4.01 / 4.03 | −0.001 [−0.006, +0.006] / 49% | 0.22 vs 0.88 (**0.25**) | +0.000 |
| r12a s1 | 3.97 / 3.97 | −0.000 [−0.003, +0.002] / 50% | 0.17 vs 0.80 (**0.21**) | +0.003 |
| r12bB | 4.01 / 4.01 | +0.001 [−0.001, +0.002] / 51% | 0.13 vs 0.79 (**0.18**) | −0.004 |
Smooth observables (no split row involved): streak-axis RMS length response **1–2 µm** vs seed floor 17–29 vs
the REAL step's 108 µm; streak centroid 1–3 vs 11–17 vs 21. **Telling the model about the step changes its
prediction by one fifth of what re-rolling the noise seeds changes it, buys 0.000 RMSE toward the real next
frame, and points in no direction. §8.68 stands, and no longer rests on `sep_um`.** The models ARE ~8% better
than persistence (4.0 vs 4.36) — context tracking, as §8.68 said; on long-streak frames they are often WORSE
than persistence in NCC (filmstrip col 2: 0.55–0.68 vs 0.76), i.e. the dynamics degrades the rendering there.

**Two corrections to the noise framing, both measured on real frames (T5, T7):**
- **Polarity is not fixed.** Pooled over 156 plateau-mean steps, Δsep has the sign of ΔL2 in **44.9%**
  (binomial p = 0.23); 69% of episodes respond with one polarity, 31% with the other. ∂sep/∂L2 changes sign
  with operating point (compression side / which band is which). The streak RMS length DOES carry a consistent
  polarity (63% opposite to ΔL2, n = 160, p ≈ 0.001), as does the energy RMS (64%). Consequence: any score
  "signed by ΔL2" is meaningless for sep; §8.68 correctly signed by the REALIZED Δsep. (A per-episode
  "sign consistency 0.70" I first computed was dropped: its null for ~5 steps/episode is 0.69.)
- **At one step the ±0.25° step is clearly visible in the real data.** Lag-1 change, boundary vs plateau:
  sep 305 vs 122 µm (**2.50×**), streak RMS length 112 vs 49 (**2.27×**), raw pixel RMSE **4.43 vs 2.99
  (1.48×)**, NCC 0.906 → 0.766. In pixel terms ~55% of a boundary pair's change is step-attributable
  (√(4.43² − 2.99²) ≈ 3.3 counts) — against which the models' counterfactual response is 0.13–0.22, i.e.
  **~5% of the real step's pixel effect.** §8.41's per-step SNR 0.60 used the whole-run per-shot sd (254 µm),
  which includes slow drift a context-conditioned one-step predictor never has to fight; the lag-1 innovation is
  p50 122 / robust sd 181 µm. Boundaries are 2.8% of transitions (160/5750 held-out) and `boundary_frac` was 0
  with F = 16 in every run of this window, so pooled over the corpus the step is ~1.5% of residual variance —
  §8.39's "2%" is a SAMPLING fact, not a physics fact. Per boundary sample the step is the majority of the
  residual. BUT this lever has been pulled before: window enrichment 0.35 → 0.5 (runs 3–9 era, ~39–53% of
  windows) "changes nothing about response" (record line ~345) and r7's per-transition flip weighting built
  only a large-signal mode (§8.34 M-B). With F = 16 a boundary transition is still 1 of 16 supervised terms
  even in an enriched window, so neither raised the boundary share of the LOSS much above ~10%. The
  paired-counterfactual loss (§8.47 pt 3), which targets the response itself rather than the boundary error,
  remains the one untested in-house lever; the DAQ request stands for the response FUNCTION (magnitude vs
  step size), not because the in-distribution step is invisible.

**Evaluation rules from this section.** (i) Primary response metrics: the pixel-space paired counterfactual
(response vs seed floor, benefit, alignment) and the streak RMS length — neither depends on a split row and
the RMS length has consistent polarity and a 2.27× lag-1 contrast with far lighter tails than sep. (ii) `sep_um`
stays as the two-bunch spacing where it means that: two-band frames (valley < 0.5), both frames valid, reported
with its heavy tail (27% of plateau steps > 300 µm). (iii) Never sign a response by ΔL2; sign by the realized
change or use pixel alignment. (iv) Display rule: an imputed store value is not a measurement — figures must
check the ok flag on every frame they annotate.
