# lego — running log

Every run on this dataset in order, what it showed, and what we decided next. Newest at the bottom.
Dataset: `swoosh-data/lego_assemblies` (private), 74 episodes / 341,494 frames / 30 Hz, dual xArm7
bimanual VR teleoperation, 6 cameras, LeRobot v2.1. `RecordedEnv` — no simulator, so control eval is
permanently off (`[env-contract] step ✗ no-sim`).

**The goal, as stated by the user (2026-08-31):** *two arms, one GPU each — **scene prediction** on one
camera, and **multicamera** on three.* Arm B is scored on the SAME scene head Arm A predicts, so the
study answers one question: **do the wrist cameras improve scene prediction?**

Branch `lego`. Live checklist with per-item evidence: `design/lego_progress.md`.

---

## 1. The dataset ships with two defects, and neither announces itself

Before any model, two things measured over all 341,494 frames.

**(a) Both rotation representations are discontinuous.** The dataset stores TCP orientation twice, in two
forms, and both break:

| stream | defect | count |
|---|---|---|
| `observation.state` rpy | wraparound at ±180° | 702 roll + 48 yaw (right), 802 + 30 (left) |
| `action` quaternion | sign flip (`q` ≡ `−q`) | 29 per arm, in 29 of 74 episodes |

Euler is the worse one, and not because of wraparound alone: `pitch` reaches ±78°, close enough to the
±90° gimbal singularity that roll and yaw go ill-conditioned and trade off. The sharpest single example
in the data is **rpy jumping 359.8° while the underlying rotation moves 0.38°** — a 360-unit spike in the
network's input for a hair of real motion.

The quaternion cannot be canonicalised away by forcing `qw ≥ 0`. The rotations span essentially all of
SO(3): max geodesic distance from the Karcher mean is 172° (right) / 180° (left), 5.0% / 6.5% of frames
beyond 90°, and `qw < 0` for 74% / 71% of frames. Forcing a hemisphere would flip most of the dataset and
manufacture a discontinuity at every `qw = 0` crossing — strictly worse than the 29 flips already there.

**(b) `action` is not in the state's frame.** Found while trying to identify the Euler convention, which
is why `EULER_SEQ` remains an unverified assumption:

- **No convention fits.** All 12 sequences tried; best is 126.6° median geodesic error between the state
  rotation and the action rotation. The median geodesic distance between two *uniform random* rotations is
  ~126°, so the best fit is indistinguishable from noise.
- **Different spaces.** `action` xyz spans x ∈ [−0.87, 0.58], y ∈ [0.22, 2.69], z ∈ [−0.33, 0.92] —
  metres, y centred on 1.47, consistent with a VR/room frame. State TCP is x ∈ [276, 756] mm,
  y ∈ [−270, 376] mm, z ∈ [−89, 665] mm — robot base frame. Best-axis correlation 0.45, and cross-axis.
- **No fixed extrinsic.** Kabsch fit per session (65 sessions): scale should be exactly 1000 (m→mm),
  measured median 1069 over a 0–2114 range; median residual 133 mm, up to 270 mm. Pairwise rotation
  between session extrinsics 26°–176°.
- **Not deltas either.** Per-episode Kabsch on differences: scales 1859–4920, extrinsics 19°–91° apart.
- **Lag does not explain it.** Sweeping 0–60 frames, correlation peaks weakly (0.589) at ~1 s — far too
  long for teleop tracking.
- **Grippers anti-correlated:** −0.72 (right), −0.57 (left).

**Decisions:** (a) re-encode both rotations to the continuous 6D form in the PROCESSOR, so no consumer can
touch an Euler angle; (b) filed as [quickdraw#15](https://github.com/isaac-ward/quickdraw/issues/15) — the
action channel is learning noise, so **run `_oneoff_action_sensitivity` early and read a near-zero result
as this bug, not as a model failure**. The proprio+image dynamics are unaffected. Also filed the stale
dataset card (65 episodes vs the metadata's 74) as dataset discussion #1.

## 2. 6D rotation encoding (`data/rotations.py`)

First two columns of the rotation matrix (Zhou et al., CVPR 2019), decoded by Gram-Schmidt.
Sign-invariant by construction, no singularity, and closed under averaging/interpolation — which matters
if the action model ever predicts chunks, since averaging quaternions of arbitrary sign is meaningless.

    state   28 -> 34    [tcp_xyz(3) 6d(6) j1..j7(7) grip(1)] x {right, left}
    action  16 -> 20    [xyz(3) 6d(6) grip(1)]               x {right, left}

`tests/test_rotations.py`, 13 episodes / 65,817 real frames, 7/7:

| check | result |
|---|---|
| round-trip geodesic error | 5.4e-06 deg (float32 storage) |
| `6d(q)` vs `6d(-q)` | **0.0 exactly** |
| Lipschitz bound `‖Δ6D‖ ≤ 2√2·sin(θ/2)` | holds everywhere, worst violation 2.4e-08 |
| orthonormal, det=+1 | 8.9e-16 |
| robust to unnormalised/non-orthogonal input | 8.9e-16 |

The continuity check is exact rather than threshold-tuned, and that mattered: two earlier threshold-based
versions of it produced false failures. Since `‖R₁−R₂‖_F = 2√2·sin(θ/2)` and the 6D encoding is two of
three columns, continuity *means* the bound holds for every consecutive pair. Euler obeys no such bound —
that is the bug, stated precisely.

**Proprio width decided (user):** keep all 34. The 7 joints per arm are redundant with TCP pose only up to
null-space configuration — they are the one signal separating two arm poses that reach the same point —
and 34 dims is negligible beside 32 image tokens. `data.obs_keep` stays null.

## 3. `data.subsample` = 6, derived (`smoke/lego_subsample.py`)

The most consequential number in the build, and `bsp32mse.yaml` says so. Record §13 on robocasa found the
per-step image change was **0.61× the reconstruction error of the autoencoder being predicted through** —
the target sat below the codec's noise floor, so predicting zero motion was the *correct* minimiser and
every run scored `motion_ratio` 0.13–0.17 regardless of action conditioning.

Frozen-TAESD floor on THIS data at square 128px: **RMSE 0.0647 / PSNR 23.79 dB**. Robocasa's was 0.0637 /
23.92 dB — near-identical, so §13's table is a valid reference here rather than a loose analogy.

Frame-Δ RMSE vs that floor, 11 episodes spread across sessions, same `[-1,1]` units:

| stride | Hz | frame-Δ RMSE | vs floor | |
|---|---|---|---|---|
| 1 | 30.0 | 0.0500 | **0.77×** | below the codec's own error |
| 2 | 15.0 | 0.0770 | 1.19× | marginal |
| 3 | 10.0 | 0.0963 | 1.49× | clears the floor |
| 4 | 7.5 | 0.1109 | 1.72× | |
| **6** | **5.0** | **0.1323** | **2.05×** | **CHOSEN** |
| 8 | 3.8 | 0.1479 | 2.29× | |
| 16 | 1.9 | 0.1854 | 2.87× | |

**Rule:** smallest stride BOTH clear of the floor (≥1.25×, where robocasa crossed into a learnable target)
AND inside the 2–5 Hz band every published long-rollout system converges on (V-JEPA-2-AC 4 fps, IRASim
~4 fps, HMA 2 Hz; robocasa chose 4 Hz). "First to clear 1.25×" alone gives stride 3, which sits on the
marginal edge and buys the shortest horizon — the very thing subsampling exists to fix. At stride 6,
`F=64` spans **12.8 s** of robot time instead of 2.1 s.

**The prediction was wrong and the reason matters.** Stride 1 was expected to be *worse* than robocasa's
0.61× because 30 Hz > 20 Hz. It measured **0.77× — better**. Real bimanual teleop moves more per frame
than robocasa's data, and that more than offsets the frame rate. **Measure; do not scale by fps.**

Caveat: TAESD is a REFERENCE codec, not vl64's (which trains a bespoke AE from scratch, so its floor is a
training outcome). Worth re-checking against vl64's trained floor now that Arm A has epochs.

## 4. The processor, and where the wall-clock actually went

`lego_assemblies()` in `data/processors.py`. Two design calls forced by scale:

- **Frames stage to disk as JPG PATHS, not arrays.** `build_recorded_dataset` pickles `Episode.frames` to
  its encode workers; the longest episode here is 10,769 frames, so in-memory arrays would push GBs per
  job through IPC. starling already uses the lazy-path pattern for exactly this reason. Resumable via a
  per-episode `.done` marker written only on a FULL decode.
- **Aspect PRESERVED at staging (16:9), not squashed to square.** The squash to the recipe's `img_size`
  belongs downstream in `load_fpv_frames`; baking it in would force a re-decode of all 444 source clips.

**Measured build cost, one camera, 341,494 frames:**

| step | time | note |
|---|---|---|
| stage (16 workers) | **508 s** | 4 eps / 22,393 frames in 79 s vs ~17 min serial (~13×) |
| encode 74 clips | 148 s | |
| **lerobot split write** | **4,369 s** | dominates — round-trips EVERY frame through a PNG, serial per split |

The staging estimate (4.5 h serial) was the thing I optimised; the real cost turned out to be
`write_lerobot_split`, at a fixed ~85 min per camera that no core count fixes at 2 splits. The 3-camera
build took 11,615 s, i.e. 3× that, as predicted. **If this dataset is rebuilt often, the lever is teaching
that writer to ingest the pre-encoded clips already sitting in `media/` rather than re-deriving frames.**

## 5. Multi-camera: what was already generic, and what was not

**Already generic (no change):** the spine (`multimodal.py:322,331` iterate `self.layout`), the training
step (`lit.py:108-111` builds `obs` by looping over every non-proprio modality), eval
(`routines.py:1027-1039` writes one mp4 per image head), `_subsample_episodes` (already decimates every
stream past `ep[1]`), and `_resident_frame_bytes()` (already sums over all image specs, so autobatch sizes
3 cameras correctly).

**Not generic — changed:** `ModalitySpec.cam` so each head declares its camera directory (the head name is
a model-side label, the directory a dataset-side one; positional pairing across two config files is the
kind of silent misalignment worth refusing); `MMWindowLoader` holding one resident uint8 store per head
sharing ONE window index; `load_split_episodes_mm` taking a camera list; `write_lerobot_split` /
`build_recorded_dataset` / `Episode.frames` taking N cameras.

**Two `next((m for m in modalities if kind == "image"))` sites that silently used the FIRST image head:**
the AUTO checkpoint monitor (so `best.ckpt` would be selected blind to two of three cameras — the same
class of miss the comment above it already records for `bott_bott16`) and `model.size` presets. The first
now RAISES rather than guessing; the second applies to every head, which also fixed a latent bug where a
knob clash on a NON-FIRST head was undetected entirely.

Verified: `lego_multicam_smoke` completed a full epoch with 3 heads — `train_loss=19.8193
val_loss=23.0602`, 798.5 s/ep, batch 3.

## 6. `data.action_aggregate` — subsample was SUMMING an absolute action

Caught in a live run's log, not by reading code. `_subsample_episodes` combines the actions it skips over
and assumed they are DELTA-like: sum them, with an auto take-last for dims that look binary (≤2 unique
values). **Both halves are false here.**

1. **The action is ABSOLUTE** (see §1b) — summing six absolute poses gives six times the position.
2. **The binary guard never fires.** The gripper is at exactly 0 or 1 for **94%** of frames but has
   **~1000 unique values** because it ramps between them. The live log read `take-last on dims []`.

Measured at stride 6 on a synthetic absolute ramp plus a gripper with one ramp value:

| mode | dim0 | gripper |
|---|---|---|
| `sum` | [0.25, **5.75**] | [0.00, **6.00**] |
| `last` | [0.08, 1.00] | [0.00, 1.00] |

**Decision:** `data.action_aggregate` = `sum` \| `last`, defaulting to `sum` so every existing dataset is
bit-identical; `last` in `conf/data/lego.yaml`. It would never have crashed — it would have fed the action
encoder and the normaliser a 6×-out-of-scale action and produced a plausible-looking run.

## 7. `subsample_all_phases` — a 50-epoch schedule that needed 30 days

First full-data Arm A attempt, killed 5 min into epoch 0 once its cost was measured. With
`subsample_all_phases: true` it loaded **276,231** train windows = ~14,540 steps/epoch at batch 19, and at
the measured ~0.28 steps/s that is **~14 h/epoch, ~30 days for 50 epochs**, first eval ~3 days out.

Pairing it with a 6× smaller `max_epochs` (~8), which is what the record's own reasoning implies, does not
work either: the eval cadence `{5, 9, 15, 19, …}` yields exactly ONE eval point at 8 epochs.

**Decision:** OFF. ~46k windows → ~2,094 steps/epoch → the stock 50 epochs with all seven eval points.
**What it costs, honestly:** at matched wall-clock the ON variant sees ~6× more DISTINCT windows and
should generalise better. A long final run may want it back with a step-based eval cadence.

## 8. ARM A — `vl64_scene`, one camera (RUNNING)

`logs/train_world_model_2026_08_31_08_33_07_lego_arm_a_scene` · GPU 1 · `data.batch=22`

autobatch: budget 95 GB = card 102.0 − reserve 4.0 − resident frame store ~2.8 (the store is 15.28 GB with
`all_phases` on; off, only every 6th frame is retained). 46,066 train / 5,565 val windows, 2,094
batches/epoch, **9,457 s/epoch (2.63 h)**, `run_eta` finish 09-05 18:49.

`best.ckpt` selects on `val/metric/scene_right/mse` — the image head, not the proprio metric, and named
`scene_right` rather than `image` so Arm B logs the same key and the A/B is directly joinable.

**First validation, epoch 3** (`check_val_every_n_epoch: 4`, so this is the first val point):

| metric | value |
|---|---|
| `val/loss/total` | 188.3930 (train 176.8348) |
| `val/loss/dynamics/latent` | **182.46** — dominates the total |
| `val/metric/scene_right/psnr` | **15.61 dB** |
| `val/metric/scene_right/mse` | 0.028267 → 15.49 dB |
| `val/loss/codec/roundtrip_scene_right_mse` | 0.028615 → **15.43 dB** |
| `val/metric/proprio/pointwise_error` | **396.8 mm** (position-only L2, right-arm TCP) |
| `val/metric/proprio/obs_error` | 1.2609 (normalised, full vector) |
| `grad/nonfinite_skipped` | 0 |

**READ 1 — the image head is CODEC-limited, not dynamics-limited.** Prediction MSE (0.028267) is
essentially identical to the codec's own round-trip MSE (0.028615) — 15.49 dB vs 15.43 dB. The dynamics
cannot be blamed for the image number because the autoencoder it predicts through is itself only at
15.4 dB. For scale, the frozen-TAESD reference floor on this data is **23.79 dB** (§3). vl64's AE trains
from scratch, so this is expected to climb for many epochs; until it does, image metrics measure the AE.

**READ 2 — proprio is worse than a constant predictor.** `pointwise_error` 396.8 mm against a workspace
where the mean distance from the centroid is **249 mm** (right TCP spans 359 / 488 / 691 mm in x/y/z,
per-axis std 95 / 87 / 227). Predicting the centroid every step would score ~249 mm. At epoch 3 of 50
over a 12.8 s open-loop rollout this is not alarming, but it is the number to watch: if it has not gone
below ~249 mm by the epoch 9–15 evals, the model is not learning position dynamics at all, and §1b
(action not in the state's frame) is the first suspect.

### ⛔ EVAL 5 — COLLAPSED. The documented vl64 signature, at eval 5 instead of eval 12.

vl64's own header: *"IT COLLAPSED AT EVAL 12, into a degenerate stable state (latent_cos -0.13, motion
0.31), after one non-finite gradient step... Watch `latent_cos`: going negative is the signature."*

| metric | robocasa collapse (eval 12) | **lego, eval 5** |
|---|---|---|
| `latent_cos_mean` | −0.13 | **−0.376** |
| `motion_ratio_mean` | 0.31 | 0.269 |

`latent_cos` is negative at EVERY horizon: −0.310 @+1, −0.120 @+16, −0.453 @+64, −0.476 @+230. The
prediction is ANTI-correlated with the true latent — worse than predicting nothing.

The mechanism is visible in one more number: **`latent_motion_ratio_mean` = 72.6**, and **495 at @+1**.
The latent is moving ~72x more than the true latent does, while the decoded image moves 0.27x as much as
it should. So the latent wanders enormously and the decoder maps all of that wandering onto nearly the
same picture. That is exactly "a degenerate stable state".

**It does not look broken from the image metrics, which is the trap.** open-loop PSNR 15.92, SSIM 0.544,
one-step 17.64 — all sitting right at the codec floor of ~15.4 dB, because a near-static prediction
scores fine when the codec is the binding constraint. Only `latent_cos` and `latent_motion_ratio` show it.

**No non-finite gradient step preceded it** (`grad/nonfinite_skipped` = 0, nothing in progress.log), so
unlike the robocasa case this was not triggered by a bad step — it is the objective's own minimiser.

**Suspected cause — the loss balance, not the recipe per se.** At epoch 3, `val/loss/dynamics/latent` was
**182.46** of a 188.39 total, against `decode/scene_right` 0.45 and the round-trip anchor 0.446. The
dynamics term is ~400x the anchor. vl64's header states the mechanism precisely: *"The anchor is the ONLY
term forcing the latent to stay a faithful encoding; every other loss can be reduced by making the latent
EASIER TO PREDICT, and the easiest such latent is degenerate."* At this ratio `latent_loss_weight: 10` —
tuned on robocasa at 96px where the latent loss was not 400x the anchor — cannot hold the latent.

**Do NOT simply raise `latent_loss_weight`.** It is bracketed on both sides in vl64: 0.4 erased 5.4 dB in
one epoch, 25 froze motion and drove gradients to inf. It is a RATIO knob. What changed here is the
DENOMINATOR — the latent loss scale on this dataset — so the ratio has to be re-derived, not nudged.

Eval epochs `{5, 9, 15, 19, 29, 39, 49}` → first eval was ~23:20 on 08-31.

### The images the model trains on ARE SQUARE (aspect squashed 1.78x)

Asked by the user. Staging preserves 16:9 (`512x288`), but `load_fpv_frames` turns an INT `img_size`
into a square shape — `dataset.py:234`, `hw = (size, size) if isinstance(size, int)` — and the recipe
sets `img_size: 128`. So:

    source  1920 x 1080  (16:9)  ->  staged  512 x 288  (16:9)  ->  MODEL  128 x 128  (1:1)

The staging decision to preserve aspect was real but the recipe squashes it one layer down, so the net
effect is a **1.78x horizontal compression**. `load_fpv_frames` already accepts an `(H, W)` tuple, so
`img_size: [128, 228]` would preserve aspect — but `img_size` also drives the AE pyramid depth
(`n_levels = int(log2(short_side // bottleneck))`, and up64's header warns that truncation there
silently changes the codec), so a non-square value needs checking against that before use.

Unlikely to be the collapse cause — a consistent anisotropic scale is something conv nets handle — but
it is a real distortion, it was not an explicit decision, and it is worth fixing for any rerun.

**To watch:** `latent_cos` going negative is vl64's documented collapse signature (it collapsed at eval 12
on robocasa after one non-finite step). And **read frames, not just LPIPS** — vl64 is documented to ghost
("perceptually plausible, spatially wrong") and its rollout is *worse* than up64's (OL PSNR 13.42 vs
14.86) even as its codec is better.

## 9. ARM B — `vl64_multicam`, three heads (RUNNING)

`logs/train_world_model_2026_08_31_11_32_38_lego_arm_b_multicam` · GPU 0 · `data.batch=3`
Heads: `scene_right`←`head_right`, `arm_top_left`←`gripper_left_top`, `arm_top_right`←`gripper_right_top`.

**Arm B is NOT "vl64 three times", and the sizing was measured, not assumed.** From Arm A's own autobatch
at one camera: 3.796 GB/sample, batch 19–22, worst eval probe 33.3 GB. Predicting ~10 GB/sample for three
heads was **wrong by 3.5×**:

| batch | reserved |
|---|---|
| 2 | 28.3 GB |
| 3 | 35.5 GB |
| 4 | **98.0 GB** (budget 89.6) |

So batch 3, leaving 54 GB unused. Eval capping (`eval.decode_chunk=16`, `eval.closed_loop_steps=[1]`) did
work: worst probe **33.3 GB → 4.0 GB**.

**`decode_chunk_train` did NOT help — a useful negative result.** `modalities.py:128` advertises it as the
one lever that buys batch size (gradient checkpointing, "0 = OFF (bit-identical)", ~1.33× decode compute,
decoder = "~78% of per-sample training memory"). Setting 24 → 8 produced **byte-identical** autobatch
numbers. The config landed and `TokenGridDecoder` does receive `chunk=_chunk`, so two things follow:
`_chunked_velocity` (`flow.py:102`) short-circuits on `x.shape[0] <= c`; and more importantly, identical
memory means **the decoder is not the bottleneck at these settings** — that 78% was measured at ONE head
with `num_tokens: 32`, whereas here each head has 16 tokens and there are three encoders, a 50-token bag
and three LPIPS-VGG passes instead.

**UNEXPLAINED and worth one probe:** b=2→3 costs +7.2 GB but b=3→4 costs **+62.5 GB**. A ~7 GB/sample
marginal predicts ~43 GB at b=4, not 98. Something changes qualitatively between 3 and 4 — a kernel or
allocation path, not linear scaling.

Recipe deltas from vl64, and only these: `weight` 1.0 → **1/3 per head** (so the aggregate image:proprio
ratio matches vl64 and the run isolates more-views as the single variable — the confound vl64's own header
flags in its own results); `num_tokens` 32 → 16; `decode_chunk_train` 64 → 8; `visual_frames` 128 → 64 per
head (3×64 = 192 still exceeds Arm A's 128 in aggregate). `img_size` STAYS at 128 — 96 would buy real
headroom but make the arms incomparable, defeating the study. `latent_loss_weight` stays **10 per head**,
deliberately not divided by 3: vl64's header records it as a RATIO knob bracketed on both sides (0.4
erased 5.4 dB in one epoch; 25 froze motion and drove gradients to inf), and each head has its own codec
to keep invertible, so the anchor is not a shared budget to split.

**14.3 h/epoch** (15,356 batches at batch 3), so first eval (epoch 5) lands ~01:40 on 09-04 and 50 epochs
would be ~30 days.

**NO VALIDATION DATA YET, and none for ~2 days.** As of 08-31 22:16 Arm B is still inside epoch 0
(`metrics.jsonl` has 0 `val/` entries). With the stock `check_val_every_n_epoch: 4` its first val point
is epoch 3 ≈ **09-02 20:00**, and its first eval is epoch 5 ≈ 09-04. Every read in §8 — the
codec-limited image head, the proprio error — is **ARM A ONLY, one camera, one val point**. Nothing is
yet known about the three-head model beyond that it trains without collapsing.

This is the strongest argument for raising the eval/val cadence: at 14.3 h/epoch, a cadence tuned for
2.6 h epochs leaves Arm B blind for days. See §11.

### ⚠️ THE ARMS ARE NOT BATCH-MATCHED — the open question

Both arms load **46,066 train / 5,565 val windows** — identical episodes, seed-0 split and subsample, so
the DATA is matched. Only the batch differs, and by 7.3×:

| | batch | batches/epoch | h/epoch | first eval |
|---|---|---|---|---|
| Arm A | **22** | 2,094 | 2.63 | ~23:20 08-31 |
| Arm B | **3** | 15,356 | 14.3 | ~01:40 09-04 |

That is a second variable moving alongside "more views". `accumulate_grad_batches` cannot rescue it —
`conf/trainer/default.yaml` LOCKS it at 1 with "NEVER use gradient accumulation".

Three options, in the order worth trying:

1. **Explain the b=4 spike first.** If it is a kernel/allocation artifact rather than real scaling, Arm B
   could reach batch 8–22 and the confound dissolves for free. Cheapest, highest value.
2. **Match down** — rerun Arm A at `data.batch=3 data.autobatch=false`. Clean comparison, ~7× slower per
   epoch, putting Arm A on Arm B's ~30-day schedule.
3. **Accept and document** the batch difference as a caveat. Weakest.

Left running as-is: both arms produce useful single-arm results either way, and this is a research call.

## 12. THE IMAGE CODEC IS COLLAPSED TO A CONSTANT IMAGE. That is the whole failure.

Not "underperforming" — **outputting one fixed frame forever**. Verified three independent ways.

**(a) Direct, from the run's own saved eval frames** (`armA_w10` ep1,
`logs/epoch_0001/eval_ae_floor/scene_right/raw_filmstrip_frames_0.npz`, 460 timesteps):

| | across-frame std | mean \|frame_t − frame_0\| |
|---|---|---|
| ground truth | 0.068056 | 0.051555 |
| **prediction** | **0.000335** | **0.000242** |

203x and 213x less. The model emits the same image at every timestep, for two different episodes.

**(b) A constant image beats the trained autoencoder.** Measured on 200 val frames: the train-mean
frame scores **17.29 dB**, the val-mean frame **18.16 dB**. The trained `ae_floor` is **15.44 dB**.
Doing nothing is 1.9 dB BETTER than what training produced.

**(c) Every mystery number falls out of it.** The 15.3–15.5 dB "ceiling" static across three runs, two
anchor weights and two aspect ratios; `motion_ratio` 0.023 (a constant has no motion); prediction PSNR
16.31 "above" the 15.44 codec ceiling (both are the same constant, scored on different frame sets).

### Where it collapses, and why the anchor cannot fix it

Layer-by-layer on the trained encoder — the conv trunk still sees the scene; the **Perceiver readout**
(`vision.py:378`) throws it away:

| stage | across-frame std / overall |
|---|---|
| conv trunk through `to_d+pos` | 0.383 / 4.08 — **9.4%**, frame info intact |
| `to_latent` cross-attention | 0.00097 / 4.58 — **0.021%**, a 440x suppression |

The readout is `latent_q + proj(attn(...))` where `latent_q` is input-INDEPENDENT (`vision.py:367`).
Training drove the input-dependent branch to ~0 and inflated the constant branch ~14x.

**It is self-sealing.** The decoder's Jacobian w.r.t. the latent is dead: feeding it a fully RANDOM
latent changes its output by 3.9%, and its latent-sensitivity is ~70x BELOW the same architecture at
random init. Measured `grad/norm/encode_scene_right` = 0.005 against `decode_scene_right` = 53.3 — a
**10,000:1** split, versus ~1:1.6 at init. So the roundtrip anchor's gradient reaches the encoder
through a Jacobian that no longer exists. **`latent_loss_weight` cannot fix this at ANY value** — which
retrospectively explains why 10 and 100 gave identical floors, and why my whole anchor-weight line of
inquiry was chasing a coefficient on a dead path.

### Why this dataset and not robocasa

The constant-image basin is far more attractive here:
- ONE fixed camera, one table, one wall across all 65 sessions, so the constant solution already scores
  17.3 dB — **above robocasa's entire from-scratch AE wall of 14.3 dB** (§1 of that record).
- At stride 6 only **~4.6%** of pixels change more than the TAESD floor RMSE, and the **top 1% of pixels
  carry 53.6%** of the frame-delta energy. An L1/LPIPS objective is nearly indifferent to the arms.
- Total headroom from constant image to TAESD-quality is only **2.45x** in the loss mix — all of it
  gated behind the hard part, while the decoder-bias path captures the first ~40% for free.

`design/collapse.md` predicts exactly this ("the realized failure sheds exactly the information that is
hard to predict") — here it sheds ALL of it, because the hard part is only ~5% of pixels.

### The normalization hypothesis: REFUTED for pixels/LPIPS, partially confirmed for the latent LN

Traced end-to-end and consistent: frames are [0,1] (`dataset.py:363`), the AE contract is [0,1]
(`vision.py:13,114,373`), `VisualLoss` expects [0,1] (`visual_loss.py:51`) and its torchmetrics LPIPS
uses `normalize=True`, which rescales internally — `visual_loss.py:88-93` deliberately does not
pre-scale. Pixel stats healthy (mean 0.525, std 0.262, p1/p99 0/0.95). Identical to what worked on
robocasa.

The one real normalization finding: the non-invertible latent LayerNorm (`multimodal.py:31`) costs
**8.9 dB at fit-start on this data versus 3.51 dB on robocasa/TAESD** (23.83 → 14.95 dB, replicated with
frozen TAESD on lego frames). But restoring dataset-AVERAGE per-position stats — exactly what a decoder
bias can learn — recovers 22.54 dB, so only ~1.3 dB is truly destroyed. It deepens the early hole the
decoder escapes via its constant path, worsening the race, but it is not the wall.

### `ae_bottleneck: 16` is the WRONG first lever

This architecture reached 18.7–20.4 dB at 128px/bottleneck-8 on robocasa. We are 3+ dB below that, so
capacity is not binding yet. The 7.0 px/float at 128x224 (vs 2.25 at robocasa's 96px) only starts to
bind AFTER the collapse is fixed. I had this queued as the next experiment; it would have measured
nothing.

### Gradient clipping is an enabler, not the cause

`gradient_clip_val=1.0` is hardcoded (`train_world_model.py:317`). Measured `grad/norm_preclip` maxima:
372.8 → 293.4 (probes), 58.1 (`armA_w10`), **896** (`armA_w100` — it scales with the anchor weight, so
the anchor owns the norm). At init the anchor's gradient is enc 380 / dec 618 at w=10, so **the clip
cuts 50–900x from step 0** and the surviving unit-norm direction is decoder-dominated. The encoder must
learn a 28,672-px → 4,096-float code at ~1/300 of nominal LR while the decoder need only learn a bias.
Stated honestly: init gradients are large at 96px too (260) and with pure MSE (478), and robocasa's
healthy 0.17–0.27 is from a CONVERGED run — so clipping alone does not separate sim from real.

### THE TRIPWIRE, for every future run

`eval_ae_floor/<head>/motion_ratio_mean` ≈ 0.02 is the one-number signature of this state, and it is far
more specific than PSNR — PSNR looked like a plausible 15.4 dB "codec floor" for three runs while the
codec was emitting a still image.

## 13. flow_hidden=128 does NOT transfer. CAPACITY does — `num_tokens=64` is the first thing to move the floor

Screen of vl128_scene on lego (4 epochs x 600 batches, evals at 1 and 3, autobatch on).

| config | ep | motion | **ae_psnr** | val_psnr | proprio mm |
|---|---|---|---|---|---|
| base (l1=3, tok32, bott8) | 1 / 3 | 0.0386 / 0.0320 | 15.22 / 15.35 | 15.38 / 15.71 | 124 / 139 |
| `visual_l1=1.0` (= st_fh128) | 1 / 3 | 0.0337 / 0.0311 | 15.38 / 15.26 | 15.71 / 15.86 | 120 / 117 |
| **`num_tokens=64`** | 1 | 0.0396 | **17.58** | **16.88** | 138 |
| *flow_hidden=512 reference* | | *0.0231* | *15.44* | | |

**FINDING 1: the robocasa fix does not transfer.** `flow_hidden=128` broke every record on sim; on real
data it leaves the floor at 15.2-15.4 dB, indistinguishable from `flow_hidden=512`'s 15.44. Both loss
balances (l1=1 and l1=3) sit on the same wall, and motion_ratio DECREASES across epochs (0.0386 → 0.0320)
— drifting toward collapse, not away.

**FINDING 2: capacity IS binding, and I said it was not.** `num_tokens` 32 → 64 moved the AE floor
**+2.2 dB** — the first movement in that number across ~10 runs. Both the subagent's analysis and my own
summary said capacity "only binds after the collapse is fixed" and that `ae_bottleneck`/`num_tokens` were
the wrong first levers. That was wrong, and the arithmetic was available the whole time: lego is
128x224 = 28,672 px through the same 32x128 token bag as robocasa's 96px = 9,216 px, i.e. **7.0 px/float
against 2.25 — 3.1x harder**. The subagent computed that number and we both filed it as "not the binding
constraint yet".

Still open at this point: motion_ratio is 0.0396, better than the 0.0231 collapsed reference but far from
healthy, so `tok64` has raised the ceiling without yet proving the codec tracks motion. `bott16` has no
eval point. `tok64+bott16` and `l1=5` are running.

**Process lesson worth more than any of the above:** six earlier screen runs died at the epoch-0 eval
because `data.autobatch=false data.batch=26` was copied from vl128's header, where 26 was calibrated at
96px square. At 128x224 the train step fit at 93.87 GiB and the eval's LPIPS pass OOMed. Pinning the
batch also disabled the one mechanism that probes eval memory. **Do not transplant a batch size across
resolutions; let autobatch size it, and cap the eval with `eval.decode_chunk` when it is tight.**

## 14. THE FIX: `num_tokens=64` **WITH** `ae_bottleneck=16` — the pair, not either knob (09-02)

The codec finally reconstructs motion. One config diff over `vl128_scene`, nothing else changed:

    model.modalities.1.num_tokens=64          # was 32
    +model.modalities.1.ae_bottleneck=16      # was 8 (the default)

Run: `logs/train_world_model_2026_09_02_06_13_10_r4_tok64_bott16` (4 epochs x 600 batches, ~55 min/epoch).

| config (ep3) | motion_ratio | ae_psnr |
|---|---|---|
| `tok32` `bott8` (base) | 0.0320 | 15.35 |
| `tok64` `bott8` | 0.0396 | 17.58 |
| `tok128` `bott8` | 0.0267 | 15.51 ← **worse** |
| aeonly (`lambda_flow=0`) `tok64` | 0.0627 | 17.85 |
| aeonly `tok128` `bott16` | 0.1077 | 17.52 |
| **`tok64` `bott16`** | **0.3218** | **19.93** |
| *collapsed reference* | *0.0231* | *15.44* |

**Verified independently from the saved frames**, not the logged metric — across-frame std of `pred`
against `gt` in `raw_filmstrip_frames_0.npz`:

| run | pred std | gt std | motion retained |
|---|---|---|---|
| `armA_w10` (collapsed) | 0.000335 | 0.068056 | **0.5%** |
| `r4` epoch 1 | 0.005556 | 0.068056 | **8.2%** |
| `r4` epoch 3 | 0.040987 | 0.068056 | **60.2%** |

19.93 dB also puts the codec INSIDE the 18.7-20.4 dB band this architecture reached on robocasa — it is
finally performing to its own precedent instead of sitting 3 dB below it.

### Why it is the PAIR, and the mechanism `vision.py` already stated

`ae_bottleneck` is the conv pyramid's target spatial size. At 8 it pools `128 -> 8`, a **256x spatial
reduction**, so the arms and blocks are gone BEFORE the token queries see the feature map. At 16 it is
`128 -> 16`, a **64x reduction** — 4x more detail surviving to the readout. `num_tokens` then decides how
much of that reaches the latent. Neither alone works, and `tok128` alone went BACKWARDS.

`vision.py` says it in one clause that was read past twice: *"Raising this to 16 makes the bottleneck
16x16 (a 64x spatial reduction at 128px instead of 256x) and **gives the token budget something to
carry**."*

The pairing is EXACT rather than lucky: `n_levels = log2(128//16) = 3` and `224/8 = 28`, so the
bottleneck is 16x28 with no silent truncation. A non-power-of-2 ratio would have landed elsewhere
with no warning (up64's header documents that trap).

### Three of my calls this record should not repeat

1. **"`ae_bottleneck` is the wrong first lever."** Written in §12 on a subagent's read that capacity only
   binds after the collapse is fixed. It WAS the fix.
2. **`lambda_flow=0` was my bet and it lost.** Training the codec with NO dynamics pressure reached only
   motion 0.0627 — so joint training was never the blocker and the 400:1 dynamics:anchor ratio was a
   symptom, not a cause. The winner has dynamics fully on at `lambda_flow=1.0`.
3. **"Capacity knobs cannot help while the codec emits a constant image."** Said an hour before the
   capacity pair fixed it.

The arithmetic that pointed here was available from the start and both a subagent and I filed it as "not
binding yet": lego is 128x224 = 28,672 px through the same token bag robocasa filled with 96px = 9,216 px,
i.e. **7.0 px/float against 2.25**.

### Consequence for scheduling

Autobatch drops to **batch 12** (from 18) — 4x the bottleneck area costs real memory. At 46,066 train
windows that is **3,839 batches/epoch**, and the screen measured ~5.6 s/batch at this config, so a FULL
epoch is roughly **6 hours**. The 50-epoch schedule is not reachable; useful eval points are, at epochs
1 and 3.

## 11. Eval cadence — raise it, and Arm B is why

Asked by the user (2026-08-31): *can we eval more often, epoch 1 3 5?*

`at_epochs` is NOT the mechanism — `conf/eval/default.yaml` warns "DO NOT add per-epoch early points
like [1,3] (that made eval effectively 'every 2' and clogged training)". The cadence knob already does
it exactly, because it fires on `(epoch+1) % every == 0`:

    eval.during_train.every_epochs=2  trainer.check_val_every_n_epoch=2      -> epochs 1, 3, 5, 7, ...

Matching `check_val_every_n_epoch` keeps eval on the same phase as validation and the checkpoint, which
the config says is load-bearing. NOTE `check_val_every_n_epoch: 4` carries "DECIDED (user, 2026-07-20):
DO NOT CHANGE without asking" — this IS that ask, recorded.

Two reasons the original warning may not apply here: `distributed.md` §2 measured eval at **1.0% of
wall time**, and on `recorded` the expensive routines (MPPI control) self-skip because there is no
simulator. Validation alone measured 11 min against Arm A's 158 min epoch (7%).

**The case is much stronger for Arm B than Arm A.** At 14.3 h/epoch the stock cadence gives it a first
val at ~2 days and a first eval at ~3.5 days; at `every_epochs=2` those become ~1 day and ~1.2 days.
Arm A, at 2.6 h/epoch, already gets a first eval the same night.

TODO: measure the actual eval cost at Arm A's epoch-5 eval (~23:20 on 08-31) before committing the
change, so the "1.0%" figure is confirmed on THIS dataset rather than inherited from robocasa.

## 10. Open items

- The b=4 memory spike (§9) — the highest-value unknown.
- `_oneoff_action_sensitivity` on Arm A — a near-zero result is quickdraw#15, not a model failure.
  Not yet run; both cards busy.
- Re-derive the subsample ratio against vl64's TRAINED codec floor rather than frozen TAESD (§3).
- **Cosmetic:** Arm A's startup prints a torus split inventory (`train 256 traj × 256 steps`) because
  `conf/data/lego.yaml` inherits `torus.yaml`. The real numbers are on the next line
  (`46066 train / 5565 val windows`). Harmless but misleading to a reader.
- `EULER_SEQ = "xyz"` is unverified and cannot be verified from this dataset (§1b).
