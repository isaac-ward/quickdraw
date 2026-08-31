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

Eval epochs `{5, 9, 15, 19, 29, 39, 49}` → first eval ~23:20 on 08-31, second ~09:50 on 09-01.

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
