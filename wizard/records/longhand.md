# longhand — Swoosh right-arm world-model dataset

Branch `longhand`, forked from `main` @ `934ea19`. Lego is finished and pushed to the `lego`
branch; nothing from it is merged here and nothing here depends on it.

Goal: turn the Swoosh right-arm collection (Google Drive, campaigns 1–9) into a
quickdraw-compatible HuggingFace dataset via a new `longhand` processor.

---

## 1. What is actually in the corpus

Pulled 2026-09-10 with `utils/gdrive_pull.sh` → `scratch/longhand`. **21 GB, 1228 files.**
`usable` = the overlap window where every stream (both arm streams and all four cameras) is
live; that is what the processor can actually emit, and it is 1–2 s shorter than run.json's
wall-clock `duration_s` because the cameras open staggered.

| campaign | runs | 32/32 green | usable | shortest | longest | disposition |
|---|---|---|---|---|---|---|
| `campaign1-tests` | 0 | – | – | – | – | **excluded** (empty; only `campaign.json`) |
| `campaign2-play` | 3 | 2 | 3.1 min | 46 s | 84 s | **excluded** (operator) |
| `campaign3-play` | 15 | 15 | 30.3 min | 37 s | 175 s | train/val |
| `campaign4-rgb` | 10 | 10 | 6.3 min | 25 s | 73 s | train/val |
| `campaign5-play-long` | 11 | 11 | 57.2 min | 21 s | **606 s** | train/val |
| `campaign6-combos` | 3 | 3 | 12.1 min | 205 s | 303 s | train/val |
| `campaign7-precision` | 6 | 6 | 9.5 min | 53 s | 149 s | train/val |
| `campaign8-purple-play` | 5 | 5 | 5.0 min | 58 s | 64 s | **eval only** |
| `campaign9-purple-stack` | 6 | 6 | 3.0 min | 28 s | 32 s | **eval only** |
| `_rt`, `_rt2`, `_rt3`, `_rt4` | 4 | 0 | 0.3 min | – | – | **excluded** (junk, see below) |
| `shakedown` | 4 | 1 | 1.7 min | 14 s | 45 s | **excluded** (bring-up) |
| `audit` | 2 | 0 | 0.5 min | 11 s | 19 s | **excluded** (bring-up) |

**Train/val pool: 45 runs, 115.3 min. Eval: 11 runs, 8.0 min. Total usable 2.15 h.**

**Every single run in campaigns 3–9 passes all 32 of the collection-side validation checks.**
Every non-green run in the corpus is one we exclude anyway. That is the cleanest raw input
this project has had, and it is worth saying plainly.

`_rt*` are **my own test artefacts** — `tests/test_roundtrip.py` in the collection repo writes
scratch runs into `campaigns/`, and the Drive sync swept them up. That is a bug in the test (it
should use a temp dir outside `campaigns/`); fix it there so the next pull is clean.

## 2. Getting the data down

`utils/gdrive_pull.sh` (rclone, `--drive-scope=drive.readonly`). One browser OAuth round trip is
needed once; the token lives in `~/.config/rclone/rclone.conf`, outside the repo, and is
read-only so it cannot modify the Drive. Sequence is documented at the top of that script.

**The Drive MCP connector cannot substitute.** `download_file_content` returns base64 *into the
model's context*; one camera mp4 is 40–53 MB (~70 MB encoded) against a 21 GB corpus. Fine for
`run.json` and directory listings, which is how the audit above was scoped before the pull.

## 3. What the processor emits

`src/quickdraw/data/swoosh.py` reads one run; `longhand()` in `processors.py` orchestrates.

**Grid: 30 Hz** (= camera fps = the collection side's `export_rate_hz`), spanning the window
where every stream is live. **Nearest-in-time, never interpolated** — every source runs at
1.7–3.3× the grid, so interpolation would invent precision, and `joints_real_deg`, rotation
signs and the gripper's 20 Hz poll all interpolate badly.

**No time shifting is applied, deliberately.** All streams share one monotonic origin
(`t_loop0`) and camera stamps come from the V4L2 kernel buffer at capture, so the ~21 ms
userspace read lag is already removed upstream.

### `action` (5) — the Xbox controller, per spec

`move_x, move_y, height, yaw, gripper`, post-deadzone/expo, from `raw/controller.jsonl`.

Not the commanded pose and not the SDK arguments: both are consequences of the action plus the
integrator's internal state, so conditioning on them hands the model the answer. Both remain in
the raw run directories and are recoverable at any time.

### `state` (17) — what the arm reported

| idx | field | source |
|---|---|---|
| 0:3 | `ee_{x,y,z}_mm` | `pose_world_xyz_mm`, world frame, mm |
| 3:9 | `ee_rot6_*` | `pose_base_mm_deg[3:6]` → rotation matrix → world frame → first two columns |
| 9 | `gripper` | `gripper_pos` normalised, **1 = open** |
| 10:17 | `joint{1..7}_rad` | `joints_real_deg`, radians |

Three choices that are not obvious, all learned on lego:

- **`joints_real_deg`, not `joints_deg`** — the latter is the controller's *planned* angle and
  was measured diverging from the real one by up to 5.02°.
- **6D rotation, not euler** — euler wrapped 702 times in a single lego arm stream. Verified
  orthonormal on real data: column norms exactly 1.0, column dot 7e-08.
- **`target_yaw_world_deg` is NOT absolute** — it resets on re-home, clear-errors and servo
  recovery. Absolute orientation only ever comes from `pose_base_mm_deg[3:6]`.

Gripper **polarity is inverted between the two streams**: action `gripper` 1 = squeeze, while
`gripper_pos` 850 = open. The state is normalised so 1 = open (monotonic in aperture) and the
action is left alone — they are different quantities and collapsing them would be a lie.

### Cameras

All four (`scene_left`, `scene_right`, `gripper_right_bottom`, `gripper_right_top`) at
**144×192** — 4:3 like the 640×480 source, both sides divisible by 16. `gripper_right_bottom`
is flipped in the recorded mp4 already; do not flip again.

## 4. The split

**Confirmed by the operator: 90/10, longest trajectories in val, and val being dominated by
`campaign5-play-long` is accepted.**

`build_recorded_dataset` could not express this — it split train/val itself at `VAL_FRAC=0.1`,
random by episode, seed 0. It now takes an optional `val_ids` from the processor, and records
`split_rule` in `dataset_card.json` so a longest-first split is distinguishable from a random
one after the fact. Five-element processor returns keep the old random behaviour untouched.

`_longest_first_val(lengths, frac, min_eps=2)`:

- **Longest-first**, because open-loop rollout evaluation can only run as far as the *shortest*
  val episode. On lego a random seed-0 draw pulled a short episode into val and capped every
  long-horizon number at a fifth of the horizon the data supported. This is the whole point.
- **Closest prefix, not accumulate-until-crossed.** Always-overshoot turned the 10% request into
  16.8% here, because the longest episodes are far longer than the rest. Landing either side of
  the target is the honest reading of "about 10%".
- **Floor of two episodes.** A one-episode val has no across-session variance at all: one
  recording's lighting, layout and operator mood become the entire validation signal.

Predicted on the real pool (45 episodes, 207,589 frames):

| | episodes | frames | share |
|---|---|---|---|
| train | 43 | 172,792 | 83.2% |
| val | 2 | 34,797 | 16.8% |

Both val episodes are `campaign5-play-long` (606 s and 554 s). **The shortest val episode is
16,605 frames ≈ 554 s**, so open-loop rollouts are evaluable out to nine minutes — against
lego, where this was the binding constraint, that is the headline number.

**Read val loss accordingly:** it measures long-horizon fidelity on long play trajectories, not
a representative i.i.d. sample of training. The per-episode `task` label carries the campaign
into `<split>/meta/tasks.parquet` for anyone who wants to slice it.

Campaigns 8 and 9 become their own splits, `eval_purple_play` and `eval_purple_stack`, written
verbatim with no random splitting — the same `eval_<suffix>` convention `starling_bags` uses.

## 5. Sync quality

`read_run` reports per-stream nearest-neighbour displacement as `max`, `p99` and `over` — the
fraction of steps displaced past *that stream's own half-period*, which is the best any
nearest-neighbour resample can do. Reporting the max alone was misleading: one dropped sample in
a two-minute run puts it at 3× typical and makes a clean run look broken.

On `campaign3-play/recording_2026_09_10_05_31_26` (125 s, 32/32 green):

| stream | max | p99 | own bound | past bound |
|---|---|---|---|---|
| controller | 22.3 ms | 7.0 ms | 5.0 ms | 3.3% |
| arm_state | 33.7 ms | 9.4 ms | 10.0 ms | 0.4% |
| scene_left | 17.7 ms | 17.6 ms | 16.0 ms | 25.3% |
| scene_right | 7.8 ms | 7.8 ms | 16.0 ms | 0.0% |
| gripper_right_bottom | 18.0 ms | 18.0 ms | 16.0 ms | 20.6% |
| gripper_right_top | 4.3 ms | 4.2 ms | 16.0 ms | 0.0% |

The camera rows are **phase, not fault**: a camera whose true rate differs from the 30 Hz grid
by a fraction of a percent drifts slowly through the grid's phase, so its displacement sweeps
the full ±half-period and sits just past the bound for part of the run. `max ≈ half the camera
period` is the arithmetic floor. The controller and arm maxima are three isolated stream
hiccups (2 and 1 respectively, `dt > 3× median`), not a systematic rate problem.

What would be alarming and is not present: `over` near 100%, or `max` several times the
stream's own period.

## 6. The built dataset

`logs/recording_2026_09_10_10_04_45_longhand` — 4.4 GB of splits + 2.8 GB of preview media.
Build took 44 min (56 runs read on 6 workers, 224 clips encoded, 4 lerobot splits written).

| split | episodes | frames | hours | windows (P=8,F=64) | shortest ep |
|---|---|---|---|---|---|
| train | 43 | 172,835 | 1.600 | 169,782 | 621 |
| val | 2 | 34,799 | 0.322 | 34,657 | **16,606** |
| eval_purple_play | 5 | 9,048 | 0.084 | 8,693 | – |
| eval_purple_stack | 6 | 5,323 | 0.049 | 4,897 | – |

Train's LONGEST episode is 15,996 steps and val's SHORTEST is 16,606 — the longest-first split
is strictly ordered, which is the invariant that matters.

`check_dataset` passes: obs_dim 17, action_dim 5, all splits fit `P+F`.

### Verified against the source, not just self-consistent

`tests/test_longhand_roundtrip.py`, run on val episode 0 (18,193 steps):

- **states and actions are bit-exact** — `max |diff| == 0.0` between the parquet and a fresh
  `read_run` of the source directory. No silent recast, reorder or truncation.
- **frames align with rows.** Mean |Δpixel| against the source at integer shifts −3…+3:

  | camera | −3 | −2 | −1 | **0** | +1 | +2 | +3 |
  |---|---|---|---|---|---|---|---|
  | scene_left | 6.81 | 5.69 | 4.31 | **3.02** | 4.24 | 5.55 | 6.61 |
  | gripper_right_top | 5.46 | 4.59 | 3.76 | **3.28** | 3.84 | 4.66 | 5.51 |

  The residual 3/255 at shift 0 is AV1 re-encode loss. **The shape is the actual result**: a
  strict minimum at 0 rising cleanly either side. A small number alone would prove nothing —
  a dataset misaligned by one row also scores "small" on a slow scene, which is exactly how
  lego's misalignment survived into training. The test asserts both the argmin AND that the
  curve is steep enough to discriminate, so it cannot pass vacuously.

**Note for training on this box:** lerobot's own loader needs `torchcodec`, which cannot load
here (`libavutil.so.60` missing). quickdraw's `data/dataset.py` reads the mp4s with imageio
instead, so training is unaffected — but anything reaching for `LeRobotDataset` directly will
fail until ffmpeg's shared libs are installed.

## 8. Training runs — the stride ablation

`vl128_blockstack_2cam` throughout: robocasa's record-holding `vl128_2cam` (record §24) with only
what the dataset forces changed — proprio 17 / action 5, `img_size [96,128]` non-square at 3:4 to
match the 144×192 source, `ae_bottleneck 6`. Cameras `scene_right` + `gripper_right_top`,
`action_aggregate=mean`, evals every 2 epochs. **The one variable is the temporal stride.**

### 8.1 First bracket: strides 10 and 15 (2026-09-11 03:19 → 17:40, killed)

| | `l1_mean` @ep9 | ae_floor RMSE | motion_ratio | median of halves |
|---|---|---|---|---|
| **stride10** | **0.0327** | 0.0347 | 0.858 | 0.0349 → 0.0327 (**−6.3%**) |
| stride15 | 0.0361 | 0.0373 | 0.823 | 0.0348 → 0.0361 (**+3.6%**) |

**stride10 won by 9.4% at matched epoch 9.** Judged against this data's own noise, not robocasa's
±0.02 rule (calibrated on values 4× larger): settled epoch-to-epoch jitter is **σ = 0.00054**, so
the gap is **6.3σ**. stride10 also won at matched real time at 275 s, 411 s and 550 s.

stride15 peaked around ep5–7 and regressed for ~12 epochs after — the GameNGen pattern §13 cites
(below ~10⁷ examples, quality peaks early then degrades), with fewer windows and a harder problem.

**`motion_ratio` 0.82–0.86 in both.** Robocasa's degenerate failure was 0.13, so neither run hedged
to zero. The bracket's actual purpose — stay above the floor — succeeded.

### 8.2 The thing the first bracket could not know at launch

**The measured codec floor is ~0.0350 RMSE** (`eval_ae_floor/cam_scene/mse_mean`, √, ep9). The
bracket was chosen to span 0.045–0.0637. So it was not conservative, it was *low*:

| | floor | frame-delta | achieved SNR |
|---|---|---|---|
| stride10 | 0.0347 | 0.0659 | **1.90×** |
| stride15 | 0.0352 | 0.0762 | **2.16×** |
| | | | *1.24–1.36× = robocasa's productive zone* |

Both landed **above** the zone, not below it. Erring high was right in principle — below the floor
is the degenerate failure — and overshot in fact. And the ordering confirms the direction: the
**lower**-SNR arm won, so the optimum is further down.

At the measured floor: stride 5 → 1.36× (the robocasa-equivalent point), stride 6 → 1.49×,
stride 3 → 1.02×, stride 1 → 0.44× (below the floor).

### 8.3 Second bracket: strides 1 and 5 (launched 2026-09-11 17:50)

| arm | GPU | stride | rate | F=64 spans | SNR | batch |
|---|---|---|---|---|---|---|
| `bs_stride1` | 0 | 1 | 30 Hz | 2.4 s | **0.44×** | 11 |
| `bs_stride5` | 1 | 5 | 6 Hz | 12.0 s | **1.36×** | 13 |

stride 5 is the predicted optimum. **stride 1 is a deliberate below-floor anchor** — record §13
predicts the degenerate zero-motion solution there, and `motion_ratio` collapsing toward 0.13 would
confirm the mechanism holds on this dataset rather than only on robocasa. Autobatch chose 11 rather
than 13 for stride 1; note it when reading the comparison.

With §8.1's retained logs this gives a **1 / 5 / 10 / 15** ablation. Caveat: 10 and 15 were killed
at ep9/ep17, so comparisons beyond ep9 are not matched across all four.

### 8.4 Then

1. **Action aggregation** — `./wizard/scripts/blockstack-aggregate.sh <winning stride>`, by hand.
   See §8.5. Do not run it before the stride is settled: the within-window variance separating
   `mean` from `concat` is itself stride-dependent (14.7% of `move_x` at stride 10, 22.7% at 15).
2. **Four cameras** (robocasa queue item 2 — does the camera lever scale, or was the wrist view
   special for being gripper-mounted?). Block-stack can uniquely answer it: two scene, two gripper.
3. `decode_out_act=sigmoid` **already ruled out**: 0.00% of scene and 0.07% of gripper pixels
   saturated, against torus's 66.9% (worth 3.4×) and robocasa's marginal 0.4%.

### 8.5 The action-aggregation sweep (pinned, not queued)

| arm | `action_aggregate` | model sees |
|---|---|---|
| `bs_agg_sub<N>_sum` | `sum` | action_dim 5 |
| `bs_agg_sub<N>_concat` | `concat` | action_dim 5 × N |

**The default aggregation is wrong for this data.** block-stack actions are Xbox stick POSITIONS —
starling's `joy_axis_*` class — not the EEF deltas summing was written for. Lag-1 autocorrelation is
0.96–0.99 on every axis, so summing scales rather than cancels, and `normalization_stats.json` is
computed on raw stride-1 actions and never sees the aggregation:

| stride | rule | action z-std | max \|z\| |
|---|---|---|---|
| 10 | `sum` (default) | 7.6 – 9.6 | 30.2 |
| 15 | `sum` (default) | 11.0 – 14.2 | 45.3 |
| 10 | `mean` | 0.76 – 0.96 | 3.0 |

Under `sum` the two bracket arms would have differed in action input scale by ~1.5×, confounding the
stride comparison itself. Caught only by pulling `ba8ff09` from main.

`sum` is exactly N × `mean` — identical information, so that arm isolates whether input SCALE alone
hurts. No clamp is in the way (`action_fourier_freqs=0`, `action_squash=none`), so a linear `act_enc`
could absorb the factor; expect a modest effect. `mean` vs `concat` is the information-bearing
comparison. Robocasa §12 found the model uses action DISTRIBUTION not ORDER, which is exactly
concat's advantage — concat winning would mean §12 does not hold here. A tie is a real result.

### 8.6 Things that cost time, worth not repeating

- `check_val_every_n_epoch` is the VAL cadence ONLY. The eval SUITE is
  `eval.during_train.every_epochs` (default 10 + `at_epochs: [5,15]`). Setting only the first left
  evals at {5,9,15,19,...} — nothing for the first five epochs.
- **There is no `@+128`.** Horizons are derived from episode length: `@+1,8,16,32,64,412,824,1236,
  1651` at stride 10. Robocasa's headline number does not exist here; use `l1_mean`, or convert
  steps to seconds (`steps × stride / 30`) before comparing across strides.
- `metrics.jsonl` is LONG format (`step`/`tag`/`value`), not one object per epoch.
- `WANDB_API_KEY` lives in `~/.env`, outside the repo. A script that does not source it logs locally
  only and says so in one line nobody notices for hours.
- `pkill -f <pattern>` matches the calling shell's OWN command line whenever the pattern's literal
  text appears anywhere in that command. Killed the shell mid-script three times, each time leaving
  every later step silently unrun.
- A launch script with no argument guard, run bare to test it, **launched a duplicate run onto a
  busy GPU**. Both bracket scripts now refuse without arguments and refuse while training is live.
- An apostrophe in any `run_summary` value breaks the hydra-level quoting
  (`"+run_summary.x='$VAL'"`) and fails with a bare grammar error naming one character. Guarded.
- Editing a running bash script corrupts it — bash reads by byte offset as it executes.

### 8.7 The derivative-loss arm (launched 2026-09-12 02:35)

`bs_deriv_w10` on GPU 1. Term at weight 10 on all three heads, proprio moved `flow` -> `mse` to be
eligible. Baseline is the retained `bs_stride10`.

**The pre-flight the design doc asks for: PASSED.** `||dpred||/||dtrue||` vs `cos(dpred, dtrue)`,
computed post-hoc from the retained runs' `raw_filmstrip` npz, no GPU:

| run | ep1 -> last | ratio | cos |
|---|---|---|---|
| stride10 | 1 -> 9 | 0.253 -> **0.872** | +0.066 -> **+0.053** |
| stride15 | 1 -> 17 | 0.197 -> **0.921** | +0.103 -> **+0.074** |

Motion of roughly the right MAGNITUDE in essentially RANDOM DIRECTIONS = the doc's "incoherent, this
IS flicker, the target case". The ep1 reading looks like the wrong row purely from undertraining, so
do not diagnose off an early eval.

**The weight is measured, and the doc's figure does not transfer.** Its 25%-of-decode guidance came
from RANDOM 112x192 data (ratio 0.62). On real block-stack frames, three offsets, untrained:

| head | decode | derivative | ratio | weight for 25% |
|---|---|---|---|---|
| proprio | 0.62–0.85 | 0.008–0.014 | 0.013–0.017 | 14.7–19.9 |
| cam_scene | 2.31–2.37 | 0.048–0.072 | 0.021–0.031 | 8.0–12.1 |
| cam_wrist | 2.79–2.81 | 0.084–0.124 | 0.030–0.044 | 5.6–8.3 |

20–50x smaller than the doc's, because real consecutive frames barely differ — the low-motion
problem itself. **Anyone copying 0.25 would run the term at ~2% of intended strength.** A flat 10
(user's call, for simplicity) puts proprio at 13–17% of decode, cam_scene 21–31%, cam_wrist 30–44%.

**KNOWN CONFOUND, accepted (user, 2026-09-12).** The arm differs from `bs_stride10` in TWO ways:
`derivative_weight` 0 -> 10 AND proprio `decode_kind` flow -> mse. A win cannot be attributed to the
term alone. Mitigating: both runs have IDENTICAL image heads (`mse`, same weights, same cameras), so
proprio's objective reaches `cam_scene`/`cam_wrist` only through the shared trunk — the image
comparison is far less contaminated than a proprio one, and `decode/proprio` is not comparable
across the two at all. A matched `weight=0, proprio=mse` baseline was offered and declined; GPU 0
stays on `bs_stride1`.

**Reading it.** DECIDER: `eval_ood_horizon/open_loop/cam_scene/lpips` at `@+824` (275 s) / `@+1236`
(412 s) plus `lpips_mean`, matched epochs vs `bs_stride10`. TRIPWIRE: `motion_ratio_mean` both heads
vs baseline 0.858 (scene) / 0.810 (wrist) — the term is MEAN-SEEKING and the mean of "the block might
go left or right" is NO MOTION, so an over-weighted term FREEZES the prediction; `cam_wrist` runs
hottest so it shows there first, fallback 7. NEVER rank on `derivative/<head>`.

**Per-head eval metrics ARE valid** — eval passes `image_head_cams(cfg)` to the loader, so each head
is scored against its own camera. The `vl128_2cam` header's "read only the first head" warning is
stale. Confirmed on `bs_stride10` ep9: `cam_wrist` OL l1 0.1041 vs `cam_scene` 0.0327, the same
reconstructs-best / predicts-worst split robocasa §24.2 found.

### 8.8 What the derivative term actually is

`p` = predicted frames, `g` = ground truth, both (B,F,H,W,C) in [0,1]; `k` = 1 (locked).

- vector head: `MSE(p[:,k:] - p[:,:-k], g[:,k:] - g[:,:-k])`
- image head: `w_l1 * L1(dp, dg) + w_lpips * SUM_l ||(phi_l(p_t+k) - phi_l(p_t)) - (phi_l(g_t+k) - phi_l(g_t))||^2`

`phi_l` = layer-`l` activations of a frozen VGG (LPIPS's own extractor); `w_l1=3.0, w_lpips=1.0`,
the same mix the decode term uses. In one line: **does the frame-to-frame change in the prediction
match the frame-to-frame change in the truth**, scored in pixels and in perceptual features.

It is the **difference of embeddings**, not the embedding of the difference — the latter would feed
VGG a signed near-zero tensor it never saw in training. For the pixel term the two coincide
(subtraction commutes with identity), which is why the distinction is easy to miss. L1 rather than
L2 on pixels because a temporal difference image is sparse and L2 would let the largest change swamp
the rest.

### 8.9 Derivative arms: the result

`bs_deriv_w10` (all heads, weight 10) LOST and was killed at ep4. `bs_deriv_w1` ran to ep3+.

Open-loop LPIPS, cam_scene, matched epochs (all arms subsample 10, steps x 0.333 s):

| ep 3 | 21 s | 137 s | 275 s | 412 s | 550 s | mean |
|---|---|---|---|---|---|---|
| baseline (LPIPS) | 0.1117 | **0.0844** | 0.1344 | 0.1071 | 0.1000 | 0.1078 |
| **deriv w1** | **0.1058** | 0.1006 | **0.1099** | **0.0901** | **0.0959** | **0.1035** |
| deriv w10 | 0.1279 | 0.1103 | 0.1377 | 0.1169 | 0.1376 | 0.1289 |

`deriv w1` beats baseline on the mean at both ep1 (0.1143 vs 0.1191) and ep3, and wins 4 of 5
horizons at ep3 including all three long ones — a ~4% effect, i.e. real but small.

**`deriv w10` failed exactly as the design doc predicted.** The mechanism ENGAGED — directional
coherence roughly doubled, `cos` 0.149 vs the baseline's 0.070 — but it was bought by moving 32%
less (`ratio` 0.475 vs 0.696), the mean-seeking freeze. A frozen prediction compounds, so the damage
grew with horizon: +37.5% on lpips at 550 s, and worsening with training (+7.9% at ep1 -> +19.6% at
ep3). The `l1`-improves-while-`lpips`-degrades split was the giveaway.

### 8.10 DINOv3 arms: the result so far

`bs_dino_w25` (lpips 0, dino 25) **DIVERGED by epoch 1** and was killed. Decoder output on
`cam_scene` reached max **308**, min **-98**, with **58% of pixels outside [0,1]**;
`roundtrip_cam_scene_mse` 1326 against the healthy 0.0037; `dynamics/latent` 155 against 0.289.

**The cause is the property the term was chosen for.** A cosine is EXACTLY scale-invariant
(measured: scaling the true change by 0.3, 0.5 or 2.0 all score 1.000), so it exerts ZERO gradient
pressure on output MAGNITUDE. With `visual_lpips=0` the only thing bounding an unbounded conv decoder
is L1 at weight 3; at dino 2.5 L1 still wins, at dino 25 the cosine outvotes it 8:1 and the decoder
runs away. **A purely scale-invariant perceptual term must not be the ONLY perceptual term.**

| ep 1 | lpips_mean | psnr | ssim | l1 | motion | ae_floor |
|---|---|---|---|---|---|---|
| baseline | 0.1191 | 22.41 | 0.818 | 0.0353 | 0.259 | 0.0622 |
| dino 2.5 (lpips 0) | 0.2050 | 22.18 | 0.793 | 0.0386 | 0.156 | 0.0678 |
| **dino 2.5 + lpips 1** | **0.1175** | **22.47** | 0.814 | **0.0347** | 0.243 | **0.0608** |

At ep3 `dino 2.5` had closed from +72% to +40% on lpips_mean and had OVERTAKEN baseline on psnr
(23.04 vs 22.39) and l1 (0.0316 vs 0.0345) — the metrics neither arm optimises. Discount its lpips
gap heavily: it is graded on the quantity it gave up while the baseline optimises it directly.

**The weight is not 1.** With the shared l1 subtracted, LPIPS at w=1 contributes 0.8041 on cam_scene
and DINO at w=1 contributes 0.3545, so **2.5 is the matched-contribution point**. Swapping at 1.0
would have run the perceptual term at ~40% strength and measured a weakened version of it.

### 8.11 THE ACTUAL DIAGNOSIS — and it is not a loss-weighting problem

Operator observation, 2026-09-13, watching the rollouts: **the arm motion is good; the blocks pop in
and out of existence when interacted with.**

That asymmetry is the whole answer, and the literature names it. A deterministic MSE-trained decoder
emits `E[obs | tokens]`, the AVERAGE OVER POSSIBLE FUTURES. So:

* the **arm** is commanded directly — one possible future, and the average of one thing is that
  thing. It renders sharply.
* the **blocks** move only on contact, and contact outcomes are multimodal — tip left or right,
  slide or stick. The MSE-optimal prediction is the average of those, and a block averaged across
  two positions is a faint smear, i.e. IT VANISHES.

Published verbatim: *"deterministic models using loss functions like MSE will average together
possible futures, producing blurry predictions"*, and *"PredRNN produces blurry frames with objects
disappearing while still achieving a low MSE"*.

**So vanishing is the CORRECT answer to the objective we wrote** — the same structural trap as
record §13 one level up, and the reason three loss-term experiments moved so little. LPIPS vs DINO,
derivative terms and patch weighting are all ways of REWEIGHTING a loss whose optimum already puts a
ghost there. You cannot reweight your way out of a conditional mean.

Our **dynamics** is already stochastic (`mm_flow`, `stochastic_eval: true`, locked by the user
2026-08-10). Our **decoder is not**: `decode_kind: mse` on both image heads. We sample the latent
trajectory and then render it through a head trained to emit conditional means.

### 8.12 The fork this creates

`decode_kind=flow` and the derivative term are MUTUALLY EXCLUSIVE, verified in code
(`modalities.py:270`): a noised decoder has no clean single-pass prediction, and
`design/derivative_loss.md §5.1` measured that 62% of a temporal difference of its predictions is
the tau draw rather than motion. The eligibility guard raises.

Second cost: `decode_arch: up` is `decode_kind=mse` ONLY — it is a pure decoder with no analysis
path, and the dispatch raises rather than silently falling back. Flow needs `unet` or `vit`. But
`decode_arch: up` was itself an experimental win (robocasa §20), so switching moves TWO variables.

**Prior evidence is thin and does not settle it.** `_oneoff_decode_kind.py` compared decode heads on
FROZEN latents — flow beat mse by 16% on LPIPS at 6000 steps but `decode_stochastic` lost, with the
reasoning *"a converged head on a good code is nearly a point mass"* and *"a drifted latent makes the
mean WRONG, not UNCERTAIN"*. **That argument is about the DECODER's uncertainty given fixed tokens;
the failure here is uncertainty in the DYNAMICS — where the block ends up.** Different distributions.
The experiment also self-reports that both heads were MEMORISING. **No full training run has ever
used `decode_kind=flow` on an image head.**

### 8.13 The other two candidate fixes

* **Object-centric / SlotDiffusion** (arXiv 2305.11281). Represent a frame as a few SLOTS, each
  binding to one object, rather than a grid of patches. A block becomes a persistent ENTITY with
  attributes, so when the gripper occludes it the slot survives even though the pixels do not —
  object permanence becomes structural instead of something to be learned. A diffusion decoder
  renders slots back to pixels. The principled fix; it would replace the token bag.
* **Stochastic Adversarial Video Prediction** (arXiv 1804.01523). Keep the architecture, change the
  objective: a latent variable for WHICH future plus an adversarial loss, so samples must look real
  rather than average-real. A discriminator rejects a smeared block because no real frame contains
  one. Cheaper than object-centric, but adds adversarial instability.
* Worth knowing before promising a fix: **MemoBench (arXiv 2606.27537) reports that NO current video
  generation model reliably maintains object memory across occlusion.** Our blocks are occluded by
  the gripper at the moment of contact, so this may be partly unsolved rather than merely unsolved
  by us.

### 8.14 Tooling built along the way

* `VisualLoss` term registry — one declaration reaches the decode loss, the roundtrip anchor AND the
  derivative term. `smoke/visual_loss_parity.py` holds a 32-cell golden asserted with `torch.equal`;
  it caught that the old code multiplied `w_lpips` into the running total ONCE PER VGG LAYER, a 1e-7
  fp32 difference that showed up in exactly one cell.
* `models/dino_loss.py` + `DinoV3Term`, default off and byte-identical. `smoke/dino_loss.py` 17/17.
* `eval_control` now asks the env before running (clean `NotImplementedError` skip instead of dying
  inside MPPI on a KeyError); `eval.manifold_max_steps` caps what manifold forwards.
* Issue #19 (normalization diagnostic rollout) and #20 (the LPIPS dilution finding) on the repo.

### 8.15 THE REAL SYMPTOM: object IDENTITY, not permanence (2026-09-13)

Operator, watching rollouts: *"the arm motion is really good, but the blocks kinda pop in and out of
existence when interacted with"* — and later, decisively, that **untouched stationary objects** do it
too.

Rendering ground truth against prediction settled what "popping" means. **Blocks do not fade, blur
or vanish. They CHANGE COLOUR and merge.** Truth holds a blue and a green block; the baseline turns
green→red by 8 s and is down to a single red block by 21 s, with the arm tracking well and every
frame sharp throughout. The model knows *a block is there* and not *which block*.

This invalidated the framing we had both been using for two days, and it explains why a stationary-
object metric said "objects persist, 1.2× error growth": **a patch where green becomes red still
holds a block of similar size and brightness, so L1 barely moves.** I was measuring presence when
the failure is identity.

### 8.16 Seven hypotheses, measured and killed

All on trained checkpoints, no training runs. Tooling: `_oneoff_rollout_diagnosis.py`.

| # | hypothesis | verdict |
|---|---|---|
| 1 | loss reweighting (LPIPS / DINOv3 / derivative) | moved little; high doses actively hurt |
| 2 | mode collapse in the flow | **no** — 8 draws spread to 2.8× their own step-motion |
| 3 | we score away real diversity | **no** — best-of-8 gains 2.5% at 21 s, and the gain SHRINKS with horizon, the opposite of what multimodality predicts |
| 4 | codec capacity | **no** — per-frame encode→decode of the TRUE frames keeps blue blue and green green for the whole clip. Recon error is 0.29× the patch's own variation on dynamic patches vs 2.48× on still ones |
| 5 | off-manifold decode | **no** — rolled frames are 0.90–1.01× as sharp as real ones; encode→decode is 0.95× |
| 6 | identity is a low-variance latent direction | **no** — a recolour moves the latent **2.3× MORE per pixel** than a reposition, so the L2 flow loss is not blind to it |
| 7 | the flow never learned identity | **no** — at steps 0–12 (0–4 s) the rollout matches the codec ceiling with correct colours. **IT COMPOUNDS** |

Plus two compounding knobs run as real arms to ep13: `p_tf_dynamics=0.8` and `df_scale=0.1`, both
**6–8% worse** on OL LPIPS at every epoch, with no instability (`grad/norm/flow` 0.99 / 0.53).

**Notably the robocasa gradient blow-ups did NOT reproduce** — 2.4 here against 3.33e13 there at the
same dose. That vindicates the reading that §21.1's catastrophe was its `flow_hidden=512` base (§23),
not substitution per se. Two separable claims the record had fused: substitution is *not unstable*
here, it is simply *not helpful*.

### 8.17 The window hypothesis (running)

Temporal attention is a **sliding window** of `window` steps (`spacetime.py:6`, per token-slot,
causal + RoPE). With P=8 context frames, at rollout step *t* the window holds 8 context + *t*
predicted — so **the last TRUE frame scrolls out at t = window − P.**

| window | anchor leaves at | |
|---|---|---|
| **32** (all runs to date) | step 24 | **8 s** |
| 64 | step 56 | 19 s |
| 128 | never, within P+F=72 | — |

**Observed colour swaps begin at steps 24–32, i.e. 8–10 s.** That is exactly where the anchor leaves
at window=32. From that point the model attends only to its own predictions with nothing real to
anchor identity against — and **no reweighting of any objective can repair a model that has
structurally forgotten what the scene contained**, which would explain why every loss-side
intervention did nothing.

FALSIFIABLE: if the break point MOVES with the window, confirmed; if it stays at ~8 s, the timing
match was coincidence.

**Arms:** `bs_win64` and `bs_win128`, otherwise identical to `bs_stride10` — proprio stays
`decode_kind=flow` (proprio=mse existed ONLY in the derivative arms, forced by the eligibility
guard), `p_tf_dynamics=1.0`, `df_scale=0.0`.

**CONFOUND, stated not hidden: autobatch cut the batch 13 → 8 → 4** and epoch cost rose to 5,490 s
and 7,901 s. Epochs are not compute-matched either (1,094 / 1,778 / 3,556 batches). So aggregate
LPIPS from these arms is unreliable, especially at batch 4. The readable claim is narrow: **does the
colour-swap onset move?** That is qualitative and a batch difference should not manufacture it.

Early, and not the decider: `win64` ep3 `lpips_mean` **0.1008 vs the baseline's 0.1078** — the first
arm in the whole sequence ahead at a matched epoch, on a third fewer samples per step. One point.

**ep1 is too early to read the break point at all** — baseline and win64 are equally mushy there,
neither renders crisp blocks, so there are no clean colours to watch swap. The test needs ~ep9.

### 8.17b THE WINDOW HYPOTHESIS IS FALSIFIED (2026-09-14)

Both arms rendered at the first epoch where they draw crisp blocks. **The break point did not move.**

| | window | anchor leaves at | PREDICTED break | ACTUAL break | OL LPIPS, compute-matched |
|---|---|---|---|---|---|
| baseline | 32 | step 24 | 8 s | ~8 s | **0.0930 @10k steps** |
| `bs_win64` | 64 | step 56 | **19 s** | **~3–5 s** | 0.0980 @10k |
| `bs_win128` | 128 | never (in training) | **never** | **~1–3 s** | 0.1006 @21k |

`win128` — where the true context is NEVER out of view during training — breaks EARLIEST of the
three. A stronger manipulation producing a worse result is the signature of a hypothesis that is
wrong, not one that is under-dosed. The 8 s timing coincidence that motivated this was exactly that.

And they cost: wider windows cut the batch (13 → 8 → 4), raised epoch cost to 5,350 s and 7,750 s,
and lost on OL LPIPS at every matched step count. Killed.

**Nine hypotheses now measured and dead.** What is solid: the codec CAN hold identity; the flow HAS
learned it; it COMPOUNDS away by ~8 s, sharply and confidently, into a different scene. It is not
mode collapse, capacity, a scoring artefact, off-manifold decode, latent salience, teacher-forcing
substitution, diffusion forcing, or the attention window.

No lever survives. The remaining explanations are not knobs — **2.06 h of data and a 6.8M-parameter
model** — so the next collection session is worth more than any config change available here, and it
should target object interactions, where identity has to survive contact and occlusion.

### 8.20 The capacity sweep — and the CONTROL that made it readable (2026-09-14/15)

> **VOID — see §8.24.** Every arm here was read before its codec floor converged. `tok64`'s floor
> (0.1634) equalled its rollout (0.1627): the codec could not reconstruct a frame it was shown, so
> the arm measured codec convergence speed, not prediction. Capacity is an OPEN question again.
> The control's batch-8 ≡ batch-13 finding also needs re-checking at converged floors.

Capacity was the one axis never varied here, and §21.2 had explicitly left it open: `num_tokens=64`
was killed after 2 evals **for cost, not evidence**, having posted the best codec floor on this
dataset at the time. So two arms, two different claims about what room was missing:

| arm | change | params | what it claims |
|---|---|---|---|
| `bs_tok64` | `num_tokens` 32 → 64 per camera | 11.94M (vs 11.92M) | more **slots** — places for distinct objects |
| `bs_d256` | `model.d` 128 → 256 | 18.23M | more **dimensions** — what each slot can say |

`tok64` is the clean version of the hypothesis: it doubles the bag while adding essentially **no
parameters** (32 extra query embeddings × d=128 per camera).

**Both autobatched to 8 against the baseline's 13.** That is a confound shared by both arms and by
neither baseline — exactly the one that made the window sweep unreadable (batch collapsed 13 → 8 → 4
and it was only noticed afterwards). So the third arm was a **control**: `bs_batch8ctl`, the baseline
config with `data.autobatch=false data.batch=8` and nothing else touched.

OL LPIPS on `cam_scene`, at **matched gradient steps** (1778/epoch at batch 8, 1094 at batch 13):

| arm | gstep | OL mean | vs baseline |
|---|---|---|---|
| **`bs_batch8ctl`** | 1778 | **0.1167** | **1.0% worse** |
| `bs_d256` | 1778 / 5334 / 8890 | 0.1460 / 0.1575 / 0.1519 | 26% / 60% / 62% worse |
| `bs_tok64` | 1778 / 5334 | 0.1627 / 0.1637 | 41% / 66% worse |

The control lands *on* the baseline curve — 1.0% apart, with `@+8` of 0.0608 actually beating the
interpolated baseline. **Batch 8 is not the cause.** Both capacity axes therefore lost on merit, and
both were moving the wrong way while the control improved. `d256` blew up at ep3 (train 12.88,
recovering to 5.52 by ep5) which briefly looked like it might excuse the width axis — but its ep5
reading, taken post-recovery, is still 62% worse. Both axes closed.

**Run the control.** It cost one GPU for a few hours and converted two uninterpretable arms into a
clean result. Without it the honest conclusion would have been "capacity may or may not help, batch
may or may not be why" — which is what the window sweep had to settle for.

### 8.21 THE DIAGNOSIS: this is OVERFITTING, not capacity or architecture

> **PARTLY VOID — see §8.24.** The val/train evidence below (gap 1.09 → 1.69, val turning up at
> ep9) still stands on its own. What does NOT stand is the supporting argument from the capacity and
> window arms, and the `tok16` mirror test that appeared to adjudicate it — all read before their
> codec floors converged.

`bs_stride10`, train vs val vs ratio:

| | ep1 | ep3 | ep5 | ep7 | ep9 |
|---|---|---|---|---|---|
| train | 5.71 | 3.78 | 3.03 | 2.53 | 2.52 |
| val | 6.24 | 4.98 | 4.50 | 4.23 | **4.26** |
| val/train | 1.09 | 1.32 | 1.48 | 1.67 | **1.69** |

The gap widens monotonically and **val turns UP at ep9** (4.2276 → 4.2575) while train is flat
(2.5279 → 2.5179). That is overfitting: 56 episodes, 2.06 h, ~22k training timesteps at subsample 10,
against a 12M-parameter model.

**It retro-explains the entire pile of dead hypotheses** rather than adding a tenth to it. An overfit
dynamics model memorises plausible scenes instead of generalising transitions, so:

- extra capacity buys more memorisation — **both** axes worse, monotonically
- a longer attention window buys more still — and `win128`, the *strongest* manipulation, broke
  **earliest** (§8.17b), which was the most confusing result in the record until now
- predictions come out **sharp and confidently wrong** rather than blurred — memorised, not averaged
- best-of-8 gains 2.5% and shrinks with horizon — the failure was never a lack of diversity
- the codec is fine while the rollout drifts — per-frame reconstruction needs no generalisation

The batch-8 control corroborates the mechanism: it reaches a 1.65 gap by **ep3**, where the batch-13
baseline was at 1.32 and did not hit 1.65 until ep7. More gradient steps per epoch → faster
overfitting. The gap is step-driven, not epoch-driven.

**There is no more data to add.** `campaign1-tests` is EMPTY (0 episodes, only `campaign.json`) and
`campaign2-play` is **3.1 minutes**. Campaigns 8–9 are the held-out eval set and spending them
destroys the measurement. More data means new recording, not reprocessing.

**What is actually available:** the world model has **no dropout knob at all**. `optim.weight_decay`
is `1e-4` (very low for this gap). `variations.noise_injection.observations_encoded_pre_fusion`
exists and is OFF (`scale: 0.0`). So: weight decay, shrinking the model, or latent noise — built as
`wizard/scripts/blockstack-regularize.sh {wd|tokens|noise}`.

Note on the `noise` mode: `df_scale=0.1` (isotropic context noise) already lost 6–8%. This knob
injects at a different point, so it is not the same experiment, but the prior is poor.

The mirror test is the cheap one: the capacity axis already has two points going the *wrong* way, so
if overfitting is the story, going **down** that axis should help. Watch `eval_ae_floor` when doing
it — fewer tokens also means less room to reconstruct one frame, which would confound the rollout read.

### 8.22 Two more corrections to things that were believed

- **`detach_every` is 32, not 8, on every block-stack arm.** Asserted from memory as 8, which would
  have meant gradients spanning only 2.7 s against a ~8 s break — a compelling story for a
  compounding failure, and wrong. At 32 with F=64/subsample 10 gradients span **10.7 s**, already
  past the break. BPTT truncation is a much weaker lever here than it looked. (`vl128.yaml` does
  record that `detach_every: 8` was tried and **froze** the model.)
- **`metrics.jsonl`'s `step` field is the EPOCH INDEX, not the gradient step.** It reads 1,3,5,7,9
  for evals every 2 epochs. A comparison table was built reading it as steps before this was
  noticed. `wizard/scripts/olcmp.py` now applies batches-per-epoch by hand and prints both columns,
  with the trap in its docstring.

### 8.23 The self-kill bug, fifth occurrence — now actually fixed

`pkill -f <pattern>` matching the calling shell was fixed once by reading `/proc/<pid>/cmdline`
instead. It came back anyway: the kill loop was inside a `bash -c` whose **own command line
contained the literal text `experiment=bs_tok64`** (in the `case` pattern), so it matched itself and
died mid-script, before the launch that was supposed to follow. Four earlier occurrences silently
skipped later steps; this one left a GPU idle and no control running.

The durable guard is a **type check, not a pattern refinement**: require `/proc/<pid>/exe` to resolve
to `*python*`, which a bash shell can never satisfy.

```bash
for p in $(pgrep -f "quickdr[a]w.train_world_model"); do
  case "$(readlink /proc/$p/exe)" in *python*) : ;; *) continue ;; esac   # never a shell
  tr '\0' '\n' < "/proc/$p/cmdline" | grep -qx "experiment=$TARGET" && kill "$p"
done
```

Also: `bs_tok64` ignored SIGTERM for the full 240 s and needed SIGKILL. Always verify death with
`kill -0` in a loop; never assume `kill` worked.

Third related trap: a bare `run_summary` launch fails the **5-point assertion** (`trying_detail` is
required). Cost one relaunch when the summary was written inline rather than via the script.

### 8.24 THE READING ERROR THAT INVALIDATED §8.20 AND §8.21 (2026-09-15)

**Open-loop LPIPS is only comparable across arms whose CODEC FLOOR has converged.** It was read at
ep1-ep5 on six arms whose floors had not. Everything §8.20 concluded about capacity, and the
"mirror test" that appeared to confirm §8.21, measured codec convergence speed and not dynamics.

`eval_ae_floor/cam_scene/lpips_mean` is a per-frame encode->decode of the TRUE frames: the best the
rollout could possibly score. Pulled next to the rollout number it destroys the sweep:

| run | ep | gstep | OL | floor | OL-floor | OL/floor |
|---|---|---|---|---|---|---|
| `bs_stride10` | 1→9 | 1094→9846 | 0.1191→0.0930 | 0.0752→**0.0305** | 0.044→0.063 | 1.58→3.05 |
| `bs_batch8ctl` | 1 / 3 | 1778 / 5334 | 0.1167 / 0.0978 | 0.0484 / 0.0363 | 0.068 / 0.062 | 2.41 / 2.69 |
| `bs_wd01` | 1 / 3 | 1778 / 5334 | 0.1250 / 0.0930 | 0.0525 / 0.0306 | 0.073 / 0.062 | 2.38 / 3.04 |
| `bs_tok16` | 1 | 1778 | 0.1399 | 0.0686 | 0.071 | 2.04 |
| `bs_tok64` | 1 / 3 | 1778 / 5334 | 0.1627 / 0.1637 | **0.1634 / 0.1732** | −0.001 / −0.010 | **1.00 / 0.95** |
| `bs_d256` | 1 / 3 | 1778 / 5334 | 0.1460 / 0.1575 | 0.0653 / **0.1582** | 0.081 / −0.001 | 2.24 / **1.00** |

**`bs_tok64`'s floor was 0.1634 against a rollout of 0.1627.** Its codec could not reconstruct a
frame it was *shown*. The rollout sat exactly at the codec ceiling, so the arm carried no
information about prediction at all. `d256` reached the same degenerate state by ep3.

What died with it:

- **§8.20 is void.** "Capacity closed on both axes, with a control" was the cleanest-looking result
  in this record. Both arms were killed while their codecs were still converging, and a bigger
  bag/width converges *slower*. Capacity is once again an OPEN question.
- **The §8.21 mirror test is void.** `tok16` looked 20% worse, but its floor is 0.0686 against the
  control's 0.0484. Nothing was learned about binding-vs-overfitting from it.
- **`bs_wd01`'s apparent ep3 win is void.** OL 0.0930 beats the control's 0.0978 by 4.9%, but its
  floor is 16% better (0.0306 vs 0.0363) — the codec improved more than the rollout, so relative to
  what its own codec can express it got *worse*.

**The ratio does not rescue it.** When the floor is high there is little headroom for the rollout to
be worse, so `OL/floor` → 1 mechanically. `tok64`'s 1.00 is a ceiling effect, not good dynamics. The
baseline's own ratio climbs 1.58 → 3.05 purely because its codec improves faster than its rollout;
the ratio therefore tracks codec convergence stage too, and is not a normalisation.

**Rules going forward:**

1. Never quote OL LPIPS without the codec floor next to it. `olcmp.py` must print both.
2. An arm is not readable until its floor has plateaued. On the baseline that is ~ep7 (0.0313 →
   0.0305), i.e. ~7600 gradient steps — far beyond the ep1-ep3 where six arms were judged.
3. A floor at or above the rollout number means the arm is degenerate: report it as "codec did not
   converge", never as a dynamics result.
4. Matched gradient steps is necessary but NOT sufficient. Two arms at the same step with different
   floors are not comparable.

This is the same failure as §8.18's "look at the pictures" and the `latent_cos` correction: a number
was read without checking what it was capable of meaning. The difference is that this one produced
confident, well-tabulated, internally-consistent conclusions across two sections before it was
caught — which is what made it durable.

### 8.25 THE INVARIANT: `OL - floor` is pinned at ~0.063 and nothing has moved it

Once the codec floor is subtracted, every arm in this sweep produces the same number:

| run | ep | OL | floor | **OL − floor** |
|---|---|---|---|---|
| `bs_stride10` | 5 / 7 / 9 | 0.0980 / 0.0946 / 0.0930 | 0.0340 / 0.0313 / 0.0305 | **0.0640 / 0.0633 / 0.0625** |
| `bs_wd01` | 3 / 5 / 7 | 0.0930 / 0.0963 / 0.0910 | 0.0306 / 0.0317 / 0.0280 | **0.0624 / 0.0646 / 0.0631** |
| `bs_tok16` | 5 | 0.1014 | 0.0324 | 0.0690 |
| `bs_batch8ctl` | 1 / 3 | 0.1167 / 0.0978 | 0.0484 / 0.0363 | 0.0682 / 0.0615 |

On the baseline it RISES to ~0.063 and stops: 0.0439 → 0.0599 → 0.0640 → 0.0633 → 0.0625. The codec
keeps improving (0.0752 → 0.0305) and the rollout improves in lockstep, leaving the dynamics
contribution fixed.

**Every OL LPIPS improvement measured across this entire branch was the codec getting better.** This
is the floor-corrected version of §8.24: that section established the comparisons were invalid, this
one says what the valid comparison actually shows.

`bs_wd01` is the clean illustration. Its ep7 OL of 0.0910 is nominally below the baseline's 0.0930
plateau and would have been reported as the first real win — but its floor is also lower (0.0280 vs
0.0305) and `OL − floor` is 0.0631 vs 0.0625, i.e. identical. Weight decay bought a better CODEC,
not better prediction. Its trajectory also bounces 0.0930 → 0.0963 → 0.0910, noise around a flat
line, and ep3 alone looked like a 5.7% win.

**Thirteen arms, one invariant number.** Nothing has moved it: weight decay (1e-4 → 1e-2), capacity
in BOTH directions (16 / 32 / 64 tokens, d 128 / 256), the attention window (32 / 64 / 128), loss
reweighting (L1, LPIPS, DINOv3 per-patch cosine, first-order derivative), `p_tf_dynamics`,
`df_scale`, `detach_every`, `action_aggregate`, temporal subsampling.

That pattern is what a STRUCTURAL limit looks like, not a tuning problem. Nothing in this
architecture binds a token to an object: the bag is undifferentiated, so "which block is which" must
be re-derived diffusely at every one of ~30 rollout steps, and the per-step cost of doing that is
~0.063 LPIPS regardless of codec quality or weight norm. It also explains the one fact that has
held since §8.15 — the codec holds identity perfectly on true frames while the rollout loses it. The
codec never has to CARRY identity through a step.

**Read `OL − floor`, not `OL`.** Both plateau, but only the difference isolates the dynamics. A
change that improves the codec moves OL and means nothing for the failure being chased.

Two routes remain, and neither is another knob:
- **object binding**, needing an unsupervised formulation (§8.13 ruled out slot attention for lack
  of object trajectory labels — the question is whether a binding loss is possible without them)
- **more data** — 2.06 h is very small for a video world model, and §8.21 confirmed there is nothing
  left to reprocess, so this means recording

### 8.26 The binding probe: there is NO object binding to sharpen (2026-09-15)

§8.25 said the failure looks structural, and the standing suspicion was that nothing binds a token
to an object. Before building an unsupervised binding loss, measure whether there is any binding to
sharpen. `src/quickdraw/_oneoff_binding_probe.py` — checkpoint-only, runs alongside live arms.

**Method.** Ablate one image token at a time (replace it with its own temporal mean — in
distribution, unlike zeroing), decode, and take `|Δimage|`. That gives each token a spatial
FOOTPRINT over 24 frames / 8 s of val episode 0, on `bs_stride10` ep9 (floor converged, 0.0305).

| measure | value | reference |
|---|---|---|
| concentration (energy in top 5% of pixels) | **0.498** (min 0.291, max 0.656; 32/32 above 0.25) | uniform = 0.050 |
| overlap (mean pairwise IoU of top-10% masks) | 0.373 | 0 disjoint, 1 identical |
| centroid drift over 8 s | 7.89 px | scene's own moving content 10.24 px → **0.77×** |
| **inter-token spread at a fixed instant** | **8.33 px** | image diagonal 160 px → **0.052** |
| drift decomposition | **common-mode 6.20** vs individual 4.89 | — |
| pairwise-distance corr, frame 0 vs t | 0.433 (0.805 → 0.400) | 1.0 = rigid pan |

**Verdict: NOT BOUND.** Tokens are sharply localised — 10× more concentrated than uniform, every one
of the 32 — but they are localised to *the same region* (all footprint centroids within ~8 px on a
160 px diagonal) and their motion is mostly **common-mode**: the whole bag slides toward wherever
the action is. There is no spatial division of labour for a binding loss to sharpen.

**This was got wrong first.** Concentration 0.498 plus drift 0.77× content read as "CONTENT-BOUND:
emergent binding is present, a binding loss has something to sharpen" — and that was nearly
reported as a green light to build one. It is an artifact: if the DECODER is content dependent, then
ablating ANY token produces a footprint near the current action, so every token appears to follow
content while owning nothing. The two checks that actually discriminate are **inter-token spread at
a fixed instant** and the **common-mode / individual drift split**, and neither was in the first
version of the probe. Both are now in it, and its verdict logic requires them.

The general lesson, third time in this record after `latent_cos` and §8.24: a localisation measure
that never asks "localised to DIFFERENT places?" cannot distinguish binding from a content-dependent
readout. Measure the division of labour, not the localisation.

**What it means for the plan.** This closes the more attractive of the two routes left in §8.25. A
binding *loss* needs an existing weak division of labour to sharpen; there is none, so binding here
is an architecture change (slots with competition, or an explicit per-object latent), not a
regulariser — and §8.13 already ruled out slot attention for lack of object trajectory labels. It
also independently corroborates §8.25: with nothing tying a token to an object, identity must be
re-derived diffusely at every one of ~30 rollout steps, which is exactly a fixed per-step cost that
no amount of weight decay, capacity or window tuning would move.

That leaves **more data** as the one route not yet closed, and it needs recording, not reprocessing.

### 8.27 Fewer tokens: a transient gain that REVERTED — not a fix (2026-09-16)

> **HEADLINE RETRACTED, see §8.28.** This section was first written as "FEWER TOKENS FIXES IT" on
> four monotone points and two filmstrips. `tok16` ep13 then reverted to 0.0640, the baseline
> plateau, and the operator — watching the actual videos — reported `tok16` "still has weird
> glitching disappearing blocks, so this wasn't solved." Both are right and the claim was wrong.
> The numbers below are accurate; the conclusion drawn from them was not.

Halving the image tokens, 32 -> 16 per camera, produced a four-point improvement against the §8.25
invariant that then gave itself back. `bs_tok16`, identical to the baseline but `num_tokens: 16` on both image heads,
batch pinned to 8.

| `OL − floor` | ep5 | ep7 | ep9 | ep11 |
|---|---|---|---|---|
| `bs_stride10` (32 tok) | 0.0640 | 0.0633 | 0.0625 | — |
| `bs_tok16` | 0.0690 | **0.0590** | **0.0582** | **0.0566** | ep13: **0.0640** |

Monotone after its floor converged at ep5, reaching ~10% below the baseline plateau — and then at
ep13 returning to 0.0640, which IS the baseline plateau. The gain was transient.
The gain concentrates at the mid horizon, which is exactly where identity was failing:

| `@+412` (13.7 s) | | `@+824` (27.5 s) | `@+1651` (55 s) |
|---|---|---|---|
| `bs_stride10` ep9 | 0.0810 | 0.0896 | 0.0872 |
| `bs_tok16` ep11 | **0.0595** (−27%) | 0.0739 | 0.0955 (no gain) |
| `bs_tok16` ep13 | 0.0792 (−2%, gone) | 0.0850 | 0.0980 |

**AND THE FILMSTRIPS AGREE.** This is the first time in this branch a metric and the pictures have
pointed the same way, and §8.18 says the pictures decide.

- **Episode 0.** Baseline: 3 blocks at +1, then by **+1180 and +1415 the table is essentially
  empty** — blocks reduced to faint smudges. `tok16`: three saturated, distinctly coloured blocks in
  **all eight frames** across the full 55 s.
- **Episode 1.** Baseline: drops to a **single red block on an empty table at +708**, 1–2 faded
  blocks elsewhere. `tok16`: 2–3 blocks throughout; red is lost mid-rollout and +472/+708 soften,
  but blue and green persist the whole way.

Dramatic on ep0, moderate on ep1. Where blocks survive the colours stay correct — no swapping.
Positions still diverge from ground truth, but that is unavoidable open-loop over 55 s and was never
the complaint.

**Why this is the opposite of the intuition, and why §8.20 pointed the wrong way.** The capacity
sweep went UP first (64 tokens, d 256) and both arms looked catastrophic — but §8.24 showed they were
killed with unconverged codecs and measured nothing. The axis was right; the direction and the
reading were both wrong. With `tok16` at 0.0566 and the 32-token baseline at 0.0625, the dose curve
now reads: fewer image tokens, lower dynamics error.

It also fits §8.26. There is no object binding: all 32 footprints cluster in one region and drift
together, so identity is re-derived diffusely at every rollout step. FEWER tokens means less to
re-derive, and less opportunity for the re-derivation to disagree with itself — which predicts
exactly the observed shape, a gain at short and mid horizons that washes out by 55 s.

`bs_tok8` launched as the dose extension (ep3: `OL − floor` 0.0655 against `tok16`'s 0.0675 at the
same epoch, `@+412` already 0.0582). Its codec converged FASTER than `tok16`'s (floor 0.0409 at ep1
vs 0.0686), so the starvation risk did not bite.

**Open:** where the curve turns. 8 may beat 16, or 8 may starve the codec enough to lose the rollout
gain. And the 55 s horizon has not improved at any token count, so whatever fails at long range is a
separate problem from the one just fixed.

### 8.28 WHY §8.27 WAS CALLED WRONG, AND WHAT THE VIDEOS SAY (2026-09-16)

Three independent things had to be believed at once for §8.27's headline, and the fourth point plus
the operator's own viewing killed it.

**1. The metric reverted.** `bs_tok16` `OL − floor`: 0.0690, 0.0590, 0.0582, 0.0566, then **0.0640**
at ep13 — the baseline plateau. `@+412` went 0.0595 -> **0.0792** against the baseline's 0.0810,
i.e. the 27% mid-horizon gain evaporated in one eval. This is the SAME shape as `bs_wd01` ep3
(0.0624, looked like a 5.7% win, reverted at ep5). The only difference is that `tok16` sustained it
for four points instead of one, which is exactly what made it convincing.

**2. The operator watched the videos and disagreed.** Verbatim: `tok16` "still has weird glitching
disappearing blocks, so this wasn't solved." The filmstrips in §8.27 are 8 sampled frames out of
1651; the MP4 rollouts show the frames in between. **Glitching between sampled frames is invisible
to a filmstrip by construction.** §8.18 said "look at the pictures" — the sharper rule is look at
the MOTION, because the failure is temporal and a filmstrip cannot show it.

**3. The operator's read on the dose curve differs from the metric's.** They judge `tok8` "a little
better than tok16". At matched epoch 3, `OL − floor` is 0.0655 (tok8) vs 0.0675 (tok16) and `@+412`
is 0.0582 vs 0.0970 — so the videos and the numbers agree that 8 is at least not worse, on the
early evidence available before the instance came down.

**What the token axis actually shows, stated conservatively.** Every converged `OL − floor` across
every arm on this dataset falls in **0.057–0.069**. `tok16` wandered to the bottom of that band for
four evals and came back. Nothing has broken the band. §8.25's invariant stands.

**The error to not repeat.** Four monotone points, a 27% gain on the horizon where the failure
lives, and two corroborating filmstrips still was not enough — because the rule that had already
been earned twice (`wd01` ep3, and §8.24's whole floor problem) is that a trend on this dataset is
not a trend until it survives a point that could have broken it. The right response to four good
points was to keep the arm running and say nothing, not to write the headline.

### 8.19 An operational lesson: eval products fill the disk

`eval:ae_floor` died at `win64` ep5 with `OSError: No space left on device` — 287/290 GB. Eval
products (rollout videos and images) run **~1.3 GB per eval epoch per run**; thirteen runs over three
days filled it. Training survived only because the callback treats a failed routine as non-fatal; a
checkpoint write would have been much worse.

Recovered 50 GB by dropping `logs/epoch_*` and local `wandb/` from nine already-killed runs, keeping
every `metrics.jsonl`, `progress.log` and checkpoint — so no quoted number was lost. Burn rate with
two arms and evals every other epoch is ~1.3 GB/h, i.e. ~40 h of headroom. Budget for it, or prune
products on a schedule.

### 8.18 Reading rules earned the hard way today

- **OL LPIPS on cam_scene is the decider.** `latent_cos` is a diagnostic that explains *why* a
  number moved; it is never the number. (Told three times before it stuck.)
- **Single-epoch per-horizon breakdowns are noise.** A "coherent" in-horizon win at ep9 had
  REVERSED by ep11. Only `lpips_mean` across several epochs is stable enough to read.
- **Do not transfer a robocasa finding without checking its base and its data.** Robocasa is a
  simulator replaying scripted demos — deterministic by construction. Its "the flow learned a
  near-deterministic map" result measured 2.8× sample spread when re-run here. Its `p_tf_dynamics`
  dose curve ran on `flow_hidden=512`, which §23 later identified as the collapse cause, under pure
  L2 rather than the L1+LPIPS where §23 says 512 is fatal.
- **The configs and the records disagree, and the records are right.** `mm_flow.yaml` still said
  "Try 0.9" for `p_tf_dynamics` long after §21.1 measured that wrong. Acted on it, had to kill the
  arm at launch. Now corrected in the config with the dose table.
- **Look at the pictures.** Five measurements agreed the model was fine on object permanence. One
  filmstrip showed the actual failure was colour swapping. The metric was answering a question
  nobody had asked.

## 9. Still open

- [ ] **The identity/persistence failure is NOT fixed** (§8.27 retracted, §8.28). `tok16` gave a
      four-point improvement that reverted to the baseline plateau at ep13, and the operator
      watching the MP4s reports it "still has weird glitching disappearing blocks". `tok8` looks
      mildly better than `tok16` to the eye and marginally better at matched epoch 3, but had only
      2 eval points when the instance came down. RESUME HERE: run `tok8` to ep13+ and compare
      against `tok16`, on the VIDEOS, not filmstrips.
- [ ] **Judge these rollouts on the MP4s, not the filmstrips.** 8 sampled frames out of 1651
      cannot show glitching between samples, which is what the operator sees and what the
      filmstrips missed (§8.28).
- [ ] The 55 s horizon (`@+1651`) has not improved at ANY token count. Whatever fails at long
      range is a separate problem from the one §8.27 fixed.
- [ ] ~~The identity/persistence failure is unresolved~~, hypothesis was OVERFITTING
      (§8.21), not capacity (§8.20, closed on both axes with a control), not the attention window
      (§8.17b, falsified twice), and not any of the nine in §8.16. Open arms: `bs_wd01`
      (`weight_decay` 1e-2) and the batch-8 control as its matched reference.
- [ ] If regularisation bends the val/train gap but not OL LPIPS, then overfitting is real but not
      the *cause* of colour swapping, and the next question is object binding — nothing in this
      architecture ties one token to one object. §8.13 ruled slot attention out for lack of object
      trajectory labels; revisit whether an unsupervised binding loss is possible without them.
- [ ] Recording more block-stack data is the only way to attack the gap from the data side (§8.21).
      2.06 h is small for a video world model.
- [ ] Should `campaign4-rgb` / `campaign6-combos` / `campaign7-precision` be pooled (current
      behaviour, one distribution) or should the model condition on campaign? The label is
      already carried in `task`, so this is a training-side choice, not a re-processing one.
- [ ] Fix `tests/test_roundtrip.py` in the collection repo to write outside `campaigns/`.
- [ ] Run `swoosh-validate --campaign campaign2-play` on the lab machine for the first real
      camera-vs-arm cross-correlation number (deferred from the collection work).

---

## Log

**2026-09-10** — Branch cut from `main` @ `934ea19`; lego closed out on `lego`. Drive pulled:
21 GB, 1228 files, 2.15 h usable, campaigns 3–9 all 32/32 green. Wrote `data/swoosh.py` (run
reader) and `longhand()` (processor); added an optional `val_ids` override plus `split_rule`
provenance to the shared builder. Split confirmed at 90/10 longest-first, landing at 83/17 with
both val episodes ~9–10 min. `tests/test_longhand_split.py` 8/8. Two smoke builds green
end-to-end (4 splits, obs_dim 17, action_dim 5, campaign labels intact); full 4-camera build
complete and verified: bit-exact vectors and a strict frame-alignment minimum at shift 0.

**2026-09-14/15** — Window sweep killed (§8.17b: falsified on filmstrips *and* on compute-matched
LPIPS, with the stronger manipulation breaking earliest). Reopened capacity per §21.2 with two arms
on two axes, `num_tokens 64` and `d 256` — both autobatched to 8 against the baseline's 13, so
`bs_batch8ctl` was launched as a batch-8 control. The control came in **1.0% off the baseline** at
matched gradient steps, which closed both capacity axes on merit (§8.20) instead of leaving them
confounded the way the window sweep was.

That negative result plus the baseline's own loss curve produced the first unifying diagnosis in the
record: **overfitting** (§8.21). val/train widens 1.09 → 1.69 across ep1–9 and val turns up at ep9
while train is flat. It accounts for capacity hurting on both axes, the longer window hurting more,
sharp-and-wrong predictions, and best-of-k buying nothing — i.e. it explains the nine dead
hypotheses rather than joining them. Confirmed no data is available to add: `campaign1-tests` is
empty, `campaign2-play` is 3.1 min, campaigns 8–9 are the eval set.

Built `wizard/scripts/blockstack-regularize.sh` (`wd`/`tokens`/`noise`) and `wizard/scripts/olcmp.py`
(OL LPIPS at matched gradient steps). Launched `bs_wd01` (`weight_decay` 1e-4 → 1e-2) against the
control. Two corrections worth keeping: `detach_every` is **32** on every block-stack arm, not 8 as
asserted from memory (§8.22), and `metrics.jsonl`'s `step` is the **epoch index**, not the gradient
step — a comparison table was built on that error before it was caught.

Housekeeping: deleted `/home/isaac/data/lego_assemblies` (58 GB) after verifying recoverability by
actually downloading from `swoosh-data/lego_assemblies` with the token — 444 local mp4s against 444
on HF. 41 GB of it (`_quickdraw_frames`, a `*_h288` decode cache) was never uploaded and would need
regenerating. Disk 41 → 82 GB free. The self-kill bug recurred a fifth time and is now guarded by an
exe type check rather than a better pattern (§8.23).
