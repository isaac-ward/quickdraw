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

## 9. Still open

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
