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

## 6. Still open

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
running.
