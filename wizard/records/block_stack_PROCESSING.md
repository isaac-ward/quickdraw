# How `block-stack` was produced

Raw teleoperation recordings of a **Swoosh right arm** (UFACTORY xArm 7) doing block
manipulation, turned into LeRobot splits for world-model training.

Everything below is reproducible from this repo:

```bash
python -m quickdraw.data.processors +processor=block_stack \
    +source.dir=<raw campaign tree> +source.name=block_stack
```

Reader: [`src/quickdraw/data/block_stack.py`](https://github.com/isaac-ward/quickdraw/blob/main/src/quickdraw/data/block_stack.py) ·
Processor: `block_stack()` in `src/quickdraw/data/processors.py`

---

## 1. The robot and the teleop scheme

One xArm 7, right arm, **mounted 45° clockwise from vertical**. The mount is a fixed
rotation between the arm's base frame and the world frame:

```
R_WORLD_FROM_BASE = [[1, 0,     0    ],
                     [0, √½,   -√½   ],
                     [0, √½,    √½   ]]
```

It is recorded in every run's provenance and verified against the physical arm (jog +X/+Y/+Z
20 mm, confirm the world axis that moves). The processor asserts the recorded matrix matches
the one it assumes, so a re-mount cannot silently corrupt a rebuild.

An operator drove the arm with an **Xbox controller**:

| control | effect |
|---|---|
| left stick | planar motion parallel to the ground |
| right stick Y | height of that plane |
| right stick X | rotate the end-effector about the **world** vertical |
| right trigger | gripper, proportional |

End-effector positioning: the pose goes to `set_servo_cartesian` and the **xArm controller
solves the IK**. No IK is performed in this pipeline.

## 2. What was recorded, and at what rate

The collection side writes five independent JSONL streams plus four cameras. **All of them
share one monotonic clock origin** (`t_loop0`), recorded in `run.json`.

| stream | rate | contents |
|---|---|---|
| `controller` | 100 Hz | Xbox axes, post-deadzone (0.12) and post-expo (2.0) |
| `commanded` | 100 Hz | the integrated target pose the stick asked for |
| `xarm_command` | 100 Hz | the literal SDK arguments sent |
| `arm_state` | 50 Hz | what the arm reported back |
| `tick` | 100 Hz | control-loop timing |
| 4 × camera | 30 Hz | MJPG → MP4, plus one timestamp per frame |

Cameras are two scene views and two on the right gripper:
`scene_left`, `scene_right`, `gripper_right_bottom`, `gripper_right_top`, 640×480 @ 30 fps.

**Camera timestamps come from the V4L2 kernel buffer at capture**, not from when userspace
received the frame, so the ~21 ms read lag is removed at the source.

## 3. Synchronisation

There is **no cross-stream alignment step, deliberately** — no cross-correlation, no learned
offset, no shifting. One clock plus kernel-level camera stamps means the streams are already
on the same timeline. All the processor does is **resample**.

1. **Grid.** 30 Hz (= the camera rate), spanning `[t_lo, t_hi]` where `t_lo` is the *latest*
   start and `t_hi` the *earliest* end across all six streams. Cameras open staggered 0.25 s
   apart, so this trims 1–2 s off the front of each run — which is why usable duration is
   shorter than the wall-clock A-to-B duration in `run.json`.
2. **Nearest-in-time, never interpolated.** Every source runs 1.7–3.3× the grid rate, so
   interpolation would invent precision. It would also be actively wrong for `joints_real_deg`,
   for rotation components, and for the gripper's 20 Hz poll.
3. **Per-camera index map.** Each camera's timestamps are matched to the grid to decide which
   MP4 frame becomes row *i*, then decoded in one forward pass.

Resampling displacement is reported per stream as max, p99, and the fraction past that stream's
*own* half-period (the floor for any nearest-neighbour resample). On a representative 125 s run:

| stream | max | p99 | own bound | past bound |
|---|---|---|---|---|
| controller | 22.3 ms | 7.0 ms | 5.0 ms | 3.3% |
| arm_state | 33.7 ms | 9.4 ms | 10.0 ms | 0.4% |
| scene_left | 17.7 ms | 17.6 ms | 16.0 ms | 25.3% |
| scene_right | 7.8 ms | 7.8 ms | 16.0 ms | 0.0% |
| gripper_right_bottom | 18.0 ms | 18.0 ms | 16.0 ms | 20.6% |
| gripper_right_top | 4.3 ms | 4.2 ms | 16.0 ms | 0.0% |

The camera rows are **phase, not fault**. A camera whose true rate differs from 30 Hz by a
fraction of a percent drifts through the grid's phase, so its displacement sweeps the full
±half-period; `max ≈ half the camera period` is the arithmetic floor. The controller and arm
maxima are isolated stream hiccups (`dt > 3×` median), two and one respectively in that run.

## 4. `action` — 5 dims, the controller

```
[move_x, move_y, height, yaw, gripper]
```

`move_x, move_y, height, yaw` ∈ [−1, 1]; `gripper` ∈ [0, 1] where **1 = squeeze**.

**This is the raw human input, not a robot command.** The commanded pose and the literal SDK
arguments were both recorded and are both deliberately excluded — the target pose at time *t*
is `target[t−1] + stick × rate × dt`, so feeding it as an observation hands a world model the
answer to "where does the arm go next". Both remain in the raw recordings.

## 5. `observation_vector` — 17 dims, the arm

| idx | dims | field | source | transform |
|---|---|---|---|---|
| 0:3 | 3 | `ee_{x,y,z}_mm` | `pose_world_xyz_mm` | none; mm, world frame |
| 3:9 | 6 | `ee_rot6_*` | `pose_base_mm_deg[3:6]` | RPY → rotation matrix → world frame → first two columns (**not z-scored**, see below) |
| 9 | 1 | `gripper` | `gripper_pos` | `(x − closed)/(open − closed)`, **1 = open** |
| 10:17 | 7 | `joint{1..7}_rad` | `joints_real_deg` | degrees → radians |

Three choices that are not obvious:

- **`joints_real_deg`, not `joints_deg`.** The latter is the controller's *planned* angle;
  on an earlier corpus the two were measured diverging by up to 5.02°.
- **6D rotation, not Euler** ([Zhou et al.](https://arxiv.org/abs/1812.07035) — the first two
  columns of the rotation matrix; the third is recoverable by cross product, so nothing is
  lost). Euler angles wrapped 702 times in a single stream on an earlier corpus, and every
  wrap is a discontinuity a model must spend capacity memorising. Verified orthonormal on this
  data: column norms exactly 1.0, column dot product 7e-08.
- **Gripper polarity is inverted between the two streams.** Action `gripper` 1 = squeeze;
  `gripper_pos` 850 = open. The state is normalised so 1 = open (monotonic in aperture) and
  the action is left as recorded. They are different quantities and were not collapsed.

### A note on the rotation dimensions

The orientation is effectively **1 degree of freedom** in this corpus. Across all 56 runs the
tool axis stays **37.5°–50.5° from straight down with std 0.2°** — only its azimuth changes,
driven by the operator's right stick (111.6° of range). Singular values of the centred 6D
cloud are `[1.00, 0.31, 0.009, 0.004, 0.002, 0.001]`.

The full 6D is kept anyway, because it is exact and assumption-free: a pure-yaw
reconstruction `R = Rz(θ) · R₀` fits to mean 0.096° but **max 11.3°**, so there are real tilt
excursions that a 2-dim yaw encoding would discard.

### The rotation dimensions are NOT z-scored

`normalization_stats.json` gives dims **3-8** (`ee_rot6_0` ... `ee_rot6_5`) `mean = 0, std = 1`,
so they pass through unchanged. Every other dimension is z-scored on the **train split only**.
This is deliberate, and it is the one thing about this dataset worth reading before you train.

#### Why

A 6D rotation is six numbers in [-1, 1] obeying `|c0| = |c1| = 1` and `c0 . c1 = 0`, where
`c0 = dims[3:6]` and `c1 = dims[6:9]`. That coupling is the entire reason the representation is
worth using instead of Euler angles.

Per-dim z-scoring multiplies each of the six by a *different* factor. Here those factors would
have been `[3.5, 7.0, 324.4, 4.9, 2.5, 301.6]`. Two of them are enormous because **yaw about the
world vertical leaves the bottom row of the rotation matrix invariant** -- and `ee_rot6_2` and
`ee_rot6_5` *are* two entries of that bottom row (`c0_z` and `c1_z`). The teleop scheme only ever
rotates about world vertical, so those two are structurally pinned, not merely quiet.

Measured on the train split, applying a per-dim z-score would do this:

| | \|c0\| | \|c1\| | max \|c0 . c1\| |
|---|---|---|---|
| raw values in the parquet | 1.000 - 1.000 | 1.000 - 1.000 | 0.000000 |
| if z-scored per dim | 0.469 - 24.935 | 0.456 - 42.226 | 786.87 |
| **with the shipped stats** | **1.000 - 1.000** | **1.000 - 1.000** | **0.000000** |

A model cannot learn `|c| = 1` from a representation whose norm ranges 0.47 to 42. Secondarily it
would amplify sensor jitter: `ee_rot6_2`'s high-frequency residual is **119 percent of that dim's
total variation**, so after a 324x scaling the noise alone would be 1.25 sigma -- a
full-amplitude input channel carrying no information.

Per-dim z-scoring is correct for dims that are independent and differ in unit or scale (mm vs
radians). It is wrong for a group of dims that jointly encode one geometric object.

#### What this means for you

- **Using the shipped `normalization_stats.json`: nothing to do.** Apply `(x - mean) / std` to
  the whole vector; the rotation dims are already `(x - 0) / 1`.
- **Recomputing statistics yourself: preserve this.** Compute mean/std over train, then overwrite
  dims 3-8 with `mean = 0, std = 1`. If you z-score them you will silently destroy the rotation
  structure -- nothing will raise an error.
- **The third column is not stored** because it is recoverable exactly: `c2 = cross(c0, c1)`.
- **Only want a scalar heading?** `theta = atan2(c2_y, c2_x)`. But note the tool tilt is not
  perfectly constant: a pure-yaw reconstruction `R = Rz(theta) . R0` fits to mean 0.096 deg and
  **max 11.3 deg**, so a 2-dim yaw encoding does discard something real.

```python
import json, numpy as np
s = json.load(open("normalization_stats.json"))["observation_vector"]
mean, std = np.array(s["mean"]), np.array(s["std"])   # std[3:9] == 1, mean[3:9] == 0

x_norm = (x - mean) / std                             # x: (..., 17) raw from the parquet
x_back = x_norm * std + mean

c0, c1 = x[..., 3:6], x[..., 6:9]                     # exactly orthonormal
c2 = np.cross(c0, c1)                                 # the missing third column
R  = np.stack([c0, c1, c2], axis=-1)                  # full 3x3 world-frame rotation
```

#### The shipped statistics in full

| dim | field | mean | std | treatment |
|---|---|---|---|---|
| 0 | `ee_x_mm` | 523.3354 | 70.80737 | z-scored |
| 1 | `ee_y_mm` | -95.1079 | 88.21514 | z-scored |
| 2 | `ee_z_mm` | 119.5620 | 70.93004 | z-scored |
| 3 | `ee_rot6_0` | 0.0000 | 1.00000 | **identity — passes through unchanged** |
| 4 | `ee_rot6_1` | 0.0000 | 1.00000 | **identity — passes through unchanged** |
| 5 | `ee_rot6_2` | 0.0000 | 1.00000 | **identity — passes through unchanged** |
| 6 | `ee_rot6_3` | 0.0000 | 1.00000 | **identity — passes through unchanged** |
| 7 | `ee_rot6_4` | 0.0000 | 1.00000 | **identity — passes through unchanged** |
| 8 | `ee_rot6_5` | 0.0000 | 1.00000 | **identity — passes through unchanged** |
| 9 | `gripper` | 0.8738 | 0.17055 | z-scored |
| 10 | `joint1_rad` | -2.1194 | 0.27059 | z-scored |
| 11 | `joint2_rad` | -1.1055 | 0.22719 | z-scored |
| 12 | `joint3_rad` | 1.9915 | 0.21618 | z-scored |
| 13 | `joint4_rad` | 1.2705 | 0.33514 | z-scored |
| 14 | `joint5_rad` | 1.3487 | 0.37235 | z-scored |
| 15 | `joint6_rad` | 0.8736 | 0.34489 | z-scored |
| 16 | `joint7_rad` | 1.0363 | 0.28585 | z-scored |

| dim | action | mean | std | treatment |
|---|---|---|---|---|
| 0 | `move_x` | 0.0004 | 0.45708 | z-scored |
| 1 | `move_y` | -0.0080 | 0.56608 | z-scored |
| 2 | `height` | -0.0031 | 0.38903 | z-scored |
| 3 | `yaw` | -0.0002 | 0.33113 | z-scored |
| 4 | `gripper` | 0.5106 | 0.41864 | z-scored |

Action dims are all well-conditioned (amplification 1.8-3.0x) and z-scored normally.

## 6. What was dropped

Nothing is destroyed — every raw stream is retained upstream and any of this can be restored
by a re-run.

| stream | dropped | why |
|---|---|---|
| `commanded` | all 6 fields | a consequence of action + integrator state; leaks the answer |
| `xarm_command` | all 7 fields | same, plus SDK return codes |
| `tick` | all 4 fields | loop-health telemetry, not physics |
| `controller` | `raw`, `raw_age_s`, `input_age_s`, `connected` | pre-shaping axis values and liveness |
| `arm_state` | `joints_deg` | the *planned* angles (see above) |
| `arm_state` | `pose_base_mm_deg[0:3]` | **exactly recoverable**: `R_WORLD_FROM_BASE @ base_xyz == pose_world_xyz_mm` to the digit |
| `arm_state` | `report_alive`, `gripper_pos_t`, `gripper_pos_age_s`, `state`, `mode`, `error_code`, `warn_code` | liveness and fault telemetry, constant in all published runs |

## 7. Campaigns and splits

Nine recording campaigns plus bring-up folders. Campaigns 1–2 (bring-up), `shakedown`, `audit`
and four stray test artefacts were excluded. **Every published run passes all 32 of the
collection side's own validation checks.**

The per-episode `task` field carries the campaign name into `<split>/meta/tasks.parquet`, so
any split can be sliced by recording condition.

| split | source campaigns | episodes | frames |
|---|---|---|---|
| `train` | 3-play, 4-rgb, 5-play-long, 6-combos, 7-precision | 43 | 172,835 |
| `val` | (the longest episodes of the same pool) | 2 | 34,799 |
| `eval_purple_play` | 8-purple-play | 5 | 9,048 |
| `eval_purple_stack` | 9-purple-stack | 6 | 5,323 |

`eval_*` are held-out conditions, written verbatim with no random splitting.

### The train/val rule: longest trajectories to val

Val is **not** a random sample. It is the longest episodes, taken as the prefix whose frame
share lands closest to 10%, with a floor of two episodes.

**Why.** Open-loop rollout evaluation can only run as far as the *shortest* validation
episode — past that there is no ground truth to score against. On an earlier corpus a random
seed-0 draw pulled a short episode into val and capped every long-horizon number at a fifth of
the horizon the data actually supported.

Here train's longest episode is 15,996 steps and val's shortest is **16,606 steps ≈ 9 minutes**,
so the ordering is strict and rollouts are evaluable to nine minutes. The realised split is
**83.2 / 16.8 by frames** rather than exactly 90/10 — the two longest episodes are much longer
than the rest, and the two-episode floor binds.

**Read val loss accordingly.** Both val episodes come from `campaign5-play-long`. Val measures
long-horizon fidelity on long play trajectories; it is a rollout yardstick, not a
representative i.i.d. estimate of the training distribution.

## 8. Verification

`tests/test_block_stack_roundtrip.py` checks a built dataset against a fresh read of the
source recordings.

**Vectors are bit-exact.** `max |diff| == 0.0` for both state and action across all 18,193 rows
of val episode 0. No silent recast, reorder or truncation.

**Frames align with state rows.** Scored as a shift sweep, not a single number — mean |Δpixel|
of built frames against source frames at integer offsets:

| camera | −3 | −2 | −1 | **0** | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| `scene_left` | 6.81 | 5.69 | 4.31 | **3.02** | 4.24 | 5.55 | 6.61 |
| `gripper_right_top` | 5.46 | 4.59 | 3.76 | **3.28** | 3.84 | 4.66 | 5.51 |

The residual 3/255 at offset 0 is video re-encode loss. **The shape is the result**: a strict
minimum at 0, rising cleanly either side. A small number alone would prove nothing, because a
dataset misaligned by one row also scores "small" on a slowly-moving scene. The test asserts
both the argmin *and* that the curve is steep enough to discriminate, so it cannot pass
vacuously.

## 9. Known issues

- **`media/`** holds per-episode preview clips at build resolution. They duplicate the split
  videos and exist so a human can see what a split contains; they are not needed for training.
- **LeRobot's own reader needs `torchcodec`**, which needs ffmpeg's shared libraries. Any
  MP4-capable decoder reads these files; `quickdraw`'s loader uses `imageio`.

## 10. Summary of the numbers

| | |
|---|---|
| robot data | **2.056 hours** (7,400 s) |
| episodes | 56 |
| frames | 222,005 |
| rate | 30 Hz |
| cameras | 4 × 144×192 (from 640×480) |
| observation | 17 dims |
| action | 5 dims |
| raw corpus | 21 GB |
