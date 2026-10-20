# mbari — underwater world model in the MBARI test tank

Running record for the MOLA AUV / MBARI test tank work: a MuJoCo tank, 2 h of 30 Hz multi-view data on
HF, a world model trained on it, uncertainty-based OOD detection, and a control benchmark of learned
dynamics against true dynamics. Started 2026-10-18.

## The goal, stated once

A research output showing that a world model trained on nominal AUV footage in a tank can (a) predict
open-loop over useful horizons, (b) flag three distinct kinds of out-of-condition episode from its own
uncertainty, localised to pixels, and (c) be used as the dynamics model for closed-loop control,
measured against the true simulator dynamics on two tasks.

## Decisions taken (2026-10-18)

| | |
|---|---|
| runs | on this node — the decommission was called off |
| geometry | real Sketchfab CC-BY meshes where available, procedural otherwise |
| nominal data | 2 h at 30 Hz = 216,000 steps, cut as **240 x 30 s** episodes, 80 per behaviour class |
| frame size | **144 x 192**, matching block-stack, so `ae_bottleneck: 9` and the existing pipeline carry over unchanged, and both axes divide by 16 |
| OOD | 3 sets x **50 trajectories** each (raised from 20, 2026-10-18) |
| thruster OOD | **probabilistic glitch**: 30% of steps the commanded action is replaced by something wild, rather than a uniform 1/5 gain. A constant derate is a different *linear* plant the model could simply re-fit; an intermittent glitch is a distribution shift in the action-conditional, which is what the uncertainty is supposed to catch |

## Assets

All CC-BY from Sketchfab, pulled via the API 2026-10-18. Attribution and per-model licences in
`src/quickdraw/sim/mbari/assets/`.

| model | faces | bbox (file units) | note |
|---|---|---|---|
| MOLA AUV | 84,619 | 0.806 x 0.477 x 0.561 | needs explicit scaling; the only one not in metres |
| Docking station | 460 | 2.478 x 2.149 x 1.824 | already real metres, and ships **AprilTag 36h11 textures** — the real docking fiducials, so the docking benchmark gets its markers for free |
| Scuba diver | 56,454 | 1.790 tall | already real metres |

TWO SUBSTITUTIONS, both recorded rather than quietly made. The diver originally linked
(`skfb.ly/puCPB`) reports `isDownloadable: false` and the API 403s, so this is the lightest full-body
downloadable diver on Sketchfab. The great white shark was CGTrader, which has no public download API,
so the OOD fish is procedural unless a file is supplied. Both are OOD objects: what matters is that they
are novel shapes absent from training, not that they are those specific models.

THE ASSETS LIVE UNDER `src/` ON PURPOSE. The repo root is not bind-mounted into the container (only
`src`, `conf`, `.env`, `logs`, `scratch`, `pyproject.toml`, `uv.lock`), so anything written to a
top-level `assets/` exists only in the container's writable layer and dies with it — which is exactly
what happened on the first attempt, and the commit caught nothing but the README.

## Review round 2 (2026-10-18 16:30) — what was wrong and what was done

The first dataset was generated and a world model trained on it before review. Review found real
faults, every one of which had been invisible in the metrics:

| fault | cause | fix |
|---|---|---|
| **the AUV was a scatter of struts** | quadric decimation per colour GROUP spends the budget on the largest connected piece and deletes the rest; 84,619 -> 5,586 destroyed the hull | decimate per connected COMPONENT with a floor; 61,299 faces, every piece intact |
| the AUV was uniformly yellow | the model is not textured -- 14 parts, six `baseColorFactor` materials -- and merging them discarded that | one mesh/material/geom per colour group. The real MOLA: BLACK frame, light-grey hull, cyan nose dome, yellow thruster pods |
| parts rendered white-silver | trimesh writes a `.mtl` beside each `.obj` and MuJoCo honours it OVER the MJCF material | strip the sidecar and the `mtllib` line |
| two walls had window COLUMNS | the row index ran down the IMAGE, and the end-wall textures are portrait while the long walls are landscape | both rings placed by tank DEPTH -> continuous rings on all four walls |
| side-on camera saw concrete | it sat outside the glass, shooting through a textured wall plane | moved INSIDE the tank |
| windows read as black holes | they were painted as dark rectangles | each opens onto a RENDERED room, tinted glass |
| tag and window clutter | 7+5 windows and 21+24 tags per surface | both halved |
| tank appeared to continue above water | there was nothing above the water line but void | walls stop at z=0; enclosed dark warehouse above with eight ceiling lamps, a deck, and a gantry |
| one fish was often never seen | a single intruder crossing 15 m in 12 s misses the FPV frustum | three fish |
| diver stood upright | no orientation applied | pitched nose-down |

THE LESSON, recorded because it is the expensive one: **the metrics could not see any of this.** A
world model trained happily on a broken mesh in a cluttered tank, and the first sign of trouble was a
human looking at a picture. Renders of the asset itself -- not of the scene, of the ASSET -- belong in
the build step.

## Review round 3 (2026-10-18 19:00) — the second pass over the scene

| fault | cause | fix |
|---|---|---|
| walls lost their concrete | I had fallen back to flat shading after `type="2d"` textures tiled per box FACE and read as vertical planks | rendered the real wall geometry under four mappings side by side: 2d/uniform, 2d/non-uniform, cube/uniform, cube/non-uniform. 2d plank at every texrepeat and both uniform settings; cube+uniform projects RADIALLY into a starburst centred on each box; **cube + texuniform="false"** gives each face the whole image and reads correctly at every box size. Wall images are square now because cube faces are |
| the fish were approximated ellipsoids | no mesh had been found for them | **Great White Shark**, 3,690 faces, CC-BY. Swims along -Y raw, so the geom yaws +90 deg onto +X, which is the heading `mover` already writes. 1.6x -> 4 m |
| the hook was capsules and a sphere | approximated | **Crane Hook**, 10,795 faces, CC-BY, block and hook. Hangs along -Y raw; euler x=-90 deg stands it up, 0.28x -> 1.55 m |
| hanging buoy not neon | rgba 0.08 0.80 0.25 | 0.18 1.00 0.25, emission 0.18 |
| the AUV was a black silhouette in the chase view | the frame group's `baseColorFactor` is literally **(0,0,0)** and it is 33k of the mesh's 85k faces, so it rendered as a hole in the water whatever the lighting | floor every channel at 0.09 and raise specular to 0.55 -- what dark anodised aluminium looks like underwater. Side by side the chase view goes from a silhouette to a readable frame with thruster pods. Worth a regenerate because generation turned out to be **25 minutes**, not the 90 I had assumed |
| chase camera filled with wall | at -5.4 m it sat OUTSIDE the tank whenever the AUV neared an end wall, and the walls are solid boxes now | -4.2 m, 1.5 m up |

**The diver OOD became a purple buoy.** The source mesh is a T-pose. The posing attempt rotated
every vertex past 22% of the arm span about an estimated shoulder point (1,479 of 14,648 vertices,
62 degrees) and rendered T-pose vs posed side by side: the arms were still out to the sides in both
the top and the side view. The user's standing instruction for exactly that outcome was to swap the
condition for the hanging buoy turning purple instead of neon green, so `ood_diver` -> `ood_buoy`
everywhere. It is a cleaner OOD anyway: static, so there is nothing to confound with motion, and
it changes exactly one thing in the frame.

Every one of these was found by rendering and looking. None of them would have moved a metric.

## What `longhand` says, and what it predicts for this dataset

Read on `origin/longhand` (1143 lines; the copy on main is the 408-line short version). Four
findings bear on this experiment.

**§8.25 — the invariant.** Thirteen arms on block-stack, one number: `OL - floor` pinned at ~0.063,
unmoved by weight decay (1e-4 to 1e-2), capacity in BOTH directions (16/32/64 tokens, d 128/256),
attention window (32/64/128), four loss reweightings (L1, LPIPS, DINOv3 per-patch cosine, first-order
derivative), `p_tf_dynamics`, `df_scale`, `detach_every`, `action_aggregate`, temporal subsampling.
Every OL improvement measured on that branch was the CODEC getting better. The diagnosis: nothing in
the architecture binds a token to an object, so "which block is which" is re-derived diffusely at
each of ~30 rollout steps, costing ~0.063 LPIPS regardless.

**The prediction for MBARI.** The tank is rigid and in nominal data the only thing moving is the
camera. There are no object identities to lose, so the failure that dominates block-stack should be
largely ABSENT here. If MBARI's long-horizon quality is also stuck at a floor-offset invariant, the
object-binding diagnosis was incomplete and something more general is going on. That is a real test,
and it is free -- it falls out of the run we are doing anyway.

**§8.29 — the munge is the decode MEAN, not the dynamics.** `decode_kind: mse` emits
`E[frame | latent]`, and averaging a multimodal conditional gives half-blocks. MBARI's conditional
given the latent is close to unimodal (rigid scene, deterministic dynamics, one moving camera), so
`mse` is defensible here in a way it was not there. Revisit only if the rollouts look soft.

**§8.28 — judge on the MP4s, not the filmstrips.** Eight frames out of 1651 cannot show flicker
between samples, which is what the operator sees. Hence an mp4 per trajectory shipping with the
dataset.

**One place the record is deliberately NOT followed.** It says report `OL - floor`. The user's
standing instruction is raw OL LPIPS @+64/@+128 against the historical best -- not the floor, not a
penalty, not the mechanism. Raw is what gets reported.

**THE LEVER THIS OPENS, and why the dataset is now 6 h.** §8.21 and §8.25 both end at the same
wall: *2.06 h is very small for a video world model, and there is nothing left to reprocess, so this
means recording.* Block-stack could not get more -- it is a real robot. MBARI is simulated: 2 h costs
25 minutes of wall clock. The one lever thirteen arms could not pull is nearly free here, so the
nominal set went from 240 episodes (2.00 h) to **720 (6.00 h)**, OOD unchanged.

## Docking: 0.173 m -> 0.006 m, and three wrong guesses on the way

**The number was never a ceiling.** I reported 0.16 m mean final distance under TRUE dynamics as
"the ceiling any learned model is measured against". The user pushed back that it should be better.
It should, and it was not even a ceiling.

Guess 1: coarse actions. `repeat=6` holds each command 0.2 s, so the finest correction available is
one impulse. Built a near-goal fine mode (1-step actions, more samples, less noise) and measured
**no change at all**: coarse 0.173 m, fine 0.173 m, best-along-path actually worse (0.052 -> 0.085).

Two different planners cannot land on the same number by coincidence, so instead of a third guess I
dumped the distance-vs-time curve of a single trial:

| step | 0 | 240 | 480 | 560 |
|---|---|---|---|---|
| dist (m) | 7.87 | 5.92 | 1.61 | 0.20 |
| speed (m/s) | 0.00 | 0.52 | 0.63 | **0.57** |

**It was still in transit when the 600-step budget expired.** "Final distance" was measuring where
the clock stopped it, which is exactly why two planners returned the same figure. And it crawls:
7.9 m in 19 s, averaging 0.4 m/s against a 1.21 m/s cruise, `|a_lin|` at 0.2-0.6 of max. With a
TERMINAL-weighted cost at 2.4 s of lookahead, closing 1 m from 7.9 m out moves the objective 13%,
which sampling noise and the mean-path term swamp -- there is almost no gradient to descend until
it is already close.

Also misread: `within 0.5 m: 100%` counts trials that PASSED THROUGH 0.5 m, not trials that ended
there.

**The fix was deleting my own MPPI.** The user asked why `sim/mbari/control.py` had a bespoke
sampler when `controller/mppi.py` exists. Reading that module answered the control question too --
its config carries `chunk` ("execute this many steps of each plan before replanning"), which is the
`exec_n` I had reinvented an hour earlier, and `beta_vel: 0.0  # near-goal velocity penalty OFF: it
made the controller stall at the gate boundary`, which is the term `pose_cost` still had at 0.08.

Routed through the shared core (`_mppi_step` / `_score` / `_commit`, given a rollout_fn and a
reward_fn), same trial, same seed, same start:

| | old | ported |
|---|---|---|
| final | 0.173 m | **0.006 m** |
| best | 0.052 m | 0.002 m |
| holds position | no | yes -- last samples 0.06, 0.00, 0.00 |

**Not yet attributed.** Three things changed together: summed `-distance` instead of terminal
weighting (dense shaping, which is what the crawl predicts is missing), `beta_vel` 0.08 -> 0, and
`beta_ctrl` 0.02 added. A three-way ablation runs while the GPUs are busy so this says which one
bought the 29x instead of asserting it.

**The lesson, which is the same one this file keeps recording.** Two planners agreeing exactly was
the evidence that the metric was wrong, not the planner -- and one curve of the raw quantity
settled in minutes what three rounds of reasoning about the planner had got wrong. Look at the
trajectory, not the summary statistic.

## STATE OF PLAY (2026-10-19 01:15) — read this first if you are picking this up

**Everything upstream of training is done.** What is running, and what to do next.

### The data
- Raw episodes: `/app/logs/mbari/data/v6/{train,ood_thruster,ood_fish,ood_buoy}/ep_*.npz`
  — **720 nominal (6.00 h at 30 Hz) + 3 x 50 OOD**, 100 GB, no gaps.
- Per-trajectory videos: `/app/logs/mbari/data/previews/<split>/ep_*.mp4` — **870 of them**, four
  cameras side by side (FPV / chase / top-down / side-on), 3.0 GB.
- Processed (LeRobot) dataset: `/app/logs/recording_2026_10_19_00_30_50_mbari-tank`, **20% val**
  (576 train / 144 val). An earlier build at 10% val is on HF and is being replaced.
- Published: `https://huggingface.co/datasets/isaac-ronald-ward/mbari-tank`.

### The experiment now running: a DATA-VOLUME ablation
This is the point of generating 6 h. `longhand` §8.21/§8.25 end at the same wall — thirteen arms
failed to move `OL - floor` off 0.063 and the conclusion was *"2.06 h is very small for a video
world model and there is nothing left to reprocess, so this means recording"*. Block-stack could
not get more data; this dataset is simulated, so 6 h cost 65 minutes.

| | run A | run B |
|---|---|---|
| script | `train_mbari.sh` | `train_mbari_2h.sh` |
| GPU | 0 | 1 |
| train episodes | 576 (**6.00 h**) | 240 (**2.00 h**) via `+data.train_episodes=240` |
| epochs x steps | 14 x 35,812 = **501,368** | 40 x 11,937 = **477,480** |
| dataset / val / norm stats | **the same one, the same 144 val episodes, the same stats** |

Matched compute to within 5%, so the arms differ in DATA VOLUME and nothing else. They read ONE
dataset and run B is capped inside the loader (`set_train_episodes`, train-only) — packing a
second smaller dataset would have given it its own val split and its own normalization statistics,
so two things would move at once.

`finish_and_train.sh` is armed: it waits for the rebuild, ships the previews + PROCESSING.md into
the dataset, pushes to HF, then starts run A and (120 s later) run B.

### What remains
13. open-loop + 16-step-ahead on both arms — `run_after_train.sh <run_dir>`
14. uncertainty -> 3-way OOD classification
15. pixel-level uncertainty videos
16. control benchmark, learned arm vs true vs analytic
Plus: attribute the docking fix (`scratch/_attrib.py`, queued), and dock videos
(`scratch/_mppi_clips.py` -> `logs/mbari/control/clips/`).

### Things that will bite you, all found the hard way tonight
- **`pgrep -f "foo"` inside `bash -lc '... foo ...'` matches ITSELF.** Two queued jobs deadlocked
  on this. Use the bracket trick: `pgrep -f "[f]oo"`.
- **Killing a wrapper by pattern does not take its children.** Relaunching generation without
  checking left two processes per shard racing on the same npz; one file was corrupted (not even a
  valid zip). Count the distinct `--shard` arguments afterwards.
- **`tank.build()` writes `assets/tank.xml`, and every shard calls it.** Two episodes died on
  `ParseXML: empty file`. The write is atomic now (temp file + `os.replace`); do not undo that.
- **The eval memory knobs are not what they look like.** `eval.horizon_n_episodes` is NEVER READ on
  the ood_horizon path (n_ep is hardcoded to 8 with an image head), and rows =
  `min(max(n_ep,64), n_ep*n_seg)` is CAPPED at 64. Only `closed_loop_horizon=64` (n_seg < 8) gets
  under it: 52.0 -> 28.7 GB. Both arms must carry the identical eval shape or their closed-loop
  numbers are not comparable.
- **The autobatch budget differs per arm**: card - reserve - resident frame store, and 6 h of
  frames takes ~54 GB, so run A's budget is 42 GB where run B's is 78 GB.
- **MuJoCo cannot tile a texture on a box geom.** 2d smears along one axis, cube ignores
  `texrepeat`, cube+texuniform starbursts. Wall panels are split until near-square instead.
- **Assets written by the container are root-owned** and will block a `git merge`. `chown -R
  1000:1000` the assets dir first.

### Two methodological fixes worth not regressing
- The OOD detector calibrated its nominal reference on **training** episodes. It now uses the
  held-out val draw (`held_out_ids`, reproducing the packer's split) and logs how many episodes the
  threshold came from. Thresholds set on data the model was fit to read optimistically.
- Val is **20%**, not the house 10%, because val is both the checkpoint monitor AND that nominal
  reference, and a 90th-percentile threshold from 72 episodes rests on about 7 in the tail.

## Session update (2026-10-19 02:00) — control, cameras, and two loader fixes

### The control benchmark now scores POSE, not position
`goals()` returned only the dock entry POSITION and `run_episode` was called with `goal_q=None`, so
even the task named "reach a random POSE" was scoring position alone — and an AUV that arrives at
the dock entry pointing the wrong way has not docked. Both tasks now carry a target quaternion:
the dock's is the DOCK BODY'S OWN orientation, so the vehicle ends aligned with the entry axis.

Scoring it needed care after the MPPI port. `controller.mppi._score` builds its observation as
`cat(p_xyz, v_xyz)` — six dims, no quaternion — so a `reward_fn` physically cannot see attitude.
The shared module's other path takes a per-step `dist` (G,K,H) override, which is what language
steering uses, so attitude rides there:

    dist = ||p - p*||  +  w_att * angle(q, q*)  +  wall barrier        w_att = 1.2 rad^-1

The benchmark reports final attitude in degrees beside final distance, so a position-only
"success" cannot hide. First pose-targeted results, TRUE dynamics:

| trial | start | final | attitude |
|---|---|---|---|
| dock 0 | 7.86 m | 0.093 m | 12.5 deg |
| dock 1 | 9.03 m | 0.154 m | 8.2 deg |
| random pose 0 | 2.92 m | 0.172 m | 12.7 deg |

### Cameras for watching a control episode
`run_episode` takes a `render=` and captures frames from the PLANT's own MjData each executed
step, so an episode can be watched in the tank rather than read off a plot.

- **The chase camera is useless at the moment that matters.** It is body-mounted 4.2 m astern, and
  the dock sits 1.1 m from a corner, so at arrival the camera is inside the concrete.
- **A GOAL camera was added, and it does NOT orbit.** First attempt circled the target, which looks
  better for one second and then sweeps straight through a tank wall. It is now fixed, aimed once
  at the goal, and placed on the tank-centre side of it: MuJoCo puts a free camera at
  `lookat - d*(cos az, sin az)`, so setting the azimuth to the target's own bearing from the origin
  guarantees the camera sits between the goal and the middle of the tank, which is always water.
  `UnderwaterRenderer.look_at(target, dist=4.6, elev=-18)`, addressed as camera `dockcam`.
- **Playback is REAL TIME.** Every frame at the 30 Hz sim rate written at 30 fps. The first version
  kept every 4th frame and wrote 15 fps, so a 20 s approach played in 10 and looked hurried.
- Five panels: FPV, Chase, Top-down, Side-on, Goal cam.
- A second `UnderwaterRenderer` in one process returns an UNINITIALISED buffer — every episode
  after the first rendered as pure noise. One renderer, reused. (Hit once before on the MOLA
  material A/B, solved there with separate processes, and walked straight back into here.)

### Two loader inefficiencies, fixed at the general layer (`data/dataset.py`, not MBARI)
- **Reading more than you keep.** A run capped to 240 of 576 train episodes called `np.load` on the
  whole 43 GB frame cache and discarded 25 GB, because the cap was applied after episodes were
  built. `load_fpv_frames(mmap=True)` + filtering ids BEFORE indexing -> ~18 GB touched.
- **Two processes decoding the same video.** The cache write was already atomic (per-pid temp then
  rename), but atomicity protects the FILE, not the WORK: two runs started together each decoded
  all 648k frames. Now one builder holds `<cache>.building` and the rest wait and mmap the result;
  a lock older than 30 min with no cache is treated as a crashed builder and stolen. Verified with
  three concurrent workers: one build, two waiters.
- Deliberately NOT fixed: a capped run still builds the FULL cache. The cache is keyed only by
  camera and size, so a partial one could be silently reused by a run that needs all of it.

### The ablation, as actually launched
Both arms read ONE dataset (`recording_2026_10_19_00_30_50_mbari-tank`, 576/144 at **20% val**),
with run B capped by `+data.train_episodes=240` in the loader. Packing a second smaller dataset —
the first attempt — gave it its own val split and its own normalization stats, so two things moved
at once.

| | run A | run B |
|---|---|---|
| train | 576 eps (6.00 h) | 240 eps (2.00 h) |
| batch | 6 | 6 |
| epochs | 6 | 14 |
| steps | 477,504 | 464,240 |

**Batch is PINNED at 6.** Autobatch sizes per-arm from `card - reserve - resident frame store`, and
6 h of frames is ~43 GB resident, so it offered run A batch 6 and run B batch 12 — different
gradient noise, and step counts diverging to 1.11 M vs 663 k.

**OPEN RISK: epoch wall-clock is unmeasured.** 79,584 steps/epoch for run A, and this harness logs
nothing between epoch boundaries, so there is no step rate until the first `[ep 0]` line. If an
epoch is ~11 h then 6 epochs is 66 h and the epoch counts need cutting. A cleaner design if it has
to be redone: `trainer.limit_train_batches=20000` on both arms, 4 epochs each = 80,000 steps
exactly equal, with identical val cadence.

### What is automatic and what is not
- **Automatic**, during training, every 2 epochs: `eval.during_train.evals.ood_horizon` writes
  open-loop AND both closed-loop modes (1-step and **16-step-ahead**) with filmstrips, mp4s and
  curves. That is checklist item 13.
- **Not automatic**: items 14 (uncertainty + 3-way OOD), 15 (pixel-uncertainty videos) and 16 (the
  control benchmark's learned arm). They need a checkpoint AND the raw npz episodes, which no
  training hook knows about — `run_after_train.sh <run_dir>` runs all three.

### Published
`https://huggingface.co/datasets/isaac-ronald-ward/mbari-tank` — 576 train / 144 val / 3 x 50 OOD,
plus **870 per-trajectory preview mp4s** (one per episode, four cameras, 3.0 GB) embedded in the
dataset card so they play on the page.

## Known mismatch: the published dataset shows a SILVER hull, the code now renders YELLOW

The MOLA's large pressure-hull group ships from the source model as flat grey (0.4, 0.4, 0.4) and
is recoloured to yellow in `tank._recolour`, matched BY COLOUR rather than group index so a mesh
reconversion cannot silently break it. The change was made after the dataset was generated, so:

- `huggingface.co/datasets/isaac-ronald-ward/mbari-tank` — all 870 preview videos, and the
  chase / top-down / side-on streams inside the LeRobot splits, show the **silver** hull.
- Anything rendered from here — control videos, new previews — shows the **yellow** hull.

**Training is unaffected.** The only stream the model sees is `fpv`, the vehicle's own forward
camera, and the vehicle does not appear in its own FPV. The mismatch is cosmetic and confined to
the third-person views.

TO FIX: regenerate and re-push when the dataset is next rebuilt for any other reason. Not worth a
65-minute regeneration plus a re-upload on its own.

Two other scene changes made at the same time, both AFTER the dataset was built, so they carry the
same caveat:
- **The dock is a procedural white cradle**, flush and square against the -X wall (`DOCK_YAW=0.0`,
  entry facing +X), replacing the Sketchfab mesh. Two reasons, and the second is not cosmetic.
  The mesh's geometry is NOT aligned to its own body frame, so even with the body square the dock
  rendered visibly skewed; and because the body quaternion is the control benchmark's ATTITUDE
  TARGET, the translucent ghost marking that target could never line up with the structure the eye
  reads as the dock. `_dock_geoms` builds a 0.90 x 0.72 m entry hoop with 1.30 m of rail behind it
  and a pair of guide vanes, axis-aligned by construction -- so "square in the dock" and "matches
  the target quaternion" are the same statement.
- **A translucent green ghost of the vehicle** can be placed at the goal pose (`mjcf(ghost=True)`,
  a mocap body, 25% alpha). OPT-IN and off by default, and never built for data generation -- a
  goal marker in the training frames would be a feature the model could read instead of learning
  the dynamics. NOTE: the alpha must be set on the GEOM, not only on the material. A geom's own
  rgba defaults to (0.5, 0.5, 0.5, 1) and that default OVERRIDES the material's alpha, so the first
  version rendered fully opaque while the material said 0.25.

## Session update (2026-10-20) — the warm start that never retired, v7, and the restart

### The controller arrived and then shook. The bug was three lines of warm start.

The docking record below says 0.173 m -> 0.006 m and calls it done. Watching the videos said
otherwise: the vehicle reaches the goal and then vibrates on it indefinitely. `final_dist` could not
see this, which is the methodological point — **a single terminal frame cannot distinguish "stopped"
from "passing through"**, and every number reported up to this point was a terminal frame.

The diagnostic (`control_study probe`) dumps `dist`, `|v|`, `|w|`, `|a|` and `|da|` for the last
seconds. The answer was immediate and unambiguous: at 5 cm from the target the planner was commanding
**|a| = 0.33 of full thrust on all six axes, continuously**, with `|da|` near zero. High command,
low command-CHANGE — not search noise, a stuck constant. That names the mechanism: the warm start.

Near the goal the cost landscape is nearly flat. Every candidate scores alike, the softmax weights go
uniform, the update barely moves `mean` — so whatever thrust the plan carried on approach keeps
firing forever, because nothing ever takes it away. Two separate faults kept it alive:

1. **`np.roll` WRAPS.** The last action of a plan became the first action of the next one. Stale
   thrust was not merely retained, it was recycled in a loop.
2. **`mean_decay` was never applied at all.** It lives in `MPPIConfig` and `controller/mppi.py`'s own
   loop applies it, but `_mppi_step` leaves it to the caller — and this caller did not.

Fixed to shift, zero-pad, decay:

```python
nxt = np.concatenate([plan[1:], np.zeros((1, plan.shape[1]), plan.dtype)], 0) * mean_decay
```

`python -m quickdraw.sim.mbari.control_study sweep decay`

| mean_decay | final_m | att | \|v\| | \|a\| | \|da\| |
|---|---|---|---|---|---|
| 1.0 (the bug) | 0.031 | 1.4 | 0.180 | **0.170** | 0.042 |
| **0.7** | 0.035 | 1.1 | 0.084 | **0.072** | 0.016 |
| 0.4 | 0.060 | 1.7 | 0.124 | 0.069 | 0.013 |
| 0.0 (cold start) | **0.867** | 3.1 | 0.148 | 0.073 | 0.015 |

The lesson is NOT that warm starts are bad. A cold start is catastrophic — 96 draws of pure noise
cannot find the approach without a sensible centre to sample around. The warm start has to exist AND
has to bleed away. 0.7.

### Penalise effort, not speed

`sweep effort` and `sweep settle`. Two candidate cures for terminal speed, and only one works.

| config | final_m | att | v_end |
|---|---|---|---|
| bctrl 0.02 bvel 0.00 | **0.143** | 10.6 | 0.512 |
| bctrl 0.20 bvel 0.00 | 0.183 | 10.1 | 0.562 |
| bctrl 1.00 bvel 0.00 | 0.309 | 13.4 | 0.583 |
| bctrl 0.02 bvel 0.50 | 0.233 | 12.1 | 0.491 |

A small cost on command magnitude damps the settling jitter; a large one simply prevents arrival. The
gated near-goal VELOCITY penalty is worse than useless (0.233 against 0.143) — which is exactly what
`controller/mppi.py` already said in the comment next to `beta_vel = 0`: *"it made the controller
stall at the gate boundary"*. The shared module had measured this before we re-measured it. Reading
the comment would have been cheaper than the sweep.

### Attitude weight is almost pure cost, and the replan period is the real lever

`python -m quickdraw.sim.mbari.control_study sweep attitude`

| config | final_m | att° | wander | \|v\| | \|a\| | cost |
|---|---|---|---|---|---|---|
| w_att 0.3 | **0.021** | 1.5 | 0.024 | 0.045 | 0.041 | 1x |
| w_att 1.2 (shipped) | 0.035 | 1.1 | 0.037 | 0.084 | 0.072 | 1x |
| w_att 4.0 | 0.083 | 1.3 | 0.134 | 0.185 | 0.133 | 1x |
| **exec_n 2** | **0.016** | **1.0** | **0.021** | 0.056 | 0.064 | 3x |
| exec_n 1 | 0.020 | 1.0 | 0.018 | 0.044 | 0.067 | 6x |

**`w_att` buys nothing above ~0.3.** Position error scales 4x from 0.021 to 0.083 as the weight
rises, while attitude error does NOT improve — 1.5°, 1.1°, 1.3°, no trend. Attitude has its own floor
near 1° set by the sampling noise, and weighting it harder only spends position. Tune it DOWN until
attitude starts to degrade, rather than up until it feels important.

**`exec_n` is the knee at 2.** `repeat` is how coarse the ACTIONS are; `exec_n` is how much of a plan
is SPENT before replanning. They were the same number only because `run_episode` executed exactly
what it planned. Separating them and halving the spend takes final error 0.035 -> 0.016 and wander
0.037 -> 0.021. `exec_n 1` is no better than 2 at twice the cost.

**Locked in as the default** (`control.py`): `w_att=0.3`, and one schedule
`dict(repeat=6, horizon=12, samples=96, sigma=0.20, exec_n=2)` with `FINE = COARSE`. The two-mode
range switch is kept but both modes now point at the swept schedule, so there is no second
schedule in the codebase that nobody measured. Both benchmark arms inherit it, so item 16 still
compares only the dynamics.

COST NOTE: `exec_n 2` replans three times as often. On TrueDynamics that is CPU and free; on the
LEARNED arm every replan is a batch of latent rollouts, so item 16 gets ~3x more expensive.

### A caution on the first combined render: three trials, one noise realization

The first render of the combined config returned 0.006 m / 0.4 deg / wander 0.011 on **all three
trials, identical to three decimals**, from three different starts (4.35, 3.64, 5.09 m) and three
different goals. Different episodes — the mp4s differ in size and content — and identical
settling.

The cause is a seed, not a bug in the controller: the render passed `seed=100` for every trial
where the sweep passes `seed=100 + i`, and `run_episode` seeds the planner as `seed + k`. So all
three drew the IDENTICAL noise sequence at every step index. Once the warm start has decayed near
the goal the plan mean is ~0 and the candidate set is essentially that noise, so the residual
jitter is one realization reported three times. The episodes are independent; **the settling
statistics are n=1**.

This is the same signature as the 0.173 m "ceiling" (two schedules, one number, and the number
belonged to the harness). Identical numbers from nominally independent runs are a seed or a
budget, never a result. `control_study sweep combined` re-measures it at seeds 100/101/102.

### Collisions are now a first-class object, in the planner AND the plant

Previously the barrier covered the four walls only, at a weight comparable to a metre of distance —
so a plan that clipped a buoy to save two metres of travel **scored well**, and did it. Now:

- `W_CRASH = 50.0` per step inside any solid. No amount of saved distance can buy that back.
- The barrier covers the same `KEEPOUT` volumes `trajectories.clear` uses, so **planner and plant
  agree on what a collision IS**. One predicate, two consumers; they cannot drift apart.
- `run_episode` terminates with `crashed=True` on contact, and the recorder truncates the episode.
- Target poses that would require a collision to reach are rejected at sampling time.

### v7: 6.01 h, collision-free, balanced, on HF

Regenerated with collisions active and the hook halved. Episodes now vary in length (they end on
contact), which needed `data.dataset.set_train_steps` alongside `set_train_episodes`, and a
`MIN_STEPS = 80` floor because a 0-frame episode cannot be encoded to mp4 and crashed the packer.
25 zero-length episodes got through the first time: `_start_pose` tested `inside()` when it should
have tested `clear()`.

`python -m quickdraw.sim.mbari.verify logs/mbari/data/v7` is the acceptance check, and it says:

```
  1147 episodes, 715945 steps = 6.63 h
  episodes containing a COLLIDING frame :    0   (want 0)
  zero-length episodes                  :    0   (want 0)
  ended on contact                      :  433 (38%)
  train  circle      244  mean_len 891  2.01 h   ended_on_contact   3/244
  train  circle_yaw  248  mean_len 886  2.03 h   ended_on_contact   6/248
  train  random      489  mean_len 455  2.06 h   ended_on_contact 400/489
```

**The 1/3-1/3-1/3 balance is in HOURS, not in episodes, and it has to be.** 82% of `random`
episodes end on contact against 1% of `circle` — a piecewise-constant random wrench walks into a
wall, a fitted circle by construction does not. So random episodes are half the length (455 steps
against ~890) and it takes twice as many of them (489 against 244) to buy the same 2 h. Anything
that counts EPISODES per class will read this dataset as 2:1 imbalanced when it is balanced to
within 2.5% on the axis that matters. Train total: 6.10 h.

Top-up rounds to fill the full 6 h rather than accepting the shortfall. Two bugs there, both
arithmetic rather than physics: fixed episode indices collided with prior rounds (so later rounds
silently overwrote earlier ones), and the deficit was computed from ASSUMED yield rather than
OBSERVED, so the top-ups were systematically too small.

### The 13.5 h epoch was `limit_val_batches`, never set

Training was cycling at 13.5 h per epoch against an estimated 2.9. Cause: `limit_train_batches=8000`
was set and `limit_val_batches` was not, so validation ran its full **19,896 batches** — two and a
half times the training it was validating. Both arms were killed (with permission) and relaunched
from scratch with `+trainer.limit_val_batches=400`. 135 h -> 29 h for the full schedule.

### The two arms, as restarted 2026-10-20 03:08

Same compute, different data volume — the ablation is unchanged, only the schedule is fixed.

| | runA | runB |
|---|---|---|
| dir | `logs/train_world_model_2026_10_20_03_08_59_mbari_fpv_v1` | `..._03_11_07_mbari_fpv_2h` |
| data | 6.11 h (all 785 train episodes) | 2 h (`+data.train_steps=173000`) |
| batch | 6, pinned | 6, pinned |

`data.autobatch=false data.batch=6` is deliberate: autobatch chose 6 for one arm and 12 for the
other because the resident frame store differs with dataset size, which would have confounded the
data-volume comparison with a batch-size comparison.

### Why the sweeps are committed now (`control_study.py`)

Every table above was produced by a separate throwaway in `scratch/`, which is gitignored. Four
copies of the same twenty lines, none of them reproducible by anyone else, and a record full of
numbers nobody can re-check. They are now one module where the grids are data and the loop is
written once; each table above names the command that regenerates it. `scratch/` keeps only genuine
one-offs — asset conversion probes, render-setup screenshots, the yield arithmetic.

### Transferring this to a different vehicle

Asked for, so written down. Structural and portable: retire the warm start; two planning modes
switched on range; decouple hold from replan period; make sure the cost actually SEES orientation
(ours silently did not — `_score` builds its observation as `cat(p_xyz, v_xyz)`, six dims, no
quaternion, so a `reward_fn` physically cannot read attitude and it has to ride in the per-step
`dist` override); collisions catastrophic; measure settling not arrival; dump the distance curve
before tuning anything.

NOT portable, and all three invert for an underactuated airframe: our plant is neutrally buoyant
with heavy drag, so thrust-to-zero is a valid resting state and drag does much of the settling work.
A quadrotor must decay toward HOVER thrust, not zero; must penalise `||a - a_hover||`, not `||a||`;
and cannot treat the attitude weight as optional, because attitude IS its lateral force — the
`w_att` result above is a six-actuated-DOF result and does not transfer.

## Checklist

- [x] 1. Recon: network egress, Sketchfab API, MuJoCo EGL offscreen render (144x192 verified)
- [x] 2. Decisions: node, assets, episode split, resolution
- [x] 3. Fetch and convert the three meshes (glTF -> OBJ, recentred, per-model material dirs)
- [x] 4. Tank MJCF: 15 x 10 x 10 m, concrete walls, two window rows, ArUco grid, yellow crane hook to 1/4 depth
- [x] 5. MOLA 6-DOF dynamics: 3 forces + 3 torques, damping so momentum dies quickly
- [x] 6. Trajectory generators: random / forward-circles / circles-with-yaw, analytic, collision-free
- [x] 7. Recorder: state, actions, fpv + top-down + side-on at 144x192
- [x] 8. Generate 720 x 30 s nominal (240 per class) = **6.00 h** -- raised from 240 (see below)
- [x] 9. Generate 3 OOD sets, 50 trajectories each: glitchy thruster, sharks, purple buoy
- [x] 10. Package to the house dataset format and push to HF -- `isaac-ronald-ward/mbari-tank`
- [x] 11. Read `wizard/records/*.md` on main AND on the `longhand` branch
- [x] 12. Training config written and launched (user asleep -- decisions listed above)
- [x] 12b. Review round 3 fixes (see below); dataset regenerated as **v4**
- [x] 12c. Review rounds 4 and 5; dataset regenerated as **v6**; an mp4 per trajectory
- [x] 12d. MPPI deduplicated onto `controller/mppi.py`; docking 0.173 m -> 0.006 m
      (that 0.006 is a TERMINAL-FRAME number and overstates it -- see 12e)
- [x] 12e. Controller actually settles: warm start retired, collisions catastrophic, schedule
      swept. 0.035 m / 1.1 deg with |a| 0.33 -> 0.072 of full thrust at the goal; `exec_n 2`
      reaches 0.016 m. All tables reproducible via `control_study.py`
- [x] 12f. Dataset v7 regenerated collision-free (6.01 h) and pushed to HF
- [x] 12g. `limit_val_batches` fix; both ablation arms relaunched from scratch 2026-10-20 03:08
- [ ] 13. Train BOTH ablation arms; log full open-loop and 16-step-ahead predictions (IN PROGRESS: epoch 3/10 as of 2026-10-20 19:00, ~4.5 h/epoch)
- [ ] 14. Uncertainty: classify the three OOD types against nominal
- [ ] 15. Pixel-level uncertainty videos
- [ ] 16. Control benchmark: reach the dock, and reach a random pose, under TRUE dynamics vs the LEARNED world model

## Measured, so far

| | |
|---|---|
| AUV dynamics | cruise 1.21 m/s at full surge, coast to 5% in 0.80 s, yaw 49 deg/s |
| trajectory tracking | circle and circle_yaw both 0.040 m mean error, 0.09 m p95, no actuator railing |
| gripper of the tank | 68 ArUco 5x5_250 markers, all unique, detector round-trip 4/4 |
| render cost | 307 ms -> 65 ms per 4-camera control step (mesh decimation, shadows off, one scene build) |
| dataset | 720 x 900 = **648,000 steps = exactly 6.00 h at 30 Hz**, plus 3 x 50 x 360 = 10 min per OOD set |
| on disk | 50 GB of npz -> 1.4 GB of mp4 after packaging |
| `delta_k` (fpv) | 0.1124 at stride 1 = 1.76x a 0.064 codec floor -> **no subsampling** |
| control, shipped schedule | 0.035 m, 1.1 deg, wander 0.037, \|a\| 0.072 of full thrust at the goal |
| control, `exec_n 2` | **0.016 m, 1.0 deg**, wander 0.021, at 3x the planning cost |
| control, before the warm-start fix | 0.040 m terminal, but \|a\| **0.33** continuous on all six axes |
| planner cost | coarse call 24 x 12 x 6 x 17 = 29,376 mj_steps; fine call 48 x 18 x 17 = 14,688 |

## Decisions made while the user was asleep

Flagged here rather than buried in commits, since item 12 said to ask first and there was nobody to ask.

1. **`subsample=1`.** Measured, not assumed -- see the table. Also removes the `action_aggregate`
   question entirely, and these are absolute commands so `sum` would have been wrong.
2. **`conf/model/mbari_fpv.yaml` is `vl128_blockstack_2cam` with only the dataset-forced fields
   changed** (proprio dim, action dim, one image head at 144x192 / bottleneck 9). Every inherited
   value was bought with an experiment; changing one here would confound the transplant.
3. **`max_epochs=40`, evals every 2 epochs, manifold and all three denoising routines OFF** -- the
   records say those four read torus-only fields and do not self-skip on a recorded dataset.
4. **`checkpoint_monitor=val/metric/fpv/mse`** explicitly, because the default monitor is a proprio
   metric and would select the wrong epoch for an image result.
5. **The thruster OOD is a 30% probabilistic glitch**, not the 1/5 derate originally planned -- the
   user's suggestion, and the better one: a derate is a different linear plant a world model can
   re-fit, an intermittent glitch is a shift in the action-conditional.
6. **Substituted diver and fish**: the linked diver is not downloadable (403) and the CGTrader shark
   has no API. Both are OOD objects, so novelty is what matters, not the specific artist's model.

## Open questions

- Episode length for the OOD sets: 50 trajectories x 10 min total means 12 s each, which is short
  against 30 s nominal episodes. Flagged for the user at step 9.
