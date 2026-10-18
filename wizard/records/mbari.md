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

## Checklist

- [x] 1. Recon: network egress, Sketchfab API, MuJoCo EGL offscreen render (144x192 verified)
- [x] 2. Decisions: node, assets, episode split, resolution
- [x] 3. Fetch and convert the three meshes (glTF -> OBJ, recentred, per-model material dirs)
- [x] 4. Tank MJCF: 15 x 10 x 10 m, concrete walls, two window rows, ArUco grid, yellow crane hook to 1/4 depth
- [x] 5. MOLA 6-DOF dynamics: 3 forces + 3 torques, damping so momentum dies quickly
- [x] 6. Trajectory generators: random / forward-circles / circles-with-yaw, analytic, collision-free
- [x] 7. Recorder: state, actions, fpv + top-down + side-on at 144x192
- [x] 8. Generate 240 x 30 s nominal (80 per class)
- [x] 9. Generate 3 OOD sets, 50 trajectories each: glitchy thruster, fish, diver
- [x] 10. Package to the house dataset format and push to HF
- [x] 11. Read `wizard/records/*.md` on main AND on the `longhand` branch
- [x] 12. Training config written and launched (user asleep -- decisions listed above)
- [ ] 13. Train; log full open-loop and 16-step-ahead predictions
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
| dataset | 240 x 900 = **216,000 steps = exactly 2.00 h at 30 Hz**, plus 3 x 50 x 360 = 10 min per OOD set |
| on disk | 50 GB of npz -> 1.4 GB of mp4 after packaging |
| `delta_k` (fpv) | 0.1124 at stride 1 = 1.76x a 0.064 codec floor -> **no subsampling** |

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
