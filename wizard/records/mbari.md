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

## Checklist

- [x] 1. Recon: network egress, Sketchfab API, MuJoCo EGL offscreen render (144x192 verified)
- [x] 2. Decisions: node, assets, episode split, resolution
- [x] 3. Fetch and convert the three meshes (glTF -> OBJ, recentred, per-model material dirs)
- [x] 4. Tank MJCF: 15 x 10 x 10 m, concrete walls, two window rows, ArUco grid, yellow crane hook to 1/4 depth
- [x] 5. MOLA 6-DOF dynamics: 3 forces + 3 torques, damping so momentum dies quickly
- [x] 6. Trajectory generators: random / forward-circles / circles-with-yaw, analytic, collision-free
- [x] 7. Recorder: state, actions, fpv + top-down + side-on at 144x192
- [ ] 8. Generate 240 x 30 s nominal (80 per class)
- [ ] 9. Generate 3 OOD sets, 50 trajectories each: glitchy thruster, fish, diver
- [ ] 10. Package to the house dataset format and push to HF
- [ ] 11. Read `wizard/records/*.md` on main AND on the `longhand` branch
- [ ] 12. Propose a training config and ask before launching
- [ ] 13. Train; log full open-loop and 16-step-ahead predictions
- [ ] 14. Uncertainty: classify the three OOD types against nominal
- [ ] 15. Pixel-level uncertainty videos
- [ ] 16. Control benchmark: reach the dock, and reach a random pose, under TRUE dynamics vs the LEARNED world model

## Open questions

- Episode length for the OOD sets: 50 trajectories x 10 min total means 12 s each, which is short
  against 30 s nominal episodes. Flagged for the user at step 9.
