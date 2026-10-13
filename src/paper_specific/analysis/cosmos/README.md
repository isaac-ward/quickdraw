# cosmos — the measurements behind `wizard/records/cosmos.md`

These were one-off scripts in `scratch/` (gitignored) and are kept because every one of them is the
PROVENANCE for a number that is asserted elsewhere — in `conf/interpret/block_stack.yaml`, in the
record, or in a commit message. Without them those numbers are claims nobody can re-check.

They are not a library and nothing imports them. Each is standalone, hardcoded to block-stack, and was
run inside the container as `.venv/bin/python <file>`. Paths are absolute `/app/...` for that reason.

| file | what it establishes |
|---|---|
| `bs_axes.py` | the sign of every block-stack action axis, by correlating `action[t]` against the observation delta over all 43 train episodes (r ≈ +0.94 on each diagonal). This is why `action_axes` says "moves away from the camera" and not something guessed from the name `move_x`. |
| `bs_axes2.py` | that rot6 (obs 3:9) is COLUMN-major — `[c0\|c1]` is orthonormal to 1e-4, `[r0\|r1]` is not — hence world yaw is `atan2(rot6_1, rot6_0)` and +yaw is counter-clockwise from above. |
| `bs_cam.py` | how world axes appear in the fixed `scene_left` camera: the arm located as the centroid of frame-to-frame change, regressed on `ee_xyz`. Vertical fit R² 0.90, HORIZONTAL only 0.33 — which is why the config says "left" for +y is the best estimate available rather than a confident one. |
| `grip.py` | that the gripper channel needs no debounce: 80 grasps over 9096 steps, mean hold 5.03 s, and a median filter only destroys real ones (80 → 52 at 1.4 s). |
| `cube_track.py` | the colour-segmented cube centroids that caught the VLM naming the one cube that never moves (Finding 8). The reason object attribution is not in the prompt. |
| `errtab.py` | the averaged per-frame L1 / MSE / PSNR / LPIPS / SSIM table, via the repo's own `image_curves` so the arithmetic matches eval. |
| `spec_gifs.py` | renders the clips to the house spec (`viz.label_panels` + `viz.save_gif`) from the raw npz. |
| `backfill_prompts.py` | regenerates `prompts_<i>.txt` for a run that predates prompt logging. Safe only because `build_action_prose` is a pure function of the recorded actions, the axes file and the window boundaries. |

`errtab.py` and `spec_gifs.py` read `raw_filmstrip_frames_<ep>.npz` under a run dir in `logs/`, which is
gitignored and local-only — see the record's "What is NOT backed up" section before assuming they run.
