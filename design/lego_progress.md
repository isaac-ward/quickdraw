# lego_assemblies buildout — execution checklist

Branch `lego`. Two single-GPU arms on `swoosh-data/lego_assemblies` (74 episodes, 341,494 frames,
30 Hz, dual xArm7 bimanual teleop). **Arm A** = scene prediction, one camera. **Arm B** = multicamera,
three heads. Plan and rationale: the published plan artifact; this file is the live status.

Ticked items are DONE AND VERIFIED — each carries the evidence that closed it.

---

## P0 — silent-failure fixes (~15 min, Arm B only, but they fail quietly so land them first)

- [x] `train_world_model.py:272` — now RAISES when there are >1 image modalities and no explicit
      `trainer.checkpoint_monitor`, instead of silently selecting on the first head. Single-head
      behaviour bit-identical.
- [x] `training/setup.py:118` — presets now apply to EVERY image head. **Verified:** 0/1/3 heads all
      correct. Also fixes a latent bug — a clash on a NON-FIRST head was previously undetected
      entirely; it now raises and names the head (`model.modalities.cam2.num_tokens=999`).

## P1 — processor + 6D wiring (~half day, SHARED)

- [x] `data/rotations.py` — continuous 6D rotation encoding (Zhou et al. 2019), state 28→34,
      action 16→20. **Verified:** `tests/test_rotations.py`, 7/7 checks on 65,817 real frames;
      sign-invariance exact (0.0), Lipschitz bound `||d6D|| <= 2*sqrt(2)*sin(theta/2)` holds
      everywhere (worst violation 2.4e-08), round-trip geodesic error 5.4e-06 deg.
- [x] `data/processors.py` — new `lego_assemblies()` + `_stage_clip()`. Frames staged to disk as
      JPG **paths**, not arrays: the shared builder pickles `Episode.frames` to encode workers and the
      longest episode here is 10,769 frames. Staging is resumable via a per-episode `.done` marker.
      Aspect preserved at stage (both camera families are 16:9); the square squash to the recipe's
      `img_size` stays downstream in `load_fpv_frames` where the recipe controls it.
- [x] Register in `PROCESSORS`.
- [x] Run it on 2 episodes → `logs/recording_2026_08_31_01_49_23_lego`. **Verified:** train obs
      (4145, 34) act (4145, 20); val obs (5573, 34) act (5573, 20); `summary.json` reports obs_dim 34,
      action_dim 20, fps 30, camera head_right, image_hw [288, 512] (16:9 preserved).
- [x] `check_dataset` passes — *"OK — dataset fits the config"*, 4074 train / 5502 val windows at P=8 F=64.

## P2 — configs + dims (~1 h, SHARED)

- [x] **DECIDED: proprio width 34** (user, 2026-08-31) — keep the joints. They are the only signal
      distinguishing arm configurations that share a TCP pose (null-space config), and 34 dims is
      negligible beside 32 image tokens. `data.obs_keep` stays null.
- [x] `conf/data/lego.yaml` — INHERITS `torus.yaml` (the generic data config; robocasa used it with
      overrides) and changes only `repo_id`, `cam: head_right`, `obs_keep: null`, `subsample`
      (placeholder, loudly flagged) and `subsample_all_phases: true`.
- [x] Dim overrides documented in both configs; passed on the CLI as vl64/up64 require.
- [x] `conf/model/vl64_scene.yaml` — inherits `vl64`, restates `modalities` in full. proprio dim
      6→34; image head renamed `image`→`scene_right` so BOTH arms log `val/metric/scene_right/mse`
      and the A/B is directly joinable (the name is the batch key AND the metric key —
      `setup.py:815` passes `image_head=img.name`, independent of `data.cam`).
- [x] `model_summary` clean: `proprio (B,T,34)→(B,T,1,128)`, `action_enc (B,T,20)`,
      `scene_right encoder (B,T,128,128,3)`, decoder 4.764M (matches vl64's documented 4,760,003),
      12.93M total, codec at 50% of cap.

## P3 — derive `data.subsample` (~2 h + a short AE run, SHARED) — **GATES ARM A**

- [x] Codec floor measured — `eval_ae_floor +ae_floor.taesd=true`, 512 frames @128px:
      **PSNR 23.79 dB, MSE 0.00418, RMSE 0.0647**. Robocasa's was 23.92 dB / 0.0637, so §13's table
      is a valid reference here, not a loose analogy. (TAESD is a REFERENCE codec — vl64 trains its
      own from scratch — so re-check once Arm A has a few epochs.)
- [x] Frame-delta sweep — new `smoke/lego_subsample.py`, 11 episodes across sessions, square 128px,
      same `[-1,1]` units as the floor. Reads SOURCE mp4s, so it did NOT wait on the full stage.

      | stride | Hz | frame-Δ RMSE | vs floor | |
      |---|---|---|---|---|
      | 1 | 30.0 | 0.0500 | **0.77×** | below the codec's own error |
      | 2 | 15.0 | 0.0770 | 1.19× | marginal |
      | 3 | 10.0 | 0.0963 | 1.49× | clears the floor |
      | 4 | 7.5 | 0.1109 | 1.72× | |
      | **6** | **5.0** | **0.1323** | **2.05×** | **CHOSEN** |
      | 8 | 3.8 | 0.1479 | 2.29× | |
      | 16 | 1.9 | 0.1854 | 2.87× | |

- [x] **`data.subsample=6`** (5 Hz). Rule: smallest stride BOTH ≥1.25× AND in the 2–5 Hz band every
      published long-rollout system converges on. "First to clear 1.25×" alone gives 3, on the
      marginal edge with the shortest horizon. At 6, F=64 spans **12.8 s** instead of 2.1 s.
- [x] Set in `conf/data/lego.yaml` with the full table + `subsample_all_phases: true`.

      **The prediction was wrong and the reason matters.** I expected stride-1 SNR *worse* than
      robocasa's 0.61× because 30 Hz > 20 Hz. It measured **0.77×** — better. Real bimanual teleop
      moves more per frame than robocasa's data, more than offsetting the higher frame rate.
      Measure; do not extrapolate from fps.

## ARM A — scene prediction (gpu 0, `head_right`)

- [x] **Staging parallelised** — decode fanned out across episodes (`_stage_job` + ProcessPoolExecutor,
      16 workers). Only PATHS cross the process boundary, so the fan-out is nearly free — the same
      property that made path-based frames the right call for the encode workers. **Measured:** 4
      episodes / 22,393 frames in **79 s** vs ~17 min serially (~13×). Full single-camera stage drops
      from ~4.5 h to ~20 min; Arm B's three cameras from ~13 h to ~1 h.
- [ ] Full dataset build at stride 6 — RUNNING (74 episodes, 341,494 frames).
- [ ] Launch `model=vl64_scene`.
- [ ] `_oneoff_action_sensitivity` early — near-zero sensitivity is quickdraw#15 (action not in the
      state's frame), NOT the model. Record the number either way.
- [ ] Watch `latent_cos` (negative = the vl64 collapse signature) and read actual frames, not just
      LPIPS (vl64 ghosts: "perceptually plausible, spatially wrong").

## P4 — multi-camera data path (~1 day, ARM B only)

- [ ] `processors.py:46` — `Episode.frames` → `dict[str, ndarray]` keyed by camera.
- [ ] `processors.py:87` — `build_recorded_dataset(…, cam)` → `cams: list[str]`; one video key each.
- [ ] `dataset.py:198` — `load_fpv_frames(cam=…)` per-camera, one `.npy` cache each.
- [ ] `dataset.py:254` — `load_split_episodes_with_frames` per-camera; count assert per camera.
- [ ] `dataset.py:285-310` — `MMWindowLoader`'s singular `self.image_head`/`self.frames` → one store
      and one batch key per camera. **The load-bearing change.**
- [ ] `conf/data/*.yaml` — `cam:` scalar → list.
- [ ] Backward compat: the single-camera path (Arm A, torus, robocasa) still works unchanged.

## P5 — multicam recipe (~2 h, ARM B only)

- [ ] `conf/model/vl64_multicam.yaml` — 3 image modalities: `scene_right` (`head_right`),
      `arm_top_left` (`gripper_left_top`), `arm_top_right` (`gripper_right_top`).
- [ ] `weight: 1/3` each — holds vl64's image:proprio ratio; 1.0 each would change head count AND
      loss balance together (the confound the vl64 header flags in its own results).
- [ ] `latent_loss_weight` — start at 10, watch `latent_cos`. Bracketed on both sides in vl64
      (0.4 erased 5.4 dB in one epoch; 25 froze motion, gradients → inf). It is a RATIO knob, so do
      NOT naively divide by 3.
- [ ] Memory check before launch: ~16.8 GB/camera at 128px, GPU-resident → ~50 GB of 96 GB for three.

## ARM B — multicamera (gpu 1, 3 heads)

- [ ] Launch `model=vl64_multicam`.
- [ ] Score on `scene_right` — the SAME camera Arm A predicts — so the comparison answers one clean
      question: do the wrist cameras improve scene prediction?

---

## Carried-in caveats (measured, not speculative)

- **`action` is not in the state's frame** — no Euler convention fits (best 126.6 deg median geodesic
  error; ~126 deg is what two uniform random rotations give), per-session extrinsics differ by
  26–176 deg, scale ranges 0–2114 where physics demands exactly 1000, grippers anti-correlated
  (−0.72 / −0.57). Action-conditioning is learning noise. Filed as quickdraw#15.
- **vl64 ghosts** — "perceptually plausible, spatially wrong"; best floor LPIPS while sitting 2.1 dB
  below runs with worse perceptual scores.
- **vl64's rollout regresses** vs up64 (OL PSNR 13.42 vs 14.86) and it collapsed at eval 12 after one
  non-finite step.
- **README stats are stale** on the HF dataset (says 65 episodes, metadata says 74) — dataset
  discussion #1.
