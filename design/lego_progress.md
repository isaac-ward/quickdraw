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
- [x] Full dataset build — `logs/recording_2026_08_31_06_30_02_lego`, 67 train / 7 val episodes,
      341,494 transitions, obs_dim 34, action_dim 20, 8.2 GB. Stage 508 s + encode 148 s + lerobot
      write 4369 s (that writer round-trips every frame through a PNG; it dominates and is serial per
      split, so it is a fixed ~85 min per camera).
- [x] **GPU shakedown** — 2 epochs on the 6-episode build. Proved the path runs and returned
      autobatch `batch=23` at 128px / 1 camera. Also flushed out that training REQUIRES a 5-point
      `run_summary`, and that hydra rejects bare commas/parens in override values.
- [x] **ARM A LAUNCHED** — `logs/train_world_model_2026_08_31_07_55_29_lego_arm_a_scene`, GPU 1,
      epoch 0 started 08:06:43. autobatch: budget 82.7 GB = card 102.0 − reserve 4.0 − **resident
      frame store 15.28** (predicted 16.8), fit 3.796 GB/sample → **batch 19**. 50 epochs, evals at
      {5, 9, 15, 19, 29, 39, 49}.
- [ ] Launch `model=vl64_scene`.
- [ ] `_oneoff_action_sensitivity` early — near-zero sensitivity is quickdraw#15 (action not in the
      state's frame), NOT the model. Record the number either way.

### 🐛 BUG FOUND AND FIXED: `subsample` was SUMMING an absolute action

Caught in a live run's log while Arm A was starting. `_subsample_episodes` combines the actions it
skips over and assumed they are DELTA-like: sum them, with an auto take-last for dims that look
binary (≤2 unique values). **Both halves are false here, and it fails silently.**

1. **The action is ABSOLUTE, not a delta** — a VR-controller pose, xyz in metres in a room frame
   (y spans [0.22, 2.69], centred on 1.47 = headset height). Summing six absolute poses gives six
   times the position. Measured: it does not correlate with `d(state_tcp)` at any lag 0–60.
2. **The binary guard never fires.** The gripper is at exactly 0 or 1 for **94%** of frames but has
   **~1000 unique values** because it ramps between them, so `≤2 unique` misses it. The live log
   showed `take-last on dims []` — an empty hold list.

Measured at stride 6 on a synthetic absolute ramp + a gripper with one ramp value:

| mode | dim0 | gripper |
|---|---|---|
| `sum` | [0.25, **5.75**] | [0.00, **6.00**] |
| `last` | [0.08, 1.00] | [0.00, 1.00] |

Fix: new `data.action_aggregate` = `sum` \| `last`, defaulting to `sum` so every existing dataset is
bit-identical; `last` set in `conf/data/lego.yaml`. It would not have crashed — it would have fed the
action encoder and normalizer a 6×-out-of-scale action and produced a plausible run.

### ⚠️ WALL-CLOCK: `subsample_all_phases` makes an epoch 6x bigger, and 50 of them is ~8 days

Arm A loaded **276,231 train windows** at batch 19 = **~14,540 steps/epoch**. Measured step rate is
around 1/s (heavy step: F=64 BPTT + 128px decode + LPIPS-VGG), so that is **~4 h/epoch** and the
first eval, at epoch 5, is **~20 h away**. 50 epochs is ~8 days.

The cause is not a mistake, it is a knob interaction worth naming: `subsample_all_phases: true`
multiplies train windows by `subsample` (6), which the record calls free "because more data means
fewer epochs for equal gradient steps" — but `max_epochs` was left at the stock 50, so the run is 6x
longer than that reasoning intends.

THE TRADE, at matched wall-clock (~33 h):
  * all_phases ON  -> ~8 epochs, ~276k DISTINCT windows seen ~8x. Better data breadth, but only
    ONE or TWO eval points (cadence is {5, 9, 15, 19, ...}).
  * all_phases OFF -> 50 epochs of ~46k windows, ~40 min/epoch, **7 eval points** and first signal
    at ~3.3 h. Same total window-presentations, repeated more.

Deliberately NOT changed unilaterally: `subsample_all_phases` and the eval cadence are both recorded
user decisions, and trading data breadth for eval frequency is a research call. Arm A is left running
as specified. If faster feedback matters more than breadth, restart with
`data.subsample_all_phases=false`.
- [ ] Watch `latent_cos` (negative = the vl64 collapse signature) and read actual frames, not just
      LPIPS (vl64 ghosts: "perceptually plausible, spatially wrong").

## P4 — multi-camera data path (ARM B only) — code DONE, loader untested on GPU

- [x] `Episode.frames` → accepts `dict[cam, paths|array]`; processor emits it for a camera list.
- [x] `build_recorded_dataset(cam: str | list)` — one encode job per (camera, split, episode) in a
      shared pool; `summary.json` records every camera.
- [x] `write_lerobot_split` — N video features, per-frame entry each, per-episode decode dropped
      before the next (3 cameras × a long episode is GBs).
- [x] `load_split_episodes_mm(cam=list, img_size=list)` → `(obs, act, img0, img1, img2)`.
      `_subsample_episodes` needed **no change** — it already decimates every stream past `ep[1]`.
- [x] `MMWindowLoader` — one resident uint8 store PER head, one batch key per head, all sharing ONE
      window index so streams cannot drift. `.frames` kept as an alias at exactly one head.
- [x] `ModalitySpec.cam` — each image head declares its camera directory. Raises if a head omits it
      with >1 head, and raises if two heads resolve to the same camera.
- [x] `_resident_frame_bytes()` needed no change — already sums over all image specs, so autobatch
      sizes 3 cameras correctly.
- [x] 3-camera build verified on 2 episodes → `logs/recording_2026_08_31_08_02_12_lego3`, all three
      `observation.images.*` keys written.
- [x] **Loader VERIFIED on GPU** — `lego_multicam_smoke` completed a full epoch with 3 heads:
      `train_loss=19.8193 val_loss=23.0602`, 798.5 s/ep, 207 train batches at autobatch batch 3.
      Three resident stores, three batch keys, three codecs, no drift. P4 is done.
- [x] Full 3-camera build — `logs/recording_2026_08_31_08_17_37_lego3full`, 24 GB, 67 train / 7 val,
      305,108 transitions, all three cameras at 288x512. 11,615 s, dominated by the lerobot PNG
      round-trip (3x one camera, as predicted).

## P5 — multicam recipe (ARM B only) — DONE

- [x] `conf/model/vl64_multicam.yaml` — 3 heads, each declaring its camera. `model_summary` clean:
      3 encoders (0.401M ea) + 3 decoders (4.764M ea), 50 tokens/step vs Arm A's 34, 23.26M params.
- [x] `weight: 0.3333` each — aggregate image:proprio ratio matches vl64's, so the run isolates
      "more views" as the single variable.
- [x] `latent_loss_weight` stays **10 per head**, deliberately NOT divided by 3 — it is a ratio knob
      and each head has its own codec to keep invertible, so the anchor is not a shared budget.
- [x] **Memory sized from Arm A's own autobatch, not guessed.** At 128px/1 camera: budget 82.7 GB
      after a 15.28 GB frame store, 3.796 GB/sample → batch 19, worst eval probe 33.3 GB. Tripling
      heads → frame store ~46 GB (budget ~52), ~10 GB/sample (batch ~4, below the ~8 where
      `distributed.md` says gradient noise bites), eval probe ~100 GB (**would not fit**). So three
      knobs move: `num_tokens` 32→16 (8→64 is a documented null, so per-head capacity is not what
      we're spending), `decode_chunk_train` 64→24, `visual_frames` 128→64 (3×64 still exceeds Arm A's
      128 in aggregate). **`img_size` stays 128** — 96 would buy headroom but make the arms
      incomparable, defeating the study.
- [x] Eval capped on the CLI: `eval.decode_chunk=16 eval.closed_loop_steps=[1]`. **Verified:** worst
      eval probe drops from 33.3 GB (1 camera, uncapped) to **4.0 GB** with 3 heads.
- [ ] **⚠️ MEASURED: 3 heads cost 35.0 GB/sample, not the ~10 I predicted — batch lands at 3.**
      Multicam smoke autobatch: b=2 → 28.3 GB, b=3 → 35.5 GB, b=4 → **98.3 GB** (over the 98 GB
      budget). Strongly superlinear, so it chose batch 3 and left 62 GB unused. Batch 3 is well below
      the ~8 `design/distributed.md` names as where gradient noise bites.

      **The fix is `decode_chunk_train`, and it is free.** `modalities.py:128` documents it as
      gradient checkpointing — "0 = OFF (bit-identical)", "~1.33x decode compute" — and says the
      decoder is "~78% of per-sample training memory across TWO passes... the one lever that buys
      real batch size; everything else lives in the other 22%". SMALLER chunk = more checkpointing =
      less memory, same result. Arm B currently sets 24; dropping toward 8 or 4 should recover batch
      into the 6–10 range at ~1.33x decode time. Applied as 24 -> 8 in the recipe; Arm B's own
      autobatch will report the result.

## ARM B — multicamera (gpu 0, 3 heads)

- [x] **LAUNCHED** — `logs/train_world_model_2026_08_31_11_32_38_lego_arm_b_multicam`, GPU 0, on the
      full 3-camera build `lego3full` with `decode_chunk_train: 8`, 50 epochs,
      `eval.decode_chunk=16 eval.closed_loop_steps=[1]`,
      `trainer.checkpoint_monitor=val/metric/scene_right/mse`.
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
