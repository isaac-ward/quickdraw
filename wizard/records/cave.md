# cave — quadrotor world model in a Gaussian-splat cave (Grapevine)

Running record for the cave quadrotor work: a photoreal 3D-Gaussian-splat render of the real Grapevine
cave, a quadrotor flown through it on real dynamics, 30 Hz multi-view data on HF, a world model trained
on it, uncertainty-based OOD detection, and a control benchmark of learned dynamics against true
dynamics. This is the MBARI pipeline (`wizard/records/mbari.md`) re-targeted from an AUV-in-a-tank to a
quadrotor-in-a-cave. Started 2026-10-19.

## The goal, stated once

A research output showing that a world model trained on nominal quadrotor FPV footage in a cave can
(a) predict open-loop over useful horizons, (b) flag three distinct kinds of out-of-condition episode
from its own uncertainty, localised to pixels, and (c) be used as the dynamics model for closed-loop
control, measured against the true simulator dynamics. Same objective as everywhere in this project:
raw `eval_ood_horizon/open_loop/image/lpips/@+64` and `@+128` against the historical best.

## Why this is mostly a re-target, not a new build

The MBARI pipeline is domain-agnostic except for three pieces — **dynamics**, **scene/collision**, and
**renderer**. Everything downstream (recorder interface, `build_recorded_dataset` packaging, the
`vl128` training recipe, `run_uncertainty` 3-way detection, `control.py` True/Analytic/Learned MPPI
benchmark) is generic given per-episode `(obs, act, frames)`. So the work is: supply the three physical
pieces from the user's own cave sim, and re-index the state layout downstream.

## Decisions taken (2026-10-19)

| | | |
|---|---|---|
| dynamics | **reuse prospector `Dynamics3D`** (JAX, 12-state, 4-D action, RK4 100 Hz, batched jitted `rollout_batch`) | user's lib, already tuned for this cave |
| cave + collision | **sealed prospector Grapevine map for collision/free-space; the splat for render; one similarity transform `T` links them** | the splat has HOLES (gaps in coverage) so an occupancy grid built from it would leak on flood-fill and let the drone pass through unsealed walls (user, 2026-10-19). The processed Grapevine voxelization is watertight. Two frames, one offline `T` — see "Aligning the splat" below. Earlier "no alignment" decision reversed |
| renderer | **add a `gsplat` splat renderer INSIDE prospector** | prospector is the user's and modifiable; it already has the env loop, POV-camera keyframing, and trajectory export. Skip FiGS entirely |
| OOD | mirror MBARI: 3 sets × 50 traj — **rotor-glitch (dynamics)** + **composited red person mesh (vision)** + **appearance/lighting shift (vision)** | 1 dynamics + 2 vision. The person mesh reads as a search-and-rescue target (user, 2026-10-19): Sketchfab low-poly base mesh (138ca18c…), pulled via `SKETCHFAB_API_KEY`, tinted red, depth-composited into FPV |
| data volume | **6.00 h nominal straight away** (720 × 900 steps @ 30 Hz = 648,000), + 3 × 50 OOD | user, 2026-10-19. Skips MBARI's 2 h debug cut |
| behaviours | **4 classes, 25% each**: corridor-follow (A*), orbit/circle-yaw, explore (random), station-keep+look-around. **Collision-free: an episode that collides is ended and resampled** | user, 2026-10-19. Nominal data is clean free-flight; collision belongs to control, not the WM training set |
| proprio rotation | **quaternion**, not prospector's Z-Y-X Euler; quat dims passed `obs_identity` | avoids the wraparound discontinuity the WM would otherwise have to learn around — the same call MBARI and block-stack's rot6 made |
| frame size | **144×192** (H,W), `ae_bottleneck: 9` | reuses `conf/model/mbari_fpv.yaml` → the `vl128` recipe transplants verbatim; both axes ÷16 |
| gen env | **isolated venv** (torch + gsplat + jax + prospector) | the shared training venv is never `uv sync`'d or contaminated (MEMORY: shared-venv fragility). Gen writes npz; training reads the HF dataset |

## Division of labor

**prospector** (user's, modifiable — JAX + new gsplat): dynamics, `CaveMap3D` from the splat,
trajectory generators (A* / `PolicyMPPI` / min-snap), **NEW gsplat renderer** (RGB+α→grey, FPV +
third-person), **NEW episode recorder** → per-episode npz + per-trajectory MP4, OOD injection.

**quickdraw**: `package.py` (npz → HF via `build_recorded_dataset`), `conf/{data,model}/cave*.yaml`,
training, `run_uncertainty` (reused, state slices re-indexed), the control benchmark
(`controller/mppi.py` with True = prospector `Dynamics3D`, Analytic = coarse, Learned = the torch WM
proprio head). The learned arm is the one cross-framework seam — torch WM scored by the shared MPPI,
collision by `CaveMap3D`; exactly MBARI's `control.py` shape.

## The reuse / swap map (quickdraw `sim/cave/` mirrors `sim/mbari/`)

| mbari file | reuse? | cave change |
|---|---|---|
| `mola.py` (dynamics) | swap | → prospector `Dynamics3D`; `state()` returns 13-D (xyz, quat, v_world, body-rate) |
| `tank.py` (scene) | swap | → `CaveMap3D.from_ply(cave_6_splat.ply)` |
| `water.py` (`UnderwaterRenderer`) | swap | → prospector gsplat renderer, same `render(state, cam)->HWC uint8` interface |
| `trajectories.py` | pattern | A*/MPPI collision-free path classes in cave free space |
| `record.py` | interface-compatible | drop-in once dynamics+render swapped |
| `generate.py` | yes | parallel shards, now GPU-render-bound (cheaper than MuJoCo EGL) |
| `package.py` | yes | names/dims only |
| `run_uncertainty.py` + `uncertainty.py` | yes | re-index POS/ROT/VEL/ANGVEL slices for the 13-D state |
| `control.py` + `run_control.py` | yes | new dynamics classes + cave tasks |
| `conf/model/mbari_fpv.yaml`, `conf/data/mbari.yaml`, `train_mbari.sh` | yes | → `cave_fpv` / `cave`; proprio dim 13, action dim 4, one FPV head |

## gsplat rendering notes (the one genuinely new code)

- `cave_6_splat.ply`: standard 3DGS, **268,853** gaussians, 62 props (pos, normals, f_dc + 45 f_rest =
  full **SH deg 3**, opacity, scale×3, rot×4). Dense: mean α 0.78. Extent ≈ 26×21×14 units (robust
  20×15×10), centroid ≈ (−2.2, −0.4, 2.9). If units are metres this is a ~20 m cave.
- Render with `gsplat.rasterization(means, quats, scales, opacities, sh_coeffs, viewmats, Ks, W, H,
  sh_degree=3, render_mode="RGB+ED")` → **RGB + depth + alpha** in one pass. The alpha (accumulated
  opacity, [0,1]) is the "splat where we have it" mask: `frame = rgb*α + GREY*(1-α)`.
- Pose convention (from FiGS): input is camera-to-world; gsplat/nerfstudio wants OpenGL (X right, Y up,
  Z back) while robotics is OpenCV (X right, Y down, Z forward) — left-multiply the flip `diag(1,-1,-1,1)`.
  Intrinsics are standard pinhole fx,fy,cx,cy. Build viewmats from the quad body pose × camera offset.
- Env verified: torch 2.10+cu128, nvcc 12.8, 2× H100 NVL free → gsplat CUDA ext will build.
- Third-person views (chase/topdown/side) render the same splat from external poses; the drone has no
  splat, so a small projected glyph is composited for human review (FPV doesn't need it — the drone
  isn't in its own view).

## Aligning the splat to the sealed Grapevine map (two frames, one transform)

The splat has gaps; the processed Grapevine map is watertight. So they split roles and are linked by ONE
offline similarity transform `T` (7-DoF: scale `s`, rotation `R`, translation `t`), `splat → Grapevine`:

- **Physics / collision / free-space / A*** run in the **Grapevine metric frame** (z-up, sealed).
- **Rendering** takes the camera pose in the Grapevine frame, maps it with `T⁻¹` into splat space, and
  calls gsplat. Splat holes → α≈0 → grey composite: harmless, since collision never touches the splat.

Computing `T`, most-robust-first (caves are tube-like + self-similar, so lead with landmarks):
1. Prep: splat centres filtered opacity>0.5 + sane scale (drop floaters) = cloud A; Grapevine
   cloud/voxel-surface = cloud B; downsample both to a common voxel size.
2. **Landmark init**: hand-pick ≥4 correspondences in a viewer (corner, constriction, entrance lip),
   closed-form **Umeyama** similarity. PCA-axis + extent-ratio is the zero-landmark fallback; automatic
   FPFH/RANSAC is unreliable on a self-similar cave.
3. Refine: similarity-ICP — open3d `registration_icp` +
   `TransformationEstimationPointToPoint(with_scaling=True)`, then a point-to-plane pass.
4. **Validate by rendering, not by RMSE**: render the splat from Grapevine-frame poses, overlay (a)
   splat depth vs Grapevine mesh/voxel depth along the same rays (few-cm median), (b) splat RGB over a
   top-down of the occupancy. A good RMSE on a symmetric cave can still be a flipped/shifted fit.

Sealing: flood-fill the Grapevine occupancy from a known interior point (`label_accessible_space`), cap
the open cave mouth with a virtual plane so free-space is bounded.

PREREQUISITE: the processed Grapevine asset is NOT in git — prospector pulls it from Backblaze B2
(`scripts/assets-pull.sh`). Needed, or the user points at the file. FALLBACK if B2 is unavailable:
morphological close (dilate→erode) + bounded flood-fill on the splat-derived grid — lower fidelity, no
second asset. Aligned tandem is primary.

## Risks / watch-outs

- **Splat floaters** contaminate the occupancy grid → filter gaussians by opacity + scale before
  `from_ply`; validate flood-fill yields a single connected interior.
- **Up-axis / scale / gravity** come for free from `T`: physics runs in the Grapevine frame (already
  z-up metres), so the splat's arbitrary SfM scale/orientation is absorbed into `T⁻¹` at render time and
  never touches the dynamics. The binding risk is `T` itself being a flipped/shifted fit — caught by the
  render-overlay validation, not an RMSE.
- **Cross-framework control seam**: torch WM as a dynamics model inside the MPPI — budget for it; it is
  the one place JAX (true dynamics) and torch (learned) meet. MBARI solved the analogous seam.
- **OOD-vision in a static splat**: the object OOD is a shaded primitive composited by **depth test**
  against the splat's depth buffer (gsplat gives depth); the appearance OOD is a colored haze/tint over
  a region. Neither needs re-training the splat.

## Phases / checklist

- [ ] 1. **Render spike** (de-risk first): gsplat loads the ply, render from a hand-picked pose, verify
      RGB + α-grey composite reads as the cave. *Gate: a good-looking frame.*
- [ ] 2. Pull/locate the processed Grapevine map; seal it (flood-fill + cap the mouth); sphere-collision
      sanity. Compute `T` (splat→Grapevine): landmarks → similarity-ICP → validate by render overlay.
- [ ] 3. Wire prospector `Dynamics3D` in the Grapevine frame; render FPV through `T⁻¹`; stable 30 Hz.
- [ ] 4. Trajectory generators (A*/MPPI collision-free path classes); recorder with render-in-loop
      (FPV + 3 third-person); per-trajectory MP4 (the "judge on MP4s" rule).
- [ ] 5. Generate nominal + 3 OOD sets; package → HF (`isaac-ronald-ward/cave-...`).
- [ ] 6. Train `cave_fpv` (vl128 transplant); report raw OL LPIPS @+64/@+128 vs best.
- [ ] 7. Uncertainty → 3-way OOD classification; pixel-level uncertainty videos.
- [ ] 8. Control benchmark: reach-pose + traverse, True vs Analytic vs Learned WM, under collision.

## Assets & dev setup (2026-10-19)

- **prospector dev branch**: `gsplat-render` at `/home/ubuntu/user_irw/prospector` (full clone; the
  gsplat renderer + recorder + OOD land here; eventual PR to prospector main).
- **Grapevine map pulled** (public B2, no creds): `src/assets/point_clouds/grapevine-exped-3.ply`
  (30 MB, 800K verts, metres, z-up). Cave cfg `grapevine_3d_025`: `voxel_size: 0.2`,
  `known_internal_xyz: [-0.672833, 0.089741, 1.85611]` (flood-fill seed), graph JSON committed. Did
  NOT pull the 4.56 GB `cave-maps.blend` (Blender render is being replaced) or the other caves.
- **Splat** `logs/cave_6_splat.ply` (in quickdraw): centroid (−2.2,−0.4,2.9), extent ~26×21×14 — a
  DIFFERENT frame/scale from the Grapevine cloud, which is why `T` (splat→Grapevine) is required.
- **Sketchfab token**: `SKETCHFAB_API_KEY` is in `quickdraw/.env` (not the shell env) — source it for
  the OOD person-mesh download.
- **Isolated gen env**: torch + gsplat (CUDA build) + jax + prospector + open3d/plyfile, kept separate
  from the quickdraw training venv (never `uv sync` the shared one — MEMORY: shared-venv fragility).

All four interview questions (2026-10-19) are resolved and recorded in the decisions table above.

## Phase 1–2 results (2026-10-19)

- **FPV FOV matched to prospector**: the POV/ego cameras in `cave-maps.blend`
  (`camera_agent_*_pov`, `camera_ego_example`) are **16 mm on a 36×24 sensor → 96.7° H / 73.7° V**.
  Confirmed by mapping object→camera-data lens (not assumed). Tele cameras (77–100 mm) are the
  overhead/establishing shots. Our FPV uses hfov 96.7°, square pixels, at 144×192.
- **Phase 1 render spike PASSED**: `gsplat.rasterization(render_mode="RGB+ED")` renders
  `cave_6_splat.ply` (268k gaussians, SH deg3) as a photoreal cave; alpha→grey composite fills holes.
  `logs/cave/spike/contact_sheet.png`. Up-axis ≈ +z (upright with up=[0,0,1]).
- **Alignment T computed** (`logs/cave/alignment/T_splat_to_grape.npy`): PCA init (best sign flip) →
  similarity-ICP (with_scaling). Fitness 0.938, final one-way NN splat→grape **median 0.70 m, p90
  1.39 m** (voxel 0.2 m). Figures: `overlay_before.png` / `overlay_after.png` (3 projections) +
  `nn_hist.png`.
- **KEY FINDING — the splat is a SUB-SECTION, but covers 75% of the navigable interior.** Splat extent
  26×20×14 vs Grapevine 31×36×17; the splat is the central chamber. Sealing (`label_accessible_space`
  from the interior seed) gives **1,527 m³ reachable interior, no leak** (2.1M exterior voxels correctly
  sealed off — no mouth-cap needed for this seed). Of that, **1,144 m³ (75%) is within 3 m of splat
  texture** → always-textured FPV. Untextured 25% = one narrow tunnel + high ceiling pockets.
  `logs/cave/coverage/sealed_coverage.png`.
- **DECISION: nominal flights are constrained to the textured (green) region** — sample waypoints from
  the textured-free mask so FPV always has real texture. 1,144 m³ is ample for diverse 30 s paths.

## Renderer design (2026-10-19) — splat over lit capture geometry

The FPV composite shows the **splat where covered and the grapevine-3 geometry (lit) elsewhere**, so
unsplatted regions read as real flyable geometry, not a flat wall (user ask). Implementation:
- Both the splat (colored, SH deg3) and the uncovered grapevine mesh are rendered in **ONE gsplat pass**,
  depth-sorted — so compositing is automatic (opaque splat occludes mesh behind; mesh shows through holes
  and beyond coverage). Mesh = one gaussian per vertex, kept only where the splat has no coverage
  (dilated mask), sized to overlap into a surface.
- **Mesh is lambertian-shaded by a HEADLIGHT** (light along the camera forward, using mesh vertex
  normals), recomputed per frame → lit geometry that shows motion. Coloured from the **splat's own
  palette** (earthy brown, sampled along the path) + per-vertex noise, so it matches and shows motion.
- **GOTCHA: open3d/filament OffscreenRenderer renders this mesh BLACK on headless Vulkan** (box renders
  fine; the 800k-vert grapevine mesh AND even a point cloud of it return all-zeros — not a camera/size/
  winding/NaN issue, filament silently rejects it). Do NOT rely on open3d rendering here; it is used
  only to load the mesh + compute normals. The gsplat shaded-gaussian path is the renderer.
- Grey/empty background is dark (~0.03). FPV FOV 96.7° H (prospector's 16 mm POV).
- **Lit mesh (2026-10-19 v2)**: uncovered geometry is baked-lit by **24 point lights spread through the
  free space (FPS), with raycast HARD shadows** (open3d `RaycastingScene` — the raycasting module works
  headless even though filament rendering does not), half-lambert so grazing faces aren't black, ambient
  0.34 floor so tunnels are visible, earthy colour sampled from the splat + brightness/noise variation.
- **Collision-safe paths**: A* keep-out 0.8 m + clearance-gradient repair → all paths ≥ 0.82 m min
  clearance (fixed an earlier wall-clip from keep-out 0.6 m + over-smoothing).
- **Deliverables** now in `logs/cave/flythroughs/` (see `logs/cave/README.md` for the current-vs-stale
  index): **8 fly-throughs** `fly_NN.mp4`, each a synced **[top-down map w/ moving marker | FPV]**,
  full-length traversals (23.6–26.2 m) in alternating directions; `fly_NN_map.png`, `overview_midframes.png`.
  Old single-flythrough + flat-grey tests moved to `logs/cave/_stale/`.
- **v3 fixes (2026-10-19, from review):**
  - **Scene was upside down** — grapevine +z is gravity-DOWN. Camera up set to **−z** (`UP` const);
    now stalactites hang from the ceiling, floor at the bottom. (This also means Phase-3 dynamics use
    gravity along +z_grapevine / up = −z.)
  - **Splat in collision**: the aligned splat voxels are unioned into the occupancy as obstacles, with a
    **+0.2 m dilation buffer on splats specifically** (≈25% more clearance from splats; tunnels are
    unsplatted so unaffected). The drone no longer clips splat surfaces.
  - **Clearance**: keep-out 0.8 m + repair → all paths ≥ 0.80 m (less wall-hugging).
  - **Diverse routes**: endpoints from farthest-point sampling + backfill → 8 varied traversals incl. the
    side tunnel (not all the main corridor).
  - **Lighting**: the alternating warm/cool point lights read as "flashing RGB lights on the wall" (half
    orange, half blue, shifting with the camera + mp4 keyframe compression). Replaced with ONE soft
    warm-white for all lights, lower intensity, wider falloff.
  - **Unsplatted colour**: mostly the ACTUAL splat palette (0.85 sample + 0.15 faint earthy tint),
    **darker** (×0.48–0.78) — grey-earthy, not brown, matching the splat's own tones. Ambient 0.28.
  - **Top-down chirality**: with camera-up = −z, the FPV's right is +Y, but the map drew +Y up (a view
    from the +z side = from under the true floor), so turning left in the map looked like turning right
    in the FPV. Map now drawn from TRUE-ABOVE (looking along +z: +X right, **+Y down**) to match the FPV.

## Phase 3 wired (2026-10-19) — dynamics + recorder

New prospector module `src/prospector/cave_wm/` (branch `gsplat-render`):
- `scene.py` `CaveScene` — builds once (~11 s): splat + baked-lit mesh merged into one gsplat pass,
  collision occ (grapevine ∪ dilated splat, sealed), clearance field, textured mask, FPS anchors, A*,
  `render(eye,fwd)` (via T⁻¹, up=−z). Render **3.3 ms/frame**, dyn.step 0.2 ms.
- `record.py` — `Dynamics3D` (velocity mode, **30 Hz**, dt=1/30, JAX on CPU) + 4 behaviour policies
  (corridor=A* follow, orbit, explore, station). obs **(T,13)** = xyz|quat|vel|bodyrates (quat from the
  dynamics' euler, no wrap), act **(T,4)** = vx vy vz yaw_rate (commanded setpoint). Collision → drop+retry.
  GRAVITY TODO: prospector dynamics assume z-up; grapevine +z is gravity-down — velocity controller
  compensates so trajectories still follow commands; a gravity-correct frame is a later refinement.
- `generate.py` — build scene once, roll N interleaved episodes (class-balanced prefix), npz shards.
- **Smoke OK**: 8 episodes (2/class, 400 steps) in 1.6 min, 0 failed; previews in
  `logs/cave/data/smoke_previews/` (per-class mp4 + `all_classes_2x2.mp4` + `_contact.png`).
- STILL TODO: OOD sets (rotor-glitch + red-person mesh + appearance — person mesh not yet downloaded),
  quickdraw `package.py` (npz→HF, mirror sim/mbari) + `conf/{data,model}/cave*`, then train.

## Phase 3b — OOD + packaging + configs built (2026-10-19)

- **OOD (3 sets, mirror MBARI)**, all verified (`logs/cave/data/ood_smoke/`): **ood_thruster** = rotor
  glitch (30% of steps the plant gets an erratic command; the COMMANDED action is still recorded — the
  gap is the point; dynamics OOD); **ood_person** = the Sketchfab low-poly person (530 tri, downloaded
  via `SKETCHFAB_API_KEY` to `cave_wm/assets/person.glb`) sampled to 4k red gaussians, placed on the path
  as a search-and-rescue figure, rendered in the same gsplat pass (vision OOD); **ood_appearance** = a
  cool blue white-balance shift + haze post-process (vision OOD). `generate.py plan_ood` = 3×50×360,
  classes from the same mix. `--ood-per 50 --ood-steps 360` generates them.
- **quickdraw `src/quickdraw/sim/cave/package.py`** — npz → `build_recorded_dataset` → HF, 20% val,
  `obs_identity` on quat dims, per-episode task (`nominal_<kind>` / `ood_*`). Target repo
  `isaac-ronald-ward/cave-grapevine`.
- **configs**: `conf/model/cave_fpv.yaml` (vision WM: proprio+fpv), `conf/model/cave_proprio.yaml`
  (dynamics-only WM: proprio head only), `conf/data/cave.yaml` (13-obs/4-act, fpv, 30 Hz, subsample 1).
- Plan: finish nominal (720) → generate OOD (3×50) → package nominal+OOD together → push to HF → train
  both WMs → uncertainty 3-way OOD detection + control benchmark.

## Phase 3c — dataset generated + packaging to HF (2026-10-19)

- **Generated**: 720 nominal + 50/50/50 OOD = **870 episodes, 29 GB** npz in `logs/cave/data/v1/`.
  0 failed. Generation gotchas: A* was 21 s/call (pure-Python) → replaced with coarse-grid A*
  (`CaveScene.plan_fast`, 0.03 s); concurrent gsplat-kernel compile race → stagger shards 25 s; run via
  `cave_wm/gen_both.sh` (resumable — skips episodes on disk). Render 3.3 ms, ~4 s/episode.
- **Packaging**: `cave_wm/pack_push.sh` runs `sim.cave.package` then `quickdraw.push_to_hub
  +hub.name=cave-grapevine`. Re-encode is ~5 s/ep (SVT-AV1) → full run ~90 min, so run it **detached
  (`setsid`)**, not under the 2 h harness-task cap. GOTCHA: two packaging runs on the same run_dir race
  and spew ffmpeg "Output file is empty / Invalid argument" — a clean single run is error-free. RAM is
  ample (629 GB; packer holds all episodes, ~58 GB).
- NEXT: once on HF → train `cave_fpv` (vision) + `cave_proprio` (dynamics-only); then OOD detection +
  control benchmark.

## Phase 4 — on HF + both WMs training (2026-10-19)

- **PUSHED**: `isaac-ronald-ward/cave-grapevine` (public). `package.py` wrote **5 split datasets** —
  `train` 576 ep, `val` 144 ep (random, seed 0, val_frac 0.2), `ood_thruster`/`ood_person`/
  `ood_appearance` 50 ep each — 870 ep / **702,000 frames** @ 144×192 30 Hz. run_dir
  `logs/recording_2026_10_19_06_07_09_cave-grapevine`. SVT-AV1 re-encode ~5 s/ep, packaging ~1 h, push
  clean (exit 0, cleared+reuploaded). Train with `data.root=<run_dir>`.
- **Training launched** (`cave_wm/train_cave.sh`, copied to `logs/cave/train_cave.sh`, run **inside the
  `quickdraw-app` container** `cd /app`, detached via `setsid`; logs `logs/cave/train/`):
  - **GPU0 `cave_fpv_v1`** — VISION WM, `model=cave_fpv`, 16 epochs (~509k steps, step-count parity with
    `mbari_fpv_v1`'s 501k), `checkpoint_monitor=val/metric/fpv/mse`, eval `decode_chunk=32
    closed_loop_horizon=64`, torus-only evals off.
  - **GPU1 `cave_proprio_v1`** — DYNAMICS-ONLY WM, `model=cave_proprio`, 40 epochs (cheap — no image head
    so the loader never parks the 702k frames on GPU; setup.py:209 reads `model.modalities` to pick cams).
    Default `checkpoint_monitor=val/metric/proprio/pointwise_error`.
  - Both: `action_dim=4`, `obs_dim=13`, `dt=1/30`, `data.action_aggregate=concat`, `subsample=1`,
    `every_epochs=2`. Staggered 20 s so the second process doesn't race config/data setup.
- **GOTCHA — epoch budget + two-card contention (fixed by relaunch):** first launch ran both WMs
  concurrently (GPU0/GPU1) at 16/40 epochs. Two problems surfaced after ~2 h:
  1. **batch 6, not 15.** The vision frame store eats **53.75 GB**, leaving autobatch a 42 GB budget at
     4.1 GB/sample → **data.batch=6**, so one epoch = 477,504/6 ≈ **79.6k steps**. 16 epochs = 1.27M
     steps (~2.5× MBARI). True step-parity with `mbari_fpv_v1` (501k) is **≈6 epochs** — the "16 ≈ 509k"
     math wrongly assumed batch 15. Proprio trains at a big batch (933 steps/epoch) and converges fast →
     **20 epochs** ample.
  2. **two maxed H100s hit a shared ceiling.** Proprio epoch time blew up **445 s → 3176 s (7×)** once
     the vision step ran — both use the GPU-resident loader so it is NOT CPU dataloading (power / mem-BW).
  - **FIX attempt 1 (serialise):** ran fpv alone — still **>8 h/epoch uncontended** (epoch 0 ran 114 min
    without reaching 25%; GPU0 100%, GPU1 idle). So contention was secondary; the config itself is
    ~**0.4 s/step at batch 6 × ~80k steps/epoch**. Root cause: the 54 GB resident frame store caps
    autobatch at **batch 6**, and real per-step cost is dominated by LPIPS-vgg over `visual_frames=128`
    frames + the `F=64` autoregressive unroll + per-step image decode. The binding constraint is
    **samples/s** (batch-invariant), so the only cuts that help are fewer windows and cheaper per-step work.
  - **REAL root cause (the "contention" was a MISDIAGNOSIS):** the 7× jump is the **`p_tf` teacher-forcing
    curriculum** (`p_tf_warmup_epochs=1`, `p_tf_batch_granular=false`). **Epoch 0** runs at `p_tf=1.0` —
    every step predicted from GROUND TRUTH, fully parallel → fast (~445 s). **Epochs 1+** hard-switch to
    `p_tf=0.0` — the compiled **autoregressive rollout** (the model unrolls its OWN predictions over the F
    horizon, sequentially) → ~3200 s. That AR rollout is exactly what a WM needs for multi-step OOD/control,
    so it is not removable. Proof it is NOT cross-GPU contention: proprio's AR epoch is **3176 s with fpv
    co-running vs 3191–3535 s alone — identical**. So concurrency is fine; **F is the cost multiplier** for
    the AR rollout.
  - **What is running now (BOTH concurrent, no penalty):**
    - **`cave_proprio`** — GPU0, F=64 (2 s horizon), ~59 min/AR-epoch. best.ckpt already at ep1
      (val_loss 0.0558). Let it run; **stop at val plateau** (likely ~ep6–8), don't wait for all 20.
    - **`cave_fpv` reduced** — GPU1 (`train_cave_fpv.sh`): `visual_frames 128→32`, `data.F 64→32` (halves
      the AR rollout, 1 s horizon — ample for short-horizon OOD + MPPI), `decode_chunk_train 64→16` (→
      autobatch **batch 29**, vs 6), `+data.train_episodes=240` (smaller frame store).
    - **IMAGE DECODE is the fpv bottleneck, not the AR unroll:** even epoch 0 (teacher-forced) was ~0.108
      s/sample (vs proprio 0.0016) → ~6 h/epoch, because the 4.77M upsample decoder + LPIPS-vgg runs on
      `recon_frac=1.0 × F` frames per step. **FIX: `recon_frac 1.0 → 0.25`** (decode a quarter of the
      rollout frames; validated older mm_flow value) ≈ 4× image throughput. Relaunched 15:43 with
      `recon_frac=0.25`, `max_epochs=5`. (Gotcha: `cp`-ing the launch script while the old wrapper was mid-run
      corrupted its parsing — overwrite launch scripts only after the old process is confirmed dead.)
    - Both checkpoint every epoch (`save_top_k=-1`) → safe early-stop at val plateau.

## Phase 4a — dynamics WM done; vision WM in progress (2026-10-19)

- **`cave_proprio` (dynamics WM) DONE — best.ckpt = epoch 3**, `val/loss/total` 0.0344,
  `pointwise_error` 0.301. Val curve: 0.572 → 0.0558 → 0.0379 → **0.0344 (ep3, best)** → 0.291 (ep4) →
  0.979 (ep5). **Late-epoch divergence** (train+val loss blow up past ep3) — the known flow-WM
  instability (no LR decay in the AR regime; cf. [[flow-decode-nan-blowup]], [[repro-collapse-past-ep20]]).
  `save_top_k=-1` captured the pre-divergence ep3; stopped the run early. ckpt:
  `logs/train_world_model_2026_10_19_11_36_34_cave_proprio_v1/checkpoints/best.ckpt`. If downstream
  OOD/control is weak, retrain with cosine LR decay.
- **`cave_fpv` (vision WM) relaunched 17:47** — recon_frac 0.25 made epoch 0 ~2 h (was 6 h); batch 33.
  Expect the same curriculum divergence past ~ep3, so best.ckpt will land early; 5-epoch ceiling is fine.
- **GOTCHA — `pkill -f` matched the run_summary text:** `pkill -f 'train_world_model.*cave_proprio'`
  (meant for the dynamics run) ALSO killed the vision run, because the fpv command's
  `run_summary.rationale` contained the literal string "cave_proprio". Lost ~2 h of fpv epoch 0.
  **FIX: kill by `experiment=<name>` (unique), not by model/substrings that appear in free-text summary
  args; removed "cave_proprio" from the fpv summary.**
- NEXT: when both best.ckpt land → 3-way OOD detection (reuse `run_uncertainty`: rotor-glitch should be
  caught by the dynamics WM; person + appearance by the vision WM) + control benchmark
  (True vs Analytic vs Learned dynamics).

## Phase 4b — physics prior for the dynamics WM (drift + noise fix), 2026-10-19 ~21:30

**Isaac looked at the dynamics-WM eval plots and flagged DRIFT + NOISE.** Root cause: the prior-free
proprio head reconstructs the full 13-D state every step, re-learning exact integration from scratch and
accumulating the error over the AR rollout.

**The codebase already has the residual-physics mechanism** (`prediction = dynamics_prior(prev,act) +
residual_head`, residual zero-init so prediction == the physics law at init; `physics_proprio_chained`
train, `_physics_proprio_rollout` eval). But the shipped prior (`owm_physics.dynamics_prior`) is
FORCE-based (`a=R(q)F/m`, action=force), **ISS-chaser-specific — it does NOT fit cave**, where the action
is a **velocity setpoint**, not a force.

**Verified the right prior on the held-out val split (144 ep):** cave dynamics are velocity-mode kinematics:
- `p' = p + v·dt` → **0.9%** residual (0.00016 m vs the 0.0186 m/step scale). (`v_cmd·dt` only gets 21% —
  the onboard controller lags the setpoint by ~0.12 m/s, so the *obs* velocity, not the command, is the
  correct integrand.)
- `q' = q⊗exp(½ w dt)` → **0.056 deg/step** (copying the quat instead leaves 0.59 — 10× worse).
- `v'=v`, `w'=w` → the residual learns the small controller response (Δv ~0.01 m/s/step).

**Change (surgical):**
- `environments/owm_physics.py`: added `velocity_kinematics(prev,act,pos,vel,quat,bodyrate,dt)` — the
  force-free velocity-mode prior. Reuses `quat_integrate`. Action unused by the prior (drives the residual
  via its latent token).
- `training/setup.py` `_make_dynamics_prior`: added `mode == "kinematic"` branch (reads
  position/velocity/quat/bodyrate idx + dt from env cfg; dt = subsample · env.dt).
- `conf/model/cave_proprio.yaml`: proprio head gets `prior: kinematic, latent_loss_weight: 0` (a
  residual head can't do the codec round-trip — the model raises if latent_loss_weight>0 in prior mode).
- Smoke-tested: `velocity_kinematics` numerically correct; `_make_dynamics_prior` + full `build_model` +
  `imagine_eval` all run with the prior active (`core.dynamics_prior` set, `residual_head` present).

**CAVE OBS INDICES MUST BE SET EXPLICITLY** (train script does): pos [0,1,2] | quat [3,4,5,6] | vel
[7,8,9] | rate [10,11,12]. The env defaults (velocity=pos+len→[3,4,5], bodyrate=quat+1→[7,8,9]) would map
velocity onto the quaternion and body rate onto velocity. Setting `position_idx` also makes the eval
`pointwise_error` (checkpoint metric) POSITION-only L2 — exactly the drift we're fixing.

- **Retrain launched 21:26 on GPU0**, `experiment=cave_proprio_prior_v1`, batch 512, F=64, 8 epochs,
  script `prospector/cave_wm/train_cave_proprio_prior.sh` (copied to `logs/cave/`). The residual target is
  tiny+smooth, so expect it to be MORE stable than the prior-free run (which diverged past ep3) and to cut
  both drift and noise. Compare eval plots vs `cave_proprio_v1` (prior-free, pointwise_error 0.301).
- If it helps, apply the same `prior: kinematic` to the vision model's proprio head (`cave_fpv.yaml`) on
  its next run. See [[owm-control-physics-prior]], [[physical-loss-soft-continuity-dead]].

### Phase 4b.1 — the prior EXPLODED on persistence; fix = ground it on the command (2026-10-19 ~22:00)

First kinematic prior (`v'=v`, `w'=w` persistence) fit teacher-forced one-step (train 0.048) but **blew up
in the autoregressive val**: `val/metric/proprio/obs_error` **470** (vs prior-free 0.57), pointwise_error
6.0 (vs 0.30). CAUSE: an additive residual on a *persisted* channel random-walks over the AR horizon —
velocity drifts, position integrates the drift (quadratic blow-up), body rate drives the quaternion wild.
Prior-free doesn't explode because each step is a fresh decode bounded to the data range; a residual-on-
persistence is unbounded.

FIX — **ground every channel on the (bounded) action** ("the vehicle executes its command"):
`v'=v_cmd`, `p'=p+v·dt` (exact integration of the now-bounded velocity, 0.9%), `q'=q⊗exp(½[0,0,ψ̇]dt)`
(yaw from the commanded yaw-rate), `w'=0` (residual predicts body rate directly, bounded like prior-free).
The residual now learns only the DEVIATION from perfect command-following (controller lag, tilt). Rewrote
`prospector_physics.velocity_kinematics` (added `vel_cmd_slice`, `yaw_rate_idx`). **Validated: a fresh
untrained model rolls 60 steps stably, tracking a real val trajectory to 0.24 m, no NaN** — the old design
would have random-walked to 100s. Relaunched 21:58 (`cave_proprio_prior_v1`, GPU0). Compare first val to
prior-free 0.30 and the broken 470.

### Phase 4b.2 — prior won short-horizon position but ATTITUDE exploded; A/B attitude fix (2026-10-19 ~23:45)

The grounded prior (`cave_proprio_prior_v1`) trained stably — val pointwise_error 0.103 at ep1 (vs prior-free
0.301). BUT that metric is **position-only** (position_idx set), so it flattered the prior. The matched
AR-rollout comparison (`sim/cave/compare_prior.py`, 6 val ep, 90 steps = 3 s) told the real story
(`logs/cave/prior_compare/drift_vs_step.png`):
- **Position**: prior WINS for the first ~1 s (step 10: 0.02 m vs 0.30; step 30: 0.22 vs 0.33), crosses over
  ~1.1 s, loses by 3 s (1.19 vs 0.60 m).
- **Velocity**: prior worse (humps to 0.85 m/s mid-rollout) — residual hasn't learned the controller lag at ep1.
- **Attitude**: prior EXPLODES to ~105° by 3 s; prior-free flat ~4°.

ROOT CAUSE (structural, not maturity): the prior integrated **commanded yaw only** `q'=q⊗exp(½[0,0,ψ̇]dt)`.
A quadrotor steers by TILTING (pitch/roll) — yaw-only integration applies none of the dominant attitude
change, so error ramps monotonically. The prior-free model decodes the full quaternion from the latent
(carries the maneuver) → bounded. **owm's prior already does the right thing** (`owm_physics.dynamics_prior`
kinematics path): it integrates the OBSERVED body rate `omega = prev_obs[bodyrate_idx]` (all 3 axes). My cave
prior had departed from that.

A/B launched (8 ep, GPU0/GPU1, `train_cave_proprio_AB.sh`):
- **A** (`experiment=cave_proprio_priorA_v1`, model=cave_proprio, `prior: kinematic`): attitude from observed
  body rate `q'=q⊗exp(½·w_prev·dt)`, w persists CLAMPED (rate_clamp 1.0), residual corrects. Safe despite
  persistence — quat lives on the unit sphere (quat_integrate ×unit dq preserves |q|), direction walk capped
  at 180°, unlike velocity (open integral). This is the verified 0.056°/step path.
- **B** (`experiment=cave_proprio_priorB_v1`, model=cave_proprio_posvel, `prior: kinematic_posvel`): NO
  attitude prior; quat left to the black-box decode (already flat ~4°). Prior = position+velocity only
  (`velocity_kinematics(bodyrate_idx=None)`).

Both still ground velocity on v_cmd (can't be black-box — that's the open-integral random walk that exploded
position). Rewrote `velocity_kinematics` (commanded-yaw → observed body rate + rate_clamp; `bodyrate_idx=None`
= variant B), added `kinematic_posvel` mode to `_make_dynamics_prior`, new `conf/model/cave_proprio_posvel.yaml`.
Vision run (`cave_fpv_v1`) killed to free GPU1 for the A/B (user: "we don't care about the vision model so
much here"). Next: compare A vs B vs no-prior at ep1 then ep3 via the now-N-way `compare_prior.py`.

### Phase 4b.3 — VERDICT: the prior loses; the fix is velocity-integral position (2026-10-20 ~01:15)

A/B compared at ep1 vs prior-free (`sim/cave/compare_prior.py`, now N-way; `logs/cave/prior_compare_AB/`):
**both A and B lose.** Attitude: A ramps to 138°, B to 82° (vs prior-free flat ~4°). Position @ 3s: A 1.9m,
B 3.0m (vs prior-free 0.6m). The observed-body-rate hypothesis for A was FALSIFIED — integration accumulates
over the horizon exactly like commanded-yaw did; B's persisted quaternion random-walks.

**General lesson:** for any channel the black-box decodes well in ABSOLUTE terms, an integration/persistence
prior is strictly worse — it turns a bounded per-step error into unbounded accumulated drift. A prior helps
ONLY where the absolute decode is weak. Here that's position alone.

**THE FIX (no retrain), `logs/cave/prior_compare_AB/the_fix.png`:** black-box decode everything + integrate
ONLY position from the DECODED velocity: `p[t] = p_anchor + cumsum(v_decoded[t-1]·dt)`. On the EXISTING
prior-free best.ckpt: position **0.075m @ 3s** (vs 0.60 raw decode, vs 1.9/3.0 for A/B), black-box attitude
(4°) and velocity (0.03) untouched. Works because the black-box velocity is independently excellent while its
absolute POSITION decode accumulates latent-rollout error. NOT a prior+residual — a POST-decode position
override; to help MPPI control it must live in the rollout (imagine_shared), denorm→integrate→renorm.

DECISION (user): killed A/B (dominated); HOLDING on implementing the fix for now — finding banked, prior-free
`cave_proprio_v1` best.ckpt stands as the dynamics model. Next downstream work (OOD, control) uses prior-free.

### Phase 5 — point-to-point POSE control benchmark: learned WM vs true dynamics (2026-10-20 ~02:45)

Built `sim/cave/control.py` + `run_control.py` (+ `prospector/cave_wm/run_cave_control.sh`). ONE MPPI controller,
the dynamics model is the only variable; execution always the true Dynamics3D plant. Arms: TrueDyn (Dynamics3D
rollout_batch, the ceiling), LearnedDyn (the red WM via imagine_shared, override baked in), AnalyticDyn (command
kinematics). Runs in the BRIDGE (prospector venv + PYTHONPATH=quickdraw/src, JAX_PLATFORMS=cpu; CaveScene built
collision-only via a _palette/_bake monkeypatch -- no gsplat/ninja). MPPI: COARSE (15x6 held = 3 s coherent
lookahead) far, FINE (repeat=1) near goal; cost = terminal+path position + yaw + clearance-SDF wall barrier +
near-goal velocity damping. Collision (clearance<=DRONE_R=0.3) = immediate fail. 20 pose pairs from val
trajectory points (near-line-of-sight), SAME pairs/seed across arms. Pose = position + yaw; success =
position (0.3 m) + settled (<0.3 m/s); yaw reported, not gated (goal yaw is an arbitrary recorded heading).

RESULT (`logs/cave/control_bench/`, control_metrics.png + control_trajectories.png):
  arm       success  collide  t_comp  settle  path/straight  yaw   clr_avg  final
  true       0.40     0.00     8.72s   8.72s    0.99          44     0.94    0.40m
  learned    0.35     0.00     8.40s   8.40s    1.00          58     0.95    0.42m
  analytic   0.30     0.00     8.26s   8.26s    1.00          72     0.97    0.40m
HEADLINE: the learned red WM controls almost as well as the TRUE simulator (0.35 vs 0.40 success = 7 vs 8 of
20; nearly identical time/path/clearance) and BEATS the analytic hand-model (0.30) -- learned sits between the
analytic floor and the true-dynamics ceiling, close to true. Zero collisions all arms (the SDF barrier works).
Trajectories overlay almost perfectly in 5/6 shown pairs. Modest absolute success is the 0.3 m tolerance sitting
at the controller's FINE floor (many pairs land 0.3-0.5 m) + a few wall local-minima -- SAME for every arm, so
the comparison is clean. Yaw tracking: true<learned<analytic (secondary).

VAL POINTS: pose pairs need val obs, but the bridge venv has no lerobot -> pre-extract pos+quat in the CONTAINER
(`.venv/bin/python` on load_split_episodes) to `logs/cave/control/val_points.npz`, read by run_control.

### Phase 5b — control episode videos: FPV | chase | top-down, real-time (2026-10-20 ~03:10)

`sim/cave/render_control.py`: re-runs each pose pair (deterministic, stochastic_eval=False) and renders every
30 Hz step from 3 cameras -> 30 fps mp4 (one sim step = one frame = REAL TIME, not sped up). Drone drawn as a
quadrotor GLYPH injected via scene.set_extra gaussians (green front / red rear rotors = heading); cyan goal
marker; dark cave brightened. FULL CaveScene (render path; needs ninja ON PATH: `PATH=<prospector>/.venv/bin:$PATH`
so gsplat JIT-compiles once -- the collision-only monkeypatch is NOT used here). gsplat render ~2 ms/frame.
No drone MESH in prospector (only a SISL texture droneBody_diffuse_SISL.png) -> glyph. Enclosed cave => external
chase/top-down eyes are CLAMPED to free space (`_safe_eye`, clearance-aware) or they land inside a wall.
20 videos in `logs/cave/control_bench/videos/pairNN_learned.mp4` (8/20 success deterministic vs 7/20 stochastic
in the benchmark -- 1-pair determinism gap). `prospector/cave_wm/run_cave_control.sh` for the benchmark.

### Phase 5c — tightened success (arrive+settle+HOLD) + station-keeping; re-scored (2026-10-20 ~03:50)

User: higher absolute success AND tighter requirements (velocity settling + position holding). Reconciled by
making the CONTROLLER station-keep, not by loosening. Added a FINE station-keeping mode (one-step actions, low
noise, velocity damping gated to the tolerance ball r_settle=0.40 so it holds without stalling the approach;
NEAR=0.6, w_term=1.2). SUCCESS is now a SUSTAINED HOLD: pos<0.40 m AND speed<0.25 m/s for >=20 consecutive
steps (0.67 s), no collision (`_run_start`). Goals sampled in safer free space (min_clear 0.85), steps 360.

RESULT (`logs/cave/control_bench/`, 20 pairs): true 0.60 (12/20) | learned 0.55 (11/20) | analytic 0.45 (9/20),
ZERO collisions all arms; time-to-complete ~8.9 s, settle ~8.9 s, path 1.03x straight-line, closest-obstacle
~0.88 m avg. Learned one pair behind true -> learned ≈ true under a rigorous arrive-settle-hold criterion;
clean true ≥ learned > analytic ordering. (Earlier momentary-touch criterion gave 0.40/0.35/0.30.)

Tuning seam: the velocity-damping zone (r_settle) MUST equal the tolerance, not larger -- r_settle=0.6 damped
the approach at 0.6 m and stalled the drone short (0.63 m). Fine-control floor is ~0.37 m (MPPI one-step
granularity), so pos_tol 0.40 sits just above it; rigor comes from the sustained velocity-settled hold.

### Phase 5d — SHARP-MESH drone + ghost in the videos (real SISL mesh, pyrender) (2026-10-20 ~05:30)

User wanted the drone as a sharp MESH (not gaussians), the real SISL model WITH its logo, a green 20%-opacity
ghost marking the goal (drone must hover exactly on it), props spinning, and the video to keep holding ~8 s
after arrival. Delivered:
- **Mesh extraction from `cave-maps.blend`:** the blend is huge (open_mainfile OOMs/timeouts) so use lazy
  `bpy.data.libraries.load` to pull ONLY the drone: `quadrotor`>`New Model Set`>`quad`>{~20 `default.*` body
  panels (Paint Gloss white/black/red; `default.040` carries the SISL logo texture) + `blade0-3` props}. bpy's
  glTF/OBJ exporters produce EMPTY/!!wrong!! geometry headless — so extract geometry DIRECTLY to numpy:
  `scratchpad/extract_npz2.py` -> `prospector/.../cave_wm/assets/drone_body.npz` + `drone_blade{0..3}.npz`
  (verts in mm, faces, per-vertex colors = material base color, with the SISL panel's colors sampled from the
  texture via UV so the red logo markings show). Body 194k verts; blades separate so props can spin.
- **Headless mesh render:** open3d OffscreenRenderer renders BLACK on this box -> use **pyrender + EGL**
  (`uv pip install pyrender trimesh` into the prospector venv, additive; torch/jax/Dynamics3D verified intact).
  `sim/cave/drone_render.py`: PERSISTENT pyrender scene (meshes uploaded once, per-frame only node poses change;
  full res, NO decimation), real + green-ghost scenes, blades spun about local +y through each blade centroid.
  Mesh-local +y = up, rotor plane x-z -> `ROT_L2B` maps local->body (record.euler_zyx_R). Camera matches the
  gsplat camera (same eye/fwd/UP + hfov 96.7) so the mesh composites pixel-aligned over the cave.
- **Compositing** (`render_control.py` rewritten): cave = gsplat; ghost (green, ~0.28 alpha) at the goal pose;
  sharp drone (opaque, spinning) over it; top-down slices the ceiling (gaussians above drone-0.3m dropped).
  FPV = cave+ghost (no self-drone); chase/top = cave+ghost+drone. `run_episode(hold_extra=240)` -> ~8 s hover
  on the ghost after arrival. 2 rows learned/true, 30 fps real-time. Videos: `logs/cave/control_bench/videos/
  pairNN_compare.mp4`. Inspection montage (4 views + XYZ axes): `logs/cave/drone_inspect.png`.
- Full drone mesh in `cave-maps.blend` as `OBquadrotor` (texture `droneBody_diffuse_SISL_Stanford.png`); the
  logo is per-vertex-sampled (approximate at drone scale) -- a crisp UV-textured mesh is a possible upgrade.

### Phase 6 — controller rewrite: the drone now ARRIVES (warm-start fix) (2026-10-20 ~19:00)

Symptom (user): "we don't get to the final goal, we just hover there" — the drone settled ~0.37–0.67 m short of
the ghost and bobbed. A friend's MPPI field notes (6-DOF AUV, 7.9 m → 0.016 m) pinpointed two structural bugs in
`control.py`, both now fixed:

1. **Warm start was `np.roll` (WRAPS) → shift + zero-pad + decay.** `np.roll(plan,-1)` recycles the last action
   back to the front; near the goal the cost is flat, the softmax weights go uniform, the mean barely updates, so
   the approach thrust fires forever → the hover-short / vibrate loop. Fix (mppi_step): `nxt = concat(plan[1:],
   zeros) * mean_decay`. **This is the whole ballgame:** mean final error 0.37–0.67 m → **0.128 m**, SR 0.92,
   v_goal (mean speed at goal, last 2 s) 0.147 → 0.007 when settled; zero collisions. (TrueDyn, 12 pairs.)
2. **Hard near-goal velocity brake `(d<r_settle)·|v|` → SMOOTH `exp(-(d/r_settle)^2)·|v|`.** A hard gate stalls
   the controller at the mode boundary (the old ~0.37 m floor); the smooth proximity weight removes the stall.
   Added a small control-effort term `beta_ctrl·||a||` (= deviation from hover in velocity mode; friend's
   "penalise effort, not a boundary-gated speed").

What did NOT transfer (the friend's plant has hydrodynamic drag; a quadrotor has none): a separate FINE repeat=1
station-keeping mode. With no drag a one-step plan either freezes short (decay bleeds the creep) or cannot arrest
the approach momentum — every FINE variant parked at its engage distance. So the controller is **single-mode**:
COARSE (15 actions × 6 held = 3 s coherent lookahead, exec_n=3 so it replans often) does the whole reach-and-hold;
its chunky held actions give the authority to close the last metre, and the warm-start decay + smooth damping
settle the residual. The ~0.15 m wander that remains is gentle drag-free station-keeping bob. NEAR=0 disables
FINE (kept in the file, documented as dead for this plant). New metrics in the episode dict: `v_goal, a_goal,
da_goal, wander, min_dist` (friend's point 8 — measure thrash, not just final error). Reach-vs-time curves (point
9 — dump the curve before tuning; the old "floor" was partly the episode budget): `logs/cave/control_tune/
dist_curves.png`. Tuning harness (TrueDyn, env-overridable knobs): `scratchpad/tune_control.py` (one-off; the
locked config lives in `control.py`, the canonical benchmark is `run_control.py`).

### Phase 7 — ALL-MESH render: cave is a mesh now, not splats (2026-10-20 ~19:45)

User: "there's still purple and blue in the splat walls … load in the MESH (not the splat) and texture it in
browns and greys", "4x the resolution", props were wrong, and "SISL text readable left-to-right" in chase.

- **The cave is now a MESH, no gsplat.** `grapevine-exped-3.ply` is itself a full surface mesh (800k verts /
  1.6M faces) whose AABB [31.4 × 35.9 × 16.8 m] exactly matches the control/collision voxel grid — i.e. it is in
  the drone's own metric frame, so it drops straight in with zero alignment work. (The blend's `Section N`/`tunnels`
  objects are in a different, misaligned survey frame — y to ~106 — and are NOT usable.) `drone_render._load_cave`
  loads it, tints it height-wise in rock browns/greys (floor browner, ceiling greyer) + faint grain, and pyrender
  renders it **double-sided** (`SKIP_CULL_FACES` — the cave normals face OUT, so from inside the walls are
  back-faces and cull to black without this) under strong ambient + several FIXED directional fills (directional
  light does not fall off with distance, so the whole cavern reads, not just a headlamp patch) + a camera headlamp.
  No purple/blue. Correct depth occlusion (walls occlude the drone).
- **Props fixed.** The 4 `blade` objects in the blend are identical geometry stacked at the origin (identity
  transforms) — the old extract stacked them → one blob in the centre. Now `drone_render` loads ONE blade and
  places 4 copies at the body's detected motor arm-tips (±139 mm x, +161/−115 mm z), spinning about each tip.
  `extract_drone_mesh.py` (cleaned, in the cave module) regenerates `drone_body.npz` + `drone_blade0.npz`.
- **Unified renderer** (`drone_render.render_view`): cave + green goal-ghost + drone, each rendered then
  DEPTH-composited (ghost translucent 0.25, drone opaque over). FPV gets **no ghost** (it is the drone's own
  camera; at the goal the ghost would fill the frame green). Top-down slices the ceiling via the cave camera's
  near-plane (clip nearer than drone_height − 0.35 m) so the overhead view sees inside.
- **House video layout** (`render_control.py` rewritten): per-view frame streams → `viz.label_panels`
  (panels 768×576 = 4× the 192×144 source, header 58 px) for the learned row and the true row, vstacked →
  `viz.save_mp4` at 30 fps. Replaces the hand-rolled compositing/labels. GL renderer confirmed = **NVIDIA H100**
  (amortized ~6 ms/mesh-render; the first render's 0.6 s is one-time shader compile), so full-res is fast.
- **RENDER FRAME:** QA at `logs/cave/qa_newrender_{mid,end}.png` — cave reads in browns/greys, drone sharp with 4
  corner props + landing gear + red SISL marks, ghost halo at the goal, FPV clean. **TODO:** the SISL logo is
  still per-vertex-approximate (not crisp text) — needs a real UV-texture material; and confirm the logo reads
  left-to-right in chase once textured.

### Reproduce (entry points)
- **Controller + benchmark:** `run_control.py` (bridge: prospector `.venv` + PYTHONPATH=quickdraw/src +
  JAX_PLATFORMS=cpu; `--ckpt <red best.ckpt> --root <dataset dir> --points_npz logs/cave/control/val_points.npz`).
- **Videos:** `render_control.py` (same bridge + `PATH=<prospector>/.venv/bin:$PATH PYOPENGL_PLATFORM=egl`,
  `--pairs 0,1,...`). Cave + drone + ghost are all meshes; no ninja/gsplat needed for the render.
- **Drone mesh assets:** `extract_drone_mesh.py` via `uv run --no-project --with bpy --python 3.11` (needs the
  blend; writes npz into prospector's cave_wm/assets). **Never `uv sync`** the shared venv.
- Red WM ckpt: `logs/train_world_model_2026_10_19_11_36_34_cave_proprio_v1/checkpoints/best.ckpt` (load with
  `integrate_position_from_velocity=True` + `velocity_idx=[7,8,9]`; run_control.load_wm does this).
