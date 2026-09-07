# A Gaussian-splat decoder: render the image instead of predicting it

Status: **IDEA ONLY. Nothing implemented, nothing measured beyond the diagnosis in §1.** Written
2026-09-07, from the starling flight-set results in `wizard/records/starling.md` §1.3.

The proposal is narrow: **replace the image decoder's output layer so it emits 3D Gaussians rather than
pixels, and render them through a rasteriser at the pose the proprio head already predicts.** The
encoder, the token bag, the transition head, the loss sites and `VisualLoss` are all unchanged.

---

## 1. The measurement that motivates it

On the starling drone sets, sharpness recovery (fraction of pixels with |grad| > 0.08, predicted over
ground truth), split by whether the dynamics is involved:

| | AE floor (codec only) | open-loop (codec + dynamics) |
|---|---|---|
| starling depth2 ep5 | **0.855** | **0.174** |
| starling depth4 ep5 | **0.882** | 0.174 |
| robocasa `twocam_full` ep29 | 0.890 | **0.856** |

**The codec renders sharp structure as well as robocasa's does. The rollout throws 80% of it away, where
robocasa's keeps 96%.** Ground-truth sharpness is the same in both (18.78% vs 20.29% strong edges), so it
is not the scene content. The one distinguishing fact is that **robocasa's camera is fixed** — most pixels
are unchanged step to step, so sharp structure survives by being carried forward — while on starling the
camera IS the drone and every thin line must be re-localised at a new position every step.

Corroborating, from `starling.md` §1.2: **80% of the dynamics error is spent by step 8**, and closed-loop-1
(re-ground every step) still scores 0.148 against a 0.036 floor. The failure is a SINGLE-STEP positional
failure, and the blur is its downstream symptom — a decoder that does not know where a light strip is
renders a faint smear.

## 2. Why a splat decoder attacks that specifically

Three distinct reasons, worth keeping separate because they have different strengths:

1. **The representation stops depending on viewpoint.** Today the decoder is a function of the latent
   alone, so to render step `t+k` sharply the latent must encode where every structure sits IN IMAGE
   SPACE — and after the camera moves that is a completely different arrangement. With a splat the render
   is a function of *(scene, pose)*, and the scene is the same 3D object from any viewpoint.
2. **The renderer supplies projective geometry for free.** Perspective, parallax and occlusion ordering
   are exact arithmetic in the rasteriser. Today the conv trunk must LEARN the projective geometry of
   camera motion, implicitly, from two hours of flight, in a 6.76M-parameter model. A fixed-camera dataset
   never imposes that burden — which is precisely why robocasa keeps 0.856 and starling keeps 0.174.
3. **Errors change character from blur to misregistration.** This is the claim the idea should be judged
   on. Uncertain about position, the current model hedges into a smear, and LPIPS punishes that hard. A
   splat rendered from a pose 20 cm off is a SHARP, structurally correct image from a slightly wrong
   viewpoint, which LPIPS punishes lightly — it is notably displacement-tolerant, the very property that
   hurt us elsewhere (`conf/model/vl64.yaml`: "a loss blind to spatial displacement tells the dynamics
   that landing in the wrong place is cheap").

**The honest counter.** Pose prediction is currently BAD: proprio open-loop `pointwise_error` 1.353 in
normalised units against a 0.083 codec floor, i.e. metres of position error in a 15 m room. So this trades
"blurry image of roughly the right place" for "sharp image of the wrong place". Whether that is a net LPIPS
win is empirical, not obvious. §6 is the experiment that settles it before any code is written.

## 3. What makes it tractable here: the pose is observed

`starling`'s 16-dim `observation_vector`, recovered by fingerprinting 24,277 val frames:

| dims | what | fingerprint |
|---|---|---|
| **0:3** | **position** | x ±7 m, y ±2.9 m, z 0.22–2.61 m — z always positive, so altitude in a flightroom |
| 3:6 | linear velocity (likely) | ±3.0, ±2.5, ±0.9 — z component smaller, as expected |
| **6:10** | **orientation quaternion** | \|q\| = 1.00000 ± 0.000000 exactly |
| 10:15 | rates / accelerations (unconfirmed) | — |
| 15 | specific force (likely) | mean 9.77, range 2.79–14.68 — gravity-ish |

So a **full 6-DoF camera pose is already in the observation vector, already normalised, already supervised,
and already predicted by the proprio head.** No pose has to be inferred from images. This is the fact that
turns the idea from a research project into a decoder swap. (The robocasa equivalent is recovered in
`environments/robocasa_utils.OBS_LAYOUT`; dims 0:3 + 3:7 there.)

## 4. Prior art, read 2026-09-07

**Splatter Image** (Szymanowicz, Rupprecht, Vedaldi, CVPR 2024, arXiv 2312.13150) — **the one that maps
onto our architecture.** Feed-forward: a network maps an image to a Gaussian mixture in one pass.

* Backbone is a SongUNet; the last layer is replaced with a **1x1 conv with `12 + k_c` channels**, where
  `12` = opacity(1) + 3D offset(3) + depth(1) + scale(3) + quaternion(4) and `k_c` is 3 (Lambertian) or 12
  (spherical harmonics at L=1). **One Gaussian per pixel.**
* The position parameterisation is the load-bearing trick:
  **mu = (u1*d + dx, u2*d + dy, d + dz)** — depth `d` places the Gaussian along that pixel's camera ray,
  and the learned 3D offset then lets it move OFF the ray. That is what lets a single view cover geometry
  it cannot see: the network allocates some Gaussians to the visible surface and pushes others behind it,
  and switches one off entirely with sigma = 0. The paper describes it as "an extension of depth prediction
  networks".
* Activations: `sigma = sigmoid(s)`, **`d = (z_far - z_near) * sigmoid(d_hat) + z_near`**,
  `Sigma = R(q) diag(exp(s_hat))^2 R(q)^T`.
* **Multi-view fusion is a rigid warp plus a union**: run the net per view, map each mixture into a common
  frame (`mu~ = R mu + T`, `Sigma~ = R Sigma R^T`), take the union. They also condition on relative pose
  and add cross-view attention so views coordinate.
* **Trained with L2 + LPIPS** — and they say explicitly why they can: the renderer hits 588 FPS, so whole
  images fit in every iteration, unlike NeRF which must subsample pixels. Same objective family as ours.
* Single GPU, <= 20 GB, 38 FPS reconstruction.

**Scaffold-GS** (Lu et al., CVPR 2024, arXiv 2312.00109) — **NOT a decoder, and not needed.** This is
per-scene optimisation: voxelise a COLMAP point cloud, treat voxel centres as anchors carrying a feature
`f_v` in R^32 plus `k` learnable offsets, spawn `k` Gaussians per visible anchor, and decode their opacity /
colour / scale / quaternion on the fly from (anchor feature, camera-anchor distance, viewing direction)
through small MLPs. Loss is L1 + SSIM + a volume regulariser. It is fitted to one scene over minutes; no
network maps images to a scene, so it cannot be dropped in.

Its anchor-spawns-k-Gaussians mechanism IS the right answer to "many Gaussians from few parameters", and
its view-dependent MLP decoding would suit a specular flightroom better than SH. **Hold it in reserve**:
the arithmetic says Gaussian count is not the binding constraint (§5), so we would only reach for it if
that turned out wrong.

## 5. The architecture

The Gaussians live in a **fixed reference frame** — the camera pose at the last context step — and the
render uses the RELATIVE pose to the target step. That is what buys reason 2 in §2. A per-step splat
rendered from its own camera would be an over-parameterised way of making an image and would gain nothing.

```
                    predicted token bag  (M, 33, d=128)
                              |
            +-----------------+------------------+
            | proprio token (1)                  | image tokens (32)
            v                                    v
    proprio decode (MLP flow head)      TokenGridReadout + conv up trunk   <-- UNCHANGED
            |                                    |
            v                                    v
      obs_t (16 dims)                   out_conv: Conv2d(64, 15, 3)   <-- 3 channels -> 15
            |                                    |
   pose_t = (obs[0:3], obs[6:10])       splatter map (M, 15, Hg, Wg)
            |                                    |
            |                    s_hat -> sigmoid                  opacity  1
            |                    d_hat -> znear+(zfar-znear)*sig   depth    1
            |                    delta -> tanh * offset_scale      offset   3
            |                    s     -> exp                      scale    3
            |                    q     -> normalise                quat     4
            |                    c     -> sigmoid                  rgb      3
            |                                    |
            |           mu = (u1*d+dx, u2*d+dy, d+dz)   u = ray dir per grid cell
            |           Sigma = R(q) diag(exp s)^2 R(q)^T
            |                                    |
  pi_rel = pose_t (-) pose_ref -----------> gsplat rasteriser
                                                 |
                                                 v
                                   image (M, Hg, Wg, 3) in [0,1]
                                                 |
                                                 v
                              VisualLoss(pred, target)   <-- UNCHANGED
```

**What does not change, which is most of it:** the encoder, the token bag, the transition head, the
`TokenGridReadout`, the conv up trunk, both loss sites, the shared `VisualLoss` instance, the L1/LPIPS mix,
`latent_loss_weight`. The rendered image is in [0,1] by construction (sigmoid colours + alpha
compositing), so **`decode_out_act` becomes moot** and the range diagnostics go quiet.

### Files

| file | change |
|---|---|
| `models/splat.py` **(new, ~120 lines)** | ray grid from intrinsics, channel -> Gaussian activation, relative-pose composition, the `gsplat` call. Kept out of `decoders.py` the way `robocasa_utils.py` is kept out of the env file, so the decoder stays readable |
| `models/decoders.py` | `SplatGridDecoder(TokenGridDecoder)` — inherits the trunk verbatim, overrides `out_conv` width and `velocity()`. Leaves `TokenGridDecoder` bit-identical |
| `models/modalities.py` | `decode_arch: "splat"` branch; `ModalitySpec` gains `splat_grid`, `splat_z_near`, `splat_z_far`, `splat_offset_scale` |
| `models/multimodal.py` | **the one structural change**: decode proprio FIRST, extract the pose, pass it to the image decode |
| `conf/model/vl128_starling_splat.yaml` | the recipe |

That structural change is smaller than it sounds. `self.layout` already orders proprio before image, and
both `to_obs` and `recon_losses` iterate it in order. There is also an existing precedent for pulling
physical state out of the bag with a frozen head: `multimodal.physical_state()` does exactly that for the
physics loss.

### Three things that would bite

**Intrinsics, and a clean dodge.** `mu` needs a ray direction per grid cell, which needs the FOV.
starling's is not recorded anywhere (torus has `fpv_fov: 100` in config). Rather than fitting it: make
**log-focal a single learned scalar**, initialised from a guess. One parameter, gradient arrives free
through `mu`, blocker gone.

**Memory.** 21,504 Gaussians per frame x F=64 x batch 21 is a great deal of rasterising, and decode is
already ~78% of per-sample memory (`design/decode_memory.md`). Two outs: a generator grid COARSER than the
image (56x96 = 5,376 Gaussians still carries more than the 21,504 output pixels need, at 4x the economy),
and `decode_chunk_train` already exists. Measure before assuming.

**Initialisation.** `sigmoid(0)` puts every Gaussian at mid-depth with random offsets and opacity 0.5 — a
uniform fog — and unlike the paper our decoder trains JOINTLY with the dynamics rather than standalone.
Mitigate with the project's existing trick: **zero-init the offset and scale heads**, so at step 0 the
splat is a clean depth-ordered sheet at mid-depth that renders as roughly the right image. `decode_inject`
and `LevelCrossAttn` both already do this (`models/decoders.py`).

### What this variant does NOT give you

**No persistence.** Each step re-emits its own splat, so nothing is literally carried forward; the win is
narrower — viewpoint change becomes exact arithmetic instead of something the conv trunk must learn. Two
bonuses fall out anyway: because the reference frame is fixed across the window, successive steps describe
the same scene, so the latent's residual only has to encode scene CHANGE; and the depth channel is a
**free depth map**, itself a diagnostic for whether the model has any geometric understanding at all.

The cheap push toward persistence, if wanted later, is a consistency penalty between consecutive steps'
Gaussians — no new machinery.

## 6. Two experiments to run BEFORE writing any of this

Both are days, not weeks, and either coming back negative saves building the wrong thing.

1. **True-pose oracle.** Condition the EXISTING decoder on the ground-truth future pose and measure how
   much of the 0.174 -> 0.855 sharpness gap closes. If perfect pose does not fix the blur, positional
   uncertainty is not the mechanism and this whole document is built on sand. If it closes most of the gap,
   the payoff is bounded and everything above is justified.
2. **Partition the token bag.** Hold ~24 "scene" tokens fixed through the rollout and let ~8 "ego" tokens
   evolve; the flow head already predicts per-token residuals, so this is masking which tokens get one.
   Tests the persistence claim for roughly twenty lines, with no rasteriser, no depth and no intrinsics.

## 7. Variants considered and set aside

* **Pose-only rollout.** Fuse the P context frames into ONE persistent splat via the known poses, then
  predict only pose. Maximum persistence benefit and the smallest thing to predict, but it can render
  nothing outside the context views — at +128 (21 s of flight) coverage is near zero — and it cannot
  represent the walking person in some flightroom episodes. Worth building as a DIAGNOSTIC CEILING: it
  measures the most persistence could ever buy.
* **Splat plus learned residual.** Persistent splat renders a geometric prior; the existing conv decoder
  predicts a residual on top from the rolled latent. Keeps generative capacity and degrades gracefully to
  today's behaviour where the splat has no coverage. Costs two decoders and risks the residual head
  learning to ignore the splat. The natural follow-on if §5 works.
* **Depth + pose warping** as a cheaper stand-in. Dropped: §4 shows it is the degenerate case of the
  splatter-image head — same information, worse renderer — so there is little to learn from it that
  building the real thing would not also teach.
* **Raising `num_tokens`** to widen the latent. Not the constraint: the decoder already expands 4,096
  latent floats into 64,512 output numbers (15.8x), so the latent INDEXES a generator rather than storing
  the output. This was the error in the first framing of this idea.
