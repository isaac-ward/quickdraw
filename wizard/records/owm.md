# owm-iss — orbital docking world model (numerical-v1, best-renderer)

Running record for the ISS docking world-model work on the `sislaboratory/owm-iss-numerical-v1-*-goal-dt50ms-500k`
datasets. Companion to `wizard/records/robocasa-scene4-4h.md` (the recipe origin) and the executable recipe
`conf/model/bsp32mse.yaml` + `src/quickdraw/models/recipes/bsp32mse.md`.

## Datasets

Three variants, 20 Hz (dt 50 ms), ~500k frames each, FPV + composite 512×512 renders (best renderer):
- `...-nonoise-...` (deterministic), `...-coop-...` (cooperative target), `...-noncoop-...` (evasive target).
Local: `logs/recorded_hf/<name>`. Loaded via `data.root=... data.repo_id=<name> data.cam=fpv`,
`environments=recorded`.

Episodes are long: median ~5.9k frames (~297 s), max 7201 (~360 s), ~94 val episodes.

## Observation schema — the 27-dim `observation_vector` (decoded from stats.json)

| dims | quantity | per-dim std | notes |
|---|---|---|---|
| 0–1 | **absolute epoch time** (Julian date ~2.46e6, seconds-in-day) | 14800 / 28000 | **JUNK for dynamics.** JD at 2.46e6 in float32 has ~0.25-day quantization → sub-day info is destroyed. Also stats show mean<min (degenerate). DROP. |
| 2–4 | **ego position** (m) | ~[69, 58, 51] | `position_idx=[2,3,4]`. Spans ±~250. |
| 5–7 | **ego velocity** | ~[1.06, 0.93, 0.87] | the quantity that must be integrated to get position |
| 8–11 | ego attitude **quaternion** (w-first) | ~[0.27, 0.66, 0.29, 0.59] | drives the FPV camera view |
| 12–14 | ego **body rates** (ang. vel.) | ~[0.03, 0.02, 0.02] | tiny |
| 15–17 | goal-relative **position** | ~[69, 60, 57] | ≈ ego position minus a per-episode goal offset → largely REDUNDANT with 2–4 |
| 18–20 | goal-relative **velocity** | ~[1.06, 0.93, 0.87] | ≈ ego velocity (goal ~static) → REDUNDANT with 5–7 |
| 21–23 | goal-relative **attitude error** (3, axis-angle/MRP) | ~[0.74, 0.49, 0.48] | control-relevant |
| 24–26 | goal-relative **body rates** | ~[0.03, 0.02, 0.02] | REDUNDANT with 12–14 |

`action` is 6-D (continuous thrust/torque, std ~600–710, per-dim independent). Obs is z-scored per-dim in the
loader (`data/dataset.py:22-23` from `stats.json`). There is currently NO obs-subsetting knob — selecting dims
requires a loader change (slice `obs_all` + `o_mean`/`o_std`, and reset `position_idx`).

## Finding 1 — `data.subsample`: neither delta_k nor action-decorrelation binds; span governs

- **delta_k / codec ratio** (the robocasa method): demands sub-1 Hz here (nonsense) — the from-scratch codec
  RMSE (~0.127 at 18 dB, below) exceeds per-step image motion at every sane stride.
- **action decorrelation**: the 6 action dims are independent and their ACF is FLAT ~0.72 from lag 1 out to 2 s
  (drops 1.0→0.72 in one step, then plateaus) — a persistent low-freq component (~72% var) + per-step white
  noise (~28%). NO stride decorrelates them, so it can't pick k either.
- **real-time span** `F*k/fps` is therefore the driver. Confirm with `eval_ae_floor/image/psnr/@+1` at ep1.

## Finding 2 — subsample=20 (image head) validated, then superseded by a proprio focus

Ran `model=bsp32mse` at subsample=20 (1 Hz, 64 s window), coop+noncoop, 10 epochs before pivot. The image head
behaved exactly as the recipe wants — no zero-motion collapse:

| metric (coop, ep0→ep9) | | |
|---|---|---|
| open-loop motion_ratio @+1 | 0.070 → **0.349** | rising 5× (the anti-collapse signal) |
| ae_floor motion_ratio | 0.199 → 0.484 | codec preserves 2.5× more motion |
| ae_floor psnr @+1 (from-scratch codec floor) | 17.95 → 19.0 dB | ~2 dB below robocasa's 20; RISING, not eroding |
| open-loop psnr (mean) | 10.7 → ~13.0 | climbing toward robocasa's 14.25 |
| codec roundtrip loss | 0.008 → 0.006 | weight-10 anchor holding |

So the image recipe ports cleanly. **But position prediction is poor**, which redirected the work (below).

## Finding 3 — position prediction: it's the proprio autoencoder floor + a double integrator, NOT the dynamics

The checkpoint metric `pointwise_error` is a **raw-unit, position-only L2** over the rollout, so it looks
alarming (~30–50) while `obs_error` (normalized, full vector) is a healthy ~0.2. Decomposing (coop ep9):

| eval mode | pointwise (raw pos L2) | obs_error (norm) |
|---|---|---|
| **ae_floor** (encode→decode, no dynamics) | **28.2** | 0.22 |
| **closed_loop_1** (one-step) | **29.7** | 0.21 |
| closed_loop_16 | 33.0 | 0.26 |
| open_loop (64-step AR) | 50.0 | 0.41 |

Three stacked causes:
1. **Scale illusion.** The codec reconstructs the whole vector to a uniform ~0.2σ. Position σ≈80 → 0.2σ ≈ 16/dim
   ≈ 28 in L2. Same *relative* fidelity robocasa got, but robocasa's EEF lived in a ~0.5 m box so its raw number
   was centimeters and looked "great." We're comparing raw L2 across a ~300× scale gap.
2. **One-step ≈ ae_floor (29.7 vs 28.2).** The dynamics predicts position to the codec's representational limit
   already. The bottleneck is the **proprio autoencoder floor**, not the transition model.
3. **Double integrator → open-loop drift (28→50).** In the *data* it is genuinely second-order: verified
   Δpos ≈ v·dt (per-step Δpos 0.09 ≈ velocity 0.9 × dt 0.05). But the STATE is (p, v) with both in the obs, so
   the one-step map (p,v)→p' = p + v·dt is near-linear and easy (hence one-step ≈ floor). The difficulty is only
   in AR rollout: position = p₀ + Σ v·dt, so any velocity bias ε compounds to ε·N drift.

## Current proprio autoencoder

`VectorModality` (`models/modalities.py`): `27-vec → FourierMLP(→d=128) → 1 token → FlowField(MLP) decode → 27-vec`.
Recipe had proprio `decode_kind=flow, decode_param=x0` (generative, noise-curriculum trained). Raw capacity is
NOT the bottleneck (128 floats ≫ 27 dims); the floor comes from (i) generative flow noise on a near-deterministic
target and (ii) the huge position dynamic range eating the uniform relative-fidelity budget.

## Levers (ordered easy→structural), with status

Two distinct failure modes, do not conflate them:
- **FLOOR / one-step error** (ae_floor 28 ≈ one-step 30): a *representation* problem → mse decode, dim subset,
  relative encoding.
- **DRIFT over the rollout** (28→50): an *integration* problem (Σ velocity error) → kinematic coupling, finer dt.

1. **`model.modalities.0.decode_kind=mse`** (proprio flow→mse) — **DONE, in the running arm.** Deterministic L2
   (degenerate no-noise FlowField); removes flow sampling noise from a near-deterministic target. Same argument
   the recipe makes for the image decode. Attacks the FLOOR.
2. **`data.subsample=5`** (5 Hz, F=64 → 12.8 s) — **DONE, in the running arm.** Finer dt → smaller/more-linear
   per-step extrapolation + better-resolved velocity → lower one-step error AND slower DRIFT.
3. **Config-driven obs subset, keep ego dims 2–14** — **DONE (code) + in the running arm.** New `data.obs_keep`
   knob (default null=full; see below). Keep ego pos/vel/quat/bodyrate (13 dims); drop broken time 0–1 and the
   redundant goal-relative block 15–26. Quat+bodyrate KEPT because they drive the FPV view (the image head needs
   them). Concentrates the shared latent budget on the physical ego state. Attacks the FLOOR.
4. **Relative / delta encoding of proprio** — NEXT, once mse/subset confirm the floor is codec-limited. See the
   "Relative encoding" note below. Attacks the FLOOR (not the drift).
5. **Enforce kinematic coupling** p' = p + v·dt (and ω = dθ/dt for attitude) — the DRIFT fix. See the
   "Kinematic coupling" note below for what the repo already has and what's missing. Strongest, most invasive.

## `data.obs_keep` — the config knob (implemented)

`conf/data/torus.yaml: obs_keep: null` (OFF = full vector). Set to a list of dim indices to KEEP; applied inside
BOTH episode loaders (`load_split_episodes`, `load_split_episodes_mm`) AND `Normalizer.subset_obs` (so training
and every eval see the same subset — same no-missed-call-site discipline as `subsample`). When set you MUST also
set `model.modalities.0.dim`, `environments.obs_dim`, and `environments.position_idx` to the SUBSET indexing.
For owm-iss `obs_keep=[2..14]` → obs_dim 13, position triple now at `[0,1,2]`.

## Kinematic coupling — what the repo has, and why it's not usable as-is

`training/variations.py: PhysicalLoss` already has a **CONTINUITY** term = exactly `v = dp/dt`, a soft
central-difference residual `||v̂_t − (p_{t+1}−p_{t−1})/(2·dt)||² / v_scale²`. BUT:
- It is **gated on `env.physical_loss(obs_phys)`**, an env-provided analytic residual hook. `RecordedEnv`
  (`environments/recorded.py`) does **not** implement it → the whole variation **skips** (`{"skipped":1.0}`);
  the env-contract log line reads `physical_loss ✗ off`. So it is OFF for all recorded data today.
- Its ALGEBRAIC terms (`d_off` on-surface, `v_off` tangent) are **torus-specific** — irrelevant to docking.
- It is a **SOFT penalty**, not the HARD structural enforcement ("network never independently regresses
  position"). Hard enforcement = a structured transition/decode where position is COMPUTED as `p_prev + ∫v`,
  never a free output — a model change, not a loss.
- CONTINUITY as written is Euclidean; **attitude** needs `ω = dθ/dt` in quaternion-log form, not implemented.

To use it for docking: implement a generic, env-independent continuity residual driven by `position_idx` +
`velocity_idx` (+ a quaternion variant for attitude), OR go the hard-enforcement route.

## Relative encoding — what "delta targets" means here (concretely)

The model today encodes ABSOLUTE position (±300) and the decoder must reproduce a number near ±300; a uniform
0.2σ codec error on σ≈80 is ~16/dim of raw error — that IS the floor. "Relative/delta target" = a change of
coordinates, not a physics change: pick a reference = the last observed context position `p₀`, subtract it from
every frame's position before the model sees it (`p̃_t = p_t − p₀`). Now the model encodes/predicts `p̃`, which
starts at 0 and only grows to the accumulated motion over the 12.8 s window (tens of units, not ±300) — far
smaller dynamic range, so the SAME relative fidelity yields a much smaller raw floor. At eval, reconstruct
absolute position by adding `p₀` back (`p_t = p̃_t + p₀`); `p₀` is known (the context's last position). It is a
single subtraction, needs no per-step decode (fits the latent-space model), and removes the ±300 DC offset — the
offset the actions don't even control (it's just where the episode started) — from both the AE floor and the
latent. NOTE: it fixes the FLOOR, not the integration DRIFT (velocity prediction, hence Σε, is unchanged).

## Experiments (run log — EVERY run goes here, with the NEW feature(s) it tests)

Discipline: log every launch with its date, the new feature(s) under test vs the prior run, and the reading.

| launched | tag | dataset | new feature(s) vs prior | status | reading |
|---|---|---|---|---|---|
| 08-15..08-16 | `bsp32mse_{coop,noncoop}` | coop, noncoop | port bsp32mse; subsample=20, full obs27, proprio flow | done 10 ep, killed | image head validated (motion_ratio rising); **position poor** → diagnosis |
| 08-18 ~15:48 | `bsp32mse_{coop,noncoop}_s5mse` | coop, noncoop | subsample 20→5; proprio decode **flow→mse** | superseded pre-ep1 | (pivoted to ego13) |
| 08-18 16:00 | `bsp32mse_{coop,noncoop}_s5mse_ego13` (1st) | coop, noncoop | **obs_keep=[2..14]** (13-dim ego) via new `data.obs_keep` knob | killed — evals crashed 27-vs-13 | ep0 floor (standalone, fixed code): **5.6 / 4.4**, ~5× drop |
| 08-18 19:04 | same (relaunch) | coop, noncoop | eval-loader `obs_keep` fix (param version) | killed — poisoned by mid-run code edit | ep0 eval failed again (module version skew) |
| 08-18 21:46 | `bsp32mse_coop_s5mse_ego13` **(anchor-0 baseline)** | coop | settled `set_obs_keep` global (obs_keep fix, final form) | COLLAPSED, killed 08-19 (freed GPU0 for rt10) | proprio floor 6.7→19.5→18.1 (ep0/1/2, collapsed at p_tf→0); image *motion* 0.31→0.12 (collapsing); image *codec* floor HELD (rt_img ~0.01, anchored) — **in-run proof: anchored modality holds, unanchored collapses** |
| 08-18 21:46 | `bsp32mse_noncoop_s5mse_ego13` | noncoop | (paired w/ above) | killed 08-19 to free GPU1 | ep0→ep1 ae_floor 4.2→15.0 (same collapse) |
| 08-19 02:2x | `coop_ego13_rt1` (1st) | coop | proprio anchor weight 1 — **but silently unanchored** (VectorModality wiring bug, below) | killed, invalid | == baseline objective; no anchor engaged |
| 08-19 04:17 | `coop_ego13_rt1` (relaunch) **(A/B, weight 1)** | coop | proprio anchor **1** + VectorModality wiring fix | **RUNNING** GPU1 | ep0→ep1 ae_floor 6.5→**8.2 HELD** (vs baseline 6.7→19.5); cl1 7.0→11.3; open_loop 41→**33** (better rollout) |
| 08-19 04:2x | `coop_ego13_rt10` **(A/B, weight 10)** | coop | proprio anchor **10** (upper bracket) | **RUNNING** GPU0 | ep0→ep1 ae_floor 5.4→**6.5 HELD tighter**; cl1 6.8→7.6; open_loop 62→**44** (worse rollout — over-constrained) |

**Finding 7 — the proprio anchor PREVENTS the floor collapse (confirmed at ep1).** Both anchored weights held
`ae_floor` at ~6–8 through the `p_tf→0` transition vs the unanchored baseline's 3× collapse to 19.5 — the
diagnosis (Finding 6) and fix are validated. The 1-vs-10 bracket shows the predicted tradeoff: **weight 10 holds
reconstruction tighter (floor 6.5, cl1 7.6) but has WORSE open_loop (44 vs 33)** — the strong anchor
over-constrains the codec and the rollout pays. Weight 1 looks like the better balance so far (holds floor AND
rollout). Caveats: only ep1 (open_loop noisy); and **image-motion still collapses in both** (`OL_imgMR` 0.3→0.14,
same as baseline) — the proprio anchor doesn't touch the image-dynamics motion loss, a separate open problem.

**Finding 6 — the proprio codec had NO roundtrip anchor, and the anchor flag was silently a no-op.** The recipe
anchors only the IMAGE codec (`latent_loss_weight=10`); proprio's was unset → 0. So once `p_tf→0`, the unanchored
proprio codec co-adapts to prediction and its floor collapses (baseline: 6.7→19.5→18.1), exactly `design/collapse.md`.
Fix = a proprio anchor. **But a second bug:** `+model.modalities.0.latent_loss_weight=1` set the SPEC value, yet
`VectorModality.__init__` never copied it to the module, and `roundtrip_losses` reads the weight OFF THE MODULE
(`getattr(module,"latent_loss_weight",0)`) — so it stayed 0 and the first rt1 run was silently unanchored (only
`roundtrip_image` ever logged, never `roundtrip_proprio`). Fixed: `VectorModality` now exposes
`self.latent_loss_weight` (default 0.0 = unchanged for every other run). Confirmed: spec 0→module 0 (no anchor),
1/10→in roundtrip heads. Note the flag needs `+` (key not in the proprio struct by default).

**Active A/B (08-19), anchor-strength bracket (baseline collapse already captured):** `coop_ego13_rt1` (proprio
anchor 1, GPU1) vs `coop_ego13_rt10` (proprio anchor 10, GPU0). Weight 1 = matched to dynamics scale; weight 10
= image-matched/strong (proprio roundtrip ~15× the image one, so 10 may over-constrain → informative ceiling).
Read: does either hold `ae_floor` flat through `p_tf→0` (vs the baseline's 6.7→19.5 collapse), and does 10 wreck
the dynamics/image-motion? OLD note superseded: coop baseline (proprio anchor 0, GPU0) vs `coop_ego13_rt1` (anchor 1,
GPU1, from ep0). Same dataset + config otherwise. Hypothesis: the anchor holds the position floor (ae_floor)
flat through the `p_tf→0` transition instead of eroding 6.7→19.5. Weight 1 (not 10) because the proprio roundtrip
MSE is ~10× the image one, so 10 would swamp the objective; 1 sits near the dynamics-loss scale.

Features now available (default OFF): `data.obs_keep` (process-wide obs subset, `set_obs_keep`); proprio
`decode_kind=mse`; proprio `latent_loss_weight` (roundtrip anchor, needs `+` — not in the proprio struct by default).
NOT implemented: relative encoding (parked — floor fix, low value), kinematic coupling.

## Finding 4 — ego-13 + proprio-mse cut the position floor ~5×; the bottleneck shifted to DRIFT

Standalone eval on the ep0 (1-epoch) checkpoints with the fixed code:

| coop, raw pos L2 | old (full-27, sub20) | **ego-13 + mse (sub5)** |
|---|---|---|
| ae_floor (codec floor) | 28.2 | **5.6** |
| closed_loop_1 (one-step) | 29.7 | **8.2** |
| closed_loop_16 | 33.0 | 10.6 |
| open_loop (64-step) | 50.0 | 36.6 |

(noncoop ae_floor 22.0 → **4.4**.) The mse-decode + ego-subset did the heavy lifting — the codec IS adaptive (its
`FourierMLP` encoder self-scales the small centered signal), so the earlier worry that we needed relative
encoding to beat an absolute floor was wrong. **Two consequences:**
1. **Relative encoding is now low-value.** The floor is 5.6 and one-step (8.2) already EXCEEDS it — lowering the
   floor further has diminishing returns. (It stays a cheap option, not a priority.)
2. **The bottleneck moved from floor to DRIFT.** Before, one-step ≈ floor (floor-limited). Now one-step (8.2) >
   floor (5.6), and open_loop (36.6) ≫ one-step = the autoregressive rollout compounds error. Caveat: ep0, one
   epoch — the one-step gap may narrow with training; confirm over the relaunch.

## Finding 5 — the goal, and how to close the open_loop→floor gap (framing corrected)

Target (user): **open_loop position as good as `ae_floor`** (~6, vs open_loop ~38). This IS closable in
principle — `numerical-v1` is deterministic physics and the action sequence is given, so the future is fully
DETERMINED by (p0, v0, actions); it is not irreducible uncertainty. `ae_floor` is a lower bound on open_loop
(you can't decode position better than the codec), so the goal = make the rollout add ~nothing.

**Correction to earlier framing (important):** the current model DECODES position directly from the predicted
latent — it does NOT integrate velocity. So "drift = integrated velocity error" was wrong for THIS architecture;
the drift is generic autoregressive **latent** drift. Velocity matters as an *input* (the latent must carry the
rate to predict next position) but NOT as an *output* — so "up-weight velocity" is a non-lever here. It only
becomes a lever under kinematic coupling.

**Why the gap is hard:** the data is a double integrator (position = ∫∫thrust). Predicting position DIRECTLY
over 100s of steps is ill-conditioned — nothing forces the rollout onto the physical manifold, so it wanders
(the ep0 open_loop plot is a wild scribble vs a short GT segment). **Key insight:** for a double integrator
driven by KNOWN actions, the hard quantity (position) is the double-integral of an EASY quantity (acceleration ≈
the thrust/action, a near-algebraic map). Predicting position directly fights the conditioning; predicting
acceleration and integrating respects it.

**Levers, corrected & ordered (current architecture):**
1. **Kinematic coupling (structural) — highest leverage.** Restructure the readout so `v' = v + â·dt`,
   `p' = p + v'·dt`, with the net predicting acceleration `â` (≈ the action). Then long-horizon position is a
   stable double-integral of an easy target; drift is bounded by acceleration error + the initial-condition
   floor. This is the principled path to open_loop → floor. Soft precursor: a generic continuity residual
   `‖v̂ − Δp/Δt‖²` (the `physical_loss` idea, but env-independent, driven by position_idx=[0,1,2] +
   velocity_idx=[3,4,5]; the repo's version is env-gated + torus-specific, OFF for recorded data).
2. **Diffusion forcing — robustness, complementary.** Train on NOISED context so the latent dynamics tolerates
   its own accumulated error (exposure-bias / amplification fix). Architecture-agnostic; helps the drift we have
   but doesn't fix the ill-conditioning. `p_tf_warmup=1` already gives partial in-rollout exposure.
3. (deprioritized: relative encoding = floor fix; contraction = caps amplification but not correctness;
   velocity-weighting = only meaningful inside kinematic coupling.)

**Realistic ceiling:** kinematic coupling could plausibly pull open_loop from ~38 into low-teens/single digits,
but an EXACT floor match over 100s of steps is unlikely (initial-condition codec floor + acceleration error
integrating). Confirm the "just train" baseline first — does open_loop fall over 50 epochs, or plateau like the
old high-floor arm (~50 across 9 ep)? If it plateaus, kinematic coupling is the move.

## Bug fixed — the "missed call site" trap (obs_keep in eval)

`data.obs_keep` was first added as a PARAM threaded through every loader call site — and the eval routines'
OWN `load_split_episodes_mm` calls (`routines.py` ×8) were missed, so training ran 13-dim but every eval crashed
`RuntimeError: size of tensor a (27) must match b (13)` (non-fatal → trained blind, no eval metrics). The
param-threading was the wrong shape (correctness spread over 11 sites). **Final fix: a process-wide global
`set_obs_keep` mirroring `set_subsample`** — applied INSIDE both loaders (`_apply_obs_keep`) and the Normalizer
(`subset_obs`), set once per entrypoint (`train_world_model`, `standalone`, `check_dataset`, `train_action_model`).
No call site can bypass it; `None` = bit-identical full obs. (A second, separate incident: editing these files
mid-run poisoned the 19:04 process via cached-vs-fresh module skew → the 21:46 relaunch on settled code.)

## Planned — OOD detection on dock attempts (world-model as anomaly detector)

**Goal.** Use the trained world model to flag off-nominal docking. **OOD = another ship docked at a port
location NOT seen in training** — i.e. a *visual scene novelty* (a second vessel present at a novel port), not a
dynamics change. So the OOD signal should live primarily in the **image** modality (the model's frame
predictions are surprised by the novel docked-ship configuration); proprio may or may not move.

**Primary metric — Violation of Expectation (VoE).** The model predicts the next observation (its expectation);
score = the deviation of the ACTUAL observed frame/state from that prediction (one-step / short-horizon
prediction error, re-grounded — i.e. the `closed_loop_1` quantity, read primarily on the IMAGE, and on proprio
as a check). Nominal → low (in-distribution); OOD → high (novel scene). Chosen as the FIRST metric because it
needs no calibrated uncertainty — it just measures "was the model wrong" — which is exactly what a visual scene
novelty triggers. One rollout per episode, compared to the observed trajectory.

**Secondary metric — sampling variance.** K stochastic flow rollouts (diffusion `stochastic_eval`, different
noise) from the same context+actions; per-step across-sample variance; episode score = max over the horizon.
Epistemic-uncertainty signal; use to complement/backstop VoE. Caveat: if the committed rollout is
near-deterministic the spread may be too small to discriminate — verify meaningful spread before relying on it.

**Protocol (conformal-style, both metrics).**
1. Score every episode (nominal + attempts) by the metric, same model + procedure.
2. Split the 100 nominals: reserve **N** for the balanced test set (N = #OOD episodes), **calibrate the 90th
   percentile** of the score on the remaining ~(100−N) nominals ⇒ ~10% false-positive rate by construction.
3. Classify the balanced test set (N reserved nominals + N OOD attempts): score > threshold ⇒ OOD.
4. Report TPR on attempts at the fixed ~10% FPR, plus threshold-free ROC/AUC.

**Splits + selection (FINALIZED 08-19 after the divergence check + visibility scan — supersedes the earlier
collision-pair design; the "rungs"/graded tiers are DROPPED per user: overcomplicated).**
- **Nominal (in-distribution)** = `dock-success-100ep`: base ISS, **no** berthed vehicle (the training scene),
  all 8 ports. (Training ports {harmony_fwd_pma2, harmony_nadir_cbm, zvezda_aft, pirs_nadir, rassvet_nadir};
  held out {harmony_zenith_cbm, poisk_zenith, unity_nadir_cbm}.)
- **OOD (visual)** = `dock-attempts-by-port` episodes that (a) SUCCESSFULLY dock a FREE port (so the dynamics ==
  nominal) but (b) have an extra vehicle berthed elsewhere that IS IN THE FPV background — a pure *visual* scene
  novelty. NOT the 3 collision pairs (target-occupied → degenerate FPV pointing at space → dropped).
- **Divergence check (`logs/ood/divergence/`):** nominal-vs-OOD absolute-position trajectories are IDENTICAL
  (endpoints match 2–7 cm; fully interleaved approach funnels) because the berthed vehicle is off the approach
  corridor to a free port. ⇒ proprio carries NO OOD signal → it is a **NEGATIVE CONTROL**; the signal is
  image-only.
- **Visibility scan (`logs/ood/visibility_scan/`):** the vehicle is out-of-FPV for most (ship,port); visible only
  for specific approach corridors. Usable pairs (user-confirmed by eye):
    · **cygnus** @ harmony_fwd_pma2, rassvet_nadir, zvezda_aft — all TRAINING ports, strong.
    · **dragon** @ harmony_zenith_cbm — HELD-OUT port, accepted as MILD novelty (the WM sees much docking, a new
      port is not full OOD; the vehicle is the dominant novelty).
    · **soyuz** — not reliably visible → DROPPED.
  ⇒ **4 matched ports**: harmony_fwd_pma2, rassvet_nadir, zvezda_aft, harmony_zenith_cbm.
- **Matched-port protocol (kills the port-content confound):** calibrate + nominal-test only on these 4 ports, so
  the ONLY nominal-vs-OOD difference is the vehicle. Nominal matched pool ≈ 51 (harmony_fwd 13, rassvet_nadir 12,
  zvezda_aft 13, harmony_zenith 13). Split ≈ 34 calibrate / 17 nominal-test. OOD test = **30 balanced**
  (15 cygnus from its 60, 15 dragon from its 20).
- **Per-head VoE:** episode score = ~90th-pct-over-steps of the one-step re-grounded (closed_loop_1) prediction
  error, computed PER HEAD. IMAGE head = the detector; PROPRIO head = negative control (expect it NOT to fire —
  a built-in sanity check that the divergence result holds through the model).
- **Reads:** TPR at the ~10% FPR set by the 90% quantile; threshold-free ROC/AUC; TPR per ship + per port;
  per-head; and the **per-port NOMINAL VoE baseline** (to see whether held-out harmony_zenith's nominal is
  elevated — the mild port-novelty we accepted).
- Per-episode labels from `rollout.json` (docked/collided/escaped).

**Prerequisite.** The ANCHORED arm-A model (`relenc_A_coop`) — nominal VoE must be LOW/stable so OOD spikes stand
out. Eval script: `src/quickdraw/scripts/ood_voe.py` (numeric only, NO VTK/video renders — avoids the standalone
GL hang, memory `standalone-eval-diffusion-gl-hang`).

## Implementation status (08-19) — relative encoding + physical-loss hook (BUILT, default-off, pre-launch)

Both A/B ingredients are coded, default-off (rt10/rt1 unaffected), unit/model-level tested. NOT launched yet
(pending a "less data for faster iteration" decision + the real-data full-pipeline smoke on a freed GPU).

- **Relative position encoding** (`model.relative_position` + `model.relative_scale`, `position_idx` from env):
  anchor = window/context/trajectory first-step position, threaded as an ARG (never stored) through
  encode_state/to_obs/recon/roundtrip/forward/rollout/loss_terms/_step/imagine_eval/ae_floor. recon compares in
  the relative frame; roundtrip + val/eval de-relativize to absolute. `s_rel≈[0.107,0.133,0.113]` (norm
  within-window std; 8.6× magnitude shrink → floor 5.5→~0.6 projected). Model-level smoke: all paths run off+on,
  backward intact, off bit-identical. multimodal.py + lit.py + routines.py + setup.py.
- **Physical-loss durable hook**: `environments/base.py:continuity_residual` (shared v=dp/dt helper);
  `RecordedEnv.physical_loss` returns `{"continuity": …}` (velocity_idx auto-derived); `PhysicalLoss` variation
  generalized to Huber-penalize EVERY returned key; `physical_state` de-relativizes for arm B. Tested: residual
  0 for v=dp/dt, nonzero when broken. Docs: base.py hook contract + design/models/variations.md.
- **Process safeguard still owed before launch:** real-data full-pipeline smoke (train step + every eval routine,
  flags on) on a freed GPU — the obs_keep-bug catcher.

## A/B launched (08-19) — relative encoding (floor) × physical-loss continuity (drift), 1/3 data

Real-data smoke of both features PASSED after fixing 3 bugs the CPU/fp32 model-test missed: bf16 index_put
dtype (cast RHS to dest dtype), `VarContext.obs_seq` not `.obs`, and config-struct (`relative_position`/
`relative_scale` added to conf/model/mm_flow.yaml so no `+` needed). Both features engaged in the smoke
(roundtrip_proprio logged = anchor+relative active; physical_loss/continuity logged = drift term active).

| launched | tag | GPU | new feature vs prior | reads |
|---|---|---|---|---|
| 08-19 ~19:5x | `relenc_A_coop` | 0 | anchor@10 + **relative_position** (floor fix) | vs rt10 (anchor-only floor ~5.5) → relative's floor gain (proj ~0.6) |
| 08-19 ~19:5x | `relenc_B_coop` | 1 | A + **physical_loss.continuity=0.5** (soft drift lever, warmup 2ep) | vs arm A → drift gain on open_loop |

Both: bsp32mse ego-13 (obs_keep[2..14], sub5, proprio-mse) + `limit_train_batches=0.333` (3× faster epochs).
Eval suite = smoke-validated only (ae_floor + ood_horizon + manifold; denoising OFF). Levers stack (orthogonal
flags); relative/obs_keep/mse affect inference, anchor + physical-loss are TRAINING-ONLY (no inference effect).
LESSON (rt10): never `pkill -f` on a name that's a PREFIX of another (`coop_ego13_rt1` matched `..._rt10`) — use
exact match.

## Finding 6 (08-20) — the control failure is WM model-exploitation; a residual-PHYSICS proprio WM is the fix

MPPI overshoots because it maximizes reward on the WM's IMAGINED rollout, and the black-box WM mis-learns the
action-response (open-loop plots: the y/cross-track axis goes the WRONG way). It then commits to actions that move
reality the opposite way. Reward shaping is downstream noise; the lever is the WM's action-response fidelity.

**Verified the chaser dynamics against the ground-truth sim CODE** (`outofthisworldmodel-envs/.../envs/
iss_numerical/dynamics.py`), not just data:
- **Thrust is EXACT**: line 416 `accel_chaser += rotate_body_to_world(q_bi, u[0:3]) / mass` == `a = R(q)·F/m`,
  `q_bi=[w,x,y,z]` body→ECI, `mass=cfg.physics.mass=12000`. Data-fit agreed: cos 0.996, magratio 1.00.
- **NOT Clohessy-Wiltshire**: the env integrates BOTH chief+chaser as full ECI states (zonal J2–J6 + third-body
  sun/moon + drag, RK4) and DIFFERENCES them; the obs is the LVLH `relative_view`. CW is only the linearization of
  that relative motion → **CW dropped**; the residual learns the (near-linear) relative orbital drift itself.
- **Attitude**: `q̇ = ½ q⊗ω` (line 431); obs quat is body→world (q_bw), rates are ω_rel.
- **Subsampled (sub=5)**: `Δv = R(q)·(F_summed/mass)·raw_dt` (raw_dt 0.05); `p' = p + v'·dt_eff` (dt_eff=sub·raw_dt=0.25).

**#14 fixed**: `absolutize_proprio` de-relativization forced fp32 (was bf16 → ~1 m quantization on the ~100s-of-m
absolute position, which capped proprio error AND fed MPPI coarse states).

## Experiments 08-20 — physics-prior proprio WM (residual-physics), coop + noncoop

Recipe (all): mm_flow single proprio modality dim 13, d128 depth4 heads8 window32, `latent_norm=layernorm`
(REQUIRED for bespoke), `relative_position=true` scale [0.107,0.133,0.113], anchor `latent_loss_weight=10`,
subsample 5, F 64, batch 256 (autobatch off), cam=fpv (proprio-only loader skips frames), manifold off, 10 epochs.
Physics ON = `+environments.dynamics_prior=true +environments.quat_idx=[6,7,8,9]` (bodyrate idx auto-derived
[10,11,12]); gated in `owm_physics.py`, byte-identical off. eval OL horizon 128, closed-loop {1,16}. Launcher:
scratchpad `launch_proprio.sh` (`GPU=<n> DATA=<dataset> bash launch_proprio.sh EXP FOURIER PHYSICS <5 summaries>`).

| run | data | phys | fourier | extras | result |
|---|---|---|---|---|---|
| A_baseline | coop | off | off | — | val_ptw→3.03; eval OL 10.43 (black-box) |
| B_fourier | coop | off | 8 | — | worse (fourier HURTS): OL 10.96, ptw 3.36 |
| C_physics (v1) | coop | on | off | thrust-only, teacher-forced loss | best of the 4; OL 9.78 (BB); **chained** OL **2.94**, cl1 **0.066**, cl16 **0.072** (tracks GT on x/z; y plateaus = thrust-only) |
| D_physicsfourier | coop | on | 8 | — | fourier hurts again: OL 10.98, ptw 3.76 |
| **C2** (v2) | coop | on | off | **+attitude kinematics +p_tf-chained loss** | RUNNING (GPU0). val_phys 32.9→0.25→0.065→0.050 (ep0-3) |
| **C2_noncoop** | noncoop | on | off | same as C2 | RUNNING (GPU1) — for MPPI benchmark's evasive-target scenario |

**Two fixes over C_physics v1 (→ C2):** (a) BAKE exact attitude kinematics `q'=q⊗exp(½ω dt)` — verified on data
**0.006°/step vs 0.42° copy (49×)**; (b) p_tf-RESPECTING chained physics loss (`physics_proprio_chained`): the
residual trains on the SAME autoregressive compounding regime the eval rollout uses, closing the
teacher-forced-train / AR-eval exposure gap that left C_physics-v1's y-axis plateauing.

**C2 is a PHYSICS-CHAIN-ONLY model.** With the weight-1.0 physics loss the model routes proprio accuracy through
the physics chain, so the black-box latent decoder degrades: `val_ptw` (pointwise err via the black-box decode,
NOT the physics path) 22→17→12→6.5 while ABCD sit ~3. This is FINE — MPPI uses the physics chain. The metrics that
matter are `val/loss/physics/proprio` (the AR physics-chain rollout error, normalized) and the physics-chained
`eval_ood_horizon` (meters; fires at epochs 5 and 9). `val_ptw` is a red herring for physics models — and
`checkpoint_metric=pointwise_error` selects best.ckpt on it, so for MPPI use the epoch with lowest val_phys /
best chained eval, NOT best.ckpt. (Open q: we still TRAIN decode/proprio + codec roundtrip though the chain doesn't
use the decoder — droppable in a future run; the residual reads the raw latent token.)

**imagine_eval / imagine_shared UNIFIED + two MPPI bugs fixed.** They were two divergent rollout paths;
`imagine_shared` (the ONLY rollout MPPI uses, controller/mppi.py) (1) never passed `norm` → scored the black-box
dynamics, and (2) never threaded the rel-position anchor → fed absolute positions through a relative encoder for
relative_position=true = garbage. **Prior MPPI runs on this model are invalid.** Fix: one core `_imagine`
(imagine_eval=K=1, imagine_shared=K≥1 fan-out), physics shared via `_physics_proprio_rollout` (INVARIANT note),
anchor threaded for all K, mppi.py passes `norm`. PARITY VERIFIED bit-identical (imagine_eval==imagine_shared(K=1),
max|d|=0). Branch `physics-prior` @ 6d982b6 (pushed to origin).

**MPPI benchmark handoff (owm-bench agent):** (1) use the C2 checkpoint chosen by val_phys, NOT best.ckpt; (2)
build the model WITH the hook (`+environments.dynamics_prior=true +environments.quat_idx=[6,7,8,9]`) — else
`dynamics_prior=None` and it silently black-boxes; (3) ALWAYS pass `norm` to imagine_shared/imagine_eval; (4) feed
fp32 ctx obs, no outer bf16 autocast (#14); (5) I/O contract: ctx `(B,P,13)` normalized fp32, actions
`(B·K,P-1+H,6)` normalized (action for step t at index P-1+t), returns normalized proprio → denorm → reward; (6)
re-baseline from scratch (old numbers invalid).

---

## 08-24 — DreamerV3-style probabilistic latent heads (bimodal A/B), + B GAUSSIAN-KL COLLAPSE

New modular prob-head component (`src/quickdraw/models/dist_heads.py` + `MultiModalDistribution` in `multimodal.py`,
config name `mm_dist`): parameterizes an EXPLICIT distribution over the next JOINT latent bag (proprio+image, all
n_state tokens), samples it, decodes — mutually exclusive with the flow head. Categorical (DreamerV3: 32×d/16 groups,
unimix 1%, straight-through, balanced KL β_dyn 1.0/β_rep 0.1, free-bits 1 nat) is primary; Gaussian-KL (stochastic
latent) and Gaussian-NLL (Ward-2026, deterministic latent) are siblings. Posterior conditions on the fused
single-frame obs (action-excluded); prior from the transformer spine. Recon = **posterior-decode single term, no
separate anchor** (modality `weight:0` drops predicted-bag decode; `latent_loss_weight:1` keeps `codec/roundtrip`
= decode(sample(posterior(o_t)))→o_t). Physics-proprio override inherited unchanged. Smoked 41/41 (KV-parity
Δ~1e-6) + 2 independent fable audits.

**A/B (16ep coop, physics on, taesd image + ego-13 proprio, batch 26, compile):** A=categorical (GPU0),
B=gaussian-KL (GPU1). Killed at ep3 for audits.

| run | head | ep3 val total | kl/dyn raw | codec_proprio | status |
|---|---|---|---|---|---|
| **A** | categorical | 14.8 | **12.6** | 0.91 | HEALTHY |
| **B** | gaussian-KL | 28213 | **32109** | 8.68 | **COLLAPSED** |

**B collapse — CONFIRMED empirically (loaded ep3 ckpt + 1 val batch), NOT a units artifact:** ~99.97% of B's loss
is the KL term. Posterior σ collapsed to the `min_std=0.1` floor (mean 0.103) AND posterior mean exploded to ±64
(bf16 saturation) while the prior stayed moderate (σ~0.72, μ~14) → KL `(μq−μp)²/2σp²` runs away (bag std 4.9,
|max| 62, **Inf grads**). ROOT CAUSES (2 audits): (1) `latent_norm:affine` does NOT normalize a POSTERIOR-SAMPLED
bag — `_ln` is gated off for affine, whose premise (bag = invertible AE latent) is false for a net-output sample;
(2) Gaussian posterior mean = unbounded `Linear(feats)`, no `max_std`, variance→floor; (3) 128-dim SUMMED KL vs
mean-mse recon → ~2500:1 gradient imbalance. Categorical is IMMUNE (bounded simplex + unimix) — **this A/B
reproduces DreamerV3's reason for choosing categorical over Gaussian for the WM latent.** Fix B before re-running:
force LN on the sampled bag (or `latent_norm:layernorm`) + `max_std` + rescale dyn/rep to recon units.

**Bugs found by the audits (this session):** (a) FIXED `multimodal.py:_physics_proprio_rollout` — bf16 (autocast)
bag token fed to fp32 `residual_head` OUTSIDE autocast → dtype crash; only the STANDALONE eval path hit it (train/val
run under Lightning autocast). Caught by running eval_ood_horizon on A's ep3 ckpt (first eval ever on mm_dist).
(b) OPEN sibling: `physics_proprio` (dead code) has the same uncast token. (c) OPEN `eval_manifold.py:44` never
`norm_act`s → silent garbage for ALL MM models on owm. (d) OPEN GaussianHead losses computed in bf16 while
categorical's softmax is fp32 → a precision CONFOUND in the very A/B. (e) `p_tf_batch_granular` absent from both
dist yamls → silently True (proven recipe locks false). (f) recon-posterior-decode decision verified in force to 4
decimals. checkpoint_monitor=val/loss/physics/proprio confirmed wired.

**Decode/physics UNIFICATION landed (08-24).** `physics/proprio` is no longer a bolt-on — it's `decode/proprio`
with a per-modality `prior ∈ {none, identity, physics}` (env supplies the fn). Bit-identical verified
(decode/proprio physics == old physics/proprio 2.494110, Δ<5e-7; smoke 41/41). Round-trip illegal in prior mode
(build-time raise; encoder grounded by the decode loss). Legacy `dynamics_prior=true` configs byte-identical (dual
path). Queued categorical dreamer runs (coop+noncoop, bespoke conv/unet AE 32 tok, layernorm, prior:physics FULL,
decode/image ON = image-collapse fix, evals, 16ep). Loss terms: kl/dyn(1)+kl/rep(0.1)+decode/proprio(1,physics
chain)+decode/image(1,absolute)+codec/roundtrip_image(1). No proprio round-trip, no standalone physics term.

**PINNED — physics-rung sweep (later).** Measured: FULL 6-DOF physics leaves the residual ~nothing to learn
(residual 0.4% of position, no error reduction). Test how much physics to give by toggling the prior rung, all
else fixed: FULL (default) · kinematics-only (integrate v→p & ω→q; WM learns force→v AND τ→ω — the symmetric
"less") · identity (obs=prev+residual; pure delta) · none (absolute). **SKIP translation-only** (boring +
quat-norm hazard). Needs small config flags on the physics fn (rotational_dynamics / attitude_kinematics on/off)
to build the reduced priors — not yet wired.
