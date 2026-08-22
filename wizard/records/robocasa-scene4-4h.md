# robocasa-scene4-4h — running log

Every run on this dataset in order, what it showed, and what we decided next. Newest at the bottom.
Dataset: `isaac-ronald-ward/robocasa-scene4-4h`, 128px, camera `robot0_agentview_left`, `RecordedEnv`
(replayed trajectories — it **cannot** `step`, so control eval is permanently off).

**The goal, as stated by the user (2026-08-10):** *slightly sharper predictions that stay coherent for
32–64 steps, with confidence that motion is actually being modelled.* 32–64 is the trained rollout length
(`F=64`). Everything before 2026-08-10 was scored on a 791–2048 step horizon, which is ~12× past that and
was actively misleading.

---

## 1. The 14.3 dB wall (pre-2026-08-09)

Four configurations — varying encoder, decoder, BPTT window, sampler — all pinned at **14.3–14.8 dB** image
PSNR. The bespoke conv/U-Net autoencoder's own reconstruction floor was ~14.3 dB.

**Decision:** the world model was saturating its tokenizer; all headroom was in the tokenizer, not the
dynamics. Move to a pretrained one.

## 2. TAESD + a broken adapter (issue #12, first attempt)

Frozen TAESD bridged to the token bag by a *learned Perceiver* adapter. Round-tripped at **10.49 dB after
nine epochs** — while a *randomly initialised* projection scores 11.10. Training the adapter bought
literally nothing and capped every image metric far below the tokenizer's ceiling.

**Decision:** rewrite the adapter as a parameter-free index rearrangement + zero-init residual, so it is
bit-exact identity at init. Verified 23.92 dB at init = the raw frozen AE on the same frames.

## 3. `taesd_exact` — the wall breaks, then collapses (08-09 04:56)

`logs/robocasa-tok/train_world_model_2026_08_09_04_56_48_taesd_exact`

| epoch | one-step PSNR | open-loop PSNR |
|---|---|---|
| 0 | 11.76 | — |
| 1–3 | 17.80 → 18.36 | 12.21 → 12.92 |
| 4 | **18.39** | 12.30 |
| 5 | **11.63** | 10.55 ← collapse, never recovered |

**Two findings that shaped everything after.**
- **Single-step and long-horizon are decoupled.** The tokenizer swap bought **+6.6 dB single-step and
  +0.1 dB at horizon**. The largest per-step win available moved the horizon metric not at all.
- **`loss/roundtrip/image` read exactly `0.0` at every epoch** — it was measuring a norm-free path the model
  never runs, so the adapter had never been trained to invert LayerNorm.

**Decisions:** (a) any lever that only improves single-step is a poor bet for horizon — dropped
`flow_hidden` and `depth`; (b) fix the round-trip to route through `encode_state`/`to_obs`; (c) the collapse
was un-diagnosable because `last.ckpt` had frozen at epoch 4 — fixed so it is written unconditionally
every epoch.

## 4. num_tokens ↔ d see-saw (08-09 22:15, fit-start only)

`logs/robocasa-tok/*nt{4,8,16,32,64}*` — floors at init, frozen TAESD:

| bag | mode | floor | LN tax |
|---|---|---|---|
| 4×256 | EXACT | 20.79 | 3.13 |
| **8×128** | EXACT | **20.41** | 3.51 |
| 16×64 | EXACT | 20.01 | 3.91 |
| 32×32 | EXACT | 19.72 | 4.20 |
| 64×16 | EXACT | 18.68 | 5.24 |
| 32×128 | **PADDED** | **16.03** | 7.89 |

**Decisions:** (a) `num_tokens × d == L` exactly, **never pad** — padding is *active damage*, not idle
width, because LayerNorm is per-token so the zeros enter the statistics the real floats are divided by
(−4.4 dB, worse than any token-count choice); (b) storage is definitively not the constraint — 8×128 holds
all 1024 latent floats; (c) no clean `num_tokens` A/B exists at a fixed budget, since raising tokens always
shrinks the backbone (`~d²`). Written up in `design/capacity.md`.

## 5. `seesaw_wide` vs `seesaw_ctrl` — capacity (08-09 22:35, killed ep1/ep4)

4×256 (~4.2M backbone) vs 8×128 (1.06M) at a fixed 1024-float budget.

Control led at **every** epoch (ep0 15.22 vs 13.71). **Decision:** dynamics capacity via the width/token
see-saw is not the lever; killed `wide`. Also first sighting of `codec/roundtrip_image` **rising**
(0.0043 → 0.0058) — the dynamics dragging the adapter off fidelity.

## 6. `df_off` vs `df_on` — diffusion forcing (08-10, killed ep4/30)

`logs/robocasa-df/train_world_model_2026_08_10_08_58_57_{df_off,df_on}` (an earlier 04:03 pair of the same
arms died at ep1 — see *incidents*). Full results are in `robocasa-df.sh`.

|  | ep0 | ep1 | ep2 | ep3 |
|---|---|---|---|---|
| one-step, off | 12.50 | 17.68 | 17.95 | **18.20** |
| one-step, on | 11.04 | 17.38 | 17.65 | 17.79 |
| open-loop, off | 9.20 | 12.05 | 12.28 | **12.70** |
| open-loop, on | 8.82 | 12.48 | 12.82 | 12.25 |
| **val `dynamics/latent`, off** | 0.2833 | **0.2961** | **0.2961** | **0.2966** |
| val `dynamics/latent`, on | 0.8644 | 0.3870 | 0.4202 | 0.5256 |
| **ae_floor (codec NOW), off** | 23.52 | 22.23 | 21.88 | **21.63** |
| ae_floor, on | 23.02 | 21.75 | 21.58 | 21.29 |

**Three findings.**
1. **DF at 0.1 does not help.** Trails on one-step at every epoch, LPIPS a wash, and its train-noised vs
   val-clean gap *widens* every epoch — reproducing what `conf/config.yaml` already recorded for 0.25/1.0.
   It also does **not** simulate autoregression: it mixes in *isotropic Gaussian* noise, while real rollout
   error is structured and drift-shaped.
2. **The dynamics stopped learning at epoch 1.** `dynamics/latent` flat to 4 s.f. across three epochs, and
   it is ~96% of the objective. Every gain in `val loss total` came from `decode/image`, not the dynamics.
3. ***The codec is eroding.*** `eval_ae_floor` — the *current* encode→decode quality — falls ~2.3 dB in four
   epochs, and `closed_loop_1_steps` tracks it almost exactly (22.45 → 21.13). **One-step prediction was
   pinned to a ceiling training was destroying.** `dynamics/latent` and `codec/roundtrip` share the same
   0.134M trainable adapter params and the dynamics wins.

**Decisions:** drop DF. Attack the erosion. And fix the measurement, because —

## 7. The measurement gap (08-10)

Readouts fired at **quarters of H** only, so at H=791 they were `@+197/394/591/791` — **no number at step 32
or 64 anywhere.** Every headline figure quoted for two days was a mean over 791 steps. And nothing tested
whether *motion* was modelled: no persistence baseline existed, so a model predicting "next = current"
would score respectably.

**Landed:** readouts at `@+1,8,16,32,64` ∪ quarters; `eval.horizon` 2048 → 128; `psnr_frozen` (hold frame 0
— the do-nothing baseline); `motion_ratio` (‖Δpred‖/‖Δtrue‖, 1 = right amount of motion, <1 = freezing).
Validated: a frozen predictor scores `motion_ratio` 0.000 and `psnr` *exactly* equal to `psnr_frozen`; a
perfect one scores 1.000.

## 8. `tf_ln` vs `tf_affine` — token-mixing denoiser (08-10 19:36, KILLED ep1/ep2)

`logs/robocasa-tfaff/train_world_model_2026_08_10_19_36_25_{tf_ln,tf_affine}` — see
`robocasa-tf-affine.sh`. Both `flow_arch=transformer`; arms differ only in `latent_norm`
(layernorm floor 20.41 vs affine floor 23.92). Score arm B as **gap-to-floor**.

Rationale: a dynamics loss flat to 4 s.f. is what a *representational ceiling* looks like, and the
per-token MLP denoiser (0.072M) treats the 8 tokens independently given frozen conditioning — it cannot
represent correlated motion across the bag. The transformer is also **the only lever with evidence from
outside this machine** (reported significantly better than mlp on the user's other setup). Affine removes
the 3.51 dB LayerNorm tax, which is the only reason the adapter residual must learn anything at all.

---

## 9. `tf_affine` vs `tf_bespoke` — frozen vs learned tokenizer (08-11 03:31, killed ep3/ep2)

`logs/robocasa-tfaff/train_world_model_2026_08_11_03_31_45_{tf_affine,tf_bespoke}`

Both `flow_arch=transformer`. Arm A frozen TAESD + affine; arm B a bespoke conv/U-Net AE trained from scratch
(+ layernorm, because affine presupposes a FIXED latent a learned encoder does not have).

**The inversion, at ep0.** The bespoke codec is **8.6 dB worse** (15.29 vs 23.88) and it was **2 dB BETTER at 64
steps** (9.06 vs 7.06), with `motion_ratio@+64` **1.117 vs 0.589**. Sharp-but-static versus blurry-but-moving,
exactly as read off the videos. A reconstruction-optimised frozen latent is a hostile prediction target.

**Then BOTH diverged**, and the cause was not the tokenizer:

```
val/loss/total    tf_affine  0.711 0.420 0.863 1.825      tf_bespoke  0.824 1.502 3.202
grad/norm/flow    tf_affine  0.98  1.48  766   21         tf_bespoke  2.10  65.6  10151
ae_floor          tf_affine  23.88 22.42 18.94 16.37      tf_bespoke  15.29 10.69 11.47
lpips@+1          tf_affine  0.173 0.254 0.450 0.627
```

`_TokenMixBlock` did not zero-init its residual branches, so it perturbed the velocity field from step 0 inside
a graph ~32 BPTT steps x 6 ODE steps deep — gain >= 1 per application, compounded ~192 times.
`gradient_clip_val=1.0` then **laundered** it: `norm_postclip` was exactly 1.000 every epoch, so training took
full-size confident steps along the exploded direction and degraded SMOOTHLY instead of NaN-ing. That is why it
read as a modelling failure for two days.

**Decisions:** zero-init both residual branches (identity at init, wakes after one step); log
`grad/clip_ratio = preclip/clip_val` as the number that makes this legible; and the per-token MLP denoiser is
the only one with four epochs of stable gradients (0.43-0.85), so it is the safe baseline.

## 10. `mlp_affine` vs `tfz_affine` — mlp vs FIXED transformer (08-11 16:03, RUNNING)

`logs/robocasa-tfaff/train_world_model_2026_08_11_16_03_52_{mlp_affine,tfz_affine}`

Byte-identical except `flow_arch`. **The first clean mlp-vs-transformer test** — every earlier comparison was
confounded by normalization and/or batch size.

|  | mlp ep0 -> ep1 | transformer(zero-init) ep0 -> ep1 |
|---|---|---|
| **ae_floor (CODEC)** | 24.31 -> **21.52**  (-2.8) | 25.06 -> **24.51**  (-0.55) |
| val `codec/roundtrip` | 0.0038 -> 0.0070 (+84%) | 0.0033 -> 0.0037 (+12%) |
| **val 1-step PSNR (PREDICTIVE)** | 12.47 -> 16.74 | 15.55 -> **18.29** |
| closed_loop_1 | 22.61 -> 20.95 | 22.92 -> **22.67** |
| open_loop @+32 | 10.39 -> 13.91 | 11.17 -> **14.39** |
| open_loop @+64 | 8.11 -> 12.97 | 9.83 -> **13.64** |
| (bar: `psnr_frozen@+64`) | 10.40 | 10.40 |
| motion_ratio @+64 | 0.769 -> 0.192 | 0.700 -> **0.126** |
| lpips @+1 | 0.207 -> 0.384 | 0.132 -> **0.166** |
| val loss total | 0.708 -> 0.517 | 0.484 -> **0.254** |
| grad/norm/flow | 0.602 -> 0.906 | 0.593 -> **0.571** |

**1. The zero-init fix holds so far.** `grad/norm/flow` 0.593 -> 0.571, where the broken run went
0.98 -> 1.48 -> 766. It is now BELOW the mlp. ep2 is the real test (that is where 766 appeared).

**2. A STRONGER DENOISER PROTECTS THE CODEC — the headline result.** The transformer holds `ae_floor` at 24.51
(-0.55 dB) while the mlp erodes to 21.52 (-2.8 dB), and `codec/roundtrip` rises 12% vs 84%. Interpretation: the
weaker per-token MLP cannot fit the dynamics, so it leans on the shared adapter to reshape the latent into
something easier — paying codec fidelity for dynamics loss. The transformer is capable enough not to need that.
This also re-frames section 8: what was called "the dynamics destroying the codec" is really "an underpowered
denoiser destroying the codec".

**3. Both arms now BEAT the do-nothing baseline at 64 steps** for the first time: 12.97 and 13.64 against
`psnr_frozen@+64 = 10.40`. Predictive PSNR is genuinely better, and the transformer leads everywhere
(+1.55 dB one-step, +0.67 dB at 64 steps, half the val loss, less than half the lpips@+1).

**4. But motion is COLLAPSING as they train** — `motion_ratio@+64` 0.769 -> 0.192 (mlp) and 0.700 -> 0.126
(transformer). They are getting SHARPER AND MORE STATIC. Beating persistence by predicting a
slightly-less-frozen scene is not the goal, and the transformer is the *worse* of the two on motion.

**Consequence -> action conditioning** (design/action_conditioning.md): `grad/norm/act_enc` was 0.0010 against
the flow's 0.5932, i.e. **0.17% of the total gradient**. `readout()` was slicing off the action token's output
entirely, so actions reached a prediction ONLY via attention onto 1 of 10 slots.
`diffusion.concat_action_embedding` (default ON) now concatenates the RAW pre-backbone `act_enc(act)` onto every
state token; `model.action_fourier_freqs` and `modalities.<i>.fourier_freqs` (both default 0/off) add sin/cos
bands. NOT bit-identical to these runs — they predate it.

## 11. `tfz_act` vs `tfz_act_fourier` — action conditioning (08-12, 3rd launch RUNNING)

The first test of the §10 consequence. Both arms carry the corrected conditioning; they differ **only** in
`model.action_fourier_freqs` (0 vs 16). `_cond` now returns **3 channels** per state token — the state, the
action token's backbone output (action × state, no longer sliced away), and the raw pre-backbone
`act_enc(act)` — so the flow's `in_dim` went 288 → 544.

Launched three times: 01:21 (restarted at 50% of ep0), 02:22 (**completed ep0 and ep1, then killed** — see
below), 07:19 (current, 12 epochs).

### The result so far — the mechanism engages, the payoff is small

Salvaged from the killed 02:22 pair, against §10's `tfz_affine` which has no action conditioning:

| | `tfz_affine` ep0→ep1 | `tfz_act` ep0→ep1 | `tfz_act_fourier` ep0→ep1 |
|---|---|---|---|
| `grad/norm/act_enc` | 0.0010 → 0.0089 | 0.0117 → **0.0498** | 0.0206 → **0.0661** |
| `motion_ratio@+64` | 0.700 → 0.126 | 0.589 → **0.161** | 0.665 → **0.163** |
| val 1-step PSNR | 15.55 → 18.29 | 15.87 → 18.28 | 15.83 → **18.55** |
| `open_loop psnr@+64` | 9.83 → 13.64 | 10.28 → 13.69 | 10.37 → **13.79** |
| val loss total | 0.4841 → 0.2544 | 0.3757 → 0.2651 | 0.3979 → 0.2540 |

**5.6× more gradient reaches the action pathway (7.4× with Fourier)** — at no cost to sharpness or loss.

### CORRECTION (08-12 later): the motion gain did NOT replicate. Only the gradient did.

The 3rd launch is a **same-config, same-seed replicate** of the killed 2nd launch (verified by diffing the
resolved configs — the only difference is the new `action_squash: none` key, which is prior behaviour). So
there are now n=2 per arm, and the replicate spread is **larger than every effect claimed above**:

| ep1 | `tfz_affine` | `tfz_act` 2nd / 3rd | `tfz_act_fourier` 2nd / 3rd |
|---|---|---|---|
| `motion_ratio@+64` | 0.126 | 0.161 / **0.131** | 0.163 / **0.131** |
| `open_loop psnr@+64` | 13.64 | 13.69 / **13.60** | 13.79 / **12.98** |
| val 1-step PSNR | 18.29 | 18.28 / **18.55** | 18.55 / **18.06** |
| `grad/norm/act_enc` | 0.0089 | 0.0498 / **0.0520** | 0.0661 / **0.1441** |

Same config gives `motion_ratio` 0.161 vs 0.131 and `open_loop@+64` 13.79 vs 12.98 (**0.81 dB of pure
noise**). The live pair's motion (0.131) is indistinguishable from the no-action-conditioning baseline
(0.126). **Noise floor: ±0.03 on `motion_ratio@+64`, ±0.8 dB on `open_loop@+64`, ±0.5 dB on 1-step.** Every
future single-seed claim on this dataset must clear those.

Only `grad/norm/act_enc` is reproducible: 5.8–16× in both replicates of both arms. **The plumbing works and
moves no outcome metric.** §12 explains why.

Also retracted: the reading that ep0 (`p_tf=1`) "moves the arm" and the model trades motion for PSNR.
`motion_ratio = ‖Δpred‖/‖Δtrue‖` is a **magnitude ratio — direction-blind** — so noise scores high, and ep0
scores `psnr@+64 = 9.64` against a do-nothing baseline of 10.40, i.e. worse than holding frame 0. `tf_ln`
ep0 reads 2.060 and `tf_bespoke` 1.117; those are noise, not motion. ep0 is an unconverged predictor, not a
moving one, so the universal ep0→ep1 motion drop is much weaker evidence than it looks.

### Why it died, and the two fixes

`eval_denoising_filmstrip` hand-built the conditioning as `h[0, :m.n_state, :]` (width `d`) instead of
calling `_cond`, so the moment `in_dim` became 544 it raised
`mat1 and mat2 shapes cannot be multiplied (9x288 and 544x128)` — every eval, deterministically. It now
routes through `m._cond(h, act[:, t_ctx])`. **`_cond` is the single source of truth for the conditioning
width; any call site that reimplements it is a latent break.**

Then `EVAL_FAIL_LIMIT` did what §10's incident asked for and **raised**, destroying two arms that had
4.5 h of good training, over a *filmstrip*. That trade is wrong: the escalation exists to stop a run
silently emitting nothing, and **disabling the routine** achieves that without discarding the expensive
part. It now adds the routine to `_disabled`, logs one loud line + `eval/disabled/<name>`, and continues.

### Hardening done before leaving them overnight (user asked: "anticipate further problems")

- **All six enabled routines probed green** against a same-architecture checkpoint, out-of-process
  (`ae_floor`, `ood_horizon`, `manifold`, `denoising_filmstrip`, `denoising_multistep`,
  `denoising_aggregate`). The suspected second instance of the filmstrip bug — `manifold_clouds` calls
  `m.flow.sample(h_pro)` at width `d` — is **unreachable** here: `denoising_multistep`/`_aggregate`
  clean-skip on a joint transformer flow before reaching it. Full eval cost ~4 min/epoch.
- **`wizard/scripts/wd/watchdog.sh`** resumes a dead arm from its own rolling checkpoint (≤3 attempts,
  15 min settling window, stalls are reported but never killed). Two traps found by testing it:
  - **A naive resume OOMs instantly.** `train_world_model` skips autobatch on resume but nothing re-injects
    the chosen batch, so `data.batch` falls back to the config default of **1024** against a chosen 32. The
    watchdog reads the chosen value out of `config.resolved.yaml` and refuses to resume if it can't.
  - **`last.ckpt` goes stale after a resume.** ~~Lightning writes the rolling checkpoint as `last-v1.ckpt`
    (then `-v2`) and leaves `last.ckpt` frozen at the pre-resume epoch.~~ **REFUTED 2026-08-19 by audit.**
    That only happens when the run dir was MOVED or COPIED. `ModelCheckpoint` restores
    `best_model_score`/`last_model_path` only if its `dirpath` EQUALS the one stored in the checkpoint
    (lightning `model_checkpoint.py:556-572`); on a moved dir that state is lost, which is both why the
    score read `nan` and why the version counter bumped to `-v1`. Measured IN PLACE: the rolling file is
    REUSED (no `-v2`), `last.ckpt` is untouched, and the monitor restores as a real number (0.26734).
    **The real lesson is the opposite one: pass the resume path ABSOLUTE and identical to the original, or
    top-k and best-checkpoint tracking silently reset.** Taking the newest `last*.ckpt` is still kept as
    cheap insurance. Not `epoch=*.ckpt` — those are top-k by val metric and can be stale.
  - Verified end-to-end: a resume of a **copy** of the dead run restored to epoch 1, wrote
    `epoch=1-step=7585.ckpt` (7582 + 3 limited batches — the step counter restores), exit 0. It also
    OOM'd `manifold` (8.07 GiB in one allocation, the largest of any routine) because the test shared a
    GPU — an unplanned but real demonstration that an eval OOM is now survived rather than fatal.

### Timing

ep0 ~1.33 h, but **ep1+ cost ~2.9 h**: ep0 runs at `p_tf=1` (teacher-forced, hits the P1 fast path) and
`p_tf_warmup_epochs=1` drops it to 0 afterwards, so the full AR rollout only starts at ep1. 12 epochs is
therefore ~33 h, not ~17 h.

**Known record-keeping flaw:** the launch script writes `$OUT/<arm>.out` with `>`, so relaunching an arm
overwrites the dead run's stdout — the reason the 01:21 pair was restarted is no longer recoverable. The
watchdog appends to a distinct `<arm>.resumeN.out` instead.

## 12. Action-sensitivity probe — the model uses action DISTRIBUTION, not action ORDER (08-12)

A one-off on the §11 ep3 checkpoints (`src/quickdraw/_oneoff_action_sensitivity.py`, a NEW file so it could
not tear the live runs' cached imports). Asks the question `motion_ratio` cannot: hold the context fixed,
change only the **commanded future**, and see whether the imagined future changes — in the latent bag and in
pixels separately, because `to_obs` decodes the very bags `_rollout` returns.

**The control that makes it mean anything:** `stochastic_eval: true`, so two rollouts with identical actions
already differ. Every variant is rolled under the same reseeded RNG, and **3 same-actions/different-seed
draws** give the noise floor. Floor spread is only 1.04–1.28× on pixel MSE, so ratios above ~1.3 are real.
Run in fp32 (no autocast) — bf16 nondeterminism is another noise source. Only actions from index `P-1` on
are perturbed, so the context and its aligned actions stay intact.

### Finding 1 — there is NO latent/pixel gap, so recon capacity is NOT the lever

Latent and pixel divergence track each other at every horizon, and where they differ the decoder
**amplifies** the latent difference rather than washing it out (Fourier arm @+64: latent 2.18× floor →
pixel 9.81×). The "dynamics is action-sensitive but the decoder hides it" hypothesis is dead.

### Finding 2 — order-only perturbations are free at +64; distribution changes are not

A random permutation is a WEAK perturbation here and the first run of this probe was confounded by it:
robocasa actions are near-smooth (**lag-1 r = +0.988**, consecutive steps differ 13%, dim 3 constant, dim 4
binary, dims 8–10 near-dead), so a permutation is 0.771 relative L2 against 1.936 for a clip swap and
**exactly 0 in >10% of windows**. Fixed by adding reversal + half-window shift; all five perturbations then
sit at 1.00–1.43 relative L2, i.e. matched. `tfz_act` @+64, response PER UNIT of action change:

| perturbation | keeps action multiset? | pixel/pert | ΔPSNR vs GT (reseed band 16.96–17.11) |
|---|---|---|---|
| `shuffled_time` | yes | 0.13 | 17.03 — **inside noise** |
| `shift_half` | yes | 0.12 | 17.07 — **inside noise** |
| `reversed_time` | yes | 0.28 | 17.11 — **inside noise** |
| `other_clip` | no | **1.34** | 16.15 (−0.83 dB) |
| `zero` | no | **1.29** | 16.50 (−0.48 dB) |

**At 64 steps you can REVERSE the action sequence and the prediction is no worse.** 5–10× differential at
matched perturbation size. And the horizon structure matches the freeze: at **+16** `reversed_time` costs
−0.6 dB (18.41 → 17.84) at 1.27× floor, so ordering does matter there. **The model tracks action timing for
~16 steps, then falls back to action statistics** — the same horizon where `motion_ratio` bottoms out.

### Finding 3 — CFG would amplify the WRONG axis, and Fourier already demonstrates it

CFG amplifies `v_action − v_null`, which is exactly the true-vs-`zero` axis. The two arms differ only by the
Fourier expansion and are wildly apart on it:

| | `zero` pixel/pert @+64 | `zero` ΔPSNR @+64 | order-only pixel/pert @+64 | `motion_ratio@+64` |
|---|---|---|---|---|
| `tfz_act` | 1.29 | −0.48 dB | 0.12–0.28 | 0.168 |
| `tfz_act_fourier` | **9.81** | **−3.96 dB** | 0.20–0.26 | 0.154 |

**7.6× more response on precisely CFG's axis, and it bought nothing**: same order-insensitivity, no better
motion, no better PSNR. Most likely 384 sin/cos bands make a sustained zeroed action sequence OOD, so
−3.96 dB is brittleness, not comprehension. **CFG + action dropout is therefore NOT the recommended next
move** — it was proposed (§11 follow-up, `design/ideas.md`) on the theory that the model needed an
action-conditional difference manufactured; it has one, on the axis that does not matter.

### What this supports, and the limit that can't be worked around here

The lever the evidence points at is **long-horizon order sensitivity**: penalise the rollout for being
invariant to action order (roll true + permuted actions, penalise their similarity). It optimises exactly
the measured quantity, costs ~2× rollout, and is neither motion-weighted recon nor inverse dynamics.

**But the ceiling on it is unmeasurable on this dataset.** With lag-1 r = 0.988 the true future may itself
depend only weakly on action ordering, and `RecordedEnv` cannot `step`, so the counterfactual true future
under reordered actions cannot be generated. We would be optimising order-sensitivity without knowing how
much is warranted — an argument for a small steppable-sim dataset, not a bigger run here.

**Cheapest action regardless:** promote this probe to a tracked eval routine. ~2 min, and a far sharper
diagnostic than `motion_ratio` (direction-blind, whole-frame). Blocked only because `evaluation/*` cannot be
edited while the runs are live.

## 13. THE MECHANISM: at 20 Hz the motion we ask for is BELOW our own codec's error floor (08-12)

Prompted by a literature review (agent, 08-12) whose headline was that every working robot world model
subsamples to 2–5 Hz. Checked our own data and it is decisive.

**`fps = 20`** (`meta/info.json`, both splits). So `F=64` — everything we have called "long horizon" all
week — is **3.2 seconds of real time**. Published long-horizon controllability runs at 2–5 Hz, where 64
steps is 13–30 s.

Measured on val, against the codec ceiling of 23.92 dB (per-pixel RMSE **0.0637**):

| stride | seconds | frame-delta RMSE | vs codec floor | % pixels moving > codec error |
|---|---|---|---|---|
| **1 (what we train)** | 0.05 | **0.0389** | **0.61×** | **3.18%** |
| 2 | 0.10 | 0.0570 | 0.90× | — |
| 4 | 0.20 | 0.0788 | 1.24× | 8.08% |
| 5 | 0.25 | 0.0864 | 1.36× | — |
| 8 | 0.40 | 0.1028 | 1.61× | 12.21% |
| 32 | 1.60 | 0.1506 | 2.36× | 21.24% |

**At 20 Hz the per-step motion is 0.61× the reconstruction error of the autoencoder we predict through, and
only 3.18% of pixels move more than that error.** The signal is below the noise floor. Hedging to zero is
not a pathology of the model, it is the correct solution to the objective we wrote down.

This single fact explains every observation in §11 and §12 at once:
- `motion_ratio` → 0.13: predicting ~zero minimises whole-frame MSE when the target is sub-floor.
- Action ORDER free, DISTRIBUTION costly (§12): ordering selects *which* sub-noise-floor motion occurs, so
  it cannot register; only the aggregate action statistics survive above the floor.
- 5.8–16× more action gradient changing nothing (§11): no quantity of action information helps when the
  regression target is below the model's own precision.
- One-step 1.59 dB from the codec ceiling: 97% of the frame is static background the codec nails.

Subsampling to 4–5 Hz flips SNR from 0.61× to 1.24–1.36×, makes 64 steps 13–16 s, and costs **4–5× LESS**
per epoch (fewer windows). Literature convergence: V-JEPA-2-AC 4 fps with *integrated* EEF deltas
(2506.09985), HMA resamples 40 datasets to 2 Hz (2502.04296), IRASim ~4 fps (2406.14540); FAST (2501.09747)
names high-frequency action correlation as the cause — our lag-1 r = 0.988 is NORMAL for 20 Hz teleop, not
anomalous.

### Corollary: a BETTER codec is a motion lever

The floor is 0.0637 *because* the codec is good; anything that lowers reconstruction error (256px, per
`capacity.md`) lowers the floor and makes per-step motion learnable. The "excellent AE floor + no motion"
pairing the user spotted is not a coincidence — the same number is both.

### Live confirmation: both arms peaked at ep2 and REGRESSED at ep3

| | ep0 | ep1 | ep2 | ep3 |
|---|---|---|---|---|
| `tfz_act` 1-step / ol@64 / mot@64 / vloss | 15.14 / 9.64 / 0.559 / .5118 | 18.55 / 13.60 / 0.131 / .2534 | **18.83 / 13.90 / 0.168 / .2441** | 17.98 / 12.64 / 0.118 / .2552 |
| `tfz_act_fourier` same | 15.71 / 9.82 / 0.750 / .4760 | 18.06 / 12.98 / 0.131 / .2774 | **18.40 / 13.44 / 0.154 / .2706** | 18.09 / 13.36 / 0.119 / .2664 |

`motion_ratio@+64` fell in BOTH arms at ep3 (0.168→0.118, 0.154→0.119) — two independent runs, same
direction. **This contradicts §12's "motion may be recovering, 12 epochs will resolve it" reading; it is
resolving negatively.** `tfz_act`'s val loss also worsened (0.2441→0.2552). Matches GameNGen's small-data
ablation (2408.14837): below ~10^7 examples, test quality peaks EARLY then degrades. Runs left alive to
finish (user instruction), but they are past peak.

### Also from the review — established, and load-bearing

- **91k steps is NORMAL, not short**: V-JEPA-2-AC 94.5k, Vid2World 100k, Genie 125k, WHAM 200k. And **no
  published controllability-vs-steps curve exists** — "controllability emerges late, we just aren't there"
  is unsupported by anything in the literature. Do not buy more epochs at this design.
- **Scale**: 4.17M params is ~220× below the cohort median (~0.9–1B) and 10× below the smallest working
  action-conditioned video world model (HMA-Base 44M — pretrained on >2.5B frames). 4 h is ~700× below
  median and 22× below the smallest from-scratch single-scene success (DIAMOND CS:GO 87 h / 381M params,
  itself described as brittle). **No published <50M-param, <50 h, from-scratch, 30–64-step controllable arm
  video result exists.**
- **Our action injection is on the losing side of three independent ablations.** Cosmos-Predict2.5
  (2511.00062, Table 20): TimeEmbedding-add **24.95 PSNR / 146 FVD** > CrossAttention 24.41/159 >
  ChannelConcat **23.11/267**. HMA (2502.04296): per-layer modulation > token-concat ("token concatenation
  along the sequence dimension does not have enough expressiveness"). IRASim (2406.14540): frame-level AdaLN
  **28.82 vs 23.89** PSNR. Our design is a hybrid of the two losing arms. **NOTE: this conflicts with the
  user's 08-11 call ("i don't like film adaln") — that decision was made about PROPRIO going in as a bag,
  and this evidence is about the ACTION specifically. User's call, but the evidence should be on the table.**
- **`action_squash: symlog` is wrong for actions.** DreamerV3 symlogs observations/rewards/values, NOT
  actions. The literature standard is per-dim **quantile normalisation** (1st/99th → [-1,1]; FAST, OpenVLA,
  π0), which also handles our near-dead dims by stretching them. Nobody drops dead dims.
- **Adopt standard controllability metrics** for comparability: Genie **ΔPSNR** (true vs random actions;
  published values 1.3–2.1 dB), dWorldEval **Δ-LPIPS + shuffle**, ActSWM step-drift gap. Our reversal probe
  (§12) is **sharper than anything published** — the review found no paper that perturbs action ORDER — so
  keep it, but report ΔPSNR alongside.
- Our failure mode is named elsewhere: **"context collapse"** (ActSWM), **"visual inertia"** (Astra),
  **"stagnation"** (Steady-Forcing). Caveat: ActSWM's pathology is recorded-vs-**zero** being identical, and
  we already differ there (−0.48 dB, −3.96 dB for the Fourier arm), so their hinge fix targets an axis we
  partly have. The order/distribution dissociation is ours.

**CAVEAT on citations:** the 2026-dated arXiv IDs in the review (26xx.*) were verified by the agent only via
fetched pages and are past this assistant's knowledge cutoff — re-verify before citing anywhere external.
The load-bearing ones (2511.00062, 2502.04296, 2406.14540, 2501.09747, 2506.09985, 2408.14837) are older and
checkable.

### Metric bug found

`psnr_frozen@+1` logs **120.00 dB** — the clamp for *identical images*. At +1 the frozen baseline is
comparing the held frame with itself, so the frozen curve is misaligned by one step against the model curve.
`psnr_frozen@+1` is unusable as written. The +64 comparison (13.90 vs 10.40) is not materially affected, but
note that at +8 the model is now **tied** with persistence (15.13 vs 15.11 at ep3).

### Recommended next run (not started)

**Subsample to 4–5 Hz with integrated actions**: keep every 4th–5th frame; the conditioning action is the
SUM of the skipped EEF deltas (rotations composed, gripper = last), optionally plus absolute EEF pose.
Cheaper than baseline, violates no standing constraint, and attacks the measured mechanism rather than a
symptom. Pre-check first, for free: regress the true next-state delta on the action at stride 1 vs stride 4/5
and compare R² — if subsampling does not raise the action's explanatory power, do not spend the run.

### CONFIRMED against torus-world (08-12) — the mechanism predicts BOTH outcomes

The user pushed back: torus-world learns video prediction from very little data, so little data cannot be
the blocker. Correct, and §13 predicts exactly that. Measured with the SAME frozen TAESD at 128px
(`_torus_check.py`, self-validated by first reproducing robocasa's 23.41 dB before trusting the torus number):

| | codec floor | per-step delta | **delta / floor** | % pixels moving |
|---|---|---|---|---|
| **torus-world** (60 fps) | 0.0259 (31.73 dB) | 0.0505 | **1.95x** | 3.68% |
| **robocasa** (20 Hz) | 0.0675 (23.41 dB) | 0.0389 | **0.61x** | 3.18% |
| robocasa @ 4 Hz (new runs) | 0.0675 | 0.0863 | **1.35x** | 8.08% |

Torus's per-step SNR is **3.2x** robocasa's, from two compounding causes: its codec floor is **2.6x lower**
(synthetic flat-textured frames round-trip at 31.73 dB vs 23.41) and its per-step motion is 1.3x larger in
absolute terms *despite running at 3x the frame rate*. One mechanism explains both results: torus sits at
SNR ~2, robocasa-at-20-Hz at SNR ~0.6. Data quantity is not what separates them.

**REFINEMENT — the moving-pixel fraction is NOT the discriminator.** §13 above cites 3.18% of pixels moving
as part of the problem; torus has essentially the same fraction (3.68%) and learns motion fine. The
discriminator is the **delta-to-codec-floor RATIO** alone. The pixel fraction explains why whole-frame MSE
is an insensitive objective in both datasets, but it does not explain the robocasa failure.

**Implication for the target rate:** matching torus's 1.95x would need robocasa stride ~8-10 (2.5-2 Hz), not
5. Stride 5 (1.35x) is the conservative first step and matches V-JEPA-2-AC's 4 fps; if 4 Hz shows motion but
weakly, stride 8-10 is the indicated follow-up rather than any change to conditioning.

**Corollary already noted, now quantified:** a better codec is a motion lever, and it is the LARGER of the
two terms here (2.6x vs 1.3x). 256px, or any encoder with lower reconstruction error on robocasa frames,
buys more SNR than subsampling does.

### Correction: this repo sets NO training seed

§11 called the 2nd and 3rd launches "same-config, same-seed replicates". Wrong: there is no
`seed_everything` or `manual_seed` anywhere in the training path, and no top-level `seed` key. The +-0.8 dB
spread is therefore genuine seed-to-seed variance, which is the correct yardstick for architectural claims
(and means two runs of one config ARE a 2-seed replicate -- what the 4 Hz pair is doing).

### The 20 Hz runs COLLAPSED at ep4 (not merely regressed)

Killed at user instruction to free the GPUs; ep4 metrics were already logged and show a full collapse, so
nothing was lost:

| | ep2 (peak) | ep3 | ep4 |
|---|---|---|---|
| `tfz_act` 1-step / ol@64 | 18.83 / 13.90 | 17.98 / 12.64 | **9.07 / 7.78** |
| `tfz_act_fourier` 1-step / ol@64 | 18.40 / 13.44 | 18.09 / 13.36 | **8.54 / 7.26** |

`motion_ratio@+64` ROSE as they collapsed (0.170 and 0.325) -- independent confirmation that it is
direction-blind and that a diverged model scores high on it. Never judge a run on it alone. Peak-at-ep2 then
collapse matches GameNGen's small-data ablation (2408.14837).

## 14. `hz4_seedA` / `hz4_seedB` — 4 Hz, two seeds (08-12 21:31, RUNNING)

`data.subsample=5` (new knob, `conf/data/torus.yaml`; 1 = off = bit-identical). Applied inside BOTH episode
loaders so training windows and all 5 eval call sites cannot diverge in rate -- a missed call site would
leave eval at 20 Hz and look like the model failing. Actions are SUMMED across skipped frames except
auto-detected near-binary dims (measured [3, 4, 11] = the constant, the flag, the gripper), which take-last.
Smoke-verified: lengths T//5, obs/images exactly every 5th frame, additive dims equal the group sum, hold
dims differ from it, and the resulting per-step delta is 0.0863 = 1.35x floor, matching the prediction.

235/235 train episodes survive (min length 123 >= P+F=72) -> 35,085 windows, ~1,100 batches/epoch vs 7,582,
so ~25 min/epoch. 40 epochs (~17 h), in the 30-200 range the small-data literature uses. F=64 now spans
**16 s** of real time instead of 3.2 s. Fourier bands OFF (§12: no benefit, likely OOD brittleness),
symlog OFF (user).

TWO SEEDS, NOT AN A/B, on purpose: every A/B this week had an effect inside the +-0.8 dB noise floor, so this
measures the effect against the well-established 20 Hz baseline AND establishes the floor at the new rate.

Judge on: `motion_ratio@+64` (target >0.4, ceiling ~0.92) **together with** PSNR and the §12 action-order
gap -- never motion_ratio alone, per the collapse above.

## 15. The anchor WORKED, and SNR 1.43x is still not enough (08-13)

`latent_loss_weight=10` did exactly what it was supposed to. `anch128`, 9 epochs:

| | roundtrip | codec floor | 1step | ol@64 | mot@64 |
|---|---|---|---|---|---|
| unanchored (w=1), ep0->ep11 | 0.0038 -> 0.0087 | 23.9 -> **19.9** (-4.0) | 16.45 | 13.66 | 0.177 |
| **anchored (w=10), ep0->ep8** | 0.0033 -> 0.0035 | 24.79 -> **24.39** (-0.40) | 16.64 | 13.94 | 0.167 |

Erosion 3.55 dB -> **0.40 dB**. The mechanism and the fix are both confirmed, and the SNR the subsampling
bought is now retained (0.85x -> 1.43x).

**And it changed nothing about motion.** `anch128` plateaued from ep5: 1step 16.4->16.6, ol@64 13.8->13.94,
`mot@64` flat at **0.16-0.20** for nine epochs. So:

> **SNR 1.43x is NOT sufficient for motion.** That is the single most useful number of the week -- it puts a
> floor under what any future arm has to beat, and it was measured with the codec held stable so nothing else
> can be blamed.

`anch256` (SNR 2.33x, floor holding 28.4 -> 28.2 dB) is the arm that says whether ~2x IS sufficient. Left
running through its ep15 judgement.

### The proprio head is the existence proof inside our own model (08-13, user spotted it)

The user noticed the open-loop PROPRIO rollouts look excellent while the image rollouts are frozen. Same
dynamics, same bag, same actions -- the difference is entirely the readout's bottleneck:

```
image     128*128*3 = 49,152 values -> 1,024 floats  = 48x COMPRESSION (lossy, frozen TAESD)
proprio          16 values ->   128 floats           =  8x EXPANSION   (no bottleneck at all)
```

| head | per-step change | codec floor | change/floor |
|---|---|---|---|
| image | 0.0863 | 0.0598 | **1.44x** |
| **proprio** | 0.1835 | **0.0160** | **11.47x** |

**8x the SNR, and it visibly moves.** The dynamics can model this arm; the image readout cannot express it
above its own reconstruction error. Ranks every case correctly: proprio 11.5x works, torus 1.95x works,
256px 2.33x TBD, 128px 1.43x fails, 20 Hz 0.61x fails.

**A hypothesis of mine died here:** I expected "proprio has no static background to hide behind" to explain
it. It does not -- the IMAGE changes a larger fraction of its own spread per step (31.4% vs 20.0%). The
moving-fraction favours proprio only 2.6x (25.2% of dims vs 9.8% of pixels) against a 7.9x floor advantage.
**The codec floor dominates; background dilution is secondary.**

## 16. HOLIDAY PROGRAM: non-pretrained flow image decoder (08-13, RUNNING UNATTENDED)

User direction: a bunch of tests on a non-pretrained flow image decoder, layernorm to start but try things,
15-epoch monitors, new configs when one does not work, no new code except bug fixes, branch
`holiday-bespoke-flow-decoder`, do not commit to main.

### It needs NO code. Verified by building it.

`pretrained: false` gives the bespoke `ImageModality`, which encodes **directly** to `(num_tokens, d)` -- no
adapter, and therefore **no `num_tokens * d == L` constraint at all**, so the compression ratio becomes a
design choice instead of TAESD's 48x. It also honours `decode_kind: flow` (the pretrained path hardcodes
`self.decode_kind = "mse"`), giving `ImageUNetFlowHead` -- a trainable generative decoder, structurally the
same class of head as the proprio flow MLP that works. Built and forward-passed both arms:

```
nt=8   4.46M params  ImageUNetFlowHead  1024 floats  48.0x compression  bag (B,32,9,128)
nt=32  6.04M params  ImageUNetFlowHead  4096 floats  12.0x compression  bag (B,32,33,128)
roundtrip anchor wired at weight 10.0 in both
```

Two config declarations were needed, both the same bug as gotcha #1 (`compile_rollout`): `latent_loss_weight`
and now `encode_base`/`decode_base` existed only as dataclass defaults, so hydra rejected bare overrides with
"not in struct". Declared with values matching the code defaults -> bit-identical.

**Forced constraint:** bespoke must use `latent_norm=layernorm`; `affine` RAISES without a frozen pretrained
latent (it calibrates fixed per-channel stats once, and a learned encoder drifts its own scale). So
bespoke-vs-TAESD is confounded with LN-vs-affine. `bsp8` vs `bsp32` is clean.

### The honest odds, stated up front

`anch128` held 24.4 dB and still failed. The only historical bespoke measurement is **~15 dB** (48x
compression, unanchored, and it eroded to 10.6). **So a bespoke arm must find ~+9 dB over history merely to
reach a configuration that already failed.** Its two routes are less compression (`num_tokens`) and AE
capacity (`encode_base`/`decode_base`), which is exactly how the queue is ordered. The counter-argument for
running it: nobody has ever run a learned encoder at 12x compression WITH the anchor, and the old bespoke
failures are contaminated by the erosion we only diagnosed yesterday.

### Judgement bar (auditable, calibrated on measured baselines)

At ep15, judged once, all five numbers logged:

```
KILL if 1step < 12       -> collapsed (every 20 Hz run did this by ep4)
KILL if ae_floor < 21 dB -> codec hopeless (24.4 dB ALREADY fails, so <21 cannot win)
KILL if mot@64 < 0.22    -> the same static failure as all 14 prior runs (baseline 0.16-0.20)
else KEEP to ep40
```

Validated against five real runs before launch: `anch128`@ep8 -> KILL (no motion), `hz4_seedA`@ep11 -> KILL
(floor 19.96), the collapsed `tfz_act`@ep4 -> KILL (COLLAPSED). The ep>=15 gate is what stops `anch256` being
killed at ep1 on a number that is always low early.

### Queue (`wizard/scripts/holiday/queue.txt`, popped onto whichever GPU frees)

bsp32 (12x compression, primary) | bsp32wide (base 64, AE capacity) | bsp8 (48x control) | bsp16 | bsp32none
(no latent norm) | bsp32mse (is the flow decoder earning its keep?) | bsp32llw30 | bsp32vit | bsp32deep |
bsp32widest (base 96) | bsp64 (6x compression) | bsp32sub8 (2.5 Hz)

### Orchestrator

`wizard/scripts/holiday/orchestrator.sh`, polls 10 min: free GPU -> pop next config; running and ep<15 ->
leave alone; ep>=15 -> judge once, KEEP or KILL-and-free; died -> log and move on. Adopts `anch256` under the
same rule. Every pgrep/pkill uses the bracket trick ([t]rain not train) because a bare pattern also matches
the orchestrator's own command line -- a plain `pkill -f watchdog.sh` killed one of my own shells today.

### Still open / not done

- `motion_ratio` is direction-blind and whole-frame; a COLLAPSED model scores HIGHER on it (the ep4 collapse
  read 0.17-0.33). It is in the kill rule only as a floor, never as evidence of success on its own.
- The §12 action-sensitivity probe is NOT promoted to an eval routine (would be new code). Run it post-hoc:
  `src/quickdraw/_oneoff_action_sensitivity.py` against any checkpoint.
- Decoder-only fine-tune of TAESD (keep its encoder's 9 dB prior, train its decoder on this one scene) is
  the untried lever I rate highest, and it needs a small code change (split `freeze`, narrow the affine
  guard to the ENCODER -- the guard's own rationale is about encoder drift, so it is stricter than needed).
- A 16-channel VAE would be 12x compression at 128px with a pretrained prior -- the "less compression"
  lever without paying the 9 dB.
- No watchdog is running. The orchestrator relaunches from the queue but does NOT resume a crashed run from
  its checkpoint.

## 17. THE 8×8 BOTTLENECK — why 14 runs could not move the AE floor (08-20/21)

**The trigger.** The work was presented and the feedback was that *both* the AE floor and the dynamics
look blurry. That is the same complaint §16's winner (`bsp32mse`) was supposed to have answered, so the
question became: what is actually pinning the bespoke reconstruction floor at 18.7–20.4 dB?

**The finding.** Both conv pyramids computed their level count from an *independent copy* of the same rule:

```python
# ConvImageEncoder  (models/vision.py, after a stride-2 stem)
n_levels = max(1, int(math.log2(max(8, min(h0, w0)) // 8)))
# ConditionalUNet   (models/vision.py:370, from FULL resolution — a second, separate copy)
n_levels = max(1, int(math.log2(max(8, min(H,  W )) // 8)))
```

The `8` is a hardcoded **target**: the pyramid pools until the short side reaches ~8, *at every
resolution*. 64px → 8×8. 128px → 8×8. 256px → 8×8. (192/384px → 12×12.) So the encoder discarded all
spatial detail below 8×8 **before the `num_tokens` learned queries ever cross-attended the map**.

**This retroactively explains every null result on the codec axis.** Four separate sweeps, all flat:

| swept | range | effect on the AE floor |
|---|---|---|
| `num_tokens` | 8 → 64 (an **8× range of latent floats**) | none, 18.7–20.4 dB |
| `decode_base` | 32 → 64 (2.3× total params) | none on the floor |
| `ae_depth` | 4 → 6 | none |
| `latent_loss_weight` | 10 → 30 | none |

None of them touch the binding constraint, so none of them could have worked. It also **predicts that
256px would have been a waste** — the pyramid would still land on 8×8, so we would have paid 4× the
compute for the same bottleneck. (256px is out of the running anyway per the user, 08-21.)

Consistent supporting evidence already in hand: the **best bespoke floor of the whole holiday program,
21.26 dB, came from `decode_arch=vit`** — the one decoder with *no conv pyramid at all*.

**The parameter split is also backwards for a world model.** On `bsp32mse` (6.373M total):

| block | params | share |
|---|---|---|
| image decoder (U-Net) | 4.376M | 68.7% |
| dynamics backbone | 1.063M | 16.7% |
| flow denoiser | 0.485M | 7.6% |
| image encoder | 0.403M | 6.3% |
| | | **dynamics = 24.3%** |

Two thirds of the model is a decoder that reads an 8×8 bottleneck. Rebalancing (`depth` 4→8,
`flow_arch_depth` 2→4, `d` 128→192) is a separate, deferred experiment.

### The knob (implemented 08-21)

`ae_bottleneck` on `ModalitySpec` → `VisionAEConfig.bottleneck`, replacing the hardcoded `8` in **both**
formulas. Default 8. Verified bit-identical:

```
bottleneck=None  total 6.373M | ae 0.403M | decode_head 4.376M | enc bott_hw (8, 8)
bottleneck=8     total 6.373M | ae 0.403M | decode_head 4.376M | enc bott_hw (8, 8)   <- identical
bottleneck=16    total 5.321M | ae 0.189M | decode_head 3.538M | enc bott_hw (16,16)
```

**Exposed as a shared TARGET, not as `n_levels`** (the user's first suggestion). The encoder pools
*after* a stride-2 stem and the decoder pools from full resolution, so at 128px they need 3 and 4 levels
respectively — one shared `n_levels` value would silently desynchronise them. A shared target cannot.

**Note the confound, which runs in the favourable direction.** Raising the target removes one pyramid
level, i.e. removes the deepest and widest channel block, so `bottleneck=16` has **fewer** parameters
(5.321M vs 6.373M, and the AE proper drops 0.403M → 0.189M). A win is therefore unambiguous — spatial
resolution beating channel width at 0.83× the params. A loss is ambiguous, and the follow-up would be
`bottleneck=16` + `decode_base=64`.

### The A/B — `bott_recon1` vs `bott_bott16` (08-21, RUNNING)

`wizard/scripts/robocasa-bottleneck.sh`, `logs/robocasa-bottleneck/`, 40 epochs, both on `model=bsp32mse`
at `data.subsample=5`.

| arm | GPU | change | tests |
|---|---|---|---|
| `bott_recon1` | 0 | `model.recon_frac=1.0` | is the blur a shortage of **supervision**? 0.25 was inherited, never measured; `bsp32mse.yaml` names it as the most obvious untested sharpness lever |
| `bott_bott16` | 1 | `+model.modalities.1.ae_bottleneck=16` | is the blur the **8×8 bottleneck**? 64× spatial reduction instead of 256× |

**Based on `long` (decode_base 32), not `sharp` (64)** — the user asked why. `sharp` wins LPIPS by ~6%
(0.303 vs 0.323) but costs 2.3× the params and 11% more wall-clock, and *loses* PSNR (best OL@+64 14.32
vs 14.82). `decode_base` is believed orthogonal to the bottleneck, so `long` is the cheaper, faster base
and keeps `decode_base` available as the follow-up lever if `bott16` loses.

**Batch is left to autobatch, which is ON — and pinning it was RETRACTED.** The first plan pinned
`data.batch=8` to match the baselines (they ran at 8 under the old mis-measured 35%-headroom finder). The
user pushed back and was right: comparability was already broken by the epoch budget, and "they peaked at
ep17" is an EPOCH count, not a step count, so a fixed batch never bought the step-for-step comparison
claimed for it. So this doubles as the rewritten autobatch's first live outing, and it landed well:

| arm | GB/sample | batch | reserved | frag |
|---|---|---|---|---|
| `bott_recon1` | 13.169 | **7** | 92.5/93 GB (99%) | 3% |
| `bott_bott16` | — | **17** | 91.0/93 GB (98%) | 0% |

Both confirmed on the COMPILED step, not just eager. Note `recon_frac=1.0` costs 13.2 GB/sample and lands
*below* the baseline's batch 8, while `bott16` more than doubles it — the arms differ in batch by 2.4x.
That is a confound BETWEEN the arms (not against the baselines), accepted rather than equalised because
equalising means running both at 7 and idling half of GPU 1. `max_epochs` 25 -> 40 to compensate for the
larger batch being fewer gradient steps per epoch.

Measured cost: `bott_bott16` 0.86 h/ep (2064 batches), `bott_recon1` 2.11 h/ep (5013 batches) — against the
baseline's 1.588 h/ep. So the bottleneck arm finishes ~08-23 and the recon arm ~08-25.

**Baselines to beat** (both 50 ep, batch 8, subsample 5):

| run | best LPIPS@+64 | best OL@+64 | best AE floor |
|---|---|---|---|
| `logs/holiday/train_world_model_2026_08_18_09_16_26_bsp32mse_long` | **0.323** @ep17 | **14.82 dB** @ep9 | 20.14 dB @ep13 |
| `logs/holiday/train_world_model_2026_08_18_09_18_26_bsp32mse_sharp` | **0.303** @ep17 | 14.32 dB @ep2 | 19.83 dB @ep18 |

Read it on `ae_floor` PSNR (did the **codec** get sharper) and LPIPS@+32/+64 (did the **rollout** get
sharper). **Not** on `motion_ratio` alone — it is direction-blind and a *collapsed* model scores higher.

**RESULT AT ep13 — the 8x8 bottleneck WAS the binding constraint, and fixing it did NOT fix the blur.**

The AE floor moved for the first time in 15 runs. Epoch-matched on `psnr_mean`:

| ae_floor PSNR mean | e8 | e9 | e10 | e11 | e12 |
|---|---|---|---|---|---|
| `bott_bott16` (16x16, 5.32M) | 19.88 | 19.86 | 19.89 | 19.97 | **20.04** |
| `BASE long` (8x8, 6.37M) | 19.45 | 19.50 | 19.59 | 19.41 | 19.52 |
| `BASE sharp` (8x8, 14.83M) | 19.42 | 19.39 | 19.34 | 19.44 | 19.36 |

Ahead at every matched epoch by +0.4 to +0.6 dB, **still rising monotonically** where both baselines have
gone flat and begun oscillating, and doing it on **fewer parameters than either**. The confound registered
in advance (a dropped pyramid level costs 1.05M params) therefore ran the favourable way, so this is the
unambiguous case: spatial resolution beats channel width on this codec.

**But LPIPS goes the other way, and LPIPS is the metric the complaint was about.**

| ae_floor LPIPS@+64 | e8 | e10 | e12 |
|---|---|---|---|
| `bott_bott16` | 0.267 | 0.244 | 0.245 |
| `BASE long` | 0.229 | 0.222 | **0.207** |
| `BASE sharp` | 0.169 | 0.165 | **0.157** |

Worse than `long` at every epoch and much worse than `sharp`. Same on OL LPIPS@+64 (0.363 vs 0.336 vs
0.340 @e12). Motion is also lower and rising more slowly: 0.389 vs 0.437 and 0.480. OL PSNR@+64 ties
`long` exactly (14.79 @e9 vs 14.82 @e9).

So more spatial resolution buys PSNR and costs perceptual sharpness — the distortion-perception tradeoff
again, pointing the way it has on every lever in this project. **`bott16` wins the metric the experiment
was designed around and loses the one that motivated it.**

**A CORRECTION TO THIS RECORD.** The baseline floor was quoted throughout as "20.14 dB @ep13" (including
in the baselines table below). That figure is `ae_floor/image/psnr/@+1` — the ONE-STEP reconstruction —
not `psnr_mean`. Like-for-like on `psnr_mean`, `long`'s best is **19.586 @ep10**. So `bott16`'s 20.04 is a
real +0.45 dB gain rather than a wash against 20.14. Every comparison in this section is same-tag; the
earlier number was not, and it made a win look like a tie.

**`bott_recon1` is losing on everything** (6 epochs): OL PSNR@+64 peaked 14.34 @ep1 and has decayed to
13.60; OL LPIPS 0.415-0.444, worse than every arm and every baseline. `recon_frac=1.0` looks like a
straight loss at 2.4x the cost per epoch (2.11 h/ep, batch 7). Let it reach ~ep10 to confirm, then it is
the obvious kill candidate.

**FOLLOW-UP, STAGED AND NOT RUN** (`wizard/scripts/robocasa-bottleneck-2.sh`, needs a free GPU):
`ae_bottleneck=16` **+** `decode_base=64`. `bott16` gains PSNR from resolution but loses LPIPS to the
1.05M params the dropped level cost; `sharp` shows `decode_base=64` is worth ~0.05 LPIPS on its own.
Combining them tests whether resolution and channel width are ADDITIVE, and it is the first config with a
plausible shot at both. This is exactly why the A/B was based on `long` rather than `sharp` — it kept
`decode_base` free as the next lever.

`bott_recon1` (ep0-2) is ahead of the baseline on open-loop PSNR@+64 at every matched epoch (12.36/14.34/
13.78 vs 11.02/13.66/13.60) and behind on OL LPIPS@+128 (0.651/0.525/0.485 vs 0.622/0.491/0.479) — the
distortion-perception tradeoff again, pointing the way it always does on this problem, with the gaps
converging as if `recon_frac` buys early-training speed rather than a different endpoint.

### Multi-GPU: not set up, deliberately — `design/distributed.md` (08-21)

`devices=1` is pinned at `train_world_model.py:270` as a **guard**, not an oversight. Written up in full;
the short version is that three things would produce a wrong-but-plausible run rather than an error:

1. **`MMWindowLoader` is a hand-rolled iterator**, not a `DataLoader`, so Lightning cannot inject a
   `DistributedSampler` — and `seed_everything` gives every rank the *same* `torch.randperm`. Both ranks
   would train on **identical batches**, DDP would average a gradient with itself, and the run would be
   mathematically identical to 1 GPU at 2× the power. Nothing raises.
2. **Zero rank-awareness anywhere in `src/quickdraw`** (`grep global_rank|is_global_zero|world_size` is
   empty). Both ranks would append to the same `metrics.jsonl`, run every eval routine twice, and race on
   the same video/checkpoint paths.
3. **Autobatch would hang, not OOM.** Per-rank batch is a *micro*-batch (effective = `batch × world_size`,
   the only knob we have left with `accumulate_grad_batches` and `window_stride` both LOCKED), and if two
   ranks probe *different* batches they run different step counts and deadlock in the next collective.

**And it is the wrong trade today.** DDP buys wall-clock on one hypothesis; two cards buy two hypotheses
in the same wall-clock. Every result in this record came from the second mode, and the binding constraint
on this project is that effects are small against the ±0.8 dB noise floor — we need *more arms*, not
faster arms. DDP becomes right when one config stops fitting (a much bigger decoder, 256px, F ≫ 64).

## Appendix — folded in from wizard/scripts/*.md (2026-08-11)

These lived next to the launch scripts, where `.gitignore` kept them unsynced. Content preserved verbatim.

### from `wizard/scripts/handoff-eval-action-dist.md`

# Handoff: 4 fixes in the action-distribution eval + one config declaration

Context: running `mm_flow` on a **recorded, no-simulator** dataset with a **12-dim** action space
(`isaac-ronald-ward/robocasa-scene4-4h`, `environments=recorded`). Line numbers are against the commit
`2c1ea51`. Items 1–3 are bugs; item 4 is a feature request with a design.

Motivating measurement — the 12 action dims over all 259,299 train frames. Note how degenerate a real
robot action space is, which is what makes the current magnitude-only products uninformative:

| dim | std | n_unique | reading |
|---|---|---|---|
| 0,1,2 | 0.23 / 0.21 / 0.19 | ~550–700 | EEF delta position |
| **3** | **0.000** | **1** | exactly constant — dead |
| **4** | 0.71 | **2** | binary flag (~85% at −1) |
| 5,6,7 | 0.31 / 0.24 / 0.30 | ~700 | rotation deltas |
| 8,9,10 | 0.081 / 0.086 / 0.072 | ~710 | near-dead (mobile base, unused in this scene) |
| **11** | 0.97 | **2** | binary — gripper |

---

## 1. `model.compile_rollout` is undeclared, so the bare override is rejected

`src/quickdraw/training/setup.py:81` reads it with `bool(m.get("compile_rollout", False))`, but
`conf/model/mm_flow.yaml` never declares the key. So:

```
model.compile_rollout=true    -> ERROR: "Key 'compile_rollout' is not in struct"
+model.compile_rollout=true   -> works
```

This is the same trap class as the old `trainer.fast_dev_run` (read but undeclared ⇒ needs `+`). It is
easy to hit because every *other* model knob in that file takes the bare form.

**Fix:** declare it in `conf/model/mm_flow.yaml` next to `grad_checkpoint`, defaulting off, with the
guard conditions in the comment:

```yaml
compile_rollout: false   # true -> torch.compile the AR rollout STEP (~6x, accelerations.md Exp 9).
#                          Requires head_dim (d/heads) >= 16 and is mutually exclusive with
#                          variations.contraction (its Jacobian power-iteration needs the eager graph).
```

Measured on this dataset for reference: **3.90 s/batch eager → 0.67 s/batch compiled** at
`d=128 depth=4 F=64 batch=32` (5.8x), and 1.07 s/batch at `d=256 depth=6`.

---

## 2. CRASH: bare `ecfg.a_max` in `eval_action_distribution` (3 sites)

`src/quickdraw/evaluation/routines.py` lines **679, 690, 694**:

```python
fig = viz.fig_action_by_state(arr, x, ecfg.a_max, ...)              # 679
frames = viz.anim_action_distribution(true_a, pred_a, ecfg.a_max, ...)   # 690
frames_bx = viz.anim_action_by_state(true_a, pred_a, x, ecfg.a_max, ...) # 694
```

`a_max` is a **torus-only** knob (`conf/environments/torus.yaml:7`, `a_max: 4.0`).
`conf/environments/recorded.yaml` has no such key ⇒ `AttributeError` the moment this eval runs on any
recorded/non-torus env. This is the identical bare-attribute pattern already fixed at `_ood_axis`
(line 166), just further down the same file.

`a_max` is only ever used as a histogram **x-limit** — `viz.py:1197`, `1287`:
`hi = min(float(a_max), <data max> * 1.05)`. So it has a natural data-driven fallback.

**Fix:** pass `getattr(ecfg, "a_max", None)` and make the three viz functions treat `None` as "use the
data range":

```python
a_max = getattr(ecfg, "a_max", None)   # torus-only knob; None -> derive the limit from the data
```
and in `viz.py` (`fig_action_distribution:1160`, `anim_action_distribution:1185`,
`anim_action_by_state:1225`, `fig_action_by_state:1276`):
```python
data_hi = max(float(tm.max()), float(pm.max())) * 1.05
hi = data_hi if a_max is None else min(float(a_max), data_hi)
```

### 2b. Same bug, one line up, still live: `ecfg.init_speed`

`routines.py:168` — inside the function whose `R`/`r` default you *just* made lazy on line 166:

```python
se.get("init_speed", ecfg.init_speed),   # <-- bare attribute, evaluated EAGERLY
```

`se.get(k, default)` always evaluates `default`, so this raises on any env without `init_speed` —
`recorded.yaml` has none — even when the dataset card supplies `init_speed`. Line 166 was fixed with
`or {...getattr...}`; line 168 was missed.

**Fix:** `se.get("init_speed", getattr(ecfg, "init_speed", None))`, and confirm `_openloop_split`
tolerates `None` (it is reached only by the `ood_visual/geometric/dynamics` axes, so this is latent
rather than blocking — but it is the same footgun).

---

## 3. The by-state ("by-x") products must be torus-only

`routines.py:663`:

```python
x = np.stack([o[1:L, 0] for o, _, _ in eps]).astype(np.float32)   # ambient x at each action's state
```

This hardcodes **observation dim 0** and the two by-state products then split rows on `x<0` / `x>=0`,
captioned as the torus's *slow basin / fast basin* (`viz.py:1226`: "TOP x<0 slow, BOTTOM x>=0 fast").

On any other env, obs dim 0 is an unrelated quantity — for this robocasa state it is a joint value with
mean 2.86, std 1.16, so `x<0` selects almost nothing and the split is silently meaningless. It does not
crash, which is worse: the panels render and look authoritative.

**Minimal fix** — gate both by-state products on geometry, exactly like the denoising routines now do:

```python
has_geom = getattr(ecfg, "R", None) is not None      # by-x split is torus semantics
...
if has_geom:
    for name, arr in (("true", true_a), ("pred", pred_a)):
        fig = viz.fig_action_by_state(...)           # 678-681
    frames_bx = viz.anim_action_by_state(...)        # 694-695
```

**Preferred fix** — make it an *optional env hook*, matching the 7-hook contract in `docs/byo.md` so
each hook independently unlocks one product with a graceful fallback. Add to the `WorldEnv` protocol in
`src/quickdraw/environments/base.py`:

```python
def action_dist_split(self, obs):        # OPTIONAL
    """-> (labels: bool array over obs rows, low_name: str, high_name: str) | None.
    Splits action-distribution panels by a MEANINGFUL state feature. None -> panels are pooled only."""
```

`TorusEnv` returns `(obs[..., 0] < 0, "slow", "fast")`; every other env inherits `None` and the by-state
products self-skip. That also deletes the hardcoded `o[..., 0]` and lets the panel captions come from the
env instead of being baked into `viz.py`.

---

## 4. FEATURE: per-dim action marginals (dataset/env-agnostic)

**Why the current products are not enough.** Every existing action plot reduces to a **magnitude**:
`np.linalg.norm(..., axis=-1)` at `viz.py:1192-1193`, `1231-1232`, `1283`. That is dimension-agnostic (so
12-D does *not* crash once item 2 is fixed) but for a real action space it is dominated by whichever dims
happen to have the largest scale. Here |a| is driven by the two **binary** dims (std 0.97 and 0.71),
while the six dims that actually carry continuous control have std 0.07–0.31. So the headline plot mostly
shows the gripper toggling, and a head could match |a| well while getting every continuous dim wrong.

Likewise `true_pred_w1` (`routines.py:684`) is a single scalar on pooled |a| — it cannot say *which* dim
is wrong.

### 4a. New viz function

In `src/quickdraw/logging/viz.py`, beside the existing action functions (~1160–1320):

```python
def fig_action_marginals(true_a, pred_a, names=None, max_cols=4, discrete_max=10, q=(0.001, 0.999)):
    """Per-dim action marginals: recorded (filled) vs head (outline) on SHARED bins, one panel per dim.
    true_a/pred_a: (E, T, A) physical actions. names: list[str] | None -> 'a[i]'.
    Dimension-agnostic and env-agnostic: no a_max, no state split, no geometry."""
```

Per-dim behaviour, decided from the **true** actions so panels stay comparable across runs:

- **constant** (`n_unique <= 1`): don't fake a histogram — render a text tile `"a[3] constant @ 0.000"`.
  Keeps the grid aligned and makes degeneracy *visible* instead of hidden.
- **discrete** (`n_unique <= discrete_max`): grouped bar chart of value frequencies, true vs pred.
  A 60-bin histogram of a ±1 flag is unreadable; this is the correct rendering for dims 4 and 11.
- **continuous**: shared bins over `[lo, hi]` = the `q` quantiles of the true dim, padded ~5%. Overlay
  true (filled, alpha) and pred (step outline). Robust quantiles, not min/max, so one outlier can't
  flatten the panel.

Shared bins between true and pred are the important detail — separately-binned histograms are not
visually comparable.

### 4b. Per-dim quantitative metric

Reuse the existing quantile-difference Wasserstein already at `routines.py:683-684` (same `q` grid),
applied per dim instead of to pooled magnitude:

```python
q = np.linspace(0.0, 1.0, 512)
w1_per_dim = [float(np.mean(np.abs(np.quantile(true_a[..., i], q) - np.quantile(pred_a[..., i], q))))
              for i in range(true_a.shape[-1])]
live = [i for i in range(true_a.shape[-1]) if true_a[..., i].std() > 1e-6]   # skip dead dims
writer.scalars({f"eval_action_distribution/w1/dim_{i}": w for i, w in enumerate(w1_per_dim)}, step)
writer.scalar("eval_action_distribution/w1_mean", float(np.mean([w1_per_dim[i] for i in live])), step)
```

**`live` matters:** a constant dim has W1 ≈ 0 by construction. Averaging it in would flatter the head —
on this dataset 4 of 12 dims are dead or near-dead, so a naive mean is diluted by a third. Keep the
existing pooled-magnitude `true_pred_w1` unchanged for continuity with prior runs.

`writer.scalars(dict, step)` already exists (`logging/writer.py:135`).

### 4c. Dim names, from the dataset (not the env)

lerobot carries them: `<root>/train/meta/info.json` → `features.action.names`. It is `null` for this
dataset, so the fallback must be graceful — but wiring it now means a dataset that *does* label its
actions gets readable panels for free. Read once in the routine (root via `resolve_data_root(cfg)`,
already imported at `routines.py:20`), fall back to `None` ⇒ `a[i]`.

### 4d. Call site

In `eval_action_distribution`, right after `true_a` / `pred_a` are built (`routines.py:671-672`) and
**unconditionally** — it needs no geometry, no `a_max`, no state split, so it works for torus (2-D),
pendulum (1-D) and robocasa (12-D) alike:

```python
fig = viz.fig_action_marginals(true_a, pred_a, names=action_names)
writer.figure("eval_action_distribution/marginals", fig, step); plt.close(fig)
```

This should become the **primary** product of the eval; the magnitude animations stay as secondary
views, and the by-state ones become torus-only per item 3.

---

## Acceptance criteria

1. `python -m quickdraw.eval_action_distribution checkpoint=<run> data.root=<root> data.repo_id=robocasa-scene4-4h data.cam=robot0_agentview_left environments=recorded environments.obs_dim=16 environments.action_dim=12` completes with **no `AttributeError`**.
2. `eval_action_distribution/marginals` shows **12 panels**: histograms for dims 0,1,2,5,6,7,8,9,10; bar
   charts for dims 4 and 11; a "constant" tile for dim 3.
3. No `by_state_*` / `animation_byx` products are emitted for `environments=recorded`; they still are for torus.
4. `w1_mean` excludes dim 3, and per-dim `w1/dim_*` scalars exist for all 12.
5. Torus runs are **unchanged** — same products, same `true_pred_w1`, `a_max` still respected as the limit.

Note this eval requires an action head, which is trained **post-hoc** on a frozen WM via
`train_action_model` (joint training is locked off — it killed control and destabilised the WM). So
reproducing needs a `train_action_model` run first; there is no action prior in a stock WM checkpoint.

### from `wizard/scripts/robocasa-scene4-4h.md`

# Wizard record — `isaac-ronald-ward/robocasa-scene4-4h`

Script: `wizard/scripts/robocasa-scene4-4h.sh` (both gitignored). Dataset inspected 2026-08-04;
**experiment revised 2026-08-05** from a decode-objective A/B to a **capacity sweep** at the user's
direction ("do the vit mse on both, a smaller 3.3M on one and a larger one on the other… 128x128 on both").

## Part A — what the dataset actually is

**Inspected before asking anything.** The key finding: this repo is **not a raw robocasa dump** — it is
already a **processed quickdraw recording**, pushed via `push_to_hub` from
`logs/recording_2026_08_04_06_54_40_robocasa-scene4-4h`. It has `train/` + `val/` split dirs each with
`meta/info.json`, `data/chunk-000/`, `videos/observation.images.<cam>/`, plus top-level `summary.json`,
`dataset_card.json`, `normalization_stats.json`, `progress.log`, and a `media/` dir of per-episode mp4s.

**So there is no `data_generation` stage and no `data/processors.py` stage.** Those are already done. (The
raw source was `madang6/quickdraw-robocasa-scene4-4h`, which the `robocasa` processor reads.)

| property | value | source |
|---|---|---|
| `obs_dim` | **16** | `train/meta/info.json` → `observation_vector.shape` |
| `action_dim` | **12** | `train/meta/info.json` → `action.shape` |
| `dt` / fps | **0.05 / 20** | `summary.json`, `dataset_card.json` |
| camera key | **`robot0_agentview_left`** — the ONLY one | `videos/observation.images.robot0_agentview_left` |
| image | **256×256×3**, av1 / yuv420p | `info.json` video info |
| train | 235 eps · 259,299 frames · **3.60 h** · len 615/1103/1749 | `summary.json` + `check_dataset` |
| val | 26 eps · 29,294 frames · **0.41 h** · len 644/1126/2066 | `summary.json` + `check_dataset` |
| windows @ P=8 F=64 | **242,614 train / 27,448 val** | `check_dataset` |
| robot_type | `null` | `info.json` |

**Column of `docs/byo.md`: 1 — recorded data, no simulator.** Confirmed by the run's own contract report:

```
[env-contract] recorded | obs_dim=16 action_dim=12 | reset ✗ no-sim | step ✗ no-sim | reward ✗ no-control |
render_obs ✗ dataset-images-only | rollout_metrics ✗ default(pointwise) | render_diagnostics ✗ filmstrip |
control_goals ✗ reward-only | physical_loss ✗ off | checkpoint_metric=pointwise_error | policies=-
```

- **Available:** WM training + validation, the `ood_horizon` pointwise metric, the `pred`/GT image
  filmstrip, and (deferred, not in this script) `eval_interpret` + `train_reward_model`, which need only the
  frozen WM + data + a VLM.
- **Unavailable:** MPPI control — both the goal race *and* language steering — plus the oracle baseline.
  All three must *step* the env. Hence `eval.during_train.evals.control=false`.

**Only one camera**, so the multi-trunk (one `image` modality per camera) question is moot. Multiple
cameras would mean re-running the processor against the source repo with `+source.camera=<leaf>`; the
`robocasa` docstring notes the source has 3 cameras.

## The user's choices

| topic | chosen | notes |
|---|---|---|
| image decode | **ViT-MSE on BOTH arms** | `decode_kind=mse`, `decode_arch=vit` — the header's winner |
| image resolution | **128×128 on both** (from 256) | the resolution the recipe's numbers were measured at |
| action head | **off on both** | WM only; no `train_action_model` |
| model size | **arm A ~3.2M / arm B ~14.2M** | the sweep axis |
| pipeline scope | **`train_world_model` only** | interpret + reward head deferred to the winning arm |
| GPUs | **both — one arm each** | GPU 0 = base, GPU 1 = large, otherwise byte-identical |

## The experiment: capacity, with everything else held fixed

The single independent variable is `d`/`depth`/`heads`. Everything else — data, schedule, decode objective,
teacher forcing, epochs — is identical between arms, so a difference in outcome is attributable to capacity
alone.

| arm | `d` | `depth` | `heads` | `head_dim` | params | GPU |
|---|---|---|---|---|---|---|
| base | 128 | 4 | 8 | 16 ✓ | **3.22M** | 0 |
| large | 256 | 6 | 16 | 16 ✓ | **14.21M** | 1 |

`head_dim = d/heads` must be a power of 2 for FlexAttention — both arms land on 16. (`d=192, heads=8`
would give 24 ✗; `d=256, heads=16` was chosen over `d=192, heads=12` for a cleaner 4.4× scale-up.)

`model_summary` for the large arm — the growth is almost entirely backbone, since `num_tokens=8` fixes the
per-frame token budget and the ViT heads scale only with `d`:

```
[train] MultiModalFlow 14.21M params
[arch] d=256 window=32 | per-step bag = 9 state token(s) + 1 action = 10 tokens
```

Base arm, per component (the 3.22M reference):

```
  proprio encoder (mlp)              (B,T,16) -> (B,T,1,128)                              0.009M
  image encoder (vit)                (B,T,128,128,3) -> (B,T,8,128)                       0.968M
  action_enc                         (B,T,12) -> (B,T,1,128)                              0.018M
  space-time backbone                (B,T,10,128) -> same [spatial 10 tok/step + causal]   1.060M
  predict_next: flow (rectified, per-token) (B,T,9,128) -> same                            0.072M
  proprio decode (mlp flow)          (B,T,1,128) -> (B,T,16)                              0.019M
  image decode (vit mse/no-noise)    (B,T,8,128) -> (B,T,128,128,3)                       1.072M
```

**Why 128×128 and not native 256.** At 256 the same config is 3.27M (encoder 0.992M / decode 1.097M) —
parameters barely move, but `patch=16` means **256 patches/frame vs 64**, ~4× the ViT encode+decode work
across the F=64 rollout, and the GPU-resident frame store grows 12.75 GB → 51 GB. The cost is compute and
memory, not parameters. 128 also matches the resolution every quoted metric was measured at.

## Learnings applied (quoted from `conf/model/mm_flow.yaml`)

- **`decode_kind=mse` + `decode_arch=vit`** — "ViT-MSE image decode (decode_kind=mse, decode_arch=vit)
  slightly BEAT U-Net/flow on EVERY axis here": val PSNR **18.9 vs 18.3**, OOD pointwise **0.24 vs 0.32**,
  OOD PSNR **14.5 vs 12.6**, control **4.12 vs 3.5**. The image is a deterministic render, so "the
  conditional mean IS the target."
- **`action_head.enabled=false`** — "the JOINT action head KILLS control (goals ~0 vs 3.88 baseline) AND
  destabilizes the WM (unetflow NaN'd ~ep31, vitmse collapsed to a frozen 11.1)." The control half of that
  doesn't even apply here (no steppable env); the destabilization half does.
- **`p_tf_end=0.0` (in-rollout), `p_tf_warmup_epochs=4`, Diffusion Forcing OFF** — full teacher forcing
  causes "autoregressive MEAN-COLLAPSE (proprio→origin, images→one frozen frame)"; in-rollout took "val
  0.81→0.32, pointwise 0.61." DF "at scale 0.25/1.0 it made collapse WORSE" — "The anti-collapse lever is
  in-rollout, not DF."
- **`diffusion.shortcut=true`** — K=1 sampling via self-consistency, safe because next-state dynamics are
  near-deterministic (a straight noise→target field, so "1×(2d) == 2×(d)").
- **`recon_frac=0.25`, `detach_every=16`, `num_tokens=8`, `encode_arch=vit`** — from the FULL recipe line;
  the header also forbids "NO conv encoder, NO num_tokens>8, NO grad-accum, NO window_stride>1".
- **`lr_warmup_steps=300`, `weight_decay=1e-4`** — already the defaults in `conf/optim/adamw.yaml`, so no
  override was added. Warmup exists because "flow heads regress a clean target from near-pure noise →
  high-variance early gradients."
- **Forbidden levers left alone:** `trainer.accumulate_grad_batches` (LOCKED at 1 — Lightning SUMS
  micro-batch grads → ~N× effective LR → mean-collapse) and `data.window_stride` (LOCKED at 1).
- **`trainer.max_epochs=20`** — the header's metrics are "@ep10-15" and it records collapse past ep20
  (NaN ~ep31, frozen 11.1), so 20 is the informative budget. Config default is 50; raise it only to study
  the collapse itself.

## Repo gotchas found while wiring this up

These are the reason the script looks the way it does. **None required a code change.**

1. **`data.root`, not `data.hf_repo`.** Training resolves either (`resolve_data_root`), but every eval
   routine reads `cfg.data.root` **directly** — `evaluation/routines.py:89,163,218,264,361,414,636,639`.
   With `hf_repo` alone, `data.root` stays `null` and `ood_horizon` would die at the first eval epoch (5).
   The script resolves the snapshot path itself and passes `data.root`.
2. **Frame-cache race.** `load_fpv_frames` writes `<root>/<split>/<cam>_128.npy` with a non-atomic
   `np.save` (`data/dataset.py:69,108`). Two simultaneous cold starts would both decode 288k frames and
   could tear the file. Stage 0 warms it serially: **train 12.75 GB in 158s, val 1.44 GB in 19s**. This is
   the single most important reason the two arms cannot simply be launched back-to-back from cold.
3. **`manifold` / `denoising_*` must be disabled.** `eval_manifold` (`routines.py:218`) and both denoising
   routines call `load_split_episodes_mm(cfg.data.root, "val", img_size=img_size)` with **no `cam` and no
   `repo_id`**, so they default to `fpv` / `"torus"` and cannot find this dataset. Worse, the denoising
   routines gate on `isinstance(m, MultiModalFlow)` — which **is** our model, so they do *not* self-skip —
   and then read torus-only `ecfg.R` / `ecfg.r`, absent from `recorded.yaml`.
4. **`ood_horizon` is safe** and stays on: it passes `cam` + `repo_id` (`routines.py:89`) and takes
   geometry via `getattr(ecfg, "R", None)`, so the torus scene path degrades gracefully. "OOD" here means
   horizon ≫ trained (H up to 2048 vs F=64) on the **val** split — not a distribution-shifted split, which
   this dataset doesn't have.
5. **`recorded.yaml` ships starling's dims AND its rate.** `obs_dim: 16` happens to match, but
   `action_dim: 4` and `dt: 0.0333333` (30 Hz) do not — this dataset is 12-dim at 20 Hz. `dt` feeds
   `fps = round(1.0/ecfg.dt)` in `eval_ood_horizon`, so without `environments.dt=0.05` the filmstrip and
   rollout videos play 1.5× too fast.
6. **`trainer.fast_dev_run` does not exist in this repo.** `Trainer(...)`
   (`train_world_model.py:178-190`) never passes it, so plain `trainer.fast_dev_run=true` is rejected by
   the struct and `+trainer.fast_dev_run=true` would be silently ignored. The wired equivalents are
   `+trainer.limit_train_batches` / `+trainer.limit_val_batches`, which the script uses instead.
   **`wizard/prompt.md`'s pre-flight step 3 is wrong for this codebase** and should be reworded.
7. **Hydra override quoting.** `run_summary` prose containing `(` or `,` fails the override grammar
   ("mismatched input ' ('"). Each arg is shell-single-quoted with the value double-quoted.
8. **The duplicate-summary check is inert under `QUICKDRAW_LOG_ROOT`.** `_assert_summary_unique` is called
   with its default `root="logs"` (`train_world_model.py:90`), globbing `logs/*/auto_run_summary.txt`,
   while runs land in `logs/robocasa-scene4-4h/<run>/`. The summaries here are unique anyway; the smoke
   stage carries `allow_duplicate=true` so re-running the script is safe.
9. **Stale startup banner.** `[startup] data inventory` echoes `conf/data/torus.yaml`'s `splits:` block
   (256 traj × 256 steps, `eval_ood_visual/geometric/dynamics`) — meaningless on this path. The real
   numbers are the line above: `data ready in 65.9s: 242614 train / 27448 val windows`.
10. **`snapshot_download` pulls the 1.13 GB `media/` dir** (2.54 GB total) that the loader never reads —
    the lerobot videos it uses are under `train/videos/`. Cosmetic bandwidth waste only.
11. **`docker compose exec` does not forward host env** (found 2026-08-05). The script `export`s
    `QUICKDRAW_LOG_ROOT`, but that only affects the *host* shell; without an explicit `-e` on every `exec`,
    `make_run_dir` falls back to plain `logs/` inside the container and the two arms scatter outside the
    grouped subfolder. Every `exec` in the script now passes `-e QUICKDRAW_LOG_ROOT=…`.

Items 1, 3 and 5 are latent bugs on the recorded+non-torus path rather than user error — worth fixing in
the repo (thread `resolve_data_root` and `cam`/`repo_id` through the eval routines; make the denoising
routines require torus geometry rather than just `MultiModalFlow`) if this dataset becomes a regular target.
Item 6 is a documentation bug in `wizard/prompt.md` itself.

## Pre-flight results

**1. `check_dataset`** — passed:

```
[check] data=/caches/hf/hub/datasets--isaac-ronald-ward--robocasa-scene4-4h/snapshots/5a3df71e...
        P=8 F=64  (an episode needs >= P+F = 72 steps for 1 window)  repo_id=robocasa-scene4-4h
[check] env 'recorded': obs_dim=16 action_dim=12
[check] train:  235 episodes | len min/mean/max = 615/1103/1749 | obs_dim=16 action_dim=12 -> 242614 training windows
[check] val  :   26 episodes | len min/mean/max = 644/1126/2066 | obs_dim=16 action_dim=12 -> 27448 training windows

[check] OK — dataset fits the config
```

**2. `model_summary`** — 3.22M (base) / 14.21M (large); see the arch tables above.

**3. Smoke (2 train + 2 val batches)** — both configs pass, no nonfinite grads:

| arm | run dir | train_loss | val_loss | nonfinite grads |
|---|---|---|---|---|
| base (3.22M, ViT-MSE) | `train_world_model_2026_08_04_20_52_19_rc4h_smoke_mse` | **2.5908** | **2.3219** | 0 |
| large (14.21M, ViT-MSE) | `train_world_model_2026_08_05_03_07_14_rc4h_smoke_large` | **2.7071** | **2.3088** | 0 |

Large-arm grad norms at step 1 — the image heads dominate and warmup is doing its job (preclip 2.62 clipped
to 1.0): `decode_image 2.03`, `encode_image 1.63`, `decode_proprio 0.218`, `encode_proprio 0.073`,
`flow 0.064`, `backbone 0.045`, `act_enc 0.0008`. The 14.2M arm fits at `batch=32` alongside the 14.2 GB
resident frame store, which was the open risk.

## Expected cost

242,614 windows ÷ batch 32 = **7,582 steps/epoch**; the base smoke measured ~0.5 s/batch → **~63 min/epoch**,
plus validation every 4 epochs ("~as long as the train epoch"). So **20 epochs ≈ 1 day for the base arm**;
the large arm is wider and will be somewhat slower. Both run in parallel on separate GPUs. `best.ckpt`
tracks `pointwise_error` (no `checkpoint_metric` hook on `RecordedEnv`). Resume with
`+resume=<run_dir>/checkpoints/last.ckpt`.

## What to compare when they finish

Both runs write to `logs/robocasa-scene4-4h/train_world_model_<ts>_rc4h_{base,large}/`. The decisive
products, all under `eval_ood_horizon/`:

- `<head>/psnr|ssim|mse|l1` error-vs-step curves and the `@+x` scalars — the primary capacity readout. A
  large PSNR gap means the base arm was underfitting the kitchen scene.
- `image/filmstrip_<i>` — decoded vs ground-truth frames. Read this **alongside** PSNR: a conditional-mean
  decoder is structurally favoured by PSNR, so check whether the larger model actually resolves movable
  objects and the gripper rather than just lowering average error.
- `eval_ood_horizon` / `pointwise_error` — the proprio-side scalar `best.ckpt` is chosen on.
- Watch both for the documented failure mode: autoregressive **mean-collapse** (images → one frozen frame,
  proprio → origin) and any nonfinite-grad skips, especially past ep10.

**Then:** run `eval_interpret` (needs a drafted `conf/interpret/robocasa.yaml` — semantic factors + VLM
prompt, the one non-automatic interpret input — plus `OPENAI_API_KEY`) and `train_reward_model` on whichever
arm wins. Both are env-free and therefore available on this recorded dataset.

## Standing decisions

- `num_tokens × d == L` exactly. **Never pad.**
- Latent normalization is **required**; only the mechanism is a choice (`layernorm | affine | none`).
- `p_tf_end=0` (in-rollout). Never full teacher forcing.
- `accumulate_grad_batches=1` and `window_stride=1` are **LOCKED**.
- Never `uv sync` (strips umap/sklearn).
- **Never edit `src/` while a run is live** — see incidents.
- Action head **off** on this dataset: collapsed 2/2 (ep15, ep7, identical signature).
- Control eval **off**: `RecordedEnv` cannot `step`.

## Incidents worth not repeating

- **Live source edit poisoned a run.** The 04:03 df pair lost *every* eval to
  `TypeError: fig_error_vs_step() got an unexpected keyword 'split_bottom'`. `logging/viz` was imported at
  process start with the old signature; `evaluation/products` imports lazily at the first eval and got the
  new code — one process running two generations. It would have burned 46 GPU-hours emitting nothing, as a
  "FAILED non-fatally ... CONTINUING training" line. **Fix:** 2 consecutive failures of one routine now
  escalate, plus `smoke/eval_products.py` covering that seam.
- **The escalation then over-corrected and killed two healthy runs** (§11, 08-12): it *raised*, so a
  `denoising_filmstrip` that could not render destroyed 4.5 h of completed training on both arms. 2
  consecutive failures now **DISABLE that one routine** (loud line + `eval/disabled/<name>`) and training
  continues. Training is the expensive part; a diagnostic never justifies discarding it.
- **Epoch-0 evals were skipped** as an "untrained baseline" — wrong, `on_train_epoch_end` fires after a full
  epoch (6066 batches), and ep0 is the `p_tf=1` teacher-forced baseline every later epoch should be read
  against. Now evaluated.
- **`autobatch` miscalibrated for the AR path.** It probes at `p_tf=1` (61 GB) but the AR epochs ran at
  88.7/93 GB, over the 75 GB it targeted. Headroom raised 0.25 → 0.35.
- **Metrics that measured the wrong path.** `eval_ae_floor` certified 23.92 dB while the model ran at 20.41;
  `roundtrip` read 0.0 for every run; `kvcache/latent_max_abs_diff` became pure sampling noise the moment
  `stochastic_eval` defaulted true. All three were "correct code measuring a path the model never executes".
