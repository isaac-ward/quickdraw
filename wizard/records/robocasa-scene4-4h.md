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

> **READ THIS BEFORE SETTING `ae_bottleneck` (added 2026-08-27).** It is a **TARGET, not a guarantee**, and it
> must be **PAIRED WITH `img_size`**. Both pyramids size themselves with
> `n_levels = int(log2(short_side // bottleneck))` (`vision.py:346` encoder, after a stride-2 stem; `:385`
> decoder, from full resolution) and `int()` **TRUNCATES**, so any ratio that is not a power of 2 lands
> silently somewhere else:
>
> | img_size | ae_bottleneck | encoder | decoder | actual |
> |---|---|---|---|---|
> | **128** | **8** (the default) | 3 lvls → 8px | 4 lvls → 8px | **8×8, EXACT** |
> | **96** | **6** | 3 lvls → 6px | 4 lvls → 6px | **6×6, EXACT** |
> | 96 | 8 (the default) | 2 lvls → 12px | 3 lvls → 12px | **12×12 — a 64× reduction, not 256×** |
> | 128 | 6 | 3 lvls → 8px | 4 lvls → 8px | truncates back to 8×8 |
>
> **128/8 and 96/6 are the matched pair** — same level counts, same 256× spatial reduction — so results
> transfer between those two resolutions. **At 96px the DEFAULT of 8 silently builds a materially less
> compressed codec** (one fewer level per side), with no warning of any kind. 128px is forgiving; 96px is not.
> Every §20 measurement was taken at 96px with `ae_bottleneck: 6` for exactly this reason.

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

**`bott_bott16` EROSION — its useful life ended around e18-e19.** After plateauing from e14 it began a
slow decline, visible in the eval metrics by e22-e23:

| | e19 | e20 | e21 | e22 | e23 |
|---|---|---|---|---|---|
| OL LPIPS@+64 | 0.344 | 0.342 | 0.338 | 0.354 | **0.368** |
| OL PSNR@+64 | 14.26 | 14.08 | 13.98 | 14.23 | **13.87** |
| ae_floor psnr_mean | 20.07 | 20.04 | 19.83 | 19.89 | — |
| val/loss/total | 0.510 | 0.534 | 0.562 | 0.634 | **0.728** |

At e23 this was characterised as the §16 EROSION pattern and "not the §14 hard collapse", on the grounds
that one-step PSNR@+1 was still 16.88 and the gradients were clean (0 NaNs, 0 infs, 0 skipped, and
`norm_preclip == norm_postclip` at 0.17-0.27, i.e. the clip never engaging). **That reading was right about
the mechanism and WRONG about it being benign — by e32 it is a full collapse:**

| | e28 | e29 | e30 | e31 | e32 |
|---|---|---|---|---|---|
| OL LPIPS@+64 | 0.333 | 0.645 | 0.515 | 0.724 | **0.734** |
| OL PSNR@+64 | 14.20 | 12.46 | 13.49 | 12.21 | **11.56** |
| ae_floor psnr_mean | 19.83 | 19.10 | 18.80 | **18.54** | — |
| ae_floor LPIPS@+64 | 0.232 | 0.267 | 0.268 | **0.303** | — |
| `grad/norm_preclip` | 0.43 | 3.88 | 1.04 | **18.33** | **17.05** |

OL LPIPS has more than DOUBLED off its e21 best (0.338 -> 0.734) and OL PSNR@+64 has lost 2.7 dB. The floor
is 1.56 dB below its e14 peak.

**The mechanism is a gradient blow-up, and the e23 gradient reading has to be updated.** `norm_preclip` is
18.33 at e31 — about **70x** the 0.17-0.27 seen at e23 — so the clip at 1.0 is now truncating essentially
the whole gradient every step. There are still no NaNs or infs, so this is not numerical failure: the loss
landscape is genuinely blowing up and the clip is all that prevents outright divergence while the model
degrades. Onset is between e28 (0.43) and e29 (3.88), exactly where OL LPIPS jumped 0.333 -> 0.645.

**LESSON: on this config `grad/norm_preclip` is the early-warning metric, not val_loss.** val_loss had been
rising since e15 while every eval metric held flat, then oscillated +-0.3 for ten epochs — it never
cleanly marked the turn. The gradient norm did, in one epoch.

Motion is the lone exception, still creeping up (0.447 @e23) while the codec metrics decline — the
signature of the dynamics continuing to fit through an eroding codec, which is exactly what
`latent_loss_weight=10` was raised to slow (§16). At a 16x16 bottleneck it evidently needs to be higher
still: **`latent_loss_weight` > 10 is the natural companion knob to `ae_bottleneck` > 8**, and is untested.

Practical consequence: read `bott16`'s numbers as its BESTS (floor 20.10 @e14, ae_floor LPIPS 0.227 @e18,
OL LPIPS 0.338 @e21) — `best.ckpt` holds them — and treat epochs past ~e19 as actively harmful. The
40-epoch schedule was too long for this arm.

**`bott_recon1` — A RECORDED CLAIM HERE WAS WRONG AND IS NOW REVERSED.** At ep5 this section said
"`recon_frac=1.0` looks like a straight loss ... the obvious kill candidate". That was a **methodological
error**: `recon1`'s EARLY values were compared against the baselines' BEST-over-all-epochs values. Once
epoch-matched — the way `bott16` was analysed — it leads on the metrics that actually matter here:

| epoch-matched, e7-e8 | `bott_recon1` | `BASE long` | `bott_bott16` |
|---|---|---|---|
| ae_floor LPIPS@+64 (e7) | **0.228** | 0.241 | 0.268 |
| OL LPIPS@+64 (e8) | **0.373** | 0.374 | 0.387 |
| OL motion_ratio mean (e8) | **0.440** | 0.378 | 0.341 |
| ae_floor psnr_mean (e6) | 19.59 | 19.51 | **19.71** |
| OL PSNR@+64 (e6) | 14.52 | 14.63 | 14.53 |

It leads on perceptual distance AND motion — precisely the two axes `bott16` lost and the two the original
complaint was about — and its motion at e8 (0.440) already exceeds `bott16`'s best across 20 epochs
(0.435). Even at ep5, epoch-matched, it was already ahead of `long` on LPIPS (0.415-0.444 vs 0.421-0.488);
the "losing" call was an artifact of the comparison, not the data.

**This is the same error shape as the `20.14 dB` tag mixup recorded above: comparing across different
aggregations.** Both times it turned a win into an apparent loss. RULE: on this dataset, compare
EPOCH-MATCHED and SAME-TAG, and quote best-over-epochs only against another best-over-epochs.

**THE CLEAN FOUR-WAY AT e11** (epoch-matched, same-tag — the comparison rule this section had to learn
twice). Every metric has a different winner, and the split is not arbitrary:

| metric @e11 | `recon1` | `bott16` | `long` | `sharp` | winner |
|---|---|---|---|---|---|
| ae_floor psnr_mean | 19.59 | **19.97** | 19.41 | 19.44 | `bott16` |
| ae_floor LPIPS@+64 | 0.213 | 0.241 | 0.216 | **0.173** | `sharp` |
| OL PSNR@+64 | **14.33** | 14.06 | 14.22 | 14.13 | `recon1` |
| OL LPIPS@+64 | 0.363 | 0.364 | 0.368 | **0.317** | `sharp` |
| OL LPIPS@+128 | **0.332** | 0.363 | 0.384 | 0.353 | `recon1` |
| OL motion_ratio mean | **0.488** | 0.375 | 0.413 | 0.446 | `recon1` |

**A HORIZON CROSSOVER, which is the new mechanism-level finding.** `sharp` wins perceptual distance at
horizon 64 (0.317 vs `recon1`'s 0.363) but LOSES it at horizon 128 (0.353 vs 0.332). So the two levers buy
sharpness at DIFFERENT horizons, and the reason is mechanical: `recon_frac=1.0` supervises EVERY step of
the rollout against its true frame, so its benefit should compound with horizon — which is exactly what a
crossover between +64 and +128 looks like. `decode_base` buys a stronger decoder, which helps most where
the latent is still accurate, i.e. early.

**This matters for the standing goal** ("slightly sharper results that stay coherent for like 32-64 steps
... and I want to make sure the movement is being modeled", user, throughout). `recon1` wins BOTH halves of
that sentence at e11: long-horizon perceptual quality and motion (0.488 against 0.375-0.446, the highest
any arm has reached at this epoch). `sharp`'s LPIPS win is real but concentrated at the short end.

**Revised standing picture — four levers, four different wins, and they look orthogonal:**



| lever | owns |
|---|---|
| `ae_bottleneck=16` | the reconstruction **floor** (+0.51 dB, stable, fewer params) |
| `decode_base=64` | **SHORT-horizon** sharpness (ae_floor LPIPS 0.142; OL LPIPS@+64 0.303) |
| `recon_frac=1.0` | **LONG-horizon** sharpness (OL LPIPS@+128) **+ motion** + rollout PSNR; also the only arm with NO codec erosion |
| `latent_loss_weight` > 10 | UNTESTED companion to a raised bottleneck (see the erosion note above) |

### FINAL: `bott_bott16`, 40/40 epochs (08-23 04:00)

| metric | best | final e39 | `long` best | `sharp` best |
|---|---|---|---|---|
| ae_floor psnr_mean | **20.100** @e14 | 11.98 | 19.586 | 19.499 |
| ae_floor LPIPS@+64 | 0.227 @e18 | 0.719 | 0.192 | **0.142** |
| OL PSNR@+64 | 14.789 @e9 | 11.60 | **14.817** | 14.323 |
| OL LPIPS@+64 | 0.333 @e28 | 0.753 | 0.323 | **0.303** |
| OL LPIPS@+128 | 0.336 @e16 | 0.754 | 0.333 | **0.310** |
| OL motion mean | 0.492 @e29 | 0.373 | 0.516 | **0.571** |

The collapse ran to completion: floor 20.10 -> 11.98 (**-8 dB**), every LPIPS above 0.7. The last 20 epochs
were purely destructive. **It won exactly one metric — the AE reconstruction floor (+0.51 dB over the best
of 15 prior runs) — and lost every rollout metric.** So the 8x8 bottleneck was a real binding constraint on
CODEC FIDELITY, and relieving it does not improve the ROLLOUT, which is what the goal is about.

**A CHECKPOINT-SELECTION PROBLEM WORTH FIXING BEFORE ANY OF THESE CKPTS ARE REUSED.** `best.ckpt` for this
run is pinned to **e8** (`best monitor=0.53341`) — not e14 (floor peak), not e18 (ae_floor LPIPS peak), not
e16/e28 (OL LPIPS peaks). The monitor is `val/metric/proprio/<env.checkpoint_metric>`, a **proprio** metric
(`training/lit.py:190`, wired at `train_world_model.py:248`), so on this dataset `best.ckpt` selects for
proprio quality and is **blind to every image metric this record judges on**. Not introduced by this batch,
but it means "load best.ckpt and look at the images" silently gets e8. Either monitor an image metric on
image-modality runs, or always select the epoch by hand from metrics.jsonl.

### `bott_recon1` at ep17 of 40 (08-23 06:12) — the best ROLLOUT model measured on this dataset

Its own trend is flat-or-improving with no sign of `bott16`'s e19 turn: OL LPIPS@+128 0.332 -> 0.308 across
e12-e17, motion 0.481 -> 0.558 (peaking 0.564 @e15), ae_floor LPIPS pinned at 0.205-0.207, 1-step PSNR
16.84-16.96. Gradient norms 0.25-0.32 and FALLING; val_loss flat at 0.467 for seventeen epochs.

Best-vs-best (recon1 has 17 of 40 epochs; the other three are complete):

| metric | `recon1` | `bott16` | `long` | `sharp` |
|---|---|---|---|---|
| ae_floor psnr_mean | 19.68 | **20.10** | 19.59 | 19.50 |
| ae_floor LPIPS@+64 | 0.205 | 0.227 | 0.192 | **0.142** |
| **OL PSNR@+64** | **14.99** | 14.79 | 14.82 | 14.32 |
| OL LPIPS@+64 | 0.318 | 0.333 | 0.323 | **0.303** |
| **OL LPIPS@+128** | **0.308** | 0.336 | 0.333 | 0.310 |
| OL motion mean | 0.564 | 0.492 | 0.516 | **0.571** |

`recon1` holds the best open-loop PSNR@+64 (14.99) and best long-horizon LPIPS (0.308) of ANY run here, and
matched `sharp` on motion (0.564 vs 0.571) at e15 rather than e18.

**THE PROPERTY THAT MATTERS MOST, and it is new:** `recon1`'s LPIPS is nearly FLAT ACROSS HORIZON —
@+64 0.318 vs @+128 0.308, i.e. it gets *slightly better* at the longer horizon. Compare `sharp`
0.303 -> 0.310 and `bott16` 0.333 -> 0.336, both degrading. Perceptual quality that does not decay from 64
to 128 steps is precisely what "stay coherent for like 32-64 steps" asks for, and no other lever produces
it. The e11 horizon crossover has narrowed but held.

**FOLLOW-UP, STAGED AND NOT RUN** (`wizard/scripts/robocasa-bottleneck-2.sh`, needs a free GPU):
`ae_bottleneck=16` **+** `decode_base=64`. `bott16` gains PSNR from resolution but loses LPIPS to the
1.05M params the dropped level cost; `sharp` shows `decode_base=64` is worth ~0.05 LPIPS on its own.
Combining them tests whether resolution and channel width are ADDITIVE. This is exactly why the A/B was
based on `long` rather than `sharp` — it kept `decode_base` free as the next lever.

**CAVEAT ADDED after the `recon_frac` reversal above:** this is no longer obviously the best next config.
`recon_frac=1.0` is orthogonal to both and is currently the strongest single signal on the actual
complaint (sharpness AND motion). The config the evidence points at is
`ae_bottleneck=16 + decode_base=64 + recon_frac=1.0`, but that stacks THREE levers at once and would be
uninterpretable if it failed. Decide between "clean 2-lever test" and "stack everything" before launching.

**Do NOT kill `bott_recon1`** — it is the most informative arm running and at 2.11 h/ep will not reach
`long`'s ep17 LPIPS peak until ~08-23 22:00. `bott_bott16` is the arm that has said everything it has to
say: plateaued on every metric from e14 with 20 epochs left. If a GPU is needed, stop that one.

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

## 18. THE DYNAMICS LOSS NEVER FOLLOWED p_tf — found, fixed, and the fix COLLAPSES (08-24/25)

### The defect

`MultiModalFlow` ran TWO forward computations. The rollout fed the **decode** loss; a second, fully
teacher-forced parallel pass fed the **dynamics** loss and never saw the rollout — `loss_terms` received
`pred_bag` and never referenced it. So `p_tf` ramped 1 -> 0, the decode loss duly became autoregressive, and
**the transition function stayed teacher-forced for the entire run.** It was never once asked to step from a
latent it produced itself. Full write-up: `design/flow.md`.

Not unnoticed: `MultiModalDistribution.loss_terms` documents the same behaviour for itself and says "the
p_tf ramp affects only the rolled decode losses ... **as for Flow**". A considered position, not an oversight.

### What made it visible

Two new metrics (commit `60dc373`), because every metric we had was either pixel-space (which saturates on a
scene that is mostly right) or `motion_ratio`, which the repo documents as direction-blind:

* `latent_cos` = cos(z_pred, z_true) per horizon — how far OFF COURSE the rollout is
* `latent_motion_ratio` = per-step angular step size vs the truth

On `bott_recon1`: cos 0.98 / 0.71 / 0.48 / 0.26 / **0.14** at h=1/8/16/32/64 — by 64 steps the latent is
~orthogonal to the truth, ~82 degrees off, while UNDER-rotating (ratio 0.95 -> 0.56). Per-step error ~11
degrees compounding as a random walk toward the 90-degree ceiling.

Ruled out first, each by measurement rather than argument: KV-cache train/eval divergence (null, +-0.01 dB at
every horizon), action off-by-one (clean, audited), `stochastic_eval` (refuted — DETERMINISTIC is worse, cos
0.037 vs 0.128 @h64), `sampling_steps` (null for K>=2; K=1 badly broken), `latent_norm` (layernorm measured
better than none), and the decoder (decoding TRUE latents gives LPIPS 0.19 vs 0.31 for predicted — the
decoder renders sharply when handed a correct latent).

### The fix and the A/B

`model.dynamics_follows_p_tf` (commit `e9de2b0`): the dynamics loss conditions on the rollout's FEEDS — what
each step actually STOOD ON — instead of clean latents, so one schedule governs both losses. No new loss
term, no new schedule, no new weight; `p_tf` untouched. Feeds, not raw predictions, because at `p_tf>=1` no
rollout runs, so there are no feeds and the context is clean, which IS teacher forcing — no special-case
gate. Three audits caught four drafting errors (in-place view corruption of `z`, a DF-ordering trap, an
inverted detach justification, a smoke caller passing a full-L tensor).

`dfptf_on` vs `dfptf_off`, identical binaries, batch 27 both, 20 epochs.

**THE MECHANISM FIRES.** `latent_cos@+32` 0.542 vs 0.256 (**2.1x**), `@+64` 0.221 vs 0.159 (**1.4x**), pixel
`motion_ratio` 0.654 vs 0.341 (**1.9x**). The pre-registered falsification condition is cleared.

**AND THEN IT DESTROYS ITSELF.** The arm collapsed between e5 and e6 -- not a quality tradeoff, a
TRAINING-STABILITY FAILURE:

| | e5 (its best) | e6 | e7 |
|---|---|---|---|
| `grad/norm_preclip` | 0.709 @e3 | — | **1.84e+07** |
| ae_floor psnr_mean | 19.1 | **11.4** | — |
| OL LPIPS@+64 | 0.514 | 0.748 | 0.744 |
| **latent_cos@+64** | **0.263** | **-0.055** | **-0.112** |

Gradient norm to **18 million**, with num_nans/num_infs both 0 -- the clip at 1.0 laundering the blow-up,
exactly as §17 documented for `bott16`. And `latent_cos` went NEGATIVE: the predicted latent points AWAY
from the truth, worse than random. The control at e7 is untouched (grad 0.464, floor 19.4, train_loss 0.307
vs 9.12, and OL LPIPS@+64 improving monotonically 0.676 -> 0.381 across e0-e6).

**Up to e5 the fix worked**: latent_cos@+64 0.263 vs the control's 0.146, a 1.8x improvement. Then it went.

This is the exact risk `design/flow.md` flagged in its own Decisions section -- "the target stops being a
fixed function of the data (it moves with the model)" -- where I claimed p_tf's warmup would mitigate it.
It did not: warmup is ONE epoch and the collapse came at six.

**SO DROPPING THE `frac` MIXING KNOB WAS A MISTAKE.** It was removed on the argument that it was "redundant
with p_tf". That was right about the SCHEDULE and wrong about the STRENGTH: p_tf controls WHEN you switch to
hard mode, not HOW MUCH of the context is the model's own. Substituting all F-1 positions is too strong, and
the evidence now converges from two directions:

| context corruption | motion | OL LPIPS@+64 | stable? |
|---|---|---|---|
| none (control) | 0.333 | 0.450 | yes |
| **DF, isotropic lambda=0.1** | 0.362 | **0.419** | yes |
| **feeds, FULL strength** | 0.672 | 0.514 then collapse | **NO** |

A dose-response curve whose optimum is well below full strength. The corruption FAMILY works; this dose does
not. Next: a mixing FRACTION (a strength knob, not a schedule), DF and the fix together, or simply rerun DF.

Earlier reading at e2-e3, kept because it was the first signal and it pointed the right way:

| e2 -> e3 | OL LPIPS@+64 | OL LPIPS@+128 | OL PSNR@+64 |
|---|---|---|---|
| dfptf ON | 0.496 -> **0.533** | 0.531 -> **0.570** | 13.77 -> 13.55 |
| dfptf OFF | 0.450 -> 0.420 | 0.478 -> 0.438 | 13.89 -> 14.11 |

Reading: **over-committing.** It moves 1.9x more but not accurately enough, and LPIPS punishes
confident-but-wrong harder than hedging. Only 4 epochs, and at batch 27 both arms have 3.9x fewer gradient
steps than any b7 run — but the direction is consistent and the gap is widening.

### DIFFUSION FORCING IS THE BEST THING RUN SO FAR, and it is the gentler version of the same idea

`df_recon1` (`variations.noise_injection.observations_encoded_pre_fusion.scale=0.1`) corrupts the SAME
context, with isotropic noise instead of the model's own structured errors:

| @e2, matched epochs | OL LPIPS@+64 | OL LPIPS@+128 | motion |
|---|---|---|---|
| **DF 0.1** (b7) | **0.419** | **0.450** | 0.362 |
| recon1 plain (b7) | 0.474 | 0.485 | 0.278 |
| dfptf OFF (b27) | 0.450 | 0.478 | 0.333 |
| dfptf ON (b27) | 0.496 | 0.531 | 0.672 |

So the corruption FAMILY works; the STRUCTURED, full-strength version overshoots. The obvious next
experiments are the middle ground: partial feed mixing, or DF and the fix together. **`df_recon1` was killed
at e2 to free a GPU and should be rerun** — it was winning.

Cross-run note: the two controls land in the same place despite 4x different batch (`dfptf OFF` b27 vs
`recon1 plain` b7 at e2: LPIPS@+128 0.478 vs 0.485, PSNR 13.89 vs 13.78), so cross-run comparison here is
NOISIER than within-A/B, not useless. DF's 0.450 is a real target.

### The filmstrip's "noise" panel is the PREDICTION plus noise, not GT plus noise

`prev = hist[0, -1]` is the ROLLOUT's own latent at that horizon and `decode_bag` computes
`LN(prev + x)`, so the `noise` column is `decode(LN(rollout_latent_at_h + unit Gaussian))`. GT never enters
it. Three consequences, all of which look like bugs and are not:

* blurry even at k=0 — `prev` is ALREADY wrong (~82 degrees off at h=64), so `decode(prev)` is blurry before
  any noise is added;
* blurrier with horizon — `prev` degrades with horizon; the row index IS the accumulated drift;
* the k-steps never approach GT — the flow outputs a SMALL residual, so `LN(prev + x_k)` converges to
  `decode(prev)`. **The refinement's ceiling is the rollout's own current state.** It can only undo the noise
  it was handed; it cannot repair drift baked into `prev`, which the filmstrip holds fixed.

So the k-gain (+1.2 to +2.0 dB at h<=32, **-0.16 dB at h=64**) measures how much injected noise the flow can
undo — and at h=64 it undoes none, i.e. the velocity field is uninformative there. The filmstrip tests the
flow's LOCAL FIELD, not rollout quality; rollout quality is `latent_cos` and OL LPIPS.

### Instrumentation defects found and fixed along the way

* `eval_ood_horizon` was the only eval routine without `@torch.no_grad()` — its latent pass built a full
  autograd tape over a 128-step rollout every eval epoch (`c599869`).
* `latent_cos` described a DIFFERENT TRAJECTORY than the LPIPS curves beside it: `latent_pass` rolled out a
  second time, and with `stochastic_eval: true` each rollout draws fresh eps. Now reuses the same rollout's
  bag (`c599869`).
* `latent_motion_ratio` logged **9.4e11**: it took the mean of per-episode ratios, so one episode with a
  near-zero true delta blew up the average. Now a ratio of means, as pixel `motion_ratio` always did
  (`0c27d9c`). **Both live dfptf arms carry the broken version** — that curve is unusable for this A/B;
  `latent_cos` is unaffected.

### Throughput: decode chunk+checkpoint (`decode_chunk_train`)

`recon_frac=1.0` cost 13.17 GB/sample -> batch 7. ~78% of per-sample memory is the DECODER, across TWO
passes (the decode loss AND the roundtrip anchor — an adversarial audit caught that the first patch chunked
only one, which would have landed at batch ~10 instead of ~29). Chunking the velocity FORWARD and catting
outputs keeps ONE loss call, so there is no mean-of-means weighting to get wrong:

    measured on recon_losses:  peak 7.716 -> 1.696 GB (-78.0%), loss equal to 6.6e-07 RELATIVE
    live:                      13.169 -> 3.178 GB/sample, batch 7 -> 27, 5013 -> 1300 batches/epoch
    epoch:                     ~6970s (val-adjusted b7) -> 5149s = ~26% faster

Costs ~1.33x decode compute; ep0 (parallel path, no AR loop to amortise against) is 23% SLOWER per sample,
and the win only appears once the dispatch-bound AR rollout amortises across 4x the samples.

**First measured phase breakdown in this project** (`record_function` markers, `c599869`) — every prior "X is
N% of the step" figure, in design docs and audits alike, was inferred from epoch totals:

    qd/ar_rollout   63.5%   |   qd/recon_losses  31.0%   |   loss_terms 4.1%   |   shared_encode 1.3%

which revises the AR rollout up from an inferred ~51% and decode down from an inferred ~40%.

## 19. ABSOLUTE vs RESIDUAL, on a generative decoder — and the collapse diagnosed by probe, not by run (08-25/26)

Two runs of ~10 h each were spent on collapses before anything was measured about the mechanism. This section
is mostly about the ~10 GPU-MINUTES that replaced the third.

### 19.1 What §18 left: two changes, one failure mode

`dynamics_follows_p_tf=true` (§18) and `predict: absolute` both produced the SAME triad, two epochs apart:

| | latent_cos@+64 pre | grad/norm/flow | floor PSNR | latent_cos after |
|---|---|---|---|---|
| §18 dfptf (e5→e6) | 2.1× control | 0.42 → 1.3e7 | 19.1 → 11.4 | NEGATIVE |
| `abs_pred` (e2→e3) | 0.261 (2.0× control) | 0.341 → **108.9** | 18.35 → **8.94** | **−0.056** |

And the same FINGERPRINT: `grad/norm/{flow,backbone,encode_*,act_enc}` all `inf`, while `decode_image` (5.75)
and `decode_proprio` (4.00) stayed FINITE. Everything inside the recurrent path, nothing outside it.

`abs_pred` was WINNING when it died — epoch-matched against its control at e0/e1/e2 it took every open-loop
metric, with a **3× smaller dynamics penalty** (0.028 vs 0.083 at e1) on a *worse* floor, which is exactly the
profile the standing goal prefers. Its latent_cos advantage also WIDENED with depth (1.4× at +32, 2.1× at +64,
3.3× at +96), the signature of re-deriving the state rather than accumulating increments.

### 19.2 The Jacobian probe — 4 GPU-minutes that killed the planned fix

`_oneoff_jacobian_probe.py` power-iterates the per-step operator norm ‖∂bag_{h+1}/∂bag_h‖ over 32 rollout
steps. `_oneoff_jacobian_init.py` does the same on FRESH weights, same seed, only `predict` differing.

| | mean | max | min |
|---|---|---|---|
| residual e3 (trained) | **0.9918** | 1.0258 | 0.9654 |
| residual e11 (trained) | **0.9687** | 1.2544 | 0.9088 |
| absolute e3 (trained, post-collapse) | 1.6246 | **11.1942** | 0.0980 |
| residual **at init** | 0.7978 | 0.9086 | — |
| absolute **at init** | **0.2507** | 0.3002 | — |

**Absolute is MORE contractive at init, not less.** That refutes the structural story that had been driving the
plan ("absolute drops the identity skip → `J` instead of `I + J` → less contractive by construction"). The 11×
gains are LEARNED, within three epochs. The real difference is an ANCHOR: residual's map is pinned near unit
gain by construction — 3% spread at e3, still pinned at e11 — and training can only perturb it. Absolute's is
free, and it drifted.

**This killed `detach_every=16`, which was to have been the anti-collapse lever in both arms.** `detach_every`
bounds the LENGTH of the gradient chain. Residual doesn't need it (per-step gain 0.97–0.99 → the 32-step
product CONTRACTS: measured amplification 0.59 at e3, 0.41 at e11 — which is also why `mm_flow.yaml:150` found
32 stable and 16 not). Absolute isn't helped by it either: **halving how many steps you cross does not fix one
step whose gain is 11.19.** Reverted to the validated 32.

Corollary for instrumentation: `abs_pred` had NO pre-collapse checkpoint (val every 4 epochs → first save at
e3, post-mortem), so the trained-absolute row above cannot separate cause from consequence. Hence
`check_val_every_n_epoch=1` in §19.4.

### 19.3 The generative decoder — a prior "null" that was a CONVERGENCE artifact

`bsp32mse.yaml:52` records `decode_kind=flow` as buying "~nothing on sharpness (LPIPS 0.385 vs 0.390), cost
1.1 dB of PSNR". `_oneoff_decode_kind.py` retested it properly: frozen trained encoder → identical real
latents → two fresh decode heads, same seed, same data, same steps.

| steps | mse LPIPS | flow committed | flow sampled |
|---|---|---|---|
| 1200 | **0.2537** | 0.2852 | 0.3328 |
| 3600 | 0.0922 | **0.0839** | 0.0883 |
| 6000 | 0.0604 (31.04 dB) | **0.0505 (32.43 dB)** | 0.0547 (32.10 dB) |

**Flow is not worse, it is SLOWER** — it crosses mse between 2400 and 3600 steps and finishes 16% better on
LPIPS and +1.4 dB. The harder objective (denoise at every τ, not only τ=1) costs convergence speed and buys
quality. The old null is explained: that run was at `recon_frac=0.25`, i.e. a QUARTER of the decode gradient
steps — almost certainly still pre-crossover.

**But `decode_stochastic` must be OFF, and its sign FLIPS with budget.** At 500 steps sampling beat committing
(0.5507 vs 0.6475 LPIPS) — the distortion–perception tradeoff, and the entire reason the flag was written. At
6000 it LOSES on both (0.0547 vs 0.0505, 32.10 vs 32.43 dB). The tradeoff only pays when `p(obs|tokens)` has
real spread; a converged head on a good code is nearly a point mass, so sampling adds noise around an accurate
mean. **This does not reverse in open loop** — a drifted latent makes the mean WRONG, not UNCERTAIN, and
sampling in pixel space explores around the wrong point rather than toward the true frame.

Caveat, because the numbers look far too good: 6000 × batch 16 over 192 frames is ~500 passes over a tiny set,
so both heads are MEMORISING and LPIPS 0.05 is nowhere near the real floor (~0.24). It is a capacity/
convergence comparison, not a generalisation one. The sampled-vs-committed half is the sturdier result — same
weights, same data, only the decode path differs.

Two more things this line of work found:
- **`decode_kind=flow` was a NO-OP at inference before `decode_stochastic` existed** (`21786e2`). `decode()`
  hardcoded `deterministic=True` and, for `param=x0`, `steps=1`; that makes the first iteration
  `velocity(zeros, τ=1, cond)` — literally the `no_noise`/mse line. Measured max|difference| **0.0**.
- **The flag then leaked into the TRAINING objective** (`5b09034`), via `roundtrip_losses → to_obs → decode`
  at `latent_loss_weight: 10`. Since `E‖x̂−t‖² = ‖E x̂−t‖² + Var(x̂)`, scoring a SAMPLE there trains the
  sampler's variance toward ZERO — the anchor would have cancelled the sharpness at 10× the weight of the loss
  that wanted it. Fixed with `commit=True`; the anchor measures the CODEC, which is deterministic by definition.

**No reweighting.** The mse/flow raw-loss ratio moved 0.839 (@500) → 1.210 (@6000), so it is not a stable
property of the objective. `model.modalities.1.weight` stays 1.0. (This retires the concern — raised in the
adversarial audit and endorsed here — that swapping `decode_kind` silently reweights the only autoregressive
gradient in the model.)

### 19.4 The pair (`absres_*`, launched 08-26 00:37, RUNNING)

Shared: 128px, `ae_bottleneck` 8, `recon_frac` 1.0, **`detach_every` 32**, generative conv U-Net flow decoder
(`decode_kind=flow decode_arch=unet decode_param=x0 decode_steps=6 decode_base=32`), `decode_stochastic` FALSE,
`p_tf_dynamics` 1.0, **`check_val_every_n_epoch=1`**. Autobatch chose **27 for both arms**.

| arm | GPU | `model.diffusion.predict` |
|---|---|---|
| `absres_residual` | 0 | residual — the control |
| `absres_absolute` | 1 | absolute |

ONE variable. **This pair is NOT expected to prevent the collapse** — nothing in it anchors the step map, and
`detach_every` was just shown to be the wrong lever. It is expected to REPRODUCE it with per-epoch checkpoints,
so the Jacobian of a HEALTHY absolute model (e1/e2) can finally be measured against the residual arm's.

Pre-registered expectations, so the read is scored rather than rationalised:
- **Both arms:** floor LPIPS 0.20–0.22 (old control 0.237), floor PSNR 20–20.5 (old 19.3), arriving by e0 —
  the real run feeds the decode head 27×64 = 1728 frames/step, clearing §19.3's crossover budget in ~30 steps.
  *Floor WORSE than 0.237 by e2 ⇒ the probe did not transfer and `bsp32mse.yaml:52` was right.*
- **Residual:** penalty ~0.10, unchanged — the decoder should move floor and OL together. *Penalty < 0.08 ⇒ the
  decoder is helping the DYNAMICS, via cleaner gradients through `recon_frac=1.0`. That would be new.*
- **Absolute:** beats residual on every OL metric at e0–e2 (latent_cos@+64 ~0.30 vs ~0.14, penalty ~0.03 vs
  ~0.10), then collapses at ~e3. *Surviving past e5 ⇒ the generative decoder changed the gradient landscape,
  and a ~0.03 penalty held to e8 is the largest win here since `recon_frac=1.0`.*

### 19.5 Instrumentation added this round

- **Filmstrip FLOOR column** (`868a35f`) — each row now decodes the TRUE latent for its frame, so codec error
  (floor↔GT) and dynamics error (kK↔floor) separate BY EYE, per horizon, ON THE SAME FRAME. Previously an OL
  number could only be compared against a floor averaged over DIFFERENT frames. Panel PSNR text removed (45
  numbers is noise); the grid + `ep_idx`/`t_ctx` provenance go to `logs/epoch_<step>/eval_flow/
  denoising_filmstrip_<i>.npz`.
- **`seed = step` is a TRAP for cross-epoch reads.** Each epoch draws a different episode and `t_ctx`, so the
  control's h=64 reading 16.99 (e2) → 13.71 (e11) is two different scenes, NOT degradation. Same-epoch
  cross-run comparisons ARE matched. Pin `eval.denoising_seed` to make a within-run series comparable.
- **`p_tf_dynamics`** (`46dca7b`) — the boolean became a probability, the strength knob §18 concluded was
  needed. 1.0 bit-identical to the historical always-clean path (verified: 1.108671 either way), null/0.0 the
  full substitution that collapsed (1.866729), 0.9 in between (1.129133). RNG only drawn for 0<q<1.

## 20. THE DECODER WAS THE PROBLEM AFTER ALL — `decode_arch="up"` wins the objective (08-26/27)

Two measured pathologies in the mse image decoder, one rewrite, and a result that reversed twice before
settling. Also the session's methodological low point, recorded because it cost three probes.

### 20.1 The two pathologies

`vision.ConditionalUNet` served BOTH `decode_kind: mse` (a tokens->image DECODER) and `decode_kind: flow` (an
image DENOISER) through one `velocity(x, temb, cond, demb)`. Consequences, measured on CPU at the live geometry:

**(a) THE DOWN PATH CONVOLVES ZEROS.** In mse mode `x` is an all-zero tensor (`flow.py` no_noise branches in
both `loss` and `_sample`). Skip-tensor INTERIOR spatial std is **EXACTLY 0.000000** at levels 0-2; the only
spatial structure is a zero-padding border halo, which engulfs the whole map by level 3 (std 0.207). So the up
path's sole absolute-position signal was a padding artifact -- the StyleGAN3 / Xu et al. 2021 "positional
information hidden in padding" pathology. Cost: **693,888 params (15.9% of 4,373,763)** and **~33% of decoder
activation volume**, at FULL resolution, TWICE per step (decode loss + roundtrip anchor).

**(b) A RANK-640 CHOKE ON A 4,096-FLOAT LATENT.** Exactly two routes from `cond` to pixels:
`cond_to_spatial = Linear(4096 -> 512)` (**2,097,664 params = 48.0% of the decoder**) reshaped to a 2x2 map and
nearest-upsampled to the bottleneck, plus `g = cond.mean(1)` (rank <=128, the token MEAN, the ONLY per-block
conditioning every FiLM block receives). **640 of 4,096 floats = 15.6% visible**; ~3,456 latent directions
produce PIXEL-IDENTICAL images.

**(b) quantitatively explains section 17's nulls.** At `num_tokens=8` the bag is already 1,024 floats > 640, so
the ENTIRE 8->64 sweep saturated the readout -- predicted null, observed null (floor flat 18.7-20.4 dB). And
`ae_depth` 4->6's null was TRIVIAL: `modalities.py:200-216` routes that knob to the ViT paths only, so for the
conv arm it never reached the decoder at all.

### 20.2 The fix (`c7801d2`, models/decoders.py)

`decode_arch: "up"` -- UP ONLY, no analysis path, no skips. `TokenGridReadout` (learned bott_hw x bott_hw query
grid cross-attending the token bag + a ViTBlock mixer) replaces the dense flatten; `TokenPool` (attention
pooling) replaces `cond.mean(1)`. Readout bandwidth **36x128 = 4,608 vs 640**, which MATCHES the 4,096-float
latent. Params **1,515,907 @base32 / 4,760,003 @base64** vs the U-Net's 4,373,763.

The mid self-attention is LOAD-BEARING, not decoration: `grid_q` is content-INDEPENDENT, so before its keys
learn slot discrimination the attention is near-uniform and every cell receives approximately the token MEAN --
the module would momentarily reproduce the pathology it exists to remove.

**WIRING LANDMINE FIXED.** The dispatch was `if unet ... else vit`, so `decode_arch: "up"` would have SILENTLY
built the ViT head and the run would have tested nothing. Now explicit with a RAISE on unknowns (4 of
`smoke/up_decoder.py`'s 14 checks assert those raises).

**THE PRIOR "ViT DECODER FAILED" RECORD IS EVIDENTIALLY ROTTEN.** `bsp32mse.yaml:57` blames "motion 0.141, and
it collapsed". The run logs say `bsp32vit` posted the best bespoke floor of the program and then died at **e15
from a RECURRENT-PATH gradient explosion** -- inf on flow/backbone/encoders, BOTH decoders finite at ~1.9 --
the same fingerprint as bott16 and predict=absolute, neither of which had a ViT decoder. The 0.141 is epoch 7's
single lowest value of a metric the record itself calls direction-blind. That comment should be rewritten; it
nearly killed this line of work.

### 20.3 THE RESULT — `dec_unet32` vs `dec_up64`, 96px, PARAM-MATCHED (4,375,875 vs 4,760,003, 8.8% apart)

Epoch-matched raw **OL LPIPS@+128**, gap = unet minus up (positive = up better):

| ep | unet@32 | up@64 | gap |
|---|---|---|---|
| 3 | 0.3712 | 0.3770 | −0.006 |
| 6 | **0.2957** | 0.3203 | −0.025 |
| 7 | 0.3036 | 0.2954 | +0.008 |
| 8 | 0.2878 | 0.2744 | +0.013 |
| 9 | 0.3121 | 0.2587 | +0.053 |
| 10 | 0.3065 | 0.2503 | +0.056 |
| 11 | 0.3206 | 0.2536 | +0.067 |
| 12 | 0.2935 | **0.2429** | +0.051 |

**Six consecutive matched epochs for up@64, margin growing 0.008 -> 0.067.** Before e7 the sign ALTERNATED, so
this is a crossover, not a run of luck. Best-so-far @+128: **unet 0.2878 (e8), up 0.2429 (e12)**.

Horizon curves at e11 -- up@64 is better at EVERY horizon and its curve DESCENDS past +32 while the control's
flattens:

|  | +1 | +8 | +16 | +32 | +64 | +96 | +128 |
|---|---|---|---|---|---|---|---|
| unet@32 | 0.222 | 0.379 | 0.354 | 0.329 | 0.314 | 0.394 | 0.321 |
| up@64 | 0.191 | 0.333 | 0.304 | 0.279 | 0.273 | 0.320 | **0.254** |

**And the floors DIVERGE — the control's codec is eroding:**

    unet@32  e7:0.1889 e8:0.1924 e9:0.2067 e10:0.2138 e11:0.2116     <- WORSE for four epochs
    up@64    e7:0.1563 e8:0.1499 e9:0.1475 e10:0.1452 e11:0.1465     <- flat, 0.065 better

That is the codec-erosion signature the roundtrip anchor only SLOWS (measured elsewhere at 0.40 dB per 9 epochs
at weight 10). The plausible causal chain for the objective gap: a decoder that is not degrading feeds a better
autoregressive gradient back into the dynamics via the decode loss, which `design/flow.md` notes is the only
autoregressive gradient in the model and reaches the dynamics solely through the decoder's Jacobian.

**CAVEATS, on the record before the number gets tempting:**
- **0.2429 at 96px is NOT better than 0.2783 at 128px.** LPIPS is resolution-dependent and coarser frames are
  easier. Nothing at 96px can challenge the record; only a 128px `up` run can.
- Param-matched but NOT cost-matched: up@64 has no down path, so it is cheaper per step and ran ~1 epoch ahead
  throughout. Epoch-matched comparison handles this; an equal-wall-clock comparison would flatter it.
- Both arms ran `p_tf_dynamics: 1.0`, which `mm_flow.yaml:125` calls "the DEFECT". Equally handicapped, so the
  comparison holds, but neither arm is at its best.

### 20.4 THE METHODOLOGICAL FAILURE — three probes that could not answer the question

Before the A/B, three frozen-encoder probes were run (`_oneoff_decoder_ab.py`): load the record holder, FREEZE
its encoder, train fresh decode heads. All three said up LOST -- floor 0.3649 vs 0.3156, @+128 0.5490 vs
0.5005, and the ordering held to 14k steps so it was not a convergence artifact. Conclusion drawn at the time:
"the readout-rank hypothesis is not supported; trunk capacity dominates."

**The user identified the flaw: that encoder was CO-TRAINED with the U-Net being replaced.** Measured on the
checkpoint (`_oneoff_latent_rank.py`):

    effective rank        participation ratio 29.5 | 90% of variance in 90 comps | 99% in 509 comps (of 4,096)
    readout subspace      REAL 53.23% of variance captured | RANDOM same-dim 15.75% | concentration 3.38x

99% of the latent's variance sits in **509 dims, INSIDE the 640 the U-Net already reads**, and the encoder put
3.4x more variance in that specific subspace than chance. So there was nothing outside the readout for extra
bandwidth to find: **the probes had no statistical power against the hypothesis they were built to test.**

The adversarial audit upheld this with one correction worth keeping: the encoder is NOT gradient-free in the
null directions -- `dynamics_detach_encoder: false` means the dynamics context path and the AR recon chain are
both full-rank at the encoder. What is absent is only PIXEL gradient. And it noted that the information ceiling
alone predicts up ~= unet, not up LOSING -- so the frozen result did contain a real signal about trunk size,
data starvation (570 frames), and the U-Net getting the padding-halo positional prior for free while `grid_q`
must learn it.

**Lesson:** a frozen-encoder A/B cannot evaluate a decoder whose whole claim is that the encoder would use it
differently. Only co-training tests that. Three probes and ~2 GPU-hours went to a rigged question.

### 20.5 Also corrected this session

- **"DF at 0.1 is stable AND the best result measured" — WRONG, repeated for hours.** Record line 97: "**DF at
  0.1 does not help.** Trails on one-step at every epoch, LPIPS a wash." `mm_flow.yaml:56-59`: keep DF
  scale=0.0 off, "the anti-collapse lever is in-rollout, not DF". `df_recon1` only ever reached e2 at 0.4499.
  The source of the error was a section-18 table where DF 0.1 topped ONE latent column (0.419/0.450), promoted
  to a global claim and then reused as a premise.
- **"Encoder capacity is measured-null three times over" — WRONG.** Only `num_tokens` 8->64 is a genuine
  encoder-capacity null. `ae_depth` never reaches the conv encoder. **`encode_base` has NEVER been swept** --
  record line 628 says it existed only as a dataclass default until it was exposed, then never tested.
- **`latent_loss_weight` is a PIXEL mse** (`multimodal.py:406`), not a latent loss, despite the name.
- **`latent_norm=layernorm` provides essentially NO anti-collapse**: per-token non-affine LN forbids only a
  within-token-constant vector; a time-constant or constant-direction bag passes untouched.
- **`model.d` is NOT a dynamics knob.** It is the token width, so it changes the latent size, the decoder's
  conditioning width and the encoder output. The isolated dynamics knobs are `flow_hidden` (currently defaults
  to d=128; 512 gives a **13.7x** flow head, 484,928 -> 6,651,584, touching nothing else), `flow_arch_depth`,
  `depth`, and `window`. `heads` is capped at 8: head_dim 128/16 = 8 violates the compiled rollout's
  >=16-and-power-of-2 constraint.

### 20.6 Queue

1. **Phase-shifted subsampling** — at `subsample=5` frames 1-4, 6-9, 11-14 are DISCARDED ENTIRELY and windows
   slide only over phase 0. Emitting all 5 phases gives **7.3x the windows at exactly 4 Hz** (35,085 ->
   255,774 measured at stride 1), same per-step motion, same real-time horizon, fully comparable to history --
   and free, because more data means fewer epochs for equal gradient steps. ~15 lines in `_subsample_episodes`.
   Motivation: `bott_recon1` ended at train 0.0398 vs val 0.5015, a 12x gap on 3.6 h of video.
2. **`flow_hidden=512`** on top of (1) — the only isolated dynamics-capacity knob, and it should be near-free
   in wall-clock because the throughput audit measured the step at ~24 TFLOP/s = **2.5% of H100 bf16 peak**,
   i.e. serialization-bound. Width is parallel work; depth is serial. WIDEN, DO NOT DEEPEN.
3. **128px with `decode_arch: up`** — the only run that can be scored against 0.2783.
4. `p_tf_dynamics=0.9` — cheap, fold into any of the above.

## 21. THE `p_tf_dynamics` DOSE CURVE IS A CLEAN NEGATIVE — and `tok64` was stopped on COST, not evidence (08-27/29)

Two things closed on 08-29, both by `kill -TERM` inside `quickdraw-app-1` (the trainers run as **root in the
container**; a host-side `kill` from `ubuntu` returns EPERM and, if stderr is suppressed, looks like it worked).

### 21.1 The dose curve: every substitution dose LOST, and gradient health degrades monotonically

`p_tf_dynamics` = P(the dynamics loss conditions on the TRUTH). 1.0 = always clean, which `mm_flow.yaml` calls
"the DEFECT"; lower = more of the model's own rolled-out latents in the conditioning. The theory said the
defect should hurt. Five doses on the identical `dyn512` base (96px, `ae_bottleneck=6`, `decode_arch=up@64`,
`flow_hidden=512`, `recon_frac=1.0`):

| `p_tf_dynamics` | evals | best OL LPIPS@+128 | best codec floor | max `grad/norm` |
|---|---|---|---|---|
| **1.00** (`dyn512`, the "defect") | 26 | **0.2392** | 0.1554 | **2.1** |
| 0.9375 (`q09375_dyn512`) | 16 | 0.2864 | 0.1772 | 1.47e4 |
| 0.875 (`q0875_dyn512`) | 24 | 0.2632 | **0.1442** | 3.33e13 |
| 0.75 (`q075_dyn512`) | 7 | 0.4060 | 0.2134 | **inf** |
| 0.50 (`q050_dyn512`) | 2 | 0.6206 | 0.3908 | **inf** |
| 0.0 (section 18) | — | blow-up | — | — |

Read the last column, not the third. **The objective column is noisy and non-monotone (0.9375 scored worse
than 0.875); the gradient column is monotone across five doses and spans thirteen orders of magnitude.** That
is the real finding: substitution re-bases the dynamics target to `z[t+1] - feed`, whose magnitude grows with
the model's own error, and the resulting positive feedback shows up in the gradient norm long before it shows
up in LPIPS. `q050` reached `inf` by eval 1; `q075` by eval 5.

Two things this does NOT say:
- **It is not evidence that clean conditioning is CORRECT.** `dyn512` still ends in the same late collapse
  (tail 0.364 / **0.239** / 0.606 / 0.253 — the record 0.2392 sits between two blow-ups), and `q0875` collapsed
  the same way (0.283 / 0.306 / 0.623 / 0.496). Both fail; the defect just fails later and lower.
- **It is not a floor result.** `q0875`'s floor of 0.1442 is the second best ever recorded on this dataset
  (behind `dec_up64`'s 0.1432). Substitution left the CODEC alone and damaged the DYNAMICS, exactly where the
  mechanism predicts.

**Consequence:** `mm_flow.yaml:125`'s "Try 0.9" advice is now measured wrong and should be corrected. Keep
`p_tf_dynamics=1.0`. The late collapse is real but `p_tf_dynamics` is not its lever — see section 19, which
already showed `detach_every` is not either.

### 21.2 `tok64` — killed at 2 evals, and it was AHEAD

`num_tokens` 32 -> 64 on the `up@64` base. Section 17's null (`num_tokens` 8->64 moved nothing) was explained
in section 20 by the U-Net's rank-640 readout; with the query-grid readout the cap is now `grid x d` = 4,608,
so 64 tokens is 8,192 floats = **178% of cap** — over, though for a query grid "over" buys selection rather
than bandwidth. A literature review predicted a null (TiTok renders 256px from 32 tokens).

It was killed after 2 evals. **At matched eval index 1 it had the BEST codec floor of any run on this dataset
and the second-best @+128:**

| run @ eval idx 1 | @+128 | floor |
|---|---|---|
| **`tok64`** | 0.4331 | **0.2577** |
| `dec_unet32` | **0.4251** | 0.2909 |
| `q0875_dyn512` | 0.4603 | 0.2803 |
| `dyn512` | 0.4658 | 0.2998 |
| `dec_up64` | 0.4772 | 0.3027 |

Two evals is noise-dominated and this record's own rule is best-so-far over matched series, so it is not
evidence that 64 tokens WINS. But it is the opposite of the evidence needed to call it null, and the run was
**not** stopped for that reason. It was stopped for **cost**: 2,339 batches/epoch at ~72 min against
`dyn512`'s 1,254, i.e. ~48 h for 40 epochs on one of two GPUs, blocking the entire visual-loss queue below.

**If `num_tokens=64` is retried, it is an OPEN question, not a closed one.** Log it that way.

### 21.3 What the GPUs were freed for

The reconstruction loss, which has never been varied on this dataset. Both pixel-space terms are plain L2:

| site | loss | weight |
|---|---|---|
| AR decode (`flow.py`, `no_noise` branch) | `F.mse_loss` | 1.0 |
| roundtrip codec anchor (`multimodal.py:407`) | `F.mse_loss` | **10** |

L2's minimiser under uncertainty is the conditional MEAN, i.e. blur, and the weight-10 term is the dominant
pixel pressure in the model. Every published latent codec (VQGAN, SD-VAE, IRIS, MAGVIT) uses L1 + LPIPS
instead. Planned: one shared `VisualLoss` (`w_l2*L2 + w_l1*L1 + w_lpips*LPIPS`) instantiated ONCE per image
modality and used at BOTH sites, with the x10 applied outside it.

**TRAP, caught before launch.** `_install_perceptual` calls `evaluation.openloop._lpips_net`, which builds
**SqueezeNet** — the exact network `image_curves` uses to compute the reported metric. Training that arm as
written would have optimised the eval metric's own features and produced a number not comparable to any of
the 25 historical runs. **Train with VGG, keep SqueezeNet for eval.**

## 22. THE RECONSTRUCTION LOSS WAS NEVER VARIED — and changing it beat the record at EPOCH 1 (08-29/30)

STATUS: round 1 is RUNNING (`vl_keep10`, `vl_iris`). Everything below through 22.5 is settled; 22.6 is the
live result at eval 1 of 40 and will be extended.

### 22.1 The thing nobody had looked at

Two call sites train the image decoder, and both had been hardcoded `F.mse_loss` since the beginning:

| site | where | weight |
|---|---|---|
| (a) AR decode loss | `flow.TransportHead.loss`, `no_noise` branch | 1.0 |
| (b) roundtrip codec anchor | `multimodal.roundtrip_losses` | **10** |

L2's minimiser under uncertainty is the conditional MEAN, i.e. blur. design/collapse.md had already written the
consequence down in this project's own words -- *"MSE loves blur. Dropping detail moves the prediction toward a
smooth mean; MSE barely penalizes that... LPIPS was 0.18 the whole time -- the blur was there from the start;
MSE never saw it."* So for 25 runs we optimised a loss structurally blind to sharpness and then ranked every
run on LPIPS. Nobody had tried changing it.

### 22.2 What the field actually uses (code-verified, not paraphrased)

| system | pixel | perceptual | GAN |
|---|---|---|---|
| VQGAN | **L1 @ 1.0** | LPIPS-**VGG16** @ **1.0** | yes, delayed |
| LDM / SD-VAE | L1 @ 1.0 | LPIPS-VGG16 @ 1.0 | from step 50k |
| **IRIS** | **L1 @ 1.0** | LPIPS-VGG16 @ **1.0** | **none** |
| ViTok stage 1 | L2 @ 1.0 | LPIPS @ 1.0 (swept 0 / 0.5 / 1.0) | none |
| SoftVQ-VAE | L2 @ 1.0 | LPIPS-VGG @ 1.0 | 0.2 |
| MAGVIT-v2 | L2 @ 5.0 | **0.1 -- and NOT LPIPS** (ResNet50 logits) | 0.1, from step 0 |
| TiTok stage 2 | L2 @ 1.0 | **0.1 -- ConvNeXt-S, not LPIPS** | 0.01, from 20k |

Two families, and the 0.1 family is a TRAP for us: both members have a GAN carrying sharpness AND use a
non-LPIPS perceptual net. Every GAN-free system uses 1:1. IRIS is the closest published relative -- bespoke
encoder, 64px frames, no GAN, L1 + LPIPS-VGG16 at exactly 1:1. ViTok measured MSE -> +LPIPS taking rFID
2.1 -> 0.95. All of them enable the perceptual term from STEP 0; warmup is reserved for the GAN.

### 22.3 THE TRAP THAT WAS CAUGHT BEFORE LAUNCH: train on vgg, evaluate on squeeze

The first version of this work called `evaluation.openloop._lpips_net`, which builds **SqueezeNet** -- the
backbone of the REPORTED metric and of all 25 historical numbers. Training on it optimises the metric's own
features and yields a number comparable to nothing. `_lpips_net` now takes `net_type`, cached per
(device, net_type), DEFAULT UNCHANGED at squeeze; training defaults to **vgg**, which is also what
VQGAN/LDM/IRIS/SoftVQ all hardcode. Measured on the converged control, vgg/squeeze = **1.713** -- so the two
are not interchangeable even in scale.

### 22.4 ONE `VisualLoss`, BOTH SITES -- and why sharing it is what makes it safe

`models/visual_loss.py`: `w_l2*L2 + w_l1*L1 + w_lpips*LPIPS(net)`. `ImageModality` owns ONE instance,
registered once, handed to site (a) through the new `TransportHead.loss(recon_loss=)` and to site (b) directly.
Site weights apply OUTSIDE. Defaults (`w_l2=1`, rest 0) are bit-identical to `F.mse_loss`.

The sharing is not tidiness. If site (b) kept extra PURE pixel loss outside the module, its 10x weight would
dominate a small perceptual term at site (a) and re-blur the decoder. Because the site weight scales the whole
mix, the pixel:perceptual RATIO is identical at both sites and only the magnitude differs.

`recon_loss` REPLACES the head's internal `F.mse_loss` rather than adding to it -- the earlier `aux` hook
summed on top, which would have double-counted L2. Ignored for `param="v"` (no clean prediction; the dynamics
FlowField shares that method).

TWO BUGS WORTH REMEMBERING, both invisible to a passing test suite:
  * **Rank.** Site (a) flattens to (M,H,W,C); site (b) scores `to_obs()` output and keeps its (B,F) lead, so it
    arrives (B,F,H,W,C). `mse_loss`/`l1_loss` reduce over everything and are rank-agnostic -- which is exactly
    why a pure-MSE anchor never cared, and why this was invisible for the entire life of the project until
    LPIPS needed `permute(0,3,1,2)`. It mattered twice: the permute, and `_subsample` drawing over FRAMES
    rather than whole trajectories. The smoke passed 16/16 on broken code because it only drove the 4-D path.
  * **`_perceptual` as a method returning None.** A bound method is always truthy, so `aux is not None` passed
    and the head added None to the loss. It must be a None ATTRIBUTE. (Superseded by `recon_loss`, kept here
    because the same shape of mistake will recur.)

### 22.5 MEASURED, not estimated: the term magnitudes, and the memory cost

`_oneoff_visual_terms.py` on `dyn512`'s best.ckpt, 256 held-out val frames through the real encode->decode:

| term | converged value |
|---|---|
| L2 | 0.01305 (18.84 dB) |
| L1 | 0.05732 |
| LPIPS-squeeze | 0.16124 |
| LPIPS-**vgg** | **0.27613** |

So the literature mix L1+LPIPS is **25.5x** the anchor's current magnitude. Note the ratio is NOT constant:
~3.7x at random init and ~25.5x at convergence, because L2 collapses ~29x over training while L1 falls ~9x and
LPIPS only ~3x. No single scalar preserves the anchor's magnitude throughout.

MEMORY -- and the first reading of it was WRONG, off BROKEN code. The pre-fix probes fit 5.636 GB/sample and
chose **batch 16**, and a `visual_frames=32` probe returned the identical figure, which looked like proof that
the frame subsample controls only a fixed cost. Both measurements were taken with the 22.4 rank bug live. With
5-D input at the anchor, `_subsample` compared `frames=128` against `pred.shape[0]` = the number of
TRAJECTORIES (26), so `0 < 128 < 26` was false and **the subsample silently did not apply at the anchor at
all** -- VGG ran on every frame there. Fixing the rank fixed the memory:

    02:32 (broken)  probe b=16: 92.9GB   b=8: 47.8GB   fit 5.636 GB/sample  -> batch 16
    05:41 (fixed)   probe b=16: 58.6GB   b=8: 32.0GB   fit 3.324 GB/sample  -> batch 26

**Both live arms run at batch 26** (verified in `config.resolved.yaml`, not inferred from a log line), and
1350 x 26 = **35,100 windows/epoch against the control's 30,716**. So the arms are NOT handicapped: batch is
within 7% of the control's 28 and an epoch is 14% MORE data, not 30% less. An earlier version of this section
claimed the opposite; the error was reading `fit chose data.batch=16` out of the CRASHED launches and never
re-reading it after the successful relaunch.

Whether `visual_frames` controls memory on the FIXED path is now UNMEASURED -- do not quote the 32-vs-128
result, it was taken on the broken path.

The one asymmetry that does remain: the arms run 40 epochs to the control's 26, so best-so-far draws from more
samples of a jittery distribution. Compare best-so-far TRUNCATED to 26 evals.

### 22.6 ROUND 1: the anchor weight, and I got the invariant wrong

Both arms: `visual_l1=1.0 visual_l2=0.0 visual_lpips=1.0` on vgg. ONE variable, the site-(b) weight.

| eval | `vl_keep10` (w=10) @+128 / floor / dB | `vl_iris` (w=0.4) @+128 / floor / dB |
|---|---|---|
| 0 | 0.3217 / 0.2412 / 15.02 | 0.4302 / 0.2517 / 14.84 |
| 1 | **0.2035** / **0.1916** / 16.27 | 0.3598 / 0.4264 / **9.41** |

**`vl_keep10` beat the all-time record (0.2392, which took `dyn512` 23 evals) at EVAL 1**, from behind on both
data-per-epoch and batch size. Its floor at eval 1 (0.1916) is better than the control's at eval 1 (0.2998) and
closing on the control's all-time best floor (0.1554).

**`vl_iris` collapsed in one epoch.** Floor PSNR 14.84 -> 9.41 (5.4 dB), floor LPIPS 0.2517 -> 0.4264, dynamics
latent loss 0.0989 -> **94.27** (~950x), `grad/norm/flow` -> nan (the guard caught it, `nonfinite_skipped 1`),
motion 1.398 (diverging), raw roundtrip MSE 0.0267 -> 0.0824.

WHAT THAT SETTLES. `up64.yaml` says "latent_loss_weight: 10 -- DO NOT lower: at 1.0 the codec eroded 3.55 dB in
4 epochs." It was overridden to 0.4 on the argument that the anchor's MAGNITUDE was preserved. That argument is
WRONG, and the warning was conservative: 5.4 dB in ONE epoch at 0.4 versus 3.55 dB in four at 1.0.

**`latent_loss_weight` is not a magnitude knob. It is a RATIO knob: "stay invertible" against "be
predictable".** The anchor `Dec(Enc(x))->x` is the ONLY term forcing the latent to remain a faithful encoding.
Every other loss -- the dynamics flow loss, the AR decode loss -- can be reduced by making the latent EASIER TO
PREDICT, and the easiest-to-predict latent is degenerate. So the anchor is the sole counterweight to
representational collapse, and what the encoder responds to is its pressure RELATIVE to everything pulling the
other way. Preserving its loss VALUE cut that by 25x. The failure is a feedback loop, not a threshold: weak
anchor -> encoder drifts -> the dynamics' target distribution goes non-stationary -> flow loss climbs ->
larger gradients -> more drift -> nan. Same topology as section 21's substitution loop, different route.

THE KNOB IS NOW BRACKETED FROM BELOW TWICE AND NEVER FROM ABOVE:
    w=1.0 (pure MSE)   3.55 dB erosion / 4 epochs
    w=0.4 (L1+LPIPS)   5.40 dB erosion / 1 epoch
    w=10  (L1+LPIPS)   healthy, floor improving, record at eval 1
10 was INHERITED, never optimised. "10 is enough" and "10 is optimal" are different claims and only the first
is shown.

### 22.7 THE CONFOUND IN THE WINNING ARM -- stated plainly because it decides the next run

The launch script called `vl_keep10` the arm that "changes ONLY the loss shape." **That is wrong.** Holding the
WEIGHT at 10 while the loss VALUE grew ~25x means the anchor's actual contribution grew ~25x too. So
`vl_keep10` changes the loss shape AND applies ~25x more codec pressure, and either could be producing the
record. One run separates them: **pure L2 at `latent_loss_weight`~250** -- same pressure, no perceptual term.

Also note it is NOT a free lunch, and the mechanism is visible in the raw-MSE series (`codec/roundtrip_*_mse`,
logged at weight 0.0 precisely so any mix stays comparable to the 25 historical runs):

    dyn512     0.01720  0.01437  0.01365  0.01344  0.01315     <- better on MSE, as it must be: it optimises MSE
    vl_keep10  0.02649  0.02269

We traded pixel error for perceptual quality. That is the trade working as designed, not an artifact.

### 22.8 `vl_iris` DOES NOT REFUTE IRIS -- the arm was misnamed

IRIS's recipe is the MIX (L1 1.0 + LPIPS-VGG16 1.0, no GAN). BOTH arms run exactly that, and the one that kept
the incumbent anchor weight is breaking records. What failed was the anchor-weight override, which is not part
of IRIS's recipe at all -- **IRIS has ONE reconstruction site and therefore no such knob to set**. The arm
should have been called `vl_w04`.

This is the cleanest confirmation of the literature review's warning that our two-site structure is off the
published map: every cited system applies its reconstruction loss at exactly one site, and token world models
(IRIS, Genie, MAGVIT lineage) avoid the problem structurally by training dynamics as cross-entropy in token
space, never decoding to pixels. The literature-faithful part transferred immediately and strongly; the part no
paper could advise on is precisely where it broke, on a judgment call rather than on anything cited.

### 22.10 GHOSTING — the objective is now 83% the one term that cannot see it (user, eyes-on, 08-30)

The user reported a **ghosting artifact in the rendered frames that no previous run had at all**. It is real,
it is measurable, and it is in the CODEC (`eval_ae_floor`, i.e. encode->decode of a REAL frame), not in the
dynamics. At matched-or-better floor LPIPS:

| run | ev | floor LPIPS | floor PSNR | floor SSIM |
|---|---|---|---|---|
| **`vl_keep10`** | 2 | **0.1347** (best EVER; prior best 0.1432) | **17.42** | **0.581** |
| `dec_up64` | 6 | 0.1592 | 19.51 | 0.671 |
| `dyn512` | 6 | 0.1743 | 19.64 | 0.669 |

`vl_keep10` has the best perceptual floor ever recorded while sitting **2.1 dB and 0.09 SSIM BELOW** runs with
WORSE perceptual scores. Good LPIPS at bad PSNR/SSIM is the signature of "perceptually plausible, spatially
wrong", and ghosting is what that looks like on screen.

MECHANISM. LPIPS scores VGG features after several POOLING stages, so it is comparatively insensitive to small
spatial displacement. Where the decoder is unsure of an edge's position, rendering TWO FAINT COPIES costs LPIPS
almost nothing -- both are plausible in feature space -- while a SQUARED pixel term would punish both copies
hard. We set `visual_l2 = 0.0`, so there is nothing squared left in the objective at all. And the balance is
worse than it sounds: at convergence L1 = 0.057 and LPIPS-vgg = 0.276, so the objective is **83% LPIPS and 17%
L1** -- 83% of it is the one term blind to ghosting, and the 17% that can see it penalises only LINEARLY.

Every historical run was 100% squared error, which is why none of them ghosted and why all of them were BLURRY
instead. We did not remove an artifact; we traded one for another.

WHAT THIS MEANS FOR TRUSTING THE NUMBER. The literature review flagged exactly this in advance: E-LPIPS
(arXiv 1906.03973) and R-LPIPS (2307.15157) show that OPTIMISING against an LPIPS network finds
metric-specific minima that contradict human judgment, and different backbones (our vgg-train / squeeze-eval
split) attenuate that without eliminating it -- both are ImageNet CNN feature stacks. **The user's eyes are the
check on the metric, and here they disagreed with it.** Treat `vl_keep10`'s 0.2035 as real but INFLATED until
it is reproduced with a squared term in the loss. This is also the concrete reason the vgg/squeeze split was
worth insisting on: without it the contamination would be total rather than partial.

THE FIX, and the weight matters more than the term. At `w_l2=1.0` the squared term would be 0.013 against
LPIPS's 0.276 -- 4% of the loss, cosmetic. The precedent for SCALING it is HiFiC (arXiv 2006.09965), which
computes MSE on [0,255] with k_M = 0.075*2^-5, i.e. ~152x MSE on [0,1], explicitly to bring the pixel and
perceptual terms to the same order of magnitude. For us:

    visual_l2 = 20   ->  0.013 * 20 = 0.26,  comparable to LPIPS 0.276
    visual_l1 = 1.0
    visual_lpips = 1.0

`visual_lpips=0.5` (ViTok's own swept value) is the weaker alternative -- it moves LPIPS's share only from 83%
to 71%, probably not enough.

QUEUED as `vl_l2back` (l1=1.0, l2=20.0, lpips=1.0, latent_loss_weight=10, everything else identical to
`vl_keep10`). The user elected to let the two running arms finish first. NOTE that both live arms share this
loss and will BOTH ghost, so the anchor-weight sweep in 22.9 is being read on a partly metric-gamed objective.

### 22.9 Queue

0. **`visual_l1=3.0` with `visual_lpips=1.0`** (PINNED 2026-09-01). Currently 1.0/1.0, which is NOT balanced:
   measured on `vl_keep10`'s own checkpoint the terms are L1 0.0587 and LPIPS-vgg 0.1835, so the objective is
   **24% pixel / 76% perceptual**. At `l1=3.0` it is 49/51 -- state it as "weighted 3:1 so the pixel and
   perceptual terms contribute equally, measured on the trained codec". DO NOT write 3.13: the third digit is
   one measurement on one checkpoint and the ratio drifts more than 4% during training.
   MOTIVATION, measured (`_oneoff_colour_cast.py`, 128 val frames, real encode->decode):

   | | channel bias R/G/B | tint (spread) | saturation | MAE |
   |---|---|---|---|---|
   | `dyn512` (pure L2) | -0.0018 / -0.0017 / -0.0007 | 0.0011 | 0.985 matched | 0.0567 |
   | `vl_keep10` (L1+LPIPS) | **+0.0024 / -0.0026 / +0.0018** | **0.0050** (4.5x) | **1.100** | 0.0584 |

   Same MAE, differently SHAPED error: a magenta cast (R,B up / G down) and **10% oversaturation**. LPIPS
   scores VGG features, which barely move under a global colour shift, and oversaturating makes edges "pop"
   in feature space at almost no metric cost. The tint itself is ~1.3/255 and probably invisible; the
   SATURATION is the part the user could see. NOTE this also kills the "add L2 back" idea: a 0.0025 bias
   against a 0.058 per-pixel error is 4% under L1 and negligible under L2, so no squared term fixes a small
   global cast at any sane weight. Raising L1 (rather than lowering LPIPS) is the right direction because it
   raises pixel pressure WITHOUT lowering total pressure -- which is how `vl_lp025` wrecked its floor.
   `l1=5.0` (62/38, pixel-dominant) is the follow-up if parity is not enough.

1. **`latent_loss_weight=25` on the `vl_keep10` base** -- the knob has never been swept upward, and 22.6 makes
   it the highest-EV single run available. Watch for the OPPOSITE failure: an over-anchored encoder pinned to
   being a good autoencoder at the expense of being predictable, which would show as a good floor with rising
   `grad/norm/flow` and falling `latent_cos`. Note this does NOT resolve 22.7's confound.
2. **pure L2 at `latent_loss_weight`~250** -- the confound-breaker. Is the win LPIPS, or just pressure?
3. **`decode_inject`** (models/decoders.py, feature 2) on whichever loss wins -- 115,584 params, zero-init so
   the model is a strict superset at step 0.
4. **`decode_xattn_max_res=24`** (feature 3) -- only if 3 moves the needle; same hypothesis, 2.3x the cost.
5. `df_scale` toward 0.7 -- the dynamics axis, now a one-number change (section 21 established our DF already
   has per-position random levels AND a learned Fourier level embedding; only the magnitude was small).

WATCH ITEM on `vl_keep10`: `grad/norm/flow` rose 0.44 -> 1.86 across the first two evals. Every historical
collapse spiked there first.

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
