# Capacity: where the world model's headroom actually is

Findings on the robocasa-scene4-4h dataset (128px, frozen TAESD, EXACT adapter), 2026-08-09.
Everything here is measured on this repo; each claim says how.

---

## 1. The floor numbers below are UNTRAINED. Read them as homework, not as ceilings.

`eval_ae_floor` fires once at `on_fit_start`. At that moment the adapter's refine branch is
zero-initialised (`vision._zero_init_last`), so it is a no-op: the gate measures the
**parameter-free reshape + LayerNorm** and nothing else.

So a floor of 20.41 dB does **not** mean "this configuration reconstructs at 20.41 dB". It means
"before any training, the LayerNorm has removed 3.51 dB that the adapter residual now has to learn
back". It is the size of the job, not the outcome of the job.

What supervises that job is `loss/roundtrip/<mod>`, which routes through `encode_state -> to_obs`
(the real path, LayerNorm included). Prior to 2026-08-09 that term measured the modality's own
norm-free `up(down(g))`, read `0.0` at every epoch of every run, and produced no gradient at all —
so **no run to date has actually trained the adapter to invert LayerNorm.** The only evidence it
moves is a 3-epoch smoke: 0.00907 -> 0.00861.

**Consequence for the table in §3: it ranks the starting deficits, NOT the achievable ceilings.**
An arrangement with more homework may still finish higher. Nothing here licenses "more tokens
reconstruct worse" — that is untested.

---

## 2. Single-step quality and long-horizon quality are decoupled

From the `taesd_exact` run (`logs/robocasa-tok/..._taesd_exact/logs/metrics.jsonl`):

| epoch | one-step val PSNR | open-loop long-horizon PSNR |
|------:|------------------:|----------------------------:|
| 0 | 11.76 | — |
| 1 | 17.80 | 12.21 |
| 2 | 18.12 | 12.28 |
| 3 | 18.36 | **12.92** |
| 4 | **18.39** | 12.30 |
| 5 | 11.63 | 10.55 (collapse) |

Replacing the bespoke autoencoder with frozen TAESD bought **+6.6 dB single-step and +0.1 dB at
horizon**. That was the largest per-step improvement available — it broke a 14.3–14.8 dB wall that
four prior configurations were pinned at — and the horizon metric did not move.

Long-horizon sits at ~12.3 dB no matter how good the one-step map gets. That is the signature the
teacher-forcing notes describe for the unconditional-mean attractor (`conf/model/mm_flow.yaml`,
p_tf block): the rollout drifts to a blurry average frame, and once there, per-step fidelity is
irrelevant.

**Therefore: any lever that only improves single-step accuracy is a poor bet for long-horizon.**
That includes `flow_hidden` and `depth`.

Related, unquantified: training rolls `data.F=64`; `eval.horizon` is 2048 capped by episode
length. Long-horizon is measured up to ~32x outside the trained rollout length.

---

## 3. At a fixed float budget, num_tokens and d see-saw — and both directions cost

The adapter requires `num_tokens * d == L` (the AE latent size) for EXACT. At 128px, TAESD's latent
is `4*16*16 = 1024` floats, so the two knobs are locked together.

Measured (fit-start floor, frozen TAESD, robocasa 128px; raw AE reference 23.92 dB):

| bag | mode | floor | LN homework | spatial granularity/token | backbone params (~d²) |
|---|---|---|---|---|---|
| 4x256 | EXACT | 20.79 | 3.13 dB | 4 rows | ~4.2M |
| **8x128** | EXACT | **20.41** | 3.51 dB | **2 rows (default)** | 1.06M |
| 16x64 | EXACT | 20.01 | 3.91 dB | 1 row | ~265K |
| 32x32 | EXACT | 19.72 | 4.20 dB | ½ row | ~66K |
| 64x16 | EXACT | 18.68 | 5.24 dB | ¼ row | ~17K |
| 32x128 | PADDED | 16.03 | 7.89 dB | ½ row, 96/128 idle | 1.06M |

Two independent effects push the same way as tokens get finer:

- **LayerNorm cost grows.** `_ln` is per token and non-affine, so it discards each token's mean and
  std — 2 scalars per token. 8 tokens drops 16 of 1024 floats; 64 tokens drops 128.
- **Backbone shrinks.** Transformer parameters scale with `d²`, so 8x128 -> 32x32 is a ~16x smaller
  backbone. Attention cost meanwhile scales with `num_tokens²`.

What finer tokens buy is **spatial addressing**. See §4.

**There is no clean num_tokens A/B at a fixed budget** — raising tokens always drags backbone
capacity down with it. Two ways around it, neither perfect:

- Grow the latent (256px -> 4096 floats -> `32x128` is EXACT with `d` unchanged). Confounded by
  input resolution and by a higher raw ceiling (~27.8 dB vs 23.9 dB measured on this dataset).
- Score **gap-to-floor** (PSNR minus the run's own measured `eval_ae_floor`) instead of absolute
  PSNR, which normalises the ceiling change out.

### Storage is definitively not the constraint

At 8x128 the bag holds all 1024 latent floats exactly, and the raw-AE reference (23.92 dB) is
reached by the adapter alone. Whatever limits this model, it is not the number of floats carried.

---

## 4. A token is a raster stripe, and that is an arbitrary choice

`GridToTokens.base` flattens the latent spatial-major with channels innermost
(`grid.permute(0,2,3,1).reshape(B,-1)`) and cuts it into `num_tokens` contiguous chunks. At
8x128 each chunk is 128 floats = 32 grid cells = **2 full rows of the 16x16 latent**. Tokens are
horizontal bands.

Two consequences worth testing, neither measured yet:

- **Anisotropy.** Vertical motion crosses token boundaries immediately; horizontal motion stays
  inside one token. The dynamics sees the two directions differently for no principled reason.
- **LayerNorm removes a LOCAL statistic.** Each token's mean is the mean brightness of its own 2-row
  band, so LN destroys 8 distinct local statistics. Under a layout where each token samples the
  whole grid (strided/polyphase interleave), every token's mean is approximately the *global* mean —
  the discarded scalars are then largely redundant across tokens, so less unique information is
  lost. This predicts a lower LN homework figure at the same `num_tokens`, and is cheap to test:
  the floor gate needs no training.

Alternative layouts, in increasing distance from the current one: 2D tiles (square-ish patches,
fixes anisotropy), channel-major (token = one latent channel), strided interleave (every token
global). All are pure index rearrangements — no parameters, no capacity change, EXACT preserved.

---

## 5. Padding is active damage, not idle capacity

`adapter_mode` reports `identity_at_init: True` for PADDED, and for the adapter in isolation that is
correct: zero-pad then strip is a bijection.

But the model runs `encode -> LayerNorm -> decode`, and **LayerNorm is per token, not per element**.
For a token holding 32 real floats and 96 zeros, the mean and std are computed across all 128
entries, so the zeros enter the statistics that the real values are normalised by. The real floats
come out scaled by a factor derived from a mostly-zero distribution. Stripping the pad afterwards
does not undo it.

Measured: 32x128 (75% pad) floors at **16.03 dB** against **20.41 dB** for EXACT 8x128 — **-4.4 dB**,
a bigger loss than any `num_tokens` choice in §3.

This is the same failure class as the `eval_ae_floor` bug fixed on 2026-08-09: a property that holds
for a sub-path the model never runs in isolation, reported as if it held end-to-end. Fixed by
scoping the claim in `vision.adapter_mode`'s docstring and making the `[adapter] WASTING` warning in
`logging/callback.py` state the measured end-to-end cost rather than only the idle-float percentage.

**Rule, now recorded in `conf/model/mm_flow.yaml`: keep `num_tokens * d == L`. Never pad.**

---

## Open

- Root cause of the epoch-5 collapse. No post-collapse weights existed at the time; `last.ckpt` is
  now written unconditionally every epoch, so the next one is inspectable.
- Whether the adapter residual actually recovers the LayerNorm homework once `loss/roundtrip` has a
  real gradient. First run to test it is in flight.
- Whether spatial addressing (§4) or dynamics capacity (§3) drives the long-horizon plateau.
- The `F=64` trained vs 2048 evaluated horizon gap (§2).
