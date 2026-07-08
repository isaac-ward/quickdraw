# Language-steered MPPI via a learned reward `R(latent, text)` — build plan

**Status: to build.** Steer the MPPI planner with a natural-language request ("red") by scoring candidate
rollouts **in latent space** with a learned reward `R(latent, text)`, **distilled** from `eval_interpret`'s
`(latent, VLM-label)` pairs. Decode-free, real-time, multi-modal-native — chosen over the text→target
variant because a learned reward needs no target points, no `k`, no centroids.

Naming: the world-model trainer is `train_world.py` (was `train.py`); the reward trainer is `train_reward.py`
(new), reusing `train_world.py`'s logging path (`make_writer` → local mirror + wandb, identical keys).

## Reward head

- Text: frozen `sentence-transformers/paraphrase-MiniLM-L3-v2` → `text_emb` (384-d), embedded **once per request**.
- Latent: the flattened token bag (`n_state·d`) — the exact object MPPI rolls.
- Two small trainable MLPs into a shared space (`d_e ≈ 128`): `f_z: latent→e`, `f_t: text_emb→e`.
- **Reward `R = cos(f_z(z), f_t(text)) ∈ [-1,1]`** — bounded, smooth (good for MPPI ranking), multi-modal by
  construction (any `z` aligned with the text direction scores high, so "red region A/B" both win).
- Only `f_z, f_t` train; MiniLM and the world model stay frozen.

## Data + train/val split

- Source: one `eval_interpret` run's **per_step** `(latent, label)` pairs — the *rollout* latents, i.e. exactly
  what MPPI queries (no train/deploy distribution gap). `projections/latents.npy` + `clip_index.npy` → `labels.json`.
- **Split by CLIP** (~85/15), not by point, so a val clip's 30 per-step latents never leak into train.
- Text inputs: per attribute value, a few templated **paraphrases** ("red", "the red band", "red surface",
  "reddish") pointing at that value's latents → generalizes over phrasing. MiniLM handles arbitrary length
  (pools to a fixed 384-d), so nothing on our side deals with sequence length.

## Training objective

Cross-entropy over the fixed vocabulary's text prototypes. Precompute `f_t(text_k)` for the K attribute values
(each `text_k` a mean over its paraphrases); for a latent `z_i` with label `c_i`, form cosine logits
`ℓ_k = cos(f_z(z_i), f_t(text_k)) / τ` and minimise cross-entropy toward `c_i`. This handles many latents
sharing a label cleanly (classification over the fixed text set), and — because `f_t` is a *learned* map from
MiniLM's 384-d space — it still generalises to novel phrasings ("crimson" → near "red"). Paraphrase
augmentation (a random phrasing per value per step) makes `f_t` robust to wording. No off-manifold negatives
in v1 (see Guardrails). `loss/total == loss/infonce ==` this cosine-CE.

## Guardrails against reward hacking — DEFERRED (observe first)

A learned reward can be exploited off-manifold, but the rollout is **dynamics-constrained** — MPPI can only
reach latents the world-model dynamics actually produce from the seed+actions, so much off-manifold space is
simply unreachable. **v1 ships no guardrail**; we first check whether MPPI actually hacks the reward. If it
does, add (preferred order): (1) **off-manifold negatives in training** — affine blends of real latents
(random `α`, incl. `α∉[0,1]`) + random Gaussian, with a margin loss pushing their reward down; fixes `R` at
the source, no MPPI retuning. (2) **stay-on-manifold penalty** `cost = −R + λ·offmanifold(z)` as a backstop.

## Sanity gate (before ever planning with it)

**Held-out `argmax` accuracy:** on val latents, score each against *every* attribute text and `argmax` →
predicted attribute; compare to the VLM label. If `R` recovers the label this way it learned the alignment;
if it's ~chance (1/#classes), `R` is broken — fix head/data, don't plan with it. This is `val/acc/argmax`.

## Logging (plain loop reusing `make_writer`; every scalar under BOTH `train/` and `val/`)

Scalars (each logged as `train/<key>` and `val/<key>`):
- `loss/` : `total` (= `infonce`, the cosine-CE)
- `acc/`  : `argmax` (headline gate — score each latent against every text, argmax == label?), `rank_mean`
  (mean rank of the correct text; per-class detail lives in the confusion figure, not scalar lines)
- `sim/`  : `pos_mean` (mean cos to the CORRECT text, →1 good), `neg_mean` (mean cos to WRONG texts, low good),
  `gap` (= pos_mean − neg_mean, the separation margin)

Figures (val):
- `confusion/color` — argmax-predicted vs true (reuses `viz.fig_confusion`); this IS the per-class view.
- `embedding/color_2d`, `embedding/color_3d` — `f_z(latents)` projected, colored by label (does the reward
  space itself separate?)

Bookkeeping (train-only): `optim/{lr,grad_norm}`, `time/{epoch_seconds,total_minutes}`,
`data/{n_train_clips,n_val_clips,n_points,counts/<bucket>}`, `model/params`.
Headline metrics: `val/acc/argmax` and `val/sim/gap`. A **`guide.md`** is written into the run's log folder
explaining each metric briefly (with the cos / cross-entropy equations).

## Dependency (setup blocker)

Needs a tiny text encoder — `sentence-transformers/paraphrase-MiniLM-L3-v2` (~17M) — via `sentence-transformers`
(or `transformers` + mean-pooling). NOT currently in the image; adding the package and pulling the model into
`/caches/hf` is the first setup step (requires the container to reach the Hub once).

## MPPI integration (per step, decode-free)

- Once per episode: `t_e = f_t(MiniLM(request))`.
- Per candidate per step: `r = cos(f_z(z_t), t_e)`; `cost = −Σ_t r (+ λ·offmanifold)`.
- Configurable:
  ```yaml
  reward:
    embed_dim: 128
    paraphrases: true
    distance: cosine            # the head trains on cosine; kept for parity with the text→target variant
    offmanifold_lambda: 0.0     # >0 turns on guardrail #2 in the cost
    temperature: 1.0            # optional cosine sharpening
  ```

## Blast radius / build order

New: `train_reward.py` (trains `f_z/f_t`, saves the head + `f_t` precompute), `language/reward.py` (loads the
head, resolves request→`t_e`, the cost fn). Modified, 3 surgical spots:
1. `controller/mppi.py` — make the cost **pluggable** (`cost_fn`); today it's hardwired to −distance-to-goal.
2. `imagine_shared` — expose the rolled **latent bag** to the scorer (today it only decodes proprio).
3. `eval_control.py` — accept `language.request=...` and select the reward cost.
World-model spine/training untouched.

Build order: (a) `train_reward.py` + logging + the held-out gate on an eval_interpret run; (b) wire the
pluggable cost + latent-bag exposure; (c) `eval_control language.request=...` demo; (d) optional guardrail #2
and the LDA-subspace distance refinement.
