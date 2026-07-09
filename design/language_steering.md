# Language-steered MPPI via a learned reward `R(latent, text)` — build plan

**Status: built.** Steer the MPPI planner with a natural-language request ("red") by scoring candidate
rollouts **in latent space** with a learned reward `R(latent, text)`, **distilled** from `eval_interpret`'s
`(latent, VLM-label)` pairs. Decode-free, real-time, multi-modal-native — chosen over the text→target
variant because a learned reward needs no target points, no `k`, no centroids. It is **not** a separate eval:
it is `eval_control` with the oracle turned off and a reward objective swapped in (see MPPI integration).

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

- Once per episode: `t_e = f_t(MiniLM(request))` (known-vocab requests reuse the precomputed prototype, so no
  text encoder is needed at plan time; open-vocab would need live MiniLM — deferred).
- Per candidate per step: `r = cos(f_z(z_t), t_e)`; the candidate's MPPI return is `Σ_t r` (higher = better,
  softmax-weighted like the goal-distance return). No separate cost sign to manage — it slots straight into the
  existing `_mppi_step` where a per-candidate return replaces the goal-distance `_score`.
- Config is just the shared `language:` block in `conf/config.yaml` (no `reward:` cost block):
  ```yaml
  language:
    request: red     # a bucket in the reward head's vocab
    head: null       # path to a train_reward run's reward_head.pt; null -> normal goal-race eval_control
  ```

## Blast radius / build order — as shipped

The steering path **reuses `run_control`'s spine** via an `oracle` toggle rather than a parallel controller.
New: `train_reward.py` (trains `f_z/f_t`, saves the head + `f_t` prototypes), `language/reward.py`
(`LanguageReward`: loads the head, `request→t_e`, `score(latent, t_e)`). Modified, surgically:
1. `controller/mppi.py` — `run_control(reward=None, request=None, oracle=True)`. `oracle=True` builds the
   `true`+`pred` controllers (unchanged dual race); `oracle=False` builds only the learned `pred` controller.
   The rollout functions return a third value `dist` — a per-step distance `(G,K,H)`, or `None` → use the
   Euclidean goal distance. **Reward-as-distance:** in reward mode the learned rollout sets `dist = 1 - R(bag_t,
   t_e)` and feeds it through the SAME `_score` as the goal controller, so `beta_vel`/`r_settle` (near-target
   velocity braking) and `beta_ctrl` apply identically — the agent brakes as it nears the request instead of
   orbiting it, and the "score" reads honestly as a distance (0 = perfectly on the request). Goal advancement /
   settle / early-break are gated to goal mode. `reward=None, oracle=True` is byte-for-byte the old behavior.
2. `imagine_shared(..., return_bag=True)` — exposes the rolled **latent bag** (`_bag`, `(B·K,H,n_state,d)`) so
   the reward can score it; default `return_bag=False` decodes proprio only (unchanged).
3. `controller/run.py` — one `run_and_log_control` handles both modes: it detects `cfg.language.{request,head}`,
   applies `language.overrides` (n_episodes=1, max_steps=500) onto the control config, loads `LanguageReward`,
   calls `run_control(oracle=False, reward=…)`, and logs the SAME products (`control_video_0` without goal rings,
   `<head>/rollout_0` via the shared `products.log_image_head`) plus a `distance_to_request_0` curve (realized
   `1 - R`) in place of `distance_to_goal_0`. (The earlier standalone `language_control.py` / `_run_language`
   were deleted.)
World-model spine/training untouched.

The realized `dist_curve` re-grounds on the real last-P states and rolls ONE dynamics step (`core._rollout`,
the exact path the reward was trained on) rather than `encode_state`, so the curve is measured in the same
latent subspace the planner optimizes.

Deferred: guardrail #2 (off-manifold penalty); the LDA-subspace distance refinement; a `r_settle`/`beta_vel`
sweep for the cosine-distance scale (default `r_settle=0.5` only brakes at `R>0.5` — likely want ~1.0).
