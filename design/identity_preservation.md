# Identity preservation — anchoring the rollout to its grounded start

Sibling to `design/combatting_drift.md`. Drift (Ceiling B) is "the latent walks off"; this doc is about the
specific symptom we care most about — **object identity lost in transit** (cubes recolour/merge over the
rollout, §8.15). The idea here is orthogonal to the drift losses: give the autoregressive chain a **persistent
reference to the grounded initial state** so identity has something true to re-anchor against, instead of only
its own drifting recent past.

Context: our rollout runs a causal transformer with `use_kv_cache=True` over a sliding **window**. The initial
grounded context `z_0` (the first P encoded frames) is present at the start but **slides out of the window**
after `window` steps — after which nothing ties later predictions to the true initial appearance of the objects.
Both ideas below are ways to **re-inject `z_0` after it leaves the window**; they differ in mechanism.

We do NOT act on this yet — parked for later. Two candidate mechanisms:

## Idea 1 — Initial-condition conditioning (the easy one)

Keep the grounded bag `z_0` and feed it into EVERY step's backbone input, via concat or FiLM (a small change to
`_to_input`/`_cond`, plus the rollout carrying `z_0` alongside the window). The initial appearance is then always
present in the conditioning, guaranteed to influence every prediction.

- **Mechanism:** explicit input — `z_0` is concatenated/FiLM'd into each step, so it is *guaranteed used* (it is
  literally in the input).
- **Pros:** simplest; robust; a good first test of "does a persistent anchor help at all". Composes with delta
  prediction (predict the step relative to `z_0` rather than relative to the previous, drifting bag — we already
  predict residuals `next = prev + d`; anchoring the residual to the INITIAL bag is a small variant).
- **Cons:** crude — the anchor is a fixed broadcast signal, not selectively retrievable. The model can't ask
  "what colour was cube 3 specifically" and pull just that; it gets the whole `z_0` as context every step.
- **Config:** `anchor_init: bool`, `anchor_mode: concat|film`. Off = bit-identical.
- **Blast radius: LOW-MEDIUM** — the backbone input assembly + the rollout carrying `z_0`.

## Idea 2 — Attention sink (the expressive one)

NOT a special transformer feature — a **KV-cache management trick** (StreamingLLM, Xiao et al. 2023). In our
sliding-window cache, normally the oldest tokens are evicted; the sink **never evicts `z_0`'s keys/values** (and
maybe the first P context frames), so every future query can *attend* to the grounded start with full
query-key selectivity. Exploits the empirical finding that models dump attention mass onto the first tokens, and
that keeping them stabilizes long/streaming generation.

- **Mechanism:** attention — `z_0`'s K/V stay attendable forever; the model *learns to retrieve* specific
  tokens/features from it per query. This is exactly the associative "what colour was that cube" lookup that
  identity needs — the better-matched mechanism in principle.
- **Pros:** expressive, selective retrieval; the right shape for identity.
- **Cons:** a sink usually **only helps if you TRAIN with it** — bolting it onto a model trained without it often
  does little, so it's not a free inference tweak; it's a training-regime change. Touches the KV-cache/window code.
- **Config:** `attention_sink_tokens: int` (how many initial bags to pin), applied in the rollout cache.
- **Blast radius: MEDIUM** — the KV-cache eviction policy in the rollout, and it wants to be trained-with.

## Which is stronger?

- **Init-conditioning** is the easier, lower-risk FIRST test — guaranteed-used, simple, and tells us whether a
  persistent anchor moves the identity failure at all.
- **Attention sink** is potentially stronger for identity specifically (selective key-value retrieval matches
  the "recall this object's appearance" operation), but needs training-with-it and touches the cache.

Recommendation: **init-conditioning first** (cheap, robust); if it helps and we want selective retrieval,
graduate to the sink. Both are really "re-inject `z_0` after it leaves the window" — conditioning via the input,
the sink via attention.

## Relation to the literature (from the scan)

- **Attention sink** = Rolling Forcing (arXiv 2509.25161, code TencentARC/RollingForcing).
- **Sparse context memory** (EnerVerse-AC, arXiv 2501.01895) generalizes the sink: keep a SPARSE SET of retained
  earlier real latents as anchors, not just `z_0`. Same axis, more anchors; a retention policy on the rollout
  context. Slightly more than the sink.
- **Delta prediction** (Δ-IRIS, arXiv 2406.19320): carry identity in a persistent (recurrent) state summary and
  predict only the per-step DELTA, so unchanged cubes are never re-synthesized/recoloured. Our residual
  prediction anchored to `z_0` (Idea 1's delta variant) is the cheap approximation of this.

## NOT to be confused with our existing DINO loss

We already have a DINO term (`models/dino_loss.py`, `DinoV3Term`, per-patch cosine) and tried it — as a DECODE
perceptual loss (DINO features of the decoded frame vs GT, a Ceiling-A render term); `bs_dino_w25` diverged at
ep1, `dino 2.5 + lpips 1` was the matched-contribution point. That is a *render* loss, distinct from (and not a
substitute for) the *dynamics-space anchoring* here. The DINO-WM idea (predict dynamics IN frozen-DINO feature
space) is a third, separate thing and a bigger lift; given DINO's already been finicky here it is NOT a near-term
pick.
