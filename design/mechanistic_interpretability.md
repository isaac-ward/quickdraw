# Mechanistic interpretability — a diagnostic toolkit for the drift/identity failure

Motivated by **"World Modeling in Transformers"** (Beckmann, Queloz, Freitas; arXiv 2609.21748; code MIT
`github.com/bepierre/world-modeling`). They train **TaxiGPT** on random walks through Manhattan and — against
Vafa et al.'s "the implicit world model is incoherent" claim (arXiv 2406.03689) — show mechanistically that the
model DOES store a faithful map, tracks position, and has a "goal compass"; its navigation FAILURES come from
**interference between superposed intersection features**, not a missing map. Their core lesson is directly ours:
**behavioral failure ≠ no world model.** We have been diagnosing our drift/identity failure only BEHAVIORALLY
(OL−floor, filmstrips, the 0.063 invariant). This doc records their toolkit and how to apply it to our latent WM.

## Primer: what mechanistic interpretability is

Reverse-engineering a network's INTERNAL computation, not just its input→output behavior. Two organizing ideas:
- **Features** = concepts stored as DIRECTIONS in activation space.
- **Circuits** = how features are computed and combined across layers.

## The toolbox (methods, and how the TaxiGPT paper used each)

- **Linear probe** — train a linear map activations→concept; good decode ⇒ the concept is linearly represented.
  CORRELATIONAL only (decodable ≠ used). *TaxiGPT:* intersection identity from the layer-18 residual stream via
  a softmax probe (train 80/test 20) → 99.6% of 4,497 intersections decode ≥0.9; street connectivity via 8
  per-move ridge probes → 89.2% at layer 15; legal moves via a ridge probe (F1 0.989, layer 18).
- **Diff-means direction** — the cheapest probe: the vector between class-mean activations, `u_v = c_v − c̄`
  (class mean minus global mean). No training, robust, interpretable. *TaxiGPT:* nearest-centroid on `u_v` →
  99.7% intersection decode.
- **Causal intervention (ablation / patching / steering)** — EDIT an activation along a direction and measure the
  behavior change. This closes the "probing is only correlational" gap: **ablate** (remove a direction),
  **patch** (swap in another value/example's), **steer** (set to a target). *TaxiGPT:* (a) **minimal
  teleportation** at layer 11, `h ← h + α·u_T − β·u_X` (add target intersection dir, subtract current), α=β=1.5
  → 99.3% of nodes then pick a move legal FOR THE NEW intersection — proving the map is USED, not just
  decodable; (b) **compass steering** — set the compass plane to a target bearing → the model follows it (median
  18.1° error); (c) **compass ablation** at layer 18 keeps legal moves ~99.5% but drops goal-reaching 94%→23%.
- **Goal compass** — a 2D CIRCULAR representation, not a single vector. *TaxiGPT:* center states within each
  intersection (remove identity), average into 16 bearing bins `μ_b`, build axes `v_cos = Σ cos(θ_b)μ_b`,
  `v_sin = Σ sin(θ_b)μ_b` → an orthonormal 2D plane at layer 16, median angular error 18.1°.
- **Look-back window ablation** — which past positions does the model actually READ to localize? *TaxiGPT:*
  replace past positions' position-subspace (top diff-means singular dirs covering 90% variance) in the cached
  K/V with move-average centroids → current-position decode collapses 99.8%→14.8%; last-6 → 91.7%, last-20 → 99.2%.
- **Superposition analysis** — networks pack more features than dimensions as overlapping, non-orthogonal
  directions, which then INTERFERE (Anthropic toy-models). Measured via **participation ratio**
  `(Σλ_i)²/(Σλ_i²)` and nearest-neighbor angles. *TaxiGPT:* 244 dims capture 90% of variance among 4,516
  intersection dirs (participation ratio 101 vs 1,600 full); NN angle 48.1° (vs ~90° orthogonal) → packed and
  interfering. Artificially weakening the write + adding interference noise on CLEAN states REPRODUCES the
  illegal-move rate → causal link from superposition-interference to behavioral failure.
- **Logit lens** — read an intermediate activation through the output head to see what the model "would say"
  there. *TaxiGPT:* verifies the legal-move set.
- **Sparse autoencoders (SAEs)** — decompose superposed activations into sparse monosemantic features. NOT used
  by TaxiGPT; the heavier standard tool when diff-means/probes are insufficient.

Code layout (MIT): one analysis per module, `python -m experiments.<name>` over `ckpts/random-walks/model.ckpt`,
shared utils in `common/` (intersections / streets / legal_moves / compass / lookback / teleport / superposition).

## How to apply it to quickdraw (our latent WM)

Our analogue of "intersection identity" is **cube identity/colour/pose**; our residual stream is the per-step
**latent token bag** (32 tokens × d=128 per cam); our "rollout" is the dynamics flow. Read-only on existing
checkpoints — no training, cheap.

1. **Identity probes across the rollout.** Diff-means / linear probes on the latent bag for cube colour / id /
   position at each rollout step. Does "cube-3-is-red" decode at step 1 and COLLAPSE by step ~60? Pins the drift
   to a specific representational decay, and tells us if identity is even linearly encoded (vs already superposed).
2. **Superposition / interference on the bag** — participation ratio + NN-angle on per-cube feature directions.
   **If our cubes are superposed and interfering like their intersections, that is a testable explanation for
   recolour/merge — and for WHY every dynamics knob flatlines at 0.063: the ceiling may be REPRESENTATIONAL
   (superposition), not dynamical.** That would redirect effort from dynamics regularizers (overshoot/DF/
   straightening) toward Ceiling-A CAPACITY (more tokens / bigger bottleneck → more orthogonal features).
3. **Activation patching to localize Ceiling A vs B.** Patch a CLEAN (encoded-true) latent into the rolled bag
   at step t and decode: renders the right cubes ⇒ drift lives in the DYNAMICS (latent), not the decoder. This
   is the rigorous version of "is the decoder masking latent drift?" — and it validates or KILLS the premise
   behind overshoot before spending arms on it.
4. **Look-back window ablation of our rollout.** Their exact method: ablate past latents in the dynamics context
   to see which the model actually reads — does it over-rely on its own recent (drifted) predictions vs the
   grounded initial context? Empirically settles whether the attention-sink / z0-anchor idea
   (`design/identity_preservation.md`) would help BEFORE building it.
5. **An action "compass."** Is the action's effect a clean steerable direction in our latents, and does it
   degrade under drift?

**Highest-leverage transplant: (2)+(3) together** — probe whether cube identity is superposed/interfering, and
patch clean-vs-rolled latents to localize the failure. If the answer is "superposition interference in the
latent," it is a genuinely NEW diagnosis of the 0.063 invariant and redirects effort from dynamics knobs toward
representational capacity. Non-GPU-heavy, runs on existing checkpoints.

Relation to our other docs: `combatting_drift.md` (dynamics/Ceiling-B levers) and `identity_preservation.md`
(z0-anchoring) both ASSUME the failure is dynamical; this toolkit is how we'd TEST that assumption before
committing more training arms to it.
