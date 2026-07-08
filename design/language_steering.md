# Language-steered MPPI — design note

**Status: planned, not implemented.** Goal: steer the MPPI planner with a natural-language request
("red", "fast and red", eventually compositional) by scoring candidate rollouts **in latent space** — no
image decode in the loop, so it stays real-time. The supervision comes for free from `eval_interpret`, which
already emits `(latent, VLM-label, imagination)` per clip; a language reward/target is a **distillation** of
that (too-slow) VLM signal into a fast latent-space function the MPPI inner loop can call millions of times.

## Text encoder

`sentence-transformers/paraphrase-MiniLM-L3-v2` (~17M, 384-dim), **frozen**. We *train* the latent↔text
alignment, so the encoder only needs semantic structure + smooth generalization to unseen phrasings — MiniLM
suffices; no CLIP text tower, negation not handled for now. Crucially it embeds the request **once per
episode** (the request is fixed during control), so the LM never runs inside the MPPI loop — its size is
irrelevant to planning speed. Only the tiny in-loop scorer (below) runs per candidate-step.

## The two axes

There are really two independent choices, which is why "1 vs 2" felt confusing:

- **Vocabulary:** closed (known buckets) vs open (compositional, novel phrasings).
- **Target shape:** a single target point (distance cost) vs a reward field over the whole latent space.

Options 1 and 2 differ ONLY on vocabulary; both are single-target, so both suffer the same **centroid
mislocation** — averaging a multi-modal request (e.g. "red" appearing in two disconnected latent regions)
lands the target in the empty space *between* the modes. Option 3 is the reward-field answer.

## The three options

| | 1. closed-vocab text→latent | 2. open-vocab text→latent | 3. R(latent, text) per step |
|---|---|---|---|
| in-loop cost / step | `‖latent − target‖` (cheapest) | same | small MLP forward (highest, still ≪ rollout) |
| LM at plan time | once/request (amortized) | once/request | once/request |
| signal shape | smooth, monotone to a point | smooth, monotone to a point | learned field — can be flat/spiky |
| OOD / reward-hacking | low | low | **high** (MPPI exploits off-manifold pockets) |
| vocabulary | closed | open / compositional | open / compositional |
| semantics | "reach X" | "reach X" | reach / avoid / maintain / relative |
| multi-modal target | no (averages) | no (averages) | **yes** (field high at every mode) |
| MPPI reuse | drop-in (swap the spatial goal) | drop-in | new cost surface → **retune** λ/σ |
| training | none (centroids) | train `g` (offline) | train head (offline) |

- **1 — closed centroid.** Fastest scorer, zero training, reuses the tuned distance cost → known-good control
  on day one. But closed vocab, "reach" only, and centroid mislocation on multi-modal buckets.
- **2 — learned `g: text_emb → target latent`.** In-loop cost identical to 1 (the map runs once/request), so
  open-vocab steering at no extra planning cost, still on the reused distance machinery. Same single-target
  limits. Data: the `(latent, label)` pairs from `eval_interpret`, target = per-label centroid (or, better,
  per-mode; see below).
- **3 — reward head `R(proj(latent), text_emb)`.** Most expressive: avoid/maintain/compositional/relative all
  fall out as a dense reward, and it's natively multi-modal. Costs (all real-time-relevant): (a) the most
  in-loop compute (keep the head 1–2 layers); (b) **no smoothness guarantee** — can saturate (flat → no
  gradient for MPPI) or hide spurious high-reward pockets **off the data manifold that MPPI will drive to**,
  worsened by rollout drift into OOD latents where the head never trained; (c) changes the cost surface, so
  MPPI's `lambda_`/`noise_sigma` need re-tuning. Mitigate with a stay-on-manifold penalty (reuse the existing
  contraction / physical losses, or a density term).

## Fixing centroid mislocation without going all the way to 3

Replace distance-to-**centroid** with **min-distance-to-a-set-of-modes**: cluster each request's matching
latents (k-means within the bucket) into a few exemplars and take the distance to the *nearest* one. That is
"go to one or the other, not the average," and it's still real-time (a min over a handful of subtractions).
- Closed vocab: trivial — you have the bucket's latents.
- Open vocab: you must *retrieve* matching latents for a novel phrase, and that retrieval is essentially a
  similarity `R(latent, text)` — i.e. it converges on option 3. So multi-modal + open-vocab naturally tends
  toward the reward field.

## Real-time summary

For all three, the LM cost is amortized to once-per-request; the in-loop cost is a subtraction (1/2) or a
tiny MLP (3) on the latent MPPI already carries — **cheaper than the world-model step it scores**, and it
never decodes an image. The only plumbing: expose the rolled **latent bag** to the scorer (today
`imagine_shared` decodes proprio for scoring; the language scorer reads the bag directly, skipping even that).

## Recommended staging

1. **Stand up 1** (closed-vocab centroids, min-distance-to-modes) to prove latent-space steering end-to-end in
   MPPI with almost no new code and the known-good distance cost.
2. **Move to 2** for open vocab — same in-loop path, add the learned `g` (or retrieval).
3. **Escalate to 3** only when you need avoid/maintain/relative, or when open-vocab retrieval is doing enough
   work that a reward field is simpler — and pair it with a stay-on-manifold penalty to blunt reward hacking.

The interpretability pipeline (`eval_interpret`: latents + VLM labels + saved projections + per-clip
imaginations) is exactly the substrate this needs — language steering is a natural next module, not a rewrite.
