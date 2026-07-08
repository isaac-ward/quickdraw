# Interpretability module — `eval_interpret/` (design, not yet built)

**Goal:** probe whether the world-model's LATENT space organizes by human-legible semantic factors — **color,
speed, direction** — by VLM-labeling short imagined clips and recoloring the `eval_manifold` projections by those
labels. If the latent separates by a factor, the model has learned a (partly) legible representation of it.

**When:** run ONCE, post-training. It IS a registered eval routine (so it's organized + reusable) but is kept
OUT of the during-train cadence — see "Routine structure" below.

**Precondition:** exactly one vision trunk. Raise a clear error otherwise (it's inherently a vision probe).

---

## Routine structure (answering "do we have an eval_routine class?")
No class — eval routines are **functions** with a shared signature `(cfg, model, norm, ecfg, writer, device, step)`
registered in `routines.REGISTRY`, and `conf/eval/default.yaml`'s `during_train.evals` list selects which run
*during* training. So `eval_interpret` slots in as a REGISTRY entry (organized, callable) but is simply **left out
of `during_train.evals`**, and gets a **standalone entrypoint** `python -m quickdraw.eval_interpret checkpoint=...`
(mirrors `eval_diffusion`). Available on demand, never in the train loop. No refactor to a class needed.

## Pipeline
1. **Sample N=1024 action sequences from VAL** and **imagine** each: seed with P context frames (real), roll the
   model forward a **0.5 s clip = 30 frames @ 60 Hz**, keep decoded FPV frames + carried latents. Short enough for
   one motion.
2. **Label each clip by factor, per a config-declared SOURCE.** Two factors: **color** and **speed** (direction
   was dropped — 0.5 s carries too little turn signal, agreement sat at chance).
   - **color → `source: vlm`.** OpenAI `gpt-4o-mini` reads the RENDERED image (seamstress harness: Responses API,
     K base64 frames as `input_image` + `json_schema` output, `OPENAI_API_KEY`) and reports the color of the
     surface the agent is STANDING ON (bottom-center patch). This is an INDEPENDENT probe — the image is a separate
     decode head, not the proprio the latent came from.
   - **speed → `source: analytic`.** Computed exactly from the imagined velocity, bucketed by **quantile terciles**
     (slow/medium/fast split at the 33rd/66th percentiles of the clip set) — self-calibrating, no magic thresholds.
   - The analytic hue-at-position doubles as the **cross-check** for the VLM color (confusion + agreement).
3. **One point per clip** = the mean over the imagined rollout of the model's **internal predictive state** (the
   carried token bag from `_rollout`, NOT a re-encoding of the decoded output). So we color the state that PRODUCES
   the open-loop prediction by that prediction's OUTPUT (decoded velocity / rendered color) — an output-vs-state
   probe, not encoding-vs-its-own-input.
4. **Project once, recolor per factor:** for each reducer (umap/tsne/pca) × dim (2D/3D), fit the projection ONCE on
   the embeddings, reuse those coordinates for each factor's coloring (color/speed overlay the same cloud).
5. **Output** `eval_interpret/{umap,tsne,pca}/{color,speed}_{2,3}d.png` — categorical color + **legend**.

## One point per clip
The point is the **mean over the imagined rollout of the internal predictive state** (the carried token bag). The
label is per-clip, so one labeled point directly answers "does the latent cluster by this factor," and 1024 points
read cleanly with a categorical legend.

## Config-driven label sets (per environment)
Mirror seamstress's per-env config → **`conf/interpret/torus.yaml`** (swap for other environments). It defines:
- the **sets** and their **discrete buckets**: `color: [red, orange, yellow, green, blue, purple]` (the surface
  the agent is STANDING ON — the patch at the bottom-center of the FPV, not ambient visible color),
  `speed: [stopped, slow, fast]`, `direction: [forward, left, right]`;
- the **VLM prompt** + `model: gpt-4o-mini`, **frames-per-clip fed to the VLM** (K), clip length (30);
- a categorical **color map** per bucket for the legend.
So changing environments = a new `conf/interpret/<env>.yaml`, no code change.

## Reuse eval_manifold — general code + particularizations
Factor the manifold plotting so **both** routines share it:
- `reduce_dims(umap/tsne/pca)` is already general.
- Add a **general point-cloud plotter** that takes an optional `labels`+`legend` (categorical color + legend box)
  OR `None` (structure-only). Extend `fig_points_2d` / `fig_points_6view` to a categorical variant.
- **`eval_manifold`** = the particularization with no labels (structure only, current behavior).
- **`eval_interpret`** = the particularization with categorical labels + legend, same reducers/dims, "project once
  reuse coords."

## Cross-check (VLM trust check)
All three factors are analytically derivable from the model's **imagined proprio** (the imagination gives proprio
+ image), so we can grade the VLM against ground truth *of the same imagination*:
- **speed** — bucket `|imagined velocity|` (proprio dims 3:6) with the config thresholds → stopped/slow/fast.
- **direction** — sign of the heading change (Δ of the velocity's tangent angle) over the clip → forward/left/right.
- **color** — apply the known torus **coloring function** to the imagined surface position → expected dominant color.

For each clip compare VLM label vs analytic label, per factor. **Output:** a per-factor **agreement %** (logged +
`eval_interpret/crosscheck/summary.json`) and a **confusion-matrix figure** `crosscheck/{color,speed,direction}_confusion.png`.
High agreement ⇒ trust the VLM. Low ⇒ either the VLM is noisy *or* the model's rendered video disagrees with its own
predicted proprio (itself an interesting finding). It also gives a labeling fallback if the VLM is unreliable.

## Outputs & how you'll look at it
Standalone run → a fresh `logs/eval_interpret_<ts>_<label>/` dir (like `eval_diffusion`). Under it:

    eval_interpret/
      umap/  {color,speed,direction}_{2,3}d.png     # 6: same projection, 3 recolorings x 2 dims
      tsne/  {color,speed,direction}_{2,3}d.png      # 6
      pca/   {color,speed,direction}_{2,3}d.png      # 6   (18 manifold plots total)
      crosscheck/  {color,speed,direction}_confusion.png  + summary.json
      labels.json        # per clip: {episode, start, vlm:{...}, analytic:{...}}  -> drill into any point
      examples/{factor}/{bucket}.mp4   # a 4x4 GRID composite (up to 16 example clips per bucket), ~1s playback

**Reading it:** each plot is the SAME projected cloud of 512 clip-points, recolored + legended by one factor —
so `pca/direction_3d.png` vs `pca/color_3d.png` are the identical geometry, different colors. If points of one
bucket group together, the latent encodes that factor; **PCA is the honest "are they really separated" read**,
UMAP/t-SNE show neighborhood/cluster structure. Cross-check confusion figs tell you whether to trust the labels.
`labels.json` + `examples/` let you click into specific clips.

**Progress log** (illustrative):

    [eval_interpret @ep100] start: 512 clips x 30 frames (0.5s) from val | vision trunk=image_fpv
    [eval_interpret @ep100] imagining 512 clips... 25% | 50% | 75% | done in 41s
    [eval_interpret @ep100] VLM labeling (gpt-4o-mini, K frames/clip)... 128/512 | 256/512 | 512/512 (ETA ~HH:MM)
    [eval_interpret @ep100] label counts: color{red 61,orange 44,...} speed{stopped 90,slow 210,fast 212} direction{fwd 240,left 141,right 131}
    [eval_interpret @ep100] cross-check agreement: color 87% | speed 93% | direction 88%
    [eval_interpret @ep100] projecting umap/tsne/pca (2d+3d) + rendering 18 plots + 3 confusions...
    [eval_interpret @ep100] done in Xs -> eval_interpret/

## Status / needs
- Reuse seamstress `vlm_labeling.py` (Responses API + json_schema + base64 frames) + its per-env config pattern.
- `OPENAI_API_KEY` (Isaac providing) in the run env.
- Deliverable to review once written: the 18 plots (3 reducers × 2 dims × 3 sets) under `eval_interpret/`.
