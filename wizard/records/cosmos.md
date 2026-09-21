# cosmos — NVIDIA Cosmos Video2World as an external world model

Running record for evaluating a model this repo did not train, through the evals this repo already has.
Companion to `src/quickdraw/models/external.py` (the ABC), `external_cosmos.py` (the adapter) and
`conf/model/cosmos{,_blockstack}.yaml`. Started 2026-09-20.

**The claim being tested.** Not "is Cosmos good" — it is a 2B/7B video generator trained on internet-scale
data and we are a small world model trained on one robot. The claim is narrower: *an external model can be
put through `eval_ood_horizon` / `eval_manifold` unchanged, and the resulting number means what the column
says*. Everything below is either that plumbing or a reason a number did not mean what it said.

## Setup

- `diffusers 0.35.2`, already in the container. No NVIDIA monorepo, no transformer-engine/apex/megatron.
- Two pipeline families behind one code path (`external.pipeline`):
  `predict2` = `Cosmos2VideoToWorldPipeline`, default `nvidia/Cosmos-Predict2-2B-Video2World` (4x temporal VAE)
  `cosmos1` = `CosmosVideoToWorldPipeline`, `nvidia/Cosmos-1.0-Diffusion-7B-Video2World` (8x temporal VAE, 7B)
- Both HF repos are `gated=auto`. Predict2 needed one click by `isaac-ronald-ward` before it would download.
- Container additions, `uv pip install` only (never `uv sync`): `cosmos-guardrail` **--no-deps** (it declares
  `transformers>=5`, which would drag us off 4.57 and away from diffusers 0.35; it runs fine on 4.57), plus
  `better-profanity peft retinaface-py scikit-image sentencepiece`, and `nltk<3.10` (3.10's pathsec refuses
  to open a symlink and every file in the HF cache is one). Recorded as the `[cosmos]` extra in pyproject.
- The guardrail is a licence term, not an option: the pipeline raises if `safety_checker is None`. It is
  installed and active (Qwen3Guard-Gen-0.6B on the prompt + RetinaFace face blur on the output; the SigLIP
  video filter is disabled upstream as too false-positive-prone).

## Two upstream bugs, patched narrowly in the adapter

1. `cosmos_guardrail 0.3.1` swapped Llama-Guard for Qwen3Guard, and its `device`/`dtype` properties now read
   attributes a plain `nn.Module` does not have. `DiffusionPipeline.device` touches every registered
   component, so it raised `AttributeError` and the pipeline never ran at all. `_patch_guardrail` derives
   both from parameters.
2. `safety_checker` sorts FIRST among the components (`_get_signature_keys` returns them sorted) and the
   pipeline parks it back on the CPU at the end of every call. So from the SECOND call on, diffusers
   believed the whole pipeline was on CPU and put the token ids there. A one-shot script never sees this;
   an eval loop always does. `_pin_device` reports the denoiser's device instead, via a throwaway subclass.

## Finding 1 — resolution is the whole story, and 144x192 is unusable

The pipeline's only hard constraint is `height % 16 == 0 and width % 16 == 0`, which both our sizes satisfy
(112/16=7, 144/16=9, 192/16=12). So Cosmos CAN run at our native size with no resampling either way. It
should not. block-stack val, open-loop, H=128, 2 eps:

| resolution | LPIPS @+1 | @+32 | @+128 | motion_ratio @+1 | SSIM @+128 |
|---|---|---|---|---|---|
| 144x192 (native, fps 16) | 0.561 | 0.778 | 0.758 | 4.09 | 0.215 |
| 144x192 (native, fps 2)  | 0.646 | 0.770 | 0.729 | 5.67 | 0.167 |
| 704x1280 (its trained size, stretched) | **0.058** | **0.243** | **0.266** | **0.91** | **0.667** |

`motion_ratio` is the tell: at 144x192 the prediction carries 4-6x the frame-to-frame motion of the ground
truth, and LPIPS is high AND FLAT from +1 — it never locks on, so there is nothing to compound. At 704x1280
motion_ratio is ~0.9, i.e. the right amount of motion. The filmstrip says it more plainly than any scalar:
at 144x192 the output is saturated psychedelic noise from step 1.

Cost: 590 s per 88-frame call at 704x1280 vs 8.6 s at 144x192 — **56x**.

CAVEAT ON THAT ROW, now resolved: 144x192 is 3:4 and 704x1280 is 0.55, so the ABC bilinearly STRETCHED the
context going in and squashed the output coming back — that row confounded "more pixels" with "distorted".
It was the pixels. See Finding 1b.

## Finding 1b — the aspect-preserving ladder: no knee below 720x960, cost is 34x

All 3:4 like the data, all divisible by 16, open-loop, H=64, 2 eps, prose prompts (2026-09-21):

| resolution | px vs native | LPIPS @+1 | @+16 | @+64 | SSIM @+64 | motion_ratio @+1 | rollout (2 calls) |
|---|---|---|---|---|---|---|---|
| 144x192 | 1x    | 0.477 | 0.697 | 0.839 | 0.198 | 4.10 | 21.9 s |
| 288x384 | 4x    | 0.497 | 0.747 | 0.776 | 0.224 | 7.32 | 57.8 s |
| 432x576 | 9x    | 0.208 | 0.463 | 0.521 | 0.387 | 4.67 | 150.1 s |
| 720x960 | 25x   | **0.041** | **0.226** | **0.218** | **0.730** | **1.35** | 742.9 s |

Nothing happens until 432x576 and it is still improving at 720x960 — there is no plateau in range, so
"use the biggest you can afford" is the honest rule. Cost is close to linear in pixels (25x pixels ->
34x time), which makes the trade easy to price. 432x576 is the cheap usable point (LPIPS @+64 0.52 at
1/5 the cost); 720x960 is the one to quote.

The 720x960 filmstrip is the first one worth looking at: the table, the three blocks and their colours
and positions are all correct and stable, and an arm is present and moving. What it gets WRONG is
instructive and invisible to LPIPS alone — it hallucinates extra arms and grippers, and the camera
DRIFTS (the viewpoint swings at +46/+55) even though scene_left is bolted down. It is generating a
plausible video of this scene rather than continuing this particular episode under these particular
commands.

## Finding 2 — the frame rate is NOT the story (hypothesis, tested, rejected)

Our steps are 8 frames of 30 Hz footage apart = 3.75 Hz; the pipeline defaults to fps=16. 16/3.75 = 4.3,
which matched the measured motion_ratio of ~4 almost exactly. It was a coincidence. Sweeping the declared
fps over {16, 8, 4, 2} at 144x192 moved nothing:

| fps | 16 | 8 | 4 | 2 |
|---|---|---|---|---|
| LPIPS @+128 | 0.758 | 0.765 | 0.757 | 0.729 |
| motion_ratio @+1 | 4.09 | 6.04 | 6.47 | 5.67 |

## Finding 3 — context granularity is a real handicap, and it is quantised

Conditioning is pinned at LATENT frames, and latent 0 covers pixel frame 0 alone, so only a `kt+1`-frame
context survives the denoiser untouched. At the default `data.P=8`: Predict2 (4x) gets **5** context frames,
Cosmos-1.0 (8x) gets **ONE**, against our 8. `_cond_len` snaps down and drops the same count off the output
so nothing echoed is ever scored, and prints what it used. `data.P=9/17/33` buys an equal-context run.

## Finding 4 — the action channel is text, and the FORMAT matters

Video2World has no action input; the prompt is the entire action channel. The future actions are recorded
ground truth (the same slice our model gets as a tensor) rendered into words by
`evaluation/interpret.build_action_prose` — a deterministic template, no model in the loop.

The bridge originally handed it `build_action_text`, a grid of floats. That is the right format for a VLM
being asked to LABEL a clip it can already see, and the wrong one for a generator whose T5 encoder only saw
scene descriptions — under CFG at 7.0 an out-of-distribution prompt embedding is not ignored, it is pushed
toward. block-stack val, default eval config, 144x192:

| | open_loop LPIPS @+1 | @+2048 | cl_1 LPIPS @+1 | cl_1 SSIM @+1 | cl_1 PSNR @+1 |
|---|---|---|---|---|---|
| table | 0.632 | 0.744 | 0.669 | 0.356 | 11.0 dB |
| prose | **0.561** | 0.799 | **0.397** | **0.515** | **13.7 dB** |

Prose wins clearly at short lead and loses slightly long. Both were at 144x192, so both are measuring the
resolution failure more than anything else — this table needs redoing at a working resolution.

And PER CHUNK, not per rollout: one sentence for 2048 steps says nothing about the 88 frames a given call
generates, and a stick that reverses inside a span averages to "holds still". `_actions` returns a
`TextActions` (a `list` subclass, so every other adapter is untouched) carrying a `window(i, lo, hi)`
renderer; the adapter re-prompts each call with its own frames' actions.

The axis wordings are MEASURED, not read off the names — see `conf/interpret/block_stack.yaml` for the
correlation table (r = +0.94 on every diagonal), the rot6 column-major determination, and the weak (R2 0.33)
horizontal camera fit that "left" for +y rests on.

## Bug fixed — an image-only model got no visual products at all

`score_and_emit` returned `{}` as soon as a model had no proprio head, and that early return sat BEFORE
`emit_openloop` — which writes the filmstrips, rollout mp4s and error-vs-step curves for IMAGE heads too.
Every Cosmos run before 2026-09-21 produced scalars and nothing to look at. Introduced by the
heads-configurable change; fixed by making each proprio product optional on its own input. A second copy of
the same mistake: the `continue` when there is no obs to render skipped the image-head loop after it.

## The prompts are logged next to the rollout

`<mode>/prompts_<i>.txt`, one per plotted episode, one block per generated chunk, in the order the model
saw them. For a text-conditioned model the prompt IS the action channel and it is rendered at run time,
so without this it is unrecoverable from the mp4 it produced. `writer.text` (local only, like
`writer.array`); adapters append to `ExternalWorldModel.prompt_log` and the routine drains it. A model
that takes actions as numbers leaves it empty and nothing is written.

## Experiments (run log)

| date | what | config | result |
|---|---|---|---|
| 09-20 | first light, cosmos1 | starling2 112x192 H=128 2 eps | LPIPS @+1 0.501 @+128 0.778; 44 s rollout |
| 09-20 | timing, predict2 | starling2 112x192 H=64 | 6.73 s / 88-frame call, 16.7 GB, 5/8 context |
| 09-20 | timing, predict2 | block-stack 144x192 H=64 | 8.60 s / call, 16.8 GB |
| 09-20 | **val, table prompt** | block-stack default eval (H=2048 + cl_1 + cl_16) | 4938 s; see Finding 4 |
| 09-20 | **val, prose prompt** | same | 4115 s; see Finding 4 |
| 09-20 | fps sweep | 144x192 H=128 2 eps, fps 16/8/4/2 | no effect — Finding 2 |
| 09-20 | native resolution | 704x1280 H=128 2 eps | LPIPS @+128 **0.266** — Finding 1 |
| 09-21 | products smoke | 144x192 H=64 2 eps | mp4s + filmstrips emit; output is visibly noise |
| 09-21 | **val, 720x960** | block-stack, H=2048 open-loop + cl_16, prose | RUNNING (~8 h) |
| 09-21 | aspect ladder | 288x384, 432x576, 720x960 H=64 2 eps | no knee; 720x960 LPIPS @+64 **0.218** — Finding 1b |

## Costs, for planning

Per 88-frame pipeline call, Predict2-2B, batch 1, 35 steps: 11 s at 144x192, 29 s at 288x384, 75 s at
432x576, 371 s at 720x960, 590 s at 704x1280.

A full val (2 eps) is 24 calls/ep open-loop at H=2048 plus 16 calls/ep for cl_16 = **80 calls**; cl_1 adds
512 calls/ep and is off the table above native size (it was ~90% of the 70-minute 144x192 val, and would be
~84 h at 720x960). So, open-loop at full horizon + `closed_loop_steps=[16]`:

| resolution | full val (2 eps) |
|---|---|
| 432x576 | ~1.7 h |
| 720x960 | ~8.2 h |

## Open

- There is no resolution knee below 720x960 (Finding 1b); is there one above? 1088x1440 (3:4, %16) would
  cost ~2.3x again. Worth one H=64 probe before committing to a long run.
- The camera drift at 720x960 is the sharpest failure and no current metric names it. A fixed-camera
  dataset makes it measurable: background-only optical flow should be ~0 and is not.
- `provides_latents`: what counts as "the latent" for Cosmos — tokenizer latent or diffusion intermediate?
  Until that is decided, `eval_manifold` and `eval_interpret` are skipped by name, which is correct but
  leaves two columns empty.
- Redo the prompt-format A/B (Finding 4) at a working resolution.
- `eval_purple_play` / `eval_purple_stack` (5 and 6 held-out eps) load with no code change via
  `eval.horizon_split=`. Deliberately NOT the default: they are the OOD test for anything trained here.
