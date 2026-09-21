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

**This is now the default.** `model.render_size` is what Cosmos runs at (720x960 for block-stack, an
extrapolated 672x1152 for starling — both an exact integer multiple of the data size, so identical aspect
and both axes %16); `modalities.img_size` stays what the data is decoded at and what every metric is
computed at. Order of precedence: `external.height/width` (explicit CLI) > `model.render_size` > the
modality size. Nothing runs at native size by accident any more.

## Finding 2 — RETRACTED: the fps sweep was run at the broken resolution and proves nothing

Our steps are 8 frames of 30 Hz footage apart = 3.75 Hz; the pipeline defaults to fps=16. 16/3.75 = 4.3,
which matched the measured motion_ratio of ~4 almost exactly. It was a coincidence. Sweeping the declared
fps over {16, 8, 4, 2} at 144x192 moved nothing:

| fps | 16 | 8 | 4 | 2 |
|---|---|---|---|---|
| LPIPS @+128 | 0.758 | 0.765 | 0.757 | 0.729 |
| motion_ratio @+1 | 4.09 | 6.04 | 6.47 | 5.67 |

**That sweep was run at 144x192, where the output is noise whatever you do (Finding 1/1b), so it
separates nothing.** "No effect" was the resolution failure swamping the variable, not evidence that fps
is inert. Worse, reading the code shows fps is NOT a passive conditioning scalar — it rescales the
temporal ROTARY POSITION EMBEDDING directly (`transformer_cosmos.py`, `base_fps = 24`):

    emb_t = torch.outer(seq[:pe_size[0]] / fps * self.base_fps, temporal_freqs)

so the declared fps sets how far apart the model believes consecutive latent frames are. Our steps are
8 frames of 30 Hz footage = 3.75 Hz, and we declare 16, i.e. we tell it the frames are 4.3x closer
together in time than they are. The principled value is 4 (nearest int to 3.75). REDO AT 720x960.

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

## Finding 5 — our prompts against the card's spec: too thin, and describing 4.7x too much time

The card asks for "fewer than 300 words ... a scene description, key objects or characters, background,
and any specific actions or motions to be depicted **within the 5-second duration**". Its own example is
~100 words. Ours (block-stack val ep0, 24 windows) run **20-53 words, mean 37**.

Two gaps, and the second is structural:

1. CONTENT. We give the scene and the motion. We do not name the key objects (three cubes: green, blue,
   red) or describe the background (grey wall, chairs, tiled floor) — both explicitly requested. There is
   room: we are using an eighth of the budget. Enriching `scene_prompt` is free and should also carry
   "the camera never moves", which is a direct instruction against the measured drift.

2. DURATION, which no amount of wording fixes. One chunk is 88 predicted steps, and at stride 8 on 30 Hz
   footage that is **23.5 seconds of real time** presented to the model as one 5-second clip — 4.7x more
   action than the format is designed to hold. It cannot depict 23.5 s in 5 s; it compresses or ignores.

   The fix is to stop asking, by choosing the STRIDE so our frames land near the model's own spacing.
   30 Hz does not divide by 16, so 15 Hz -- `data.subsample=2` -- is the closest this footage gets:

   | data.subsample | we feed | declared fps | 5 ctx frames | 93-frame clip | vs the 5.8 s design |
   |---|---|---|---|---|---|
   | 8 (what we ran) | 3.75 Hz | 4 | 1.33 s | 23.3 s | 4.0x |
   | 4 | 7.5 Hz | 8 | 0.67 s | 11.6 s | 2.0x |
   | **2** | **15 Hz** | **15** | **0.33 s** | **6.2 s** | **1.1x** |
   | 1 | 30 Hz | 30 | 0.17 s | 3.1 s | 0.5x |

   Stride 2 is the sweet spot and it beats the obvious "run at native 30 Hz" on both axes: closer to the
   design point (1.1x vs 0.5x) and half the cost. For the same WALL-CLOCK horizon it is 4x the calls:
   34 s of footage goes 0.4 h -> 1.2 h, the full 9.1 min goes 4.9 h -> 19.4 h. The catch is that step
   counts stop being comparable across models -- our +128 at stride 8 is Cosmos's +512 at stride 2, both
   34 s -- so the comparison has to be made at matched SECONDS.

   FPS IS NOW DERIVED, not defaulted. The adapter reads the dataset's own fps from summary.json (the same
   source env_cfg uses for dt), divides by data.subsample, and declares that. Whatever stride is chosen,
   what we tell the model now matches what we send it; `external.fps` overrides to test that claim.

## Bug fixed — an image-only model got no visual products at all

`score_and_emit` returned `{}` as soon as a model had no proprio head, and that early return sat BEFORE
`emit_openloop` — which writes the filmstrips, rollout mp4s and error-vs-step curves for IMAGE heads too.
Every Cosmos run before 2026-09-21 produced scalars and nothing to look at. Introduced by the
heads-configurable change; fixed by making each proprio product optional on its own input. A second copy of
the same mistake: the `continue` when there is no obs to render skipped the image-head loop after it.

## The prompts are logged next to the rollout

`<mode>/action/prompts_<i>.txt`, one per plotted episode, one block per generated chunk, in the order the
model saw them — nested under `action/` the same way the pixels are nested under `image/`, because the
prompt IS the action channel for these models. For a text-conditioned model the prompt IS the action channel and it is rendered at run time,
so without this it is unrecoverable from the mp4 it produced. `writer.text` (local only, like
`writer.array`); adapters append to `ExternalWorldModel.prompt_log` indexed by their position in the
call's sub-batch, and `rollout_regrounded` translates that to the global (episode, segment) row — only
it knows `r0` and the row layout. Step ranges are absolute, so a cl_16 run reads +1..+16, +17..+32, ...
rather than thirteen segments all claiming +1..+16. A model that takes actions as numbers logs nothing.

## Finding 6 — stride 2 works, and the noise floor says what the numbers can and cannot show

At `data.subsample=2` (15 Hz fed, fps 15 declared, 6.2 s clips, 720x960) the prediction is finally a
picture of the actual episode: the table, all three cubes in their right places holding their colours,
one arm, a stable viewpoint, through 11.7 s of pure open-loop rollout. Read at MATCHED SECONDS, which is
the only fair way across strides:

| footage | stride 8 | stride 2 |
|---|---|---|
| 2.13 s | @+8 LPIPS 0.199 | @+32 LPIPS **0.187** |
| 4.27 s | @+16 LPIPS 0.226 | @+64 LPIPS **0.140** |
| 11.7 s | — | @+176 LPIPS 0.355 |

THE NOISE FLOOR, from two runs identical but for the seed (H=176, 2 eps):

| metric | mean abs diff between seeds | worst lead |
|---|---|---|
| LPIPS | **0.024** | 0.059 |
| SSIM | **0.033** | 0.096 |
| motion_ratio | **0.640** | 1.306 |

So at 2 episodes a LPIPS gap under ~0.05 means nothing, and **motion_ratio is not usable as a
discriminator at all** — the same config gave 1.507 and 0.201 at +64. Every earlier reading of
motion_ratio at 720x960 (the "camera drift" inference included) is therefore unsupported; the 4-6x
values at 144x192 were consistent across many runs and survive, but nothing at working resolution does.

Two episodes is the whole block-stack val split, so the fix is not more compute at this split. Note that
`eval_purple_play` (5 eps) and `eval_purple_stack` (6 eps) cost NOTHING to spend on Cosmos — it never
saw any of this data, so every split is equally out of distribution for it. That reservation applies to
models trained here, not to an external one.

## Finding 7 — the prompt described stick directions, not the task

Reading a real prompt against the footage showed two things wrong, both mine.

The SCENE half asserted a configuration: "stands at the right edge of the table", "sit on the tabletop".
Neither survives ten seconds of an episode -- the arm moves and the cubes get stacked on each other. It
now describes only what is always true, and states the task: the arm "picks up coloured cubes, lifts
them and stacks them on top of each other".

The ACTION half never mentioned the manipulation at all. The gripper was averaged like a rate axis, so
a state got reported once per phase -- "opens its gripper, then opens its gripper and lowers, then opens
its gripper and rises" -- while the grasp, the lift and the place were never said. An axis can now
declare `role: gripper` and is read as TRANSITIONS:

    closes its gripper on the cube, lowers, then moves left across the bench and rotates
    counter-clockwise seen from above, then opens its gripper and releases the cube, rises

"while holding the cube" is said once per hold rather than on every phase it spans.

MEASURED, not assumed: the gripper channel is not noisy. 80 grasps over 9096 steps (10 min), mean hold
5.03 s, one grasp every 7.6 s. A median filter only destroys real grasps (80 -> 52 at 1.4 s), so there
is no debounce.

WHAT IT STILL CANNOT SAY IS WHICH CUBE. block-stack records the arm -- ee pose, rot6, gripper, joints --
and no object poses, so there is nothing to read; naming a cube off the future frames would be handing
the model the answer. "the cube" is the honest limit. The model can see the cubes in its context frames;
what it cannot infer is the intent, and that is what the prompt is for.

## Finding 8 — the VLM can name the cube, and names the wrong one. Tried, measured, dropped.

`label_clip` was given exactly what Cosmos gets and nothing more: the 8 ground-truth CONTEXT frames plus
`build_action_text` (the full stick table, gripper included) from t=0 to the end of the chunk. No future
frames, so nothing leaks -- it is re-expressing information the model already has. It answered fluently:

    [chunk 1] objects=['green'] confident=True
      "The robot arm moves toward the green cube, lowers, and closes its gripper, picking it up. It then
       carries the green cube left across the table and places it next to the blue cube."

Checked against colour-segmented cube centroids rather than against my eyes (288x384, x = right):

    green  (243,175)  (242,176)  1px   (241,176)  1px   (242,176)  2px
    blue   (299,182)  (300,181)  1px   (288,195) 18px   (304,170) 29px
    red    (334,196)  (275,200) 58px   (276,211) 11px   (276,200) 12px
    marks:       +0        +88              +176             +264

Green moves ONE PIXEL across the whole window -- it is the only cube that never moves -- and the VLM
named it as the one picked up and carried. Red, which moved 58 px, went unmentioned. Chunks 2 and 3 name
red while blue is the mover. Wrong cube three times out of three, `confident: true` three times out of
three. That is strictly worse than saying "the cube": the generator would be instructed to move a cube
that stays still. Dropped.

The failure is reasonable -- integrating 88 steps of joystick into a trajectory and binding it to objects
in a 45-degree view is hard, and it fell back on generic block-stacking narration. The route that could
work does not need a VLM for the hard part: cube pixel positions from colour segmentation on the context
frames (works, above), the gripper path by integrating the sticks (exact, and non-leaky -- PROCESSING.md
gives the integrator rule), then nearest-cube at grasp time. The weak link is projecting world mm to
pixels: the fit is R2 0.90 vertical but 0.33 HORIZONTAL, and horizontal is the axis that separates the
cubes. Not attempted.

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
| 09-21 | val, 720x960, stride 8 | block-stack, H=2048 + cl_16 | KILLED at 52 min — superseded by stride 2 |
| 09-21 | negative-prompt A/B, stride 8 | 720x960, H=200, 3 arms | KILLED at 24 min — same reason: the negative's motion clauses interact with motion-per-frame, so it has to be run at the stride we will use |
| 09-21 | stride-2 smoke | 720x960, H=176, 2 eps, fps auto 15 | the scene, stable, 11.7 s — Finding 6 |
| 09-21 | seed repeat | same, `+external.seed=1` | noise floor: LPIPS 0.024, motion_ratio 0.640 |
| 09-21 | val ablation, stride 2 | 720x960, H=8192 + cl_16 | KILLED at 2 min — the prompt described stick directions and never said a cube was picked up (Finding 7) |
| 09-21 | VLM object attribution | gpt-4o, context frames + stick table | wrong cube 3/3, confident 3/3 — Finding 8, dropped |
| 09-21 | **val ablation, stride 2, v2** | 720x960, H=8192 (9.1 min) + cl_16, 94 calls/ep, grasp-event prompts. GPU0 scene only, GPU1 scene + negative | RUNNING (~19 h) |
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
- ~~Enrich `scene_prompt`~~ DONE 09-21: 14 words -> 71, naming the three cubes, the wall, the chairs and
  the floor, and ending "The camera is bolted down and never moves" against the measured drift.
- H=128 head-to-head: current scheme vs native-rate-and-decimate (Finding 5.2).
- Redo the fps sweep at 720x960 (Finding 2 is retracted). fps=4 is the principled value; the question is
  whether matching the temporal RoPE to our real step rate beats staying at the trained 16.
- `nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned` takes NUMERIC per-step actions ("end-effector
  displacement and gripper width"), 640x480 at 4 fps — block-stack's semantics at our step rate, and an
  apples-to-apples row the text bridge can never be. Runtime is the cosmos-predict2 monorepo, not
  diffusers (a raw .pt + tokenizer .pth, no model_index.json), so it costs the NVIDIA stack we avoided.
- `eval_purple_play` / `eval_purple_stack` (5 and 6 held-out eps) load with no code change via
  `eval.horizon_split=`. Deliberately NOT the default: they are the OOD test for anything trained here.
