# End-to-end workflow

Every step is a Hydra entrypoint (override any field on the CLI) and writes its own run folder
`logs/<step>_<timestamp>_<experiment>/` — dataset or checkpoints, plots, videos, and a local mirror of
everything sent to wandb. Run the steps in order; `RUN` is your experiment name. Training entrypoints require a **unique** 5-point `run_summary`
(`+run_summary.problem=... +run_summary.tried=... +run_summary.trying=... +run_summary.trying_detail=...
+run_summary.rationale=...`) — train fails fast if it is missing or copies a previous run's note.

## 1. Generate data

```bash
uv run python -m quickdraw.data_generation experiment=$RUN
```

Simulates every split (`train`/`val` + the `eval_ood_*` axes from `conf/data/torus.yaml`) by rolling the
configured environment (`environments.name`, default the torus) with the configured behavior policy
(`data.action_sampler`: `ornstein_uhlenbeck` | `bimodal` | `random`), renders each trajectory's egocentric
clip, and writes one lerobot dataset per split (parquet vectors + mp4 `observation.images.fpv`).
Produces `logs/data_generation_<ts>_$RUN/` — set `DATA=` that path; everything downstream takes
`data.root=$DATA`. `environments.name` selects the env (torus default); recorded trajectories (no
simulator) can instead be converted with `recording_to_lerobot` — see
[docs/byo_environment.md](byo_environment.md).

## 2. Push to the Hub

```bash
uv run python -m quickdraw.push_to_hub data.root=$DATA +hub.name=torus-world
```

Uploads the entire run folder as ONE HF dataset repo (all splits' parquet/meta + normalization stats +
media) with an auto-generated dataset card. The upload is a **clean reupload**: remote files not in this
upload are deleted in the same commit, so a regenerated dataset fully replaces the old one. Auth is
`HF_TOKEN` from the environment. A pushed dataset can later be trained on directly via
`data.hf_repo=<namespace>/<name>` (downloaded + cached) instead of a local `data.root`.

## 3. Train the world model

```bash
uv run python -m quickdraw.train_world_model experiment=$RUN data.root=$DATA \
    +run_summary.problem=... +run_summary.tried=... +run_summary.trying=... \
    +run_summary.trying_detail=... +run_summary.rationale=...
```

Trains the configured model (`model=...`) on `P`-context / `F`-horizon windows. Produces
`logs/train_world_model_<ts>_$RUN/` with `checkpoints/` (top-k + last) — set `CKPT=` that run dir. Two things
happen on a cadence during training:

- **Validation** — every 4 epochs (`trainer.check_val_every_n_epoch`). Val is an autoregressive rollout
  roughly as long as a train epoch; it tracks the in-distribution rollout loss on held-out episodes and
  selects the best checkpoint.
- **Evaluation** — the subscribed eval routines (`conf/eval/default.yaml` `during_train`) run at epochs
  {5, 10, 15, 20, 30, 40, 50, ...}. On by default: `ood_horizon` (long-horizon open-loop rollout — how fast
  accuracy decays past the trained horizon, proprio + image heads), `control` (dual MPPI, oracle vs
  learned — whether the model is good enough to plan with; goal-based when the env supplies goals, else
  reward-only, maximizing `env.reward`), `manifold` (latent-space projections — whether
  the latent has collapsed), and the denoising visuals (diffusion models only). The OOD-split axes
  (visual/geometric/dynamics) are off by default and run post-hoc:
  `uv run python -m quickdraw.eval_ood experiment=$RUN data.root=$DATA checkpoint=$CKPT`.

## 4. Train the action model

```bash
uv run python -m quickdraw.train_action_model checkpoint=$CKPT data.root=$DATA experiment=$RUN \
    +run_summary.problem=... # (same 5 fields)
```

Trains the action-distribution head (the learned play/behavior prior used as the MPPI proposal)
**post-hoc, on a FROZEN world-model checkpoint** — only the action-flow head gets gradients, on context
features computed under `no_grad`. It is post-hoc because joint training destabilized the world model
(control collapsed to ~0 goals vs 3.88 and the WM NaN'd). Produces `logs/train_action_model_<ts>_$RUN/` with
full-model checkpoints (frozen WM + trained head) that load like any other checkpoint, and runs the
action-distribution eval itself (learned prior vs the true data action distribution).

## 5. Interpret — VLM labeling

```bash
uv run python -m quickdraw.eval_interpret checkpoint=$CKPT data.root=$DATA   # needs OPENAI_API_KEY; image world models only
```

This is the step that attaches human-meaningful labels to the world model's latent space with a
vision-language model. It imagines `n_clips` short clips from the frozen WM, decodes them to frames, and
for each clip a VLM (`gpt-4o`) both (a) **labels** it by the semantic factors in `conf/interpret/<env>.yaml`
(torus: the `color` the agent stands on, read from the image; its vertical `positioning`, analytic) and
(b) writes `n_captions` diverse free-form **captions** of the clip. Every step's latent is paired with its
clip's label + captions, and the latent manifold is projected (PCA/UMAP/t-SNE) recolored per factor.
Produces `logs/eval_interpret_<ts>_$RUN/` (per-point latents + per-factor labels + per-clip captions) — set
`INTERP=` that path.

The captions are the text side of the reward model's contrastive distillation (step 6), so this labeling
step is what **grounds language control**: a richer/broader interpret run → a better-generalizing, more
steerable reward head. Point `interpret=<env>` at a `conf/interpret/<env>.yaml` to label a different env.

## 6. Train the reward model

```bash
uv run python -m quickdraw.train_reward_model reward.interpret_run=$INTERP experiment=$RUN
```

Distills a language reward head `R(latent, text) = cos(f_z(latent), f_t(text))` from the interpret run's
captions via CLIP-style contrastive learning (MiniLM embeds the text; the latents come from the frozen
WM). Produces `logs/train_reward_model_<ts>_$RUN/reward_head.pt` — the decode-free scorer MPPI uses to steer by
a text request.

## 7. Language control

```bash
uv run python -m quickdraw.eval_control checkpoint=$CKPT data.root=$DATA \
    language.head=logs/train_reward_model_<ts>_$RUN/reward_head.pt \
    language.request='top red' \
    language.interpret_run=$INTERP
```

Setting `language.head` switches `eval_control` from the goal race to language steering: MPPI maximizes
the reward head's score of the imagined latents against `language.request`. `language.interpret_run`
additionally animates the agent moving through the saved latent projections. Steering-specific knobs live
under `language.overrides` (`n_episodes`, `max_steps`, `lambda_`, ...) so the goal-race defaults are
untouched. Products per run: `control_video_<i>.mp4` (+ a combined tile of all inits),
`reward_trace_<i>.png` (imagined vs achieved, reward head vs ground truth), and the latent-space
animations.
