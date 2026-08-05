# Training & Evaluation

Train the base model (and later variants) with PyTorch Lightning + Hydra; evaluate open-loop
(prediction) and closed-loop (control). Every knob is set from hierarchical YAML. CUDA-accelerated
throughout: `torch.compile`, bf16-mixed, FlexAttention, TF32, fused optimizer.

## Config layout (Hydra groups, sub-yamls — seamstress style)

```
conf/
  config.yaml            # defaults: list composing the groups below
  model/mm_*.yaml        # d, depth, heads, window W, p_tf (mm_dsar / mm_lsar / mm_flow, + _proprio variants)
  data/torus.yaml        # dataset name@version, P, F, batch, workers
  optim/adamw.yaml       # lr, weight_decay, betas, schedule, grad_clip
  trainer/default.yaml   # max_epochs, precision, check_val_every_n_epoch, save_top_k
  environments/torus.yaml  # R, r, dt, gamma, a_max  (shared by generation + control)
  logging/wandb.yaml     # project, group, cadence
  eval/default.yaml      # rollout horizon + during_train: {every_epochs, evals: {ood_horizon: true, control: true, ...}}
  control/mppi.yaml      # horizon, num_samples, noise_sigma, lambda, tol, max_steps, beta_vel, r_settle
```
`config.yaml` composes these via a `defaults:` list; override any field on the CLI
(`model.depth=6 control.num_samples=1024`). One knob, one place, fully hierarchical.

## Training loop

- `LightningModule` wraps `BaseWorldModel`; `Trainer(max_epochs, precision="bf16-mixed",
  gradient_clip_val=1.0, check_val_every_n_epoch=4)` → **validate every 4 epochs** (the locked
  cadence, `conf/trainer/default.yaml`).
- `ModelCheckpoint(monitor="val/manifold_distance_error", mode="min", save_top_k=k, save_last=True)`.
  Resolved config + normalization stats written to the run folder (`data.md`, `logging.md`).
- `training_step`: build fused step-tokens, run the `p_tf` rollout, MSE on delta, log `train/*`.
- `validation_step`: windowed val loss/metrics. Separately, `EvalCallback` runs the **subscribed**
  eval routines (`cfg.eval.during_train.evals`, a dict of bools) every `every_epochs` — train/val are unaffected.

## Speed (CUDA)

- `torch.compile(...)`; `precision="bf16-mixed"`; `torch.set_float32_matmul_precision("high")` (TF32).
  > SUPERSEDED (see `accelerations.md` Exp 8): **multimodal models SKIP whole-model compile** (the per-batch
  > image gather + ViT AE complicate it). The **parallel** forward (epoch-0, p_tf=1) still gets a fused
  > FlexAttention kernel — but the **serial AR rollout runs attention UNFUSED** (it emits `flex_attention called
  > without torch.compile()`), which is why the opt-in `model.compile_rollout` (compile the step, ~6×) is a real
  > win, not a no-op — see Exp 9. Non-mm models compile in **default** mode (not `max-autotune`: the
  > parallel forward runs ~1 epoch under the p_tf curriculum, so the long autotune search isn't worth it).
- **FlexAttention** for the causal + sliding-window (`W`) mask: one `mask_mod` (causal AND within
  `W`) compiled to a block-sparse kernel that skips out-of-window blocks — faster than a dense SDPA
  mask. Built once, reused every layer and rollout step.
- Fused `AdamW(fused=True)`; DataLoader pinned + workers + prefetch (`data.md`).
- The AR rollout fn is shared by training, open-loop eval, and control.
  > The rollout runs **EAGER by default**. The AR step is **dispatch-bound** (the serial F-step loop, GPU ~20%
  > util), so the throughput levers are batch (nearly free) and compiling the step. Opt-in `model.compile_rollout`
  > does the latter: `torch.compile(step, mode="default")` — **~6×, parity-safe** (Exp 9). It both fuses the
  > per-step attention (which is UNFUSED in the eager rollout) and collapses the ~256 dispatches. Note
  > `mode="reduce-overhead"` (CUDA graphs) does NOT work — incompatible with the retained-BPTT rollout. The old
  > "~57-shape recompile thrash" that made rollout-compile look hopeless is gone (fixed-window `pad_block_mask`
  > caps the shapes). See `accelerations.md` Exp 8/9 + `design/rollout_throughput.md`.

## Pipeline (4 steps; run-dir prefix = step)

`data_generation` → `train` (train/val + subscribed in-loop evals) → standalone `eval_ood_horizon` /
`eval_ood` / `eval_control` at the best checkpoint. Runs land in `logs/<step>_<timestamp>_<experiment>/`.

## Eval routines (defined once; subscribed in training AND runnable standalone)

~10 routines in `evaluation/routines.py`, registered by name in `REGISTRY`: `ood_horizon` (long-horizon
open-loop rollout), `ood_visual`, `ood_geometric`, `ood_dynamics` (open-loop on the OOD splits, each scored
on its **own geometry** from the dataset card), `control` (MPPI with the world model as dynamics — below),
`denoising_multistep`, `denoising_aggregate`, `manifold`, `interpret`, `action_distribution`.

Each routine is `(cfg, model, norm, env, run_dir, device, wandb_run) -> summary`; it saves plots and
logs to W&B. Two invocations, no duplication:
- **During training** — `EvalCallback` runs the routines enabled in `cfg.eval.during_train.evals`
  (a dict of bools, one per registered routine) every `every_epochs`. `0` or all-false disables.
- **Standalone at best ckpt** — `quickdraw.eval_ood_horizon | eval_ood | eval_control`, each loading
  the train run's `best.ckpt` via `checkpoint=<train_run_dir>`.

## control routine — MPPI with learned dynamics

The trained world model is the predictive dynamics inside a receding-horizon MPPI loop (reuses the
same (eager) rollout fn — no duplication). Own routines, own W&B group `eval/control/`.

**Goals (8).** 4 toroidal compass directions × {outer, inner} ring (`torus_utils.control_goals`):
compass = E `θ=0`, N `θ=π/2`, W `θ=π`, S `θ=3π/2`; rings = outer `φ=0`, inner `φ=π` — all in the
`z=0` plane. Named e.g. `out_N`, `in_S`.

**Loop (per control step).** Sample `num_samples` action sequences over horizon `H` (Gaussian around
the shifted previous mean, `noise_sigma`); roll all through the world model from the current context;
reward `r = −‖p̂ − target‖ − β·w(d)·‖ṗ̂‖`, where the velocity penalty is gated by `w(d)` — it ramps
in as `d = ‖p̂ − target‖` drops below `r_settle`, so the controller approaches fast and **settles**
on arrival rather than overshooting (`β, r_settle` in `control/mppi.yaml`);
weights `= softmax(returns/λ)`; mean `=` weighted average; execute the
first action on the **true `TorusEnv`**; append to context; repeat until `‖p − target‖ < tol` or
`max_steps`. All `num_samples` and all 8 goals are batched into one GPU model call.

**Logging.** Control-loop Hz, time-to-completion, success, final distance, the target atlas, and the
per-target control videos are all logged under `eval/control/` — see `logging.md`.

**Acceleration.** `num_samples × 8` rollouts batched through the compiled bf16 model; Hz reported
honestly (sequential over `H`, parallel over samples and targets). Goal: real-time-ish, run fast.

## Modularity

One rollout fn (`BaseWorldModel.imagine`) serves training (`p_tf`), open-loop eval, and MPPI. Metric
functions imported from the env; torus plotting from `logging.md`. MPPI is a thin controller around
`imagine`, so swapping base → variants needs no control-code change.
