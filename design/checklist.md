# Execution Checklist — quickdraw

## STATUS (implemented)
Phases 0–8 are **coded and committed to the tree**. All 25 modules `compileall`-clean; all 9 Hydra
configs parse. Core logic **verified on real torch (CPU smoke test)**: env stays on-manifold,
16 targets on-surface, parallel forward, `imagine` rollout indexing, `rollout_train` gradient flow,
and the three metrics. (Seamstress source was ported into the package; re-pull from the seamstress
repo for the image-stage ViT when needed.)

**Validate on first Docker/GPU run** (could not be exercised in the sandbox — no GPU/heavy deps):
FlexAttention (flex-only, fails hard off-GPU — confirmed), the lerobot read/write API in
`data/generate.py:write_lerobot_split` + `data/dataset.py:load_split_episodes`, Lightning/W&B
wiring, and `torch.compile`. Image stage (Phase 9) and variants (Phase 10) remain deferred.

Pipeline is modular: `docker compose up -d` starts an idle container; `exec` in and run the 4 steps
yourself — `data_generation → train (val + in-dist open-loop) → eval_ood → eval_control`. The two
eval steps load the train run's `best.ckpt`. Runs: `logs/<step>_<timestamp>_<experiment>/`.

---

Build order for the vector-only shoot-out. **Phases 0–8 are fully specified and ready to execute
now**; Phases 9–10 are deferred (image stage, variants). `📋` = port/adapt from the seamstress
**scratch clone** — flag Isaac before copying (the clone is in a session-temp dir and is not
permanent; those files must be pulled into the repo at these steps or they're lost).

## Phase 0 — Scaffolding & infra
- [ ] Rename package `template` → `quickdraw`; layout `src/quickdraw/{env,data,models,train,eval,control,logging,viz}`.
- [ ] `pyproject.toml`: add `torch, lightning, wandb, lerobot, torchcodec, imageio[ffmpeg]`; `uv lock`.
- [ ] `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.env.template`, README quickstart (infrastructure.md).
- [ ] verify: `docker compose run --rm app python -c "import torch;print(torch.cuda.is_available())"` → True.

## Phase 1 — Config (Hydra groups)
- [ ] `conf/config.yaml` + groups `model/ data/ optim/ trainer/ env/ logging/ eval/ control/` (training.md).
- [ ] every doc knob surfaced; CLI overrides work.
- [ ] verify: `train --cfg job` prints the fully-resolved config.

## Phase 2 — Environment (new code; no port)
- [ ] `TorusEnv` (torch): state `(θ,φ,θ̇,φ̇)`, semi-implicit Euler, OU actions, batched `B` envs,
      per-episode `torch.Generator` seed; nonzero initial speed.
- [ ] metric fns `signed_dist, n̂, manifold_distance_error, pointwise_error, tangent_velocity_error,
      phase_drift` — the single source of truth imported everywhere.
- [ ] 16 named targets (4 rings × 4 compass).
- [ ] verify: true rollouts have `manifold_distance_error ≈ 0`; seed reproduces an episode.

## Phase 3 — Data (lerobot)
- [ ] generation: `TorusEnv` → `LeRobotDataset.add_frame/save_episode`; splits
      `train, val, eval_ind, eval_ood_{visual,geometric,dynamics}`.
- [ ] train-only normalization stats, applied to all splits (override lerobot per-dataset stats).
- [ ] windowed loader (`delta_timestamps`, P=32,F=32) + full-trajectory eval loader.
- [ ] verify: seed regenerates identical data; train stats ≈ N(0,1); OOD shows shift.

## Phase 4 — Base model  📋
- [ ] stream encoders (`obs_vector` MLP, `action` MLP) + `TokenStreamFuser` 📋 (≈verbatim).
- [ ] transformer block: port `blocks.py`/`attention.py`/`positional.py` 📋, adapt → **pre-norm + causal + RoPE**.
- [ ] FlexAttention causal + sliding-window (`W`) `mask_mod`.
- [ ] delta head; shared `imagine` rollout fn.
- [ ] verify: shapes correct; one-step overfit on a tiny batch.

## Phase 5 — Training (Lightning + speed)
- [ ] `LightningModule`: `p_tf` rollout, truncated BPTT, MSE on delta.
- [ ] `Trainer`: bf16-mixed, `check_val_every_n_epoch=1`, grad_clip, `ModelCheckpoint(save_top_k)`.
- [ ] speed: `torch.compile(max-autotune)`, TF32, fused `AdamW`.
- [ ] verify: 1 epoch trains → validates; top-k checkpoints saved.

## Phase 6 — Logging  📋
- [ ] run folder + timestamp (`make_log_dir`) 📋; `config.resolved.yaml`; `metrics.csv` disk mirror.
- [ ] torus plotting helper (matplotlib 3D; shared by static plot + video frames).
- [ ] `LoggingCallback`: `train/ val/ diag/ data/` families + the rollout report.
- [ ] verify: W&B shows all families; disk mirror matches W&B.

## Phase 7 — Open-loop eval
- [ ] eval callback: full-trajectory rollouts on `ind` + 3 OOD every K epochs.
- [ ] `error_vs_step` plots, trajectory plots/videos, summary scalars (`@500/1000/2000`, `auc`).
- [ ] verify: a stay-put baseline gives a monotonically rising `manifold_distance_error`.

## Phase 8 — Control eval (MPPI)  📋
- [ ] MPPI controller 📋 (from `run_control.py`) around `imagine`: sample → roll → reward
      (`−‖p̂−target‖ − β·w(d)·‖ṗ̂‖`) → softmax weights → execute on `TorusEnv`.
- [ ] 16 targets, batched (`num_samples × targets`); report Hz, time-to-completion, success.
- [ ] target atlas (`GridSpec(4,3)`) + per-target control videos (logging.md `eval/control/`).
- [ ] verify: reaches several targets; Hz reported; videos logged.

## Phase 9 — Image stage (later)
- [ ] egocentric renderer (Darboux cam, rainbow texture, nvdiffrast) → MP4 per episode.
- [ ] image encoder 📋 (port `image_tokenizer_vit.py`) → 1 fused token; torchcodec decode in loader.
- [ ] `obs_image_video` logging.
- [ ] verify: decoded frame == render; image flows through the unchanged loader.

## Phase 10 — Shoot-out / variants (later)
- [ ] lock base results across InD / OOD / control.
- [ ] add model variants (`models.md`) on the same harness; W&B groups for cross-run comparison.

## Seamstress ports (📋) — from the scratch clone, in order of need
| File | Used at | Treatment |
|---|---|---|
| `token_stream_fuser.py` | Phase 4 | copy ≈verbatim |
| `blocks.py`, `attention.py`, `positional.py` | Phase 4 | port + adapt (pre-norm, causal, FlexAttention) |
| `custom_logging.py` (run-folder, timestamp) | Phase 6 | port helpers |
| MPPI from `run_control.py` | Phase 8 | port + adapt to `imagine` |
| `image_tokenizer_vit.py` | Phase 9 | port (later) |
