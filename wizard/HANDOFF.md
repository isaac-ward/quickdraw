# Handoff — live state as of 2026-09-07

Read this, then the record for whichever dataset you are working on. **This file is the only thing that
goes stale**; everything durable lives in `wizard/records/<dataset>.md`, `conf/`, and the memory dir.
Delete or rewrite it when the state below stops being true.

Per-dataset findings and queues:
`wizard/records/starling.md` (drone flight) · `wizard/records/robocasa-scene4-4h.md` (robot kitchen) ·
`wizard/records/owm.md` (torus — a DUMMY sanity dataset, not a research target).

**The objective, everywhere: raw `eval_ood_horizon/open_loop/<head>/lpips/@+128`.** Not the AE floor, not
`latent_cos`, not any derived penalty. `starling.md` §1.5 has the measurement showing `latent_cos` moved
30% while the objective did not follow, so it is not even a proxy.

---

## Running right now

| run | GPU | recipe | data | state |
|---|---|---|---|---|
| `starling_heavy` | 0 | `vl128_starling_heavy` (flow_arch_depth 4) | `starling` | ep8, best @+128 **0.3092** |
| `starling2_heavy` | 1 | `vl128_starling_heavy` | `starling2` | ep7, best @+128 **0.3672** |

Baselines they are being judged against (finished, killed at plateau):
`starling_vl128` best **0.2754** (ep7) · `starling2_vl128` best **0.3505** (ep11).

**The depth-4 arm is losing.** Better than depth-2 at only 2/7 and 2/6 matched epochs at @+128. It moves
`latent_cos` decisively and does not convert. Candidate to replace with the queue's item 1.

Kill by PID, targeted, NEVER a blanket `pkill` — another agent session shares this container and both
GPUs. Container processes run as root, so a host-side kill silently fails:
`docker compose exec app bash -lc "kill -TERM <pid>"`.

## Published models

| repo | from | epoch | @+128 |
|---|---|---|---|
| `isaac-ronald-ward/quickdraw-wm-robocasa-vl128` | single-cam robocasa | 11 | 0.13697 |
| `isaac-ronald-ward/quickdraw-wm-robocasa-vl128-2cam` | `twocam_full` | 29 | **0.1161** |
| `isaac-ronald-ward/quickdraw-wm-torus-vl128` | `torus_vl128c` | 17 | 0.0958 |

`push_model` now picks the OBJECTIVE's epoch from the run's own `metrics.jsonl` (not `best.ckpt`, which
tracks a proxy), and embeds the rollout video per image head on the card. `save_top_k=-1` everywhere, so
no epoch is ever pruned — that regression cost us `torus_vl128c` ep18, the best epoch it ever had.

## Environment state (this container)

The training venv `/app/.venv` now has robocasa live: `numpy 2.2.5` (robocasa hard-asserts EXACTLY this),
`mujoco 3.3.1` (also exact), `robosuite 1.5.2` **editable from `/caches/sim/robosuite`** — the PyPI wheel
raises `unexpected keyword argument 'load_model_on_init'` — and `robocasa 1.0.1` editable from
`/caches/sim/robocasa`. `opencv-python-headless`, not `opencv-python` (the latter wants GTK libs this
container lacks). torch 2.10.0+cu128, umap 0.5.12, sklearn 1.9.0 all verified intact afterwards.

**NEVER `uv sync`** — it strips umap/sklearn. `uv pip install` is fine.

## Untested / open work

1. **`eval_control` on robocasa has never been run end-to-end.** The env (`environments=robocasa`) passes
   18/18 in `smoke/robocasa_env.py`, but the MPPI control path against it is unexercised. Note there is
   NO ORACLE on robocasa (see `examples/robocasa.py` — forking 1024 mujoco kitchens is infeasible), so any
   control number is reward-only and a poor score cannot be attributed to the world model.
2. `docs/using_pretrained_models.ipynb` has never been re-executed against the live Hub repos.
3. No committed roundtrip smoke for `load_pretrained` (verified live only).
4. `preserve_ol.py` / `push_torus_watch.py` in `/tmp` are stale (pinned to dead runs) and were written by
   another agent session. `save_top_k=-1` makes `preserve_ol` unnecessary.

## LOCKED / do not change

`accumulate_grad_batches=1` · `data.window_stride=1` · no pretrained autoencoder pieces (LPIPS as a LOSS
network is fine) · `action_squash=none` (symlog off) · no `psnr_frozen` · 96px is robocasa's working
resolution, not 128 · `latent_loss_weight=10` (bracketed on both sides, §22/§23) · never kill a run
without asking.
