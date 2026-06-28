#!/usr/bin/env bash
# Launch the 6-way world-model shoot-out: DSAR + LSAR{naked, recon, ema, sigreg, vicreg}.
# 3 per GPU, full BPTT (detach_every=0), F=64, batch 256 (probed to fit 3/GPU), in-loop eval
# (ood_horizon + control, per conf/eval defaults) every 10 epochs.
# ============================================================================================
# !!! DO NOT disable compile here. DO NOT set TORCHDYNAMO_DISABLE=1. !!!
# Why: the transformer uses FlexAttention, which REQUIRES torch.compile to build its attention
# kernel (it self-compiles even in the eager rollout). Disabling compile globally forces a slow
# eager-attention fallback: ~260% CPU, GPU idle, no training progress. The one-time compile at
# startup (the "slow start") is the NECESSARY cost of FlexAttention, not a waste. Leave compile on.
# (If startup is too slow, the lever is the COMPILE MODE in train.py — e.g. drop max-autotune to the
#  default mode — NOT disabling compile. That is a train.py change, coordinate before touching it.)
# ============================================================================================
#
# Usage:  bash src/quickdraw/scripts/launch_shootout.sh
# Kills any existing so_* runs first, then launches all 6 detached inside the container.
# Monitor: logs/train_*so_*/progress.log  or wandb.

set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1   # -> repo root (this script lives in src/quickdraw/scripts/)

DATA="logs/data_generation_2026_06_27_04_49_59_regen_dyn_v8"  # v8: base gamma0.3 (calmer), slower OOD (g0.5/m2.0), only ood_horizon long, eval@[20,40]

# 5-point run summary (plain words + periods only; Hydra's override grammar rejects ; , - : = etc.)
# REWRITE THESE EVERY CAMPAIGN: train.py rejects a run_summary identical to any previous run's. The note
# must speak to THIS launch's current hypothesis + what changed since last time. The per-variant `trying`
# (built in launch() with the run name) also tags each run so the 6 are unique. No stale copy-pastes.
P="Long horizon rollouts drift off the torus. We compare data space versus latent space world models and which anti collapse mechanism best holds the manifold."
T="DSAR refactor verified bit identical. LSAR with five collapse mechanisms implemented and smoke tested. Memory probed so batch 256 fits three per GPU at full BPTT."
Y="Six way shoot out. DSAR plus LSAR naked recon ema sigreg vicreg. Full BPTT F64 batch 256. Three per GPU on the regen dataset."
D="Shared backbone d96 dz16 window64. In loop eval ood horizon and control every ten epochs. Standalone ood splits at best checkpoints later."
R="Real training reaches full rollout where collapse appears. Collapse panel and long horizon eval show which mechanisms hold the manifold."

COMMON=( data.root="$DATA" data.batch=256 model.detach_every=0 )  # eval at conf/eval at_epochs=[20,40]
# `trying` is set per-variant in launch() (appends the run name) so each of the 6 notes is unique.
RS=( run_summary.problem="$P" run_summary.tried="$T" run_summary.trying_detail="$D" run_summary.rationale="$R" )

echo "[shootout] killing any existing so_ runs..."
docker compose exec -T app pkill -9 -f "experiment=so_" 2>/dev/null || true
sleep 4

launch () {  # $1=gpu  $2=experiment-name  $3..=model overrides
  local gpu="$1" name="$2"; shift 2
  echo "[shootout] launching $name on GPU $gpu"
  # compile stays ON (FlexAttention needs it — see banner). PER-RUN compile caches: 6 concurrent runs
  # sharing the default inductor/triton cache contend on it and stall for ~10 min; isolated caches let
  # each compile independently (single run with its own cache compiles + trains in ~1 min / 18s-epoch).
  # TORCHINDUCTOR_COMPILE_THREADS=1: compile inductor kernels in-process. The default async SubprocPool
  # forks ~32 compile workers from a parent holding a live CUDA context + big thread pool -> fork-after-
  # CUDA deadlock that wedges the run at 0% GPU right after the FlexAttention compile (it never reaches
  # epoch 0). train.py also sets this in code; the env is belt-and-suspenders in case inductor inits early.
  docker compose exec -T -d -e CUDA_VISIBLE_DEVICES="$gpu" -e TORCHINDUCTOR_COMPILE_THREADS=1 \
    -e TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_$name" -e TRITON_CACHE_DIR="/tmp/triton_$name" app \
    uv run python -m quickdraw.train "$@" "${COMMON[@]}" experiment="$name" "${RS[@]}" \
      run_summary.trying="$Y This run is the $name variant."
}

# Light stagger (interleaving GPUs). The FlexAttention per-shape recompile thrash that used to stall
# startup for ~10 min is FIXED (fixed-window rollout in sequence.py + cache_size_limit in train.py), so
# this is no longer about compile contention. It just desyncs the first in-loop eval (epoch 0:
# ood_horizon + control + VTK video render, which is CPU-heavy ~minutes) so 6 renders don't all peak
# together. Training itself is fast now (~18s/epoch).
STAGGER=45
launch 0 so_dsar        model=base;                                                sleep $STAGGER
launch 1 so_lsar_ema    model=latent_space_autoregressor +collapse=ema;            sleep $STAGGER
launch 0 so_lsar_naked  model=latent_space_autoregressor +collapse=naked;          sleep $STAGGER
launch 1 so_lsar_sigreg model=latent_space_autoregressor +collapse=sigreg;         sleep $STAGGER
launch 0 so_lsar_recon  model=latent_space_autoregressor +collapse=reconstruction; sleep $STAGGER
launch 1 so_lsar_vicreg model=latent_space_autoregressor +collapse=vicreg

echo "[shootout] launched 6 (3 per GPU, staggered). Monitor: logs/train_*so_*/progress.log or wandb."
