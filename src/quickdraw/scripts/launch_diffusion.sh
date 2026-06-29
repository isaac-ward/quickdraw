#!/usr/bin/env bash
# Diffusion mini-shootout: the latent flow-matching world model in TWO variants — plain flow + shortcut
# (design/models/diffusion.md). Backbone/data/BPTT match the variation campaign (recon LSAR) so the
# diffusion class is comparable to the rest of the shoot-out. In-loop eval = the shared OOD + control
# routines PLUS the diffusion flow-field viz (diffusion/{streamline,quiver} + flow_endpoint_error).
# ============================================================================================
# !!! DO NOT disable compile / DO NOT set TORCHDYNAMO_DISABLE=1 — FlexAttention REQUIRES torch.compile
# to build its kernel. The diffusion forward (incl. the flow sampler) compiles fine. See launch_shootout.sh.
# ============================================================================================
#
# RUN SUMMARY IS NOT HARDCODED. The operator supplies the note FRESH each launch via env vars (a baked-in
# note goes stale and train.py rejects duplicates). Plain words + periods only (Hydra rejects ; , - : = ).
# Shared fields: RS_PROBLEM RS_TRIED RS_DETAIL RS_RATIONALE. Per-run `trying`: RS_TRYING_flow RS_TRYING_shortcut.
#
# Usage: RS_PROBLEM=.. RS_TRIED=.. RS_DETAIL=.. RS_RATIONALE=.. RS_TRYING_flow=.. RS_TRYING_shortcut=.. \
#          bash src/quickdraw/scripts/launch_diffusion.sh
# NOTE: needs free GPU capacity. If the variation campaign is still running (both GPUs ~full), either
# wait, free a GPU, or drop data.batch below in COMMON.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1   # -> repo root

DATA="logs/data_generation_2026_06_27_04_49_59_regen_dyn_v8"

: "${RS_PROBLEM:?author + export the run_summary fresh, not hardcoded; missing RS_PROBLEM}"
: "${RS_TRIED:?missing RS_TRIED}"; : "${RS_DETAIL:?missing RS_DETAIL}"; : "${RS_RATIONALE:?missing RS_RATIONALE}"

# fixed across both: diffusion model, full backbone, batch 256, v8 data, truncated BPTT (16) — the
# variation-campaign settings, so diffusion is comparable. diffusion_field viz on (the headline artifact).
COMMON=( model=diffusion data.root="$DATA" data.batch=256 model.detach_every=16
         eval.during_train.evals.diffusion_field=true )
RS=( run_summary.problem="$RS_PROBLEM" run_summary.tried="$RS_TRIED"
     run_summary.trying_detail="$RS_DETAIL" run_summary.rationale="$RS_RATIONALE" )

echo "[diffusion] killing any existing so_diff_ runs..."
docker compose exec -T app pkill -9 -f "experiment=so_diff_" 2>/dev/null || true
sleep 4

launch () {  # $1=gpu  $2=experiment-name  $3=trying-env-var-name  $4..=model overrides
  local gpu="$1" name="$2" tvar="$3"; shift 3
  local trying="${!tvar:?missing $tvar the run-specific trying note}"
  echo "[diffusion] launching $name on GPU $gpu"
  docker compose exec -T -d -e CUDA_VISIBLE_DEVICES="$gpu" -e TORCHINDUCTOR_COMPILE_THREADS=1 \
    -e TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_$name" -e TRITON_CACHE_DIR="/tmp/triton_$name" app \
    uv run python -m quickdraw.train "${COMMON[@]}" "$@" experiment="$name" "${RS[@]}" \
      run_summary.trying="$trying"
}

# plain flow (sampling_steps 6) + shortcut (step-size cond + self-consistency -> K=1 sampling, in-rollout-cheap)
launch 0 so_diff_flow     RS_TRYING_flow     model.diffusion.shortcut=false model.diffusion.sampling_steps=6
sleep 45
launch 1 so_diff_shortcut RS_TRYING_shortcut model.diffusion.shortcut=true  model.diffusion.sampling_steps=1

echo "[diffusion] launched 2 (plain flow on GPU0, shortcut on GPU1). Monitor: logs/train_*so_diff_*/progress.log or wandb."
