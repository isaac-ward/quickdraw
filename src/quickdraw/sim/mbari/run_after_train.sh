#!/bin/bash
# Everything that needs a trained checkpoint. Takes the training run directory as $1.
#   1. open-loop + 16-step-ahead predictions   -> logs/mbari/worldmodel/
#   2. uncertainty scoring + OOD classification -> logs/mbari/uncertainty/
#   3. pixel-level uncertainty videos           -> logs/mbari/uncertainty/pixel/
#   4. control benchmark, learned arm included  -> logs/mbari/control/
set -u
cd /app
R=${1:?usage: run_after_train.sh <train_run_dir> [tag] [gpu]}
TAG=${2:-$(basename $R)}
GPU=${3:-0}
L=/app/logs/mbari
mkdir -p $L/worldmodel $L/uncertainty/$TAG $L/control/$TAG

# ITEM 13 IS ALREADY DONE BY THE TRAINING RUN. eval.during_train.evals.ood_horizon writes the
# open-loop AND both closed-loop modes (1-step and 16-step-ahead) every 2 epochs, into the run's
# own directory. Stage 1 below re-runs it standalone on the FINAL checkpoint at the full horizon,
# which the during-train copy does not do; if you only want the curves, they are already there.
#
# TAG and GPU exist so the two ablation arms can run this side by side without overwriting each
# other -- outputs land under uncertainty/<tag>/ and control/<tag>/.

echo "=== 1. open-loop + 16-step-ahead  $(date -u +%H:%M) ==="
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u -m quickdraw.eval_ood_horizon \
  checkpoint=$R model=mbari_fpv data=mbari environments=recorded \
  environments.obs_dim=13 environments.action_dim=6 environments.dt=0.0333333 \
  data.subsample=1 data.action_aggregate=concat \
  eval.horizon=900 "eval.closed_loop_steps=[16]" \
  experiment=mbari_openloop_$TAG > $L/worldmodel/openloop_$TAG.log 2>&1
grep -iE "lpips|psnr|wrote|run dir" $L/worldmodel/openloop_$TAG.log | tail -12

echo "=== 2. uncertainty + OOD classification  $(date -u +%H:%M) ==="
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u -c "
from quickdraw.sim.mbari.run_uncertainty import run
import json
rep, _ = run('$R', out='$L/uncertainty/$TAG', log=lambda m: print(m, flush=True))
print(json.dumps(rep['three_way'], indent=1, default=str))
" > $L/uncertainty/$TAG/uncertainty.log 2>&1
tail -25 $L/uncertainty/$TAG/uncertainty.log

echo "=== 3. pixel uncertainty videos  $(date -u +%H:%M) ==="
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u -m quickdraw.sim.mbari.run_pixel_uncertainty \
  --ckpt $R --out $L/uncertainty/$TAG/pixel > $L/uncertainty/$TAG/pixel.log 2>&1
tail -6 $L/uncertainty/$TAG/pixel.log

echo "=== 4. control benchmark: true vs analytic vs learned  $(date -u +%H:%M) ==="
CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl .venv/bin/python -u -m quickdraw.sim.mbari.run_control \
  --ckpt $R --out $L/control/$TAG > $L/control/$TAG/benchmark.log 2>&1
grep "^  ==" $L/control/$TAG/benchmark.log
echo "=== all done  $(date -u +%H:%M) ==="
