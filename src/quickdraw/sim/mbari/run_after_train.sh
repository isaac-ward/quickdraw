#!/bin/bash
# Everything that needs a trained checkpoint. Takes the training run directory as $1.
#   1. open-loop + 16-step-ahead predictions   -> logs/mbari/worldmodel/
#   2. uncertainty scoring + OOD classification -> logs/mbari/uncertainty/
#   3. pixel-level uncertainty videos           -> logs/mbari/uncertainty/pixel/
#   4. control benchmark, learned arm included  -> logs/mbari/control/
set -u
cd /app
R=${1:?usage: run_after_train.sh <train_run_dir>}
L=/app/logs/mbari
mkdir -p $L/worldmodel $L/uncertainty $L/control

echo "=== 1. open-loop + 16-step-ahead  $(date -u +%H:%M) ==="
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m quickdraw.eval_ood_horizon \
  checkpoint=$R model=mbari_fpv data=mbari environments=recorded \
  environments.obs_dim=13 environments.action_dim=6 environments.dt=0.0333333 \
  data.subsample=1 data.action_aggregate=concat \
  eval.horizon=900 "eval.closed_loop_steps=[16]" \
  experiment=mbari_openloop > $L/worldmodel/openloop.log 2>&1
grep -iE "lpips|psnr|wrote|run dir" $L/worldmodel/openloop.log | tail -12

echo "=== 2. uncertainty + OOD classification  $(date -u +%H:%M) ==="
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -c "
from quickdraw.sim.mbari.run_uncertainty import run
import json
rep, _ = run('$R', out='$L/uncertainty', log=lambda m: print(m, flush=True))
print(json.dumps(rep['three_way'], indent=1, default=str))
" > $L/uncertainty/uncertainty.log 2>&1
tail -25 $L/uncertainty/uncertainty.log

echo "=== 3. pixel uncertainty videos  $(date -u +%H:%M) ==="
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m quickdraw.sim.mbari.run_pixel_uncertainty \
  --ckpt $R --out $L/uncertainty/pixel > $L/uncertainty/pixel.log 2>&1
tail -6 $L/uncertainty/pixel.log

echo "=== 4. control benchmark: true vs analytic vs learned  $(date -u +%H:%M) ==="
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl .venv/bin/python -u -m quickdraw.sim.mbari.run_control \
  --ckpt $R --out $L/control > $L/control/benchmark.log 2>&1
grep "^  ==" $L/control/benchmark.log
echo "=== all done  $(date -u +%H:%M) ==="
