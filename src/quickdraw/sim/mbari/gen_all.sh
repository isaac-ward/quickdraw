#!/bin/bash
# Full MBARI dataset: 240 x 30 s nominal + 3 x 50 OOD trajectories, sharded across independent
# processes. Shards alternate GPUs. Each shard skips episodes already on disk, so re-running the
# script resumes rather than redoing.
cd /app
N=24
OUT=/app/logs/mbari/data/v6
mkdir -p /app/logs/mbari/data/shardlogs
for k in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$((k % 2)) MUJOCO_GL=egl \
  .venv/bin/python -u -m quickdraw.sim.mbari.generate \
      --out $OUT --shard $k --shards $N \
      --nominal 720 --steps 900 --ood-per 50 --ood-steps 360 \
      > /app/logs/mbari/data/shardlogs/shard_$k.log 2>&1 &
done
wait
echo "ALL SHARDS DONE"
