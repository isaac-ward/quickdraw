#!/bin/bash
# A four-camera preview mp4 for EVERY episode, sharded across processes. ~3.3 h serial, ~20 min
# at 12 shards. Re-running resumes: episodes whose mp4 already exists are skipped.
cd /app
N=${2:-12}
V=${1:-/app/logs/mbari/data/v6}
OUT=/app/logs/mbari/data/previews
mkdir -p /app/logs/mbari/data/prevlogs
for k in $(seq 0 $((N-1))); do
  .venv/bin/python -u -m quickdraw.sim.mbari.previews \
      --root $V --out $OUT --shard $k --shards $N \
      > /app/logs/mbari/data/prevlogs/prev_$k.log 2>&1 &
done
wait
echo "ALL PREVIEWS DONE  $(ls $OUT/*/*.mp4 2>/dev/null | wc -l) files, $(du -sh $OUT | cut -f1)"
