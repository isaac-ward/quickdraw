#!/bin/bash
# The full dataset, generated to a STEP target rather than an episode count.
#
# Episodes now end on contact, so a 900-step job does not reliably yield 900 steps -- the random
# class keeps about half. Asking for a fixed number of episodes therefore delivers less data than
# intended, and unevenly: the planned circles rarely truncate while the random class usually does,
# so an episode-balanced set is no longer the STEP-balanced one the spec asked for. So: one base
# pass, then top-up rounds until every behaviour class in every split has reached its step target.
cd /app
N=24
OUT=${1:-/app/logs/mbari/data/v7}
PER_CLASS=${2:-216000}        # 2.00 h per class -> 6.00 h nominal
OOD_PER_CLASS=${3:-6000}      # 3.33 min per class -> 10 min per OOD set
mkdir -p /app/logs/mbari/data/shardlogs

run_shards () {   # $1 = extra args
  for k in $(seq 0 $((N-1))); do
    CUDA_VISIBLE_DEVICES=$((k % 2)) MUJOCO_GL=egl \
    .venv/bin/python -u -m quickdraw.sim.mbari.generate \
        --out $OUT --shard $k --shards $N --per-class $PER_CLASS --ood-per-class $OOD_PER_CLASS \
        $1 > /app/logs/mbari/data/shardlogs/shard_$k.log 2>&1 &
  done
  wait
}

echo "=== base pass $(date -u +%H:%M) ==="
run_shards "--nominal 720 --steps 900 --ood-per 50 --ood-steps 360"

for round in 1 2 3 4 5 6; do
  SHORT=$(.venv/bin/python - <<PY
from quickdraw.sim.mbari.generate import steps_by_class
from quickdraw.sim.mbari.trajectories import CLASSES
tot = 0
for sp, tgt in (("train", $PER_CLASS), ("ood_thruster", $OOD_PER_CLASS),
                ("ood_fish", $OOD_PER_CLASS), ("ood_buoy", $OOD_PER_CLASS)):
    g = steps_by_class("$OUT", sp)
    tot += sum(max(0, tgt - int(g.get(c, 0))) for c in CLASSES)
print(tot)
PY
)
  echo "=== round $round: $SHORT steps short  $(date -u +%H:%M) ==="
  [ "$SHORT" -le 2000 ] && { echo "TARGET MET"; break; }
  run_shards "--topup"
done

.venv/bin/python - <<PY
from quickdraw.sim.mbari.generate import steps_by_class
for sp in ("train", "ood_thruster", "ood_fish", "ood_buoy"):
    g = steps_by_class("$OUT", sp)
    t = sum(g.values())
    print(f"  {sp:13s} {dict(g)}  total {t} steps = {t/30/3600:.2f} h")
PY
echo "ALL DONE $(date -u +%H:%M)"
