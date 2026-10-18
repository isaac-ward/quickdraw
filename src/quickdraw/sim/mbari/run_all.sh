#!/bin/bash
# Everything downstream of generation, chained so it runs unattended:
#   wait for the shards -> preview videos -> pack all four cameras -> push to HF -> train.
# Each stage logs under /app/logs/mbari/, and each is skippable by hand if one needs redoing.
set -u
cd /app
V=/app/logs/mbari/data/v5
L=/app/logs/mbari

echo "=== waiting for generation shards  $(date -u +%H:%M) ==="
while pgrep -f "sim.mbari.generate" >/dev/null; do sleep 60; done
for s in train ood_thruster ood_fish ood_buoy; do
  printf "  %-14s %4s episodes\n" $s $(ls $V/$s/*.npz 2>/dev/null | wc -l)
done
du -sh $V

echo "=== preview videos  $(date -u +%H:%M) ==="
.venv/bin/python -u -m quickdraw.sim.mbari.previews --root $V --out $L/data/previews --n-per 2 \
  > $L/data/previews.log 2>&1
tail -6 $L/data/previews.log

echo "=== packaging all four cameras  $(date -u +%H:%M) ==="
.venv/bin/python -u -m quickdraw.sim.mbari.package --root $V --name mbari-tank \
  > $L/data/package.log 2>&1
R=$(grep "^run_dir:" $L/data/package.log | tail -1 | awk '{print $2}')
echo "  run_dir = $R"
[ -z "$R" ] && { echo "PACKAGING FAILED"; tail -20 $L/data/package.log; exit 1; }
cp $L/data/PROCESSING.md $R/PROCESSING.md 2>/dev/null
echo "$R" > $L/data/run_dir.txt

echo "=== push to hugging face  $(date -u +%H:%M) ==="
.venv/bin/python -u -m quickdraw.push_to_hub data.root=$R +hub.name=mbari-tank \
  > $L/data/push.log 2>&1
tail -3 $L/data/push.log

echo "=== training  $(date -u +%H:%M) ==="
sed -i "s|^R=.*|R=$R|" /app/src/quickdraw/sim/mbari/train_mbari.sh
bash /app/src/quickdraw/sim/mbari/train_mbari.sh > $L/worldmodel/train_v2.log 2>&1
echo "=== pipeline done  $(date -u +%H:%M) ==="
