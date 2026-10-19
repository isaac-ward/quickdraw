#!/bin/bash
# Waits for the dataset rebuild, pushes it to HF with the 20% val split, then starts BOTH arms of
# the data-volume ablation on that one dataset. Run A (6.00 h) on GPU 0, run B (2.00 h, the same
# dataset capped to its first 240 train episodes) on GPU 1.
set -u
cd /app
R=/app/logs/recording_2026_10_19_00_30_50_mbari-tank
L=/app/logs/mbari

echo "=== waiting for the dataset rebuild  $(date -u +%H:%M) ==="
while pgrep -f "[m]bari.package" >/dev/null; do sleep 60; done
for s in train val ood_thruster ood_fish ood_buoy; do
  printf "  %-14s %s episodes\n" $s \
    "$(python3 -c "import json;print(json.load(open('$R/$s/meta/info.json'))['total_episodes'])" 2>/dev/null)"
done

echo "=== shipping the previews and PROCESSING.md into the dataset  $(date -u +%H:%M) ==="
cp $L/data/PROCESSING.md $R/PROCESSING.md 2>/dev/null
mkdir -p $R/media/previews && cp -r $L/data/previews/* $R/media/previews/ 2>/dev/null
echo "  $(find $R/media/previews -name '*.mp4' | wc -l) preview mp4, $(du -sh $R/media/previews | cut -f1)"

echo "=== push to hugging face  $(date -u +%H:%M) ==="
.venv/bin/python -u -m quickdraw.push_to_hub data.root=$R +hub.name=mbari-tank \
  > $L/data/push.log 2>&1
tail -2 $L/data/push.log

echo "=== run A: 6.00 h on GPU 0  $(date -u +%H:%M) ==="
nohup bash /app/src/quickdraw/sim/mbari/train_mbari.sh > $L/worldmodel/train_6h.log 2>&1 &
sleep 120
echo "=== run B: 2.00 h on GPU 1  $(date -u +%H:%M) ==="
nohup bash /app/src/quickdraw/sim/mbari/train_mbari_2h.sh > $L/worldmodel/train_2h.log 2>&1 &
wait
echo "=== both arms exited  $(date -u +%H:%M) ==="
