#!/bin/bash
# v7 -> previews -> pack -> HF -> both training arms. Run from an immutable copy: editing a shell
# script while bash is executing it shifts the byte offsets under the interpreter and corrupts the
# run (seen tonight -- a syntax error mid-flight killed a generation pass).
set -u
cd /app
V=/app/logs/mbari/data/v7
L=/app/logs/mbari

echo "=== previews, one per episode  $(date -u +%H:%M) ==="
bash /app/src/quickdraw/sim/mbari/gen_previews.sh $V 16 > $L/data/previews_v7.log 2>&1
tail -1 $L/data/previews_v7.log

echo "=== packing (fpv keys; all four cameras ship as clips)  $(date -u +%H:%M) ==="
.venv/bin/python -u -m quickdraw.sim.mbari.package --root $V --name mbari-tank --cams fpv \
  > $L/data/package_v7.log 2>&1
R=$(grep "^run_dir:" $L/data/package_v7.log | tail -1 | awk '{print $2}')
[ -z "$R" ] && { echo "PACKAGING FAILED"; tail -20 $L/data/package_v7.log; exit 1; }
echo "  run_dir = $R"
echo "$R" > $L/data/run_dir.txt
for s in train val ood_thruster ood_fish ood_buoy; do
  printf "  %-14s %s eps\n" $s \
    "$(python3 -c "import json;print(json.load(open('$R/$s/meta/info.json'))['total_episodes'])" 2>/dev/null)"
done

echo "=== shipping previews + PROCESSING.md  $(date -u +%H:%M) ==="
cp $L/data/PROCESSING.md $R/PROCESSING.md 2>/dev/null
mkdir -p $R/media/previews && cp -r $L/data/previews/* $R/media/previews/ 2>/dev/null
echo "  $(find $R/media/previews -name '*.mp4' | wc -l) preview mp4"

echo "=== push to hugging face  $(date -u +%H:%M) ==="
.venv/bin/python -u -m quickdraw.push_to_hub data.root=$R +hub.name=mbari-tank > $L/data/push_v7.log 2>&1
tail -2 $L/data/push_v7.log

sed -i "s|^R=.*|R=$R|" /app/src/quickdraw/sim/mbari/train_mbari.sh
sed -i "s|^R=.*|R=$R|" /app/src/quickdraw/sim/mbari/train_mbari_2h.sh
echo "=== run A: 6.00 h on GPU 0  $(date -u +%H:%M) ==="
nohup bash /app/src/quickdraw/sim/mbari/train_mbari.sh > $L/worldmodel/train_6h.log 2>&1 &
sleep 150
echo "=== run B: 2.00 h on GPU 1  $(date -u +%H:%M) ==="
nohup bash /app/src/quickdraw/sim/mbari/train_mbari_2h.sh > $L/worldmodel/train_2h.log 2>&1 &
wait
echo "=== both arms exited  $(date -u +%H:%M) ==="
