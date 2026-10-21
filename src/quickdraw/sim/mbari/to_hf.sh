#!/bin/bash
# Wait for generation, then previews -> pack -> HF -> both training arms. Launched from an
# immutable copy (editing a running shell script shifts bash's byte offsets and corrupts it).
set -u
cd /app
V=/app/logs/mbari/data/v7
L=/app/logs/mbari

echo "=== waiting for generation  $(date -u +%H:%M) ==="
while pgrep -f "[s]im.mbari.generate" >/dev/null; do sleep 30; done
while pgrep -f "[g]en_all_run.sh" >/dev/null; do sleep 30; done
.venv/bin/python - <<PY
from quickdraw.sim.mbari.generate import steps_by_class
for sp in ("train", "ood_thruster", "ood_fish", "ood_buoy"):
    g = steps_by_class("$V", sp); t = sum(g.values())
    print(f"  {sp:13s} {dict(g)}  {t/30/3600:.2f} h")
PY

echo "=== previews (resumes; skips what exists)  $(date -u +%H:%M) ==="
bash /app/src/quickdraw/sim/mbari/gen_previews.sh $V 16 > $L/data/previews_v7.log 2>&1
echo "  $(find $L/data/previews -name '*.mp4' | wc -l) preview mp4"

echo "=== packing  $(date -u +%H:%M) ==="
.venv/bin/python -u -m quickdraw.sim.mbari.package --root $V --name mbari-tank --cams fpv \
  > $L/data/package_v7.log 2>&1
R=$(grep "^run_dir:" $L/data/package_v7.log | tail -1 | awk '{print $2}')
[ -z "$R" ] && { echo "PACKAGING FAILED"; tail -25 $L/data/package_v7.log; exit 1; }
echo "  run_dir = $R"
for s in train val ood_thruster ood_fish ood_buoy; do
  printf "  %-14s %s eps\n" $s \
    "$(python3 -c "import json;print(json.load(open('$R/$s/meta/info.json'))['total_episodes'])" 2>/dev/null)"
done

cp $L/data/PROCESSING.md $R/PROCESSING.md 2>/dev/null
mkdir -p $R/media/previews && cp -r $L/data/previews/* $R/media/previews/ 2>/dev/null
echo "=== PUSH TO HUGGING FACE  $(date -u +%H:%M) ==="
.venv/bin/python -u -m quickdraw.push_to_hub data.root=$R +hub.name=mbari-tank > $L/data/push_v7.log 2>&1
tail -2 $L/data/push_v7.log
echo "=== PUSH DONE  $(date -u +%H:%M) ==="

sed -i "s|^R=.*|R=$R|" /app/src/quickdraw/sim/mbari/train_mbari.sh
sed -i "s|^R=.*|R=$R|" /app/src/quickdraw/sim/mbari/train_mbari_2h.sh
echo "=== run A: 6 h on GPU 0  $(date -u +%H:%M) ==="
nohup bash /app/src/quickdraw/sim/mbari/train_mbari.sh > $L/worldmodel/train_6h.log 2>&1 &
sleep 150
echo "=== run B: 2 h on GPU 1  $(date -u +%H:%M) ==="
nohup bash /app/src/quickdraw/sim/mbari/train_mbari_2h.sh > $L/worldmodel/train_2h.log 2>&1 &
wait
