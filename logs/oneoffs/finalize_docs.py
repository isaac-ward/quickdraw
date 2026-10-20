import glob, os, shutil
SNAP=glob.glob("/caches/hf/hub/datasets--isaac-ronald-ward--block-stack/snapshots/*/")[0]
STAGE="/app/logs/_hf_stage"
DOC="""

---

## Update 2026-09-24 — cube object-state features + `observation.state` rename

**`observation_vector` was renamed to `observation.state`** (lerobot-canonical; matches our other datasets).
Values are byte-identical to the old `observation_vector [17]`; only the field name changed.

**New field: `observation.objects.cube` — shape `[32]`, float32.** Per-frame cube object-state, extracted by
per-episode color segmentation (hue-peak palette discovery, so only colors actually present in a split are
tracked — R/G/B for the standard task, R/B/P for the `eval_purple_*` splits).

Layout: **4 cube slots (canonical order R, G, B, P) x 2 cameras (cam_scene=scene_right, cam_wrist=gripper_right_top)
x 4 features**, flattened as [(cube j, cam c) for j in RGBP for c in (scene, wrist)] -> [f0..f3]. Per (cube,camera):

| idx | feature | meaning |
|----|---------|---------|
| 0 | cx / W | centroid x / image width, in [0,1] |
| 1 | cy / H | centroid y / image height, in [0,1] |
| 2 | sqrt(area / TYP) | apparent linear extent (~ 1/depth proximity cue); TYP=120 px |
| 3 | soft_vis | clip(area/TYP, 0, 1); 0 = not currently observed |

**Occlusion:** if a cube is not detected but was seen earlier, its **position is held at last-seen** while
soft_vis (and size) go to **0** -> `[cx, cy, 0, 0]` marks a held/occluded position. A cube color **absent** from a
split (P in train/val; G in the purple splits) is **all-zeros** every frame. So soft_vis>0 = current;
soft_vis=0 with nonzero position = carried through occlusion; all-zero = absent.

Normalization stats for observation.objects.cube (normalization_stats.json + per-split meta/stats.json) are on
the **train** split. Cameras/action/indices and the four observation.images.* video streams are unchanged.
Extraction is deterministic classical CV (no learned detector, no labels).
"""
for fn in ["PROCESSING.md","README.md"]:
    shutil.copy2(os.path.join(SNAP,fn), os.path.join(STAGE,fn))
    with open(os.path.join(STAGE,fn),"a") as f: f.write(DOC)
    print(f"appended cube doc to {fn}")
print("DONE docs")
