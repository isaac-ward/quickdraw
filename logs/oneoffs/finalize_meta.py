"""Finalize the staged cube dataset's META + DOCS (no upload). Per split: rewrite meta/info.json
(observation_vector -> observation.state; add observation.objects.cube [32]) and meta/stats.json
(rename + add cube min/max/mean/std/count computed on that split). Top-level: normalization_stats.json
(rename + add cube from train) and a documented block appended to PROCESSING.md + README.md."""
import glob, json, os, shutil, numpy as np, pyarrow.parquet as pq
SNAP=glob.glob("/caches/hf/hub/datasets--isaac-ronald-ward--block-stack/snapshots/*/")[0]
STAGE="/app/logs/_hf_stage"
SPLITS=["train","val","eval_purple_play","eval_purple_stack"]
CUBE_FEAT={"dtype":"float32","shape":[32],"names":None}
def rename_add(features):
    out={}
    for k,v in features.items():
        out["observation.state" if k=="observation_vector" else k]=v
    out["observation.objects.cube"]=dict(CUBE_FEAT)
    return out
for sp in SPLITS:
    md=os.path.join(STAGE,sp,"meta"); os.makedirs(md,exist_ok=True)
    src=os.path.join(SNAP,sp,"meta")
    # copy unchanged meta bits
    for f in ["tasks.parquet"]:
        if os.path.exists(os.path.join(src,f)): shutil.copy2(os.path.join(src,f),os.path.join(md,f))
    if os.path.isdir(os.path.join(src,"episodes")): shutil.copytree(os.path.join(src,"episodes"),os.path.join(md,"episodes"),dirs_exist_ok=True)
    # info.json
    info=json.load(open(os.path.join(src,"info.json"))); info["features"]=rename_add(info["features"])
    json.dump(info,open(os.path.join(md,"info.json"),"w"),indent=4)
    # cube stats from the staged parquet
    cube=np.array(pq.read_table(os.path.join(STAGE,sp,"data","chunk-000","file-000.parquet"))
                  .column("observation.objects.cube").to_pylist(),dtype=np.float64)
    cstat={"min":cube.min(0).tolist(),"max":cube.max(0).tolist(),"mean":cube.mean(0).tolist(),
           "std":cube.std(0).tolist(),"count":[int(len(cube))]}
    # stats.json rename + add
    st=json.load(open(os.path.join(src,"stats.json")))
    if "observation_vector" in st: st["observation.state"]=st.pop("observation_vector")
    st["observation.objects.cube"]=cstat
    json.dump(st,open(os.path.join(md,"stats.json"),"w"),indent=4)
    print(f"{sp}: info.json + stats.json finalized")
# top-level normalization_stats.json
ns=json.load(open(os.path.join(SNAP,"normalization_stats.json")))
if "observation_vector" in ns: ns["observation.state"]=ns.pop("observation_vector")
cs=json.load(open(os.path.join(STAGE,"_cube_stats.json")))
ns["observation.objects.cube"]={"mean":cs["observation.objects.cube_mean"],"std":cs["observation.objects.cube_std"]}
json.dump(ns,open(os.path.join(STAGE,"normalization_stats.json"),"w"),indent=2)
print("normalization_stats.json finalized (observation.state + observation.objects.cube)")
print("DONE meta")
