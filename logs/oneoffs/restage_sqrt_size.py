"""Transform the cube 'size' channel in the staged parquets from area/TYP -> sqrt(area/TYP) (linear apparent
extent ~ 1/depth; tames the heavy quadratic tail). size channel = feat index 2 of each (cube,cam) block of 4.
Recompute observation.objects.cube TRAIN stats afterwards. Cheap: operates on staged arrays, no re-extraction."""
import glob, json, numpy as np, pyarrow as pa, pyarrow.parquet as pq
SIZE_IDX=[(j*2+ci)*4+2 for j in range(4) for ci in range(2)]   # [2,6,10,14,18,22,26,30]
train=[]
for sp in ["train","val","eval_purple_play","eval_purple_stack"]:
    p=f"/app/logs/_hf_stage/{sp}/data/chunk-000/file-000.parquet"; t=pq.read_table(p)
    cube=np.array(t.column("observation.objects.cube").to_pylist(),dtype=np.float32)  # (N,32)
    cube[:,SIZE_IDX]=np.sqrt(np.clip(cube[:,SIZE_IDX],0,None))                        # area/TYP -> sqrt
    cols={n:t.column(n) for n in t.column_names if n!="observation.objects.cube"}
    nt=pa.table(cols).append_column("observation.objects.cube",
        pa.array([r.tolist() for r in cube],type=pa.list_(pa.float32())))
    pq.write_table(nt,p)
    if sp=="train": train.append(cube)
    print(f"{sp}: size->sqrt done, cube max now {cube.max():.2f}")
tc=np.concatenate(train,0)
json.dump({"observation.objects.cube_mean":tc.mean(0).tolist(),
           "observation.objects.cube_std":(tc.std(0)+1e-6).tolist(),"train_frames":int(len(tc))},
          open("/app/logs/_hf_stage/_cube_stats.json","w"))
print(f"recomputed train stats over {len(tc)} frames; cube range now [{tc.min():.2f},{tc.max():.2f}]")
