import numpy as np, pyarrow.parquet as pq
n=pq.read_table("/app/logs/_hf_stage/train/data/chunk-000/file-000.parquet")
cube=np.array(n.column("observation.objects.cube").to_pylist()).reshape(-1,4,2,4)  # (T,cube,cam,feat)
for ci,cam in enumerate(["cam_scene","cam_wrist"]):
    size=cube[:,:,ci,2].reshape(-1); nz=size[size>0]
    print(f"{cam}: n={len(nz)} median={np.median(nz):.2f} p90={np.percentile(nz,90):.2f} p99={np.percentile(nz,99):.2f} max={nz.max():.2f} frac>3={np.mean(nz>3):.3f}")
