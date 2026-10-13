import numpy as np, pyarrow.parquet as pq
for sp in ["train","eval_purple_stack"]:
    n=pq.read_table(f"/app/logs/_hf_stage/{sp}/data/chunk-000/file-000.parquet")
    cube=np.array(n.column("observation.objects.cube").to_pylist()).reshape(-1,4,2,4)  # (T,cube,cam,feat)
    size=cube[...,2]  # size/TYP
    flat=size.reshape(-1); nz=flat[flat>0]
    print(f"{sp}: size/TYP nonzero n={len(nz)}  median={np.median(nz):.2f}  p90={np.percentile(nz,90):.2f}  "
          f"p99={np.percentile(nz,99):.2f}  max={nz.max():.2f}  frac>2={np.mean(nz>2):.3f}  frac>5={np.mean(nz>5):.4f}")
