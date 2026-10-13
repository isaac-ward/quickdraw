import glob, numpy as np, pyarrow.parquet as pq
snap=glob.glob("/caches/hf/hub/datasets--isaac-ronald-ward--block-stack/snapshots/*/")[0]
for sp in ["train","val","eval_purple_play","eval_purple_stack"]:
    o=pq.read_table(f"{snap}{sp}/data/chunk-000/file-000.parquet")
    n=pq.read_table(f"/app/logs/_hf_stage/{sp}/data/chunk-000/file-000.parquet")
    assert o.num_rows==n.num_rows, sp
    for c in ["action","timestamp","frame_index","episode_index","index","task_index"]:
        assert o.column(c).to_pylist()==n.column(c).to_pylist(), f"{sp}:{c} changed!"
    ov=np.array(o.column("observation_vector").to_pylist()); st=np.array(n.column("observation.state").to_pylist())
    assert np.array_equal(ov,st), f"{sp}: state != observation_vector"
    cube=np.array(n.column("observation.objects.cube").to_pylist())
    assert cube.shape==(o.num_rows,32), f"{sp} cube shape {cube.shape}"
    assert "observation_vector" not in n.column_names, f"{sp}: observation_vector still present (should be renamed)"
    print(f"{sp}: rows={o.num_rows} | orig cols preserved | state==obs_vec | cube {cube.shape} range[{cube.min():.2f},{cube.max():.2f}]")
print("VERIFY PASSED: non-destructive except intended rename obs_vector->state + added cube")
