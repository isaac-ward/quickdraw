"""Upload ONLY the changed files (parquets + meta + stats + docs) to isaac-ronald-ward/block-stack main, as one
atomic commit. Videos / episodes / tasks.parquet / other files are NOT touched (non-destructive superset)."""
import os
from huggingface_hub import HfApi, CommitOperationAdd
STAGE="/app/logs/_hf_stage"; REPO="isaac-ronald-ward/block-stack"
SPLITS=["train","val","eval_purple_play","eval_purple_stack"]
paths=[]
for sp in SPLITS:
    paths += [f"{sp}/data/chunk-000/file-000.parquet", f"{sp}/meta/info.json", f"{sp}/meta/stats.json"]
paths += ["normalization_stats.json","PROCESSING.md","README.md"]
ops=[]
for p in paths:
    local=os.path.join(STAGE,p)
    assert os.path.exists(local), f"missing staged file {local}"
    ops.append(CommitOperationAdd(path_in_repo=p, path_or_fileobj=local))
print(f"uploading {len(ops)} files to {REPO}@main ...")
api=HfApi()
info=api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, revision="main",
    commit_message="Add observation.objects.cube [32] + rename observation_vector->observation.state (non-destructive superset)",
    commit_description="Per-frame cube object-state (R/G/B/P x 2 cams x [cx/W,cy/H,sqrt(area/TYP),soft_vis]) via per-episode color seg; hold-last+vis=0 on occlusion; absent color=zeros. observation_vector renamed to lerobot-canonical observation.state (values identical). Cameras/action/indices/videos unchanged. Stats + PROCESSING.md/README updated.")
print("COMMIT:", getattr(info,"commit_url",info))
