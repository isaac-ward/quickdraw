import json
from huggingface_hub import HfApi, hf_hub_download
REPO="isaac-ronald-ward/block-stack"; api=HfApi()
files=api.list_repo_files(REPO, repo_type="dataset")
vids=[f for f in files if f.endswith(".mp4")]; parq=[f for f in files if f.endswith(".parquet")]
print(f"remote files: {len(files)} total | {len(vids)} mp4 videos still present | {len(parq)} parquet")
# check the new train info.json features
p=hf_hub_download(REPO, "train/meta/info.json", repo_type="dataset")
feats=list(json.load(open(p))["features"])
print("train features now:", feats)
print("observation.state present:", "observation.state" in feats,
      "| observation.objects.cube present:", "observation.objects.cube" in feats,
      "| observation_vector GONE:", "observation_vector" not in feats)
# normalization_stats
p2=hf_hub_download(REPO, "normalization_stats.json", repo_type="dataset")
ns=json.load(open(p2)); print("normalization_stats keys:", list(ns), "| cube std len:", len(ns["observation.objects.cube"]["std"]))
