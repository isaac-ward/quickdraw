"""Push a generated dataset run to the HF Hub as ONE dataset repo (the whole run folder).

  python -m quickdraw.push_to_hub data.root=logs/data_generation_<ts>_<exp> +hub.name=torus-world

Datasets are PUBLIC by default; pass `+hub.private=true` to keep one private.
Uploads the entire run directory (all splits' parquet/meta + normalization stats + media + summary)
under a single repo `<namespace>/<name>`, so it is one thing to browse on the Hub. A given split
loads back with `LeRobotDataset("torus/<split>", root="<downloaded-repo>/<split>")`. Auth uses
HF_TOKEN from the environment (.env)."""

from __future__ import annotations

import json
import os

import hydra
from huggingface_hub import HfApi


def _generic_card(name: str, s: dict) -> str:
    """Card for a run whose summary.json has NO torus geometry fields (e.g. a data.processors run):
    obs/action dims (from the norm stats), splits, fps — no manifold-specific prose."""
    counts = s["counts"]
    splits = list(counts)
    norm = s.get("normalization_stats", {})
    obs_dim = len(norm.get("observation_vector", {}).get("mean", [])) or "?"
    act_dim = len(norm.get("action", {}).get("mean", [])) or "?"
    fps, cam = s.get("fps", "?"), s.get("camera", "cam")
    hw = s.get("image_hw")
    hw_txt = f"{hw[0]}×{hw[1]}×3, video" if hw else "video"
    cfgs = "\n".join(f"  - config_name: {sp}\n    data_files: {sp}/data/**/*.parquet" for sp in splits)
    rows = "\n".join(f"| `{sp}` | {counts[sp]['episodes']} | {counts[sp]['steps_per_episode']} | "
                     f"{counts[sp]['transitions']} |" for sp in splits)
    return f"""---
license: mit
pretty_name: {name}
tags:
- world-models
- robotics
configs:
{cfgs}
---

# {name}

Recorded trajectories (no simulator) packaged as LeRobot splits for world-model training.
Generated with [quickdraw](https://github.com/isaac-ward/quickdraw) (`data.processors`).

## Observation / action
- **observation_vector** ({obs_dim}): the recorded state
- **observation.images.{cam}** ({hw_txt}): the recorded camera stream, stored as MP4, aligned 1:1
  with the vector frames — the image modality for vision models
- **action** ({act_dim}): the recorded actions, at {fps} Hz

## Splits
| split | episodes | steps/ep (mean) | frames |
|---|---|---|---|
{rows}

Normalization statistics are computed on **train only** and applied to every split.
"""


def _make_card(root: str, name: str) -> str:
    """Build a dataset card (README.md) from the run's own summary.json: YAML frontmatter with a HF
    viewer `configs` block (so each split's parquet is browsable) + a human-readable description.
    Torus-generated runs get the full torus card; runs without torus fields get a generic card."""
    s = json.load(open(os.path.join(root, "summary.json")))
    counts, split_env, coloring = s["counts"], s["split_env"], s["coloring"]
    splits = list(counts)
    if not all("R" in (split_env.get(sp) or {}) for sp in splits):   # no torus geometry -> generic card
        return _generic_card(name, s)
    action_sampler = s.get("action_sampler", "ornstein_uhlenbeck")
    action_desc = {
        "ornstein_uhlenbeck": "an **Ornstein–Uhlenbeck** action process (temporally-correlated, unimodal, zero-mean)",
        "bimodal": "a **two-basin** action process with a **bimodal action-magnitude** distribution "
                   "(low- vs high-thrust rings) and temporal smoothing — occasional smooth hops between basins",
    }.get(action_sampler, f"a `{action_sampler}` action process")

    cfgs = "\n".join(f"  - config_name: {sp}\n    data_files: {sp}/data/**/*.parquet" for sp in splits)
    rows = "\n".join(
        f"| `{sp}` | {counts[sp]['episodes']} | {counts[sp]['steps_per_episode']} | "
        f"{counts[sp]['transitions']} | R={split_env[sp]['R']}, r={split_env[sp]['r']}, "
        f"γ={split_env[sp]['gamma']}, a_max={split_env[sp]['a_max']} | {coloring[sp]} |"
        for sp in splits
    )
    return f"""---
license: mit
pretty_name: {name}
tags:
- world-models
- robotics
- torus
- long-horizon
configs:
{cfgs}
---

# {name}

A toy world-model benchmark: a damped particle driven by {action_desc} around a
**torus** manifold. The benchmark measures long-horizon consistency as *staying on the data
manifold*. Generated with [quickdraw](https://github.com/isaac-ward/quickdraw).

## Observation / action
- **observation_vector** (6): `[position (x,y,z); velocity (ẋ,ẏ,ż)]`
- **observation.images.fpv** (256×256×3, video): the egocentric / first-person-view RGB frame, stored
  as MP4 (torchcodec-decoded), aligned 1:1 with the vector frames — the image modality for vision models
- **action** (2): `(a_θ, a_φ)` — angular pushes in the two torus coordinates

## Action distribution
The action process is `{action_sampler}`. Its per-timestep action-magnitude distribution (8 timesteps,
regenerated with every dataset build) — read the two peaks for the bimodal sampler:

![action distribution](media/action_distribution.png)

## Splits
| split | episodes | steps/ep | transitions | environment | coloring |
|---|---|---|---|---|---|
{rows}

`train`/`val` are in-distribution. Each `eval_ood_*` changes exactly one axis: **visual** (texture
only), **geometric** (R, r only), **dynamics** (γ, a_max only); `eval_ood_horizon` is the in-distribution
long-horizon eval.

## Metrics (long-horizon rollout)
- **manifold_distance_error** = `|signed_dist(p̂)|` — perpendicular distance off the torus surface (0 = on-surface)
- **pointwise_error** = `‖p̂ − p‖` — distance to the true point at the same step
- **tangent_velocity_error** = `|⟨ṗ̂, n̂⟩|` — predicted velocity's component off the surface (0 = tangent)

Normalization statistics are computed on **train only** and applied to every split.
"""


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    root = cfg.data.root
    if not root or not os.path.exists(os.path.join(root, "dataset_card.json")):
        raise FileNotFoundError(f"No dataset run at data.root={root!r} (run data_generation first).")

    hub = cfg.get("hub", {}) or {}
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    namespace = hub.get("namespace") or api.whoami()["name"]
    name = hub.get("name") or os.path.basename(os.path.normpath(root))  # default: the run-folder name
    private = bool(hub.get("private", False))   # PUBLIC by default; +hub.private=true to keep private
    repo_id = f"{namespace}/{name}"

    with open(os.path.join(root, "README.md"), "w") as f:  # dataset card -> uploaded with the folder
        f.write(_make_card(root, name))
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    # delete_patterns="*" -> a CLEAN reupload: every remote file not in this upload is removed in the same
    # commit, so a regenerated dataset fully replaces the old one (no stale splits/media left behind).
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=root, delete_patterns="*")
    print(f"[push_to_hub] {root} -> https://huggingface.co/datasets/{repo_id} (private={private}, cleared+reuploaded)")


if __name__ == "__main__":
    main()
