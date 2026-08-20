"""Validate a dataset FITS the training config before you launch a run — the two failures that otherwise
show up minutes in: (1) episodes too short for `P+F` windows (0 training windows), and (2) obs/action dims
that don't match the chosen env. Reads the proprio vectors only (no video decode), so it's fast and CPU-only.
Exits non-zero if anything is wrong, so the wizard / a script can gate on it.

  uv run python -m quickdraw.check_dataset data.root=<run_dir> data.repo_id=<name> environments=recorded
  uv run python -m quickdraw.check_dataset data.hf_repo=<ns>/<name> data.repo_id=<name> environments=recorded
"""

from __future__ import annotations

import glob
import os
import sys

import hydra
import numpy as np

from .data.dataset import load_split_episodes, set_obs_keep
from .training.setup import env_cfg, resolve_data_root


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    root = resolve_data_root(cfg)
    set_obs_keep(cfg.data.get("obs_keep", None))   # check the EFFECTIVE obs (subset), so dims match env/model
    P, F = int(cfg.data.P), int(cfg.data.F)
    need = P + F
    repo = str(cfg.data.get("repo_id", "torus"))
    print(f"[check] data={root}  P={P} F={F}  (an episode needs >= P+F = {need} steps for 1 window)  repo_id={repo}")

    env_obs = env_act = None
    try:
        e = env_cfg(cfg)
        env_obs, env_act = int(e.obs_dim), int(e.action_dim)
        print(f"[check] env '{cfg.environments.name}': obs_dim={env_obs} action_dim={env_act}")
    except Exception as ex:   # noqa: BLE001 — env dims are best-effort; keep checking windows regardless
        print(f"[check] (no env dims to cross-check: {type(ex).__name__}: {ex})")

    present = {os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(root, "*", "meta"))}
    splits = [s for s in ("train", "val", "eval") if s in present] or ["train", "val"]

    problems = []
    for split in splits:
        try:
            eps = load_split_episodes(root, split, repo_id=repo)
        except Exception as ex:   # noqa: BLE001
            problems.append(f"{split}: failed to load ({type(ex).__name__}: {ex})")
            continue
        if not eps:
            problems.append(f"{split}: no episodes")
            continue
        lens = [o.shape[0] for o, _ in eps]
        wins = sum(max(0, ln - need + 1) for ln in lens)
        od, ad = int(eps[0][0].shape[-1]), int(eps[0][1].shape[-1])
        print(f"[check] {split:5s}: {len(eps):4d} episodes | len min/mean/max = "
              f"{min(lens)}/{int(np.mean(lens))}/{max(lens)} | obs_dim={od} action_dim={ad} "
              f"-> {wins} training windows")
        if wins == 0:
            problems.append(f"{split}: 0 training windows — longest episode {max(lens)} < P+F {need}; "
                            f"lower data.F / data.P, or use longer episodes")
        if env_obs is not None and od != env_obs:
            problems.append(f"{split}: data obs_dim {od} != env obs_dim {env_obs} "
                            f"(fix environments.obs_dim and model.modalities.0.dim)")
        if env_act is not None and ad != env_act:
            problems.append(f"{split}: data action_dim {ad} != env action_dim {env_act} "
                            f"(fix environments.action_dim and model.action_dim)")

    if problems:
        print("\n[check] PROBLEMS:")
        for p in problems:
            print(f"  x {p}")
        sys.exit(1)
    print("\n[check] OK — dataset fits the config")


if __name__ == "__main__":
    main()
