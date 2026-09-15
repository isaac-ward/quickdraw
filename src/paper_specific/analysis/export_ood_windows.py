"""Export the reviewed OOD anomaly windows as ONE self-documenting file, and push it to the dataset repo.

starling-2 ships the OOD clips but nothing that says WHERE in each clip the anomaly is, or which clips to
drop -- so anyone pulling it has to re-review by hand. This writes that annotation next to the data.

    python scratch/export_ood_windows.py [--push]
"""
from __future__ import annotations

import json
import os
import sys

from omegaconf import OmegaConf

from quickdraw.data.dataset import load_split_episodes, set_action_aggregate, set_subsample
from quickdraw.data.ood_windows import EXCLUDE_EPISODES, FPS, WINDOWS
from quickdraw.training.setup import resolve_data_root

REPO = "isaac-ronald-ward/starling-2"
NAME = "ood_anomaly_windows.json"

HOW = (
    "Human-reviewed anomaly windows for the OOD splits of starling-2, and the episodes to drop. Reviewed "
    "2026-09-15 by watching every clip beside its proprio and action traces. Use this to clean the data "
    "before measuring anything on these splits.\n\n"
    "eval_ood_noodle is a VISUAL anomaly: a pink pool noodle is waved in front of the lens while the drone "
    "holds a single constant command. It is invisible in the proprio. eval_ood_leafblower is a DYNAMIC "
    "anomaly: an off-camera leafblower pushes the drone, again under one constant command, so it is "
    "invisible in the image and shows up as motion the sticks never asked for. Both splits hold exactly one "
    "unique action row for their whole duration, which is what makes any departure attributable.\n\n"
    "Each episode entry gives `ignore` and, when not ignored, the window in FRAMES at 15 Hz and in seconds. "
    "`model_steps_at_stride_4` is the same window for a model trained with data.subsample=4. Ignored "
    "episodes are ones where the anomaly never really materialised -- the noodle stayed out of frame, or "
    "the blower produced no motion above the drone's own hover noise -- and scoring them would dilute the "
    "result with clips that contain no event.\n\n"
    "eval_memory_backwall1 / backwall2 carry no window: the whole clip is the experiment (turn away from a "
    "scene and back). They carry one exclusion instead, listed under `exclude_episodes`, for an episode "
    "whose return comes too late in the clip to fall inside a rollout that also needs context frames."
)


def main(*args: str) -> int:
    cfg = OmegaConf.create(json.load(open(
        "logs/paper_icra_2027/model_backups/train_action_2026_09_14_04_41_17_s2_ah_chunk32_full/logs/config.json")))
    set_subsample(1); set_action_aggregate("concat")
    root = resolve_data_root(cfg)
    doc = {"how_to_use": HOW, "fps": FPS, "reviewed": "2026-09-15", "splits": {}}
    for sp in ("eval_ood_noodle", "eval_ood_leafblower", "eval_memory_backwall1", "eval_memory_backwall2"):
        eps = load_split_episodes(root, sp, repo_id="starling-2")
        w = WINDOWS.get(sp, {})
        rows = []
        for i, (o, _) in enumerate(eps):
            n = len(o)
            e = {"episode": i, "frames": n, "duration_s": round(n / FPS, 2)}
            if sp in WINDOWS:
                win = w.get(i)
                e["ignore"] = win is None
                if win is not None:
                    a, b = win[0], min(win[1], n)
                    e.update({"start_frame": a, "end_frame": b,
                              "start_s": round(a / FPS, 2), "end_s": round(b / FPS, 2),
                              "model_steps_at_stride_4": [a // 4, -(-b // 4)]})
            else:
                e["ignore"] = i in EXCLUDE_EPISODES.get(sp, [])
            rows.append(e)
        kind = ("visual (pink pool noodle in frame; not visible in proprio)" if "noodle" in sp else
                "dynamic (off-camera leafblower; not visible in the image)" if "leafblower" in sp else
                "memory (turn away from a scene and back; whole clip is the experiment)")
        doc["splits"][sp] = {"anomaly_kind": kind, "n_episodes": len(eps),
                             "n_kept": sum(1 for r in rows if not r["ignore"]), "episodes": rows}
        print(f"  {sp:24s} {len(eps):2d} episodes, {doc['splits'][sp]['n_kept']:2d} kept")
    out = os.path.join("logs", "paper_icra_2027", NAME)
    json.dump(doc, open(out, "w"), indent=1)
    print(f"\n  wrote {out} ({os.path.getsize(out)} bytes)")
    if "--push" in args:
        from huggingface_hub import upload_file
        info = upload_file(path_or_fileobj=out, path_in_repo=NAME, repo_id=REPO, repo_type="dataset",
                           commit_message="ood_anomaly_windows.json: reviewed anomaly windows + episodes to drop")
        print(f"  pushed to {REPO}: {info}")
    else:
        print("  (not pushed; re-run with --push)")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
