"""Load a published world model and imagine with it — the consumer-facing entry point.

WHY THIS EXISTS. Loading a trained model was eight lines of tribal knowledge repeated by hand in nine
one-off scripts, and two of those lines are not guessable:

    model = build_model(cfg)                                   # cfg, not the checkpoint, defines the shape
    sd = torch.load(ckpt)["state_dict"]
    model.load_state_dict({k[6:]: v for k, v in sd.items()      # strip the Lightning "model." prefix
                           if k.startswith("model.")}, strict=False)
    norm = Normalizer.from_file(resolve_data_root(cfg))          # THE STATS LIVE IN THE DATASET

The last one is the trap. A checkpoint carries `state_dict` and an EMPTY `hyper_parameters`, so the weights
hold no record of the architecture that produced them AND no normalisation statistics. Without the right
stats every rollout is silently wrong rather than obviously broken — and for a private dataset a consumer
cannot obtain them at all. A published model repo therefore bundles its own copy, and this loader reads it
from there rather than from a dataset it may not have.

    from quickdraw import load_pretrained
    model, norm, cfg = load_pretrained("isaac-ronald-ward/quickdraw-wm-robocasa-vl128")

See docs/using_pretrained_models.md for the imagination walkthrough and the three traps that bite
(images are [0,1] and NOT normalised while vectors ARE; the action tensor needs P+H-1 steps, not H;
`decode_chunk` or a long horizon exhausts the image decoder, ~78% of per-sample memory).
"""

from __future__ import annotations

import json
import os

import torch
from omegaconf import OmegaConf

WEIGHTS = "weights.safetensors"
CONFIG = "config.resolved.yaml"
STATS = "normalization_stats.json"
CONTEXT = "example_context.npz"
TRAIN_STATE = "training_state.ckpt"


def _snapshot(repo_or_path: str, revision: str | None = None) -> str:
    """Local directory for a model repo id, or the path itself if it is already a directory."""
    if os.path.isdir(repo_or_path):
        return repo_or_path
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo_or_path, repo_type="model", revision=revision)


def _load_weights(path: str) -> dict:
    """safetensors if present, else the raw Lightning checkpoint (strip the `model.` prefix)."""
    st = os.path.join(path, WEIGHTS)
    if os.path.exists(st):
        from safetensors.torch import load_file
        return load_file(st)
    ck = os.path.join(path, TRAIN_STATE)
    if not os.path.exists(ck):
        raise FileNotFoundError(
            f"{path} has neither {WEIGHTS} nor {TRAIN_STATE}. A published model repo needs at least one; "
            f"see quickdraw.push_model for what publish writes.")
    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    return {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}


def load_pretrained(repo_or_path: str, device: str | torch.device = "cpu",
                    revision: str | None = None, strict: bool = False):
    """Return `(model, normalizer, cfg)` for a published quickdraw world model.

    `repo_or_path` is a Hub model id (`<namespace>/<name>`) or a local directory. `device` moves the model
    and puts it in eval mode. `strict` controls `load_state_dict`; the default False tolerates buffers that
    a newer code version added, and the count of missing/unexpected keys is printed either way so a real
    mismatch is visible rather than silent.

    The normalizer comes from the repo's own `normalization_stats.json`, NOT from a dataset — that is the
    whole point of bundling it. Its `subset_obs()` is applied so norm/denorm match whatever obs subset the
    training run used (a no-op unless `set_obs_keep` is active).
    """
    path = _snapshot(repo_or_path, revision)
    cfg_path = os.path.join(path, CONFIG)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"{path} has no {CONFIG} — without it the weights cannot be given a shape")
    cfg = OmegaConf.load(cfg_path)

    from .data.dataset import Normalizer
    from .training.setup import build_model
    model = build_model(cfg)
    miss, unexp = model.load_state_dict(_load_weights(path), strict=strict)
    if miss or unexp:
        print(f"[load_pretrained] {len(miss)} missing / {len(unexp)} unexpected keys "
              f"(strict={strict}). First few: missing {list(miss)[:3]} unexpected {list(unexp)[:3]}",
              flush=True)
    model = model.to(device).eval()

    stats_path = os.path.join(path, STATS)
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"{path} has no {STATS}. Without the training normalisation statistics every rollout is "
            f"SILENTLY WRONG rather than visibly broken, so this is a hard error, not a warning.")
    with open(stats_path) as f:
        norm = Normalizer(json.load(f)).subset_obs()
    return model, norm, cfg


def load_example_context(repo_or_path: str, revision: str | None = None) -> dict:
    """The small (obs, act, frames) sample a published repo ships, as a dict of numpy arrays.

    Exists so the quickstart runs for someone who does not have the training dataset — which for the
    robocasa model is everyone outside this project, because that dataset is private.
    """
    import numpy as np
    path = _snapshot(repo_or_path, revision)
    p = os.path.join(path, CONTEXT)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{path} has no {CONTEXT} — publish it with quickdraw.push_model")
    with np.load(p) as z:
        return {k: z[k] for k in z.files}
