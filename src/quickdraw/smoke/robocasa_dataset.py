"""Focused RoboCasa adapter + model smoke.

Run inside the QuickDraw container:
  uv run python -m quickdraw.smoke.robocasa_dataset /datasets/robocasa
"""

from __future__ import annotations

import sys

import imageio.v2 as imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir

from quickdraw.data.dataset import (
    derive_episode_partition,
    episode_inventory,
    load_split_episodes,
    normalization_stats,
    validate_lerobot_contract,
)
from quickdraw.training.setup import build_model, validate_model_data_dimensions


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} failed{': ' + detail if detail else ''}")
    print(f"[OK] {name}{' — ' + detail if detail else ''}", flush=True)


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "/datasets/robocasa"
    with initialize_config_dir(config_dir="/app/conf", version_base=None):
        cfg = compose(config_name="robocasa_world_model", overrides=[f"data.root={root}"])

    info, records = validate_lerobot_contract(root, str(cfg.data.source_split), cfg.data.schema)
    check("metadata totals", len(records) == 261 and int(info["total_frames"]) == 288_593)
    partition = derive_episode_partition(records, cfg.data.validation)
    train_ids, val_ids = partition["train_episode_indices"], partition["val_episode_indices"]
    check("deterministic 234/27 episode holdout", len(train_ids) == 234 and len(val_ids) == 27)
    check("episode holdout is disjoint", not set(train_ids) & set(val_ids))
    check("all eight groups represented", len(partition["groups"]) == 8 and
          all(item["train_episodes"] and item["val_episodes"] for item in partition["groups"].values()))

    episodes = load_split_episodes(root, "train", "observation.state", "action")
    check("all state/action episodes load", len(episodes) == 261)
    check("state/action boundary contract", all(
        obs.dtype == np.float32 and act.dtype == np.float32 and
        obs.shape[1:] == (16,) and act.shape[1:] == (12,)
        for obs, act in episodes
    ))
    by_id = {int(record["episode_index"]): episode for record, episode in zip(records, episodes, strict=True)}
    train = [by_id[index] for index in train_ids]
    val = [by_id[index] for index in val_ids]
    stats = normalization_stats(train, "observation.state", "action")
    check("normalization widths", len(stats["observation_vector"]["mean"]) == 16 and
          len(stats["action"]["mean"]) == 12)
    train_inv, val_inv = episode_inventory(train, 8, 64), episode_inventory(val, 8, 64)
    check("dense window total", train_inv["windows"] + val_inv["windows"] == 270_062,
          f"{train_inv['windows']} train + {val_inv['windows']} val")

    # Decode one complete authoritative video to check template, RGB, and frame alignment without
    # constructing the 13.2 GiB full-corpus cache (the bounded trainer smoke exercises that path).
    row = records[0]
    episode_index = int(row["episode_index"])
    rel = info["video_path"].format(
        episode_chunk=episode_index // int(info["chunks_size"]),
        episode_index=episode_index,
        video_key=str(cfg.data.schema.image_key),
    )
    reader = imageio.get_reader(f"{root}/train/{rel}")
    count, first = 0, None
    try:
        for frame in reader:
            if first is None:
                first = np.asarray(frame)[..., :3]
            count += 1
    finally:
        reader.close()
    check("selected camera aligns to episode length", count == int(row["length"]), str(count))
    check("selected camera is 256x256 RGB", first is not None and first.shape == (256, 256, 3))

    validate_model_data_dimensions(cfg, info)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).train()
    B, P, F = 1, 2, 2
    L = P + F
    obs = {
        "proprio": torch.randn(B, L, 16, device=device),
        "image": torch.rand(B, L, 128, 128, 3, device=device),
    }
    act = torch.randn(B, L, 12, device=device)
    preds = model({key: value[:, :-1] for key, value in obs.items()}, act[:, :-1])[:, P - 1:]
    future = {key: value[:, P:] for key, value in obs.items()}
    raw, weights = model.loss_terms(preds, future, obs, 1.0, act)
    recon = model.recon_losses(preds, future)
    modality_weights = {mod.name: float(mod.weight) for mod in model.modalities.values()}
    loss = sum(weights[key] * raw[key] for key in raw)
    loss = loss + sum(modality_weights[key.split("/")[-1]] * value for key, value in recon.items())
    loss.backward()
    check("prescribed model forward/backward", bool(torch.isfinite(loss).item()),
          f"loss={float(loss.detach()):.4f}")
    print("[robocasa_dataset] ALL OK", flush=True)


if __name__ == "__main__":
    main()
