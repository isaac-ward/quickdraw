"""Shared builders: cfg -> dataclasses, model, datasets, loaders. Used by all entrypoints."""

from __future__ import annotations

import hashlib
import json
import os

import torch

from ..data.dataset import (
    MMWindowLoader, Normalizer, TrajectoryDataset,
    derive_episode_partition, episode_inventory, load_split_episodes, load_split_episodes_mm,
    normalization_stats, validate_lerobot_contract,
)
from ..environments.torus import TorusConfig


def env_cfg(cfg) -> TorusConfig:
    e = cfg.environments
    return TorusConfig(R=e.R, r=e.r, dt=e.dt, gamma=e.gamma, a_max=e.a_max, init_speed=e.init_speed, mass=e.mass)


def _modality_specs(cfg):
    """cfg.model.modalities -> list[ModalitySpec]. Defaults to a single proprio (6-vec) modality when none
    is configured, so proprio-only models are just the ONE spine with one modality (no separate vector path)."""
    from ..models.modalities import ModalitySpec
    ms = cfg.model.get("modalities", None)
    if not ms:
        return [ModalitySpec(name="proprio", kind="vector", dim=int(cfg.model.get("obs_dim", 6)))]
    return [ModalitySpec(**dict(e)) for e in ms]


def build_model(cfg):
    """Dispatch on cfg.model.name: data-space (DSAR) or latent-space (LSAR + a collapse mechanism) or
    diffusion; if cfg.model.modalities is set, build the MULTIMODAL variant (token-bag spine)."""
    m = cfg.model
    name = str(m.get("name", "base"))

    specs = _modality_specs(cfg)
    if specs is not None:
        from ..models.multimodal import MultiModalFlow, MultiModalDSAR, MultiModalLSAR
        common = dict(specs=specs, d=m.d, depth=m.depth, heads=m.heads, window=m.window,
                      mlp_ratio=m.mlp_ratio, rope_theta=m.rope_theta, action_dim=m.get("action_dim", 2),
                      grad_checkpoint=bool(m.get("grad_checkpoint", False)))
        # diffusion forcing (variations.noise_injection.observations_encoded_pre_fusion) — "corrupt-and-tell"
        # noise on the pre-fusion context tokens. Flow models ONLY (needs the backbone level embedding) -> gate.
        ni = (cfg.get("variations") or {}).get("noise_injection", {}) or {}
        oe = (ni.get("observations_encoded_pre_fusion", {}) if hasattr(ni, "get") else {}) or {}
        oeg = (lambda k, v: oe.get(k, v)) if hasattr(oe, "get") else (lambda k, v: getattr(oe, k, v))
        df_scale = float(oeg("scale", 0.0) or 0.0)
        df_granularity = str(oeg("granularity", "timestep"))
        if df_scale > 0.0 and name not in ("mm_flow", "flow"):
            raise ValueError(f"variations.noise_injection.observations_encoded_pre_fusion (diffusion forcing) "
                             f"requires a flow model (model.name in mm_flow/flow); got {name!r}.")
        if name in ("mm_dsar", "dsar", "base"):
            return MultiModalDSAR(**common)
        if name in ("mm_lsar", "lsar"):
            from ..models.collapse import EMA, Reconstruction, make_collapse
            if cfg.get("collapse", None) is not None:          # conf/collapse group: naked/recon/ema/sigreg/vicreg
                strat = make_collapse(cfg.collapse)
            elif bool(m.get("ema", False)):                    # legacy mm_lsar_ema config -> EMA strategy
                strat = EMA(tau=float(m.get("ema_decay", 0.996)))
            else:
                strat = Reconstruction()
            return MultiModalLSAR(**common, lambda_pred_latent=m.get("lambda_pred_latent", 1.0),
                                  collapse=strat, lambda_reg=m.get("lambda_reg", 1.0))
        if name in ("mm_flow", "flow"):
            cv = (cfg.get("variations") or {}).get("contraction", {}) or {}
            cw = float((cv.get("weight", 0.0) if hasattr(cv, "get") else getattr(cv, "weight", 0.0)) or 0.0)
            if cw > 0.0:   # contraction differentiates the one-step map, which for diffusion runs through the ODE sampler
                raise ValueError("variations.contraction is mutually exclusive with the diffusion model "
                                 "(disable contraction, weight=0, to train diffusion).")
            d = m.get("diffusion", {})
            dfg = (lambda k, v: d.get(k, v)) if hasattr(d, "get") else (lambda k, v: getattr(d, k, v))
            ah = m.get("action_head", {}) or {}                # action-distribution prior (opt-in)
            ahg = (lambda k, v: ah.get(k, v)) if hasattr(ah, "get") else (lambda k, v: getattr(ah, k, v))
            return MultiModalFlow(**common, sampling_steps=int(dfg("sampling_steps", 6)),
                                       shortcut=bool(dfg("shortcut", False)), predict=str(dfg("predict", "residual")),
                                       stochastic_eval=bool(dfg("stochastic_eval", False)),
                                       time_sampling=str(dfg("time_sampling", "uniform")),
                                       flow_hidden=int(dfg("flow_hidden", 0)),
                                       lambda_flow=m.get("lambda_flow", 1.0),
                                       lambda_consistency=m.get("lambda_consistency", 1.0),
                                       df_scale=df_scale, df_granularity=df_granularity,
                                       action_head_enabled=bool(ahg("enabled", False)),
                                       action_head_weight=float(ahg("weight", 1.0)),
                                       action_head_shortcut=bool(ahg("shortcut", True)),
                                       action_head_detach_gradient=bool(ahg("detach_gradient", False)),
                                       dynamics_detach_encoder=bool(m.get("dynamics_detach_encoder", False)))
        raise ValueError(f"unknown model.name: {name!r}")


_hf_root_cache: dict = {}   # hf_repo -> snapshot path (avoid re-resolving/downloading per call)


def resolve_data_root(cfg) -> str:
    """data.hf_repo unset -> cfg.data.root verbatim (local path, unchanged behavior). Set -> download+cache
    the HF dataset repo (the whole run folder: per-split subdirs + normalization_stats.json) and return that
    local snapshot path — same layout as a local run dir, so everything downstream is unchanged."""
    repo = cfg.data.get("hf_repo", None)
    local = cfg.data.get("root", None)
    if repo and local:
        raise ValueError("data.root and data.hf_repo are mutually exclusive; configure exactly one")
    if not repo:
        return local
    if repo not in _hf_root_cache:
        from huggingface_hub import snapshot_download   # HF_TOKEN read from env
        _hf_root_cache[repo] = snapshot_download(repo_id=repo, repo_type="dataset")
    return _hf_root_cache[repo]


def normalizer(cfg) -> Normalizer:
    if str(cfg.data.get("kind", "torus")) == "robocasa":
        raise RuntimeError("RoboCasa normalization is derived from its logical train episodes; use prepare_training_data")
    return Normalizer.from_file(resolve_data_root(cfg))


def window_loaders(cfg, norm: Normalizer):
    """The ONE GPU-resident loader for every model. Loads the FPV frame store only when an image modality
    is present; proprio-only just loads (obs, act) — no frames touched."""
    P, F = cfg.data.P, cfg.data.F
    root = resolve_data_root(cfg)
    specs = _modality_specs(cfg)
    img = next((s for s in specs if s.kind == "image"), None)   # image modality (if any) -> resident frame store
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {}
    for split, shuffle in (("train", True), ("val", False)):
        stride = int(cfg.data.get("window_stride", 1)) if split == "train" else 1   # subsample TRAIN windows only; val stays dense
        if img is not None:
            eps = load_split_episodes_mm(root, split, img_size=img.img_size)
            loaders[split] = MMWindowLoader(eps, P, F, norm, cfg.data.batch, shuffle, dev, image_head=img.name, stride=stride)
        else:                                                    # proprio-only: (obs, act) pairs, no FPV frames
            eps = load_split_episodes(root, split)
            loaders[split] = MMWindowLoader(eps, P, F, norm, cfg.data.batch, shuffle, dev, stride=stride)
    return loaders


def validate_model_data_dimensions(cfg, info: dict | None = None) -> None:
    """Fail before model construction if selected model widths disagree with the data contract."""
    state_dim, action_dim = int(cfg.data.schema.state_dim), int(cfg.data.schema.action_dim)
    configured_action = int(cfg.model.get("action_dim", action_dim))
    if configured_action != action_dim:
        raise ValueError(f"model.action_dim={configured_action} does not match data action_dim={action_dim}")
    proprio = [entry for entry in cfg.model.get("modalities", []) if str(entry.name) == "proprio"]
    if len(proprio) != 1 or int(proprio[0].dim) != state_dim:
        got = None if not proprio else int(proprio[0].dim)
        raise ValueError(f"model proprio dim={got} does not match data state_dim={state_dim}")
    if info is not None:
        features = info["features"]
        for key, dim in ((str(cfg.data.schema.state_key), state_dim),
                         (str(cfg.data.schema.action_key), action_dim)):
            metadata_dim = list(features[key]["shape"])
            if metadata_dim != [dim]:
                raise ValueError(f"configured {key} width {dim} does not match metadata {metadata_dim}")


def _write_json(path: str, value: dict) -> None:
    with open(path, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def prepare_training_data(cfg, run_dir: str):
    """Build the complete training data state once and persist the exact split/stats used by the run."""
    if str(cfg.data.get("kind", "torus")) != "robocasa":
        norm = normalizer(cfg)
        loaders = window_loaders(cfg, norm)
        inventory = {}
        for split in ("train", "val"):
            spec = cfg.data.splits[split]
            n, steps = int(spec.n_traj), int(spec.steps)
            stride = int(cfg.data.get("window_stride", 1)) if split == "train" else 1
            inventory[split] = {
                "episodes": n,
                "frames": n * steps,
                "transitions": n * max(0, steps - 1),
                "windows": loaders[split].N,
                "window_stride": stride,
            }
        validate_model_data_dimensions(cfg)
        return norm, loaders, inventory

    root = resolve_data_root(cfg)
    source_split = str(cfg.data.source_split)
    info, records = validate_lerobot_contract(root, source_split, cfg.data.schema)
    validate_model_data_dimensions(cfg, info)
    partition = derive_episode_partition(records, cfg.data.validation)
    partition["source"] = {
        "root": os.path.realpath(root),
        "hf_repo": cfg.data.get("hf_repo", None),
        "source_split": source_split,
    }
    _write_json(os.path.join(run_dir, "dataset_split.json"), partition)

    state_key, action_key = str(cfg.data.schema.state_key), str(cfg.data.schema.action_key)
    train_ids = partition["train_episode_indices"]
    val_ids = partition["val_episode_indices"]
    train_state = load_split_episodes(root, source_split, state_key, action_key, train_ids)
    val_state = load_split_episodes(root, source_split, state_key, action_key, val_ids)
    stats = normalization_stats(train_state, state_key, action_key)
    _write_json(os.path.join(run_dir, "normalization_stats.json"), stats)
    norm = Normalizer(stats)

    P, F = int(cfg.data.P), int(cfg.data.F)
    train_stride = int(cfg.data.get("window_stride", 1))
    inventory = {
        "train": episode_inventory(train_state, P, F, train_stride),
        "val": episode_inventory(val_state, P, F, 1),
    }
    specs = _modality_specs(cfg)
    images = [spec for spec in specs if spec.kind == "image"]
    if len(images) > 1:
        raise ValueError("RoboCasa dataset support currently accepts exactly one configured image modality")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = int(cfg.data.batch)
    if images:
        image = images[0]
        if image.name != str(cfg.data.schema.image_name):
            raise ValueError(
                f"model image modality {image.name!r} does not match data image_name={cfg.data.schema.image_name!r}"
            )
        if int(image.img_size) != int(cfg.data.schema.image_size):
            raise ValueError(
                f"model image size {image.img_size} does not match data image_size={cfg.data.schema.image_size}"
            )
        partition_hash = hashlib.sha256(json.dumps({
            "policy": partition["policy"],
            "train": train_ids,
            "val": val_ids,
        }, sort_keys=True).encode()).hexdigest()[:16]
        common = {
            "img_size": int(cfg.data.schema.image_size),
            "state_key": state_key,
            "action_key": action_key,
            "image_key": str(cfg.data.schema.image_key),
            "cache_root": cfg.data.get("cache_root", None),
            "cache_identity": partition_hash,
        }
        train_eps = load_split_episodes_mm(
            root, source_split, episode_indices=train_ids, **common
        )
        val_eps = load_split_episodes_mm(
            root, source_split, episode_indices=val_ids, **common
        )
        loaders = {
            "train": MMWindowLoader(train_eps, P, F, norm, batch, True, device,
                                    image_head=str(cfg.data.schema.image_name), stride=train_stride),
            "val": MMWindowLoader(val_eps, P, F, norm, batch, False, device,
                                  image_head=str(cfg.data.schema.image_name), stride=1),
        }
    else:
        loaders = {
            "train": MMWindowLoader(train_state, P, F, norm, batch, True, device, stride=train_stride),
            "val": MMWindowLoader(val_state, P, F, norm, batch, False, device, stride=1),
        }
    for split in loaders:
        if int(loaders[split].N) != int(inventory[split]["windows"]):
            raise AssertionError(f"{split} loader window count disagrees with episode inventory")
    return norm, loaders, inventory


def eval_episodes(cfg, norm: Normalizer, split: str):
    return TrajectoryDataset(load_split_episodes(resolve_data_root(cfg), split), norm)


def data_exists(cfg) -> bool:
    root = resolve_data_root(cfg)
    if str(cfg.data.get("kind", "torus")) == "robocasa":
        split = str(cfg.data.get("source_split", "train"))
        return bool(root) and os.path.isfile(os.path.join(root, split, "meta", "info.json"))
    return bool(root) and os.path.exists(os.path.join(root, "normalization_stats.json"))


def load_checkpoint(model, path: str):
    """Load a Lightning checkpoint into a bare BaseWorldModel, stripping wrapper prefixes.

    `path` may be a .ckpt file or a train run dir (resolved to <dir>/checkpoints/best.ckpt).
    """
    import torch

    if path and os.path.isdir(path):
        path = os.path.join(path, "checkpoints", "best.ckpt")
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    clean = {}
    for k, v in sd.items():
        if k.startswith("model."):
            k = k[len("model."):]
        k = k.replace("_orig_mod.", "")  # strip torch.compile wrapper anywhere (whole-model or submodule)
        clean[k] = v
    model.load_state_dict(clean, strict=False)
    return model
