"""Shared builders: cfg -> dataclasses, model, datasets, loaders. Used by all entrypoints."""

from __future__ import annotations

import os

import torch

from ..data.dataset import (
    MMWindowLoader, Normalizer, TrajectoryDataset,
    load_split_episodes, load_split_episodes_mm,
)
from ..environments.torus_utils import TorusConfig


def _recorded_dt(cfg, fallback: float) -> float:
    """A recorded env's dt is DATASET-SPECIFIC (velocity/physics quantities scale with it), but
    conf/environments/recorded.yaml ships one default (starling's 30 Hz). Prefer the dataset's OWN frame
    rate: read `fps` from the run folder's summary.json (dt = 1/fps). Fall back to environments.dt with a
    LOUD warning when the dataset's fps is unknown, so a silent Hz mismatch can't quietly corrupt training."""
    import json
    try:
        root = resolve_data_root(cfg)
        fps = json.load(open(os.path.join(root, "summary.json"))).get("fps")
    except Exception as ex:   # noqa: BLE001
        print(f"[env_cfg] WARNING: could not read fps from the dataset ({type(ex).__name__}); using "
              f"environments.dt={fallback}. If that's not your data's 1/fps, set +environments.dt.", flush=True)
        return fallback
    if fps:
        dt = 1.0 / float(fps)
        if abs(dt - fallback) > 1e-6:
            print(f"[env_cfg] recorded dt <- dataset fps {fps} => dt={dt:.5f} "
                  f"(overrides environments.dt={fallback})", flush=True)
        return dt
    print(f"[env_cfg] WARNING: dataset summary.json has no fps; using environments.dt={fallback}. "
          f"Set +environments.dt to your data's 1/fps if that's wrong.", flush=True)
    return fallback


def env_cfg(cfg):
    """cfg.environments -> the env's config dataclass (torus: TorusConfig, unchanged; recorded: RecordedConfig)."""
    e = cfg.environments
    if str(e.get("name", "torus_world")).lower() in ("torus_world", "torus", "torusworld-v0"):
        return TorusConfig(R=e.R, r=e.r, dt=e.dt, gamma=e.gamma, a_max=e.a_max, init_speed=e.init_speed, mass=e.mass)
    if str(e.name).lower() == "pendulum":
        from ..environments.examples.pendulum import PendulumConfig
        return PendulumConfig(dt=float(e.dt), max_torque=float(e.max_torque), g=float(e.g),
                              m=float(e.m), l=float(e.l))
    if str(e.name).lower() == "recorded":
        from ..environments.recorded import RecordedConfig
        return RecordedConfig(obs_dim=int(e.obs_dim), action_dim=int(e.action_dim),
                              dt=_recorded_dt(cfg, float(e.dt)))
    raise ValueError(f"env_cfg: no config dataclass for environments.name={e.name!r}")


def _modality_specs(cfg):
    """cfg.model.modalities -> list[ModalitySpec]. Defaults to a single proprio (6-vec) modality when none
    is configured, so proprio-only models are just the ONE spine with one modality (no separate vector path)."""
    from ..models.modalities import ModalitySpec
    ms = cfg.model.get("modalities", None)
    if not ms:
        return [ModalitySpec(name="proprio", kind="vector", dim=int(cfg.model.get("obs_dim", 6)))]
    specs = []
    for e in ms:
        kw = dict(e)
        if not isinstance(kw.get("img_size", 128), int):    # yaml [H, W] (ListConfig) -> plain tuple
            kw["img_size"] = tuple(int(s) for s in kw["img_size"])
        specs.append(ModalitySpec(**kw))
    return specs


def build_model(cfg):
    """Dispatch on cfg.model.name: data-space (DSAR) or latent-space (LSAR + a collapse mechanism) or
    diffusion; if cfg.model.modalities is set, build the MULTIMODAL variant (token-bag spine)."""
    m = cfg.model
    name = str(m.get("name", "base"))

    specs = _modality_specs(cfg)
    if specs is not None:
        from ..models.multimodal import MultiModalFlow, MultiModalDSAR, MultiModalLSAR
        compile_rollout = bool(m.get("compile_rollout", False))
        if compile_rollout:
            # FlexAttention has NO double-backward under torch.compile (design/variations.md), so the compiled
            # rollout and the contraction penalty (needs the eager sdpa-MATH double-backward path) are mutually
            # exclusive. Fail fast so the user picks one (don't silently disable either).
            cv = (cfg.get("variations") or {}).get("contraction", {}) or {}
            cw = float((cv.get("weight", 0.0) if hasattr(cv, "get") else getattr(cv, "weight", 0.0)) or 0.0)
            if cw > 0.0:
                raise ValueError("model.compile_rollout is mutually exclusive with variations.contraction "
                                 "(weight>0): FlexAttention has no double-backward under torch.compile, which the "
                                 "contraction Jacobian power-iteration requires. Disable one (compile_rollout=false "
                                 "or contraction.weight=0).")
            # The compiled FlexAttention Triton kernel requires per-head dim >= 16 (inductor raises
            # "NYI: embedding dimension ... must be at least 16" mid-compile otherwise). The EAGER rollout has
            # no such floor (it runs the unfused fallback), so this only blocks compile_rollout. Fail fast here
            # at build time, not with a cryptic inductor error ~15 min in at the first p_tf==0 compile.
            head_dim = int(m.d) // int(m.heads)
            if head_dim < 16:
                raise ValueError(
                    f"model.compile_rollout needs head_dim (d/heads) >= 16 for the compiled FlexAttention kernel, "
                    f"but d={m.d}/heads={m.heads} -> head_dim={head_dim}. Raise d or lower heads so d/heads >= 16 "
                    f"(e.g. d=128/heads=8), or disable compile_rollout (eager has no floor).")
        common = dict(specs=specs, d=m.d, depth=m.depth, heads=m.heads, window=m.window,
                      mlp_ratio=m.mlp_ratio, rope_theta=m.rope_theta, action_dim=m.get("action_dim", 2),
                      grad_checkpoint=bool(m.get("grad_checkpoint", False)),
                      compile_rollout=compile_rollout)
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


def autobatch_find(cfg, device, log=print) -> int:
    """Binary-search the largest `data.batch` whose AR training step fits `VRAM*(1-headroom)`, capped at
    `autobatch_max`. The AR step is DISPATCH-bound (accelerations.md Exp 8), so bigger batch is nearly-free
    throughput until memory binds — pick the largest that fits with headroom. Builds a THROWAWAY model +
    SYNTHETIC batches (memory is shape- not value-dependent), probes real `rollout_train` fwd + decode + bwd,
    then frees. Headroom covers the optimizer states, eval-phase spikes, and allocator reserve — raise
    `data.autobatch_headroom` if an eval OOMs. Config: data.autobatch{,_headroom,_max,_base}."""
    import gc
    headroom = float(cfg.data.get("autobatch_headroom", 0.15))
    cap = int(cfg.data.get("autobatch_max", 512))
    base = int(cfg.data.get("autobatch_base", 16))
    total = torch.cuda.get_device_properties(device).total_memory
    budget = int(total * (1.0 - headroom))
    P, F = int(cfg.data.P), int(cfg.data.F); L = P + F
    de = int(cfg.model.get("detach_every", 16)); rf = float(cfg.model.get("recon_frac", 1.0))
    specs = _modality_specs(cfg); adim = int(cfg.model.get("action_dim", 2))
    model = build_model(cfg).to(device).train()
    # The eager binary search sizes MEMORY, which is ~compile-independent (validation: 22.9 GB compiled vs eager),
    # so probe EAGER — else the compiled path recompiles at every probe batch size (~96 s each = thrash). If
    # compile_rollout is on, confirm_compiled() re-enables it and validates the chosen batch once on the real
    # compiled step (stepping down on OOM), so the returned batch is checked on the exact path training will run.
    compiled_run = bool(cfg.model.get("compile_rollout", False))
    model.compile_rollout = False

    def synth(B):
        obs = {}
        for s in specs:
            if getattr(s, "kind", "vector") == "image":
                sz = getattr(s, "img_size", 128)
                hw = (sz, sz) if isinstance(sz, int) else tuple(int(x) for x in sz)
                obs[s.name] = torch.rand(B, L, hw[0], hw[1], 3, device=device)
            else:
                obs[s.name] = torch.randn(B, L, int(getattr(s, "dim", 6)), device=device)
        act = torch.randn(B, L - 1, adim, device=device)
        return {k: v[:, :P] for k, v in obs.items()}, act, {k: v[:, P:] for k, v in obs.items()}

    def probe(B):   # peak bytes for one real AR step, or None on OOM
        torch.cuda.synchronize(device); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        try:
            ctx, act, fut = synth(B)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = model.rollout_train(ctx, act, fut, 0.0, de)
                loss = preds.float().pow(2).mean()
                try:   # add the decode-backward footprint (recon_frac subset) — deterministic decoders only
                    k = max(1, int(round(rf * preds.shape[1])))
                    loss = loss + sum(v.float().pow(2).mean() for v in model.to_obs(preds[:, :k]).values())
                except Exception:   # flow/other decoders that don't decode cleanly here — rollout-only estimate
                    pass
            loss.backward(); model.zero_grad(set_to_none=True)
            return torch.cuda.max_memory_allocated(device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not (isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()):
                raise
            model.zero_grad(set_to_none=True); torch.cuda.empty_cache(); return None

    def fits(B):
        p = probe(B); return (p is not None and p <= budget), p

    def done(b):   # free probe activations before the caller builds loaders + the real model
        gc.collect(); torch.cuda.empty_cache(); return b

    def confirm_compiled(b):   # validate the eager-chosen batch on the REAL compiled step (recompiles per shape)
        if not compiled_run:
            return b
        model.compile_rollout = True
        for _ in range(4):
            log(f"[autobatch] compiled-confirm: probing batch {b} on the compiled step (one-time ~1-2 min compile)...")
            ok_, p_ = fits(b)
            if ok_:
                log(f"[autobatch] compiled-confirm OK: batch {b} ({(p_ or 0)/1e9:.1f}/{budget/1e9:.0f}GB compiled)")
                return b
            if b - 8 < base:
                log(f"[autobatch] compiled step over budget down to base {base}; using {base}")
                return base
            log(f"[autobatch] compiled step over budget at {b} — stepping down to {b - 8}")
            b -= 8
        return b

    ok, p = fits(base)
    if not ok:
        log(f"[autobatch] base batch {base} already over budget ({(p or 0)/1e9:.1f}/{budget/1e9:.0f}GB) — using {base}")
        return done(base)
    lo = hi = base
    while hi * 2 <= cap:                       # double to bracket [lo fits, hi over]
        ok, p = fits(hi * 2)
        if ok: lo = hi = hi * 2
        else: hi = hi * 2; break
    if lo == hi:                               # fit to the cap without going over
        log(f"[autobatch] fits to cap: data.batch={lo} (VRAM {total/1e9:.0f}GB @ {int(headroom*100)}% headroom, cap {cap})")
        return done(confirm_compiled(lo))
    while hi - lo > 8:                          # bisect to a multiple of 8
        mid = (((lo + hi) // 2) // 8) * 8
        if mid <= lo or mid >= hi: break
        ok, p = fits(mid)
        if ok: lo = mid
        else: hi = mid
    log(f"[autobatch] chose data.batch={lo}  (VRAM {total/1e9:.0f}GB, budget {budget/1e9:.0f}GB @ "
        f"{int(headroom*100)}% headroom; probed rollout_train fwd+decode+bwd)")
    return done(confirm_compiled(lo))


def resolve_data_root(cfg) -> str:
    """data.hf_repo unset -> cfg.data.root verbatim (local path, unchanged behavior). Set -> download+cache
    the HF dataset repo (the whole run folder: per-split subdirs + normalization_stats.json) and return that
    local snapshot path — same layout as a local run dir, so everything downstream is unchanged."""
    repo = cfg.data.get("hf_repo", None)
    if not repo:
        return cfg.data.root
    if repo not in _hf_root_cache:
        from huggingface_hub import snapshot_download   # HF_TOKEN read from env
        _hf_root_cache[repo] = snapshot_download(repo_id=repo, repo_type="dataset")
    return _hf_root_cache[repo]


def normalizer(cfg) -> Normalizer:
    return Normalizer.from_file(resolve_data_root(cfg))


def window_loaders(cfg, norm: Normalizer):
    """The ONE GPU-resident loader for every model. Loads the FPV frame store only when an image modality
    is present; proprio-only just loads (obs, act) — no frames touched."""
    P, F = cfg.data.P, cfg.data.F
    root = resolve_data_root(cfg)
    specs = _modality_specs(cfg)
    img = next((s for s in specs if s.kind == "image"), None)   # image modality (if any) -> resident frame store
    cam, repo = str(cfg.data.get("cam", "fpv")), str(cfg.data.get("repo_id", "torus"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {}
    for split, shuffle in (("train", True), ("val", False)):
        stride = int(cfg.data.get("window_stride", 1)) if split == "train" else 1   # subsample TRAIN windows only; val stays dense
        if img is not None:
            eps = load_split_episodes_mm(root, split, img_size=img.img_size, cam=cam, repo_id=repo)
            loaders[split] = MMWindowLoader(eps, P, F, norm, cfg.data.batch, shuffle, dev, image_head=img.name, stride=stride)
        else:                                                    # proprio-only: (obs, act) pairs, no camera frames
            eps = load_split_episodes(root, split, repo_id=repo)
            loaders[split] = MMWindowLoader(eps, P, F, norm, cfg.data.batch, shuffle, dev, stride=stride)
    return loaders


def eval_episodes(cfg, norm: Normalizer, split: str):
    return TrajectoryDataset(load_split_episodes(resolve_data_root(cfg), split,
                                                 repo_id=str(cfg.data.get("repo_id", "torus"))), norm)


def data_exists(cfg) -> bool:
    root = resolve_data_root(cfg)
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
